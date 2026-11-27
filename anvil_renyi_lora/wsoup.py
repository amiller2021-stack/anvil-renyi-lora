"""FP32 multi-trajectory W_soup LoRA checkpoint merger."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import torch

_PEFT_A_RE = re.compile(r"^(.*)\.lora_A(?:\.[^.]+)?\.weight$")
_PEFT_B_RE = re.compile(r"^(.*)\.lora_B(?:\.[^.]+)?\.weight$")


@dataclass
class LoRATrajectory:
    """Single LoRA training trajectory to merge into a base checkpoint."""

    name: str
    state_dict: Dict[str, torch.Tensor]
    rank: int
    alpha: float
    weight: float = 1.0


@dataclass
class MergeReport:
    """Telemetry produced by merge_wsoup_fp32."""

    mode: str
    num_trajectories: int
    merged_layers: int
    mean_pairwise_cosine: float
    global_delta_l2: float
    global_relative_perturbation: float
    merged_sha256: str
    layer_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)


def _pair_lora_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Tuple[str, str]]:
    """Map base weight key -> (lora_a_key, lora_b_key) for native or PEFT dicts."""
    a_map: Dict[str, str] = {}
    b_map: Dict[str, str] = {}

    for key in state_dict:
        if key.endswith(".lora_a"):
            prefix = key[: -len(".lora_a")]
            a_map[f"{prefix}.weight"] = key
            a_map[f"{prefix}.base.weight"] = key
        elif key.endswith(".lora_b"):
            prefix = key[: -len(".lora_b")]
            b_map[f"{prefix}.weight"] = key
            b_map[f"{prefix}.base.weight"] = key
        else:
            m_a = _PEFT_A_RE.match(key)
            if m_a:
                prefix = m_a.group(1)
                a_map[f"{prefix}.weight"] = key
                a_map[f"{prefix}.base.weight"] = key
            m_b = _PEFT_B_RE.match(key)
            if m_b:
                prefix = m_b.group(1)
                b_map[f"{prefix}.weight"] = key
                b_map[f"{prefix}.base.weight"] = key

    pairs: Dict[str, Tuple[str, str]] = {}
    for target_key, a_key in a_map.items():
        if target_key in b_map:
            pairs[target_key] = (a_key, b_map[target_key])
    return pairs


def _ties_merge_fp32(
    deltas: List[torch.Tensor],
    weights: List[float],
    density: float = 0.5,
) -> torch.Tensor:
    """TIES-Merging in FP32: trim low-magnitude entries, elect sign, disjoint mean."""
    if not 0.0 < density <= 1.0:
        raise ValueError(f"ties_density must be in (0, 1], got {density}")

    trimmed = []
    for d in deltas:
        if density < 1.0:
            flat = d.abs().reshape(-1)
            k = max(1, int(round(flat.numel() * density)))
            thresh = torch.topk(flat, k, sorted=False).values.min()
            d = torch.where(d.abs() >= thresh, d, torch.zeros_like(d))
        trimmed.append(d)

    stacked = torch.stack(trimmed, dim=0)
    w_vec = torch.tensor(weights, dtype=torch.float32).view(-1, *([1] * deltas[0].ndim))

    # Sign election by weighted magnitude sum
    elected_sign = torch.sign((stacked * w_vec).sum(dim=0))
    agree_mask = (torch.sign(stacked) == elected_sign) & (stacked != 0)

    weighted_sum = (stacked * w_vec * agree_mask).sum(dim=0)
    weight_denom = (w_vec * agree_mask).sum(dim=0).clamp_min(1e-12)
    return torch.where(agree_mask.any(dim=0), weighted_sum / weight_denom, torch.zeros_like(deltas[0]))


def _mean_pairwise_cosine(deltas: List[torch.Tensor]) -> float:
    if len(deltas) < 2:
        return 1.0
    flats = [d.reshape(-1) for d in deltas]
    sims = []
    for i in range(len(flats)):
        ni = flats[i].norm().item()
        for j in range(i + 1, len(flats)):
            nj = flats[j].norm().item()
            if ni > 1e-12 and nj > 1e-12:
                sims.append(float(torch.dot(flats[i], flats[j]).item() / (ni * nj)))
    return sum(sims) / len(sims) if sims else 0.0


def _state_dict_sha256(state_dict: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        h.update(key.encode("utf-8"))
        t = state_dict[key].detach().cpu().contiguous()
        h.update(str(t.dtype).encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def merge_wsoup_fp32(
    base_state_dict: Dict[str, torch.Tensor],
    trajectories: Sequence[LoRATrajectory],
    mode: Literal["linear", "ties"] = "linear",
    ties_density: float = 0.5,
    output_dtype: Optional[torch.dtype] = None,
) -> Tuple[Dict[str, torch.Tensor], MergeReport]:
    """Merge N LoRA trajectories into base weights using strict FP32 accumulation.

    Reconstructs each trajectory's DeltaW_k = (alpha_k / r_k) * (B_k @ A_k) in
    float32 to avoid low-precision cancellation across trajectories, then casts
    back to each weight's original dtype (or output_dtype) only on write-out.
    """
    if not trajectories:
        raise ValueError("trajectories must contain at least one LoRATrajectory")

    total_w = sum(t.weight for t in trajectories)
    if total_w <= 0.0:
        raise ValueError("Sum of trajectory weights must be positive")
    norm_weights = [t.weight / total_w for t in trajectories]

    traj_pairs = [_pair_lora_keys(t.state_dict) for t in trajectories]
    merged: Dict[str, torch.Tensor] = {}

    layer_stats: Dict[str, Dict[str, float]] = {}
    all_cosines: List[float] = []
    sum_delta_sq = 0.0
    sum_base_sq = 0.0
    merged_layers = 0

    for key, base_tensor in base_state_dict.items():
        deltas: List[torch.Tensor] = []
        active_weights: List[float] = []

        for traj, pairs, w in zip(trajectories, traj_pairs, norm_weights):
            if key in pairs:
                a_key, b_key = pairs[key]
                a = traj.state_dict[a_key].detach().to(device="cpu", dtype=torch.float32)
                b = traj.state_dict[b_key].detach().to(device="cpu", dtype=torch.float32)
                scale = traj.alpha / float(traj.rank)
                deltas.append((b @ a) * scale)
                active_weights.append(w)

        if not deltas:
            merged[key] = base_tensor.detach().cpu().clone()
            continue

        # Re-normalize active weights if a subset of trajectories adapted this layer
        w_sum = sum(active_weights)
        active_weights = [w / w_sum for w in active_weights]

        if mode == "linear":
            combined_delta = torch.zeros_like(deltas[0])
            for d, w in zip(deltas, active_weights):
                combined_delta.add_(d, alpha=w)
        elif mode == "ties":
            combined_delta = _ties_merge_fp32(deltas, active_weights, density=ties_density)
        else:
            raise ValueError(f"Unsupported merge mode '{mode}'. Choose 'linear' or 'ties'.")

        base_fp32 = base_tensor.detach().to(device="cpu", dtype=torch.float32)
        updated_fp32 = base_fp32 + combined_delta

        target_dtype = output_dtype if output_dtype is not None else base_tensor.dtype
        merged[key] = updated_fp32.to(dtype=target_dtype)

        d_norm = float(combined_delta.norm().item())
        b_norm = float(base_fp32.norm().item())
        cos = _mean_pairwise_cosine(deltas)
        all_cosines.append(cos)
        sum_delta_sq += d_norm * d_norm
        sum_base_sq += b_norm * b_norm
        merged_layers += 1
        layer_stats[key] = {
            "delta_l2": d_norm,
            "relative_l2": d_norm / max(b_norm, 1e-12),
            "pairwise_cosine": cos,
        }

    global_delta_l2 = sum_delta_sq ** 0.5
    global_rel = global_delta_l2 / max(sum_base_sq ** 0.5, 1e-12)
    report = MergeReport(
        mode=mode,
        num_trajectories=len(trajectories),
        merged_layers=merged_layers,
        mean_pairwise_cosine=(sum(all_cosines) / len(all_cosines)) if all_cosines else 1.0,
        global_delta_l2=global_delta_l2,
        global_relative_perturbation=global_rel,
        merged_sha256=_state_dict_sha256(merged),
        layer_stats=layer_stats,
    )
    return merged, report
