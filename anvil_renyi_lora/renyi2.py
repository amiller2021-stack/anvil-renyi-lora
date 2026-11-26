"""SVD-free Renyi-2 effective-rank regularizer for LoRA adapters.

For a LoRA update DeltaW = B @ A with A in R^{r x d_in} and B in R^{d_out x r}:
  G_A = A @ A.T   (r x r)
  G_B = B.T @ B   (r x r)
  M   = G_B @ G_A (r x r)

The non-zero eigenvalues of M equal the squared singular values {sigma_i^2} of DeltaW:
  tr(M)   = sum_i sigma_i^2
  tr(M^2) = sum_i sigma_i^4 = sum_{i,j} M_{ij} M_{ji}
  erank_2(DeltaW) = tr(M)^2 / tr(M^2) in [1, r]

Complexity: O(r^2 * (d_in + d_out) + r^3), zero SVD calls, fully differentiable.
Reference: Sawada & Goyal (2026), arXiv:2611.17191.
"""

from __future__ import annotations

import math
from typing import Dict, Iterator, Literal, Optional, Tuple

import torch
import torch.nn as nn


def renyi2_effective_rank(
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    eps: float = 1e-12,
    mode: Literal["log", "normalized"] = "log",
    shift_to_nonnegative: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """Compute exact Renyi-2 effective rank and penalty from LoRA factors (A, B).

    Args:
        lora_a: Down-projection weight A of shape (r, d_in).
        lora_b: Up-projection weight B of shape (d_out, r).
        eps: Numerical stabilizer for division and log.
        mode:
            - "log": -log(erank_2(DeltaW)) = log(tr(M^2)) - 2*log(tr(M)), optionally
              shifted by +log(r) so the penalty lies in [0, log(r)].
            - "normalized": 1 - erank_2(DeltaW) / r in [0, 1).
        shift_to_nonnegative: Adds +log(r) in "log" mode (identical gradient).

    Returns:
        (erank2, penalty, is_active). When B is zero-initialized (tr(M) <= eps),
        returns (0.0, 0.0, False) cleanly without NaN gradients.
    """
    if lora_a.ndim != 2 or lora_b.ndim != 2:
        raise ValueError(
            f"Expected 2D LoRA factors, got A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}"
        )
    if lora_a.shape[0] != lora_b.shape[1]:
        raise ValueError(
            f"Rank mismatch between A={tuple(lora_a.shape)} and B={tuple(lora_b.shape)}"
        )

    rank = lora_a.shape[0]
    a = lora_a.float()
    b = lora_b.float()

    # r x r symmetric Gram matrices
    g_a = a @ a.transpose(0, 1)
    g_b = b.transpose(0, 1) @ b

    # tr(G_B G_A) = <G_B, G_A>_F because G_A and G_B are symmetric
    tr_m = (g_b * g_a).sum()
    if tr_m.detach().item() <= eps:
        zero = torch.zeros((), device=lora_a.device, dtype=torch.float32)
        return zero, zero, False

    # M = G_B @ G_A is generally non-symmetric, so tr(M^2) = sum_{i,j} M_{ij} M_{ji}
    m = g_b @ g_a
    tr_m2 = (m * m.transpose(0, 1)).sum()

    erank2 = (tr_m * tr_m) / (tr_m2 + eps)

    if mode == "log":
        penalty = torch.log(tr_m2 + eps) - 2.0 * torch.log(tr_m + eps)
        if shift_to_nonnegative:
            penalty = penalty + math.log(float(rank))
    elif mode == "normalized":
        penalty = 1.0 - (erank2 / float(rank))
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Choose 'log' or 'normalized'.")

    return erank2, penalty, True


def iter_lora_factor_pairs(
    model: nn.Module,
    adapter_name: Optional[str] = None,
) -> Iterator[Tuple[str, torch.Tensor, torch.Tensor]]:
    """Yield (module_name, lora_a_weight, lora_b_weight) for custom or PEFT LoRA layers."""
    for name, module in model.named_modules():
        # Native LoRALinear convention
        if hasattr(module, "lora_a") and hasattr(module, "lora_b"):
            a_w = module.lora_a.weight if isinstance(module.lora_a, nn.Module) else module.lora_a
            b_w = module.lora_b.weight if isinstance(module.lora_b, nn.Module) else module.lora_b
            if isinstance(a_w, torch.Tensor) and isinstance(b_w, torch.Tensor):
                yield name, a_w, b_w
            continue

        # Hugging Face PEFT convention (ModuleDict keyed by adapter name)
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            la, lb = module.lora_A, module.lora_B
            if isinstance(la, nn.ModuleDict) and isinstance(lb, nn.ModuleDict):
                keys = [adapter_name] if adapter_name is not None else list(la.keys())
                for k in keys:
                    if k in la and k in lb:
                        yield f"{name}.{k}", la[k].weight, lb[k].weight
            elif hasattr(la, "weight") and hasattr(lb, "weight"):
                yield name, la.weight, lb.weight


class Renyi2RankRegularizer(nn.Module):
    """Drop-in loss regularizer encouraging full-rank capacity utilization across LoRA layers."""

    def __init__(
        self,
        lambda_weight: float = 0.01,
        mode: Literal["log", "normalized"] = "log",
        shift_to_nonnegative: bool = True,
        eps: float = 1e-12,
        adapter_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.lambda_weight = lambda_weight
        self.mode = mode
        self.shift_to_nonnegative = shift_to_nonnegative
        self.eps = eps
        self.adapter_name = adapter_name

    def forward(self, model: nn.Module) -> Tuple[torch.Tensor, Dict[str, float]]:
        penalties = []
        eranks = []
        utils = []
        device = next(model.parameters()).device

        for _name, a_w, b_w in iter_lora_factor_pairs(model, adapter_name=self.adapter_name):
            if not (a_w.requires_grad or b_w.requires_grad):
                continue
            erank2, pen, active = renyi2_effective_rank(
                a_w,
                b_w,
                eps=self.eps,
                mode=self.mode,
                shift_to_nonnegative=self.shift_to_nonnegative,
            )
            if active:
                penalties.append(pen)
                er_val = float(erank2.detach().item())
                eranks.append(er_val)
                utils.append(er_val / float(a_w.shape[0]))

        if not penalties:
            zero = torch.zeros((), device=device, dtype=torch.float32)
            return zero, {"active_layers": 0.0, "mean_erank2": 0.0, "mean_utilization": 0.0, "raw_penalty": 0.0}

        mean_penalty = torch.stack(penalties).mean()
        weighted_loss = self.lambda_weight * mean_penalty
        metrics = {
            "active_layers": float(len(penalties)),
            "mean_erank2": sum(eranks) / len(eranks),
            "mean_utilization": sum(utils) / len(utils),
            "raw_penalty": float(mean_penalty.detach().item()),
        }
        return weighted_loss, metrics
