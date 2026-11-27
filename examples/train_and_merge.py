"""Minimal runnable example: Anvil2 polar LoRA + Renyi-2 regularization + FP32 W_soup merge."""

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from anvil_renyi_lora import (
    Anvil2,
    LoRATrajectory,
    Renyi2RankRegularizer,
    inject_lora,
    merge_wsoup_fp32,
    split_anvil2_params,
)
from anvil_renyi_lora.lora import extract_lora_state_dict


class TinyTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * 4, bias=False)
        self.fc2 = nn.Linear(d_model * 4, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        h = self.ln1(x)
        q = self.q_proj(h).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o_proj(attn.transpose(1, 2).contiguous().view(b, t, c))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 128, d_model: int = 128, n_heads: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(64, d_model)
        self.blocks = nn.ModuleList([TinyTransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.ln_f(x))


def make_synthetic_char_batches(seed: int, n_seqs: int = 192, seq_len: int = 33) -> torch.Tensor:
    """Deterministic multi-pattern public byte-level corpus (no external download required)."""
    templates = [
        "def add(a, b): return a + b\n",
        "for i in range(8): acc += i * 2\n",
        "class Node: next_id = 42; active = True\n",
        "SELECT id, score FROM evals ORDER BY score DESC;\n",
    ]
    raw = ("".join(templates) * 40).encode("ascii")
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, len(raw) - seq_len - 1, (n_seqs,), generator=g)
    return torch.stack([torch.tensor(list(raw[s : s + seq_len]), dtype=torch.long) for s in starts])


@torch.no_grad()
def eval_loss(model: nn.Module, data: torch.Tensor, device: torch.device) -> float:
    model.eval()
    x = data[:, :-1].to(device)
    y = data[:, 1:].to(device)
    logits = model(x)
    return float(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())


def train_trajectory(
    base_state: dict,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    seed: int,
    rank: int,
    alpha: float,
    renyi_lambda: float,
    steps: int,
    device: torch.device,
) -> tuple[LoRATrajectory, float, dict]:
    torch.manual_seed(seed)
    model = TinyCausalLM().to(device)
    model.load_state_dict(copy.deepcopy(base_state))
    inject_lora(model, rank=rank, alpha=alpha)

    polar_params, _ = split_anvil2_params(model)
    opt = Anvil2(polar_params, lr=1.5e-2, momentum=0.9, ns_steps=3, weight_decay=0.01)
    reg = Renyi2RankRegularizer(lambda_weight=renyi_lambda, mode="log")

    model.train()
    last_metrics = {"mean_erank2": 0.0, "mean_utilization": 0.0}
    for step in range(steps):
        batch = train_data[(step * 24) % len(train_data) : ((step * 24) % len(train_data)) + 24].to(device)
        if batch.size(0) == 0:
            batch = train_data[:24].to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(batch[:, :-1])
        ce_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
        reg_loss, last_metrics = reg(model)
        (ce_loss + reg_loss).backward()
        opt.step()

    vloss = eval_loss(model, val_data, device)
    traj = LoRATrajectory(
        name=f"seed_{seed}",
        state_dict=extract_lora_state_dict(model),
        rank=rank,
        alpha=alpha,
        weight=1.0,
    )
    return traj, vloss, last_metrics


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    base_model = TinyCausalLM().to(device)
    base_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}

    train_data = make_synthetic_char_batches(seed=10, n_seqs=192)
    val_data = make_synthetic_char_batches(seed=99, n_seqs=64)
    base_val = eval_loss(base_model, val_data, device)
    print(f"Base model val loss: {base_val:.4f} (ppl={math.exp(base_val):.2f})")

    rank, alpha, steps = 16, 32.0, 40

    # Quick ablation: lambda=0.0 vs lambda=0.05 on single seed
    _, v_unreg, m_unreg = train_trajectory(base_state, train_data, val_data, 1, rank, alpha, 0.0, steps, device)
    _, v_renyi, m_renyi = train_trajectory(base_state, train_data, val_data, 1, rank, alpha, 0.05, steps, device)
    print(
        f"Ablation (r={rank}) | lambda=0.00: val={v_unreg:.4f}, erank2={m_unreg['mean_erank2']:.2f} "
        f"({m_unreg['mean_utilization']*100:.1f}% util)"
    )
    print(
        f"Ablation (r={rank}) | lambda=0.05: val={v_renyi:.4f}, erank2={m_renyi['mean_erank2']:.2f} "
        f"({m_renyi['mean_utilization']*100:.1f}% util)"
    )

    # 3-trajectory FP32 W_soup merge
    trajs = []
    for s in (11, 22, 33):
        tr, vl, m = train_trajectory(base_state, train_data, val_data, s, rank, alpha, 0.02, steps, device)
        trajs.append(tr)
        print(f"  Trajectory seed={s}: val={vl:.4f} | mean_erank2={m['mean_erank2']:.2f}/{rank}")

    merged_sd, report = merge_wsoup_fp32(base_state, trajs, mode="linear")
    merged_model = TinyCausalLM().to(device)
    merged_model.load_state_dict(merged_sd)
    soup_val = eval_loss(merged_model, val_data, device)

    print(
        f"FP32 W_soup (N={report.num_trajectories}) | val={soup_val:.4f} | "
        f"mean_cos={report.mean_pairwise_cosine:.3f} | rel_delta={report.global_relative_perturbation:.4f} | "
        f"sha256={report.merged_sha256[:12]}..."
    )


if __name__ == "__main__":
    main()
