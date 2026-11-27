"""Minimal LoRALinear module with clean state_dict export for W_soup merging."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with trainable low-rank factors (A, B)."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 32,
        alpha: float = 64.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / float(rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.lora_a = nn.Parameter(
            torch.empty(rank, self.in_features, device=base.weight.device, dtype=base.weight.dtype)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(self.out_features, rank, device=base.weight.device, dtype=base.weight.dtype)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b)
        return out + delta * self.scaling


def inject_lora(
    model: nn.Module,
    target_suffixes: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"),
    rank: int = 32,
    alpha: float = 64.0,
    dropout: float = 0.0,
) -> int:
    """Replace matching nn.Linear modules in-place with LoRALinear wrappers."""
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(sfx) for sfx in target_suffixes):
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        replaced += 1
    return replaced


def extract_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Export only LoRA adapter tensors (detached CPU clones)."""
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if k.endswith(".lora_a") or k.endswith(".lora_b")
    }
