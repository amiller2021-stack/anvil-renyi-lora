"""Anvil2: 3-step Newton-Schulz polar optimizer for 2D weight and LoRA matrices."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import torch
from torch.optim import Optimizer

# Quintic Newton-Schulz polynomial coefficients (a*X + b*(X X^T)X + c*(X X^T)^2 X)
DEFAULT_NS_COEFFS: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def newton_schulz_polar(
    grad: torch.Tensor,
    steps: int = 3,
    coeffs: Tuple[float, float, float] | Sequence[Tuple[float, float, float]] = DEFAULT_NS_COEFFS,
    eps: float = 1e-7,
    compute_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Approximate the orthogonal polar factor U V^T of a 2D matrix without SVD.

    Always forms the Gram product X @ X.T along the smaller dimension min(m, n),
    which makes orthogonalizing low-rank LoRA factors (r << d) O(r^2 * d).
    """
    if grad.ndim != 2:
        raise ValueError(f"newton_schulz_polar expects a 2D tensor, got shape {tuple(grad.shape)}")

    orig_dtype = grad.dtype
    if compute_dtype is None:
        if grad.is_cuda and torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float32

    x = grad.to(dtype=compute_dtype)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.transpose(-2, -1).contiguous()

    # Frobenius normalization bounds singular values to (0, 1] before polynomial iteration
    x = x / (x.norm() + eps)

    if isinstance(coeffs[0], (tuple, list)):
        schedule = list(coeffs)
        if len(schedule) != steps:
            raise ValueError(f"Expected {steps} coefficient tuples, got {len(schedule)}")
    else:
        schedule = [coeffs] * steps  # type: ignore[list-item]

    for a, b, c in schedule:
        gram = x @ x.transpose(-2, -1)
        poly = b * gram + c * (gram @ gram)
        x = a * x + poly @ x

    if transposed:
        x = x.transpose(-2, -1).contiguous()

    return x.to(dtype=orig_dtype)


class Anvil2(Optimizer):
    """Polar orthogonalized momentum optimizer for 2D hidden and LoRA matrices.

    1D parameters (biases, LayerNorm/RMSNorm scales) and embeddings should be
    routed to a companion AdamW optimizer using ``split_anvil2_params``.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 2e-2,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 3,
        ns_coeffs: Tuple[float, float, float] | Sequence[Tuple[float, float, float]] = DEFAULT_NS_COEFFS,
        eps: float = 1e-7,
        rms_scale: bool = True,
        compute_dtype: Optional[torch.dtype] = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if ns_steps < 1:
            raise ValueError(f"ns_steps must be >= 1, got {ns_steps}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            ns_coeffs=ns_coeffs,
            eps=eps,
            rms_scale=rms_scale,
            compute_dtype=compute_dtype,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            mu: float = group["momentum"]
            nesterov: bool = group["nesterov"]
            wd: float = group["weight_decay"]
            ns_steps: int = group["ns_steps"]
            ns_coeffs = group["ns_coeffs"]
            eps: float = group["eps"]
            rms_scale: bool = group["rms_scale"]
            compute_dtype: Optional[torch.dtype] = group["compute_dtype"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError(
                        f"Anvil2 only optimizes 2D matrices, received shape {tuple(p.shape)}. "
                        "Use split_anvil2_params() to route 1D/embedding params to AdamW."
                    )

                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]

                buf.mul_(mu).add_(g)
                update_in = g.add(buf, alpha=mu) if nesterov else buf

                ortho = newton_schulz_polar(
                    update_in,
                    steps=ns_steps,
                    coeffs=ns_coeffs,
                    eps=eps,
                    compute_dtype=compute_dtype,
                )

                scale = math.sqrt(max(1.0, p.shape[0] / p.shape[1])) if rms_scale else 1.0
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(ortho, alpha=-lr * scale)

        return loss


def split_anvil2_params(
    model: torch.nn.Module,
    exclude_substrings: Tuple[str, ...] = ("embed", "lm_head", "wte", "wpe"),
) -> Tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split trainable parameters into (polar_2d_params, adamw_aux_params)."""
    polar_params: list[torch.nn.Parameter] = []
    aux_params: list[torch.nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        low = name.lower()
        if p.ndim == 2 and not any(tok in low for tok in exclude_substrings):
            polar_params.append(p)
        else:
            aux_params.append(p)

    return polar_params, aux_params
