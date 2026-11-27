"""Mathematical verification suite for Anvil2, Renyi-2 Gram regularizer, and FP32 W_soup."""

import math
import torch

from anvil_renyi_lora import (
    Anvil2,
    LoRATrajectory,
    Renyi2RankRegularizer,
    inject_lora,
    merge_wsoup_fp32,
    newton_schulz_polar,
)
from anvil_renyi_lora.renyi2 import renyi2_effective_rank


def test_newton_schulz_polar_compresses_spectrum_both_orientations():
    torch.manual_seed(0)
    for shape in [(16, 128), (128, 16)]:
        x = torch.randn(*shape, dtype=torch.float32)
        ortho = newton_schulz_polar(x, steps=5, compute_dtype=torch.float32)
        assert ortho.shape == shape
        svals = torch.linalg.svdvals(ortho)
        # Quintic polynomial maps singular values tightly around 1.0
        assert svals.min().item() > 0.70
        assert svals.max().item() < 1.30


def test_anvil2_optimizer_step_updates_2d_params():
    torch.manual_seed(1)
    p = torch.nn.Parameter(torch.randn(8, 32))
    p_before = p.detach().clone()
    opt = Anvil2([p], lr=1e-2, momentum=0.9, ns_steps=3)
    (p.pow(2).sum()).backward()
    opt.step()
    assert not torch.allclose(p, p_before)
    assert torch.isfinite(p).all()


def test_renyi2_gram_matches_exact_svd_and_backprops():
    torch.manual_seed(42)
    r, d_in, d_out = 16, 96, 64
    a = torch.randn(r, d_in, requires_grad=True)
    # Create an intentionally skewed spectrum so G_A and G_B do not commute
    scale = torch.linspace(0.1, 3.0, r)
    b = (torch.randn(d_out, r) * scale).detach().requires_grad_(True)

    erank2, penalty, active = renyi2_effective_rank(a, b, mode="log", shift_to_nonnegative=True)
    assert active

    # Reference via dense SVD of DeltaW = B @ A
    svals = torch.linalg.svdvals((b @ a).detach().float())
    s2 = svals.pow(2)
    expected_erank2 = (s2.sum().pow(2) / s2.pow(2).sum()).item()

    assert abs(erank2.item() - expected_erank2) < 1e-4
    assert penalty.item() >= -1e-6
    assert abs(penalty.item() - (math.log(r) - math.log(expected_erank2))) < 1e-4

    penalty.backward()
    assert a.grad is not None and torch.isfinite(a.grad).all() and a.grad.norm() > 0
    assert b.grad is not None and torch.isfinite(b.grad).all() and b.grad.norm() > 0


def test_renyi2_zero_init_safety():
    a = torch.randn(8, 32, requires_grad=True)
    b = torch.zeros(32, 8, requires_grad=True)
    erank2, penalty, active = renyi2_effective_rank(a, b)
    assert not active
    assert erank2.item() == 0.0
    assert penalty.item() == 0.0


def test_renyi2_regularizer_discovers_lora_layers():
    model = torch.nn.Sequential()
    model.add_module("q_proj", torch.nn.Linear(32, 32, bias=False))
    inject_lora(model, target_suffixes=("q_proj",), rank=8, alpha=16.0)
    with torch.no_grad():
        model.q_proj.lora_b.normal_(std=0.05)

    reg = Renyi2RankRegularizer(lambda_weight=0.1)
    loss, stats = reg(model)
    assert stats["active_layers"] == 1.0
    assert 1.0 <= stats["mean_erank2"] <= 8.0
    loss.backward()
    assert model.q_proj.lora_a.grad is not None
    assert model.q_proj.lora_b.grad is not None


def test_wsoup_fp32_linear_and_ties_merge():
    torch.manual_seed(7)
    base = {"layer.weight": torch.randn(16, 16, dtype=torch.bfloat16)}
    a1, b1 = torch.randn(4, 16), torch.randn(16, 4)
    a2, b2 = torch.randn(4, 16), torch.randn(16, 4)

    t1 = LoRATrajectory("t1", {"layer.lora_a": a1, "layer.lora_b": b1}, rank=4, alpha=8.0, weight=1.0)
    t2 = LoRATrajectory("t2", {"layer.lora_a": a2, "layer.lora_b": b2}, rank=4, alpha=8.0, weight=3.0)

    merged, rep = merge_wsoup_fp32(base, [t1, t2], mode="linear", output_dtype=torch.float32)
    expected = base["layer.weight"].float() + 0.25 * (2.0 * (b1 @ a1)) + 0.75 * (2.0 * (b2 @ a2))

    assert rep.merged_layers == 1
    assert torch.allclose(merged["layer.weight"], expected, atol=1e-5)

    # Verify deterministic manifest hash and TIES execution
    merged_ties, rep_ties = merge_wsoup_fp32(base, [t1, t2], mode="ties", ties_density=0.5)
    assert merged_ties["layer.weight"].dtype == torch.bfloat16
    assert len(rep_ties.merged_sha256) == 64
