# anvil-renyi-lora

Standalone PyTorch implementations of three post-training primitives for small-model LoRA adaptation:

1. **Anvil2 Polar Optimizer (`anvil2.py`)**: 3-step quintic Newton–Schulz polar orthogonalization applied along the smaller LoRA dimension ($O(r^2 d)$) with Nesterov momentum, aspect-ratio RMS scaling, and decoupled weight decay.
2. **SVD-Free Rényi-2 Effective-Rank Regularizer (`renyi2.py`)**: Differentiable capacity-utilization penalty computed strictly from $r \times r$ Gram matrices $G_A = A A^\top$ and $G_B = B^\top B$ (Sawada & Goyal, 2026), preventing LoRA rank collapse without materializing $\Delta W$ or running an SVD.
3. **FP32 `W_soup` Multi-Trajectory Merger (`wsoup.py`)**: Strict `float32` reconstruction and linear / TIES-merging of multi-trajectory LoRA adapters with pairwise trajectory cosine diagnostics and SHA-256 weight manifests.

## Mathematical Summary

### 1. 3-Step Polar Update (`Anvil2`)
Given momentum matrix $M_t \in \mathbb{R}^{m \times n}$, `newton_schulz_polar` transposes if $m > n$ so the inner Gram product is $\min(m, n) \times \min(m, n)$, normalizes $X_0 = M_t / (\|M_t\|_F + \varepsilon)$, and runs 3 quintic steps:

$$X_{k+1} = a X_k + \big(b (X_k X_k^\top) + c (X_k X_k^\top)^2\big) X_k$$

with default coefficients $(a, b, c) = (3.4445, -4.7750, 2.0315)$ and aspect-ratio step scaling $\sqrt{\max(1, m/n)}$.

### 2. SVD-Free Rényi-2 Effective-Rank Penalty
For $\Delta W = B A$ ($A \in \mathbb{R}^{r \times d_{\text{in}}}$, $B \in \mathbb{R}^{d_{\text{out}} \times r}$), the eigenvalues of $M = G_B G_A = (B^\top B)(A A^\top) \in \mathbb{R}^{r \times r}$ match the non-zero squared singular values $\{\sigma_i^2\}$ of $\Delta W$:

$$\mathrm{erank}_2(\Delta W) = \frac{\big(\mathrm{tr}(M)\big)^2}{\mathrm{tr}(M^2)} \in [1, r], \qquad \mathcal{L}_{\text{Rényi-2}} = \log r + \log\!\big(\mathrm{tr}(M^2) + \varepsilon\big) - 2\log\!\big(\mathrm{tr}(M) + \varepsilon\big)$$

Because $M = G_B G_A$ is generally non-symmetric, $\mathrm{tr}(M^2)$ is computed via $\sum_{i,j} M_{ij} M_{ji}$ (`(M * M.T).sum()`). Zero-initialized $B$ layers ($\mathrm{tr}(M) \le \varepsilon$) are masked automatically on step 0.

### 3. FP32 Weight-Soup Merging (`W_soup`)
Each trajectory's update $\Delta W_k = \frac{\alpha_k}{r_k} B_k A_k$ is reconstructed and averaged in `torch.float32` before casting back to the checkpoint dtype, eliminating low-precision cancellation across trajectories.

## Completed Training & Merge Metrics (1.7B Backbone)

The Anvil2 training loop and FP32 `W_soup` merger were validated on a 1.7B dense Transformer backbone trained on a 200M-token multi-domain corpus and evaluated on a 2,000-prompt blind multi-domain suite (zero runtime errors across 6,000 total model completions):

| Stage / Configuration | Rank & Steps | Deterministic Contract Avg (2,000 prompts) | Blind Preference Wins (500-prompt A/B/C) | Blind Mean Score (0–10) |
|---|---|---:|---:|---:|
| Un-tuned 1.5B Baseline | — | 26.05% | 69 / 500 (13.8%) | 2.22 |
| Un-tuned 2.0B Hybrid Baseline | — | 36.73% | 154 / 500 (30.8%) | 3.04 |
| **1.7B + Anvil2 Bootstrap (`r=64`, 2,500 steps) → 3-Trajectory FP32 `W_soup v3` (10,200 cumulative steps)** | `r=64` base / `r=32` adapters | **35.82%** (97.5% cross-domain isolation, 61.0% code/diff structure) | **275 / 500 (55.0%)** | **4.73** |

**Scope note on `renyi2.py`:** The Gram-matrix Rényi-2 regularizer is mathematically verified against exact `torch.linalg.svdvals` to `< 1e-4` error in `tests/test_math.py` and benchmarked in `examples/train_and_merge.py`. Full-scale 1.7B/2.0B multi-adapter sweeps with Rényi-2 enabled are planned as the next experimental iteration and are not included in the table above.

## Quick Start

```bash
pip install -e ".[dev]"
pytest -v
python examples/train_and_merge.py
