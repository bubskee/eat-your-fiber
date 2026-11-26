# eat-your-fiber

**Singular learning theory on zero-loss fibers: Lean 4 certificates paired with real-network empiricism.**

Different parameter vectors computing the *exact same function* can sit in valleys with macroscopically different local learning coefficients $(\lambda, m)$. Starting from [residual-slt-in-a-haystack](https://github.com/bubskee/residual-slt-in-a-haystack) and building on [slt-bridge](https://github.com/bubskee/slt-bridge), this repository develops two tracks side by side—both aiming for large, macroscopic $\Delta\lambda$ gaps that can be **kernel-certified in Lean 4** and **measured live in PyTorch with SGLD**:

1. **Order-2 (LoRA / Reduced-Rank / Deep Linear):** Exact local learning coefficients across the zero-loss fiber $\{BA = M^*\}$, including the **LoRA fiber phase transition** (flat Aoyagi–Watanabe plateau for $q \le q_c$, followed by an exact quadratic rise $\frac{(q - q_c)^2}{2}$ for $q > q_c$, putting standard LoRA initialization $q = k$ at the top of the parabola).
2. **Order-3 (Bilinear MLPs / 3-Tensors):** Exact fiber geometry and border-rank singularities of trainable bilinear MLPs $x \mapsto W_{\text{out}}((W_1 x) \odot (W_2 x))$, leveraging vendored 3-tensor rank and geometric border-rank formalizations from [`openai/math`](https://github.com/openai/math).

## Epistemic labels

We keep claim statuses strictly separated across `notes/`, `lean/`, and `experiments/`:

- **`[LC]`** — Lean-certified (builds cleanly, audited via `#print axioms` using only `propext`, `Classical.choice`, `Quot.sound`, no `sorry`).
- **`[paper]`** — Proved on paper, not yet formalized in Lean.
- **`[empirical]`** — Numerical / PyTorch SGLD / SGD observation.
- **`[conjectural]`** — Heuristic or open conjecture.

## Current highlights

- **`[LC]` Order-3 Bilinear MLPs, Border-Rank Singularity & Stretched-Exponential SLT (`lean/EatYourFiber/BorderRankSLT.lean`):**
  - Bridges width-$k$ bilinear MLPs $T(A, B, C) = \sum_{r=1}^k a_r \otimes b_r \otimes c_r$ directly to OpenAI's vendored `Tensor`, `tensorRank`, and `geometricBorderRank` (`exists_zero_loss_iff_tensorRank_le`, `geometricBorderRank_le_of_approx_loss`).
  - Certifies the exact border-rank gap of the canonical $W$-tensor $W = e_0\otimes e_0\otimes e_1 + e_0\otimes e_1\otimes e_0 + e_1\otimes e_0\otimes e_0$ over every field $F$ (`wTensor_geometricBorderRank`: `= 2` via OpenAI's `card_le_geometricBorderRank_of_minor`; `wTensor_tensorRank`: `= 3` via 2×2 slice identities).
  - Proves the sharp quantitative escape law $\inf_{\|\theta\|_2 \le R} \|T(\theta) - W\|_F^2 \asymp R^{-12}$ (`wTensor_loss_ge_of_paramNormSq`, `wTensor_escape_upper`) achieved by the centered finite-difference curve $\tilde W_\varepsilon = W + \varepsilon^2 e_1^{\otimes 3}$, the exact weight-decay law $\inf_\theta (\|T(\theta)-W\|_F^2 + \gamma \|\theta\|_2^2) \asymp \gamma^{6/7}$ (`wTensor_weightDecay_lower`, `wTensor_weightDecay_upper`), and **stretched-exponential Bayesian free energy** $-\log Z_W^{(2)}(N) \asymp N^{1/7}$ (`wPartitionZ_two_le_stretchedExp`, `wPartitionZ_two_ge_stretchedExp`) contrasted against **width-3 realizable polynomial decay** $-\log Z_W^{(3)}(N) \le 9\log N + C_3$ (`wPartitionZ_three_ge_poly`).
- **`[empirical]` Order-3 border-rank scaling, SMC+HMC free energy & quadratic MLP (`experiments/bilinear_border_rank.py`, `notes/bilinear_border_rank.md`):**
  - Multi-start L-BFGS spontaneously discovers the centered curve $\tilde W_\varepsilon$ (`e111` residual fraction `1.0000`, `obj/pred = 1.0028`, virial ratio `6.05` vs `6.00`, loss slope `0.851` vs `6/7 = 0.857`, norm² slope `-0.139` vs `-1/7 = -0.143`), plus a sharp width-3 first-order branch transition from the border-rank branch at high $\gamma$ to the rank-3 fiber at low $\gamma$.
  - SMC + HMC across $\beta \in [0.5, 10^5]$ shows $\hat\lambda(\beta) = \beta \mathbb{E}_\beta[L]$ plateauing flat at $\lambda = 4.03 \pm 0.08$ for $k=3$, while **diverging as $\beta^{1/7}$** (slope `0.154` vs `0.143`, reaching `8.65 ± 0.14`) for $k=2$, giving width 3 a `10.74`-nat lower Bayesian free energy despite 6 more parameters.
- **`[LC]` Full Aoyagi–Watanabe (2005) Theorem 1 & Complete Unconditional LoRA Fiber Phase Transition (`lean/EatYourFiber/AoyagiWatanabe.lean`):**
  - Elementary Lean 4 proof of **all four regimes** of Aoyagi–Watanabe (2005) Theorem 1 (`aoyagi_watanabe_rank0`: $m + n \le k$, $k + n \le m$, $k + m \le n$, and the interior triangle for both even parity $m_{\log} = 1$ and odd parity $m_{\log} = 2$) for all $m, k, n \in \mathbb{N}$ via a single structural induction on $(m, k)$ using projective blowup of $B$ at its max-magnitude entry, two unit-Jacobian affine translations, and `SLT.NC.stepG` (zero Hironaka, zero SVD/Weyl Jacobian, zero polar coordinates).
  - Combined with the Schur-complement Laplace reduction (`lora_schur_fiber_reduction`), unconditionally certifies the **entire LoRA zero-loss fiber phase transition** for all $m, n, k, q \in \mathbb{N}$:
    - `lora_fiber_exact_pair`: exact pair $(\frac{mq}{2} + \lambda_{\mathrm{AW}}(m, n-q, k-q),\; m_{\mathrm{AW}}(m, n-q, k-q))$ for all $q \le \min(k, n)$.
    - `lora_subcritical_plateau_exact_pair`: **flat subcritical plateau** $(\lambda_{\mathrm{AW}}(m, n, k), m_{\mathrm{AW}}(m, n, k))$ for all interior $(m, n, k)$ and $m + 2q < n + k$.
    - `lora_supercritical_parabola_exact_pair` (also independently proved in `AoyagiWatanabeBoundary.lean` via orthogonal projection + AM–GM cube bound): **supercritical parabola** $(\lambda_{\mathrm{AW}}^{\mathrm{even}}(m, n, k) + \frac{1}{2}(q - q_c)^2, 1)$ for all $n + k \le m + 2q$.
- **`[LC]` Self-Attention LoRA Coupling Dichotomy (`lean/EatYourFiber/AttentionLoRA.lean`):**
  - Certifies to **all orders** (via exact nonlinear left-multiplication diffeomorphisms `attnVO_factor`, `attnQK_factor` and an exact two-sided Schur complement reduction `twoSided_schur_reduction`) the architectural dichotomy between joint `V+O` and joint `Q+K` LoRA on single-head self-attention:
    - **Joint `V+O`** (`attention_VO_exact_pair`, `attention_VO_subcritical_plateau`, `attention_VO_supercritical_parabola`): flat plateau $(dH - H^2/2, 1)$ for $2q \le H$, followed by a **$4\times$-amplified supercritical parabola** $(dH - H^2/2 + 2(q - H/2)^2, 1)$ for $H \le 2q \le d$ (jump $+H^2/2$ at standard LoRA init $q = H$).
    - **Joint `Q+K`** (`attention_QK_gauge_orbit`, `attention_QK_exact_pair`, `attention_QK_flat_pair`): exact all-orders $GL(q)$ zero-loss gauge symmetry **completely annihilates the supercritical parabola**, giving identically flat $(dH - H^2/2, 1)$ for **all** $q \le H \le d$ (`attention_coupling_dichotomy_standard_init`).
- **`[empirical]` MALA phase-transition curve (`experiments/lora_fiber_sgld.py`, `notes/lora_fiber_phase_transition.md`):**
  - Reproduces both the flat plateau ($q \le q_c$) and the exact $\frac{1}{2}(q - q_c)^2$ parabola ($q > q_c$) on `rank0_8x8x8` and `lora_16x16_r2_H10`.
- **`[empirical]` Nonlinear MLP & Self-Attention LoRA fiber dynamics (`experiments/lora_nonlinear_fiber.py`, `notes/lora_nonlinear_fiber.md`):**
  - **MLP invariance & dead-neuron shift:** GELU, ReLU, and SwiGLU 2-layer MLPs reproduce the exact linear phase transition $\lambda(q) = 24 + \frac{1}{2}(q - 4)_+^2$ within $1\sigma$ (`q=8: 31.97–32.01 ± 0.07`); killing 2 ReLU neurons ($m_{\mathrm{eff}} = 6$) shifts $q_c: 4 \to 5$, plateau $24.0 \to 19.5$, and top $32.0 \to 24.0$ (`23.99 ± 0.04`).
  - **Attention $V$-$O$ vs. $Q$-$K$ coupling dichotomy (`[LC]` single-head + `[empirical]` + `[paper]` multi-head):** Single-head joint `V+O` matches the certified $4\times$-amplified parabola ($\lambda_{V+O}(q) = 24 + 2(q-2)_+^2$, MALA `q=4: 32.00 ± 0.06`), while joint `Q+K` matches the certified flat $GL(q)$ gauge ceiling ($\lambda_{Q+K}(q) \equiv 24.0$, MALA `q=4: 23.99 ± 0.05`). In **2-head attention** ($d_h + q \le d$), cross-head transversality breaks the $GL(q)$ cancellation and restores the full supercritical top (`32.10 ± 0.14`).
  - **Training & honest noisy-data nuance:** On realizable GELU MLP and full $Q,K,V,O$ Attention tasks, Adam/SGD without WD stay supercritical while AdamW and SGD+WD descend onto the subcritical plateau. On low-sample noisy fine-tuning ($S=48, \sigma=0.25$), noise excites all 8 LoRA ranks (destroying the inactive fiber); moderate decoupled AdamW shrinks $\|A_{\perp B}\|_F$ without improving test MSE, whereas strong WD, coupled SGD+WD, or subcritical rank-$2$ init cut test MSE by $2\text{--}8\times$.
- **`[empirical]` Real SGD / Adam / AdamW dynamics on the LoRA fiber (`experiments/lora_sgd_dynamics.py`, `notes/lora_sgd_dynamics.md`):**
  - Without weight decay, full-batch GD/Adam and mini-batch SGD/Adam from standard LoRA init ($B = 0, A \sim \mathcal{N}$) remain pinned at the supercritical top $q = k = 8$ ($\hat\lambda \approx 85.8\text{--}86.0$, $+8$ above $\lambda_{\mathrm{AW}} = 78$).
  - Adding weight decay (`AdamW` or `SGD+WD`) contracts $A_{\mathrm{in}}$ below $q_c = 4$, sliding $\hat\lambda(t)$ down the parabola onto the global-minimum plateau ($\hat\lambda \approx 75.7\text{--}75.9$, matching the $q = 0$ MALA baseline).

## Layout

- `lean/` — Lean `v4.35.0` project depending on `slt-bridge` (`SLT`) and Mathlib, plus:
  - `lean/EatYourFiber/` — new SLT fiber-geometry certificates (`LoRASchur`, `LoRASchurLaplace`, `AoyagiWatanabeBoundary`, `AoyagiWatanabe`, `AttentionLoRA`, `BorderRankSLT`).
  - `lean/Vendor/OAI/` — 8 self-contained Apache-2.0 tensor / border-rank / block-rank modules vendored from [`openai/math`](https://github.com/openai/math) (commit `3344da59991ed94f9ba711217b29c1d760badfa6`). See [NOTICE](NOTICE).
- `experiments/` — reproducible Python / PyTorch SGLD and SGD scripts (`uv`-managed).
- `notes/` — mathematical derivations and experiment logs.

## Reproduce

### Lean 4 build & axiom audit

```sh
cd lean
lake exe cache get
lake build
lake env lean -DwarningAsError=true -DautoImplicit=false Audit.lean
```

### PyTorch experiments (Track 1 & Track 2)

```sh
cd experiments
uv run python bilinear_border_rank.py
uv run python lora_fiber_sgld.py
uv run python lora_sgd_dynamics.py
uv run python lora_nonlinear_fiber.py
```

## License & Attribution

Code and Lean proofs are licensed under the [Apache License 2.0](LICENSE). Vendored tensor and border-rank files in `lean/Vendor/OAI/` are adapted from OpenAI's [`openai/math`](https://github.com/openai/math) repository under Apache License 2.0; see [NOTICE](NOTICE) for full citation and commit provenance.
