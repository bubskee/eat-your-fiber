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

- **`[LC]` Unconditional supercritical LoRA fiber parabola (`lean/EatYourFiber/AoyagiWatanabeBoundary.lean`):**
  - Elementary Lean 4 proofs of Aoyagi–Watanabe (2005) boundary regimes $k + n \le m \implies (\frac{kn}{2}, 1)$ (`rank0_hasLaplacePair_of_le_m`) and $k + m \le n \implies (\frac{km}{2}, 1)$ (`rank0_hasLaplacePair_of_le_n`) via inductive orthogonal-projection Gaussian integration and an AM–GM cube bound (zero resolution of singularities, zero polar coordinates).
  - Combined with the Schur-complement Laplace reduction (`lora_schur_fiber_reduction`), yields `lora_supercritical_parabola_exact_pair`: exact local learning pair $(\lambda_{\mathrm{AW}}^{\mathrm{even}}(m, n, k) + \frac{1}{2}(q - q_c)^2, 1)$ unconditionally for all $m, n, k, q$ with $q \le k, q \le n, k < m + n, n + k \le m + 2q$.
- **`[empirical]` MALA phase-transition curve (`experiments/lora_fiber_sgld.py`, `notes/lora_fiber_phase_transition.md`):**
  - Reproduces both the flat plateau ($q \le q_c$) and the exact $\frac{1}{2}(q - q_c)^2$ parabola ($q > q_c$) on `rank0_8x8x8` and `lora_16x16_r2_H10`.
- **`[empirical]` Real SGD / Adam / AdamW dynamics on the LoRA fiber (`experiments/lora_sgd_dynamics.py`, `notes/lora_sgd_dynamics.md`):**
  - Without weight decay, full-batch GD/Adam and mini-batch SGD/Adam from standard LoRA init ($B = 0, A \sim \mathcal{N}$) remain pinned at the supercritical top $q = k = 8$ ($\hat\lambda \approx 85.8\text{--}86.0$, $+8$ above $\lambda_{\mathrm{AW}} = 78$).
  - Adding weight decay (`AdamW` or `SGD+WD`) contracts $A_{\mathrm{in}}$ below $q_c = 4$, sliding $\hat\lambda(t)$ down the parabola onto the global-minimum plateau ($\hat\lambda \approx 75.7\text{--}75.9$, matching the $q = 0$ MALA baseline).

## Layout

- `lean/` — Lean `v4.35.0` project depending on `slt-bridge` (`SLT`) and Mathlib, plus:
  - `lean/EatYourFiber/` — new SLT fiber-geometry certificates (`LoRASchur`, `LoRASchurLaplace`, `AoyagiWatanabeBoundary`).
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

### PyTorch MALA & SGD/Adam LoRA fiber experiments

```sh
cd experiments
uv run python lora_fiber_sgld.py
uv run python lora_sgd_dynamics.py
```

## License & Attribution

Code and Lean proofs are licensed under the [Apache License 2.0](LICENSE). Vendored tensor and border-rank files in `lean/Vendor/OAI/` are adapted from OpenAI's [`openai/math`](https://github.com/openai/math) repository under Apache License 2.0; see [NOTICE](NOTICE) for full citation and commit provenance.
