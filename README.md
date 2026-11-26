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

## Layout

- `lean/` — Lean `v4.35.0` project depending on `slt-bridge` (`SLT`) and Mathlib, plus:
  - `lean/EatYourFiber/` — new SLT fiber-geometry certificates.
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

### PyTorch MALA LoRA fiber experiment

```sh
cd experiments
uv run python lora_fiber_sgld.py
```

## License & Attribution

Code and Lean proofs are licensed under the [Apache License 2.0](LICENSE). Vendored tensor and border-rank files in `lean/Vendor/OAI/` are adapted from OpenAI's [`openai/math`](https://github.com/openai/math) repository under Apache License 2.0; see [NOTICE](NOTICE) for full citation and commit provenance.
