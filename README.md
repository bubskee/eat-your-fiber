# eat-your-fiber

Lean 4 proofs and PyTorch experiments on one question from singular learning theory (SLT): **if two parameter vectors compute exactly the same function, do they have the same local learning coefficient (LLC)?**

Often they don't. The set of parameters that reach zero loss (the *zero-loss fiber*) is usually not a single point, and the LLC can change a lot as you move along it. This repo works that out in small models where it can be proved, then checks it in models people actually train.

> **Status:** early preprint, exploratory. Proofs, code and write-ups are made largely with AI agents, directed by Bobby Faber. `[LC]` results are checked by the Lean kernel. The `[paper]` arguments, experiment code and write-ups are not machine-checked; read them as drafts. Corrections are welcome.

## Why

- Interpretability and DevInterp work read LLC curves as a measure of model complexity, and read changes in them as developmental stages.
- The LLC belongs to a point in parameter space, not to a function. So it is worth knowing how much of an LLC change is the function changing, and how much is the optimizer moving along the fiber.
- The aim is to marry proof to practice:
  - prove exact statements in Lean where the model is simple enough;
  - measure the same quantities in PyTorch in less simple models;
  - label every claim by how it is supported.

## How claims are labelled

- **`[LC]`, Lean-certified:**
  - Builds with `-DwarningAsError=true -DautoImplicit=false` and has no `sorry`.
  - All 107 declarations in [`lean/Audit.lean`](lean/Audit.lean) (104 `EatYourFiber`, 3 vendored `openai/math`) depend only on `[propext, Classical.choice, Quot.sound]`.
- **`[paper]`:** argued by hand (local linearization, pullback, branch comparison), not formalized.
- **`[empirical]`:** measured (MALA, SMC + HMC, L-BFGS, SGD / Adam / AdamW runs).
- **`[conjectural]`:** heuristic or open.

## What's here

There are two tracks.

### Track 2: inside the fiber (order 2: LoRA, reduced-rank regression, attention)

- **LoRA fiber phase transition** `[LC]` + `[empirical]`:
  - Along the LoRA zero-loss fiber $\{BA = M^*\}$, the LLC is flat (the Aoyagi–Watanabe value) up to a critical rank $q_c$.
  - Above $q_c$ it rises on a parabola, $\lambda_{\mathrm{AW}}^{\mathrm{even}} + \frac{1}{2}(q - q_c)^2$.
  - Standard LoRA init ($B = 0$, $q = k$) sits at the top of that parabola.
  - Lean includes an elementary proof of all four regimes of Aoyagi–Watanabe (2005) Theorem 1.
  - Note: [`lora_fiber_phase_transition.md`](notes/lora_fiber_phase_transition.md).
- **Full $(r^*, p, q)$ stratification** `[LC]`:
  - Lean certifies the local learning pair on every stratum of the reduced-rank-regression fiber.
  - It also shows the pair equals a minimum over irreducible-component codimensions.
  - This is the formula proposed for MAIS-O70 in [`ztreisman/arc-llc`](https://github.com/ztreisman/arc-llc), proved over $\mathbb{R}$ without arc spaces.
  - Note: [`lora_fiber_phase_transition.md`](notes/lora_fiber_phase_transition.md) §3.1.
- **Attention: `V+O` vs. `Q+K`** `[LC]` + `[empirical]`:
  - These results are for single-head weight-product losses, to all orders.
  - Joint `V+O` LoRA makes the parabola 4× steeper.
  - Joint `Q+K` has an exact $GL(q)$ gauge orbit that keeps the LLC flat.
  - With 2 or more heads the two cases decouple: this part is `[paper]` + `[empirical]`.
  - Note: [`lora_nonlinear_fiber.md`](notes/lora_nonlinear_fiber.md).
- **Optimizers pick the point** `[empirical]`:
  - GD, SGD and Adam without weight decay stay at the supercritical top.
  - AdamW and SGD + WD move down onto the plateau.
  - Note: [`lora_sgd_dynamics.md`](notes/lora_sgd_dynamics.md).
- **A real 2-layer causal Transformer** `[empirical]` + `[paper]`:
  - **Plateau and parabolas:** 25 checkpoints computing the same function ($\max\lvert\Delta\mathrm{logit}\rvert \le 1.5\times 10^{-14}$) reproduce the plateau and both parabolas (tops `61.65 ± 0.50` / `61.98 ± 0.40` vs. $62$).
  - **Same function, different futures:** checkpoints computing the same function differ in plasticity vs. stability when fine-tuned.
  - **A late LLC drop that is mostly fiber sliding:** a late AdamW drop of `-10.79` is about 95% movement along the fiber.
  - Note: [`transformer_fiber_surgery.md`](notes/transformer_fiber_surgery.md).
- **DevInterp check** `[empirical]` + `[paper]`:
  - We did a 3-seed rerun of Hoogland et al.'s `L2H4Minf` regression Transformer, with their Table 2 SGLD settings.
  - In our runs the late LLC drop (`-74.88 ± 3.62`) is absent at function-identical balanced points (`+0.58 ± 1.16`).
  - The drop follows a LayerNorm ↔ `W_QKV` rescaling that Adam drifts along.
  - Our drop is larger than the published one (`75` vs. `49`), and we have not tested their checkpoints.
  - Notes: [`devinterp_fiber_decomposition.md`](notes/devinterp_fiber_decomposition.md), [`devinterp_for_authors.md`](notes/devinterp_for_authors.md).

### Track 1: at the edge of the fiber (order 3: bilinear MLPs and 3-tensors)

- **Empty fibers with zero infimum** `[LC]`:
  - For width-$k$ bilinear models fitted to a 3-tensor, the rank-$\le k$ set is not closed.
  - For the $W$-tensor at $k = 2$, $\inf L = 0$, but no finite parameter reaches it.
  - The border-rank gap $\underline{\mathrm{rk}}(W) = 2 < 3 = \mathrm{rk}(W)$ is certified over every field, building on vendored [`openai/math`](https://github.com/openai/math).
- **A stretched-exponential free energy** `[LC]` + `[empirical]`:
  - Under a Gaussian prior, $-\log Z_W^{(2)}(N) \asymp N^{1/7}$, between the realizable $\Theta(\log N)$ and unrealizable $\Theta(N)$ cases.
  - Width 3 is realizable: $-\log Z_W^{(3)}(N) \le 4 \log N + C_3$.
  - Note: [`bilinear_border_rank.md`](notes/bilinear_border_rank.md).
- **A positive-measure version** `[LC]` + `[paper]` + `[empirical]`:
  - On $\operatorname{Det}(T^*) < 0$ ($1 - \pi/4 \approx 21.46\%$ of Gaussian $2\times2\times2$ targets) there is no best rank-2 approximation.
  - There the laws are $\Delta L \asymp R^{-6}$, $\gamma^{3/4}$ and $\beta^{1/4}$.
  - All three are `[LC]` with no hypotheses on an explicit open ball `t3Ball`, and under stated hypotheses elsewhere.
  - Note: [`generic_border_rank.md`](notes/generic_border_rank.md).

## Results index

| Result | Statement | Status | Lean | Code / note |
| :--- | :--- | :---: | :--- | :--- |
| Aoyagi–Watanabe (2005) Thm 1, all regimes and parities | $(\lambda_{\mathrm{AW}}(m,n,k),\, m_{\mathrm{AW}}(m,n,k))$ for all $m,k,n$ | `[LC]` | [`aoyagi_watanabe_rank0`](lean/EatYourFiber/AoyagiWatanabe.lean) | [note](notes/lora_fiber_phase_transition.md) |
| LoRA fiber phase transition | Plateau $\lambda_{\mathrm{AW}}$ ($q \le q_c$); parabola $\lambda_{\mathrm{AW}}^{\mathrm{even}} + \frac{1}{2}(q-q_c)^2$ ($q > q_c$, $m_{\log}=1$) | `[LC]` + `[empirical]` | [`lora_fiber_exact_pair`](lean/EatYourFiber/AoyagiWatanabe.lean), [`lora_subcritical_plateau_exact_pair`](lean/EatYourFiber/AoyagiWatanabe.lean), [`lora_supercritical_parabola_exact_pair`](lean/EatYourFiber/AoyagiWatanabeBoundary.lean) | [`lora_fiber_sgld.py`](experiments/lora_fiber_sgld.py), [note](notes/lora_fiber_phase_transition.md) |
| $(r^*, p, q)$ RRR fiber stratification and component codimension | $\lambda = \frac{MN - m'n'}{2} + \lambda_{\mathrm{AW}}(m',n',k')$; $2\lambda_{\mathrm{exc}} = \min_{\alpha \in [q, k-p]} \operatorname{codim} Z_{\alpha,k-\alpha}$ | `[LC]` | [`rrr_fiber_exact_pair`](lean/EatYourFiber/RRRFiberStratification.lean), [`rrr_fiber_subcritical_plateau`](lean/EatYourFiber/RRRFiberStratification.lean), [`rrr_fiber_A_supercritical_parabola`](lean/EatYourFiber/RRRFiberStratification.lean), [`rrr_fiber_B_supercritical_parabola`](lean/EatYourFiber/RRRFiberStratification.lean), [`rrr_fiber_eq_min_component_codim`](lean/EatYourFiber/RRRFiberStratification.lean) | [note §3.1](notes/lora_fiber_phase_transition.md) |
| Single-head attention `V+O` vs. `Q+K` | `V+O`: $dH - \frac{H^2}{2} + 2(q-\frac{H}{2})_+^2$ ($2q \le d$); `Q+K`: flat $dH - \frac{H^2}{2}$ via $GL(q)$ gauge, any $U,V \in O(d)$ | `[LC]` + `[empirical]` | [`attention_VO_supercritical_parabola`](lean/EatYourFiber/AttentionLoRA.lean), [`attention_QK_general_gauge_orbit`](lean/EatYourFiber/AttentionLoRA.lean), [`attention_QK_general_flat_pair`](lean/EatYourFiber/AttentionLoRA.lean), [`attention_coupling_dichotomy_standard_init`](lean/EatYourFiber/AttentionLoRA.lean) | [`lora_nonlinear_fiber.py`](experiments/lora_nonlinear_fiber.py), [note](notes/lora_nonlinear_fiber.md) |
| 2-head attention decoupling ($d=8, d_h=4, H=4$) | Both pairs decouple to $2 \times \lambda_{\mathrm{single}}(q)$: plateau $24\to 28$, top $32$ | `[paper]` + `[empirical]` | — | [`lora_nonlinear_fiber.py`](experiments/lora_nonlinear_fiber.py), [note](notes/lora_nonlinear_fiber.md) |
| Nonlinear MLPs, dead ReLUs, optimizer dynamics | GELU / ReLU / SwiGLU match the linear curve; $d_{\mathrm{dead}}$ shifts $m_{\mathrm{eff}} = m - d_{\mathrm{dead}}$; weight decay moves LoRA down the parabola | `[paper]` + `[empirical]` | — | [`lora_sgd_dynamics.py`](experiments/lora_sgd_dynamics.py), [`lora_nonlinear_fiber.py`](experiments/lora_nonlinear_fiber.py), [note](notes/lora_sgd_dynamics.md) |
| 2-layer causal Transformer fiber surgery | 25 iso-functional $(p,q)$ checkpoints reproduce plateau and parabolas; same-function checkpoints differ in plasticity vs. stability; late AdamW $\Delta\hat\lambda = -10.79$ is $\approx 95\%$ fiber sliding; Pre-LN $\mathbf{1}_d$ kernel splits tops $57$ vs. $61$ | `[empirical]` + `[paper]` | Tests the RRR and attention results (no new Lean) | [`transformer_fiber_surgery.py`](experiments/transformer_fiber_surgery.py), [note](notes/transformer_fiber_surgery.md) |
| Bilinear model ↔ tensor rank / border rank | Nonempty finite fiber $\iff \mathrm{rk}(T^*)\le k$; $\inf L = 0 \implies \underline{\mathrm{rk}}(T^*)\le k$ (one direction only) | `[LC]` | [`exists_zero_loss_iff_tensorRank_le`](lean/EatYourFiber/BorderRankSLT.lean), [`geometricBorderRank_le_of_approx_loss`](lean/EatYourFiber/BorderRankSLT.lean) | [note](notes/bilinear_border_rank.md) |
| $W$-tensor border-rank gap (every field) | $\underline{\mathrm{rk}}(W) = 2 < 3 = \mathrm{rk}(W)$ | `[LC]` | [`wTensor_geometricBorderRank`](lean/EatYourFiber/BorderRankSLT.lean), [`wTensor_tensorRank`](lean/EatYourFiber/BorderRankSLT.lean), [`wTensor_borderRank_lt_tensorRank`](lean/EatYourFiber/BorderRankSLT.lean) | [note](notes/bilinear_border_rank.md) |
| Escape and weight-decay laws ($W$, $k=2$) | $\inf_{\|\theta\|_2\le R}\|T(\theta)-W\|_F^2 \asymp R^{-12}$; $\inf_\theta(L + \gamma\|\theta\|_2^2) \asymp \gamma^{6/7}$ | `[LC]` + `[empirical]` | [`wTensor_loss_ge_of_paramNormSq`](lean/EatYourFiber/BorderRankSLT.lean), [`wTensor_escape_upper`](lean/EatYourFiber/BorderRankSLT.lean), [`wTensor_weightDecay_lower`](lean/EatYourFiber/BorderRankSLT.lean), [`wTensor_weightDecay_upper`](lean/EatYourFiber/BorderRankSLT.lean) | [`bilinear_border_rank.py`](experiments/bilinear_border_rank.py), [note](notes/bilinear_border_rank.md) |
| Free energy, width 2 vs. width 3 | $-\log Z_W^{(2)}(N) \asymp N^{1/7}$; $-\log Z_W^{(3)}(N) \le 4\log N + C_3$ (Lean gives $\lambda \le 4$; $\lambda = 4$ is `[paper]` + `[empirical]`) | `[LC]` + `[empirical]` | [`wPartitionZ_two_le_stretchedExp`](lean/EatYourFiber/BorderRankSLT.lean), [`wPartitionZ_two_ge_stretchedExp`](lean/EatYourFiber/BorderRankSLT.lean), [`wPartitionZ_three_ge_poly_sharp`](lean/EatYourFiber/BorderRankSLT.lean) | [`bilinear_border_rank.py`](experiments/bilinear_border_rank.py), [note](notes/bilinear_border_rank.md) |
| Generic $\operatorname{Det}(T^*) < 0$ class | $\operatorname{Det}(T(\theta)) = (\det A\det B\det C)^2$; $\Delta L \asymp R^{-6}$, $\Delta L^*(\gamma) \asymp \gamma^{3/4}$, $-\log Z - \beta L_{\mathrm{wall}} \asymp \beta^{1/4}$ | `[LC]` on `t3Ball` and under stated hypotheses; rest `[paper]` + `[empirical]` | [`cayleyDet_bilinearTensor`](lean/EatYourFiber/GenericBorderRank.lean), [`volume_t3Ball_pos`](lean/EatYourFiber/GenericBorderRank.lean), [`t3Ball_twoSided_laws`](lean/EatYourFiber/GenericBorderRank.lean), [`generic_escape_twoSided`](lean/EatYourFiber/GenericBorderRank.lean), [`generic_weightDecay_twoSided`](lean/EatYourFiber/GenericBorderRank.lean), [`genericPartitionZ_two_stretchedExp_twoSided`](lean/EatYourFiber/GenericBorderRank.lean), [`generic_nearMinimizer_scaling`](lean/EatYourFiber/GenericBorderRank.lean) | [`generic_border_rank.py`](experiments/generic_border_rank.py), [note](notes/generic_border_rank.md) |
| DevInterp: `L2H4Minf` late LLC drop | `46k → 364k`: raw $\Delta\hat\lambda = $ `-74.88 ± 3.62`, balanced `+0.58 ± 1.16`; RMSProp-pSGLD still drops `-65.02 ± 4.09` | `[empirical]` + `[paper]` | — | [`devinterp_fiber_decomposition.py`](experiments/devinterp_fiber_decomposition.py), [`verify_upstream_icl_parity.py`](experiments/verify_upstream_icl_parity.py), [note](notes/devinterp_fiber_decomposition.md), [authors' note](notes/devinterp_for_authors.md) |

The notes have the full numbers, settings and caveats for each row.

## Limitations

- **Small models:** the Transformer runs use $d_{\mathrm{model}} = 12$, 2 layers. Carrying the results over to large models is `[conjectural]`.
- **Lean stops at simplified losses:**
  - The `[LC]` attention results are for Frobenius weight-product losses (no softmax).
  - Transfer to next-token KL is `[paper]`.
  - Multi-head decoupling, non-orthogonal `V+O` overlap and nonlinear-MLP pullback are `[paper]` + `[empirical]`.
- **Non-compact priors (Track 1):**
  - The $N^{1/7}$ class needs a non-compact Gaussian prior.
  - On a compact ball the width-2 problem is unrealizable, with linear free energy.
  - See [`bilinear_border_rank.md`](notes/bilinear_border_rank.md) §2.4.
- **Generic border rank:**
  - `GenericWallHyp` and $\mathrm{GL}(2)^3$ wall attainment outside `t3Ball` are `[paper]`, as are the virial limit and the Laplace offsets.
  - The 77.5–90% degeneration rates at $d = 4$ come from an L-BFGS detector, not an exact classification.
  - SwiGLU was tested on one native teacher ([`generic_border_rank.md`](notes/generic_border_rank.md) §3.4).
- **Estimators:**
  - All LLC numbers are finite-$\beta$ / finite-$\gamma$ estimates.
  - We use Metropolis-adjusted samplers where we can: unadjusted SGLD showed large upward bias on stiff directions ([`lora_nonlinear_fiber.md`](notes/lora_nonlinear_fiber.md) §1.3).
  - Subcritical plateaus read slightly low at finite $\beta$.
- **Noisy fine-tuning:**
  - With small noisy data there may be no inactive fiber at all.
  - The raw empirical LLC can go strongly negative there ([`lora_nonlinear_fiber.md`](notes/lora_nonlinear_fiber.md) §2.2).
- **Pre-LN vs. Pre-RMSNorm:** the Pre-LN kernel argument is `[paper]`. It was checked on one Pre-LN model that only partly learned its task.
- **Optimizer dependence:** the plasticity–stability split largely washes out under Adam from a near-plateau start.
- **DevInterp:**
  - Our raw drop is larger than the published one, and their checkpoints are private.
  - We can't rule out a small change in the true LLC.
  - The estimate's change at fixed function depends on the localization metric (about `−16` to `+1`).
  - Extending this to their language models is `[conjectural]` ([`devinterp_fiber_decomposition.md`](notes/devinterp_fiber_decomposition.md) §9).

## Layout

- `lean/`: Lean `v4.35.0` project, depending on [`slt-bridge`](https://github.com/bubskee/slt-bridge) and Mathlib.
  - `lean/EatYourFiber/`: `LoRASchur`, `LoRASchurLaplace`, `AoyagiWatanabeBoundary`, `AoyagiWatanabe`, `RRRFiberStratification`, `AttentionLoRA`, `BorderRankSLT`, `GenericBorderRank`.
  - `lean/Audit.lean`: `#print axioms` for 107 headline declarations.
  - `lean/Vendor/OAI/`: 8 Apache-2.0 tensor / border-rank modules from [`openai/math`](https://github.com/openai/math) (commit `3344da59991ed94f9ba711217b29c1d760badfa6`). See [NOTICE](NOTICE).
- `experiments/`: PyTorch scripts (managed with `uv`), with JSON / PNG outputs in `experiments/results/`.
- `notes/`: one write-up per experiment, each with derivations, settings, results and a claim-status header.

## Reproduce

Lean build and axiom audit:

```sh
cd lean
lake exe cache get
lake build
lake env lean -DwarningAsError=true -DautoImplicit=false Audit.lean
```

Experiments (approximate wall-clock times are recorded in each results JSON and depend on the machine):

```sh
cd experiments
uv run python bilinear_border_rank.py        # ~6 min
uv run python generic_border_rank.py         # ~11 min
uv run python lora_fiber_sgld.py
uv run python lora_sgd_dynamics.py
uv run python lora_nonlinear_fiber.py        # ~11 min
uv run python transformer_fiber_surgery.py   # ~12 min
uv run python verify_upstream_icl_parity.py --icl-repo /path/to/icl
uv run python devinterp_fiber_decomposition.py --from-checkpoints --eval-seeds 0,1,2   # --replot redraws from JSON
```

## Related work

- **Watanabe (2009, 2018):** *Algebraic Geometry and Statistical Learning Theory*; *Mathematical Theory of Bayesian Statistics*. The SLT background: RLCTs, free-energy asymptotics, WBIC.
- **Aoyagi & Watanabe (2005):** "Stochastic complexities of reduced rank regression in Bayesian estimation," *Neural Networks* 18(7). Their Theorem 1 is formalized here (`aoyagi_watanabe_rank0`).
- **Aoyagi (2024):** "Consideration on the learning efficiency of multiple-layered neural networks with linear units," *Neural Networks*. RLCTs of deep linear networks.
- **Lau, Furman, Wang, Murfet & Wei (2025):** "The Local Learning Coefficient: A Singularity-Aware Complexity Measure," AISTATS ([arXiv:2308.12108](https://arxiv.org/abs/2308.12108)). LLC definition and SGLD estimation.
- **Treisman (2025/2026):** [`ztreisman/arc-llc`](https://github.com/ztreisman/arc-llc). Arc-space computation of local RLCTs for reduced-rank regression, and the MAIS-O70 formula certified here.
- **Hoogland, Wang, Farrugia-Roberts, Carroll, Wei & Murfet (2024/2025):** "Loss Landscape Degeneracy and Stagewise Development in Transformers," to appear in TMLR ([arXiv:2402.02364](https://arxiv.org/abs/2402.02364)). The `L2H4Minf` model and LLC curves we rerun.
- **Hitchcock & Hoogland (2025):** "From Global to Local: A Scalable Benchmark for Local Posterior Sampling" ([arXiv:2507.21449](https://arxiv.org/abs/2507.21449)). RMSProp-preconditioned SGLD, used in our DevInterp comparison.
- **Hu et al. (2022):** "LoRA," ICLR ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685)). Standard $(B=0, A \sim \mathcal{N})$ init.
- **Hayou, Ghosh & Yu (2024):** "LoRA+," ICML ([arXiv:2402.12354](https://arxiv.org/abs/2402.12354)). Separate learning rates $\eta_B > \eta_A$.
- **de Silva & Lim (2008):** "Tensor rank and the ill-posedness of the best low-rank approximation problem," *SIAM J. Matrix Anal. Appl.* 30(3) ([arXiv:math/0607647](https://arxiv.org/abs/math/0607647)). Border rank and the positive-volume set with no best rank-2 approximation.
- **Bergqvist (2013):** "Exact probabilities for typical ranks of 2×2×2 and 3×3×2 tensors," *Linear Algebra Appl.* 438(2). The $1 - \pi/4$ probability.
- **Sharkey (2023)** ([arXiv:2305.03452](https://arxiv.org/abs/2305.03452)) and **Pearce, Dooms, Rigg, Oramas & Sharkey (2024)** ([arXiv:2410.08417](https://arxiv.org/abs/2410.08417)). Bilinear MLPs for interpretability.
- **Related repos:** [`residual-slt-in-a-haystack`](https://github.com/bubskee/residual-slt-in-a-haystack), [`slt-bridge`](https://github.com/bubskee/slt-bridge).

## License

Code and Lean proofs are under the [Apache License 2.0](LICENSE). Files in `lean/Vendor/OAI/` are adapted from [`openai/math`](https://github.com/openai/math) under Apache 2.0; see [NOTICE](NOTICE).
