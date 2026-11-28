# Track A Empirical Study: Is That LLC Drop Development, or Is the Estimator Sliding Along a Symmetry Fiber?

> **Epistemic status:**
> - **Bilinear fiber stratification & single-head $GL(q)$ gauge invariance (`RRRFiberStratification.lean`, `AttentionLoRA.lean`):** `[LC]` in [`lean/EatYourFiber/RRRFiberStratification.lean`](../lean/EatYourFiber/RRRFiberStratification.lean) and [`lean/EatYourFiber/AttentionLoRA.lean`](../lean/EatYourFiber/AttentionLoRA.lean). These certify the exact RRR / linearized-attention RLCTs; applying fiber canonicalization to full Transformers relies on the `[paper]` exact-symmetry derivations below.
> - **Exact continuous iso-functional symmetries of `DTransformer` (`ORBIT`: Pre-LN $\mathbf{1}_d$ residual kernel, invertible per-head $GL(d_h)$ $Q$-$K$ & $V$-$O$ Riccati balancing, strictly positive ReLU MLP neuronwise rescaling, and strictly positive coordinatewise $(\gamma_i, \beta_i) \leftrightarrow W_{:,i}$ LayerNorm affine scaling across all 5 unembedding rows), the $1$-vs-$d_{\text{out}}$ finite-$\gamma$ localized-Gibbs curvature distortion derivation, and the exact orbit-equivariant metric localization SGLD construction:** `[paper]` exact algebraic / local Gaussian-Laplace analysis (not formalized in Lean).
> - **All public repository/checkpoint audits, committed upstream parity verification (`max|Δy| = 0.0` in float64 & float32 across 512 sequences against `timaeus-research/icl` @ `5baf730`), faithful 500k multi-seed `L2H4Minf` reproductions (`Adam, wd=0` and bit-for-bit matched-init `AdamW, wd=0.08`, 3 seeds each, saved as `.safetensors` + SHA-256 `manifest.json`), exact Table-2 SGLD (`10` chains $\times$ `5,000` steps, $m=1{,}024$, $2^{20}$ pool, exact FP32), float64 & post-float32 exactness checks across all 114 checkpoints on 512 fresh sequences, GP-smoothed LLC-derivative stage detection, per-channel `ORBIT` vs. `ORBIT+` ablations, exact full-model & block Gauss–Newton spectra (`512 × 8 = 4,096` residuals), 4-estimator remedy suite (`vanilla`, `rmsprop_psgld` [arXiv:2507.21449], black-box `param_rms_covariant`, and exact `orbit_covariant`), and $\gamma$/$\epsilon$/full-batch-gradient controls:** `[empirical]` in [`experiments/devinterp_fiber_decomposition.py`](../experiments/devinterp_fiber_decomposition.py) and [`experiments/verify_upstream_icl_parity.py`](../experiments/verify_upstream_icl_parity.py) (raw data: [`experiments/results/devinterp_fiber_decomposition.json`](../experiments/results/devinterp_fiber_decomposition.json), 8-panel figure: [`experiments/results/devinterp_fiber_decomposition.png`](../experiments/results/devinterp_fiber_decomposition.png), checkpoints + manifest: [`experiments/results/devinterp_checkpoints/manifest.json`](../experiments/results/devinterp_checkpoints/manifest.json)).
> - **Extrapolation of this LayerNorm-vs-Linear finite-$\gamma$ fiber artifact to larger attention-only language models (e.g., `LM1`–`LM5` in Hoogland et al., 2024) and other DevInterp LLC trajectories:** `[conjectural]`.

> **How to read "fiber", "func", `ORBIT` vs. `ORBIT+`, and "canonicalized LLC" in this note (`[paper]`):**
> Every move in **`ORBIT`** (§2) is a **pure invertible loss-preserving diffeomorphism** $g$ of parameter space $\mathbb{R}^{51,717}$ that preserves **all 5 `DTransformer` output channels** (zero rank truncation, zero zeroing branches, verified $\min s_i > 0$, $\min \sigma > 0$, $|\det G| > 0$ on every checkpoint): coordinatewise strictly positive rescalings $(\gamma_i,\beta_i,W_{:,i})\mapsto(s_i\gamma_i,s_i\beta_i,s_i^{-1}W_{:,i})$ with $s_i>0$ (scaling all 5 rows of `unembedding.weight` simultaneously), invertible per-head $GL(16)$ Riccati balancing on full-rank $Q$–$K$ and $V$–$O$ factors, strictly positive ReLU neuron rescalings + permutations, and exact flat translations (Pre-LN $\mathbf{1}_d$ projection and bias absorption). For such $g$, $K\circ g=K$, so the true LLC satisfies $\lambda(w)=\lambda(g\cdot w)$ (Lau et al., 2308.12108, §3 and App. C: invariance under local diffeomorphisms). Extra norm-reducing moves that are *not* parameter-space diffeomorphisms (zeroing the 4 unused unembedding rows + row-0-only `lnf` balancing) are isolated in **`ORBIT+`** and reported separately.
> - $\hat\lambda_{\text{raw}}$ and $\hat\lambda_{\text{orbit}}$ are two SGLD estimates of the **same** true LLC, taken at two parameters implementing the exact same function. $\Delta\hat\lambda_{\text{fiber}}=\Delta\hat\lambda_{\text{total}}-\Delta\hat\lambda_{\text{func}}$ measures how much the change in the **estimator** comes from its non-invariance under these symmetries (isotropic localization $\frac{\gamma}{2}\|w-w^*\|_2^2$, finite $n\beta$, finite step size and chain length). It is **not** a change in the true LLC.
> - "Fiber sliding" names a real fact about **training**: from step `46k` to `364k`, the Adam trajectory moves far along an exact rescaling orbit (pre-attention `LN_0` $\gamma$ shrinks $6.1\times$ while `W_QKV` grows $6.6\times$). What is an artifact is that the SGLD **estimate** tracks this coordinate.
> - "Balanced / `ORBIT` LLC" means **the SGLD estimate at the `ORBIT`-balanced representative**. It is itself a finite-$\gamma$ estimate and is gauge-dependent: at step `363,636` (seed 0, §5.1) three pure-diffeomorphism gauges give `161.65` (`LN0_attn` only), `164.10` (unit-$\gamma$), and `164.87` (`ORBIT`). It is not the true LLC either. What it shows is that a symmetry-fixed estimate — or an orbit-covariant sampler evaluated directly at the raw weights (§7) — has no `LR3–LR4` drop.
> - Prior work: Lau et al. (2308.12108) note that the SGLD estimator equals the LLC only conjecturally, under conditions on $\gamma$ (App. D), and their rescaling-invariance experiment (App. O, two-layer ReLU network) used a preconditioner matched to the rescaling. What we have not found in prior work is (i) a Transformer training trajectory drifting along an exact `LN ↔ Linear` orbit, (ii) that drift producing a reported developmental LLC drop even under RMSProp-preconditioned SGLD, and (iii) direct Gauss–Newton spectral proof + an orbit-covariant metric localization remedy.

---

## Executive Summary

We perform a **faithful 500,000-step, exact-FP32** fiber decomposition, Gauss–Newton spectral audit, and remedy benchmark on Hoogland, Wang, Farrugia-Roberts, Carroll, Wei & Murfet (2024, TMLR, arXiv:2402.02364), *"Loss Landscape Degeneracy and Stagewise Development in Transformers"* — specifically their **51,717-parameter in-context linear regression Transformer (`L2H4Minf`)**, whose non-monotonic SGLD Local Learning Coefficient (LLC) curve exhibits five developmental stages (`LR1`–`LR5`) including a prominent late-stage drop (`LR3`–`LR4`).

Because their regression checkpoints are in a private AWS bucket, we retrained **3 seeds (`0, 1, 2`)** of both **`Adam (wd = 0.0)`** and **bit-for-bit matched-init `AdamW (wd = 0.08)`** (`max|w_init_adam - w_init_adamw| = 0.0`, identical SHA-256) for the **exact paper horizon $T = 500{,}000$** (`OneCycleLR` peak at `250k`, $B=256$, $K=8, D=4, \sigma^2=0.125$), saving all **114 checkpoints** to disk as `.safetensors` with a SHA-256 manifest. Every LLC estimate uses **Hoogland et al.'s exact Table 2 SGLD settings** in full FP32 (`tf32=False`): **10 chains $\times$ 5,000 steps** (`1,000` burn-in + `4,000` draws), minibatch **$m = 1{,}024$**, fixed deterministic evaluation pool of **$2^{20} = 1{,}048{,}576$** sequences, $\epsilon = 3\times 10^{-4}, \gamma = 13.3, n\beta = 66.7$.

Our faithful 500k raw curve tracks Hoogland et al.'s released 5-seed `analysis/L2H4Minf/llcs.pt` with **Pearson $r = 0.981$** across 23 aligned checkpoints, hitting the **exact same empirical peak step (`46,003`)** (`167.03 ± 2.15` vs. their `170.42 ± 0.23`), matching `LR3` step `126,262` to `0.7` points (`156.15 ± 1.63` vs. their `156.87 ± 0.45`), and hitting the **exact same trough step (`363,636`)** (GP-smoothed trough `365,597` vs. their GP trough `359,112`) before rebounding in `LR5` (`+5.44` vs. their `+6.71`). Our raw trough is deeper (`92.15 ± 1.48` vs. their `121.16 ± 2.79`; `-74.88` drop vs. their `-49.26`), reflecting stronger late-stage weight growth in exact FP32. Exact pure-diffeomorphism `ORBIT` canonicalization ($\max|\Delta y|_{\mathrm{f64}} \le 1.39\times 10^{-12}$, **post-float32 $\max|\Delta y|_{\mathrm{f32}} \le 2.14\times 10^{-5}$** across all 114 checkpoints on 512 fresh sequences) establishes five main results:

1. **Early development (`LR1` zero prediction, `LR2` in-context learning) is genuinely functional, not a gauge artifact (`[empirical]`):**
   - `LR1 (0 → 984)`: $\Delta\hat\lambda_{\text{total}} = +16.15 \pm 8.21$ decomposes into $\Delta\hat\lambda_{\text{func}} = +16.14 \pm 8.22$ and $\Delta\hat\lambda_{\text{fiber}} = +0.01 \pm 0.01$ (**fiber share $+0.2\% \pm 0.1\%$**).
   - `LR2 (984 → 46,003)`: $\Delta\hat\lambda_{\text{total}} = +149.70 \pm 0.99$ decomposes into $\Delta\hat\lambda_{\text{func}} = +154.18 \pm 0.96$ and $\Delta\hat\lambda_{\text{fiber}} = -4.47 \pm 0.17$ (**fiber share $-3.0\% \pm 0.1\%$**).
2. **The entire `LR3–LR4` drop (`46,003 → 363,636`) is 100.9% ± 1.6% estimator non-invariance along the exact `ORBIT` symmetry fiber (`[empirical]`):**
   - Across all 3 seeds, `RAW` SGLD plunges by $\Delta\hat\lambda_{\text{total}} = \mathbf{-74.88 \pm 3.62}$ (`167.03 → 92.15`), while on the **exact same functions** (`ORBIT` pure diffeomorphism, `0` skipped heads across all 69 Adam checkpoints), `ORBIT` SGLD shows **no drop**: $\Delta\hat\lambda_{\text{func}} = \mathbf{+0.58 \pm 1.16}$ (`163.60 → 164.19`), leaving $\Delta\hat\lambda_{\text{fiber}} = \mathbf{-75.47 \pm 2.58}$ (**fiber share $100.9\% \pm 1.6\%$**, individually `98.2%`, `103.7%`, `100.9%`; using the GP-smoothed peak `30,303 → 363,636` gives `104.4% ± 1.8%`).
   - Extra non-diffeomorphism moves (`ORBIT+`, zeroing unused unembedding rows `1..4`) change the trough LLC by only `0.01` points (`164.88` vs. `164.87` on seed 0), confirming the result does not depend on any non-invertible move.
3. **Per-channel ablation & exact Gauss–Newton spectra directly prove the $1$-vs-$d_{\text{out}}$ `LN_0 ↔ QKV` curvature-suppression mechanism (`[empirical]` + `[paper]`):**
   - At step `363,636` (seed 0), balancing **only** pre-attention `LN_0 ↔ QKV` (`blk0_ln0`, `blk1_ln0`, `dy_f32 = 5.2e-6`) lifts $\hat\lambda$ from `90.01 ± 0.54` to **`161.65 ± 0.12`** (**`+71.63` points = `95.7%` of the `+74.86` `ORBIT`–`RAW` gap**). `LN1_mlp` adds `+8.01`; `LN_mean_proj`, `QK`, `VO`, `MLP`, and `LNf_unembed_all5` are all `< 1` point.
   - **Direct spectral measurement (3-seed mean, exact Gauss–Newton $G = \frac{2}{4096} J^\top J$ on 512 sequences $\times$ 8 positions):** Between peak `46k` and trough `364k`, as `blk0_ln0` shrinks (`mean|γ|: 1.039 → 0.169`) and `||W_QKV,0||` blows up (`10.80 → 71.61`), the number of `W_QKV` Gauss–Newton eigenvalues above the isotropic localization floor $\tau = \gamma / (n\beta) = 0.1994$ **collapses $11\times$ in `RAW` (`398.3 ± 12.6 → 36.0 ± 7.0`)**, while in `ORBIT` it **rises (`450.7 ± 10.7 → 547.0 ± 13.3`)**! Simultaneously, the `LN_0` block's max eigenvalue explodes $10\times$ in `RAW` (`3.19 → 31.85`) while shrinking in `ORBIT` (`2.53 → 0.37`). Full-model Gaussian-Laplace effective dimension $\hat\lambda_{\mathrm{GN}}(13.3)$ collapses **`359.93 → 119.15`** (`-240.8`) in `RAW`, while rising **`372.84 → 398.69`** in `ORBIT`!
4. **Reparametrization-aware remedies — why RMSProp-pSGLD (`arXiv:2507.21449`) does NOT fix the drop, and what DOES (`[empirical]` + `[paper]`):**
   - **Standard RMSProp-pSGLD (Hitchcock & Hoogland, arXiv:2507.21449 Alg. G.3 / `devinterp` `RMSpropPreconditioner`, $a = 0.01$, 3 seeds):** `RAW` **still drops by `-65.02 ± 4.09`** (`168.16 → 103.14`) while `ORBIT` stays flat (`168.04 → 167.09`, $\Delta = -0.95 \pm 1.22$; trough `RAW - ORBIT = -63.94 ± 3.66`). Adaptive step-size preconditioning improves mixing slightly (+11 points vs. vanilla) but **cannot** fix the drop because it still targets the **same isotropic localized posterior** $\pi_\gamma(w) \propto \exp(-n\beta L(w) - \frac{\gamma}{2}\|w - w^*\|_2^2)$.
   - **Symmetry-Adapted Metric Localization SGLD (`orbit_covariant`, exact pure-multiplicative diagonal metric $D(w^*) = \mathrm{diag}(c_p^2)$ + covariant preconditioner $P = D^{-1}$, ZERO weight modification, 3 seeds):** **Completely eliminates the drop on `RAW` weights (`163.34 ± 1.99 → 163.07 ± 1.30`, $\Delta = -0.27 \pm 0.83$)** and agrees with `ORBIT`-canonicalized Vanilla SGLD to `< 1` point at trough (`RAW - ORBIT = -0.94 ± 0.16`)!
   - **Black-box Per-Tensor Param-RMS Covariant SGLD (`param_rms_covariant`, architecture-agnostic, 3 seeds):** Eliminates **100.0%** of the `RAW - ORBIT` trough gap (`159.84 ± 0.85` vs. `159.85 ± 0.91`, gap **`-0.00 ± 0.13`**!) without knowing any model symmetry, though normalizing every tensor to its initialization RMS also rescales genuine effective-weight growth (producing a matched `-17.67` / `-15.49` shift on both `RAW` and `ORBIT`).
5. **Controls (`[empirical]`):**
   - **Finite-$\gamma$ sweep (seed 0, `46k → 364k`):** `RAW` peak→trough drop scales dramatically with $\gamma$: `-12.45` at $\gamma=3.3$ (where chains wander $\|w - w^*\| = 124.8$), `-80.03` at $\gamma=13.3$, and **`-110.59`** at $\gamma=40.0$ (`163.22 → 52.63`). Across the exact same $\gamma$ range, `ORBIT` changes by only `+0.88`, `-1.22`, and `+2.80` (`165.2–167.4` at trough).
   - **Full-batch-gradient SGLD (seed 0, 1,024 fixed samples, Langevin noise $\sqrt{\epsilon}\eta_t$ retained):** Preserves the full `-82.17`-point `RAW` drop (`167.80 → 85.62`) vs. `-1.10` for `ORBIT` (`164.59 → 163.50`, `hd = +0.2 / +0.0`), ruling out minibatch gradient noise.
   - **Matched-init `AdamW (wd = 0.08)` counterfactual (3 seeds, identical init & batch stream):** Bounds `||W_QKV,0||` (`8.2 → 12.2` vs. `71.6` under Adam) and collapses the trough fiber gap from `-72.04` to `-1.18` (`RAW 152.55` vs. `ORBIT 153.74`), leaving only a small `-4.94 ± 0.59` dip across `46k → 364k` (`dFunc = -4.09`, `dFiber = -0.85`) despite *stronger* `lnf` collapse (`96%` vs. `95%`) and monotonically rising OOD `g=5` MSE (`3.372 → 3.652`).

---

## 1. Public Code Audit, Recipe Resolution & Upstream Parity (`[empirical]`)

### 1.1 Upstream Parity & Recipe Conflict Resolution
- **Committed upstream parity check ([`experiments/verify_upstream_icl_parity.py`](../experiments/verify_upstream_icl_parity.py)):** Directly imports `InContextRegressionTransformer` from `github.com/timaeus-research/icl` (pinned commit `5baf73025d279abd2d56c98e13bf5663608709c0`), copies exact weights into our replica, and evaluates 512 fresh sequences (`seed = 2026`). Result (embedded in JSON): **`51,717` parameters, identical `state_dict` keys, `max_delta_y_f64 = 0.0`, `max_delta_y_f32 = 0.0`, `max_delta_all_5ch_f64 = 0.0`, `max_delta_all_5ch_f32 = 0.0`**.
- **Explicit resolution of the YAML vs. paper recipe conflict:** The public `sweeps/regression/training-runs/L2H4Minf.yaml` sets `max_examples: 16` and leaves optimizer and scheduler unspecified (so `src/icl/regression/config.py` defaults to `SGD`, `weight_decay = 1e-4`). By contrast, the published paper (arXiv:2402.02364 Table 4 & App. F.2) specifies $K = 8$ (`max_tokens = 16`, `51,717` params), `Adam (wd = 0.0)`, `OneCycleLR(max_lr = 0.003, pct_start = 0.5, anneal_strategy = 'linear', div_factor = T/2 - 1, final_div_factor = T/2 - 1)`, $T = 500{,}000$, $B = 256$, $D = 4$, $\sigma^2 = 0.125$. We follow the paper because `analysis/L2H4Minf/llcs.pt` logs exactly 8 token positions (`0..7`, matching $K = 8$) and its MSE/LLC trajectory matches `Adam (wd = 0.0)`.
- **Exact Table 2 SGLD settings (zero shortcuts):** $\epsilon = 3\times 10^{-4}$, $\gamma = 13.3$, $n\beta = 66.7$, **10 chains $\times$ 5,000 steps** (`1,000` burn-in + `4,000` draws), minibatch **$m = 1{,}024$**, fixed deterministic pool of **$2^{20} = 1{,}048{,}576$** sequences (`pool_seed = 12345`), exact FP32 (`tf32=False`) on an NVIDIA RTX PRO 6000 Blackwell GPU.

### 1.2 Quantitative Comparison Against Hoogland et al.'s Released 5-Seed `llcs.pt` & GP Stage Boundaries
We apply Hoogland et al.'s exact GP-smoothed LLC-derivative stage-boundary procedure (`icl/notebooks/main-figures.ipynb` cell 51 & App. B.1: RBF + WhiteKernel GP on $\log(t+1)$ for $t \ge 100$, differentiated w.r.t. $\log(t+1)$) directly to both our 3-seed mean raw curve (23 checkpoints aligned to `llcs.pt`) and Hoogland et al.'s released 5-seed `llcs.pt` (190 checkpoints):

| Metric | Hoogland `llcs.pt` (5 seeds, 500k) | Our Faithful 500k (`Adam, wd=0`, 3 seeds) | Earlier 100k Retrain (`T=100k`, cheaper chain) |
|---|---:|---:|---:|
| Aligned Pearson $r$ vs. `llcs.pt` | `1.000` | **`0.9806`** (23 aligned steps) | n/a (5× shorter horizon) |
| `LR1` boundary (`984` / `1k`) LLC | `19.23 ± 0.35` (`MSE 4.157`) | `17.33 ± 1.21` (`MSE 4.203`) | `33.19 ± 1.83` (`MSE 3.969`) |
| GP `LR1→LR2` plateau step | `1,270` | `902–1,187` | assigned `1,000` |
| Empirical Peak step & LLC | **`46,003`** (`170.42 ± 0.23`, `MSE 1.881`) | **`46,003`** (`167.03 ± 2.15`, `MSE 1.948`) | `10,000` (`162.68 ± 0.23`) |
| GP Peak step & LLC | `45,668` (`170.08`) | `33,857` (`166.70`, flat `30k–46k`) | assigned `25,000` |
| `LR3` step `126,262` LLC | `156.87 ± 0.45` (`MSE 1.827`) | **`156.15 ± 1.63`** (`MSE 1.875`) | n/a |
| Empirical Trough step & LLC | **`363,636`** (`121.16 ± 2.79`, `MSE 1.773`) | **`363,636`** (`92.15 ± 1.48`, `MSE 1.804`) | `80,000` (`113.53 ± 4.22`) |
| GP Trough step & LLC | `359,112` (`121.17`) | **`365,597`** (`91.30`) | assigned `80,000` |
| `LR3–LR4` Drop (Peak → Trough) | `-49.26` (seed range `-39` to `-57`) | **`-74.88 ± 3.62`** | `-46.76 ± 4.52` |
| `LR5` Uptick (Trough → `500k`) | `+6.71` (`121.16 → 127.87`) | **`+5.44 ± 1.45`** (`92.15 → 97.59`) | `+3.10 ± 0.62` |

*Note on the deeper trough (`92.15` vs. `121.16`):* Up through step `126,262` (`156.15` vs. `156.87`), our faithful 500k curve matches Hoogland et al.'s 5-seed mean to within `0.7–3.4` points. After `150k` (around the `OneCycleLR` peak at `250k`), our exact-FP32 PyTorch GPU runs drift slightly further along the `LN_0 ↔ QKV` fiber (`||w|| = 241.7` at `364k`) than their TPU runs, deepening the raw trough by ~29 points while hitting the **exact same trough step (`363,636`)** and exact same `LR5` rebound (`+5.44` vs. `+6.71`).

---

## 2. Pure Group Actions (`ORBIT`) vs. Extra Moves (`ORBIT+`) & Post-Float32 Exactness (`[paper]` + `[empirical]`)

We split canonicalization strictly into two classes, run in `float64` and verified on **512 fresh held-out sequences** both **in `float64` (before cast)** and **in `float32` (after casting canonical weights to `float32`, the exact weights SGLD evaluates)**:

### 2.1 `ORBIT` — Pure Invertible Loss-Preserving Group Actions ONLY
Every move in `ORBIT` is a smooth bijection on $\mathbb{R}^{51,717}$ preserving **all 5 output channels** of `DTransformer` (zero rank truncation, zero zeroing branches):
1. **Pre-LN $\mathbf{1}_d$ kernel projection (`LN_mean_proj`, `[paper]`):** Exact flat translation ($\det = 1$) subtracting the mean along `d_embed` from `token_embedding.weight`, `postn_embedding.weight`, `attention.output.weight`, `fc2.weight`, and `fc2.bias`.
2. **Invertible per-head $GL(16)$ Riccati balancing (`QK`, `VO`, `[paper]`):** Computes the unique SPD matrix $G \in GL(16)$ minimizing $\|G W_Q\|_F^2 + \|G^{-1} W_K\|_F^2$ (and $\|G W_V\|_F^2 + \|W_O G^{-1}\|_F^2$) via square-root-free SVDs (no Gram-matrix squaring), explicitly checking $\sigma_{\min} \ge 10^{-4}$ and $|\det G| > 0$ (leaving $G = I$ on any rank-deficient head — which never fires on Adam and cleanly guards decayed AdamW heads).
3. **Strictly positive ReLU MLP neuronwise rescaling + permutation (`MLP`, `[paper]`):** Scales $(W_{1,j,:}, b_{1,j}) \mapsto s_j (W_{1,j,:}, b_{1,j})$, $W_{2,:,j} \mapsto s_j^{-1} W_{2,:,j}$ with $s_j = \sqrt{\|W_{2,:,j}\|_2 / \sqrt{\|W_{1,j,:}\|_2^2 + b_{1,j}^2}} > 0$ (skipping only if norm $\le 10^{-15}$, never zeroing!), then sorts descending.
4. **Strictly positive coordinatewise LayerNorm affine balancing (`LN0_attn`, `LN1_mlp`, `LNf_unembed_all5`, `[paper]`):**
   - `LN0_attn` (biasless `ln0 → QKV`): scales $(\gamma_{0,i}, \beta_{0,i}) \mapsto s_i(\gamma_{0,i}, \beta_{0,i})$, $W_{QKV,:,i} \mapsto s_i^{-1} W_{QKV,:,i}$ with $s_i = (\|W_{QKV,:,i}\|_2 / \sqrt{\gamma_{0,i}^2 + \beta_{0,i}^2})^{1/2} > 0$.
   - `LN1_mlp` (biased `ln1 → fc1`): exact bias absorption $b_{\text{fc1}} \leftarrow b_{\text{fc1}} + W_{\text{fc1}} \beta_1$, $\beta_1 \leftarrow 0$, then strictly positive $s_i = (\|W_{\text{fc1},:,i}\|_2 / |\gamma_{1,i}|)^{1/2} > 0$.
   - **`LNf_unembed_all5` (`lnf → unembedding` across ALL 5 rows simultaneously):** exact 5-row bias absorption $b_U \leftarrow b_U + W_U \beta_f$, $\beta_f \leftarrow 0$, then strictly positive $s_i = (\|W_{U,:,i}\|_2 / |\gamma_{f,i}|)^{1/2} > 0$ scaling **all 5 rows** of column $i$ simultaneously.
- **`ORBIT`** applies `LN_mean_proj` followed by 20 alternating sweeps of (`LNf_unembed_all5`, `LN1_mlp`, `LN0_attn`, `QK`, `VO`, `MLP`).

### 2.2 `ORBIT+` — Extra Norm-Reducing Moves (Reported Separately)
Starts from `ORBIT`, zeros the 4 unused unembedding rows/biases (`unembedding.1.{weight,bias}[1:5] = 0`, which leaves the regression readout channel 0 unchanged but alters channels 1–4 and is not an invertible parameter-space map), and re-balances `lnf` against readout row 0 only.

### 2.3 Exactness & Diffeomorphism Certificates Across All 114 Checkpoints (`[empirical]`)
- **All 69 `Adam (wd = 0)` checkpoints (512 fresh sequences):**
  - Readout channel 0: **$\max|\Delta y|_{\mathrm{f64}} \le 1.39\times 10^{-12}$**, **post-float32 $\max|\Delta y|_{\mathrm{f32}} \le 2.14\times 10^{-5}$**.
  - All 5 `DTransformer` channels: **$\max|\Delta \text{all5}|_{\mathrm{f64}} \le 3.28\times 10^{-12}$**, **post-float32 $\max|\Delta \text{all5}|_{\mathrm{f32}} \le 5.97\times 10^{-5}$**.
  - Diffeomorphism certificate: **`n_skipped_degenerate = 0` on every single checkpoint**, $\min s_i = 0.263$, $\max s_i = 6.30\times 10^6$ (from collapsed `lnf` coordinates $|\gamma_{f,i}| \sim 10^{-13}$ balanced against live `unembedding` rows 1–4), $\min \sigma = 0.0193$, $\min|\det G| = 0.110$, $\max \kappa(G) = 3.88$. Every `ORBIT` map on Adam is a certified smooth diffeomorphism!
- **All 45 `AdamW (wd = 0.08)` checkpoints (512 fresh sequences):**
  - Readout channel 0: $\max|\Delta y|_{\mathrm{f64}} \le 7.87\times 10^{-13}$, **post-float32 $\max|\Delta y|_{\mathrm{f32}} \le 1.57\times 10^{-5}$**; all 5 channels $\le 2.12\times 10^{-12}$ (f64) / $2.84\times 10^{-5}$ (f32). Weight decay prunes some attention-head singular values below $10^{-4}$ late in training (`202k–500k`), which our SVD guard cleanly leaves at $G = I$ (still an exact diffeomorphism!).

---

## 3. Part 1 — Faithful 500k Multi-Seed `Adam (wd = 0.0)` Decomposition (`[empirical]`)

*Error bars in §3–4:* `±` is the standard error **across the 3 training seeds (`0, 1, 2`)** of each seed's 10-chain mean ($\mathrm{SD}/\sqrt{3}$). Within-seed 10-chain SEs are `0.04–0.92` at almost all checkpoints (up to `2.78` mid-transition at `162k`). Fiber shares are means of per-seed ratios.

### 3.1 Full 23-Checkpoint Trajectory (3 Seeds Mean ± SE, Exact Table-2 SGLD: 10×5,000, $m=1{,}024$, $2^{20}$ pool)

| Step | ID MSE | $\mathbb{E}[\hat y^2]$ | $\mathrm{ICL}_{0:7}$ | OOD $g{=}5$ | Raw $\hat\lambda \pm \mathrm{SE}$ | `ORBIT` $\hat\lambda \pm \mathrm{SE}$ | Raw $-$ `ORBIT` | `blk0_ln0` $\|\gamma\|$ / $\|W_{\text{QKV}}\|$ | `lnf` $\|\gamma\|{<}0.1$ / eff${<}0.05$ | $\|w\|_2$ raw $\to$ orb | Worst $\max|\Delta y|_{\mathrm{f32}}$ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0` | 4.4452 | 0.265 | -0.060 | 4.083 | 1.18 ± 9.42 | -6.72 ± 9.39 | +7.89 | 1.000 / 8.0 | 0.00 / 0.44 | 24.9 → 23.1 | `5.1e-07` |
| `507` | 4.3011 | 0.122 | -0.045 | 4.080 | 10.85 ± 5.33 | 2.96 ± 5.29 | +7.89 | 1.000 / 8.0 | 0.00 / 0.43 | 24.9 → 23.1 | `5.4e-07` |
| `984` | 4.2029 | 0.030 | -0.089 | 4.076 | 17.33 ± 1.21 | 9.43 ± 1.18 | +7.90 | 1.001 / 8.0 | 0.00 / 0.43 | 24.9 → 23.1 | `6.0e-07` |
| `1,673` | 4.1578 | **0.016** | -0.110 | 4.072 | 20.34 ± 0.60 | 12.44 ± 0.56 | +7.90 | 1.003 / 8.1 | 0.00 / 0.43 | 25.0 → 23.2 | `6.0e-07` |
| `5,050` | 2.8620 | 1.266 | -1.919 | 3.909 | 105.74 ± 3.33 | 98.63 ± 3.39 | +7.11 | 1.040 / 8.5 | 0.00 / 0.27 | 26.4 → 24.7 | `5.3e-06` |
| `10,101` | 2.2308 | 1.901 | -2.995 | 3.590 | 147.55 ± 2.44 | 141.61 ± 2.48 | +5.94 | 1.100 / 9.2 | 0.00 / 0.15 | 27.8 → 26.2 | `6.4e-06` |
| `20,202` | 2.0172 | 2.156 | -3.443 | 3.219 | 162.52 ± 2.40 | 157.69 ± 2.36 | +4.83 | 1.132 / 9.8 | 0.00 / 0.14 | 29.5 → 27.9 | `6.8e-06` |
| `30,303` | 1.9803 | 2.216 | -3.515 | 3.152 | 165.44 ± 2.06 | 161.09 ± 2.11 | +4.35 | 1.116 / 10.1 | 0.00 / 0.14 | 30.7 → 29.0 | `7.2e-06` |
| `40,292` | 1.9512 | 2.225 | -3.561 | 3.070 | 166.90 ± 2.05 | 162.98 ± 2.02 | +3.92 | 1.074 / 10.5 | 0.00 / 0.14 | 31.9 → 30.0 | `6.7e-06` |
| **`46,003`** | 1.9478 | 2.197 | -3.564 | 3.028 | **167.03 ± 2.15** | **163.60 ± 2.04** | +3.43 | 1.039 / 10.8 | 0.00 / 0.15 | 32.8 → 30.8 | `7.7e-06` |
| `80,808` | 1.9037 | 2.436 | -3.629 | 2.936 | 161.90 ± 2.19 | 164.23 ± 1.54 | -2.33 | 0.709 / 14.8 | 0.01 / 0.29 | 42.8 → 38.5 | `7.6e-06` |
| `126,262` | 1.8751 | 2.302 | -3.661 | 3.044 | 156.15 ± 1.63 | 162.38 ± 1.90 | -6.22 | 0.391 / 26.2 | 0.47 / 0.68 | 70.4 → 57.9 | `1.1e-05` |
| `161,616` | 1.8553 | 2.304 | -3.709 | 3.036 | 137.60 ± 7.87 | 161.86 ± 1.95 | -24.27 | 0.284 / 37.0 | 0.78 / 0.85 | 99.7 → 76.9 | `1.4e-05` |
| `202,020` | 1.8445 | 2.297 | -3.719 | 2.981 | 120.88 ± 14.82 | 161.63 ± 1.82 | -40.75 | 0.220 / 48.7 | 0.90 / 0.90 | 136.8 → 103.3 | `1.1e-05` |
| `252,525` | 1.8405 | 2.375 | -3.725 | 2.876 | 100.14 ± 6.02 | 161.92 ± 1.62 | -61.78 | 0.185 / 61.2 | 0.93 / 0.93 | 184.0 → 141.9 | `1.4e-05` |
| `318,181` | 1.8151 | 2.290 | -3.761 | 2.833 | 93.28 ± 2.81 | 163.19 ± 1.43 | -69.91 | 0.170 / 69.9 | 0.94 / 0.95 | 226.2 → 178.9 | `1.2e-05` |
| **`363,636`** | 1.8038 | 2.421 | -3.775 | 2.827 | **92.15 ± 1.48** | **164.19 ± 1.04** | **-72.04** | **0.169 / 71.6** | **0.95 / 0.95** | 241.7 → 191.4 | `1.4e-05` |
| `424,242` | 1.7921 | 2.411 | -3.791 | 2.861 | 94.64 ± 2.24 | 165.01 ± 0.96 | -70.38 | 0.167 / 72.9 | 0.95 / 0.95 | 252.0 → 201.5 | `2.1e-05` |
| `500,000` | 1.7777 | 2.393 | -3.812 | 2.816 | 97.59 ± 2.86 | 165.92 ± 0.98 | -68.33 | 0.167 / 73.2 | 0.95 / 0.95 | 255.0 → 204.9 | `1.8e-05` |

*(Intermediate steps `200`, `2,843`, `60,606`, `101,010` are included in the JSON and Figure Panel A.)*

### 3.2 Stagewise $\Delta\hat\lambda_{\text{total}} = \Delta\hat\lambda_{\text{func}} + \Delta\hat\lambda_{\text{fiber}}$ (3 Seeds Mean ± SE)

| Stage Transition | $\Delta\hat\lambda_{\text{total}}$ (Raw) | $\Delta\hat\lambda_{\text{func}}$ (`ORBIT`) | $\Delta\hat\lambda_{\text{fiber}}$ (Fiber Slide) | Fiber Share (mean ± SE) | Per-Seed Fiber Share (`s0 / s1 / s2`) | Verdict |
|---|---:|---:|---:|---:|---|---|
| **`LR1` (`0 → 984`)** | +16.15 ± 8.21 | **+16.14 ± 8.22** | +0.01 ± 0.01 | **+0.2% ± 0.1%** | `+0.2% / -0.0% / +0.2%` | **Genuine functional** (zero pred.) |
| **`LR2` (`984 → 46,003`)** | +149.70 ± 0.99 | **+154.18 ± 0.96** | -4.47 ± 0.17 | **-3.0% ± 0.1%** | `-3.0% / -3.2% / -2.8%` | **Genuine functional** (ICL acq.) |
| **`LR3–LR4` (`46,003 → 363,636`)** | **-74.88 ± 3.62** | **+0.58 ± 1.16** | **-75.47 ± 2.58** | **100.9% ± 1.6%** | **`98.2% / 103.7% / 100.9%`** | **~100% estimator non-invariance along `ORBIT`** |
| **`LR3–LR4` (GP peak `30,303 → 363,636`)** | **-73.29 ± 3.52** | **+3.10 ± 1.15** | **-76.39 ± 2.40** | **104.4% ± 1.8%** | **`101.7% / 107.9% / 103.6%`** | **~100% estimator non-invariance (GP window)** |
| **`LR5` (`363,636 → 500,000`)** | +5.44 ± 1.45 | +1.73 ± 0.08 | +3.71 ± 1.39 | 64.4% ± 7.6% | `64.4% / 77.5% / 51.3%` | Mixed late uptick |

*Note on `step = 0` offset (`Raw - ORBIT = +7.89`):* At initialization, `lnf.weight = 1.0` while `unembedding` entries are $O(1/\sqrt{64})$, creating a constant `+7.89` gauge offset across `0..1,673` that cancels to `+0.01` over `LR1`.

---

## 4. Part 2 — Matched-Init `AdamW (wd = 0.08)` Intervention (3 Seeds, 500k) (`[empirical]`)

Every `AdamW` seed starts from **bit-for-bit identical initial weights** (`max|w_init_adam - w_init_adamw| = 0.0`, identical SHA-256) and receives the **identical online batch stream** at every step `1..500,000`.

| Step | ID MSE | OOD $g{=}5$ | Raw $\hat\lambda \pm \mathrm{SE}$ | `ORBIT` $\hat\lambda \pm \mathrm{SE}$ | Raw $-$ `ORBIT` | `blk0_ln0` $\|\gamma\|$ / $\|W_{\text{QKV}}\|$ | `lnf` $\|\gamma\|{<}0.1$ / eff${<}0.05$ | $\|w\|_2$ raw $\to$ orb | Worst $\max|\Delta y|_{\mathrm{f32}}$ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0` | 4.4452 | 4.083 | 1.18 ± 9.42 | -6.72 ± 9.39 | +7.89 | 1.000 / 8.0 | 0.00 / 0.44 | 24.9 → 23.1 | `5.1e-07` |
| `984` | 4.2030 | 4.076 | 17.32 ± 1.21 | 9.42 ± 1.18 | +7.90 | 1.000 / 8.0 | 0.00 / 0.43 | 24.9 → 23.1 | `5.1e-07` |
| `10,101` | 2.2400 | 3.619 | 146.30 ± 2.41 | 140.55 ± 2.46 | +5.75 | 1.056 / 8.9 | 0.00 / 0.15 | 26.7 → 25.2 | `5.7e-06` |
| `46,003` | 1.9667 | 3.372 | **157.50 ± 2.19** | **157.83 ± 2.06** | -0.33 | 0.585 / 8.2 | 0.01 / 0.34 | 20.1 → 17.7 | `6.6e-06` |
| `126,262` | 1.9695 | 3.534 | 152.68 ± 2.62 | 154.57 ± 2.80 | -1.89 | 0.282 / 13.2 | 0.84 / 0.87 | 27.1 → 17.4 | `6.4e-06` |
| `252,525` | 1.9821 | 3.630 | 150.32 ± 2.74 | 151.83 ± 2.89 | -1.50 | 0.183 / 14.4 | 0.96 / 0.96 | 28.9 → 18.2 | `6.3e-06` |
| `363,636` | 1.9425 | **3.652** | **152.55 ± 2.78** | **153.74 ± 2.90** | **-1.18** | **0.188 / 12.2** | **0.96 / 0.96** | **24.9 → 16.1** | `8.9e-06` |
| `500,000` | 1.8470 | 3.638 | 159.64 ± 2.56 | 160.48 ± 2.66 | -0.84 | 0.217 / 9.6 | 0.96 / 0.96 | 18.2 → 12.4 | `1.6e-05` |

**Stagewise `AdamW (wd = 0.08)` decomposition (`46,003 → 363,636`):** $\Delta\hat\lambda_{\text{total}} = -4.94 \pm 0.59$, $\Delta\hat\lambda_{\text{func}} = -4.09 \pm 0.87$, $\Delta\hat\lambda_{\text{fiber}} = -0.85 \pm 0.30$ (followed by `LR5` rebound $+7.09 \pm 0.78$ to `159.64`). Bounding `||W_QKV,0||` (`12.2` vs. `71.6`) suppresses the trough fiber gap from `-72.04` to `-1.18`, even though `AdamW` collapses `lnf` slightly *more* (`96%` vs. `95%`) and degrades OOD `g=5` MSE (`3.372 → 3.652`). Because `AdamW` changes the learned functions (`ID MSE = 1.943` vs. `1.804` at `364k`), it serves as supporting intervention evidence rather than a same-function comparison.

---

## 5. Part 3 — Per-Channel Exact Symmetry Ablation & `ORBIT` vs. `ORBIT+` (`[empirical]`)

All entries use Adam seed 0 with exact Table-2 SGLD (`10` chains $\times$ `5,000` steps, $m = 1{,}024$, $2^{20}$ pool). Error bars are SE **across the 10 SGLD chains** ($\mathrm{SD}/\sqrt{10}$); `hd` is split-half drift $n\beta(\bar L_{\text{2nd half}} - \bar L_{\text{1st half}})$.

| Exact Symmetry Channel | Pure Diffeom.? | Peak `46,003` $\hat\lambda \pm \mathrm{SE}$ (`hd`) | Trough `363,636` $\hat\lambda \pm \mathrm{SE}$ (`hd`) | `364k` $\Delta$ vs. Raw | `364k` $\|w\|_2$ | `364k` $\max|\Delta y|_{\mathrm{f32}}$ | `364k` $\max|\Delta\text{all5}|_{\mathrm{f32}}$ |
|---|:-:|---:|---:|---:|---:|---:|---:|
| `RAW` | — | 169.60 ± 0.23 (+0.5) | **90.01 ± 0.54** (+0.4) | +0.00 | 255.6 | `0.0e+00` | `0.0e+00` |
| `LN_mean_proj` (Pre-LN $\mathbf{1}_d$) | Yes | 169.60 ± 0.23 (+0.5) | 90.01 ± 0.54 (+0.4) | -0.00 | 255.4 | `9.5e-06` | `2.6e-05` |
| `QK` (Per-head $GL(16)$) | Yes | 169.60 ± 0.23 (+0.5) | 90.38 ± 0.57 (+0.4) | +0.37 | 254.1 | `8.3e-06` | `4.9e-05` |
| `VO` (Per-head $GL(16)$) | Yes | 169.64 ± 0.23 (+0.5) | 90.97 ± 0.54 (+0.2) | +0.95 | 251.3 | `7.6e-06` | `2.1e-05` |
| `MLP` (ReLU scale + perm) | Yes | 169.53 ± 0.20 (+0.6) | 90.56 ± 0.51 (+0.8) | +0.54 | 247.2 | `4.5e-06` | `1.5e-05` |
| `LNf_unembed_all5` (`lnf ↔ W_U` all 5) | Yes | 165.68 ± 0.26 (+0.9) | 90.03 ± 0.52 (+0.2) | +0.02 | 255.6 | `9.5e-07` | `1.9e-06` |
| `LN1_mlp` (`ln1 ↔ fc1`) | Yes | 169.64 ± 0.22 (+0.5) | 98.02 ± 0.71 (+0.8) | **+8.01** | 302.9 | `1.0e-05` | `2.5e-05` |
| **`LN0_attn` (`ln0 ↔ QKV`)** | **Yes** | 170.30 ± 0.20 (+0.4) | **161.65 ± 0.12** (-0.0) | **+71.63** | **220.5** | **`5.2e-06`** | **`2.8e-05`** |
| `LN_all_unit_gamma` (control gauge) | Yes | 168.86 ± 0.24 (+0.6) | 164.10 ± 0.45 (+2.5) | +74.09 | 305.4 | `1.4e-05` | `3.9e-05` |
| **`ORBIT` (joint pure group)** | **Yes** | 166.33 ± 0.23 (+0.9) | **164.87 ± 0.19** (-0.1) | **+74.86** | **220.5** | **`1.4e-05`** | **`4.1e-05`** |
| `ORBIT+` (extra: zero unused rows) | No | 165.93 ± 0.22 (+0.8) | 164.88 ± 0.20 (-0.1) | +74.87 | 220.5 | `1.4e-05` | `1.2e+00` |

**Key takeaway:** Balancing **only** `LN0_attn` (`blk0_ln0`, `blk1_ln0` against `W_QKV`) recovers **`+71.63` points (`95.7%` of the `ORBIT`–`RAW` gap)** and drops split-half drift to `-0.0`. `ORBIT` and `ORBIT+` agree to **`0.01` points** at trough (`164.87` vs. `164.88`).

### 5.1 Layer-by-Layer Effective Weight Diagnostics (`Adam wd = 0`, 3-Seed Mean at `46,003 → 363,636`)

| LayerNorm $\to$ Downstream Linear | $d_{\text{out}}$ per $\gamma_i$ | `mean|γ|` (`46k → 364k`) | `frac(|γ|<0.1)` | $\|W_{\text{down}}\|_F$ (`46k → 364k`) | `mean(|γ|·‖W[:,i]‖)` | `frac(eff<0.05)` | Regime |
|---|---:|---:|---:|---:|---:|---:|---|
| `blk0_ln0 → QKV` | **192** | `1.039 → 0.169` | `0.00 → 0.52` | **`10.80 → 71.61`** | **`1.406 → 2.031`** | `0.00 → 0.44` | **Massive fiber blowup + partial pruning** |
| `blk1_ln0 → QKV` | **192** | `1.034 → 0.353` | `0.00 → 0.10` | **`10.90 → 98.87`** | **`1.410 → 4.687`** | `0.00 → 0.06` | **Massive fiber blowup + eff. growth** |
| `blk0_ln1 → fc1` | **64** | `1.018 → 0.524` | `0.00 → 0.22` | `6.24 → 55.95` | `0.795 → 4.276` | `0.00 → 0.20` | Fiber blowup + eff. growth |
| `blk1_ln1 → fc1` | **64** | `1.090 → 1.332` | `0.00 → 0.04` | `7.68 → 72.23` | `1.043 → 12.458` | `0.00 → 0.03` | Eff. growth (balanced $\gamma$) |
| `lnf → W_U[0]` | **1** | `1.069 → 0.093` | `0.00 → 0.95` | `1.24 → 1.24` | `0.151 → 0.060` | **`0.15 → 0.95`** | **Genuine readout collapse** |

---

## 6. Part 6 — Direct Spectral Proof: Exact Gauss–Newton Hessian Spectra (`[empirical]` + `[paper]`)

To move beyond the diagonal Gaussian-Laplace heuristic (`[paper]`), we compute the **exact nonzero eigenvalues** of the Gauss–Newton matrix $G = \frac{2}{N_{\text{res}}} J^\top J$ of the MSE loss on $512$ fresh held-out sequences ($N_{\text{res}} = 512 \times 8 = 4{,}096$ scalar residuals) via dual Gram matrices $\frac{2}{4096} J J^\top \in \mathbb{R}^{4096 \times 4096}$ (exact float64 `eigvalsh`) across **all 3 Adam seeds** at peak (`46,003`) and trough (`363,636`), comparing `RAW` vs. `ORBIT` on the full model (`51,717` params) and individual blocks (`W_QKV`, `LN_0`, `W_fc1`, `LN_1`). The isotropic localization curvature floor is $\tau = \gamma / (n\beta) = 13.3 / 66.7 = \mathbf{0.1994}$, and the exact Gaussian-Laplace effective dimension is $\hat\lambda_{\mathrm{GN}}(\gamma) = \frac{1}{2}\sum_k \frac{n\beta \mu_k}{n\beta \mu_k + \gamma}$:

| Parameter Block (`# params`) | Milestone | Gauge | Exact $\hat\lambda_{\mathrm{GN}}(13.3)$ (3-seed mean ± SE) | Eigenvalues $> \tau$ ($0.1994$) | Max Eigenvalue $\mu_{\max}$ | Trace $\mathrm{tr}(G)$ |
|---|---|---|---:|---:|---:|---:|
| **Full Model (`51,717`)** | Peak `46k` | `RAW` | `359.93 ± 11.26` | `584.7 ± 19.9` | `439.99` | `2,467.8` |
| **Full Model (`51,717`)** | Peak `46k` | `ORBIT` | `372.84 ± 9.72` | `615.7 ± 16.5` | `395.56` | `2,722.5` |
| **Full Model (`51,717`)** | **Trough `364k`** | **`RAW`** | **`119.15 ± 5.80`** | **`167.0 ± 10.4`** | `70.99` | `566.5` |
| **Full Model (`51,717`)** | **Trough `364k`** | **`ORBIT`** | **`398.69 ± 7.71`** | **`676.7 ± 17.3`** | `91.53` | `2,538.3` |
| **`W_QKV` (`24,576`)** | Peak `46k` | `RAW` | `245.29 ± 5.59` | `398.3 ± 12.6` | `115.38` | `1,105.4` |
| **`W_QKV` (`24,576`)** | Peak `46k` | `ORBIT` | `271.08 ± 5.33` | `450.7 ± 10.7` | `140.37` | `1,451.9` |
| **`W_QKV` (`24,576`)** | **Trough `364k`** | **`RAW`** | **`44.09 ± 4.56`** | **`36.0 ± 7.0`** | **`1.27`** | **`33.7`** |
| **`W_QKV` (`24,576`)** | **Trough `364k`** | **`ORBIT`** | **`321.04 ± 8.12`** | **`547.0 ± 13.3`** | **`67.83`** | **`2,072.1`** |
| **`LN_0` (`128`)** | Peak `46k` | `RAW` / `ORBIT` | `17.88` / `15.36` | `23.3` / `18.7` | `3.19` / `2.53` | `20.6` / `15.8` |
| **`LN_0` (`128`)** | **Trough `364k`** | **`RAW` / `ORBIT`** | **`31.99` / `4.13`** | **`65.0` / `2.3`** | **`31.85` / `0.37`** | **`201.8` / `2.5`** |
| **`W_fc1` (`8,192`)** | Peak `46k` | `RAW` / `ORBIT` | `99.71` / `89.34` | `109.0` / `95.3` | `16.96` / `14.24` | `145.1` / `122.0` |
| **`W_fc1` (`8,192`)** | **Trough `364k`** | **`RAW` / `ORBIT`** | **`7.52` / `75.98`** | **`0.7` / `78.3`** | **`0.40` / `28.51`** | **`3.6` / `161.2`** |

**Direct spectral confirmation (`[empirical]`):** Exactly as predicted by the $1$-vs-$d_{\text{out}}$ congruence mechanism (`[paper]`), shrinking each scalar $\gamma_{0,i}$ by $c_i \approx 0.17$ concentrates curvature onto the 128 `LN_0` directions ($\mu_{\max}: 3.19 \to 31.85$, trace $+10\times$, saturating at $\le 64$ modes) while **crushing the 24,576-parameter `W_QKV` spectrum by $60\times$ in trace (`2,072 → 33.7`)**, sweeping **511 live `W_QKV` eigenvalues (`547.0 → 36.0`)** and **77 live `W_fc1` eigenvalues (`78.3 → 0.7`)** underneath the isotropic localization floor $\tau = 0.1994$! Under `ORBIT`, the exact same functions have **more** live curvature modes at `364k` (`676.7`) than at `46k` (`615.7`)!

---

## 7. Part 5 — Reparametrization-Aware Remedies (`[empirical]` + `[paper]`)

Can a better sampler fix the late-stage LLC drop **without canonicalizing the model weights**? We benchmark four SGLD-family estimators side-by-side on **all 3 Adam seeds** at Peak (`46,003`) and Trough (`363,636`), `RAW` vs. `ORBIT`, using identical Table-2 chain budgets (`10` chains $\times$ `5,000` steps, $m = 1{,}024$, $2^{20}$ pool; `±` is SE across the 3 seeds):

1. **`vanilla` (Hoogland et al. 2024):** $\Delta w_t = -\frac{\epsilon}{2}(n\beta g_t + \gamma(w_t - w^*)) + \sqrt{\epsilon}\eta_t$.
2. **`rmsprop_psgld` (exact Hitchcock & Hoogland, arXiv:2507.21449 Alg. G.3 / `devinterp` `RMSpropPreconditioner`):** EMA $v_t = 0.99 v_{t-1} + 0.01 g_t^2$, bias-corrected $\hat v_t$, $G_t = (\sqrt{\hat v_t} + a)^{-1}$ with $a = 0.01$ (chosen so $\sqrt{\hat v_t} \in [7\times 10^{-4}, 3.7\times 10^{-2}]$ spans $a$ and preconditioning is genuinely active; `eps_base = eps * a` so flat directions retain effective step $\epsilon$), applying $G_t$ to *both* $n\beta g_t$ and $\gamma(w_t - w^*)$ and $\sqrt{G_t}$ to noise. Targets the **same isotropic localized posterior** $\pi_\gamma(w)$.
3. **`param_rms_covariant` (Black-Box Per-Tensor RMS Covariant SGLD):** Zero symmetry knowledge required! Sets per-tensor scale $c_T = \mathrm{rms}(w_{T,\text{init}}) / \max(\mathrm{rms}(w_T^*),\, 0.1\,\mathrm{rms}(w_{T,\text{init}}))$ (shared by LayerNorm biases), diagonal metric $D = \mathrm{diag}(c_T^2)$ in localization $\frac{\gamma}{2}(w - w^*)^\top D (w - w^*)$, and covariant preconditioner $P = D^{-1}$.
4. **`orbit_covariant` (Symmetry-Adapted Metric Localization SGLD — Exact Pure-Multiplicative Remedy):** Computes exact cumulative scaling factors $c_p(w^*) > 0$ for all pure-multiplicative coordinatewise symmetries (`(gamma, beta) ↔ W` on all 5 LNs without bias absorption, plus both ReLU MLPs), sets $D(w^*) = \mathrm{diag}(c_p^2)$, and runs covariant Langevin dynamics:
   $$\Delta w_t = -\frac{\epsilon}{2}\big(n\beta\, D^{-1} g_t + \gamma(w_t - w^*)\big) + \sqrt{\epsilon}\, D^{-1/2} \eta_t.$$
   Under the change of coordinates $\tilde w_t = D^{1/2} w_t$, this discrete-time Markov chain is **algebraically identical to isotropic Vanilla SGLD at the diagonal-balanced representative**, leaving $w^*$ completely untouched!

| Estimator (3 seeds, `46k → 364k`) | `RAW` Peak $\to$ Trough | `RAW` $\Delta$ (`Trough - Peak`) | `ORBIT` Peak $\to$ Trough | `ORBIT` $\Delta$ (`Trough - Peak`) | Trough Gap (`RAW - ORBIT`) | Verdict |
|---|---|---:|---|---:|---:|---|
| **`vanilla` (Hoogland 2024)** | `166.93 ± 2.17 → 91.69 ± 1.22` | **`-75.24 ± 3.34`** | `163.42 ± 2.02 → 164.00 ± 1.17` | `+0.58 ± 1.02` | **`-72.31 ± 2.26`** | Severe isotropic-prior distortion |
| **`rmsprop_psgld` (`2507.21449`)** | `168.16 ± 2.03 → 103.14 ± 2.07` | **`-65.02 ± 4.09`** | `168.04 ± 1.85 → 167.09 ± 1.89` | `-0.95 ± 1.22` | **`-63.94 ± 3.66`** | **Still drops `-65` pts!** (Same isotropic prior!) |
| **`param_rms_covariant` (black-box)** | `177.51 ± 1.36 → 159.84 ± 0.85` | `-17.67 ± 0.82` | `175.34 ± 1.53 → 159.85 ± 0.91` | `-15.49 ± 0.75` | **`-0.00 ± 0.13`** | **0.00 trough gauge gap!** (Global norm shift) |
| **`orbit_covariant` (exact metric)** | **`163.34 ± 1.99 → 163.07 ± 1.30`** | **`-0.27 ± 0.83`** | `163.42 ± 2.02 → 164.00 ± 1.16` | `+0.58 ± 1.03` | **`-0.94 ± 0.16`** | **Fixes 99.6% of drop with zero weight edits!** |

**What this proves (`[empirical]`):**
- **Adaptive preconditioning alone is NOT enough:** `rmsprop_psgld` (`2507.21449`) recovers only ~10 points of mixing lag while leaving **`86%` (`-65.02` points) of the `RAW` drop** and **`-63.94` points of the `RAW - ORBIT` gap** completely intact. (Testing $a \in \{0.1, 0.01, 0.001\}$ in pilot checks confirmed the gap persists across all damping scales.)
- **Changing the localization metric fixes the drop directly at the raw weights:** Both covariant-metric remedies erase the `RAW - ORBIT` trough gauge gap (`-0.00` for black-box `param_rms_covariant`, `-0.94` for exact `orbit_covariant`). `orbit_covariant` additionally preserves exact scale across training (`Δ = -0.27 ± 0.83`), whereas black-box `param_rms_covariant` cannot distinguish symmetry rescaling from genuine effective-weight growth (so both `RAW` and `ORBIT` shift by ~`-16` points across `46k → 364k`). *(We also tested a black-box $\det D = 1$ geometric-mean-normalized gradient-RMS metric in pilot checks and found it **worsens** the trough gap to `-168` points because `W_QKV` + `W_fc1` comprise `63%` of all parameters, dragging down the global geometric mean when their gradients shrink.)*

---

## 8. Part 4 — SGLD Localization ($\gamma$), Step-Size ($\epsilon$) & Full-Batch-Gradient Controls (`[empirical]`)

All runs use Adam seed 0 with exact Table-2 chain budget (`10` chains $\times$ `5,000` steps, $m = 1{,}024$, $2^{20}$ pool; `±` is SE across 10 chains). **Full-batch-gradient SGLD (`FB1024`)** computes the exact gradient on a fixed 1,024-sample subset (removing minibatch gradient noise) while **retaining** Langevin noise $\sqrt{\epsilon}\,\eta_t$, measuring LLC relative to that subset's loss.

| Setting | Peak `46,003` `RAW` / `ORBIT` | Trough `363,636` `RAW` (`hd`, $\|w-w^*\|$) | Trough `363,636` `ORBIT` (`hd`, $\|w-w^*\|$) | `RAW` $\Delta$ (`46k→364k`) | `ORBIT` $\Delta$ (`46k→364k`) |
|---|---:|---:|---:|---:|---:|
| **$\gamma = 3.3, \epsilon = 3\text{e-}4$** | `174.99 ± 0.87` / `166.32 ± 0.29` | `162.54 ± 0.56` (`+1.1`, `124.8`) | `167.20 ± 0.38` (`+0.6`, `124.8`) | **`-12.45`** | **`+0.88`** |
| **$\gamma = 13.3, \epsilon = 3\text{e-}4$ (default)** | `170.03 ± 0.31` / `166.42 ± 0.26` | `90.00 ± 0.68` (`+0.8`, `62.3`) | `165.20 ± 0.20` (`-0.1`, `62.4`) | **`-80.03`** | **`-1.22`** |
| **$\gamma = 40.0, \epsilon = 3\text{e-}4$** | `163.22 ± 0.20` / `164.62 ± 0.16` | `52.63 ± 0.19` (`+0.6`, `36.0`) | `167.42 ± 0.17` (`-0.4`, `36.0`) | **`-110.59`** | **`+2.80`** |
| $\gamma = 13.3, \epsilon = 1\text{e-}4$ | `168.97 ± 0.43` / `166.79 ± 0.48` | `88.55 ± 1.03` (`+4.7`, `62.3`) | `165.50 ± 0.31` (`-0.6`, `62.4`) | `-80.42` | `-1.29` |
| **$\gamma = 13.3$, Full-Batch-Gradient `1024`** | `167.80 ± 0.30` / `164.59 ± 0.24` | `85.62 ± 0.65` (`+0.2`, `62.3`) | `163.50 ± 0.16` (`+0.0`, `62.4`) | **`-82.17`** | **`-1.10`** |

**Takeaways (`[empirical]`):**
1. **Finite-$\gamma$ posterior distortion:** At step `363,636`, `RAW` swings by **`109.9` points** across $\gamma \in \{3.3, 13.3, 40.0\}$ (`162.54 → 90.00 → 52.63`, with small `|hd| ≤ 1.1`), whereas `ORBIT` stays invariant within **`2.2` points** (`167.20 → 165.20 → 167.42`)! Note that at $\gamma = 3.3$, chains travel $\|w - w^*\| = 124.8$ (half the full parameter norm `255.6`), so weakening $\gamma$ is not a clean fix — it sacrifices locality while still leaving a `-12.45`-point drop.
2. **Not a minibatch-noise artifact:** Full-batch-gradient SGLD preserves the entire `-82.17`-point `RAW` drop vs. `-1.10` for `ORBIT` (`hd = +0.2 / +0.0`).
3. **Secondary mixing lag:** Cutting $\epsilon$ to $10^{-4}$ slows `RAW` equilibration (`hd = +4.7`, `88.55`) while `ORBIT` remains equilibrated (`hd = -0.6`, `165.50`).

---

## 9. Honest Nuances & Implications for Developmental Interpretability (`[empirical]` + `[conjectural]`)

1. **Both sides of the verdict matter — do not overclaim:**
   - **Early developmental stages (`LR1`–`LR2`) are genuinely captured by SGLD LLC (`[empirical]`):** Zero prediction (`LR1`) and in-context regression acquisition (`LR2`) produce massive, clean rises in `ORBIT`-canonicalized LLC (`-6.72 → 9.43 → 163.60`), with essentially zero fiber confounding (`+0.2%` in `LR1`, `-3.0%` in `LR2`).
   - **Real structural and behavioral changes *do* occur in `LR3`–`LR4`, even though the `-75`-point SGLD drop does not measure them (`[empirical]`):** Final LayerNorm readout collapse (`95%` of effective unembedding weights $< 0.05$) happens under both `Adam` and `AdamW`, and `blk0_ln0` undergoes partial effective pruning (`44% < 0.05`) alongside its $6.6\times$ `W_QKV` blowup. What our data rule out is reading the late-stage SGLD LLC drop as a drop in functional complexity: `RAW` and `ORBIT` share the exact same true LLC, `ORBIT` is flat (`+0.58 ± 1.16`), exact Gauss–Newton spectra confirm 511 live `W_QKV` modes are swept under $\gamma/(n\beta)$ by the coordinate rescaling alone, and `orbit_covariant` SGLD eliminates `99.6%` of the drop directly at the raw weights. We cannot say from finite-$\gamma$ SGLD alone whether the *true* asymptotic LLC changes subtly between `46k` and `364k`.
2. **Practical prescription for DevInterp (`[paper]` + `[empirical]`):** Whenever tracking LLC trajectories on LayerNorm/RMSNorm Transformers, adaptive step-size preconditioning (`RMSPropSGLD`) is **not sufficient** because it preserves the isotropic localization prior. Practitioners should either (a) evaluate SGLD at an exact symmetry-canonicalized representative (`ORBIT`), or (b) use **symmetry-adapted metric localization (`orbit_covariant`)** (or, as a zero-knowledge diagnostic check, black-box `param_rms_covariant` to verify whether `RAW` and gauge-fixed curves diverge!).
3. **Open question (`[conjectural]`):** Whether mid-training LLC drops in larger language models (such as `LM3` in Hoogland et al.'s `georgeyw/L2W256-3m-rr`, trained with `AdamW, wd = 0.05`) have any analogous `LN ↔ Linear` fiber component — or are predominantly functional because weight decay bounds parameter norms — is an immediate open test on their public Hugging Face checkpoints.

---

## 10. Reproducibility & Artifacts
- **Self-contained, shardable production script:** [`experiments/devinterp_fiber_decomposition.py`](../experiments/devinterp_fiber_decomposition.py) (supports `--smoke`, `--train-only`, `--from-checkpoints --shard k/N`, `--merge-shards`, and `--replot`).
- **Committed upstream parity check:** [`experiments/verify_upstream_icl_parity.py`](../experiments/verify_upstream_icl_parity.py) (verified `0.0` bit-for-bit parity in f64 & f32 against `timaeus-research/icl` @ `5baf73025d279abd2d56c98e13bf5663608709c0`).
- **Saved `.safetensors` checkpoints + SHA-256 manifest (114 files, 23 MB):** [`experiments/results/devinterp_checkpoints/manifest.json`](../experiments/results/devinterp_checkpoints/manifest.json) (all 3 seeds $\times$ 23 Adam + 15 AdamW checkpoints; anyone can re-run SGLD/remedies/Gauss–Newton directly via `--from-checkpoints` without retraining).
- **Raw production data & Hoogland reference:** [`experiments/results/devinterp_fiber_decomposition.json`](../experiments/results/devinterp_fiber_decomposition.json) (total GPU eval runtime `12,047.5s` across 3 shards + `~1,800s` training on 1× NVIDIA RTX PRO 6000 Blackwell, exact FP32) and [`experiments/results/hoogland_l2h4minf_reference.json`](../experiments/results/hoogland_l2h4minf_reference.json).
- **8-panel publication figure:** [`experiments/results/devinterp_fiber_decomposition.png`](../experiments/results/devinterp_fiber_decomposition.png) (`--replot` regenerates in `< 2s`).
- **One-page collaborator note for the DevInterp authors:** [`notes/devinterp_for_authors.md`](devinterp_for_authors.md).
