# Track A Empirical Study: Is That LLC Drop Development, or Is It Fiber Sliding?

> **Epistemic status:**
> - **Bilinear fiber stratification & single-head $GL(q)$ gauge invariance (`RRRFiberStratification.lean`, `AttentionLoRA.lean`):** `[LC]` in [`lean/EatYourFiber/RRRFiberStratification.lean`](../lean/EatYourFiber/RRRFiberStratification.lean) and [`lean/EatYourFiber/AttentionLoRA.lean`](../lean/EatYourFiber/AttentionLoRA.lean). These certify the exact RRR / linearized-attention RLCTs; applying fiber canonicalization to full Transformers relies on the `[paper]` exact-symmetry derivations below.
> - **Exact continuous iso-functional symmetries of `DTransformer` (Pre-LN $\mathbf{1}_d$ residual kernel, per-head $GL(d_h)$ $Q$-$K$ & $V$-$O$ balanced SVD, ReLU MLP positive neuronwise rescaling, and coordinatewise $(\gamma_i, \beta_i) \leftrightarrow W_{:,i}$ LayerNorm affine scaling) and the $1$-vs-$d_{\text{out}}$ finite-$\gamma$ localized-Gibbs curvature distortion derivation:** `[paper]` exact algebraic / local Gaussian-Laplace analysis (not formalized in Lean).
> - **All public repository/checkpoint audits, multi-seed `L2H4Minf` reproductions (`Adam, wd=0` and `AdamW, wd=0.08`, 3 seeds each), float64 exactness checks ($\max|\Delta y| \le 1.47\times 10^{-12}$ across all 48 checkpoints), stagewise $\Delta\hat\lambda_{\text{total}} = \Delta\hat\lambda_{\text{func}} + \Delta\hat\lambda_{\text{fiber}}$ decompositions, per-channel symmetry ablations, and $T_{\mathrm{SGLD}} = 5{,}000$ localization/step-size/minibatch/full-batch controls:** `[empirical]` in [`experiments/devinterp_fiber_decomposition.py`](../experiments/devinterp_fiber_decomposition.py) (raw data: [`experiments/results/devinterp_fiber_decomposition.json`](../experiments/results/devinterp_fiber_decomposition.json), 6-panel figure: [`experiments/results/devinterp_fiber_decomposition.png`](../experiments/results/devinterp_fiber_decomposition.png)).
> - **Extrapolation of this LayerNorm-vs-Linear finite-$\gamma$ fiber artifact to larger attention-only language models (e.g., `LM1`–`LM5` in Hoogland et al., 2024) and other DevInterp LLC trajectories:** `[conjectural]`.

---

## Executive Summary

We apply our exact full-model iso-functional fiber decomposition (`experiments/transformer_fiber_surgery.py`, Part 3) to Hoogland, Wang, Farrugia-Roberts, Carroll, Wei & Murfet (2024, TMLR, arXiv:2402.02364), *"Loss Landscape Degeneracy and Stagewise Development in Transformers"* — specifically their **51,717-parameter in-context linear regression Transformer (`L2H4Minf`)**, whose non-monotonic SGLD Local Learning Coefficient (LLC) curve exhibits five plateaus and a striking late-stage LLC drop (`LR1`–`LR5`) interpreted as developmental stages.

Across **3 independent training seeds** of the exact `L2H4Minf` architecture, task distribution, optimizer (`Adam, wd=0`, `OneCycleLR`), and SGLD LLC estimator ($\epsilon = 3\times 10^{-4}, \gamma = 13.3, n\beta = 66.7$), exact float64 iso-functional canonicalization ($\max|\Delta y| \le 1.47\times 10^{-12}$ across all 48 multi-seed checkpoints) yields a **sharp, unambiguous dichotomy between early development and the late-stage LLC drop (`[empirical]`):**

1. **Early stages (`LR1` zero prediction and `LR2` in-context learning acquisition) are $\approx 100\%$ genuine functional development (`[empirical]`):**
   - `LR1 (0 → 1k)`: $\Delta\hat\lambda_{\text{total}} = +28.09 \pm 5.35$ decomposes into $\Delta\hat\lambda_{\text{func}} = +28.11 \pm 5.32$ and $\Delta\hat\lambda_{\text{fiber}} = -0.02 \pm 0.05$ (**fiber share $-0.1\% \pm 0.2\%$**).
   - `LR2 (1k → 25k)`: $\Delta\hat\lambda_{\text{total}} = +127.10 \pm 0.98$ decomposes into $\Delta\hat\lambda_{\text{func}} = +135.54 \pm 1.35$ and $\Delta\hat\lambda_{\text{fiber}} = -8.44 \pm 0.40$ (**fiber share $-6.6\% \pm 0.3\%$**, i.e., genuine functional complexity increases slightly *more* than raw $\hat\lambda$ shows).
2. **The striking `LR3–LR4` LLC drop (`25k → 80k`) is $101.2\% \pm 1.1\%$ pure iso-functional fiber sliding — the canonicalized LLC never drops at all (`[empirical]`):**
   - Across all 3 seeds, raw SGLD LLC plummets by $\Delta\hat\lambda_{\text{total}} = \mathbf{-46.76 \pm 4.52}$ (`160.29 → 113.53`), closely matching Hoogland et al.'s $-47$-point drop (`~170 → ~123`).
   - On the **exact same functions** (canonicalized in float64 to $\max|\Delta y| \le 1.09\times 10^{-12}$), the balanced LLC does not drop: $\Delta\hat\lambda_{\text{func}} = \mathbf{+0.64 \pm 0.55}$ (`159.45 → 160.09`, consistent with zero/flat), leaving $\Delta\hat\lambda_{\text{fiber}} = \mathbf{-47.40 \pm 4.86}$ (**fiber share $101.2\% \pm 1.1\%$**, individually `103.4%`, `100.5%`, `99.8%` across seeds `0, 1, 2`).
3. **Per-channel ablation pinpoints a single exact symmetry as responsible for `96.3%` of the drop: pre-attention `LN_0 ↔ QKV` coordinatewise rescaling (`[empirical]` + `[paper]`):**
   - At step `80k` (seed 0, exact $T_{\mathrm{SGLD}} = 5{,}000$), `RAW` gives `109.06 ± 1.15` vs. `FULL` `160.72 ± 0.38` (`+51.65`).
   - Balancing **only** the two pre-attention LayerNorms (`blk0_ln0`, `blk1_ln0`) against `W_QKV` via the exact symmetry $(\gamma_{0,i}, \beta_{0,i}) \mapsto s_i(\gamma_{0,i}, \beta_{0,i})$, $W_{QKV,:,i} \mapsto s_i^{-1} W_{QKV,:,i}$ ($\max|\Delta y| = 6.22\times 10^{-15}$) lifts LLC from `109.06` to **`158.82 ± 0.31`** (**`+49.76` points, `96.3%` of the full gap**), while reducing split-half drift from `+2.2` to `+0.2`.
   - `LN1_mlp` (`Pre-MLP LN_1 ↔ fc1`) accounts for `+10.94` individually; `LN_mean_proj` (`+0.00`), `QK` (`-0.39`), `VO` (`+0.92`), `MLP` (`-0.22`), and `LNf_unembed` (`-0.30`) are all $< 1$ point.
4. **Mechanistic resolution of Hoogland et al.'s "LayerNorm collapse" conjecture (`[paper]` + `[empirical]`):**
   - Final unembedding LayerNorm (`lnf`) **genuinely collapses** in effective weight (`77%` of $|\gamma_{f,i}| < 0.1$, `82%` of effective products $|\gamma_{f,i} W_{U,0,i}| < 0.05$) alongside OOD magnitude fragility (`g=5` normalized MSE reverses from `2.927` at 50k to `2.986` at 80k). However, `lnf` involves only 64 scalar weights and does **not** drive the $-47$-point LLC drop.
   - By contrast, the earlier pre-attention LayerNorms (`blk0_ln0`, `blk1_ln0`) shrink (`mean|γ|: 0.797 → 0.246` and `0.913 → 0.422`) **in anti-correlation with a $3.2\text{--}3.6\times$ blowup of downstream `W_QKV` (`13.27 → 42.18` and `14.19 → 51.14`)**, keeping `93%–100%` of effective products live (`mean_eff: 1.337 → 1.398` and `1.634 → 2.747`). Because each scalar $\gamma_{0,i}$ multiplies $3 d_{\text{embed}} = 192$ downstream weights (`24,576` parameters total — `47.5%` of the model!), shrinking $\gamma_i$ by $c_i < 1$ scales $192$ Hessian eigenvalues down by $c_i^2$, pushing them below the isotropic SGLD localization floor $\gamma / (n\beta)$ and slashing $\hat\lambda_{\text{SGLD}}$ without changing the implemented function.
5. **Two decisive independent confirmations — finite-$\gamma$ scaling, full-batch SGLD, and weight decay (`[empirical]`):**
   - **Finite-$\gamma$ scaling ($T_{\mathrm{SGLD}} = 5{,}000$, step 80k):** Weakening localization $4\times$ to $\gamma = 3.3$ (closer to the asymptotic $\gamma \to 0$ RLCT) **erases the drop even on `RAW`** (`161.13 ± 0.84` vs. `FULL` `164.11 ± 0.59`, gap `-2.98`), while strengthening localization to $\gamma = 40.0$ blows the gap up to **`-94.78` points** (`RAW` `64.10 ± 0.31` vs. `FULL` `158.88 ± 0.52`).
   - **Minibatch & zero-noise full-batch controls (step 80k, $\gamma = 13.3$, $T_{\mathrm{SGLD}} = 5{,}000$):** `m = 1024` (Hoogland exact) reproduces `m = 256` to $0.2$ points (`RAW 109.96 ± 1.61` vs. `FULL 160.90 ± 0.25`, gap `-50.94`), and **zero-noise full-batch SGLD (`FB = 1024`)** preserves the full gap (`RAW 106.40 ± 1.68` vs. `FULL 152.88 ± 0.20`, gap `-46.48`, split-half drift `-0.2`), proving the drop is an equilibrium finite-$\gamma$ fiber distortion rather than minibatch gradient noise or non-convergence.
   - **Weight-decay counterfactual (`AdamW, wd = 0.08`, 3 seeds):** Bounding weight norms (`||w||: 24.9 → 27.3` vs. `108.2` under `wd=0`) prevents the `LN_0 ↔ QKV` blowup (`||W_QKV,0|| = 14.0` vs. `42.2`) and **eliminates the `LR3–LR4` LLC drop completely** ($\Delta\hat\lambda_{\text{total}} = +2.48 \pm 1.61$, $\Delta\hat\lambda_{\text{func}} = +2.37 \pm 1.81$, $\Delta\hat\lambda_{\text{fiber}} = +0.11 \pm 0.23$) — **even though `AdamW` collapses `lnf` even further (`94% < 0.1`, `94%` effective `< 0.05`) and degrades OOD `g=5` MSE (`3.259 → 3.543`)!**

---

## 1. Public Code, Checkpoints & Exact Reproduction Audit (`[empirical]`)

### 1.1 Public Availability Audit
- **GitHub (`github.com/timaeus-research/icl`):** Publicly accessible; contains the complete model (`src/icl/regression/{dtransformer,model}.py`), synthetic linear-regression task (`tasks.py`), training & optimizer configuration (`config.py`, `sweeps/regression/training-runs/L2H4Minf.yaml`), SGLD implementation (`src/icl/analysis/{sgld,sample,slt}.py`), and pre-computed summary tables (`analysis/L2H4Minf/{llcs,evals_over_time,ln_stats,unembeddings}.pt`) for 5 seeds (`0..4`).
- **Checkpoints (Hugging Face `timaeus` & `georgeyw`, AWS S3):** Only the language-model checkpoints (e.g., `georgeyw/L2W256-3m-rr`, `timaeus/tetrahedron-3m-og`) are publicly hosted on Hugging Face. The in-context linear regression (`L2H4Minf`) checkpoints were stored in a private AWS bucket (`AWS_REGRESSION_BUCKET_NAME` in `config.py`) and are not publicly downloadable.
- **Exact Architecture & Training Recipe Recovered from Code + Paper (arXiv:2402.02364 Table 2, Table 4, Appendix F.2.1):**
  - `InContextRegressionTransformer` (`DTransformer`): $D = 4$, $K = 8$ (`max_tokens = 16`, `token_size = 5`), $L = 2$ layers, $H = 4$ heads ($d_h = 16$), $d_{\text{embed}} = 64$, $d_{\text{mlp}} = 64$, Pre-LayerNorm (`nn.LayerNorm`, $\varepsilon = 10^{-5}$), **ReLU** MLP, `include_output = True`, no biases on `token_embedding`, `postn_embedding`, `QKV`, or `W_O`, biases on `fc1`, `fc2`, and `unembedding`. Note that `DTransformer.forward` applies `x = x + block(x)` where `block(x)` already includes internal residual additions; we preserve this verbatim. Total trainable parameters: **`51,717`** (matching Table F.2.1 exactly; float64 replica matches `timaeus-research/icl` to $\max|\Delta y| = 6.66\times 10^{-16}$).
  - Online data: $w \sim \mathcal{N}(0, I_4)$, $x_k \sim \mathcal{N}(0, I_4)$, $y_k = w^\top x_k + \mathcal{N}(0, 0.125)$, batch size $B = 256$.
  - Optimizer: **`Adam` with `weight_decay = 0.0`**, $\beta_{1,2} = (0.9, 0.999)$, `OneCycleLR(max_lr = 0.003, pct_start = 0.5, anneal_strategy = 'linear', div_factor = T/2 - 1, final_div_factor = T/2 - 1)`.
  - SGLD LLC hyperparameters (Table 2): $\epsilon = 0.0003$, $\gamma = 13.3$, $n\beta = 66.7$, $m = 1024$ (and Figure A.2 shows $m = 256$ matches $m = 1024$, which we verify directly in Part 4), $T_{\mathrm{SGLD}} = 5{,}000$ (`1,000` burn-in + `4,000` draws).

### 1.2 Fidelity of Our 3-Seed `T = 100,000` Reproduction vs. Hoogland et al.'s `T = 500,000` `llcs.pt`
Hoogland et al.'s saved `analysis/L2H4Minf/llcs.pt` (5 seeds, `T = 500k`) shows: step `0` LLC `~0–12` (MSE `4.26`) $\to$ `LR1` `1k` LLC `~19.2` (MSE `4.15`) $\to$ `LR2` peak `40k` LLC `~170.0` (MSE `1.90`) $\to$ `LR3` `126k` LLC `~156.9` (MSE `1.82`) $\to$ `LR4` trough `320k` LLC `~123.4` (`-46.6` drop, MSE `1.77`) $\to$ `LR5` `500k` LLC `~127.9` (`+4.5` uptick, MSE `1.74`).

Our 3-seed `T = 100,000` reproduction (`Adam, wd = 0.0`, same `OneCycleLR` shape, 5× shorter horizon) faithfully reproduces every single quantitative and qualitative stage:
- **LR1 (`0 → 1k`):** `E[y_hat^2]` drops to **`0.017`** at step `500` (context-independent zero prediction); MSE `4.389 → 3.969`; `RAW` LLC rises `5.10 ± 3.65 → 33.19 ± 1.83`.
- **LR2 (`1k → 25k`):** ICL score (`MSE_7 - MSE_0`) plunges `-0.237 → -3.553`, ID MSE `3.969 → 1.935`, `RAW` LLC surges to peak **`162.68 ± 0.23`** (`10k`) / **`160.29 ± 0.89`** (`25k`).
- **LR3–LR4 (`25k → 80k`):** ID MSE keeps improving (`1.935 → 1.843`), while OOD `g=5` normalized MSE reverses (`2.927` at 50k $\to$ `2.986` at 80k), final LayerNorm `lnf` collapses (`0% → 77% < 0.1`, `82%` effective `< 0.05`), and **`RAW` LLC drops by `-46.76 ± 4.52` points (`160.29 → 113.53`)** — matching Hoogland et al.'s `-46.6`-point drop!
- **LR5 (`80k → 100k`):** ID MSE improves to `1.811`, `RAW` LLC stabilizes with a slight uptick (`113.53 → 116.63`, `+3.10 ± 0.62`), matching Hoogland et al.'s `+4.5` uptick.

---

## 2. Full-Model Exact Iso-Functional Canonicalization & Float64 Verification (`[paper]` + `[empirical]`)

We canonicalize every exact continuous symmetry of `InContextRegressionTransformer` in `float64` (using 15 alternating sweeps of the coupled channels to converge to the joint Frobenius-norm-minimizing representative on the exact iso-functional fiber):

1. **Pre-LN $\mathbf{1}_d$ kernel projection (`LN_mean_proj`, `[paper]`):** Because every residual-stream reader (`blk0.ln0`, `blk0.ln1`, `blk1.ln0`, `blk1.ln1`, `lnf`) passes through LayerNorm first, projecting out the constant vector $\mathbf{1}_{64}$ from every column of `token_embedding`, `postn_embedding`, `attention.output`, `fc2.weight`, and `fc2.bias` leaves all outputs identically invariant.
2. **Per-head $GL(d_h)$ $Q$-$K$ balancing (`QK`, `[paper]`):** For each layer and head $h \in \{0,\dots,3\}$, replace $(W_Q^{(h)}, W_K^{(h)})$ with the exact rank-$d_h$ balanced SVD $( \Sigma_{:d_h}^{1/2} U_{:,:d_h}^\top,\; \Sigma_{:d_h}^{1/2} V_{:,:d_h}^\top )$ of $(W_Q^{(h)})^\top W_K^{(h)}$. Minimizes $\|W_Q^{(h)}\|_F^2 + \|W_K^{(h)}\|_F^2$ and collapses any rank-deficient inactive fiber directions.
3. **Per-head $GL(d_h)$ $V$-$O$ balancing (`VO`, `[paper]`):** Replace $(W_O^{(h)}, W_V^{(h)})$ with the exact rank-$d_h$ balanced SVD of $W_O^{(h)} W_V^{(h)}$.
4. **ReLU MLP positive neuronwise rescaling & sorting (`MLP`, `[paper]`):** Because `DTransformer` uses ReLU, scaling $(w_{1,j}, b_{1,j}) \mapsto s_j (w_{1,j}, b_{1,j})$ and $w_{2,:,j} \mapsto s_j^{-1} w_{2,:,j}$ with $s_j = (\|w_{2,:,j}\|_2 / \sqrt{\|w_{1,j}\|_2^2 + b_{1,j}^2})^{1/2}$ minimizes squared norm while preserving outputs identically; neurons are then canonically sorted by descending effective product.
5. **LayerNorm affine $(\gamma, \beta) \leftrightarrow W_{\text{next}}$ coordinatewise balancing (`LNf_unembed`, `LN1_mlp`, `LN0_attn`, `[paper]`):**
   - For biased downstream layers (`lnf → unembedding[0]` and `ln1 → fc1`): absorb $\beta$ into downstream bias ($b' = b + W\beta$, $\beta' = 0$) and balance $|\gamma_i'| = \|W'_{:,i}\|_2 = \sqrt{|\gamma_i|\|W_{:,i}\|_2}$ (setting both to $0$ if the product is $0$, placing dead coordinates at the minimal-LLC singular origin $(0,0)$ rather than the artificially supercritical $(1,0)$ locus of unit-$\gamma$).
   - For biasless `ln0 → QKV`: simultaneous coordinatewise rescaling $(\gamma_{0,i}, \beta_{0,i}) \mapsto s_i(\gamma_{0,i}, \beta_{0,i})$ and $W_{QKV,:,i} \mapsto s_i^{-1} W_{QKV,:,i}$ with $s_i = (\|W_{QKV,:,i}\|_2 / \sqrt{\gamma_{0,i}^2 + \beta_{0,i}^2})^{1/2}$ is **exact even without an attention bias**.
   - *Non-exact move excluded:* Residual-stream global rescaling is **not** exact because PyTorch `LayerNorm` has $\varepsilon = 10^{-5} > 0$, so it is intentionally excluded from our exact canonicalizer.

**Exactness verification across every single move and checkpoint (`[empirical]`):**
- Individual moves at step `80k` (seed 0, float64 on 64 held-out sequences): `LN_mean_proj` `7.11e-15`, `QK` `3.13e-14`, `VO` `3.91e-14`, `MLP` `4.33e-15`, `LNf_unembed` `1.78e-15`, `LN1_mlp` `8.83e-15`, `LN0_attn` `6.22e-15`, `LN_all_unit_gamma` `1.15e-14`, `FULL` `2.48e-13`.
- Global worst-case across **all 48 multi-seed checkpoints** (`27` Adam + `21` AdamW): **$\max|\Delta y| = 1.47\times 10^{-12}$** (Adam max: `1.09e-12`; AdamW max: `1.47e-12`).

---

## 3. Part 1 — Multi-Seed `Adam (wd=0)` Stagewise Decomposition (`[empirical]`)

### 3.1 Full Trajectory (3 Seeds Mean ± SE, $\epsilon=3\times 10^{-4}, \gamma=13.3, n\beta=66.7$, $1{,}000$ burn + $2{,}500$ draws)

| Step | ID MSE | $\mathbb{E}[\hat y^2]$ | $\mathrm{ICL}_{0:7}$ | OOD $g{=}5$ ($\mathrm{MSE}/25$) | Raw $\hat\lambda \pm \mathrm{SE}$ | Balanced $\hat\lambda \pm \mathrm{SE}$ | Raw $-$ Bal | `blk0_ln0` $\|\gamma\|$ / $\|W_{\text{QKV}}\|$ | `lnf` $\|\gamma\|{<}0.1$ / eff${<}0.05$ | $\|w\|_2$ raw $\to$ bal | $\max|\Delta y|$ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0` | 4.3892 | 0.211 | -0.029 | 4.077 | 5.10 ± 3.65 | -4.20 ± 3.71 | +9.30 ± 0.07 | 1.00 / 8.0 | 0.00 / 0.41 | 24.9 → 22.8 | `2.43e-14` |
| `500` | 4.1926 | **0.017** | -0.083 | 4.072 | 18.63 ± 0.25 | 9.33 ± 0.34 | +9.30 ± 0.11 | 1.00 / 8.0 | 0.00 / 0.40 | 24.9 → 22.9 | `2.04e-14` |
| `1,000` | 3.9691 | 0.107 | -0.237 | 4.057 | 33.19 ± 1.83 | 23.91 ± 1.77 | +9.27 ± 0.12 | 1.01 / 8.1 | 0.00 / 0.39 | 25.1 → 23.1 | `5.83e-14` |
| `2,500` | 2.5501 | 1.559 | -2.455 | 3.826 | 125.42 ± 2.01 | 117.62 ± 2.08 | +7.81 ± 0.11 | 1.06 / 8.7 | 0.00 / 0.20 | 26.8 → 24.9 | `5.63e-13` |
| `10,000` | 1.9834 | 2.209 | -3.493 | 3.123 | **162.68 ± 0.23** | 157.52 ± 0.33 | +5.16 ± 0.22 | 1.11 / 10.0 | 0.00 / 0.11 | 29.7 → 28.0 | `7.35e-13` |
| `25,000` | 1.9345 | 2.106 | -3.553 | 2.965 | **160.29 ± 0.89** | **159.45 ± 0.42** | +0.84 ± 0.50 | 0.80 / 13.3 | 0.00 / 0.20 | 37.4 → 34.0 | `1.09e-12` |
| `50,000` | 1.8981 | 2.310 | -3.620 | 2.927 | 131.58 ± 4.05 | 157.36 ± 0.37 | -25.78 ± 4.24 | 0.33 / 30.9 | 0.43 / 0.71 | 76.5 → 59.3 | `7.47e-13` |
| `80,000` | 1.8425 | 2.348 | -3.720 | **2.986** | **113.53 ± 4.22** | **160.09 ± 0.61** | **-46.56 ± 4.83** | **0.25 / 42.2** | **0.77 / 0.82** | 108.2 → 81.0 | `5.22e-13` |
| `100,000` | 1.8112 | 2.360 | -3.768 | 2.977 | 116.63 ± 3.66 | 161.75 ± 0.80 | -45.12 ± 4.46 | 0.24 / 42.9 | 0.78 / 0.82 | 111.1 → 83.6 | `7.65e-13` |

### 3.2 Stagewise $\Delta\hat\lambda_{\text{total}} = \Delta\hat\lambda_{\text{func}} + \Delta\hat\lambda_{\text{fiber}}$ (3 Seeds Mean ± SE)

| Stage Transition | $\Delta\hat\lambda_{\text{total}}$ (Raw) | $\Delta\hat\lambda_{\text{func}}$ (Balanced) | $\Delta\hat\lambda_{\text{fiber}}$ (Fiber Slide) | Fiber Share (mean ± SE) | Per-Seed Fiber Share (`s0 / s1 / s2`) | Verdict |
|---|---:|---:|---:|---:|---|---|
| **`LR1` (`0 → 1k`)** | +28.09 ± 5.35 | **+28.11 ± 5.32** | -0.02 ± 0.05 | **-0.1% ± 0.2%** | `-0.6% / +0.1% / +0.1%` | **Genuine functional** (zero pred.) |
| **`LR2` (`1k → 25k`)** | +127.10 ± 0.98 | **+135.54 ± 1.35** | -8.44 ± 0.40 | **-6.6% ± 0.3%** | `-7.0% / -6.1% / -6.7%` | **Genuine functional** (ICL acq.) |
| **`LR3–LR4` (`25k → 80k`)** | **-46.76 ± 4.52** | **+0.64 ± 0.55** | **-47.40 ± 4.86** | **101.2% ± 1.1%** | **`103.4% / 100.5% / 99.8%`** | **100% iso-functional fiber sliding** |
| **`LR5` (`80k → 100k`)** | +3.10 ± 0.62 | +1.66 ± 0.19 | +1.44 ± 0.48 | 44.2% ± 6.3% | `56.7% / 38.8% / 37.0%` | Small mixed late-stage drift |

*Note on `step = 0` offset (`Raw - Bal = +9.30`):* At initialization, `lnf.weight = 1.0` while `unembedding[0]` entries are small ($\sim 1/\sqrt{64}$), placing inactive readout coordinates on the supercritical `(1, 0)` locus; balancing `lnf` drops init LLC by `~9` points uniformly across steps `0..1k`, cancelling to `-0.02 ± 0.05` across `LR1`.

---

## 4. Part 2 — Weight-Decay Counterfactual (`AdamW, wd = 0.08`, 3 Seeds) (`[empirical]`)

| Step | ID MSE | OOD $g{=}5$ ($\mathrm{MSE}/25$) | Raw $\hat\lambda \pm \mathrm{SE}$ | Balanced $\hat\lambda \pm \mathrm{SE}$ | Raw $-$ Bal | `blk0_ln0` $\|\gamma\|$ / $\|W_{\text{QKV}}\|$ | `lnf` $\|\gamma\|{<}0.1$ / eff${<}0.05$ | $\|w\|_2$ raw $\to$ bal | $\max|\Delta y|$ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0` | 4.3992 | 4.078 | 4.64 ± 3.04 | -4.81 ± 2.93 | +9.45 ± 0.20 | 1.00 / 8.0 | 0.00 / 0.41 | 24.9 → 22.8 | `1.97e-14` |
| `1,000` | 3.9830 | 4.059 | 32.23 ± 4.92 | 22.79 ± 4.73 | +9.44 ± 0.19 | 1.01 / 8.0 | 0.00 / 0.38 | 25.1 → 23.0 | `1.21e-13` |
| `10,000` | 2.0458 | 3.259 | 155.78 ± 1.29 | 151.39 ± 1.35 | +4.39 ± 0.10 | 0.94 / 8.9 | 0.00 / 0.18 | 25.1 → 23.7 | `6.49e-13` |
| `25,000` | 1.9833 | 3.358 | **151.62 ± 1.56** | **153.28 ± 1.72** | -1.67 ± 0.16 | 0.46 / 11.1 | 0.06 / 0.61 | 24.0 → 19.0 | `6.75e-13` |
| `60,000` | 1.9296 | 3.526 | 151.68 ± 0.34 | 153.60 ± 0.18 | -1.92 ± 0.18 | 0.27 / 15.9 | 0.90 / 0.91 | 31.6 → 19.7 | `5.65e-13` |
| `80,000` | 1.9030 | **3.543** | **154.10 ± 0.11** | **155.65 ± 0.10** | **-1.55 ± 0.08** | **0.28 / 14.0** | **0.94 / 0.94** | **27.3 → 17.7** | `6.08e-13` |
| `100,000` | 1.8290 | 3.458 | 159.12 ± 0.39 | 160.48 ± 0.39 | -1.35 ± 0.06 | 0.29 / 12.4 | 0.94 / 0.95 | 23.5 → 15.9 | `1.47e-12` |

**Stagewise `AdamW (wd = 0.08)` decomposition (`25k → 80k`):** $\Delta\hat\lambda_{\text{total}} = \mathbf{+2.48 \pm 1.61}$, $\Delta\hat\lambda_{\text{func}} = \mathbf{+2.37 \pm 1.81}$, $\Delta\hat\lambda_{\text{fiber}} = \mathbf{+0.11 \pm 0.23}$.

**Why this counterfactual is decisive:** Under `AdamW (wd = 0.08)`, both **final LayerNorm collapse (`94% < 0.1`, `94%` effective `< 0.05`)** and **OOD `g=5` magnitude degradation (`3.259 → 3.543`)** are *stronger* than under `Adam (wd = 0)` — yet **there is no `LR3–LR4` LLC drop at all** (`151.62 → 154.10`) because weight decay prevents the `||W_QKV||` blowup (`14.0` vs. `42.2`) and keeps `RAW` within `1.55` points of `FULL`!

---

## 5. Part 3 — Per-Channel Exact Symmetry Ablation & Mechanistic Proof (`[empirical]` + `[paper]`)

### 5.1 Per-Channel Ablation (Adam Seed 0, Exact Hoogland $T_{\mathrm{SGLD}} = 5{,}000$: 1,000 burn + 4,000 draws)

| Exact Symmetry Channel | Step `25k` $\hat\lambda \pm \mathrm{SE}$ (`hd`) | `25k` $\Delta$ vs Raw | Step `80k` $\hat\lambda \pm \mathrm{SE}$ (`hd`) | `80k` $\Delta$ vs Raw | `80k` $\|w\|_2$ | `80k` $\max|\Delta y|$ |
|---|---:|---:|---:|---:|---:|---:|
| `RAW` | 159.04 ± 0.67 (-0.3) | +0.00 | **109.06 ± 1.15** (+2.2) | +0.00 | 106.6 | `0.00e+00` |
| `LN_mean_proj` (Pre-LN $\mathbf{1}_d$) | 159.04 ± 0.67 (-0.3) | -0.00 | 109.06 ± 1.15 (+2.2) | +0.00 | 106.6 | `7.11e-15` |
| `QK` (Per-head $GL(16)$) | 159.03 ± 0.63 (-0.2) | -0.01 | 108.67 ± 0.91 (+0.7) | -0.39 | 106.1 | `3.13e-14` |
| `VO` (Per-head $GL(16)$) | 159.26 ± 0.67 (-0.2) | +0.22 | 109.98 ± 0.34 (+2.9) | +0.92 | 106.0 | `3.91e-14` |
| `MLP` (ReLU neuron scale) | 159.20 ± 0.69 (-0.3) | +0.15 | 108.84 ± 1.13 (+1.0) | -0.22 | 104.8 | `4.33e-15` |
| `LNf_unembed` (`lnf ↔ W_U[0]`) | 156.52 ± 0.64 (-0.2) | -2.52 | 108.77 ± 1.16 (+2.8) | -0.30 | 106.6 | `1.78e-15` |
| `LN1_mlp` (`ln1 ↔ fc1`) | 159.09 ± 0.68 (-0.3) | +0.04 | 120.01 ± 2.92 (+5.9) | **+10.94** | 110.9 | `8.83e-15` |
| **`LN0_attn` (`ln0 ↔ QKV`)** | 161.66 ± 0.61 (-0.6) | +2.61 | **158.82 ± 0.31** (+0.2) | **+49.76** | **86.9** | **`6.22e-15`** |
| `LN_all_unit_gamma` (control) | 161.28 ± 0.80 (-0.2) | +2.24 | 163.24 ± 1.17 (+0.6) | +54.17 | 101.8 | `1.15e-14` |
| **`FULL` (joint norm-min)** | 158.93 ± 0.74 (-0.3) | -0.11 | **160.72 ± 0.38** (-0.1) | **+51.65** | **80.6** | **`2.48e-13`** |

### 5.2 Layer-by-Layer Effective Weight Diagnostics (Adam `wd = 0`, 3-Seed Mean at `25k → 80k`)

| LayerNorm $\to$ Downstream Linear | Params $d_{\text{out}}$ per $\gamma_i$ | `mean|γ|` (`25k → 80k`) | `frac(|γ|<0.1)` (`25k → 80k`) | $\|W_{\text{down}}\|_F$ (`25k → 80k`) | Effective `mean(|γ|·‖W[:,i]‖)` (`25k → 80k`) | `frac(eff<0.05)` (`25k → 80k`) | Regime |
|---|---:|---:|---:|---:|---:|---:|---|
| `blk0_ln0 → QKV` | **192** | `0.797 → 0.246` | `0.00 → 0.19` | `13.27 → 42.18` | **`1.337 → 1.398`** | `0.00 → 0.07` | **Pure fiber blowup** |
| `blk1_ln0 → QKV` | **192** | `0.913 → 0.422` | `0.00 → 0.01` | `14.19 → 51.14` | **`1.634 → 2.747`** | `0.00 → 0.00` | **Pure fiber blowup** |
| `blk0_ln1 → fc1` | **64** | `0.861 → 0.453` | `0.00 → 0.20` | `7.79 → 26.32` | **`0.848 → 1.641`** | `0.00 → 0.14` | **Pure fiber blowup** |
| `blk1_ln1 → fc1` | **64** | `1.083 → 0.959` | `0.00 → 0.02` | `9.92 → 30.03` | `1.340 → 3.801` | `0.00 → 0.02` | Growth (balanced $\gamma$) |
| `lnf → W_U[0]` | **1** | `1.014 → 0.230` | `0.00 → 0.77` | `1.12 → 1.25` | **`0.129 → 0.072`** | **`0.20 → 0.82`** | **Genuine readout collapse** |

### 5.3 Mathematical Mechanism: The $1$-vs-$d_{\text{out}}$ Finite-$\gamma$ Curvature Distortion (`[paper]`)
For a coordinate $i$ entering a linear layer via $\gamma_i W_{:, i} \in \mathbb{R}^{d_{\text{out}}}$, rescaling $(\gamma_i, W_{:, i}) \mapsto (c_i \gamma_i, c_i^{-1} W_{:, i})$ with $c_i \ll 1$ leaves the function identical, yet transforms the local Gauss–Newton Hessian eigenvalues $H_k$ asymmetrically:
- The **single** direction $\delta \gamma_i$ sees its curvature multiplied by $c_i^{-2} \gg 1$ (already saturated at $\frac{1}{2} \frac{n\beta H}{n\beta H + \gamma} \approx \frac{1}{2}$).
- All **$d_{\text{out}}$** directions of $\delta W_{:, i}$ ($d_{\text{out}} = 192$ for `QKV`, $64$ for `fc1`) see their curvature multiplied by **$c_i^2 \ll 1$** ($\sim 10\text{--}16\times$ flatter for `blk0_ln0` where $c_i \approx 0.25$!).
Under DevInterp's isotropic Gaussian localization $\frac{\gamma}{2}\|w - w^*\|_2^2$, each mode contributes approximately $\frac{1}{2}\frac{n\beta H_k}{n\beta H_k + \gamma}$ to $\hat\lambda$. Pushing $d_{\text{out}} = 192$ live curvature eigenvalues below $\gamma / (n\beta)$ collapses their contribution toward $0$ — creating a massive apparent LLC drop solely from non-orthogonal coordinate rescaling along an exact iso-functional fiber, while simultaneously inflating the local condition number by $c_i^{-4}$ and slowing SGLD mixing!

---

## 6. Part 4 — SGLD Localization ($\gamma$), Step-Size ($\epsilon$), $m{=}1024$ & Full-Batch Controls (`[empirical]`)

All runs use Adam seed 0 at exact Hoogland chain length $T_{\mathrm{SGLD}} = 5{,}000$ (`1,000` burn + `4,000` draws, 4 chains, $n\beta = 66.7$):

| Checkpoint | Setting | `RAW` $\hat\lambda \pm \mathrm{SE}$ (`hd`) | `RAW` Trace `[300:900 / 1k:2k / 4k:5k]` | `FULL` $\hat\lambda \pm \mathrm{SE}$ (`hd`) | `FULL` Trace `[1k:2k / 4k:5k]` | `Raw - Full` |
|---|---|---:|---|---:|---|---:|
| **`25k` (LR2 peak)** | $\gamma=3.3, \epsilon=3\text{e-}4, m=256$ | 168.02 ± 0.86 (+2.2) | `164.7 / 167.8 / 169.1` | 160.50 ± 0.44 (+1.1) | `160.2 / 160.7` | +7.52 |
| **`25k` (LR2 peak)** | $\gamma=13.3, \epsilon=3\text{e-}4, m=256$ | 159.27 ± 0.28 (+0.1) | `159.2 / 159.9 / 158.5` | 158.92 ± 0.26 (+0.4) | `159.1 / 158.9` | +0.35 |
| **`25k` (LR2 peak)** | $\gamma=40.0, \epsilon=3\text{e-}4, m=256$ | 131.55 ± 0.66 (+1.5) | `129.2 / 131.6 / 131.2` | 154.09 ± 0.21 (+0.9) | `153.5 / 154.6` | -22.55 |
| **`25k` (LR2 peak)** | $\gamma=13.3, \epsilon=1\text{e-}4, m=256$ | 158.34 ± 0.45 (-0.0) | `150.9 / 158.5 / 157.6` | 159.23 ± 0.28 (-1.0) | `160.1 / 158.3` | -0.90 |
| **`80k` (LR4 trough)** | **$\gamma=3.3, \epsilon=3\text{e-}4, m=256$** | **161.13 ± 0.84** (+1.5) | `146.1 / 160.6 / 160.9` | **164.11 ± 0.59** (+1.6) | `162.7 / 163.7` | **-2.98** |
| **`80k` (LR4 trough)** | **$\gamma=13.3, \epsilon=3\text{e-}4, m=256$** | **110.17 ± 1.60** (+0.7) | `106.9 / 112.0 / 110.2` | **160.81 ± 0.26** (-0.4) | `161.0 / 159.7` | **-50.64** |
| **`80k` (LR4 trough)** | **$\gamma=40.0, \epsilon=3\text{e-}4, m=256$** | **64.10 ± 0.31** (+1.2) | `64.7 / 63.8 / 64.7` | **158.88 ± 0.52** (-0.8) | `159.5 / 158.7` | **-94.78** |
| **`80k` (LR4 trough)** | $\gamma=13.3, \epsilon=1\text{e-}4, m=256$ | 104.76 ± 2.25 (+3.3) | `85.3 / 103.6 / 108.0` | 161.42 ± 0.63 (-0.3) | `161.9 / 160.2` | -56.66 |
| **`80k` (LR4 trough)** | **$\gamma=13.3, \epsilon=3\text{e-}4, m=1024$** | **109.96 ± 1.61** (+0.8) | `106.9 / 111.6 / 110.1` | **160.90 ± 0.25** (-0.2) | `160.8 / 160.2` | **-50.94** |
| **`80k` (LR4 trough)** | **$\gamma=13.3, \epsilon=3\text{e-}4$, Full-Batch `1024`** | **106.40 ± 1.68** (-0.2) | `102.8 / 108.6 / 105.6` | **152.88 ± 0.20** (-0.4) | `153.0 / 151.9` | **-46.48** |

**Key diagnostic takeaways (`[empirical]`):**
1. **Equilibrium finite-$\gamma$ distortion (not merely slow mixing):** At step `80k` across all three $\gamma$ values, the `1k:2k` and `4k:5k` windows are flat (`|hd| ≤ 1.6`), proving chain equilibrium. Yet `RAW` swings by **`97` points** across $\gamma \in \{3.3, 13.3, 40.0\}$ (`161.13 → 110.17 → 64.10`) while exact iso-functional `FULL` varies by only `5.2` points (`164.11 → 160.81 → 158.88`)!
2. **Not a minibatch-noise artifact:** Switching from `m = 256` to Hoogland et al.'s exact `m = 1024` changes `RAW` by `0.21` and `FULL` by `0.09`; switching to **zero-noise full-batch SGLD (`FB = 1024`)** preserves the `-46.48`-point fiber deficit with `hd = -0.2`.
3. **Secondary step-size sensitivity:** At $\epsilon = 10^{-4}$, the ill-conditioned `RAW` checkpoint lags early (`300:900 = 85.3`, `hd = +3.3`), whereas `FULL` converges immediately (`161.42`, `hd = -0.3`).

---

## 7. Honest Nuances & Implications for Developmental Interpretability (`[empirical]` + `[conjectural]`)

1. **Both sides of the verdict are real — do not overclaim:**
   - **Developmental structure in `LR1`–`LR2` is genuinely captured by SGLD LLC (`[empirical]`):** Zero prediction (`LR1`) and in-context regression acquisition (`LR2`) drive massive, clean rises in *canonicalized* LLC (`-4.20 → 23.91 → 159.45`), with essentially zero fiber confounding (`-0.1%` in `LR1`, `-6.6%` in `LR2`).
   - **Real structural/behavioral changes *do* happen in `LR3`–`LR4`, even though the LLC *drop* does not measure them (`[empirical]`):** Final LayerNorm readout collapse (`82%` of effective unembedding weights $< 0.05$) and OOD magnitude fragility (`g=5` MSE degradation) genuinely occur in `LR3–LR4` (under *both* `Adam` and `AdamW`). What fails is attributing the **`-47`-point SGLD LLC drop** to functional simplification: our canonicalization + `AdamW` counterfactual prove that the `-47`-point LLC drop is caused `101%` by anti-correlated `LN_0 ↔ QKV` fiber sliding under finite-$\gamma$ isotropic localization.
2. **Practical prescription for DevInterp (`[paper]` + `[empirical]`):** Whenever tracking DevInterp SGLD LLC across training of LayerNorm/RMSNorm Transformers (especially with `wd = 0` or large LR schedules), practitioners should **canonicalize exact continuous symmetries (especially `LN ↔ Linear` scaling) before estimating LLC** — or equivalently evaluate $\Delta\hat\lambda_{\text{func}}$ alongside $\Delta\hat\lambda_{\text{total}}$ — to separate genuine developmental complexity transitions from isotropic-prior fiber-sliding artifacts.
3. **Open question (`[conjectural]`):** Hoogland et al.'s 2-layer language model (`georgeyw/L2W256-3m-rr`, trained with `AdamW, wd = 0.05`) also exhibits a mid-training LLC drop (`LM3`); testing our exact full-model canonicalizer directly on those public Hugging Face checkpoints is an immediate natural next step.

---

## 8. Reproducibility & Artifacts
- Self-contained code: [`experiments/devinterp_fiber_decomposition.py`](../experiments/devinterp_fiber_decomposition.py) (end-to-end runtime `6,512.0s` on Apple Silicon MPS + 8 CPU cores; `--replot` regenerates the figure directly from JSON in `< 2s`).
- Raw multi-seed data: [`experiments/results/devinterp_fiber_decomposition.json`](../experiments/results/devinterp_fiber_decomposition.json).
- 6-panel summary figure: [`experiments/results/devinterp_fiber_decomposition.png`](../experiments/results/devinterp_fiber_decomposition.png).
