# Track A: The Late LLC Drop in Hoogland et al.'s Regression Transformer Follows a LayerNorm Rescaling Orbit

> **Epistemic status**
> - `[LC]`: bilinear fiber stratification and single-head $GL(q)$ gauge invariance ([`RRRFiberStratification.lean`](../lean/EatYourFiber/RRRFiberStratification.lean), [`AttentionLoRA.lean`](../lean/EatYourFiber/AttentionLoRA.lean)). These cover reduced-rank regression and linearized attention only. Nothing about the full Transformer in this note is formalized.
> - `[paper]` (derived by hand, not in Lean): that each move in `ORBIT` (§2) is an exact loss-preserving diffeomorphism of the 51,717-dimensional parameter space; the curvature-suppression mechanism (§6); and the change of variables showing that `orbit_covariant` SGLD is vanilla SGLD at the diagonally balanced point (§7).
> - `[empirical]`: everything else. That covers the upstream parity check, the 3-seed 500k reruns, the SGLD estimates, the per-channel ablation, the Gauss–Newton spectra, the estimator comparison and the controls. Code: [`devinterp_fiber_decomposition.py`](../experiments/devinterp_fiber_decomposition.py), [`verify_upstream_icl_parity.py`](../experiments/verify_upstream_icl_parity.py). Data: [`devinterp_fiber_decomposition.json`](../experiments/results/devinterp_fiber_decomposition.json). Figure: [`devinterp_fiber_decomposition.png`](../experiments/results/devinterp_fiber_decomposition.png). Checkpoints: [`devinterp_checkpoints/manifest.json`](../experiments/results/devinterp_checkpoints/manifest.json).
> - `[conjectural]`: whether anything similar happens in their language models (`LM1`–`LM5`) or in other DevInterp LLC curves.

## The Short Version

The true LLC is invariant under loss-preserving diffeomorphisms of parameter space (Lau et al., arXiv:2308.12108). The SGLD estimator is not. Its localization term $\frac{\gamma}{2}\|w - w^*\|_2^2$ pulls equally hard along every coordinate. So moving $w^*$ along a symmetry orbit changes the estimate, even though the function and the true LLC stay the same.

In our reruns of Hoogland et al.'s `L2H4Minf` (arXiv:2402.02364), Adam (`wd = 0`) moves a long way along one such orbit between steps `46k` and `364k`. The pre-attention LayerNorm $\gamma$ in block 0 shrinks about $6\times$ while $\|W_{\text{QKV}}\|$ grows about $6.6\times$, so the product of the two does not shrink. Hoogland et al. see this LayerNorm collapse too, "particularly the layer normalization module before the first attention block" (§7.3). They conjecture that it explains part of the LR3–LR4 LLC decrease through added degeneracy, and note that the argument "hinges on the collapse being complete" (App. D.2.6). In our runs the block-0 collapse is only partial: for most coordinates the downstream weights grow to compensate. That makes it a loss-preserving reparametrization, which leaves the true LLC unchanged, and the drop in the estimate follows it. Block 1's pre-attention LayerNorm does the same ($\gamma$ shrinks about $3\times$, $\|W_{\text{QKV}}\|$ grows about $9\times$).

The scales show why the estimator reacts. At $\gamma = 13.3$ the chains end up about `62.3` from $w^*$, essentially what the localization term alone gives ($\sqrt{51{,}717/13.3} = 62.4$), or about `0.27` per coordinate. At the trough, block 0's $\gamma$ entries average `0.17`, so the raw chain moves across $\gamma \approx 0$ in many coordinates.

**What we find** (3 seeds unless stated):

1. **The drop disappears at a function-identical point.** Between the peak (`46,003`) and trough (`363,636`), the raw SGLD estimate falls from `167.0` to `92.2`. We then map each checkpoint to a balanced point on its symmetry orbit (`ORBIT`, §2). The map is exact, and the readout changes by at most `2.1e-5` after casting to float32. At the balanced points the estimate goes from `163.6` to `164.2`. Balancing only the two pre-attention `LN_0 ↔ W_QKV` pairs (blocks 0 and 1) recovers about 96% of the trough gap (seed 0).
2. **The gap is a finite-$\gamma$ effect, and it shrinks as $\gamma$ weakens** (seed 0, trough). With $\gamma = 40, 13.3, 3.3$ the raw estimate is `52.6`, `90.0`, `162.5`, while the balanced estimate stays between `165.2` and `167.4`. This is the main reason to think the balanced estimate is the closer one to the true LLC. It is not a strong argument. At $\gamma = 3.3$ the chains wander $\|w - w^*\| \approx 125$, so they are no longer very local. A weakly localized raw chain can also slide along the flat orbit toward the balanced region, which makes the convergence partly expected. The Gauss–Newton rank check in §6 would not depend on the gauge.
3. **RMSProp-preconditioned SGLD cannot fix this, by construction.** Preconditioning changes how the chain moves but, up to the usual bias of adaptive preconditioning without a correction term, not the distribution it samples, and that distribution still has the isotropic localization term. Empirically, the raw estimate still drops `65` points under RMSProp-pSGLD (Hitchcock & Hoogland, arXiv:2507.21449).
4. **Changing the localization metric does fix it.** We build a diagonal metric from the balancing scales and run SGLD at the raw weights. The estimate is then flat (`−0.27 ± 0.83`). By a change of variables this is the same computation as vanilla SGLD at the diagonally balanced point (§7). So it is not independent evidence; it is a way to get the balanced estimate without editing the checkpoint.
5. **The early stages are unaffected.** Balancing changes the LR1 and LR2 rises by 3% or less.

**What we do not claim:**

- **That the true LLC is flat from `46k` to `364k`.** Both estimates are finite-$\gamma$, and how much the estimate moves at fixed function depends on the metric. Vanilla SGLD at the balanced point gives `+0.6`. A black-box per-tensor RMS metric gives about `−16` on both raw and balanced weights. The robust results are that the raw-versus-balanced gap closes, and that most of the raw drop goes with it.
- **That this explains Hoogland et al.'s own curve.** Our raw drop is `75` points and theirs is `49`, and the curves separate after about `150k`, inside the window in question. If their runs drift less along the orbit, more of their drop could be real. Comparing weight norms at the trough (ours: `241.7`) would be a first check; running the balancing on their original checkpoints, if they still have them, would settle it.
- **That nothing real happens in LR3–LR4.** As they report, the final LayerNorm readout genuinely collapses: `95%` of effective unembedding weights fall below `0.05`. About `44%` of block 0's pre-attention coordinates are also effectively pruned, and robustness to out-of-distribution inputs changes. The raw estimate's drop just does not track these changes. Balancing `lnf` moves the trough estimate by `0.02`.

### Main numbers

| `46,003 → 363,636` (3 seeds, mean ± SE across seeds) | Raw weights | Balanced (`ORBIT`) | Trough gap (raw − balanced) |
|---|---:|---:|---:|
| Vanilla SGLD (Table 2 settings) | `167.03 → 92.15` ($\Delta = -74.88 \pm 3.62$) | `163.60 → 164.19` ($\Delta = +0.58 \pm 1.16$) | `−72.04` |
| RMSProp-pSGLD ($a = 0.01$) | $\Delta = -65.02 \pm 4.09$ | $\Delta = -0.95 \pm 1.22$ | `−63.94 ± 3.66` |
| Metric-localized SGLD (`orbit_covariant`, §7) | $\Delta = -0.27 \pm 0.83$ | $\Delta = +0.58 \pm 1.03$ | `−0.94 ± 0.16` |
| Per-tensor RMS metric (`param_rms_covariant`, §7) | $\Delta = -17.67 \pm 0.82$ | $\Delta = -15.49 \pm 0.75$ | `−0.00 ± 0.13` |
| Matched-init AdamW (`wd = 0.08`), §4 — *different functions* | $\Delta = -4.94 \pm 0.59$ | $\Delta = -4.09 \pm 0.87$ | `−1.18` |

The vanilla row comes from the main trajectory sweep (§3). The §7 estimator benchmark reran vanilla with its own chains and got $\Delta = -75.24 \pm 3.34$.

### Terms used in this note

- **`ORBIT`** composes only invertible, loss-preserving maps that keep all 5 output channels. These are: positive coordinatewise LayerNorm ↔ next-linear rescalings, per-head $GL(16)$ balancing of $Q$–$K$ and $V$–$O$, positive ReLU neuron rescalings and permutations, and exact flat translations (the pre-LN $\mathbf{1}_d$ direction and bias absorption). For such a map $g$, $K \circ g = K$, so $\lambda(w) = \lambda(g \cdot w)$.
- **`ORBIT+`** adds moves that are not diffeomorphisms: zeroing the 4 unused unembedding rows and rebalancing `lnf` against row 0 only. It is reported separately and changes the trough estimate by `0.01`.
- **"Balanced estimate"** means the SGLD estimate at the `ORBIT` representative. It is itself a finite-$\gamma$, gauge-dependent number. At the seed-0 trough, three different exact gauges give `161.65` (`LN0_attn` only), `164.10` (unit $\gamma$) and `164.87` (`ORBIT`).
- **$\Delta\hat\lambda_{\text{func}}$ and $\Delta\hat\lambda_{\text{fiber}}$** in the tables below are shorthand. $\Delta\hat\lambda_{\text{func}}$ is the change in the balanced estimate. $\Delta\hat\lambda_{\text{fiber}} = \Delta\hat\lambda_{\text{total}} - \Delta\hat\lambda_{\text{func}}$ is the part of the raw change that balancing removes. The split depends on the gauge and is not a decomposition of the true LLC.
- **Prior work.** Lau et al. (App. D) note that the SGLD estimator approximates the LLC only under conditions on $\gamma$. Their rescaling experiment (App. O, a two-layer ReLU network) used a preconditioner matched to the rescaling. We have not found prior work showing a Transformer training run that drifts along an exact `LN ↔ Linear` orbit far enough to produce a reported LLC drop.

---

## 1. Upstream Parity, Recipe and Comparison with the Released Curve (`[empirical]`)

### 1.1 Upstream Parity and Recipe
- **Upstream parity check ([`experiments/verify_upstream_icl_parity.py`](../experiments/verify_upstream_icl_parity.py)):** Directly imports `InContextRegressionTransformer` from `github.com/timaeus-research/icl` (pinned commit `5baf73025d279abd2d56c98e13bf5663608709c0`), copies exact weights into our replica, and evaluates 512 fresh sequences (`seed = 2026`). Result (embedded in JSON): **`51,717` parameters, identical `state_dict` keys, `max_delta_y_f64 = 0.0`, `max_delta_y_f32 = 0.0`, `max_delta_all_5ch_f64 = 0.0`, `max_delta_all_5ch_f32 = 0.0`**.
- **Recipe: the public YAML and the paper disagree.** The public `sweeps/regression/training-runs/L2H4Minf.yaml` sets `max_examples: 16` and leaves optimizer and scheduler unspecified (so `src/icl/regression/config.py` defaults to `SGD`, `weight_decay = 1e-4`). By contrast, the published paper (arXiv:2402.02364 Table 4 & App. F.2) specifies $K = 8$ (`max_tokens = 16`, `51,717` params), `Adam (wd = 0.0)`, `OneCycleLR(max_lr = 0.003, pct_start = 0.5, anneal_strategy = 'linear', div_factor = T/2 - 1, final_div_factor = T/2 - 1)`, $T = 500{,}000$, $B = 256$, $D = 4$, $\sigma^2 = 0.125$. We follow the paper because `analysis/L2H4Minf/llcs.pt` logs exactly 8 token positions (`0..7`, matching $K = 8$) and its MSE/LLC trajectory matches `Adam (wd = 0.0)`.
- **SGLD settings (Hoogland et al. Table 2):** $\epsilon = 3\times 10^{-4}$, $\gamma = 13.3$, $n\beta = 66.7$, **10 chains $\times$ 5,000 steps** (`1,000` burn-in + `4,000` draws), minibatch **$m = 1{,}024$**, fixed deterministic pool of **$2^{20} = 1{,}048{,}576$** sequences (`pool_seed = 12345`), exact FP32 (`tf32=False`) on an NVIDIA RTX PRO 6000 Blackwell GPU.

### 1.2 Comparison with Hoogland et al.'s Released 5-Seed `llcs.pt`
We apply Hoogland et al.'s exact GP-smoothed LLC-derivative stage-boundary procedure (`icl/notebooks/main-figures.ipynb` cell 51 & App. B.1: RBF + WhiteKernel GP on $\log(t+1)$ for $t \ge 100$, differentiated w.r.t. $\log(t+1)$) directly to both our 3-seed mean raw curve (23 checkpoints aligned to `llcs.pt`) and Hoogland et al.'s released 5-seed `llcs.pt` (190 checkpoints):

| Metric | Hoogland `llcs.pt` (5 seeds, 500k) | Our rerun (`Adam, wd=0`, 3 seeds) | Earlier 100k Retrain (`T=100k`, cheaper chain) |
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

*Note on the match.* Through step `126,262` our curve is within `0.7–3.4` points of the released 5-seed mean, and the peak and trough land on the same logged steps. After about `150k` it falls further: our trough is `92.2` against their `121.2`, and our peak-to-trough drop is `75` points against their `49`. At the trough our raw parameter norm is `241.7`, so our runs have drifted a long way along the `LN_0 ↔ QKV` orbit, but we have not seen their checkpoints and cannot compare drift directly. They trained on TPUs (App. F.2.3) and do not state the precision; we trained in FP32 on one GPU. We do not know which difference, if any, causes the gap. The Pearson $r = 0.98$ in the table is dominated by the large LR2 rise, which any reasonable rerun reproduces, so it says little about the late-stage match.

---

## 2. The Balancing Maps: `ORBIT` and `ORBIT+` (`[paper]` + `[empirical]`)

Balancing runs in `float64`. We check it on 512 fresh sequences both in `float64` and after casting the balanced weights to `float32`, which are the weights SGLD actually uses.

### 2.1 `ORBIT`: Invertible Loss-Preserving Maps Only
Every move in `ORBIT` is a smooth bijection on $\mathbb{R}^{51,717}$ preserving **all 5 output channels** of `DTransformer` (zero rank truncation, zero zeroing branches):
1. **Pre-LN $\mathbf{1}_d$ kernel projection (`LN_mean_proj`, `[paper]`):** Exact flat translation ($\det = 1$) subtracting the mean along `d_embed` from `token_embedding.weight`, `postn_embedding.weight`, `attention.output.weight`, `fc2.weight`, and `fc2.bias`.
2. **Invertible per-head $GL(16)$ Riccati balancing (`QK`, `VO`, `[paper]`):** Computes the unique SPD matrix $G \in GL(16)$ minimizing $\|G W_Q\|_F^2 + \|G^{-1} W_K\|_F^2$ (and $\|G W_V\|_F^2 + \|W_O G^{-1}\|_F^2$) via square-root-free SVDs (no Gram-matrix squaring), explicitly checking $\sigma_{\min} \ge 10^{-4}$ and $|\det G| > 0$ (leaving $G = I$ on any rank-deficient head — which never fires on Adam and cleanly guards decayed AdamW heads).
3. **Strictly positive ReLU MLP neuronwise rescaling + permutation (`MLP`, `[paper]`):** Scales $(W_{1,j,:}, b_{1,j}) \mapsto s_j (W_{1,j,:}, b_{1,j})$, $W_{2,:,j} \mapsto s_j^{-1} W_{2,:,j}$ with $s_j = \sqrt{\|W_{2,:,j}\|_2 / \sqrt{\|W_{1,j,:}\|_2^2 + b_{1,j}^2}} > 0$ (skipping only if norm $\le 10^{-15}$, never zeroed), then sorts descending.
4. **Strictly positive coordinatewise LayerNorm affine balancing (`LN0_attn`, `LN1_mlp`, `LNf_unembed_all5`, `[paper]`):**
   - `LN0_attn` (biasless `ln0 → QKV`): scales $(\gamma_{0,i}, \beta_{0,i}) \mapsto s_i(\gamma_{0,i}, \beta_{0,i})$, $W_{QKV,:,i} \mapsto s_i^{-1} W_{QKV,:,i}$ with $s_i = (\|W_{QKV,:,i}\|_2 / \sqrt{\gamma_{0,i}^2 + \beta_{0,i}^2})^{1/2} > 0$.
   - `LN1_mlp` (biased `ln1 → fc1`): exact bias absorption $b_{\text{fc1}} \leftarrow b_{\text{fc1}} + W_{\text{fc1}} \beta_1$, $\beta_1 \leftarrow 0$, then strictly positive $s_i = (\|W_{\text{fc1},:,i}\|_2 / |\gamma_{1,i}|)^{1/2} > 0$.
   - **`LNf_unembed_all5` (`lnf → unembedding` across ALL 5 rows simultaneously):** exact 5-row bias absorption $b_U \leftarrow b_U + W_U \beta_f$, $\beta_f \leftarrow 0$, then strictly positive $s_i = (\|W_{U,:,i}\|_2 / |\gamma_{f,i}|)^{1/2} > 0$ scaling **all 5 rows** of column $i$ simultaneously.
- **`ORBIT`** applies `LN_mean_proj` followed by 20 alternating sweeps of (`LNf_unembed_all5`, `LN1_mlp`, `LN0_attn`, `QK`, `VO`, `MLP`).

### 2.2 `ORBIT+`: Extra Moves, Reported Separately
Starts from `ORBIT`, zeros the 4 unused unembedding rows/biases (`unembedding.1.{weight,bias}[1:5] = 0`, which leaves the regression readout channel 0 unchanged but alters channels 1–4 and is not an invertible parameter-space map), and re-balances `lnf` against readout row 0 only.

### 2.3 Exactness Checks on All 114 Checkpoints (`[empirical]`)
- **All 69 `Adam (wd = 0)` checkpoints (512 fresh sequences):**
  - Readout channel 0: **$\max|\Delta y|_{\mathrm{f64}} \le 1.39\times 10^{-12}$**, **post-float32 $\max|\Delta y|_{\mathrm{f32}} \le 2.14\times 10^{-5}$**.
  - All 5 `DTransformer` channels: **$\max|\Delta \text{all5}|_{\mathrm{f64}} \le 3.28\times 10^{-12}$**, **post-float32 $\max|\Delta \text{all5}|_{\mathrm{f32}} \le 5.97\times 10^{-5}$**.
  - Diffeomorphism certificate: **`n_skipped_degenerate = 0` on every single checkpoint**, $\min s_i = 0.263$, $\max s_i = 6.30\times 10^6$ (from collapsed `lnf` coordinates $|\gamma_{f,i}| \sim 10^{-13}$ balanced against live `unembedding` rows 1–4), $\min \sigma = 0.0193$, $\min|\det G| = 0.110$, $\max \kappa(G) = 3.88$. No balancing step was skipped as degenerate on any Adam checkpoint, so every `ORBIT` map there is a composition of the invertible maps above.
- **All 45 `AdamW (wd = 0.08)` checkpoints (512 fresh sequences):**
  - Readout channel 0: $\max|\Delta y|_{\mathrm{f64}} \le 7.87\times 10^{-13}$, **post-float32 $\max|\Delta y|_{\mathrm{f32}} \le 1.57\times 10^{-5}$**; all 5 channels $\le 2.12\times 10^{-12}$ (f64) / $2.84\times 10^{-5}$ (f32). Weight decay prunes some attention-head singular values below $10^{-4}$ late in training (`202k–500k`), which our SVD guard cleanly leaves at $G = I$ (still an exact diffeomorphism).

---

## 3. Adam (`wd = 0`), 3 Seeds, 500k Steps (`[empirical]`; JSON `part1_adam_wd0`)

*Error bars in §3–4:* `±` is the standard error **across the 3 training seeds (`0, 1, 2`)** of each seed's 10-chain mean ($\mathrm{SD}/\sqrt{3}$). Within-seed 10-chain SEs are `0.04–0.92` at almost all checkpoints (up to `2.78` mid-transition at `162k`). Fiber shares are means of per-seed ratios.

### 3.1 Trajectory (3-Seed Mean ± SE, Table 2 SGLD: 10 × 5,000, $m=1{,}024$, $2^{20}$ pool)

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

### 3.2 Stagewise Changes, Raw vs. Balanced (3-Seed Mean ± SE)

The share column is the per-seed ratio $\Delta\hat\lambda_{\text{fiber}} / \Delta\hat\lambda_{\text{total}}$, averaged. Values above 100% mean the balanced estimate rose slightly while the raw one fell. Read them as "essentially all", not as a precise fraction: the split depends on the gauge (see "Terms used in this note").

| Stage Transition | $\Delta\hat\lambda_{\text{total}}$ (Raw) | $\Delta\hat\lambda_{\text{func}}$ (`ORBIT`) | $\Delta\hat\lambda_{\text{fiber}}$ | Share removed by balancing | Per seed (`s0 / s1 / s2`) | Reading |
|---|---:|---:|---:|---:|---|---|
| **`LR1` (`0 → 984`)** | +16.15 ± 8.21 | **+16.14 ± 8.22** | +0.01 ± 0.01 | **+0.2% ± 0.1%** | `+0.2% / -0.0% / +0.2%` | Little change from balancing |
| **`LR2` (`984 → 46,003`)** | +149.70 ± 0.99 | **+154.18 ± 0.96** | -4.47 ± 0.17 | **-3.0% ± 0.1%** | `-3.0% / -3.2% / -2.8%` | Little change from balancing |
| **`LR3–LR4` (`46,003 → 363,636`)** | **-74.88 ± 3.62** | **+0.58 ± 1.16** | **-75.47 ± 2.58** | **100.9% ± 1.6%** | **`98.2% / 103.7% / 100.9%`** | Drop absent at balanced points |
| **`LR3–LR4` (GP peak `30,303 → 363,636`)** | **-73.29 ± 3.52** | **+3.10 ± 1.15** | **-76.39 ± 2.40** | **104.4% ± 1.8%** | **`101.7% / 107.9% / 103.6%`** | Same, GP-smoothed peak |
| **`LR5` (`363,636 → 500,000`)** | +5.44 ± 1.45 | +1.73 ± 0.08 | +3.71 ± 1.39 | 64.4% ± 7.6% | `64.4% / 77.5% / 51.3%` | Mixed |

*Note on `step = 0` offset (`Raw - ORBIT = +7.89`):* At initialization, `lnf.weight = 1.0` while `unembedding` entries are $O(1/\sqrt{64})$, creating a constant `+7.89` gauge offset across `0..1,673` that cancels to `+0.01` over `LR1`.

---

## 4. Matched-Init AdamW (`wd = 0.08`), 3 Seeds, 500k Steps (`[empirical]`; JSON `part2_adamw`)

Each AdamW seed starts from the same initial weights as the matching Adam seed (identical SHA-256) and sees the same batch stream at every step.

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

**Raw vs. balanced, `46,003 → 363,636`:** raw $\Delta = -4.94 \pm 0.59$, balanced $\Delta = -4.09 \pm 0.87$, difference $-0.85 \pm 0.30$; then an LR5 rebound of $+7.09 \pm 0.78$ to `159.64`. With weight decay, $\|W_{\text{QKV},0}\|$ stays bounded (`12.2` vs. `71.6` under Adam) and the trough gap shrinks from `−72.04` to `−1.18`. This happens even though `lnf` collapses slightly more under AdamW (`96%` vs. `95%`) and OOD `g = 5` error rises (`3.372 → 3.652`). AdamW learns different functions (ID MSE `1.943` vs. `1.804` at `364k`), so this is an intervention that supports the mechanism, not a same-function comparison.

---

## 5. Per-Channel Ablation (`[empirical]`; JSON `part3_channel_ablation`)

All entries use Adam seed 0 with Table 2 SGLD (`10` chains $\times$ `5,000` steps, $m = 1{,}024$, $2^{20}$ pool). Error bars are SE **across the 10 SGLD chains** ($\mathrm{SD}/\sqrt{10}$); `hd` is split-half drift $n\beta(\bar L_{\text{2nd half}} - \bar L_{\text{1st half}})$.

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

Balancing only `LN0_attn` (`blk0_ln0` and `blk1_ln0` against `W_QKV`) recovers `+71.63` of the `+74.86` gap (`96%`) and brings split-half drift to `0.0`. `LN1_mlp` adds `+8.01`; every other single channel moves the trough estimate by less than one point. `ORBIT` and `ORBIT+` agree to `0.01`.

### 5.1 Effective Weights by Layer (Adam `wd = 0`, 3-Seed Mean, `46,003 → 363,636`)

| LayerNorm $\to$ Downstream Linear | $d_{\text{out}}$ per $\gamma_i$ | `mean|γ|` (`46k → 364k`) | `frac(|γ|<0.1)` | $\|W_{\text{down}}\|_F$ (`46k → 364k`) | `mean(|γ|·‖W[:,i]‖)` | `frac(eff<0.05)` | Regime |
|---|---:|---:|---:|---:|---:|---:|---|
| `blk0_ln0 → QKV` | **192** | `1.039 → 0.169` | `0.00 → 0.52` | **`10.80 → 71.61`** | **`1.406 → 2.031`** | `0.00 → 0.44` | Large rescaling + partial pruning |
| `blk1_ln0 → QKV` | **192** | `1.034 → 0.353` | `0.00 → 0.10` | **`10.90 → 98.87`** | **`1.410 → 4.687`** | `0.00 → 0.06` | Large rescaling + effective growth |
| `blk0_ln1 → fc1` | **64** | `1.018 → 0.524` | `0.00 → 0.22` | `6.24 → 55.95` | `0.795 → 4.276` | `0.00 → 0.20` | Rescaling + effective growth |
| `blk1_ln1 → fc1` | **64** | `1.090 → 1.332` | `0.00 → 0.04` | `7.68 → 72.23` | `1.043 → 12.458` | `0.00 → 0.03` | Eff. growth (balanced $\gamma$) |
| `lnf → W_U[0]` | **1** | `1.069 → 0.093` | `0.00 → 0.95` | `1.24 → 1.24` | `0.151 → 0.060` | **`0.15 → 0.95`** | Real readout collapse |

---

## 6. Gauss–Newton Spectra (`[empirical]` + `[paper]`; JSON `part6_gauss_newton`)

We compute the nonzero eigenvalues of the Gauss–Newton matrix $G = \frac{2}{N_{\text{res}}} J^\top J$ of the MSE loss on 512 fresh sequences ($N_{\text{res}} = 512 \times 8 = 4{,}096$ scalar residuals), via the dual Gram matrix $\frac{2}{4096} J J^\top$ (float64 `eigvalsh`). We do this for all 3 Adam seeds at the peak (`46,003`) and trough (`363,636`), raw and balanced, for the full model and for the `W_QKV`, `LN_0`, `W_fc1` and `LN_1` blocks. The isotropic localization floor is $\tau = \gamma / (n\beta) = 13.3 / 66.7 = 0.1994$: directions with curvature below it are dominated by the localization term. The Gaussian–Laplace effective dimension is $\hat\lambda_{\mathrm{GN}}(\gamma) = \frac{1}{2}\sum_k \frac{n\beta \mu_k}{n\beta \mu_k + \gamma}$. These counts and effective dimensions depend on the gauge, as the isotropic floor does; they explain the estimator's behavior rather than measure the true LLC.

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

**What the spectra show.** The mechanism (`[paper]`) is a change of coordinates. Shrinking $\gamma_{0,i}$ by a factor $c_i \approx 0.17$ and growing $W_{\text{QKV},:,i}$ by $1/c_i$ moves curvature from the 192 `W_QKV` weights fed by coordinate $i$ onto the single $\gamma_{0,i}$. In the raw trough, the `LN_0` block's largest eigenvalue rises from `3.19` to `31.85`, while the raw `W_QKV` trace at the trough is `60×` smaller than the balanced one (`33.7` vs. `2,072`). The number of `W_QKV` eigenvalues above $\tau$ falls from `547.0` (balanced) to `36.0` (raw), and for `W_fc1` from `78.3` to `0.7`. At the balanced points the full model has slightly more eigenvalues above $\tau$ at `364k` (`676.7`) than at `46k` (`615.7`). A check that does not depend on the gauge would be the numerical rank of $G$ at each checkpoint, which any loss-preserving diffeomorphism preserves. We have not run it.

---

## 7. Other Estimators (`[empirical]` + `[paper]`; JSON `part5_remedies`)

Can a different sampler remove the drop without editing the checkpoint? We compare four SGLD-family estimators on all 3 Adam seeds at the peak (`46,003`) and trough (`363,636`), raw and balanced, with the Table 2 chain budget (`10` chains × `5,000` steps, $m = 1{,}024$, $2^{20}$ pool; `±` is SE across seeds):

1. **`vanilla` (Hoogland et al. 2024):** $\Delta w_t = -\frac{\epsilon}{2}(n\beta g_t + \gamma(w_t - w^*)) + \sqrt{\epsilon}\eta_t$.
2. **`rmsprop_psgld` (Hitchcock & Hoogland, arXiv:2507.21449 Alg. G.3; `devinterp` `RMSpropPreconditioner`):** EMA $v_t = 0.99 v_{t-1} + 0.01 g_t^2$, bias-corrected $\hat v_t$, $G_t = (\sqrt{\hat v_t} + a)^{-1}$ with $a = 0.01$, applied to both $n\beta g_t$ and $\gamma(w_t - w^*)$, with $\sqrt{G_t}$ on the noise. We chose $a = 0.01$ because $\sqrt{\hat v_t}$ ranges over $[7\times 10^{-4}, 3.7\times 10^{-2}]$, so the preconditioner is active. One departure from the default: we scale the base step to $\epsilon a$, so that flat directions keep step $\epsilon$. This sampler targets the same isotropic localized posterior $\pi_\gamma(w)$ as vanilla SGLD.
3. **`param_rms_covariant` (per-tensor RMS metric, no symmetry knowledge):** per-tensor scale $c_T = \mathrm{rms}(w_{T,\text{init}}) / \max(\mathrm{rms}(w_T^*),\, 0.1\,\mathrm{rms}(w_{T,\text{init}}))$ (shared with LayerNorm biases), metric $D = \mathrm{diag}(c_T^2)$ in the localization term $\frac{\gamma}{2}(w - w^*)^\top D (w - w^*)$, and preconditioner $P = D^{-1}$.
4. **`orbit_covariant` (metric from the balancing scales):** computes the cumulative scale $c_p(w^*) > 0$ of each coordinate under the coordinatewise rescaling symmetries (`(γ, β) ↔ W` for all 5 LayerNorms without bias absorption, plus both ReLU MLPs), sets $D(w^*) = \mathrm{diag}(c_p^2)$, and runs
   $$\Delta w_t = -\frac{\epsilon}{2}\big(n\beta\, D^{-1} g_t + \gamma(w_t - w^*)\big) + \sqrt{\epsilon}\, D^{-1/2} \eta_t.$$
   In coordinates $\tilde w = D^{1/2} w$ this is vanilla SGLD started at the diagonally balanced point, with identical noise. Because the rescalings are exact symmetries, $L(D^{-1/2}\tilde w) = L(\tilde w)$. So `orbit_covariant` is a way to compute the balanced estimate without editing $w^*$; it is not independent evidence for it. It omits the $GL(16)$ head balancing and bias absorption in `ORBIT`, which is why its trough value sits `0.9` below the `ORBIT` value.

| Estimator (3 seeds, `46k → 364k`) | `RAW` Peak $\to$ Trough | `RAW` $\Delta$ (`Trough - Peak`) | `ORBIT` Peak $\to$ Trough | `ORBIT` $\Delta$ (`Trough - Peak`) | Trough Gap (`RAW - ORBIT`) | Reading |
|---|---|---:|---|---:|---:|---|
| **`vanilla` (Hoogland 2024)** | `166.93 ± 2.17 → 91.69 ± 1.22` | **`-75.24 ± 3.34`** | `163.42 ± 2.02 → 164.00 ± 1.17` | `+0.58 ± 1.02` | **`-72.31 ± 2.26`** | Raw drops, balanced flat |
| **`rmsprop_psgld` (`2507.21449`)** | `168.16 ± 2.03 → 103.14 ± 2.07` | **`-65.02 ± 4.09`** | `168.04 ± 1.85 → 167.09 ± 1.89` | `-0.95 ± 1.22` | **`-63.94 ± 3.66`** | Most of the drop remains (same target distribution) |
| **`param_rms_covariant` (black-box)** | `177.51 ± 1.36 → 159.84 ± 0.85` | `-17.67 ± 0.82` | `175.34 ± 1.53 → 159.85 ± 0.91` | `-15.49 ± 0.75` | **`-0.00 ± 0.13`** | Gap closed; both fall ~16 |
| **`orbit_covariant` (exact metric)** | **`163.34 ± 1.99 → 163.07 ± 1.30`** | **`-0.27 ± 0.83`** | `163.42 ± 2.02 → 164.00 ± 1.16` | `+0.58 ± 1.03` | **`-0.94 ± 0.16`** | Gap closed; equals balanced vanilla by change of variables |

**Reading the table (`[empirical]`):**
- **RMSProp-pSGLD leaves most of the drop.** The raw estimate falls `65.02` points, compared with about `75` for vanilla, and the trough gap is `−63.94`. This is expected: preconditioning changes how fast the chain mixes and, up to the bias of adaptive preconditioning without a correction term, not which distribution it samples. Faster mixing may account for the 10-point difference from vanilla. Pilot runs at $a \in \{0.1, 0.01, 0.001\}$ also left the gap in place; those pilots are not in the committed JSON.
- **Changing the localization metric closes the gap.** Both metric-based estimators close the trough gap (`−0.00` for `param_rms_covariant`, `−0.94` for `orbit_covariant`). They disagree about the drop itself. `orbit_covariant` is flat (`−0.27 ± 0.83`), while `param_rms_covariant` falls about `16` points on both raw and balanced weights. The per-tensor metric also rescales real growth in effective weights. So which metric to trust is a modelling choice, and the change in what the function's complexity actually is lies somewhere between about `−16` and `+1` points by these estimators.
- A pilot with a $\det D = 1$ geometric-mean-normalized gradient-RMS metric made the gap worse (`−168`). `W_QKV` and `W_fc1` hold 63% of the parameters and drag the geometric mean down when their gradients shrink. This pilot is not in the committed JSON either.

---

## 8. Controls: $\gamma$, Step Size and Full-Batch Gradients (`[empirical]`; JSON `part4_sgld_controls`)

All runs use Adam seed 0 with the Table 2 chain budget (`10` chains $\times$ `5,000` steps, $m = 1{,}024$, $2^{20}$ pool; `±` is SE across 10 chains). **Full-batch-gradient SGLD (`FB1024`)** computes the exact gradient on a fixed 1,024-sample subset (removing minibatch gradient noise) while **retaining** Langevin noise $\sqrt{\epsilon}\,\eta_t$, measuring LLC relative to that subset's loss.

| Setting | Peak `46,003` `RAW` / `ORBIT` | Trough `363,636` `RAW` (`hd`, $\|w-w^*\|$) | Trough `363,636` `ORBIT` (`hd`, $\|w-w^*\|$) | `RAW` $\Delta$ (`46k→364k`) | `ORBIT` $\Delta$ (`46k→364k`) |
|---|---:|---:|---:|---:|---:|
| **$\gamma = 3.3, \epsilon = 3\text{e-}4$** | `174.99 ± 0.87` / `166.32 ± 0.29` | `162.54 ± 0.56` (`+1.1`, `124.8`) | `167.20 ± 0.38` (`+0.6`, `124.8`) | **`-12.45`** | **`+0.88`** |
| **$\gamma = 13.3, \epsilon = 3\text{e-}4$ (default)** | `170.03 ± 0.31` / `166.42 ± 0.26` | `90.00 ± 0.68` (`+0.8`, `62.3`) | `165.20 ± 0.20` (`-0.1`, `62.4`) | **`-80.03`** | **`-1.22`** |
| **$\gamma = 40.0, \epsilon = 3\text{e-}4$** | `163.22 ± 0.20` / `164.62 ± 0.16` | `52.63 ± 0.19` (`+0.6`, `36.0`) | `167.42 ± 0.17` (`-0.4`, `36.0`) | **`-110.59`** | **`+2.80`** |
| $\gamma = 13.3, \epsilon = 1\text{e-}4$ | `168.97 ± 0.43` / `166.79 ± 0.48` | `88.55 ± 1.03` (`+4.7`, `62.3`) | `165.50 ± 0.31` (`-0.6`, `62.4`) | `-80.42` | `-1.29` |
| **$\gamma = 13.3$, Full-Batch-Gradient `1024`** | `167.80 ± 0.30` / `164.59 ± 0.24` | `85.62 ± 0.65` (`+0.2`, `62.3`) | `163.50 ± 0.16` (`+0.0`, `62.4`) | **`-82.17`** | **`-1.10`** |

**Takeaways (`[empirical]`, seed 0):**
1. **The gap depends on $\gamma$ and shrinks as $\gamma$ weakens.** At the trough the raw estimate is `52.63`, `90.00`, `162.54` for $\gamma = 40, 13.3, 3.3$ (all $|\mathrm{hd}| \le 1.1$). The balanced estimate stays within `2.2` points (`165.2–167.4`). Weakening $\gamma$ is not a clean fix, though: at $\gamma = 3.3$ the chains travel $\|w - w^*\| = 124.8$, about half the parameter norm, and the raw estimate still drops `12.45` points.
2. **Minibatch noise is not the cause.** With full-batch gradients the raw drop is `82.17` points, against `1.10` balanced.
3. **Smaller steps slow the raw chain.** At $\epsilon = 10^{-4}$ the raw trough chain still drifts (`hd = +4.7`), while the balanced chain is equilibrated (`hd = −0.6`).

---

## 9. What This Means for DevInterp Practice (`[empirical]` + `[conjectural]`)

1. **The early stages hold up.** LR1 and LR2 are clean rises in the balanced estimate (`−6.72 → 9.43 → 163.60`), and balancing changes them by 3% or less.
2. **The late drop is mostly a reading of the rescaling.** At the trough, about 96% of the raw-versus-balanced gap comes from one `LN ↔ Linear` pair. That is the same pre-attention LayerNorm the paper names, but here its collapse is largely compensated downstream. Real changes still happen in LR3–LR4: readout collapse, partial pruning in block 0, and changes in OOD behavior. Finite-$\gamma$ SGLD alone cannot tell us whether the true LLC changes a little over `46k → 364k`.
3. **Practical suggestion.** On LayerNorm or RMSNorm Transformers trained without weight decay, compare the raw LLC curve with one taken at a balanced point, or with a metric-localized sampler. RMSProp preconditioning alone does not address this. The per-tensor RMS metric is a cheap first check, because it needs no knowledge of the symmetries.
4. **Open question (`[conjectural]`).** Their language models were trained with AdamW (`wd = 0.05`). Do mid-training LLC drops there, such as `LM3`, have any `LN ↔ Linear` component? Our AdamW run suggests weight decay limits the drift, but that is one small model. Their public Hugging Face checkpoints would answer it.

---

## 10. Reproducibility & Artifacts
- **Script:** [`experiments/devinterp_fiber_decomposition.py`](../experiments/devinterp_fiber_decomposition.py) (supports `--smoke`, `--train-only`, `--from-checkpoints --shard k/N`, `--merge-shards`, and `--replot`).
- **Upstream parity check:** [`experiments/verify_upstream_icl_parity.py`](../experiments/verify_upstream_icl_parity.py) (verified `0.0` bit-for-bit parity in f64 & f32 against `timaeus-research/icl` @ `5baf73025d279abd2d56c98e13bf5663608709c0`).
- **Saved `.safetensors` checkpoints + SHA-256 manifest (114 files, 23 MB):** [`experiments/results/devinterp_checkpoints/manifest.json`](../experiments/results/devinterp_checkpoints/manifest.json) (all 3 seeds $\times$ 23 Adam + 15 AdamW checkpoints; SGLD, the estimator comparison and the spectra can be rerun with `--from-checkpoints`, without retraining).
- **Data and reference curve:** [`experiments/results/devinterp_fiber_decomposition.json`](../experiments/results/devinterp_fiber_decomposition.json) (total GPU eval runtime `12,047.5s` across 3 shards + `~1,800s` training on 1× NVIDIA RTX PRO 6000 Blackwell, exact FP32) and [`experiments/results/hoogland_l2h4minf_reference.json`](../experiments/results/hoogland_l2h4minf_reference.json).
- **Figure:** [`experiments/results/devinterp_fiber_decomposition.png`](../experiments/results/devinterp_fiber_decomposition.png) (`--replot` regenerates in `< 2s`).
- **Short note for the DevInterp authors:** [`notes/devinterp_for_authors.md`](devinterp_for_authors.md).
