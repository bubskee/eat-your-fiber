# Note for the DevInterp Authors (Hoogland, Wang, Farrugia-Roberts, Carroll, Wei, Murfet & Hitchcock)

*A concise, collaborative summary of our exact-symmetry decomposition, Gauss–Newton spectral check, and reparametrization-aware SGLD remedy experiments on the `L2H4Minf` in-context linear regression Transformer (arXiv:2402.02364 & arXiv:2507.21449).*

---

## 1. What We Did

We love the developmental interpretability program and the `L2H4Minf` in-context linear regression study (Hoogland et al., 2024, TMLR). Because our group has been studying exact continuous symmetry fibers in neural networks, we wanted to check how much of `L2H4Minf`'s non-monotonic SGLD LLC trajectory (`LR1`–`LR5`) reflects functional/structural changes versus non-invariance of the finite-$\gamma$ localized SGLD estimator along exact iso-functional symmetry orbits.

Since the `L2H4Minf` regression checkpoints live in a private S3 bucket, we:
1. **Verified bit-for-bit model parity** against `github.com/timaeus-research/icl` (commit `5baf7302`) on 512 held-out sequences (`max|Δy| = 0.0` in both float64 and float32 across all 5 `DTransformer` output channels).
2. **Retrained 3 seeds (`0, 1, 2`) for the exact 500,000-step paper recipe** (`K = 8, D = 4, σ² = 0.125`, `51,717` params, `Adam wd = 0.0`, `OneCycleLR(max_lr = 0.003)` peaking at `250k`, $B = 256$; noting that public `L2H4Minf.yaml` sets `max_examples: 16` and leaves optimizer unset, so we followed Table 4 / App. F.2 matching the 8 token positions in `llcs.pt`), plus **bit-for-bit matched-init `AdamW (wd = 0.08)`** runs (`max|w_init_diff| = 0.0`), saving all 114 checkpoints as `.safetensors`.
3. **Evaluated every checkpoint with your exact Table 2 SGLD settings** in exact FP32 (`10` chains $\times$ `5,000` steps = `1,000` burn + `4,000` draws, $m = 1{,}024$, fixed $2^{20}$ pool, $\epsilon = 3\times 10^{-4}, \gamma = 13.3, n\beta = 66.7$), comparing:
   - **Raw weights (`RAW`)** vs. **pure exact invertible group canonicalization (`ORBIT`)** (Pre-LN $\mathbf{1}_d$ projection, invertible per-head $GL(16)$ balancing on $Q$-$K$ & $V$-$O$, strictly positive ReLU MLP scaling, and strictly positive coordinatewise $(\gamma_i,\beta_i)\leftrightarrow W_{:,i}$ balancing across all 5 LayerNorms — including all 5 rows of `unembedding` — with zero rank truncation or zeroing branches, verified to $\max|\Delta y|_{\mathrm{f64}} \le 1.39\times 10^{-12}$ and **post-float32 $\max|\Delta y|_{\mathrm{f32}} \le 2.14\times 10^{-5}$** on 512 fresh sequences).
   - **Exact Gauss–Newton spectra** ($G = \frac{2}{4096} J^\top J$ on 512 sequences $\times$ 8 positions) for the full model (`51,717`) and individual blocks (`W_QKV`, `LN_0`, `W_fc1`).
   - **Four SGLD-family estimators** on `RAW` vs. `ORBIT` across `46k → 364k`: Vanilla SGLD, **RMSProp-pSGLD** (Hitchcock & Hoogland, arXiv:2507.21449 Alg. G.3 / `devinterp` `RMSpropPreconditioner`), a black-box **Per-Tensor Param-RMS Covariant** SGLD, and an exact **Symmetry-Adapted Metric Localization SGLD (`orbit_covariant`)**.

---

## 2. What We Found

Our faithful 500k `RAW` trajectory tracks your released 5-seed `llcs.pt` closely (Pearson $r = 0.981$, exact same empirical peak at `46,003` [`167.0` vs. `170.4`], `LR3` step `126,262` at `156.2` vs. `156.9`, exact same trough at `363,636`, and `LR5` uptick `+5.4` vs. `+6.7`; our raw trough is deeper at `92.2` vs. `121.2`). Comparing `RAW` and `ORBIT` reveals a crisp split:

1. **Good news for `LR1` and `LR2` — early developmental stages are rock-solid:**
   - `LR1 (0 → 984)` (`+16.15` raw vs. `+16.14` `ORBIT`, fiber share `+0.2% ± 0.1%`) and `LR2 (984 → 46,003)` (`+149.70` raw vs. `+154.18` `ORBIT`, fiber share `-3.0% ± 0.1%`) are completely genuine functional complexity increases, unaffected by symmetry gauge.
2. **The `LR3–LR4` drop (`46,003 → 363,636`) is `100.9% ± 1.6%` estimator non-invariance along the exact `LN_0 ↔ QKV` rescaling orbit:**
   - `RAW` drops **`-74.88 ± 3.62`** (`167.03 → 92.15`), while on the exact same functions (same true LLC by local diffeomorphism invariance), `ORBIT` is flat (**`+0.58 ± 1.16`**, `163.60 → 164.19`).
   - Per-channel ablation isolates **pre-attention `LN_0 ↔ QKV` rescaling alone** as `+71.63` of the `+74.86` trough gap (`95.7%`). By contrast, final `lnf` readout collapse genuinely happens (`95%` of effective unembedding weights $< 0.05$), yet balancing `lnf` changes trough LLC by only `+0.02` points.
3. **Direct Gauss–Newton spectral proof:**
   - As `blk0_ln0` shrinks (`1.04 → 0.17`) while `||W_QKV,0||` grows (`10.8 → 71.6`), the exact number of `W_QKV` Gauss–Newton eigenvalues above the isotropic localization floor $\tau = \gamma/(n\beta) = 0.1994$ collapses **$11\times$ in `RAW` (`398 → 36`)** while *rising* in `ORBIT` (`451 → 547`) — driving full-model Gaussian-Laplace effective dimension from `359.9 → 119.2` in `RAW` vs. `372.8 → 398.7` in `ORBIT`.
4. **Why `RMSPropSGLD` (`2507.21449`) still drops, and a clean fix for `devinterp`:**
   - **Exact `RMSPropSGLD` (`a = 0.01`, 3 seeds):** `RAW` **still drops `-65.02 ± 4.09`** (`168.16 → 103.14`) vs. `-0.95 ± 1.22` for `ORBIT` (trough gap `-63.94 ± 3.66`). Because `RMSpropPreconditioner` preconditions *both* gradient and isotropic prior by $G_t$, it still targets the same isotropic localized posterior $\pi_\gamma(w) \propto \exp(-n\beta L(w) - \frac{\gamma}{2}\|w - w^*\|_2^2)$.
   - **Exact Symmetry-Adapted Metric Localization (`orbit_covariant`, 3 seeds):** Replacing the isotropic prior with $\frac{\gamma}{2}(w - w^*)^\top D(w^*)(w - w^*)$ (where $D = \mathrm{diag}(c_p^2)$ encodes the pure-multiplicative `LN ↔ Linear` + `MLP` balancing scales) and preconditioning by $P = D^{-1}$ gives **`RAW`: `163.34 → 163.07` ($\Delta = -0.27 \pm 0.83$, trough gap `-0.94 ± 0.16`)** — fixing `99.6%` of the drop **without modifying the model weights**!
   - **Zero-symmetry diagnostic (`param_rms_covariant`):** Even a 5-line per-tensor parameter-RMS metric reduces the trough `RAW - ORBIT` gap to **`-0.00 ± 0.13`**, making it a cheap black-box sanity check for whether an LLC drop is gauge-driven.

---

## 3. What We're Still Unsure Of

1. **Your original TPU checkpoints:** Our 500k exact-FP32 raw trough (`92.2`) is ~29 points deeper than your released `llcs.pt` (`121.2`), even though peak (`46k`), `LR3` (`126k`), and trough (`364k`) timing match closely. Testing our `ORBIT` canonicalizer or `orbit_covariant` sampler directly on one of your original S3 checkpoints (seed 1 at `46k` and `320k/364k`) would settle immediately whether ~100% of your original `-49` drop has the exact same cause!
2. **True asymptotic LLC change:** Both `RAW` and `ORBIT` are finite-$\gamma$ estimates; we cannot rule out a subtle change in the true $\gamma \to 0$ LLC between `46k` and `364k` (e.g., from `lnf` readout collapse or partial `blk0` pruning).
3. **Larger language models with weight decay (`LM1`–`LM5`):** In our matched-init `AdamW (wd = 0.08)` runs, weight decay kept `RAW - ORBIT` within `1.2` points across all 500k steps. Whether mid-training LLC drops in `georgeyw/L2W256-3m-rr` (`AdamW, wd = 0.05`) are predominantly functional or partially gauge-driven is an open test we'd love to compare notes on.

---

## 4. One-Command Reproduction

All 114 trained checkpoints (`23 MB`, `.safetensors` + SHA-256 `manifest.json`) are committed in `experiments/results/devinterp_checkpoints/`. Everything can be verified or re-run without retraining:

```bash
# 1. Verify bit-for-bit parity against timaeus-research/icl (0.0 diff on 512 seqs, f64 & f32)
python experiments/verify_upstream_icl_parity.py --icl-repo /path/to/timaeus-research/icl

# 2. Fast ~60s end-to-end smoke test (training, ORBIT vs ORBIT+, remedies, Gauss-Newton, plot)
python experiments/devinterp_fiber_decomposition.py --smoke --out-dir /tmp/eyf_smoke

# 3. Re-evaluate directly from our saved 500k checkpoints (shardable across N GPUs via --shard k/N)
python experiments/devinterp_fiber_decomposition.py --from-checkpoints --eval-seeds 0,1,2 --shard 0/1

# 4. Regenerate the 8-panel summary figure from saved JSON (< 2s)
python experiments/devinterp_fiber_decomposition.py --replot
```

Full tables, proofs, and caveats: [`notes/devinterp_fiber_decomposition.md`](devinterp_fiber_decomposition.md) and [`experiments/results/devinterp_fiber_decomposition.png`](../experiments/results/devinterp_fiber_decomposition.png). We'd love your thoughts — and if an `OrbitCovariantPrior` / `MetricPreconditioner` hook would be useful in `devinterp`, we'd be happy to help prototype it!
