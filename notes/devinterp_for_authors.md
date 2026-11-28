# The Late LLC Drop in `L2H4Minf`: A Note for the Authors

*For Hoogland, Wang, Farrugia-Roberts, Carroll, Wei, Murfet and Hitchcock (arXiv:2402.02364, arXiv:2507.21449). Early preprint material; corrections welcome.*

## Question

- App. D.2.6 says the degeneracy argument for the LR3–LR4 decrease "hinges on the collapse being complete".
- We reran `L2H4Minf` to check how complete the LayerNorm collapse is, and what the SGLD estimate does when it is not.

## Short answer

- In our runs the collapse is mostly offset by growth in the next layer's weights.
- That makes it a loss-preserving reparametrization, so the true LLC is unchanged.
- Most of the drop in the SGLD estimate follows that reparametrization.
- We have not shown that this holds for your runs.

## Setup

- **Model:** matches `timaeus-research/icl` at `5baf730` exactly (zero output difference, float64 and float32, 512 sequences).
- **Training:** your Table 4 recipe (Adam, `wd = 0`, 500k steps), 3 seeds.
- **Estimator:** your Table 2 SGLD settings at every checkpoint.
- **Comparison with your curve:**
  - Same peak and trough steps (`46,003`, `363,636`).
  - Within a few points of your released curve through `126k`.
  - After that ours falls further: a 75-point drop against your 49. We don't know why. Our runs used FP32 on one GPU.

## Mechanism

- **Estimator vs. LLC:**
  - The LLC is unchanged by loss-preserving reparametrizations.
  - The SGLD estimate is not, because its localization term pulls equally along every coordinate.
- **What Adam does between peak and trough:**
  - Block 0 pre-attention LN: $\gamma$ shrinks about $6\times$, $\|W_{\text{QKV}}\|$ grows about $6.6\times$.
  - Block 1 pre-attention LN: $\gamma$ shrinks about $3\times$, $\|W_{\text{QKV}}\|$ grows about $9\times$.
  - This is the collapse in your §7.3 and App. D.2.5. In block 0 about 44% of coordinates are effectively pruned; for the rest, $W_{\text{QKV}}$ compensates.
- **Why the estimator reacts:**
  - At $\gamma = 13.3$ the chains sit about `62` from $w^*$. That is the localization term alone ($\sqrt{51{,}717/13.3}$), about `0.27` per coordinate.
  - At the trough, block 0's $\gamma$ entries average `0.17`.
  - So the raw chain crosses $\gamma \approx 0$ in many coordinates. At the balanced point the same function sits at a scale the localization treats evenly.

## Results

- **Balanced points (exact maps, readout change ≤ `2.1e-5` in float32), 3 seeds:**
  - Raw estimate: `167.0 → 92.2`.
  - Balanced estimate: `163.6 → 164.2`.
  - Balancing only the two pre-attention `LN ↔ W_QKV` pairs closes about 96% of the trough gap (seed 0).
  - LR1 and LR2 change by 3% or less.
- **Localization strength (trough, seed 0):**
  - Raw: `52.6`, `90.0`, `162.5` at $\gamma = 40, 13.3, 3.3$.
  - Balanced: `165.2`–`167.4` throughout.
  - Caveat: at $\gamma = 3.3$ the chains wander about `125` and are not local. A weak pull also lets the raw chain slide along the orbit toward the balanced region. So this is weak evidence for the balanced value.
- **RMSProp-pSGLD ($a = 0.01$, base step scaled to $\epsilon a$):**
  - The raw drop is `−65` (vanilla: about `−75`).
  - Expected: preconditioning targets the same localized distribution (up to the usual bias without a correction term). It can speed up mixing but cannot remove a gap in the target.
- **Metric in the localization term:**
  - Use a diagonal metric built from the balancing scales. SGLD at the raw weights is then flat (`−0.3`).
  - This is the balanced estimate by a change of variables, not separate evidence.

## Open questions

- **Does this explain your curve?**
  - Our runs may drift further than yours. If so, more of your 49 points could be real.
  - Quick check without checkpoints: our raw parameter norm at the trough is `241.7`. Compare with your weight-norm logs.
  - Decisive check: balance one of your seeds at `46k` and near `364k`, if you still have the checkpoints. Our code takes a plain state dict.
- **Is the balanced point the right reference?**
  - The $\gamma$ sweep is the only direct argument.
  - The numerical rank of the Gauss–Newton matrix does not depend on the gauge. We haven't run it yet.
- **Does the true LLC change over this window?**
  - The change at fixed function depends on the metric: `+0.6` (balanced, isotropic) to about `−16` (per-tensor RMS metric).
  - What is robust: the raw-vs-balanced gap closes.
- **Language models?**
  - They were trained with AdamW.
  - In our AdamW (`wd = 0.08`) rerun, the gap stayed within about one point.

## Materials

- **Repository:** [github.com/bubskee/eat-your-fiber](https://github.com/bubskee/eat-your-fiber).
- **Full write-up:** [`notes/devinterp_fiber_decomposition.md`](https://github.com/bubskee/eat-your-fiber/blob/main/notes/devinterp_fiber_decomposition.md).
- **Figure:** [`devinterp_fiber_decomposition.png`](https://github.com/bubskee/eat-your-fiber/blob/main/experiments/results/devinterp_fiber_decomposition.png).
- **Checkpoints:** all 114 are committed (`.safetensors` with a SHA-256 manifest), so nothing needs retraining:

```bash
python experiments/verify_upstream_icl_parity.py --icl-repo /path/to/timaeus-research/icl
python experiments/devinterp_fiber_decomposition.py --from-checkpoints --eval-seeds 0,1,2
python experiments/devinterp_fiber_decomposition.py --replot
```

*Code, experiments and write-up were done largely with AI agents, directed by Bobby Faber.*
