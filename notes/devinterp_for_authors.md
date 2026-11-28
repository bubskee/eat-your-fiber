# The Late LLC Drop in `L2H4Minf`: A Note for the Authors

*For Hoogland, Wang, Farrugia-Roberts, Carroll, Wei, Murfet and Hitchcock. It concerns arXiv:2402.02364 and arXiv:2507.21449.*

We reran your in-context linear regression Transformer (`L2H4Minf`). In our runs, most of the LR3–LR4 drop in the SGLD estimate tracks a LayerNorm rescaling that leaves the function unchanged, rather than a fall in complexity. We would value your view. The cleanest test needs two of your original checkpoints.

## Setup

- **Model and training.** Our model code matches `timaeus-research/icl` at `5baf730` exactly: zero output difference in float64 and float32 on 512 sequences. We trained 3 seeds with your Table 4 recipe (Adam, `wd = 0`, 500k steps).
- **Estimator.** We evaluated every checkpoint with your Table 2 SGLD settings.
- **How our curve compares.** It peaks and bottoms out at the same logged steps as yours (`46,003` and `363,636`). Our drop is larger, though: 75 points against your 49. We don't know why. Our runs used FP32 on one GPU.

## The observation

The true LLC does not change under loss-preserving reparametrizations, but the SGLD estimate does. Its localization term pulls equally hard along every coordinate, so moving $w^*$ along a symmetry orbit changes the estimate.

Between the peak and the trough, Adam moves a long way along one such orbit. Block 0's pre-attention LayerNorm $\gamma$ shrinks about $6\times$, while $\|W_{\text{QKV}}\|$ grows about $6.6\times$. This is the collapse you describe in §7.3 and App. D.2.5. In our runs it is only partly complete: about 44% of the coordinates are effectively pruned, and for the rest the downstream weights grow to compensate.

## Findings

1. **At balanced points the drop disappears.** We map each checkpoint to a balanced point on its symmetry orbit. The maps are exact; the readout changes by at most `2.1e-5` in float32. At the balanced points the estimate goes from `163.6` to `164.2`, against `167.0` to `92.2` at the raw weights. Balancing that one `LN ↔ W_QKV` pair accounts for about 96% of the trough gap. LR1 and LR2 change by 3% or less.
2. **The gap is a finite-$\gamma$ effect.** At the trough (seed 0), the raw estimate is `52.6`, `90.0` and `162.5` for $\gamma = 40$, `13.3` and `3.3`. The balanced estimate stays between `165` and `167`. Your §4 notes that stable estimates need strong localization. Here, stronger localization makes the raw drop larger.
3. **RMSProp-preconditioned SGLD leaves most of the drop** (`−65` points). We would expect that: preconditioning changes how the chain mixes, not the distribution it targets.
4. **A metric in the localization term removes the drop.** We build a diagonal metric from the balancing scales and put it in the localization term. SGLD at the raw weights then gives a flat curve (`−0.3`). By a change of variables this is the same computation as the balanced estimate, so it is not separate evidence. It could still be a small `devinterp` option, and we would be glad to help prototype it.

## What we are unsure about

- **Whether this explains your curve.** Our drop is larger than yours, and we can't compare how far the runs drift without your checkpoints. Running the balancing on one of your seeds at `46k` and near `364k` would settle it. Our code works on a plain state dict.
- **Whether the true LLC changes over this window.** How much the estimate moves at a fixed function depends on the metric. It ranges from `+0.6` (balanced point, isotropic localization) to about `−16` (a per-tensor RMS metric). The robust result is that the raw-versus-balanced gap closes.
- **Whether the language models show anything similar.** They were trained with AdamW. In our AdamW (`wd = 0.08`) rerun, the gap stayed within about one point.

## Materials

The full tables and caveats are in [`devinterp_fiber_decomposition.md`](devinterp_fiber_decomposition.md), and the figure is [`devinterp_fiber_decomposition.png`](../experiments/results/devinterp_fiber_decomposition.png). All 114 checkpoints are committed as `.safetensors` with a SHA-256 manifest, so nothing needs retraining:

```bash
python experiments/verify_upstream_icl_parity.py --icl-repo /path/to/timaeus-research/icl
python experiments/devinterp_fiber_decomposition.py --from-checkpoints --eval-seeds 0,1,2
python experiments/devinterp_fiber_decomposition.py --replot
```

*Much of the code, the experiments and the writing were done with AI agents, directed by Bobby Faber. Corrections are very welcome.*
