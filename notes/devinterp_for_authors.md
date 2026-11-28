# The Late LLC Drop in `L2H4Minf`: A Note for the Authors

*For Hoogland, Wang, Farrugia-Roberts, Carroll, Wei, Murfet and Hitchcock. It concerns arXiv:2402.02364 and arXiv:2507.21449.*

We reran your in-context linear regression Transformer (`L2H4Minf`) to test the caveat in your App. D.2.6: that the degeneracy argument for the LR3–LR4 decrease "hinges on the collapse being complete". In our runs the collapse is mostly compensated by growth in the next layer's weights. That makes it a loss-preserving reparametrization, which leaves the true LLC unchanged. Most of the drop in the SGLD estimate tracks that reparametrization. We have not shown that the same holds for your runs, and we would value your view.

## Setup

- **Model and training.** Our model code matches `timaeus-research/icl` at `5baf730` exactly: zero output difference in float64 and float32 on 512 sequences. We trained 3 seeds with your Table 4 recipe (Adam, `wd = 0`, 500k steps).
- **Estimator.** We evaluated every checkpoint with your Table 2 SGLD settings.
- **How our curve compares.** It peaks and bottoms out at the same logged steps as yours (`46,003` and `363,636`), and through `126k` it stays within a few points of your released curve. After that it falls further: our drop is 75 points against your 49, so we drift into exactly the window in question. We don't know why. Our runs used FP32 on one GPU.

## The observation

The true LLC does not change under loss-preserving reparametrizations, but the SGLD estimate does. Its localization term pulls equally hard along every coordinate, so moving $w^*$ along a symmetry orbit changes the estimate.

Between the peak and the trough, Adam moves a long way along one family of such orbits: the pre-attention LayerNorms against $W_{\text{QKV}}$. In block 0, $\gamma$ shrinks about $6\times$ while $\|W_{\text{QKV}}\|$ grows about $6.6\times$. In block 1, $\gamma$ shrinks about $3\times$ while $\|W_{\text{QKV}}\|$ grows about $9\times$. This is the collapse you describe in §7.3 and App. D.2.5. In block 0 it is only partly complete: about 44% of the coordinates are effectively pruned, and for the rest $W_{\text{QKV}}$ grows to compensate.

The scales explain why this matters for the estimator. At $\gamma = 13.3$ the chains end up about `62` from $w^*$, which is what the localization term alone gives ($\sqrt{51{,}717/13.3}$), so each coordinate wanders about `0.27`. At the trough, block 0's $\gamma$ entries average `0.17`, so the raw chain moves across $\gamma \approx 0$ in many coordinates. At the balanced point the same function sits at a scale the localization treats much more evenly.

## Findings

1. **At balanced points the drop disappears.** We map each checkpoint to a balanced point on its symmetry orbit. The maps are exact; the readout changes by at most `2.1e-5` in float32. At the balanced points the estimate goes from `163.6` to `164.2`, against `167.0` to `92.2` at the raw weights (3 seeds). In seed 0, balancing only the two pre-attention `LN ↔ W_QKV` pairs accounts for about 96% of the trough gap. LR1 and LR2 change by 3% or less.
2. **The gap shrinks as $\gamma$ weakens.** At the trough (seed 0), the raw estimate is `52.6`, `90.0` and `162.5` for $\gamma = 40$, `13.3` and `3.3`. The balanced estimate stays between `165.2` and `167.4`. This is weaker evidence than it looks. At $\gamma = 3.3$ the chains wander about `125` and are no longer local, the raw estimate still drops `12.5` points, and a weakly localized chain can slide along the flat orbit toward the balanced region anyway.
3. **RMSProp-preconditioned SGLD leaves most of the drop** (`−65` points, against about `−75` for vanilla). We used $a = 0.01$, with the base step scaled to $\epsilon a$ so that flat directions keep step $\epsilon$. Up to the usual bias of adaptive preconditioning without a correction term, this targets the same localized distribution. So it can speed up mixing, which may account for the 10-point difference, but it cannot remove a gap that is in the target.
4. **A metric in the localization term removes the drop.** We build a diagonal metric from the balancing scales and put it in the localization term. SGLD at the raw weights then gives a flat curve (`−0.3`). By a change of variables this is the same computation as the balanced estimate, so it is not separate evidence.

## What we are unsure about

- **Whether this explains your curve.** Our runs may simply drift further than yours. If so, more of your 49 points could be real. A first check needs no checkpoints: our raw parameter norm at the trough is `241.7`, and if your weight-norm logs are handy, comparing against them would show whether your runs drift as far. If you still have the checkpoints, running the balancing on one seed at `46k` and near `364k` would settle it. Our code works on a plain state dict.
- **Whether the balanced point is the right reference.** The $\gamma$ sweep is our only direct argument for it. A check that doesn't depend on the gauge would be the numerical rank of the Gauss–Newton matrix at the peak and the trough, which is unchanged by any loss-preserving reparametrization. We haven't run it yet.
- **Whether the true LLC changes over this window.** How much the estimate moves at a fixed function depends on the metric. It ranges from `+0.6` (balanced point, isotropic localization) to about `−16` (a per-tensor RMS metric). The robust result is that the raw-versus-balanced gap closes.
- **Whether the language models show anything similar.** They were trained with AdamW. In our AdamW (`wd = 0.08`) rerun, the gap stayed within about one point.

## Materials

The code, full tables and caveats are at [github.com/bubskee/eat-your-fiber](https://github.com/bubskee/eat-your-fiber): see [`notes/devinterp_fiber_decomposition.md`](https://github.com/bubskee/eat-your-fiber/blob/main/notes/devinterp_fiber_decomposition.md) and the [figure](https://github.com/bubskee/eat-your-fiber/blob/main/experiments/results/devinterp_fiber_decomposition.png). All 114 checkpoints are committed as `.safetensors` with a SHA-256 manifest, so nothing needs retraining:

```bash
python experiments/verify_upstream_icl_parity.py --icl-repo /path/to/timaeus-research/icl
python experiments/devinterp_fiber_decomposition.py --from-checkpoints --eval-seeds 0,1,2
python experiments/devinterp_fiber_decomposition.py --replot
```

*Much of the code, the experiments and the writing were done with AI agents, directed by Bobby Faber. Corrections are very welcome.*
