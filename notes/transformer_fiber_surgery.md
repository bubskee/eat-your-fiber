# Track 2: LoRA Fiber Surgery on a 2-Layer Causal Transformer

## Status of Each Claim

- `[LC]` (Lean):
  - **Complete unconditional 2D $(r^*, p, q)$ RRR zero-loss fiber stratification:** `rrr_fiber_exact_pair`, `rrr_fiber_subcritical_plateau`, `rrr_fiber_A_supercritical_parabola`, `rrr_fiber_B_supercritical_parabola`, `rrr_supercritical_mutually_exclusive`, `rrr_fiber_eq_min_component_codim` in [`lean/EatYourFiber/RRRFiberStratification.lean`](../lean/EatYourFiber/RRRFiberStratification.lean).
    - Axiom audit: each depends only on axioms $\subseteq$ `{propext, Classical.choice, Quot.sound}`.
  - **Single-head self-attention all-orders dichotomy:** $V{+}O$ ($4\times$ supercritical parabola) vs. $Q{+}K$ (exact $GL(q)$ gauge symmetry and flat cancellation $\lambda_{Q+K}(q) \equiv dH - H^2/2$): `attention_VO_exact_pair`, `attention_QK_flat_pair`, `attention_QK_general_flat_pair`, `attention_coupling_dichotomy_standard_init` in [`lean/EatYourFiber/AttentionLoRA.lean`](../lean/EatYourFiber/AttentionLoRA.lean).
  - **Scope:** these certify the RRR / linearized-attention losses. Transferring their $\lambda$ values to the Transformer's next-token KL relies on the `[paper]` pullback-invariance argument below.
- `[paper]` (local differential / Gauss–Newton analysis, not formalized in Lean):
  - Pullback bilipschitz invariance of local learning coefficients through nonlinear Transformer layers (RMSNorm, causal softmax, GELU MLP, LM head, next-token KL divergence) whenever $\mathcal{H}_{\mathrm{KL}} \succ 0$.
  - Pre-LayerNorm exact $\mathbf{1}_d$-kernel output-dimension drop ($m_{\mathrm{eff}} = d - 1$).
  - Multi-head shared-LoRA transversality decoupling ($d_h + q \le d$).
  - Fiber-tangent plasticity operator $\Delta(\Delta W) = -\eta(G A^\top A + B B^\top G)$.
- `[empirical]`:
  - Real 2-layer causal Transformer pretraining; 25-point 2D $(p, q)$ exact iso-functional fiber surgery; Pre-LayerNorm symmetry-breaking check; 5-seed continual-learning / noisy-overfitting / directional-plasticity experiments (SGD & Adam); 5-optimizer 2D phase-diagram trajectories; late-stage $\Delta\hat\lambda$ functional-vs.-fiber decompositions; $n_{\mathrm{heads}} \in \{1, 2, 4\}$ attention coupling sweeps.
  - Code: [`experiments/transformer_fiber_surgery.py`](../experiments/transformer_fiber_surgery.py). Raw data: [`experiments/results/transformer_fiber_surgery.json`](../experiments/results/transformer_fiber_surgery.json). 6-panel figure: [`experiments/results/transformer_fiber_surgery.png`](../experiments/results/transformer_fiber_surgery.png).
- `[conjectural]`: extrapolation of the continual-learning plasticity–stability splits and the late-stage DevInterp LLC fiber-sliding decompositions to billion-parameter Transformers under stochastic mini-batch schedules.

---

## Summary

### Setup

- **Goal:** test the Lean-certified Track 2 fiber geometry (`RRRFiberStratification.lean`, `AttentionLoRA.lean`) on a pretrained 2-layer causal Transformer language model.
- **Model:** token + positional embeddings, Pre-RMSNorm (plus a Pre-LayerNorm GPT-2 twin), causal multi-head self-attention, residual stream, $2\times$ GELU MLP, LM head.
- **Training:** real next-token cross-entropy, to `100.0%` accuracy on a 3-pair associative-recall + modular-composition sequence task.
- **Loss studied:** next-token functional KL divergence $K(\theta) = \mathbb{E}_x[\mathrm{KL}(p_{\theta^*}(\cdot|x) \| p_\theta(\cdot|x))]$, averaged over 32 fixed evaluation sequences.
- **Estimator:** float64 MALA with a fixed full-matrix preconditioner. It keeps exact detailed balance and removes the $\sim 10^4\times$ Transformer Fisher anisotropy.

### Findings

1. **The 2D $(p, q)$ phase diagram reappears in the Transformer (`[empirical]`).**
   - **Setup:** 25 exact iso-functional surgery checkpoints on Layer 1 $W_O$ ($M=N=12, H=10, r^*=2$, excess $(m,n,k)=(10,10,8)$, $p_c = q_c = 4$). Every checkpoint has $\max|\Delta\mathrm{logit}| = 1.5\times 10^{-14}$ on held-out sequences.
   - **Subcritical plateau ($\lambda = 54.0$):** MALA sits on or below it; points run $0.4$–$4.9$ low at finite $\beta$. Full-norm balanced diagonal $(4,4) \to 53.11 \pm 0.57$.
   - **$A$-supercritical parabola $+\frac{1}{2}(q-4)^2$:** up to $(0,8) \to 61.65 \pm 0.50$ (theory $62.0$).
   - **$B$-supercritical parabola $+\frac{1}{2}(p-4)^2$:** up to $(8,0) \to 61.98 \pm 0.40$ (theory $62.0$).
   - **Off-axis mutual exclusivity:** $(1,6), (2,6), (6,1), (6,2) \to 55.09\text{--}56.14$ (theory $56.0$).
2. **Pre-LayerNorm breaks the $p \leftrightarrow q$ symmetry through an exact $\mathbf{1}_d$ Fisher kernel (`[paper]` + `[empirical]`).**
   - **Mechanism:** Layer-1 $\mathrm{LN}_2$ and final $\mathrm{LN}_f$ both subtract the mean vector $\mathbf{1}_d$. Any rank-1 perturbation $\mathbf{1}_d u^\top$ to residual-stream writes ($W_O$ or $W_{\mathrm{down}}$) changes output logits by identically $0$.
   - **Fisher check:** the exact $144 \times 144$ Gauss–Newton Fisher matrix of $W_O$ has $0$ zero eigenvalues under Pre-RMSNorm ($\lambda_{\min} = 7.33\times 10^{-5}$) and exactly $12$ under Pre-LayerNorm ($\lambda_{\min} = -2.55\times 10^{-17}$, $\lambda_{13} = 2.79\times 10^{-5}$).
   - **Consequence:** $m_{\mathrm{eff}} = d - 1 = 11$, so $(p_c, q_c)$ moves from $(4, 4) \to (3.5, 4.5)$, and the two tops differ by $4.0$ points: $\lambda(0,8) = 57.0$ vs. $\lambda(8,0) = 61.0$.
   - **Measured:** $56.38 \pm 0.72$ vs. $60.07 \pm 0.63$; exact Hessian `gauss_lam = 57.0 vs. 61.0`.
3. **Function-matched Task-A checkpoints split on Task B, and SGD and Adam differ (`[empirical]`).**
   - **Setup:** 5 Task-A checkpoints. Four are exact surgery points ($\max|\Delta\mathrm{logit}| = 1.7\times 10^{-14}$); the near-plateau `(0,0)+1e-3` differs by $6.2\times 10^{-4}$ (initial $\mathrm{KL}_A = 1.3\times 10^{-10}$). 5-seed fine-tuning on a shifted real-token associative-recall task (`shift_rule = 3`, real cross-entropy).
   - **SGD, clean Task B:**
     - `Exact (0,0)` and `Near-Plateau (0,0)+1e-3` stay near rank 2 ($\mathrm{erank}(\Delta W) = 2.18\text{--}2.41$), with recall accuracy $0.211\text{--}0.253$ and low Task-A forgetting ($\mathrm{KL}_A = 0.385\text{--}0.397$).
     - `Balanced (4,4)` ($0.384$), `A-Supercrit (0,8)` ($0.448$) and `B-Supercrit (8,0)` ($0.473$, $2.2\times$ higher Task-B accuracy) adapt faster, with higher Task-A forgetting ($\mathrm{KL}_A = 0.450\text{--}0.464$).
   - **SGD and Adam, noisy limited-sample Task B ($S=32$, 25% corruption):** `Exact (0,0)` has a $2.6\times$–$5.3\times$ smaller overfitting gap ($+0.126$ SGD, $+0.081$ Adam) and lower test CE ($1.872$ / $1.851$) than supercritical `(0,8)` / `(8,0)` (gaps $+0.333\text{--}0.433$, test CE $1.964\text{--}1.999$).
   - **SGD vs. Adam:**
     - `Exact (0,0)` locks both SGD and Adam to rank 2.
     - With a `1e-3` residual norm (`Near-Plateau (0,0)+1e-3`), SGD stays trapped ($0.253$ acc, $+0.179$ noisy gap).
     - Adam's $1/\sqrt{v}$ normalization breaks the saddle immediately ($0.595$ acc, $+0.371$ noisy gap, matching `(4,4)`, `(0,8)`, `(8,0)`).
   - **Directional plasticity:**
     - On orthogonal $V_\perp$-input vs. $U_\perp$-output shifts, `(0,8)` and `(8,0)` mirror each other under SGD ($0.785$ vs. $0.489$ on $V_\perp$; $0.718$ vs. $0.819$ on $U_\perp$) and Adam ($0.937$ vs. $0.848$; $0.915$ vs. $0.960$).
     - `Balanced (4,4)` combines a low subcritical LLC ($\lambda = 54.0$) with fast plasticity on both axes ($0.658 / 0.752$ SGD; $0.941 / 0.957$ Adam).
4. **Real optimizer trajectories map onto the 2D $(p_{\mathrm{eff}}, q_{\mathrm{eff}})$ phase diagram, and the late-stage AdamW LLC drop is $\approx 95\%$ ($\pm 6$ pts) iso-functional fiber sliding (`[empirical]`).**
   - **`LoRA + Adam (wd=0)`:** stays in the $A$-supercritical region at $(p_{\mathrm{eff}}, q_{\mathrm{eff}}) = (0.24, 5.57)$, $\hat\lambda = 58.44 \pm 0.53$.
   - **LoRA+ ($\eta_B = 12\,\eta_A$, Hayou et al. 2024):** further along the $A$-supercritical axis at $(0.00, 6.73)$, $\hat\lambda = 61.47 \pm 0.46$. Shrinking $\eta_A$ freezes $A_{\mathrm{in}}$ more tightly at its full-rank init.
   - **`LoRA + AdamW`, `LoRA + SGD+WD`, `Balanced Small Init`:** inside the 2D subcritical plateau ($\hat\lambda = 45.72\text{--}47.62$ vs. exact $(0,0)$ ref $49.96 \pm 0.49$).
   - **Late-stage decomposition** (exact balanced-SVD surgery, $\max|\Delta\mathrm{logit}| \le 5.6\times 10^{-14}$):
     - `LoRA + AdamW (2500 -> 8000)`, after Task-A KL has converged (`1.53e-8 -> 3.48e-8`): $\Delta\hat\lambda_{\mathrm{total}} = -10.79 \pm 0.83$, split into $\Delta\hat\lambda_{\mathrm{func}} = -0.51 \pm 0.69$ (zero within $1\sigma$) and $\Delta\hat\lambda_{\mathrm{fiber}} = -10.28 \pm 1.07$.
     - Fiber share: $95.3\% \pm 6.4$ pts, defined as $1 - \Delta\hat\lambda_{\mathrm{func}}/\Delta\hat\lambda_{\mathrm{total}}$, with a delta-method 1-SE error from the independent raw and balanced MALA runs. Consistent with 100%.
     - `LoRA + Adam (ctrl)` is flat ($+0.85 \pm 0.76$).
     - `LoRA + SGD+WD (1500 -> 9000)` starts from a still-overparameterized momentum-SGD overshoot (`dW_err = 2.3e-2`). It shows a large functional component ($\Delta\hat\lambda_{\mathrm{func}} = -14.83 \pm 0.87$, `dW_err` $2.3\times 10^{-2} \to 4.5\times 10^{-3}$) alongside fiber collapse.
5. **The single-head $V{+}O$ vs. $Q{+}K$ dichotomy and the multi-head reversal hold at the $q = 4$ tops (`[empirical]`).**
   - **Setup:** $n_{\mathrm{heads}} \in \{1, 2, 4\}$ ($d=12, H=4$), through causal softmax, RMSNorm, GELU MLP, LM head and next-token KL.
   - **$n_{\mathrm{heads}} = 1$:**
     - $V{+}O$ rises to $\hat\lambda(4) = 45.10 \pm 0.40$ (`#H = 96` Hessian eigenvalues above $\gamma$, `g_lam = 48.0`).
     - $Q{+}K$ is capped by the $16$-dimensional $GL(4)$ gauge symmetry at $\hat\lambda(4) = 38.48 \pm 0.37$ (`#H = 80`, `g_lam = 39.8`).
   - **$n_{\mathrm{heads}} = 2$ and $4$:**
     - Cross-head transversality ($d_h + q \le 12$) breaks $GL(4)$ completely: `#H = 96` for both channels.
     - MALA lands on $48.0$ for both $V{+}O$ ($47.98 \pm 0.12$ at 2-head, $47.95 \pm 0.18$ at 4-head) and $Q{+}K$ ($47.86 \pm 0.31$ at 2-head, $48.47 \pm 0.28$ at 4-head).
   - **Caveat:** 1-head $Q{+}K$ is not flat at finite $\beta$ (`27.09 → 38.48`).

---

## 0. Model and Pretraining Setup (`[empirical]`)

- **Architecture (`CausalTransformerLM`):**
  - Vocabulary $V = 24$, context length $T = 12$, $d_{\mathrm{model}} = 12$, $d_{\mathrm{mlp}} = 24$ ($2\times$ GELU MLP).
  - 2 causal Pre-RMSNorm Transformer blocks (plus a Pre-LayerNorm twin for Part 1's comparison), final normalization, untied LM head.
  - Parameters: `3,180` for Pre-RMSNorm, `3,240` for Pre-LayerNorm (all model parameters, LoRA factors excluded).
- **Pretraining task:** 3-pair Associative Recall + Shifted Modular Composition, `[k0, v0, k1, v1, k2, v2, mod, q1, r1, c1, q2, r2]`.
  - 3 distinct keys (`0..7`), 3 values (`8..15`), modifier (`16..23`), two distinct queries (`q1, q2`).
  - Targets: recalled values `r1 = val(q1)`, `r2 = val(q2)`, and modular sum `c1 = 8 + ((r1 - 8 + mod - 16) % 8)`.
- **Training:** online batches (`bs = 128`, 3,000 steps), AdamW `lr = 1.2e-2`, cosine schedule, `wd = 2e-3`, real next-token cross-entropy across all 11 positions.

| Pretrained Model | Norm | $n_{\mathrm{heads}}$ | $d_h$ | Val CE | Recall `r1` Acc | Comp `c1` Acc | Recall `r2` Acc | Time |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Base (Parts 1–4) | Pre-RMSNorm | 2 | 6 | `1.2873` | 1.000 | 1.000 | 1.000 | 10.9s |
| Pre-LN twin (Part 1) | Pre-LayerNorm | 2 | 6 | `1.5484` | 0.461 | 0.643 | 0.520 | 10.9s |
| 1-Head twin (Part 4) | Pre-RMSNorm | 1 | 12 | `1.4535` | 0.465 | 1.000 | 0.494 | 7.8s |
| 4-Head twin (Part 4) | Pre-RMSNorm | 4 | 3 | `1.4396` | 0.451 | 1.000 | 0.551 | 12.8s |

- **MALA with exact full-matrix linear preconditioning:**
  - **Problem:** the next-token KL Gauss–Newton Fisher $\mathcal{H}_{\mathrm{KL}}$ of Layer 1 $W_O$ has condition number $\sim 5.8\times 10^3$ ($\lambda_{\min} = 7.33\times 10^{-5}$, $\lambda_{\max} = 4.25\times 10^{-1}$), rotated into the model's learned eigenbasis. Diagonal preconditioning alone mixes slowly along soft off-diagonal eigenmodes.
  - **Preconditioner:** a fixed linear map $w = w_0 + P z$, computed once before burn-in from the exact local Hessian $H_0 = \nabla^2(\beta K)(w_0)$ (plus quartic-scale probes in the soft/null eigenspace). Float64 MALA runs in $z$-space.
  - **Detailed balance:** $P$ is a constant invertible linear map independent of the chain state, so MALA satisfies exact detailed balance for $\pi(w) \propto \exp(-\beta K(w) - \frac{\gamma}{2}\|w - w_0\|_2^2)$.
  - **Caching:** frozen Layer-0/pre-$W_O$ activations are cached; this matches the full 2-layer forward pass to $\max|\Delta\mathrm{logit}| = 1.07\times 10^{-14}$.

---

## 1. Part 1: Iso-Functional 2D $(p, q)$ Fiber Surgery (`[empirical]`)

### 1.1 Setup and Function-Identity Check

- **Adapter:** the pretrained 2-layer Pre-RMSNorm Transformer (`h=2`) is frozen; a rank-$H = 10$ LoRA adapter $(B, A)$ is attached to Layer 1's $12 \times 12$ attention output projection $W_O$.
- **Task A:** next-token KL distillation toward a rank-$r^* = 2$ adaptation $\Delta W_A^* = U_A[:, :2]\operatorname{diag}(0.8, 0.6) V_A[:, :2]^\top$.
  - $\mathrm{KL}(\text{Task A} \| \text{base}) = 0.4134\text{ nats}$; converged LoRA reaches $\mathrm{KL}_A = 3.18\times 10^{-7}$.
- **Predicted values** (`RRRFiberStratification.lean`, `[LC]` for the RRR loss; using them for the Transformer KL is the `[paper]` pullback step):
  - Excess dimensions $(m, n, k) = (10, 10, 8)$ ($m+n+k = 28$ even, $m_{\log} = 1$), so $p_c = q_c = 4$.
  - Active $\lambda_{\mathrm{act}} = 22.0$; excess plateau $\lambda_{\mathrm{AW}}(10,10,8) = 32.0$; total plateau $\lambda(p, q) = 54.0$ for $p \le 4, q \le 4$.
  - Symmetric supercritical parabolas $\lambda = 54.0 + \frac{1}{2}(q - 4)_+^2$ ($q > 4$) and $54.0 + \frac{1}{2}(p - 4)_+^2$ ($p > 4$), peaking at $62.0$.
- **Checkpoints:** canonical iso-functional $(B', A')$ with inactive singular scale $s_{\mathrm{in}} = 2.0$, at 25 points of the $(p, q)$ diagram.
  - $B'A' = \Delta W_A^*$ to machine precision.
  - $\max|\Delta\mathrm{logit}| = 1.5\times 10^{-14}$ across all 64 held-out sequences at every checkpoint.

### 1.2 25-Point MALA Results

- **Settings:** $\beta = 10^9, \gamma = 5.0$, 20 chains, 1,100 burn-in + 1,500 sample steps.

| Slice | $(p, q)$ | Lean `[LC]` $\lambda$ | MALA $\hat\lambda \pm \mathrm{SE}$ | Error | Acc | Split-half `hdiff` | Local `gauss_lam` | $\max|\Delta\mathrm{logit}|$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| $A$-axis | `(0, 0)` | 54.0 | 49.08 ± 0.45 | -4.92 | 0.39 | +1.18 | 22.0 | `1.5e-14` |
| $A$-axis | `(0, 1)` | 54.0 | 49.14 ± 0.55 | -4.86 | 0.39 | -0.42 | 27.0 | `1.5e-14` |
| $A$-axis | `(0, 2)` | 54.0 | 50.45 ± 0.65 | -3.55 | 0.38 | -0.16 | 32.0 | `1.5e-14` |
| $A$-axis | `(0, 3)` | 54.0 | 51.86 ± 0.61 | -2.14 | 0.38 | +1.47 | 37.0 | `1.5e-14` |
| $A$-axis | `(0, 4)` | 54.0 | 52.24 ± 0.46 | -1.76 | 0.38 | -0.28 | 42.0 | `1.5e-14` |
| $A$-axis | `(0, 5)` | 54.5 | 53.75 ± 0.48 | -0.75 | 0.38 | +1.64 | 47.0 | `1.5e-14` |
| $A$-axis | `(0, 6)` | 56.0 | 56.10 ± 0.55 | +0.10 | 0.40 | -0.26 | 52.0 | `1.5e-14` |
| $A$-axis | `(0, 7)` | 58.5 | 58.15 ± 0.25 | -0.35 | 0.37 | -0.34 | 57.0 | `1.5e-14` |
| $A$-axis | `(0, 8)` | 62.0 | 61.65 ± 0.50 | -0.35 | 0.35 | +1.22 | 62.0 | `1.5e-14` |
| $B$-axis | `(1, 0)` | 54.0 | 50.79 ± 0.44 | -3.21 | 0.37 | +1.25 | 27.0 | `1.5e-14` |
| $B$-axis | `(2, 0)` | 54.0 | 50.52 ± 0.48 | -3.48 | 0.41 | -0.26 | 32.0 | `1.5e-14` |
| $B$-axis | `(3, 0)` | 54.0 | 52.39 ± 0.34 | -1.61 | 0.39 | -1.11 | 37.0 | `1.5e-14` |
| $B$-axis | `(4, 0)` | 54.0 | 53.42 ± 0.50 | -0.58 | 0.42 | +1.85 | 42.0 | `1.5e-14` |
| $B$-axis | `(5, 0)` | 54.5 | 54.70 ± 0.44 | +0.20 | 0.39 | -0.20 | 47.0 | `1.5e-14` |
| $B$-axis | `(6, 0)` | 56.0 | 55.61 ± 0.36 | -0.39 | 0.40 | -0.14 | 52.0 | `1.5e-14` |
| $B$-axis | `(7, 0)` | 58.5 | 58.56 ± 0.42 | +0.06 | 0.40 | -0.02 | 57.0 | `1.5e-14` |
| $B$-axis | `(8, 0)` | 62.0 | 61.98 ± 0.40 | -0.02 | 0.39 | -0.10 | 62.0 | `1.5e-14` |
| Balanced diag | `(1, 1)` | 54.0 | 51.10 ± 0.40 | -2.90 | 0.38 | +1.16 | 31.5 | `1.5e-14` |
| Balanced diag | `(2, 2)` | 54.0 | 52.20 ± 0.41 | -1.80 | 0.35 | +0.91 | 40.0 | `1.5e-14` |
| Balanced diag | `(3, 3)` | 54.0 | 53.57 ± 0.36 | -0.43 | 0.40 | -0.88 | 47.5 | `1.5e-14` |
| Balanced diag | `(4, 4)` | 54.0 | 53.11 ± 0.57 | -0.89 | 0.37 | -0.83 | 54.0 | `1.5e-14` |
| Off-axis | `(1, 6)` | 56.0 | 55.09 ± 0.28 | -0.91 | 0.38 | +0.06 | 54.0 | `1.5e-14` |
| Off-axis | `(2, 6)` | 56.0 | 55.48 ± 0.47 | -0.52 | 0.39 | -0.17 | 56.0 | `1.5e-14` |
| Off-axis | `(6, 1)` | 56.0 | 55.64 ± 0.27 | -0.36 | 0.43 | +0.22 | 54.0 | `1.5e-14` |
| Off-axis | `(6, 2)` | 56.0 | 56.14 ± 0.32 | +0.14 | 0.39 | +0.35 | 56.0 | `1.5e-14` |

Each feature of `RRRFiberStratification.lean` is reproduced on the Transformer:

- **Subcritical plateau:** the expected negative finite-$\beta$ logarithmic offset ($-0.4$ to $-4.9$).
  - Largest ($-2.9$ to $-4.9$) at the six most singular points, where local `gauss_lam` is only $22\text{--}32$ and the remaining $22\text{--}32$ points come purely from quartic singular integrals.
- **Supercritical parabolas:** within $\le 0.39$ at all 6 on-axis supercritical points $r \in \{6, 7, 8\}$.
- **Flat plateau up to the full-norm balanced corner `(4, 4)`:** `gauss_lam = 54.0`, MALA `53.11 ± 0.57`.
- **Subcritical-coordinate independence:** holds at all four off-axis points `(1,6), (2,6), (6,1), (6,2)`.

### 1.3 Pre-LayerNorm $\mathbf{1}_d$-Kernel and $p \leftrightarrow q$ Asymmetry (`[paper]` + `[empirical]`)

In a Pre-LayerNorm Transformer, both $\mathrm{LN}_2$ and $\mathrm{LN}_f$ project out the constant vector $\mathbf{1}_d$:
$$\mathrm{LN}_2(x_1 + c\,\mathbf{1}_d) \equiv \mathrm{LN}_2(x_1) \implies \mathrm{LN}_f\!\bigl(x_1 + c\,\mathbf{1}_d + \mathrm{MLP}(\mathrm{LN}_2(x_1 + c\,\mathbf{1}_d))\bigr) \equiv \mathrm{LN}_f(x_{\mathrm{final}}).$$

- **Claim:** the $d$-dimensional subspace $\{\mathbf{1}_d u^\top : u \in \mathbb{R}^d\}$ lies identically in $\ker(\mathcal{H}_{\mathrm{KL}})$ of Layer 1 $W_O$.
- **Exact Hessian check:**
  - Pre-RMSNorm `Fisher(W_O)`: `0` zero eigenvalues ($\lambda_{\min} = 7.33\times 10^{-5}$, $\lambda_{\mathrm{med}} = 4.69\times 10^{-3}$, $\lambda_{\max} = 4.25\times 10^{-1}$).
  - Pre-LayerNorm `Fisher(W_O)`: exactly `12` zero eigenvalues ($\lambda_{\min} = -2.55\times 10^{-17}$, $\lambda_{13} = 2.79\times 10^{-5}$, $\lambda_{\max} = 5.31\times 10^{-1}$).
- **Consequence:**
  - Effective output dimension drops to $M_{\mathrm{eff}} = 11$ ($m = 9, n = 10, k = 8$; $m+n+k = 27$ odd $\implies m_{\log} = 2$ on the plateau).
  - $p_c = 3.5 \ne q_c = 4.5$, giving a $4.0$-point asymmetry between $(0, 8)$ and $(8, 0)$.

| Pre-LayerNorm $(p, q)$ | Theory ($M_{\mathrm{eff}}=11, N=12$) | $m_{\log}$ | Local `gauss_lam` | MALA $\hat\lambda \pm \mathrm{SE}$ | Acc | `hdiff` |
|---:|---:|---:|---:|---:|---:|---:|
| `(0, 0)` | 51.0 | 2 | 21.0 | 47.28 ± 0.42 | 0.39 | +0.64 |
| `(0, 8)` | 57.0 | 1 | 57.0 | 56.38 ± 0.72 | 0.40 | -0.96 |
| `(8, 0)` | 61.0 | 1 | 61.0 | 60.07 ± 0.63 | 0.39 | +0.03 |

- **Result:** Pre-LayerNorm lowers the standard LoRA ($A$-supercritical) top by $5.0$ points ($62 \to 57$) and the $B$-supercritical top by only $1.0$ point ($62 \to 61$).
- **Reading:** LayerNorm's shift invariance acts like a structural dead neuron on residual-stream writes.

---

## 2. Part 2: Continual Learning from Function-Matched Checkpoints (`[empirical]`)

### 2.1 Mechanism: The Fiber Tangent Operator (`[paper]`)

- **Same start:** at $t = 0$ of Task B fine-tuning, all iso-functional checkpoints share the effective weight $\Delta W_0 = B' A' = \Delta W_A^*$. So they have the same Task-B loss $L_B(\Delta W_0)$ and the same weight-space gradient $G_0 = \nabla_{\Delta W} L_B(\Delta W_0)$.
- **Different first step:** the first-order effective-weight update under GD/SGD depends on the fiber coordinates:
$$\Delta(\Delta W) = -\eta\Bigl(G_0\,(A')^\top A' + B' (B')^\top G_0\Bigr) + O(\eta^2).$$
- **By checkpoint:**
  - `(0, 0)`: $B'_{\mathrm{in}} = A'_{\mathrm{in}} = 0$ leaves all $k = 8$ inactive slots on an exact invariant singular subspace ($\nabla_{B_{\mathrm{in}}} = \nabla_{A_{\mathrm{in}}} = 0$).
  - `(0, 8)`: $(A'_{\mathrm{in}})^\top A'_{\mathrm{in}}$ gives immediate rank-8 plasticity on the 10-dim $V_\perp$.
  - `(8, 0)`: $B'_{\mathrm{in}}(B'_{\mathrm{in}})^\top$ gives immediate rank-8 plasticity on the 10-dim $U_\perp$.
  - `(4, 4)`: inside the flat 2D plateau, with the same low $\lambda = 54.0$ as `(0, 0)`; both $B'_{\mathrm{in}}$ and $A'_{\mathrm{in}}$ carry rank-4 live plasticity.
- **Experimental setup:**
  - 5 checkpoints, all verified at $\max|\Delta\mathrm{logit}| = 1.7\times 10^{-14}$ on held-out Task-A sequences, except `Near-Plateau (0,0)+1e-3` at $\max|\Delta\mathrm{logit}| = 6.2\times 10^{-4}$.
  - 5 random seeds, each independently rotating the inactive orthogonal complements $U_\perp, V_\perp$ and resampling the Task-B datasets.
  - Both SGD and Adam.

### 2.2 Regime 2a: Clean Shifted-Recall Task B

- **Settings:** real-token shifted recall, `shift_rule = 3`, $S_{\mathrm{train}}=192, S_{\mathrm{test}}=256$; initial `CE_B = 3.5363`, `r1_acc = 0.051`.

| Opt | Checkpoint | Test CE@20 | Final Test CE | Task-B `r1` Acc | Task-A Forgetting $\mathrm{KL}_A$ | Update $\|\Delta W\|_F$ | $\mathrm{erank}(\Delta W)$ |
|---|---|---:|---:|---:|---:|---:|---:|
| SGD | `Exact (0,0)` | 1.785 ± 0.014 | 1.699 ± 0.006 | 0.211 ± 0.014 | 0.385 ± 0.004 | 4.03 | 2.18 |
| SGD | `Near (0,0)+1e-3` | 1.785 ± 0.014 | 1.690 ± 0.023 | 0.253 ± 0.035 | 0.397 ± 0.021 | 4.77 | 2.41 |
| SGD | `Balanced (4,4)` | 1.799 ± 0.011 | 1.686 ± 0.014 | 0.384 ± 0.036 | 0.450 ± 0.011 | 6.90 | 3.41 |
| SGD | `A-Supercrit (0,8)` | 1.791 ± 0.014 | 1.646 ± 0.011 | 0.448 ± 0.016 | 0.458 ± 0.024 | 7.09 | 3.47 |
| SGD | `B-Supercrit (8,0)` | 1.865 ± 0.091 | 1.638 ± 0.010 | 0.473 ± 0.044 | 0.464 ± 0.015 | 7.14 | 3.35 |
| Adam | `Exact (0,0)` | 1.851 ± 0.004 | 1.679 ± 0.004 | 0.228 ± 0.019 | 0.349 ± 0.004 | 3.16 | 2.32 |
| Adam | `Near (0,0)+1e-3` | 1.824 ± 0.004 | 1.591 ± 0.004 | 0.595 ± 0.014 | 0.524 ± 0.017 | 6.55 | 3.53 |
| Adam | `Balanced (4,4)` | 1.798 ± 0.005 | 1.586 ± 0.002 | 0.595 ± 0.005 | 0.548 ± 0.013 | 6.83 | 3.79 |
| Adam | `A-Supercrit (0,8)` | 1.786 ± 0.001 | 1.594 ± 0.006 | 0.609 ± 0.006 | 0.532 ± 0.014 | 6.70 | 3.70 |
| Adam | `B-Supercrit (8,0)` | 1.810 ± 0.006 | 1.591 ± 0.004 | 0.608 ± 0.013 | 0.554 ± 0.023 | 6.95 | 3.92 |

### 2.3 Regime 2b: Noisy Limited-Sample Task B

- **Settings:** real-token Task B, $S_{\mathrm{train}}=32$, 25% target corruption, clean $S_{\mathrm{test}}=256$.

| Opt | Checkpoint | Train CE | Clean Test CE | Overfit Gap (Te $-$ Tr) | Task-A $\mathrm{KL}_A$ | $\mathrm{erank}(\Delta W)$ |
|---|---|---:|---:|---:|---:|---:|
| SGD | `Exact (0,0)` | 1.746 | 1.872 ± 0.021 | +0.126 ± 0.015 | 0.520 ± 0.020 | 2.03 |
| SGD | `Near (0,0)+1e-3` | 2.143 | 2.323 ± 0.423* | +0.179 ± 0.026 | 1.023 ± 0.455* | 2.27 |
| SGD | `Balanced (4,4)` | 1.600 | 1.992 ± 0.027 | +0.392 ± 0.040 | 0.733 ± 0.032 | 4.54 |
| SGD | `A-Supercrit (0,8)` | 1.605 | 1.984 ± 0.032 | +0.379 ± 0.040 | 0.729 ± 0.039 | 5.09 |
| SGD | `B-Supercrit (8,0)` | 1.643 | 1.976 ± 0.038 | +0.333 ± 0.062 | 0.711 ± 0.045 | 4.95 |
| Adam | `Exact (0,0)` | 1.770 | 1.851 ± 0.012 | +0.081 ± 0.017 | 0.459 ± 0.012 | 1.87 |
| Adam | `Near (0,0)+1e-3` | 1.590 | 1.960 ± 0.016 | +0.371 ± 0.033 | 0.736 ± 0.023 | 4.07 |
| Adam | `Balanced (4,4)` | 1.567 | 1.989 ± 0.031 | +0.422 ± 0.048 | 0.763 ± 0.043 | 4.15 |
| Adam | `A-Supercrit (0,8)` | 1.559 | 1.964 ± 0.042 | +0.405 ± 0.055 | 0.746 ± 0.046 | 4.07 |
| Adam | `B-Supercrit (8,0)` | 1.565 | 1.999 ± 0.021 | +0.433 ± 0.048 | 0.821 ± 0.033 | 4.64 |

- **\*Caveat (`Near (0,0)+1e-3`, noisy SGD):** 1 of 5 seeds briefly spiked during late SGD steps on the small noisy batch (`lr=0.8`).
  - This inflates the mean/SE of test CE (`±0.423`) and `KL_A` (`±0.455`).
  - Its generalization gap stays tightly bounded (`+0.179 ± 0.026`) and its rank low (`2.27`).

### 2.4 Regime 2c: Directional Plasticity ($V_\perp$-Input vs. $U_\perp$-Output Task B)

- **Metric:** relative KL drop at step 25 (`(KL_0 - KL_25) / KL_0`, mean ± SE over 5 seeds); final residual KL fraction is `KL_100 / KL_0`.

| Task-B Alignment | Opt | `Exact (0,0)` | `Near (0,0)+1e-3` | `Balanced (4,4)` | `A-Supercrit (0,8)` | `B-Supercrit (8,0)` |
|---|---|---:|---:|---:|---:|---:|
| $V_\perp$ (input-aligned) | SGD | 0.475 ± 0.055 | 0.475 ± 0.055 | 0.658 ± 0.067 | 0.785 ± 0.031 | 0.489 ± 0.070 |
| $U_\perp$ (output-aligned) | SGD | 0.626 ± 0.071 | 0.626 ± 0.071 | 0.752 ± 0.083 | 0.718 ± 0.053 | 0.819 ± 0.059 |
| $V_\perp$ (input-aligned) | Adam | 0.600 ± 0.048 | 0.926 ± 0.013 | 0.941 ± 0.007 | 0.937 ± 0.003 | 0.848 ± 0.021 |
| $U_\perp$ (output-aligned) | Adam | 0.725 ± 0.031 | 0.938 ± 0.008 | 0.957 ± 0.006 | 0.915 ± 0.013 | 0.960 ± 0.002 |

### 2.5 Summary of Part 2

1. **`Exact (0,0)` traps both optimizers.**
   - The invariant subspace $B_{\mathrm{in}} = A_{\mathrm{in}} = 0$ keeps both SGD and Adam at rank $\approx 2$.
   - Relative to `(4,4)`, `(0,8)` and `(8,0)`: the noisy overfitting gap is $2.6\times$–$5.3\times$ smaller and clean-Task-B forgetting of Task A is $14$–$37\%$ lower.
   - Cost: $1.8\times$–$2.7\times$ lower clean Task-B recall accuracy.
2. **A near-plateau checkpoint separates SGD from Adam.**
   - `(0,0)+1e-3` is representative of finite-step AdamW.
   - SGD stays strongly trapped ($0.253$ acc, $+0.179$ gap).
   - Adam's adaptive normalization breaks the saddle immediately ($0.595$ acc, $+0.371$ gap, indistinguishable from `(0,8)` on isotropic Task B).
   - So in practice, fiber-induced plasticity–stability locking is strong under SGD and exact-zero pruning, but largely washed out on non-directional tasks when continuing with unregularized Adam from a nonzero `1e-3` residual.
3. **Directional fiber geometry survives under Adam.**
   - On directional shifts (Regime 2c), `A-Supercrit (0,8)` and `B-Supercrit (8,0)` show a mirror-image asymmetry under both SGD and Adam.
   - `Balanced (4,4)` sits inside the low-complexity 2D plateau ($\lambda = 54.0$, zero supercritical penalty) and adapts fast and symmetrically on both input and output shifts ($0.941$ and $0.957$ under Adam; $0.658$ and $0.752$ under SGD).

---

## 3. Part 3: Optimizer Trajectories and the Functional vs. Iso-Functional $\Delta\hat\lambda$ Decomposition (`[empirical]`)

### 3.1 Five Optimizer Trajectories on Task A

- **Settings:** $\beta = 5\times 10^8, \gamma = 5.0$; exact references `(0,0) = 49.96 ± 0.49`, `(0,8) = 61.76 ± 0.37`.

| Optimizer / Init | Final Step | Final $\mathrm{KL}_A$ | $\|\Delta\hat W - \Delta W_A^*\|_F$ | Final $(p_{\mathrm{eff}}, q_{\mathrm{eff}})$ | $\|B_{\mathrm{in}}\|_F$ | $\|A_{\mathrm{in}}\|_F$ | Final $\hat\lambda \pm \mathrm{SE}$ | Acc / `hdiff` | Phase Region |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `LoRA + Adam (wd=0)` | 4,500 | `4.72e-09` | `2.97e-03` | `(0.24, 5.57)` | 0.249 | 5.971 | 58.44 ± 0.53 | 0.43 / -0.75 | $A$-supercritical |
| `LoRA+ (eta_B=12*eta_A)` | 4,500 | `8.16e-08` | `1.21e-02` | `(0.00, 6.73)` | 0.030 | 6.971 | 61.47 ± 0.46 | 0.39 / -0.35 | Deep $A$-supercritical |
| `LoRA + AdamW (wd=0.12)` | 8,000 | `3.48e-08` | `3.95e-03` | `(0.04, 0.19)` | 0.087 | 0.214 | 47.03 ± 0.46 | 0.48 / -0.48 | 2D subcritical plateau |
| `LoRA + SGD+WD (mom=0.9)` | 9,000 | `6.61e-08` | `4.48e-03` | `(0.01, 0.01)` | 0.050 | 0.051 | 45.72 ± 0.58 | 0.53 / +2.14 | 2D subcritical plateau |
| `Balanced Small Init (2e-3)` | 4,500 | `8.68e-09` | `2.41e-03` | `(0.40, 0.16)` | 0.351 | 0.189 | 47.62 ± 0.54 | 0.43 / +0.40 | 2D subcritical plateau |

- **LoRA+ (Hayou et al. 2024):**
  - **Expectation:** raising $\eta_B / \eta_A = 12$ might push LoRA+ toward the $B$-supercritical side.
  - **Observed:** from standard LoRA initialization ($B_0 = 0, A_0 \sim \mathcal{N}(0, 0.7^2)$), shrinking $\eta_A = 0.003$ freezes $A_{\mathrm{in}}$ more tightly at its full-rank initialization ($\|A_{\mathrm{in}}\|_F = 6.971$, $q_{\mathrm{eff}} = 6.73$). The fast $B$ updates align with the active column space $U_{\mathrm{act}}$ ($\|B_{\mathrm{in}}\|_F = 0.030$, $p_{\mathrm{eff}} = 0.00$).
  - **Net:** LoRA+ ends further into the $A$-supercritical cone ($\hat\lambda = 61.47$ vs. $58.44$ for equal-LR Adam).

### 3.2 Late-Stage $\Delta\hat\lambda$ Decomposition: Functional Change vs. Iso-Functional Fiber Sliding

- **Balanced counterpart:** at any checkpoint $\theta_t = (B_t, A_t)$, the exact balanced-SVD surgery counterpart $\theta^{\mathrm{bal}}(f_t) = (U_H \Sigma_H^{1/2}, \Sigma_H^{1/2} V_H^\top)$ satisfies $B_{\mathrm{bal}} A_{\mathrm{bal}} = B_t A_t$ to machine precision ($\max|\Delta\mathrm{logit}| \le 5.6\times 10^{-14}$), with zero inactive fiber norm.
- **Decomposition** between late-stage converged checkpoints $t_1 \to t_2$:
$$\Delta\hat\lambda_{\mathrm{total}} = \underbrace{\bigl[\hat\lambda(\theta^{\mathrm{bal}}(f_{t_2})) - \hat\lambda(\theta^{\mathrm{bal}}(f_{t_1}))\bigr]}_{\Delta\hat\lambda_{\mathrm{func}}\text{ (functional change)}} + \underbrace{\bigl[(\hat\lambda(\theta_{t_2}) - \hat\lambda(\theta^{\mathrm{bal}}(f_{t_2}))) - (\hat\lambda(\theta_{t_1}) - \hat\lambda(\theta^{\mathrm{bal}}(f_{t_1})))\bigr]}_{\Delta\hat\lambda_{\mathrm{fiber}}\text{ (pure iso-functional fiber sliding)}}.$$

| Run ($t_1 \to t_2$) | $\mathrm{KL}_A(t_1 \to t_2)$ | $\hat\lambda_{\mathrm{raw}}(t_1 \to t_2)$ | $\hat\lambda_{\mathrm{bal}}(t_1 \to t_2)$ | $\Delta\hat\lambda_{\mathrm{total}}$ | $\Delta\hat\lambda_{\mathrm{func}}$ | $\Delta\hat\lambda_{\mathrm{fiber}}$ | Fiber Share |
|---|---:|---:|---:|---:|---:|---:|---:|
| `LoRA + AdamW` (`2500→8000`) | `1.53e-8 → 3.48e-8` | `57.83 → 47.03` | `42.17 → 41.66` | -10.79 ± 0.83 | -0.51 ± 0.69 | -10.28 ± 1.07 | 95.3% ± 6.4 |
| `LoRA + SGD+WD` (`1500→9000`) | `5.36e-7 → 6.61e-8` | `57.93 → 45.72` | `65.10 → 50.28` | -12.21 ± 0.68 | -14.83 ± 0.87 | +2.62 ± 1.11 | *func. rank pruning* |
| `LoRA + Adam (ctrl)` (`2500→4500`) | `6.75e-9 → 4.72e-9` | `57.60 → 58.44` | `35.57 → 34.75` | +0.85 ± 0.76 | -0.82 ± 0.61 | +1.67 ± 0.97 | *(flat control)* |

Relevance for Developmental Interpretability (DevInterp):

- **`LoRA + AdamW`: the drop is fiber sliding.**
  - After Task-A loss has converged (`KL_A ~ 1.5e-8`), a standard LLC curve drops by $-10.79 \pm 0.83$.
  - $\approx 95\%$ of that drop ($95.3\% \pm 6.4$ pts; $-10.28 \pm 1.07$) is iso-functional fiber sliding, with no detectable change in the balanced-checkpoint LLC ($\Delta\hat\lambda_{\mathrm{func}} = -0.51 \pm 0.69$; Task-A KL `1.5e-8 → 3.5e-8`).
  - Implication: without an iso-functional surgery check, a weight-decay fiber slide on an LLC curve could be misread as a late-stage circuit/developmental phase transition.
- **`LoRA + SGD+WD`: the drop is a functional change.**
  - At step 1500 of momentum SGD (`dW_err = 2.31e-2`, transient overshoot $\|B_{\mathrm{in}}\|_F = 2.43$), $B_{1500} A_{1500}$ is still off target, so $\hat\lambda_{\mathrm{bal}}(1500) = 65.10$ sits near the full-rank value.
  - Coupled `SGD+WD` then shrinks that residual to `dW_err = 4.5e-3` (Task-A KL `5.4e-7 → 6.6e-8`), lowering $\hat\lambda_{\mathrm{bal}}$ to `50.28` ($\Delta\hat\lambda_{\mathrm{func}} = -14.83 \pm 0.87$), while $(p_{\mathrm{eff}}, q_{\mathrm{eff}})$ collapses from $(1.91, 4.76) \to (0.01, 0.01)$.
  - The decomposition separates this case from the AdamW case.
- **Caveat (finite $\beta$):** for `LoRA_Adam` and `LoRA_AdamW`, $\hat\lambda_{\mathrm{bal}}$ sits at `35–42`, below the exact rank-2 zero-residual `(0,0)` reference `49.96`.
  - Cause: finite-step optimization leaves small residual singular values $\sigma_{3..10} \sim 10^{-3}$. Taking $\sqrt{\sigma_j} \sim 0.03$ on both factors places the inactive balanced slots at the finite-$\beta$ transition scale $\sim (\beta \mathcal{H}_{\mathrm{med}})^{-1/4}$, where MALA has a larger finite-$\beta$ dip.
  - `dW_err` is essentially matched at $t_1$ (`5.0e-3`) and $t_2$ (`3.9e-3`), so this offset cancels in $\Delta\hat\lambda_{\mathrm{func}} = -0.51 \pm 0.69$.

---

## 4. Part 4: Multi-Head $V{+}O$ vs. $Q{+}K$ Coupling in the 2-Layer Transformer (`[empirical]`)

- **Setup:**
  - Joint rank-$H = 4$ LoRA adapters on Layer 1's causal multi-head self-attention ($V{+}O$ and $Q{+}K$).
  - Three pretrained 2-layer Pre-RMSNorm Transformers with $n_{\mathrm{heads}} \in \{1, 2, 4\}$ ($d_{\mathrm{model}} = 12, d_h \in \{12, 6, 3\}$).
  - Exact next-token KL through causal softmax, Layer-1 RMSNorm + GELU MLP, final RMSNorm and LM head.
  - $\gamma = 60.0, s_{\mathrm{in}} = 4.5, \beta_{VO} = 10^8, \beta_{QK} = 8\times 10^8$, 20 chains, 1,100 burn-in + 1,500 sample steps.
- **Single-head theory (`[LC]`, `AttentionLoRA.lean`, $d=12, H=4$):**
  - $V{+}O$: $\lambda_{V+O}(q) = 40 + 2(q - 2)_+^2$ ($40 \to 48$, jump $+8.0$).
  - $Q{+}K$: $\lambda_{Q+K}(q) \equiv 40.0$, flat across all $q \in \{0, 2, 4\}$ (exact $16$-dim $GL(4)$ gauge cancellation at $q=4$: Hessian rank $80$ vs. $96$).
- **Multi-head theory (`[paper]`, $n_{\mathrm{heads}} \in \{2, 4\}$, $d_h + q \le 12$):**
  - Distinct per-head attention patterns decouple the pair into the additive single-projection regime $2 \times (22 + \frac{1}{2}(q - 2)_+^2) = 44 + (q - 2)_+^2$ ($44 \to 48$, jump $+4.0$).
  - Hessian rank $96$ for both $V{+}O$ and $Q{+}K$.

| $n_{\mathrm{heads}}$ ($d_h$) | $q$ | Theory $V{+}O$ | $V{+}O$ MALA $\hat\lambda \pm \mathrm{SE}$ | $V{+}O$ `#H` (`g_lam`) | $V{+}O$ Acc / `hd` | Theory $Q{+}K$ | $Q{+}K$ MALA $\hat\lambda \pm \mathrm{SE}$ | $Q{+}K$ `#H` (`g_lam`) | $Q{+}K$ Acc / `hd` |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| 1-Head (`12`) | 0 | 40.0 | 34.69 ± 0.36 | 0 (0.0) | 0.37 / +0.94 | 40.0 | 27.09 ± 0.38 | 0 (0.0) | 0.46 / +1.68 |
| 1-Head (`12`) | 2 | 40.0 | 36.77 ± 0.38 | 48 (24.0) | 0.39 / +0.13 | 40.0 | 31.04 ± 0.52 | 44 (21.9) | 0.47 / +0.29 |
| 1-Head (`12`) | 4 | 48.0 | 45.10 ± 0.40 | 96 (48.0) | 0.47 / -0.30 | 40.0 | 38.48 ± 0.37 | 80 (39.8) | 0.43 / +0.34 |
| 2-Head (`6`) | 0 | 44.0 | 40.43 ± 0.43 | 0 (0.0) | 0.43 / +1.08 | 44.0 | 34.17 ± 0.37 | 0 (0.0) | 0.46 / +2.66 |
| 2-Head (`6`) | 2 | 44.0 | 42.35 ± 0.42 | 48 (24.0) | 0.42 / +1.23 | 44.0 | 39.64 ± 0.60 | 48 (24.0) | 0.46 / -0.01 |
| 2-Head (`6`) | 4 | 48.0 | 47.98 ± 0.12 | 96 (48.0) | 0.57 / -0.03 | 48.0 | 47.86 ± 0.31 | 96 (48.0) | 0.55 / -0.02 |
| 4-Head (`3`) | 0 | 44.0 | 39.69 ± 0.36 | 0 (0.0) | 0.44 / +0.87 | 44.0 | 38.83 ± 0.40 | 0 (0.0) | 0.40 / +1.74 |
| 4-Head (`3`) | 2 | 44.0 | 42.16 ± 0.62 | 48 (24.0) | 0.43 / +0.48 | 44.0 | 41.30 ± 0.57 | 48 (24.0) | 0.40 / +0.45 |
| 4-Head (`3`) | 4 | 48.0 | 47.95 ± 0.18 | 96 (48.0) | 0.58 / -0.46 | 48.0 | 48.47 ± 0.28 | 96 (48.0) | 0.57 / -0.81 |

Findings:

1. **Single-head $GL(4)$ gauge cancellation, checked by Hessian and MALA.**
   - At $n_{\mathrm{heads}} = 1, q = 4$, the finite-difference next-token KL Hessian $\beta\nabla^2 K$ has 96 eigenvalues above $\gamma = 60$ (`g_lam = 48.0`) on $V{+}O$ and 80 (`g_lam = 39.8`) on $Q{+}K$.
   - The $16$ missing eigenvalues match the $16$-dimensional $GL(4)$ gauge nullspace, through causal softmax, RMSNorm, GELU MLP and LM head.
   - MALA shows the same $6.6$-point gap ($45.10 \pm 0.40$ vs. $38.48 \pm 0.37$). $V{+}O$'s $-2.9$ residual below $48.0$ comes from mild finite-$\gamma$ Hessian rotation on the less-conditioned 1-head $W_V$.
2. **Multi-head decoupling at $n_{\mathrm{heads}} = 2$ and $4$.**
   - The $q = 0$ plateau rises for both channels ($V{+}O$: $34.69 \to 40.43 / 39.69$; $Q{+}K$: $27.09 \to 34.17 / 38.83$).
   - Both $q = 4$ tops match the decoupled value $48.0$ (`#H = 96` for both): $47.98 \pm 0.12$ / $47.86 \pm 0.31$ (2-head) and $47.95 \pm 0.18$ / $48.47 \pm 0.28$ (4-head).
   - Split-half drift $|\mathrm{hdiff}| \le 0.81$ throughout.
3. **Caveat (finite $\beta$, $Q{+}K$ at $q = 0$).**
   - $Q{+}K$ enters through softmax, whose Fisher spectrum is expected to be much softer than $V{+}O$'s. The eigenvalue ratio is not recorded in the saved JSON.
   - This gives a larger negative finite-$\beta$ logarithmic offset at $q=0$: `27.09` at 1-head, `34.17` at 2-head, `38.83` at 4-head, where the 4-head $Q{+}K$ Fisher is best conditioned.
   - So the main asymptotic test of the single-head vs. multi-head dichotomy is the $q = 4$ regular/gauge-regular top (`45.10 vs. 38.48` at 1-head; `47.98 vs. 47.86` at 2-head; `47.95 vs. 48.47` at 4-head), together with the above-$\gamma$ Hessian eigenvalue counts (`96 vs. 80` at 1-head; `96 vs. 96` at 2- and 4-head).

---

## 5. Reproducibility and Artifacts

- **Code:** [`experiments/transformer_fiber_surgery.py`](../experiments/transformer_fiber_surgery.py).
  - Self-contained PyTorch/NumPy/Matplotlib, fixed seeds, float64 MALA with exact detailed-balance full-matrix linear preconditioning.
  - End-to-end runtime `736.2s` on an Apple Silicon CPU; `--replot` redraws the figure from the saved JSON.
- **Raw data:** [`experiments/results/transformer_fiber_surgery.json`](../experiments/results/transformer_fiber_surgery.json).
- **Multi-panel summary figure:** [`experiments/results/transformer_fiber_surgery.png`](../experiments/results/transformer_fiber_surgery.png).
