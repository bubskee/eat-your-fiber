# Track 2 Flagship Empirical Bridge: Exact LoRA Fiber Surgery, Continual Learning, Optimizer Phase-Diagram Trajectories & Multi-Head Coupling on a Real 2-Layer Causal Transformer

> **Epistemic status:**
> - **Complete unconditional 2D $(r^*, p, q)$ RRR zero-loss fiber stratification (`rrr_fiber_exact_pair`, `rrr_fiber_subcritical_plateau`, `rrr_fiber_A_supercritical_parabola`, `rrr_fiber_B_supercritical_parabola`, `rrr_supercritical_mutually_exclusive`, `rrr_fiber_eq_min_component_codim`):** `[LC]` in [`lean/EatYourFiber/RRRFiberStratification.lean`](../lean/EatYourFiber/RRRFiberStratification.lean) (axiom audit: each depends only on axioms $\subseteq$ `{propext, Classical.choice, Quot.sound}`).
> - **Single-head Self-Attention all-orders $V{+}O$ ($4\times$ supercritical parabola) vs. $Q{+}K$ (exact $GL(q)$ gauge symmetry and flat cancellation $\lambda_{Q+K}(q) \equiv dH - H^2/2$) dichotomy (`attention_VO_exact_pair`, `attention_QK_flat_pair`, `attention_QK_general_flat_pair`, `attention_coupling_dichotomy_standard_init`):** `[LC]` in [`lean/EatYourFiber/AttentionLoRA.lean`](../lean/EatYourFiber/AttentionLoRA.lean). These certify the RRR / linearized-attention losses; transferring their $\lambda$ values to the Transformer's next-token KL relies on the `[paper]` pullback-invariance argument below.
> - **Pullback bilipschitz invariance of local learning coefficients through nonlinear Transformer layers (RMSNorm, causal softmax, GELU MLP, LM head, next-token KL divergence) whenever $\mathcal{H}_{\mathrm{KL}} \succ 0$; Pre-LayerNorm exact $\mathbf{1}_d$-kernel output-dimension drop ($m_{\mathrm{eff}} = d - 1$); multi-head shared-LoRA transversality decoupling ($d_h + q \le d$); and fiber-tangent plasticity operator $\Delta(\Delta W) = -\eta(G A^\top A + B B^\top G)$:** `[paper]` local differential / Gauss–Newton analysis (not formalized in Lean).
> - **All real 2-layer causal Transformer pretraining, 25-point 2D $(p, q)$ exact iso-functional fiber surgery, Pre-LayerNorm symmetry-breaking check, 5-seed continual-learning / noisy-overfitting / directional-plasticity experiments (SGD & Adam), 5-optimizer 2D phase-diagram trajectories, late-stage $\Delta\hat\lambda$ functional-vs.-fiber decompositions, and $n_{\mathrm{heads}} \in \{1, 2, 4\}$ attention coupling sweeps:** `[empirical]` in [`experiments/transformer_fiber_surgery.py`](../experiments/transformer_fiber_surgery.py) (raw data: [`experiments/results/transformer_fiber_surgery.json`](../experiments/results/transformer_fiber_surgery.json), 6-panel figure: [`experiments/results/transformer_fiber_surgery.png`](../experiments/results/transformer_fiber_surgery.png)).
> - **Extrapolation of continual-learning plasticity–stability splits and late-stage DevInterp LLC fiber-sliding decompositions to billion-parameter Transformers under stochastic mini-batch schedules:** `[conjectural]`.

---

## Executive Summary

This study bridges our Lean-certified Track 2 fiber geometry (`RRRFiberStratification.lean`, `AttentionLoRA.lean`) onto a **genuine pretrained 2-layer causal Transformer language model** (token + positional embeddings, Pre-RMSNorm [plus a Pre-LayerNorm GPT-2 twin], causal multi-head self-attention, residual stream, $2\times$ GELU MLP, LM head, trained with real next-token cross-entropy loss to `100.0%` accuracy on a 3-pair associative-recall + modular-composition sequence task). Evaluating the next-token functional KL divergence $K(\theta) = \mathbb{E}_x[\mathrm{KL}(p_{\theta^*}(\cdot|x) \| p_\theta(\cdot|x))]$ (averaged over 32 fixed evaluation sequences) with fixed full-matrix preconditioned float64 MALA (preserving exact detailed balance while eliminating $\sim 10^4\times$ Transformer Fisher anisotropy), we establish five main empirical findings:

1. **The full 2D $(p, q)$ phase diagram reappears inside a real 2-layer causal Transformer (`[empirical]`):** Across **25 exact iso-functional surgery checkpoints** on Layer 1 $W_O$ ($M=N=12, H=10, r^*=2$, excess $(m,n,k)=(10,10,8)$, $p_c = q_c = 4$, **every checkpoint verified $\max|\Delta\mathrm{logit}| = 1.5\times 10^{-14}$** on held-out sequences), MALA sits on or below the flat 2D subcritical plateau ($\lambda = 54.0$; points run $0.4$–$4.9$ low at finite $\beta$, the full-norm balanced diagonal $(4,4) \to 53.11 \pm 0.57$), the $A$-supercritical parabola $+\frac{1}{2}(q-4)^2$ up to $(0,8) \to \mathbf{61.65 \pm 0.50}$ (theory $62.0$), the symmetric $B$-supercritical parabola $+\frac{1}{2}(p-4)^2$ up to $(8,0) \to \mathbf{61.98 \pm 0.40}$ (theory $62.0$), and off-axis mutual exclusivity at $(1,6), (2,6), (6,1), (6,2) \to 55.09\text{--}56.14$ (theory $56.0$).
2. **Architectural discovery — Pre-LayerNorm breaks $p \leftrightarrow q$ fiber symmetry via an exact $\mathbf{1}_d$ Fisher kernel (`[paper]` + `[empirical]`):** Because both Layer-1 $\mathrm{LN}_2$ and final $\mathrm{LN}_f$ subtract the mean vector $\mathbf{1}_d$, adding any rank-1 perturbation $\mathbf{1}_d u^\top$ to residual-stream writes ($W_O$ or $W_{\mathrm{down}}$) changes output logits by identically $0$. The exact $144 \times 144$ Gauss–Newton Fisher matrix of $W_O$ has **$0$ zero eigenvalues under Pre-RMSNorm** ($\lambda_{\min} = 7.33\times 10^{-5}$) vs. **exactly $12$ zero eigenvalues under Pre-LayerNorm** ($\lambda_{\min} = -2.55\times 10^{-17}$, $\lambda_{13} = 2.79\times 10^{-5}$). This reduces $m_{\mathrm{eff}} = d - 1 = 11$, shifting $(p_c, q_c)$ from $(4, 4) \to (3.5, 4.5)$ and breaking symmetry by $4.0$ points: $\lambda(0,8) = 57.0$ vs. $\lambda(8,0) = 61.0$ (measured: **$56.38 \pm 0.72$ vs. $60.07 \pm 0.63$**, exact Hessian `gauss_lam = 57.0 vs. 61.0`).
3. **Function-matched Task-A checkpoints exhibit a sharp plasticity–stability, noisy-overfitting, and directional-plasticity split on Task B — with an honest SGD vs. Adam contrast (`[empirical]`):** Starting from 5 Task-A checkpoints (four exact surgery points at $\max|\Delta\mathrm{logit}| = 1.7\times 10^{-14}$; the realistic near-plateau `(0,0)+1e-3` differs by $6.2\times 10^{-4}$, initial $\mathrm{KL}_A = 1.3\times 10^{-10}$), 5-seed fine-tuning on a shifted real-token associative-recall task (`shift_rule = 3`, real cross-entropy) reveals:
   - **Under SGD (clean Task B):** `Exact (0,0)` and `Near-Plateau (0,0)+1e-3` remain locked near rank 2 ($\mathrm{erank}(\Delta W) = 2.18\text{--}2.41$, recall accuracy $0.211\text{--}0.253$, low Task-A forgetting $\mathrm{KL}_A = 0.385\text{--}0.397$), whereas `Balanced (4,4)` ($0.384$), `A-Supercrit (0,8)` ($0.448$), and `B-Supercrit (8,0)` ($0.473$, **$2.2\times$ higher Task-B accuracy**) adapt rapidly at the cost of higher Task-A forgetting ($\mathrm{KL}_A = 0.450\text{--}0.464$).
   - **Under SGD & Adam (noisy limited-sample Task B, $S=32$, 25% corruption):** `Exact (0,0)` exhibits a **$2.6\times$–$5.3\times$ smaller overfitting gap** ($+0.126$ SGD, $+0.081$ Adam) and lower test CE ($1.872$ / $1.851$) than supercritical `(0,8)` / `(8,0)` (gaps $+0.333\text{--}0.433$, test CE $1.964\text{--}1.999$).
   - **Honest SGD vs. Adam nuance:** While `Exact (0,0)` locks **both** SGD and Adam to rank 2, adding realistic `1e-3` residual norm (`Near-Plateau (0,0)+1e-3`) keeps **SGD** trapped ($0.253$ acc, $+0.179$ noisy gap) whereas **Adam's $1/\sqrt{v}$ normalization immediately breaks the saddle** ($0.595$ acc, $+0.371$ noisy gap, matching `(4,4)`, `(0,8)`, `(8,0)`).
   - **Mirror-image directional plasticity & the `(4,4)` sweet spot:** On orthogonal $V_\perp$-input vs. $U_\perp$-output shifts, `(0,8)` and `(8,0)` flip verbatim under both SGD ($0.785$ vs. $0.489$ on $V_\perp$; $0.718$ vs. $0.819$ on $U_\perp$) and Adam ($0.937$ vs. $0.848$; $0.915$ vs. $0.960$), while **Balanced `(4,4)` achieves the best of both worlds: low subcritical LLC ($\lambda = 54.0$) AND fast symmetric plasticity on both axes ($0.658 / 0.752$ SGD; $0.941 / 0.957$ Adam)**.
4. **Real optimizer trajectories map cleanly onto the 2D $(p_{\mathrm{eff}}, q_{\mathrm{eff}})$ phase diagram, and the late-stage AdamW LLC drop is $\approx 95\%$ ($\pm 6$ pts) pure iso-functional fiber sliding (`[empirical]`):**
   - `LoRA + Adam (wd=0)` freezes in the $A$-supercritical region at $(p_{\mathrm{eff}}, q_{\mathrm{eff}}) = (0.24, 5.57)$, $\hat\lambda = 58.44 \pm 0.53$; **LoRA+ ($\eta_B = 12\,\eta_A$, Hayou et al. 2024) stays even higher on the $A$-supercritical axis** at $(0.00, 6.73)$, $\hat\lambda = 61.47 \pm 0.46$ (shrinking $\eta_A$ freezes $A_{\mathrm{in}}$ more tightly at its full-rank init!).
   - `LoRA + AdamW`, `LoRA + SGD+WD`, and `Balanced Small Init` all land inside the 2D subcritical plateau ($\hat\lambda = 45.72\text{--}47.62$ vs. exact $(0,0)$ ref $49.96 \pm 0.49$).
   - Decomposing late-stage $\Delta\hat\lambda$ via exact balanced-SVD surgery ($\max|\Delta\mathrm{logit}| \le 5.6\times 10^{-14}$): after Task-A KL has already converged (`1.53e-8 -> 3.48e-8`), `LoRA + AdamW (2500 -> 8000)` drops by $\Delta\hat\lambda_{\mathrm{total}} = \mathbf{-10.79 \pm 0.83}$, decomposing into $\Delta\hat\lambda_{\mathrm{func}} = \mathbf{-0.51 \pm 0.69}$ (zero within $1\sigma$) plus $\Delta\hat\lambda_{\mathrm{fiber}} = \mathbf{-10.28 \pm 1.07}$ (**fiber share $95.3\% \pm 6.4$ pts**, i.e. $1 - \Delta\hat\lambda_{\mathrm{func}}/\Delta\hat\lambda_{\mathrm{total}}$ with a delta-method 1-SE error from the independent raw and balanced MALA runs; consistent with 100%). By contrast, `LoRA + Adam (ctrl)` is flat ($+0.85 \pm 0.76$), and coupled `LoRA + SGD+WD (1500 -> 9000)` starting from a still-overparameterized momentum-SGD overshoot (`dW_err = 2.3e-2`) shows a large functional component ($\Delta\hat\lambda_{\mathrm{func}} = -14.83 \pm 0.87$, `dW_err` $2.3\times 10^{-2} \to 4.5\times 10^{-3}$) alongside fiber collapse.
5. **The single-head $V{+}O$ vs. $Q{+}K$ coupling dichotomy and multi-head reversal survive at the $q = 4$ tops inside the full pretrained 2-layer causal Transformer (`[empirical]`; 1-head $Q{+}K$ is not flat at finite $\beta$, `27.09 → 38.48`):** Across $n_{\mathrm{heads}} \in \{1, 2, 4\}$ ($d=12, H=4$, through causal softmax, RMSNorm, GELU MLP, LM head, and next-token KL):
   - At **$n_{\mathrm{heads}} = 1$**: $V{+}O$ jumps sharply to $\hat\lambda(4) = \mathbf{45.10 \pm 0.40}$ (`#H = 96` Hessian eigenvalues above $\gamma$, `g_lam = 48.0`), while $Q{+}K$ is capped by the $16$-dimensional $GL(4)$ gauge symmetry at $\hat\lambda(4) = \mathbf{38.48 \pm 0.37}$ (`#H = 80`, `g_lam = 39.8`).
   - At **$n_{\mathrm{heads}} = 2$ and $4$**: cross-head transversality ($d_h + q \le 12$) breaks $GL(4)$ completely — `#H = 96` for **both** channels, and MALA lands cleanly on $48.0$ for **both** $V{+}O$ ($47.98 \pm 0.12$ at 2-head, $47.95 \pm 0.18$ at 4-head) and $Q{+}K$ ($47.86 \pm 0.31$ at 2-head, $48.47 \pm 0.28$ at 4-head).

---

## 0. Real 2-Layer Causal Transformer & Pretraining Setup (`[empirical]`)

- **Architecture (`CausalTransformerLM`):** Vocabulary $V = 24$, context length $T = 12$, $d_{\mathrm{model}} = 12$, $d_{\mathrm{mlp}} = 24$ ($2\times$ GELU MLP), 2 causal Pre-RMSNorm Transformer blocks (plus a Pre-LayerNorm twin for Part 1's architectural comparison), final normalization, and untied LM head (`3,180` parameters for Pre-RMSNorm; `3,240` for Pre-LayerNorm; all model parameters, LoRA factors excluded).
- **Pretraining task:** 3-pair structured **Associative Recall + Shifted Modular Composition** (`[k0, v0, k1, v1, k2, v2, mod, q1, r1, c1, q2, r2]`): 3 distinct keys (`0..7`), 3 values (`8..15`), modifier (`16..23`), two distinct queries (`q1, q2`), exact recalled values `r1 = val(q1)`, `r2 = val(q2)`, and modular sum `c1 = 8 + ((r1 - 8 + mod - 16) % 8)`. Trained with online batches (`bs = 128`, 3,000 steps, AdamW `lr = 1.2e-2`, cosine schedule, `wd = 2e-3`, real next-token cross-entropy across all 11 positions):

| Pretrained Model | Norm | $n_{\mathrm{heads}}$ | $d_h$ | Val CE | Recall `r1` Acc | Comp `c1` Acc | Recall `r2` Acc | Time |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Flagship base (Parts 1–4) | Pre-RMSNorm | 2 | 6 | `1.2873` | **1.000** | **1.000** | **1.000** | 10.9s |
| Pre-LN twin (Part 1 bonus) | Pre-LayerNorm | 2 | 6 | `1.5484` | 0.461 | 0.643 | 0.520 | 10.9s |
| 1-Head twin (Part 4) | Pre-RMSNorm | 1 | 12 | `1.4535` | 0.465 | **1.000** | 0.494 | 7.8s |
| 4-Head twin (Part 4) | Pre-RMSNorm | 4 | 3 | `1.4396` | 0.451 | **1.000** | 0.551 | 12.8s |

- **MALA Methodology (Exact Full-Matrix Linear Preconditioning):** On a real Transformer, the next-token KL Gauss–Newton Fisher matrix $\mathcal{H}_{\mathrm{KL}}$ of Layer 1 $W_O$ has condition number $\sim 5.8\times 10^3$ ($\lambda_{\min} = 7.33\times 10^{-5}$, $\lambda_{\max} = 4.25\times 10^{-1}$) rotated into the model's learned eigenbasis; diagonal preconditioning alone mixes slowly along soft off-diagonal eigenmodes. We therefore compute a **fixed** full-matrix linear preconditioner $w = w_0 + P z$ once before burn-in from the exact local Hessian $H_0 = \nabla^2(\beta K)(w_0)$ (plus quartic-scale probes in the soft/null eigenspace), and run float64 MALA in $z$-space. Because $P$ is a constant invertible linear map independent of the Markov chain state, MALA satisfies **exact detailed balance** for $\pi(w) \propto \exp(-\beta K(w) - \frac{\gamma}{2}\|w - w_0\|_2^2)$. Caching frozen Layer-0/pre-$W_O$ activations matches the full 2-layer causal Transformer forward pass to $\max|\Delta\mathrm{logit}| = 1.07\times 10^{-14}$.

---

## 1. Part 1 — Exact Iso-Functional 2D $(p, q)$ Fiber Surgery (`[empirical]`)

### 1.1 Setup & Function-Identity Verification
We freeze the pretrained 2-layer Pre-RMSNorm Transformer (`h=2`) and attach a rank-$H = 10$ LoRA adapter $(B, A)$ to Layer 1's $12 \times 12$ attention output projection $W_O$. Task A is next-token KL distillation toward a rank-$r^* = 2$ adaptation $\Delta W_A^* = U_A[:, :2]\operatorname{diag}(0.8, 0.6) V_A[:, :2]^\top$ ($\mathrm{KL}(\text{Task A} \| \text{base}) = 0.4134\text{ nats}$; converged LoRA reaches $\mathrm{KL}_A = 3.18\times 10^{-7}$).

Excess dimensions are $(m, n, k) = (10, 10, 8)$ ($m+n+k = 28$ even, $m_{\log} = 1$), giving critical ranks $p_c = q_c = 4$, active $\lambda_{\mathrm{act}} = 22.0$, excess plateau $\lambda_{\mathrm{AW}}(10,10,8) = 32.0$, total plateau $\lambda(p, q) = 54.0$ for $p \le 4, q \le 4$, and symmetric supercritical parabolas $\lambda = 54.0 + \frac{1}{2}(q - 4)_+^2$ ($q > 4$) and $54.0 + \frac{1}{2}(p - 4)_+^2$ ($p > 4$) peaking at $62.0$ (`RRRFiberStratification.lean`, `[LC]` for the RRR loss; using these values for the Transformer KL is the `[paper]` pullback step).

Constructing canonical iso-functional checkpoints $(B', A')$ with inactive singular scale $s_{\mathrm{in}} = 2.0$ across 25 points of the 2D $(p, q)$ phase diagram preserves $B'A' = \Delta W_A^*$ to machine precision and yields **$\max|\Delta\mathrm{logit}| = 1.5\times 10^{-14}$** across all 64 held-out sequences at every single checkpoint.

### 1.2 Full 25-Point 2D $(p, q)$ MALA Results ($\beta = 10^9, \gamma = 5.0$, 20 chains, 1,100 burn-in + 1,500 sample steps)

| Slice | $(p, q)$ | Lean `[LC]` $\lambda$ | MALA $\hat\lambda \pm \mathrm{SE}$ | Error | Acc | Split-half `hdiff` | Local `gauss_lam` | $\max|\Delta\mathrm{logit}|$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| $A$-axis | `(0, 0)` | 54.0 | 49.08 ± 0.45 | -4.92 | 0.39 | +1.18 | 22.0 | `1.5e-14` |
| $A$-axis | `(0, 1)` | 54.0 | 49.14 ± 0.55 | -4.86 | 0.39 | -0.42 | 27.0 | `1.5e-14` |
| $A$-axis | `(0, 2)` | 54.0 | 50.45 ± 0.65 | -3.55 | 0.38 | -0.16 | 32.0 | `1.5e-14` |
| $A$-axis | `(0, 3)` | 54.0 | 51.86 ± 0.61 | -2.14 | 0.38 | +1.47 | 37.0 | `1.5e-14` |
| $A$-axis | `(0, 4)` | 54.0 | 52.24 ± 0.46 | -1.76 | 0.38 | -0.28 | 42.0 | `1.5e-14` |
| $A$-axis | `(0, 5)` | 54.5 | 53.75 ± 0.48 | -0.75 | 0.38 | +1.64 | 47.0 | `1.5e-14` |
| $A$-axis | `(0, 6)` | 56.0 | **56.10 ± 0.55** | +0.10 | 0.40 | -0.26 | 52.0 | `1.5e-14` |
| $A$-axis | `(0, 7)` | 58.5 | **58.15 ± 0.25** | -0.35 | 0.37 | -0.34 | 57.0 | `1.5e-14` |
| $A$-axis | `(0, 8)` | 62.0 | **61.65 ± 0.50** | -0.35 | 0.35 | +1.22 | 62.0 | `1.5e-14` |
| $B$-axis | `(1, 0)` | 54.0 | 50.79 ± 0.44 | -3.21 | 0.37 | +1.25 | 27.0 | `1.5e-14` |
| $B$-axis | `(2, 0)` | 54.0 | 50.52 ± 0.48 | -3.48 | 0.41 | -0.26 | 32.0 | `1.5e-14` |
| $B$-axis | `(3, 0)` | 54.0 | 52.39 ± 0.34 | -1.61 | 0.39 | -1.11 | 37.0 | `1.5e-14` |
| $B$-axis | `(4, 0)` | 54.0 | 53.42 ± 0.50 | -0.58 | 0.42 | +1.85 | 42.0 | `1.5e-14` |
| $B$-axis | `(5, 0)` | 54.5 | **54.70 ± 0.44** | +0.20 | 0.39 | -0.20 | 47.0 | `1.5e-14` |
| $B$-axis | `(6, 0)` | 56.0 | **55.61 ± 0.36** | -0.39 | 0.40 | -0.14 | 52.0 | `1.5e-14` |
| $B$-axis | `(7, 0)` | 58.5 | **58.56 ± 0.42** | +0.06 | 0.40 | -0.02 | 57.0 | `1.5e-14` |
| $B$-axis | `(8, 0)` | 62.0 | **61.98 ± 0.40** | -0.02 | 0.39 | -0.10 | 62.0 | `1.5e-14` |
| Balanced diag | `(1, 1)` | 54.0 | 51.10 ± 0.40 | -2.90 | 0.38 | +1.16 | 31.5 | `1.5e-14` |
| Balanced diag | `(2, 2)` | 54.0 | 52.20 ± 0.41 | -1.80 | 0.35 | +0.91 | 40.0 | `1.5e-14` |
| Balanced diag | `(3, 3)` | 54.0 | **53.57 ± 0.36** | -0.43 | 0.40 | -0.88 | 47.5 | `1.5e-14` |
| Balanced diag | `(4, 4)` | 54.0 | **53.11 ± 0.57** | -0.89 | 0.37 | -0.83 | 54.0 | `1.5e-14` |
| Off-axis | `(1, 6)` | 56.0 | **55.09 ± 0.28** | -0.91 | 0.38 | +0.06 | 54.0 | `1.5e-14` |
| Off-axis | `(2, 6)` | 56.0 | **55.48 ± 0.47** | -0.52 | 0.39 | -0.17 | 56.0 | `1.5e-14` |
| Off-axis | `(6, 1)` | 56.0 | **55.64 ± 0.27** | -0.36 | 0.43 | +0.22 | 54.0 | `1.5e-14` |
| Off-axis | `(6, 2)` | 56.0 | **56.14 ± 0.32** | +0.14 | 0.39 | +0.35 | 56.0 | `1.5e-14` |

Every feature of `RRRFiberStratification.lean` holds quantitatively on the real 2-layer causal Transformer: the expected negative finite-$\beta$ logarithmic offset on the deeply singular subcritical plateau ($-0.4$ to $-4.9$; largest, $-2.9$ to $-4.9$, at the six most singular points where local `gauss_lam` is only $22\text{--}32$ and the remaining $22\text{--}32$ points come purely from quartic singular integrals), agreement along both supercritical parabolas (within $\le 0.39$ at all 6 on-axis supercritical points $r \in \{6, 7, 8\}$), flat plateau behavior all the way to the full-norm balanced corner `(4, 4)` (`gauss_lam = 54.0`, MALA `53.11 ± 0.57`), and subcritical-coordinate independence on all four off-axis points `(1,6), (2,6), (6,1), (6,2)`.

### 1.3 Architectural Discovery: Pre-LayerNorm Exact $\mathbf{1}_d$-Kernel & $p \leftrightarrow q$ Symmetry Breaking (`[paper]` + `[empirical]`)
In a **Pre-LayerNorm** Transformer, both $\mathrm{LN}_2$ and $\mathrm{LN}_f$ project out the constant vector $\mathbf{1}_d$:
$$\mathrm{LN}_2(x_1 + c\,\mathbf{1}_d) \equiv \mathrm{LN}_2(x_1) \implies \mathrm{LN}_f\!\bigl(x_1 + c\,\mathbf{1}_d + \mathrm{MLP}(\mathrm{LN}_2(x_1 + c\,\mathbf{1}_d))\bigr) \equiv \mathrm{LN}_f(x_{\mathrm{final}}).$$
Consequently, the entire $d$-dimensional subspace $\{\mathbf{1}_d u^\top : u \in \mathbb{R}^d\}$ lies identically in $\ker(\mathcal{H}_{\mathrm{KL}})$ of Layer 1 $W_O$, confirmed numerically by exact Hessian computation:
- **Pre-RMSNorm `Fisher(W_O)`:** `0` zero eigenvalues ($\lambda_{\min} = 7.33\times 10^{-5}$, $\lambda_{\mathrm{med}} = 4.69\times 10^{-3}$, $\lambda_{\max} = 4.25\times 10^{-1}$).
- **Pre-LayerNorm `Fisher(W_O)`:** **exactly `12` zero eigenvalues** ($\lambda_{\min} = -2.55\times 10^{-17}$, $\lambda_{13} = 2.79\times 10^{-5}$, $\lambda_{\max} = 5.31\times 10^{-1}$).

This drops effective output dimension to $M_{\mathrm{eff}} = 11$ ($m = 9, n = 10, k = 8$; $m+n+k = 27$ odd $\implies m_{\log} = 2$ on the plateau), shifting $p_c = 3.5 \ne q_c = 4.5$ and creating a **$4.0$-point asymmetry** between $(0, 8)$ and $(8, 0)$:

| Pre-LayerNorm $(p, q)$ | Theory ($M_{\mathrm{eff}}=11, N=12$) | $m_{\log}$ | Local `gauss_lam` | MALA $\hat\lambda \pm \mathrm{SE}$ | Acc | `hdiff` |
|---:|---:|---:|---:|---:|---:|---:|
| `(0, 0)` | **51.0** | 2 | 21.0 | **47.28 ± 0.42** | 0.39 | +0.64 |
| `(0, 8)` | **57.0** | 1 | **57.0** | **56.38 ± 0.72** | 0.40 | -0.96 |
| `(8, 0)` | **61.0** | 1 | **61.0** | **60.07 ± 0.63** | 0.39 | +0.03 |

Thus Pre-LayerNorm lowers the standard LoRA ($A$-supercritical) top by $5.0$ points ($62 \to 57$) while lowering the $B$-supercritical top by only $1.0$ point ($62 \to 61$)—a direct architectural consequence of LayerNorm's shift invariance acting like a structural dead neuron on residual-stream writes.

---

## 2. Part 2 — Continual Learning / Plasticity–Stability Split from Function-Matched Checkpoints (`[empirical]`)

### 2.1 Mechanism: The Fiber Tangent Operator (`[paper]`)
At $t = 0$ of Task B fine-tuning, all iso-functional checkpoints share the exact same effective weight $\Delta W_0 = B' A' = \Delta W_A^*$, hence the identical Task-B loss $L_B(\Delta W_0)$ and identical weight-space gradient $G_0 = \nabla_{\Delta W} L_B(\Delta W_0)$. Yet their first-order effective-weight update under GD/SGD depends explicitly on their fiber coordinates:
$$\Delta(\Delta W) = -\eta\Bigl(G_0\,(A')^\top A' + B' (B')^\top G_0\Bigr) + O(\eta^2).$$
At `(0, 0)`, $B'_{\mathrm{in}} = A'_{\mathrm{in}} = 0$ leaves all $k = 8$ inactive slots on an exact invariant singular subspace ($\nabla_{B_{\mathrm{in}}} = \nabla_{A_{\mathrm{in}}} = 0$). At `(0, 8)`, $(A'_{\mathrm{in}})^\top A'_{\mathrm{in}}$ gives immediate rank-8 plasticity on the 10-dim $V_\perp$; at `(8, 0)`, $B'_{\mathrm{in}}(B'_{\mathrm{in}})^\top$ gives immediate rank-8 plasticity on the 10-dim $U_\perp$; at `(4, 4)` (inside the flat 2D plateau, same low $\lambda = 54.0$ as `(0, 0)`!), **both** $B'_{\mathrm{in}}$ and $A'_{\mathrm{in}}$ carry rank-4 live plasticity.

We compare 5 checkpoints (all verified $\max|\Delta\mathrm{logit}| = 1.7\times 10^{-14}$ on held-out Task-A sequences, except realistic `Near-Plateau (0,0)+1e-3` with $\max|\Delta\mathrm{logit}| = 6.2\times 10^{-4}$), across **5 random seeds** (independently rotating the inactive orthogonal complements $U_\perp, V_\perp$ and resampling Task-B datasets) for **both SGD and Adam**.

### 2.2 Regime 2a: Clean Real-Token Shifted Recall Task B (`shift_rule = 3`, $S_{\mathrm{train}}=192, S_{\mathrm{test}}=256$, initial `CE_B = 3.5363`, `r1_acc = 0.051`)

| Opt | Checkpoint | Test CE@20 | Final Test CE | Task-B `r1` Acc | Task-A Forgetting $\mathrm{KL}_A$ | Update $\|\Delta W\|_F$ | $\mathrm{erank}(\Delta W)$ |
|---|---|---:|---:|---:|---:|---:|---:|
| **SGD** | `Exact (0,0)` | 1.785 ± 0.014 | 1.699 ± 0.006 | **0.211 ± 0.014** | **0.385 ± 0.004** | 4.03 | **2.18** |
| **SGD** | `Near (0,0)+1e-3` | 1.785 ± 0.014 | 1.690 ± 0.023 | **0.253 ± 0.035** | **0.397 ± 0.021** | 4.77 | **2.41** |
| **SGD** | `Balanced (4,4)` | 1.799 ± 0.011 | 1.686 ± 0.014 | **0.384 ± 0.036** | 0.450 ± 0.011 | 6.90 | 3.41 |
| **SGD** | `A-Supercrit (0,8)` | 1.791 ± 0.014 | **1.646 ± 0.011** | **0.448 ± 0.016** | 0.458 ± 0.024 | 7.09 | 3.47 |
| **SGD** | `B-Supercrit (8,0)` | 1.865 ± 0.091 | **1.638 ± 0.010** | **0.473 ± 0.044** | 0.464 ± 0.015 | 7.14 | 3.35 |
| **Adam** | `Exact (0,0)` | 1.851 ± 0.004 | 1.679 ± 0.004 | **0.228 ± 0.019** | **0.349 ± 0.004** | 3.16 | **2.32** |
| **Adam** | `Near (0,0)+1e-3` | 1.824 ± 0.004 | **1.591 ± 0.004** | **0.595 ± 0.014** | 0.524 ± 0.017 | 6.55 | 3.53 |
| **Adam** | `Balanced (4,4)` | 1.798 ± 0.005 | **1.586 ± 0.002** | **0.595 ± 0.005** | 0.548 ± 0.013 | 6.83 | 3.79 |
| **Adam** | `A-Supercrit (0,8)` | 1.786 ± 0.001 | **1.594 ± 0.006** | **0.609 ± 0.006** | 0.532 ± 0.014 | 6.70 | 3.70 |
| **Adam** | `B-Supercrit (8,0)` | 1.810 ± 0.006 | **1.591 ± 0.004** | **0.608 ± 0.013** | 0.554 ± 0.023 | 6.95 | 3.92 |

### 2.3 Regime 2b: Noisy Limited-Sample Real-Token Task B ($S_{\mathrm{train}}=32$, 25% target corruption, clean $S_{\mathrm{test}}=256$)

| Opt | Checkpoint | Train CE | Clean Test CE | Overfit Gap (Te $-$ Tr) | Task-A $\mathrm{KL}_A$ | $\mathrm{erank}(\Delta W)$ |
|---|---|---:|---:|---:|---:|---:|
| **SGD** | `Exact (0,0)` | 1.746 | **1.872 ± 0.021** | **+0.126 ± 0.015** | **0.520 ± 0.020** | **2.03** |
| **SGD** | `Near (0,0)+1e-3` | 2.143 | 2.323 ± 0.423* | **+0.179 ± 0.026** | 1.023 ± 0.455* | **2.27** |
| **SGD** | `Balanced (4,4)` | 1.600 | 1.992 ± 0.027 | +0.392 ± 0.040 | 0.733 ± 0.032 | 4.54 |
| **SGD** | `A-Supercrit (0,8)` | 1.605 | 1.984 ± 0.032 | +0.379 ± 0.040 | 0.729 ± 0.039 | 5.09 |
| **SGD** | `B-Supercrit (8,0)` | 1.643 | 1.976 ± 0.038 | +0.333 ± 0.062 | 0.711 ± 0.045 | 4.95 |
| **Adam** | `Exact (0,0)` | 1.770 | **1.851 ± 0.012** | **+0.081 ± 0.017** | **0.459 ± 0.012** | **1.87** |
| **Adam** | `Near (0,0)+1e-3` | 1.590 | 1.960 ± 0.016 | +0.371 ± 0.033 | 0.736 ± 0.023 | 4.07 |
| **Adam** | `Balanced (4,4)` | 1.567 | 1.989 ± 0.031 | +0.422 ± 0.048 | 0.763 ± 0.043 | 4.15 |
| **Adam** | `A-Supercrit (0,8)` | 1.559 | 1.964 ± 0.042 | +0.405 ± 0.055 | 0.746 ± 0.046 | 4.07 |
| **Adam** | `B-Supercrit (8,0)` | 1.565 | 1.999 ± 0.021 | +0.433 ± 0.048 | 0.821 ± 0.033 | 4.64 |

*\*Honest note on `Near (0,0)+1e-3` under noisy SGD: 1 of 5 seeds briefly spiked during late SGD steps on the small noisy batch (`lr=0.8`), inflating mean/SE of test CE (`±0.423`) and `KL_A` (`±0.455`) while keeping its generalization gap tightly bounded (`+0.179 ± 0.026`) and rank low (`2.27`).*

### 2.4 Regime 2c: Directional Plasticity Split ($V_\perp$-Input vs. $U_\perp$-Output Orthogonal Task B, 5 seeds)
Relative KL drop at step 25 (`(KL_0 - KL_25) / KL_0`, mean ± SE) and final residual KL fraction (`KL_100 / KL_0`):

| Task-B Alignment | Opt | `Exact (0,0)` | `Near (0,0)+1e-3` | `Balanced (4,4)` | `A-Supercrit (0,8)` | `B-Supercrit (8,0)` |
|---|---|---:|---:|---:|---:|---:|
| **$V_\perp$ (Input-aligned)** | **SGD** | 0.475 ± 0.055 | 0.475 ± 0.055 | **0.658 ± 0.067** | **0.785 ± 0.031** | 0.489 ± 0.070 |
| **$U_\perp$ (Output-aligned)** | **SGD** | 0.626 ± 0.071 | 0.626 ± 0.071 | **0.752 ± 0.083** | 0.718 ± 0.053 | **0.819 ± 0.059** |
| **$V_\perp$ (Input-aligned)** | **Adam** | 0.600 ± 0.048 | 0.926 ± 0.013 | **0.941 ± 0.007** | **0.937 ± 0.003** | 0.848 ± 0.021 |
| **$U_\perp$ (Output-aligned)** | **Adam** | 0.725 ± 0.031 | 0.938 ± 0.008 | **0.957 ± 0.006** | 0.915 ± 0.013 | **0.960 ± 0.002** |

### 2.5 Honest Synthesis of Part 2
1. **Exact singular saddle trapping vs. Adam's $1/\sqrt{v}$ normalization:** At `Exact (0,0)`, the invariant subspace $B_{\mathrm{in}} = A_{\mathrm{in}} = 0$ traps **both** SGD and Adam to rank $\approx 2$ (relative to `(4,4)`, `(0,8)` and `(8,0)`: cutting the noisy overfitting gap by $2.6\times$–$5.3\times$ and clean-Task-B forgetting of Task A by $14$–$37\%$, at the cost of $1.8\times$–$2.7\times$ lower clean Task-B recall accuracy). Crucially, however, at a **realistic near-plateau checkpoint** `(0,0)+1e-3` (representative of finite-step AdamW), **SGD remains strongly trapped** ($0.253$ acc, $+0.179$ gap) **whereas Adam's adaptive normalization immediately breaks the saddle** ($0.595$ acc, $+0.371$ gap, indistinguishable from `(0,8)` on isotropic Task B!). Thus, in practice, **fiber-induced plasticity–stability locking is strong under SGD and exact-zero pruning, but largely washed out on non-directional tasks if continuing with unregularized Adam from a nonzero `1e-3` residual**.
2. **Even under Adam, directional fiber geometry survives — and Balanced `(4,4)` dominates:** On directional shifts (Regime 2c), `A-Supercrit (0,8)` and `B-Supercrit (8,0)` still show a clean mirror-image asymmetry under **both** SGD and Adam, whereas **`Balanced (4,4)` sits inside the low-complexity 2D plateau ($\lambda = 54.0$, zero supercritical penalty) while delivering fast, symmetric adaptation on both input and output shifts ($0.941$ and $0.957$ under Adam; $0.658$ and $0.752$ under SGD)**.

---

## 3. Part 3 — Real Optimizer Trajectories on the 2D $(p, q)$ Phase Diagram & Functional vs. Iso-Functional $\Delta\hat\lambda$ Decomposition (`[empirical]`)

### 3.1 Five Optimizer Trajectories on Task A ($\beta = 5\times 10^8, \gamma = 5.0$; exact references: `(0,0) = 49.96 ± 0.49`, `(0,8) = 61.76 ± 0.37`)

| Optimizer / Init | Final Step | Final $\mathrm{KL}_A$ | $\|\Delta\hat W - \Delta W_A^*\|_F$ | Final $(p_{\mathrm{eff}}, q_{\mathrm{eff}})$ | $\|B_{\mathrm{in}}\|_F$ | $\|A_{\mathrm{in}}\|_F$ | Final $\hat\lambda \pm \mathrm{SE}$ | Acc / `hdiff` | Phase Region |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `LoRA + Adam (wd=0)` | 4,500 | `4.72e-09` | `2.97e-03` | `(0.24, 5.57)` | 0.249 | 5.971 | **58.44 ± 0.53** | 0.43 / -0.75 | $A$-supercritical |
| `LoRA+ (eta_B=12*eta_A)` | 4,500 | `8.16e-08` | `1.21e-02` | `(0.00, 6.73)` | 0.030 | **6.971** | **61.47 ± 0.46** | 0.39 / -0.35 | Deep $A$-supercritical |
| `LoRA + AdamW (wd=0.12)` | 8,000 | `3.48e-08` | `3.95e-03` | `(0.04, 0.19)` | 0.087 | **0.214** | **47.03 ± 0.46** | 0.48 / -0.48 | 2D Subcritical plateau |
| `LoRA + SGD+WD (mom=0.9)` | 9,000 | `6.61e-08` | `4.48e-03` | `(0.01, 0.01)` | 0.050 | **0.051** | **45.72 ± 0.58** | 0.53 / +2.14 | 2D Subcritical plateau |
| `Balanced Small Init (2e-3)` | 4,500 | `8.68e-09` | `2.41e-03` | `(0.40, 0.16)` | 0.351 | **0.189** | **47.62 ± 0.54** | 0.43 / +0.40 | 2D Subcritical plateau |

**Surprising empirical finding on LoRA+ (Hayou et al. 2024):** One might naively expect raising $\eta_B / \eta_A = 12$ to push LoRA+ toward the $B$-supercritical side. Instead, starting from standard LoRA initialization ($B_0 = 0, A_0 \sim \mathcal{N}(0, 0.7^2)$), shrinking $\eta_A = 0.003$ freezes $A_{\mathrm{in}}$ **even more tightly** at its full-rank initialization ($\|A_{\mathrm{in}}\|_F = 6.971$, $q_{\mathrm{eff}} = 6.73$) while fast $B$ updates align rapidly with the active column space $U_{\mathrm{act}}$ ($\|B_{\mathrm{in}}\|_F = 0.030$, $p_{\mathrm{eff}} = 0.00$), leaving **LoRA+ even deeper in the $A$-supercritical cone** ($\hat\lambda = 61.47$ vs. $58.44$ for equal-LR Adam)!

### 3.2 Exact Late-Stage $\Delta\hat\lambda$ Decomposition: Functional Change vs. Pure Iso-Functional Fiber Sliding
At any checkpoint $\theta_t = (B_t, A_t)$, its exact balanced-SVD surgery counterpart $\theta^{\mathrm{bal}}(f_t) = (U_H \Sigma_H^{1/2}, \Sigma_H^{1/2} V_H^\top)$ satisfies $B_{\mathrm{bal}} A_{\mathrm{bal}} = B_t A_t$ to machine precision ($\max|\Delta\mathrm{logit}| \le 5.6\times 10^{-14}$) with zero inactive fiber norm. Between late-stage converged checkpoints $t_1 \to t_2$:
$$\Delta\hat\lambda_{\mathrm{total}} = \underbrace{\bigl[\hat\lambda(\theta^{\mathrm{bal}}(f_{t_2})) - \hat\lambda(\theta^{\mathrm{bal}}(f_{t_1}))\bigr]}_{\Delta\hat\lambda_{\mathrm{func}}\text{ (functional change)}} + \underbrace{\bigl[(\hat\lambda(\theta_{t_2}) - \hat\lambda(\theta^{\mathrm{bal}}(f_{t_2}))) - (\hat\lambda(\theta_{t_1}) - \hat\lambda(\theta^{\mathrm{bal}}(f_{t_1})))\bigr]}_{\Delta\hat\lambda_{\mathrm{fiber}}\text{ (pure iso-functional fiber sliding)}}.$$

| Run ($t_1 \to t_2$) | $\mathrm{KL}_A(t_1 \to t_2)$ | $\hat\lambda_{\mathrm{raw}}(t_1 \to t_2)$ | $\hat\lambda_{\mathrm{bal}}(t_1 \to t_2)$ | $\Delta\hat\lambda_{\mathrm{total}}$ | $\Delta\hat\lambda_{\mathrm{func}}$ | $\Delta\hat\lambda_{\mathrm{fiber}}$ | Fiber Share |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`LoRA + AdamW` (`2500→8000`)** | `1.53e-8 → 3.48e-8` | `57.83 → 47.03` | `42.17 → 41.66` | **-10.79 ± 0.83** | **-0.51 ± 0.69** | **-10.28 ± 1.07** | **95.3% ± 6.4** |
| `LoRA + SGD+WD` (`1500→9000`) | `5.36e-7 → 6.61e-8` | `57.93 → 45.72` | `65.10 → 50.28` | -12.21 ± 0.68 | **-14.83 ± 0.87** | +2.62 ± 1.11 | *func. rank pruning* |
| `LoRA + Adam (ctrl)` (`2500→4500`) | `6.75e-9 → 4.72e-9` | `57.60 → 58.44` | `35.57 → 34.75` | +0.85 ± 0.76 | -0.82 ± 0.61 | +1.67 ± 0.97 | *(flat control)* |

**Diagnostic significance for Developmental Interpretability (DevInterp) & honest finite-$\beta$ caveat:**
- On **`LoRA + AdamW`**, after Task-A loss has already converged (`KL_A ~ 1.5e-8`), a standard LLC curve drops dramatically by **$-10.79 \pm 0.83$**—yet **$\approx 95\%$ of that drop ($95.3\% \pm 6.4$ pts; $\mathbf{-10.28 \pm 1.07}$) is pure iso-functional fiber sliding** with no detectable change in balanced-checkpoint LLC ($\Delta\hat\lambda_{\mathrm{func}} = -0.51 \pm 0.69$; Task-A KL `1.5e-8 → 3.5e-8`). Without checking iso-functional fiber surgery, a practitioner reading an LLC curve could misattribute a weight-decay fiber slide to a late-stage circuit/developmental phase transition!
- Conversely, our decomposition **cleanly distinguishes** pure fiber sliding (`AdamW`) from a genuine functional change (`SGD+WD`): at step 1500 of momentum-SGD (`dW_err = 2.31e-2`, transient overshoot $\|B_{\mathrm{in}}\|_F = 2.43$), $B_{1500} A_{1500}$ is still off target, so $\hat\lambda_{\mathrm{bal}}(1500) = 65.10$ sits near the full-rank value; coupled `SGD+WD` then shrinks that residual to `dW_err = 4.5e-3` (Task-A KL `5.4e-7 → 6.6e-8`), lowering $\hat\lambda_{\mathrm{bal}}$ to `50.28` ($\Delta\hat\lambda_{\mathrm{func}} = -14.83 \pm 0.87$) while collapsing $(p_{\mathrm{eff}}, q_{\mathrm{eff}})$ from $(1.91, 4.76) \to (0.01, 0.01)$.
- *Honest methodological nuance:* For `LoRA_Adam` and `LoRA_AdamW`, $\hat\lambda_{\mathrm{bal}}$ sits at `35–42` (below the exact rank-2 zero-residual `(0,0)` reference `49.96`) because finite-step optimization leaves tiny residual singular values $\sigma_{3..10} \sim 10^{-3}$; taking $\sqrt{\sigma_j} \sim 0.03$ on both factors places inactive balanced slots right at the finite-$\beta$ transition scale $\sim (\beta \mathcal{H}_{\mathrm{med}})^{-1/4}$ where MALA experiences a larger finite-$\beta$ dip. Because `dW_err` is essentially matched at $t_1$ (`5.0e-3`) and $t_2$ (`3.9e-3`), this offset cancels cleanly in $\Delta\hat\lambda_{\mathrm{func}} = -0.51 \pm 0.69$.

---

## 4. Part 4 — Multi-Head Attention $V{+}O$ vs. $Q{+}K$ Coupling Dichotomy inside the Real 2-Layer Causal Transformer (`[empirical]`)

We attach joint rank-$H = 4$ LoRA adapters to Layer 1's causal multi-head self-attention ($V{+}O$ and $Q{+}K$) across our three pretrained 2-layer Pre-RMSNorm Transformers with $n_{\mathrm{heads}} \in \{1, 2, 4\}$ ($d_{\mathrm{model}} = 12, d_h \in \{12, 6, 3\}$), evaluating the exact next-token KL divergence through causal softmax, Layer-1 RMSNorm + GELU MLP, final RMSNorm, and LM head ($\gamma = 60.0, s_{\mathrm{in}} = 4.5, \beta_{VO} = 10^8, \beta_{QK} = 8\times 10^8$, 20 chains, 1,100 burn-in + 1,500 sample steps).

- **Single-head `[LC]` theory (`AttentionLoRA.lean`, $d=12, H=4$):** $V{+}O$ obeys $\lambda_{V+O}(q) = 40 + 2(q - 2)_+^2$ ($40 \to 48$, jump $+8.0$); $Q{+}K$ obeys $\lambda_{Q+K}(q) \equiv 40.0$ flat across all $q \in \{0, 2, 4\}$ (exact $16$-dim $GL(4)$ gauge cancellation at $q=4$: Hessian rank $80$ vs. $96$).
- **Multi-head `[paper]` theory ($n_{\mathrm{heads}} \in \{2, 4\}$, $d_h + q \le 12$):** distinct per-head attention patterns decouple the pair into the additive single-projection regime $2 \times (22 + \frac{1}{2}(q - 2)_+^2) = 44 + (q - 2)_+^2$ ($44 \to 48$, jump $+4.0$, Hessian rank $96$ for **both** $V{+}O$ and $Q{+}K$).

| $n_{\mathrm{heads}}$ ($d_h$) | $q$ | Theory $V{+}O$ | $V{+}O$ MALA $\hat\lambda \pm \mathrm{SE}$ | $V{+}O$ `#H` (`g_lam`) | $V{+}O$ Acc / `hd` | Theory $Q{+}K$ | $Q{+}K$ MALA $\hat\lambda \pm \mathrm{SE}$ | $Q{+}K$ `#H` (`g_lam`) | $Q{+}K$ Acc / `hd` |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| **1-Head (`12`)** | 0 | 40.0 | 34.69 ± 0.36 | 0 (0.0) | 0.37 / +0.94 | 40.0 | 27.09 ± 0.38 | 0 (0.0) | 0.46 / +1.68 |
| **1-Head (`12`)** | 2 | 40.0 | 36.77 ± 0.38 | 48 (24.0) | 0.39 / +0.13 | 40.0 | 31.04 ± 0.52 | 44 (21.9) | 0.47 / +0.29 |
| **1-Head (`12`)** | **4** | **48.0** | **45.10 ± 0.40** | **96 (48.0)** | 0.47 / -0.30 | **40.0** | **38.48 ± 0.37** | **80 (39.8)** | 0.43 / +0.34 |
| **2-Head (`6`)** | 0 | 44.0 | 40.43 ± 0.43 | 0 (0.0) | 0.43 / +1.08 | 44.0 | 34.17 ± 0.37 | 0 (0.0) | 0.46 / +2.66 |
| **2-Head (`6`)** | 2 | 44.0 | 42.35 ± 0.42 | 48 (24.0) | 0.42 / +1.23 | 44.0 | 39.64 ± 0.60 | 48 (24.0) | 0.46 / -0.01 |
| **2-Head (`6`)** | **4** | **48.0** | **47.98 ± 0.12** | **96 (48.0)** | 0.57 / -0.03 | **48.0** | **47.86 ± 0.31** | **96 (48.0)** | 0.55 / -0.02 |
| **4-Head (`3`)** | 0 | 44.0 | 39.69 ± 0.36 | 0 (0.0) | 0.44 / +0.87 | 44.0 | 38.83 ± 0.40 | 0 (0.0) | 0.40 / +1.74 |
| **4-Head (`3`)** | 2 | 44.0 | 42.16 ± 0.62 | 48 (24.0) | 0.43 / +0.48 | 44.0 | 41.30 ± 0.57 | 48 (24.0) | 0.40 / +0.45 |
| **4-Head (`3`)** | **4** | **48.0** | **47.95 ± 0.18** | **96 (48.0)** | 0.58 / -0.46 | **48.0** | **48.47 ± 0.28** | **96 (48.0)** | 0.57 / -0.81 |

**Key findings & honest finite-$\beta$ observation:**
1. **Exact Hessian & MALA confirmation of the single-head $GL(4)$ gauge cancellation inside a full causal Transformer:** At $n_{\mathrm{heads}} = 1, q = 4$, the finite-difference next-token KL Hessian $\beta\nabla^2 K$ has **96** eigenvalues above $\gamma = 60$ (`g_lam = 48.0`) on $V{+}O$ vs. **80** (`g_lam = 39.8`) on $Q{+}K$—$16$ fewer, matching the $16$-dimensional $GL(4)$ gauge nullspace through causal softmax, RMSNorm, GELU MLP, and LM head! MALA reflects the same $6.6$-point gap ($45.10 \pm 0.40$ vs. $38.48 \pm 0.37$, with $V{+}O$'s $-2.9$ residual below $48.0$ coming from mild finite-$\gamma$ Hessian rotation on the less-conditioned 1-head $W_V$).
2. **Sharp multi-head decoupling reversal at $n_{\mathrm{heads}} = 2$ and $4$:** Immediately upon moving to $2$ or $4$ heads, **both** the $q = 0$ plateau rises ($V{+}O$: $34.69 \to 40.43 / 39.69$; $Q{+}K$: $27.09 \to 34.17 / 38.83$) and **both** $q = 4$ tops lock onto the exact decoupled theoretical value $48.0$ (`#H = 96` for both): $47.98 \pm 0.12$ / $47.86 \pm 0.31$ (2-head) and $47.95 \pm 0.18$ / $48.47 \pm 0.28$ (4-head), all with split-half drift $|\mathrm{hdiff}| \le 0.81$.
3. *Honest finite-$\beta$ note on $Q{+}K$ at $q = 0$:* At $q = 0$, $Q{+}K$ enters through softmax, whose Fisher spectrum is expected to be much softer than $V{+}O$'s (the eigenvalue ratio is not recorded in the saved JSON), producing a larger negative finite-$\beta$ logarithmic offset at $q=0$ (`27.09` at 1-head, `34.17` at 2-head, converging to `38.83` at 4-head where the 4-head $Q{+}K$ Fisher is best conditioned). Consequently, the most decisive asymptotic test of the single-head vs. multi-head dichotomy is the $q = 4$ regular/gauge-regular top (`45.10 vs. 38.48` at 1-head; `47.98 vs. 47.86` at 2-head; `47.95 vs. 48.47` at 4-head), corroborated by the above-$\gamma$ Hessian eigenvalue counts (`96 vs. 80` at 1-head; `96 vs. 96` at 2- and 4-head).

---

## 5. Reproducibility & Artifacts
- Code: [`experiments/transformer_fiber_surgery.py`](../experiments/transformer_fiber_surgery.py) (self-contained PyTorch/NumPy/Matplotlib, fixed seeds, float64 MALA with exact detailed-balance full-matrix linear preconditioning, end-to-end runtime `736.2s` on an Apple Silicon CPU; `--replot` redraws the figure from the saved JSON).
- Raw data: [`experiments/results/transformer_fiber_surgery.json`](../experiments/results/transformer_fiber_surgery.json).
- Multi-panel summary figure: [`experiments/results/transformer_fiber_surgery.png`](../experiments/results/transformer_fiber_surgery.png).
