# LoRA Fibers in Nonlinear MLPs and Self-Attention (Track 2b)

## Epistemic Status

- **Linear LoRA fiber phase transition (baseline):** `[LC]`.
  - $\lambda(q) = \lambda_{\mathrm{act}} + \lambda_{\mathrm{AW}}(m, n, k)$ for $q \le q_c$, and $\lambda_{\mathrm{act}} + \lambda_{\mathrm{AW}}^{\mathrm{even}}(m, n, k) + \frac{1}{2}(q - q_c)^2$ for $q > q_c$.
  - In `lean/EatYourFiber/{LoRASchur,LoRASchurLaplace,AoyagiWatanabeBoundary,AoyagiWatanabe}.lean` (see `notes/lora_fiber_phase_transition.md`).
- **Single-head self-attention, $V{+}O$ vs. $Q{+}K$ exact all-orders coupling dichotomy:** `[LC]` in `lean/EatYourFiber/AttentionLoRA.lean`, audited on `[propext, Classical.choice, Quot.sound]` in `lean/Audit.lean`.
  - $V{+}O$: $4\times$-amplified supercritical parabola $\lambda_{V+O}(q) = dH - \frac{H^2}{2} + 2(q - \frac{H}{2})_+^2$.
  - $Q{+}K$: exact all-orders $GL(q)$ gauge orbit, two-sided Schur complement reduction, and flat cancellation $\lambda_{Q+K}(q) \equiv dH - \frac{H^2}{2}$ across all $q \le H \le d$, including arbitrary orthogonal bases $U, V \in O(d)$ for `Q+K`.
  - Theorems: `attnVO_factor`, `attnQK_factor`, `attention_QK_gauge_orbit`, `attention_VO_exact_pair`, `attention_VO_subcritical_plateau`, `attention_VO_supercritical_parabola`, `twoSided_schur_reduction`, `attention_QK_exact_pair`, `attention_QK_flat_pair`, `attention_coupling_dichotomy_standard_init`, `attention_QK_general_gauge_orbit`, `attention_QK_general_exact_pair`, `attention_QK_general_flat_pair`.
- **Pullback bilipschitz invariance through nonlinearities and dead-neuron effective output dimension:** `[paper]` (local quadratic expansion, not yet formalized in Lean).
  - Non-degenerate Gauss-Newton Fisher $\mathcal{H} \succ 0 \implies$ identical $\lambda(q)$.
  - Dead neurons: $m_{\mathrm{eff}} = m - d_{\mathrm{dead}}$.
- **All experiments:** `[empirical]` in `experiments/lora_nonlinear_fiber.py` (data in `experiments/results/lora_nonlinear_fiber.json`, figure in `experiments/results/lora_nonlinear_fiber.png`).
  - MLP (GELU, ReLU, SwiGLU, DeadReLU2), teacher-student training, noisy low-sample fine-tuning.
  - Attention ($V, Q, V{+}O, Q{+}K$, 2-head, and full $Q,K,V,O$ training).
- **Multi-head shared-LoRA transversality and finite-noisy-data generalization tradeoffs:** `[empirical]` + `[paper]` local linearization; `[conjectural]` for deep multi-layer transformers at scale.
  - Transversality condition: $d_h + q \le d \implies Q$-$K$ gauge cancellation broken.
- **See also:** [transformer_fiber_surgery.md](transformer_fiber_surgery.md) carries the single-block attention and MLP results into a pretrained 2-layer causal Transformer (`[empirical]`):
  - the `V+O` vs. `Q+K` dichotomy and multi-head reversal at $n_{\mathrm{heads}} \in \{1, 2, 4\}$ (with exact Hessian ranks);
  - 2D $(p, q)$ fiber surgery and optimizer trajectories under next-token KL.

---

## Summary

- **Question:** does the LoRA fiber phase transition, and weight-decay-driven descent onto the subcritical plateau, survive outside the linear bilinear loss $\|B A - M^*\|_F^2$?
- **Short answer:** yes, with the refinements below, each seen empirically and explained algebraically.

**MLP nonlinearities preserve the transition** (`[empirical]`, §1)

- **Setup:** GELU, healthy ReLU and SwiGLU 2-layer MLPs, $M=N=H=8, r^*=0$.
- **Result:** MALA gives the same subcritical plateau ($q \le q_c = 4$) and supercritical parabola $\frac{1}{2}(q - 4)_+^2$, up to $\hat\lambda(8) = 31.97\text{--}32.01 \pm 0.07$ (theory $32.00$).
- **Dead neurons:** with $d_{\mathrm{dead}} = 2$ dead ReLU neurons, $m_{\mathrm{eff}} = 6$.
  - $q_c$ shifts $4 \to 5$, plateau $24.0 \to 19.5$, top $32.0 \to 24.0$ (MALA: $\hat\lambda(8) = 23.99 \pm 0.04$).

**MLP training: weight decay is what reaches the plateau** (`[empirical]`, §2.1)

- **Setup:** teacher-student GELU MLP, $M=N=H=10, r^*=2$. Theory plateau $42.0$; MALA references $q=0$: $39.52 \pm 0.19$, LoRA $q=8$: $50.06 \pm 0.07$.
- **Nonlinearity partly breaks subspace invariance:** under Adam, nonlinear gradients make $B_{\mathrm{in}}$ slightly nonzero ($\|B_{\mathrm{in}}\|_F = 0.106$), forcing 1 singular value of $A_{\mathrm{in}}$ to $0$.
  - Adam ($\hat\lambda = 45.59 \pm 0.23$) sits slightly below SGD ($\hat\lambda = 47.49 \pm 0.22$).
  - Both stay $+6.1$ to $+8.0$ above the plateau.
- **With weight decay:** `AdamW` ($\mathrm{wd}=0.04$: $\hat\lambda = 39.49 \pm 0.27$) and `SGD+WD` ($\hat\lambda = 39.72 \pm 0.30$) collapse $\|A_{\mathrm{in}}\|_F \to 0$ and land on the subcritical plateau.

**Low-sample noisy fine-tuning: a negative result** (`[empirical]`, §2.2)

- **Setup:** $d=16, H=8, S_{\mathrm{train}}=48, \sigma=0.25$.
- **No inactive fiber:** all $H = 8$ LoRA ranks fit noise ($\mathrm{rank}(\widehat{\Delta W}) = 8$), so $k_{\mathrm{eff}} = 0$ and the functional LLC saturates at the regular full-rank value $r(2d-r)/2 = 96.0$.
- **Moderate decoupled AdamW** ($\mathrm{wd} \in [0.05, 0.5]$) shrinks $\|A_{\perp B}\|_F$ ($5.10 \to 2.26$) but does not prune the noise-fitting ranks or improve test MSE ($1.03 \to 1.00$).
- **What does prevent noise overfitting:**
  - strong AdamW ($\mathrm{wd}=3.0$, test MSE $0.625$);
  - coupled SGD+WD ($\mathrm{wd}=0.03$, test MSE $0.482$);
  - subcritical rank-deficient init ($q=2$, test MSE $0.130$, functional $\hat\lambda = 61.65$).
- **Empirical LLC goes negative:** at the rank-$2$ generalizing solution, the raw centered train LLC $\beta\mathbb{E}[L_{\mathrm{train}}(\theta) - L_{\mathrm{train}}(\theta^*)]$ is $-314.7$, because that point is a strict saddle of the noisy training loss.

**Self-attention: $V$-$O$ vs. $Q$-$K$ coupling** (`[LC]` all orders for single-head weight-product losses; multi-head `[paper]` + `[empirical]`; §3)

- **Single projection:** LoRA on $V$ alone or $Q$ alone (inside softmax) follows the linear formula $\lambda(q) = 14 + \frac{1}{2}(q - 2)_+^2$ ($d=8, H=4$; MALA $q=4$: $16.04 \pm 0.04$ for $V$, $15.96 \pm 0.05$ for $Q$).
- **Single-head joint $V{+}O$:** both $A$-factors sit on the same side of $W_O W_V$, so it acts as one rank-$2H$ adapter with right-rank $2q$.
  - $\lambda_{V+O}(q) = 24 + 2(q - 2)_+^2$: the supercritical jump is $4\times$ larger ($+8.0$ vs. $+2.0$ single; MALA $q=4$: $32.00 \pm 0.06$).
- **Single-head joint $Q{+}K$:** $A_Q^\top$ sits left and $A_K$ right of $W_Q^\top W_K$, giving a balanced two-sided $(q, q)$ fiber with an exact all-orders $GL(q)$ gauge symmetry.
  - The supercritical parabola cancels: $\lambda_{Q+K}(q) \equiv 24.0$ for all $q \in \{0,\dots,4\}$ (MALA $q=4$: $23.99 \pm 0.05$).
- **2-head attention reverses the dichotomy** ($d=8, n_{\mathrm{heads}}=2, d_h=4, H=4$):
  - Distinct attention patterns $P^{(1)} \ne P^{(2)}$ make the loss constrain $(W_O^{(1)} W_V^{(1)}, W_O^{(2)} W_V^{(2)})$ (resp. $Q, K$) separately, so both `V+O` and `Q+K` become additive single-projection fibers.
  - Plateau: $24.0 \to 2 \times 14.0 = 28.0$ (MALA $q=0$: `26.59` / `26.78`).
  - Top: since $d_h + q = 8 \le d$, cross-head transversality gives full rank $64$ at $q = H = 4$, so both tops are $2 \times 16.0 = 32.0$ (`32.06 ± 0.07` and `32.10 ± 0.14`; jump $+4.0$ each vs. single-head $+8.0$ / $0.0$).
- **Full $Q,K,V,O$ training** ($d=10, H=5, T=8, r^*=1$):
  - `LoRA + Adam (wd=0)` keeps large excess fiber norms on all 4 projections ($\|A_{\perp B}\|_F \in [1.31, 2.37]$) and a raised joint LLC: $\hat\lambda_{\mathrm{ALL}} = 86.27 \pm 0.40$; $\hat\lambda_{V+O} = 43.78 \pm 0.22$ (above the $42.0$ plateau), $\hat\lambda_{Q+K} = 42.15 \pm 0.15$.
  - `AdamW`, `SGD+WD`, `Subcrit (q=2)` and `Small Balanced` all reach the subcritical plateau ($\hat\lambda_{\mathrm{ALL}} \in [78.30, 79.89]$).

---

## 1. Nonlinear MLP Fiber Sweep and Estimator Checks

### 1.1 Architecture and Pullback Equivalence (`[paper]`)

- **Model:** rank-$H$ LoRA adapter $(B, A)$ on the first hidden layer of a frozen 2-layer MLP:
  $$f(x; B, A) = W_2\,\phi\!\bigl((W_1 + B A) x + b_1\bigr), \qquad W_1, W_2 \in O(8),\; M = N = H = 8,\; r^* = 0,$$
- **Data:** $S = 256$ fixed i.i.d. Gaussian inputs $x_i \sim \mathcal{N}(0, I_8)$ (seed `7`).
- **Local expansion:** around $BA = 0$, the loss $L(B, A) = \frac{1}{S}\sum_{i=1}^S \|f(x_i; B, A) - f(x_i; 0, 0)\|^2$ expands as $\langle BA, \mathcal{H}\, BA\rangle + O(\|BA\|^3)$, with $\mathcal{H}$ the Gauss-Newton Fisher operator.
- **Claim:** if $\mathcal{H} \succ 0$ on $\mathbb{R}^{M \times N}$ (bilipschitz equivalence $c_1 \|BA\|_F^2 \le L(B, A) \le c_2 \|BA\|_F^2$), then $\lambda(q)$ is invariant and equals the linear theorem:
  $$\lambda(q) = 24.0 + \tfrac{1}{2}(q - 4)_+^2, \qquad q_c = 4.$$
- **Dead neurons:** if $d_{\mathrm{dead}}$ ReLU neurons never fire on the data ($\phi'(z_j(x_i)) \equiv 0$), those rows of $BA$ lie in $\ker(\mathcal{H})$, and $m_{\mathrm{eff}} = M - d_{\mathrm{dead}} = 6$:
  $$q_c' = \frac{N + H - m_{\mathrm{eff}}}{2} = 5, \qquad \lambda_{\mathrm{dead}}(q) = 19.5 + \tfrac{1}{2}(q - 5)_+^2.$$
  - Set via $b_{1,0}=b_{1,1}=-20$; measured firing rates `[0.0, 0.0, 0.480, 0.496, 0.457, 0.504, 0.520, 0.504]`.

### 1.2 Fiber Sweep Results (`[empirical]`)

- **Sampler:** vectorized float64 MALA, $\beta = 10^5, \gamma = 20$, 28 chains, 2,200 burn-in + 3,200 sample steps.
- **Diagnostics:** acceptance $0.46\text{--}0.57$, split-half drift $|\mathrm{hdiff}| \le 0.85$.

| $q$ | Theory ($m=8$) | GELU $\hat\lambda \pm \mathrm{SE}$ | ReLU $\hat\lambda \pm \mathrm{SE}$ | SwiGLU $\hat\lambda \pm \mathrm{SE}$ | Theory ($m_{\mathrm{eff}}=6$) | DeadReLU2 $\hat\lambda \pm \mathrm{SE}$ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 24.00 | 23.17 ± 0.09 | 23.27 ± 0.09 | 23.31 ± 0.10 | 19.50 | 18.73 ± 0.07 |
| 1 | 24.00 | 22.74 ± 0.17 | 22.82 ± 0.18 | 22.50 ± 0.24 | 19.50 | 18.29 ± 0.17 |
| 2 | 24.00 | 22.52 ± 0.18 | 22.90 ± 0.17 | 22.51 ± 0.22 | 19.50 | 18.27 ± 0.14 |
| 3 | 24.00 | 23.20 ± 0.18 | 23.23 ± 0.19 | 23.20 ± 0.20 | 19.50 | 18.56 ± 0.18 |
| 4 | 24.00 | 23.66 ± 0.17 | 23.71 ± 0.19 | 23.50 ± 0.26 | 19.50 | 18.62 ± 0.17 |
| 5 | 24.50 | 24.67 ± 0.14 | 24.63 ± 0.16 | 24.58 ± 0.20 | 19.50 | 19.33 ± 0.17 |
| 6 | 26.00 | 26.06 ± 0.14 | 26.17 ± 0.15 | 26.12 ± 0.16 | 20.00 | 20.16 ± 0.11 |
| 7 | 28.50 | 28.42 ± 0.06 | 28.54 ± 0.07 | 28.48 ± 0.10 | 21.50 | 21.51 ± 0.07 |
| 8 | 32.00 | 31.97 ± 0.07 | 31.97 ± 0.07 | 32.01 ± 0.07 | 24.00 | 23.99 ± 0.04 |

- **Healthy nonlinearities agree:** GELU, ReLU and SwiGLU agree pairwise within $1.6\sigma$ at every $q$ (largest gap: GELU vs. ReLU at $q = 2$).
- **Match to theory:** within $\le 0.04$ at $q = 8$.
- **Caveat:** on the singular plateau $q < q_c$ there is the expected $-0.7$ to $-1.5$ finite-$\beta$ logarithmic offset, the same as in the linear `rank0_8x8x8` baseline.
- **Dead ReLU:** `DeadReLU2` matches the predicted shift $q_c: 4 \to 5$, plateau $24.0 \to 19.5$, top $32.0 \to 24.0$ ($\hat\lambda(8) = 23.99 \pm 0.04$).

### 1.3 Estimator Stability, MALA vs. Unadjusted SGLD (`[empirical]`)

- **Setup:** GELU MLP ($M=N=H=8$) at $q = 0$ (theory $24.0$) and $q = 8$ (theory $32.0$); true $\Delta\lambda = +8.0$.

| Sampler | $\beta$ | $\gamma$ | Step $\epsilon$ | $\hat\lambda(q=0)$ | $\hat\lambda(q=8)$ | $\Delta\hat\lambda$ |
|---|---:|---:|---:|---:|---:|---:|
| MALA | $2\times 10^4$ | 20 | adaptive | 22.39 ± 0.10 | 32.00 ± 0.09 | +9.60 |
| MALA | $5\times 10^4$ | 20 | adaptive | 22.77 ± 0.11 | 32.00 ± 0.08 | +9.23 |
| MALA | $10^5$ | 20 | adaptive | 23.25 ± 0.14 | 32.00 ± 0.09 | +8.76 |
| MALA | $10^5$ | 10 | adaptive | 23.53 ± 0.10 | 32.02 ± 0.08 | +8.48 |
| MALA | $10^5$ | 40 | adaptive | 22.72 ± 0.10 | 32.03 ± 0.08 | +9.30 |
| SGLD (unadjusted) | $5\times 10^4$ | 20 | $5.0\times 10^{-6}$ | 22.14 ± 0.23 | 36.81 ± 0.08 | +14.66 |
| SGLD (unadjusted) | $5\times 10^4$ | 20 | $1.5\times 10^{-5}$ | 23.52 ± 0.15 | 54.54 ± 0.90 | +31.02 |

- **MALA is stable** across $5\times$ variation in $\beta$ and $4\times$ variation in $\gamma$:
  - $\hat\lambda(8) = 32.00\text{--}32.03$ throughout.
  - $\hat\lambda(0)$ rises monotonically toward $24.0$ as $\beta/\gamma$ increases ($22.39 \to 23.53$).
- **Unadjusted SGLD is biased upward** on the stiff supercritical directions: $36.81$ at $\epsilon = 5\times 10^{-6}$, $54.54$ at $\epsilon = 1.5\times 10^{-5}$.
- **Consequence:** Metropolis adjustment (MALA) is needed for quantitative LLC comparisons across fiber regimes.

---

## 2. Nonlinear GELU MLP Training Dynamics (`[empirical]`)

### 2.1 Teacher-Student Realizable Fiber, $M=N=H=10, r^*=2$ (Part 2a)

- **Setup:**
  - Frozen 2-layer GELU base ($W_1, W_2 \in O(10)$).
  - Rank-$2$ teacher shift $\Delta W^* = U_*\operatorname{diag}(2.0, 1.5) V_*^\top$.
  - $S_{\mathrm{train}} = 256$, $S_{\mathrm{test}} = 512$ (seed `17`, init seed `123`).
- **Theory:** excess dimensions $(m, n, k) = (8, 8, 8)$, $q_c = 4$, plateau $\lambda_{\mathrm{AW}} = 42.0$, standard LoRA top $\lambda(8) = 50.0$.
- **MALA references** ($\beta=10^5, \gamma=20$): $q=0 \to 39.52 \pm 0.19$, $q=8 \to 50.06 \pm 0.07$.

| Run | Init | Optimizer | Steps | Final Test Loss | $\|B_{\mathrm{in}}\|_F$ | $\|A_{\mathrm{in}}\|_F$ | $\mathrm{rank}_{>0.05}(A_{\mathrm{in}})$ | Excess $\mathrm{erank}$ | Final $\hat\lambda \pm \mathrm{SE}$ | Regime |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `MLP_LoRA_SGD` | LoRA ($q=10$) | SGD ($\mathrm{wd}=0$) | 12k | $5.38\times 10^{-11}$ | 0.000 | 4.012 | 8 | 4.69 | 47.49 ± 0.22 | Supercritical |
| `MLP_LoRA_Adam` | LoRA ($q=10$) | Adam ($\mathrm{wd}=0$, cos) | 8k | $6.54\times 10^{-32}$ | 0.106 | 3.640 | 7 | 4.01 | 45.59 ± 0.23 | Supercritical |
| `MLP_LoRA_AdamW` | LoRA ($q=10$) | AdamW ($\mathrm{wd}=0.04$, cos) | 22k | $2.02\times 10^{-11}$ | 0.093 | 0.019 | 0 | -0.11 | 39.49 ± 0.27 | Subcritical plateau |
| `MLP_LoRA_SGD_WD` | LoRA ($q=10$) | SGD+WD ($0.02 \to 0$ at 20k) | 24k | $5.37\times 10^{-30}$ | 0.000 | 0.000 | 0 | 0.06 | 39.72 ± 0.30 | Subcritical plateau |
| `MLP_Subcrit_Adam` | Subcrit ($B=0, q=2$) | Adam ($\mathrm{wd}=0$, cos) | 8k | $1.45\times 10^{-31}$ | 0.000 | 0.000 | 0 | -0.20 | 37.88 ± 0.34 | Subcritical plateau |
| `MLP_SmallBal_Adam` | Small bal ($10^{-3}$) | Adam ($\mathrm{wd}=0$, cos) | 8k | $2.60\times 10^{-13}$ | 0.166 | 0.266 | 3 | 0.27 | 39.92 ± 0.25 | Subcritical plateau |

- **Difference from the linear case:**
  - Linear: $U_\perp^\top B(t) = 0$ is an exact invariant of GD/Adam from $B_0 = 0$.
  - GELU: the transient gradient $\nabla_{\Delta W} L = 2\mathbb{E}[\operatorname{diag}(\phi'(z)) W_2^\top (f - f^*) x^\top]$ has full rank and makes $B_{\mathrm{in}}$ nonzero under Adam ($\sigma_1(B_{\mathrm{in}}) = 0.106$).
- **Effect on $A_{\mathrm{in}}$:** to keep $B_{\mathrm{in}} A_{\mathrm{in}} = 0$ at convergence, one inactive singular value of $A_{\mathrm{in}}$ goes to $0.000$ ($\mathrm{rank}(A_{\mathrm{in}}) = 7$ instead of $8$).
  - This lowers `MLP_LoRA_Adam` ($45.59$) slightly relative to `MLP_LoRA_SGD` ($47.49$).
  - For `MLP_LoRA_SGD`, $\sigma_8(A_{\mathrm{in}}) = 0.105 < \gamma^{-1/2}$ accounts for the gap below $50.06$.
- **Weight decay decides the regime:**
  - Adam and SGD without weight decay stay supercritical.
  - `AdamW` ($\hat\lambda = 39.49$) and `SGD+WD` ($\hat\lambda = 39.72$) reach the $q=0$ MALA plateau ($39.52 \pm 0.19$).
- **Caveat:** `MLP_Subcrit_Adam` ($37.88 \pm 0.34$) sits slightly below $39.52$.
  - Cause: finite-$\beta$ active-factor imbalance, $\sigma(A_{\mathrm{act}}) = [2.66, 1.46]$ vs. $\sigma(B_{\mathrm{act}}) = [1.18, 0.66]$.
  - All 8 inactive units stay identically $0$.

### 2.2 Low-Sample Noisy Fine-Tuning, $d=16, H=8, S_{\mathrm{train}}=48, \sigma=0.25$ (Part 2b)

- **Setup:**
  - 5 seeds (`1000..1004`), frozen 2-layer GELU base (unadapted test MSE $= 3.0939$).
  - True rank-$2$ shift $\Delta W^*$ with additive target noise $\sigma_\epsilon = 0.25$; $S_{\mathrm{test}} = 512$ clean held-out samples.
- **Estimators:**
  - Functional KL LLC $\hat\lambda_{\mathrm{func}}$: $K_{\mathrm{KL}}(\theta) = \mathbb{E}_{x\sim\text{test}}[\|f_\theta(x) - f_{\theta^*}(x)\|^2]$, $\beta=10^5, \gamma=20$.
  - Centered empirical LLC $\hat\lambda_{\mathrm{emp}}$: $\beta\mathbb{E}[L_{\mathrm{train}}(\theta) - L_{\mathrm{train}}(\theta^*)]$, $\beta=2000, \gamma=20$, only when $\|\nabla L_{\mathrm{train}}(\theta^*)\| < 10^{-2}$.

| Method | Test MSE (5-seed mean ± SE) | $\|\Delta\hat W - \Delta W^*\|_F$ | $\|A_{\perp B}\|_F$ | $\mathrm{erank}(BA)$ | Excess $\mathrm{erank}$ | $\hat\lambda_{\mathrm{func}}$ (seed 0) | $\hat\lambda_{\mathrm{emp}}$ (seed 0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `LoRA + Adam (wd=0)` | 1.0263 ± 0.1198 | 1.472 | 5.101 | 3.68 | 2.64 | 93.66 ± 0.43 | 87.08 ± 0.41 |
| `LoRA + AdamW (wd=0.05)` | 1.0246 ± 0.1197 | 1.471 | 2.625 | 3.68 | 1.98 | 95.81 ± 0.28 | 91.88 ± 0.62 |
| `LoRA + AdamW (wd=0.5)` | 0.9955 ± 0.1040 | 1.451 | 2.261 | 3.65 | 2.03 | 96.23 ± 0.29 | 92.53 ± 0.51 |
| `LoRA + AdamW (wd=3.0)` | 0.6249 ± 0.0431 | 1.152 | 1.798 | 3.27 | 1.66 | 96.20 ± 0.33 | N/A ($\|\nabla\| > 10^{-2}$) |
| `LoRA + SGD+WD (wd=0.03)` | 0.4817 ± 0.0242 | 1.006 | 1.369 | 2.91 | 2.52 | 95.79 ± 0.25 | N/A ($\|\nabla\| > 10^{-2}$) |
| `Subcrit LoRA (q=2) + Adam` | 0.1304 ± 0.0139 | 0.505 | 0.000 | 1.95 | 0.01 | 61.65 ± 0.58 | -314.73 ± 1.41 (saddle) |
| `Small Bal (1e-3) + Adam` | 0.9954 ± 0.1160 | 1.452 | 1.851 | 3.65 | 1.93 | 96.21 ± 0.23 | 93.03 ± 0.49 |

1. **Noise removes the inactive zero-loss fiber.**
   - Unregularized Adam (and `SmallBal_Adam`) uses all 8 LoRA ranks to fit training noise (train MSE $0.680$, $\mathrm{erank}(BA) = 3.68$).
   - So $k_{\mathrm{eff}} = H - \mathrm{rank}(BA) = 0$, and the functional LLC saturates at the regular full-rank dimension $r(2d-r)/2 = 8(32-8)/2 = 96.0$ (measured $93.66\text{--}96.23$).
2. **Decoupled AdamW vs. coupled SGD+WD.**
   - Moderate AdamW ($\mathrm{wd} = 0.05, 0.5$) shrinks $\|A_{\perp B}\|_F$ ($5.10 \to 2.26$) but leaves the noise-fitting product $BA$ unchanged ($\mathrm{erank}(BA) = 3.65$, test MSE $\approx 1.00$).
   - Reason: Adam's adaptive normalization keeps noisy updates $O(\eta)$ regardless of gradient magnitude.
   - Only very strong AdamW ($\mathrm{wd}=3.0$, test MSE $0.625$) or coupled L2 in `SGD+WD` ($\mathrm{wd}=0.03$, equivalent to nuclear-norm regularization on $BA$, test MSE $0.482$) prunes the noisy singular values.
3. **Subcritical rank-$2$ initialization is an exact generalizing saddle.**
   - **Setup:** $B_0 = 0, \mathrm{rank}(A_0) = 2$ keeps the 6 excess rows/columns on the invariant subspace $B_{:,2:}=0, A_{2:,:}=0$.
   - **Result:** matches the true rank $r^*=2$ ($\mathrm{erank}(BA) = 1.95$); test MSE drops $8\times$ ($1.026 \to 0.130$); functional LLC drops $93.7 \to 61.65$ (theory for $(16,16,8,r^*=2,q=0)$: $30 + 37.5 = 67.5$).
   - **Saddle:** on the noisy training loss this point has $\|\nabla L_{\mathrm{train}}\| = 7.2\times 10^{-16}$ but is a strict saddle ($L_{\mathrm{train}} = 0.895 > 0.680$).
   - **Consequence:** raw centered empirical MALA moves downhill into the noise-fitting basin ($\hat\lambda_{\mathrm{emp}} = -314.7$). This is why functional KL LLC (Watanabe's $K(w)$) is the right complexity measure for generalizing regularized/low-rank solutions.

---

## 3. Small Attention Block Coupling and Training

### 3.1 $V$-$O$ vs. $Q$-$K$ Coupling (`[LC]` Single-Head All Orders; `[paper]` Multi-Head)

- **Setup:** single-head self-attention $Y = \operatorname{softmax}(X W_Q^\top W_K X^\top / \sqrt{d}) X W_V^\top W_O^\top$.
  - Frozen orthogonal base weights $W_Q, W_K, W_V, W_O \in O(d)$.
  - Rank-$H$ LoRA adapters at the zero-loss fiber ($B_0 = 0, \mathrm{rank}(A_0) = q$).
  - Up to orthogonal gauge (base weights set to $I_d$), the exact all-orders matrix-product losses are certified in `lean/EatYourFiber/AttentionLoRA.lean`.

- **Single projection ($V$ alone or $Q$ alone; `[paper]` pullback + `[LC]` linear):**
  - Both have non-degenerate Gauss-Newton Fisher matrices (through softmax for $Q$).
  - Predicted: the standard one-sided transition $\lambda(q) = \lambda_{\mathrm{AW}}(d, d, H) + \frac{1}{2}(q - q_c)_+^2$ (even-parity form; for odd parity the parabola is based at $\lambda_{\mathrm{AW}}^{\mathrm{even}}$, $\tfrac18$ below the plateau).
  - For $d = 8, H = 4$: $q_c = 2$, $\lambda(q) = 14 + \frac{1}{2}(q - 2)_+^2$ (jump $+2.0$).

- **Joint $(V, O)$, same-side stacking (`[LC]`):** exact to all orders (`attnVO_factor`),
  $$(I_d + B_O A_O)(I_d + B_V A_V) - I_d = B_O A_O + \bigl((I_d + B_O A_O) B_V\bigr) A_V.$$
  - Left-multiplying $B_V$ by $M = I_d + B_O A_O$ has Jacobian $|\det M|^{-H} \in [2^{-H}, 2^H]$ near $0$.
  - This reduces the nonlinear loss exactly to a single rank-$2H$ LoRA fiber of right-rank $2q$ (`attention_VO_exact_pair`, `attention_VO_subcritical_plateau`, `attention_VO_supercritical_parabola`):
  $$\lambda_{V+O}(q) = dH - \frac{H^2}{2} + 2\left(q - \frac{H}{2}\right)_+^2, \qquad m_{\log} = 1.$$
  - For $d=8, H=4$: plateau $24.0$ (vs. additive $28.0$), top $32.0$ (jump $+8.0$, $4\times$ the single-projection jump).

- **Joint $(Q, K)$, two-sided balanced fiber with exact $GL(q)$ gauge symmetry (`[LC]`):** $W_Q$ enters transposed in $W_Q^\top W_K$; exact to all orders (`attnQK_factor`),
  $$(I_d + B_Q A_Q)^\top (I_d + B_K A_K) - I_d = B_K A_K + A_Q^\top \bigl((I_d + B_K A_K)^\top B_Q\bigr)^\top.$$
  - Left-multiplying $B_Q$ by $M = (I_d + B_K A_K)^\top$ reduces the nonlinear loss exactly to a balanced two-sided $(q, q)$ fiber.
  - **Gauge orbit** (`attention_QK_gauge_orbit`): for every $C \in \mathbb{R}^{q\times q}$ with $I_q + C$ invertible, choosing $D = (I_q + C)^{-\top} - I_q$ gives $(I_d + B_Q A_{QK})^\top (I_d + B_K A_{QK}) = I_d$ identically.
  - **Two-sided Schur complement reduction** (`twoSided_schur_reduction`) then gives (`attention_QK_exact_pair`, `attention_QK_flat_pair`):
  $$\lambda_{Q+K}(q) = \frac{q^2 + 2q(d-q)}{2} + \lambda_{\mathrm{AW}}(d - q, d - q, 2H - 2q) = dH - \frac{H^2}{2}, \qquad m_{\log} = 1$$
  - Identically flat for all $q \in \{0, \dots, H\}$ whenever $H \le d$. For $d = 8, H = 4$: $\lambda_{Q+K}(q) \equiv 24.0$, no supercritical penalty. Standard-init dichotomy: `attention_coupling_dichotomy_standard_init`.
  - **General orthogonal bases** (`[LC]` for `Q+K`; `[paper]` for non-orthogonal `V+O` overlap):
    - `Q+K`: in `AttentionLoRA.lean` (§8), `attention_QK_general_gauge_orbit`, `attention_QK_general_exact_pair` and `attention_QK_general_flat_pair` certify the all-orders $GL(q)$ gauge symmetry and flat cancellation $\lambda_{Q+K}(q) \equiv dH - \frac{H^2}{2}$ for arbitrary orthogonal initializations $A_{Q,0} = A_{\mathrm{can}} U$, $A_{K,0} = A_{\mathrm{can}} V$, any $U, V \in O(d)$ (including orthogonal or independently rotated $Q, K$ row spaces).
    - Proof idea: independent left/right orthogonal invariance of $\|U^\top X V\|_F^2$ after the nonlinear left-multiplication diffeomorphism.
    - `V+O` (`[paper]`): overlapping row spaces reduce the stacked right-rank to $q_{\mathrm{eff}} = \operatorname{rank}([A_{O,0};\, A_{V,0}]) \in [q, 2q]$, which lowers the supercritical jump smoothly to $\frac{1}{2}(q_{\mathrm{eff}} - H)_+^2$.

- **Multi-head decoupling and reversal** ($n_{\mathrm{heads}} \ge 2$, shared LoRA across heads; `[paper]` + `[empirical]`):
  - **Setup:** $n_{\mathrm{heads}} = 2$ ($d = 8, d_h = 4, H = 4$) with distinct per-head attention patterns $P^{(1)} \ne P^{(2)}$.
  - The loss constrains the two head products separately: $(W_O^{(1)} W_V^{(1)}, W_O^{(2)} W_V^{(2)})$ (resp. $(W_Q^{(1)\top} W_K^{(1)}, W_Q^{(2)\top} W_K^{(2)})$). This breaks single-head coupling at both ends of the fiber.
  - **At $q = 0$:** cross-adapter cancellation is restricted to block-diagonal head gauges $\mathfrak{gl}(d_h) \oplus \mathfrak{gl}(d_h)$ (dim $2 d_h^2 = 32$ instead of $d^2 = 64$). Their rank-constrained fiber has higher codimension than the decoupled branch, so the plateau rises from $24.0$ to the additive sum $2 \times \lambda_{\mathrm{AW}}(8, 8, 4) = 28.0$.
  - **At $q = H = 4$:** since $d_h + q = 8 \le d$, the per-head subspace (dim $d_h$) and shared adapter row space (dim $q$) are generically transverse. The first-order map in $(B_Q, B_K)$ or $(B_V, B_O)$ is then injective (full rank $2 d H = 64$), so both `V+O` and `Q+K` sit at $\lambda(4) = 64/2 = 32.0$.
    - Supercritical jump $+4.0 = 2 \times (+2.0)$ each, vs. single-head $+8.0$ for `V+O` and $0.0$ for `Q+K`.

### 3.2 Attention Coupling Sweep (Part 3a, `[empirical]`)

- **Setup:** single-head, $d=8, H=4, T=6, S=48$, $\gamma=20$; $\beta=2\times 10^5$ for $V, V{+}O$; $\beta=5\times 10^5$ for $Q, Q{+}K$.

| $q$ | Theory Single | `V_only` | `Q_only` (softmax) | Theory $V{+}O$ | `VO_joint` | Theory $Q{+}K$ | `QK_joint` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 14.00 | 13.56 ± 0.06 | 13.48 ± 0.09 | 24.00 | 23.23 ± 0.08 | 24.00 | 23.19 ± 0.07 |
| 1 | 14.00 | 13.18 ± 0.16 | 13.20 ± 0.17 | 24.00 | 22.53 ± 0.16 | 24.00 | 22.11 ± 0.22 |
| 2 | 14.00 | 13.88 ± 0.16 | 13.40 ± 0.14 | 24.00 | 23.26 ± 0.16 | 24.00 | 21.88 ± 0.28 |
| 3 | 14.50 | 14.78 ± 0.17 | 14.37 ± 0.14 | 26.00 | 26.30 ± 0.16 | 24.00 | 22.68 ± 0.22 |
| 4 | 16.00 | 16.04 ± 0.04 | 15.96 ± 0.05 | 32.00 | 32.00 ± 0.06 | 24.00 | 23.99 ± 0.05 |

- **2-head check:** $d=8, n_{\mathrm{heads}}=2, d_h=4, H=4$; decoupled additive theory $2 \times \lambda_{\mathrm{single}}(q) = 28 + (q - 2)_+^2$.

| $q$ | Decoupled Theory (`[paper]`) | 2-Head `VO_joint` | 2-Head `QK_joint` |
|---:|---:|---:|---:|
| 0 | 28.00 | 26.59 ± 0.13 | 26.78 ± 0.15 |
| 2 | 28.00 | 26.01 ± 0.30 | 25.88 ± 0.44 |
| 4 | 32.00 | 32.06 ± 0.07 | 32.10 ± 0.14 |

- **Single head:** the $V{+}O$ (`32.00`) vs. $Q{+}K$ (`23.99`) dichotomy matches the `[LC]` theory to `0.01`.
- **Two heads:** the reversal matches.
  - Both plateaus rise `24.0 -> 28.0` (finite-$\beta$ MALA `26.59`/`26.78`, matching $2\times$ the single-projection finite-$\beta$ value, $2 \times 13.5 \approx 27.0$).
  - Both supercritical tops land at `32.00` (`32.06` and `32.10`).
- **Caveat (pilot run):** with random non-orthogonal `[A_V; A_O W_V]` at $q=4$, $\sigma_{\min}(\widetilde{A}_0) = 0.095 < \gamma^{-1/2}$ reduced finite-$\gamma$ `VO_joint` to `28.81 ± 0.20`, matching $q_{\mathrm{eff}}=7$ theory `28.5`. The main sweep uses orthogonalized stacking to isolate the asymptotic `32.00`.

### 3.3 Full $Q, K, V, O$ LoRA Training (Part 3b, `[empirical]`)

- **Model:** 1-layer residual self-attention block, $d=10, H=5, T=8$, $S_{\mathrm{train}}=96, S_{\mathrm{test}}=192$, unadapted test loss $3.6161$.
- **Teacher:** rank-$1$ shifts on all four projections $(W_Q, W_K, W_V, W_O)$.
  - Net rank-$2$ delta on $W_O W_V$ and $W_Q^\top W_K$; excess $(8, 8, 8)$ per pair $\implies$ plateau $42.0$ per pair, $V{+}O$ LoRA top $50.0$, $Q{+}K$ flat $42.0$.
- **Sampler:** fixed-diagonal-preconditioned MALA (`precond=True`), 24–28 chains, 3,000 burn-in + 3,500 sample steps.
  - Split-half drift $|\mathrm{hdiff}| \le 1.6$.
  - Preconditioning removes a slow-mixing bias seen in unpreconditioned MALA at $\epsilon \sim 3\times 10^{-7}$.

| Run | Test Loss | $\|A_{\perp B}\|_F$ $(Q, K, V, O)$ | $\hat\lambda_{V+O}$ | $\hat\lambda_{Q+K}$ | Joint $\hat\lambda_{\mathrm{ALL}}$ |
|---|---:|---|---:|---:|---:|
| `Attn_LoRA_Adam` ($\mathrm{wd}=0$) | $1.13\times 10^{-30}$ | `(2.23, 1.31, 2.37, 2.04)` | 43.78 ± 0.22 | 42.15 ± 0.15 | 86.27 ± 0.40 |
| `Attn_LoRA_AdamW` ($\mathrm{wd}=0.04$) | $5.03\times 10^{-10}$ | `(0.51, 0.28, 0.42, 0.27)` | 40.37 ± 0.23 | 40.79 ± 0.32 | 79.89 ± 0.39 |
| `Attn_LoRA_SGD_WD` ($\mathrm{wd}=0.02\to 0$) | $3.13\times 10^{-7}$ | `(0.00, 0.00, 0.00, 0.08)` | 39.20 ± 0.19 | 39.55 ± 0.35 | 79.04 ± 0.57 |
| `Attn_Subcrit_Adam` ($q=2, \mathrm{wd}=0$) | $1.04\times 10^{-30}$ | `(0.00, 0.00, 0.00, 0.00)` | 39.88 ± 0.26 | 39.73 ± 0.31 | 78.30 ± 0.56 |
| `Attn_SmallBal_Adam` ($10^{-3}, \mathrm{wd}=0$) | $1.56\times 10^{-9}$ | `(0.22, 0.26, 0.20, 0.36)` | 40.12 ± 0.31 | 40.22 ± 0.36 | 78.70 ± 0.54 |

1. **Weight decay collapses excess fiber rank on all 4 projections** and lowers joint $\hat\lambda_{\mathrm{ALL}}$ by $6.4\text{--}7.2$ points.
   - `LoRA + Adam` keeps a high excess fiber norm (`1.31–2.37`) and $\hat\lambda_{\mathrm{ALL}} = 86.27 \pm 0.40$.
   - `AdamW` ($79.89$), `SGD+WD` ($79.04$), `Subcrit q=2` ($78.30$) and `SmallBal` ($78.70$) all reach the same subcritical plateau, $\approx 39.5\text{--}40.5$ per pair.
   - This matches the finite-$\beta$ excess-$(8,8,8)$ plateau $39.52$ from Part 2a.
2. **The two channels differ as the coupling theory predicts.** Under `Attn_LoRA_Adam`:
   - $\hat\lambda_{V+O} = 43.78 \pm 0.22$ is above the $42.0$ asymptotic plateau: a partial supercritical rise, moderated by random $[A_V; A_O W_V]$ overlap and gauge-coupled nonzero $B_{\mathrm{in}}$.
   - $\hat\lambda_{Q+K} = 42.15 \pm 0.15$ stays at the $42.0$ flat gauge ceiling.
   - Its $+2.4$ gap above `Subcrit` ($39.73$) is the same finite-$\beta$ singular-vs-regular offset seen in Part 3a at $q=0$ vs. $q=4$, not a supercritical parabola.

---

## 4. Reproducibility and Artifacts

- Code: `experiments/lora_nonlinear_fiber.py` (fixed deterministic seeds, float64 PyTorch CPU, preconditioned MALA + split-half convergence diagnostics).
- Raw data: `experiments/results/lora_nonlinear_fiber.json`.
- Summary figure: `experiments/results/lora_nonlinear_fiber.png`.
