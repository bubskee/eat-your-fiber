# Nonlinear MLP & Self-Attention LoRA Fiber Dynamics (Track 2b)

> **Epistemic status:**
> - **Linear LoRA fiber phase transition baseline ($\lambda(q) = \lambda_{\mathrm{act}} + \lambda_{\mathrm{AW}}(m, n, k)$ for $q \le q_c$, $\lambda_{\mathrm{act}} + \lambda_{\mathrm{AW}}^{\mathrm{even}}(m, n, k) + \frac{1}{2}(q - q_c)^2$ for $q > q_c$):** `[LC]` in `lean/EatYourFiber/{LoRASchur,LoRASchurLaplace,AoyagiWatanabeBoundary,AoyagiWatanabe}.lean` (see `notes/lora_fiber_phase_transition.md`).
> - **Single-head Self-Attention $V{+}O$ vs. $Q{+}K$ exact all-orders coupling dichotomy ($4\times$-amplified supercritical parabola $\lambda_{V+O}(q) = dH - \frac{H^2}{2} + 2(q - \frac{H}{2})_+^2$, exact all-orders $GL(q)$ gauge orbit on $Q{+}K$, two-sided Schur complement reduction, and complete flat cancellation $\lambda_{Q+K}(q) \equiv dH - \frac{H^2}{2}$ across all $q \le H \le d$, including arbitrary orthogonal bases $U, V \in O(d)$ for `Q+K`):** `[LC]` in `lean/EatYourFiber/AttentionLoRA.lean` (`attnVO_factor`, `attnQK_factor`, `attention_QK_gauge_orbit`, `attention_VO_exact_pair`, `attention_VO_subcritical_plateau`, `attention_VO_supercritical_parabola`, `twoSided_schur_reduction`, `attention_QK_exact_pair`, `attention_QK_flat_pair`, `attention_coupling_dichotomy_standard_init`, `attention_QK_general_gauge_orbit`, `attention_QK_general_exact_pair`, `attention_QK_general_flat_pair`), audited strictly on `[propext, Classical.choice, Quot.sound]` in `lean/Audit.lean`.
> - **Pullback bilipschitz invariance through nonlinearities (non-degenerate Gauss-Newton Fisher $\mathcal{H} \succ 0 \implies$ identical $\lambda(q)$) and dead-neuron effective output dimension $m_{\mathrm{eff}} = m - d_{\mathrm{dead}}$:** local quadratic expansion (`[paper]`, not yet formalized in Lean).
> - **All MLP (GELU, ReLU, SwiGLU, DeadReLU2), teacher-student training, noisy low-sample fine-tuning, and Attention ($V, Q, V{+}O, Q{+}K$, 2-head, and full $Q,K,V,O$ training) experiments:** `[empirical]` in `experiments/lora_nonlinear_fiber.py` (data in `experiments/results/lora_nonlinear_fiber.json`, figure in `experiments/results/lora_nonlinear_fiber.png`).
> - **Multi-head shared-LoRA transversality condition ($d_h + q \le d \implies Q$-$K$ gauge cancellation broken) and generalization tradeoffs under finite noisy data:** `[empirical]` + `[paper]` local linearization (`[conjectural]` for deep multi-layer transformers at scale).
>
> **See also:** [transformer_fiber_surgery.md](transformer_fiber_surgery.md) moves the single-block attention and MLP results here into a real pretrained 2-layer causal Transformer: the `V+O` vs. `Q+K` dichotomy and multi-head reversal at $n_{\mathrm{heads}} \in \{1, 2, 4\}$ (with exact Hessian ranks), plus 2D $(p, q)$ fiber surgery and optimizer trajectories under next-token KL (`[empirical]`).

---

## Executive Summary

Does the LoRA fiber phase transition and weight-decay-driven descent onto the subcritical plateau survive outside the linear bilinear loss $\|B A - M^*\|_F^2$? **Yes — with three important architectural and statistical refinements discovered empirically and explained algebraically:**

1. **Smooth and piecewise-linear MLP nonlinearities preserve the exact phase transition verbatim (`[empirical]`):** Across GELU, healthy ReLU, and SwiGLU 2-layer MLPs ($M=N=H=8, r^*=0$), MALA reproduces the identical subcritical plateau ($q \le q_c = 4$) and supercritical parabola $\frac{1}{2}(q - 4)_+^2$ up to $\hat\lambda(8) = 31.97\text{--}32.01 \pm 0.07$ (theory $32.00$). When $d_{\mathrm{dead}} = 2$ ReLU neurons are dead, the effective output dimension drops to $m_{\mathrm{eff}} = 6$, shifting the critical threshold right from $q_c = 4 \to 5$ and dropping the plateau from $24.0 \to 19.5$ and top from $32.0 \to 24.0$ (MALA: $\hat\lambda(8) = 23.99 \pm 0.04$).
2. **Nonlinearity partially breaks subspace invariance during MLP training, yet Adam/SGD without WD remain supercritical while AdamW/SGD+WD descend to the plateau (`[empirical]`):** On a teacher-student GELU MLP ($M=N=H=10, r^*=2$, theory plateau $42.0$, MALA $q=0$ ref $39.52 \pm 0.19$, LoRA $q=8$ ref $50.06 \pm 0.07$), nonlinear gradients kick $B_{\mathrm{in}}$ slightly nonzero under Adam ($\|B_{\mathrm{in}}\|_F = 0.106$, forcing 1 singular value of $A_{\mathrm{in}}$ to $0$), lowering Adam ($\hat\lambda = 45.59 \pm 0.23$) slightly below SGD ($\hat\lambda = 47.49 \pm 0.22$) — yet **both remain $+6.1$ to $+8.0$ above the plateau**. Both `AdamW` ($\mathrm{wd}=0.04$: $\hat\lambda = 39.49 \pm 0.27$) and `SGD+WD` ($\hat\lambda = 39.72 \pm 0.30$) collapse $\|A_{\mathrm{in}}\|_F \to 0$ and land cleanly on the subcritical plateau.
3. **Honest negative/nuanced finding on low-sample noisy fine-tuning (`[empirical]`):** When finite noisy training data ($d=16, H=8, S_{\mathrm{train}}=48, \sigma=0.25$) excites all $H = 8$ LoRA ranks into fitting noise ($\mathrm{rank}(\widehat{\Delta W}) = 8$), there is **no inactive zero-loss fiber** ($k_{\mathrm{eff}} = 0$) and the functional LLC saturates at the regular full-rank value $r(2d-r)/2 = 96.0$. Moderate decoupled AdamW ($\mathrm{wd} \in [0.05, 0.5]$) shrinks the nullspace $\|A_{\perp B}\|_F$ ($5.10 \to 2.26$) **without** pruning the noise-fitting ranks or improving test MSE ($1.03 \to 1.00$). Only strong AdamW ($\mathrm{wd}=3.0$, test MSE $0.625$), coupled SGD+WD ($\mathrm{wd}=0.03$, test MSE $0.482$), or subcritical rank-deficient init ($q=2$, test MSE $0.130$, functional $\hat\lambda = 61.65$) prevents noise overfitting. Moreover, at the rank-$2$ generalizing solution, the raw centered empirical train LLC $\beta\mathbb{E}[L_{\mathrm{train}}(\theta) - L_{\mathrm{train}}(\theta^*)]$ becomes **strongly negative** ($-314.7$) because the low-rank generalizing point is a strict saddle of the noisy training loss.
4. **Self-Attention reveals a sharp $V$-$O$ vs. $Q$-$K$ bilinear coupling dichotomy — Lean-certified to all orders (`[LC]`) for single-head weight-product losses, with a multi-head reversal (`[paper]` + `[empirical]`):**
   - Single-projection LoRA on $V$ alone **or $Q$ alone (inside softmax)** obeys the exact linear formula $\lambda(q) = 14 + \frac{1}{2}(q - 2)_+^2$ ($d=8, H=4$, MALA $q=4$: $16.04 \pm 0.04$ for $V$, $15.96 \pm 0.05$ for $Q$).
   - **Single-head joint $V{+}O$** stacks both $A$-factors on the same side of $W_O W_V$, behaving as a single rank-$2H$ adapter with right-rank $2q$: $\lambda_{V+O}(q) = 24 + 2(q - 2)_+^2$, **quadrupling** the supercritical jump ($+8.0$ vs. $+2.0$ single; MALA $q=4$: $32.00 \pm 0.06$).
   - **Single-head joint $Q{+}K$** places $A_Q^\top$ on the left and $A_K$ on the right of $W_Q^\top W_K$, forming a balanced two-sided $(q, q)$ fiber with an **exact all-orders $GL(q)$ gauge symmetry** that **completely annihilates the supercritical parabola**: $\lambda_{Q+K}(q) \equiv 24.0$ flat across all $q \in \{0,\dots,4\}$ (MALA $q=4$: $23.99 \pm 0.05$).
   - **2-head Attention ($d=8, n_{\mathrm{heads}}=2, d_h=4, H=4$) decouples the two adapters and reverses the dichotomy:** because the two heads have distinct attention patterns $P^{(1)} \ne P^{(2)}$, the loss constrains $(W_O^{(1)} W_V^{(1)}, W_O^{(2)} W_V^{(2)})$ (resp. $Q, K$) separately. Both `V+O` and `Q+K` collapse to the additive single-projection regime: plateau shifts from $24.0 \to 2 \times 14.0 = \mathbf{28.0}$ (MALA $q=0$: `26.59` / `26.78`) and because $d_h + q = 8 \le d$, cross-head transversality gives full rank $64$ at $q = H = 4$, putting **both** tops at $2 \times 16.0 = \mathbf{32.0}$ (`32.06 ± 0.07` and `32.10 ± 0.14`, jump $+4.0$ each vs. single-head $+8.0$ / $0.0$).
   - In **full $Q,K,V,O$ Attention training** ($d=10, H=5, T=8, r^*=1$), standard `LoRA + Adam (wd=0)` retains large excess fiber norms across all 4 projections ($\|A_{\perp B}\|_F \in [1.31, 2.37]$) and elevated joint LLC ($\hat\lambda_{\mathrm{ALL}} = 86.27 \pm 0.40$; $\hat\lambda_{V+O} = 43.78 \pm 0.22$ above the $42.0$ plateau, $\hat\lambda_{Q+K} = 42.15 \pm 0.15$), whereas `AdamW`, `SGD+WD`, `Subcrit (q=2)`, and `Small Balanced` all collapse onto the subcritical plateau ($\hat\lambda_{\mathrm{ALL}} \in [78.30, 79.89]$).

---

## 1. Nonlinear MLP Exact Fiber Sweep & Estimator Diagnostics

### 1.1 Architecture & Pullback Equivalence (`[paper]`)
We attach a rank-$H$ LoRA adapter $(B, A)$ to the first hidden layer of a frozen 2-layer MLP:
$$f(x; B, A) = W_2\,\phi\!\bigl((W_1 + B A) x + b_1\bigr), \qquad W_1, W_2 \in O(8),\; M = N = H = 8,\; r^* = 0,$$
evaluated on $S = 256$ fixed i.i.d. Gaussian inputs $x_i \sim \mathcal{N}(0, I_8)$ (seed `7`). Around $BA = 0$, the empirical squared-error loss $L(B, A) = \frac{1}{S}\sum_{i=1}^S \|f(x_i; B, A) - f(x_i; 0, 0)\|^2$ expands as $\langle BA, \mathcal{H}\, BA\rangle + O(\|BA\|^3)$, where $\mathcal{H}$ is the Gauss-Newton Fisher operator. Whenever $\mathcal{H} \succ 0$ on $\mathbb{R}^{M \times N}$ (bilipschitz equivalence $c_1 \|BA\|_F^2 \le L(B, A) \le c_2 \|BA\|_F^2$), the local learning coefficient $\lambda(q)$ is invariant and must match the linear theorem:
$$\lambda(q) = 24.0 + \tfrac{1}{2}(q - 4)_+^2, \qquad q_c = 4.$$
If $d_{\mathrm{dead}}$ ReLU neurons never fire on the data ($\phi'(z_j(x_i)) \equiv 0$, verified firing rates `[0.0, 0.0, 0.480, 0.496, 0.457, 0.504, 0.520, 0.504]` via $b_{1,0}=b_{1,1}=-20$), those $d_{\mathrm{dead}}$ rows of $BA$ lie identically in $\ker(\mathcal{H})$, reducing the effective output dimension to $m_{\mathrm{eff}} = M - d_{\mathrm{dead}} = 6$:
$$q_c' = \frac{N + H - m_{\mathrm{eff}}}{2} = 5, \qquad \lambda_{\mathrm{dead}}(q) = 19.5 + \tfrac{1}{2}(q - 5)_+^2.$$

### 1.2 Exact Fiber Sweep Results (`[empirical]`)
Vectorized float64 MALA ($\beta = 10^5, \gamma = 20$, 28 chains, 2,200 burn-in + 3,200 sample steps; acceptance $0.46\text{--}0.57$, split-half drift $|\mathrm{hdiff}| \le 0.85$):

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

All three healthy nonlinearities (GELU, ReLU, SwiGLU) agree with each other pairwise within $1.6\sigma$ at every $q$ (largest gap GELU vs. ReLU at $q = 2$) and match the linear theory to $\le 0.04$ at $q = 8$ (with the expected $-0.7$ to $-1.5$ finite-$\beta$ logarithmic correction on the singular plateau $q < q_c$, identical to the linear `rank0_8x8x8` baseline). `DeadReLU2` confirms the exact predicted shift $q_c: 4 \to 5$, plateau $24.0 \to 19.5$, and supercritical top $32.0 \to 24.0$ ($\hat\lambda(8) = 23.99 \pm 0.04$).

### 1.3 Estimator Stability & MALA vs. Unadjusted SGLD (`[empirical]`)
Evaluated on GELU MLP ($M=N=H=8$) at $q = 0$ (theory $24.0$) vs. $q = 8$ (theory $32.0$; true $\Delta\lambda = +8.0$):

| Sampler | $\beta$ | $\gamma$ | Step $\epsilon$ | $\hat\lambda(q=0)$ | $\hat\lambda(q=8)$ | $\Delta\hat\lambda$ |
|---|---:|---:|---:|---:|---:|---:|
| MALA | $2\times 10^4$ | 20 | adaptive | 22.39 ± 0.10 | 32.00 ± 0.09 | +9.60 |
| MALA | $5\times 10^4$ | 20 | adaptive | 22.77 ± 0.11 | 32.00 ± 0.08 | +9.23 |
| MALA | $10^5$ | 20 | adaptive | 23.25 ± 0.14 | 32.00 ± 0.09 | +8.76 |
| MALA | $10^5$ | 10 | adaptive | 23.53 ± 0.10 | 32.02 ± 0.08 | +8.48 |
| MALA | $10^5$ | 40 | adaptive | 22.72 ± 0.10 | 32.03 ± 0.08 | +9.30 |
| SGLD (unadjusted) | $5\times 10^4$ | 20 | $5.0\times 10^{-6}$ | 22.14 ± 0.23 | 36.81 ± 0.08 | +14.66 |
| SGLD (unadjusted) | $5\times 10^4$ | 20 | $1.5\times 10^{-5}$ | 23.52 ± 0.15 | 54.54 ± 0.90 | +31.02 |

MALA is rock-solid across $5\times$ variation in $\beta$ and $4\times$ variation in $\gamma$: $\hat\lambda(8) = 32.00\text{--}32.03$ identically, while $\hat\lambda(0)$ converges monotonically toward $24.0$ as $\beta/\gamma$ increases ($22.39 \to 23.53$). By contrast, **unadjusted SGLD exhibits severe upward discretization bias on the stiff supercritical directions** ($36.81$ at $\epsilon = 5\times 10^{-6}$, blowing up to $54.54$ at $\epsilon = 1.5\times 10^{-5}$), reinforcing why Metropolis adjustment (MALA) is essential for quantitative LLC comparisons across fiber regimes.

---

## 2. Nonlinear GELU MLP Training Dynamics (`[empirical]`)

### 2.1 Part 2a: Exact Teacher-Student Realizable Fiber ($M=N=H=10, r^*=2$)
Frozen 2-layer GELU base ($W_1, W_2 \in O(10)$), rank-$2$ teacher shift $\Delta W^* = U_*\operatorname{diag}(2.0, 1.5) V_*^\top$, $S_{\mathrm{train}} = 256$, $S_{\mathrm{test}} = 512$ (seed `17`, init seed `123`). Excess dimensions $(m, n, k) = (8, 8, 8)$, $q_c = 4$, theoretical plateau $\lambda_{\mathrm{AW}} = 42.0$, standard LoRA top $\lambda(8) = 50.0$. Exact MALA references at $\beta=10^5, \gamma=20$: $q=0 \to 39.52 \pm 0.19$, $q=8 \to 50.06 \pm 0.07$.

| Run | Init | Optimizer | Steps | Final Test Loss | $\|B_{\mathrm{in}}\|_F$ | $\|A_{\mathrm{in}}\|_F$ | $\mathrm{rank}_{>0.05}(A_{\mathrm{in}})$ | Excess $\mathrm{erank}$ | Final $\hat\lambda \pm \mathrm{SE}$ | Regime |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `MLP_LoRA_SGD` | LoRA ($q=10$) | SGD ($\mathrm{wd}=0$) | 12k | $5.38\times 10^{-11}$ | 0.000 | 4.012 | 8 | 4.69 | 47.49 ± 0.22 | Supercritical |
| `MLP_LoRA_Adam` | LoRA ($q=10$) | Adam ($\mathrm{wd}=0$, cos) | 8k | $6.54\times 10^{-32}$ | **0.106** | 3.640 | 7 | 4.01 | 45.59 ± 0.23 | Supercritical |
| `MLP_LoRA_AdamW` | LoRA ($q=10$) | AdamW ($\mathrm{wd}=0.04$, cos) | 22k | $2.02\times 10^{-11}$ | 0.093 | **0.019** | **0** | -0.11 | **39.49 ± 0.27** | Subcritical plateau |
| `MLP_LoRA_SGD_WD` | LoRA ($q=10$) | SGD+WD ($0.02 \to 0$ at 20k) | 24k | $5.37\times 10^{-30}$ | 0.000 | **0.000** | **0** | 0.06 | **39.72 ± 0.30** | Subcritical plateau |
| `MLP_Subcrit_Adam` | Subcrit ($B=0, q=2$) | Adam ($\mathrm{wd}=0$, cos) | 8k | $1.45\times 10^{-31}$ | 0.000 | 0.000 | 0 | -0.20 | **37.88 ± 0.34** | Subcritical plateau |
| `MLP_SmallBal_Adam` | Small bal ($10^{-3}$) | Adam ($\mathrm{wd}=0$, cos) | 8k | $2.60\times 10^{-13}$ | 0.166 | 0.266 | 3 | 0.27 | **39.92 ± 0.25** | Subcritical plateau |

**Key nonlinear finding:** Unlike the linear problem (where $U_\perp^\top B(t) = 0$ is an exact invariant of GD/Adam from $B_0 = 0$), in a GELU MLP the transient gradient $\nabla_{\Delta W} L = 2\mathbb{E}[\operatorname{diag}(\phi'(z)) W_2^\top (f - f^*) x^\top]$ has full rank and kicks $B_{\mathrm{in}}$ nonzero under Adam ($\sigma_1(B_{\mathrm{in}}) = 0.106$). To satisfy $B_{\mathrm{in}} A_{\mathrm{in}} = 0$ at convergence, one inactive singular value of $A_{\mathrm{in}}$ collapses to $0.000$ ($\mathrm{rank}(A_{\mathrm{in}}) = 7$ instead of $8$), slightly lowering `MLP_LoRA_Adam` ($45.59$) relative to `MLP_LoRA_SGD` ($47.49$, where $\sigma_8(A_{\mathrm{in}}) = 0.105 < \gamma^{-1/2}$ accounts for the gap below $50.06$). Crucially, **both Adam and SGD without weight decay remain strongly supercritical**, whereas `AdamW` ($\hat\lambda = 39.49$) and `SGD+WD` ($\hat\lambda = 39.72$) descend cleanly onto the exact $q=0$ MALA plateau ($39.52 \pm 0.19$). Note also that `MLP_Subcrit_Adam` ($37.88 \pm 0.34$) sits slightly below $39.52$ due to finite-$\beta$ active-factor imbalance ($\sigma(A_{\mathrm{act}}) = [2.66, 1.46]$ vs. $\sigma(B_{\mathrm{act}}) = [1.18, 0.66]$) while keeping all 8 inactive units identically $0$.

### 2.2 Part 2b: Realistic Low-Sample Noisy Fine-Tuning ($d=16, H=8, S_{\mathrm{train}}=48, \sigma=0.25$)
5 seeds (`1000..1004`), frozen 2-layer GELU base (unadapted test MSE $= 3.0939$), true rank-$2$ shift $\Delta W^*$ with additive target noise $\sigma_\epsilon = 0.25$, $S_{\mathrm{test}} = 512$ clean held-out samples. Functional KL LLC $\hat\lambda_{\mathrm{func}}$ uses $K_{\mathrm{KL}}(\theta) = \mathbb{E}_{x\sim\text{test}}[\|f_\theta(x) - f_{\theta^*}(x)\|^2]$ ($\beta=10^5, \gamma=20$); centered empirical LLC $\hat\lambda_{\mathrm{emp}}$ uses $\beta\mathbb{E}[L_{\mathrm{train}}(\theta) - L_{\mathrm{train}}(\theta^*)]$ ($\beta=2000, \gamma=20$) when $\|\nabla L_{\mathrm{train}}(\theta^*)\| < 10^{-2}$.

| Method | Test MSE (5-seed mean ± SE) | $\|\Delta\hat W - \Delta W^*\|_F$ | $\|A_{\perp B}\|_F$ | $\mathrm{erank}(BA)$ | Excess $\mathrm{erank}$ | $\hat\lambda_{\mathrm{func}}$ (seed 0) | $\hat\lambda_{\mathrm{emp}}$ (seed 0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `LoRA + Adam (wd=0)` | 1.0263 ± 0.1198 | 1.472 | 5.101 | 3.68 | 2.64 | 93.66 ± 0.43 | 87.08 ± 0.41 |
| `LoRA + AdamW (wd=0.05)` | 1.0246 ± 0.1197 | 1.471 | 2.625 | 3.68 | 1.98 | 95.81 ± 0.28 | 91.88 ± 0.62 |
| `LoRA + AdamW (wd=0.5)` | 0.9955 ± 0.1040 | 1.451 | 2.261 | 3.65 | 2.03 | 96.23 ± 0.29 | 92.53 ± 0.51 |
| `LoRA + AdamW (wd=3.0)` | **0.6249 ± 0.0431** | 1.152 | 1.798 | 3.27 | 1.66 | 96.20 ± 0.33 | N/A ($\|\nabla\| > 10^{-2}$) |
| `LoRA + SGD+WD (wd=0.03)` | **0.4817 ± 0.0242** | 1.006 | 1.369 | 2.91 | 2.52 | 95.79 ± 0.25 | N/A ($\|\nabla\| > 10^{-2}$) |
| `Subcrit LoRA (q=2) + Adam` | **0.1304 ± 0.0139** | **0.505** | **0.000** | **1.95** | **0.01** | **61.65 ± 0.58** | **-314.73 ± 1.41** (saddle!) |
| `Small Bal (1e-3) + Adam` | 0.9954 ± 0.1160 | 1.452 | 1.851 | 3.65 | 1.93 | 96.21 ± 0.23 | 93.03 ± 0.49 |

**Honest negative & structural findings:**
1. **Noise destroys the inactive zero-loss fiber:** With $S_{\mathrm{train}} = 48$ noisy samples, unregularized Adam (and `SmallBal_Adam`) uses **all 8 LoRA ranks** to fit the training noise (train MSE $0.680$, $\mathrm{erank}(BA) = 3.68$). Because $k_{\mathrm{eff}} = H - \mathrm{rank}(BA) = 0$, the functional LLC saturates at the regular full-rank dimension $r(2d-r)/2 = 8(32-8)/2 = 96.0$ (measured $93.66\text{--}96.26$).
2. **Decoupled AdamW vs. coupled SGD+WD on noisy tasks:** Moderate AdamW ($\mathrm{wd} = 0.05, 0.5$) shrinks $\|A_{\perp B}\|_F$ ($5.10 \to 2.26$) but leaves the noise-fitting product $BA$ untouched ($\mathrm{erank}(BA) = 3.65$, test MSE $\approx 1.00$) because Adam's adaptive normalization keeps noisy updates $O(\eta)$ regardless of gradient magnitude. Only very strong AdamW ($\mathrm{wd}=3.0$, test MSE $0.625$) or coupled L2 in `SGD+WD` ($\mathrm{wd}=0.03$, equivalent to nuclear-norm regularization on $BA$, test MSE $0.482$) prunes the noisy singular values.
3. **Subcritical rank-$2$ initialization is an exact generalizing saddle:** Initializing $B_0 = 0, \mathrm{rank}(A_0) = 2$ keeps the 6 excess rows/columns on the invariant subspace $B_{:,2:}=0, A_{2:,:}=0$, matching the true rank $r^*=2$ ($\mathrm{erank}(BA) = 1.95$), slashing test MSE by $8\times$ ($1.026 \to 0.130$), and dropping functional LLC from $93.7 \to 61.65$ (theory for $(16,16,8,r^*=2,q=0)$: $30 + 37.5 = 67.5$). Notably, on the noisy training loss this point has $\|\nabla L_{\mathrm{train}}\| = 7.2\times 10^{-16}$ yet sits at a **strict saddle** ($L_{\mathrm{train}} = 0.895 > 0.680$), causing raw centered empirical MALA to fall downhill into the noise-fitting basin ($\hat\lambda_{\mathrm{emp}} = -314.7$). This cleanly demonstrates why functional KL LLC (Watanabe's $K(w)$) is the right complexity measure for generalizing regularized/low-rank solutions.

---

## 3. Small Attention Block: $Q, K, V, O$ Coupling Geometry & Training Dynamics

### 3.1 Algebraic Coupling Dichotomy: $V$-$O$ vs. $Q$-$K$ (`[LC]` single-head exact all-orders; `[paper]` multi-head)
Consider single-head self-attention $Y = \operatorname{softmax}(X W_Q^\top W_K X^\top / \sqrt{d}) X W_V^\top W_O^\top$ with frozen orthogonal base weights $W_Q, W_K, W_V, W_O \in O(d)$ and rank-$H$ LoRA adapters at the zero-loss fiber ($B_0 = 0, \mathrm{rank}(A_0) = q$). Up to orthogonal gauge (setting base weights to $I_d$), the exact all-orders matrix-product losses are certified in `lean/EatYourFiber/AttentionLoRA.lean`:

- **Single projection ($V$ alone or $Q$ alone, `[paper]` pullback + `[LC]` linear):** Both have non-degenerate Gauss-Newton Fisher matrices (even through softmax for $Q$), predicting the standard one-sided transition $\lambda(q) = \lambda_{\mathrm{AW}}(d, d, H) + \frac{1}{2}(q - q_c)_+^2$ (even-parity form; for odd parity the parabola is based at $\lambda_{\mathrm{AW}}^{\mathrm{even}}$, $\tfrac18$ below the plateau). For $d = 8, H = 4$: $q_c = 2$, $\lambda(q) = 14 + \frac{1}{2}(q - 2)_+^2$ (jump $+2.0$).
- **Joint $(V, O)$ coupling (same-side stacking, `[LC]`):** Exact to **all orders** (`attnVO_factor`),
  $$(I_d + B_O A_O)(I_d + B_V A_V) - I_d = B_O A_O + \bigl((I_d + B_O A_O) B_V\bigr) A_V.$$
  Left-multiplying $B_V$ by $M = I_d + B_O A_O$ has Jacobian $|\det M|^{-H} \in [2^{-H}, 2^H]$ near $0$, reducing the nonlinear loss verbatim to a single rank-$2H$ LoRA fiber of right-rank $2q$ (`attention_VO_exact_pair`, `attention_VO_subcritical_plateau`, `attention_VO_supercritical_parabola`):
  $$\lambda_{V+O}(q) = dH - \frac{H^2}{2} + 2\left(q - \frac{H}{2}\right)_+^2, \qquad m_{\log} = 1.$$
  For $d=8, H=4$: plateau $24.0$ (vs. additive $28.0$), top $32.0$ ($\mathbf{4\times}$ single jump $+8.0$!).
- **Joint $(Q, K)$ coupling (two-sided balanced fiber & exact $GL(q)$ gauge symmetry, `[LC]`):** Because $W_Q$ enters **transposed** in $W_Q^\top W_K$, exact to **all orders** (`attnQK_factor`),
  $$(I_d + B_Q A_Q)^\top (I_d + B_K A_K) - I_d = B_K A_K + A_Q^\top \bigl((I_d + B_K A_K)^\top B_Q\bigr)^\top.$$
  Here left-multiplying $B_Q$ by $M = (I_d + B_K A_K)^\top$ reduces the nonlinear loss verbatim to a **balanced two-sided $(q, q)$ fiber** with an **exact all-orders $GL(q)$ zero-loss gauge orbit** (`attention_QK_gauge_orbit`: for every $C \in \mathbb{R}^{q\times q}$ with $I_q + C$ invertible, choosing $D = (I_q + C)^{-\top} - I_q$ gives $(I_d + B_Q A_{QK})^\top (I_d + B_K A_{QK}) = I_d$ identically). Applying the **exact two-sided Schur complement reduction** (`twoSided_schur_reduction`) yields (`attention_QK_exact_pair`, `attention_QK_flat_pair`):
  $$\lambda_{Q+K}(q) = \frac{q^2 + 2q(d-q)}{2} + \lambda_{\mathrm{AW}}(d - q, d - q, 2H - 2q) = dH - \frac{H^2}{2}, \qquad m_{\log} = 1$$
  **identically flat for ALL $q \in \{0, \dots, H\}$ whenever $H \le d$!** For $d = 8, H = 4$: $\lambda_{Q+K}(q) \equiv 24.0$ (zero supercritical penalty; standard-init dichotomy certified in `attention_coupling_dichotomy_standard_init`).
  - *General orthogonal bases (`[LC]` for `Q+K`; `[paper]` for non-orthogonal `V+O` overlap):* In `AttentionLoRA.lean` (§8), `attention_QK_general_gauge_orbit`, `attention_QK_general_exact_pair`, and `attention_QK_general_flat_pair` certify `Q+K` all-orders $GL(q)$ gauge symmetry and exact flat cancellation $\lambda_{Q+K}(q) \equiv dH - \frac{H^2}{2}$ for **arbitrary orthogonal initializations** $A_{Q,0} = A_{\mathrm{can}} U$, $A_{K,0} = A_{\mathrm{can}} V$ with any $U, V \in O(d)$ (including orthogonal or independently rotated $Q, K$ row spaces), exploiting independent left/right orthogonal invariance $U, V$ of $\|U^\top X V\|_F^2$ after the nonlinear left-multiplication diffeomorphism. Conversely, for `V+O` (`[paper]`), overlapping row spaces reduce the stacked right-rank to $q_{\mathrm{eff}} = \operatorname{rank}([A_{O,0};\, A_{V,0}]) \in [q, 2q]$, moderating the supercritical jump smoothly to $\frac{1}{2}(q_{\mathrm{eff}} - H)_+^2$.
- **Multi-head decoupling & reversal ($n_{\mathrm{heads}} \ge 2$, shared LoRA across heads, `[paper]` + `[empirical]`):** With $n_{\mathrm{heads}} = 2$ ($d = 8, d_h = 4, H = 4$) and distinct per-head attention patterns $P^{(1)} \ne P^{(2)}$, the loss separately constrains the two head products $(W_O^{(1)} W_V^{(1)}, W_O^{(2)} W_V^{(2)})$ (resp. $(W_Q^{(1)\top} W_K^{(1)}, W_Q^{(2)\top} W_K^{(2)})$). This breaks single-head coupling at *both* ends of the fiber:
  - At $q = 0$, cross-adapter cancellation is restricted to block-diagonal head gauges $\mathfrak{gl}(d_h) \oplus \mathfrak{gl}(d_h)$ (dim $2 d_h^2 = 32$ instead of $d^2 = 64$), whose rank-constrained fiber has higher codimension than the decoupled branch, so the plateau shifts up from $24.0$ to the additive sum $2 \times \lambda_{\mathrm{AW}}(8, 8, 4) = \mathbf{28.0}$.
  - At $q = H = 4$, because $d_h + q = 8 \le d$, the per-head subspace (dim $d_h$) and shared adapter row space (dim $q$) are generically transverse, making the first-order map in $(B_Q, B_K)$ or $(B_V, B_O)$ injective (full rank $2 d H = 64$), so **both** `V+O` and `Q+K` sit at $\lambda(4) = 64/2 = \mathbf{32.0}$ (supercritical jump $+4.0 = 2 \times (+2.0)$ each, versus single-head $+8.0$ for `V+O` and $0.0$ for `Q+K`).

### 3.2 Part 3a: Exact Attention Coupling Sweep (`[empirical]`)
Single-head ($d=8, H=4, T=6, S=48$, $\gamma=20$; $\beta=2\times 10^5$ for $V, V{+}O$; $\beta=5\times 10^5$ for $Q, Q{+}K$):

| $q$ | Theory Single | `V_only` | `Q_only` (softmax) | Theory $V{+}O$ | `VO_joint` | Theory $Q{+}K$ | `QK_joint` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 14.00 | 13.56 ± 0.06 | 13.48 ± 0.09 | 24.00 | 23.23 ± 0.08 | 24.00 | 23.19 ± 0.07 |
| 1 | 14.00 | 13.18 ± 0.16 | 13.20 ± 0.17 | 24.00 | 22.53 ± 0.16 | 24.00 | 22.11 ± 0.22 |
| 2 | 14.00 | 13.88 ± 0.16 | 13.40 ± 0.14 | 24.00 | 23.26 ± 0.16 | 24.00 | 21.88 ± 0.28 |
| 3 | 14.50 | 14.78 ± 0.17 | 14.37 ± 0.14 | 26.00 | 26.30 ± 0.16 | 24.00 | 22.68 ± 0.22 |
| 4 | 16.00 | **16.04 ± 0.04** | **15.96 ± 0.05** | 32.00 | **32.00 ± 0.06** | 24.00 | **23.99 ± 0.05** |

**2-Head Attention check** ($d=8, n_{\mathrm{heads}}=2, d_h=4, H=4$; decoupled additive theory $2 \times \lambda_{\mathrm{single}}(q) = 28 + (q - 2)_+^2$):

| $q$ | Decoupled Theory (`[paper]`) | 2-Head `VO_joint` | 2-Head `QK_joint` |
|---:|---:|---:|---:|
| 0 | 28.00 | 26.59 ± 0.13 | 26.78 ± 0.15 |
| 2 | 28.00 | 26.01 ± 0.30 | 25.88 ± 0.44 |
| 4 | 32.00 | **32.06 ± 0.07** | **32.10 ± 0.14** |

Every single theoretical prediction is confirmed—both the single-head $V{+}O$ (`32.00`) vs. $Q{+}K$ (`23.99`) dichotomy (matching `[LC]` theory to `0.01`) and the 2-head decoupling reversal, where *both* the plateau rises from `24.0 -> 28.0` (finite-$\beta$ MALA `26.59`/`26.78`, matching $2\times$ the single-projection finite-$\beta$ value $2 \times 13.5 \approx 27.0$) and *both* supercritical tops land at `32.00` (`32.06` and `32.10`). *(Caveat: in the pilot with random non-orthogonal `[A_V; A_O W_V]` at $q=4$, $\sigma_{\min}(\widetilde{A}_0) = 0.095 < \gamma^{-1/2}$ reduced finite-$\gamma$ `VO_joint` to `28.81 ± 0.20`, matching $q_{\mathrm{eff}}=7$ theory `28.5`; orthogonalized stacking in the main sweep isolates the exact asymptotic `32.00`.)*

### 3.3 Part 3b: Full $Q, K, V, O$ Attention LoRA Training Dynamics (`[empirical]`)
1-layer residual self-attention block ($d=10, H=5, T=8$, $S_{\mathrm{train}}=96, S_{\mathrm{test}}=192$, unadapted test loss $3.6161$), teacher has rank-$1$ shifts on all four projections $(W_Q, W_K, W_V, W_O)$ (net rank-$2$ delta on $W_O W_V$ and $W_Q^\top W_K$, excess $(8, 8, 8)$ per pair $\implies$ plateau $42.0$ per pair, $V{+}O$ LoRA top $50.0$, $Q{+}K$ flat $42.0$). Evaluated with **fixed-diagonal-preconditioned MALA** (`precond=True`, 24–28 chains, 3,000 burn-in + 3,500 sample steps; stationarity verified via split-half drift $|\mathrm{hdiff}| \le 1.6$, eliminating the slow-mixing bias we caught in unpreconditioned MALA at $\epsilon \sim 3\times 10^{-7}$):

| Run | Test Loss | $\|A_{\perp B}\|_F$ $(Q, K, V, O)$ | $\hat\lambda_{V+O}$ | $\hat\lambda_{Q+K}$ | Joint $\hat\lambda_{\mathrm{ALL}}$ |
|---|---:|---|---:|---:|---:|
| `Attn_LoRA_Adam` ($\mathrm{wd}=0$) | $1.13\times 10^{-30}$ | `(2.23, 1.31, 2.37, 2.04)` | **43.78 ± 0.22** | 42.15 ± 0.15 | **86.27 ± 0.40** |
| `Attn_LoRA_AdamW` ($\mathrm{wd}=0.04$) | $5.03\times 10^{-10}$ | `(0.51, 0.28, 0.42, 0.27)` | 40.37 ± 0.23 | 40.79 ± 0.32 | **79.89 ± 0.39** |
| `Attn_LoRA_SGD_WD` ($\mathrm{wd}=0.02\to 0$) | $3.13\times 10^{-7}$ | `(0.00, 0.00, 0.00, 0.08)` | 39.20 ± 0.19 | 39.55 ± 0.35 | **79.04 ± 0.57** |
| `Attn_Subcrit_Adam` ($q=2, \mathrm{wd}=0$) | $1.04\times 10^{-30}$ | `(0.00, 0.00, 0.00, 0.00)` | 39.88 ± 0.26 | 39.73 ± 0.31 | **78.30 ± 0.56** |
| `Attn_SmallBal_Adam` ($10^{-3}, \mathrm{wd}=0$) | $1.56\times 10^{-9}$ | `(0.22, 0.26, 0.20, 0.36)` | 40.12 ± 0.31 | 40.22 ± 0.36 | **78.70 ± 0.54** |

**Key findings:**
1. **Weight decay collapses excess fiber rank across all 4 attention projections and drops joint $\hat\lambda_{\mathrm{ALL}}$ by $6.4\text{--}7.2$ points:** Standard `LoRA + Adam` remains stuck at high excess fiber norm (`1.31–2.37`) and $\hat\lambda_{\mathrm{ALL}} = 86.27 \pm 0.40$, whereas `AdamW` ($79.89$), `SGD+WD` ($79.04$), `Subcrit q=2` ($78.30$), and `SmallBal` ($78.70$) all converge onto the exact same subcritical plateau ($\approx 39.5\text{--}40.5$ per pair, matching the finite-$\beta$ excess-$(8,8,8)$ plateau $39.52$ from Part 2a!).
2. **Channel asymmetry confirms the $V{+}O$ vs. $Q{+}K$ coupling theory in training:** Under `Attn_LoRA_Adam`, $\hat\lambda_{V+O} = 43.78 \pm 0.22$ clearly breaks **above** the $42.0$ asymptotic plateau (partial supercritical rise moderated by random $[A_V; A_O W_V]$ overlap and gauge-coupled nonzero $B_{\mathrm{in}}$), while $\hat\lambda_{Q+K} = 42.15 \pm 0.15$ stays right at the $42.0$ flat gauge ceiling (its $+2.4$ gap above `Subcrit` $39.73$ reflects the same finite-$\beta$ singular-vs-regular offset seen in Part 3a at $q=0$ vs. $q=4$, rather than a supercritical parabola).

---

## 4. Reproducibility & Artifacts
- Code: `experiments/lora_nonlinear_fiber.py` (fixed deterministic seeds, float64 PyTorch CPU, preconditioned MALA + split-half convergence diagnostics).
- Raw data: `experiments/results/lora_nonlinear_fiber.json`.
- Summary figure: `experiments/results/lora_nonlinear_fiber.png`.
