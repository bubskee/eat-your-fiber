# Gradient-Based Training Dynamics on the LoRA Fiber

> **Epistemic status:**
> - **Phase-transition baseline (`lora_16x16_r2_H10`, $\lambda_{\mathrm{AW}} = 78$, $\lambda(q=8) = 86$, $q_c = 4$):**
>   - `[LC]` across the whole fiber (the `q ≤ q_c` plateau and the `q > q_c` parabola) in `lean/EatYourFiber/LoRASchur.lean`, `LoRASchurLaplace.lean`, `AoyagiWatanabeBoundary.lean` and `AoyagiWatanabe.lean` (audited via `lean/Audit.lean`).
>   - `[empirical]` MALA curve in `experiments/lora_fiber_sgld.py`.
> - **SGD / Adam / AdamW landing and trajectory experiments (`full_batch`, `minibatch`):** `[empirical]` in `experiments/lora_sgd_dynamics.py`; results in `experiments/results/lora_sgd_dynamics.json` and `experiments/results/lora_sgd_dynamics.png`.
> - **Subspace invariance ($U_\perp^\top B(t) = 0$ for standard LoRA init under GD/SGD):** elementary gradient algebra (`[paper]`).

## 1. Setup and Inactive Subspace Projection

- **Question:** where does gradient-based optimization land on the zero-loss fiber $\{B A = M^*\}$?
- **Configuration:** `lora_16x16_r2_H10` from `notes/lora_fiber_phase_transition.md`:
$$M = 16,\quad N = 16,\quad H = 10,\quad r^* = 2 \quad \Longrightarrow \quad (m, n, k) = (14, 14, 8),\quad q_c = 4,$$
- **Target:** rank-$2$, $M^* = U_* \operatorname{diag}(2.0, 1.5) V_*^\top$ with random orthogonal factors $U, V$.
- **Reference values:**
  - Global minimum: $\lambda_{\mathrm{AW}} = 78.00$; MALA $q=0$ baseline $75.68 \pm 0.19$ at $\beta = 10^5, \gamma = 20$.
  - Standard LoRA initialization: $\lambda(k=8) = 86.00$; MALA $q=8$ baseline $85.96 \pm 0.05$.
- **Inactive blocks:** at each checkpoint $(B, A)$, rotate into the SVD basis of $M^*$ and take the inactive $m \times k$ and $k \times n$ residual blocks:
$$A_{\mathrm{in}} := Q_{\ker(U_*^\top B)}^\top (A V_\perp) \in \mathbb{R}^{k \times n}, \qquad B_{\mathrm{in}} := (U_\perp^\top B)\, Q_{\operatorname{col}(A V_*)^\perp} \in \mathbb{R}^{m \times k}.$$
  - When $B A = M^*$: $U_*^\top B A V_\perp = 0$ and $U_\perp^\top B A V_* = 0$.
  - So $A_{\mathrm{in}}$ and $B_{\mathrm{in}}$ isolate the exact $k$-dimensional inactive fiber coordinates.
- **Estimator:** vectorized MALA (`mala_estimate_llc` in `experiments/lora_fiber_sgld.py`), 48 chains, $\beta = 10^5$, $\gamma = 20$, 6,000 burn-in + 8,000 sample steps.

## 2. Full-Batch Runs: Initialization and Weight Decay (`[empirical]`)

All runs optimize $K(B, A) = \|B A - M^*\|_F^2$ (`experiments/lora_sgd_dynamics.py`).

| Run | Init | Optimizer | Steps | Final $K(B, A)$ | $\|B_{\mathrm{in}}\|_F$ | $\|A_{\mathrm{in}}\|_F$ | $\operatorname{rank}(A_{\mathrm{in}} > 0.05)$ | $\hat\lambda \pm \mathrm{SE}$ | Regime |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `InitA_GD` | LoRA ($B{=}0, A{\sim}\mathcal{N}(0, 0.5^2)$) | GD ($\mathrm{wd}=0$) | 6,000 | $5.56 \times 10^{-30}$ | $5.02 \times 10^{-16}$ | $5.249$ | $8$ | $85.93 \pm 0.07$ | Supercritical ($q=8$) |
| `InitA_Adam` | LoRA ($B{=}0, A{\sim}\mathcal{N}(0, 0.5^2)$) | Adam ($\mathrm{wd}=0$, cosine lr) | 6,000 | $3.08 \times 10^{-32}$ | $2.24 \times 10^{-16}$ | $5.006$ | $8$ | $85.96 \pm 0.12$ | Supercritical ($q=8$) |
| `InitB_GD` | Small balanced ($\sigma = 10^{-3}$) | GD ($\mathrm{wd}=0$) | 20,000 | $1.28 \times 10^{-9}$ | $1.12 \times 10^{-2}$ | $0.009$ | $0$ | $76.00 \pm 0.16$ | Subcritical plateau ($q=0$) |
| `InitB_Adam` | Small balanced ($\sigma = 10^{-3}$) | Adam ($\mathrm{wd}=0$, cosine lr) | 8,000 | $5.46 \times 10^{-19}$ | $2.35 \times 10^{-1}$ | $0.427$ | $3$ | $76.32 \pm 0.17$ | Subcritical plateau ($q \le q_c$) |
| `InitC_AdamW` | LoRA ($B{=}0, A{\sim}\mathcal{N}(0, 0.5^2)$) | AdamW ($\mathrm{wd}=10^{-2}$, cosine lr) | 50,000 | $2.57 \times 10^{-15}$ | $9.15 \times 10^{-3}$ | $0.095$ | $2$ | $75.87 \pm 0.18$ | Subcritical plateau ($q \le q_c$) |
| `InitC_SGD_WD` | LoRA ($B{=}0, A{\sim}\mathcal{N}(0, 0.5^2)$) | SGD+WD ($\mathrm{wd}=10^{-2} \to 0$ at $45\text{k}$) | 50,000 | $1.09 \times 10^{-30}$ | $1.63 \times 10^{-14}$ | $7.18 \times 10^{-6}$ | $0$ | $75.83 \pm 0.16$ | Subcritical plateau ($q=0$) |

### Observations

1. **Standard LoRA init, plain GD or Adam: stays at the supercritical top ($\hat\lambda \approx 86$).**
   - **Mechanism:**
     - $B(0) = 0$ gives $U_\perp^\top B(0) = 0$.
     - The gradient $\nabla_B K = 2(BA - M^*)A^\top$ then has zero projection onto $U_\perp$, so $B_{\mathrm{in}}(t) = 0$ for all $t$.
     - Once $BA = M^*$, the inactive rows $A_{\mathrm{in}}$ lie in $\ker(B_{\mathrm{act}})$ and get zero gradient.
   - **Result:** all $8$ inactive singular values stay large: $\sigma(A_{\mathrm{in}}) \in [0.564, 2.970]$ (GD), $[0.335, 2.870]$ (Adam).
   - **Estimate:** $85.93 \pm 0.07$ (GD), $85.96 \pm 0.12$ (Adam); theoretical $q=8$ value $\lambda(8) = 86.00$, i.e. $\Delta\lambda = +8$ above $\lambda_{\mathrm{AW}}$.

2. **Small balanced init: stays on the subcritical plateau ($\hat\lambda \approx 76$, matching the $q=0$ MALA baseline).**
   - **`InitB_GD`:** inactive singular values stay $\le 0.0043$; $\hat\lambda = 76.00 \pm 0.16$, within $0.32$ of the $q=0$ MALA reference $75.68 \pm 0.19$.
   - **`InitB_Adam`:**
     - Adaptive normalization amplifies small early cross-gradients into 3 nonzero inactive modes ($\sigma_1 = 0.364, \sigma_2 = 0.174, \sigma_3 = 0.134$).
     - It still lands on the plateau ($\hat\lambda = 76.32 \pm 0.17$), since $q = 3 \le q_c = 4$.

3. **Weight decay moves standard LoRA down the $(q - q_c)_+^2/2$ parabola to the global minimum.**
   - The parabola is the even-parity form, as in all examples here.
   - In both runs $B A \approx M^*$ is reached early; weight decay then contracts $A_{\mathrm{in}}$ exponentially along the fiber.
   - **`InitC_AdamW`:**
     - $\hat\lambda(t)$: $83.29 \pm 0.17$ ($t=2\text{k}$, $\sigma_1=2.097$) → $75.92 \pm 0.16$ ($t=20\text{k}$, $\sigma_1=0.223$) → $75.87 \pm 0.18$ ($t=50\text{k}$, $\sigma_1=0.054$).
     - Cosine lr decay brings $K(B, A)$ to $2.57 \times 10^{-15}$ with no separate polish stage.
   - **`InitC_SGD_WD`:**
     - $\hat\lambda(t)$: $85.14 \pm 0.16$ ($t=2\text{k}$) → $75.47 \pm 0.12$ ($t=10\text{k}$) → $75.83 \pm 0.16$ ($t=50\text{k}$).
     - Constant coupled L2 ($\mathrm{wd}=10^{-2}$) holds $K(B, A)$ at $5.00 \times 10^{-5}$ (active shrinkage bias).
     - After $\mathrm{wd}$ is annealed to $0$ at step 45,000, $K$ drops to $1.09 \times 10^{-30}$ with $\|A_{\mathrm{in}}\|_F = 7.18 \times 10^{-6}$.
   - **Crossover scale:** at finite MALA localization $\gamma = 20$, the radius $\gamma^{-1/2} \approx 0.224$ marks where inactive singular values cross into the plateau. Once $\sigma_i(A_{\mathrm{in}}) \lesssim \gamma^{-1/2}$ for more than $k - q_c = 4$ modes, $\hat\lambda$ settles onto the plateau.

## 3. Mini-Batch SGD on Finite Samples (`[empirical]`)

- **Setup:** $S = 64$ Gaussian samples $X \in \mathbb{R}^{16 \times 64}$, batch size $16$, loss $\frac{1}{16}\|B A X_b - Y_b\|_F^2$, standard LoRA initialization (`Init A`).

| Run | Targets | Optimizer | Epochs | Pop. Loss $\|BA-M^*\|_F^2$ | $\|A_{\mathrm{in}}\|_F$ | $\operatorname{rank}(A_{\mathrm{in}} > 0.05)$ | $\hat\lambda \pm \mathrm{SE}$ |
|---|---|---|---:|---:|---:|---:|---:|
| `MB_SGD_clean` | $Y = M^* X$ | Mini-batch SGD ($\mathrm{wd}=0$) | 1,500 | $1.01 \times 10^{-31}$ | $5.155$ | $8$ | $85.83 \pm 0.09$ |
| `MB_Adam_clean` | $Y = M^* X$ | Mini-batch Adam ($\mathrm{wd}=0$, cosine lr) | 1,500 | $7.74 \times 10^{-32}$ | $4.454$ | $8$ | $85.75 \pm 0.10$ |
| `MB_SGD_noisy` | $Y = M^* X + 0.05\Xi$ | Mini-batch SGD ($\mathrm{wd}=0$, cosine lr) | 2,000 | $1.03 \times 10^{-2}$ | $4.908$ | $8$ | $86.03 \pm 0.07$ |
| `MB_AdamW_clean` | $Y = M^* X$ | Mini-batch AdamW ($\mathrm{wd}=10^{-2}$, cosine lr) | 5,000 | $1.82 \times 10^{-11}$ | $0.202$ | $3$ | $75.70 \pm 0.21$ |

- **Mechanism:** for noiseless targets, $U_\perp^\top B(0) = 0$ gives $U_\perp^\top \nabla_B L_{\text{batch}} = 0$ on every mini-batch.
- **No weight decay:** mini-batch SGD and Adam stay at the supercritical edge $q = k = 8$ ($\hat\lambda \approx 85.8$).
- **Label noise (`MB_SGD_noisy`, $\sigma_\xi = 0.05$):** does not relax $A_{\mathrm{in}}$ below $q_c$ within 2,000 epochs ($\hat\lambda = 86.03 \pm 0.07$).
- **Weight decay (`MB_AdamW_clean`):** the only run here that collapses the inactive modes below $q_c$.
  - $\sigma_{\max}(A_{\mathrm{in}}) = 0.176$, $\operatorname{rank}_{>0.05} = 3 \le q_c = 4$.
  - $\hat\lambda$ drops to $75.70 \pm 0.21$, on the global-minimum plateau.
- **Files:** `experiments/results/lora_sgd_dynamics.png` (4-panel figure) and `experiments/results/lora_sgd_dynamics.json` (full checkpoint traces and singular-value spectra).
