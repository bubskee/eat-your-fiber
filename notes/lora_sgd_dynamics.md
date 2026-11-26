# Gradient-Based Training Dynamics on the LoRA Fiber

> **Epistemic status:**
> - **Phase transition baseline (`lora_16x16_r2_H10`, $\lambda_{\mathrm{AW}} = 78$, $\lambda(q=8) = 86$, $q_c = 4$):** `[LC]` Schur reduction (`lean/EatYourFiber/LoRASchur.lean`, `LoRASchurLaplace.lean`) and `[LC]` unconditional supercritical parabola for all $q \ge q_c$ (`lean/EatYourFiber/AoyagiWatanabeBoundary.lean`, audited via `lean/Audit.lean`); `[paper]` Aoyagi–Watanabe interior triangle for $q < q_c$; `[empirical]` MALA curve in `experiments/lora_fiber_sgld.py`.
> - **SGD / Adam / AdamW fiber landing & trajectory experiments (`full_batch`, `minibatch`):** `[empirical]` in `experiments/lora_sgd_dynamics.py` (results in `experiments/results/lora_sgd_dynamics.json` and `experiments/results/lora_sgd_dynamics.png`).
> - **Subspace invariance mechanism ($U_\perp^\top B(t) = 0$ for standard LoRA init under GD/SGD):** elementary gradient algebra (`[paper]`).

## 1. Setup and Inactive Subspace Projection

We test where gradient-based optimization lands on the zero-loss fiber $\{B A = M^*\}$ for the `lora_16x16_r2_H10` configuration from `notes/lora_fiber_phase_transition.md`:
$$M = 16,\quad N = 16,\quad H = 10,\quad r^* = 2 \quad \Longrightarrow \quad (m, n, k) = (14, 14, 8),\quad q_c = 4,$$
with global minimum $\lambda_{\mathrm{AW}} = 78.00$ (MALA $q=0$ baseline $75.68 \pm 0.19$ at $\beta = 10^5, \gamma = 20$) and standard LoRA initialization $\lambda(k=8) = 86.00$ (MALA $q=8$ baseline $85.96 \pm 0.05$). The rank-$2$ target $M^* = U_* \operatorname{diag}(2.0, 1.5) V_*^\top$ uses random orthogonal factors $U, V$.

At any checkpoint $(B, A)$, we rotate into the SVD basis of $M^*$ and extract the inactive $m \times k$ and $k \times n$ residual blocks:
$$A_{\mathrm{in}} := Q_{\ker(U_*^\top B)}^\top (A V_\perp) \in \mathbb{R}^{k \times n}, \qquad B_{\mathrm{in}} := (U_\perp^\top B)\, Q_{\operatorname{col}(A V_*)^\perp} \in \mathbb{R}^{m \times k}.$$
When $B A = M^*$, $U_*^\top B A V_\perp = 0$ and $U_\perp^\top B A V_* = 0$, so $A_{\mathrm{in}}$ and $B_{\mathrm{in}}$ isolate the exact $k$-dimensional inactive fiber coordinates. We evaluate $\hat\lambda$ using the vectorized MALA estimator (`mala_estimate_llc`, 48 chains, $\beta = 10^5$, $\gamma = 20$, 6,000 burn-in + 8,000 sample steps) from `experiments/lora_fiber_sgld.py`.

## 2. Full-Batch Results: Initialization vs. Weight Decay (`[empirical]`)

All runs below optimize $K(B, A) = \|B A - M^*\|_F^2$ (`experiments/lora_sgd_dynamics.py`):

| Run | Init | Optimizer | Steps | Final $K(B, A)$ | $\|B_{\mathrm{in}}\|_F$ | $\|A_{\mathrm{in}}\|_F$ | $\operatorname{rank}(A_{\mathrm{in}} > 0.05)$ | $\hat\lambda \pm \mathrm{SE}$ | Regime |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `InitA_GD` | LoRA ($B{=}0, A{\sim}\mathcal{N}(0, 0.5^2)$) | GD ($\mathrm{wd}=0$) | 6,000 | $5.56 \times 10^{-30}$ | $5.02 \times 10^{-16}$ | $5.249$ | $8$ | $85.93 \pm 0.07$ | Supercritical ($q=8$) |
| `InitA_Adam` | LoRA ($B{=}0, A{\sim}\mathcal{N}(0, 0.5^2)$) | Adam ($\mathrm{wd}=0$, cosine lr) | 6,000 | $3.08 \times 10^{-32}$ | $2.24 \times 10^{-16}$ | $5.006$ | $8$ | $85.96 \pm 0.12$ | Supercritical ($q=8$) |
| `InitB_GD` | Small balanced ($\sigma = 10^{-3}$) | GD ($\mathrm{wd}=0$) | 20,000 | $1.28 \times 10^{-9}$ | $1.12 \times 10^{-2}$ | $0.009$ | $0$ | $76.00 \pm 0.16$ | Subcritical plateau ($q=0$) |
| `InitB_Adam` | Small balanced ($\sigma = 10^{-3}$) | Adam ($\mathrm{wd}=0$, cosine lr) | 8,000 | $5.46 \times 10^{-19}$ | $2.35 \times 10^{-1}$ | $0.427$ | $3$ | $76.32 \pm 0.17$ | Subcritical plateau ($q \le q_c$) |
| `InitC_AdamW` | LoRA ($B{=}0, A{\sim}\mathcal{N}(0, 0.5^2)$) | AdamW ($\mathrm{wd}=10^{-2}$, cosine lr) | 50,000 | $2.57 \times 10^{-15}$ | $9.15 \times 10^{-3}$ | $0.095$ | $2$ | $75.87 \pm 0.18$ | Subcritical plateau ($q \le q_c$) |
| `InitC_SGD_WD` | LoRA ($B{=}0, A{\sim}\mathcal{N}(0, 0.5^2)$) | SGD+WD ($\mathrm{wd}=10^{-2} \to 0$ at $45\text{k}$) | 50,000 | $1.09 \times 10^{-30}$ | $1.63 \times 10^{-14}$ | $7.18 \times 10^{-6}$ | $0$ | $75.83 \pm 0.16$ | Subcritical plateau ($q=0$) |

### Key Observations

1. **Plain GD and Adam from standard LoRA init stay pinned at the supercritical top ($\hat\lambda \approx 86$):**
   Because $B(0) = 0$ implies $U_\perp^\top B(0) = 0$, the gradient $\nabla_B K = 2(BA - M^*)A^\top$ has zero projection onto $U_\perp$, preserving $B_{\mathrm{in}}(t) = 0$ identically. Once $BA = M^*$, the inactive rows $A_{\mathrm{in}}$ lie in $\ker(B_{\mathrm{act}})$ and receive zero gradient, leaving all $8$ singular values large ($\sigma(A_{\mathrm{in}}) \in [0.564, 2.970]$ for GD, $[0.335, 2.870]$ for Adam). Consequently, $\hat\lambda$ remains at $85.93 \pm 0.07$ (GD) and $85.96 \pm 0.12$ (Adam), matching the theoretical $q=8$ value $\lambda(8) = 86.00$ ($\Delta\lambda = +8$ above $\lambda_{\mathrm{AW}}$).

2. **Small balanced initialization stays on the subcritical plateau ($\hat\lambda \approx 76$, matching $q=0$ MALA baseline):**
   Under `InitB_GD`, inactive singular values remain $\le 0.0043$ and $\hat\lambda = 76.00 \pm 0.16$ (within $0.32$ of the exact $q=0$ MALA reference $75.68 \pm 0.19$). Notably, `InitB_Adam` amplifies tiny early cross-gradients via adaptive normalization into 3 nonzero inactive modes ($\sigma_1 = 0.364, \sigma_2 = 0.174, \sigma_3 = 0.134$), yet still lands on the subcritical plateau ($\hat\lambda = 76.32 \pm 0.17$) because $q = 3 \le q_c = 4$.

3. **Weight decay drives standard LoRA down the $(q - q_c)_+^2/2$ parabola to the global minimum:**
   In `InitC_AdamW` and `InitC_SGD_WD`, $B A \approx M^*$ is reached early while weight decay exponentially contracts $A_{\mathrm{in}}$ along the fiber:
   - `InitC_AdamW`: $\hat\lambda(t)$ drops from $83.29 \pm 0.17$ ($t=2\text{k}$, $\sigma_1=2.097$) to $75.92 \pm 0.16$ ($t=20\text{k}$, $\sigma_1=0.223$) and $75.87 \pm 0.18$ ($t=50\text{k}$, $\sigma_1=0.054$), while cosine lr decay brings $K(B, A)$ to $2.57 \times 10^{-15}$ without needing a separate polish stage.
   - `InitC_SGD_WD`: $\hat\lambda(t)$ drops from $85.14 \pm 0.16$ ($t=2\text{k}$) to $75.47 \pm 0.12$ ($t=10\text{k}$) and $75.83 \pm 0.16$ ($t=50\text{k}$). Note that constant coupled L2 ($\mathrm{wd}=10^{-2}$) holds $K(B, A)$ at $5.00 \times 10^{-5}$ (active shrinkage bias) until $\mathrm{wd}$ is annealed to $0$ at step 45,000, after which $K$ snaps to $1.09 \times 10^{-30}$ with $\|A_{\mathrm{in}}\|_F = 7.18 \times 10^{-6}$.
   - Notice that at finite MALA localization $\gamma = 20$, the localization radius $\gamma^{-1/2} \approx 0.224$ marks where inactive singular values cross over into the subcritical plateau: once $\sigma_i(A_{\mathrm{in}}) \lesssim \gamma^{-1/2}$ for more than $k - q_c = 4$ modes, $\hat\lambda$ settles onto the plateau.

## 3. Finite-Sample Mini-Batch SGD Check (`[empirical]`)

We also train on $S = 64$ Gaussian samples $X \in \mathbb{R}^{16 \times 64}$ with batch size $16$ minimizing $\frac{1}{16}\|B A X_b - Y_b\|_F^2$ from standard LoRA initialization (`Init A`):

| Run | Targets | Optimizer | Epochs | Pop. Loss $\|BA-M^*\|_F^2$ | $\|A_{\mathrm{in}}\|_F$ | $\operatorname{rank}(A_{\mathrm{in}} > 0.05)$ | $\hat\lambda \pm \mathrm{SE}$ |
|---|---|---|---:|---:|---:|---:|---:|
| `MB_SGD_clean` | $Y = M^* X$ | Mini-batch SGD ($\mathrm{wd}=0$) | 1,500 | $1.01 \times 10^{-31}$ | $5.155$ | $8$ | $85.83 \pm 0.09$ |
| `MB_Adam_clean` | $Y = M^* X$ | Mini-batch Adam ($\mathrm{wd}=0$, cosine lr) | 1,500 | $7.74 \times 10^{-32}$ | $4.454$ | $8$ | $85.75 \pm 0.10$ |
| `MB_SGD_noisy` | $Y = M^* X + 0.05\Xi$ | Mini-batch SGD ($\mathrm{wd}=0$, cosine lr) | 2,000 | $1.03 \times 10^{-2}$ | $4.908$ | $8$ | $86.03 \pm 0.07$ |
| `MB_AdamW_clean` | $Y = M^* X$ | Mini-batch AdamW ($\mathrm{wd}=10^{-2}$, cosine lr) | 5,000 | $1.82 \times 10^{-11}$ | $0.202$ | $3$ | $75.70 \pm 0.21$ |

Because $U_\perp^\top B(0) = 0$ makes $U_\perp^\top \nabla_B L_{\text{batch}} = 0$ on every mini-batch for noiseless targets, **mini-batch SGD and Adam without weight decay remain stuck at the supercritical edge $q = k = 8$ ($\hat\lambda \approx 85.8$)**, and even label noise (`MB_SGD_noisy`, $\sigma_\xi = 0.05$) does not relax $A_{\mathrm{in}}$ below $q_c$ within 2,000 epochs ($\hat\lambda = 86.03 \pm 0.07$). Only explicit weight decay (`MB_AdamW_clean`) collapses the inactive modes below $q_c$ ($\sigma_{\max}(A_{\mathrm{in}}) = 0.176$, $\operatorname{rank}_{>0.05} = 3 \le q_c = 4$) and drops $\hat\lambda$ to $75.70 \pm 0.21$ on the global-minimum plateau.

See `experiments/results/lora_sgd_dynamics.png` for the 4-panel visualization and `experiments/results/lora_sgd_dynamics.json` for full checkpoint traces and singular-value spectra.
