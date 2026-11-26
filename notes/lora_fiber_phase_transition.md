# The LoRA / Reduced-Rank Fiber Phase Transition

> **Epistemic status:**
> - **Schur-complement algebraic identity & two-sided loss comparison (`schur_block_right`, `schur_loss_lower`, `schur_loss_upper`):** `[LC]` in `lean/EatYourFiber/LoRASchur.lean`.
> - **Measure-theoretic Laplace Schur-complement reduction (`lora_schur_fiber_reduction`) & unconditional standard LoRA initialization pair (`lora_standard_init_exact_pair`):** `[LC]` in `lean/EatYourFiber/LoRASchurLaplace.lean` (audited via `lean/Audit.lean`).
> - **Algebraic phase transition identities (`lora_plateau_lambda/mult`, `lora_parabola_lambda/mult`):** `[LC]` in `lean/EatYourFiber/LoRASchur.lean`.
> - **Aoyagi–Watanabe (2005) RLCT theorem for rank-0 `‖B₂ Δ₂₂‖_F^2`:** cited / `[paper]` (Lean formalization next).
> - **PyTorch MALA verification (`rank0_8x8x8`, `lora_16x16_r2_H10`):** `[empirical]` in `experiments/lora_fiber_sgld.py`.

## 1. Setup

Consider a 2-layer linear network / LoRA adapter
$$f(B, A) = B A, \qquad B \in \mathbb{R}^{M \times H},\quad A \in \mathbb{R}^{H \times N},$$
with isotropic input covariance so the population squared-error loss for a rank-$r^*$ target $M^* \in \mathbb{R}^{M \times N}$ is
$$K(B, A) = \|B A - M^*\|_F^2.$$
Write the excess dimensions as $m = M - r^*$, $n = N - r^*$, and $k = H - r^*$.

After rotating to the SVD basis of $M^*$, a realization on the zero-loss fiber $\{B A = M^*\}$ with $r^*$ active units and $k$ inactive units has residual blocks $B_{\mathrm{in}} \in \mathbb{R}^{m \times k}$, $A_{\mathrm{in}} \in \mathbb{R}^{k \times n}$ satisfying $B_{\mathrm{in}} A_{\mathrm{in}} = 0$.

## 2. Exact Schur-Complement Reduction (`[paper]`)

Fix $B_0 = 0 \in \mathbb{R}^{m \times k}$ and $A_0 \in \mathbb{R}^{k \times n}$ of rank $q \in \{0, \dots, \min(k, n)\}$:
- $q = 0$: **fully dead** excess units ($B_0 = 0, A_0 = 0$).
- $q = k$: **standard LoRA initialization** ($B_0 = 0$, $A_0$ full row rank $k$).

In an orthogonal basis where $A_0 = \begin{pmatrix} I_q & 0 \\ 0 & 0 \end{pmatrix}$, perturb
$$B = \begin{pmatrix} B_1 & B_2 \end{pmatrix}, \qquad A = \begin{pmatrix} I_q + \Delta_{11} & \Delta_{12} \\ \Delta_{21} & \Delta_{22} \end{pmatrix}.$$
Then $B A = \begin{pmatrix} M_L & M_R \end{pmatrix}$ with
$$M_L = B_1(I_q + \Delta_{11}) + B_2 \Delta_{21}, \qquad M_R = B_1 \Delta_{12} + B_2 \Delta_{22}.$$
On a neighborhood $\|\Delta_{11}\| < \tfrac12$, $\|\Delta_{12}\| < \tfrac14$, define the smooth local diffeomorphism
$$\tilde B_1 := B_1(I_q + \Delta_{11}) + B_2 \Delta_{21}, \qquad \tilde\Delta_{22} := \Delta_{22} - \Delta_{21}(I_q + \Delta_{11})^{-1}\Delta_{12},$$
leaving $(B_2, \Delta_{11}, \Delta_{12}, \Delta_{21})$ unchanged. Its Jacobian determinant is $\det(I_q + \Delta_{11})^m$, bounded above and away from zero locally. In the new coordinates,
$$M_L = \tilde B_1, \qquad M_R = \tilde B_1 (I_q + \Delta_{11})^{-1}\Delta_{12} + B_2 \tilde\Delta_{22}.$$
Because $\|(I_q + \Delta_{11})^{-1}\Delta_{12}\| \le \tfrac12$,
$$\tfrac14\bigl(\|\tilde B_1\|_F^2 + \|B_2 \tilde\Delta_{22}\|_F^2\bigr) \le \|BA\|_F^2 \le 4\bigl(\|\tilde B_1\|_F^2 + \|B_2 \tilde\Delta_{22}\|_F^2\bigr).$$
Therefore:
1. $\Delta_{11}, \Delta_{12}, \Delta_{21}$ decouple as flat coordinates of finite local volume.
2. $\tilde B_1 \in \mathbb{R}^{m \times q}$ contributes a regular Gaussian factor of exact learning coefficient $mq/2$ (multiplicity $1$).
3. $(B_2, \tilde\Delta_{22}) \in \mathbb{R}^{m \times (k-q)} \times \mathbb{R}^{(k-q) \times (n-q)}$ is a rank-$0$ reduced-rank regression problem of size $(m, n - q, k - q)$.

Hence the local learning pair at $(0, A_0)$ of rank $q$ is
$$\lambda(q) = \frac{mq}{2} + \lambda_{\mathrm{AW}}(m, n - q, k - q), \qquad m_{\log}(q) = m_{\mathrm{AW}}(m, n - q, k - q).$$

## 3. The Phase Transition (`[paper]`)

Recall Aoyagi–Watanabe (2005) Theorem 1 for $\|BA\|_F^2$ with $B \in \mathbb{R}^{m \times k}, A \in \mathbb{R}^{k \times n}$:
- If $k \ge m + n$: $(\lambda, m_{\log}) = (mn/2, 1)$.
- If $m \ge k + n$: $(\lambda, m_{\log}) = (kn/2, 1)$.
- If $n \ge k + m$: $(\lambda, m_{\log}) = (km/2, 1)$.
- Otherwise (strict triangle inequality):
  $$\lambda_{\mathrm{AW}}^{\mathrm{even}}(m, n, k) = \frac{2(mn + nk + km) - m^2 - n^2 - k^2}{8},$$
  with $(\lambda_{\mathrm{AW}}, m_{\log}) = (\lambda_{\mathrm{AW}}^{\mathrm{even}}, 1)$ if $m+n+k$ is even and $(\lambda_{\mathrm{AW}}^{\mathrm{even}} + \tfrac18, 2)$ if $m+n+k$ is odd.

Assuming $(m, n, k)$ lies in the interior triangle ($m + n > k$, $m + k > n$, $n + k > m$), define the critical rank
$$q_c := \frac{n + k - m}{2}.$$
Because $(m, n - q, k - q)$ exits the interior triangle into the $m \ge (n-q)+(k-q)$ regime precisely at $q = q_c$, we obtain:

- **Subcritical plateau ($q \le q_c$):**
  $$\lambda(q) = \lambda_{\mathrm{AW}}(m, n, k) \quad \text{(constant in } q\text{!)},$$
  with $m_{\log}(q) = 2$ if $m+n+k$ is odd and $1$ if even.
- **Supercritical parabola ($q > q_c$):**
  $$\lambda(q) = \lambda_{\mathrm{AW}}^{\mathrm{even}}(m, n, k) + \frac{(q - q_c)^2}{2}, \qquad m_{\log}(q) = 1.$$

At standard LoRA initialization ($q = k$, assuming $k \le n$), the excess learning coefficient over the global minimum (when $m+n+k$ is even) is
$$\Delta\lambda = \lambda(k) - \lambda(0) = \frac{(m + k - n)^2}{8}.$$
For $m = n = k$ (square rank-0 LoRA), $q_c = k/2$ and $\Delta\lambda = k^2 / 8$—scaling quadratically in the LoRA rank $k$.

## 4. Empirical Verification (`[empirical]`)

Running vectorized Metropolis-Adjusted Langevin Algorithm (MALA, 48 chains, $\beta = 10^5$, localization $\gamma = 20$, 6,000 burn-in + 8,000 sample steps; see `experiments/lora_fiber_sgld.py`) on two configurations reproduces both the subcritical flat plateau ($q \le q_c$) and the exact supercritical parabola $\frac{1}{2}(q - q_c)^2$ ($q > q_c$):

### `rank0_8x8x8` ($M=N=H=8, r^*=0$, $q_c = 4$)

| $q$ | Theory $\lambda(q)$ | MALA $\hat\lambda \pm \mathrm{SE}$ | Error |
|---:|---:|---:|---:|
| 0 | 24.00 | 23.41 ± 0.04 | -0.59 |
| 1 | 24.00 | 22.93 ± 0.11 | -1.07 |
| 2 | 24.00 | 23.13 ± 0.10 | -0.87 |
| 3 | 24.00 | 23.20 ± 0.11 | -0.80 |
| 4 | 24.00 | 23.68 ± 0.10 | -0.32 |
| 5 | 24.50 | 24.70 ± 0.09 | +0.20 |
| 6 | 26.00 | 26.10 ± 0.05 | +0.10 |
| 7 | 28.50 | 28.54 ± 0.03 | +0.04 |
| 8 | 32.00 | 32.01 ± 0.02 | +0.01 |

### `lora_16x16_r2_H10` ($M=N=16, H=10, r^*=2$, excess $(14,14,8)$, $q_c = 4$)

| $q$ | Theory $\lambda(q)$ | MALA $\hat\lambda \pm \mathrm{SE}$ | Error |
|---:|---:|---:|---:|
| 0 | 78.00 | 75.68 ± 0.19 | -2.32 |
| 1 | 78.00 | 75.92 ± 0.26 | -2.08 |
| 2 | 78.00 | 76.32 ± 0.22 | -1.68 |
| 3 | 78.00 | 76.32 ± 0.20 | -1.68 |
| 4 | 78.00 | 77.62 ± 0.17 | -0.38 |
| 5 | 78.50 | 78.62 ± 0.17 | +0.12 |
| 6 | 80.00 | 80.51 ± 0.14 | +0.51 |
| 7 | 82.50 | 82.74 ± 0.11 | +0.24 |
| 8 | 86.00 | 85.96 ± 0.05 | -0.04 |

See `experiments/results/lora_fiber_phase_transition.png` for the plotted curves.

