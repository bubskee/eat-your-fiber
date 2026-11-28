# Phase Transition Along the Zero-Loss Fiber of LoRA and Reduced-Rank Regression

## Status of Each Claim

- `[LC]` in `lean/EatYourFiber/LoRASchur.lean`:
  - Schur-complement algebraic identity, two-sided loss comparison, and phase-transition formulas.
  - `schur_block_right`, `schur_loss_lower`, `schur_loss_upper`, `lora_plateau_lambda/mult`, `lora_parabola_lambda/mult`.
- `[LC]` in `lean/EatYourFiber/LoRASchurLaplace.lean`:
  - Measure-theoretic Laplace Schur-complement reduction (`lora_schur_fiber_reduction`).
  - Standard LoRA initialization pair (`lora_standard_init_exact_pair`).
- `[LC]` in `lean/EatYourFiber/AoyagiWatanabeBoundary.lean`:
  - Independent elementary proof of the Aoyagi–Watanabe boundary regimes (`rank0_hasLaplacePair_of_le_m`, `rank0_hasLaplacePair_of_le_n`).
  - Supercritical LoRA parabola (`lora_supercritical_parabola_exact_pair`).
- `[LC]` in `lean/EatYourFiber/AoyagiWatanabe.lean`:
  - Full Aoyagi–Watanabe (2005) Theorem 1 across all four regimes and parities (`aoyagi_watanabe_rank0`).
  - Complete LoRA fiber exact pair (`lora_fiber_exact_pair`).
  - Unconditional subcritical plateau (`lora_subcritical_plateau_exact_pair`).
- `[LC]` in `lean/EatYourFiber/RRRFiberStratification.lean`:
  - Complete unconditional $(r^*, p, q)$ RRR zero-loss fiber stratification (`rrr_fiber_exact_pair`, `rrr_fiber_subcritical_plateau`, `rrr_fiber_A_supercritical_parabola`, `rrr_fiber_B_supercritical_parabola`, `rrr_supercritical_mutually_exclusive`).
  - Irreducible-component codimension characterization (`rrr_fiber_eq_min_component_codim`, synthesizing `MAIS-O70` / `ztreisman/arc-llc`).
- **Axioms:** all 107 audited declarations in `lean/Audit.lean` depend only on axioms in `[propext, Classical.choice, Quot.sound]`.
- `[empirical]` in `experiments/lora_fiber_sgld.py`: PyTorch MALA verification (`rank0_8x8x8`, `lora_16x16_r2_H10`).
- `[empirical]` in `experiments/lora_sgd_dynamics.py`: real GD / SGD / Adam / AdamW training dynamics on the fiber (see `notes/lora_sgd_dynamics.md`).
- **See also:** [transformer_fiber_surgery.md](transformer_fiber_surgery.md) (`[empirical]`).
  - Tests the full 2D $(p, q)$ stratification of §3.1 inside a real pretrained 2-layer causal Transformer (LoRA on $W_O$, next-token KL) via iso-functional fiber surgery.
  - Adds a Pre-LayerNorm symmetry-breaking check, continual-learning consequences, and a functional-vs.-fiber decomposition of late-stage $\Delta\hat\lambda$.

## 1. Setup

- **Model:** a 2-layer linear network / LoRA adapter
  $$f(B, A) = B A, \qquad B \in \mathbb{R}^{M \times H},\quad A \in \mathbb{R}^{H \times N},$$
- **Loss:** with isotropic input covariance, the population squared-error loss for a rank-$r^*$ target $M^* \in \mathbb{R}^{M \times N}$ is
  $$K(B, A) = \|B A - M^*\|_F^2.$$
- **Excess dimensions:** $m = M - r^*$, $n = N - r^*$, $k = H - r^*$.
- **Residual blocks:** after rotating to the SVD basis of $M^*$, a realization on the zero-loss fiber $\{B A = M^*\}$ with $r^*$ active units and $k$ inactive units has residual blocks $B_{\mathrm{in}} \in \mathbb{R}^{m \times k}$, $A_{\mathrm{in}} \in \mathbb{R}^{k \times n}$ with $B_{\mathrm{in}} A_{\mathrm{in}} = 0$.

## 2. Exact Schur-Complement Reduction (`[LC]`)

- **Base point:** $B_0 = 0 \in \mathbb{R}^{m \times k}$ and $A_0 \in \mathbb{R}^{k \times n}$ of rank $q \in \{0, \dots, \min(k, n)\}$.
  - $q = 0$: fully dead excess units ($B_0 = 0, A_0 = 0$).
  - $q = k$: standard LoRA initialization ($B_0 = 0$, $A_0$ full row rank $k$).
- **Perturbation:** in an orthogonal basis where $A_0 = \begin{pmatrix} I_q & 0 \\ 0 & 0 \end{pmatrix}$, set
  $$B = \begin{pmatrix} B_1 & B_2 \end{pmatrix}, \qquad A = \begin{pmatrix} I_q + \Delta_{11} & \Delta_{12} \\ \Delta_{21} & \Delta_{22} \end{pmatrix}.$$
  Then $B A = \begin{pmatrix} M_L & M_R \end{pmatrix}$ with
  $$M_L = B_1(I_q + \Delta_{11}) + B_2 \Delta_{21}, \qquad M_R = B_1 \Delta_{12} + B_2 \Delta_{22}.$$
- **Change of coordinates:** on the neighborhood $\|\Delta_{11}\| < \tfrac12$, $\|\Delta_{12}\| < \tfrac14$, define the smooth local diffeomorphism
  $$\tilde B_1 := B_1(I_q + \Delta_{11}) + B_2 \Delta_{21}, \qquad \tilde\Delta_{22} := \Delta_{22} - \Delta_{21}(I_q + \Delta_{11})^{-1}\Delta_{12},$$
  - It leaves $(B_2, \Delta_{11}, \Delta_{12}, \Delta_{21})$ unchanged.
  - Its Jacobian determinant is $\det(I_q + \Delta_{11})^m$, bounded above and away from zero locally.
  - In the new coordinates,
    $$M_L = \tilde B_1, \qquad M_R = \tilde B_1 (I_q + \Delta_{11})^{-1}\Delta_{12} + B_2 \tilde\Delta_{22}.$$
- **Two-sided bound:** since $\|(I_q + \Delta_{11})^{-1}\Delta_{12}\| \le \tfrac12$,
  $$\tfrac14\bigl(\|\tilde B_1\|_F^2 + \|B_2 \tilde\Delta_{22}\|_F^2\bigr) \le \|BA\|_F^2 \le 4\bigl(\|\tilde B_1\|_F^2 + \|B_2 \tilde\Delta_{22}\|_F^2\bigr).$$
- **Consequences:**
  1. $\Delta_{11}, \Delta_{12}, \Delta_{21}$ decouple as flat coordinates of finite local volume.
  2. $\tilde B_1 \in \mathbb{R}^{m \times q}$ contributes a regular Gaussian factor of exact learning coefficient $mq/2$ (multiplicity $1$).
  3. $(B_2, \tilde\Delta_{22}) \in \mathbb{R}^{m \times (k-q)} \times \mathbb{R}^{(k-q) \times (n-q)}$ is a rank-$0$ reduced-rank regression problem of size $(m, n - q, k - q)$.
- **Result:** the local learning pair at $(0, A_0)$ of rank $q$ is
  $$\lambda(q) = \frac{mq}{2} + \lambda_{\mathrm{AW}}(m, n - q, k - q), \qquad m_{\log}(q) = m_{\mathrm{AW}}(m, n - q, k - q).$$

## 3. The Phase Transition (`[LC]`)

- **Aoyagi–Watanabe (2005) Theorem 1** for $\|BA\|_F^2$ with $B \in \mathbb{R}^{m \times k}, A \in \mathbb{R}^{k \times n}$:
  - If $k \ge m + n$: $(\lambda, m_{\log}) = (mn/2, 1)$.
  - If $m \ge k + n$: $(\lambda, m_{\log}) = (kn/2, 1)$.
  - If $n \ge k + m$: $(\lambda, m_{\log}) = (km/2, 1)$.
  - Otherwise (strict triangle inequality):
    $$\lambda_{\mathrm{AW}}^{\mathrm{even}}(m, n, k) = \frac{2(mn + nk + km) - m^2 - n^2 - k^2}{8},$$
    with $(\lambda_{\mathrm{AW}}, m_{\log}) = (\lambda_{\mathrm{AW}}^{\mathrm{even}}, 1)$ if $m+n+k$ is even and $(\lambda_{\mathrm{AW}}^{\mathrm{even}} + \tfrac18, 2)$ if $m+n+k$ is odd.
- **Assumption:** $(m, n, k)$ lies in the interior triangle ($m + n > k$, $m + k > n$, $n + k > m$).
- **Critical rank:**
  $$q_c := \frac{n + k - m}{2}.$$
- **Reason for the transition:** $(m, n - q, k - q)$ leaves the interior triangle into the $m \ge (n-q)+(k-q)$ regime exactly at $q = q_c$. This gives two regimes:
  - **Subcritical plateau ($q \le q_c$):**
    $$\lambda(q) = \lambda_{\mathrm{AW}}(m, n, k) \quad \text{(constant in } q\text{!)},$$
    with $m_{\log}(q) = 2$ if $m+n+k$ is odd and $1$ if even.
  - **Supercritical parabola ($q > q_c$):**
    $$\lambda(q) = \lambda_{\mathrm{AW}}^{\mathrm{even}}(m, n, k) + \frac{(q - q_c)^2}{2}, \qquad m_{\log}(q) = 1.$$
- **Standard LoRA initialization** ($q = k$, assuming $k \le n$): the excess learning coefficient over the global minimum (when $m+n+k$ is even) is
  $$\Delta\lambda = \lambda(k) - \lambda(0) = \frac{(m + k - n)^2}{8}.$$
  - For $m = n = k$ (square rank-0 LoRA), $q_c = k/2$ and $\Delta\lambda = k^2 / 8$, quadratic in the LoRA rank $k$.

### 3.1 Full $(r^*, p, q)$ Stratification and Component Codimension (`[LC]`)

- **Source:** `lean/EatYourFiber/RRRFiberStratification.lean`, unconditional.
- **Scope:** all $r^*, p, q, m', n', k' \in \mathbb{N}$ and every invertible active block $D \in GL(r^*, \mathbb{R})$, with
  - $M = r^* + p + m'$, $N = r^* + q + n'$, $H = r^* + q + p + k'$;
  - $m = p + m'$, $n = q + n'$, $k = p + q + k'$.

1. **Active plus asymmetric $(p, q)$ Schur reduction (`rrr_fiber_exact_pair`, `rrr_lambda_active_plus_excess`, `[LC]`):**
   - At any canonical $(r^*, p, q)$ fiber point (`rrrBaseB_mul_rrrBaseA`), a block-triangular local diffeomorphism (`rrrTriMap`) decouples the active $r^*$ and right-inactive $q$ blocks.
   - This leaves two regular Gaussian factors plus a transposed `LoRASchurLaplace` rank-$p$ problem.
   - Exact learning pair:
     $$\lambda(r^*, p, q) = \frac{MN - m'n'}{2} + \lambda_{\mathrm{AW}}(m', n', k') = \frac{r^*(M + N - r^*)}{2} + \frac{pn + qm - pq}{2} + \lambda_{\mathrm{AW}}(m - p,\, n - q,\, k - p - q),$$
     with multiplicity $m_{\log}(r^*, p, q) = m_{\mathrm{AW}}(m - p,\, n - q,\, k - p - q)$.
2. **2D phase diagram and mutual exclusivity (`[LC]`):**
   - In the interior regime, define $p_c := \frac{k + m - n}{2}$ and $q_c := \frac{k + n - m}{2}$.
   - Since $p_c + q_c = k \ge p + q$, the supercritical regimes $p > p_c$ and $q > q_c$ are mutually exclusive (`rrr_supercritical_mutually_exclusive`).
   - **2D subcritical plateau ($p < p_c$ and $q < q_c$, `rrr_fiber_subcritical_plateau`):** $\lambda(r^*, p, q) = \lambda_{\mathrm{act}} + \lambda_{\mathrm{AW}}(m, n, k)$, $m_{\log} = m_{\mathrm{AW}}(m, n, k)$.
   - **$A$-supercritical parabola ($q \ge q_c$, `rrr_fiber_A_supercritical_parabola`):** $\lambda(r^*, p, q) = \lambda_{\mathrm{act}} + \lambda_{\mathrm{AW}}^{\mathrm{even}}(m, n, k) + \frac{1}{2}(q - q_c)^2$, $m_{\log} = 1$.
   - **$B$-supercritical parabola ($p \ge p_c$, `rrr_fiber_B_supercritical_parabola`):** $\lambda(r^*, p, q) = \lambda_{\mathrm{act}} + \lambda_{\mathrm{AW}}^{\mathrm{even}}(m, n, k) + \frac{1}{2}(p - p_c)^2$, $m_{\log} = 1$.
3. **Irreducible-component codimension (`rrr_fiber_eq_min_component_codim`, `[LC]`, synthesizing `MAIS-O70` / [`ztreisman/arc-llc`](https://github.com/ztreisman/arc-llc)):**
   - **Components:** the rank-0 excess fiber $\{B_{\mathrm{in}} A_{\mathrm{in}} = 0\}$ decomposes into irreducible components $Z_{\alpha, k - \alpha} = \{\operatorname{rank}(A_{\mathrm{in}}) \le \alpha,\; \operatorname{rank}(B_{\mathrm{in}}) \le k - \alpha\}$ with $\operatorname{codim} Z_{\alpha, k-\alpha} = m\alpha + n(k-\alpha) - \alpha(k-\alpha)$.
   - **Components through a point:** those passing through a fiber point $(p, q)$ are indexed by integers $\alpha \in [q, k - p]$.
   - **Result:** `rrr_fiber_eq_min_component_codim` certifies
     $$2\lambda_{\mathrm{exc}}(p, q) = \min_{\alpha \in [q,\, k - p] \cap \mathbb{Z}} \bigl(m\alpha + n(k - \alpha) - \alpha(k - \alpha)\bigr).$$
   - **Geometric reading:** the plateau-to-parabola transition is the window $[q, k - p]$ moving past the integer minimizer $\alpha^* \approx q_c$ of this discrete parabola as $q$ grows.
   - **Relation to `arc-llc`:**
     - `arc-llc` studies the exact $(r^*, a, b)$ local RRR RLCT formula via complex arc spaces (Smith normal form over $\mathbb{C}[[t]]$ and Docampo's formula).
     - Here the same formula has an unconditional, real-valued Lean 4 proof that uses no Hironaka resolution, complexification or arc-space machinery.
     - It is offered as a candidate solution to `MAIS-O70` in [`ztreisman/arc-llc`](https://github.com/ztreisman/arc-llc).

- **Two observations elsewhere in the repository that this explains:**
  - Single-head `Q+K` LoRA stays flat across all $q \le H$: it lies on the balanced diagonal $p = q \le H = p_c = q_c$ inside the 2D plateau (see `AttentionLoRA.lean`).
  - Small nonzero $B_{\mathrm{in}}$ generated during nonlinear Adam training (`notes/lora_nonlinear_fiber.md` §2.1) does not raise $\lambda$ above the $A$-supercritical parabola.

## 4. Empirical Verification (`[empirical]`)

- **Sampler:** vectorized Metropolis-Adjusted Langevin Algorithm (MALA), 48 chains, $\beta = 10^5$, localization $\gamma = 20$, 6,000 burn-in + 8,000 sample steps (`experiments/lora_fiber_sgld.py`).
- **Result:** on both configurations below, MALA reproduces the subcritical plateau ($q \le q_c$) and the supercritical parabola $\frac{1}{2}(q - q_c)^2$ ($q > q_c$).
- **Figure:** `experiments/results/lora_fiber_phase_transition.png`.

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

### 4.1 Finite-$\beta$ Bias and Related Work

- **Observation:** MALA sits slightly below theory for $q < q_c$ and matches theory at $q = k$.
- **Supercritical top ($q = k$):**
  - All active coordinates enter the loss quadratically (regular Gaussian directions, $m_{\log} = 1$).
  - So $\beta \mathbb{E}_\beta[K] = \lambda(k) + O(\gamma/\beta)$, which matches `32.00` / `86.00` to $\le 0.04$.
- **Singular subcritical plateau ($q < q_c$):**
  - Higher-order quartic/determinantal directions converge slowly, at $O(1 / \log(\beta/\gamma))$, with negative finite-$\beta$ bias.
  - Evidence: the $(\beta, \gamma)$ stability sweep in `notes/lora_nonlinear_fiber.md` §1.3, where $\hat\lambda(q=0)$ rises monotonically from `22.39 -> 23.53` toward `24.0` as $\beta/\gamma$ increases.
  - **Caveat:** unadjusted SGLD has a severe *upward* discretization bias on stiff supercritical modes.
- **Relation to prior SLT work:**
  - `aoyagi_watanabe_rank0` is a mechanized, elementary Lean 4 proof of Aoyagi & Watanabe (2005, *Neural Networks* 18(7): 924–933, Theorem 1).
  - It complements Aoyagi's (2024, *Neural Networks*) deep linear network RLCTs.
  - It complements the local learning coefficient framework of Lau, Furman, Wang, Murfet & Wei (2025, AISTATS; `arXiv:2308.12108`) and Furman & Lau (2024, `arXiv:2402.03698`, since merged into `arXiv:2308.12108`).
  - What it adds: the exact phase transition *along* the zero-loss fiber of a fixed function.
