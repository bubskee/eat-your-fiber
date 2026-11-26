# Order-3 Bilinear MLPs, Border-Rank Singularity, and Stretched-Exponential SLT

**Epistemic status summary:**
- **`[LC]`** (Lean 4 kernel-certified, zero `sorry`, strictly `[propext, Classical.choice, Quot.sound]` in `lean/EatYourFiber/BorderRankSLT.lean`):
  1. Bilinear MLP ↔ OpenAI `Tensor` rank/border-rank bridge (`exists_zero_loss_iff_tensorRank_le`, `geometricBorderRank_le_of_approx_loss`).
  2. Exact algebraic border-rank gap of the canonical $W$-tensor over *every* field $F$: $\mathrm{geometricBorderRank}(W) = 2 < 3 = \mathrm{tensorRank}(W)$ (`wTensor_geometricBorderRank`, `wTensor_tensorRank`, `wTensor_borderRank_lt_tensorRank`), building directly on OpenAI's vendored `BorderRank.lean` (`card_le_geometricBorderRank_of_minor`).
  3. Sharp quantitative border-rank escape law $\inf_{\|\theta\|_2 \le R} \|T(\theta) - W\|_F^2 \asymp R^{-12}$ (`wTensor_loss_ge_of_paramNormSq`, `wTensor_escape_upper`), achieved by the **centered finite-difference** optimal border-rank curve $\tilde W_\varepsilon = W + \varepsilon^2 e_1^{\otimes 3}$ (beating the textbook one-sided curve $W_\varepsilon$ by a full factor of 2 in the exponent: $R^{-12}$ vs $R^{-6}$).
  4. Exact weight-decay scaling law $\inf_\theta (\|T(\theta) - W\|_F^2 + \gamma \|\theta\|_2^2) \asymp \gamma^{6/7}$ (`wTensor_weightDecay_lower`, `wTensor_weightDecay_upper`).
  5. **Stretched-exponential Bayesian free energy** $c_1 N^{1/7} \le -\log Z_W^{(2)}(N) \le c_2 N^{1/7}$ (`wPartitionZ_two_le_stretchedExp`, `wPartitionZ_two_ge_stretchedExp`) vs. **sharp width-3 codimension-8 RLCT contrast** $-\log Z_W^{(3)}(N) \le 4 \log N + C_3$ (`wPartitionZ_three_ge_poly_sharp`, sharpening the elementary $N^{-9}$ bound `wPartitionZ_three_ge_poly` to the exact $\lambda = 4$ exponent).
- **`[empirical]`** (`experiments/bilinear_border_rank.py`, `experiments/results/bilinear_border_rank.{json,png}`):
  - Multi-start L-BFGS weight-decay scaling ($k=2$ vs $k=3$), SMC + HMC Bayesian free energy and diverging effective LLC $\hat\lambda(\beta) = \beta \mathbb{E}_\beta[L(\theta)]$, and real 1-hidden-layer quadratic MLP $f(x) = W_{\mathrm{out}}(W_{\mathrm{in}} x)^{\odot 2}$ training dynamics.

---

## 1. Why Order-3 Breaks Order-2 SLT

In **Order-2** (matrices / LoRA, `{BA = M*}`), the locus of matrices of rank $\le k$ is Zariski-closed: border rank equals rank. Consequently, a target $M^*$ is either realizable at finite weights ($\lambda < \infty$, Watanabe polynomial free energy $-\log Z(N) \sim \lambda \log N$) or strictly bounded away from zero loss ($\inf L = L^* > 0$, linear free energy $-\log Z(N) \sim N L^*$).

In **Order-3** (bilinear MLPs $T(A, B, C) = \sum_{r=1}^k a_r \otimes b_r \otimes c_r \in \mathbb{R}^{d_1 \times d_2 \times d_3}$), the rank-$\le k$ locus is **not closed**. Whenever a target tensor $T^*$ lies in the strict border-rank gap
$$\underline{\mathrm{rk}}(T^*) \le k < \mathrm{rk}(T^*),$$
the model has $\inf_{\theta \in \mathbb{R}^P} L(\theta) = 0$ while its finite zero-loss fiber is **empty**. Under a Gaussian prior (or $\ell_2$ weight decay), the posterior must escape to infinity along a border-rank degeneration tube, placing the Bayesian free energy in a **third SLT universality class** strictly between realizable Watanabe $\Theta(\log N)$ and unrealizable $\Theta(N)$: **stretched-exponential** $-\log Z(N) \asymp N^{\alpha}$ with fractional exponent $\alpha \in (0, 1)$.

---

## 2. Exact Theory on the $W$-Tensor (`[LC]`)

Consider the canonical $2 \times 2 \times 2$ $W$-tensor
$$W = e_0 \otimes e_0 \otimes e_1 + e_0 \otimes e_1 \otimes e_0 + e_1 \otimes e_0 \otimes e_0.$$

### 2.1 Algebraic gap over every field (`[LC]`)
- `wTensor_geometricBorderRank`: $\mathrm{geometricBorderRank}(W) = 2$.
  - Lower bound ($\ge 2$): direct application of OpenAI's `card_le_geometricBorderRank_of_minor` on the $l = 0$ flattening slice $M_0 = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$ ($\det M_0 = -1 \ne 0$).
  - Upper bound ($\le 2$): via the explicit 1-parameter algebraic curve $W_t = t^{-1}((e_0 + t e_1)^{\otimes 3} - e_0^{\otimes 3})$ over the algebraically closed (hence infinite) field $\overline{F}$.
- `wTensor_tensorRank`: $\mathrm{tensorRank}(W) = 3$.
  - For any rank-2 tensor $T = a_0 \otimes b_0 \otimes c_0 + a_1 \otimes b_1 \otimes c_1$, its two $l$-slices satisfy the polynomial identities (`rank2_slice_identities`):
    $$\det M_0 = d_{AB} c_{00} c_{01}, \qquad \det M_1 = d_{AB} c_{10} c_{11}, \qquad \operatorname{tr}(\operatorname{adj}(M_0) M_1) = d_{AB}(c_{01} c_{10} + c_{00} c_{11}),$$
    where $d_{AB} = \det A \det B$. On $W$, $\det M_0 = -1$, $\det M_1 = 0$, $\operatorname{tr}(\operatorname{adj}(M_0) M_1) = 0$, forcing $c_{10} = c_{11} = 0$, contradicting $(M_1)_{00} = 1$.

### 2.2 Optimal centered border-rank curve & sharp $R^{-12}$ escape law (`[LC]`)
The textbook one-sided border-rank curve $W_\varepsilon = \varepsilon^{-1}((e_0 + \varepsilon e_1)^{\otimes 3} - e_0^{\otimes 3})$ has $\|W_\varepsilon - W\|_F^2 = 3\varepsilon^2 + \varepsilon^4 \asymp R^{-6}$ (for $\|\theta\|_2 \asymp R \asymp \varepsilon^{-1/3}$).

Using the **centered finite-difference** curve (`centeredCurve`):
$$\tilde W_\varepsilon = \frac{1}{2\varepsilon}\Bigl((e_0 + \varepsilon e_1)^{\otimes 3} - (e_0 - \varepsilon e_1)^{\otimes 3}\Bigr) = W + \varepsilon^2 e_1^{\otimes 3}$$
cancels the $O(\varepsilon)$ error completely, yielding exact loss $L(\theta_\varepsilon) = \varepsilon^4$ and exact squared norm $\|\theta_\varepsilon\|_2^2 = 6(2\varepsilon)^{-2/3}(1 + \varepsilon^2)$, so
$$L(\theta_\varepsilon) \asymp R^{-12}.$$
Using the same three 2×2 slice identities quantitatively, `wTensor_loss_ge_of_paramNormSq` proves that **$R^{-12}$ is globally optimal** across all $\theta \in \mathbb{R}^{12}$:
$$\min\!\left(\frac{1}{64},\; \frac{1}{4624\,\|\theta\|_2^{12}}\right) \le \|T(\theta) - W\|_F^2.$$

### 2.3 Weight-decay scaling (`γ^{6/7}`) and stretched-exponential free energy (`N^{1/7}`) (`[LC]`)
Balancing $L \asymp R^{-12}$ against $\gamma R^2$ via weighted AM–GM gives the exact scaling laws:
- **Regularized ERM** (`wTensor_weightDecay_lower`, `wTensor_weightDecay_upper`):
  $$\frac{1}{64}\,\gamma^{6/7} \le \inf_{\theta \in \mathbb{R}^{12}}\bigl(\|T(\theta) - W\|_F^2 + \gamma \|\theta\|_2^2\bigr) \le 13\,\gamma^{6/7} \qquad (\forall \gamma \in (0, 1]),$$
  with sharp leading asymptote along `centeredCurve`:
  $$J^*(\gamma) \sim 7 \cdot 2^{-4/7}\gamma^{6/7}, \qquad L^*(\gamma) \sim 2^{-4/7}\gamma^{6/7}, \qquad \|\theta^*(\gamma)\|_2^2 \sim 6 \cdot 2^{-4/7}\gamma^{-1/7}, \qquad \frac{\gamma \|\theta^*\|_2^2}{L^*} \to 6.$$
- **Bayesian partition function** $Z_W^{(k)}(N) = \int \exp\bigl(-N\|T(\theta) - W\|_F^2 - \frac{1}{2}\|\theta\|_2^2\bigr)\, d\theta$:
  - **Width $k = 2$** (`wPartitionZ_two_le_stretchedExp`, `wPartitionZ_two_ge_stretchedExp`):
    $$\left(\frac{1}{32}\right)^{12} e^{-90 N^{1/7}} \le Z_W^{(2)}(N) \le (4\pi)^6 e^{-a_1 N^{1/7}} \qquad (\forall N \ge 1,\; a_1 = 64^{-1} \cdot 4^{-6/7}),$$
    certifying $-\log Z_W^{(2)}(N) \asymp N^{1/7}$ (so the DevInterp / WBIC effective learning coefficient $\hat\lambda(N) = N \mathbb{E}_N[L(\theta)] = N \frac{d}{dN}(-\log Z_W^{(2)}(N))$ **diverges as $N^{1/7}$**!).
  - **Width $k = 3$** (`wPartitionZ_three_ge_poly`, `wPartitionZ_three_ge_poly_sharp`):
    - Elementary $N^{-P/2}$ bound (`wPartitionZ_three_ge_poly`): $2^{18} e^{-10404} N^{-9} \le Z_W^{(3)}(N)$ ($\forall N \ge 1$).
    - **Sharp codimension-8 RLCT bound (`wPartitionZ_three_ge_poly_sharp`, `[LC]`):**
      $$\left(\frac{1}{64}\right)^{10} \cdot 256 \cdot e^{-32807025}\, N^{-4} \le Z_W^{(3)}(N) \qquad (\forall N \ge 1),$$
      matching the exact codimension-8 regular RLCT $\lambda = 8/2 = 4.0$ (proved via natural-order zero-Jacobian Fubini and an explicit 10D rational fiber chart around the submersive width-3 realization $W = \frac{1}{2}(e_0+e_1)^{\otimes 3} - \frac{1}{2}(e_0-e_1)^{\otimes 3} - e_1^{\otimes 3}$; note that the sparse canonical decomposition `wRealize3` itself has a rank-7 singular Jacobian whose missing $e_1^{\otimes 3}$ direction enters quadratically, still yielding $\lambda = 7/2 + 1/2 = 4.0$). Thus adding **one single neuron** ($k = 2 \to 3$) collapses $-\log Z_W^{(k)}(N)$ from stretched-exponential $\Theta(N^{1/7})$ back to Watanabe polynomial $\le 4\log N + C_3$, even though both widths have $\inf L = 0$.

### 2.4 Non-Compactness, Prior Tails, and Watanabe's Framework (`[paper]` + `[LC]`)

Standard Singular Learning Theory (Watanabe 2009, 2018) assumes a **compact** parameter space $\Theta \subset \mathbb{R}^P$ with compactly supported prior $\varphi(\theta)$. It is important to be explicit about how Track 1 sits relative to that assumption:

| Prior support / tail $\pi(\theta)$ | Width $k = 2$ on $W$-tensor | Free-energy scaling $-\log Z_W^{(2)}(N)$ | Effective LLC $\hat\lambda(N) = N \mathbb{E}_N[L]$ |
| :--- | :--- | :--- | :--- |
| **Compact ball** $\|\theta\|_2 \le R$ | Strictly unrealizable ($L^* \asymp R^{-12} > 0$) | $\Theta(N)$ (linear) | $\Theta(N)$ |
| **Gaussian** $e^{-\frac{1}{2}\|\theta\|_2^2}$ ($\ell_2$ WD) `[LC]` | Escapes along $\tilde W_\varepsilon$, $R \asymp N^{1/14}$ | $\Theta(N^{1/7})$ (stretched-exponential) | $\sim 2^{-10/7} N^{1/7} + \frac{27}{7}$ (`[paper]` Laplace) |
| **Stretched-exp** $e^{-c\|\theta\|_2^\alpha}$ ($\alpha > 0$) `[paper]` | Escapes with $R \asymp N^{1/(12+\alpha)}$ | $\Theta(N^{\alpha/(12+\alpha)})$ | $\Theta(N^{\alpha/(12+\alpha)})$ |
| **Heavy-tailed** $(1 + \|\theta\|_2^2)^{-\nu/2}$ ($\nu > P$) `[paper]` | Escapes with $R \asymp N^{1/12}$ | $\Theta(\log N)$ (**prior-dependent** $\lambda$) | $\Theta(1)$ (depends on $\nu$) |
| **Width $k = 3$ (any positive prior)** `[LC]` | Realizable at finite $\theta^*$ | $\Theta(\log N)$ ($\lambda = 4.0$, prior-independent) | $\to 4.0$ |

Thus, the stretched-exponential universality class $-\log Z(N) \asymp N^{1/7}$ is a genuine **non-compact** SLT phenomenon arising from the interaction between a non-closed algebraic image (strict border-rank gap $\underline{\mathrm{rk}}(T^*) \le k < \mathrm{rk}(T^*)$) and the Gaussian prior ($\ell_2$ weight decay) standard in deep learning and Bayesian neural networks.

---

## 3. PyTorch Empiricism (`[empirical]`)

Script: [bilinear_border_rank.py](../experiments/bilinear_border_rank.py) | Data: [bilinear_border_rank.json](../experiments/results/bilinear_border_rank.json) | Figure: [bilinear_border_rank.png](../experiments/results/bilinear_border_rank.png)

![Order-3 Border-Rank Singularity Results](../experiments/results/bilinear_border_rank.png)

### 3.1 Experiment 1: Weight-decay scaling & width-3 branch transition (L-BFGS, $\gamma \in [10^{-6}, 10^{-1}]$)

| Quantity (fit on $\gamma \le 3\times 10^{-4}$) | `[LC]` / Leading Theory | Empirical ($k=2$) | Empirical ($k=3$) |
| :--- | :---: | :---: | :---: |
| Loss slope $d\log L^* / d\log\gamma$ | $6/7 = 0.8571$ ($k=2$), $2.000$ ($k=3$) | **`0.8513`** | **`1.9996`** |
| Total objective slope $d\log J^* / d\log\gamma$ | $6/7 = 0.8571$ ($k=2$) | **`0.8600`** | — |
| Weight norm² slope $d\log \|\theta^*\|_2^2 / d\log\gamma$ | $-1/7 = -0.1429$ ($k=2$), $0$ ($k=3$) | **`-0.1386`** | plateaus at **`8.679`** |
| Ratio $J^*(\gamma) / (7\cdot 2^{-4/7}\gamma^{6/7})$ at $\gamma=10^{-6}$ | `1.0000` | **`1.0028`** | — |
| Virial ratio $\gamma \|\theta^*\|_2^2 / L^*$ at $\gamma=10^{-6}$ | `6.000` | **`6.05`** | $1.69 \times 10^6$ |
| Residual fraction in $e_1^{\otimes 3}$ at $\gamma=10^{-6}$ | `1.0000` (centered $\tilde W_\varepsilon$) | **`1.0000`** | `0.4576` |

- **Purely unseeded spontaneous discovery of the centered curve:** In a standalone random-start check (16 independent i.i.d. Gaussian starts per $\gamma$, **zero** warm-start and **zero** analytic seeding), L-BFGS on $k=2$ spontaneously converges onto the centered finite-difference border-rank curve $\tilde W_\varepsilon = W + \varepsilon^2 e_1^{\otimes 3}$ in **16/16 (100%)** starts at $\gamma \in \{10^{-2}, 10^{-3}, 10^{-4}\}$ (`e111 = 0.9992` at $10^{-4}$) and **10/16 (62.5%)** at $\gamma = 10^{-5}$ (`e111 = 0.9999`). The full $\gamma \in [10^{-1}, 10^{-6}]$ sweep uses only random starts plus warm-start continuation (no analytic seeding) and reaches `e111_residual_frac = 0.999985`, `obj/pred = 1.0028`, `virial = 6.05`.
- **Bonus empirical finding — Width-3 first-order branch transition:** At high weight decay ($\gamma = 10^{-1}$), even the width-3 model collapses onto the exact width-2 border-rank solution ($L = 8.066\times 10^{-2}$, $\|\theta\|_2^2 = 6.989$, matching $k=2$ to 5 significant digits) because the rank-3 realization costs more norm penalty ($\|\theta\|_2^2 \approx 8.68$). Between $\gamma = 10^{-1}$ and $\gamma = 10^{-2}$, $k=3$ undergoes a sharp transition onto the exact rank-3 fiber.

### 3.2 Experiment 2: Exact Laplace Asymptotics & Equilibrated SMC + HMC ($\beta \in [0.5, 10^6]$)

#### Exact 4-Scale Laplace Analysis for $k = 2$ (`[paper]`)
At the Newton-refined saddle point $\theta^*(\beta)$ of $U_\beta(\theta) = \beta L(\theta) + \frac{1}{2}\|\theta\|_2^2$, the $12\times 12$ Hessian $H_U(\beta) = \beta \nabla^2 L(\theta^*) + I_{12}$ exhibits an exact 4-scale spectrum (condition number $\kappa \approx 8.9\times 10^6$ at $\beta = 10^5$ and $1.7\times 10^8$ at $\beta = 10^6$):
- **4 torus gauge eigenvalues** $= 2 = O(1)$,
- **1 radial border-rank escape eigenvalue** $\to 14 = O(1)$,
- **3 intermediate eigenvalues** $\propto \beta^{6/7}$,
- **4 stiff eigenvalues** $\propto \beta^{9/7}$.

Differentiating the Laplace free energy $F_{\mathrm{Lap}}(\beta) = U_\beta(\theta^*) + \frac{1}{2}\log\det H_U(\beta) - \log 32$ (accounting for the $32 = 2 \times 4^2$ discrete symmetry modes) gives the exact subleading fluctuation offsets:
$$\hat\lambda_{\mathrm{Lap}}(\beta) = \beta \frac{d F_{\mathrm{Lap}}}{d\beta} = \beta L^*(\beta) + \frac{27}{7} + o(1) = 2^{-10/7}\beta^{1/7} + \frac{27}{7} + o(1) \qquad \left(\frac{27}{7} = \frac{1}{2}\!\left(3\cdot\frac{6}{7} + 4\cdot\frac{9}{7}\right) \approx 3.8571\right),$$
$$\mathbb{E}_\beta[\|\theta\|_2^2] = \|\theta^*(\beta)\|_2^2 + \frac{30}{7} + o(1) \qquad \left(\frac{30}{7} \approx 4.2857\right).$$

#### Equilibrated SMC + HMC & Cold-Start Verification (`[empirical]`)
Using 192 parallel SMC chains across 44 geometric temperatures up to $\beta = 10^6$ with **row-diagonal preconditioned HMC**, exact **torus gauge-balancing MH moves** ($\det J = 1$), and an exact reversible **1-parameter Lie-group border-rank move** $S_u = \operatorname{diag}(e^u, e^{-2u})$ (which leaves $W$ invariant and scales $e_1^{\otimes 3}$ by $e^{-6u}$ with Jacobian $\log|\det J| = -3ku$, eliminating the $\kappa \sim 10^8$ radial equilibration lag):

| $\beta$ | $k=2$ Exact Laplace $\hat\lambda$ | $k=2$ SMC+HMC $\hat\lambda$ | $k=2$ Cold-Start $\hat\lambda$ ($z$) | $k=2$ Laplace $R^2$ | $k=2$ SMC $R^2$ | $k=2$ Cold $R^2$ ($z$) |
|---:|---:|---:|---:|---:|---:|---:|
| $10^4$ | 5.208 | 5.167 ± 0.057 | 5.261 ± 0.043 ($z=1.32$) | 21.15 | 21.03 ± 0.15 | 20.64 ± 0.15 ($z=1.85$) |
| $10^5$ | 5.764 | 5.787 ± 0.062 | 5.842 ± 0.055 ($z=0.67$) | 27.51 | 27.24 ± 0.16 | 27.09 ± 0.14 ($z=0.71$) |
| $10^6$ | 6.522 | 6.467 ± 0.070 | 6.426 ± 0.058 ($z=0.46$) | 36.44 | 36.21 ± 0.16 | 36.64 ± 0.16 ($z=1.91$) |

- **Width $k = 3$ (realizable):** $\hat\lambda(\beta) = \beta \mathbb{E}_\beta[L(\theta)]$ hits a **rock-flat Watanabe plateau at $\hat\lambda = 3.999 \pm 0.061$** across three full decades $\beta \in [10^3, 10^6]$ (matching the exact codimension-8 regular fiber RLCT $\lambda = 8/2 = 4.0$, with cold-start checks agreeing within $z < 0.9$).
- **Width $k = 2$ (border-rank):** SMC+HMC tracks the exact Laplace curve $\beta L^*(\beta) + \frac{27}{7}$ across all four high-$\beta$ decades and matches independent cold-start chains within $< 2\sigma$ on both $\hat\lambda$ and $\mathbb{E}_\beta[\|\theta\|_2^2]$:
  - High-$\beta$ ($\beta \ge 10^3$) excess LLC slope of $\hat\lambda(\beta) - \frac{27}{7}$: **`+0.1535`** (theory $+1/7 = +0.1429$); raw slope `0.0447` vs. exact Laplace `0.0448`.
  - High-$\beta$ excess norm² slope of $\mathbb{E}_\beta[\|\theta\|_2^2] - \frac{30}{7}$: **`+0.1415`** (theory $+1/7 = +0.1429$).
  - **Free energy & Bayes factor:** At $\beta = 10^5$, $F_{\mathrm{SMC}}^{(k=2)} = 60.93$ (exact Laplace `60.45`) vs. $F_{\mathrm{SMC}}^{(k=3)} = 52.68$ ($\Delta F = 8.25$ nats, $3.8\times 10^3\times$ Bayes factor). At $\beta = 10^6$, $F_{\mathrm{SMC}}^{(k=2)} = 75.06$ (exact Laplace `74.54`) vs. $F_{\mathrm{SMC}}^{(k=3)} = 61.85$ — a **13.21 nat** Bayesian preference ($e^{13.21} \approx 5.4 \times 10^5\times$ Bayes factor) for the *larger* width-3 model over the smaller width-2 border-rank model!

### 3.3 Experiment 3: Real 1-hidden-layer quadratic MLP $f(x) = W_{\mathrm{out}}(W_{\mathrm{in}} x)^{\odot 2}$
Targeting $f^*(x) = (2 x_0 x_1,\; x_0^2)$ on $N = 4096$ Gaussian inputs $x \sim \mathcal{N}(0, I_2)$ (exact symmetric restriction of $W$):
- **Regularized MSE + $\gamma \|\theta\|_2^2$ sweep:** $k = 2$ exhibits MSE slope **`0.8489`** (theory `0.8571`) and norm² slope **`-0.1367`** (theory `-0.1429`); $k = 3$ exhibits MSE slope **`1.9995`** (theory `2.000`).
- **Unregularized Adam (20,000 steps):** $k = 3$ reaches machine-precision zero loss ($10^{-31}$ before Adam step-size jitter; $1.25 \times 10^{-7}$ at step 20k) at bounded norm² ($7.02$), while $k = 2$ stalls at $7.49 \times 10^{-5}$ with continuously exploding weights ($\|\theta(t)\|_2^2 \to 12.61$).
