# Order-3 Bilinear MLPs, Border-Rank Singularity, and Stretched-Exponential SLT

**Epistemic status summary:**
- **`[LC]`** (Lean 4 kernel-certified, zero `sorry`, strictly `[propext, Classical.choice, Quot.sound]` in `lean/EatYourFiber/BorderRankSLT.lean`):
  1. Bilinear MLP ↔ OpenAI `Tensor` rank/border-rank bridge (`exists_zero_loss_iff_tensorRank_le`, `geometricBorderRank_le_of_approx_loss`).
  2. Exact algebraic border-rank gap of the canonical $W$-tensor over *every* field $F$: $\mathrm{geometricBorderRank}(W) = 2 < 3 = \mathrm{tensorRank}(W)$ (`wTensor_geometricBorderRank`, `wTensor_tensorRank`, `wTensor_borderRank_lt_tensorRank`), building directly on OpenAI's vendored `BorderRank.lean` (`card_le_geometricBorderRank_of_minor`).
  3. Sharp quantitative border-rank escape law $\inf_{\|\theta\|_2 \le R} \|T(\theta) - W\|_F^2 \asymp R^{-12}$ (`wTensor_loss_ge_of_paramNormSq`, `wTensor_escape_upper`), achieved by the **centered finite-difference** optimal border-rank curve $\tilde W_\varepsilon = W + \varepsilon^2 e_1^{\otimes 3}$ (beating the textbook one-sided curve $W_\varepsilon$ by a full factor of 2 in the exponent: $R^{-12}$ vs $R^{-6}$).
  4. Exact weight-decay scaling law $\inf_\theta (\|T(\theta) - W\|_F^2 + \gamma \|\theta\|_2^2) \asymp \gamma^{6/7}$ (`wTensor_weightDecay_lower`, `wTensor_weightDecay_upper`).
  5. **Stretched-exponential Bayesian free energy** $c_1 N^{1/7} \le -\log Z_W^{(2)}(N) \le c_2 N^{1/7}$ (`wPartitionZ_two_le_stretchedExp`, `wPartitionZ_two_ge_stretchedExp`) vs. **width-3 realizable polynomial contrast** $-\log Z_W^{(3)}(N) \le 9 \log N + C_3$ (`wPartitionZ_three_ge_poly`).
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
  - **Width $k = 3$** (`wPartitionZ_three_ge_poly`):
    $$2^{18} e^{-10404} N^{-9} \le Z_W^{(3)}(N) \qquad (\forall N \ge 1),$$
    so adding **one single neuron** ($k = 2 \to 3$) collapses $-\log Z_W^{(k)}(N)$ from stretched-exponential $\Theta(N^{1/7})$ back to Watanabe polynomial $O(\log N)$, even though both widths have $\inf L = 0$.

---

## 3. PyTorch Empiricism (`[empirical]`)

Script: [bilinear_border_rank.py](file:///Users/bubskee/workspace/eat-your-fiber/experiments/bilinear_border_rank.py) | Data: [bilinear_border_rank.json](file:///Users/bubskee/workspace/eat-your-fiber/experiments/results/bilinear_border_rank.json) | Figure: [bilinear_border_rank.png](file:///Users/bubskee/workspace/eat-your-fiber/experiments/results/bilinear_border_rank.png)

![Order-3 Border-Rank Singularity Results](file:///Users/bubskee/workspace/eat-your-fiber/experiments/results/bilinear_border_rank.png)

### 3.1 Experiment 1: Weight-decay scaling & width-3 branch transition (L-BFGS, $\gamma \in [10^{-6}, 10^{-1}]$)

| Quantity (fit on $\gamma \le 3\times 10^{-4}$) | `[LC]` / Leading Theory | Empirical ($k=2$) | Empirical ($k=3$) |
| :--- | :---: | :---: | :---: |
| Loss slope $d\log L^* / d\log\gamma$ | $6/7 = 0.8571$ ($k=2$), $2.000$ ($k=3$) | **`0.8513`** | **`1.9996`** |
| Total objective slope $d\log J^* / d\log\gamma$ | $6/7 = 0.8571$ ($k=2$) | **`0.8600`** | — |
| Weight norm² slope $d\log \|\theta^*\|_2^2 / d\log\gamma$ | $-1/7 = -0.1429$ ($k=2$), $0$ ($k=3$) | **`-0.1386`** | plateaus at **`8.679`** |
| Ratio $J^*(\gamma) / (7\cdot 2^{-4/7}\gamma^{6/7})$ at $\gamma=10^{-6}$ | `1.0000` | **`1.0028`** | — |
| Virial ratio $\gamma \|\theta^*\|_2^2 / L^*$ at $\gamma=10^{-6}$ | `6.000` | **`6.05`** | $1.69 \times 10^6$ |
| Residual fraction in $e_1^{\otimes 3}$ at $\gamma=10^{-6}$ | `1.0000` (centered $\tilde W_\varepsilon$) | **`1.0000`** | `0.4576` |

- **Spontaneous discovery of the centered curve:** Without any symmetry constraint, L-BFGS on $k=2$ spontaneously converges onto the centered finite-difference border-rank curve $\tilde W_\varepsilon = W + \varepsilon^2 e_1^{\otimes 3}$ (`e111_residual_frac = 1.0000`, `obj/pred = 1.0028`, `virial = 6.05`).
- **Bonus empirical finding — Width-3 first-order branch transition:** At high weight decay ($\gamma = 10^{-1}$), even the width-3 model collapses onto the exact width-2 border-rank solution ($L = 8.066\times 10^{-2}$, $\|\theta\|_2^2 = 6.989$, matching $k=2$ to 5 significant digits) because the rank-3 realization costs more norm penalty ($\|\theta\|_2^2 \approx 8.68$). Between $\gamma = 10^{-1}$ and $\gamma = 10^{-2}$, $k=3$ undergoes a sharp transition onto the exact rank-3 fiber.

### 3.2 Experiment 2: SMC + HMC Bayesian free energy & diverging LLC ($\beta \in [0.5, 10^5]$)
Using 128 parallel SMC chains with exact systematic importance resampling, 6-step HMC rejuvenation, and exact torus gauge-balancing MH moves (`ESS ~ 80–90 / 128`, `acc ~ 0.65–0.74`):

- **Width $k = 3$ (realizable):** $\hat\lambda(\beta) = \beta \mathbb{E}_\beta[L(\theta)]$ hits a **rock-flat Watanabe plateau at $\hat\lambda = 4.03 \pm 0.08$** across three full decades $\beta \in [10^2, 10^5]$ (matching the exact codimension-8 regular fiber RLCT $\lambda = 8/2 = 4.0$), with posterior weight norm² frozen at $\mathbb{E}_\beta[\|\theta\|_2^2] = 15.58$.
- **Width $k = 2$ (border-rank):** $\hat\lambda(\beta)$ **peels upward and diverges** along the predicted $\beta^{1/7}$ power law:
  - High-$\beta$ ($\beta \ge 10^3$) slope of $\mathbb{E}_\beta[L]$: **`-0.8464`** (theory $-6/7 = -0.8571$).
  - High-$\beta$ slope of $\hat\lambda(\beta) = \beta \mathbb{E}_\beta[L]$: **`+0.1536`** (theory $+1/7 = +0.1429$).
  - At $\beta = 10^5$: $\hat\lambda^{(k=2)} = 8.65 \pm 0.14$ (more than $2\times$ the width-3 RLCT!), $\mathbb{E}_\beta[\|\theta\|_2^2] = 23.73$ (still growing), and the exact SMC free energy is $F_{\mathrm{SMC}}^{(k=2)} = 63.71$ vs. $F_{\mathrm{SMC}}^{(k=3)} = 52.97$ — a **10.74 nat** Bayesian preference ($e^{10.74} \approx 46{,}000\times$ Bayes factor) for the *larger* width-3 model over the smaller width-2 border-rank model!

### 3.3 Experiment 3: Real 1-hidden-layer quadratic MLP $f(x) = W_{\mathrm{out}}(W_{\mathrm{in}} x)^{\odot 2}$
Targeting $f^*(x) = (2 x_0 x_1,\; x_0^2)$ on $N = 4096$ Gaussian inputs $x \sim \mathcal{N}(0, I_2)$ (exact symmetric restriction of $W$):
- **Regularized MSE + $\gamma \|\theta\|_2^2$ sweep:** $k = 2$ exhibits MSE slope **`0.8489`** (theory `0.8571`) and norm² slope **`-0.1367`** (theory `-0.1429`); $k = 3$ exhibits MSE slope **`1.9995`** (theory `2.000`).
- **Unregularized Adam (20,000 steps):** $k = 3$ reaches machine-precision zero loss ($10^{-31}$ before Adam step-size jitter; $1.25 \times 10^{-7}$ at step 20k) at bounded norm² ($7.02$), while $k = 2$ stalls at $7.49 \times 10^{-5}$ with continuously exploding weights ($\|\theta(t)\|_2^2 \to 12.61$).
