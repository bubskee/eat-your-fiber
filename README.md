# eat-your-fiber

**Singular learning theory on—and at the boundary of—zero-loss fibers: Lean 4 kernel certificates paired with PyTorch empiricism.**

Different parameter vectors computing the *exact same function* can sit in valleys with macroscopically different local learning coefficients $(\lambda, m_{\log})$—and in higher-order multilinear models, the zero-loss fiber can disappear to infinity entirely while $\inf_\theta L(\theta) = 0$. Building on [residual-slt-in-a-haystack](https://github.com/bubskee/residual-slt-in-a-haystack) and [slt-bridge](https://github.com/bubskee/slt-bridge), this repository studies two complementary regimes side by side, aiming for macroscopic $\Delta\lambda$ phenomena that are simultaneously **kernel-certified in Lean 4** and **measured live in PyTorch**:

1. **Order-2 — Inside the zero-loss fiber (LoRA / Reduced-Rank / Self-Attention):** The rank-$\le k$ matrix locus is Zariski-closed, so every realizable target $M^*$ has a nonempty finite zero-loss fiber $\{BA = M^*\}$. Across that fiber, $(\lambda, m_{\log})$ undergoes an exact **plateau-to-parabola phase transition** (flat Aoyagi–Watanabe plateau for $q \le q_c$, exact quadratic rise $\frac{1}{2}(q - q_c)^2$ for $q > q_c$, placing standard LoRA initialization $q = k$ at the top of the parabola). In single-head self-attention, nonlinear composition creates a sharp **architectural coupling dichotomy**: joint `V+O` quadruples the supercritical jump to $2(q - H/2)_+^2$, whereas joint `Q+K` develops an all-orders $GL(q)$ gauge symmetry that **completely annihilates** the parabola (before multi-head transversality decouples both pairs).
2. **Order-3 — Boundary of the zero-loss fiber (Bilinear MLPs / 3-Tensors):** For width-$k$ bilinear MLPs $x \mapsto W_{\mathrm{out}}((W_1 x) \odot (W_2 x))$, the rank-$\le k$ 3-tensor locus is **not closed**. When a target sits in a strict border-rank gap $\underline{\mathrm{rk}}(T^*) \le k < \mathrm{rk}(T^*)$, $\inf_\theta L(\theta) = 0$ yet the finite zero-loss fiber is **empty**. Leveraging vendored 3-tensor border-rank formalizations from [`openai/math`](https://github.com/openai/math), we certify that under a Gaussian prior ($\ell_2$ weight decay), the canonical $W$-tensor forces a third non-compact SLT universality class—**stretched-exponential free energy** $-\log Z_W^{(2)}(N) \asymp N^{1/7}$—strictly between realizable Watanabe $\Theta(\log N)$ and compact unrealizable $\Theta(N)$.

---

## Epistemic labels

We keep claim statuses strictly separated across `lean/`, `notes/`, and `experiments/`:

- **`[LC]`** — Lean-certified: builds cleanly under `-DwarningAsError=true -DautoImplicit=false` with zero `sorry`; all 60 theorems in [`lean/Audit.lean`](lean/Audit.lean) depend strictly on `[propext, Classical.choice, Quot.sound]`.
- **`[paper]`** — Proved on paper (local linearization, pullback, or branch comparison), not yet formalized in Lean.
- **`[empirical]`** — Numerical / PyTorch MALA, SMC+HMC, L-BFGS, or SGD/Adam/AdamW observation.
- **`[conjectural]`** — Heuristic or open conjecture.

---

## Master theorem & evidence index

| Result | Exact Formula / Scaling Law | Status | Lean Certificate | Empirics & Notes |
| :--- | :--- | :---: | :--- | :--- |
| **Aoyagi–Watanabe (2005) Thm 1** (all 4 regimes & parities, elementary projective blowup) | $(\lambda_{\mathrm{AW}}(m,n,k),\, m_{\mathrm{AW}}(m,n,k))$ for all $m,k,n\in\mathbb{N}$ | `[LC]` | [`aoyagi_watanabe_rank0`](lean/EatYourFiber/AoyagiWatanabe.lean) | [`lora_fiber_phase_transition.md`](notes/lora_fiber_phase_transition.md) |
| **LoRA zero-loss fiber phase transition** (rank-$q$ half-dead $(0, A_0)$) | Plateau $\lambda_{\mathrm{AW}}$ ($q \le q_c$); parabola $+\frac{1}{2}(q-q_c)^2$ ($q > q_c$, $m_{\log}=1$) | `[LC]` + `[empirical]` | [`lora_fiber_exact_pair`](lean/EatYourFiber/AoyagiWatanabe.lean), [`lora_subcritical_plateau_exact_pair`](lean/EatYourFiber/AoyagiWatanabe.lean), [`lora_supercritical_parabola_exact_pair`](lean/EatYourFiber/AoyagiWatanabeBoundary.lean) | [`lora_fiber_sgld.py`](experiments/lora_fiber_sgld.py), [`lora_fiber_phase_transition.md`](notes/lora_fiber_phase_transition.md) |
| **Complete $(r^*, p, q)$ RRR fiber stratification & component codimension (`MAIS-O70`)** | $\lambda = \frac{MN - m'n'}{2} + \lambda_{\mathrm{AW}}(m',n',k')$; 2D plateau; $2\lambda_{\mathrm{exc}} = \min_{\alpha \in [q, k-p]} \operatorname{codim} Z_{\alpha,k-\alpha}$ | `[LC]` | [`rrr_fiber_exact_pair`](lean/EatYourFiber/RRRFiberStratification.lean), [`rrr_fiber_subcritical_plateau`](lean/EatYourFiber/RRRFiberStratification.lean), [`rrr_fiber_A_supercritical_parabola`](lean/EatYourFiber/RRRFiberStratification.lean), [`rrr_fiber_B_supercritical_parabola`](lean/EatYourFiber/RRRFiberStratification.lean), [`rrr_fiber_eq_min_component_codim`](lean/EatYourFiber/RRRFiberStratification.lean) | [`lora_fiber_phase_transition.md`](notes/lora_fiber_phase_transition.md) (§3.1) |
| **Single-head Attention `V+O` vs. `Q+K` coupling dichotomy** (exact all-orders; general $U,V \in O(d)$ for `Q+K`) | `V+O`: $dH - \frac{H^2}{2} + 2(q-\frac{H}{2})_+^2$ (`4×` jump); `Q+K`: identically flat $dH - \frac{H^2}{2}$ via $GL(q)$ gauge | `[LC]` + `[empirical]` | [`attention_VO_supercritical_parabola`](lean/EatYourFiber/AttentionLoRA.lean), [`attention_QK_general_gauge_orbit`](lean/EatYourFiber/AttentionLoRA.lean), [`attention_QK_general_flat_pair`](lean/EatYourFiber/AttentionLoRA.lean), [`attention_coupling_dichotomy_standard_init`](lean/EatYourFiber/AttentionLoRA.lean) | [`lora_nonlinear_fiber.py`](experiments/lora_nonlinear_fiber.py), [`lora_nonlinear_fiber.md`](notes/lora_nonlinear_fiber.md) |
| **2-Head Attention decoupling & reversal** ($d=8, d_h=4, H=4$) | Both `V+O` & `Q+K` decouple to $2 \times \lambda_{\mathrm{single}}(q)$: plateau $24\to 28$, top $32$ | `[paper]` + `[empirical]` | — | [`lora_nonlinear_fiber.py`](experiments/lora_nonlinear_fiber.py), [`lora_nonlinear_fiber.md`](notes/lora_nonlinear_fiber.md) |
| **Nonlinear MLP invariance, dead-ReLU shift & optimizer dynamics** | GELU/ReLU/SwiGLU match linear curve; $d_{\mathrm{dead}}$ shifts $m_{\mathrm{eff}} = m - d_{\mathrm{dead}}$; WD drives LoRA down parabola | `[paper]` + `[empirical]` | — | [`lora_sgd_dynamics.py`](experiments/lora_sgd_dynamics.py), [`lora_nonlinear_fiber.py`](experiments/lora_nonlinear_fiber.py), [`lora_sgd_dynamics.md`](notes/lora_sgd_dynamics.md) |
| **Bilinear MLP $\leftrightarrow$ 3-tensor rank / border-rank bridge** | Nonempty finite fiber $\iff \mathrm{rk}(T^*)\le k$; $\inf L = 0 \implies \underline{\mathrm{rk}}(T^*)\le k$ | `[LC]` | [`exists_zero_loss_iff_tensorRank_le`](lean/EatYourFiber/BorderRankSLT.lean), [`geometricBorderRank_le_of_approx_loss`](lean/EatYourFiber/BorderRankSLT.lean) | [`bilinear_border_rank.md`](notes/bilinear_border_rank.md) |
| **$W$-tensor exact border-rank gap** (over every field $F$) | $\underline{\mathrm{rk}}(W) = 2 < 3 = \mathrm{rk}(W)$ | `[LC]` | [`wTensor_geometricBorderRank`](lean/EatYourFiber/BorderRankSLT.lean), [`wTensor_tensorRank`](lean/EatYourFiber/BorderRankSLT.lean), [`wTensor_borderRank_lt_tensorRank`](lean/EatYourFiber/BorderRankSLT.lean) | [`bilinear_border_rank.md`](notes/bilinear_border_rank.md) |
| **Sharp centered-curve escape & weight-decay scaling** ($W$-tensor, $k=2$) | $\inf_{\|\theta\|_2\le R}\|T(\theta)-W\|_F^2 \asymp R^{-12}$; $\inf_\theta(L + \gamma\|\theta\|_2^2) \asymp \gamma^{6/7}$ | `[LC]` + `[empirical]` | [`wTensor_loss_ge_of_paramNormSq`](lean/EatYourFiber/BorderRankSLT.lean), [`wTensor_escape_upper`](lean/EatYourFiber/BorderRankSLT.lean), [`wTensor_weightDecay_lower`](lean/EatYourFiber/BorderRankSLT.lean), [`wTensor_weightDecay_upper`](lean/EatYourFiber/BorderRankSLT.lean) | [`bilinear_border_rank.py`](experiments/bilinear_border_rank.py), [`bilinear_border_rank.md`](notes/bilinear_border_rank.md) |
| **Stretched-exponential Bayesian free energy vs. sharp width-3 RLCT contrast** | $-\log Z_W^{(2)}(N) \asymp N^{1/7}$ vs. $-\log Z_W^{(3)}(N) \le 4\log N + C_3$ (exact $\lambda = 4.0$) | `[LC]` + `[empirical]` | [`wPartitionZ_two_le_stretchedExp`](lean/EatYourFiber/BorderRankSLT.lean), [`wPartitionZ_two_ge_stretchedExp`](lean/EatYourFiber/BorderRankSLT.lean), [`wPartitionZ_three_ge_poly_sharp`](lean/EatYourFiber/BorderRankSLT.lean) | [`bilinear_border_rank.py`](experiments/bilinear_border_rank.py), [`bilinear_border_rank.md`](notes/bilinear_border_rank.md) |

---

## Current highlights

### Track 1 — Order-3 Bilinear MLPs, Border-Rank Singularity & Stretched-Exponential SLT
- **`[LC]` Algebraic & analytic certificates ([`BorderRankSLT.lean`](lean/EatYourFiber/BorderRankSLT.lean)):**
  - Bridges width-$k$ bilinear MLPs $T(A, B, C) = \sum_{r=1}^k a_r \otimes b_r \otimes c_r$ directly to OpenAI's vendored `Tensor`, `tensorRank`, and `geometricBorderRank`.
  - Certifies $\mathrm{geometricBorderRank}(W) = 2 < 3 = \mathrm{tensorRank}(W)$ over every field $F$ for $W = e_0\otimes e_0\otimes e_1 + e_0\otimes e_1\otimes e_0 + e_1\otimes e_0\otimes e_0$ (using `card_le_geometricBorderRank_of_minor` from `openai/math` and 2×2 slice adjugate identities).
  - Proves the sharp quantitative escape law $\inf_{\|\theta\|_2 \le R} \|T(\theta) - W\|_F^2 \asymp R^{-12}$, achieved by the **centered finite-difference curve** $\tilde W_\varepsilon = \frac{1}{2\varepsilon}((e_0+\varepsilon e_1)^{\otimes 3} - (e_0-\varepsilon e_1)^{\otimes 3}) = W + \varepsilon^2 e_1^{\otimes 3}$ (beating the textbook one-sided curve $R^{-6}$ by $2\times$ in the exponent), the exact weight-decay law $\inf_\theta (\|T(\theta)-W\|_F^2 + \gamma \|\theta\|_2^2) \asymp \gamma^{6/7}$, and **stretched-exponential Bayesian free energy** $-\log Z_W^{(2)}(N) \asymp N^{1/7}$ contrasted against the **sharp width-3 codimension-8 RLCT bound** $-\log Z_W^{(3)}(N) \le 4\log N + C_3$ (`wPartitionZ_three_ge_poly_sharp`, proved via an explicit 10D rational fiber chart around the submersive realization $\frac{1}{2}(e_0+e_1)^{\otimes 3} - \frac{1}{2}(e_0-e_1)^{\otimes 3} - e_1^{\otimes 3}$).
- **`[empirical]` PyTorch verification ([`bilinear_border_rank.py`](experiments/bilinear_border_rank.py), [`bilinear_border_rank.md`](notes/bilinear_border_rank.md)):**
  - Unseeded purely random multi-start L-BFGS spontaneously discovers the centered curve $\tilde W_\varepsilon$ (**16/16** at $\gamma \in \{10^{-2}, 10^{-3}, 10^{-4}\}$, **10/16** at $10^{-5}$; unseeded sweep reaches `e111 = 0.999985`, `obj/pred = 1.0028`, virial `6.05` vs `6.00`, loss slope `0.851` vs `6/7`, norm² slope `-0.139` vs `-1/7`), plus a sharp width-3 first-order branch transition from the border-rank branch at high $\gamma$ to the rank-3 fiber at low $\gamma$.
  - Preconditioned SMC + HMC with an exact reversible Lie-group border-rank move up to $\beta = 10^6$ (verified against independent cold-start chains within $<2\sigma$) shows $\hat\lambda(\beta)$ plateauing flat at **`3.999 ± 0.061`** ($\lambda = 4.0$) for $k=3$, while for $k=2$ tracking the exact 4-scale Laplace prediction $\hat\lambda_{\mathrm{Lap}}(\beta) = 2^{-10/7}\beta^{1/7} + \frac{27}{7}$ (`6.467 ± 0.070` vs `6.522` at $\beta=10^6$; excess slope `0.1535` vs `1/7`; excess norm² slope `0.1415` vs `1/7`), yielding a **13.21-nat** Bayesian preference ($5.4 \times 10^5\times$ Bayes factor) for the *larger* width-3 model at $\beta = 10^6$.
  - A real 1-hidden-layer quadratic MLP $f(x) = W_{\mathrm{out}}(W_{\mathrm{in}} x)^{\odot 2}$ targeting $(2x_0 x_1, x_0^2)$ exhibits the same $\gamma^{6/7}$ vs. $\gamma^2$ laws and unregularized Adam weight explosion at $k=2$ vs. bounded convergence at $k=3$.

### Track 2 — Order-2 LoRA Fibers, Full Aoyagi–Watanabe & Self-Attention Coupling
- **`[LC]` Full Aoyagi–Watanabe (2005) Theorem 1 & Unconditional LoRA Fiber Phase Transition ([`AoyagiWatanabe.lean`](lean/EatYourFiber/AoyagiWatanabe.lean), [`AoyagiWatanabeBoundary.lean`](lean/EatYourFiber/AoyagiWatanabeBoundary.lean), [`LoRASchurLaplace.lean`](lean/EatYourFiber/LoRASchurLaplace.lean)):**
  - Elementary Lean 4 proof of **all four regimes** and both parities ($m_{\log} \in \{1, 2\}$) of Aoyagi–Watanabe (2005) Theorem 1 for all $m, k, n \in \mathbb{N}$ via a single structural induction on $(m, k)$ using projective blowup of $B$ at its max-magnitude entry, two unit-Jacobian affine shears, and `SLT.NC.stepG` (zero Hironaka, zero Weyl/SVD Jacobian, zero polar coordinates).
  - Combined with the measure-theoretic Schur-complement Laplace reduction (`lora_schur_fiber_reduction`), unconditionally certifies the entire rank-0 LoRA fiber phase transition: flat subcritical plateau $(\lambda_{\mathrm{AW}}(m,n,k), m_{\mathrm{AW}}(m,n,k))$ for $m + 2q < n + k$ and exact supercritical parabola $(\lambda_{\mathrm{AW}}^{\mathrm{even}}(m,n,k) + \frac{1}{2}(q - q_c)^2, 1)$ for $n + k \le m + 2q$.
- **`[LC]` Complete Unconditional $(r^*, p, q)$ RRR Zero-Loss Fiber Stratification ([`RRRFiberStratification.lean`](lean/EatYourFiber/RRRFiberStratification.lean)):**
  - Unconditionally certifies the exact local learning pair $((MN - m'n')/2 + \lambda_{\mathrm{AW}}(m', n', k'),\, m_{\mathrm{AW}}(m', n', k'))$ at *every* canonical $(r^*, p, q)$ stratum of the zero-loss fiber $\{BA = M^*\}$ for all $r^*, p, q, m', n', k' \in \mathbb{N}$ and **every** invertible $D \in GL(r^*, \mathbb{R})$ (`rrr_fiber_exact_pair`), including the 2D flat subcritical plateau (`rrr_fiber_subcritical_plateau`), mutually exclusive $A$- and $B$-supercritical parabolas (`rrr_fiber_A_supercritical_parabola`, `rrr_fiber_B_supercritical_parabola`, `rrr_supercritical_mutually_exclusive`), and the exact irreducible-component codimension identity $2\lambda_{\mathrm{exc}}(p,q) = \min_{\alpha \in [q, k-p]} (m\alpha + n(k-\alpha) - \alpha(k-\alpha))$ (`rrr_fiber_eq_min_component_codim`).
  - Provides an unconditional, real-valued Lean 4 kernel certificate of the local RRR RLCT formula studied as a candidate solution to **MAIS-O70** in [`ztreisman/arc-llc`](https://github.com/ztreisman/arc-llc) (without Hironaka, complexification, or Docampo's arc-space formula).
- **`[LC]` Self-Attention LoRA Coupling Dichotomy ([`AttentionLoRA.lean`](lean/EatYourFiber/AttentionLoRA.lean)):**
  - Certifies to **all orders** (via exact nonlinear left-multiplication diffeomorphisms `attnVO_factor`, `attnQK_factor` and an exact two-sided Schur complement reduction `twoSided_schur_reduction`) that on single-head self-attention:
    - Joint **`V+O`** stacks both $A$-factors on the same side, producing a **$4\times$-amplified supercritical parabola** $(dH - H^2/2 + 2(q - H/2)_+^2, 1)$ (jump $+H^2/2$ at $q = H$).
    - Joint **`Q+K`** forms a balanced two-sided $(q, q)$ fiber with an exact all-orders **$GL(q)$ zero-loss gauge orbit**, giving identically flat $(dH - H^2/2, 1)$ across all $q \le H \le d$ — certified not only for canonical shared bases (`attention_QK_flat_pair`) but for **arbitrary orthogonal bases $U, V \in O(d)$** (`attention_QK_general_gauge_orbit`, `attention_QK_general_exact_pair`, `attention_QK_general_flat_pair`).
- **`[empirical]` MALA, Nonlinear MLPs, Multi-Head Attention & Optimizer Dynamics ([`lora_fiber_sgld.py`](experiments/lora_fiber_sgld.py), [`lora_sgd_dynamics.py`](experiments/lora_sgd_dynamics.py), [`lora_nonlinear_fiber.py`](experiments/lora_nonlinear_fiber.py)):**
  - MALA reproduces the plateau + parabola on linear (`rank0_8x8x8`, `lora_16x16_r2_H10`) and nonlinear MLPs (GELU, ReLU, SwiGLU within $1\sigma$; killing 2 ReLU neurons shifts $q_c: 4 \to 5$, plateau $24.0 \to 19.5$, top $32.0 \to 24.0$, measured `23.99 ± 0.04`).
  - Single-head attention MALA confirms `V+O` (`32.00 ± 0.06`) vs. `Q+K` (`23.99 ± 0.05`), while **2-head attention** decouples both pairs to $2 \times \lambda_{\mathrm{single}}(q)$ (plateau $24 \to 28$, MALA `26.59`/`26.78`; both tops at `32.0`, MALA `32.06 ± 0.07` and `32.10 ± 0.14`).
  - On realizable linear, GELU MLP, and full $Q,K,V,O$ attention tasks, GD/SGD/Adam without weight decay remain pinned at the supercritical regime ($\hat\lambda \approx 85.8\text{--}86.0$ linear, `45.6–47.5` GELU, `86.27` attention), while `AdamW` and `SGD+WD` contract $A_{\mathrm{in}}$ below $q_c$ and descend onto the subcritical plateau.

---

## Scope, assumptions & honest limitations

1. **Non-compact prior vs. Watanabe's compact parameter space (Track 1):** Standard SLT assumes compact prior support $\Theta \subset \mathbb{R}^P$; on any compact ball $\|\theta\|_2 \le R$, width-2 $W$-tensor fitting is strictly unrealizable ($L^*(R) \asymp R^{-12} > 0$) with linear free energy $-\log Z \sim N L^*(R)$. The stretched-exponential class $-\log Z_W^{(2)}(N) \asymp N^{1/7}$ specifically requires a **non-compact Gaussian prior** ($\ell_2$ weight decay); general tails $e^{-c R^\alpha}$ give $N^{\alpha/(12+\alpha)}$, and heavy-tailed priors $R^{-\nu}$ yield prior-dependent $\Theta(\log N)$ scaling (see [`notes/bilinear_border_rank.md`](notes/bilinear_border_rank.md) §2.4).
2. **Formalization boundaries on Order-2 fibers (Track 2):**
   - Complete $(r^*, p, q)$ RRR fiber stratification for arbitrary invertible active block $D \in GL(r^*, \mathbb{R})$ (`RRRFiberStratification.lean`) and single-head `Q+K` flat cancellation for arbitrary orthogonal bases $U, V \in O(d)$ (`AttentionLoRA.lean`) are **`[LC]`**.
   - Non-orthogonal `V+O` row-space overlap (`q_eff = rank([A_{O,0}; A_{V,0}])`), multi-head decoupling ($n_{\mathrm{heads}} \ge 2$), and nonlinear MLP pullback invariance through a non-degenerate Gauss-Newton Fisher $\mathcal{H} \succ 0$ remain `[paper]` + `[empirical]` ([`notes/lora_nonlinear_fiber.md`](notes/lora_nonlinear_fiber.md)).
3. **Low-sample noisy fine-tuning nuance:** When finite noisy training data ($S = 48, \sigma = 0.25$) excites all $H$ LoRA ranks into fitting noise, there is no inactive zero-loss fiber ($k_{\mathrm{eff}} = 0$) and moderate decoupled AdamW shrinks $\|A_{\perp B}\|_F$ without improving test MSE; strong AdamW, coupled SGD+WD, or subcritical rank-$2$ initialization cut test MSE by $2\text{--}8\times$. Moreover, raw centered empirical LLC turns strongly negative (`-314.7`) at the generalizing rank-2 saddle, illustrating why functional KL LLC is essential ([`notes/lora_nonlinear_fiber.md`](notes/lora_nonlinear_fiber.md) §2.2).
4. **Finite-$\beta$ estimation & sampler choice:** Across all empirical sweeps, Metropolis-adjusted samplers (MALA / preconditioned MALA / SMC+HMC) are used because unadjusted SGLD incurs severe upward discretization bias on stiff supercritical directions (`36.81` to `54.54` vs. `32.00` on GELU MLP; [`notes/lora_nonlinear_fiber.md`](notes/lora_nonlinear_fiber.md) §1.3). On singular subcritical plateaus ($q < q_c$), finite-$\beta$ MALA retains the expected negative $O(1/\log(\beta/\gamma))$ offset that closes monotonically as $\beta/\gamma$ grows.

---

## Relation to SLT & ML literature

- **Watanabe (2009, 2018):** *Algebraic Geometry and Statistical Learning Theory* (CUP) and *Mathematical Theory of Bayesian Statistics* (CRC) — foundational SLT free-energy asymptotics, RLCTs $(\lambda, m)$, and WBIC.
- **Aoyagi & Watanabe (2005):** *"Stochastic complexities of reduced rank regression in Bayesian estimation,"* *Neural Networks* 18(7): 924–933 — Theorem 1 gives the exact RLCTs of matrix factorization, here unconditionally mechanized in Lean 4 via elementary projective blowups (`aoyagi_watanabe_rank0`).
- **Aoyagi (2024):** *"Consideration on the learning efficiency of multiple-layered neural networks with linear units,"* *Neural Networks* — global RLCTs of deep linear networks.
- **Lau, Furman, Wang, Murfet & Wei (2025):** *"The Local Learning Coefficient: A Singularity-Aware Complexity Measure,"* AISTATS 2025 ([`arXiv:2308.12108`](https://arxiv.org/abs/2308.12108)) and **Furman & Lau (2024)** ([`arXiv:2402.03698`](https://arxiv.org/abs/2402.03698)) — local learning coefficient theory and scalable estimation on deep linear and neural networks; our Track 2 maps exact local LLC phase transitions *within* a fixed-function zero-loss fiber.
- **Treisman (2025/2026):** [`ztreisman/arc-llc`](https://github.com/ztreisman/arc-llc) — arc-space computation of local RLCTs for reduced-rank regression via Smith normal form over $\mathbb{C}[[t]]$ and Docampo's formula, proposing the $(r^*, a, b)$ local RLCT formula as a candidate solution to **MAIS-O70**; our [`RRRFiberStratification.lean`](lean/EatYourFiber/RRRFiberStratification.lean) certifies this formula unconditionally over $\mathbb{R}$ in Lean 4 and proves its equivalence to minimizing irreducible fiber-component codimensions.
- **Hu et al. (2022):** *"LoRA: Low-Rank Adaptation of Large Language Models,"* ICLR 2022 ([`arXiv:2106.09685`](https://arxiv.org/abs/2106.09685)) — standard asymmetric $(B=0, A \sim \mathcal{N})$ initialization sits at the supercritical top $q = k$.
- **Sharkey (2023)** ([`arXiv:2305.03452`](https://arxiv.org/abs/2305.03452)) & **Pearce, Dooms, Rigg, Oramas & Sharkey (2024)** ([`arXiv:2410.08417`](https://arxiv.org/abs/2410.08417)) — bilinear MLPs and 3-tensor weight-based interpretability.
- **OpenAI (2026):** [`openai/math`](https://github.com/openai/math) (commit `3344da59991ed94f9ba711217b29c1d760badfa6`) — vendored Lean 4 tensor and geometric border-rank formalizations in `lean/Vendor/OAI/`.

---

## Layout

- `lean/` — Lean `v4.35.0` project depending on [`slt-bridge`](https://github.com/bubskee/slt-bridge) (`SLT`) and Mathlib:
  - `lean/EatYourFiber/` — SLT fiber-geometry certificates (`LoRASchur`, `LoRASchurLaplace`, `AoyagiWatanabeBoundary`, `AoyagiWatanabe`, `RRRFiberStratification`, `AttentionLoRA`, `BorderRankSLT`).
  - `lean/Audit.lean` — `#print axioms` verification for all 60 exported theorems.
  - `lean/Vendor/OAI/` — 8 self-contained Apache-2.0 tensor / border-rank / block-rank modules vendored from [`openai/math`](https://github.com/openai/math). See [NOTICE](NOTICE).
- `experiments/` — reproducible Python / PyTorch scripts (`uv`-managed) and saved JSON/PNG outputs in `experiments/results/`.
- `notes/` — mathematical derivations, epistemic status headers, and full experiment logs.

## Reproduce

### Lean 4 build & axiom audit

```sh
cd lean
lake exe cache get
lake build
lake env lean -DwarningAsError=true -DautoImplicit=false Audit.lean
```

### PyTorch experiments (Track 1 & Track 2)

```sh
cd experiments
uv run python bilinear_border_rank.py
uv run python lora_fiber_sgld.py
uv run python lora_sgd_dynamics.py
uv run python lora_nonlinear_fiber.py
```

## License & Attribution

Code and Lean proofs are licensed under the [Apache License 2.0](LICENSE). Vendored tensor and border-rank files in `lean/Vendor/OAI/` are adapted from OpenAI's [`openai/math`](https://github.com/openai/math) repository under Apache License 2.0; see [NOTICE](NOTICE) for full citation and commit provenance.
