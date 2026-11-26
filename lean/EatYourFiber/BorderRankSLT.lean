/-
Copyright (c) 2026 Bobby Faber. All rights reserved.
Released under Apache 2.0 license.
-/

import Vendor.OAI.LinearAlgebra.SpectralBooks.BorderRank
import EatYourFiber.AoyagiWatanabe
import Mathlib.Analysis.SpecialFunctions.Gaussian.GaussianIntegral
import Mathlib.Analysis.MeanInequalities
import Mathlib.Algebra.Polynomial.Eval.Defs
import Mathlib.Algebra.Polynomial.Roots
import Mathlib.FieldTheory.IsAlgClosed.Basic

/-!
# Order-3 Bilinear MLPs, Border-Rank Singularity, and Stretched-Exponential SLT

This module establishes the first kernel-certified (`[LC]`) connection between **3-tensor border
rank** (building directly on OpenAI's vendored `Tensor`, `tensorRank`, and `geometricBorderRank`
formalizations in `Vendor.OAI.LinearAlgebra.SpectralBooks.BorderRank`) and **Singular Learning
Theory (SLT)** for bilinear MLPs `T(A, B, C) = ∑_{r=1}^k a_r ⊗ b_r ⊗ c_r`:

1. **Bilinear MLP ↔ Tensor Bridge:**
   - `exists_zero_loss_iff_tensorRank_le`: a target 3-tensor `T*` has a nonempty finite zero-loss
     fiber for a width-`k` bilinear MLP iff `tensorRank T* ≤ k`.
   - `geometricBorderRank_le_of_approx_loss`: if a width-`k` bilinear MLP can approximate `T*` to
     arbitrarily small Frobenius loss (`inf L = 0`), then `geometricBorderRank T* ≤ k`.
2. **Exact Border-Rank Gap of the `W`-Tensor**
   `W = e₀⊗e₀⊗e₁ + e₀⊗e₁⊗e₀ + e₁⊗e₀⊗e₀ ∈ F^{2×2×2}` (over *every* field `F`):
   - `wTensor_geometricBorderRank`: `geometricBorderRank (wTensor F) = 2` (lower bound via OpenAI's
     `card_le_geometricBorderRank_of_minor`; upper bound via the algebraic border-rank curve).
   - `wTensor_tensorRank`: `tensorRank (wTensor F) = 3` (lower bound via the 2×2 adjugate/trace
     nilpotency obstruction).
3. **Sharp Quantitative Border-Rank Escape Law (`R^{-12}`):**
   - `wTensor_loss_ge_of_paramNormSq`: for *every* width-2 parameter `θ`,
     `min (1/64) (1 / (4624 * ‖θ‖₂^{12})) ≤ ‖T(θ) - W‖_F²`.
   - `wTensor_escape_upper`: matched by the **centered finite-difference** optimal border-rank path
     `W̃_ε = (2ε)⁻¹ ((e₀ + ε e₁)^{⊗3} - (e₀ - ε e₁)^{⊗3}) = W + ε² e₁^{⊗3}`, achieving
     `‖T(θ) - W‖_F² ≍ R^{-12}` (beating the textbook one-sided border-rank path `R^{-6}` by a
     factor of 2 in the exponent!).
4. **Exact Weight-Decay Scaling Law (`γ^{6/7}`):**
   - `wTensor_weightDecay_lower` & `wTensor_weightDecay_upper`:
     `inf_θ (‖T(θ) - W‖_F² + γ ‖θ‖₂²) ≍ γ^{6/7}` for all `γ ∈ (0, 1]`.
5. **Stretched-Exponential Bayesian Free Energy (`N^{1/7}`) vs Width-3 Polynomial Contrast:**
   - `wPartitionZ_two_le_stretchedExp` & `wPartitionZ_two_ge_stretchedExp`:
     for the Gaussian-prior global partition function `Z_W^{(2)}(N)`,
     `c₁ N^{1/7} ≤ -log Z_W^{(2)}(N) ≤ c₂ N^{1/7}` (a **third SLT universality class** strictly
     between realizable Watanabe `Θ(log N)` and unrealizable `Θ(N)`!).
   - `wPartitionZ_three_ge_poly`: adding a single neuron (`k = 2 → 3`) makes `W` realizable and
     collapses the free energy back to polynomial decay `Z_W^{(3)}(N) ≥ c₃ N^{-9}`
     (`-log Z_W^{(3)}(N) ≤ 9 log N + C₃`), even though **both** `k = 2` and `k = 3` have
     `inf_θ L(θ) = 0`!
-/

namespace EatYourFiber

open MeasureTheory Set Filter Finset Real
open OAI.MatrixMultiplication.Foundation
open OAI.LinearAlgebra.SpectralBooks
open scoped ENNReal NNReal BigOperators

/-! ### 1. Bilinear MLP ↔ OpenAI `Tensor` bridge -/

/-- Parameter space of a width-`k` bilinear MLP `(A, B, C) ∈ ℝ^{d₁×k} × ℝ^{d₂×k} × ℝ^{d₃×k}`. -/
abbrev BilinearParam (d1 d2 d3 k : ℕ) := RMat d1 k × RMat d2 k × RMat d3 k

/-- The 3-tensor computed by a width-`k` bilinear MLP `(A, B, C)`:
`T(A, B, C)_{i,j,l} = ∑_r A_{i,r} B_{j,r} C_{l,r}`. -/
def bilinearTensor {d1 d2 d3 k : ℕ} (θ : BilinearParam d1 d2 d3 k) :
    Tensor ℝ (Fin d1) (Fin d2) (Fin d3) :=
  fun i j l => ∑ r : Fin k, θ.1 i r * θ.2.1 j r * θ.2.2 l r

/-- Squared Frobenius norm of a real 3-tensor. -/
def tensorFrobSq {d1 d2 d3 : ℕ} (T : Tensor ℝ (Fin d1) (Fin d2) (Fin d3)) : ℝ :=
  ∑ i : Fin d1, ∑ j : Fin d2, ∑ l : Fin d3, (T i j l) ^ 2

/-- Squared Frobenius loss `‖T(θ) - T*‖_F²` of a bilinear MLP targeting `T*`. -/
def bilinearLoss {d1 d2 d3 k : ℕ} (Tstar : Tensor ℝ (Fin d1) (Fin d2) (Fin d3))
    (θ : BilinearParam d1 d2 d3 k) : ℝ :=
  tensorFrobSq (fun i j l => bilinearTensor θ i j l - Tstar i j l)

/-- Squared Euclidean norm `‖A‖_F² + ‖B‖_F² + ‖C‖_F²` of a bilinear MLP parameter vector. -/
def paramNormSq {d1 d2 d3 k : ℕ} (θ : BilinearParam d1 d2 d3 k) : ℝ :=
  (∑ i : Fin d1, ∑ r : Fin k, (θ.1 i r) ^ 2) +
  (∑ j : Fin d2, ∑ r : Fin k, (θ.2.1 j r) ^ 2) +
  (∑ l : Fin d3, ∑ r : Fin k, (θ.2.2 l r) ^ 2)

theorem tensorFrobSq_nonneg {d1 d2 d3 : ℕ} (T : Tensor ℝ (Fin d1) (Fin d2) (Fin d3)) :
    0 ≤ tensorFrobSq T :=
  Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ =>
    Finset.sum_nonneg fun _ _ => sq_nonneg _

theorem bilinearLoss_nonneg {d1 d2 d3 k : ℕ}
    (Tstar : Tensor ℝ (Fin d1) (Fin d2) (Fin d3)) (θ : BilinearParam d1 d2 d3 k) :
    0 ≤ bilinearLoss Tstar θ :=
  tensorFrobSq_nonneg _

theorem paramNormSq_nonneg {d1 d2 d3 k : ℕ} (θ : BilinearParam d1 d2 d3 k) :
    0 ≤ paramNormSq θ := by
  unfold paramNormSq
  positivity

theorem tensorFrobSq_eq_zero_iff {d1 d2 d3 : ℕ} (T : Tensor ℝ (Fin d1) (Fin d2) (Fin d3)) :
    tensorFrobSq T = 0 ↔ T = 0 := by
  constructor
  · intro h
    funext i j l
    have h1 : ∑ j' : Fin d2, ∑ l' : Fin d3, (T i j' l') ^ 2 = 0 := by
      have := (Finset.sum_eq_zero_iff_of_nonneg (fun i' _ =>
        Finset.sum_nonneg fun j' _ => Finset.sum_nonneg fun l' _ => sq_nonneg (T i' j' l'))).mp h
      exact this i (Finset.mem_univ i)
    have h2 : ∑ l' : Fin d3, (T i j l') ^ 2 = 0 := by
      have := (Finset.sum_eq_zero_iff_of_nonneg (fun j' _ =>
        Finset.sum_nonneg fun l' _ => sq_nonneg (T i j' l'))).mp h1
      exact this j (Finset.mem_univ j)
    have h3 : (T i j l) ^ 2 = 0 := by
      have := (Finset.sum_eq_zero_iff_of_nonneg (fun l' _ => sq_nonneg (T i j l'))).mp h2
      exact this l (Finset.mem_univ l)
    exact sq_eq_zero_iff.mp h3
  · rintro rfl
    simp [tensorFrobSq]

theorem sq_entry_le_tensorFrobSq {d1 d2 d3 : ℕ}
    (T : Tensor ℝ (Fin d1) (Fin d2) (Fin d3)) (i : Fin d1) (j : Fin d2) (l : Fin d3) :
    (T i j l) ^ 2 ≤ tensorFrobSq T := by
  have h1 : (T i j l) ^ 2 ≤ ∑ l' : Fin d3, (T i j l') ^ 2 :=
    Finset.single_le_sum (fun l' _ => sq_nonneg _) (Finset.mem_univ l)
  have h2 : ∑ l' : Fin d3, (T i j l') ^ 2 ≤ ∑ j' : Fin d2, ∑ l' : Fin d3, (T i j' l') ^ 2 :=
    Finset.single_le_sum (fun j' _ => Finset.sum_nonneg fun l' _ => sq_nonneg _) (Finset.mem_univ j)
  have h3 : ∑ j' : Fin d2, ∑ l' : Fin d3, (T i j' l') ^ 2 ≤ tensorFrobSq T :=
    Finset.single_le_sum (fun i' _ => Finset.sum_nonneg fun j' _ =>
      Finset.sum_nonneg fun l' _ => sq_nonneg _) (Finset.mem_univ i)
  exact h1.trans (h2.trans h3)

theorem bilinearTensor_rankAtMost {d1 d2 d3 k : ℕ} (θ : BilinearParam d1 d2 d3 k) :
    (bilinearTensor θ).RankAtMost k := by
  refine ⟨fun r i => θ.1 i r, fun r j => θ.2.1 j r, fun r l => θ.2.2 l r, ?_⟩
  funext i j l
  rfl

/-- A target tensor `T*` admits a finite zero-loss parameter at width `k` iff `T*.RankAtMost k`. -/
theorem exists_zero_loss_iff_rankAtMost {d1 d2 d3 : ℕ}
    (Tstar : Tensor ℝ (Fin d1) (Fin d2) (Fin d3)) (k : ℕ) :
    (∃ θ : BilinearParam d1 d2 d3 k, bilinearLoss Tstar θ = 0) ↔ Tstar.RankAtMost k := by
  constructor
  · rintro ⟨θ, hθ⟩
    have heq : bilinearTensor θ = Tstar := by
      have h0 := (tensorFrobSq_eq_zero_iff _).mp hθ
      funext i j l
      have := congrFun (congrFun (congrFun h0 i) j) l
      exact sub_eq_zero.mp this
    rw [← heq]
    exact bilinearTensor_rankAtMost θ
  · rintro ⟨a, b, c, hT⟩
    refine ⟨(fun i r => a r i, fun j r => b r j, fun l r => c r l), ?_⟩
    rw [bilinearLoss, tensorFrobSq_eq_zero_iff]
    funext i j l
    have := congrFun (congrFun (congrFun hT i) j) l
    simp only [bilinearTensor, Tensor.rankOne, Pi.zero_apply] at this ⊢
    linarith

/-- Exact rank-fiber bridge: a width-`k` bilinear MLP has a nonempty zero-loss fiber over `T*`
if and only if `tensorRank T* ≤ k`. -/
theorem exists_zero_loss_iff_tensorRank_le {d1 d2 d3 : ℕ}
    (Tstar : Tensor ℝ (Fin d1) (Fin d2) (Fin d3)) (k : ℕ) :
    (∃ θ : BilinearParam d1 d2 d3 k, bilinearLoss Tstar θ = 0) ↔ tensorRank Tstar ≤ k := by
  rw [exists_zero_loss_iff_rankAtMost]
  constructor
  · exact tensorRank_le_of_rankAtMost
  · intro hle
    rcases rankAtMost_tensorRank Tstar with ⟨a, b, c, hT⟩
    refine ⟨fun r => if hr : r.1 < tensorRank Tstar then a ⟨r.1, hr⟩ else 0,
            fun r => if hr : r.1 < tensorRank Tstar then b ⟨r.1, hr⟩ else 0,
            fun r => if hr : r.1 < tensorRank Tstar then c ⟨r.1, hr⟩ else 0, ?_⟩
    funext i j l
    rw [congrFun (congrFun (congrFun hT i) j) l]
    symm
    calc
      (∑ r : Fin k, Tensor.rankOne
          (if hr : r.1 < tensorRank Tstar then a ⟨r.1, hr⟩ else 0)
          (if hr : r.1 < tensorRank Tstar then b ⟨r.1, hr⟩ else 0)
          (if hr : r.1 < tensorRank Tstar then c ⟨r.1, hr⟩ else 0) i j l)
        = ∑ r ∈ (Finset.univ : Finset (Fin k)).filter (fun r => r.1 < tensorRank Tstar),
            Tensor.rankOne
              (if hr : r.1 < tensorRank Tstar then a ⟨r.1, hr⟩ else 0)
              (if hr : r.1 < tensorRank Tstar then b ⟨r.1, hr⟩ else 0)
              (if hr : r.1 < tensorRank Tstar then c ⟨r.1, hr⟩ else 0) i j l := by
          rw [← Finset.sum_filter_add_sum_filter_not (s := Finset.univ)
            (p := fun r : Fin k => r.1 < tensorRank Tstar)]
          have hzero : ∑ r ∈ Finset.univ.filter (fun r : Fin k => ¬ r.1 < tensorRank Tstar),
              Tensor.rankOne
                (if hr : r.1 < tensorRank Tstar then a ⟨r.1, hr⟩ else 0)
                (if hr : r.1 < tensorRank Tstar then b ⟨r.1, hr⟩ else 0)
                (if hr : r.1 < tensorRank Tstar then c ⟨r.1, hr⟩ else 0) i j l = 0 := by
            apply Finset.sum_eq_zero
            intro r hr
            have hnr : ¬ r.1 < tensorRank Tstar := (Finset.mem_filter.mp hr).2
            simp [hnr, Tensor.rankOne]
          rw [hzero, add_zero]
      _ = ∑ r : Fin (tensorRank Tstar), Tensor.rankOne (a r) (b r) (c r) i j l := by
          refine Finset.sum_bij
            (fun (r : Fin k) (hr : r ∈ Finset.univ.filter (fun r : Fin k => r.1 < tensorRank Tstar)) =>
              (⟨r.1, (Finset.mem_filter.mp hr).2⟩ : Fin (tensorRank Tstar)))
            (fun _ _ => Finset.mem_univ _)
            (fun r₁ _ r₂ _ h => Fin.ext (congrArg (Fin.val : Fin (tensorRank Tstar) → ℕ) h))
            (fun s _ => ⟨⟨s.1, lt_of_lt_of_le s.2 hle⟩, by simp [s.2], by ext; rfl⟩)
            (fun r hr => by
              have hr' : r.1 < tensorRank Tstar := (Finset.mem_filter.mp hr).2
              simp [hr', Tensor.rankOne])

/-- Border-rank bridge: if a width-`k` bilinear MLP can approximate `T*` to arbitrarily small
Frobenius loss (`inf L = 0`), then `geometricBorderRank T* ≤ k`. -/
theorem geometricBorderRank_le_of_approx_loss {d1 d2 d3 : ℕ}
    (Tstar : Tensor ℝ (Fin d1) (Fin d2) (Fin d3)) (k : ℕ)
    (happrox : ∀ ε > 0, ∃ θ : BilinearParam d1 d2 d3 k, bilinearLoss Tstar θ < ε) :
    geometricBorderRank Tstar ≤ k := by
  apply geometricBorderRank_le_of_atMost
  intro p hp
  -- Embed `AlgebraicClosure ℝ →ₐ[ℝ] ℂ` via `IsAlgClosed.lift` so we can pass to the limit in `ℂ`.
  let φ : AlgebraicClosure ℝ →ₐ[ℝ] ℂ := IsAlgClosed.lift
  let pC : MvPolynomial (Fin d1 × Fin d2 × Fin d3) ℂ := MvPolynomial.map φ.toRingHom p
  have hcomp : ∀ T : Tensor ℝ (Fin d1) (Fin d2) (Fin d3),
      φ (tensorEval (toAlgebraicClosure T) p) =
        MvPolynomial.eval (fun q => (T q.1 q.2.1 q.2.2 : ℂ)) pC := by
    intro T
    simp only [tensorEval, toAlgebraicClosure, pC]
    change φ.toRingHom (MvPolynomial.eval₂ (RingHom.id _) (fun q => algebraMap ℝ _ (T q.1 q.2.1 q.2.2)) p) =
      MvPolynomial.eval (fun q => (T q.1 q.2.1 q.2.2 : ℂ)) (MvPolynomial.map φ.toRingHom p)
    rw [MvPolynomial.hom_eval₂, MvPolynomial.eval_map]
    congr 1
    funext q
    change φ (algebraMap ℝ (AlgebraicClosure ℝ) (T q.1 q.2.1 q.2.2)) = _
    rw [φ.commutes]
    rfl
  -- Choose approximating sequence `θ_n` with `bilinearLoss Tstar (θ n) < 1 / (n + 1)^2`.
  choose θ hθ using fun n : ℕ =>
    happrox (1 / ((n : ℝ) + 1) ^ 2) (by positivity)
  have htend_entry : ∀ q : Fin d1 × Fin d2 × Fin d3,
      Tendsto (fun n : ℕ => (bilinearTensor (θ n) q.1 q.2.1 q.2.2 : ℂ))
        atTop (nhds (Tstar q.1 q.2.1 q.2.2 : ℂ)) := by
    intro q
    have hreal : Tendsto (fun n : ℕ => bilinearTensor (θ n) q.1 q.2.1 q.2.2)
        atTop (nhds (Tstar q.1 q.2.1 q.2.2)) := by
      rw [Metric.tendsto_atTop]
      intro ε hε
      obtain ⟨N, hN⟩ := exists_nat_gt (1 / ε)
      refine ⟨N, fun n hn => ?_⟩
      rw [Real.dist_eq]
      have hsq : (bilinearTensor (θ n) q.1 q.2.1 q.2.2 - Tstar q.1 q.2.1 q.2.2) ^ 2 <
          (1 / ((n : ℝ) + 1)) ^ 2 := by
        calc
          (bilinearTensor (θ n) q.1 q.2.1 q.2.2 - Tstar q.1 q.2.1 q.2.2) ^ 2
            ≤ bilinearLoss Tstar (θ n) :=
              sq_entry_le_tensorFrobSq (fun i j l => bilinearTensor (θ n) i j l - Tstar i j l)
                q.1 q.2.1 q.2.2
          _ < 1 / ((n : ℝ) + 1) ^ 2 := hθ n
          _ = (1 / ((n : ℝ) + 1)) ^ 2 := by rw [one_div, one_div, inv_pow]
      have habs : |bilinearTensor (θ n) q.1 q.2.1 q.2.2 - Tstar q.1 q.2.1 q.2.2| <
          1 / ((n : ℝ) + 1) := by
        have hpos : (0 : ℝ) < 1 / ((n : ℝ) + 1) := by positivity
        nlinarith [sq_abs (bilinearTensor (θ n) q.1 q.2.1 q.2.2 - Tstar q.1 q.2.1 q.2.2),
          abs_nonneg (bilinearTensor (θ n) q.1 q.2.1 q.2.2 - Tstar q.1 q.2.1 q.2.2)]
      have hlt : 1 / ((n : ℝ) + 1) < ε := by
        have hn' : (N : ℝ) ≤ n := Nat.cast_le.mpr hn
        have h1 : 1 / ε < (n : ℝ) + 1 := by linarith
        rw [div_lt_iff₀ (by positivity : (0 : ℝ) < (n : ℝ) + 1)]
        rw [div_lt_iff₀ hε] at h1
        linarith
      exact habs.trans hlt
    exact Complex.continuous_ofReal.continuousAt.tendsto.comp hreal
  have htend_eval : Tendsto
      (fun n : ℕ => MvPolynomial.eval (fun q => (bilinearTensor (θ n) q.1 q.2.1 q.2.2 : ℂ)) pC)
      atTop (nhds (MvPolynomial.eval (fun q => (Tstar q.1 q.2.1 q.2.2 : ℂ)) pC)) := by
    have hcont : Continuous (fun f : Fin d1 × Fin d2 × Fin d3 → ℂ => MvPolynomial.eval f pC) :=
      MvPolynomial.continuous_eval pC
    exact hcont.continuousAt.tendsto.comp (tendsto_pi_nhds.mpr htend_entry)
  have hzero_seq : ∀ n : ℕ,
      MvPolynomial.eval (fun q => (bilinearTensor (θ n) q.1 q.2.1 q.2.2 : ℂ)) pC = 0 := by
    intro n
    rw [← hcomp (bilinearTensor (θ n))]
    have hrank : (toAlgebraicClosure (bilinearTensor (θ n))).RankAtMost k :=
      (bilinearTensor_rankAtMost (θ n)).map (algebraMap ℝ (AlgebraicClosure ℝ))
    rw [hp _ hrank, map_zero]
  have hzero_lim : MvPolynomial.eval (fun q => (Tstar q.1 q.2.1 q.2.2 : ℂ)) pC = 0 := by
    have htend0 : Tendsto
        (fun n : ℕ => MvPolynomial.eval (fun q => (bilinearTensor (θ n) q.1 q.2.1 q.2.2 : ℂ)) pC)
        atTop (nhds 0) := by
      simp only [hzero_seq]
      exact tendsto_const_nhds
    exact tendsto_nhds_unique htend_eval htend0
  rw [← hcomp Tstar] at hzero_lim
  exact φ.toRingHom.injective (by simpa using hzero_lim)

/-! ### 2. Exact algebraic border-rank gap of the `W`-tensor (over every field `F`) -/

/-- The canonical 2×2×2 `W`-tensor `W = e₀⊗e₀⊗e₁ + e₀⊗e₁⊗e₀ + e₁⊗e₀⊗e₀`. -/
def wTensor (F : Type*) [Field F] : Tensor F (Fin 2) (Fin 2) (Fin 2) :=
  fun i j l =>
    if (i = 0 ∧ j = 0 ∧ l = 1) ∨ (i = 0 ∧ j = 1 ∧ l = 0) ∨ (i = 1 ∧ j = 0 ∧ l = 0) then 1 else 0

/-- Lower bound `2 ≤ geometricBorderRank (wTensor F)` directly via OpenAI's
`card_le_geometricBorderRank_of_minor` on the `l = 0` flattening slice `[[0, 1], [1, 0]]`. -/
theorem two_le_wTensor_geometricBorderRank (F : Type*) [Field F] :
    2 ≤ geometricBorderRank (wTensor F) := by
  let entry : Fin 2 → Fin 2 → Fin 2 × Fin 2 × Fin 2 := fun i j => (i, j, 0)
  have hne : (Matrix.of fun i j => wTensor F (entry i j).1 (entry i j).2.1 (entry i j).2.2).det ≠ 0 := by
    rw [Matrix.det_fin_two]
    simp [entry, wTensor]
  have hvanish : ∀ (S : Tensor (AlgebraicClosure F) (Fin 2) (Fin 2) (Fin 2)) (r : ℕ),
      S.RankAtMost r → r < Fintype.card (Fin 2) →
      (Matrix.of fun i j => S (entry i j).1 (entry i j).2.1 (entry i j).2.2).det = 0 := by
    intro S r hS hr
    simp only [Fintype.card_fin] at hr
    rcases hS with ⟨a, b, c, rfl⟩
    rw [Matrix.det_fin_two]
    interval_cases r
    · simp [entry, Tensor.rankOne]
    · simp only [Matrix.of_apply, entry, Fin.sum_univ_one, Tensor.rankOne]
      ring
  simpa only [Fintype.card_fin] using
    card_le_geometricBorderRank_of_minor (wTensor F) entry hne hvanish

/-- Every algebraically closed field contains an element outside any finite set (Euclid). -/
private theorem exists_notMem_finset_of_isAlgClosed {K : Type*} [Field K] [IsAlgClosed K]
    (s : Finset K) : ∃ t : K, t ∉ s := by
  classical
  let s' : Finset K := insert (0 : K) s
  let r : Polynomial K := (∏ a ∈ s', (Polynomial.X - Polynomial.C a)) + 1
  have hr_deg : r.degree ≠ 0 := by
    have hprod_monic : (∏ a ∈ s', (Polynomial.X - Polynomial.C a)).Monic :=
      Polynomial.monic_prod_of_monic _ _ fun a _ => Polynomial.monic_X_sub_C a
    have hdeg_prod : (∏ a ∈ s', (Polynomial.X - Polynomial.C a)).natDegree = s'.card := by
      rw [Polynomial.natDegree_prod_of_monic _ _ fun a _ => Polynomial.monic_X_sub_C a]
      simp [Polynomial.natDegree_X_sub_C]
    have hcard_pos : 0 < s'.card := Finset.card_pos.mpr ⟨0, Finset.mem_insert_self 0 s⟩
    have hdeg_r : r.degree = (s'.card : WithBot ℕ) := by
      rw [show r = (∏ a ∈ s', (Polynomial.X - Polynomial.C a)) + 1 from rfl,
        Polynomial.degree_add_eq_left_of_degree_lt]
      · rw [Polynomial.degree_eq_natDegree hprod_monic.ne_zero, hdeg_prod]
      · rw [Polynomial.degree_one, Polynomial.degree_eq_natDegree hprod_monic.ne_zero, hdeg_prod]
        exact_mod_cast hcard_pos
    rw [hdeg_r]
    intro h0
    have : s'.card = 0 := by exact_mod_cast h0
    omega
  obtain ⟨t, ht⟩ := IsAlgClosed.exists_root r hr_deg
  refine ⟨t, fun hts => ?_⟩
  have hts' : t ∈ s' := Finset.mem_insert_of_mem hts
  have hprod0 : (∏ a ∈ s', (Polynomial.X - Polynomial.C a)).eval t = 0 := by
    rw [Polynomial.eval_prod]
    exact Finset.prod_eq_zero hts' (by simp)
  have h1 : r.eval t = 1 := by simp [r, hprod0]
  rw [Polynomial.IsRoot.def] at ht
  rw [ht] at h1
  exact zero_ne_one h1

/-- Upper bound `geometricBorderRank (wTensor F) ≤ 2` over *every* field `F`, using the
1-parameter algebraic border-rank curve `W_t = t⁻¹ ((e₀ + t e₁)^{⊗3} - e₀^{⊗3})`. -/
theorem wTensor_geometricBorderRank_le_two (F : Type*) [Field F] :
    geometricBorderRank (wTensor F) ≤ 2 := by
  classical
  apply geometricBorderRank_le_of_atMost
  intro p hp
  let K := AlgebraicClosure F
  -- Polynomial curve `W_X ∈ (K[X])^{2×2×2}`:
  -- `W_X(i,j,l) = W(i,j,l) + X*(weight-2) + X^2*(e₁⊗e₁⊗e₁)`.
  let Wpoly : Fin 2 × Fin 2 × Fin 2 → Polynomial K := fun q =>
    match q.1.1, q.2.1.1, q.2.2.1 with
    | 0, 0, 0 => 0
    | 0, 0, 1 => 1
    | 0, 1, 0 => 1
    | 1, 0, 0 => 1
    | 0, 1, 1 => Polynomial.X
    | 1, 0, 1 => Polynomial.X
    | 1, 1, 0 => Polynomial.X
    | 1, 1, 1 => Polynomial.X ^ 2
    | _, _, _ => 0
  let q : Polynomial K := MvPolynomial.eval₂ Polynomial.C Wpoly p
  -- Evaluating `q` at any `t : K` equals `tensorEval (S t) p`.
  have hq_eval : ∀ t : K,
      q.eval t = tensorEval (fun i j l => (Wpoly (i, j, l)).eval t) p := by
    intro t
    change (Polynomial.evalRingHom t) (MvPolynomial.eval₂ Polynomial.C Wpoly p) =
      MvPolynomial.eval₂ (RingHom.id _) (fun q => (Wpoly q).eval t) p
    rw [MvPolynomial.hom_eval₂]
    congr 1
    ext x
    simp
  -- For every `t ≠ 0`, `S t` has rank ≤ 2 via `t⁻¹ ((e₀ + t e₁)^{⊗3} - e₀^{⊗3})`.
  have hq_zero_ne : ∀ t : K, t ≠ 0 → q.eval t = 0 := by
    intro t ht
    rw [hq_eval t]
    apply hp
    refine ⟨fun r i => if r = 0 then (if i = 0 then t⁻¹ else 1) else (if i = 0 then -t⁻¹ else 0),
            fun r j => if r = 0 then (if j = 0 then 1 else t) else (if j = 0 then 1 else 0),
            fun r l => if r = 0 then (if l = 0 then 1 else t) else (if l = 0 then 1 else 0), ?_⟩
    funext i j l
    fin_cases i <;> fin_cases j <;> fin_cases l <;>
      simp [Wpoly, Tensor.rankOne, Fin.sum_univ_two, ht] <;> (try field_simp; try ring)
  -- Since `K = AlgebraicClosure F` is algebraically closed (hence infinite), `q = 0`.
  have hq_zero : q = 0 := by
    by_contra hne
    obtain ⟨t, ht⟩ := exists_notMem_finset_of_isAlgClosed (insert (0 : K) q.roots.toFinset)
    have ht0 : t ≠ 0 := fun h => ht (Finset.mem_insert.mpr (Or.inl h))
    have htr : t ∉ q.roots := fun h =>
      ht (Finset.mem_insert.mpr (Or.inr (Multiset.mem_toFinset.mpr h)))
    exact htr ((Polynomial.mem_roots hne).mpr (by simpa [Polynomial.IsRoot] using hq_zero_ne t ht0))
  -- Evaluating at `t = 0` recovers `toAlgebraicClosure (wTensor F)`.
  have hW0 : (fun i j l => (Wpoly (i, j, l)).eval 0) = toAlgebraicClosure (wTensor F) := by
    funext i j l
    fin_cases i <;> fin_cases j <;> fin_cases l <;>
      simp [Wpoly, toAlgebraicClosure, wTensor]
  rw [← hW0, ← hq_eval 0, hq_zero, Polynomial.eval_zero]

/-- **Exact geometric border rank of `W` (`[LC]`):** `geometricBorderRank (wTensor F) = 2`
over every field `F`. -/
theorem wTensor_geometricBorderRank (F : Type*) [Field F] :
    geometricBorderRank (wTensor F) = 2 :=
  le_antisymm (wTensor_geometricBorderRank_le_two F) (two_le_wTensor_geometricBorderRank F)

/-- Key 2×2 slice identities for any rank-2 tensor `T = a₀⊗b₀⊗c₀ + a₁⊗b₁⊗c₁` over any
commutative ring: `det(M₀)`, `det(M₁)`, and `tr(adj(M₀) M₁)` factor cleanly. -/
theorem rank2_slice_identities {R : Type*} [CommRing R]
    (a00 a01 a10 a11 b00 b01 b10 b11 c00 c01 c10 c11 : R) :
    let m000 := a00 * b00 * c00 + a01 * b01 * c01
    let m001 := a00 * b10 * c00 + a01 * b11 * c01
    let m010 := a10 * b00 * c00 + a11 * b01 * c01
    let m011 := a10 * b10 * c00 + a11 * b11 * c01
    let m100 := a00 * b00 * c10 + a01 * b01 * c11
    let m101 := a00 * b10 * c10 + a01 * b11 * c11
    let m110 := a10 * b00 * c10 + a11 * b01 * c11
    let m111 := a10 * b10 * c10 + a11 * b11 * c11
    let dAB := (a00 * a11 - a01 * a10) * (b00 * b11 - b01 * b10)
    (m000 * m011 - m001 * m010 = dAB * (c00 * c01)) ∧
    (m100 * m111 - m101 * m110 = dAB * (c10 * c11)) ∧
    (m011 * m100 - m001 * m110 - m010 * m101 + m000 * m111 =
      dAB * (c01 * c10 + c00 * c11)) := by
  refine ⟨by ring, by ring, by ring⟩

/-- **Exact tensor rank of `W` (`[LC]`):** `tensorRank (wTensor F) = 3` over every field `F`. -/
theorem wTensor_tensorRank (F : Type*) [Field F] :
    tensorRank (wTensor F) = 3 := by
  apply le_antisymm
  · -- Upper bound `≤ 3`: explicit 3-term decomposition `e₀⊗e₀⊗e₁ + e₀⊗e₁⊗e₀ + e₁⊗e₀⊗e₀`.
    apply tensorRank_le_of_rankAtMost
    refine ⟨fun r i => if r = 2 then (if i = 1 then 1 else 0) else (if i = 0 then 1 else 0),
            fun r j => if r = 1 then (if j = 1 then 1 else 0) else (if j = 0 then 1 else 0),
            fun r l => if r = 0 then (if l = 1 then 1 else 0) else (if l = 0 then 1 else 0), ?_⟩
    funext i j l
    fin_cases i <;> fin_cases j <;> fin_cases l <;>
      simp [wTensor, Tensor.rankOne, Fin.sum_univ_three]
  · -- Lower bound `≥ 3`: if `RankAtMost (wTensor F) 2`, the 2×2 slice identities force `c10 = c11 = 0`.
    by_contra! hlt
    have hle2 : tensorRank (wTensor F) ≤ 2 := Nat.le_of_lt_succ hlt
    have hr2 : (wTensor F).RankAtMost 2 := by
      rcases rankAtMost_tensorRank (wTensor F) with ⟨a, b, c, hT⟩
      refine ⟨fun r => if hr : r.1 < tensorRank (wTensor F) then a ⟨r.1, hr⟩ else 0,
              fun r => if hr : r.1 < tensorRank (wTensor F) then b ⟨r.1, hr⟩ else 0,
              fun r => if hr : r.1 < tensorRank (wTensor F) then c ⟨r.1, hr⟩ else 0, ?_⟩
      funext i j l
      rw [congrFun (congrFun (congrFun hT i) j) l]
      interval_cases h : tensorRank (wTensor F)
      · simp [Tensor.rankOne]
      · simp [Fin.sum_univ_two, Tensor.rankOne]
      · simp [Fin.sum_univ_two, Tensor.rankOne]
    rcases hr2 with ⟨a, b, c, hW⟩
    have hentry : ∀ i j l : Fin 2,
        wTensor F i j l = a 0 i * b 0 j * c 0 l + a 1 i * b 1 j * c 1 l := by
      intro i j l
      simpa [Tensor.rankOne, Fin.sum_univ_two] using congrFun (congrFun (congrFun hW i) j) l
    have h000 : a 0 0 * b 0 0 * c 0 0 + a 1 0 * b 1 0 * c 1 0 = 0 := by
      simpa [wTensor] using (hentry 0 0 0).symm
    have h001 : a 0 0 * b 0 1 * c 0 0 + a 1 0 * b 1 1 * c 1 0 = 1 := by
      simpa [wTensor] using (hentry 0 1 0).symm
    have h010 : a 0 1 * b 0 0 * c 0 0 + a 1 1 * b 1 0 * c 1 0 = 1 := by
      simpa [wTensor] using (hentry 1 0 0).symm
    have h011 : a 0 1 * b 0 1 * c 0 0 + a 1 1 * b 1 1 * c 1 0 = 0 := by
      simpa [wTensor] using (hentry 1 1 0).symm
    have h100 : a 0 0 * b 0 0 * c 0 1 + a 1 0 * b 1 0 * c 1 1 = 1 := by
      simpa [wTensor] using (hentry 0 0 1).symm
    have h101 : a 0 0 * b 0 1 * c 0 1 + a 1 0 * b 1 1 * c 1 1 = 0 := by
      simpa [wTensor] using (hentry 0 1 1).symm
    have h110 : a 0 1 * b 0 0 * c 0 1 + a 1 1 * b 1 0 * c 1 1 = 0 := by
      simpa [wTensor] using (hentry 1 0 1).symm
    have h111 : a 0 1 * b 0 1 * c 0 1 + a 1 1 * b 1 1 * c 1 1 = 0 := by
      simpa [wTensor] using (hentry 1 1 1).symm
    obtain ⟨hd0, hd1, htr⟩ := rank2_slice_identities
      (a 0 0) (a 1 0) (a 0 1) (a 1 1)
      (b 0 0) (b 1 0) (b 0 1) (b 1 1)
      (c 0 0) (c 1 0) (c 0 1) (c 1 1)
    rw [h000, h001, h010, h011] at hd0
    rw [h100, h101, h110, h111] at hd1
    rw [h000, h001, h010, h011, h100, h101, h110, h111] at htr
    set dAB := (a 0 0 * a 1 1 - a 1 0 * a 0 1) * (b 0 0 * b 1 1 - b 1 0 * b 0 1)
    have hdAB_ne : dAB ≠ 0 := by
      intro h0; simp [h0] at hd0
    have hc00_ne : c 0 0 ≠ 0 := by
      intro h0; simp [h0] at hd0
    have hc10_ne : c 1 0 ≠ 0 := by
      intro h0; simp [h0] at hd0
    have hprod0 : c 0 1 * c 1 1 = 0 := by
      have : dAB * (c 0 1 * c 1 1) = 0 := by linear_combination -hd1
      exact (mul_eq_zero.mp this).resolve_left hdAB_ne
    have hsum0 : c 1 0 * c 0 1 + c 0 0 * c 1 1 = 0 := by
      have : dAB * (c 1 0 * c 0 1 + c 0 0 * c 1 1) = 0 := by linear_combination -htr
      exact (mul_eq_zero.mp this).resolve_left hdAB_ne
    have hc01_0 : c 0 1 = 0 := by
      rcases mul_eq_zero.mp hprod0 with h | h
      · exact h
      · rw [h, mul_zero, add_zero] at hsum0
        exact (mul_eq_zero.mp hsum0).resolve_left hc10_ne
    have hc11_0 : c 1 1 = 0 := by
      rw [hc01_0, mul_zero, zero_add] at hsum0
      exact (mul_eq_zero.mp hsum0).resolve_left hc00_ne
    rw [hc01_0, hc11_0] at h100
    norm_num at h100

/-- **Strict Border-Rank Gap (`[LC]`):** `geometricBorderRank (wTensor F) < tensorRank (wTensor F)`. -/
theorem wTensor_borderRank_lt_tensorRank (F : Type*) [Field F] :
    geometricBorderRank (wTensor F) < tensorRank (wTensor F) := by
  rw [wTensor_geometricBorderRank, wTensor_tensorRank]
  norm_num

/-! ### 3. Sharp quantitative escape law (`R^{-12}`) and exact weight-decay law (`γ^{6/7}`) -/

private theorem entry_sq_le_paramNormSq_1 (θ : BilinearParam 2 2 2 2) (i r : Fin 2) :
    (θ.1 i r) ^ 2 ≤ paramNormSq θ := by
  have h1 : (θ.1 i r) ^ 2 ≤ ∑ r' : Fin 2, (θ.1 i r') ^ 2 :=
    Finset.single_le_sum (fun _ _ => sq_nonneg _) (Finset.mem_univ r)
  have h2 : ∑ r' : Fin 2, (θ.1 i r') ^ 2 ≤ ∑ i' : Fin 2, ∑ r' : Fin 2, (θ.1 i' r') ^ 2 :=
    Finset.single_le_sum (fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _) (Finset.mem_univ i)
  have h3 : 0 ≤ (∑ j : Fin 2, ∑ r' : Fin 2, (θ.2.1 j r') ^ 2) +
      (∑ l : Fin 2, ∑ r' : Fin 2, (θ.2.2 l r') ^ 2) := by positivity
  unfold paramNormSq
  linarith

private theorem entry_sq_le_paramNormSq_2 (θ : BilinearParam 2 2 2 2) (j r : Fin 2) :
    (θ.2.1 j r) ^ 2 ≤ paramNormSq θ := by
  have h1 : (θ.2.1 j r) ^ 2 ≤ ∑ r' : Fin 2, (θ.2.1 j r') ^ 2 :=
    Finset.single_le_sum (fun _ _ => sq_nonneg _) (Finset.mem_univ r)
  have h2 : ∑ r' : Fin 2, (θ.2.1 j r') ^ 2 ≤ ∑ j' : Fin 2, ∑ r' : Fin 2, (θ.2.1 j' r') ^ 2 :=
    Finset.single_le_sum (fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _) (Finset.mem_univ j)
  have h3 : 0 ≤ (∑ i : Fin 2, ∑ r' : Fin 2, (θ.1 i r') ^ 2) := by positivity
  have h4 : 0 ≤ (∑ l : Fin 2, ∑ r' : Fin 2, (θ.2.2 l r') ^ 2) := by positivity
  unfold paramNormSq
  linarith

private theorem entry_sq_le_paramNormSq_3 (θ : BilinearParam 2 2 2 2) (l r : Fin 2) :
    (θ.2.2 l r) ^ 2 ≤ paramNormSq θ := by
  have h1 : (θ.2.2 l r) ^ 2 ≤ ∑ r' : Fin 2, (θ.2.2 l r') ^ 2 :=
    Finset.single_le_sum (fun _ _ => sq_nonneg _) (Finset.mem_univ r)
  have h2 : ∑ r' : Fin 2, (θ.2.2 l r') ^ 2 ≤ ∑ l' : Fin 2, ∑ r' : Fin 2, (θ.2.2 l' r') ^ 2 :=
    Finset.single_le_sum (fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _) (Finset.mem_univ l)
  have h3 : 0 ≤ (∑ i : Fin 2, ∑ r' : Fin 2, (θ.1 i r') ^ 2) +
      (∑ j : Fin 2, ∑ r' : Fin 2, (θ.2.1 j r') ^ 2) := by positivity
  unfold paramNormSq
  linarith

private theorem abs_mul_le_mul_of_abs_le {x y a b : ℝ} (hx : |x| ≤ a) (hy : |y| ≤ b) :
    |x * y| ≤ a * b := by
  rw [abs_mul]
  exact mul_le_mul hx hy (abs_nonneg y) ((abs_nonneg x).trans hx)

/-- **Sharp Quantitative Border-Rank Escape Lower Bound (`[LC]`):**
For *every* width-2 bilinear MLP parameter `θ ∈ ℝ^{12}`,
`min (1/64) (1 / (4624 * (paramNormSq θ)^6)) ≤ bilinearLoss (wTensor ℝ) θ`.
In particular, for `‖θ‖₂ = R ≥ 1`, `‖T(θ) - W‖_F² ≥ 1 / (4624 R^{12})`. -/
theorem wTensor_loss_ge_of_paramNormSq (θ : BilinearParam 2 2 2 2) :
    min ((1 : ℝ) / 64) (1 / (4624 * (paramNormSq θ) ^ 6)) ≤ bilinearLoss (wTensor ℝ) θ := by
  by_cases hbig : (1 : ℝ) / 64 ≤ bilinearLoss (wTensor ℝ) θ
  · exact le_trans (min_le_left _ _) hbig
  push Not at hbig
  apply le_trans (min_le_right _ _)
  set L := bilinearLoss (wTensor ℝ) θ
  have hL_nonneg : 0 ≤ L := bilinearLoss_nonneg _ _
  set δ := Real.sqrt L
  have hδ_nonneg : 0 ≤ δ := Real.sqrt_nonneg L
  have hδ_sq : δ ^ 2 = L := Real.sq_sqrt hL_nonneg
  have hδ_le : δ ≤ 1 / 8 := by
    have : δ ^ 2 ≤ (1 / 8 : ℝ) ^ 2 := by linarith
    nlinarith [hδ_nonneg]
  -- Entrywise residuals `|T(θ)_{ijl} - W_{ijl}| ≤ δ`.
  have herr : ∀ i j l : Fin 2, |bilinearTensor θ i j l - wTensor ℝ i j l| ≤ δ := by
    intro i j l
    have hsq : (bilinearTensor θ i j l - wTensor ℝ i j l) ^ 2 ≤ δ ^ 2 := by
      rw [hδ_sq]
      exact sq_entry_le_tensorFrobSq (fun i' j' l' => bilinearTensor θ i' j' l' - wTensor ℝ i' j' l')
        i j l
    have hsq' : |bilinearTensor θ i j l - wTensor ℝ i j l| ^ 2 ≤ δ ^ 2 := by
      rwa [sq_abs]
    nlinarith [abs_nonneg (bilinearTensor θ i j l - wTensor ℝ i j l), hδ_nonneg]
  -- Unpack the 8 entries of `M0` (l=0) and `M1` (l=1).
  set a00 := θ.1 0 0; set a01 := θ.1 0 1; set a10 := θ.1 1 0; set a11 := θ.1 1 1
  set b00 := θ.2.1 0 0; set b01 := θ.2.1 0 1; set b10 := θ.2.1 1 0; set b11 := θ.2.1 1 1
  set c00 := θ.2.2 0 0; set c01 := θ.2.2 0 1; set c10 := θ.2.2 1 0; set c11 := θ.2.2 1 1
  set m000 := a00 * b00 * c00 + a01 * b01 * c01
  set m001 := a00 * b10 * c00 + a01 * b11 * c01
  set m010 := a10 * b00 * c00 + a11 * b01 * c01
  set m011 := a10 * b10 * c00 + a11 * b11 * c01
  set m100 := a00 * b00 * c10 + a01 * b01 * c11
  set m101 := a00 * b10 * c10 + a01 * b11 * c11
  set m110 := a10 * b00 * c10 + a11 * b01 * c11
  set m111 := a10 * b10 * c10 + a11 * b11 * c11
  have he00 : |m000| ≤ δ := by simpa [bilinearTensor, wTensor, Fin.sum_univ_two, m000] using herr 0 0 0
  have he01 : |m001 - 1| ≤ δ := by simpa [bilinearTensor, wTensor, Fin.sum_univ_two, m001] using herr 0 1 0
  have he10 : |m010 - 1| ≤ δ := by simpa [bilinearTensor, wTensor, Fin.sum_univ_two, m010] using herr 1 0 0
  have he11 : |m011| ≤ δ := by simpa [bilinearTensor, wTensor, Fin.sum_univ_two, m011] using herr 1 1 0
  have hf00 : |m100 - 1| ≤ δ := by simpa [bilinearTensor, wTensor, Fin.sum_univ_two, m100] using herr 0 0 1
  have hf01 : |m101| ≤ δ := by simpa [bilinearTensor, wTensor, Fin.sum_univ_two, m101] using herr 0 1 1
  have hf10 : |m110| ≤ δ := by simpa [bilinearTensor, wTensor, Fin.sum_univ_two, m110] using herr 1 0 1
  have hf11 : |m111| ≤ δ := by simpa [bilinearTensor, wTensor, Fin.sum_univ_two, m111] using herr 1 1 1
  have hm001_abs : |m001| ≤ 9 / 8 := by
    have := abs_le.mp he01; rw [abs_le]; constructor <;> linarith
  have hm010_abs : |m010| ≤ 9 / 8 := by
    have := abs_le.mp he10; rw [abs_le]; constructor <;> linarith
  have hm100_abs : |m100| ≤ 9 / 8 := by
    have := abs_le.mp hf00; rw [abs_le]; constructor <;> linarith
  have hm000_abs8 : |m000| ≤ 1 / 8 := he00.trans hδ_le
  have hm101_abs8 : |m101| ≤ 1 / 8 := hf01.trans hδ_le
  -- Bound `det M0`, `det M1`, `tr(adj(M0) M1)` cleanly using triangle inequality.
  set detM0 := m000 * m011 - m001 * m010
  set detM1 := m100 * m111 - m101 * m110
  set trAdj := m011 * m100 - m001 * m110 - m010 * m101 + m000 * m111
  have hdetM0_le : detM0 ≤ -(1 / 2 : ℝ) := by
    have h1 : m000 * m011 ≤ δ ^ 2 := by
      have := abs_mul_le_mul_of_abs_le he00 he11
      rw [← sq] at this
      exact (abs_le.mp this).2
    have h01l : (7 / 8 : ℝ) ≤ m001 := by linarith [(abs_le.mp he01).1]
    have h10l : (7 / 8 : ℝ) ≤ m010 := by linarith [(abs_le.mp he10).1]
    have h2 : (7 / 8 : ℝ) * (7 / 8 : ℝ) ≤ m001 * m010 :=
      mul_le_mul h01l h10l (by norm_num) (by linarith)
    have h3 : δ ^ 2 ≤ (1 / 8 : ℝ) * (1 / 8 : ℝ) := by
      rw [sq]; exact mul_le_mul hδ_le hδ_le hδ_nonneg (by norm_num)
    linarith
  have hdetM0_abs : (1 / 2 : ℝ) ≤ |detM0| := by
    rw [abs_of_neg (by linarith)]
    linarith
  have hdetM0_ne : detM0 ≠ 0 := by linarith
  have hdetM1_abs : |detM1| ≤ (5 / 4 : ℝ) * δ := by
    have h1 : |m100 * m111| ≤ (9 / 8 : ℝ) * δ := abs_mul_le_mul_of_abs_le hm100_abs hf11
    have h2 : |m101 * m110| ≤ (1 / 8 : ℝ) * δ := abs_mul_le_mul_of_abs_le hm101_abs8 hf10
    calc |m100 * m111 - m101 * m110|
      ≤ |m100 * m111| + |m101 * m110| := abs_sub _ _
      _ ≤ (5 / 4 : ℝ) * δ := by linarith
  have htrAdj_abs : |trAdj| ≤ (7 / 2 : ℝ) * δ := by
    have h1 : |m011 * m100| ≤ δ * (9 / 8 : ℝ) := abs_mul_le_mul_of_abs_le he11 hm100_abs
    have h2 : |m001 * m110| ≤ (9 / 8 : ℝ) * δ := abs_mul_le_mul_of_abs_le hm001_abs hf10
    have h3 : |m010 * m101| ≤ (9 / 8 : ℝ) * δ := abs_mul_le_mul_of_abs_le hm010_abs hf01
    have h4 : |m000 * m111| ≤ (1 / 8 : ℝ) * δ := abs_mul_le_mul_of_abs_le hm000_abs8 hf11
    calc |m011 * m100 - m001 * m110 - m010 * m101 + m000 * m111|
      ≤ |m011 * m100 - m001 * m110 - m010 * m101| + |m000 * m111| := abs_add_le _ _
      _ ≤ |m011 * m100 - m001 * m110| + |m010 * m101| + |m000 * m111| := by
        linarith [abs_sub (m011 * m100 - m001 * m110) (m010 * m101)]
      _ ≤ |m011 * m100| + |m001 * m110| + |m010 * m101| + |m000 * m111| := by
        linarith [abs_sub (m011 * m100) (m001 * m110)]
      _ ≤ (7 / 2 : ℝ) * δ := by linarith
  -- Apply the 2×2 slice identities.
  set dAB := (a00 * a11 - a01 * a10) * (b00 * b11 - b01 * b10)
  have hd0 : detM0 = dAB * (c00 * c01) := by
    dsimp only [detM0, dAB, m000, m001, m010, m011]; ring
  have hd1 : detM1 = dAB * (c10 * c11) := by
    dsimp only [detM1, dAB, m100, m101, m110, m111]; ring
  have htr : trAdj = dAB * (c01 * c10 + c00 * c11) := by
    dsimp only [trAdj, dAB, m000, m001, m010, m011, m100, m101, m110, m111]; ring
  have hdAB_ne : dAB ≠ 0 := by
    intro h0; rw [h0, zero_mul] at hd0; exact hdetM0_ne hd0
  have hc00_ne : c00 ≠ 0 := by
    intro h0; rw [h0, zero_mul, mul_zero] at hd0; exact hdetM0_ne hd0
  have hc01_ne : c01 ≠ 0 := by
    intro h0; rw [h0, mul_zero, mul_zero] at hd0; exact hdetM0_ne hd0
  set μ0 := c10 / c00
  set μ1 := c11 / c01
  have hμ_prod : μ0 * μ1 = detM1 / detM0 := by
    rw [hd0, hd1]; simp only [μ0, μ1]; field_simp
  have hμ_sum : μ0 + μ1 = trAdj / detM0 := by
    rw [hd0, htr]; simp only [μ0, μ1]; field_simp
  have hμ_prod_abs : |μ0 * μ1| ≤ (5 / 2 : ℝ) * δ := by
    rw [hμ_prod, abs_div, div_le_iff₀ (by positivity : (0 : ℝ) < |detM0|)]
    calc |detM1| ≤ (5 / 4 : ℝ) * δ := hdetM1_abs
      _ = (5 / 2 : ℝ) * δ * (1 / 2) := by ring
      _ ≤ (5 / 2 : ℝ) * δ * |detM0| := mul_le_mul_of_nonneg_left hdetM0_abs (by positivity)
  have hμ_sum_abs : |μ0 + μ1| ≤ 7 * δ := by
    rw [hμ_sum, abs_div, div_le_iff₀ (by positivity : (0 : ℝ) < |detM0|)]
    calc |trAdj| ≤ (7 / 2 : ℝ) * δ := htrAdj_abs
      _ = 7 * δ * (1 / 2) := by ring
      _ ≤ 7 * δ * |detM0| := mul_le_mul_of_nonneg_left hdetM0_abs (by positivity)
  have hμ_sum_sq : (μ0 + μ1) ^ 2 ≤ 49 * δ ^ 2 := by
    have : |μ0 + μ1| ^ 2 ≤ (7 * δ) ^ 2 := by
      rw [sq, sq]; exact mul_le_mul hμ_sum_abs hμ_sum_abs (abs_nonneg _) (by positivity)
    rw [sq_abs] at this; linarith
  have hμ_prod_lb : -((5 / 2 : ℝ) * δ) ≤ μ0 * μ1 := (abs_le.mp hμ_prod_abs).1
  have hδ2_le : δ ^ 2 ≤ (1 / 8 : ℝ) * δ := by
    rw [sq]; exact mul_le_mul_of_nonneg_right hδ_le hδ_nonneg
  have hμ_diff_sq : (μ0 - μ1) ^ 2 ≤ 17 * δ := by
    have hid : (μ0 - μ1) ^ 2 = (μ0 + μ1) ^ 2 - 4 * (μ0 * μ1) := by ring
    linarith
  have hμ1_sq : μ1 ^ 2 ≤ 12 * δ := by
    have hid : μ0 ^ 2 + μ1 ^ 2 = (μ0 + μ1) ^ 2 - 2 * (μ0 * μ1) := by ring
    linarith [sq_nonneg μ0]
  -- Eliminate the `r = 1` term from entry `(0, 0)`: `m100 - μ1 * m000 = (μ0 - μ1) * c00 * a00 * b00`.
  have helim : m100 - μ1 * m000 = (μ0 - μ1) * (a00 * b00 * c00) := by
    simp only [m100, m000, μ0, μ1]; field_simp; ring
  have hμ1m000_sq : (μ1 * m000) ^ 2 ≤ (1 / 4 : ℝ) ^ 2 := by
    have hm000_sq : m000 ^ 2 ≤ δ ^ 2 := by
      have : |m000| ^ 2 ≤ δ ^ 2 := by
        rw [sq, sq]; exact mul_le_mul he00 he00 (abs_nonneg _) hδ_nonneg
      rwa [sq_abs] at this
    have h1 : (μ1 * m000) ^ 2 = μ1 ^ 2 * m000 ^ 2 := by ring
    rw [h1]
    have h2 : μ1 ^ 2 * m000 ^ 2 ≤ (12 * δ) * δ ^ 2 :=
      mul_le_mul hμ1_sq hm000_sq (sq_nonneg _) (by positivity)
    have h3 : (12 * δ) * δ ^ 2 ≤ (12 * (1 / 8 : ℝ)) * (1 / 64 : ℝ) :=
      mul_le_mul (by linarith) (by linarith) (sq_nonneg _) (by norm_num)
    linarith
  have hμ1m000_le : μ1 * m000 ≤ 1 / 4 := by
    by_contra hgt; push Not at hgt
    have : (1 / 4 : ℝ) ^ 2 < (μ1 * m000) ^ 2 := by
      rw [sq, sq]; exact mul_lt_mul'' hgt hgt (by norm_num) (by norm_num)
    linarith
  have helim_ge : (1 / 2 : ℝ) ≤ m100 - μ1 * m000 := by
    have hf00l : (7 / 8 : ℝ) ≤ m100 := by linarith [(abs_le.mp hf00).1]
    linarith
  -- Square and bound `(a00 * b00 * c00)^2 ≤ (paramNormSq θ)^3`.
  set R2 := paramNormSq θ
  have hR2_nonneg : 0 ≤ R2 := paramNormSq_nonneg θ
  have ha_sq : a00 ^ 2 ≤ R2 := entry_sq_le_paramNormSq_1 θ 0 0
  have hb_sq : b00 ^ 2 ≤ R2 := entry_sq_le_paramNormSq_2 θ 0 0
  have hc_sq : c00 ^ 2 ≤ R2 := entry_sq_le_paramNormSq_3 θ 0 0
  have habc_sq : (a00 * b00 * c00) ^ 2 ≤ R2 ^ 3 := by
    have h1 : (a00 * b00 * c00) ^ 2 = a00 ^ 2 * b00 ^ 2 * c00 ^ 2 := by ring
    rw [h1, show R2 ^ 3 = R2 * R2 * R2 by ring]
    exact mul_le_mul (mul_le_mul ha_sq hb_sq (sq_nonneg _) hR2_nonneg) hc_sq
      (sq_nonneg _) (mul_nonneg hR2_nonneg hR2_nonneg)
  have hquarter : (1 / 4 : ℝ) ≤ 17 * δ * R2 ^ 3 := by
    have hsq1 : (1 / 4 : ℝ) ≤ (m100 - μ1 * m000) ^ 2 := by
      rw [show (1 / 4 : ℝ) = (1 / 2) * (1 / 2) by norm_num, sq]
      exact mul_le_mul helim_ge helim_ge (by norm_num) (by linarith)
    calc (1 / 4 : ℝ) ≤ (m100 - μ1 * m000) ^ 2 := hsq1
      _ = (μ0 - μ1) ^ 2 * (a00 * b00 * c00) ^ 2 := by rw [helim]; ring
      _ ≤ (17 * δ) * R2 ^ 3 :=
        mul_le_mul hμ_diff_sq habc_sq (sq_nonneg _) (by positivity)
  have hR2_pos : 0 < R2 := by
    by_contra hle
    push Not at hle
    have hR0 : R2 = 0 := le_antisymm hle hR2_nonneg
    rw [hR0] at hquarter
    norm_num at hquarter
  have hfinal : (1 / 16 : ℝ) ≤ 289 * L * R2 ^ 6 := by
    have hsq : (1 / 4 : ℝ) ^ 2 ≤ (17 * δ * R2 ^ 3) ^ 2 := by
      rw [sq, sq]; exact mul_le_mul hquarter hquarter (by norm_num) (by positivity)
    have : (17 * δ * R2 ^ 3) ^ 2 = 289 * δ ^ 2 * R2 ^ 6 := by ring
    rw [this, hδ_sq] at hsq
    linarith
  rw [div_le_iff₀ (by positivity : (0 : ℝ) < 4624 * R2 ^ 6)]
  linarith

/-- The optimal **centered finite-difference** border-rank curve at width `k = 2`:
`W̃_ε = (2ε)⁻¹ ((e₀ + ε e₁)^{⊗3} - (e₀ - ε e₁)^{⊗3}) = W + ε² e₁^{⊗3}`. -/
noncomputable def centeredCurve (ε : ℝ) : BilinearParam 2 2 2 2 :=
  let s : ℝ := (2 * ε) ^ (-(1 / 3 : ℝ))
  ( fun i r => if i = 0 then s else (if r = 0 then s * ε else -(s * ε)),
    fun j r => if j = 0 then s else (if r = 0 then s * ε else -(s * ε)),
    fun l r => if r = 0 then (if l = 0 then s else s * ε) else (if l = 0 then -s else s * ε) )

theorem centeredCurve_tensor {ε : ℝ} (hε : 0 < ε) (i j l : Fin 2) :
    bilinearTensor (centeredCurve ε) i j l =
      wTensor ℝ i j l + (if i = 1 ∧ j = 1 ∧ l = 1 then ε ^ 2 else 0) := by
  have hs3 : ((2 * ε) ^ (-(1 / 3 : ℝ))) ^ 3 = (2 * ε)⁻¹ := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul (by positivity : (0 : ℝ) ≤ 2 * ε),
      show (-(1 / 3 : ℝ)) * (3 : ℕ) = -1 by norm_num, Real.rpow_neg_one]
  have hs3' : ((2 * ε) ^ (-(1 / 3 : ℝ))) ^ 3 * (2 * ε) = 1 := by
    rw [hs3, inv_mul_cancel₀ (by positivity)]
  unfold bilinearTensor centeredCurve wTensor
  dsimp only
  generalize (2 * ε) ^ (-(1 / 3 : ℝ)) = s at hs3' ⊢
  fin_cases i <;> fin_cases j <;> fin_cases l <;>
    simp [Fin.sum_univ_two] <;>
    first | linear_combination | linear_combination hs3' | linear_combination ε ^ 2 * hs3'

/-- Exact loss of the centered finite-difference curve: `‖W̃_ε - W‖_F² = ε⁴`. -/
theorem centeredCurve_loss {ε : ℝ} (hε : 0 < ε) :
    bilinearLoss (wTensor ℝ) (centeredCurve ε) = ε ^ 4 := by
  unfold bilinearLoss tensorFrobSq
  simp only [Fin.sum_univ_two, centeredCurve_tensor hε]
  norm_num
  ring

/-- Exact squared Euclidean norm of the centered finite-difference curve:
`‖θ(ε)‖₂² = 6 (2ε)^{-2/3} (1 + ε²)`. -/
theorem centeredCurve_paramNormSq {ε : ℝ} (hε : 0 < ε) :
    paramNormSq (centeredCurve ε) = 6 * (2 * ε) ^ (-(2 / 3 : ℝ)) * (1 + ε ^ 2) := by
  have hs2 : ((2 * ε) ^ (-(1 / 3 : ℝ))) ^ 2 = (2 * ε) ^ (-(2 / 3 : ℝ)) := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul (by positivity : (0 : ℝ) ≤ 2 * ε)]
    norm_num
  unfold paramNormSq centeredCurve
  simp only [Fin.sum_univ_two, Fin.isValue, ↓reduceIte, one_ne_zero]
  set s := (2 * ε) ^ (-(1 / 3 : ℝ))
  have h1 : s ^ 2 + s ^ 2 + ((s * ε) ^ 2 + (-(s * ε)) ^ 2) +
      (s ^ 2 + s ^ 2 + ((s * ε) ^ 2 + (-(s * ε)) ^ 2)) +
      (s ^ 2 + (-s) ^ 2 + ((s * ε) ^ 2 + (s * ε) ^ 2)) =
      6 * s ^ 2 * (1 + ε ^ 2) := by ring
  rw [h1, hs2]

/-- **Sharp Quantitative Border-Rank Escape Upper Bound (`[LC]`):**
For all `R ≥ 4`, there exists a width-2 parameter `θ` with `paramNormSq θ ≤ R²` and
`bilinearLoss (wTensor ℝ) θ ≤ 186624 / R^{12}`. -/
theorem wTensor_escape_upper {R : ℝ} (hR : 4 ≤ R) :
    ∃ θ : BilinearParam 2 2 2 2,
      paramNormSq θ ≤ R ^ 2 ∧ bilinearLoss (wTensor ℝ) θ ≤ 186624 / R ^ 12 := by
  have hR_pos : 0 < R := by linarith
  -- Choose `ε = (1/2) * (12 / R^2)^(3/2) ∈ (0, 1]`.
  let ε : ℝ := (1 / 2 : ℝ) * (12 / R ^ 2) ^ (3 / 2 : ℝ)
  have h12R : 12 / R ^ 2 ≤ 1 := by
    rw [div_le_one (by positivity)]
    nlinarith
  have hε_pos : 0 < ε := by positivity
  have hε_le : ε ≤ 1 := by
    have hpow : (12 / R ^ 2) ^ (3 / 2 : ℝ) ≤ 1 :=
      Real.rpow_le_one (by positivity) h12R (by positivity)
    simp only [ε]; linarith
  refine ⟨centeredCurve ε, ?_, ?_⟩
  · rw [centeredCurve_paramNormSq hε_pos]
    have h2ε : 2 * ε = (12 / R ^ 2) ^ (3 / 2 : ℝ) := by simp [ε]
    have hpow : (2 * ε) ^ (-(2 / 3 : ℝ)) = R ^ 2 / 12 := by
      rw [h2ε, ← Real.rpow_mul (by positivity : (0 : ℝ) ≤ 12 / R ^ 2)]
      norm_num
      rw [Real.rpow_neg_one, inv_div]
    rw [hpow]
    have hε2 : ε ^ 2 ≤ 1 := by nlinarith
    nlinarith
  · rw [centeredCurve_loss hε_pos]
    have hε4 : ε ^ 4 = (1 / 16 : ℝ) * (12 / R ^ 2) ^ 6 := by
      have h1 : ε ^ 4 = (1 / 16 : ℝ) * ((12 / R ^ 2) ^ (3 / 2 : ℝ)) ^ 4 := by
        simp only [ε]; ring
      rw [h1, ← Real.rpow_natCast, ← Real.rpow_mul (by positivity : (0 : ℝ) ≤ 12 / R ^ 2)]
      norm_num
    rw [hε4, div_pow, ← pow_mul]
    have : (1 / 16 : ℝ) * (12 ^ 6 / R ^ (2 * 6)) = 186624 / R ^ 12 := by ring
    linarith

/-- **Exact Weight-Decay Scaling Law — Lower Bound (`[LC]`):**
For all `γ ∈ (0, 1]` and all `θ : BilinearParam 2 2 2 2`,
`(1 / 64) * γ^{6/7} ≤ bilinearLoss (wTensor ℝ) θ + γ * paramNormSq θ`. -/
theorem wTensor_weightDecay_lower {γ : ℝ} (hγ0 : 0 < γ) (hγ1 : γ ≤ 1)
    (θ : BilinearParam 2 2 2 2) :
    (1 / 64 : ℝ) * γ ^ (6 / 7 : ℝ) ≤ bilinearLoss (wTensor ℝ) θ + γ * paramNormSq θ := by
  have hlb := wTensor_loss_ge_of_paramNormSq θ
  set L := bilinearLoss (wTensor ℝ) θ
  set R2 := paramNormSq θ
  have hR2_nonneg : 0 ≤ R2 := paramNormSq_nonneg θ
  have hγpow_le : γ ^ (6 / 7 : ℝ) ≤ 1 := Real.rpow_le_one hγ0.le hγ1 (by positivity)
  by_cases h1 : (1 / 64 : ℝ) ≤ L
  · have : 0 ≤ γ * R2 := mul_nonneg hγ0.le hR2_nonneg
    nlinarith
  push Not at h1
  -- Since `L < 1/64`, the second branch of `min` applies: `1 / (4624 * R2^6) ≤ L`.
  have h2 : 1 / (4624 * R2 ^ 6) ≤ L := by
    rcases le_total ((1 : ℝ) / 64) (1 / (4624 * R2 ^ 6)) with hmin | hmin
    · rw [min_eq_left hmin] at hlb; linarith
    · rwa [min_eq_right hmin] at hlb
  have hR2_pos : 0 < R2 := by
    by_contra hle
    push Not at hle
    have h0 : R2 = 0 := le_antisymm hle hR2_nonneg
    have hθ0 : ∀ i j l : Fin 2, bilinearTensor θ i j l = 0 := by
      intro i j l
      have h10 : θ.1 i 0 = 0 := sq_eq_zero_iff.mp (le_antisymm (by linarith [entry_sq_le_paramNormSq_1 θ i 0]) (sq_nonneg _))
      have h11 : θ.1 i 1 = 0 := sq_eq_zero_iff.mp (le_antisymm (by linarith [entry_sq_le_paramNormSq_1 θ i 1]) (sq_nonneg _))
      simp [bilinearTensor, Fin.sum_univ_two, h10, h11]
    have hL3 : L = 3 := by
      simp only [L, bilinearLoss, tensorFrobSq, Fin.sum_univ_two, hθ0, wTensor]
      norm_num
    linarith
  -- Weighted AM–GM: `(1/7) * (7 / (4624 * R2^6)) + (6/7) * (7 * γ * R2 / 6) ≥ ...`
  have hamgm := Real.geom_mean_le_arith_mean2_weighted
    (w₁ := (1 / 7 : ℝ)) (w₂ := (6 / 7 : ℝ))
    (p₁ := 7 / (4624 * R2 ^ 6)) (p₂ := 7 * γ * R2 / 6)
    (by norm_num) (by norm_num) (by positivity) (by positivity) (by norm_num)
  have hrhs : (1 / 7 : ℝ) * (7 / (4624 * R2 ^ 6)) + (6 / 7 : ℝ) * (7 * γ * R2 / 6) =
      1 / (4624 * R2 ^ 6) + γ * R2 := by ring
  rw [hrhs] at hamgm
  -- Simplify the weighted geometric mean: `R2` cancels completely!
  have hlhs : (7 / (4624 * R2 ^ 6)) ^ (1 / 7 : ℝ) * (7 * γ * R2 / 6) ^ (6 / 7 : ℝ) =
      ((7 / 4624 : ℝ) ^ (1 / 7 : ℝ) * (7 / 6 : ℝ) ^ (6 / 7 : ℝ)) * γ ^ (6 / 7 : ℝ) := by
    have hsplit1 : (7 / (4624 * R2 ^ 6)) ^ (1 / 7 : ℝ) =
        (7 / 4624 : ℝ) ^ (1 / 7 : ℝ) * (R2 ^ (-(6 / 7 : ℝ))) := by
      have hrew : 7 / (4624 * R2 ^ 6) = (7 / 4624 : ℝ) * (R2 ^ (-6 : ℝ)) := by
        rw [Real.rpow_neg hR2_pos.le, show (6 : ℝ) = (6 : ℕ) by norm_num, Real.rpow_natCast]
        ring
      rw [hrew, Real.mul_rpow (by positivity) (by positivity), ← Real.rpow_mul hR2_pos.le]
      norm_num
    have hsplit2 : (7 * γ * R2 / 6) ^ (6 / 7 : ℝ) =
        (7 / 6 : ℝ) ^ (6 / 7 : ℝ) * γ ^ (6 / 7 : ℝ) * R2 ^ (6 / 7 : ℝ) := by
      rw [show 7 * γ * R2 / 6 = ((7 / 6 : ℝ) * γ) * R2 by ring,
        Real.mul_rpow (by positivity) hR2_pos.le,
        Real.mul_rpow (by positivity) hγ0.le]
    rw [hsplit1, hsplit2]
    have hcancel : R2 ^ (-(6 / 7 : ℝ)) * R2 ^ (6 / 7 : ℝ) = 1 := by
      rw [← Real.rpow_add hR2_pos]
      norm_num
    calc (7 / 4624 : ℝ) ^ (1 / 7 : ℝ) * R2 ^ (-(6 / 7 : ℝ)) *
          ((7 / 6 : ℝ) ^ (6 / 7 : ℝ) * γ ^ (6 / 7 : ℝ) * R2 ^ (6 / 7 : ℝ))
      = (7 / 4624 : ℝ) ^ (1 / 7 : ℝ) * (7 / 6 : ℝ) ^ (6 / 7 : ℝ) * γ ^ (6 / 7 : ℝ) *
          (R2 ^ (-(6 / 7 : ℝ)) * R2 ^ (6 / 7 : ℝ)) := by ring
      _ = ((7 / 4624 : ℝ) ^ (1 / 7 : ℝ) * (7 / 6 : ℝ) ^ (6 / 7 : ℝ)) * γ ^ (6 / 7 : ℝ) := by
          rw [hcancel, mul_one]
  -- Show the constant `(7/4624)^(1/7) * (7/6)^(6/7) ≥ 1/64` by raising to the 7th power!
  have hconst : (1 / 64 : ℝ) ≤ (7 / 4624 : ℝ) ^ (1 / 7 : ℝ) * (7 / 6 : ℝ) ^ (6 / 7 : ℝ) := by
    have hpos : 0 ≤ (7 / 4624 : ℝ) ^ (1 / 7 : ℝ) * (7 / 6 : ℝ) ^ (6 / 7 : ℝ) := by positivity
    have hp1 : ((7 / 4624 : ℝ) ^ (1 / 7 : ℝ)) ^ 7 = 7 / 4624 := by
      rw [← Real.rpow_natCast, ← Real.rpow_mul (by positivity)]
      norm_num
    have hp2 : ((7 / 6 : ℝ) ^ (6 / 7 : ℝ)) ^ 7 = (7 / 6 : ℝ) ^ 6 := by
      rw [← Real.rpow_natCast, ← Real.rpow_mul (by positivity)]
      norm_num
    have hpow7 : ((1 / 64 : ℝ)) ^ 7 ≤
        ((7 / 4624 : ℝ) ^ (1 / 7 : ℝ) * (7 / 6 : ℝ) ^ (6 / 7 : ℝ)) ^ 7 := by
      rw [mul_pow, hp1, hp2]
      norm_num
    by_contra hlt
    push Not at hlt
    have : ((7 / 4624 : ℝ) ^ (1 / 7 : ℝ) * (7 / 6 : ℝ) ^ (6 / 7 : ℝ)) ^ 7 < ((1 / 64 : ℝ)) ^ 7 :=
      pow_lt_pow_left₀ hlt hpos (by norm_num)
    linarith
  calc (1 / 64 : ℝ) * γ ^ (6 / 7 : ℝ)
    ≤ ((7 / 4624 : ℝ) ^ (1 / 7 : ℝ) * (7 / 6 : ℝ) ^ (6 / 7 : ℝ)) * γ ^ (6 / 7 : ℝ) :=
      mul_le_mul_of_nonneg_right hconst (by positivity)
    _ = (7 / (4624 * R2 ^ 6)) ^ (1 / 7 : ℝ) * (7 * γ * R2 / 6) ^ (6 / 7 : ℝ) := hlhs.symm
    _ ≤ 1 / (4624 * R2 ^ 6) + γ * R2 := hamgm
    _ ≤ L + γ * R2 := by linarith

/-- **Exact Weight-Decay Scaling Law — Upper Bound (`[LC]`):**
For all `γ ∈ (0, 1]`, there exists a width-2 parameter `θ` achieving
`bilinearLoss (wTensor ℝ) θ + γ * paramNormSq θ ≤ 13 * γ^{6/7}`. -/
theorem wTensor_weightDecay_upper {γ : ℝ} (hγ0 : 0 < γ) (hγ1 : γ ≤ 1) :
    ∃ θ : BilinearParam 2 2 2 2,
      bilinearLoss (wTensor ℝ) θ + γ * paramNormSq θ ≤ 13 * γ ^ (6 / 7 : ℝ) := by
  let ε : ℝ := γ ^ (3 / 14 : ℝ)
  have hε_pos : 0 < ε := by positivity
  have hε_le : ε ≤ 1 := Real.rpow_le_one hγ0.le hγ1 (by positivity)
  refine ⟨centeredCurve ε, ?_⟩
  rw [centeredCurve_loss hε_pos, centeredCurve_paramNormSq hε_pos]
  have hL : ε ^ 4 = γ ^ (6 / 7 : ℝ) := by
    simp only [ε]
    rw [← Real.rpow_natCast, ← Real.rpow_mul hγ0.le]
    norm_num
  have h2pow : (2 : ℝ) ^ (-(2 / 3 : ℝ)) ≤ 1 := by
    rw [Real.rpow_neg (by norm_num : (0 : ℝ) ≤ 2)]
    exact inv_le_one_of_one_le₀ (Real.one_le_rpow (by norm_num) (by positivity))
  have hR2 : γ * (6 * (2 * ε) ^ (-(2 / 3 : ℝ)) * (1 + ε ^ 2)) ≤ 12 * γ ^ (6 / 7 : ℝ) := by
    rw [Real.mul_rpow (by norm_num : (0 : ℝ) ≤ 2) hε_pos.le]
    have hεpow : ε ^ (-(2 / 3 : ℝ)) = γ ^ (-(1 / 7 : ℝ)) := by
      simp only [ε]
      rw [← Real.rpow_mul hγ0.le]
      norm_num
    rw [hεpow]
    have hγmul : γ * γ ^ (-(1 / 7 : ℝ)) = γ ^ (6 / 7 : ℝ) := by
      calc γ * γ ^ (-(1 / 7 : ℝ))
        = γ ^ (1 : ℝ) * γ ^ (-(1 / 7 : ℝ)) := by rw [Real.rpow_one]
        _ = γ ^ ((1 : ℝ) + -(1 / 7 : ℝ)) := (Real.rpow_add hγ0 _ _).symm
        _ = γ ^ (6 / 7 : ℝ) := by norm_num
    have h1ε2 : 1 + ε ^ 2 ≤ 2 := by nlinarith
    calc γ * (6 * ((2 : ℝ) ^ (-(2 / 3 : ℝ)) * γ ^ (-(1 / 7 : ℝ))) * (1 + ε ^ 2))
      = 6 * (2 : ℝ) ^ (-(2 / 3 : ℝ)) * (1 + ε ^ 2) * (γ * γ ^ (-(1 / 7 : ℝ))) := by ring
      _ = 6 * (2 : ℝ) ^ (-(2 / 3 : ℝ)) * (1 + ε ^ 2) * γ ^ (6 / 7 : ℝ) := by rw [hγmul]
      _ ≤ 6 * 1 * 2 * γ ^ (6 / 7 : ℝ) := by
        apply mul_le_mul_of_nonneg_right _ (by positivity)
        have : 6 * (2 : ℝ) ^ (-(2 / 3 : ℝ)) ≤ 6 * 1 := by linarith
        nlinarith [show (0 : ℝ) ≤ (2 : ℝ) ^ (-(2 / 3 : ℝ)) by positivity]
      _ = 12 * γ ^ (6 / 7 : ℝ) := by ring
  linarith

/-! ### 4. Stretched-exponential Bayesian free energy (`N^{1/7}`) vs width-3 polynomial contrast -/

/-- The Gaussian-prior global Bayesian partition function for a width-`k` bilinear MLP
targeting `W`: `Z_W^{(k)}(N) = ∫ exp(-(N ‖T(θ) - W‖_F² + (1/2) ‖θ‖₂²)) dθ`. -/
noncomputable def wPartitionZ (k : ℕ) (N : ℝ) : ℝ≥0∞ :=
  ∫⁻ θ : BilinearParam 2 2 2 k,
    ENNReal.ofReal (Real.exp (-(N * bilinearLoss (wTensor ℝ) θ + (1 / 2 : ℝ) * paramNormSq θ)))

/-- 1D Gaussian `lintegral`: `∫⁻ x : ℝ, exp(-a x²) = √(π / a)` for `a > 0`. -/
private theorem lintegral_gauss1d {a : ℝ} (ha : 0 < a) :
    (∫⁻ x : ℝ, ENNReal.ofReal (Real.exp (-(a * x ^ 2)))) =
      ENNReal.ofReal (Real.sqrt (Real.pi / a)) := by
  have hint : Integrable (fun x : ℝ => Real.exp (-(a * x ^ 2))) volume := by
    simpa [neg_mul] using integrable_exp_neg_mul_sq ha
  rw [← ofReal_integral_eq_lintegral_ofReal hint
    (Filter.Eventually.of_forall fun _ => (Real.exp_pos _).le)]
  have hval : (∫ x : ℝ, Real.exp (-(a * x ^ 2))) = Real.sqrt (Real.pi / a) := by
    simpa [neg_mul] using integral_gaussian a
  rw [hval]

/-- `d`-dimensional Gaussian `lintegral` on `Fin d → ℝ`. -/
private theorem lintegral_gauss_vec {d : ℕ} {a : ℝ} (ha : 0 < a) :
    (∫⁻ v : Fin d → ℝ, ENNReal.ofReal (Real.exp (-(a * ∑ i : Fin d, (v i) ^ 2)))) =
      ENNReal.ofReal (Real.sqrt (Real.pi / a) ^ d) := by
  induction d with
  | zero =>
    simp only [Finset.univ_eq_empty, Finset.sum_empty, mul_zero, neg_zero, Real.exp_zero,
      ENNReal.ofReal_one, pow_zero, MeasureTheory.lintegral_const]
    change 1 * (Measure.pi fun _ : Fin 0 => (volume : Measure ℝ)) Set.univ = 1
    simp
  | succ n ih =>
    have hmp := MeasureTheory.volume_preserving_piFinSuccAbove (fun _ : Fin (n + 1) => ℝ) 0
    have hmeas : Measurable (fun p : ℝ × (Fin n → ℝ) =>
        ENNReal.ofReal (Real.exp (-(a * (p.1 ^ 2 + ∑ i : Fin n, (p.2 i) ^ 2))))) := by
      fun_prop
    have hrew : (∫⁻ v : Fin (n + 1) → ℝ,
        ENNReal.ofReal (Real.exp (-(a * ∑ i : Fin (n + 1), (v i) ^ 2)))) =
        ∫⁻ p : ℝ × (Fin n → ℝ),
          ENNReal.ofReal (Real.exp (-(a * (p.1 ^ 2 + ∑ i : Fin n, (p.2 i) ^ 2)))) := by
      rw [← hmp.lintegral_comp hmeas]
      congr 1
      funext v
      simp only [MeasurableEquiv.piFinSuccAbove_apply, Fin.sum_univ_succ]
      rfl
    rw [hrew, Measure.volume_eq_prod, MeasureTheory.lintegral_prod _ hmeas.aemeasurable]
    have hsplit : ∀ (x : ℝ) (v : Fin n → ℝ),
        ENNReal.ofReal (Real.exp (-(a * (x ^ 2 + ∑ i : Fin n, (v i) ^ 2)))) =
        ENNReal.ofReal (Real.exp (-(a * x ^ 2))) *
          ENNReal.ofReal (Real.exp (-(a * ∑ i : Fin n, (v i) ^ 2))) := by
      intro x v
      rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add]
      congr 2; ring
    simp_rw [hsplit]
    simp_rw [MeasureTheory.lintegral_const_mul' _ _ ENNReal.ofReal_ne_top]
    rw [ih, MeasureTheory.lintegral_mul_const' _ _ ENNReal.ofReal_ne_top, lintegral_gauss1d ha,
      ← ENNReal.ofReal_mul (Real.sqrt_nonneg _), pow_succ']

/-- `a × b` matrix Gaussian `lintegral` on `RMat a b = Fin a → Fin b → ℝ`. -/
private theorem lintegral_gauss_mat {a b : ℕ} {c : ℝ} (hc : 0 < c) :
    (∫⁻ M : RMat a b, ENNReal.ofReal (Real.exp (-(c * ∑ i : Fin a, ∑ j : Fin b, (M i j) ^ 2)))) =
      ENNReal.ofReal (Real.sqrt (Real.pi / c) ^ (a * b)) := by
  induction a with
  | zero =>
    simp only [Finset.univ_eq_empty, Finset.sum_empty, mul_zero, neg_zero, Real.exp_zero,
      ENNReal.ofReal_one, zero_mul, pow_zero, MeasureTheory.lintegral_const]
    change 1 * (Measure.pi fun _ : Fin 0 => (volume : Measure (Fin b → ℝ))) Set.univ = 1
    simp
  | succ m ih =>
    have hmp := MeasureTheory.volume_preserving_piFinSuccAbove (fun _ : Fin (m + 1) => (Fin b → ℝ)) 0
    have hmeas : Measurable (fun p : (Fin b → ℝ) × RMat m b =>
        ENNReal.ofReal (Real.exp (-(c * ((∑ j : Fin b, (p.1 j) ^ 2) +
          ∑ i : Fin m, ∑ j : Fin b, (p.2 i j) ^ 2))))) := by
      fun_prop
    have hrew : (∫⁻ M : RMat (m + 1) b,
        ENNReal.ofReal (Real.exp (-(c * ∑ i : Fin (m + 1), ∑ j : Fin b, (M i j) ^ 2)))) =
        ∫⁻ p : (Fin b → ℝ) × RMat m b,
          ENNReal.ofReal (Real.exp (-(c * ((∑ j : Fin b, (p.1 j) ^ 2) +
            ∑ i : Fin m, ∑ j : Fin b, (p.2 i j) ^ 2)))) := by
      rw [← hmp.lintegral_comp hmeas]
      congr 1
      funext M
      simp only [MeasurableEquiv.piFinSuccAbove_apply, Fin.sum_univ_succ]
      rfl
    rw [hrew, Measure.volume_eq_prod, MeasureTheory.lintegral_prod _ hmeas.aemeasurable]
    have hsplit : ∀ (v : Fin b → ℝ) (M : RMat m b),
        ENNReal.ofReal (Real.exp (-(c * ((∑ j : Fin b, (v j) ^ 2) +
          ∑ i : Fin m, ∑ j : Fin b, (M i j) ^ 2)))) =
        ENNReal.ofReal (Real.exp (-(c * ∑ j : Fin b, (v j) ^ 2))) *
          ENNReal.ofReal (Real.exp (-(c * ∑ i : Fin m, ∑ j : Fin b, (M i j) ^ 2))) := by
      intro v M
      rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add]
      congr 2; ring
    simp_rw [hsplit]
    simp_rw [MeasureTheory.lintegral_const_mul' _ _ ENNReal.ofReal_ne_top]
    rw [ih, MeasureTheory.lintegral_mul_const' _ _ ENNReal.ofReal_ne_top, lintegral_gauss_vec hc,
      ← ENNReal.ofReal_mul (pow_nonneg (Real.sqrt_nonneg _) _), ← pow_add,
      show b + m * b = (m + 1) * b by ring]

/-- 12-dimensional Gaussian integral on `BilinearParam 2 2 2 2` with rate `1/4`. -/
private theorem lintegral_gaussian_param2 :
    (∫⁻ θ : BilinearParam 2 2 2 2, ENNReal.ofReal (Real.exp (-((1 / 4 : ℝ) * paramNormSq θ)))) =
      ENNReal.ofReal ((4 * Real.pi) ^ 6) := by
  have hc : (0 : ℝ) < 1 / 4 := by norm_num
  have hmeas : Measurable (fun θ : BilinearParam 2 2 2 2 =>
      ENNReal.ofReal (Real.exp (-((1 / 4 : ℝ) * paramNormSq θ)))) := by
    unfold paramNormSq; fun_prop
  rw [Measure.volume_eq_prod, MeasureTheory.lintegral_prod _ hmeas.aemeasurable]
  have hsplit1 : ∀ (A : RMat 2 2) (BC : RMat 2 2 × RMat 2 2),
      ENNReal.ofReal (Real.exp (-((1 / 4 : ℝ) * paramNormSq (A, BC)))) =
      ENNReal.ofReal (Real.exp (-((1 / 4 : ℝ) * ∑ i : Fin 2, ∑ r : Fin 2, (A i r) ^ 2))) *
        ENNReal.ofReal (Real.exp (-((1 / 4 : ℝ) *
          ((∑ j : Fin 2, ∑ r : Fin 2, (BC.1 j r) ^ 2) +
           (∑ l : Fin 2, ∑ r : Fin 2, (BC.2 l r) ^ 2))))) := by
    intro A BC
    rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add]
    congr 2; unfold paramNormSq; ring
  simp_rw [hsplit1]
  simp_rw [MeasureTheory.lintegral_const_mul' _ _ ENNReal.ofReal_ne_top]
  have hinner : (∫⁻ BC : RMat 2 2 × RMat 2 2,
      ENNReal.ofReal (Real.exp (-((1 / 4 : ℝ) *
        ((∑ j : Fin 2, ∑ r : Fin 2, (BC.1 j r) ^ 2) +
         (∑ l : Fin 2, ∑ r : Fin 2, (BC.2 l r) ^ 2)))))) =
      ENNReal.ofReal (Real.sqrt (Real.pi / (1 / 4 : ℝ)) ^ 4 *
        Real.sqrt (Real.pi / (1 / 4 : ℝ)) ^ 4) := by
    rw [Measure.volume_eq_prod, MeasureTheory.lintegral_prod _ (by fun_prop)]
    have hsplit2 : ∀ (B C : RMat 2 2),
        ENNReal.ofReal (Real.exp (-((1 / 4 : ℝ) *
          ((∑ j : Fin 2, ∑ r : Fin 2, (B j r) ^ 2) +
           (∑ l : Fin 2, ∑ r : Fin 2, (C l r) ^ 2))))) =
        ENNReal.ofReal (Real.exp (-((1 / 4 : ℝ) * ∑ j : Fin 2, ∑ r : Fin 2, (B j r) ^ 2))) *
          ENNReal.ofReal (Real.exp (-((1 / 4 : ℝ) * ∑ l : Fin 2, ∑ r : Fin 2, (C l r) ^ 2))) := by
      intro B C
      rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add]
      congr 2; ring
    simp_rw [hsplit2]
    simp_rw [MeasureTheory.lintegral_const_mul' _ _ ENNReal.ofReal_ne_top]
    rw [lintegral_gauss_mat hc, MeasureTheory.lintegral_mul_const' _ _ ENNReal.ofReal_ne_top,
      lintegral_gauss_mat hc, ← ENNReal.ofReal_mul (by positivity)]
  simp_rw [hinner]
  rw [MeasureTheory.lintegral_mul_const' _ _ ENNReal.ofReal_ne_top, lintegral_gauss_mat hc,
    ← ENNReal.ofReal_mul (by positivity)]
  congr 1
  have hsq : Real.sqrt (Real.pi / (1 / 4 : ℝ)) ^ 2 = 4 * Real.pi := by
    rw [Real.sq_sqrt (by positivity)]; ring
  calc Real.sqrt (Real.pi / (1 / 4 : ℝ)) ^ (2 * 2) *
        (Real.sqrt (Real.pi / (1 / 4 : ℝ)) ^ 4 * Real.sqrt (Real.pi / (1 / 4 : ℝ)) ^ 4)
    = (Real.sqrt (Real.pi / (1 / 4 : ℝ)) ^ 2) ^ 6 := by ring
    _ = (4 * Real.pi) ^ 6 := by rw [hsq]

/-- **Stretched-Exponential Bayesian Free Energy — Upper Bound on `Z_W^{(2)}(N)` (`[LC]`):**
For all `N ≥ 1`,
`Z_W^{(2)}(N) ≤ (4π)⁶ · exp(-a₁ N^{1/7})` with explicit constant `a₁ = (1/64) · 4^{-6/7} > 0`.
In particular, `-log Z_W^{(2)}(N) ≥ a₁ N^{1/7} - 6 log(4π)`. -/
theorem wPartitionZ_two_le_stretchedExp {N : ℝ} (hN : 1 ≤ N) :
    wPartitionZ 2 N ≤
      ENNReal.ofReal ((4 * Real.pi) ^ 6 * Real.exp (-(((1 / 64 : ℝ) * (4 : ℝ) ^ (-(6 / 7 : ℝ))) * N ^ (1 / 7 : ℝ)))) := by
  have hN_pos : 0 < N := by linarith
  let γ : ℝ := 1 / (4 * N)
  have hγ0 : 0 < γ := by positivity
  have hγ1 : γ ≤ 1 := by
    simp only [γ]; rw [div_le_one (by positivity)]; linarith
  have hpt : ∀ θ : BilinearParam 2 2 2 2,
      ((1 / 64 : ℝ) * (4 : ℝ) ^ (-(6 / 7 : ℝ))) * N ^ (1 / 7 : ℝ) ≤
        N * bilinearLoss (wTensor ℝ) θ + (1 / 4 : ℝ) * paramNormSq θ := by
    intro θ
    have hwd := wTensor_weightDecay_lower hγ0 hγ1 θ
    have hmul : N * ((1 / 64 : ℝ) * γ ^ (6 / 7 : ℝ)) ≤
        N * (bilinearLoss (wTensor ℝ) θ + γ * paramNormSq θ) :=
      mul_le_mul_of_nonneg_left hwd hN_pos.le
    have hrhs : N * (bilinearLoss (wTensor ℝ) θ + γ * paramNormSq θ) =
        N * bilinearLoss (wTensor ℝ) θ + (1 / 4 : ℝ) * paramNormSq θ := by
      simp only [γ]; field_simp
    have hγpow : γ ^ (6 / 7 : ℝ) = (4 : ℝ) ^ (-(6 / 7 : ℝ)) * N ^ (-(6 / 7 : ℝ)) := by
      simp only [γ]
      rw [one_div, Real.inv_rpow (by positivity), Real.mul_rpow (by norm_num) hN_pos.le,
        mul_inv, ← Real.rpow_neg (by norm_num : (0 : ℝ) ≤ 4), ← Real.rpow_neg hN_pos.le]
    have hNpow : N * N ^ (-(6 / 7 : ℝ)) = N ^ (1 / 7 : ℝ) := by
      calc N * N ^ (-(6 / 7 : ℝ))
        = N ^ (1 : ℝ) * N ^ (-(6 / 7 : ℝ)) := by rw [Real.rpow_one]
        _ = N ^ ((1 : ℝ) + -(6 / 7 : ℝ)) := (Real.rpow_add hN_pos _ _).symm
        _ = N ^ (1 / 7 : ℝ) := by norm_num
    have hlhs : N * ((1 / 64 : ℝ) * γ ^ (6 / 7 : ℝ)) =
        ((1 / 64 : ℝ) * (4 : ℝ) ^ (-(6 / 7 : ℝ))) * N ^ (1 / 7 : ℝ) := by
      rw [hγpow]
      calc N * ((1 / 64 : ℝ) * ((4 : ℝ) ^ (-(6 / 7 : ℝ)) * N ^ (-(6 / 7 : ℝ))))
        = ((1 / 64 : ℝ) * (4 : ℝ) ^ (-(6 / 7 : ℝ))) * (N * N ^ (-(6 / 7 : ℝ))) := by ring
        _ = ((1 / 64 : ℝ) * (4 : ℝ) ^ (-(6 / 7 : ℝ))) * N ^ (1 / 7 : ℝ) := by rw [hNpow]
    linarith
  set a1 : ℝ := (1 / 64 : ℝ) * (4 : ℝ) ^ (-(6 / 7 : ℝ))
  have hle_int : wPartitionZ 2 N ≤
      ∫⁻ θ : BilinearParam 2 2 2 2,
        ENNReal.ofReal (Real.exp (-(a1 * N ^ (1 / 7 : ℝ)))) *
          ENNReal.ofReal (Real.exp (-((1 / 4 : ℝ) * paramNormSq θ))) := by
    apply MeasureTheory.lintegral_mono
    intro θ
    dsimp only
    rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add]
    apply ENNReal.ofReal_le_ofReal
    apply Real.exp_le_exp.mpr
    linarith [hpt θ]
  calc wPartitionZ 2 N
    ≤ ∫⁻ θ : BilinearParam 2 2 2 2,
        ENNReal.ofReal (Real.exp (-(a1 * N ^ (1 / 7 : ℝ)))) *
          ENNReal.ofReal (Real.exp (-((1 / 4 : ℝ) * paramNormSq θ))) := hle_int
    _ = ENNReal.ofReal (Real.exp (-(a1 * N ^ (1 / 7 : ℝ)))) *
          ∫⁻ θ : BilinearParam 2 2 2 2,
            ENNReal.ofReal (Real.exp (-((1 / 4 : ℝ) * paramNormSq θ))) :=
      MeasureTheory.lintegral_const_mul _ (by unfold paramNormSq; fun_prop)
    _ = ENNReal.ofReal (Real.exp (-(a1 * N ^ (1 / 7 : ℝ)))) *
          ENNReal.ofReal ((4 * Real.pi) ^ 6) := by rw [lintegral_gaussian_param2]
    _ = ENNReal.ofReal ((4 * Real.pi) ^ 6 * Real.exp (-(a1 * N ^ (1 / 7 : ℝ)))) := by
      rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, mul_comm]

/-- Translated `L^∞`-box of half-width `ρ` around `θ₀` in `BilinearParam 2 2 2 k`. -/
private def paramBox {k : ℕ} (θ₀ : BilinearParam 2 2 2 k) (ρ : ℝ) : Set (BilinearParam 2 2 2 k) :=
  {θ | (∀ i r, θ.1 i r ∈ Set.Icc (θ₀.1 i r - ρ) (θ₀.1 i r + ρ)) ∧
       (∀ j r, θ.2.1 j r ∈ Set.Icc (θ₀.2.1 j r - ρ) (θ₀.2.1 j r + ρ)) ∧
       (∀ l r, θ.2.2 l r ∈ Set.Icc (θ₀.2.2 l r - ρ) (θ₀.2.2 l r + ρ))}

private theorem volume_matBox_shifted {a b : ℕ} (M₀ : RMat a b) {ρ : ℝ} (_hρ : 0 ≤ ρ) :
    volume {M : RMat a b | ∀ i j, M i j ∈ Set.Icc (M₀ i j - ρ) (M₀ i j + ρ)} =
      ENNReal.ofReal (2 * ρ) ^ (a * b) := by
  have hset : {M : RMat a b | ∀ i j, M i j ∈ Set.Icc (M₀ i j - ρ) (M₀ i j + ρ)} =
      Set.pi Set.univ (fun i : Fin a => Set.pi Set.univ (fun j : Fin b => Set.Icc (M₀ i j - ρ) (M₀ i j + ρ))) := by
    ext M
    simp only [Set.mem_ofPred_eq, Set.mem_pi, Set.mem_univ, forall_const]
  rw [hset]
  change (Measure.pi fun _ : Fin a => (volume : Measure (Fin b → ℝ))) _ = _
  rw [Measure.pi_pi]
  have hrow : ∀ i : Fin a,
      (volume : Measure (Fin b → ℝ)) (Set.pi Set.univ fun j : Fin b => Set.Icc (M₀ i j - ρ) (M₀ i j + ρ)) =
        ENNReal.ofReal (2 * ρ) ^ b := by
    intro i
    change (Measure.pi fun _ : Fin b => (volume : Measure ℝ)) _ = _
    rw [Measure.pi_pi]
    have h1 : ∀ j : Fin b, volume (Set.Icc (M₀ i j - ρ) (M₀ i j + ρ)) = ENNReal.ofReal (2 * ρ) := by
      intro j
      rw [Real.volume_Icc]
      congr 1; ring
    rw [Finset.prod_congr rfl (fun j _ => h1 j), Finset.prod_const, Finset.card_univ, Fintype.card_fin]
  rw [Finset.prod_congr rfl (fun i _ => hrow i), Finset.prod_const, Finset.card_univ, Fintype.card_fin,
    ← pow_mul, mul_comm b a]

private theorem measurableSet_paramBox {k : ℕ} (θ₀ : BilinearParam 2 2 2 k) (ρ : ℝ) :
    MeasurableSet (paramBox θ₀ ρ) := by
  have hprod : paramBox θ₀ ρ =
      (Set.pi Set.univ fun i => Set.pi Set.univ fun r => Set.Icc (θ₀.1 i r - ρ) (θ₀.1 i r + ρ)) ×ˢ
      ((Set.pi Set.univ fun j => Set.pi Set.univ fun r => Set.Icc (θ₀.2.1 j r - ρ) (θ₀.2.1 j r + ρ)) ×ˢ
       (Set.pi Set.univ fun l => Set.pi Set.univ fun r => Set.Icc (θ₀.2.2 l r - ρ) (θ₀.2.2 l r + ρ))) := by
    ext ⟨A, B, C⟩
    simp only [paramBox, Set.mem_ofPred_eq, Set.mem_prod, Set.mem_pi, Set.mem_univ, forall_const]
  rw [hprod]
  refine MeasurableSet.prod ?_ (MeasurableSet.prod ?_ ?_) <;>
    exact MeasurableSet.univ_pi fun _ => MeasurableSet.univ_pi fun _ => measurableSet_Icc

private theorem volume_paramBox {k : ℕ} (θ₀ : BilinearParam 2 2 2 k) {ρ : ℝ} (hρ : 0 ≤ ρ) :
    volume (paramBox θ₀ ρ) = ENNReal.ofReal ((2 * ρ) ^ (6 * k)) := by
  have hprod : paramBox θ₀ ρ =
      {A : RMat 2 k | ∀ i r, A i r ∈ Set.Icc (θ₀.1 i r - ρ) (θ₀.1 i r + ρ)} ×ˢ
      ({B : RMat 2 k | ∀ j r, B j r ∈ Set.Icc (θ₀.2.1 j r - ρ) (θ₀.2.1 j r + ρ)} ×ˢ
       {C : RMat 2 k | ∀ l r, C l r ∈ Set.Icc (θ₀.2.2 l r - ρ) (θ₀.2.2 l r + ρ)}) := by
    ext ⟨A, B, C⟩; simp [paramBox]
  rw [hprod, Measure.volume_eq_prod, Measure.prod_prod, Measure.volume_eq_prod, Measure.prod_prod]
  rw [volume_matBox_shifted θ₀.1 hρ, volume_matBox_shifted θ₀.2.1 hρ, volume_matBox_shifted θ₀.2.2 hρ]
  rw [← pow_add, ← pow_add, show 2 * k + (2 * k + 2 * k) = 6 * k by ring,
    ← ENNReal.ofReal_pow (by positivity)]

private theorem cubic_diff_bound {a b c a₀ b₀ c₀ M ρ : ℝ}
    (_ha : |a| ≤ M) (hb : |b| ≤ M) (hc : |c| ≤ M)
    (ha₀ : |a₀| ≤ M) (hb₀ : |b₀| ≤ M) (_hc₀ : |c₀| ≤ M)
    (hda : |a - a₀| ≤ ρ) (hdb : |b - b₀| ≤ ρ) (hdc : |c - c₀| ≤ ρ)
    (hM : 0 ≤ M) (hρ : 0 ≤ ρ) :
    |a * b * c - a₀ * b₀ * c₀| ≤ 3 * M ^ 2 * ρ := by
  have hid : a * b * c - a₀ * b₀ * c₀ =
      (a - a₀) * b * c + a₀ * (b - b₀) * c + a₀ * b₀ * (c - c₀) := by ring
  rw [hid]
  have h1 : |(a - a₀) * b * c| ≤ M ^ 2 * ρ := by
    rw [abs_mul, abs_mul]
    calc |a - a₀| * |b| * |c| ≤ ρ * M * M := by
          exact mul_le_mul (mul_le_mul hda hb (abs_nonneg _) hρ) hc (abs_nonneg _) (by positivity)
      _ = M ^ 2 * ρ := by ring
  have h2 : |a₀ * (b - b₀) * c| ≤ M ^ 2 * ρ := by
    rw [abs_mul, abs_mul]
    calc |a₀| * |b - b₀| * |c| ≤ M * ρ * M := by
          exact mul_le_mul (mul_le_mul ha₀ hdb (abs_nonneg _) hM) hc (abs_nonneg _) (by positivity)
      _ = M ^ 2 * ρ := by ring
  have h3 : |a₀ * b₀ * (c - c₀)| ≤ M ^ 2 * ρ := by
    rw [abs_mul, abs_mul]
    calc |a₀| * |b₀| * |c - c₀| ≤ M * M * ρ := by
          exact mul_le_mul (mul_le_mul ha₀ hb₀ (abs_nonneg _) hM) hdc (abs_nonneg _) (by positivity)
      _ = M ^ 2 * ρ := by ring
  calc |(a - a₀) * b * c + a₀ * (b - b₀) * c + a₀ * b₀ * (c - c₀)|
    ≤ |(a - a₀) * b * c + a₀ * (b - b₀) * c| + |a₀ * b₀ * (c - c₀)| := abs_add_le _ _
    _ ≤ |(a - a₀) * b * c| + |a₀ * (b - b₀) * c| + |a₀ * b₀ * (c - c₀)| := by
      linarith [abs_add_le ((a - a₀) * b * c) (a₀ * (b - b₀) * c)]
    _ ≤ 3 * M ^ 2 * ρ := by linarith

/-- **Stretched-Exponential Bayesian Free Energy — Lower Bound on `Z_W^{(2)}(N)` (`[LC]`):**
For all `N ≥ 1`,
`(1/32)^{12} · exp(-90 N^{1/7}) ≤ Z_W^{(2)}(N)`.
Combined with `wPartitionZ_two_le_stretchedExp`, this certifies the exact stretched-exponential
scaling `-log Z_W^{(2)}(N) ≍ N^{1/7}` for all `N ≥ 1`. -/
theorem wPartitionZ_two_ge_stretchedExp {N : ℝ} (hN : 1 ≤ N) :
    ENNReal.ofReal (((1 / 32 : ℝ) ^ 12) * Real.exp (-(90 * N ^ (1 / 7 : ℝ)))) ≤
      wPartitionZ 2 N := by
  have hN_pos : 0 < N := by linarith
  let ε : ℝ := N ^ (-(3 / 14 : ℝ))
  have hε_pos : 0 < ε := by positivity
  have hε_le : ε ≤ 1 := by
    simp only [ε]
    rw [Real.rpow_neg hN_pos.le]
    exact inv_le_one_of_one_le₀ (Real.one_le_rpow hN (by positivity))
  let θ₀ : BilinearParam 2 2 2 2 := centeredCurve ε
  let ρ : ℝ := (1 / 64 : ℝ) * N ^ (-(4 / 7 : ℝ))
  have hρ_pos : 0 < ρ := by positivity
  have hρ_le : ρ ≤ 1 := by
    have h1 : N ^ (-(4 / 7 : ℝ)) ≤ 1 := by
      rw [Real.rpow_neg hN_pos.le]
      exact inv_le_one_of_one_le₀ (Real.one_le_rpow hN (by positivity))
    simp only [ρ]; linarith
  have hN14_ge : 1 ≤ N ^ (1 / 14 : ℝ) := Real.one_le_rpow hN (by positivity)
  have hN17_ge : 1 ≤ N ^ (1 / 7 : ℝ) := Real.one_le_rpow hN (by positivity)
  have hN14_sq : (N ^ (1 / 14 : ℝ)) ^ 2 = N ^ (1 / 7 : ℝ) := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hN_pos.le]; norm_num
  have hN37_sq : (N ^ (-(3 / 7 : ℝ))) ^ 2 = N ^ (-(6 / 7 : ℝ)) := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hN_pos.le]; norm_num
  -- Bound every coordinate of `θ₀` by `N^{1/14}`.
  have hs_le : (2 * ε) ^ (-(1 / 3 : ℝ)) ≤ N ^ (1 / 14 : ℝ) := by
    rw [Real.mul_rpow (by norm_num : (0 : ℝ) ≤ 2) hε_pos.le]
    have h2 : (2 : ℝ) ^ (-(1 / 3 : ℝ)) ≤ 1 := by
      rw [Real.rpow_neg (by norm_num : (0 : ℝ) ≤ 2)]
      exact inv_le_one_of_one_le₀ (Real.one_le_rpow (by norm_num) (by positivity))
    have hεp : ε ^ (-(1 / 3 : ℝ)) = N ^ (1 / 14 : ℝ) := by
      simp only [ε]
      rw [← Real.rpow_mul hN_pos.le]
      norm_num
    rw [hεp]
    nlinarith [show (0 : ℝ) ≤ N ^ (1 / 14 : ℝ) by positivity]
  have hs_pos : 0 < (2 * ε) ^ (-(1 / 3 : ℝ)) := by positivity
  have hs_abs : |(2 * ε) ^ (-(1 / 3 : ℝ))| ≤ N ^ (1 / 14 : ℝ) := by
    rwa [abs_of_pos hs_pos]
  have hsε_abs : |(2 * ε) ^ (-(1 / 3 : ℝ)) * ε| ≤ N ^ (1 / 14 : ℝ) := by
    rw [abs_of_pos (mul_pos hs_pos hε_pos)]; nlinarith
  have hθ₀_1 : ∀ i r : Fin 2, |θ₀.1 i r| ≤ N ^ (1 / 14 : ℝ) := by
    intro i r
    dsimp only [θ₀, centeredCurve]
    split_ifs <;> (try simp only [abs_neg]) <;> first | exact hs_abs | exact hsε_abs
  have hθ₀_2 : ∀ j r : Fin 2, |θ₀.2.1 j r| ≤ N ^ (1 / 14 : ℝ) := by
    intro j r
    dsimp only [θ₀, centeredCurve]
    split_ifs <;> (try simp only [abs_neg]) <;> first | exact hs_abs | exact hsε_abs
  have hθ₀_3 : ∀ l r : Fin 2, |θ₀.2.2 l r| ≤ N ^ (1 / 14 : ℝ) := by
    intro l r
    dsimp only [θ₀, centeredCurve]
    split_ifs <;> (try simp only [abs_neg]) <;> first | exact hs_abs | exact hsε_abs
  -- Pointwise upper bound on the exponent across `paramBox θ₀ ρ`.
  have hexp_le : ∀ θ ∈ paramBox θ₀ ρ,
      N * bilinearLoss (wTensor ℝ) θ + (1 / 2 : ℝ) * paramNormSq θ ≤ 42 * N ^ (1 / 7 : ℝ) := by
    intro θ hθ
    rcases hθ with ⟨h1, h2, h3⟩
    have hda : ∀ i r, |θ.1 i r - θ₀.1 i r| ≤ ρ := fun i r => by
      have := h1 i r; rw [Set.mem_Icc] at this; rw [abs_le]; constructor <;> linarith
    have hdb : ∀ j r, |θ.2.1 j r - θ₀.2.1 j r| ≤ ρ := fun j r => by
      have := h2 j r; rw [Set.mem_Icc] at this; rw [abs_le]; constructor <;> linarith
    have hdc : ∀ l r, |θ.2.2 l r - θ₀.2.2 l r| ≤ ρ := fun l r => by
      have := h3 l r; rw [Set.mem_Icc] at this; rw [abs_le]; constructor <;> linarith
    set M := 2 * N ^ (1 / 14 : ℝ)
    have hM_nonneg : 0 ≤ M := by positivity
    have hM2 : M ^ 2 = 4 * N ^ (1 / 7 : ℝ) := by
      simp only [M]; rw [mul_pow, hN14_sq]; ring
    have hθ_1 : ∀ i r, |θ.1 i r| ≤ M := fun i r => by
      calc |θ.1 i r| = |(θ.1 i r - θ₀.1 i r) + θ₀.1 i r| := by ring_nf
        _ ≤ |θ.1 i r - θ₀.1 i r| + |θ₀.1 i r| := abs_add_le _ _
        _ ≤ ρ + N ^ (1 / 14 : ℝ) := add_le_add (hda i r) (hθ₀_1 i r)
        _ ≤ M := by simp only [M]; linarith
    have hθ_2 : ∀ j r, |θ.2.1 j r| ≤ M := fun j r => by
      calc |θ.2.1 j r| = |(θ.2.1 j r - θ₀.2.1 j r) + θ₀.2.1 j r| := by ring_nf
        _ ≤ |θ.2.1 j r - θ₀.2.1 j r| + |θ₀.2.1 j r| := abs_add_le _ _
        _ ≤ ρ + N ^ (1 / 14 : ℝ) := add_le_add (hdb j r) (hθ₀_2 j r)
        _ ≤ M := by simp only [M]; linarith
    have hθ_3 : ∀ l r, |θ.2.2 l r| ≤ M := fun l r => by
      calc |θ.2.2 l r| = |(θ.2.2 l r - θ₀.2.2 l r) + θ₀.2.2 l r| := by ring_nf
        _ ≤ |θ.2.2 l r - θ₀.2.2 l r| + |θ₀.2.2 l r| := abs_add_le _ _
        _ ≤ ρ + N ^ (1 / 14 : ℝ) := add_le_add (hdc l r) (hθ₀_3 l r)
        _ ≤ M := by simp only [M]; linarith
    -- Entrywise tensor perturbation bound: `|T(θ)_{ijl} - T(θ₀)_{ijl}| ≤ N^{-3/7}`.
    have hM2ρ : 6 * M ^ 2 * ρ ≤ N ^ (-(3 / 7 : ℝ)) := by
      rw [hM2]
      have hmul : N ^ (1 / 7 : ℝ) * N ^ (-(4 / 7 : ℝ)) = N ^ (-(3 / 7 : ℝ)) := by
        rw [← Real.rpow_add hN_pos]; norm_num
      calc 6 * (4 * N ^ (1 / 7 : ℝ)) * ((1 / 64 : ℝ) * N ^ (-(4 / 7 : ℝ)))
        = (24 / 64 : ℝ) * (N ^ (1 / 7 : ℝ) * N ^ (-(4 / 7 : ℝ))) := by ring
        _ = (24 / 64 : ℝ) * N ^ (-(3 / 7 : ℝ)) := by rw [hmul]
        _ ≤ N ^ (-(3 / 7 : ℝ)) := by nlinarith [show (0 : ℝ) ≤ N ^ (-(3 / 7 : ℝ)) by positivity]
    have hTdiff : ∀ i j l : Fin 2, |bilinearTensor θ i j l - bilinearTensor θ₀ i j l| ≤ N ^ (-(3 / 7 : ℝ)) := by
      intro i j l
      simp only [bilinearTensor, Fin.sum_univ_two]
      have h0 := cubic_diff_bound (hθ_1 i 0) (hθ_2 j 0) (hθ_3 l 0)
        (le_trans (hθ₀_1 i 0) (by simp only [M]; linarith))
        (le_trans (hθ₀_2 j 0) (by simp only [M]; linarith))
        (le_trans (hθ₀_3 l 0) (by simp only [M]; linarith))
        (hda i 0) (hdb j 0) (hdc l 0) hM_nonneg hρ_pos.le
      have h1' := cubic_diff_bound (hθ_1 i 1) (hθ_2 j 1) (hθ_3 l 1)
        (le_trans (hθ₀_1 i 1) (by simp only [M]; linarith))
        (le_trans (hθ₀_2 j 1) (by simp only [M]; linarith))
        (le_trans (hθ₀_3 l 1) (by simp only [M]; linarith))
        (hda i 1) (hdb j 1) (hdc l 1) hM_nonneg hρ_pos.le
      have hsplit : (θ.1 i 0 * θ.2.1 j 0 * θ.2.2 l 0 + θ.1 i 1 * θ.2.1 j 1 * θ.2.2 l 1) -
          (θ₀.1 i 0 * θ₀.2.1 j 0 * θ₀.2.2 l 0 + θ₀.1 i 1 * θ₀.2.1 j 1 * θ₀.2.2 l 1) =
          (θ.1 i 0 * θ.2.1 j 0 * θ.2.2 l 0 - θ₀.1 i 0 * θ₀.2.1 j 0 * θ₀.2.2 l 0) +
          (θ.1 i 1 * θ.2.1 j 1 * θ.2.2 l 1 - θ₀.1 i 1 * θ₀.2.1 j 1 * θ₀.2.2 l 1) := by ring
      rw [hsplit]
      exact (abs_add_le _ _).trans (by linarith)
    -- Bound `bilinearLoss (wTensor ℝ) θ ≤ 18 * N^{-6/7}`.
    have hL_le : bilinearLoss (wTensor ℝ) θ ≤ 18 * N ^ (-(6 / 7 : ℝ)) := by
      have hL0 : bilinearLoss (wTensor ℝ) θ₀ = N ^ (-(6 / 7 : ℝ)) := by
        rw [centeredCurve_loss hε_pos]
        simp only [ε]
        rw [← Real.rpow_natCast, ← Real.rpow_mul hN_pos.le]
        norm_num
      have hsq_diff : ∀ i j l : Fin 2,
          (bilinearTensor θ i j l - wTensor ℝ i j l) ^ 2 ≤
            2 * (bilinearTensor θ₀ i j l - wTensor ℝ i j l) ^ 2 + 2 * N ^ (-(6 / 7 : ℝ)) := by
        intro i j l
        have h1' : (bilinearTensor θ i j l - bilinearTensor θ₀ i j l) ^ 2 ≤ N ^ (-(6 / 7 : ℝ)) := by
          have ha := hTdiff i j l
          have hsq : |bilinearTensor θ i j l - bilinearTensor θ₀ i j l| ^ 2 ≤
              (N ^ (-(3 / 7 : ℝ))) ^ 2 := by
            nlinarith [abs_nonneg (bilinearTensor θ i j l - bilinearTensor θ₀ i j l)]
          rw [sq_abs, hN37_sq] at hsq
          exact hsq
        have h2' : (bilinearTensor θ i j l - wTensor ℝ i j l) ^ 2 ≤
            2 * (bilinearTensor θ₀ i j l - wTensor ℝ i j l) ^ 2 +
            2 * (bilinearTensor θ i j l - bilinearTensor θ₀ i j l) ^ 2 := by
          nlinarith [sq_nonneg ((bilinearTensor θ₀ i j l - wTensor ℝ i j l) -
            (bilinearTensor θ i j l - bilinearTensor θ₀ i j l))]
        linarith
      calc bilinearLoss (wTensor ℝ) θ
        = ∑ i : Fin 2, ∑ j : Fin 2, ∑ l : Fin 2, (bilinearTensor θ i j l - wTensor ℝ i j l) ^ 2 := rfl
        _ ≤ ∑ i : Fin 2, ∑ j : Fin 2, ∑ l : Fin 2,
              (2 * (bilinearTensor θ₀ i j l - wTensor ℝ i j l) ^ 2 + 2 * N ^ (-(6 / 7 : ℝ))) :=
          Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ =>
            Finset.sum_le_sum fun l _ => hsq_diff i j l
        _ = 2 * bilinearLoss (wTensor ℝ) θ₀ + 16 * N ^ (-(6 / 7 : ℝ)) := by
          simp only [bilinearLoss, tensorFrobSq, Fin.sum_univ_two]; ring
        _ = 18 * N ^ (-(6 / 7 : ℝ)) := by rw [hL0]; ring
    have hNL : N * bilinearLoss (wTensor ℝ) θ ≤ 18 * N ^ (1 / 7 : ℝ) := by
      have hNmul : N * N ^ (-(6 / 7 : ℝ)) = N ^ (1 / 7 : ℝ) := by
        calc N * N ^ (-(6 / 7 : ℝ))
          = N ^ (1 : ℝ) * N ^ (-(6 / 7 : ℝ)) := by rw [Real.rpow_one]
          _ = N ^ ((1 : ℝ) + -(6 / 7 : ℝ)) := (Real.rpow_add hN_pos _ _).symm
          _ = N ^ (1 / 7 : ℝ) := by norm_num
      calc N * bilinearLoss (wTensor ℝ) θ
        ≤ N * (18 * N ^ (-(6 / 7 : ℝ))) := mul_le_mul_of_nonneg_left hL_le hN_pos.le
        _ = 18 * (N * N ^ (-(6 / 7 : ℝ))) := by ring
        _ = 18 * N ^ (1 / 7 : ℝ) := by rw [hNmul]
    -- Bound `paramNormSq θ ≤ 12 * M^2 = 48 * N^{1/7}`.
    have hR2_le : paramNormSq θ ≤ 48 * N ^ (1 / 7 : ℝ) := by
      have hsq1 : ∀ i r, (θ.1 i r) ^ 2 ≤ M ^ 2 := fun i r => by
        have h := hθ_1 i r; rw [← sq_abs, sq, sq]; exact mul_le_mul h h (abs_nonneg _) hM_nonneg
      have hsq2 : ∀ j r, (θ.2.1 j r) ^ 2 ≤ M ^ 2 := fun j r => by
        have h := hθ_2 j r; rw [← sq_abs, sq, sq]; exact mul_le_mul h h (abs_nonneg _) hM_nonneg
      have hsq3 : ∀ l r, (θ.2.2 l r) ^ 2 ≤ M ^ 2 := fun l r => by
        have h := hθ_3 l r; rw [← sq_abs, sq, sq]; exact mul_le_mul h h (abs_nonneg _) hM_nonneg
      unfold paramNormSq
      simp only [Fin.sum_univ_two]
      linarith [hsq1 0 0, hsq1 0 1, hsq1 1 0, hsq1 1 1,
        hsq2 0 0, hsq2 0 1, hsq2 1 0, hsq2 1 1,
        hsq3 0 0, hsq3 0 1, hsq3 1 0, hsq3 1 1, hM2]
    linarith
  -- Restrict integral to `paramBox θ₀ ρ` and bound below by `exp(-42 N^{1/7}) * vol(paramBox θ₀ ρ)`.
  have hlow : ENNReal.ofReal (Real.exp (-(42 * N ^ (1 / 7 : ℝ)))) * volume (paramBox θ₀ ρ) ≤
      wPartitionZ 2 N := by
    rw [(MeasureTheory.setLIntegral_const (paramBox θ₀ ρ)
      (ENNReal.ofReal (Real.exp (-(42 * N ^ (1 / 7 : ℝ)))))).symm]
    refine le_trans ?_ (MeasureTheory.setLIntegral_le_lintegral (paramBox θ₀ ρ) _)
    exact MeasureTheory.setLIntegral_mono_ae' (measurableSet_paramBox θ₀ ρ)
      (Filter.Eventually.of_forall fun θ hθ =>
        ENNReal.ofReal_le_ofReal (Real.exp_le_exp.mpr (by linarith [hexp_le θ hθ])))
  rw [volume_paramBox θ₀ hρ_pos.le, ← ENNReal.ofReal_mul (Real.exp_pos _).le] at hlow
  refine le_trans ?_ hlow
  apply ENNReal.ofReal_le_ofReal
  -- Absorb `(N^{-4/7})^{12} = N^{-48/7} ≥ exp(-48 N^{1/7})` using `log N ≤ 7 N^{1/7}`.
  have hlog : Real.log N ≤ 7 * N ^ (1 / 7 : ℝ) := by
    have h1 : Real.log (N ^ (1 / 7 : ℝ)) ≤ N ^ (1 / 7 : ℝ) - 1 :=
      Real.log_le_sub_one_of_pos (by positivity)
    rw [Real.log_rpow hN_pos] at h1
    linarith
  have hNpow_ge : Real.exp (-(48 * N ^ (1 / 7 : ℝ))) ≤ (N ^ (-(4 / 7 : ℝ))) ^ 12 := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hN_pos.le,
      show (-(4 / 7 : ℝ)) * (12 : ℕ) = -(48 / 7 : ℝ) by norm_num]
    conv_rhs => rw [Real.rpow_def_of_pos hN_pos]
    apply Real.exp_le_exp.mpr
    have : (48 / 7 : ℝ) * Real.log N ≤ 48 * N ^ (1 / 7 : ℝ) := by linarith
    linarith
  have h2ρ : (2 * ρ) ^ 12 = (1 / 32 : ℝ) ^ 12 * (N ^ (-(4 / 7 : ℝ))) ^ 12 := by
    simp only [ρ]
    rw [show 2 * ((1 / 64 : ℝ) * N ^ (-(4 / 7 : ℝ))) = (1 / 32 : ℝ) * N ^ (-(4 / 7 : ℝ)) by ring,
      mul_pow]
  rw [show 6 * 2 = 12 by rfl, h2ρ]
  have hrearr : (1 / 32 : ℝ) ^ 12 * Real.exp (-(90 * N ^ (1 / 7 : ℝ))) =
      Real.exp (-(42 * N ^ (1 / 7 : ℝ))) *
        ((1 / 32 : ℝ) ^ 12 * Real.exp (-(48 * N ^ (1 / 7 : ℝ)))) := by
    rw [mul_left_comm, ← Real.exp_add]; congr 2; ring
  rw [hrearr]
  exact mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hNpow_ge (by positivity))
    (Real.exp_pos _).le

/-- Canonical exact width-3 realization of `wTensor ℝ`:
`e₀⊗e₀⊗e₁ + e₀⊗e₁⊗e₀ + e₁⊗e₀⊗e₀`. -/
def wRealize3 : BilinearParam 2 2 2 3 :=
  ( fun i r => if r = 2 then (if i = 1 then 1 else 0) else (if i = 0 then 1 else 0),
    fun j r => if r = 1 then (if j = 1 then 1 else 0) else (if j = 0 then 1 else 0),
    fun l r => if r = 0 then (if l = 1 then 1 else 0) else (if l = 0 then 1 else 0) )

theorem wRealize3_tensor : bilinearTensor wRealize3 = wTensor ℝ := by
  funext i j l
  fin_cases i <;> fin_cases j <;> fin_cases l <;>
    simp [bilinearTensor, wRealize3, wTensor, Fin.sum_univ_three]

/-- **Width-3 Realizable Polynomial Contrast (`[LC]`):**
Adding a single hidden neuron (`k = 2 → 3`) makes `W` realizable (`tensorRank W = 3`), collapsing
the Bayesian free energy from stretched-exponential `N^{1/7}` back to Watanabe polynomial decay:
for all `N ≥ 1`, `2^{18} e^{-10404} · N^{-9} ≤ Z_W^{(3)}(N)`
(i.e. `-log Z_W^{(3)}(N) ≤ 9 log N + C₃`). -/
theorem wPartitionZ_three_ge_poly {N : ℝ} (hN : 1 ≤ N) :
    ENNReal.ofReal ((2 : ℝ) ^ 18 * Real.exp (-10404) * N ^ (-9 : ℝ)) ≤ wPartitionZ 3 N := by
  have hN_pos : 0 < N := by linarith
  let ρ : ℝ := N ^ (-(1 / 2 : ℝ))
  have hρ_pos : 0 < ρ := by positivity
  have hρ_le : ρ ≤ 1 := by
    simp only [ρ]
    rw [Real.rpow_neg hN_pos.le]
    exact inv_le_one_of_one_le₀ (Real.one_le_rpow hN (by positivity))
  have hθ₀_1 : ∀ i : Fin 2, ∀ r : Fin 3, |wRealize3.1 i r| ≤ 1 := by
    intro i r; fin_cases i <;> fin_cases r <;> simp [wRealize3]
  have hθ₀_2 : ∀ j : Fin 2, ∀ r : Fin 3, |wRealize3.2.1 j r| ≤ 1 := by
    intro j r; fin_cases j <;> fin_cases r <;> simp [wRealize3]
  have hθ₀_3 : ∀ l : Fin 2, ∀ r : Fin 3, |wRealize3.2.2 l r| ≤ 1 := by
    intro l r; fin_cases l <;> fin_cases r <;> simp [wRealize3]
  have hexp_le : ∀ θ ∈ paramBox wRealize3 ρ,
      N * bilinearLoss (wTensor ℝ) θ + (1 / 2 : ℝ) * paramNormSq θ ≤ 10404 := by
    intro θ hθ
    rcases hθ with ⟨h1, h2, h3⟩
    have hda : ∀ i r, |θ.1 i r - wRealize3.1 i r| ≤ ρ := fun i r => by
      have := h1 i r; rw [Set.mem_Icc] at this; rw [abs_le]; constructor <;> linarith
    have hdb : ∀ j r, |θ.2.1 j r - wRealize3.2.1 j r| ≤ ρ := fun j r => by
      have := h2 j r; rw [Set.mem_Icc] at this; rw [abs_le]; constructor <;> linarith
    have hdc : ∀ l r, |θ.2.2 l r - wRealize3.2.2 l r| ≤ ρ := fun l r => by
      have := h3 l r; rw [Set.mem_Icc] at this; rw [abs_le]; constructor <;> linarith
    have hθ_1 : ∀ i r, |θ.1 i r| ≤ 2 := fun i r => by
      calc |θ.1 i r| = |(θ.1 i r - wRealize3.1 i r) + wRealize3.1 i r| := by ring_nf
        _ ≤ |θ.1 i r - wRealize3.1 i r| + |wRealize3.1 i r| := abs_add_le _ _
        _ ≤ ρ + 1 := add_le_add (hda i r) (hθ₀_1 i r)
        _ ≤ 2 := by linarith
    have hθ_2 : ∀ j r, |θ.2.1 j r| ≤ 2 := fun j r => by
      calc |θ.2.1 j r| = |(θ.2.1 j r - wRealize3.2.1 j r) + wRealize3.2.1 j r| := by ring_nf
        _ ≤ |θ.2.1 j r - wRealize3.2.1 j r| + |wRealize3.2.1 j r| := abs_add_le _ _
        _ ≤ ρ + 1 := add_le_add (hdb j r) (hθ₀_2 j r)
        _ ≤ 2 := by linarith
    have hθ_3 : ∀ l r, |θ.2.2 l r| ≤ 2 := fun l r => by
      calc |θ.2.2 l r| = |(θ.2.2 l r - wRealize3.2.2 l r) + wRealize3.2.2 l r| := by ring_nf
        _ ≤ |θ.2.2 l r - wRealize3.2.2 l r| + |wRealize3.2.2 l r| := abs_add_le _ _
        _ ≤ ρ + 1 := add_le_add (hdc l r) (hθ₀_3 l r)
        _ ≤ 2 := by linarith
    have htri : ∀ x y z : ℝ, |x + y + z| ≤ |x| + |y| + |z| := fun x y z =>
      (abs_add_le (x + y) z).trans (by linarith [abs_add_le x y])
    have hTdiff : ∀ i j l : Fin 2, |bilinearTensor θ i j l - wTensor ℝ i j l| ≤ 36 * ρ := by
      intro i j l
      rw [← wRealize3_tensor]
      simp only [bilinearTensor, Fin.sum_univ_three]
      have hc0 := cubic_diff_bound (hθ_1 i 0) (hθ_2 j 0) (hθ_3 l 0)
        (le_trans (hθ₀_1 i 0) (by norm_num)) (le_trans (hθ₀_2 j 0) (by norm_num))
        (le_trans (hθ₀_3 l 0) (by norm_num)) (hda i 0) (hdb j 0) (hdc l 0) (by norm_num) hρ_pos.le
      have hc1 := cubic_diff_bound (hθ_1 i 1) (hθ_2 j 1) (hθ_3 l 1)
        (le_trans (hθ₀_1 i 1) (by norm_num)) (le_trans (hθ₀_2 j 1) (by norm_num))
        (le_trans (hθ₀_3 l 1) (by norm_num)) (hda i 1) (hdb j 1) (hdc l 1) (by norm_num) hρ_pos.le
      have hc2 := cubic_diff_bound (hθ_1 i 2) (hθ_2 j 2) (hθ_3 l 2)
        (le_trans (hθ₀_1 i 2) (by norm_num)) (le_trans (hθ₀_2 j 2) (by norm_num))
        (le_trans (hθ₀_3 l 2) (by norm_num)) (hda i 2) (hdb j 2) (hdc l 2) (by norm_num) hρ_pos.le
      have hsplit : (θ.1 i 0 * θ.2.1 j 0 * θ.2.2 l 0 + θ.1 i 1 * θ.2.1 j 1 * θ.2.2 l 1 +
            θ.1 i 2 * θ.2.1 j 2 * θ.2.2 l 2) -
          (wRealize3.1 i 0 * wRealize3.2.1 j 0 * wRealize3.2.2 l 0 +
            wRealize3.1 i 1 * wRealize3.2.1 j 1 * wRealize3.2.2 l 1 +
            wRealize3.1 i 2 * wRealize3.2.1 j 2 * wRealize3.2.2 l 2) =
          (θ.1 i 0 * θ.2.1 j 0 * θ.2.2 l 0 - wRealize3.1 i 0 * wRealize3.2.1 j 0 * wRealize3.2.2 l 0) +
          (θ.1 i 1 * θ.2.1 j 1 * θ.2.2 l 1 - wRealize3.1 i 1 * wRealize3.2.1 j 1 * wRealize3.2.2 l 1) +
          (θ.1 i 2 * θ.2.1 j 2 * θ.2.2 l 2 - wRealize3.1 i 2 * wRealize3.2.1 j 2 * wRealize3.2.2 l 2) := by ring
      rw [hsplit]
      exact (htri _ _ _).trans (by linarith)
    have hρ2 : ρ ^ 2 = N⁻¹ := by
      simp only [ρ]
      rw [← Real.rpow_natCast, ← Real.rpow_mul hN_pos.le]
      norm_num
      exact Real.rpow_neg_one N
    have hL_le : bilinearLoss (wTensor ℝ) θ ≤ 10368 * N⁻¹ := by
      have hsq : ∀ i j l : Fin 2, (bilinearTensor θ i j l - wTensor ℝ i j l) ^ 2 ≤ 1296 * N⁻¹ := by
        intro i j l
        have := hTdiff i j l
        have h1' : |bilinearTensor θ i j l - wTensor ℝ i j l| ^ 2 ≤ (36 * ρ) ^ 2 := by
          nlinarith [abs_nonneg (bilinearTensor θ i j l - wTensor ℝ i j l)]
        rw [sq_abs] at h1'
        calc (bilinearTensor θ i j l - wTensor ℝ i j l) ^ 2 ≤ 1296 * ρ ^ 2 := by linarith
          _ = 1296 * N⁻¹ := by rw [hρ2]
      calc bilinearLoss (wTensor ℝ) θ
        = ∑ i : Fin 2, ∑ j : Fin 2, ∑ l : Fin 2, (bilinearTensor θ i j l - wTensor ℝ i j l) ^ 2 := rfl
        _ ≤ ∑ _i : Fin 2, ∑ _j : Fin 2, ∑ _l : Fin 2, (1296 * N⁻¹) :=
          Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ =>
            Finset.sum_le_sum fun l _ => hsq i j l
        _ = 10368 * N⁻¹ := by simp; ring
    have hNL : N * bilinearLoss (wTensor ℝ) θ ≤ 10368 := by
      calc N * bilinearLoss (wTensor ℝ) θ ≤ N * (10368 * N⁻¹) :=
            mul_le_mul_of_nonneg_left hL_le hN_pos.le
        _ = 10368 := by field_simp
    have hR2_le : paramNormSq θ ≤ 72 := by
      have hsq1 : ∀ i r, (θ.1 i r) ^ 2 ≤ 4 := fun i r => by
        have := hθ_1 i r; rw [← sq_abs]; nlinarith [abs_nonneg (θ.1 i r)]
      have hsq2 : ∀ j r, (θ.2.1 j r) ^ 2 ≤ 4 := fun j r => by
        have := hθ_2 j r; rw [← sq_abs]; nlinarith [abs_nonneg (θ.2.1 j r)]
      have hsq3 : ∀ l r, (θ.2.2 l r) ^ 2 ≤ 4 := fun l r => by
        have := hθ_3 l r; rw [← sq_abs]; nlinarith [abs_nonneg (θ.2.2 l r)]
      unfold paramNormSq
      have hs1 : ∑ i : Fin 2, ∑ r : Fin 3, (θ.1 i r) ^ 2 ≤ 24 := by
        calc ∑ i : Fin 2, ∑ r : Fin 3, (θ.1 i r) ^ 2
          ≤ ∑ _i : Fin 2, ∑ _r : Fin 3, (4 : ℝ) :=
            Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun r _ => hsq1 i r
          _ = 24 := by norm_num
      have hs2 : ∑ j : Fin 2, ∑ r : Fin 3, (θ.2.1 j r) ^ 2 ≤ 24 := by
        calc ∑ j : Fin 2, ∑ r : Fin 3, (θ.2.1 j r) ^ 2
          ≤ ∑ _j : Fin 2, ∑ _r : Fin 3, (4 : ℝ) :=
            Finset.sum_le_sum fun j _ => Finset.sum_le_sum fun r _ => hsq2 j r
          _ = 24 := by norm_num
      have hs3 : ∑ l : Fin 2, ∑ r : Fin 3, (θ.2.2 l r) ^ 2 ≤ 24 := by
        calc ∑ l : Fin 2, ∑ r : Fin 3, (θ.2.2 l r) ^ 2
          ≤ ∑ _l : Fin 2, ∑ _r : Fin 3, (4 : ℝ) :=
            Finset.sum_le_sum fun l _ => Finset.sum_le_sum fun r _ => hsq3 l r
          _ = 24 := by norm_num
      linarith
    linarith
  have hlow : ENNReal.ofReal (Real.exp (-10404)) * volume (paramBox wRealize3 ρ) ≤
      wPartitionZ 3 N := by
    rw [(MeasureTheory.setLIntegral_const (paramBox wRealize3 ρ)
      (ENNReal.ofReal (Real.exp (-10404)))).symm]
    refine le_trans ?_ (MeasureTheory.setLIntegral_le_lintegral (paramBox wRealize3 ρ) _)
    exact MeasureTheory.setLIntegral_mono_ae' (measurableSet_paramBox wRealize3 ρ)
      (Filter.Eventually.of_forall fun θ hθ =>
        ENNReal.ofReal_le_ofReal (Real.exp_le_exp.mpr (by linarith [hexp_le θ hθ])))
  rw [volume_paramBox wRealize3 hρ_pos.le, ← ENNReal.ofReal_mul (Real.exp_pos _).le] at hlow
  refine le_trans ?_ hlow
  apply ENNReal.ofReal_le_ofReal
  have hρpow : ρ ^ 18 = N ^ (-9 : ℝ) := by
    simp only [ρ]
    rw [← Real.rpow_natCast, ← Real.rpow_mul hN_pos.le]
    norm_num
  have hρ18 : (2 * ρ) ^ (6 * 3) = (2 : ℝ) ^ 18 * N ^ (-9 : ℝ) := by
    rw [show 6 * 3 = 18 by rfl, mul_pow, hρpow]
  rw [hρ18]
  linarith

end EatYourFiber
