/-
Copyright (c) 2026 Bobby Faber. All rights reserved.
Released under Apache 2.0 license.
-/

import EatYourFiber.BorderRankSLT

/-!
# Positive-Measure Generic Border-Rank Singularity (`Det(T*) < 0`), the `γ^{3/4}` Weight-Decay Law,
and Stretched-Exponential Excess Free Energy (`β^{1/4}`)

This module extends the kernel-certified (`[LC]`) measure-zero `W`-tensor analysis in
`EatYourFiber.BorderRankSLT` (`R^{-12}`, `γ^{6/7}`, `N^{1/7}`) to the **positive-measure generic
real-rank-3 border-rank class** `{T* ∈ ℝ^{2×2×2} | Det(T*) < 0}`:

1. **Cayley's Hyperdeterminant & Positive-Measure Non-Realizability:**
   - `cayleyDet`: Cayley's degree-4 hyperdeterminant on `Tensor R (Fin 2) (Fin 2) (Fin 2)` over any
     commutative ring `R`.
   - `cayleyDet_bilinearTensor`: exact factorization
     `cayleyDet (bilinearTensor θ) = (det2 θ.1 * det2 θ.2.1 * det2 θ.2.2) ^ 2 ≥ 0`.
   - `not_rankAtMost_two_of_cayleyDet_neg`, `three_le_tensorRank_of_cayleyDet_neg`,
     `bilinearLoss_pos_of_cayleyDet_neg`: every target with `cayleyDet T* < 0` has real rank `≥ 3`
     and strictly positive width-2 loss.
   - `cayleyDet_wTensor` (`= 0`), `cayleyDet_t3Tensor` (`= -4`), `t3Tensor_tensorRank` (`= 3`, via
     the explicit rational 3-term decomposition `T₃ = 2 e₀^{⊗3} - ½ (e₀+e₁)^{⊗3} - ½ (e₀-e₁)^{⊗3}`).
   - `isOpen_cayleyDet_neg` and `volume_cayleyDet_neg_pos`: `{T | cayleyDet T < 0}` is a nonempty
     open set of **strictly positive Lebesgue measure** (explicitly containing the `L^∞`-box of
     half-width `1/8` around `t3Tensor ℝ`).

2. **Universal Unconditional `R^{-6}` Hyperdeterminant Barrier & Local/General Escape Laws:**
   - `mode1PencilSq`, `mode2PencilSq`, `mode3PencilSq`: the three `SO(2)³`-invariant squared
     mode slice-pencil invariants `2 (det M₀)² + 2 (det M₁)² + (tr(adj M₀ M₁))²`.
   - `cayleyDet_sq_ge_of_modePencils`: for *every* `θ : BilinearParam 2 2 2 2`,
     `5832 * Q₁(T(θ)) * Q₂(T(θ)) * Q₃(T(θ)) ≤ (cayleyDet (T(θ)))² * (paramNormSq θ)⁶`.
     This pinpoints the exact algebraic obstruction: rank-1 / multilinear-rank `< (2,2,2)` tensors
     have `Q₁ Q₂ Q₃ = 0` and can hit `Det = 0` at finite norm, whereas on *any* neighborhood bounded
     away from degenerate slice pencils (`Q_k ≥ m₀² > 0`), `cayleyDet (T(θ)) ≥ 76 m₀³ / ‖θ‖₂⁶`.
   - `GenericWallHyp`: clean non-degeneracy / strict-Lagrange-multiplier structure yielding the
     general lower bounds `generic_escape_lower` (`ΔL ≥ c R^{-6}`), `generic_weightDecay_lower`
     (`ΔL + γ R² ≥ c₁ γ^{3/4}`), and `genericPartitionZ_two_le_stretchedExp`
     (`-log Z(β) ≥ β L_wall + a₁ β^{1/4} - C`).
   - `gl3_escape_upper`, `gl3_weightDecay_upper`, `gl3PartitionZ_two_ge_stretchedExp`: matching
     general upper bounds (and partition-function lower bound) along the `GL(2)³`-transported
     centered border-rank curve for *any* smooth wall point `T_wall = (G₁, G₂, G₃) · W`.

3. **Unconditional Certification on Canonical `T₃ = Re((e₀ + i e₁)^{⊗3})` (`L_wall = 1`):**
   - Via an exact integer Hadamard transform into complex-unitary coordinates `(x_k, y_k)_{k=0}^3`
     and Heron's quartic factorization, we prove *globally and unconditionally* (zero unverified
     hypotheses!):
     - `t3Tensor_loss_ge_one`: `∀ T, 0 ≤ cayleyDet T → 1 ≤ ‖T - T₃‖_F²`.
     - `t3Tensor_genericWallHyp`: unconditional instance of `GenericWallHyp (t3Tensor ℝ) 1`.
     - `t3Tensor_excessLoss_ge_of_paramNormSq` & `t3Tensor_escape_upper`: `ΔL ≍ R^{-6}`.
     - `t3Tensor_weightDecay_lower` & `t3Tensor_weightDecay_upper`: `inf (ΔL + γ R²) ≍ γ^{3/4}`.
     - `t3PartitionZ_two_le_stretchedExp` & `t3PartitionZ_two_ge_stretchedExp`:
       `-log Z_{T₃}^{(2)}(β) - β ≍ β^{1/4}` for all `β ≥ 1`.
-/

namespace EatYourFiber

open MeasureTheory Set Filter Finset Real
open OAI.MatrixMultiplication.Foundation
open OAI.LinearAlgebra.SpectralBooks
open scoped ENNReal NNReal BigOperators

/-! ### 1. Cayley's hyperdeterminant, exact factorization, and positive-measure non-realizability -/

/-- Determinant of a `2 × 2` matrix `Fin 2 → Fin 2 → R`. -/
def det2 {R : Type*} [CommRing R] (M : Fin 2 → Fin 2 → R) : R :=
  M 0 0 * M 1 1 - M 0 1 * M 1 0

/-- Cayley's degree-4 hyperdeterminant on `2 × 2 × 2` tensors over any commutative ring `R`:
`Det(T) = (tr(adj(M₀) M₁))² - 4 (det M₀) (det M₁)` where `M_l(i, j) = T i j l`. -/
def cayleyDet {R : Type*} [CommRing R] (T : Tensor R (Fin 2) (Fin 2) (Fin 2)) : R :=
  (T 1 1 0 * T 0 0 1 - T 0 1 0 * T 1 0 1 - T 1 0 0 * T 0 1 1 + T 0 0 0 * T 1 1 1) ^ 2 -
    4 * (T 0 0 0 * T 1 1 0 - T 0 1 0 * T 1 0 0) * (T 0 0 1 * T 1 1 1 - T 0 1 1 * T 1 0 1)

/-- **Exact Hyperdeterminant Factorization (`[LC]`):**
For every width-2 bilinear MLP parameter `θ = (A, B, C)`,
`cayleyDet (bilinearTensor θ) = (det2 A * det2 B * det2 C)²`. -/
theorem cayleyDet_bilinearTensor (θ : BilinearParam 2 2 2 2) :
    cayleyDet (bilinearTensor θ) = (det2 θ.1 * det2 θ.2.1 * det2 θ.2.2) ^ 2 := by
  unfold cayleyDet bilinearTensor det2
  simp only [Fin.sum_univ_two]
  ring

/-- Over *any* commutative ring `R`, every tensor of rank `≤ 2` has hyperdeterminant a square. -/
theorem cayleyDet_rankAtMost_two {R : Type*} [CommRing R]
    {T : Tensor R (Fin 2) (Fin 2) (Fin 2)} (h : T.RankAtMost 2) :
    ∃ d : R, cayleyDet T = d ^ 2 := by
  rcases h with ⟨a, b, c, rfl⟩
  refine ⟨(a 0 0 * a 1 1 - a 1 0 * a 0 1) * (b 0 0 * b 1 1 - b 1 0 * b 0 1) *
      (c 0 0 * c 1 1 - c 1 0 * c 0 1), ?_⟩
  unfold cayleyDet
  simp only [Tensor.rankOne, Fin.sum_univ_two]
  ring

/-- Every real width-2 bilinear MLP tensor has nonnegative Cayley hyperdeterminant. -/
theorem cayleyDet_bilinearTensor_nonneg (θ : BilinearParam 2 2 2 2) :
    0 ≤ cayleyDet (bilinearTensor θ) := by
  rw [cayleyDet_bilinearTensor]
  exact sq_nonneg _

/-- Any real `2 × 2 × 2` tensor with `cayleyDet T* < 0` cannot have real rank `≤ 2`. -/
theorem not_rankAtMost_two_of_cayleyDet_neg
    {Tstar : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)} (h : cayleyDet Tstar < 0) :
    ¬ Tstar.RankAtMost 2 := by
  intro hr
  rcases cayleyDet_rankAtMost_two hr with ⟨d, hd⟩
  nlinarith [sq_nonneg d]

/-- Any real `2 × 2 × 2` tensor with `cayleyDet T* < 0` has real tensor rank `≥ 3`. -/
theorem three_le_tensorRank_of_cayleyDet_neg
    {Tstar : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)} (h : cayleyDet Tstar < 0) :
    3 ≤ tensorRank Tstar := by
  by_contra! hlt
  have hle2 : tensorRank Tstar ≤ 2 := Nat.le_of_lt_succ hlt
  obtain ⟨θ, hθ⟩ := (exists_zero_loss_iff_tensorRank_le Tstar 2).mpr hle2
  have h0 := (tensorFrobSq_eq_zero_iff _).mp hθ
  have heq : bilinearTensor θ = Tstar := by
    funext i j l
    have := congrFun (congrFun (congrFun h0 i) j) l
    exact sub_eq_zero.mp this
  have hnn := cayleyDet_bilinearTensor_nonneg θ
  rw [heq] at hnn
  linarith

/-- Any real `2 × 2 × 2` tensor with `cayleyDet T* < 0` has strictly positive width-2 loss
at every finite parameter `θ`. -/
theorem bilinearLoss_pos_of_cayleyDet_neg
    {Tstar : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)} (h : cayleyDet Tstar < 0)
    (θ : BilinearParam 2 2 2 2) :
    0 < bilinearLoss Tstar θ := by
  have hnn : 0 ≤ bilinearLoss Tstar θ := bilinearLoss_nonneg Tstar θ
  refine lt_of_le_of_ne hnn ?_
  intro heq
  have hr : Tstar.RankAtMost 2 := (exists_zero_loss_iff_rankAtMost Tstar 2).mp ⟨θ, heq.symm⟩
  exact not_rankAtMost_two_of_cayleyDet_neg h hr

/-- The canonical symmetric generic real-rank-3 tensor
`T₃ = Re((e₀ + i e₁)^{⊗3}) = e₀⊗e₀⊗e₀ - e₀⊗e₁⊗e₁ - e₁⊗e₀⊗e₁ - e₁⊗e₁⊗e₀`. -/
def t3Tensor (R : Type*) [CommRing R] : Tensor R (Fin 2) (Fin 2) (Fin 2) :=
  fun i j l =>
    if i = 0 ∧ j = 0 ∧ l = 0 then 1
    else if (i = 0 ∧ j = 1 ∧ l = 1) ∨ (i = 1 ∧ j = 0 ∧ l = 1) ∨ (i = 1 ∧ j = 1 ∧ l = 0) then -1
    else 0

theorem cayleyDet_wTensor (F : Type*) [Field F] : cayleyDet (wTensor F) = 0 := by
  unfold cayleyDet wTensor
  norm_num

theorem cayleyDet_t3Tensor (R : Type*) [CommRing R] : cayleyDet (t3Tensor R) = -4 := by
  unfold cayleyDet t3Tensor
  norm_num

/-- **Exact Real Tensor Rank of `T₃` (`[LC]`):** `tensorRank (t3Tensor ℝ) = 3`,
using the rational 3-term decomposition
`T₃ = 2 e₀^{⊗3} - (1/2) (e₀ + e₁)^{⊗3} - (1/2) (e₀ - e₁)^{⊗3}`. -/
theorem t3Tensor_tensorRank : tensorRank (t3Tensor ℝ) = 3 := by
  apply le_antisymm
  · apply tensorRank_le_of_rankAtMost
    refine ⟨
      fun r i => match r.1, i.1 with
        | 0, 0 => 2
        | 0, _ => 0
        | 1, _ => -(1 / 2 : ℝ)
        | _, 0 => -(1 / 2 : ℝ)
        | _, _ => 1 / 2,
      fun r j => match r.1, j.1 with
        | 0, 0 => 1
        | 0, _ => 0
        | 1, _ => 1
        | _, 0 => 1
        | _, _ => -1,
      fun r l => match r.1, l.1 with
        | 0, 0 => 1
        | 0, _ => 0
        | 1, _ => 1
        | _, 0 => 1
        | _, _ => -1,
      ?_⟩
    funext i j l
    fin_cases i <;> fin_cases j <;> fin_cases l <;>
      (simp [t3Tensor, Tensor.rankOne, Fin.sum_univ_three]; try norm_num)
  · apply three_le_tensorRank_of_cayleyDet_neg
    rw [cayleyDet_t3Tensor]
    norm_num

/-- `cayleyDet` is continuous on `Tensor ℝ (Fin 2) (Fin 2) (Fin 2)`. -/
theorem continuous_cayleyDet :
    Continuous (cayleyDet : Tensor ℝ (Fin 2) (Fin 2) (Fin 2) → ℝ) := by
  unfold cayleyDet
  fun_prop

/-- The generic real-rank-3 set `{T | cayleyDet T < 0}` is open in `ℝ^{2×2×2}`. -/
theorem isOpen_cayleyDet_neg :
    IsOpen {T : Tensor ℝ (Fin 2) (Fin 2) (Fin 2) | cayleyDet T < 0} :=
  isOpen_lt continuous_cayleyDet continuous_const

/-- Explicit `L^∞`-box of half-width `1 / 8` around `t3Tensor ℝ` lies strictly inside
`{T | cayleyDet T ≤ -7/4}` (and hence `{T | cayleyDet T < 0}`). -/
theorem cayleyDet_le_on_t3Box {T : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)}
    (hT : ∀ i j l : Fin 2, T i j l ∈ Set.Icc (t3Tensor ℝ i j l - 1 / 8) (t3Tensor ℝ i j l + 1 / 8)) :
    cayleyDet T ≤ -(7 / 4 : ℝ) := by
  have h000 : 7 / 8 ≤ T 0 0 0 ∧ T 0 0 0 ≤ 9 / 8 := by
    have h := Set.mem_Icc.mp (hT 0 0 0); simp only [t3Tensor, ↓reduceIte, and_self] at h
    constructor <;> linarith
  have h001 : -(1 / 8 : ℝ) ≤ T 0 0 1 ∧ T 0 0 1 ≤ 1 / 8 := by
    have h := Set.mem_Icc.mp (hT 0 0 1); simp only [t3Tensor] at h
    norm_num at h; constructor <;> linarith
  have h010 : -(1 / 8 : ℝ) ≤ T 0 1 0 ∧ T 0 1 0 ≤ 1 / 8 := by
    have h := Set.mem_Icc.mp (hT 0 1 0); simp only [t3Tensor] at h
    norm_num at h; constructor <;> linarith
  have h011 : -(9 / 8 : ℝ) ≤ T 0 1 1 ∧ T 0 1 1 ≤ -(7 / 8 : ℝ) := by
    have h := Set.mem_Icc.mp (hT 0 1 1); simp only [t3Tensor] at h
    norm_num at h; constructor <;> linarith
  have h100 : -(1 / 8 : ℝ) ≤ T 1 0 0 ∧ T 1 0 0 ≤ 1 / 8 := by
    have h := Set.mem_Icc.mp (hT 1 0 0); simp only [t3Tensor] at h
    norm_num at h; constructor <;> linarith
  have h101 : -(9 / 8 : ℝ) ≤ T 1 0 1 ∧ T 1 0 1 ≤ -(7 / 8 : ℝ) := by
    have h := Set.mem_Icc.mp (hT 1 0 1); simp only [t3Tensor] at h
    norm_num at h; constructor <;> linarith
  have h110 : -(9 / 8 : ℝ) ≤ T 1 1 0 ∧ T 1 1 0 ≤ -(7 / 8 : ℝ) := by
    have h := Set.mem_Icc.mp (hT 1 1 0); simp only [t3Tensor] at h
    norm_num at h; constructor <;> linarith
  have h111 : -(1 / 8 : ℝ) ≤ T 1 1 1 ∧ T 1 1 1 ≤ 1 / 8 := by
    have h := Set.mem_Icc.mp (hT 1 1 1); simp only [t3Tensor] at h
    norm_num at h; constructor <;> linarith
  set d0 := T 0 0 0 * T 1 1 0 - T 0 1 0 * T 1 0 0
  set d1 := T 0 0 1 * T 1 1 1 - T 0 1 1 * T 1 0 1
  set tr := T 1 1 0 * T 0 0 1 - T 0 1 0 * T 1 0 1 - T 1 0 0 * T 0 1 1 + T 0 0 0 * T 1 1 1
  have hd0 : d0 ≤ -(23 / 32 : ℝ) := by
    have h1 : T 0 0 0 * T 1 1 0 ≤ -(49 / 64 : ℝ) := by
      nlinarith only [h000.1, h000.2, h110.1, h110.2]
    have h2 : -(1 / 64 : ℝ) ≤ T 0 1 0 * T 1 0 0 := by
      nlinarith only [h010.1, h010.2, h100.1, h100.2]
    dsimp only [d0]; linarith
  have hd1 : d1 ≤ -(23 / 32 : ℝ) := by
    have h1 : T 0 0 1 * T 1 1 1 ≤ 1 / 64 := by
      nlinarith only [h001.1, h001.2, h111.1, h111.2]
    have h2 : 49 / 64 ≤ T 0 1 1 * T 1 0 1 := by
      nlinarith only [h011.1, h011.2, h101.1, h101.2]
    dsimp only [d1]; linarith
  have htr : -(9 / 16 : ℝ) ≤ tr ∧ tr ≤ 9 / 16 := by
    have p1 : -(9 / 64 : ℝ) ≤ T 1 1 0 * T 0 0 1 ∧ T 1 1 0 * T 0 0 1 ≤ 9 / 64 := by
      constructor <;> nlinarith only [h110.1, h110.2, h001.1, h001.2]
    have p2 : -(9 / 64 : ℝ) ≤ T 0 1 0 * T 1 0 1 ∧ T 0 1 0 * T 1 0 1 ≤ 9 / 64 := by
      constructor <;> nlinarith only [h010.1, h010.2, h101.1, h101.2]
    have p3 : -(9 / 64 : ℝ) ≤ T 1 0 0 * T 0 1 1 ∧ T 1 0 0 * T 0 1 1 ≤ 9 / 64 := by
      constructor <;> nlinarith only [h100.1, h100.2, h011.1, h011.2]
    have p4 : -(9 / 64 : ℝ) ≤ T 0 0 0 * T 1 1 1 ∧ T 0 0 0 * T 1 1 1 ≤ 9 / 64 := by
      constructor <;> nlinarith only [h000.1, h000.2, h111.1, h111.2]
    dsimp only [tr]; constructor <;> linarith
  have htr2 : tr ^ 2 ≤ (9 / 16 : ℝ) ^ 2 := by nlinarith only [htr.1, htr.2]
  have hd01 : (23 / 32 : ℝ) ^ 2 ≤ d0 * d1 := by nlinarith only [hd0, hd1]
  change tr ^ 2 - 4 * d0 * d1 ≤ -(7 / 4 : ℝ)
  nlinarith only [htr2, hd01]

/-- **Positive Lebesgue Measure of `{T | cayleyDet T < 0}` (`[LC]`):**
`ENNReal.ofReal (1 / 4) ^ 8 ≤ volume {T | cayleyDet T < 0}`, and in particular the generic
real-rank-3 border-rank locus has strictly positive 8-dimensional Lebesgue measure. -/
theorem volume_cayleyDet_neg_pos :
    0 < volume {T : Tensor ℝ (Fin 2) (Fin 2) (Fin 2) | cayleyDet T < 0} := by
  let B : Set (Tensor ℝ (Fin 2) (Fin 2) (Fin 2)) :=
    Set.pi Set.univ fun i => Set.pi Set.univ fun j => Set.pi Set.univ fun l =>
      Set.Icc (t3Tensor ℝ i j l - 1 / 8) (t3Tensor ℝ i j l + 1 / 8)
  have hsub : B ⊆ {T : Tensor ℝ (Fin 2) (Fin 2) (Fin 2) | cayleyDet T < 0} := by
    intro T hT
    have hmem : ∀ i j l : Fin 2,
        T i j l ∈ Set.Icc (t3Tensor ℝ i j l - 1 / 8) (t3Tensor ℝ i j l + 1 / 8) := by
      intro i j l
      exact hT i (Set.mem_univ i) j (Set.mem_univ j) l (Set.mem_univ l)
    have hle := cayleyDet_le_on_t3Box hmem
    change cayleyDet T < 0
    linarith
  have hvolB : volume B = ENNReal.ofReal (1 / 4 : ℝ) ^ 8 := by
    change (Measure.pi fun _ : Fin 2 =>
      (Measure.pi fun _ : Fin 2 =>
        (Measure.pi fun _ : Fin 2 => (volume : Measure ℝ)))) B = _
    simp only [B]
    rw [Measure.pi_pi]
    have hrow : ∀ i : Fin 2,
        (Measure.pi fun _ : Fin 2 => (Measure.pi fun _ : Fin 2 => (volume : Measure ℝ)))
          (Set.pi Set.univ fun j => Set.pi Set.univ fun l =>
            Set.Icc (t3Tensor ℝ i j l - 1 / 8) (t3Tensor ℝ i j l + 1 / 8)) =
          ENNReal.ofReal (1 / 4 : ℝ) ^ 4 := by
      intro i
      rw [Measure.pi_pi]
      have hcol : ∀ j : Fin 2,
          (Measure.pi fun _ : Fin 2 => (volume : Measure ℝ))
            (Set.pi Set.univ fun l =>
              Set.Icc (t3Tensor ℝ i j l - 1 / 8) (t3Tensor ℝ i j l + 1 / 8)) =
            ENNReal.ofReal (1 / 4 : ℝ) ^ 2 := by
        intro j
        rw [Measure.pi_pi]
        have hent : ∀ l : Fin 2,
            volume (Set.Icc (t3Tensor ℝ i j l - 1 / 8) (t3Tensor ℝ i j l + 1 / 8)) =
              ENNReal.ofReal (1 / 4 : ℝ) := by
          intro l; rw [Real.volume_Icc]; congr 1; ring
        rw [Finset.prod_congr rfl (fun l _ => hent l), Finset.prod_const, Finset.card_univ,
          Fintype.card_fin]
      rw [Finset.prod_congr rfl (fun j _ => hcol j), Finset.prod_const, Finset.card_univ,
        Fintype.card_fin, ← pow_mul]
    rw [Finset.prod_congr rfl (fun i _ => hrow i), Finset.prod_const, Finset.card_univ,
      Fintype.card_fin, ← pow_mul]
  have hpos : (0 : ℝ≥0∞) < ENNReal.ofReal (1 / 4 : ℝ) ^ 8 := by
    rw [← ENNReal.ofReal_pow (by norm_num : (0 : ℝ) ≤ 1 / 4)]
    exact ENNReal.ofReal_pos.mpr (by positivity)
  exact lt_of_lt_of_le hpos (by rw [← hvolB]; exact measure_mono hsub)

/-! ### 2. Unconditional universal `R^{-6}` hyperdeterminant barrier via mode slice pencils -/

/-- `SO(2)³`-invariant squared norm of the mode-3 slice pencil `det(u M₀ + v M₁)`:
`Q₃(T) = 2 (det M₀)² + 2 (det M₁)² + (tr(adj M₀ M₁))²`. -/
def mode3PencilSq (T : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)) : ℝ :=
  2 * (T 0 0 0 * T 1 1 0 - T 0 1 0 * T 1 0 0) ^ 2 +
  2 * (T 0 0 1 * T 1 1 1 - T 0 1 1 * T 1 0 1) ^ 2 +
  (T 1 1 0 * T 0 0 1 - T 0 1 0 * T 1 0 1 - T 1 0 0 * T 0 1 1 + T 0 0 0 * T 1 1 1) ^ 2

/-- `SO(2)³`-invariant squared norm of the mode-2 slice pencil. -/
def mode2PencilSq (T : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)) : ℝ :=
  2 * (T 0 0 0 * T 1 0 1 - T 0 0 1 * T 1 0 0) ^ 2 +
  2 * (T 0 1 0 * T 1 1 1 - T 0 1 1 * T 1 1 0) ^ 2 +
  (T 1 0 1 * T 0 1 0 - T 0 0 1 * T 1 1 0 - T 1 0 0 * T 0 1 1 + T 0 0 0 * T 1 1 1) ^ 2

/-- `SO(2)³`-invariant squared norm of the mode-1 slice pencil. -/
def mode1PencilSq (T : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)) : ℝ :=
  2 * (T 0 0 0 * T 0 1 1 - T 0 0 1 * T 0 1 0) ^ 2 +
  2 * (T 1 0 0 * T 1 1 1 - T 1 0 1 * T 1 1 0) ^ 2 +
  (T 0 1 1 * T 1 0 0 - T 0 0 1 * T 1 1 0 - T 0 1 0 * T 1 0 1 + T 0 0 0 * T 1 1 1) ^ 2

theorem mode1PencilSq_nonneg (T : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)) : 0 ≤ mode1PencilSq T := by
  unfold mode1PencilSq; positivity

theorem mode2PencilSq_nonneg (T : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)) : 0 ≤ mode2PencilSq T := by
  unfold mode2PencilSq; positivity

theorem mode3PencilSq_nonneg (T : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)) : 0 ≤ mode3PencilSq T := by
  unfold mode3PencilSq; positivity

/-- Elementary SOS inequality for the `2 × 2` factor matrix pencil norm. -/
private theorem factor_pencil_sos (c00 c01 c10 c11 : ℝ) :
    2 * (c00 * c01) ^ 2 + 2 * (c10 * c11) ^ 2 + (c01 * c10 + c00 * c11) ^ 2 ≤
      (1 / 2 : ℝ) * (c00 ^ 2 + c01 ^ 2 + c10 ^ 2 + c11 ^ 2) ^ 2 := by
  have hsos :
      (1 / 2 : ℝ) * (c00 ^ 2 + c01 ^ 2 + c10 ^ 2 + c11 ^ 2) ^ 2 -
        (2 * (c00 * c01) ^ 2 + 2 * (c10 * c11) ^ 2 + (c01 * c10 + c00 * c11) ^ 2) =
      (1 / 2 : ℝ) * (c00 ^ 2 - c01 ^ 2) ^ 2 + (1 / 2 : ℝ) * (c10 ^ 2 - c11 ^ 2) ^ 2 +
        (c00 * c10 - c01 * c11) ^ 2 := by ring
  nlinarith [sq_nonneg (c00 ^ 2 - c01 ^ 2), sq_nonneg (c10 ^ 2 - c11 ^ 2),
    sq_nonneg (c00 * c10 - c01 * c11)]

private theorem mode3PencilSq_bilinearTensor_le (θ : BilinearParam 2 2 2 2) :
    mode3PencilSq (bilinearTensor θ) ≤
      (1 / 2 : ℝ) * (det2 θ.1 * det2 θ.2.1) ^ 2 *
        (∑ l : Fin 2, ∑ r : Fin 2, (θ.2.2 l r) ^ 2) ^ 2 := by
  have hid : mode3PencilSq (bilinearTensor θ) =
      (det2 θ.1 * det2 θ.2.1) ^ 2 *
        (2 * (θ.2.2 0 0 * θ.2.2 0 1) ^ 2 + 2 * (θ.2.2 1 0 * θ.2.2 1 1) ^ 2 +
          (θ.2.2 0 1 * θ.2.2 1 0 + θ.2.2 0 0 * θ.2.2 1 1) ^ 2) := by
    unfold mode3PencilSq bilinearTensor det2
    simp only [Fin.sum_univ_two]
    ring
  rw [hid]
  have hsum : ∑ l : Fin 2, ∑ r : Fin 2, (θ.2.2 l r) ^ 2 =
      (θ.2.2 0 0) ^ 2 + (θ.2.2 0 1) ^ 2 + (θ.2.2 1 0) ^ 2 + (θ.2.2 1 1) ^ 2 := by
    simp only [Fin.sum_univ_two]; ring
  rw [hsum]
  have h := factor_pencil_sos (θ.2.2 0 0) (θ.2.2 0 1) (θ.2.2 1 0) (θ.2.2 1 1)
  have hmul := mul_le_mul_of_nonneg_left h (sq_nonneg (det2 θ.1 * det2 θ.2.1))
  linarith

private theorem mode2PencilSq_bilinearTensor_le (θ : BilinearParam 2 2 2 2) :
    mode2PencilSq (bilinearTensor θ) ≤
      (1 / 2 : ℝ) * (det2 θ.1 * det2 θ.2.2) ^ 2 *
        (∑ j : Fin 2, ∑ r : Fin 2, (θ.2.1 j r) ^ 2) ^ 2 := by
  have hid : mode2PencilSq (bilinearTensor θ) =
      (det2 θ.1 * det2 θ.2.2) ^ 2 *
        (2 * (θ.2.1 0 0 * θ.2.1 0 1) ^ 2 + 2 * (θ.2.1 1 0 * θ.2.1 1 1) ^ 2 +
          (θ.2.1 0 1 * θ.2.1 1 0 + θ.2.1 0 0 * θ.2.1 1 1) ^ 2) := by
    unfold mode2PencilSq bilinearTensor det2
    simp only [Fin.sum_univ_two]
    ring
  rw [hid]
  have hsum : ∑ j : Fin 2, ∑ r : Fin 2, (θ.2.1 j r) ^ 2 =
      (θ.2.1 0 0) ^ 2 + (θ.2.1 0 1) ^ 2 + (θ.2.1 1 0) ^ 2 + (θ.2.1 1 1) ^ 2 := by
    simp only [Fin.sum_univ_two]; ring
  rw [hsum]
  have h := factor_pencil_sos (θ.2.1 0 0) (θ.2.1 0 1) (θ.2.1 1 0) (θ.2.1 1 1)
  have hmul := mul_le_mul_of_nonneg_left h (sq_nonneg (det2 θ.1 * det2 θ.2.2))
  linarith

private theorem mode1PencilSq_bilinearTensor_le (θ : BilinearParam 2 2 2 2) :
    mode1PencilSq (bilinearTensor θ) ≤
      (1 / 2 : ℝ) * (det2 θ.2.1 * det2 θ.2.2) ^ 2 *
        (∑ i : Fin 2, ∑ r : Fin 2, (θ.1 i r) ^ 2) ^ 2 := by
  have hid : mode1PencilSq (bilinearTensor θ) =
      (det2 θ.2.1 * det2 θ.2.2) ^ 2 *
        (2 * (θ.1 0 0 * θ.1 0 1) ^ 2 + 2 * (θ.1 1 0 * θ.1 1 1) ^ 2 +
          (θ.1 0 1 * θ.1 1 0 + θ.1 0 0 * θ.1 1 1) ^ 2) := by
    unfold mode1PencilSq bilinearTensor det2
    simp only [Fin.sum_univ_two]
    ring
  rw [hid]
  have hsum : ∑ i : Fin 2, ∑ r : Fin 2, (θ.1 i r) ^ 2 =
      (θ.1 0 0) ^ 2 + (θ.1 0 1) ^ 2 + (θ.1 1 0) ^ 2 + (θ.1 1 1) ^ 2 := by
    simp only [Fin.sum_univ_two]; ring
  rw [hsum]
  have h := factor_pencil_sos (θ.1 0 0) (θ.1 0 1) (θ.1 1 0) (θ.1 1 1)
  have hmul := mul_le_mul_of_nonneg_left h (sq_nonneg (det2 θ.2.1 * det2 θ.2.2))
  linarith

/-- Elementary 3-variable AM–GM: `27 x y z ≤ (x + y + z)³` for `x, y, z ≥ 0`. -/
private theorem amgm3_cube {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
    27 * x * y * z ≤ (x + y + z) ^ 3 := by
  have hid : (x + y + z) ^ 3 - 27 * x * y * z =
      (1 / 2 : ℝ) * (x + y + z) * ((x - y) ^ 2 + (y - z) ^ 2 + (z - x) ^ 2) +
        3 * (x * (y - z) ^ 2 + y * (z - x) ^ 2 + z * (x - y) ^ 2) := by ring
  nlinarith [sq_nonneg (x - y), sq_nonneg (y - z), sq_nonneg (z - x),
    mul_nonneg hx (sq_nonneg (y - z)), mul_nonneg hy (sq_nonneg (z - x)),
    mul_nonneg hz (sq_nonneg (x - y)),
    mul_nonneg (by linarith : 0 ≤ x + y + z)
      (by nlinarith [sq_nonneg (x - y), sq_nonneg (y - z), sq_nonneg (z - x)] :
        0 ≤ (x - y) ^ 2 + (y - z) ^ 2 + (z - x) ^ 2)]

/-- **Universal Unconditional `R^{-6}` Hyperdeterminant Barrier (`[LC]`):**
For *every* width-2 bilinear MLP parameter `θ ∈ ℝ^{12}`,
`5832 * Q₁(T(θ)) * Q₂(T(θ)) * Q₃(T(θ)) ≤ (cayleyDet (T(θ)))² * (paramNormSq θ)⁶`.
This is sharp in the exponent (`Det(T(θ)) ≍ R^{-6}`) and isolates the exact algebraic obstruction:
rank-1 / multilinear-rank `< (2,2,2)` points vanish on the LHS (`Q₁ Q₂ Q₃ = 0`), whereas every
multilinear-rank `(2,2,2)` neighborhood enforces `Det(T(θ)) ≥ c / R⁶`. -/
theorem cayleyDet_sq_ge_of_modePencils (θ : BilinearParam 2 2 2 2) :
    5832 * mode1PencilSq (bilinearTensor θ) * mode2PencilSq (bilinearTensor θ) *
      mode3PencilSq (bilinearTensor θ) ≤
      (cayleyDet (bilinearTensor θ)) ^ 2 * (paramNormSq θ) ^ 6 := by
  set nA := ∑ i : Fin 2, ∑ r : Fin 2, (θ.1 i r) ^ 2
  set nB := ∑ j : Fin 2, ∑ r : Fin 2, (θ.2.1 j r) ^ 2
  set nC := ∑ l : Fin 2, ∑ r : Fin 2, (θ.2.2 l r) ^ 2
  have hnA : 0 ≤ nA := Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _
  have hnB : 0 ≤ nB := Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _
  have hnC : 0 ≤ nC := Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _
  have h1 := mode1PencilSq_bilinearTensor_le θ
  have h2 := mode2PencilSq_bilinearTensor_le θ
  have h3 := mode3PencilSq_bilinearTensor_le θ
  have h12 : mode1PencilSq (bilinearTensor θ) * mode2PencilSq (bilinearTensor θ) ≤
      ((1 / 2 : ℝ) * (det2 θ.2.1 * det2 θ.2.2) ^ 2 * nA ^ 2) *
        ((1 / 2 : ℝ) * (det2 θ.1 * det2 θ.2.2) ^ 2 * nB ^ 2) :=
    mul_le_mul h1 h2 (mode2PencilSq_nonneg _) (by positivity)
  have h123 : mode1PencilSq (bilinearTensor θ) * mode2PencilSq (bilinearTensor θ) *
      mode3PencilSq (bilinearTensor θ) ≤
      (((1 / 2 : ℝ) * (det2 θ.2.1 * det2 θ.2.2) ^ 2 * nA ^ 2) *
        ((1 / 2 : ℝ) * (det2 θ.1 * det2 θ.2.2) ^ 2 * nB ^ 2)) *
        ((1 / 2 : ℝ) * (det2 θ.1 * det2 θ.2.1) ^ 2 * nC ^ 2) :=
    mul_le_mul h12 h3 (mode3PencilSq_nonneg _) (by positivity)
  have ham := amgm3_cube hnA hnB hnC
  have ham2 : (27 * nA * nB * nC) ^ 2 ≤ ((nA + nB + nC) ^ 3) ^ 2 := by
    rw [sq, sq]
    exact mul_le_mul ham ham (by positivity) (by positivity)
  have hnorm : nA + nB + nC = paramNormSq θ := rfl
  rw [hnorm] at ham2
  rw [cayleyDet_bilinearTensor]
  have hrhs1 :
      5832 * ((((1 / 2 : ℝ) * (det2 θ.2.1 * det2 θ.2.2) ^ 2 * nA ^ 2) *
        ((1 / 2 : ℝ) * (det2 θ.1 * det2 θ.2.2) ^ 2 * nB ^ 2)) *
        ((1 / 2 : ℝ) * (det2 θ.1 * det2 θ.2.1) ^ 2 * nC ^ 2)) =
      ((det2 θ.1 * det2 θ.2.1 * det2 θ.2.2) ^ 2) ^ 2 * (27 * nA * nB * nC) ^ 2 := by ring
  have hmul5832 :
      5832 * mode1PencilSq (bilinearTensor θ) * mode2PencilSq (bilinearTensor θ) *
        mode3PencilSq (bilinearTensor θ) ≤
      ((det2 θ.1 * det2 θ.2.1 * det2 θ.2.2) ^ 2) ^ 2 * (27 * nA * nB * nC) ^ 2 := by
    calc 5832 * mode1PencilSq (bilinearTensor θ) * mode2PencilSq (bilinearTensor θ) *
          mode3PencilSq (bilinearTensor θ)
      = 5832 * (mode1PencilSq (bilinearTensor θ) * mode2PencilSq (bilinearTensor θ) *
          mode3PencilSq (bilinearTensor θ)) := by ring
      _ ≤ 5832 * ((((1 / 2 : ℝ) * (det2 θ.2.1 * det2 θ.2.2) ^ 2 * nA ^ 2) *
            ((1 / 2 : ℝ) * (det2 θ.1 * det2 θ.2.2) ^ 2 * nB ^ 2)) *
            ((1 / 2 : ℝ) * (det2 θ.1 * det2 θ.2.1) ^ 2 * nC ^ 2)) :=
          mul_le_mul_of_nonneg_left h123 (by norm_num)
      _ = ((det2 θ.1 * det2 θ.2.1 * det2 θ.2.2) ^ 2) ^ 2 * (27 * nA * nB * nC) ^ 2 := hrhs1
  have hpow6 : ((paramNormSq θ) ^ 3) ^ 2 = (paramNormSq θ) ^ 6 := by ring
  calc 5832 * mode1PencilSq (bilinearTensor θ) * mode2PencilSq (bilinearTensor θ) *
        mode3PencilSq (bilinearTensor θ)
    ≤ ((det2 θ.1 * det2 θ.2.1 * det2 θ.2.2) ^ 2) ^ 2 * (27 * nA * nB * nC) ^ 2 := hmul5832
    _ ≤ ((det2 θ.1 * det2 θ.2.1 * det2 θ.2.2) ^ 2) ^ 2 * ((paramNormSq θ) ^ 3) ^ 2 :=
        mul_le_mul_of_nonneg_left ham2 (by positivity)
    _ = ((det2 θ.1 * det2 θ.2.1 * det2 θ.2.2) ^ 2) ^ 2 * (paramNormSq θ) ^ 6 := by rw [hpow6]

/-- **Local `R^{-6}` Hyperdeterminant Lower Bound (`[LC]`):**
Whenever all three mode slice-pencil invariants satisfy `m₀² ≤ Q_k(T(θ))` (bounding `T(θ)` away
from rank-1 / multilinear-rank `< (2,2,2)` tensors),
`76 * m₀³ ≤ cayleyDet (bilinearTensor θ) * (paramNormSq θ)³`. -/
theorem cayleyDet_ge_of_modePencils (θ : BilinearParam 2 2 2 2) {m₀ : ℝ} (hm₀ : 0 ≤ m₀)
    (h1 : m₀ ^ 2 ≤ mode1PencilSq (bilinearTensor θ))
    (h2 : m₀ ^ 2 ≤ mode2PencilSq (bilinearTensor θ))
    (h3 : m₀ ^ 2 ≤ mode3PencilSq (bilinearTensor θ)) :
    76 * m₀ ^ 3 ≤ cayleyDet (bilinearTensor θ) * (paramNormSq θ) ^ 3 := by
  have hsq := cayleyDet_sq_ge_of_modePencils θ
  have h12 : m₀ ^ 2 * m₀ ^ 2 ≤
      mode1PencilSq (bilinearTensor θ) * mode2PencilSq (bilinearTensor θ) :=
    mul_le_mul h1 h2 (sq_nonneg _) (mode1PencilSq_nonneg _)
  have h123 : m₀ ^ 2 * m₀ ^ 2 * m₀ ^ 2 ≤
      mode1PencilSq (bilinearTensor θ) * mode2PencilSq (bilinearTensor θ) *
        mode3PencilSq (bilinearTensor θ) :=
    mul_le_mul h12 h3 (sq_nonneg _) (mul_nonneg (mode1PencilSq_nonneg _) (mode2PencilSq_nonneg _))
  have h5776 : (76 * m₀ ^ 3) ^ 2 ≤ (cayleyDet (bilinearTensor θ) * (paramNormSq θ) ^ 3) ^ 2 := by
    calc (76 * m₀ ^ 3) ^ 2
      = 5776 * (m₀ ^ 2 * m₀ ^ 2 * m₀ ^ 2) := by ring
      _ ≤ 5832 * (m₀ ^ 2 * m₀ ^ 2 * m₀ ^ 2) := by nlinarith [pow_nonneg hm₀ 6]
      _ ≤ 5832 * (mode1PencilSq (bilinearTensor θ) * mode2PencilSq (bilinearTensor θ) *
            mode3PencilSq (bilinearTensor θ)) := mul_le_mul_of_nonneg_left h123 (by norm_num)
      _ = 5832 * mode1PencilSq (bilinearTensor θ) * mode2PencilSq (bilinearTensor θ) *
            mode3PencilSq (bilinearTensor θ) := by ring
      _ ≤ (cayleyDet (bilinearTensor θ)) ^ 2 * (paramNormSq θ) ^ 6 := hsq
      _ = (cayleyDet (bilinearTensor θ) * (paramNormSq θ) ^ 3) ^ 2 := by ring
  have hnn : 0 ≤ cayleyDet (bilinearTensor θ) * (paramNormSq θ) ^ 3 :=
    mul_nonneg (cayleyDet_bilinearTensor_nonneg θ) (pow_nonneg (paramNormSq_nonneg θ) 3)
  nlinarith [pow_nonneg hm₀ 3, hnn]

/-- Division form of the local `R^{-6}` hyperdeterminant lower bound when `m₀ > 0`. -/
theorem local_cayleyDet_escape_lower (θ : BilinearParam 2 2 2 2) {m₀ : ℝ} (hm₀ : 0 < m₀)
    (h1 : m₀ ^ 2 ≤ mode1PencilSq (bilinearTensor θ))
    (h2 : m₀ ^ 2 ≤ mode2PencilSq (bilinearTensor θ))
    (h3 : m₀ ^ 2 ≤ mode3PencilSq (bilinearTensor θ)) :
    0 < paramNormSq θ ∧ 76 * m₀ ^ 3 / (paramNormSq θ) ^ 3 ≤ cayleyDet (bilinearTensor θ) := by
  have hmul := cayleyDet_ge_of_modePencils θ hm₀.le h1 h2 h3
  have hR2_pos : 0 < paramNormSq θ := by
    by_contra hle
    push Not at hle
    have h0 : paramNormSq θ = 0 := le_antisymm hle (paramNormSq_nonneg θ)
    rw [h0] at hmul
    have hpos : 0 < 76 * m₀ ^ 3 := by positivity
    linarith
  refine ⟨hR2_pos, ?_⟩
  rw [div_le_iff₀ (pow_pos hR2_pos 3)]
  exact hmul

/-! ### 3. General `GenericWallHyp` framework (`R^{-6}`, `γ^{3/4}`, `β^{1/4}`) and `GL(2)³` upper bounds -/

/-- Non-degeneracy and strict Lagrange-multiplier wall hypothesis for a generic `Det(T*) < 0`
target tensor `Tstar` with wall loss `Lwall`:
1. Every `Det(T) ≥ 0` tensor satisfies `Lwall ≤ ‖T - T*‖_F²`.
2. On the near-wall sublevel set `‖T - T*‖_F² - Lwall < δ₀`, every `Det(T) ≥ 0` tensor stays
   bounded away from degenerate multilinear-rank `< (2,2,2)` (including rank-1) points
   (`m₀² ≤ Q_k(T)`) and obeys the strict positive Lagrange-multiplier bound
   `α * cayleyDet T ≤ ‖T - T*‖_F² - Lwall`. -/
structure GenericWallHyp (Tstar : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)) (Lwall : ℝ) where
  δ₀ : ℝ
  α : ℝ
  m₀ : ℝ
  hδ₀ : 0 < δ₀
  hα : 0 < α
  hm₀ : 0 < m₀
  wall_le : ∀ T : Tensor ℝ (Fin 2) (Fin 2) (Fin 2),
    0 ≤ cayleyDet T → Lwall ≤ tensorFrobSq (fun i j l => T i j l - Tstar i j l)
  near_wall_nondeg : ∀ T : Tensor ℝ (Fin 2) (Fin 2) (Fin 2),
    0 ≤ cayleyDet T →
    tensorFrobSq (fun i j l => T i j l - Tstar i j l) - Lwall < δ₀ →
    m₀ ^ 2 ≤ mode1PencilSq T ∧ m₀ ^ 2 ≤ mode2PencilSq T ∧ m₀ ^ 2 ≤ mode3PencilSq T
  near_wall_linear : ∀ T : Tensor ℝ (Fin 2) (Fin 2) (Fin 2),
    0 ≤ cayleyDet T →
    tensorFrobSq (fun i j l => T i j l - Tstar i j l) - Lwall < δ₀ →
    α * cayleyDet T ≤ tensorFrobSq (fun i j l => T i j l - Tstar i j l) - Lwall

/-- **General `R^{-6}` Excess-Loss Lower Bound (`[LC]`):**
Under `GenericWallHyp Tstar Lwall`, every width-2 parameter `θ` satisfies
`min h.δ₀ (76 * h.α * h.m₀³ / (paramNormSq θ)³) ≤ bilinearLoss Tstar θ - Lwall`. -/
theorem generic_escape_lower {Tstar : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)} {Lwall : ℝ}
    (h : GenericWallHyp Tstar Lwall) (θ : BilinearParam 2 2 2 2) :
    min h.δ₀ (76 * h.α * h.m₀ ^ 3 / (paramNormSq θ) ^ 3) ≤ bilinearLoss Tstar θ - Lwall := by
  by_cases hbig : h.δ₀ ≤ bilinearLoss Tstar θ - Lwall
  · exact le_trans (min_le_left _ _) hbig
  push Not at hbig
  apply le_trans (min_le_right _ _)
  have hdet_nn : 0 ≤ cayleyDet (bilinearTensor θ) := cayleyDet_bilinearTensor_nonneg θ
  obtain ⟨h1, h2, h3⟩ := h.near_wall_nondeg (bilinearTensor θ) hdet_nn hbig
  have hlin := h.near_wall_linear (bilinearTensor θ) hdet_nn hbig
  obtain ⟨_hR2_pos, hdet_lb⟩ := local_cayleyDet_escape_lower θ h.hm₀ h1 h2 h3
  calc 76 * h.α * h.m₀ ^ 3 / (paramNormSq θ) ^ 3
    = h.α * (76 * h.m₀ ^ 3 / (paramNormSq θ) ^ 3) := by ring
    _ ≤ h.α * cayleyDet (bilinearTensor θ) := mul_le_mul_of_nonneg_left hdet_lb h.hα.le
    _ ≤ bilinearLoss Tstar θ - Lwall := hlin

/-- Explicit positive weight-decay constant determined by `GenericWallHyp`. -/
noncomputable def genericWDConst {Tstar : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)} {Lwall : ℝ}
    (h : GenericWallHyp Tstar Lwall) : ℝ :=
  min h.δ₀ ((4 * (76 * h.α * h.m₀ ^ 3)) ^ (1 / 4 : ℝ) * (4 / 3 : ℝ) ^ (3 / 4 : ℝ))

theorem genericWDConst_pos {Tstar : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)} {Lwall : ℝ}
    (h : GenericWallHyp Tstar Lwall) : 0 < genericWDConst h := by
  unfold genericWDConst
  refine lt_min h.hδ₀ ?_
  have hA : 0 < 4 * (76 * h.α * h.m₀ ^ 3) := by
    have := h.hα; have := h.hm₀; positivity
  positivity

/-- **General Weight-Decay Scaling Law — Lower Bound (`γ^{3/4}`, `[LC]`):**
Under `GenericWallHyp Tstar Lwall`, for all `γ ∈ (0, 1]` and all `θ : BilinearParam 2 2 2 2`,
`genericWDConst h * γ^{3/4} ≤ (bilinearLoss Tstar θ - Lwall) + γ * paramNormSq θ`. -/
theorem generic_weightDecay_lower {Tstar : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)} {Lwall : ℝ}
    (h : GenericWallHyp Tstar Lwall) {γ : ℝ} (hγ0 : 0 < γ) (hγ1 : γ ≤ 1)
    (θ : BilinearParam 2 2 2 2) :
    genericWDConst h * γ ^ (3 / 4 : ℝ) ≤
      (bilinearLoss Tstar θ - Lwall) + γ * paramNormSq θ := by
  set ΔL := bilinearLoss Tstar θ - Lwall
  set R2 := paramNormSq θ
  have hR2_nonneg : 0 ≤ R2 := paramNormSq_nonneg θ
  have hγpow_le : γ ^ (3 / 4 : ℝ) ≤ 1 := Real.rpow_le_one hγ0.le hγ1 (by positivity)
  by_cases hbig : h.δ₀ ≤ ΔL
  · have h1 : genericWDConst h ≤ h.δ₀ := min_le_left _ _
    have h2 : genericWDConst h * γ ^ (3 / 4 : ℝ) ≤ genericWDConst h :=
      mul_le_of_le_one_right (genericWDConst_pos h).le hγpow_le
    have h3 : 0 ≤ γ * R2 := mul_nonneg hγ0.le hR2_nonneg
    linarith
  push Not at hbig
  have hdet_nn : 0 ≤ cayleyDet (bilinearTensor θ) := cayleyDet_bilinearTensor_nonneg θ
  obtain ⟨h1, h2, h3⟩ := h.near_wall_nondeg (bilinearTensor θ) hdet_nn hbig
  have hlin := h.near_wall_linear (bilinearTensor θ) hdet_nn hbig
  obtain ⟨hR2_pos0, hdet_lb⟩ := local_cayleyDet_escape_lower θ h.hm₀ h1 h2 h3
  have hR2_pos : 0 < R2 := hR2_pos0
  set A := 76 * h.α * h.m₀ ^ 3
  have hA_pos : 0 < A := by have := h.hα; have := h.hm₀; positivity
  have hΔL_lb : A / R2 ^ 3 ≤ ΔL := by
    calc A / R2 ^ 3
      = h.α * (76 * h.m₀ ^ 3 / R2 ^ 3) := by ring
      _ ≤ h.α * cayleyDet (bilinearTensor θ) := mul_le_mul_of_nonneg_left hdet_lb h.hα.le
      _ ≤ ΔL := hlin
  -- Weighted AM–GM with weights `(1/4, 3/4)`: `A / R2³ + γ R2 ≥ (4A)^{1/4} (4/3)^{3/4} γ^{3/4}`.
  have hamgm := Real.geom_mean_le_arith_mean2_weighted
    (w₁ := (1 / 4 : ℝ)) (w₂ := (3 / 4 : ℝ))
    (p₁ := 4 * A / R2 ^ 3) (p₂ := 4 * γ * R2 / 3)
    (by norm_num) (by norm_num) (by positivity) (by positivity) (by norm_num)
  have hrhs : (1 / 4 : ℝ) * (4 * A / R2 ^ 3) + (3 / 4 : ℝ) * (4 * γ * R2 / 3) =
      A / R2 ^ 3 + γ * R2 := by ring
  rw [hrhs] at hamgm
  have hlhs : (4 * A / R2 ^ 3) ^ (1 / 4 : ℝ) * (4 * γ * R2 / 3) ^ (3 / 4 : ℝ) =
      ((4 * A) ^ (1 / 4 : ℝ) * (4 / 3 : ℝ) ^ (3 / 4 : ℝ)) * γ ^ (3 / 4 : ℝ) := by
    have hsplit1 : (4 * A / R2 ^ 3) ^ (1 / 4 : ℝ) =
        (4 * A) ^ (1 / 4 : ℝ) * R2 ^ (-(3 / 4 : ℝ)) := by
      have hrew : 4 * A / R2 ^ 3 = (4 * A) * R2 ^ (-3 : ℝ) := by
        rw [Real.rpow_neg hR2_pos.le, show (3 : ℝ) = (3 : ℕ) by norm_num, Real.rpow_natCast]
        ring
      rw [hrew, Real.mul_rpow (by positivity) (by positivity), ← Real.rpow_mul hR2_pos.le]
      norm_num
    have hsplit2 : (4 * γ * R2 / 3) ^ (3 / 4 : ℝ) =
        (4 / 3 : ℝ) ^ (3 / 4 : ℝ) * γ ^ (3 / 4 : ℝ) * R2 ^ (3 / 4 : ℝ) := by
      rw [show 4 * γ * R2 / 3 = ((4 / 3 : ℝ) * γ) * R2 by ring,
        Real.mul_rpow (by positivity) hR2_pos.le,
        Real.mul_rpow (by positivity) hγ0.le]
    rw [hsplit1, hsplit2]
    have hcancel : R2 ^ (-(3 / 4 : ℝ)) * R2 ^ (3 / 4 : ℝ) = 1 := by
      rw [← Real.rpow_add hR2_pos]; norm_num
    calc (4 * A) ^ (1 / 4 : ℝ) * R2 ^ (-(3 / 4 : ℝ)) *
          ((4 / 3 : ℝ) ^ (3 / 4 : ℝ) * γ ^ (3 / 4 : ℝ) * R2 ^ (3 / 4 : ℝ))
      = (4 * A) ^ (1 / 4 : ℝ) * (4 / 3 : ℝ) ^ (3 / 4 : ℝ) * γ ^ (3 / 4 : ℝ) *
          (R2 ^ (-(3 / 4 : ℝ)) * R2 ^ (3 / 4 : ℝ)) := by ring
      _ = ((4 * A) ^ (1 / 4 : ℝ) * (4 / 3 : ℝ) ^ (3 / 4 : ℝ)) * γ ^ (3 / 4 : ℝ) := by
          rw [hcancel, mul_one]
  have hconst : genericWDConst h ≤ (4 * A) ^ (1 / 4 : ℝ) * (4 / 3 : ℝ) ^ (3 / 4 : ℝ) :=
    min_le_right _ _
  calc genericWDConst h * γ ^ (3 / 4 : ℝ)
    ≤ ((4 * A) ^ (1 / 4 : ℝ) * (4 / 3 : ℝ) ^ (3 / 4 : ℝ)) * γ ^ (3 / 4 : ℝ) :=
      mul_le_mul_of_nonneg_right hconst (by positivity)
    _ = (4 * A / R2 ^ 3) ^ (1 / 4 : ℝ) * (4 * γ * R2 / 3) ^ (3 / 4 : ℝ) := hlhs.symm
    _ ≤ A / R2 ^ 3 + γ * R2 := hamgm
    _ ≤ ΔL + γ * R2 := by linarith

/-- The Gaussian-prior global Bayesian partition function for a width-`k` bilinear MLP
targeting an arbitrary `2 × 2 × 2` tensor `Tstar`:
`Z_{T*}^{(k)}(β) = ∫ exp(-(β ‖T(θ) - T*‖_F² + (1/2) ‖θ‖₂²)) dθ`. -/
noncomputable def tensorPartitionZ (Tstar : Tensor ℝ (Fin 2) (Fin 2) (Fin 2))
    (k : ℕ) (β : ℝ) : ℝ≥0∞ :=
  ∫⁻ θ : BilinearParam 2 2 2 k,
    ENNReal.ofReal (Real.exp (-(β * bilinearLoss Tstar θ + (1 / 2 : ℝ) * paramNormSq θ)))

/-! Local copies of the 12D Gaussian and box-volume helpers (private in `BorderRankSLT`). -/

private theorem gen_lintegral_gauss1d {a : ℝ} (ha : 0 < a) :
    (∫⁻ x : ℝ, ENNReal.ofReal (Real.exp (-(a * x ^ 2)))) =
      ENNReal.ofReal (Real.sqrt (Real.pi / a)) := by
  have hint : Integrable (fun x : ℝ => Real.exp (-(a * x ^ 2))) volume := by
    simpa [neg_mul] using integrable_exp_neg_mul_sq ha
  rw [← ofReal_integral_eq_lintegral_ofReal hint
    (Filter.Eventually.of_forall fun _ => (Real.exp_pos _).le)]
  have hval : (∫ x : ℝ, Real.exp (-(a * x ^ 2))) = Real.sqrt (Real.pi / a) := by
    simpa [neg_mul] using integral_gaussian a
  rw [hval]

private theorem gen_lintegral_gauss_vec {d : ℕ} {a : ℝ} (ha : 0 < a) :
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
      congr 1; funext v
      simp only [MeasurableEquiv.piFinSuccAbove_apply, Fin.sum_univ_succ]; rfl
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
    rw [ih, MeasureTheory.lintegral_mul_const' _ _ ENNReal.ofReal_ne_top, gen_lintegral_gauss1d ha,
      ← ENNReal.ofReal_mul (Real.sqrt_nonneg _), pow_succ']

private theorem gen_lintegral_gauss_mat {a b : ℕ} {c : ℝ} (hc : 0 < c) :
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
      congr 1; funext M
      simp only [MeasurableEquiv.piFinSuccAbove_apply, Fin.sum_univ_succ]; rfl
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
    rw [ih, MeasureTheory.lintegral_mul_const' _ _ ENNReal.ofReal_ne_top, gen_lintegral_gauss_vec hc,
      ← ENNReal.ofReal_mul (pow_nonneg (Real.sqrt_nonneg _) _), ← pow_add,
      show b + m * b = (m + 1) * b by ring]

private theorem gen_lintegral_gaussian_param2 :
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
    rw [gen_lintegral_gauss_mat hc, MeasureTheory.lintegral_mul_const' _ _ ENNReal.ofReal_ne_top,
      gen_lintegral_gauss_mat hc, ← ENNReal.ofReal_mul (by positivity)]
  simp_rw [hinner]
  rw [MeasureTheory.lintegral_mul_const' _ _ ENNReal.ofReal_ne_top, gen_lintegral_gauss_mat hc,
    ← ENNReal.ofReal_mul (by positivity)]
  congr 1
  have hsq : Real.sqrt (Real.pi / (1 / 4 : ℝ)) ^ 2 = 4 * Real.pi := by
    rw [Real.sq_sqrt (by positivity)]; ring
  calc Real.sqrt (Real.pi / (1 / 4 : ℝ)) ^ (2 * 2) *
        (Real.sqrt (Real.pi / (1 / 4 : ℝ)) ^ 4 * Real.sqrt (Real.pi / (1 / 4 : ℝ)) ^ 4)
    = (Real.sqrt (Real.pi / (1 / 4 : ℝ)) ^ 2) ^ 6 := by ring
    _ = (4 * Real.pi) ^ 6 := by rw [hsq]

private def genParamBox {k : ℕ} (θ₀ : BilinearParam 2 2 2 k) (ρ : ℝ) :
    Set (BilinearParam 2 2 2 k) :=
  {θ | (∀ i r, θ.1 i r ∈ Set.Icc (θ₀.1 i r - ρ) (θ₀.1 i r + ρ)) ∧
       (∀ j r, θ.2.1 j r ∈ Set.Icc (θ₀.2.1 j r - ρ) (θ₀.2.1 j r + ρ)) ∧
       (∀ l r, θ.2.2 l r ∈ Set.Icc (θ₀.2.2 l r - ρ) (θ₀.2.2 l r + ρ))}

private theorem gen_volume_matBox_shifted {a b : ℕ} (M₀ : RMat a b) {ρ : ℝ} (_hρ : 0 ≤ ρ) :
    volume {M : RMat a b | ∀ i j, M i j ∈ Set.Icc (M₀ i j - ρ) (M₀ i j + ρ)} =
      ENNReal.ofReal (2 * ρ) ^ (a * b) := by
  have hset : {M : RMat a b | ∀ i j, M i j ∈ Set.Icc (M₀ i j - ρ) (M₀ i j + ρ)} =
      Set.pi Set.univ (fun i : Fin a => Set.pi Set.univ (fun j : Fin b => Set.Icc (M₀ i j - ρ) (M₀ i j + ρ))) := by
    ext M; simp only [Set.mem_ofPred_eq, Set.mem_pi, Set.mem_univ, forall_const]
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
      intro j; rw [Real.volume_Icc]; congr 1; ring
    rw [Finset.prod_congr rfl (fun j _ => h1 j), Finset.prod_const, Finset.card_univ, Fintype.card_fin]
  rw [Finset.prod_congr rfl (fun i _ => hrow i), Finset.prod_const, Finset.card_univ, Fintype.card_fin,
    ← pow_mul, mul_comm b a]

private theorem gen_measurableSet_paramBox {k : ℕ} (θ₀ : BilinearParam 2 2 2 k) (ρ : ℝ) :
    MeasurableSet (genParamBox θ₀ ρ) := by
  have hprod : genParamBox θ₀ ρ =
      (Set.pi Set.univ fun i => Set.pi Set.univ fun r => Set.Icc (θ₀.1 i r - ρ) (θ₀.1 i r + ρ)) ×ˢ
      ((Set.pi Set.univ fun j => Set.pi Set.univ fun r => Set.Icc (θ₀.2.1 j r - ρ) (θ₀.2.1 j r + ρ)) ×ˢ
       (Set.pi Set.univ fun l => Set.pi Set.univ fun r => Set.Icc (θ₀.2.2 l r - ρ) (θ₀.2.2 l r + ρ))) := by
    ext ⟨A, B, C⟩
    simp only [genParamBox, Set.mem_ofPred_eq, Set.mem_prod, Set.mem_pi, Set.mem_univ, forall_const]
  rw [hprod]
  refine MeasurableSet.prod ?_ (MeasurableSet.prod ?_ ?_) <;>
    exact MeasurableSet.univ_pi fun _ => MeasurableSet.univ_pi fun _ => measurableSet_Icc

private theorem gen_volume_paramBox {k : ℕ} (θ₀ : BilinearParam 2 2 2 k) {ρ : ℝ} (hρ : 0 ≤ ρ) :
    volume (genParamBox θ₀ ρ) = ENNReal.ofReal ((2 * ρ) ^ (6 * k)) := by
  have hprod : genParamBox θ₀ ρ =
      {A : RMat 2 k | ∀ i r, A i r ∈ Set.Icc (θ₀.1 i r - ρ) (θ₀.1 i r + ρ)} ×ˢ
      ({B : RMat 2 k | ∀ j r, B j r ∈ Set.Icc (θ₀.2.1 j r - ρ) (θ₀.2.1 j r + ρ)} ×ˢ
       {C : RMat 2 k | ∀ l r, C l r ∈ Set.Icc (θ₀.2.2 l r - ρ) (θ₀.2.2 l r + ρ)}) := by
    ext ⟨A, B, C⟩; simp [genParamBox]
  rw [hprod, Measure.volume_eq_prod, Measure.prod_prod, Measure.volume_eq_prod, Measure.prod_prod]
  rw [gen_volume_matBox_shifted θ₀.1 hρ, gen_volume_matBox_shifted θ₀.2.1 hρ,
    gen_volume_matBox_shifted θ₀.2.2 hρ]
  rw [← pow_add, ← pow_add, show 2 * k + (2 * k + 2 * k) = 6 * k by ring,
    ← ENNReal.ofReal_pow (by positivity)]

private theorem gen_cubic_diff_bound {a b c a₀ b₀ c₀ M ρ : ℝ}
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
    calc |a - a₀| * |b| * |c| ≤ ρ * M * M :=
          mul_le_mul (mul_le_mul hda hb (abs_nonneg _) hρ) hc (abs_nonneg _) (by positivity)
      _ = M ^ 2 * ρ := by ring
  have h2 : |a₀ * (b - b₀) * c| ≤ M ^ 2 * ρ := by
    rw [abs_mul, abs_mul]
    calc |a₀| * |b - b₀| * |c| ≤ M * ρ * M :=
          mul_le_mul (mul_le_mul ha₀ hdb (abs_nonneg _) hM) hc (abs_nonneg _) (by positivity)
      _ = M ^ 2 * ρ := by ring
  have h3 : |a₀ * b₀ * (c - c₀)| ≤ M ^ 2 * ρ := by
    rw [abs_mul, abs_mul]
    calc |a₀| * |b₀| * |c - c₀| ≤ M * M * ρ :=
          mul_le_mul (mul_le_mul ha₀ hb₀ (abs_nonneg _) hM) hdc (abs_nonneg _) (by positivity)
      _ = M ^ 2 * ρ := by ring
  calc |(a - a₀) * b * c + a₀ * (b - b₀) * c + a₀ * b₀ * (c - c₀)|
    ≤ |(a - a₀) * b * c + a₀ * (b - b₀) * c| + |a₀ * b₀ * (c - c₀)| := abs_add_le _ _
    _ ≤ |(a - a₀) * b * c| + |a₀ * (b - b₀) * c| + |a₀ * b₀ * (c - c₀)| := by
      linarith [abs_add_le ((a - a₀) * b * c) (a₀ * (b - b₀) * c)]
    _ ≤ 3 * M ^ 2 * ρ := by linarith

/-- **General Stretched-Exponential Excess Free Energy — Upper Bound on `Z^{(2)}(β)` (`[LC]`):**
Under `GenericWallHyp Tstar Lwall`, for all `β ≥ 1`,
`Z_{T*}^{(2)}(β) ≤ (4π)⁶ · exp(-(β L_wall + a₁ β^{1/4}))`,
with explicit constant `a₁ = genericWDConst h · 4^{-3/4} > 0`.
In particular, `-log Z_{T*}^{(2)}(β) - β L_wall ≥ a₁ β^{1/4} - 6 log(4π)`. -/
theorem genericPartitionZ_two_le_stretchedExp
    {Tstar : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)} {Lwall : ℝ}
    (h : GenericWallHyp Tstar Lwall) {β : ℝ} (hβ : 1 ≤ β) :
    tensorPartitionZ Tstar 2 β ≤
      ENNReal.ofReal ((4 * Real.pi) ^ 6 *
        Real.exp (-(β * Lwall + (genericWDConst h * (4 : ℝ) ^ (-(3 / 4 : ℝ))) * β ^ (1 / 4 : ℝ)))) := by
  have hβ_pos : 0 < β := by linarith
  let γ : ℝ := 1 / (4 * β)
  have hγ0 : 0 < γ := by positivity
  have hγ1 : γ ≤ 1 := by
    simp only [γ]; rw [div_le_one (by positivity)]; linarith
  set a1 : ℝ := genericWDConst h * (4 : ℝ) ^ (-(3 / 4 : ℝ))
  have hpt : ∀ θ : BilinearParam 2 2 2 2,
      β * Lwall + a1 * β ^ (1 / 4 : ℝ) ≤
        β * bilinearLoss Tstar θ + (1 / 4 : ℝ) * paramNormSq θ := by
    intro θ
    have hwd := generic_weightDecay_lower h hγ0 hγ1 θ
    have hmul : β * (genericWDConst h * γ ^ (3 / 4 : ℝ)) ≤
        β * ((bilinearLoss Tstar θ - Lwall) + γ * paramNormSq θ) :=
      mul_le_mul_of_nonneg_left hwd hβ_pos.le
    have hrhs : β * ((bilinearLoss Tstar θ - Lwall) + γ * paramNormSq θ) =
        β * bilinearLoss Tstar θ - β * Lwall + (1 / 4 : ℝ) * paramNormSq θ := by
      simp only [γ]; field_simp
    have hγpow : γ ^ (3 / 4 : ℝ) = (4 : ℝ) ^ (-(3 / 4 : ℝ)) * β ^ (-(3 / 4 : ℝ)) := by
      simp only [γ]
      rw [one_div, Real.inv_rpow (by positivity), Real.mul_rpow (by norm_num) hβ_pos.le,
        mul_inv, ← Real.rpow_neg (by norm_num : (0 : ℝ) ≤ 4), ← Real.rpow_neg hβ_pos.le]
    have hβpow : β * β ^ (-(3 / 4 : ℝ)) = β ^ (1 / 4 : ℝ) := by
      calc β * β ^ (-(3 / 4 : ℝ))
        = β ^ (1 : ℝ) * β ^ (-(3 / 4 : ℝ)) := by rw [Real.rpow_one]
        _ = β ^ ((1 : ℝ) + -(3 / 4 : ℝ)) := (Real.rpow_add hβ_pos _ _).symm
        _ = β ^ (1 / 4 : ℝ) := by norm_num
    have hlhs : β * (genericWDConst h * γ ^ (3 / 4 : ℝ)) = a1 * β ^ (1 / 4 : ℝ) := by
      rw [hγpow]
      calc β * (genericWDConst h * ((4 : ℝ) ^ (-(3 / 4 : ℝ)) * β ^ (-(3 / 4 : ℝ))))
        = a1 * (β * β ^ (-(3 / 4 : ℝ))) := by simp only [a1]; ring
        _ = a1 * β ^ (1 / 4 : ℝ) := by rw [hβpow]
    linarith
  have hle_int : tensorPartitionZ Tstar 2 β ≤
      ∫⁻ θ : BilinearParam 2 2 2 2,
        ENNReal.ofReal (Real.exp (-(β * Lwall + a1 * β ^ (1 / 4 : ℝ)))) *
          ENNReal.ofReal (Real.exp (-((1 / 4 : ℝ) * paramNormSq θ))) := by
    apply MeasureTheory.lintegral_mono
    intro θ
    dsimp only
    rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add]
    apply ENNReal.ofReal_le_ofReal
    apply Real.exp_le_exp.mpr
    linarith [hpt θ]
  calc tensorPartitionZ Tstar 2 β
    ≤ ∫⁻ θ : BilinearParam 2 2 2 2,
        ENNReal.ofReal (Real.exp (-(β * Lwall + a1 * β ^ (1 / 4 : ℝ)))) *
          ENNReal.ofReal (Real.exp (-((1 / 4 : ℝ) * paramNormSq θ))) := hle_int
    _ = ENNReal.ofReal (Real.exp (-(β * Lwall + a1 * β ^ (1 / 4 : ℝ)))) *
          ∫⁻ θ : BilinearParam 2 2 2 2,
            ENNReal.ofReal (Real.exp (-((1 / 4 : ℝ) * paramNormSq θ))) :=
      MeasureTheory.lintegral_const_mul _ (by unfold paramNormSq; fun_prop)
    _ = ENNReal.ofReal (Real.exp (-(β * Lwall + a1 * β ^ (1 / 4 : ℝ)))) *
          ENNReal.ofReal ((4 * Real.pi) ^ 6) := by rw [gen_lintegral_gaussian_param2]
    _ = ENNReal.ofReal ((4 * Real.pi) ^ 6 * Real.exp (-(β * Lwall + a1 * β ^ (1 / 4 : ℝ)))) := by
      rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, mul_comm]

/-! #### General `GL(2)³`-transported centered escape curve and matching upper bounds -/

/-- Matrix multiplication of two `2 × 2` real matrices. -/
def matMul2 (G A : RMat 2 2) : RMat 2 2 :=
  fun i r => ∑ k : Fin 2, G i k * A k r

/-- Squared Frobenius norm of a `2 × 2` real matrix. -/
def matFrobSq2 (G : RMat 2 2) : ℝ :=
  ∑ i : Fin 2, ∑ j : Fin 2, (G i j) ^ 2

/-- Multilinear action of `(G₁, G₂, G₃) ∈ M₂(ℝ)³` on a `2 × 2 × 2` real tensor. -/
def applyGL3 (G1 G2 G3 : RMat 2 2) (T : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)) :
    Tensor ℝ (Fin 2) (Fin 2) (Fin 2) :=
  fun i j l => ∑ a : Fin 2, ∑ b : Fin 2, ∑ c : Fin 2, G1 i a * G2 j b * G3 l c * T a b c

/-- The centered border-rank curve transported by `(G₁, G₂, G₃)` onto the smooth wall point
`T_wall = (G₁, G₂, G₃) · W`. -/
noncomputable def gl3CenteredCurve (G1 G2 G3 : RMat 2 2) (ε : ℝ) : BilinearParam 2 2 2 2 :=
  (matMul2 G1 (centeredCurve ε).1,
   matMul2 G2 (centeredCurve ε).2.1,
   matMul2 G3 (centeredCurve ε).2.2)

/-- Exact tensor evaluation along the `GL(2)³`-transported centered curve:
`T(gl3CenteredCurve G₁ G₂ G₃ ε) = (G₁, G₂, G₃) · W + ε² (g₁,₁ ⊗ g₂,₁ ⊗ g₃,₁)`. -/
theorem gl3CenteredCurve_tensor (G1 G2 G3 : RMat 2 2) {ε : ℝ} (hε : 0 < ε) (i j l : Fin 2) :
    bilinearTensor (gl3CenteredCurve G1 G2 G3 ε) i j l =
      applyGL3 G1 G2 G3 (wTensor ℝ) i j l + ε ^ 2 * (G1 i 1 * G2 j 1 * G3 l 1) := by
  have hlin : bilinearTensor (gl3CenteredCurve G1 G2 G3 ε) i j l =
      applyGL3 G1 G2 G3 (bilinearTensor (centeredCurve ε)) i j l := by
    unfold bilinearTensor gl3CenteredCurve matMul2 applyGL3
    simp only [Fin.sum_univ_two]
    ring
  rw [hlin]
  unfold applyGL3
  simp only [Fin.sum_univ_two, centeredCurve_tensor hε]
  norm_num
  ring

private theorem matMul2_frobSq_le (G A : RMat 2 2) :
    (∑ i : Fin 2, ∑ r : Fin 2, (matMul2 G A i r) ^ 2) ≤
      matFrobSq2 G * (∑ i : Fin 2, ∑ r : Fin 2, (A i r) ^ 2) := by
  unfold matMul2 matFrobSq2
  simp only [Fin.sum_univ_two]
  nlinarith [
    sq_nonneg (G 0 0 * A 1 0 - G 0 1 * A 0 0),
    sq_nonneg (G 0 0 * A 1 1 - G 0 1 * A 0 1),
    sq_nonneg (G 1 0 * A 1 0 - G 1 1 * A 0 0),
    sq_nonneg (G 1 0 * A 1 1 - G 1 1 * A 0 1)]

/-- Parameter norm² bound for `gl3CenteredCurve G₁ G₂ G₃ ε`. -/
theorem gl3CenteredCurve_paramNormSq_le (G1 G2 G3 : RMat 2 2) (ε : ℝ) :
    paramNormSq (gl3CenteredCurve G1 G2 G3 ε) ≤
      (matFrobSq2 G1 + matFrobSq2 G2 + matFrobSq2 G3 + 1) * paramNormSq (centeredCurve ε) := by
  have h1 := matMul2_frobSq_le G1 (centeredCurve ε).1
  have h2 := matMul2_frobSq_le G2 (centeredCurve ε).2.1
  have h3 := matMul2_frobSq_le G3 (centeredCurve ε).2.2
  have hg1 : 0 ≤ matFrobSq2 G1 := by unfold matFrobSq2; positivity
  have hg2 : 0 ≤ matFrobSq2 G2 := by unfold matFrobSq2; positivity
  have hg3 : 0 ≤ matFrobSq2 G3 := by unfold matFrobSq2; positivity
  have ha : 0 ≤ ∑ i : Fin 2, ∑ r : Fin 2, ((centeredCurve ε).1 i r) ^ 2 := by positivity
  have hb : 0 ≤ ∑ j : Fin 2, ∑ r : Fin 2, ((centeredCurve ε).2.1 j r) ^ 2 := by positivity
  have hc : 0 ≤ ∑ l : Fin 2, ∑ r : Fin 2, ((centeredCurve ε).2.2 l r) ^ 2 := by positivity
  unfold paramNormSq gl3CenteredCurve
  dsimp only
  nlinarith

/-- Excess loss bound along `gl3CenteredCurve G₁ G₂ G₃ ε` for any target `Tstar` and `ε ∈ (0, 1]`. -/
theorem gl3CenteredCurve_excessLoss_le (Tstar : Tensor ℝ (Fin 2) (Fin 2) (Fin 2))
    (G1 G2 G3 : RMat 2 2) {ε : ℝ} (hε0 : 0 < ε) (hε1 : ε ≤ 1) :
    let Twall := applyGL3 G1 G2 G3 (wTensor ℝ)
    let Lwall := tensorFrobSq (fun i j l => Twall i j l - Tstar i j l)
    let vFrobSq := tensorFrobSq (fun i j l => G1 i 1 * G2 j 1 * G3 l 1)
    bilinearLoss Tstar (gl3CenteredCurve G1 G2 G3 ε) - Lwall ≤
      (Lwall + 2 * vFrobSq) * ε ^ 2 := by
  intro Twall Lwall vFrobSq
  have hε2_le1 : ε ^ 2 ≤ 1 := by nlinarith
  have hε4 : ε ^ 4 ≤ ε ^ 2 := by nlinarith [sq_nonneg ε]
  have hent : ∀ i j l : Fin 2,
      (bilinearTensor (gl3CenteredCurve G1 G2 G3 ε) i j l - Tstar i j l) ^ 2 -
        (Twall i j l - Tstar i j l) ^ 2 ≤
        ((Twall i j l - Tstar i j l) ^ 2 + 2 * (G1 i 1 * G2 j 1 * G3 l 1) ^ 2) * ε ^ 2 := by
    intro i j l
    rw [gl3CenteredCurve_tensor G1 G2 G3 hε0 i j l]
    set d := Twall i j l - Tstar i j l
    set v := G1 i 1 * G2 j 1 * G3 l 1
    have hid : (Twall i j l + ε ^ 2 * v - Tstar i j l) ^ 2 - d ^ 2 =
        2 * ε ^ 2 * (d * v) + ε ^ 4 * v ^ 2 := by dsimp only [d]; ring
    rw [hid]
    have hdv : 2 * (d * v) ≤ d ^ 2 + v ^ 2 := by nlinarith [sq_nonneg (d - v)]
    have h1 : 2 * ε ^ 2 * (d * v) ≤ ε ^ 2 * (d ^ 2 + v ^ 2) := by
      have := mul_le_mul_of_nonneg_left hdv (sq_nonneg ε); linarith
    have h2 : ε ^ 4 * v ^ 2 ≤ ε ^ 2 * v ^ 2 := mul_le_mul_of_nonneg_right hε4 (sq_nonneg v)
    linarith
  have hsum : bilinearLoss Tstar (gl3CenteredCurve G1 G2 G3 ε) - Lwall =
      ∑ i : Fin 2, ∑ j : Fin 2, ∑ l : Fin 2,
        ((bilinearTensor (gl3CenteredCurve G1 G2 G3 ε) i j l - Tstar i j l) ^ 2 -
          (Twall i j l - Tstar i j l) ^ 2) := by
    unfold bilinearLoss Lwall tensorFrobSq
    simp only [Fin.sum_univ_two]; ring
  rw [hsum]
  calc ∑ i : Fin 2, ∑ j : Fin 2, ∑ l : Fin 2,
        ((bilinearTensor (gl3CenteredCurve G1 G2 G3 ε) i j l - Tstar i j l) ^ 2 -
          (Twall i j l - Tstar i j l) ^ 2)
    ≤ ∑ i : Fin 2, ∑ j : Fin 2, ∑ l : Fin 2,
        (((Twall i j l - Tstar i j l) ^ 2 + 2 * (G1 i 1 * G2 j 1 * G3 l 1) ^ 2) * ε ^ 2) :=
      Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ =>
        Finset.sum_le_sum fun l _ => hent i j l
    _ = (Lwall + 2 * vFrobSq) * ε ^ 2 := by
      unfold Lwall vFrobSq tensorFrobSq
      simp only [Fin.sum_univ_two]; ring

/-- **General `GL(2)³` Escape Upper Bound (`R^{-6}`, `[LC]`):**
For *any* target `Tstar` and *any* wall point `T_wall = (G₁, G₂, G₃) · W` with
`L_wall = ‖T_wall - T*‖_F²`, there exist explicit constants `R₀ > 0`, `C > 0` such that
for all `R ≥ R₀`, there is a width-2 parameter `θ` with `paramNormSq θ ≤ R²` and
`bilinearLoss Tstar θ - L_wall ≤ C / R⁶`. -/
theorem gl3_escape_upper (Tstar : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)) (G1 G2 G3 : RMat 2 2) :
    let Twall := applyGL3 G1 G2 G3 (wTensor ℝ)
    let Lwall := tensorFrobSq (fun i j l => Twall i j l - Tstar i j l)
    let MG := matFrobSq2 G1 + matFrobSq2 G2 + matFrobSq2 G3 + 1
    let CL := Lwall + 2 * tensorFrobSq (fun i j l => G1 i 1 * G2 j 1 * G3 l 1)
    ∀ {R : ℝ}, Real.sqrt (12 * MG) ≤ R →
      ∃ θ : BilinearParam 2 2 2 2,
        paramNormSq θ ≤ R ^ 2 ∧
        bilinearLoss Tstar θ - Lwall ≤ (432 * CL * MG ^ 3) / R ^ 6 := by
  intro Twall Lwall MG CL R hR
  have hMG_ge1 : 1 ≤ MG := by
    have : 0 ≤ matFrobSq2 G1 := by unfold matFrobSq2; positivity
    have : 0 ≤ matFrobSq2 G2 := by unfold matFrobSq2; positivity
    have : 0 ≤ matFrobSq2 G3 := by unfold matFrobSq2; positivity
    dsimp only [MG]; linarith
  have hMG_pos : 0 < MG := by linarith
  have hR2_ge : 12 * MG ≤ R ^ 2 := by
    have hsq : Real.sqrt (12 * MG) ^ 2 ≤ R ^ 2 := by
      rw [sq, sq]; exact mul_le_mul hR hR (Real.sqrt_nonneg _) (le_trans (Real.sqrt_nonneg _) hR)
    rwa [Real.sq_sqrt (by positivity)] at hsq
  have hR_pos : 0 < R := lt_of_lt_of_le (Real.sqrt_pos.mpr (by positivity)) hR
  let ε : ℝ := (1 / 2 : ℝ) * (12 * MG / R ^ 2) ^ (3 / 2 : ℝ)
  have hfrac_le : 12 * MG / R ^ 2 ≤ 1 := by
    rw [div_le_one (by positivity)]; exact hR2_ge
  have hε_pos : 0 < ε := by positivity
  have hε_le : ε ≤ 1 := by
    have hpow : (12 * MG / R ^ 2) ^ (3 / 2 : ℝ) ≤ 1 :=
      Real.rpow_le_one (by positivity) hfrac_le (by positivity)
    simp only [ε]; linarith
  refine ⟨gl3CenteredCurve G1 G2 G3 ε, ?_, ?_⟩
  · have hnorm := gl3CenteredCurve_paramNormSq_le G1 G2 G3 ε
    rw [centeredCurve_paramNormSq hε_pos] at hnorm
    have h2ε : 2 * ε = (12 * MG / R ^ 2) ^ (3 / 2 : ℝ) := by simp [ε]
    have hpow : (2 * ε) ^ (-(2 / 3 : ℝ)) = R ^ 2 / (12 * MG) := by
      rw [h2ε, ← Real.rpow_mul (by positivity : (0 : ℝ) ≤ 12 * MG / R ^ 2)]
      norm_num
      rw [Real.rpow_neg_one, inv_div]
    rw [hpow] at hnorm
    have hε2 : ε ^ 2 ≤ 1 := by nlinarith
    have hcalc : MG * (6 * (R ^ 2 / (12 * MG)) * (1 + ε ^ 2)) = (1 / 2 : ℝ) * R ^ 2 * (1 + ε ^ 2) := by
      field_simp; ring
    have hstep : (1 / 2 : ℝ) * R ^ 2 * (1 + ε ^ 2) ≤ R ^ 2 := by nlinarith
    exact le_trans hnorm (by linarith)
  · have hloss := gl3CenteredCurve_excessLoss_le Tstar G1 G2 G3 hε_pos hε_le
    have hε2 : ε ^ 2 = (432 * MG ^ 3) / R ^ 6 := by
      have h1 : ε ^ 2 = (1 / 4 : ℝ) * ((12 * MG / R ^ 2) ^ (3 / 2 : ℝ)) ^ 2 := by
        simp only [ε]; ring
      rw [h1, ← Real.rpow_natCast, ← Real.rpow_mul (by positivity : (0 : ℝ) ≤ 12 * MG / R ^ 2)]
      norm_num
      rw [div_pow, mul_pow, ← pow_mul]
      ring
    calc bilinearLoss Tstar (gl3CenteredCurve G1 G2 G3 ε) - Lwall
      ≤ CL * ε ^ 2 := hloss
      _ = CL * ((432 * MG ^ 3) / R ^ 6) := by rw [hε2]
      _ = (432 * CL * MG ^ 3) / R ^ 6 := by ring

/-- **General `GL(2)³` Weight-Decay Upper Bound (`γ^{3/4}`, `[LC]`):**
For *any* target `Tstar`, *any* wall point `T_wall = (G₁, G₂, G₃) · W`, and all `γ ∈ (0, 1]`,
there exists a width-2 parameter `θ` achieving
`(bilinearLoss Tstar θ - L_wall) + γ * paramNormSq θ ≤ (CL + 12 * MG) * γ^{3/4}`. -/
theorem gl3_weightDecay_upper (Tstar : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)) (G1 G2 G3 : RMat 2 2)
    {γ : ℝ} (hγ0 : 0 < γ) (hγ1 : γ ≤ 1) :
    let Twall := applyGL3 G1 G2 G3 (wTensor ℝ)
    let Lwall := tensorFrobSq (fun i j l => Twall i j l - Tstar i j l)
    let MG := matFrobSq2 G1 + matFrobSq2 G2 + matFrobSq2 G3 + 1
    let CL := Lwall + 2 * tensorFrobSq (fun i j l => G1 i 1 * G2 j 1 * G3 l 1)
    ∃ θ : BilinearParam 2 2 2 2,
      (bilinearLoss Tstar θ - Lwall) + γ * paramNormSq θ ≤
        (CL + 12 * MG) * γ ^ (3 / 4 : ℝ) := by
  intro Twall Lwall MG CL
  have hMG_pos : 0 < MG := by
    have : 0 ≤ matFrobSq2 G1 := by unfold matFrobSq2; positivity
    have : 0 ≤ matFrobSq2 G2 := by unfold matFrobSq2; positivity
    have : 0 ≤ matFrobSq2 G3 := by unfold matFrobSq2; positivity
    dsimp only [MG]; linarith
  let ε : ℝ := γ ^ (3 / 8 : ℝ)
  have hε_pos : 0 < ε := by positivity
  have hε_le : ε ≤ 1 := Real.rpow_le_one hγ0.le hγ1 (by positivity)
  refine ⟨gl3CenteredCurve G1 G2 G3 ε, ?_⟩
  have hL : bilinearLoss Tstar (gl3CenteredCurve G1 G2 G3 ε) - Lwall ≤ CL * γ ^ (3 / 4 : ℝ) := by
    have h1 := gl3CenteredCurve_excessLoss_le Tstar G1 G2 G3 hε_pos hε_le
    have hε2 : ε ^ 2 = γ ^ (3 / 4 : ℝ) := by
      simp only [ε]; rw [← Real.rpow_natCast, ← Real.rpow_mul hγ0.le]; norm_num
    rwa [hε2] at h1
  have hR2 : γ * paramNormSq (gl3CenteredCurve G1 G2 G3 ε) ≤ 12 * MG * γ ^ (3 / 4 : ℝ) := by
    have hnorm := gl3CenteredCurve_paramNormSq_le G1 G2 G3 ε
    rw [centeredCurve_paramNormSq hε_pos] at hnorm
    have h2pow : (2 : ℝ) ^ (-(2 / 3 : ℝ)) ≤ 1 := by
      rw [Real.rpow_neg (by norm_num : (0 : ℝ) ≤ 2)]
      exact inv_le_one_of_one_le₀ (Real.one_le_rpow (by norm_num) (by positivity))
    have hεpow : ε ^ (-(2 / 3 : ℝ)) = γ ^ (-(1 / 4 : ℝ)) := by
      simp only [ε]; rw [← Real.rpow_mul hγ0.le]; norm_num
    have hγmul : γ * γ ^ (-(1 / 4 : ℝ)) = γ ^ (3 / 4 : ℝ) := by
      calc γ * γ ^ (-(1 / 4 : ℝ))
        = γ ^ (1 : ℝ) * γ ^ (-(1 / 4 : ℝ)) := by rw [Real.rpow_one]
        _ = γ ^ ((1 : ℝ) + -(1 / 4 : ℝ)) := (Real.rpow_add hγ0 _ _).symm
        _ = γ ^ (3 / 4 : ℝ) := by norm_num
    have h1ε2 : 1 + ε ^ 2 ≤ 2 := by nlinarith
    have hcc : γ * (6 * (2 * ε) ^ (-(2 / 3 : ℝ)) * (1 + ε ^ 2)) ≤ 12 * γ ^ (3 / 4 : ℝ) := by
      rw [Real.mul_rpow (by norm_num : (0 : ℝ) ≤ 2) hε_pos.le, hεpow]
      calc γ * (6 * ((2 : ℝ) ^ (-(2 / 3 : ℝ)) * γ ^ (-(1 / 4 : ℝ))) * (1 + ε ^ 2))
        = 6 * (2 : ℝ) ^ (-(2 / 3 : ℝ)) * (1 + ε ^ 2) * (γ * γ ^ (-(1 / 4 : ℝ))) := by ring
        _ = 6 * (2 : ℝ) ^ (-(2 / 3 : ℝ)) * (1 + ε ^ 2) * γ ^ (3 / 4 : ℝ) := by rw [hγmul]
        _ ≤ 6 * 1 * 2 * γ ^ (3 / 4 : ℝ) := by
          apply mul_le_mul_of_nonneg_right _ (by positivity)
          nlinarith [show (0 : ℝ) ≤ (2 : ℝ) ^ (-(2 / 3 : ℝ)) by positivity]
        _ = 12 * γ ^ (3 / 4 : ℝ) := by ring
    calc γ * paramNormSq (gl3CenteredCurve G1 G2 G3 ε)
      ≤ γ * (MG * (6 * (2 * ε) ^ (-(2 / 3 : ℝ)) * (1 + ε ^ 2))) :=
        mul_le_mul_of_nonneg_left hnorm hγ0.le
      _ = MG * (γ * (6 * (2 * ε) ^ (-(2 / 3 : ℝ)) * (1 + ε ^ 2))) := by ring
      _ ≤ MG * (12 * γ ^ (3 / 4 : ℝ)) := mul_le_mul_of_nonneg_left hcc hMG_pos.le
      _ = 12 * MG * γ ^ (3 / 4 : ℝ) := by ring
  linarith

/-- **General `GL(2)³` Stretched-Exponential Excess Free Energy — Lower Bound on `Z^{(2)}(β)` (`[LC]`):**
For *any* target `Tstar` and *any* wall point `T_wall = (G₁, G₂, G₃) · W` with
`L_wall = ‖T_wall - T*‖_F²`, there exist explicit positive constants `c_vol > 0` and `a₂ > 0`
such that for all `β ≥ 1`,
`c_vol · exp(-(β L_wall + a₂ β^{1/4})) ≤ tensorPartitionZ Tstar 2 β`.
Combined with `genericPartitionZ_two_le_stretchedExp`, this certifies
`-log Z_{T*}^{(2)}(β) - β L_wall ≍ β^{1/4}` for all `β ≥ 1`. -/
theorem gl3PartitionZ_two_ge_stretchedExp (Tstar : Tensor ℝ (Fin 2) (Fin 2) (Fin 2))
    (G1 G2 G3 : RMat 2 2) :
    let Twall := applyGL3 G1 G2 G3 (wTensor ℝ)
    let Lwall := tensorFrobSq (fun i j l => Twall i j l - Tstar i j l)
    ∃ (c_vol a₂ : ℝ), 0 < c_vol ∧ 0 < a₂ ∧
      ∀ {β : ℝ}, 1 ≤ β →
        ENNReal.ofReal (c_vol * Real.exp (-(β * Lwall + a₂ * β ^ (1 / 4 : ℝ)))) ≤
          tensorPartitionZ Tstar 2 β := by
  intro Twall Lwall
  -- Entrywise bound `B_G ≥ 1` on `G1, G2, G3` so `|matMul2 G A i r| ≤ 2 * B_G * max |A|`.
  let BG : ℝ := 1 + matFrobSq2 G1 + matFrobSq2 G2 + matFrobSq2 G3
  have hBG_ge1 : 1 ≤ BG := by
    have : 0 ≤ matFrobSq2 G1 := by unfold matFrobSq2; positivity
    have : 0 ≤ matFrobSq2 G2 := by unfold matFrobSq2; positivity
    have : 0 ≤ matFrobSq2 G3 := by unfold matFrobSq2; positivity
    dsimp only [BG]; linarith
  have hBG_pos : 0 < BG := by linarith
  have hG1_abs : ∀ i k : Fin 2, |G1 i k| ≤ BG := fun i k => by
    have hsq : (G1 i k) ^ 2 ≤ matFrobSq2 G1 := by
      unfold matFrobSq2
      have h1 : (G1 i k) ^ 2 ≤ ∑ k' : Fin 2, (G1 i k') ^ 2 :=
        Finset.single_le_sum (fun _ _ => sq_nonneg _) (Finset.mem_univ k)
      have h2 : ∑ k' : Fin 2, (G1 i k') ^ 2 ≤ ∑ i' : Fin 2, ∑ k' : Fin 2, (G1 i' k') ^ 2 :=
        Finset.single_le_sum (fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _) (Finset.mem_univ i)
      exact h1.trans h2
    have h2 : 0 ≤ matFrobSq2 G2 := by unfold matFrobSq2; positivity
    have h3 : 0 ≤ matFrobSq2 G3 := by unfold matFrobSq2; positivity
    nlinarith [sq_abs (G1 i k), abs_nonneg (G1 i k), sq_nonneg (|G1 i k| - 1)]
  have hG2_abs : ∀ j k : Fin 2, |G2 j k| ≤ BG := fun j k => by
    have hsq : (G2 j k) ^ 2 ≤ matFrobSq2 G2 := by
      unfold matFrobSq2
      have h1 : (G2 j k) ^ 2 ≤ ∑ k' : Fin 2, (G2 j k') ^ 2 :=
        Finset.single_le_sum (fun _ _ => sq_nonneg _) (Finset.mem_univ k)
      have h2 : ∑ k' : Fin 2, (G2 j k') ^ 2 ≤ ∑ j' : Fin 2, ∑ k' : Fin 2, (G2 j' k') ^ 2 :=
        Finset.single_le_sum (fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _) (Finset.mem_univ j)
      exact h1.trans h2
    have h1 : 0 ≤ matFrobSq2 G1 := by unfold matFrobSq2; positivity
    have h3 : 0 ≤ matFrobSq2 G3 := by unfold matFrobSq2; positivity
    nlinarith [sq_abs (G2 j k), abs_nonneg (G2 j k), sq_nonneg (|G2 j k| - 1)]
  have hG3_abs : ∀ l k : Fin 2, |G3 l k| ≤ BG := fun l k => by
    have hsq : (G3 l k) ^ 2 ≤ matFrobSq2 G3 := by
      unfold matFrobSq2
      have h1 : (G3 l k) ^ 2 ≤ ∑ k' : Fin 2, (G3 l k') ^ 2 :=
        Finset.single_le_sum (fun _ _ => sq_nonneg _) (Finset.mem_univ k)
      have h2 : ∑ k' : Fin 2, (G3 l k') ^ 2 ≤ ∑ l' : Fin 2, ∑ k' : Fin 2, (G3 l' k') ^ 2 :=
        Finset.single_le_sum (fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _) (Finset.mem_univ l)
      exact h1.trans h2
    have h1 : 0 ≤ matFrobSq2 G1 := by unfold matFrobSq2; positivity
    have h2 : 0 ≤ matFrobSq2 G2 := by unfold matFrobSq2; positivity
    nlinarith [sq_abs (G3 l k), abs_nonneg (G3 l k), sq_nonneg (|G3 l k| - 1)]
  let CL : ℝ := Lwall + 2 * tensorFrobSq (fun i j l => G1 i 1 * G2 j 1 * G3 l 1)
  have hCL_nn : 0 ≤ CL := by
    have : 0 ≤ Lwall := tensorFrobSq_nonneg _
    have : 0 ≤ tensorFrobSq (fun i j l => G1 i 1 * G2 j 1 * G3 l 1) := tensorFrobSq_nonneg _
    dsimp only [CL]; linarith
  let D0 : ℝ := 1 + Lwall + CL
  have hD0_ge1 : 1 ≤ D0 := by
    have : 0 ≤ Lwall := tensorFrobSq_nonneg _
    dsimp only [D0]; linarith
  -- Choose box half-width `ρ = (1 / (6 * (3 * BG)^2)) * β^{-1}` and explicit `c_vol, a₂`.
  let ρ0 : ℝ := 1 / (54 * BG ^ 2)
  have hρ0_pos : 0 < ρ0 := by positivity
  have hρ0_le1 : ρ0 ≤ 1 := by
    simp only [ρ0]; rw [div_le_one (by positivity)]; nlinarith
  let c_vol : ℝ := (2 * ρ0) ^ 12
  let a₂ : ℝ := CL + 8 + 16 * D0 + 54 * BG ^ 2 + 48
  refine ⟨c_vol, a₂, by positivity, by positivity, ?_⟩
  intro β hβ
  have hβ_pos : 0 < β := by linarith
  let ε : ℝ := β ^ (-(3 / 8 : ℝ))
  have hε_pos : 0 < ε := by positivity
  have hε_le : ε ≤ 1 := by
    simp only [ε]; rw [Real.rpow_neg hβ_pos.le]
    exact inv_le_one_of_one_le₀ (Real.one_le_rpow hβ (by positivity))
  let θ₀ : BilinearParam 2 2 2 2 := gl3CenteredCurve G1 G2 G3 ε
  let ρ : ℝ := ρ0 * β ^ (-1 : ℝ)
  have hρ_pos : 0 < ρ := by positivity
  have hβ_inv_le1 : β ^ (-1 : ℝ) ≤ 1 := by
    rw [Real.rpow_neg_one]; exact inv_le_one_of_one_le₀ hβ
  have hρ_le1 : ρ ≤ 1 := by
    simp only [ρ]; nlinarith
  have hβ18_ge : 1 ≤ β ^ (1 / 8 : ℝ) := Real.one_le_rpow hβ (by positivity)
  have hβ14_ge : 1 ≤ β ^ (1 / 4 : ℝ) := Real.one_le_rpow hβ (by positivity)
  have hβ18_sq : (β ^ (1 / 8 : ℝ)) ^ 2 = β ^ (1 / 4 : ℝ) := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hβ_pos.le]; norm_num
  have hβm34_le1 : β ^ (-(3 / 4 : ℝ)) ≤ 1 := by
    rw [Real.rpow_neg hβ_pos.le]; exact inv_le_one_of_one_le₀ (Real.one_le_rpow hβ (by positivity))
  -- Coordinate bounds on `centeredCurve ε` and hence on `θ₀`.
  have hs_le : (2 * ε) ^ (-(1 / 3 : ℝ)) ≤ β ^ (1 / 8 : ℝ) := by
    rw [Real.mul_rpow (by norm_num : (0 : ℝ) ≤ 2) hε_pos.le]
    have h2 : (2 : ℝ) ^ (-(1 / 3 : ℝ)) ≤ 1 := by
      rw [Real.rpow_neg (by norm_num : (0 : ℝ) ≤ 2)]
      exact inv_le_one_of_one_le₀ (Real.one_le_rpow (by norm_num) (by positivity))
    have hεp : ε ^ (-(1 / 3 : ℝ)) = β ^ (1 / 8 : ℝ) := by
      simp only [ε]; rw [← Real.rpow_mul hβ_pos.le]; norm_num
    rw [hεp]; nlinarith [show (0 : ℝ) ≤ β ^ (1 / 8 : ℝ) by positivity]
  have hs_pos : 0 < (2 * ε) ^ (-(1 / 3 : ℝ)) := by positivity
  have hs_abs : |(2 * ε) ^ (-(1 / 3 : ℝ))| ≤ β ^ (1 / 8 : ℝ) := by rwa [abs_of_pos hs_pos]
  have hsε_abs : |(2 * ε) ^ (-(1 / 3 : ℝ)) * ε| ≤ β ^ (1 / 8 : ℝ) := by
    rw [abs_of_pos (mul_pos hs_pos hε_pos)]; nlinarith
  have hcc1 : ∀ i r : Fin 2, |(centeredCurve ε).1 i r| ≤ β ^ (1 / 8 : ℝ) := fun i r => by
    dsimp only [centeredCurve]; split_ifs <;> (try simp only [abs_neg]) <;>
      first | exact hs_abs | exact hsε_abs
  have hcc2 : ∀ j r : Fin 2, |(centeredCurve ε).2.1 j r| ≤ β ^ (1 / 8 : ℝ) := fun j r => by
    dsimp only [centeredCurve]; split_ifs <;> (try simp only [abs_neg]) <;>
      first | exact hs_abs | exact hsε_abs
  have hcc3 : ∀ l r : Fin 2, |(centeredCurve ε).2.2 l r| ≤ β ^ (1 / 8 : ℝ) := fun l r => by
    dsimp only [centeredCurve]; split_ifs <;> (try simp only [abs_neg]) <;>
      first | exact hs_abs | exact hsε_abs
  have hθ₀_1 : ∀ i r : Fin 2, |θ₀.1 i r| ≤ 2 * BG * β ^ (1 / 8 : ℝ) := fun i r => by
    change |∑ k : Fin 2, G1 i k * (centeredCurve ε).1 k r| ≤ _
    rw [Fin.sum_univ_two]
    have p0 : |G1 i 0 * (centeredCurve ε).1 0 r| ≤ BG * β ^ (1 / 8 : ℝ) := by
      rw [abs_mul]; exact mul_le_mul (hG1_abs i 0) (hcc1 0 r) (abs_nonneg _) hBG_pos.le
    have p1 : |G1 i 1 * (centeredCurve ε).1 1 r| ≤ BG * β ^ (1 / 8 : ℝ) := by
      rw [abs_mul]; exact mul_le_mul (hG1_abs i 1) (hcc1 1 r) (abs_nonneg _) hBG_pos.le
    exact (abs_add_le _ _).trans (by linarith)
  have hθ₀_2 : ∀ j r : Fin 2, |θ₀.2.1 j r| ≤ 2 * BG * β ^ (1 / 8 : ℝ) := fun j r => by
    change |∑ k : Fin 2, G2 j k * (centeredCurve ε).2.1 k r| ≤ _
    rw [Fin.sum_univ_two]
    have p0 : |G2 j 0 * (centeredCurve ε).2.1 0 r| ≤ BG * β ^ (1 / 8 : ℝ) := by
      rw [abs_mul]; exact mul_le_mul (hG2_abs j 0) (hcc2 0 r) (abs_nonneg _) hBG_pos.le
    have p1 : |G2 j 1 * (centeredCurve ε).2.1 1 r| ≤ BG * β ^ (1 / 8 : ℝ) := by
      rw [abs_mul]; exact mul_le_mul (hG2_abs j 1) (hcc2 1 r) (abs_nonneg _) hBG_pos.le
    exact (abs_add_le _ _).trans (by linarith)
  have hθ₀_3 : ∀ l r : Fin 2, |θ₀.2.2 l r| ≤ 2 * BG * β ^ (1 / 8 : ℝ) := fun l r => by
    change |∑ k : Fin 2, G3 l k * (centeredCurve ε).2.2 k r| ≤ _
    rw [Fin.sum_univ_two]
    have p0 : |G3 l 0 * (centeredCurve ε).2.2 0 r| ≤ BG * β ^ (1 / 8 : ℝ) := by
      rw [abs_mul]; exact mul_le_mul (hG3_abs l 0) (hcc3 0 r) (abs_nonneg _) hBG_pos.le
    have p1 : |G3 l 1 * (centeredCurve ε).2.2 1 r| ≤ BG * β ^ (1 / 8 : ℝ) := by
      rw [abs_mul]; exact mul_le_mul (hG3_abs l 1) (hcc3 1 r) (abs_nonneg _) hBG_pos.le
    exact (abs_add_le _ _).trans (by linarith)
  -- Pointwise bound on `T(θ₀) - Tstar` entries: `|T(θ₀) i j l - Tstar i j l| ≤ D0`.
  have hL0_le : bilinearLoss Tstar θ₀ ≤ Lwall + CL * β ^ (-(3 / 4 : ℝ)) := by
    have h1 := gl3CenteredCurve_excessLoss_le Tstar G1 G2 G3 hε_pos hε_le
    have hε2 : ε ^ 2 = β ^ (-(3 / 4 : ℝ)) := by
      simp only [ε]; rw [← Real.rpow_natCast, ← Real.rpow_mul hβ_pos.le]; norm_num
    rw [hε2] at h1
    linarith
  have hres0_abs : ∀ i j l : Fin 2, |bilinearTensor θ₀ i j l - Tstar i j l| ≤ D0 := fun i j l => by
    have hsq : (bilinearTensor θ₀ i j l - Tstar i j l) ^ 2 ≤ bilinearLoss Tstar θ₀ :=
      sq_entry_le_tensorFrobSq (fun i' j' l' => bilinearTensor θ₀ i' j' l' - Tstar i' j' l') i j l
    have hle : bilinearLoss Tstar θ₀ ≤ Lwall + CL := by
      have : CL * β ^ (-(3 / 4 : ℝ)) ≤ CL := mul_le_of_le_one_right hCL_nn hβm34_le1
      linarith
    have hsq2 : |bilinearTensor θ₀ i j l - Tstar i j l| ^ 2 ≤ D0 := by
      rw [sq_abs]; dsimp only [D0]; linarith
    nlinarith [abs_nonneg (bilinearTensor θ₀ i j l - Tstar i j l), hD0_ge1]
  -- Pointwise exponent upper bound on `genParamBox θ₀ ρ`.
  have hexp_le : ∀ θ ∈ genParamBox θ₀ ρ,
      β * bilinearLoss Tstar θ + (1 / 2 : ℝ) * paramNormSq θ ≤
        β * Lwall + (CL + 8 + 16 * D0 + 54 * BG ^ 2) * β ^ (1 / 4 : ℝ) := by
    intro θ hθ
    rcases hθ with ⟨h1, h2, h3⟩
    have hda : ∀ i r, |θ.1 i r - θ₀.1 i r| ≤ ρ := fun i r => by
      have := Set.mem_Icc.mp (h1 i r); rw [abs_le]; constructor <;> linarith
    have hdb : ∀ j r, |θ.2.1 j r - θ₀.2.1 j r| ≤ ρ := fun j r => by
      have := Set.mem_Icc.mp (h2 j r); rw [abs_le]; constructor <;> linarith
    have hdc : ∀ l r, |θ.2.2 l r - θ₀.2.2 l r| ≤ ρ := fun l r => by
      have := Set.mem_Icc.mp (h3 l r); rw [abs_le]; constructor <;> linarith
    set M := 3 * BG * β ^ (1 / 8 : ℝ)
    have hM_nonneg : 0 ≤ M := by positivity
    have hM2 : M ^ 2 = 9 * BG ^ 2 * β ^ (1 / 4 : ℝ) := by
      simp only [M]; rw [mul_pow, mul_pow, hβ18_sq]; ring
    have hθ_1 : ∀ i r, |θ.1 i r| ≤ M := fun i r => by
      calc |θ.1 i r| = |(θ.1 i r - θ₀.1 i r) + θ₀.1 i r| := by ring_nf
        _ ≤ |θ.1 i r - θ₀.1 i r| + |θ₀.1 i r| := abs_add_le _ _
        _ ≤ ρ + 2 * BG * β ^ (1 / 8 : ℝ) := add_le_add (hda i r) (hθ₀_1 i r)
        _ ≤ M := by simp only [M]; nlinarith only [hρ_le1, hBG_ge1, hβ18_ge]
    have hθ_2 : ∀ j r, |θ.2.1 j r| ≤ M := fun j r => by
      calc |θ.2.1 j r| = |(θ.2.1 j r - θ₀.2.1 j r) + θ₀.2.1 j r| := by ring_nf
        _ ≤ |θ.2.1 j r - θ₀.2.1 j r| + |θ₀.2.1 j r| := abs_add_le _ _
        _ ≤ ρ + 2 * BG * β ^ (1 / 8 : ℝ) := add_le_add (hdb j r) (hθ₀_2 j r)
        _ ≤ M := by simp only [M]; nlinarith only [hρ_le1, hBG_ge1, hβ18_ge]
    have hθ_3 : ∀ l r, |θ.2.2 l r| ≤ M := fun l r => by
      calc |θ.2.2 l r| = |(θ.2.2 l r - θ₀.2.2 l r) + θ₀.2.2 l r| := by ring_nf
        _ ≤ |θ.2.2 l r - θ₀.2.2 l r| + |θ₀.2.2 l r| := abs_add_le _ _
        _ ≤ ρ + 2 * BG * β ^ (1 / 8 : ℝ) := add_le_add (hdc l r) (hθ₀_3 l r)
        _ ≤ M := by simp only [M]; nlinarith only [hρ_le1, hBG_ge1, hβ18_ge]
    have hM2ρ : 6 * M ^ 2 * ρ = β ^ (-(3 / 4 : ℝ)) := by
      rw [hM2]
      have hmul : β ^ (1 / 4 : ℝ) * β ^ (-1 : ℝ) = β ^ (-(3 / 4 : ℝ)) := by
        rw [← Real.rpow_add hβ_pos]; norm_num
      simp only [ρ, ρ0]
      have hcancel : 6 * (9 * BG ^ 2 * β ^ (1 / 4 : ℝ)) * (1 / (54 * BG ^ 2) * β ^ (-1 : ℝ)) =
          β ^ (1 / 4 : ℝ) * β ^ (-1 : ℝ) := by field_simp; ring
      rw [hcancel, hmul]
    have h2BG_le_M : 2 * BG * β ^ (1 / 8 : ℝ) ≤ M := by
      simp only [M]; nlinarith only [hBG_ge1, hβ18_ge]
    have hTdiff : ∀ i j l : Fin 2,
        |bilinearTensor θ i j l - bilinearTensor θ₀ i j l| ≤ β ^ (-(3 / 4 : ℝ)) := by
      intro i j l
      simp only [bilinearTensor, Fin.sum_univ_two]
      have hc0 := gen_cubic_diff_bound (hθ_1 i 0) (hθ_2 j 0) (hθ_3 l 0)
        ((hθ₀_1 i 0).trans h2BG_le_M) ((hθ₀_2 j 0).trans h2BG_le_M) ((hθ₀_3 l 0).trans h2BG_le_M)
        (hda i 0) (hdb j 0) (hdc l 0) hM_nonneg hρ_pos.le
      have hc1 := gen_cubic_diff_bound (hθ_1 i 1) (hθ_2 j 1) (hθ_3 l 1)
        ((hθ₀_1 i 1).trans h2BG_le_M) ((hθ₀_2 j 1).trans h2BG_le_M) ((hθ₀_3 l 1).trans h2BG_le_M)
        (hda i 1) (hdb j 1) (hdc l 1) hM_nonneg hρ_pos.le
      have hsplit : (θ.1 i 0 * θ.2.1 j 0 * θ.2.2 l 0 + θ.1 i 1 * θ.2.1 j 1 * θ.2.2 l 1) -
          (θ₀.1 i 0 * θ₀.2.1 j 0 * θ₀.2.2 l 0 + θ₀.1 i 1 * θ₀.2.1 j 1 * θ₀.2.2 l 1) =
          (θ.1 i 0 * θ.2.1 j 0 * θ.2.2 l 0 - θ₀.1 i 0 * θ₀.2.1 j 0 * θ₀.2.2 l 0) +
          (θ.1 i 1 * θ.2.1 j 1 * θ.2.2 l 1 - θ₀.1 i 1 * θ₀.2.1 j 1 * θ₀.2.2 l 1) := by ring
      rw [hsplit]
      exact (abs_add_le _ _).trans (by linarith only [hc0, hc1, hM2ρ])
    -- Entrywise loss increment: `(T - T*)² - (T₀ - T*)² ≤ (1 + 2 D0) * β^{-3/4}`.
    have hL_inc : bilinearLoss Tstar θ - bilinearLoss Tstar θ₀ ≤
        (8 + 16 * D0) * β ^ (-(3 / 4 : ℝ)) := by
      have hent : ∀ i j l : Fin 2,
          (bilinearTensor θ i j l - Tstar i j l) ^ 2 -
            (bilinearTensor θ₀ i j l - Tstar i j l) ^ 2 ≤
            (1 + 2 * D0) * β ^ (-(3 / 4 : ℝ)) := by
        intro i j l
        set δ := bilinearTensor θ i j l - bilinearTensor θ₀ i j l
        set r := bilinearTensor θ₀ i j l - Tstar i j l
        have hid : (bilinearTensor θ i j l - Tstar i j l) ^ 2 - r ^ 2 = δ ^ 2 + 2 * r * δ := by
          dsimp only [δ, r]; ring
        rw [hid]
        have hδ_le : |δ| ≤ β ^ (-(3 / 4 : ℝ)) := hTdiff i j l
        have hr_le : |r| ≤ D0 := hres0_abs i j l
        have hβm34_nn : 0 ≤ β ^ (-(3 / 4 : ℝ)) := by positivity
        have hδ2 : δ ^ 2 ≤ β ^ (-(3 / 4 : ℝ)) := by
          have h1' : |δ| ^ 2 ≤ (β ^ (-(3 / 4 : ℝ))) ^ 2 := by
            nlinarith only [hδ_le, abs_nonneg δ, hβm34_nn]
          rw [sq_abs] at h1'
          have h2' : (β ^ (-(3 / 4 : ℝ))) ^ 2 ≤ β ^ (-(3 / 4 : ℝ)) := by
            nlinarith only [hβm34_nn, hβm34_le1]
          linarith only [h1', h2']
        have hcross : r * δ ≤ D0 * β ^ (-(3 / 4 : ℝ)) := by
          calc r * δ ≤ |r * δ| := le_abs_self _
            _ = |r| * |δ| := abs_mul r δ
            _ ≤ D0 * β ^ (-(3 / 4 : ℝ)) :=
              mul_le_mul hr_le hδ_le (abs_nonneg δ) (by linarith only [hD0_ge1])
        linarith only [hδ2, hcross]
      have hsum : bilinearLoss Tstar θ - bilinearLoss Tstar θ₀ =
          ∑ i : Fin 2, ∑ j : Fin 2, ∑ l : Fin 2,
            ((bilinearTensor θ i j l - Tstar i j l) ^ 2 -
              (bilinearTensor θ₀ i j l - Tstar i j l) ^ 2) := by
        unfold bilinearLoss tensorFrobSq; simp only [Fin.sum_univ_two]; ring
      rw [hsum]
      calc ∑ i : Fin 2, ∑ j : Fin 2, ∑ l : Fin 2,
            ((bilinearTensor θ i j l - Tstar i j l) ^ 2 -
              (bilinearTensor θ₀ i j l - Tstar i j l) ^ 2)
        ≤ ∑ _i : Fin 2, ∑ _j : Fin 2, ∑ _l : Fin 2, ((1 + 2 * D0) * β ^ (-(3 / 4 : ℝ))) :=
          Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ =>
            Finset.sum_le_sum fun l _ => hent i j l
        _ = (8 + 16 * D0) * β ^ (-(3 / 4 : ℝ)) := by simp; ring
    have hL_le : bilinearLoss Tstar θ ≤ Lwall + (CL + 8 + 16 * D0) * β ^ (-(3 / 4 : ℝ)) := by
      linarith only [hL0_le, hL_inc]
    have hβmul : β * β ^ (-(3 / 4 : ℝ)) = β ^ (1 / 4 : ℝ) := by
      calc β * β ^ (-(3 / 4 : ℝ))
        = β ^ (1 : ℝ) * β ^ (-(3 / 4 : ℝ)) := by rw [Real.rpow_one]
        _ = β ^ ((1 : ℝ) + -(3 / 4 : ℝ)) := (Real.rpow_add hβ_pos _ _).symm
        _ = β ^ (1 / 4 : ℝ) := by norm_num
    have hβL : β * bilinearLoss Tstar θ ≤ β * Lwall + (CL + 8 + 16 * D0) * β ^ (1 / 4 : ℝ) := by
      calc β * bilinearLoss Tstar θ
        ≤ β * (Lwall + (CL + 8 + 16 * D0) * β ^ (-(3 / 4 : ℝ))) :=
          mul_le_mul_of_nonneg_left hL_le hβ_pos.le
        _ = β * Lwall + (CL + 8 + 16 * D0) * (β * β ^ (-(3 / 4 : ℝ))) := by ring
        _ = β * Lwall + (CL + 8 + 16 * D0) * β ^ (1 / 4 : ℝ) := by rw [hβmul]
    have hR2_le : paramNormSq θ ≤ 108 * BG ^ 2 * β ^ (1 / 4 : ℝ) := by
      have hsq1 : ∀ i r, (θ.1 i r) ^ 2 ≤ M ^ 2 := fun i r => by
        have h := hθ_1 i r; rw [← sq_abs, sq, sq]; exact mul_le_mul h h (abs_nonneg _) hM_nonneg
      have hsq2 : ∀ j r, (θ.2.1 j r) ^ 2 ≤ M ^ 2 := fun j r => by
        have h := hθ_2 j r; rw [← sq_abs, sq, sq]; exact mul_le_mul h h (abs_nonneg _) hM_nonneg
      have hsq3 : ∀ l r, (θ.2.2 l r) ^ 2 ≤ M ^ 2 := fun l r => by
        have h := hθ_3 l r; rw [← sq_abs, sq, sq]; exact mul_le_mul h h (abs_nonneg _) hM_nonneg
      unfold paramNormSq
      simp only [Fin.sum_univ_two]
      linarith only [hsq1 0 0, hsq1 0 1, hsq1 1 0, hsq1 1 1,
        hsq2 0 0, hsq2 0 1, hsq2 1 0, hsq2 1 1,
        hsq3 0 0, hsq3 0 1, hsq3 1 0, hsq3 1 1, hM2]
    linarith only [hβL, hR2_le]
  set E0 : ℝ := CL + 8 + 16 * D0 + 54 * BG ^ 2
  have hlow : ENNReal.ofReal (Real.exp (-(β * Lwall + E0 * β ^ (1 / 4 : ℝ)))) *
      volume (genParamBox θ₀ ρ) ≤ tensorPartitionZ Tstar 2 β := by
    rw [(MeasureTheory.setLIntegral_const (genParamBox θ₀ ρ)
      (ENNReal.ofReal (Real.exp (-(β * Lwall + E0 * β ^ (1 / 4 : ℝ)))))).symm]
    refine le_trans ?_ (MeasureTheory.setLIntegral_le_lintegral (genParamBox θ₀ ρ) _)
    exact MeasureTheory.setLIntegral_mono_ae' (gen_measurableSet_paramBox θ₀ ρ)
      (Filter.Eventually.of_forall fun θ hθ =>
        ENNReal.ofReal_le_ofReal (Real.exp_le_exp.mpr (by linarith [hexp_le θ hθ])))
  rw [gen_volume_paramBox θ₀ hρ_pos.le, ← ENNReal.ofReal_mul (Real.exp_pos _).le] at hlow
  refine le_trans ?_ hlow
  apply ENNReal.ofReal_le_ofReal
  -- Absorb `β^{-12} ≥ exp(-48 β^{1/4})` via `log β ≤ 4 β^{1/4}`.
  have hlog : Real.log β ≤ 4 * β ^ (1 / 4 : ℝ) := by
    have h1 : Real.log (β ^ (1 / 4 : ℝ)) ≤ β ^ (1 / 4 : ℝ) - 1 :=
      Real.log_le_sub_one_of_pos (by positivity)
    rw [Real.log_rpow hβ_pos] at h1
    linarith
  have hβpow_ge : Real.exp (-(48 * β ^ (1 / 4 : ℝ))) ≤ (β ^ (-1 : ℝ)) ^ 12 := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hβ_pos.le,
      show (-1 : ℝ) * (12 : ℕ) = -12 by norm_num]
    conv_rhs => rw [Real.rpow_def_of_pos hβ_pos]
    apply Real.exp_le_exp.mpr
    linarith
  have h2ρ : (2 * ρ) ^ 12 = c_vol * (β ^ (-1 : ℝ)) ^ 12 := by
    simp only [ρ, c_vol]; rw [show 2 * (ρ0 * β ^ (-1 : ℝ)) = (2 * ρ0) * β ^ (-1 : ℝ) by ring, mul_pow]
  rw [show 6 * 2 = 12 by rfl, h2ρ]
  have hrearr : c_vol * Real.exp (-(β * Lwall + a₂ * β ^ (1 / 4 : ℝ))) =
      Real.exp (-(β * Lwall + E0 * β ^ (1 / 4 : ℝ))) *
        (c_vol * Real.exp (-(48 * β ^ (1 / 4 : ℝ)))) := by
    rw [mul_left_comm, ← Real.exp_add]; congr 2; simp only [a₂, E0]; ring
  rw [hrearr]
  exact mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hβpow_ge (by positivity))
    (Real.exp_pos _).le

/-! ### 4. Unconditional global certification of canonical `T₃` via Hadamard–Heron identities -/

/-- The 8-coordinate integer Hadamard transform of `T ∈ ℝ^{2×2×2}` corresponding to the
complex orthonormal basis `z = (e₀ + i e₁)/√2`, scaled by `2√2`. -/
private def hX0 (T : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)) : ℝ :=
  T 0 0 0 - T 0 1 1 - T 1 0 1 - T 1 1 0
private def hY0 (T : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)) : ℝ :=
  -T 0 0 1 - T 0 1 0 - T 1 0 0 + T 1 1 1
private def hX1 (T : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)) : ℝ :=
  T 0 0 0 - T 0 1 1 + T 1 0 1 + T 1 1 0
private def hY1 (T : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)) : ℝ :=
  -T 0 0 1 - T 0 1 0 + T 1 0 0 - T 1 1 1
private def hX2 (T : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)) : ℝ :=
  T 0 0 0 + T 0 1 1 - T 1 0 1 + T 1 1 0
private def hY2 (T : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)) : ℝ :=
  -T 0 0 1 + T 0 1 0 - T 1 0 0 - T 1 1 1
private def hX3 (T : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)) : ℝ :=
  T 0 0 0 + T 0 1 1 + T 1 0 1 - T 1 1 0
private def hY3 (T : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)) : ℝ :=
  T 0 0 1 - T 0 1 0 - T 1 0 0 - T 1 1 1

/-- Exact Hadamard norm identity for `‖T - T₃‖_F²`. -/
private theorem t3_hadamard_norm_identity (T : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)) :
    4 * tensorFrobSq (fun i j l => T i j l - t3Tensor ℝ i j l) =
      (hX0 T - 4) ^ 2 + (hY0 T) ^ 2 + (hX1 T) ^ 2 + (hY1 T) ^ 2 +
      (hX2 T) ^ 2 + (hY2 T) ^ 2 + (hX3 T) ^ 2 + (hY3 T) ^ 2 := by
  unfold tensorFrobSq t3Tensor hX0 hY0 hX1 hY1 hX2 hY2 hX3 hY3
  simp only [Fin.sum_univ_two]
  norm_num
  ring

/-- Exact Hadamard identity for Cayley's hyperdeterminant:
`64 * cayleyDet T = -((R₀ - R₁ - R₂ - R₃)² - 4 (R₁ R₂ + R₂ R₃ + R₃ R₁) + 8 P)`. -/
private theorem t3_hadamard_cayleyDet_identity (T : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)) :
    let R0 := (hX0 T) ^ 2 + (hY0 T) ^ 2
    let R1 := (hX1 T) ^ 2 + (hY1 T) ^ 2
    let R2 := (hX2 T) ^ 2 + (hY2 T) ^ 2
    let R3 := (hX3 T) ^ 2 + (hY3 T) ^ 2
    let re123 := hX1 T * hX2 T * hX3 T - hX1 T * hY2 T * hY3 T -
                 hY1 T * hX2 T * hY3 T - hY1 T * hY2 T * hX3 T
    let im123 := hX1 T * hX2 T * hY3 T + hX1 T * hY2 T * hX3 T +
                 hY1 T * hX2 T * hX3 T - hY1 T * hY2 T * hY3 T
    let P := hX0 T * re123 + hY0 T * im123
    64 * cayleyDet T =
      -((R0 - R1 - R2 - R3) ^ 2 - 4 * (R1 * R2 + R2 * R3 + R3 * R1) + 8 * P) := by
  unfold cayleyDet hX0 hY0 hX1 hY1 hX2 hY2 hX3 hY3
  dsimp only
  ring

/-- Exact Hadamard lower bounds for the three `SO(2)³`-invariant mode slice-pencil squares. -/
private theorem t3_hadamard_modePencil_bounds (T : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)) :
    let R0 := (hX0 T) ^ 2 + (hY0 T) ^ 2
    let R1 := (hX1 T) ^ 2 + (hY1 T) ^ 2
    let R2 := (hX2 T) ^ 2 + (hY2 T) ^ 2
    let R3 := (hX3 T) ^ 2 + (hY3 T) ^ 2
    (R0 + R1 - R2 - R3) ^ 2 ≤ 64 * mode1PencilSq T ∧
    (R0 + R2 - R1 - R3) ^ 2 ≤ 64 * mode2PencilSq T ∧
    (R0 + R3 - R1 - R2) ^ 2 ≤ 64 * mode3PencilSq T := by
  dsimp only
  refine ⟨?_, ?_, ?_⟩
  · have hid : 64 * mode1PencilSq T =
        4 * ((hX0 T * hX1 T + hY0 T * hY1 T - (hX2 T * hX3 T - hY2 T * hY3 T)) ^ 2 +
             (hY0 T * hX1 T - hX0 T * hY1 T - (hX2 T * hY3 T + hY2 T * hX3 T)) ^ 2) +
        ((hX0 T) ^ 2 + (hY0 T) ^ 2 + ((hX1 T) ^ 2 + (hY1 T) ^ 2) -
         ((hX2 T) ^ 2 + (hY2 T) ^ 2) - ((hX3 T) ^ 2 + (hY3 T) ^ 2)) ^ 2 := by
      unfold mode1PencilSq hX0 hY0 hX1 hY1 hX2 hY2 hX3 hY3; ring
    rw [hid]
    nlinarith [sq_nonneg (hX0 T * hX1 T + hY0 T * hY1 T - (hX2 T * hX3 T - hY2 T * hY3 T)),
      sq_nonneg (hY0 T * hX1 T - hX0 T * hY1 T - (hX2 T * hY3 T + hY2 T * hX3 T))]
  · have hid : 64 * mode2PencilSq T =
        4 * ((hX0 T * hX2 T + hY0 T * hY2 T - (hX1 T * hX3 T - hY1 T * hY3 T)) ^ 2 +
             (hY0 T * hX2 T - hX0 T * hY2 T - (hX1 T * hY3 T + hY1 T * hX3 T)) ^ 2) +
        ((hX0 T) ^ 2 + (hY0 T) ^ 2 + ((hX2 T) ^ 2 + (hY2 T) ^ 2) -
         ((hX1 T) ^ 2 + (hY1 T) ^ 2) - ((hX3 T) ^ 2 + (hY3 T) ^ 2)) ^ 2 := by
      unfold mode2PencilSq hX0 hY0 hX1 hY1 hX2 hY2 hX3 hY3; ring
    rw [hid]
    nlinarith [sq_nonneg (hX0 T * hX2 T + hY0 T * hY2 T - (hX1 T * hX3 T - hY1 T * hY3 T)),
      sq_nonneg (hY0 T * hX2 T - hX0 T * hY2 T - (hX1 T * hY3 T + hY1 T * hX3 T))]
  · have hid : 64 * mode3PencilSq T =
        4 * ((hX0 T * hX3 T + hY0 T * hY3 T - (hX1 T * hX2 T - hY1 T * hY2 T)) ^ 2 +
             (hY0 T * hX3 T - hX0 T * hY3 T - (hX1 T * hY2 T + hY1 T * hX2 T)) ^ 2) +
        ((hX0 T) ^ 2 + (hY0 T) ^ 2 + ((hX3 T) ^ 2 + (hY3 T) ^ 2) -
         ((hX1 T) ^ 2 + (hY1 T) ^ 2) - ((hX2 T) ^ 2 + (hY2 T) ^ 2)) ^ 2 := by
      unfold mode3PencilSq hX0 hY0 hX1 hY1 hX2 hY2 hX3 hY3; ring
    rw [hid]
    nlinarith [sq_nonneg (hX0 T * hX3 T + hY0 T * hY3 T - (hX1 T * hX2 T - hY1 T * hY2 T)),
      sq_nonneg (hY0 T * hX3 T - hX0 T * hY3 T - (hX1 T * hY2 T + hY1 T * hX2 T))]

/-- **Heron Hyperdeterminant Upper Bound (`[LC]`):**
In terms of the four Hadamard moduli `r_k = √(hX_k(T)² + hY_k(T)²) ≥ 0`,
`64 * cayleyDet T ≤ (r₁ + r₂ + r₃ - r₀)(r₀ + r₁ + r₂ - r₃)(r₀ + r₁ - r₂ + r₃)(r₀ - r₁ + r₂ + r₃)`. -/
private theorem t3_heron_cayleyDet_le (T : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)) :
    let r0 := Real.sqrt ((hX0 T) ^ 2 + (hY0 T) ^ 2)
    let r1 := Real.sqrt ((hX1 T) ^ 2 + (hY1 T) ^ 2)
    let r2 := Real.sqrt ((hX2 T) ^ 2 + (hY2 T) ^ 2)
    let r3 := Real.sqrt ((hX3 T) ^ 2 + (hY3 T) ^ 2)
    64 * cayleyDet T ≤
      (r1 + r2 + r3 - r0) * (r0 + r1 + r2 - r3) * (r0 + r1 - r2 + r3) * (r0 - r1 + r2 + r3) := by
  set R0 := (hX0 T) ^ 2 + (hY0 T) ^ 2
  set R1 := (hX1 T) ^ 2 + (hY1 T) ^ 2
  set R2 := (hX2 T) ^ 2 + (hY2 T) ^ 2
  set R3 := (hX3 T) ^ 2 + (hY3 T) ^ 2
  set r0 := Real.sqrt R0
  set r1 := Real.sqrt R1
  set r2 := Real.sqrt R2
  set r3 := Real.sqrt R3
  have hr0_sq : r0 ^ 2 = R0 := Real.sq_sqrt (by positivity)
  have hr1_sq : r1 ^ 2 = R1 := Real.sq_sqrt (by positivity)
  have hr2_sq : r2 ^ 2 = R2 := Real.sq_sqrt (by positivity)
  have hr3_sq : r3 ^ 2 = R3 := Real.sq_sqrt (by positivity)
  set re123 := hX1 T * hX2 T * hX3 T - hX1 T * hY2 T * hY3 T -
               hY1 T * hX2 T * hY3 T - hY1 T * hY2 T * hX3 T
  set im123 := hX1 T * hX2 T * hY3 T + hX1 T * hY2 T * hX3 T +
               hY1 T * hX2 T * hX3 T - hY1 T * hY2 T * hY3 T
  set P := hX0 T * re123 + hY0 T * im123
  set Q := hX0 T * im123 - hY0 T * re123
  have hcs : P ^ 2 + Q ^ 2 = R0 * R1 * R2 * R3 := by
    dsimp only [P, Q, re123, im123, R0, R1, R2, R3]; ring
  have hP_lb : -(r0 * r1 * r2 * r3) ≤ P := by
    have hprod_nn : 0 ≤ r0 * r1 * r2 * r3 := by positivity
    have hsq : P ^ 2 ≤ (r0 * r1 * r2 * r3) ^ 2 := by
      calc P ^ 2 ≤ P ^ 2 + Q ^ 2 := by nlinarith [sq_nonneg Q]
        _ = R0 * R1 * R2 * R3 := hcs
        _ = r0 ^ 2 * r1 ^ 2 * r2 ^ 2 * r3 ^ 2 := by rw [hr0_sq, hr1_sq, hr2_sq, hr3_sq]
        _ = (r0 * r1 * r2 * r3) ^ 2 := by ring
    nlinarith
  have hdet : 64 * cayleyDet T =
      -((R0 - R1 - R2 - R3) ^ 2 - 4 * (R1 * R2 + R2 * R3 + R3 * R1) + 8 * P) :=
    t3_hadamard_cayleyDet_identity T
  have hheron :
      -((r0 ^ 2 - r1 ^ 2 - r2 ^ 2 - r3 ^ 2) ^ 2 -
        4 * (r1 ^ 2 * r2 ^ 2 + r2 ^ 2 * r3 ^ 2 + r3 ^ 2 * r1 ^ 2) - 8 * (r0 * r1 * r2 * r3)) =
      (r1 + r2 + r3 - r0) * (r0 + r1 + r2 - r3) * (r0 + r1 - r2 + r3) * (r0 - r1 + r2 + r3) := by
    ring
  dsimp only
  rw [hdet, ← hr0_sq, ← hr1_sq, ← hr2_sq, ← hr3_sq]
  linarith only [hP_lb, hheron]

/-- Whenever `0 ≤ cayleyDet T`, the Hadamard moduli satisfy `r₀ ≤ r₁ + r₂ + r₃`. -/
private theorem t3_heron_sum_nonneg {T : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)}
    (hdet : 0 ≤ cayleyDet T) :
    0 ≤ Real.sqrt ((hX1 T) ^ 2 + (hY1 T) ^ 2) + Real.sqrt ((hX2 T) ^ 2 + (hY2 T) ^ 2) +
        Real.sqrt ((hX3 T) ^ 2 + (hY3 T) ^ 2) - Real.sqrt ((hX0 T) ^ 2 + (hY0 T) ^ 2) := by
  set r0 := Real.sqrt ((hX0 T) ^ 2 + (hY0 T) ^ 2)
  set r1 := Real.sqrt ((hX1 T) ^ 2 + (hY1 T) ^ 2)
  set r2 := Real.sqrt ((hX2 T) ^ 2 + (hY2 T) ^ 2)
  set r3 := Real.sqrt ((hX3 T) ^ 2 + (hY3 T) ^ 2)
  have hr1 : 0 ≤ r1 := Real.sqrt_nonneg _
  have hr2 : 0 ≤ r2 := Real.sqrt_nonneg _
  have hr3 : 0 ≤ r3 := Real.sqrt_nonneg _
  by_contra hlt
  push Not at hlt
  have hf1 : 0 < r0 + r1 + r2 - r3 := by linarith
  have hf2 : 0 < r0 + r1 - r2 + r3 := by linarith
  have hf3 : 0 < r0 - r1 + r2 + r3 := by linarith
  have hprod_neg :
      (r1 + r2 + r3 - r0) * (r0 + r1 + r2 - r3) * (r0 + r1 - r2 + r3) * (r0 - r1 + r2 + r3) < 0 := by
    have hpos : 0 < (r0 + r1 + r2 - r3) * (r0 + r1 - r2 + r3) * (r0 - r1 + r2 + r3) := by positivity
    nlinarith
  have hheron := t3_heron_cayleyDet_le T
  dsimp only at hheron
  linarith

/-- Exact Hadamard-4 identity for `16 (‖T - T₃‖_F² - 1)` in terms of `s = r₁ + r₂ + r₃ - r₀`. -/
private theorem t3_excessLoss_hadamard4 (T : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)) :
    let r0 := Real.sqrt ((hX0 T) ^ 2 + (hY0 T) ^ 2)
    let r1 := Real.sqrt ((hX1 T) ^ 2 + (hY1 T) ^ 2)
    let r2 := Real.sqrt ((hX2 T) ^ 2 + (hY2 T) ^ 2)
    let r3 := Real.sqrt ((hX3 T) ^ 2 + (hY3 T) ^ 2)
    let s := r1 + r2 + r3 - r0
    16 * (tensorFrobSq (fun i j l => T i j l - t3Tensor ℝ i j l) - 1) =
      32 * (r0 - hX0 T) + 8 * s + s ^ 2 +
      ((4 - r0) + r1 - r2 - r3) ^ 2 +
      ((4 - r0) - r1 + r2 - r3) ^ 2 +
      ((4 - r0) - r1 - r2 + r3) ^ 2 := by
  set r0 := Real.sqrt ((hX0 T) ^ 2 + (hY0 T) ^ 2)
  set r1 := Real.sqrt ((hX1 T) ^ 2 + (hY1 T) ^ 2)
  set r2 := Real.sqrt ((hX2 T) ^ 2 + (hY2 T) ^ 2)
  set r3 := Real.sqrt ((hX3 T) ^ 2 + (hY3 T) ^ 2)
  have hr0_sq : r0 ^ 2 = (hX0 T) ^ 2 + (hY0 T) ^ 2 := Real.sq_sqrt (by positivity)
  have hr1_sq : r1 ^ 2 = (hX1 T) ^ 2 + (hY1 T) ^ 2 := Real.sq_sqrt (by positivity)
  have hr2_sq : r2 ^ 2 = (hX2 T) ^ 2 + (hY2 T) ^ 2 := Real.sq_sqrt (by positivity)
  have hr3_sq : r3 ^ 2 = (hX3 T) ^ 2 + (hY3 T) ^ 2 := Real.sq_sqrt (by positivity)
  have hnorm := t3_hadamard_norm_identity T
  have hstep : 4 * ((hX0 T - 4) ^ 2 + (hY0 T) ^ 2 + (hX1 T) ^ 2 + (hY1 T) ^ 2 +
      (hX2 T) ^ 2 + (hY2 T) ^ 2 + (hX3 T) ^ 2 + (hY3 T) ^ 2 - 4) =
      32 * (r0 - hX0 T) + 4 * ((4 - r0) ^ 2 + r1 ^ 2 + r2 ^ 2 + r3 ^ 2 - 4) := by
    linear_combination 4 * hr0_sq.symm + 4 * hr1_sq.symm + 4 * hr2_sq.symm + 4 * hr3_sq.symm
  have hhad4 : 4 * ((4 - r0) ^ 2 + r1 ^ 2 + r2 ^ 2 + r3 ^ 2 - 4) =
      8 * (r1 + r2 + r3 - r0) + (r1 + r2 + r3 - r0) ^ 2 +
      ((4 - r0) + r1 - r2 - r3) ^ 2 +
      ((4 - r0) - r1 + r2 - r3) ^ 2 +
      ((4 - r0) - r1 - r2 + r3) ^ 2 := by ring
  linarith

/-- **Unconditional Exact Wall Loss for `T₃` (`[LC]`):**
For *every* `2 × 2 × 2` real tensor `T` with `0 ≤ cayleyDet T` (in particular every rank-`≤ 2`
tensor `bilinearTensor θ`), `1 ≤ ‖T - T₃‖_F²`. -/
theorem t3Tensor_loss_ge_one (T : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)) (hdet : 0 ≤ cayleyDet T) :
    1 ≤ tensorFrobSq (fun i j l => T i j l - t3Tensor ℝ i j l) := by
  have hs := t3_heron_sum_nonneg hdet
  have hhad := t3_excessLoss_hadamard4 T
  dsimp only at hhad
  have hr0_ge : hX0 T ≤ Real.sqrt ((hX0 T) ^ 2 + (hY0 T) ^ 2) := by
    have hsq : (hX0 T) ^ 2 ≤ (Real.sqrt ((hX0 T) ^ 2 + (hY0 T) ^ 2)) ^ 2 := by
      rw [Real.sq_sqrt (by positivity)]; nlinarith [sq_nonneg (hY0 T)]
    nlinarith [Real.sqrt_nonneg ((hX0 T) ^ 2 + (hY0 T) ^ 2)]
  nlinarith [sq_nonneg (Real.sqrt ((hX1 T) ^ 2 + (hY1 T) ^ 2) +
      Real.sqrt ((hX2 T) ^ 2 + (hY2 T) ^ 2) +
      Real.sqrt ((hX3 T) ^ 2 + (hY3 T) ^ 2) -
      Real.sqrt ((hX0 T) ^ 2 + (hY0 T) ^ 2)),
    sq_nonneg ((4 - Real.sqrt ((hX0 T) ^ 2 + (hY0 T) ^ 2)) +
      Real.sqrt ((hX1 T) ^ 2 + (hY1 T) ^ 2) -
      Real.sqrt ((hX2 T) ^ 2 + (hY2 T) ^ 2) -
      Real.sqrt ((hX3 T) ^ 2 + (hY3 T) ^ 2)),
    sq_nonneg ((4 - Real.sqrt ((hX0 T) ^ 2 + (hY0 T) ^ 2)) -
      Real.sqrt ((hX1 T) ^ 2 + (hY1 T) ^ 2) +
      Real.sqrt ((hX2 T) ^ 2 + (hY2 T) ^ 2) -
      Real.sqrt ((hX3 T) ^ 2 + (hY3 T) ^ 2)),
    sq_nonneg ((4 - Real.sqrt ((hX0 T) ^ 2 + (hY0 T) ^ 2)) -
      Real.sqrt ((hX1 T) ^ 2 + (hY1 T) ^ 2) -
      Real.sqrt ((hX2 T) ^ 2 + (hY2 T) ^ 2) +
      Real.sqrt ((hX3 T) ^ 2 + (hY3 T) ^ 2))]

/-- Near-wall box bounds on the four Hadamard moduli `(r₀, r₁, r₂, r₃)` when
`0 ≤ cayleyDet T` and `‖T - T₃‖_F² - 1 < 1/64`. -/
private theorem t3_near_wall_boxes {T : Tensor ℝ (Fin 2) (Fin 2) (Fin 2)}
    (hdet : 0 ≤ cayleyDet T)
    (hsmall : tensorFrobSq (fun i j l => T i j l - t3Tensor ℝ i j l) - 1 < 1 / 64) :
    let r0 := Real.sqrt ((hX0 T) ^ 2 + (hY0 T) ^ 2)
    let r1 := Real.sqrt ((hX1 T) ^ 2 + (hY1 T) ^ 2)
    let r2 := Real.sqrt ((hX2 T) ^ 2 + (hY2 T) ^ 2)
    let r3 := Real.sqrt ((hX3 T) ^ 2 + (hY3 T) ^ 2)
    let s := r1 + r2 + r3 - r0
    (0 ≤ s ∧ 8 * s ≤ 16 * (tensorFrobSq (fun i j l => T i j l - t3Tensor ℝ i j l) - 1)) ∧
    (335 / 128 ≤ r0 ∧ r0 ≤ 27 / 8) ∧
    (5 / 8 ≤ r1 ∧ r1 ≤ 177 / 128) ∧
    (5 / 8 ≤ r2 ∧ r2 ≤ 177 / 128) ∧
    (5 / 8 ≤ r3 ∧ r3 ≤ 177 / 128) := by
  set r0 := Real.sqrt ((hX0 T) ^ 2 + (hY0 T) ^ 2)
  set r1 := Real.sqrt ((hX1 T) ^ 2 + (hY1 T) ^ 2)
  set r2 := Real.sqrt ((hX2 T) ^ 2 + (hY2 T) ^ 2)
  set r3 := Real.sqrt ((hX3 T) ^ 2 + (hY3 T) ^ 2)
  set s := r1 + r2 + r3 - r0
  set u1 := (4 - r0) + r1 - r2 - r3
  set u2 := (4 - r0) - r1 + r2 - r3
  set u3 := (4 - r0) - r1 - r2 + r3
  have hs : 0 ≤ s := t3_heron_sum_nonneg hdet
  have hhad := t3_excessLoss_hadamard4 T
  dsimp only at hhad
  have hr0_ge : hX0 T ≤ r0 := by
    have hsq : (hX0 T) ^ 2 ≤ r0 ^ 2 := by
      rw [Real.sq_sqrt (by positivity)]; nlinarith [sq_nonneg (hY0 T)]
    nlinarith [Real.sqrt_nonneg ((hX0 T) ^ 2 + (hY0 T) ^ 2)]
  have h8s_le : 8 * s ≤ 16 * (tensorFrobSq (fun i j l => T i j l - t3Tensor ℝ i j l) - 1) := by
    linarith only [hhad, hr0_ge, sq_nonneg s, sq_nonneg u1, sq_nonneg u2, sq_nonneg u3]
  have h16 : 32 * (r0 - hX0 T) + 8 * s + s ^ 2 + u1 ^ 2 + u2 ^ 2 + u3 ^ 2 < 1 / 4 := by
    linarith only [hhad, hsmall]
  have hs_le : s ≤ 1 / 32 := by
    linarith only [h16, hr0_ge, sq_nonneg s, sq_nonneg u1, sq_nonneg u2, sq_nonneg u3]
  have hu1_sq : u1 ^ 2 ≤ 1 / 4 := by
    linarith only [h16, hr0_ge, hs, sq_nonneg s, sq_nonneg u2, sq_nonneg u3]
  have hu2_sq : u2 ^ 2 ≤ 1 / 4 := by
    linarith only [h16, hr0_ge, hs, sq_nonneg s, sq_nonneg u1, sq_nonneg u3]
  have hu3_sq : u3 ^ 2 ≤ 1 / 4 := by
    linarith only [h16, hr0_ge, hs, sq_nonneg s, sq_nonneg u1, sq_nonneg u2]
  have hu1 : -(1 / 2 : ℝ) ≤ u1 ∧ u1 ≤ 1 / 2 := by constructor <;> nlinarith only [hu1_sq]
  have hu2 : -(1 / 2 : ℝ) ≤ u2 ∧ u2 ≤ 1 / 2 := by constructor <;> nlinarith only [hu2_sq]
  have hu3 : -(1 / 2 : ℝ) ≤ u3 ∧ u3 ≤ 1 / 2 := by constructor <;> nlinarith only [hu3_sq]
  have hr0_bd : 335 / 128 ≤ r0 ∧ r0 ≤ 27 / 8 := by
    have hinv : 4 * (4 - r0) = (4 + s) + u1 + u2 + u3 := by dsimp only [s, u1, u2, u3]; ring
    constructor <;> linarith only [hinv, hs, hs_le, hu1.1, hu1.2, hu2.1, hu2.2, hu3.1, hu3.2]
  have hr1_bd : 5 / 8 ≤ r1 ∧ r1 ≤ 177 / 128 := by
    have hinv : 4 * r1 = (4 + s) + u1 - u2 - u3 := by dsimp only [s, u1, u2, u3]; ring
    constructor <;> linarith only [hinv, hs, hs_le, hu1.1, hu1.2, hu2.1, hu2.2, hu3.1, hu3.2]
  have hr2_bd : 5 / 8 ≤ r2 ∧ r2 ≤ 177 / 128 := by
    have hinv : 4 * r2 = (4 + s) - u1 + u2 - u3 := by dsimp only [s, u1, u2, u3]; ring
    constructor <;> linarith only [hinv, hs, hs_le, hu1.1, hu1.2, hu2.1, hu2.2, hu3.1, hu3.2]
  have hr3_bd : 5 / 8 ≤ r3 ∧ r3 ≤ 177 / 128 := by
    have hinv : 4 * r3 = (4 + s) - u1 - u2 + u3 := by dsimp only [s, u1, u2, u3]; ring
    constructor <;> linarith only [hinv, hs, hs_le, hu1.1, hu1.2, hu2.1, hu2.2, hu3.1, hu3.2]
  exact ⟨⟨hs, h8s_le⟩, hr0_bd, hr1_bd, hr2_bd, hr3_bd⟩

/-- **Unconditional `GenericWallHyp (t3Tensor ℝ) 1` (`[LC]`):**
With explicit constants `δ₀ = 1/64`, `α = 1/8`, `m₀ = 3/8`, `t3Tensor ℝ` satisfies every
condition of `GenericWallHyp` globally on `ℝ^{2×2×2}` with zero unverified assumptions. -/
noncomputable def t3Tensor_genericWallHyp : GenericWallHyp (t3Tensor ℝ) 1 where
  δ₀ := 1 / 64
  α := 1 / 8
  m₀ := 3 / 8
  hδ₀ := by norm_num
  hα := by norm_num
  hm₀ := by norm_num
  wall_le := t3Tensor_loss_ge_one
  near_wall_nondeg := by
    intro T hdet hsmall
    obtain ⟨_, hr0_bd, hr1_bd, hr2_bd, hr3_bd⟩ := t3_near_wall_boxes hdet hsmall
    set r0 := Real.sqrt ((hX0 T) ^ 2 + (hY0 T) ^ 2)
    set r1 := Real.sqrt ((hX1 T) ^ 2 + (hY1 T) ^ 2)
    set r2 := Real.sqrt ((hX2 T) ^ 2 + (hY2 T) ^ 2)
    set r3 := Real.sqrt ((hX3 T) ^ 2 + (hY3 T) ^ 2)
    have hr0_sq : r0 ^ 2 = (hX0 T) ^ 2 + (hY0 T) ^ 2 := Real.sq_sqrt (by positivity)
    have hr1_sq : r1 ^ 2 = (hX1 T) ^ 2 + (hY1 T) ^ 2 := Real.sq_sqrt (by positivity)
    have hr2_sq : r2 ^ 2 = (hX2 T) ^ 2 + (hY2 T) ^ 2 := Real.sq_sqrt (by positivity)
    have hr3_sq : r3 ^ 2 = (hX3 T) ^ 2 + (hY3 T) ^ 2 := Real.sq_sqrt (by positivity)
    obtain ⟨hq1, hq2, hq3⟩ := t3_hadamard_modePencil_bounds T
    rw [← hr0_sq, ← hr1_sq, ← hr2_sq, ← hr3_sq] at hq1 hq2 hq3
    have hdiff1 : 3 ≤ r0 ^ 2 + r1 ^ 2 - r2 ^ 2 - r3 ^ 2 := by
      nlinarith only [hr0_bd.1, hr1_bd.1, hr2_bd.1, hr2_bd.2, hr3_bd.1, hr3_bd.2]
    have hdiff2 : 3 ≤ r0 ^ 2 + r2 ^ 2 - r1 ^ 2 - r3 ^ 2 := by
      nlinarith only [hr0_bd.1, hr2_bd.1, hr1_bd.1, hr1_bd.2, hr3_bd.1, hr3_bd.2]
    have hdiff3 : 3 ≤ r0 ^ 2 + r3 ^ 2 - r1 ^ 2 - r2 ^ 2 := by
      nlinarith only [hr0_bd.1, hr3_bd.1, hr1_bd.1, hr1_bd.2, hr2_bd.1, hr2_bd.2]
    refine ⟨?_, ?_, ?_⟩
    · nlinarith only [hq1, hdiff1]
    · nlinarith only [hq2, hdiff2]
    · nlinarith only [hq3, hdiff3]
  near_wall_linear := by
    intro T hdet hsmall
    obtain ⟨⟨hs, h8s_le⟩, hr0_bd, hr1_bd, hr2_bd, hr3_bd⟩ := t3_near_wall_boxes hdet hsmall
    set r0 := Real.sqrt ((hX0 T) ^ 2 + (hY0 T) ^ 2)
    set r1 := Real.sqrt ((hX1 T) ^ 2 + (hY1 T) ^ 2)
    set r2 := Real.sqrt ((hX2 T) ^ 2 + (hY2 T) ^ 2)
    set r3 := Real.sqrt ((hX3 T) ^ 2 + (hY3 T) ^ 2)
    set s := r1 + r2 + r3 - r0
    have hf1 : 0 ≤ r0 + r1 + r2 - r3 ∧ r0 + r1 + r2 - r3 ≤ 6 := by
      constructor <;> linarith only [hr0_bd.1, hr0_bd.2, hr1_bd.1, hr1_bd.2, hr2_bd.1, hr2_bd.2, hr3_bd.1, hr3_bd.2]
    have hf2 : 0 ≤ r0 + r1 - r2 + r3 ∧ r0 + r1 - r2 + r3 ≤ 6 := by
      constructor <;> linarith only [hr0_bd.1, hr0_bd.2, hr1_bd.1, hr1_bd.2, hr2_bd.1, hr2_bd.2, hr3_bd.1, hr3_bd.2]
    have hf3 : 0 ≤ r0 - r1 + r2 + r3 ∧ r0 - r1 + r2 + r3 ≤ 6 := by
      constructor <;> linarith only [hr0_bd.1, hr0_bd.2, hr1_bd.1, hr1_bd.2, hr2_bd.1, hr2_bd.2, hr3_bd.1, hr3_bd.2]
    have hprod : (r0 + r1 + r2 - r3) * (r0 + r1 - r2 + r3) * (r0 - r1 + r2 + r3) ≤ 216 := by
      have h12 : (r0 + r1 + r2 - r3) * (r0 + r1 - r2 + r3) ≤ 36 :=
        (mul_le_mul hf1.2 hf2.2 hf2.1 (by norm_num)).trans (by norm_num)
      exact (mul_le_mul h12 hf3.2 hf3.1 (by norm_num)).trans (by norm_num)
    have hheron := t3_heron_cayleyDet_le T
    dsimp only at hheron
    have hheron' : 64 * cayleyDet T ≤
        s * ((r0 + r1 + r2 - r3) * (r0 + r1 - r2 + r3) * (r0 - r1 + r2 + r3)) := by
      have heq : (r1 + r2 + r3 - r0) * (r0 + r1 + r2 - r3) * (r0 + r1 - r2 + r3) * (r0 - r1 + r2 + r3) =
          s * ((r0 + r1 + r2 - r3) * (r0 + r1 - r2 + r3) * (r0 - r1 + r2 + r3)) := by
        dsimp only [s]; ring
      linarith only [hheron, heq]
    have h64 : 64 * cayleyDet T ≤ 216 * s := by
      have := mul_le_mul_of_nonneg_left hprod hs
      linarith only [hheron', this]
    linarith only [h64, h8s_le, hs]

/-- **Unconditional `R^{-6}` Excess-Loss Lower Bound for `T₃` (`[LC]`):**
For *every* width-2 parameter `θ ∈ ℝ^{12}`,
`min (1/64) ((1/2) / (paramNormSq θ)³) ≤ bilinearLoss (t3Tensor ℝ) θ - 1`. -/
theorem t3Tensor_excessLoss_ge_of_paramNormSq (θ : BilinearParam 2 2 2 2) :
    min ((1 : ℝ) / 64) ((1 / 2 : ℝ) / (paramNormSq θ) ^ 3) ≤
      bilinearLoss (t3Tensor ℝ) θ - 1 := by
  have h := generic_escape_lower t3Tensor_genericWallHyp θ
  dsimp only [t3Tensor_genericWallHyp] at h
  have hconst : (1 / 2 : ℝ) ≤ 76 * (1 / 8 : ℝ) * (3 / 8 : ℝ) ^ 3 := by norm_num
  have hdiv : (1 / 2 : ℝ) / (paramNormSq θ) ^ 3 ≤
      76 * (1 / 8 : ℝ) * (3 / 8 : ℝ) ^ 3 / (paramNormSq θ) ^ 3 :=
    div_le_div_of_nonneg_right hconst (pow_nonneg (paramNormSq_nonneg θ) 3)
  exact le_trans (min_le_min_left _ hdiv) h

/-- The optimal centered border-rank escape curve targeting `T₃`:
swapping `e₀ ↔ e₁` on all three modes and negating mode 3 maps `W + ε² e₁^{⊗3}` to
`W_sym - ε² e₀^{⊗3}`, achieving exact excess loss `2 ε² + ε⁴` at unchanged norm². -/
noncomputable def t3Curve (ε : ℝ) : BilinearParam 2 2 2 2 :=
  ( fun i r => (centeredCurve ε).1 (1 - i) r,
    fun j r => (centeredCurve ε).2.1 (1 - j) r,
    fun l r => -((centeredCurve ε).2.2 (1 - l) r) )

theorem t3Curve_tensor {ε : ℝ} (hε : 0 < ε) (i j l : Fin 2) :
    bilinearTensor (t3Curve ε) i j l =
      t3Tensor ℝ i j l - (if i = 0 ∧ j = 0 ∧ l = 0 then 1 + ε ^ 2 else 0) := by
  have hrel : bilinearTensor (t3Curve ε) i j l =
      -bilinearTensor (centeredCurve ε) (1 - i) (1 - j) (1 - l) := by
    unfold bilinearTensor t3Curve
    simp only [Fin.sum_univ_two]; ring
  rw [hrel, centeredCurve_tensor hε]
  fin_cases i <;> fin_cases j <;> fin_cases l <;>
    simp [wTensor, t3Tensor]

theorem t3Curve_loss {ε : ℝ} (hε : 0 < ε) :
    bilinearLoss (t3Tensor ℝ) (t3Curve ε) = 1 + 2 * ε ^ 2 + ε ^ 4 := by
  unfold bilinearLoss tensorFrobSq
  simp only [Fin.sum_univ_two, t3Curve_tensor hε]
  norm_num
  ring

theorem t3Curve_paramNormSq {ε : ℝ} (hε : 0 < ε) :
    paramNormSq (t3Curve ε) = 6 * (2 * ε) ^ (-(2 / 3 : ℝ)) * (1 + ε ^ 2) := by
  rw [← centeredCurve_paramNormSq hε]
  unfold paramNormSq t3Curve
  simp only [Fin.sum_univ_two]
  norm_num
  ring

/-- **Unconditional `R^{-6}` Excess-Loss Upper Bound for `T₃` (`[LC]`):**
For all `R ≥ 4`, there exists a width-2 parameter `θ` with `paramNormSq θ ≤ R²` and
`bilinearLoss (t3Tensor ℝ) θ - 1 ≤ 1296 / R⁶`. -/
theorem t3Tensor_escape_upper {R : ℝ} (hR : 4 ≤ R) :
    ∃ θ : BilinearParam 2 2 2 2,
      paramNormSq θ ≤ R ^ 2 ∧ bilinearLoss (t3Tensor ℝ) θ - 1 ≤ 1296 / R ^ 6 := by
  have hR_pos : 0 < R := by linarith
  let ε : ℝ := (1 / 2 : ℝ) * (12 / R ^ 2) ^ (3 / 2 : ℝ)
  have h12R : 12 / R ^ 2 ≤ 1 := by
    rw [div_le_one (by positivity)]; nlinarith
  have hε_pos : 0 < ε := by positivity
  have hε_le : ε ≤ 1 := by
    have hpow : (12 / R ^ 2) ^ (3 / 2 : ℝ) ≤ 1 :=
      Real.rpow_le_one (by positivity) h12R (by positivity)
    simp only [ε]; linarith
  refine ⟨t3Curve ε, ?_, ?_⟩
  · rw [t3Curve_paramNormSq hε_pos]
    have h2ε : 2 * ε = (12 / R ^ 2) ^ (3 / 2 : ℝ) := by simp [ε]
    have hpow : (2 * ε) ^ (-(2 / 3 : ℝ)) = R ^ 2 / 12 := by
      rw [h2ε, ← Real.rpow_mul (by positivity : (0 : ℝ) ≤ 12 / R ^ 2)]
      norm_num
      rw [Real.rpow_neg_one, inv_div]
    rw [hpow]
    have hε2 : ε ^ 2 ≤ 1 := by nlinarith
    nlinarith
  · rw [t3Curve_loss hε_pos]
    have hε2 : ε ^ 2 = 432 / R ^ 6 := by
      have h1 : ε ^ 2 = (1 / 4 : ℝ) * ((12 / R ^ 2) ^ (3 / 2 : ℝ)) ^ 2 := by
        simp only [ε]; ring
      rw [h1, ← Real.rpow_natCast, ← Real.rpow_mul (by positivity : (0 : ℝ) ≤ 12 / R ^ 2)]
      norm_num
      rw [div_pow, ← pow_mul]
      ring
    have hε2_le1 : ε ^ 2 ≤ 1 := by nlinarith
    have hε4 : ε ^ 4 ≤ ε ^ 2 := by nlinarith [sq_nonneg ε]
    have h3 : 3 * (432 / R ^ 6) = 1296 / R ^ 6 := by ring
    linarith only [hε2, hε4, h3]

/-- **Unconditional Weight-Decay Scaling Law for `T₃` — Lower Bound (`γ^{3/4}`, `[LC]`):**
For all `γ ∈ (0, 1]` and all `θ : BilinearParam 2 2 2 2`,
`(1 / 64) * γ^{3/4} ≤ (bilinearLoss (t3Tensor ℝ) θ - 1) + γ * paramNormSq θ`. -/
theorem t3Tensor_weightDecay_lower {γ : ℝ} (hγ0 : 0 < γ) (hγ1 : γ ≤ 1)
    (θ : BilinearParam 2 2 2 2) :
    (1 / 64 : ℝ) * γ ^ (3 / 4 : ℝ) ≤
      (bilinearLoss (t3Tensor ℝ) θ - 1) + γ * paramNormSq θ := by
  have h := generic_weightDecay_lower t3Tensor_genericWallHyp hγ0 hγ1 θ
  have hconst : (1 / 64 : ℝ) ≤ genericWDConst t3Tensor_genericWallHyp := by
    unfold genericWDConst t3Tensor_genericWallHyp
    dsimp only
    apply le_min (le_refl _)
    have hpos : 0 ≤ (4 * (76 * (1 / 8 : ℝ) * (3 / 8 : ℝ) ^ 3)) ^ (1 / 4 : ℝ) *
        (4 / 3 : ℝ) ^ (3 / 4 : ℝ) := by positivity
    have hp1 : ((4 * (76 * (1 / 8 : ℝ) * (3 / 8 : ℝ) ^ 3)) ^ (1 / 4 : ℝ)) ^ 4 =
        4 * (76 * (1 / 8 : ℝ) * (3 / 8 : ℝ) ^ 3) := by
      rw [← Real.rpow_natCast, ← Real.rpow_mul (by positivity)]; norm_num
    have hp2 : ((4 / 3 : ℝ) ^ (3 / 4 : ℝ)) ^ 4 = (4 / 3 : ℝ) ^ 3 := by
      rw [← Real.rpow_natCast, ← Real.rpow_mul (by positivity)]; norm_num
    have hpow4 : ((1 / 64 : ℝ)) ^ 4 ≤
        ((4 * (76 * (1 / 8 : ℝ) * (3 / 8 : ℝ) ^ 3)) ^ (1 / 4 : ℝ) * (4 / 3 : ℝ) ^ (3 / 4 : ℝ)) ^ 4 := by
      rw [mul_pow, hp1, hp2]; norm_num
    by_contra hlt
    push Not at hlt
    have : ((4 * (76 * (1 / 8 : ℝ) * (3 / 8 : ℝ) ^ 3)) ^ (1 / 4 : ℝ) * (4 / 3 : ℝ) ^ (3 / 4 : ℝ)) ^ 4 <
        ((1 / 64 : ℝ)) ^ 4 := pow_lt_pow_left₀ hlt hpos (by norm_num)
    linarith
  exact le_trans (mul_le_mul_of_nonneg_right hconst (by positivity)) h

/-- **Unconditional Weight-Decay Scaling Law for `T₃` — Upper Bound (`γ^{3/4}`, `[LC]`):**
For all `γ ∈ (0, 1]`, there exists a width-2 parameter `θ` achieving
`(bilinearLoss (t3Tensor ℝ) θ - 1) + γ * paramNormSq θ ≤ 15 * γ^{3/4}`. -/
theorem t3Tensor_weightDecay_upper {γ : ℝ} (hγ0 : 0 < γ) (hγ1 : γ ≤ 1) :
    ∃ θ : BilinearParam 2 2 2 2,
      (bilinearLoss (t3Tensor ℝ) θ - 1) + γ * paramNormSq θ ≤ 15 * γ ^ (3 / 4 : ℝ) := by
  let ε : ℝ := γ ^ (3 / 8 : ℝ)
  have hε_pos : 0 < ε := by positivity
  have hε_le : ε ≤ 1 := Real.rpow_le_one hγ0.le hγ1 (by positivity)
  refine ⟨t3Curve ε, ?_⟩
  rw [t3Curve_loss hε_pos, t3Curve_paramNormSq hε_pos]
  have hε2 : ε ^ 2 = γ ^ (3 / 4 : ℝ) := by
    simp only [ε]; rw [← Real.rpow_natCast, ← Real.rpow_mul hγ0.le]; norm_num
  have hε2_le1 : ε ^ 2 ≤ 1 := by nlinarith
  have hε4 : ε ^ 4 ≤ ε ^ 2 := by nlinarith [sq_nonneg ε]
  have h2pow : (2 : ℝ) ^ (-(2 / 3 : ℝ)) ≤ 1 := by
    rw [Real.rpow_neg (by norm_num : (0 : ℝ) ≤ 2)]
    exact inv_le_one_of_one_le₀ (Real.one_le_rpow (by norm_num) (by positivity))
  have hR2 : γ * (6 * (2 * ε) ^ (-(2 / 3 : ℝ)) * (1 + ε ^ 2)) ≤ 12 * γ ^ (3 / 4 : ℝ) := by
    rw [Real.mul_rpow (by norm_num : (0 : ℝ) ≤ 2) hε_pos.le]
    have hεpow : ε ^ (-(2 / 3 : ℝ)) = γ ^ (-(1 / 4 : ℝ)) := by
      simp only [ε]; rw [← Real.rpow_mul hγ0.le]; norm_num
    rw [hεpow]
    have hγmul : γ * γ ^ (-(1 / 4 : ℝ)) = γ ^ (3 / 4 : ℝ) := by
      calc γ * γ ^ (-(1 / 4 : ℝ))
        = γ ^ (1 : ℝ) * γ ^ (-(1 / 4 : ℝ)) := by rw [Real.rpow_one]
        _ = γ ^ ((1 : ℝ) + -(1 / 4 : ℝ)) := (Real.rpow_add hγ0 _ _).symm
        _ = γ ^ (3 / 4 : ℝ) := by norm_num
    have h1ε2 : 1 + ε ^ 2 ≤ 2 := by nlinarith
    calc γ * (6 * ((2 : ℝ) ^ (-(2 / 3 : ℝ)) * γ ^ (-(1 / 4 : ℝ))) * (1 + ε ^ 2))
      = 6 * (2 : ℝ) ^ (-(2 / 3 : ℝ)) * (1 + ε ^ 2) * (γ * γ ^ (-(1 / 4 : ℝ))) := by ring
      _ = 6 * (2 : ℝ) ^ (-(2 / 3 : ℝ)) * (1 + ε ^ 2) * γ ^ (3 / 4 : ℝ) := by rw [hγmul]
      _ ≤ 6 * 1 * 2 * γ ^ (3 / 4 : ℝ) := by
        apply mul_le_mul_of_nonneg_right _ (by positivity)
        nlinarith [show (0 : ℝ) ≤ (2 : ℝ) ^ (-(2 / 3 : ℝ)) by positivity]
      _ = 12 * γ ^ (3 / 4 : ℝ) := by ring
  linarith

/-- **Unconditional Stretched-Exponential Excess Free Energy for `T₃` — Upper Bound on `Z_{T₃}^{(2)}(β)` (`[LC]`):**
For all `β ≥ 1`,
`Z_{T₃}^{(2)}(β) ≤ (4π)⁶ · exp(-(β + (1/64) · 4^{-3/4} · β^{1/4}))`. -/
theorem t3PartitionZ_two_le_stretchedExp {β : ℝ} (hβ : 1 ≤ β) :
    tensorPartitionZ (t3Tensor ℝ) 2 β ≤
      ENNReal.ofReal ((4 * Real.pi) ^ 6 *
        Real.exp (-(β + ((1 / 64 : ℝ) * (4 : ℝ) ^ (-(3 / 4 : ℝ))) * β ^ (1 / 4 : ℝ)))) := by
  have h := genericPartitionZ_two_le_stretchedExp t3Tensor_genericWallHyp hβ
  have hconst : (1 / 64 : ℝ) ≤ genericWDConst t3Tensor_genericWallHyp := by
    unfold genericWDConst t3Tensor_genericWallHyp
    dsimp only
    apply le_min (le_refl _)
    have hpos : 0 ≤ (4 * (76 * (1 / 8 : ℝ) * (3 / 8 : ℝ) ^ 3)) ^ (1 / 4 : ℝ) *
        (4 / 3 : ℝ) ^ (3 / 4 : ℝ) := by positivity
    have hp1 : ((4 * (76 * (1 / 8 : ℝ) * (3 / 8 : ℝ) ^ 3)) ^ (1 / 4 : ℝ)) ^ 4 =
        4 * (76 * (1 / 8 : ℝ) * (3 / 8 : ℝ) ^ 3) := by
      rw [← Real.rpow_natCast, ← Real.rpow_mul (by positivity)]; norm_num
    have hp2 : ((4 / 3 : ℝ) ^ (3 / 4 : ℝ)) ^ 4 = (4 / 3 : ℝ) ^ 3 := by
      rw [← Real.rpow_natCast, ← Real.rpow_mul (by positivity)]; norm_num
    have hpow4 : ((1 / 64 : ℝ)) ^ 4 ≤
        ((4 * (76 * (1 / 8 : ℝ) * (3 / 8 : ℝ) ^ 3)) ^ (1 / 4 : ℝ) * (4 / 3 : ℝ) ^ (3 / 4 : ℝ)) ^ 4 := by
      rw [mul_pow, hp1, hp2]; norm_num
    by_contra hlt
    push Not at hlt
    have : ((4 * (76 * (1 / 8 : ℝ) * (3 / 8 : ℝ) ^ 3)) ^ (1 / 4 : ℝ) * (4 / 3 : ℝ) ^ (3 / 4 : ℝ)) ^ 4 <
        ((1 / 64 : ℝ)) ^ 4 := pow_lt_pow_left₀ hlt hpos (by norm_num)
    linarith
  refine le_trans h (ENNReal.ofReal_le_ofReal ?_)
  apply mul_le_mul_of_nonneg_left _ (by positivity)
  apply Real.exp_le_exp.mpr
  have h4pos : 0 ≤ (4 : ℝ) ^ (-(3 / 4 : ℝ)) := by positivity
  have hβpos : 0 ≤ β ^ (1 / 4 : ℝ) := by positivity
  have hle : ((1 / 64 : ℝ) * (4 : ℝ) ^ (-(3 / 4 : ℝ))) * β ^ (1 / 4 : ℝ) ≤
      (genericWDConst t3Tensor_genericWallHyp * (4 : ℝ) ^ (-(3 / 4 : ℝ))) * β ^ (1 / 4 : ℝ) :=
    mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right hconst h4pos) hβpos
  linarith

/-- **Unconditional Stretched-Exponential Excess Free Energy for `T₃` — Lower Bound on `Z_{T₃}^{(2)}(β)` (`[LC]`):**
For all `β ≥ 1`,
`(1 / 32)^{12} · exp(-(β + 79 β^{1/4})) ≤ Z_{T₃}^{(2)}(β)`.
Combined with `t3PartitionZ_two_le_stretchedExp`, this certifies the exact generic
stretched-exponential excess free energy `-log Z_{T₃}^{(2)}(β) - β ≍ β^{1/4}` for all `β ≥ 1`. -/
theorem t3PartitionZ_two_ge_stretchedExp {β : ℝ} (hβ : 1 ≤ β) :
    ENNReal.ofReal (((1 / 32 : ℝ) ^ 12) * Real.exp (-(β + 79 * β ^ (1 / 4 : ℝ)))) ≤
      tensorPartitionZ (t3Tensor ℝ) 2 β := by
  have hβ_pos : 0 < β := by linarith
  let ε : ℝ := β ^ (-(3 / 8 : ℝ))
  have hε_pos : 0 < ε := by positivity
  have hε_le : ε ≤ 1 := by
    simp only [ε]; rw [Real.rpow_neg hβ_pos.le]
    exact inv_le_one_of_one_le₀ (Real.one_le_rpow hβ (by positivity))
  let θ₀ : BilinearParam 2 2 2 2 := t3Curve ε
  let ρ : ℝ := (1 / 64 : ℝ) * β ^ (-1 : ℝ)
  have hρ_pos : 0 < ρ := by positivity
  have hρ_le : ρ ≤ 1 := by
    have h1 : β ^ (-1 : ℝ) ≤ 1 := by rw [Real.rpow_neg_one]; exact inv_le_one_of_one_le₀ hβ
    simp only [ρ]; linarith
  have hβ18_ge : 1 ≤ β ^ (1 / 8 : ℝ) := Real.one_le_rpow hβ (by positivity)
  have hβ18_sq : (β ^ (1 / 8 : ℝ)) ^ 2 = β ^ (1 / 4 : ℝ) := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hβ_pos.le]; norm_num
  have hβm34_le1 : β ^ (-(3 / 4 : ℝ)) ≤ 1 := by
    rw [Real.rpow_neg hβ_pos.le]; exact inv_le_one_of_one_le₀ (Real.one_le_rpow hβ (by positivity))
  have hs_le : (2 * ε) ^ (-(1 / 3 : ℝ)) ≤ β ^ (1 / 8 : ℝ) := by
    rw [Real.mul_rpow (by norm_num : (0 : ℝ) ≤ 2) hε_pos.le]
    have h2 : (2 : ℝ) ^ (-(1 / 3 : ℝ)) ≤ 1 := by
      rw [Real.rpow_neg (by norm_num : (0 : ℝ) ≤ 2)]
      exact inv_le_one_of_one_le₀ (Real.one_le_rpow (by norm_num) (by positivity))
    have hεp : ε ^ (-(1 / 3 : ℝ)) = β ^ (1 / 8 : ℝ) := by
      simp only [ε]; rw [← Real.rpow_mul hβ_pos.le]; norm_num
    rw [hεp]; nlinarith [show (0 : ℝ) ≤ β ^ (1 / 8 : ℝ) by positivity]
  have hs_pos : 0 < (2 * ε) ^ (-(1 / 3 : ℝ)) := by positivity
  have hs_abs : |(2 * ε) ^ (-(1 / 3 : ℝ))| ≤ β ^ (1 / 8 : ℝ) := by rwa [abs_of_pos hs_pos]
  have hsε_abs : |(2 * ε) ^ (-(1 / 3 : ℝ)) * ε| ≤ β ^ (1 / 8 : ℝ) := by
    rw [abs_of_pos (mul_pos hs_pos hε_pos)]; nlinarith
  have hθ₀_1 : ∀ i r : Fin 2, |θ₀.1 i r| ≤ β ^ (1 / 8 : ℝ) := fun i r => by
    dsimp only [θ₀, t3Curve, centeredCurve]; split_ifs <;> (try simp only [abs_neg]) <;>
      first | exact hs_abs | exact hsε_abs
  have hθ₀_2 : ∀ j r : Fin 2, |θ₀.2.1 j r| ≤ β ^ (1 / 8 : ℝ) := fun j r => by
    dsimp only [θ₀, t3Curve, centeredCurve]; split_ifs <;> (try simp only [abs_neg]) <;>
      first | exact hs_abs | exact hsε_abs
  have hθ₀_3 : ∀ l r : Fin 2, |θ₀.2.2 l r| ≤ β ^ (1 / 8 : ℝ) := fun l r => by
    dsimp only [θ₀, t3Curve, centeredCurve]; simp only [abs_neg]
    split_ifs <;> (try simp only [abs_neg]) <;> first | exact hs_abs | exact hsε_abs
  have hexp_le : ∀ θ ∈ genParamBox θ₀ ρ,
      β * bilinearLoss (t3Tensor ℝ) θ + (1 / 2 : ℝ) * paramNormSq θ ≤
        β + 31 * β ^ (1 / 4 : ℝ) := by
    intro θ hθ
    rcases hθ with ⟨h1, h2, h3⟩
    have hda : ∀ i r, |θ.1 i r - θ₀.1 i r| ≤ ρ := fun i r => by
      have := Set.mem_Icc.mp (h1 i r); rw [abs_le]; constructor <;> linarith only [this.1, this.2]
    have hdb : ∀ j r, |θ.2.1 j r - θ₀.2.1 j r| ≤ ρ := fun j r => by
      have := Set.mem_Icc.mp (h2 j r); rw [abs_le]; constructor <;> linarith only [this.1, this.2]
    have hdc : ∀ l r, |θ.2.2 l r - θ₀.2.2 l r| ≤ ρ := fun l r => by
      have := Set.mem_Icc.mp (h3 l r); rw [abs_le]; constructor <;> linarith only [this.1, this.2]
    set M := 2 * β ^ (1 / 8 : ℝ)
    have hM_nonneg : 0 ≤ M := by positivity
    have hM2 : M ^ 2 = 4 * β ^ (1 / 4 : ℝ) := by
      simp only [M]; rw [mul_pow, hβ18_sq]; ring
    have hβ18_le_M : β ^ (1 / 8 : ℝ) ≤ M := by simp only [M]; linarith only [hβ18_ge]
    have hθ_1 : ∀ i r, |θ.1 i r| ≤ M := fun i r => by
      calc |θ.1 i r| = |(θ.1 i r - θ₀.1 i r) + θ₀.1 i r| := by ring_nf
        _ ≤ |θ.1 i r - θ₀.1 i r| + |θ₀.1 i r| := abs_add_le _ _
        _ ≤ ρ + β ^ (1 / 8 : ℝ) := add_le_add (hda i r) (hθ₀_1 i r)
        _ ≤ M := by simp only [M]; linarith only [hρ_le, hβ18_ge]
    have hθ_2 : ∀ j r, |θ.2.1 j r| ≤ M := fun j r => by
      calc |θ.2.1 j r| = |(θ.2.1 j r - θ₀.2.1 j r) + θ₀.2.1 j r| := by ring_nf
        _ ≤ |θ.2.1 j r - θ₀.2.1 j r| + |θ₀.2.1 j r| := abs_add_le _ _
        _ ≤ ρ + β ^ (1 / 8 : ℝ) := add_le_add (hdb j r) (hθ₀_2 j r)
        _ ≤ M := by simp only [M]; linarith only [hρ_le, hβ18_ge]
    have hθ_3 : ∀ l r, |θ.2.2 l r| ≤ M := fun l r => by
      calc |θ.2.2 l r| = |(θ.2.2 l r - θ₀.2.2 l r) + θ₀.2.2 l r| := by ring_nf
        _ ≤ |θ.2.2 l r - θ₀.2.2 l r| + |θ₀.2.2 l r| := abs_add_le _ _
        _ ≤ ρ + β ^ (1 / 8 : ℝ) := add_le_add (hdc l r) (hθ₀_3 l r)
        _ ≤ M := by simp only [M]; linarith only [hρ_le, hβ18_ge]
    have hM2ρ : 6 * M ^ 2 * ρ = (3 / 8 : ℝ) * β ^ (-(3 / 4 : ℝ)) := by
      rw [hM2]
      have hmul : β ^ (1 / 4 : ℝ) * β ^ (-1 : ℝ) = β ^ (-(3 / 4 : ℝ)) := by
        rw [← Real.rpow_add hβ_pos]; norm_num
      simp only [ρ]
      calc 6 * (4 * β ^ (1 / 4 : ℝ)) * ((1 / 64 : ℝ) * β ^ (-1 : ℝ))
        = (3 / 8 : ℝ) * (β ^ (1 / 4 : ℝ) * β ^ (-1 : ℝ)) := by ring
        _ = (3 / 8 : ℝ) * β ^ (-(3 / 4 : ℝ)) := by rw [hmul]
    have hTdiff : ∀ i j l : Fin 2,
        |bilinearTensor θ i j l - bilinearTensor θ₀ i j l| ≤ (3 / 8 : ℝ) * β ^ (-(3 / 4 : ℝ)) := by
      intro i j l
      simp only [bilinearTensor, Fin.sum_univ_two]
      have hc0 := gen_cubic_diff_bound (hθ_1 i 0) (hθ_2 j 0) (hθ_3 l 0)
        ((hθ₀_1 i 0).trans hβ18_le_M) ((hθ₀_2 j 0).trans hβ18_le_M) ((hθ₀_3 l 0).trans hβ18_le_M)
        (hda i 0) (hdb j 0) (hdc l 0) hM_nonneg hρ_pos.le
      have hc1 := gen_cubic_diff_bound (hθ_1 i 1) (hθ_2 j 1) (hθ_3 l 1)
        ((hθ₀_1 i 1).trans hβ18_le_M) ((hθ₀_2 j 1).trans hβ18_le_M) ((hθ₀_3 l 1).trans hβ18_le_M)
        (hda i 1) (hdb j 1) (hdc l 1) hM_nonneg hρ_pos.le
      have hsplit : (θ.1 i 0 * θ.2.1 j 0 * θ.2.2 l 0 + θ.1 i 1 * θ.2.1 j 1 * θ.2.2 l 1) -
          (θ₀.1 i 0 * θ₀.2.1 j 0 * θ₀.2.2 l 0 + θ₀.1 i 1 * θ₀.2.1 j 1 * θ₀.2.2 l 1) =
          (θ.1 i 0 * θ.2.1 j 0 * θ.2.2 l 0 - θ₀.1 i 0 * θ₀.2.1 j 0 * θ₀.2.2 l 0) +
          (θ.1 i 1 * θ.2.1 j 1 * θ.2.2 l 1 - θ₀.1 i 1 * θ₀.2.1 j 1 * θ₀.2.2 l 1) := by ring
      rw [hsplit]
      exact (abs_add_le _ _).trans (by linarith only [hc0, hc1, hM2ρ])
    have hβm34_nn : 0 ≤ β ^ (-(3 / 4 : ℝ)) := by positivity
    have hL0 : bilinearLoss (t3Tensor ℝ) θ₀ ≤ 1 + 3 * β ^ (-(3 / 4 : ℝ)) := by
      rw [show θ₀ = t3Curve ε from rfl, t3Curve_loss hε_pos]
      have hε2 : ε ^ 2 = β ^ (-(3 / 4 : ℝ)) := by
        simp only [ε]; rw [← Real.rpow_natCast, ← Real.rpow_mul hβ_pos.le]; norm_num
      have hε2_le1 : ε ^ 2 ≤ 1 := by nlinarith only [hε_pos, hε_le]
      have hε4 : ε ^ 4 ≤ ε ^ 2 := by nlinarith only [hε2_le1, sq_nonneg ε]
      linarith only [hε2, hε4]
    have hL_inc : bilinearLoss (t3Tensor ℝ) θ - bilinearLoss (t3Tensor ℝ) θ₀ ≤
        4 * β ^ (-(3 / 4 : ℝ)) := by
      have hsq_bound : ∀ i j l : Fin 2,
          (bilinearTensor θ i j l - bilinearTensor θ₀ i j l) ^ 2 ≤
            (1 / 4 : ℝ) * β ^ (-(3 / 4 : ℝ)) := fun i j l => by
        have hδ := hTdiff i j l
        have h1' : |bilinearTensor θ i j l - bilinearTensor θ₀ i j l| ^ 2 ≤
            ((3 / 8 : ℝ) * β ^ (-(3 / 4 : ℝ))) ^ 2 := by
          nlinarith only [hδ, abs_nonneg (bilinearTensor θ i j l - bilinearTensor θ₀ i j l), hβm34_nn]
        rw [sq_abs] at h1'
        have h2' : (β ^ (-(3 / 4 : ℝ))) ^ 2 ≤ β ^ (-(3 / 4 : ℝ)) := by
          nlinarith only [hβm34_nn, hβm34_le1]
        nlinarith only [h1', h2']
      have h000 : (bilinearTensor θ 0 0 0 - t3Tensor ℝ 0 0 0) ^ 2 -
          (bilinearTensor θ₀ 0 0 0 - t3Tensor ℝ 0 0 0) ^ 2 ≤ 2 * β ^ (-(3 / 4 : ℝ)) := by
        set δ := bilinearTensor θ 0 0 0 - bilinearTensor θ₀ 0 0 0
        set r := bilinearTensor θ₀ 0 0 0 - t3Tensor ℝ 0 0 0
        have hr : r = -(1 + ε ^ 2) := by
          dsimp only [r, θ₀]; rw [t3Curve_tensor hε_pos]; simp
        have hr_abs : |r| ≤ 2 := by
          rw [hr, abs_neg, abs_of_pos (by positivity)]; nlinarith only [hε_pos, hε_le]
        have hcross : r * δ ≤ 2 * ((3 / 8 : ℝ) * β ^ (-(3 / 4 : ℝ))) := by
          calc r * δ ≤ |r * δ| := le_abs_self _
            _ = |r| * |δ| := abs_mul r δ
            _ ≤ 2 * ((3 / 8 : ℝ) * β ^ (-(3 / 4 : ℝ))) :=
              mul_le_mul hr_abs (hTdiff 0 0 0) (abs_nonneg δ) (by norm_num)
        have hδ2 : δ ^ 2 ≤ (1 / 4 : ℝ) * β ^ (-(3 / 4 : ℝ)) := hsq_bound 0 0 0
        have hid : (bilinearTensor θ 0 0 0 - t3Tensor ℝ 0 0 0) ^ 2 - r ^ 2 = δ ^ 2 + 2 * r * δ := by
          dsimp only [δ, r]; ring
        rw [hid]
        linarith only [hδ2, hcross, hβm34_nn]
      have hother : ∀ i j l : Fin 2, ¬ (i = 0 ∧ j = 0 ∧ l = 0) →
          (bilinearTensor θ i j l - t3Tensor ℝ i j l) ^ 2 -
            (bilinearTensor θ₀ i j l - t3Tensor ℝ i j l) ^ 2 ≤
            (1 / 4 : ℝ) * β ^ (-(3 / 4 : ℝ)) := by
        intro i j l hne
        have h0 : bilinearTensor θ₀ i j l = t3Tensor ℝ i j l := by
          dsimp only [θ₀]; rw [t3Curve_tensor hε_pos]; simp [hne]
        rw [h0, sub_self, zero_pow (by norm_num : 2 ≠ 0), sub_zero]
        simpa [h0] using hsq_bound i j l
      unfold bilinearLoss tensorFrobSq
      simp only [Fin.sum_univ_two]
      linarith only [h000, hβm34_nn,
        hother 0 0 1 (by decide), hother 0 1 0 (by decide), hother 0 1 1 (by decide),
        hother 1 0 0 (by decide), hother 1 0 1 (by decide), hother 1 1 0 (by decide),
        hother 1 1 1 (by decide)]
    have hL_le : bilinearLoss (t3Tensor ℝ) θ ≤ 1 + 7 * β ^ (-(3 / 4 : ℝ)) := by
      linarith only [hL0, hL_inc]
    have hβmul : β * β ^ (-(3 / 4 : ℝ)) = β ^ (1 / 4 : ℝ) := by
      calc β * β ^ (-(3 / 4 : ℝ))
        = β ^ (1 : ℝ) * β ^ (-(3 / 4 : ℝ)) := by rw [Real.rpow_one]
        _ = β ^ ((1 : ℝ) + -(3 / 4 : ℝ)) := (Real.rpow_add hβ_pos _ _).symm
        _ = β ^ (1 / 4 : ℝ) := by norm_num
    have hβL : β * bilinearLoss (t3Tensor ℝ) θ ≤ β + 7 * β ^ (1 / 4 : ℝ) := by
      calc β * bilinearLoss (t3Tensor ℝ) θ
        ≤ β * (1 + 7 * β ^ (-(3 / 4 : ℝ))) := mul_le_mul_of_nonneg_left hL_le hβ_pos.le
        _ = β + 7 * (β * β ^ (-(3 / 4 : ℝ))) := by ring
        _ = β + 7 * β ^ (1 / 4 : ℝ) := by rw [hβmul]
    have hR2_le : paramNormSq θ ≤ 48 * β ^ (1 / 4 : ℝ) := by
      have hsq1 : ∀ i r, (θ.1 i r) ^ 2 ≤ M ^ 2 := fun i r => by
        have h := hθ_1 i r; rw [← sq_abs, sq, sq]; exact mul_le_mul h h (abs_nonneg _) hM_nonneg
      have hsq2 : ∀ j r, (θ.2.1 j r) ^ 2 ≤ M ^ 2 := fun j r => by
        have h := hθ_2 j r; rw [← sq_abs, sq, sq]; exact mul_le_mul h h (abs_nonneg _) hM_nonneg
      have hsq3 : ∀ l r, (θ.2.2 l r) ^ 2 ≤ M ^ 2 := fun l r => by
        have h := hθ_3 l r; rw [← sq_abs, sq, sq]; exact mul_le_mul h h (abs_nonneg _) hM_nonneg
      unfold paramNormSq
      simp only [Fin.sum_univ_two]
      linarith only [hsq1 0 0, hsq1 0 1, hsq1 1 0, hsq1 1 1,
        hsq2 0 0, hsq2 0 1, hsq2 1 0, hsq2 1 1,
        hsq3 0 0, hsq3 0 1, hsq3 1 0, hsq3 1 1, hM2]
    linarith only [hβL, hR2_le]
  have hlow : ENNReal.ofReal (Real.exp (-(β + 31 * β ^ (1 / 4 : ℝ)))) *
      volume (genParamBox θ₀ ρ) ≤ tensorPartitionZ (t3Tensor ℝ) 2 β := by
    rw [(MeasureTheory.setLIntegral_const (genParamBox θ₀ ρ)
      (ENNReal.ofReal (Real.exp (-(β + 31 * β ^ (1 / 4 : ℝ)))))).symm]
    refine le_trans ?_ (MeasureTheory.setLIntegral_le_lintegral (genParamBox θ₀ ρ) _)
    exact MeasureTheory.setLIntegral_mono_ae' (gen_measurableSet_paramBox θ₀ ρ)
      (Filter.Eventually.of_forall fun θ hθ =>
        ENNReal.ofReal_le_ofReal (Real.exp_le_exp.mpr (by linarith [hexp_le θ hθ])))
  rw [gen_volume_paramBox θ₀ hρ_pos.le, ← ENNReal.ofReal_mul (Real.exp_pos _).le] at hlow
  refine le_trans ?_ hlow
  apply ENNReal.ofReal_le_ofReal
  have hlog : Real.log β ≤ 4 * β ^ (1 / 4 : ℝ) := by
    have h1 : Real.log (β ^ (1 / 4 : ℝ)) ≤ β ^ (1 / 4 : ℝ) - 1 :=
      Real.log_le_sub_one_of_pos (by positivity)
    rw [Real.log_rpow hβ_pos] at h1
    linarith
  have hβpow_ge : Real.exp (-(48 * β ^ (1 / 4 : ℝ))) ≤ (β ^ (-1 : ℝ)) ^ 12 := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hβ_pos.le,
      show (-1 : ℝ) * (12 : ℕ) = -12 by norm_num]
    conv_rhs => rw [Real.rpow_def_of_pos hβ_pos]
    apply Real.exp_le_exp.mpr
    linarith
  have h2ρ : (2 * ρ) ^ 12 = (1 / 32 : ℝ) ^ 12 * (β ^ (-1 : ℝ)) ^ 12 := by
    simp only [ρ]
    rw [show 2 * ((1 / 64 : ℝ) * β ^ (-1 : ℝ)) = (1 / 32 : ℝ) * β ^ (-1 : ℝ) by ring, mul_pow]
  rw [show 6 * 2 = 12 by rfl, h2ρ]
  have hrearr : (1 / 32 : ℝ) ^ 12 * Real.exp (-(β + 79 * β ^ (1 / 4 : ℝ))) =
      Real.exp (-(β + 31 * β ^ (1 / 4 : ℝ))) *
        ((1 / 32 : ℝ) ^ 12 * Real.exp (-(48 * β ^ (1 / 4 : ℝ)))) := by
    rw [mul_left_comm, ← Real.exp_add]; congr 2; ring
  rw [hrearr]
  exact mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hβpow_ge (by positivity))
    (Real.exp_pos _).le

end EatYourFiber
