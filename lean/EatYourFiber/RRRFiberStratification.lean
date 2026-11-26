/-
Copyright (c) 2026 Bobby Faber. All rights reserved.
Released under Apache 2.0 license.
-/
module

public import EatYourFiber.AoyagiWatanabe

/-!
# Complete Unconditional `(r*, p, q)` Zero-Loss Fiber Stratification for Reduced-Rank Regression

This module unconditionally Lean-certifies (`[LC]`) the exact local learning coefficient pair
`(λ, m_log)` at **every** point `(B₀, A₀)` of the reduced-rank regression / 2-layer linear network
zero-loss fiber `{B A = M*}` — for **arbitrary** teacher rank `r` (`M*` with arbitrary invertible
block `D ∈ GL(r)`), arbitrary inactive rank `p` of `B₀` (`rank B₀ = r + p`), and arbitrary
inactive rank `q` of `A₀` (`rank A₀ = r + q`), with `p + q ≤ H - r`:

1. `rrr_fiber_exact_pair`: For ALL `r, p, q, m', n', k' : ℕ` (setting `M = r + p + m'`,
   `N = r + q + n'`, `H = r + q + p + k'`) and EVERY invertible `D : Matrix (Fin r) (Fin r) ℝ`,
   the exact loss `‖(B₀ + ΔB)(A₀ + ΔA) - M*‖_F²` on `[-δ, δ]^{M × H} × [-δ, δ]^{H × N}` has
   exact learning pair
   `((M * N - m' * n') / 2 + awLambda m' n' k', awMult m' n' k')`.
   This directly and unconditionally certifies over `ℝ` the full `(r*, a, b)` local RLCT formula
   (MAIS-O70 / Treisman `arc-llc` `rrr_local.py`).
2. **Complete 2D Phase Diagram of `{B A = M*}`:**
   - `rrr_fiber_subcritical_plateau`: Flat 2D plateau `(λ_act + awLambda m n k, awMult m n k)`
     whenever `2 * p < k + m - n` AND `2 * q < k + n - m`.
   - `rrr_fiber_A_supercritical_parabola`: Exact quadratic climb `+ (q - q_c)² / 2` (`m_log = 1`)
     for `k + n ≤ m + 2 * q`.
   - `rrr_fiber_B_supercritical_parabola`: Exact quadratic climb `+ (p - p_c)² / 2` (`m_log = 1`)
     for `k + m ≤ n + 2 * p`.
   - `rrr_supercritical_mutually_exclusive`: `p > p_c` and `q > q_c` can never hold simultaneously.
3. **Irreducible-Component Codimension Characterization (`rrr_fiber_eq_min_component_codim`):**
   Certifies that `2 * λ_excess(p, q)` equals the exact minimum of the component codimension
   `m * α + n * (k - α) - α * (k - α)` over all components `α ∈ [q, k - p]` passing through
   `(p, q)`.
-/

@[expose] public section

namespace EatYourFiber

open MeasureTheory Set Filter Matrix Finset Real
open scoped ENNReal NNReal BigOperators

/-! ### 1. Measure-preserving matrix block splits and basic helpers -/

private local instance (priority := high) instSigmaFiniteVolumeRMatRRR (a b : ℕ) :
    SigmaFinite (volume : Measure (RMat a b)) :=
  inferInstance

private local instance (priority := high) instSFiniteVolumeRMatRRR (a b : ℕ) :
    SFinite (volume : Measure (RMat a b)) :=
  inferInstance

private local instance (priority := high) instSigmaFiniteVolumeProdRRR {α β : Type*}
    [MeasureSpace α] [MeasureSpace β]
    [SigmaFinite (volume : Measure α)] [SigmaFinite (volume : Measure β)] :
    SigmaFinite (volume : Measure (α × β)) :=
  Measure.instSigmaFiniteProdVolume

private local instance (priority := high) instSFiniteVolumeProdRRR {α β : Type*}
    [MeasureSpace α] [MeasureSpace β]
    [SFinite (volume : Measure α)] [SFinite (volume : Measure β)] :
    SFinite (volume : Measure (α × β)) := by
  rw [Measure.volume_eq_prod]; infer_instance

private theorem rrr_matBox_mono {a b : ℕ} {ρ₁ ρ₂ : ℝ} (h : ρ₁ ≤ ρ₂) :
    matBox (a := a) (b := b) ρ₁ ⊆ matBox ρ₂ := fun _ hM =>
  mem_matBox_iff.mpr fun i j => (mem_matBox_iff.mp hM i j).trans h

private theorem rrr_prod_smul_measure {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    (μ : Measure α) (ν : Measure β) [SigmaFinite μ] [SigmaFinite ν] (c : ℝ≥0∞)
    [SigmaFinite (c • ν)] :
    μ.prod (c • ν) = c • (μ.prod ν) := by
  refine Measure.prod_eq (μ := μ) (ν := c • ν) fun s t _ _ => ?_
  simp [Measure.smul_apply, smul_eq_mul, mul_left_comm]

private theorem rrr_sigmaFinite_ofReal_pow_smul {α : Type*} [MeasurableSpace α]
    (μ : Measure α) [SigmaFinite μ] (r : ℝ) (k : ℕ) :
    SigmaFinite (ENNReal.ofReal r ^ k • μ) := by
  have : ENNReal.ofReal r ^ k • μ = (r.toNNReal ^ k : ℝ≥0) • μ := by
    rw [ENNReal.smul_def, ENNReal.ofReal, ENNReal.coe_pow]
  rw [this]
  infer_instance

private theorem rrr_mp_trans {α β γ : Type*} [MeasureSpace α] [MeasureSpace β]
    [MeasureSpace γ] {e₁ : α ≃ᵐ β} {e₂ : β ≃ᵐ γ}
    (h₁ : MeasurePreserving e₁ volume volume) (h₂ : MeasurePreserving e₂ volume volume) :
    MeasurePreserving (e₁.trans e₂) volume volume :=
  h₂.comp h₁

private theorem rrr_mp_prodCongr {α₁ α₂ β₁ β₂ : Type*}
    [MeasureSpace α₁] [MeasureSpace α₂] [MeasureSpace β₁] [MeasureSpace β₂]
    [SFinite (volume : Measure α₁)] [SFinite (volume : Measure α₂)]
    [SFinite (volume : Measure β₁)] [SFinite (volume : Measure β₂)]
    {e₁ : α₁ ≃ᵐ β₁} {e₂ : α₂ ≃ᵐ β₂}
    (h₁ : MeasurePreserving e₁ volume volume) (h₂ : MeasurePreserving e₂ volume volume) :
    MeasurePreserving (e₁.prodCongr e₂) volume volume :=
  h₁.prod h₂

private theorem rrr_mp_prodComm {α β : Type*} [MeasureSpace α] [MeasureSpace β]
    [SFinite (volume : Measure α)] [SFinite (volume : Measure β)] :
    MeasurePreserving (MeasurableEquiv.prodComm : α × β ≃ᵐ β × α) volume volume :=
  Measure.measurePreserving_swap

private theorem rrr_matMul_entry_abs_le {a b c : ℕ}
    (X : Matrix (Fin a) (Fin b) ℝ) (Y : Matrix (Fin b) (Fin c) ℝ)
    {Bx By : ℝ} (hBx : 0 ≤ Bx) (hX : ∀ i r, |X i r| ≤ Bx) (hY : ∀ r j, |Y r j| ≤ By)
    (i : Fin a) (j : Fin c) :
    |(X * Y) i j| ≤ (b : ℝ) * Bx * By := by
  calc |(X * Y) i j|
      = |∑ r : Fin b, X i r * Y r j| := by simp only [Matrix.mul_apply]
    _ ≤ ∑ r : Fin b, |X i r * Y r j| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ r : Fin b, |X i r| * |Y r j| := by simp_rw [abs_mul]
    _ ≤ ∑ _ : Fin b, Bx * By :=
        Finset.sum_le_sum fun r _ => mul_le_mul (hX i r) (hY r j) (abs_nonneg _) hBx
    _ = (b : ℝ) * Bx * By := by simp [Finset.sum_const, mul_assoc]

/-- Vertical concatenation of `X₁ : Matrix (Fin a₁) (Fin b) ℝ` and `X₂ : Matrix (Fin a₂) (Fin b) ℝ`. -/
def vcatMat {a₁ a₂ b : ℕ} (X₁ : Matrix (Fin a₁) (Fin b) ℝ) (X₂ : Matrix (Fin a₂) (Fin b) ℝ) :
    Matrix (Fin (a₁ + a₂)) (Fin b) ℝ :=
  Matrix.of fun i j => if h : i.1 < a₁ then X₁ ⟨i.1, h⟩ j else X₂ ⟨i.1 - a₁, by omega⟩ j

/-- Horizontal concatenation of `X₁ : Matrix (Fin a) (Fin b₁) ℝ` and `X₂ : Matrix (Fin a) (Fin b₂) ℝ`. -/
def hcatMat {a b₁ b₂ : ℕ} (X₁ : Matrix (Fin a) (Fin b₁) ℝ) (X₂ : Matrix (Fin a) (Fin b₂) ℝ) :
    Matrix (Fin a) (Fin (b₁ + b₂)) ℝ :=
  Matrix.of fun i j => if h : j.1 < b₁ then X₁ i ⟨j.1, h⟩ else X₂ i ⟨j.1 - b₁, by omega⟩

private theorem vcatMat_castAdd {a₁ a₂ b : ℕ}
    (X₁ : Matrix (Fin a₁) (Fin b) ℝ) (X₂ : Matrix (Fin a₂) (Fin b) ℝ)
    (i : Fin a₁) (j : Fin b) :
    vcatMat X₁ X₂ (Fin.castAdd a₂ i) j = X₁ i j := by
  simp [vcatMat, Fin.castAdd, i.2]

private theorem vcatMat_natAdd {a₁ a₂ b : ℕ}
    (X₁ : Matrix (Fin a₁) (Fin b) ℝ) (X₂ : Matrix (Fin a₂) (Fin b) ℝ)
    (i : Fin a₂) (j : Fin b) :
    vcatMat X₁ X₂ (Fin.natAdd a₁ i) j = X₂ i j := by
  have h : ¬ (a₁ + i.1 < a₁) := by omega
  simp [vcatMat, Fin.natAdd, h]

private theorem hcatMat_castAdd {a b₁ b₂ : ℕ}
    (X₁ : Matrix (Fin a) (Fin b₁) ℝ) (X₂ : Matrix (Fin a) (Fin b₂) ℝ)
    (i : Fin a) (j : Fin b₁) :
    hcatMat X₁ X₂ i (Fin.castAdd b₂ j) = X₁ i j := by
  simp [hcatMat, Fin.castAdd, j.2]

private theorem hcatMat_natAdd {a b₁ b₂ : ℕ}
    (X₁ : Matrix (Fin a) (Fin b₁) ℝ) (X₂ : Matrix (Fin a) (Fin b₂) ℝ)
    (i : Fin a) (j : Fin b₂) :
    hcatMat X₁ X₂ i (Fin.natAdd b₁ j) = X₂ i j := by
  have h : ¬ (b₁ + j.1 < b₁) := by omega
  simp [hcatMat, Fin.natAdd, h]

private theorem vcatMat_add {a₁ a₂ b : ℕ}
    (X₁ Y₁ : Matrix (Fin a₁) (Fin b) ℝ) (X₂ Y₂ : Matrix (Fin a₂) (Fin b) ℝ) :
    vcatMat X₁ X₂ + vcatMat Y₁ Y₂ = vcatMat (X₁ + Y₁) (X₂ + Y₂) := by
  ext i j
  by_cases hi : i.1 < a₁
  · have hieq : i = Fin.castAdd a₂ ⟨i.1, hi⟩ := by ext; rfl
    rw [hieq, Matrix.add_apply, vcatMat_castAdd, vcatMat_castAdd, vcatMat_castAdd, Matrix.add_apply]
  · have hieq : i = Fin.natAdd a₁ ⟨i.1 - a₁, by omega⟩ := by ext; dsimp; omega
    rw [hieq, Matrix.add_apply, vcatMat_natAdd, vcatMat_natAdd, vcatMat_natAdd, Matrix.add_apply]

private theorem hcatMat_add {a b₁ b₂ : ℕ}
    (X₁ Y₁ : Matrix (Fin a) (Fin b₁) ℝ) (X₂ Y₂ : Matrix (Fin a) (Fin b₂) ℝ) :
    hcatMat X₁ X₂ + hcatMat Y₁ Y₂ = hcatMat (X₁ + Y₁) (X₂ + Y₂) := by
  ext i j
  by_cases hj : j.1 < b₁
  · have hjeq : j = Fin.castAdd b₂ ⟨j.1, hj⟩ := by ext; rfl
    rw [hjeq, Matrix.add_apply, hcatMat_castAdd, hcatMat_castAdd, hcatMat_castAdd, Matrix.add_apply]
  · have hjeq : j = Fin.natAdd b₁ ⟨j.1 - b₁, by omega⟩ := by ext; dsimp; omega
    rw [hjeq, Matrix.add_apply, hcatMat_natAdd, hcatMat_natAdd, hcatMat_natAdd, Matrix.add_apply]

private theorem hcatMat_sub {a b₁ b₂ : ℕ}
    (X₁ Y₁ : Matrix (Fin a) (Fin b₁) ℝ) (X₂ Y₂ : Matrix (Fin a) (Fin b₂) ℝ) :
    hcatMat X₁ X₂ - hcatMat Y₁ Y₂ = hcatMat (X₁ - Y₁) (X₂ - Y₂) := by
  ext i j
  by_cases hj : j.1 < b₁
  · have hjeq : j = Fin.castAdd b₂ ⟨j.1, hj⟩ := by ext; rfl
    rw [hjeq, Matrix.sub_apply, hcatMat_castAdd, hcatMat_castAdd, hcatMat_castAdd, Matrix.sub_apply]
  · have hjeq : j = Fin.natAdd b₁ ⟨j.1 - b₁, by omega⟩ := by ext; dsimp; omega
    rw [hjeq, Matrix.sub_apply, hcatMat_natAdd, hcatMat_natAdd, hcatMat_natAdd, Matrix.sub_apply]

private theorem frobSq_vcatMat {a₁ a₂ b : ℕ}
    (X₁ : Matrix (Fin a₁) (Fin b) ℝ) (X₂ : Matrix (Fin a₂) (Fin b) ℝ) :
    frobSq (vcatMat X₁ X₂) = frobSq X₁ + frobSq X₂ := by
  unfold frobSq
  rw [Fin.sum_univ_add]
  congr 1
  · refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
    rw [vcatMat_castAdd]
  · refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
    rw [vcatMat_natAdd]

private theorem frobSq_hcatMat {a b₁ b₂ : ℕ}
    (X₁ : Matrix (Fin a) (Fin b₁) ℝ) (X₂ : Matrix (Fin a) (Fin b₂) ℝ) :
    frobSq (hcatMat X₁ X₂) = frobSq X₁ + frobSq X₂ := by
  unfold frobSq
  simp_rw [Fin.sum_univ_add, Finset.sum_add_distrib]
  congr 1
  · refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
    rw [hcatMat_castAdd]
  · refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
    rw [hcatMat_natAdd]

private theorem frobSq_transpose_rrr {a b : ℕ} (X : Matrix (Fin a) (Fin b) ℝ) :
    frobSq Xᵀ = frobSq X := by
  unfold frobSq
  simp only [Matrix.transpose_apply]
  exact Finset.sum_comm

private theorem hcatMat_mul_vcatMat {a b₁ b₂ c : ℕ}
    (X₁ : Matrix (Fin a) (Fin b₁) ℝ) (X₂ : Matrix (Fin a) (Fin b₂) ℝ)
    (Y₁ : Matrix (Fin b₁) (Fin c) ℝ) (Y₂ : Matrix (Fin b₂) (Fin c) ℝ) :
    hcatMat X₁ X₂ * vcatMat Y₁ Y₂ = X₁ * Y₁ + X₂ * Y₂ := by
  ext i j
  simp only [Matrix.mul_apply, Matrix.add_apply]
  rw [Fin.sum_univ_add]
  congr 1
  · refine Finset.sum_congr rfl fun r _ => ?_
    rw [hcatMat_castAdd, vcatMat_castAdd]
  · refine Finset.sum_congr rfl fun r _ => ?_
    rw [hcatMat_natAdd, vcatMat_natAdd]

private theorem mul_hcatMat {a b c₁ c₂ : ℕ}
    (X : Matrix (Fin a) (Fin b) ℝ)
    (Y₁ : Matrix (Fin b) (Fin c₁) ℝ) (Y₂ : Matrix (Fin b) (Fin c₂) ℝ) :
    X * hcatMat Y₁ Y₂ = hcatMat (X * Y₁) (X * Y₂) := by
  ext i j
  by_cases hj : j.1 < c₁
  · have hjeq : j = Fin.castAdd c₂ ⟨j.1, hj⟩ := by ext; rfl
    rw [hjeq, hcatMat_castAdd]
    simp only [Matrix.mul_apply]
    refine Finset.sum_congr rfl fun r _ => ?_
    rw [hcatMat_castAdd]
  · have hjeq : j = Fin.natAdd c₁ ⟨j.1 - c₁, by omega⟩ := by ext; dsimp; omega
    rw [hjeq, hcatMat_natAdd]
    simp only [Matrix.mul_apply]
    refine Finset.sum_congr rfl fun r _ => ?_
    rw [hcatMat_natAdd]

private theorem vcatMat_mul {a₁ a₂ b c : ℕ}
    (X₁ : Matrix (Fin a₁) (Fin b) ℝ) (X₂ : Matrix (Fin a₂) (Fin b) ℝ)
    (Y : Matrix (Fin b) (Fin c) ℝ) :
    vcatMat X₁ X₂ * Y = vcatMat (X₁ * Y) (X₂ * Y) := by
  ext i j
  by_cases hi : i.1 < a₁
  · have hieq : i = Fin.castAdd a₂ ⟨i.1, hi⟩ := by ext; rfl
    rw [hieq, vcatMat_castAdd]
    simp only [Matrix.mul_apply]
    refine Finset.sum_congr rfl fun r _ => ?_
    rw [vcatMat_castAdd]
  · have hieq : i = Fin.natAdd a₁ ⟨i.1 - a₁, by omega⟩ := by ext; dsimp; omega
    rw [hieq, vcatMat_natAdd]
    simp only [Matrix.mul_apply]
    refine Finset.sum_congr rfl fun r _ => ?_
    rw [vcatMat_natAdd]

/-- Transpose measurable equivalence on `RMat a b ≃ᵐ RMat b a`. -/
private def rrrTransposeEquiv (a b : ℕ) : RMat a b ≃ᵐ RMat b a where
  toFun M j i := M i j
  invFun M i j := M j i
  left_inv _ := rfl
  right_inv _ := rfl
  measurable_toFun := Measurable.of_eval fun j => Measurable.of_eval fun i =>
    (measurable_pi_apply j).comp (measurable_pi_apply i)
  measurable_invFun := Measurable.of_eval fun i => Measurable.of_eval fun j =>
    (measurable_pi_apply i).comp (measurable_pi_apply j)

private theorem rrr_mp_transpose (a b : ℕ) :
    MeasurePreserving (rrrTransposeEquiv a b) volume volume :=
  measurePreserving_transpose a b

private theorem rrr_transpose_mem_matBox {a b : ℕ} {δ : ℝ} {M : RMat a b} :
    rrrTransposeEquiv a b M ∈ matBox δ ↔ M ∈ matBox δ := by
  simp only [mem_matBox_iff, rrrTransposeEquiv, MeasurableEquiv.coe_mk, Equiv.coe_fn_mk]
  exact ⟨fun h i j => h j i, fun h j i => h i j⟩

/-- Left multiplication `X ↦ M * X` on `RMat d H` scales Lebesgue measure by `|det M|^{-H}`. -/
private theorem rrr_map_mul_left_volume {d H : ℕ} (M : Matrix (Fin d) (Fin d) ℝ) (hM : M.det ≠ 0) :
    Measure.map (fun X : RMat d H => toRMat (M * Matrix.of X)) volume =
      ENNReal.ofReal |(M.det)⁻¹| ^ H • volume := by
  have htr : (fun X : RMat d H => toRMat (M * Matrix.of X)) =
      (rrrTransposeEquiv H d) ∘ (fun Y : RMat H d => toRMat (Matrix.of Y * Mᵀ)) ∘
        (rrrTransposeEquiv d H) := by
    funext X i r
    simp only [Function.comp_apply, rrrTransposeEquiv, MeasurableEquiv.coe_mk, Equiv.coe_fn_mk,
      toRMat_apply, Matrix.mul_apply, Matrix.of_apply, Matrix.transpose_apply]
    refine Finset.sum_congr rfl fun j _ => mul_comm _ _
  have hdetT : Mᵀ.det ≠ 0 := by rwa [Matrix.det_transpose]
  have hmeas_mul : Measurable (fun Y : RMat H d => toRMat (Matrix.of Y * Mᵀ)) := by
    refine Measurable.of_eval fun r => Measurable.of_eval fun i => ?_
    simp only [toRMat_apply, Matrix.mul_apply, Matrix.of_apply]
    fun_prop
  rw [htr, ← Measure.map_map (rrrTransposeEquiv H d).measurable
      (hmeas_mul.comp (rrrTransposeEquiv d H).measurable),
    ← Measure.map_map hmeas_mul (rrrTransposeEquiv d H).measurable,
    (rrr_mp_transpose d H).map_eq,
    map_mul_right_volume (m := H) Mᵀ hdetT, Matrix.det_transpose,
    Measure.map_smul, (rrr_mp_transpose H d).map_eq]
  · exact (rrrTransposeEquiv H d).measurable.aemeasurable

/-- Split rows of `RMat (a₁ + a₂) b` into `RMat a₁ b × RMat a₂ b`. -/
private def rrrSplitRows (a₁ a₂ b : ℕ) : RMat (a₁ + a₂) b ≃ᵐ RMat a₁ b × RMat a₂ b where
  toFun M := (fun i => M (Fin.castAdd a₂ i), fun i => M (Fin.natAdd a₁ i))
  invFun p i := if h : i.1 < a₁ then p.1 ⟨i.1, h⟩ else p.2 ⟨i.1 - a₁, by omega⟩
  left_inv M := by
    funext i j
    by_cases h : i.1 < a₁
    · simp only [h, ↓reduceDIte]; rfl
    · simp only [h, ↓reduceDIte]; congr 1; ext; dsimp; omega
  right_inv p := by
    ext i j
    · simp [Fin.castAdd, i.2]
    · have h : ¬ (a₁ + i.1 < a₁) := by omega
      simp [Fin.natAdd, h]
  measurable_toFun := Measurable.prodMk
    (Measurable.of_eval fun i => measurable_pi_apply (Fin.castAdd a₂ i))
    (Measurable.of_eval fun i => measurable_pi_apply (Fin.natAdd a₁ i))
  measurable_invFun := Measurable.of_eval fun i => by
    change Measurable (fun p : RMat a₁ b × RMat a₂ b =>
        if h' : i.1 < a₁ then p.1 ⟨i.1, h'⟩ else p.2 ⟨i.1 - a₁, by omega⟩)
    by_cases h : i.1 < a₁
    · have heq : (fun p : RMat a₁ b × RMat a₂ b =>
          if h' : i.1 < a₁ then p.1 ⟨i.1, h'⟩ else p.2 ⟨i.1 - a₁, by omega⟩) =
          fun p => p.1 ⟨i.1, h⟩ := by funext p; simp only [h, ↓reduceDIte]
      rw [heq]; exact (measurable_pi_apply (⟨i.1, h⟩ : Fin a₁)).comp measurable_fst
    · have heq : (fun p : RMat a₁ b × RMat a₂ b =>
          if h' : i.1 < a₁ then p.1 ⟨i.1, h'⟩ else p.2 ⟨i.1 - a₁, by omega⟩) =
          fun p => p.2 ⟨i.1 - a₁, by omega⟩ := by funext p; simp only [h, ↓reduceDIte]
      rw [heq]; exact (measurable_pi_apply (⟨i.1 - a₁, by omega⟩ : Fin a₂)).comp measurable_snd

private theorem rrr_mp_splitRows (a₁ a₂ b : ℕ) :
    MeasurePreserving (rrrSplitRows a₁ a₂ b) volume volume := by
  have hsymm : MeasurePreserving (rrrSplitRows a₁ a₂ b).symm volume volume := by
    refine ⟨(rrrSplitRows a₁ a₂ b).symm.measurable, ?_⟩
    symm
    refine Measure.pi_eq fun s hs => ?_
    rw [Measure.map_apply (rrrSplitRows a₁ a₂ b).symm.measurable (MeasurableSet.univ_pi hs)]
    have hpre : (rrrSplitRows a₁ a₂ b).symm ⁻¹' (Set.univ.pi s) =
        (Set.univ.pi fun i : Fin a₁ => s (Fin.castAdd a₂ i)) ×ˢ
        (Set.univ.pi fun i : Fin a₂ => s (Fin.natAdd a₁ i)) := by
      ext ⟨M₁, M₂⟩
      simp only [Set.mem_preimage, Set.mem_pi, Set.mem_univ, forall_const, Set.mem_prod]
      change (∀ i : Fin (a₁ + a₂),
          (if h : i.1 < a₁ then M₁ ⟨i.1, h⟩ else M₂ ⟨i.1 - a₁, by omega⟩) ∈ s i) ↔ _
      constructor
      · intro h
        refine ⟨fun i => ?_, fun i => ?_⟩
        · simpa [Fin.castAdd, i.2] using h (Fin.castAdd a₂ i)
        · have hnot : ¬ (a₁ + i.1 < a₁) := by omega
          simpa [Fin.natAdd, hnot] using h (Fin.natAdd a₁ i)
      · rintro ⟨h₁, h₂⟩ i
        by_cases h : i.1 < a₁
        · simp only [h, ↓reduceDIte]; exact h₁ ⟨i.1, h⟩
        · simp only [h, ↓reduceDIte]
          have heq : Fin.natAdd a₁ ⟨i.1 - a₁, by omega⟩ = i := by ext; dsimp; omega
          simpa [heq] using h₂ ⟨i.1 - a₁, by omega⟩
    rw [hpre, Measure.volume_eq_prod, Measure.prod_prod]
    change (Measure.pi fun _ : Fin a₁ => (volume : Measure (Fin b → ℝ))) _ *
      (Measure.pi fun _ : Fin a₂ => (volume : Measure (Fin b → ℝ))) _ = _
    rw [Measure.pi_pi, Measure.pi_pi, Fin.prod_univ_add (fun i => volume (s i))]
  exact hsymm.symm

private theorem rrr_splitRows_mem_matBox {a₁ a₂ b : ℕ} {δ : ℝ} {M : RMat (a₁ + a₂) b} :
    rrrSplitRows a₁ a₂ b M ∈ matBox δ ×ˢ matBox δ ↔ M ∈ matBox δ := by
  simp only [Set.mem_prod, mem_matBox_iff, rrrSplitRows, MeasurableEquiv.coe_mk, Equiv.coe_fn_mk]
  constructor
  · rintro ⟨h₁, h₂⟩ i j
    by_cases h : i.1 < a₁
    · exact h₁ ⟨i.1, h⟩ j
    · have heq : Fin.natAdd a₁ ⟨i.1 - a₁, by omega⟩ = i := by ext; dsimp; omega
      simpa [heq] using h₂ ⟨i.1 - a₁, by omega⟩ j
  · intro h
    exact ⟨fun i j => h (Fin.castAdd a₂ i) j, fun i j => h (Fin.natAdd a₁ i) j⟩

private theorem rrrSplitRows_symm_eq_vcatMat {a₁ a₂ b : ℕ} (M₁ : RMat a₁ b) (M₂ : RMat a₂ b) :
    Matrix.of ((rrrSplitRows a₁ a₂ b).symm (M₁, M₂)) = vcatMat (Matrix.of M₁) (Matrix.of M₂) := by
  ext i j
  change (if h : i.1 < a₁ then M₁ ⟨i.1, h⟩ else M₂ ⟨i.1 - a₁, by omega⟩) j = _
  simp only [vcatMat, Matrix.of_apply]
  split_ifs <;> rfl

/-- Split columns of `RMat a (b₁ + b₂)` into `RMat a b₁ × RMat a b₂`. -/
private def rrrSplitCols (a b₁ b₂ : ℕ) : RMat a (b₁ + b₂) ≃ᵐ RMat a b₁ × RMat a b₂ :=
  (rrrTransposeEquiv a (b₁ + b₂)).trans
    ((rrrSplitRows b₁ b₂ a).trans ((rrrTransposeEquiv b₁ a).prodCongr (rrrTransposeEquiv b₂ a)))

private theorem rrr_mp_splitCols (a b₁ b₂ : ℕ) :
    MeasurePreserving (rrrSplitCols a b₁ b₂) volume volume :=
  ((rrr_mp_transpose b₁ a).prod (rrr_mp_transpose b₂ a)).comp
    ((rrr_mp_splitRows b₁ b₂ a).comp (rrr_mp_transpose a (b₁ + b₂)))

private theorem rrr_splitCols_mem_matBox {a b₁ b₂ : ℕ} {δ : ℝ} {M : RMat a (b₁ + b₂)} :
    rrrSplitCols a b₁ b₂ M ∈ matBox δ ×ˢ matBox δ ↔ M ∈ matBox δ := by
  change (rrrTransposeEquiv b₁ a (rrrSplitRows b₁ b₂ a (rrrTransposeEquiv a (b₁ + b₂) M)).1,
    rrrTransposeEquiv b₂ a (rrrSplitRows b₁ b₂ a (rrrTransposeEquiv a (b₁ + b₂) M)).2) ∈
      matBox δ ×ˢ matBox δ ↔ M ∈ matBox δ
  simp only [Set.mem_prod, rrr_transpose_mem_matBox]
  exact (rrr_splitRows_mem_matBox (M := rrrTransposeEquiv a (b₁ + b₂) M)).trans
    rrr_transpose_mem_matBox

private theorem rrrSplitCols_symm_eq_hcatMat {a b₁ b₂ : ℕ} (M₁ : RMat a b₁) (M₂ : RMat a b₂) :
    Matrix.of ((rrrSplitCols a b₁ b₂).symm (M₁, M₂)) = hcatMat (Matrix.of M₁) (Matrix.of M₂) := by
  ext i j
  by_cases hj : j.1 < b₁
  · have hjeq : j = Fin.castAdd b₂ ⟨j.1, hj⟩ := by ext; rfl
    rw [hjeq, hcatMat_castAdd]
    have h := congrFun (congrFun (congrArg Prod.fst ((rrrSplitCols a b₁ b₂).apply_symm_apply (M₁, M₂))) i) ⟨j.1, hj⟩
    exact h
  · have hjeq : j = Fin.natAdd b₁ ⟨j.1 - b₁, by omega⟩ := by ext; dsimp; omega
    rw [hjeq, hcatMat_natAdd]
    have h := congrFun (congrFun (congrArg Prod.snd ((rrrSplitCols a b₁ b₂).apply_symm_apply (M₁, M₂))) i) ⟨j.1 - b₁, by omega⟩
    exact h

private theorem rrr_Z_eq_of_mp {α β : Type*} [MeasureSpace α] [MeasureSpace β]
    (e : α ≃ᵐ β) (he : MeasurePreserving e volume volume)
    (S₁ : Set α) (S₂ : Set β) (hS₁ : MeasurableSet S₁) (hS₂ : MeasurableSet S₂)
    (hS : ∀ x, e x ∈ S₂ ↔ x ∈ S₁)
    (L₁ : α → ℝ) (L₂ : β → ℝ) (hL₂ : Measurable L₂) (hL : ∀ x, L₁ x = L₂ (e x)) (N : ℝ) :
    SLT.Z (volume.restrict S₁) L₁ N = SLT.Z (volume.restrict S₂) L₂ N := by
  unfold SLT.Z
  have hg : Measurable fun y : β => ENNReal.ofReal (Real.exp (-(N * L₂ y))) := by fun_prop
  rw [← lintegral_indicator hS₁, ← lintegral_indicator hS₂,
    ← he.lintegral_comp (hg.indicator hS₂)]
  refine lintegral_congr fun x => ?_
  by_cases hx : x ∈ S₁
  · rw [Set.indicator_of_mem hx, Set.indicator_of_mem ((hS x).mpr hx), hL x]
  · rw [Set.indicator_of_notMem hx, Set.indicator_of_notMem (fun h => hx ((hS x).mp h))]

/-! ### 2. Canonical `(r, p, q)` fiber point `(B₀, A₀)` and exact RRR fiber loss -/

/-- The canonical left base block `B₀_L = M*_L ∈ ℝ^{(r + (p + m')) × (r + q)}`, placing
`D : Matrix (Fin r) (Fin r) ℝ` in the top-left `r × r` corner and `0` elsewhere. -/
def rrrBaseBL {r p q m' : ℕ} (D : Matrix (Fin r) (Fin r) ℝ) :
    Matrix (Fin (r + (p + m'))) (Fin (r + q)) ℝ :=
  vcatMat (hcatMat D (0 : Matrix (Fin r) (Fin q) ℝ))
          (0 : Matrix (Fin (p + m')) (Fin (r + q)) ℝ)

/-- The canonical lower-right base block `B₀_lo ∈ ℝ^{(p + m') × (p + k')}`, placing `I_p`
in the top-left `p × p` corner and `0` elsewhere. -/
def rrrBaseBlo (p m' k' : ℕ) : Matrix (Fin (p + m')) (Fin (p + k')) ℝ :=
  vcatMat (hcatMat (1 : Matrix (Fin p) (Fin p) ℝ) (0 : Matrix (Fin p) (Fin k') ℝ))
          (0 : Matrix (Fin m') (Fin (p + k')) ℝ)

/-- The canonical right base block `B₀_R ∈ ℝ^{(r + (p + m')) × (p + k')}`. -/
def rrrBaseBR (r p m' k' : ℕ) : Matrix (Fin (r + (p + m'))) (Fin (p + k')) ℝ :=
  vcatMat (0 : Matrix (Fin r) (Fin (p + k')) ℝ) (rrrBaseBlo p m' k')

/-- Canonical zero-loss fiber matrix `B₀ ∈ ℝ^{M × H}` of rank `r + p`
(with `M = r + (p + m')`, `H = (r + q) + (p + k')`). -/
def rrrBaseB {r p q m' k' : ℕ} (D : Matrix (Fin r) (Fin r) ℝ) :
    Matrix (Fin (r + (p + m'))) (Fin ((r + q) + (p + k'))) ℝ :=
  hcatMat (rrrBaseBL (p := p) (q := q) (m' := m') D) (rrrBaseBR r p m' k')

/-- Canonical zero-loss fiber matrix `A₀ ∈ ℝ^{H × N}` of rank `r + q`
(with `H = (r + q) + (p + k')`, `N = (r + q) + n'`). -/
def rrrBaseA (r p q n' k' : ℕ) :
    Matrix (Fin ((r + q) + (p + k'))) (Fin ((r + q) + n')) ℝ :=
  vcatMat (hcatMat (1 : Matrix (Fin (r + q)) (Fin (r + q)) ℝ) (0 : Matrix (Fin (r + q)) (Fin n') ℝ))
          (0 : Matrix (Fin (p + k')) (Fin ((r + q) + n')) ℝ)

/-- Canonical target matrix `M* = B₀ * A₀ ∈ ℝ^{M × N}` of rank `r`. -/
def rrrTarget {r p q m' n' : ℕ} (D : Matrix (Fin r) (Fin r) ℝ) :
    Matrix (Fin (r + (p + m'))) (Fin ((r + q) + n')) ℝ :=
  hcatMat (rrrBaseBL (p := p) (q := q) (m' := m') D) (0 : Matrix (Fin (r + (p + m'))) (Fin n') ℝ)

/-- Verification that `(B₀, A₀)` lies exactly on the zero-loss fiber `{B A = M*}`. -/
theorem rrrBaseB_mul_rrrBaseA {r p q m' n' k' : ℕ} (D : Matrix (Fin r) (Fin r) ℝ) :
    rrrBaseB (p := p) (q := q) (m' := m') (k' := k') D * rrrBaseA r p q n' k' =
      rrrTarget (p := p) (q := q) (m' := m') (n' := n') D := by
  unfold rrrBaseB rrrBaseA rrrTarget
  rw [hcatMat_mul_vcatMat, Matrix.mul_zero, add_zero, mul_hcatMat, Matrix.mul_one, Matrix.mul_zero]

/-- **Exact Reduced-Rank Regression Fiber Loss** around `(B₀, A₀)` of ranks `(r + p, r + q)`:
`K(ΔB, ΔA) = ‖(B₀ + ΔB) (A₀ + ΔA) - M*‖_F²`. -/
def rrrFiberLoss {r p q m' n' k' : ℕ} (D : Matrix (Fin r) (Fin r) ℝ)
    (θ : RMat (r + (p + m')) ((r + q) + (p + k')) ×
         RMat ((r + q) + (p + k')) ((r + q) + n')) : ℝ :=
  frobSq ((rrrBaseB D + Matrix.of θ.1) * (rrrBaseA r p q n' k' + Matrix.of θ.2) - rrrTarget D)

/-! ### 3. Symmetry of `awLambda` / `awMult` and transpose-LoRA exact pair -/

theorem awLambda_comm (m n k : ℕ) : awLambda m n k = awLambda n m k := by
  unfold awLambda awLambdaEven
  have hpar : Even (m + n + k) ↔ Even (n + m + k) := by rw [Nat.add_comm m n]
  by_cases c1 : m + n ≤ k
  · have c1' : n + m ≤ k := by omega
    simp only [c1, c1', ↓reduceIte]; ring
  · have c1' : ¬ (n + m ≤ k) := by omega
    by_cases c2 : k + n ≤ m
    · by_cases c3 : k + m ≤ n
      · have hk0 : k = 0 := by omega
        subst hk0; simp only [c1, c1', c2, c3, ↓reduceIte]; norm_num
      · simp only [c1, c1', c2, c3, ↓reduceIte]
    · by_cases c3 : k + m ≤ n
      · simp only [c1, c1', c2, c3, ↓reduceIte]
      · simp only [c1, c1', c2, c3, ↓reduceIte, hpar]
        split_ifs <;> ring

theorem awMult_comm (m n k : ℕ) : awMult m n k = awMult n m k := by
  unfold awMult
  have hpar : Even (m + n + k) ↔ Even (n + m + k) := by rw [Nat.add_comm m n]
  by_cases c1 : m + n ≤ k
  · have c1' : n + m ≤ k := by omega
    simp only [c1, c1', ↓reduceIte]
  · have c1' : ¬ (n + m ≤ k) := by omega
    by_cases c2 : k + n ≤ m
    · by_cases c3 : k + m ≤ n
      · simp only [c1, c1', c2, c3, ↓reduceIte]
      · simp only [c1, c1', c2, c3, ↓reduceIte]
    · by_cases c3 : k + m ≤ n
      · simp only [c1, c1', c2, c3, ↓reduceIte]
      · simp only [c1, c1', c2, c3, ↓reduceIte, hpar]

/-- The lower-right transpose-LoRA loss `‖(B₀_lo + ΔB₂_lo) * Ã₂₂‖_F²` on
`RMat (p + m') (p + k') × RMat (p + k') n'`. -/
private def rrrLoLoss {p m' n' k' : ℕ}
    (w : RMat (p + m') (p + k') × RMat (p + k') n') : ℝ :=
  frobSq ((rrrBaseBlo p m' k' + Matrix.of w.1) * Matrix.of w.2)

private theorem measurable_rrrLoLoss {p m' n' k' : ℕ} :
    Measurable (rrrLoLoss (p := p) (m' := m') (n' := n') (k' := k')) := by
  unfold rrrLoLoss frobSq
  simp only [Matrix.mul_apply, Matrix.add_apply, Matrix.of_apply]
  fun_prop

/-- Exact learning pair for `rrrLoLoss` via transpose reduction to `lora_fiber_exact_pair`. -/
private theorem rrrLoLoss_exact_pair (p m' n' k' : ℕ) :
    ∃ δ₀ : ℝ, 0 < δ₀ ∧ ∀ δ : ℝ, 0 < δ → δ ≤ δ₀ →
      SLT.HasLaplacePair
        (SLT.Z (volume.restrict (matBox (a := p + m') (b := p + k') δ ×ˢ
                                 matBox (a := p + k') (b := n') δ))
          (rrrLoLoss (p := p) (m' := m') (n' := n') (k' := k')))
        ((p : ℝ) * n' / 2 + awLambda m' n' k')
        (awMult m' n' k') := by
  let eB : RMat (p + m') (p + k') ≃ᵐ (RMat p p × RMat p m') × (RMat k' p × RMat k' m') :=
    (rrrTransposeEquiv (p + m') (p + k')).trans
      ((rrrSplitRows p k' (p + m')).trans
        ((rrrSplitCols p p m').prodCongr (rrrSplitCols k' p m')))
  let eA : RMat (p + k') n' ≃ᵐ RMat n' p × RMat n' k' :=
    (rrrTransposeEquiv (p + k') n').trans (rrrSplitCols n' p k')
  let eS1 : ((RMat p p × RMat p m') × (RMat k' p × RMat k' m')) ≃ᵐ
      (RMat p p × (RMat p m' × RMat k' p)) × RMat k' m' :=
    MeasurableEquiv.prodAssoc.trans
      (((MeasurableEquiv.refl _).prodCongr MeasurableEquiv.prodAssoc.symm).trans
        MeasurableEquiv.prodAssoc.symm)
  let eS2 : RMat k' m' × (RMat n' p × RMat n' k') ≃ᵐ CoreBlock n' p k' m' :=
    MeasurableEquiv.prodComm.trans MeasurableEquiv.prodAssoc
  let eShuffle :
      ((RMat p p × RMat p m') × (RMat k' p × RMat k' m')) × (RMat n' p × RMat n' k') ≃ᵐ
      DeltaSide p k' m' × CoreBlock n' p k' m' :=
    (eS1.prodCongr (MeasurableEquiv.refl _)).trans
      (MeasurableEquiv.prodAssoc.trans ((MeasurableEquiv.refl _).prodCongr eS2))
  let eTot : RMat (p + m') (p + k') × RMat (p + k') n' ≃ᵐ
      DeltaSide p k' m' × CoreBlock n' p k' m' :=
    (eB.prodCongr eA).trans eShuffle
  have heB_mp : MeasurePreserving eB volume volume :=
    rrr_mp_trans (rrr_mp_transpose (p + m') (p + k'))
      (rrr_mp_trans (rrr_mp_splitRows p k' (p + m'))
        (rrr_mp_prodCongr (rrr_mp_splitCols p p m') (rrr_mp_splitCols k' p m')))
  have heA_mp : MeasurePreserving eA volume volume :=
    rrr_mp_trans (rrr_mp_transpose (p + k') n') (rrr_mp_splitCols n' p k')
  have heS1_mp : MeasurePreserving eS1 volume volume :=
    rrr_mp_trans MeasureTheory.volume_preserving_prodAssoc
      (rrr_mp_trans
        (rrr_mp_prodCongr (MeasurePreserving.id _) MeasureTheory.volume_preserving_prodAssoc.symm)
        MeasureTheory.volume_preserving_prodAssoc.symm)
  have heS2_mp : MeasurePreserving eS2 volume volume :=
    rrr_mp_trans rrr_mp_prodComm MeasureTheory.volume_preserving_prodAssoc
  have heShuf_mp : MeasurePreserving eShuffle volume volume :=
    rrr_mp_trans (rrr_mp_prodCongr heS1_mp (MeasurePreserving.id _))
      (rrr_mp_trans MeasureTheory.volume_preserving_prodAssoc
        (rrr_mp_prodCongr (MeasurePreserving.id _) heS2_mp))
  have heTot_mp : MeasurePreserving eTot volume volume :=
    rrr_mp_trans (rrr_mp_prodCongr heB_mp heA_mp) heShuf_mp
  have heTot_box : ∀ {δ : ℝ} (w : RMat (p + m') (p + k') × RMat (p + k') n'),
      eTot w ∈ fiberBox (m := n') (q := p) (k' := k') (n' := m') δ ↔
      w ∈ matBox δ ×ˢ matBox δ := fun {δ} ⟨ΔB, Ã⟩ => by
    have hB : eB ΔB ∈ (matBox δ ×ˢ matBox δ) ×ˢ (matBox δ ×ˢ matBox δ) ↔ ΔB ∈ matBox δ := by
      change (rrrSplitCols p p m' (rrrSplitRows p k' (p + m') (rrrTransposeEquiv (p + m') (p + k') ΔB)).1,
              rrrSplitCols k' p m' (rrrSplitRows p k' (p + m') (rrrTransposeEquiv (p + m') (p + k') ΔB)).2) ∈
          (matBox δ ×ˢ matBox δ) ×ˢ (matBox δ ×ˢ matBox δ) ↔ ΔB ∈ matBox δ
      rw [Set.mem_prod, rrr_splitCols_mem_matBox, rrr_splitCols_mem_matBox, ← Set.mem_prod]
      exact (rrr_splitRows_mem_matBox (M := rrrTransposeEquiv (p + m') (p + k') ΔB)).trans
        rrr_transpose_mem_matBox
    have hA : eA Ã ∈ matBox δ ×ˢ matBox δ ↔ Ã ∈ matBox δ := by
      change rrrSplitCols n' p k' (rrrTransposeEquiv (p + k') n' Ã) ∈ matBox δ ×ˢ matBox δ ↔ Ã ∈ matBox δ
      exact (rrr_splitCols_mem_matBox (M := rrrTransposeEquiv (p + k') n' Ã)).trans
        rrr_transpose_mem_matBox
    change eShuffle (eB ΔB, eA Ã) ∈ deltaBox δ ×ˢ coreBox δ ↔ ΔB ∈ matBox δ ∧ Ã ∈ matBox δ
    rw [← hB, ← hA]
    rcases eB ΔB with ⟨⟨d11, d12⟩, ⟨d21, d22⟩⟩
    rcases eA Ã with ⟨b1, b2⟩
    change ((d11, (d12, d21)), (b1, (b2, d22))) ∈ deltaBox δ ×ˢ coreBox δ ↔ _
    simp only [deltaBox, coreBox, Set.mem_prod]
    tauto
  have heTot_loss : ∀ w : RMat (p + m') (p + k') × RMat (p + k') n',
      rrrLoLoss w = loraFiberLoss (eTot w) := fun ⟨ΔB, Ã⟩ => by
    unfold rrrLoLoss
    rw [← frobSq_transpose_rrr, Matrix.transpose_mul, Matrix.transpose_add]
    set X := (Matrix.of Ã)ᵀ
    set Y := (rrrBaseBlo p m' k')ᵀ + (Matrix.of ΔB)ᵀ
    rcases hB_eq : eB ΔB with ⟨⟨d11, d12⟩, ⟨d21, d22⟩⟩
    rcases hA_eq : eA Ã with ⟨b1, b2⟩
    have hX_eq : X = hcatMat (Matrix.of b1) (Matrix.of b2) := by
      have h1 : rrrSplitCols n' p k' (rrrTransposeEquiv (p + k') n' Ã) = (b1, b2) := hA_eq
      have h2 := congrArg (rrrSplitCols n' p k').symm h1
      rw [MeasurableEquiv.symm_apply_apply] at h2
      have h3 := congrArg Matrix.of h2
      rw [rrrSplitCols_symm_eq_hcatMat] at h3
      exact h3
    have hΔBT_eq : (Matrix.of ΔB)ᵀ =
        vcatMat (hcatMat (Matrix.of d11) (Matrix.of d12))
                (hcatMat (Matrix.of d21) (Matrix.of d22)) := by
      have hp1 : rrrSplitCols p p m' (rrrSplitRows p k' (p + m') (rrrTransposeEquiv (p + m') (p + k') ΔB)).1 =
          (d11, d12) := congrArg Prod.fst hB_eq
      have hp2 : rrrSplitCols k' p m' (rrrSplitRows p k' (p + m') (rrrTransposeEquiv (p + m') (p + k') ΔB)).2 =
          (d21, d22) := congrArg Prod.snd hB_eq
      have h1 : rrrSplitRows p k' (p + m') (rrrTransposeEquiv (p + m') (p + k') ΔB) =
          ((rrrSplitCols p p m').symm (d11, d12), (rrrSplitCols k' p m').symm (d21, d22)) := by
        refine Prod.ext ?_ ?_
        · have := congrArg (rrrSplitCols p p m').symm hp1
          rwa [MeasurableEquiv.symm_apply_apply] at this
        · have := congrArg (rrrSplitCols k' p m').symm hp2
          rwa [MeasurableEquiv.symm_apply_apply] at this
      have h2 := congrArg (fun M => Matrix.of ((rrrSplitRows p k' (p + m')).symm M)) h1
      rw [MeasurableEquiv.symm_apply_apply, rrrSplitRows_symm_eq_vcatMat,
        rrrSplitCols_symm_eq_hcatMat, rrrSplitCols_symm_eq_hcatMat] at h2
      exact h2
    have hBaseT_eq : (rrrBaseBlo p m' k')ᵀ =
        vcatMat (hcatMat (1 : Matrix (Fin p) (Fin p) ℝ) (0 : Matrix (Fin p) (Fin m') ℝ))
                (hcatMat (0 : Matrix (Fin k') (Fin p) ℝ) (0 : Matrix (Fin k') (Fin m') ℝ)) := by
      ext i j
      simp only [Matrix.transpose_apply, rrrBaseBlo]
      by_cases hi : i.1 < p <;> by_cases hj : j.1 < p
      · have hieq : i = Fin.castAdd k' ⟨i.1, hi⟩ := by ext; rfl
        have hjeq : j = Fin.castAdd m' ⟨j.1, hj⟩ := by ext; rfl
        rw [hieq, hjeq, vcatMat_castAdd, hcatMat_castAdd, vcatMat_castAdd, hcatMat_castAdd]
        simp [Matrix.one_apply, Fin.ext_iff, eq_comm]
      · have hieq : i = Fin.castAdd k' ⟨i.1, hi⟩ := by ext; rfl
        have hjeq : j = Fin.natAdd p ⟨j.1 - p, by omega⟩ := by ext; dsimp; omega
        rw [hieq, hjeq, vcatMat_natAdd, vcatMat_castAdd, hcatMat_natAdd]; rfl
      · have hieq : i = Fin.natAdd p ⟨i.1 - p, by omega⟩ := by ext; dsimp; omega
        have hjeq : j = Fin.castAdd m' ⟨j.1, hj⟩ := by ext; rfl
        rw [hieq, hjeq, vcatMat_castAdd, hcatMat_natAdd, vcatMat_natAdd, hcatMat_castAdd]; rfl
      · have hieq : i = Fin.natAdd p ⟨i.1 - p, by omega⟩ := by ext; dsimp; omega
        have hjeq : j = Fin.natAdd p ⟨j.1 - p, by omega⟩ := by ext; dsimp; omega
        rw [hieq, hjeq, vcatMat_natAdd, vcatMat_natAdd, hcatMat_natAdd]; rfl
    have hY_eq : Y =
        vcatMat (hcatMat ((1 : Matrix (Fin p) (Fin p) ℝ) + Matrix.of d11) (Matrix.of d12))
                (hcatMat (Matrix.of d21) (Matrix.of d22)) := by
      change (rrrBaseBlo p m' k')ᵀ + (Matrix.of ΔB)ᵀ = _
      rw [hBaseT_eq, hΔBT_eq, vcatMat_add, hcatMat_add, hcatMat_add, zero_add, zero_add, zero_add]
    have hXY : X * Y =
        hcatMat (Matrix.of b1 * ((1 : Matrix (Fin p) (Fin p) ℝ) + Matrix.of d11) +
                   Matrix.of b2 * Matrix.of d21)
                (Matrix.of b1 * Matrix.of d12 + Matrix.of b2 * Matrix.of d22) := by
      rw [hX_eq, hY_eq, hcatMat_mul_vcatMat, mul_hcatMat, mul_hcatMat, hcatMat_add]
    have heTot_val : eTot (ΔB, Ã) = ((d11, (d12, d21)), (b1, (b2, d22))) := by
      change eShuffle (eB ΔB, eA Ã) = _
      rw [hB_eq, hA_eq]; rfl
    rw [hXY, frobSq_hcatMat, heTot_val]
    rfl
  obtain ⟨δ₀, hδ₀, hred⟩ :=
    lora_fiber_exact_pair (m := n') (n := p + m') (k := p + k') (q := p) (by omega) (by omega)
  refine ⟨δ₀, hδ₀, fun δ hδ hδle => ?_⟩
  have hsub1 : p + m' - p = m' := by omega
  have hsub2 : p + k' - p = k' := by omega
  have hpair := hred δ hδ hδle
  rw [hsub1, hsub2, awLambda_comm n' m' k', awMult_comm n' m' k', mul_comm (n' : ℝ) (p : ℝ)] at hpair
  have hmeas_L : Measurable (loraFiberLoss (m := n') (q := p) (k' := k') (n' := m')) := by
    unfold loraFiberLoss frobSq
    simp only [Matrix.add_apply, Matrix.mul_apply, Matrix.of_apply]
    fun_prop
  have hmeas_F : MeasurableSet (fiberBox (m := n') (q := p) (k' := k') (n' := m') δ) :=
    ((measurableSet_matBox δ).prod ((measurableSet_matBox δ).prod (measurableSet_matBox δ))).prod
      ((measurableSet_matBox δ).prod ((measurableSet_matBox δ).prod (measurableSet_matBox δ)))
  have hZeq : ∀ N : ℝ,
      SLT.Z (volume.restrict (matBox (a := p + m') (b := p + k') δ ×ˢ
                              matBox (a := p + k') (b := n') δ)) rrrLoLoss N =
      SLT.Z (volume.restrict (fiberBox (m := n') (q := p) (k' := k') (n' := m') δ))
        loraFiberLoss N := fun N =>
    rrr_Z_eq_of_mp eTot heTot_mp _ _
      ((measurableSet_matBox δ).prod (measurableSet_matBox δ)) hmeas_F
      heTot_box rrrLoLoss loraFiberLoss hmeas_L heTot_loss N
  exact hpair.congr (Eventually.of_forall fun N => (hZeq N).symm)

/-! ### 4. Triangular 3-step Schur reduction for general `(r, p, q)` and invertible `D ∈ GL(r)` -/

/-- Passive coordinates `u = (ΔA₁₁, ΔA₁₂_mid, ΔA₂₁, ΔB₂_top)`. -/
private abbrev RRRPassive (r p q n' k' : ℕ) :=
  (RMat (r + q) (r + q) × RMat q n') × (RMat (p + k') (r + q) × RMat r (p + k'))

/-- Active coordinates `v = (ΔB₁, ΔA₁₂_top, (ΔB₂_lo, ΔA₂₂))`. -/
private abbrev RRRActive (r p q m' n' k' : ℕ) :=
  RMat (r + (p + m')) (r + q) × (RMat r n' × (RMat (p + m') (p + k') × RMat (p + k') n'))

private def rrrPassiveBox {r p q n' k' : ℕ} (δ : ℝ) : Set (RRRPassive r p q n' k') :=
  (matBox δ ×ˢ matBox δ) ×ˢ (matBox δ ×ˢ matBox δ)

private def rrrActiveBox {r p q m' n' k' : ℕ} (ρ₁ ρ₂ ρ₃ : ℝ) :
    Set (RRRActive r p q m' n' k') :=
  matBox ρ₁ ×ˢ (matBox ρ₂ ×ˢ (matBox ρ₃ ×ˢ matBox ρ₃))

private theorem measurableSet_rrrPassiveBox {r p q n' k' : ℕ} (δ : ℝ) :
    MeasurableSet (rrrPassiveBox (r := r) (p := p) (q := q) (n' := n') (k' := k') δ) :=
  ((measurableSet_matBox δ).prod (measurableSet_matBox δ)).prod
    ((measurableSet_matBox δ).prod (measurableSet_matBox δ))

private theorem measurableSet_rrrActiveBox {r p q m' n' k' : ℕ} (ρ₁ ρ₂ ρ₃ : ℝ) :
    MeasurableSet (rrrActiveBox (r := r) (p := p) (q := q) (m' := m') (n' := n') (k' := k') ρ₁ ρ₂ ρ₃) :=
  (measurableSet_matBox ρ₁).prod
    ((measurableSet_matBox ρ₂).prod ((measurableSet_matBox ρ₃).prod (measurableSet_matBox ρ₃)))

private theorem volume_rrrPassiveBox_pos {r p q n' k' : ℕ} {δ : ℝ} (hδ : 0 < δ) :
    0 < volume (rrrPassiveBox (r := r) (p := p) (q := q) (n' := n') (k' := k') δ) := by
  unfold rrrPassiveBox
  rw [Measure.volume_eq_prod, Measure.prod_prod, Measure.volume_eq_prod, Measure.prod_prod,
    Measure.volume_eq_prod, Measure.prod_prod]
  exact ENNReal.mul_pos
    (ENNReal.mul_pos (volume_matBox_pos hδ).ne' (volume_matBox_pos hδ).ne').ne'
    (ENNReal.mul_pos (volume_matBox_pos hδ).ne' (volume_matBox_pos hδ).ne').ne'

private theorem volume_rrrPassiveBox_lt_top {r p q n' k' : ℕ} {δ : ℝ} (hδ : 0 ≤ δ) :
    volume (rrrPassiveBox (r := r) (p := p) (q := q) (n' := n') (k' := k') δ) < ⊤ := by
  unfold rrrPassiveBox
  rw [Measure.volume_eq_prod, Measure.prod_prod, Measure.volume_eq_prod, Measure.prod_prod,
    Measure.volume_eq_prod, Measure.prod_prod]
  exact ENNReal.mul_lt_top (ENNReal.mul_lt_top (volume_matBox_lt_top hδ) (volume_matBox_lt_top hδ))
    (ENNReal.mul_lt_top (volume_matBox_lt_top hδ) (volume_matBox_lt_top hδ))

/-- Measure-preserving block split `(ΔB, ΔA) ≃ᵐ (u, v)` into passive and active blocks. -/
private def rrrSplitEquiv (r p q m' n' k' : ℕ) :
    (RMat (r + (p + m')) ((r + q) + (p + k')) × RMat ((r + q) + (p + k')) ((r + q) + n')) ≃ᵐ
      RRRPassive r p q n' k' × RRRActive r p q m' n' k' :=
  let eB : RMat (r + (p + m')) ((r + q) + (p + k')) ≃ᵐ
      RMat (r + (p + m')) (r + q) × (RMat r (p + k') × RMat (p + m') (p + k')) :=
    (rrrSplitCols (r + (p + m')) (r + q) (p + k')).trans
      ((MeasurableEquiv.refl _).prodCongr (rrrSplitRows r (p + m') (p + k')))
  let eA : RMat ((r + q) + (p + k')) ((r + q) + n') ≃ᵐ
      (RMat (r + q) (r + q) × (RMat r n' × RMat q n')) ×
      (RMat (p + k') (r + q) × RMat (p + k') n') :=
    (rrrSplitRows (r + q) (p + k') ((r + q) + n')).trans
      (((rrrSplitCols (r + q) (r + q) n').trans
          ((MeasurableEquiv.refl _).prodCongr (rrrSplitRows r q n'))).prodCongr
       (rrrSplitCols (p + k') (r + q) n'))
  let eB' : RMat (r + (p + m')) (r + q) × (RMat r (p + k') × RMat (p + m') (p + k')) ≃ᵐ
      RMat r (p + k') × (RMat (r + (p + m')) (r + q) × RMat (p + m') (p + k')) :=
    MeasurableEquiv.prodAssoc.symm.trans
      ((MeasurableEquiv.prodComm.prodCongr (MeasurableEquiv.refl _)).trans
        MeasurableEquiv.prodAssoc)
  let eA1 : RMat (r + q) (r + q) × (RMat r n' × RMat q n') ≃ᵐ
      (RMat (r + q) (r + q) × RMat q n') × RMat r n' :=
    ((MeasurableEquiv.refl _).prodCongr MeasurableEquiv.prodComm).trans
      MeasurableEquiv.prodAssoc.symm
  let eA' : (RMat (r + q) (r + q) × (RMat r n' × RMat q n')) ×
      (RMat (p + k') (r + q) × RMat (p + k') n') ≃ᵐ
      ((RMat (r + q) (r + q) × RMat q n') × RMat (p + k') (r + q)) ×
      (RMat r n' × RMat (p + k') n') :=
    (eA1.prodCongr (MeasurableEquiv.refl _)).trans
      (MeasurableEquiv.prodAssoc.trans
        (((MeasurableEquiv.refl _).prodCongr
            (MeasurableEquiv.prodAssoc.symm.trans
              ((MeasurableEquiv.prodComm.prodCongr (MeasurableEquiv.refl _)).trans
                MeasurableEquiv.prodAssoc))).trans
          MeasurableEquiv.prodAssoc.symm))
  let eMid1 : (RMat r (p + k') × (RMat (r + (p + m')) (r + q) × RMat (p + m') (p + k'))) ×
      (((RMat (r + q) (r + q) × RMat q n') × RMat (p + k') (r + q)) ×
       (RMat r n' × RMat (p + k') n')) ≃ᵐ
      ((((RMat (r + q) (r + q) × RMat q n') × RMat (p + k') (r + q)) × RMat r (p + k')) ×
       ((RMat (r + (p + m')) (r + q) × RMat (p + m') (p + k')) ×
        (RMat r n' × RMat (p + k') n'))) :=
    MeasurableEquiv.prodAssoc.trans
      (((MeasurableEquiv.refl _).prodCongr
          (MeasurableEquiv.prodAssoc.symm.trans
            ((MeasurableEquiv.prodComm.prodCongr (MeasurableEquiv.refl _)).trans
              MeasurableEquiv.prodAssoc))).trans
        (MeasurableEquiv.prodAssoc.symm.trans
          (MeasurableEquiv.prodComm.prodCongr (MeasurableEquiv.refl _))))
  let eAct : (RMat (r + (p + m')) (r + q) × RMat (p + m') (p + k')) ×
      (RMat r n' × RMat (p + k') n') ≃ᵐ RRRActive r p q m' n' k' :=
    MeasurableEquiv.prodAssoc.trans
      ((MeasurableEquiv.refl _).prodCongr
        (MeasurableEquiv.prodAssoc.symm.trans
          ((MeasurableEquiv.prodComm.prodCongr (MeasurableEquiv.refl _)).trans
            MeasurableEquiv.prodAssoc)))
  ((eB.trans eB').prodCongr (eA.trans eA')).trans
    (eMid1.trans (MeasurableEquiv.prodAssoc.prodCongr eAct))

private theorem rrr_mp_splitEquiv (r p q m' n' k' : ℕ) :
    MeasurePreserving (rrrSplitEquiv r p q m' n' k') volume volume := by
  unfold rrrSplitEquiv
  dsimp only
  refine rrr_mp_trans (rrr_mp_prodCongr ?_ ?_) (rrr_mp_trans ?_ (rrr_mp_prodCongr ?_ ?_))
  · refine rrr_mp_trans (rrr_mp_trans (rrr_mp_splitCols _ _ _)
        (rrr_mp_prodCongr (MeasurePreserving.id _) (rrr_mp_splitRows _ _ _))) ?_
    exact rrr_mp_trans MeasureTheory.volume_preserving_prodAssoc.symm
      (rrr_mp_trans (rrr_mp_prodCongr rrr_mp_prodComm (MeasurePreserving.id _))
        MeasureTheory.volume_preserving_prodAssoc)
  · refine rrr_mp_trans (rrr_mp_trans (rrr_mp_splitRows _ _ _)
        (rrr_mp_prodCongr
          (rrr_mp_trans (rrr_mp_splitCols _ _ _)
            (rrr_mp_prodCongr (MeasurePreserving.id _) (rrr_mp_splitRows _ _ _)))
          (rrr_mp_splitCols _ _ _))) ?_
    refine rrr_mp_trans
      (rrr_mp_prodCongr
        (rrr_mp_trans (rrr_mp_prodCongr (MeasurePreserving.id _) rrr_mp_prodComm)
          MeasureTheory.volume_preserving_prodAssoc.symm)
        (MeasurePreserving.id _)) ?_
    exact rrr_mp_trans MeasureTheory.volume_preserving_prodAssoc
      (rrr_mp_trans
        (rrr_mp_prodCongr (MeasurePreserving.id _)
          (rrr_mp_trans MeasureTheory.volume_preserving_prodAssoc.symm
            (rrr_mp_trans (rrr_mp_prodCongr rrr_mp_prodComm (MeasurePreserving.id _))
              MeasureTheory.volume_preserving_prodAssoc)))
        MeasureTheory.volume_preserving_prodAssoc.symm)
  · exact rrr_mp_trans MeasureTheory.volume_preserving_prodAssoc
      (rrr_mp_trans
        (rrr_mp_prodCongr (MeasurePreserving.id _)
          (rrr_mp_trans MeasureTheory.volume_preserving_prodAssoc.symm
            (rrr_mp_trans (rrr_mp_prodCongr rrr_mp_prodComm (MeasurePreserving.id _))
              MeasureTheory.volume_preserving_prodAssoc)))
        (rrr_mp_trans MeasureTheory.volume_preserving_prodAssoc.symm
          (rrr_mp_prodCongr rrr_mp_prodComm (MeasurePreserving.id _))))
  · exact MeasureTheory.volume_preserving_prodAssoc
  · exact rrr_mp_trans MeasureTheory.volume_preserving_prodAssoc
      (rrr_mp_prodCongr (MeasurePreserving.id _)
        (rrr_mp_trans MeasureTheory.volume_preserving_prodAssoc.symm
          (rrr_mp_trans (rrr_mp_prodCongr rrr_mp_prodComm (MeasurePreserving.id _))
            MeasureTheory.volume_preserving_prodAssoc)))

private theorem rrrSplitEquiv_mem_box {r p q m' n' k' : ℕ} {δ : ℝ}
    (θ : RMat (r + (p + m')) ((r + q) + (p + k')) × RMat ((r + q) + (p + k')) ((r + q) + n')) :
    rrrSplitEquiv r p q m' n' k' θ ∈ rrrPassiveBox δ ×ˢ rrrActiveBox δ δ δ ↔
    θ ∈ matBox δ ×ˢ matBox δ := by
  rcases θ with ⟨ΔB, ΔA⟩
  set cB := rrrSplitCols (r + (p + m')) (r + q) (p + k') ΔB
  set rB2 := rrrSplitRows r (p + m') (p + k') cB.2
  set rA := rrrSplitRows (r + q) (p + k') ((r + q) + n') ΔA
  set cA1 := rrrSplitCols (r + q) (r + q) n' rA.1
  set rA12 := rrrSplitRows r q n' cA1.2
  set cA2 := rrrSplitCols (p + k') (r + q) n' rA.2
  have hB : cB.1 ∈ matBox δ ∧ rB2.1 ∈ matBox δ ∧ rB2.2 ∈ matBox δ ↔ ΔB ∈ matBox δ := by
    have h1 := rrr_splitCols_mem_matBox (a := r + (p + m')) (b₁ := r + q) (b₂ := p + k') (δ := δ) (M := ΔB)
    have h2 := rrr_splitRows_mem_matBox (a₁ := r) (a₂ := p + m') (b := p + k') (δ := δ) (M := cB.2)
    simp only [Set.mem_prod] at h1 h2
    tauto
  have hA : cA1.1 ∈ matBox δ ∧ rA12.1 ∈ matBox δ ∧ rA12.2 ∈ matBox δ ∧
      cA2.1 ∈ matBox δ ∧ cA2.2 ∈ matBox δ ↔ ΔA ∈ matBox δ := by
    have h1 := rrr_splitRows_mem_matBox (a₁ := r + q) (a₂ := p + k') (b := (r + q) + n') (δ := δ) (M := ΔA)
    have h2 := rrr_splitCols_mem_matBox (a := r + q) (b₁ := r + q) (b₂ := n') (δ := δ) (M := rA.1)
    have h3 := rrr_splitRows_mem_matBox (a₁ := r) (a₂ := q) (b := n') (δ := δ) (M := cA1.2)
    have h4 := rrr_splitCols_mem_matBox (a := p + k') (b₁ := r + q) (b₂ := n') (δ := δ) (M := rA.2)
    simp only [Set.mem_prod] at h1 h2 h3 h4
    tauto
  change (((cA1.1, rA12.2), (cA2.1, rB2.1)), (cB.1, (rA12.1, (rB2.2, cA2.2)))) ∈
      rrrPassiveBox δ ×ˢ rrrActiveBox δ δ δ ↔ ΔB ∈ matBox δ ∧ ΔA ∈ matBox δ
  simp only [rrrPassiveBox, rrrActiveBox, Set.mem_prod]
  tauto

/-- Block form of the RRR fiber loss on `RRRPassive × RRRActive`. -/
private def rrrBlockLoss {r p q m' n' k' : ℕ} (D : Matrix (Fin r) (Fin r) ℝ)
    (uv : RRRPassive r p q n' k' × RRRActive r p q m' n' k') : ℝ :=
  let ΔA₁₁ := Matrix.of uv.1.1.1
  let ΔA₁₂_mid := Matrix.of uv.1.1.2
  let ΔA₂₁ := Matrix.of uv.1.2.1
  let ΔB₂_top := Matrix.of uv.1.2.2
  let ΔB₁ := Matrix.of uv.2.1
  let ΔA₁₂_top := Matrix.of uv.2.2.1
  let ΔB₂_lo := Matrix.of uv.2.2.2.1
  let ΔA₂₂ := Matrix.of uv.2.2.2.2
  let S := (1 : Matrix (Fin (r + q)) (Fin (r + q)) ℝ) + ΔA₁₁
  let ΔA₁₂ := vcatMat ΔA₁₂_top ΔA₁₂_mid
  let B₂ := vcatMat ΔB₂_top (rrrBaseBlo p m' k' + ΔB₂_lo)
  let B₀L := rrrBaseBL (p := p) (q := q) (m' := m') D
  frobSq (ΔB₁ * S + B₀L * ΔA₁₁ + B₂ * ΔA₂₁) +
    frobSq ((B₀L + ΔB₁) * ΔA₁₂ + B₂ * ΔA₂₂)

private theorem rrrFiberLoss_eq_blockLoss {r p q m' n' k' : ℕ} (D : Matrix (Fin r) (Fin r) ℝ)
    (θ : RMat (r + (p + m')) ((r + q) + (p + k')) × RMat ((r + q) + (p + k')) ((r + q) + n')) :
    rrrFiberLoss D θ = rrrBlockLoss D (rrrSplitEquiv r p q m' n' k' θ) := by
  rcases θ with ⟨ΔB, ΔA⟩
  set cB := rrrSplitCols (r + (p + m')) (r + q) (p + k') ΔB
  set rB2 := rrrSplitRows r (p + m') (p + k') cB.2
  set rA := rrrSplitRows (r + q) (p + k') ((r + q) + n') ΔA
  set cA1 := rrrSplitCols (r + q) (r + q) n' rA.1
  set rA12 := rrrSplitRows r q n' cA1.2
  set cA2 := rrrSplitCols (p + k') (r + q) n' rA.2
  have hΔB_eq : Matrix.of ΔB = hcatMat (Matrix.of cB.1) (vcatMat (Matrix.of rB2.1) (Matrix.of rB2.2)) := by
    have h1 : (rrrSplitRows r (p + m') (p + k')).symm (rB2.1, rB2.2) = cB.2 :=
      (rrrSplitRows r (p + m') (p + k')).symm_apply_apply cB.2
    have h2 : (rrrSplitCols (r + (p + m')) (r + q) (p + k')).symm (cB.1, cB.2) = ΔB :=
      (rrrSplitCols (r + (p + m')) (r + q) (p + k')).symm_apply_apply ΔB
    have h3 := congrArg Matrix.of h2
    rw [rrrSplitCols_symm_eq_hcatMat, ← h1, rrrSplitRows_symm_eq_vcatMat] at h3
    exact h3.symm
  have hΔA_eq : Matrix.of ΔA =
      vcatMat (hcatMat (Matrix.of cA1.1) (vcatMat (Matrix.of rA12.1) (Matrix.of rA12.2)))
              (hcatMat (Matrix.of cA2.1) (Matrix.of cA2.2)) := by
    have h1 : (rrrSplitRows r q n').symm (rA12.1, rA12.2) = cA1.2 :=
      (rrrSplitRows r q n').symm_apply_apply cA1.2
    have h2 : (rrrSplitCols (r + q) (r + q) n').symm (cA1.1, cA1.2) = rA.1 :=
      (rrrSplitCols (r + q) (r + q) n').symm_apply_apply rA.1
    have h3 : (rrrSplitCols (p + k') (r + q) n').symm (cA2.1, cA2.2) = rA.2 :=
      (rrrSplitCols (p + k') (r + q) n').symm_apply_apply rA.2
    have h4 : (rrrSplitRows (r + q) (p + k') ((r + q) + n')).symm (rA.1, rA.2) = ΔA :=
      (rrrSplitRows (r + q) (p + k') ((r + q) + n')).symm_apply_apply ΔA
    have h5 := congrArg Matrix.of h4
    rw [rrrSplitRows_symm_eq_vcatMat, ← h2, ← h3, rrrSplitCols_symm_eq_hcatMat,
      rrrSplitCols_symm_eq_hcatMat, ← h1, rrrSplitRows_symm_eq_vcatMat] at h5
    exact h5.symm
  unfold rrrFiberLoss
  rw [hΔB_eq, hΔA_eq]
  unfold rrrBaseB rrrBaseBR rrrBaseA rrrTarget
  rw [hcatMat_add, vcatMat_add, zero_add]
  have hzeroA : (0 : Matrix (Fin (p + k')) (Fin ((r + q) + n')) ℝ) =
      hcatMat (0 : Matrix (Fin (p + k')) (Fin (r + q)) ℝ)
              (0 : Matrix (Fin (p + k')) (Fin n') ℝ) := by
    ext i j
    by_cases hj : j.1 < r + q
    · have hjeq : j = Fin.castAdd n' ⟨j.1, hj⟩ := by ext; rfl
      rw [hjeq, hcatMat_castAdd]; rfl
    · have hjeq : j = Fin.natAdd (r + q) ⟨j.1 - (r + q), by omega⟩ := by ext; dsimp; omega
      rw [hjeq, hcatMat_natAdd]; rfl
  rw [hzeroA, vcatMat_add, hcatMat_add, hcatMat_add, zero_add, zero_add, zero_add]
  rw [hcatMat_mul_vcatMat, mul_hcatMat, mul_hcatMat, hcatMat_add, hcatMat_sub, sub_zero]
  rw [frobSq_hcatMat]
  have hML : (rrrBaseBL (p := p) (q := q) (m' := m') D + Matrix.of cB.1) * (1 + Matrix.of cA1.1) +
        vcatMat (Matrix.of rB2.1) (rrrBaseBlo p m' k' + Matrix.of rB2.2) * Matrix.of cA2.1 -
        rrrBaseBL (p := p) (q := q) (m' := m') D =
      Matrix.of cB.1 * (1 + Matrix.of cA1.1) + rrrBaseBL (p := p) (q := q) (m' := m') D * Matrix.of cA1.1 +
        vcatMat (Matrix.of rB2.1) (rrrBaseBlo p m' k' + Matrix.of rB2.2) * Matrix.of cA2.1 := by
    simp only [Matrix.add_mul, Matrix.mul_add, Matrix.mul_one]
    ext i j; simp only [Matrix.add_apply, Matrix.sub_apply]; ring
  rw [hML]
  rfl

/-- Exact Schur-complement rewrite of `M_R` in `rrrBlockLoss`. -/
private theorem rrr_schur_block_right {r p q m' n' k' : ℕ}
    (B₀L ΔB₁ : Matrix (Fin (r + (p + m'))) (Fin (r + q)) ℝ)
    (B₂ : Matrix (Fin (r + (p + m'))) (Fin (p + k')) ℝ)
    (ΔA₁₁ : Matrix (Fin (r + q)) (Fin (r + q)) ℝ)
    (hS : IsUnit ((1 : Matrix (Fin (r + q)) (Fin (r + q)) ℝ) + ΔA₁₁).det)
    (ΔA₁₂ : Matrix (Fin (r + q)) (Fin n') ℝ)
    (ΔA₂₁ : Matrix (Fin (p + k')) (Fin (r + q)) ℝ)
    (ΔA₂₂ : Matrix (Fin (p + k')) (Fin n') ℝ) :
    let S := (1 : Matrix (Fin (r + q)) (Fin (r + q)) ℝ) + ΔA₁₁
    let M_L := ΔB₁ * S + B₀L * ΔA₁₁ + B₂ * ΔA₂₁
    (B₀L + ΔB₁) * ΔA₁₂ + B₂ * ΔA₂₂ =
      M_L * (S⁻¹ * ΔA₁₂) + (B₀L * (S⁻¹ * ΔA₁₂) + B₂ * (ΔA₂₂ - ΔA₂₁ * (S⁻¹ * ΔA₁₂))) := by
  intro S M_L
  have hSS : S * S⁻¹ = 1 := Matrix.mul_nonsing_inv S hS
  have hML' : M_L = (B₀L + ΔB₁) * S + B₂ * ΔA₂₁ - B₀L := by
    simp only [M_L, S, Matrix.add_mul, Matrix.mul_add, Matrix.mul_one]
    abel
  rw [hML']
  calc (B₀L + ΔB₁) * ΔA₁₂ + B₂ * ΔA₂₂
      = (B₀L + ΔB₁) * (S * S⁻¹) * ΔA₁₂ + B₂ * ΔA₂₂ := by rw [hSS, Matrix.mul_one]
    _ = ((B₀L + ΔB₁) * S + B₂ * ΔA₂₁ - B₀L) * (S⁻¹ * ΔA₁₂) +
          (B₀L * (S⁻¹ * ΔA₁₂) + B₂ * (ΔA₂₂ - ΔA₂₁ * (S⁻¹ * ΔA₁₂))) := by
        simp only [Matrix.sub_mul, Matrix.add_mul, Matrix.mul_sub, Matrix.mul_assoc]
        abel

/-- Top-left `r × r` block of a `(r + q) × (r + q)` matrix. -/
private def topLeftBlock {r q : ℕ} (X : Matrix (Fin (r + q)) (Fin (r + q)) ℝ) :
    Matrix (Fin r) (Fin r) ℝ :=
  Matrix.of fun i j => X (Fin.castAdd q i) (Fin.castAdd q j)

/-- Top-right `r × q` block of a `(r + q) × (r + q)` matrix. -/
private def topRightBlock {r q : ℕ} (X : Matrix (Fin (r + q)) (Fin (r + q)) ℝ) :
    Matrix (Fin r) (Fin q) ℝ :=
  Matrix.of fun i j => X (Fin.castAdd q i) (Fin.natAdd r j)

private theorem rrrBaseBL_mul_inv_vcat {r p q m' n' : ℕ}
    (D : Matrix (Fin r) (Fin r) ℝ)
    (X : Matrix (Fin (r + q)) (Fin (r + q)) ℝ)
    (A_top : Matrix (Fin r) (Fin n') ℝ) (A_mid : Matrix (Fin q) (Fin n') ℝ) :
    rrrBaseBL (p := p) (q := q) (m' := m') D * (X * vcatMat A_top A_mid) =
      vcatMat (D * topLeftBlock X * A_top + D * topRightBlock X * A_mid)
              (0 : Matrix (Fin (p + m')) (Fin n') ℝ) := by
  unfold rrrBaseBL
  rw [vcatMat_mul, Matrix.zero_mul]
  congr 1
  ext i j
  simp only [Matrix.mul_apply, Matrix.add_apply, topLeftBlock, topRightBlock, Matrix.of_apply]
  rw [Fin.sum_univ_add]
  have hzero : ∑ x : Fin q, hcatMat D (0 : Matrix (Fin r) (Fin q) ℝ) i (Fin.natAdd r x) *
      (∑ k : Fin (r + q), X (Fin.natAdd r x) k * vcatMat A_top A_mid k j) = 0 := by
    refine Finset.sum_eq_zero fun x _ => ?_
    rw [hcatMat_natAdd]; simp
  rw [hzero, add_zero]
  have h1 : ∀ a : Fin r,
      hcatMat D (0 : Matrix (Fin r) (Fin q) ℝ) i (Fin.castAdd q a) *
        (∑ k : Fin (r + q), X (Fin.castAdd q a) k * vcatMat A_top A_mid k j) =
      D i a * (∑ b : Fin r, X (Fin.castAdd q a) (Fin.castAdd q b) * A_top b j) +
      D i a * (∑ c : Fin q, X (Fin.castAdd q a) (Fin.natAdd r c) * A_mid c j) := fun a => by
    rw [hcatMat_castAdd, Fin.sum_univ_add, mul_add]
    congr 1
    · congr 1; refine Finset.sum_congr rfl fun b _ => by rw [vcatMat_castAdd]
    · congr 1; refine Finset.sum_congr rfl fun c _ => by rw [vcatMat_natAdd]
  simp_rw [h1, Finset.sum_add_distrib]
  congr 1
  · simp_rw [Finset.mul_sum, Finset.sum_mul, ← mul_assoc]
    exact Finset.sum_comm
  · simp_rw [Finset.mul_sum, Finset.sum_mul, ← mul_assoc]
    exact Finset.sum_comm

/-- The decoupled core loss on `RRRActive`: `‖W₁‖_F² + ‖W₂‖_F² + rrrLoLoss lo`. -/
private def rrrDecoupledLoss {r p q m' n' k' : ℕ}
    (v : RRRActive r p q m' n' k') : ℝ :=
  frobSq (Matrix.of v.1) + (frobSq (Matrix.of v.2.1) + rrrLoLoss v.2.2)

private theorem measurable_rrrDecoupledLoss {r p q m' n' k' : ℕ} :
    Measurable (rrrDecoupledLoss (r := r) (p := p) (q := q) (m' := m') (n' := n') (k' := k')) := by
  unfold rrrDecoupledLoss
  exact (measurable_frobSq.comp measurable_fst).add
    ((measurable_frobSq.comp (measurable_fst.comp measurable_snd)).add
      (measurable_rrrLoLoss.comp (measurable_snd.comp measurable_snd)))

/-- The 3-step triangular affine Schur map `Φ_u = T₃ ∘ T₂ ∘ T₁` on `RRRActive` for fixed `u`. -/
private noncomputable def rrrTriMap {r p q m' n' k' : ℕ} (D : Matrix (Fin r) (Fin r) ℝ)
    (u : RRRPassive r p q n' k') :
    RRRActive r p q m' n' k' → RRRActive r p q m' n' k' :=
  let ΔA₁₁ := Matrix.of u.1.1
  let ΔA₁₂_mid := Matrix.of u.1.2
  let ΔA₂₁ := Matrix.of u.2.1
  let ΔB₂_top := Matrix.of u.2.2
  let S := (1 : Matrix (Fin (r + q)) (Fin (r + q)) ℝ) + ΔA₁₁
  let P₁₁ := topLeftBlock S⁻¹
  let P₁₂ := topRightBlock S⁻¹
  let B₀L := rrrBaseBL (p := p) (q := q) (m' := m') D
  fun v =>
    let ΔB₁ := Matrix.of v.1
    let ΔA₁₂_top := Matrix.of v.2.1
    let ΔB₂_lo := Matrix.of v.2.2.1
    let ΔA₂₂ := Matrix.of v.2.2.2
    let ΔA₁₂ := vcatMat ΔA₁₂_top ΔA₁₂_mid
    let B₂ := vcatMat ΔB₂_top (rrrBaseBlo p m' k' + ΔB₂_lo)
    let Ã₂₂ := ΔA₂₂ - ΔA₂₁ * (S⁻¹ * ΔA₁₂)
    let W₂ := D * P₁₁ * ΔA₁₂_top + D * P₁₂ * ΔA₁₂_mid + ΔB₂_top * Ã₂₂
    let W₁ := ΔB₁ * S + B₀L * ΔA₁₁ + B₂ * ΔA₂₁
    (toRMat W₁, (toRMat W₂, (v.2.2.1, toRMat Ã₂₂)))

private theorem measurable_rrrTriMap {r p q m' n' k' : ℕ} (D : Matrix (Fin r) (Fin r) ℝ)
    (u : RRRPassive r p q n' k') :
    Measurable (rrrTriMap (m' := m') D u) := by
  unfold rrrTriMap
  have hA12t_m : ∀ (i : Fin r) (j : Fin n'),
      Measurable (fun v : RRRActive r p q m' n' k' => v.2.1 i j) := fun i j =>
    (measurable_pi_apply j).comp ((measurable_pi_apply i).comp (measurable_fst.comp measurable_snd))
  have hB1_m : ∀ (i : Fin (r + (p + m'))) (j : Fin (r + q)),
      Measurable (fun v : RRRActive r p q m' n' k' => v.1 i j) := fun i j =>
    (measurable_pi_apply j).comp ((measurable_pi_apply i).comp measurable_fst)
  have hB2lo_m : ∀ (i : Fin (p + m')) (j : Fin (p + k')),
      Measurable (fun v : RRRActive r p q m' n' k' => v.2.2.1 i j) := fun i j =>
    (measurable_pi_apply j).comp ((measurable_pi_apply i).comp
      (measurable_fst.comp (measurable_snd.comp measurable_snd)))
  have hA22_m : ∀ (i : Fin (p + k')) (j : Fin n'),
      Measurable (fun v : RRRActive r p q m' n' k' => v.2.2.2 i j) := fun i j =>
    (measurable_pi_apply j).comp ((measurable_pi_apply i).comp
      (measurable_snd.comp (measurable_snd.comp measurable_snd)))
  have hvcatA_m : ∀ (i : Fin (r + q)) (j : Fin n'),
      Measurable (fun v : RRRActive r p q m' n' k' =>
        vcatMat (Matrix.of v.2.1) (Matrix.of u.1.2) i j) := fun i j => by
    simp only [vcatMat, Matrix.of_apply]
    split_ifs
    · exact hA12t_m _ _
    · exact measurable_const
  have hvcatB_m : ∀ (i : Fin (r + (p + m'))) (j : Fin (p + k')),
      Measurable (fun v : RRRActive r p q m' n' k' =>
        vcatMat (Matrix.of u.2.2) (rrrBaseBlo p m' k' + Matrix.of v.2.2.1) i j) := fun i j => by
    simp only [vcatMat, Matrix.add_apply, Matrix.of_apply]
    split_ifs
    · exact measurable_const
    · exact measurable_const.add (hB2lo_m _ _)
  have hÃ_m : ∀ (i : Fin (p + k')) (j : Fin n'),
      Measurable (fun v : RRRActive r p q m' n' k' =>
        (Matrix.of v.2.2.2 - Matrix.of u.2.1 * (((1 : Matrix (Fin (r + q)) (Fin (r + q)) ℝ) +
          Matrix.of u.1.1)⁻¹ * vcatMat (Matrix.of v.2.1) (Matrix.of u.1.2))) i j) := fun i j => by
    simp only [Matrix.sub_apply, Matrix.mul_apply, Matrix.of_apply]
    refine (hA22_m i j).sub (Finset.measurable_sum _ fun t _ => measurable_const.mul
      (Finset.measurable_sum _ fun s _ => measurable_const.mul (hvcatA_m s j)))
  refine Measurable.prodMk ?_ (Measurable.prodMk ?_ (Measurable.prodMk ?_ ?_))
  · refine Measurable.of_eval fun i => Measurable.of_eval fun j => ?_
    simp only [toRMat_apply, Matrix.add_apply, Matrix.mul_apply, Matrix.of_apply]
    exact ((Finset.measurable_sum _ fun t _ => (hB1_m i t).mul measurable_const).add
      measurable_const).add (Finset.measurable_sum _ fun t _ => (hvcatB_m i t).mul measurable_const)
  · refine Measurable.of_eval fun i => Measurable.of_eval fun j => ?_
    simp only [toRMat_apply, Matrix.add_apply, Matrix.mul_apply, Matrix.of_apply]
    exact ((Finset.measurable_sum _ fun t _ => measurable_const.mul (hA12t_m t j)).add
      measurable_const).add (Finset.measurable_sum _ fun t _ => measurable_const.mul (hÃ_m t j))
  · exact measurable_fst.comp (measurable_snd.comp measurable_snd)
  · refine Measurable.of_eval fun i => Measurable.of_eval fun j => ?_
    exact hÃ_m i j

/-- Pushforward of `volume` on `RRRActive` by `rrrTriMap D u` scales Lebesgue measure by
`|det S|^{-M} · |det(D * P₁₁)|^{-n'}`. -/
private theorem map_rrrTriMap_volume {r p q m' n' k' : ℕ} (D : Matrix (Fin r) (Fin r) ℝ)
    (u : RRRPassive r p q n' k')
    (hS : ((1 : Matrix (Fin (r + q)) (Fin (r + q)) ℝ) + Matrix.of u.1.1).det ≠ 0)
    (hDP : (D * topLeftBlock ((1 : Matrix (Fin (r + q)) (Fin (r + q)) ℝ) + Matrix.of u.1.1)⁻¹).det ≠ 0) :
    let S := (1 : Matrix (Fin (r + q)) (Fin (r + q)) ℝ) + Matrix.of u.1.1
    let DP := D * topLeftBlock S⁻¹
    Measure.map (rrrTriMap (m' := m') D u) volume =
      (ENNReal.ofReal |(S.det)⁻¹| ^ (r + (p + m')) *
        ENNReal.ofReal |(DP.det)⁻¹| ^ n') • volume := by
  intro S DP
  set ΔA₁₁ := Matrix.of u.1.1
  set ΔA₁₂_mid := Matrix.of u.1.2
  set ΔA₂₁ := Matrix.of u.2.1
  set ΔB₂_top := Matrix.of u.2.2
  set P₁₂ := topRightBlock S⁻¹
  set B₀L := rrrBaseBL (p := p) (q := q) (m' := m') D
  let shiftA : RMat r n' → RMat (p + k') n' := fun A_top =>
    toRMat (ΔA₂₁ * (S⁻¹ * vcatMat (Matrix.of A_top) ΔA₁₂_mid))
  let T₁ : RRRActive r p q m' n' k' → RRRActive r p q m' n' k' :=
    fun v => (v.1, (v.2.1, (v.2.2.1, v.2.2.2 - shiftA v.2.1)))
  let shiftW₂ : RMat (p + k') n' → RMat r n' := fun A22 =>
    toRMat (D * P₁₂ * ΔA₁₂_mid + ΔB₂_top * Matrix.of A22)
  let T₂ : RRRActive r p q m' n' k' → RRRActive r p q m' n' k' :=
    fun v => (v.1, (toRMat (DP * Matrix.of v.2.1) + shiftW₂ v.2.2.2, v.2.2))
  let shiftW₁ : RMat (p + m') (p + k') → RMat (r + (p + m')) (r + q) := fun B2lo =>
    toRMat (B₀L * ΔA₁₁ + vcatMat ΔB₂_top (rrrBaseBlo p m' k' + Matrix.of B2lo) * ΔA₂₁)
  let T₃ : RRRActive r p q m' n' k' → RRRActive r p q m' n' k' :=
    fun v => (toRMat (Matrix.of v.1 * S) + shiftW₁ v.2.2.1, v.2)
  have hcomp : rrrTriMap (m' := m') D u = T₃ ∘ T₂ ∘ T₁ := by
    funext ⟨B1, A12t, B2lo, A22⟩
    unfold rrrTriMap T₃ T₂ T₁ shiftW₁ shiftW₂ shiftA
    refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext rfl rfl))
    · funext i j; simp [toRMat, Matrix.add_apply]; ring
    · funext i j
      change (D * topLeftBlock S⁻¹ * Matrix.of A12t +
          D * topRightBlock S⁻¹ * ΔA₁₂_mid +
          ΔB₂_top * (Matrix.of A22 - ΔA₂₁ * (S⁻¹ * vcatMat (Matrix.of A12t) ΔA₁₂_mid))) i j =
        (DP * Matrix.of A12t) i j +
          (D * P₁₂ * ΔA₁₂_mid + ΔB₂_top * Matrix.of (A22 - shiftA A12t)) i j
      have hsh : Matrix.of (A22 - shiftA A12t) =
          Matrix.of A22 - ΔA₂₁ * (S⁻¹ * vcatMat (Matrix.of A12t) ΔA₁₂_mid) := by
        ext a b; rfl
      rw [hsh]
      simp only [Matrix.add_apply, DP, P₁₂]
      ring
  have hshiftA_m : Measurable shiftA := by
    refine Measurable.of_eval fun i => Measurable.of_eval fun j => ?_
    simp only [shiftA, toRMat_apply, Matrix.mul_apply, vcatMat, Matrix.of_apply]
    refine Finset.measurable_sum _ fun _ _ => measurable_const.mul ?_
    refine Finset.measurable_sum _ fun s _ => measurable_const.mul ?_
    split_ifs
    · exact (measurable_pi_apply _).comp (measurable_pi_apply _)
    · exact measurable_const
  have hshiftW2_m : Measurable shiftW₂ := by
    refine Measurable.of_eval fun i => Measurable.of_eval fun j => ?_
    simp only [shiftW₂, toRMat_apply, Matrix.add_apply, Matrix.mul_apply, Matrix.of_apply]
    exact measurable_const.add (Finset.measurable_sum _ fun _ _ =>
      measurable_const.mul ((measurable_pi_apply _).comp (measurable_pi_apply _)))
  have hshiftW1_m : Measurable shiftW₁ := by
    have hB2lo_at : ∀ (a : Fin (p + m')) (b : Fin (p + k')),
        Measurable (fun X : RMat (p + m') (p + k') => X a b) := fun a b =>
      (measurable_pi_apply b).comp (measurable_pi_apply a)
    refine Measurable.of_eval fun i => Measurable.of_eval fun j => ?_
    simp only [shiftW₁, toRMat_apply, Matrix.add_apply, Matrix.mul_apply, vcatMat, Matrix.of_apply]
    refine measurable_const.add (Finset.measurable_sum _ fun t _ => ?_)
    split_ifs with hi
    · exact measurable_const
    · exact (measurable_const.add (hB2lo_at ⟨i.1 - r, by omega⟩ t)).mul measurable_const
  have hmulDP_m : Measurable (fun X : RMat r n' => toRMat (DP * Matrix.of X)) := by
    refine Measurable.of_eval fun i => Measurable.of_eval fun j => ?_
    simp only [toRMat_apply, Matrix.mul_apply, Matrix.of_apply]; fun_prop
  have hmulS_m : Measurable (fun X : RMat (r + (p + m')) (r + q) => toRMat (Matrix.of X * S)) := by
    refine Measurable.of_eval fun i => Measurable.of_eval fun j => ?_
    simp only [toRMat_apply, Matrix.mul_apply, Matrix.of_apply]; fun_prop
  have hT1_m : Measurable T₁ :=
    measurable_fst.prodMk ((measurable_fst.comp measurable_snd).prodMk
      ((measurable_fst.comp (measurable_snd.comp measurable_snd)).prodMk
        ((measurable_snd.comp (measurable_snd.comp measurable_snd)).sub
          (hshiftA_m.comp (measurable_fst.comp measurable_snd)))))
  have hT2_m : Measurable T₂ :=
    measurable_fst.prodMk
      (((hmulDP_m.comp (measurable_fst.comp measurable_snd)).add
        (hshiftW2_m.comp (measurable_snd.comp (measurable_snd.comp measurable_snd)))).prodMk
       (measurable_snd.comp measurable_snd))
  have hT3_m : Measurable T₃ :=
    ((hmulS_m.comp measurable_fst).add
      (hshiftW1_m.comp (measurable_fst.comp (measurable_snd.comp measurable_snd)))).prodMk
      measurable_snd
  have hT1_vol : Measure.map T₁ volume = volume := by
    let T1_inner : RMat r n' × (RMat (p + m') (p + k') × RMat (p + k') n') →
        RMat r n' × (RMat (p + m') (p + k') × RMat (p + k') n') :=
      fun y => (y.1, (y.2.1, y.2.2 - shiftA y.1))
    have hT1_in_m : Measurable T1_inner :=
      measurable_fst.prodMk ((measurable_fst.comp measurable_snd).prodMk
        ((measurable_snd.comp measurable_snd).sub (hshiftA_m.comp measurable_fst)))
    have hT1_in : Measure.map T1_inner volume = volume := by
      refine Measure.ext fun s hs => ?_
      rw [Measure.map_apply hT1_in_m hs]
      change (volume.prod volume) (T1_inner ⁻¹' s) = (volume.prod volume) s
      rw [Measure.prod_apply (hT1_in_m hs), Measure.prod_apply hs]
      refine lintegral_congr fun a => ?_
      have hpre : (Prod.mk a) ⁻¹' (T1_inner ⁻¹' s) =
          (Prod.map id (fun X => X - shiftA a)) ⁻¹' ((Prod.mk a) ⁻¹' s) := rfl
      rw [hpre]
      have hmp := (MeasurePreserving.id (volume : Measure (RMat (p + m') (p + k')))).prod
        (measurePreserving_sub_right (volume : Measure (RMat (p + k') n')) (shiftA a))
      exact hmp.measure_preimage (hs.preimage measurable_prodMk_left).nullMeasurableSet
    change Measure.map (Prod.map id T1_inner) (volume.prod volume) = volume.prod volume
    have hsf : SigmaFinite (Measure.map T1_inner volume) := by rw [hT1_in]; infer_instance
    rw [← Measure.map_prod_map _ _ measurable_id hT1_in_m, Measure.map_id, hT1_in]
  have hT2_vol : Measure.map T₂ volume = ENNReal.ofReal |(DP.det)⁻¹| ^ n' • volume := by
    let T2_inner : RMat r n' × (RMat (p + m') (p + k') × RMat (p + k') n') →
        RMat r n' × (RMat (p + m') (p + k') × RMat (p + k') n') :=
      fun y => (toRMat (DP * Matrix.of y.1) + shiftW₂ y.2.2, y.2)
    have hT2_in_m : Measurable T2_inner :=
      ((hmulDP_m.comp measurable_fst).add (hshiftW2_m.comp (measurable_snd.comp measurable_snd))).prodMk
        measurable_snd
    have hT2_in : Measure.map T2_inner volume = ENNReal.ofReal |(DP.det)⁻¹| ^ n' • volume := by
      refine Measure.ext fun s hs => ?_
      rw [Measure.map_apply hT2_in_m hs, Measure.smul_apply, smul_eq_mul]
      change (volume.prod volume) (T2_inner ⁻¹' s) = _ * (volume.prod volume) s
      rw [Measure.prod_apply_symm (hT2_in_m hs), Measure.prod_apply_symm hs,
        ← lintegral_const_mul' _ _ (ENNReal.pow_lt_top ENNReal.ofReal_lt_top).ne]
      refine lintegral_congr fun y => ?_
      have hpre : (Prod.mk · y) ⁻¹' (T2_inner ⁻¹' s) =
          (fun X : RMat r n' => toRMat (DP * Matrix.of X)) ⁻¹'
            ((fun X => X + shiftW₂ y.2) ⁻¹' ((Prod.mk · y) ⁻¹' s)) := rfl
      rw [hpre]
      have hmeas_s : MeasurableSet ((Prod.mk · y) ⁻¹' s) := hs.preimage measurable_prodMk_right
      have hmeas_sh : MeasurableSet ((fun X => X + shiftW₂ y.2) ⁻¹' ((Prod.mk · y) ⁻¹' s)) :=
        hmeas_s.preimage (measurable_id.add measurable_const)
      have hmap := congrArg (fun μ => μ ((fun X => X + shiftW₂ y.2) ⁻¹' ((Prod.mk · y) ⁻¹' s)))
        (rrr_map_mul_left_volume (H := n') DP hDP)
      rw [Measure.map_apply hmulDP_m hmeas_sh, Measure.smul_apply, smul_eq_mul] at hmap
      rw [hmap]
      congr 1
      exact (measurePreserving_add_right volume (shiftW₂ y.2)).measure_preimage hmeas_s.nullMeasurableSet
    have hsf_c : SigmaFinite (ENNReal.ofReal |(DP.det)⁻¹| ^ n' •
        (volume : Measure (RMat r n' × (RMat (p + m') (p + k') × RMat (p + k') n')))) :=
      rrr_sigmaFinite_ofReal_pow_smul _ _ _
    have hsf : SigmaFinite (Measure.map T2_inner volume) := by rw [hT2_in]; exact hsf_c
    change Measure.map (Prod.map id T2_inner) (volume.prod volume) = _
    rw [← Measure.map_prod_map _ _ measurable_id hT2_in_m, Measure.map_id, hT2_in,
      rrr_prod_smul_measure]
    rfl
  have hT3_vol : Measure.map T₃ volume = ENNReal.ofReal |(S.det)⁻¹| ^ (r + (p + m')) • volume := by
    refine Measure.ext fun s hs => ?_
    rw [Measure.map_apply hT3_m hs, Measure.smul_apply, smul_eq_mul]
    change (volume.prod volume) (T₃ ⁻¹' s) = _ * (volume.prod volume) s
    rw [Measure.prod_apply_symm (hT3_m hs), Measure.prod_apply_symm hs,
      ← lintegral_const_mul' _ _ (ENNReal.pow_lt_top ENNReal.ofReal_lt_top).ne]
    refine lintegral_congr fun y => ?_
    have hpre : (Prod.mk · y) ⁻¹' (T₃ ⁻¹' s) =
        (fun X : RMat (r + (p + m')) (r + q) => toRMat (Matrix.of X * S)) ⁻¹'
          ((fun X => X + shiftW₁ y.2.1) ⁻¹' ((Prod.mk · y) ⁻¹' s)) := rfl
    rw [hpre]
    have hmeas_s : MeasurableSet ((Prod.mk · y) ⁻¹' s) := hs.preimage measurable_prodMk_right
    have hmeas_sh : MeasurableSet ((fun X => X + shiftW₁ y.2.1) ⁻¹' ((Prod.mk · y) ⁻¹' s)) :=
      hmeas_s.preimage (measurable_id.add measurable_const)
    have hmap := congrArg (fun μ => μ ((fun X => X + shiftW₁ y.2.1) ⁻¹' ((Prod.mk · y) ⁻¹' s)))
      (map_mul_right_volume (m := r + (p + m')) S hS)
    rw [Measure.map_apply hmulS_m hmeas_sh, Measure.smul_apply, smul_eq_mul] at hmap
    rw [hmap]
    congr 1
    exact (measurePreserving_add_right volume (shiftW₁ y.2.1)).measure_preimage hmeas_s.nullMeasurableSet
  rw [hcomp, ← Measure.map_map hT3_m (hT2_m.comp hT1_m), ← Measure.map_map hT2_m hT1_m,
    hT1_vol, hT2_vol, Measure.map_smul, hT3_vol, smul_smul, mul_comm]
  · exact hT3_m.aemeasurable

/-- Two-sided loss comparison between `rrrBlockLoss D (u, v)` and `rrrDecoupledLoss (rrrTriMap D u v)`
whenever `S = 1 + ΔA₁₁` is invertible and `frobSq (S⁻¹ * ΔA₁₂) ≤ 1/4`. -/
private theorem rrr_loss_comparison {r p q m' n' k' : ℕ} (D : Matrix (Fin r) (Fin r) ℝ)
    (u : RRRPassive r p q n' k') (v : RRRActive r p q m' n' k')
    (hS : IsUnit ((1 : Matrix (Fin (r + q)) (Fin (r + q)) ℝ) + Matrix.of u.1.1).det)
    (hC : frobSq (((1 : Matrix (Fin (r + q)) (Fin (r + q)) ℝ) + Matrix.of u.1.1)⁻¹ *
      vcatMat (Matrix.of v.2.1) (Matrix.of u.1.2)) ≤ 1 / 4) :
    (1 / 4 : ℝ) * rrrDecoupledLoss (rrrTriMap D u v) ≤ rrrBlockLoss D (u, v) ∧
    rrrBlockLoss D (u, v) ≤ 2 * rrrDecoupledLoss (rrrTriMap D u v) := by
  set S := (1 : Matrix (Fin (r + q)) (Fin (r + q)) ℝ) + Matrix.of u.1.1
  set ΔA₁₂ := vcatMat (Matrix.of v.2.1) (Matrix.of u.1.2)
  set B₂ := vcatMat (Matrix.of u.2.2) (rrrBaseBlo p m' k' + Matrix.of v.2.2.1)
  set B₀L := rrrBaseBL (p := p) (q := q) (m' := m') D
  set M_L := Matrix.of v.1 * S + B₀L * Matrix.of u.1.1 + B₂ * Matrix.of u.2.1
  set M_R := (B₀L + Matrix.of v.1) * ΔA₁₂ + B₂ * Matrix.of v.2.2.2
  set C := S⁻¹ * ΔA₁₂
  set Ã₂₂ := Matrix.of v.2.2.2 - Matrix.of u.2.1 * C
  set R := B₀L * C + B₂ * Ã₂₂
  have hMR : M_R = M_L * C + R :=
    rrr_schur_block_right B₀L (Matrix.of v.1) B₂ (Matrix.of u.1.1) hS ΔA₁₂ (Matrix.of u.2.1)
      (Matrix.of v.2.2.2)
  have hR_frob : frobSq R =
      frobSq (Matrix.of (rrrTriMap D u v).2.1) + rrrLoLoss (rrrTriMap D u v).2.2 := by
    have hsplit : R = vcatMat
        (D * topLeftBlock S⁻¹ * Matrix.of v.2.1 + D * topRightBlock S⁻¹ * Matrix.of u.1.2 +
          Matrix.of u.2.2 * Ã₂₂)
        ((rrrBaseBlo p m' k' + Matrix.of v.2.2.1) * Ã₂₂) := by
      dsimp only [R, B₀L, C, ΔA₁₂, B₂]
      rw [rrrBaseBL_mul_inv_vcat, vcatMat_mul, vcatMat_add, zero_add]
    rw [hsplit, frobSq_vcatMat]
    rfl
  have hdec : rrrDecoupledLoss (rrrTriMap D u v) = frobSq M_L + frobSq R := by
    unfold rrrDecoupledLoss
    rw [hR_frob]; rfl
  have hblk : rrrBlockLoss D (u, v) = frobSq M_L + frobSq M_R := rfl
  rw [hdec, hblk]
  constructor
  · have hR_eq : R = M_R - M_L * C := eq_sub_of_add_eq' hMR.symm
    have h1 : frobSq R ≤ 2 * frobSq M_R + 2 * frobSq (M_L * C) := by
      rw [hR_eq]; exact frobSq_sub_le M_R (M_L * C)
    have h2 : frobSq (M_L * C) ≤ frobSq M_L * frobSq C := frobSq_mul_le _ _
    have h3 : frobSq M_L * frobSq C ≤ frobSq M_L * (1 / 4) :=
      mul_le_mul_of_nonneg_left hC (frobSq_nonneg M_L)
    linarith [frobSq_nonneg M_L, frobSq_nonneg M_R]
  · have h1 : frobSq M_R ≤ 2 * frobSq (M_L * C) + 2 * frobSq R := by
      rw [hMR]; exact frobSq_add_le _ _
    have h2 : frobSq (M_L * C) ≤ frobSq M_L * frobSq C := frobSq_mul_le _ _
    have h3 : frobSq M_L * frobSq C ≤ frobSq M_L * (1 / 4) :=
      mul_le_mul_of_nonneg_left hC (frobSq_nonneg M_L)
    linarith [frobSq_nonneg M_L, frobSq_nonneg R]

/-- Continuity at `0` produces uniform radius `δ₀ > 0`, positive constants `η, c_lo, C_hi`, and
Jacobian bounds for `rrrTriMap D u` across all `0 < δ ≤ δ₀`. -/
private theorem exists_good_rrr_radius {r p q m' n' k' : ℕ}
    (D : Matrix (Fin r) (Fin r) ℝ) (hD : IsUnit D.det) :
    ∃ (δ₀ η c_lo C_hi : ℝ), 0 < δ₀ ∧ 0 < η ∧ η ≤ 1 ∧ 0 < c_lo ∧ c_lo ≤ 1 / 2 ∧ 2 ≤ C_hi ∧
      ∀ δ : ℝ, 0 < δ → δ ≤ δ₀ → ∀ u ∈ rrrPassiveBox (r := r) (p := p) (q := q) (n' := n') (k' := k') δ,
        let S := (1 : Matrix (Fin (r + q)) (Fin (r + q)) ℝ) + Matrix.of u.1.1
        let DP := D * topLeftBlock S⁻¹
        IsUnit S.det ∧ IsUnit DP.det ∧
        (1 / 2 : ℝ) ≤ |S.det| ∧ |S.det| ≤ 2 ∧
        (|D.det| / 2) ≤ |DP.det| ∧ |DP.det| ≤ 2 * |D.det| ∧
        (∀ v ∈ rrrActiveBox (r := r) (p := p) (q := q) (m' := m') (n' := n') (k' := k') δ δ δ,
          frobSq (S⁻¹ * vcatMat (Matrix.of v.2.1) (Matrix.of u.1.2)) ≤ 1 / 4 ∧
          rrrTriMap (m' := m') D u v ∈ rrrActiveBox (C_hi * δ) (C_hi * δ) (2 * δ)) ∧
        (u ∈ rrrPassiveBox (η * δ) →
          ∀ w ∈ rrrActiveBox (r := r) (p := p) (q := q) (m' := m') (n' := n') (k' := k')
              (c_lo * δ) (c_lo * δ) (c_lo * δ),
            ∃ v ∈ rrrActiveBox δ δ δ,
              frobSq (S⁻¹ * vcatMat (Matrix.of v.2.1) (Matrix.of u.1.2)) ≤ 1 / 4 ∧
              rrrTriMap (m' := m') D u v = w) := by
  have hD_ne : D.det ≠ 0 := isUnit_iff_ne_zero.mp hD
  have hD_abs_pos : 0 < |D.det| := abs_pos.mpr hD_ne
  obtain ⟨B_D, hB_D_def⟩ : ∃ x : ℝ,
      x = 1 + (∑ i : Fin r, ∑ j : Fin r, |D i j|) + ∑ i : Fin r, ∑ j : Fin r, |D⁻¹ i j| :=
    ⟨_, rfl⟩
  have hB_D_ge1 : 1 ≤ B_D := by
    rw [hB_D_def]
    have h1 : 0 ≤ ∑ i : Fin r, ∑ j : Fin r, |D i j| :=
      Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => abs_nonneg _
    have h2 : 0 ≤ ∑ i : Fin r, ∑ j : Fin r, |D⁻¹ i j| :=
      Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => abs_nonneg _
    linarith
  have hD_entry : ∀ i j, |D i j| ≤ B_D := fun i j => by
    have h1 : |D i j| ≤ ∑ j' : Fin r, |D i j'| :=
      Finset.single_le_sum (f := fun j' => |D i j'|) (fun _ _ => abs_nonneg _) (Finset.mem_univ j)
    have h2 : ∑ j' : Fin r, |D i j'| ≤ ∑ i' : Fin r, ∑ j' : Fin r, |D i' j'| :=
      Finset.single_le_sum (f := fun i' => ∑ j' : Fin r, |D i' j'|)
        (fun _ _ => Finset.sum_nonneg fun _ _ => abs_nonneg _) (Finset.mem_univ i)
    have h3 : 0 ≤ ∑ i' : Fin r, ∑ j' : Fin r, |D⁻¹ i' j'| :=
      Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => abs_nonneg _
    rw [hB_D_def]; linarith
  have hDinv_entry : ∀ i j, |D⁻¹ i j| ≤ B_D := fun i j => by
    have h1 : |D⁻¹ i j| ≤ ∑ j' : Fin r, |D⁻¹ i j'| :=
      Finset.single_le_sum (f := fun j' => |D⁻¹ i j'|) (fun _ _ => abs_nonneg _) (Finset.mem_univ j)
    have h2 : ∑ j' : Fin r, |D⁻¹ i j'| ≤ ∑ i' : Fin r, ∑ j' : Fin r, |D⁻¹ i' j'| :=
      Finset.single_le_sum (f := fun i' => ∑ j' : Fin r, |D⁻¹ i' j'|)
        (fun _ _ => Finset.sum_nonneg fun _ _ => abs_nonneg _) (Finset.mem_univ i)
    have h3 : 0 ≤ ∑ i' : Fin r, ∑ j' : Fin r, |D i' j'| :=
      Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => abs_nonneg _
    rw [hB_D_def]; linarith
  obtain ⟨dimM, hdimM_def⟩ : ∃ x : ℝ,
      x = (r : ℝ) + (p : ℝ) + (q : ℝ) + (m' : ℝ) + (n' : ℝ) + (k' : ℝ) + 2 :=
    ⟨_, rfl⟩
  have hdimM_ge2 : 2 ≤ dimM := by
    rw [hdimM_def]
    linarith [Nat.cast_nonneg (α := ℝ) r, Nat.cast_nonneg (α := ℝ) p, Nat.cast_nonneg (α := ℝ) q,
      Nat.cast_nonneg (α := ℝ) m', Nat.cast_nonneg (α := ℝ) n', Nat.cast_nonneg (α := ℝ) k']
  let Sfn : RMat (r + q) (r + q) → Matrix (Fin (r + q)) (Fin (r + q)) ℝ :=
    fun Δ => (1 : Matrix (Fin (r + q)) (Fin (r + q)) ℝ) + Matrix.of Δ
  have hS_cont : Continuous Sfn := continuous_const.add continuous_id
  have hS0 : Sfn 0 = 1 := by ext i j; simp [Sfn]
  have hSinv_contAt : ContinuousAt (fun Δ => (Sfn Δ)⁻¹) 0 := by
    have heq : (fun Δ => (Sfn Δ)⁻¹) = fun Δ => ((Sfn Δ).det)⁻¹ • (Sfn Δ).adjugate := by
      funext Δ; ext i j; simp [Matrix.inv_def, Ring.inverse_eq_inv]
    rw [heq]
    have h0 : (Sfn 0).det ≠ 0 := by rw [hS0, Matrix.det_one]; exact one_ne_zero
    exact (hS_cont.matrix_det.continuousAt.inv₀ h0).smul hS_cont.matrix_adjugate.continuousAt
  have hSinv_entry_contAt : ∀ i j, ContinuousAt (fun Δ => (Sfn Δ)⁻¹ i j) 0 := fun i j =>
    ((continuous_apply j).comp (continuous_apply i) : Continuous fun M : Matrix (Fin (r + q)) (Fin (r + q)) ℝ => M i j).continuousAt.comp
      hSinv_contAt
  let DPfn : RMat (r + q) (r + q) → Matrix (Fin r) (Fin r) ℝ :=
    fun Δ => D * topLeftBlock (Sfn Δ)⁻¹
  have hDP_entry_contAt : ∀ i j, ContinuousAt (fun Δ => DPfn Δ i j) 0 := fun i j => by
    let f : Matrix (Fin (r + q)) (Fin (r + q)) ℝ → ℝ :=
      fun M => ∑ t : Fin r, D i t * M (Fin.castAdd q t) (Fin.castAdd q j)
    have hf : Continuous f := continuous_finsetSum _ fun t _ =>
      continuous_const.mul ((continuous_apply (Fin.castAdd q j)).comp
        (continuous_apply (Fin.castAdd q t)))
    have hfn : (fun Δ => DPfn Δ i j) = fun Δ => f (Sfn Δ)⁻¹ := by
      funext Δ; simp [f, DPfn, Matrix.mul_apply, topLeftBlock]
    rw [hfn]
    exact hf.continuousAt.comp hSinv_contAt
  have hDP_contAt : ContinuousAt DPfn 0 :=
    continuousAt_pi.mpr fun i => continuousAt_pi.mpr fun j => hDP_entry_contAt i j
  have hSinv0 : (Sfn 0)⁻¹ = 1 := by rw [hS0]; simp
  have hP11_0 : topLeftBlock (Sfn 0)⁻¹ = 1 := by
    rw [hSinv0]
    ext i j; simp [topLeftBlock, Matrix.one_apply, Fin.ext_iff]
  have hDP0 : DPfn 0 = D := by simp [DPfn, hP11_0]
  have hDPinv_contAt : ContinuousAt (fun Δ => (DPfn Δ)⁻¹) 0 := by
    have heq : (fun Δ => (DPfn Δ)⁻¹) = fun Δ => ((DPfn Δ).det)⁻¹ • (DPfn Δ).adjugate := by
      funext Δ; ext i j; simp [Matrix.inv_def, Ring.inverse_eq_inv]
    rw [heq]
    have hdet_c : ContinuousAt (fun Δ => (DPfn Δ).det) 0 :=
      (continuous_id.matrix_det).continuousAt.comp hDP_contAt
    have hadj_c : ContinuousAt (fun Δ => (DPfn Δ).adjugate) 0 :=
      (continuous_id.matrix_adjugate).continuousAt.comp hDP_contAt
    have h0 : (DPfn 0).det ≠ 0 := by rw [hDP0]; exact hD_ne
    exact (hdet_c.inv₀ h0).smul hadj_c
  have hDPinv_entry_contAt : ∀ i j, ContinuousAt (fun Δ => (DPfn Δ)⁻¹ i j) 0 := fun i j =>
    ((continuous_apply j).comp (continuous_apply i) : Continuous fun M : Matrix (Fin r) (Fin r) ℝ => M i j).continuousAt.comp
      hDPinv_contAt
  have hnhds_Sdet : ∀ᶠ Δ in nhds (0 : RMat (r + q) (r + q)),
      (1 / 2 : ℝ) < |(Sfn Δ).det| ∧ |(Sfn Δ).det| < 2 := by
    have hc : ContinuousAt (fun Δ => |(Sfn Δ).det|) 0 := hS_cont.matrix_det.continuousAt.abs
    have hv : |(Sfn 0).det| = 1 := by rw [hS0, Matrix.det_one, abs_one]
    have hm : Set.Ioo (1 / 2 : ℝ) 2 ∈ nhds |(Sfn 0).det| := by
      rw [hv]; exact Ioo_mem_nhds (by norm_num) (by norm_num)
    exact hc.preimage_mem_nhds hm
  have hnhds_DPdet : ∀ᶠ Δ in nhds (0 : RMat (r + q) (r + q)),
      |D.det| / 2 < |(DPfn Δ).det| ∧ |(DPfn Δ).det| < 2 * |D.det| := by
    have hc : ContinuousAt (fun Δ => |(DPfn Δ).det|) 0 :=
      ((continuous_id.matrix_det).continuousAt.comp hDP_contAt).abs
    have hv : |(DPfn 0).det| = |D.det| := by rw [hDP0]
    have hm : Set.Ioo (|D.det| / 2) (2 * |D.det|) ∈ nhds |(DPfn 0).det| := by
      rw [hv]; exact Ioo_mem_nhds (by linarith) (by linarith)
    exact hc.preimage_mem_nhds hm
  have hnhds_Sinv : ∀ᶠ Δ in nhds (0 : RMat (r + q) (r + q)),
      ∀ i j, |(Sfn Δ)⁻¹ i j| < 2 := by
    rw [Filter.eventually_all]; intro i; rw [Filter.eventually_all]; intro j
    have hc : ContinuousAt (fun Δ => |(Sfn Δ)⁻¹ i j|) 0 := (hSinv_entry_contAt i j).abs
    have hv : |(Sfn 0)⁻¹ i j| < 2 := by
      rw [hSinv0]; by_cases hij : i = j <;> simp [Matrix.one_apply, hij]
    exact hc.preimage_mem_nhds (Iio_mem_nhds hv)
  have hnhds_DPentry : ∀ᶠ Δ in nhds (0 : RMat (r + q) (r + q)),
      ∀ i j, |DPfn Δ i j| < 2 * B_D := by
    rw [Filter.eventually_all]; intro i; rw [Filter.eventually_all]; intro j
    have hc : ContinuousAt (fun Δ => |DPfn Δ i j|) 0 := (hDP_entry_contAt i j).abs
    have hv : |DPfn 0 i j| < 2 * B_D := by rw [hDP0]; linarith [hD_entry i j]
    exact hc.preimage_mem_nhds (Iio_mem_nhds hv)
  have hnhds_DPinv : ∀ᶠ Δ in nhds (0 : RMat (r + q) (r + q)),
      ∀ i j, |(DPfn Δ)⁻¹ i j| < 2 * B_D := by
    rw [Filter.eventually_all]; intro i; rw [Filter.eventually_all]; intro j
    have hc : ContinuousAt (fun Δ => |(DPfn Δ)⁻¹ i j|) 0 := (hDPinv_entry_contAt i j).abs
    have hv : |(DPfn 0)⁻¹ i j| < 2 * B_D := by rw [hDP0]; linarith [hDinv_entry i j]
    exact hc.preimage_mem_nhds (Iio_mem_nhds hv)
  have hnhds_P12 : ∀ᶠ Δ in nhds (0 : RMat (r + q) (r + q)),
      ∀ i j, |topRightBlock (Sfn Δ)⁻¹ i j| < 1 / (8 * dimM ^ 3 * B_D ^ 2) := by
    rw [Filter.eventually_all]; intro i; rw [Filter.eventually_all]; intro j
    have hc : ContinuousAt (fun Δ => |topRightBlock (Sfn Δ)⁻¹ i j|) 0 :=
      (hSinv_entry_contAt (Fin.castAdd q i) (Fin.natAdd r j)).abs
    have hv : |topRightBlock (Sfn 0)⁻¹ i j| = 0 := by
      rw [hSinv0]
      have hne : Fin.castAdd q i ≠ Fin.natAdd r j := by
        intro h; have := congrArg Fin.val h; dsimp at this; omega
      simp [topRightBlock, Matrix.one_apply, hne]
    have hm : Set.Iio (1 / (8 * dimM ^ 3 * B_D ^ 2)) ∈ nhds |topRightBlock (Sfn 0)⁻¹ i j| := by
      rw [hv]; exact Iio_mem_nhds (by positivity)
    exact hc.preimage_mem_nhds hm
  have hall := hnhds_Sdet.and (hnhds_DPdet.and (hnhds_Sinv.and (hnhds_DPentry.and
    (hnhds_DPinv.and hnhds_P12))))
  rw [Metric.eventually_nhds_iff] at hall
  obtain ⟨r₁, hr₁, hr₁_prop⟩ := hall
  let C_hi : ℝ := 8 * dimM ^ 2 * B_D + 4
  let c_lo : ℝ := 1 / (12 * dimM ^ 2 * B_D)
  let η : ℝ := 1 / (24 * dimM ^ 3 * B_D)
  let δ₀ : ℝ := min (r₁ / 2) (1 / (32 * dimM ^ 4 * B_D ^ 2))
  have hdim_pos : 0 < dimM := by linarith only [hdimM_ge2]
  have hB_D_pos : 0 < B_D := by linarith only [hB_D_ge1]
  have hdim2_ge1 : (1 : ℝ) ≤ dimM ^ 2 := one_le_pow₀ (by linarith only [hdimM_ge2])
  have hdim3_ge1 : (1 : ℝ) ≤ dimM ^ 3 := one_le_pow₀ (by linarith only [hdimM_ge2])
  have hdim4_ge1 : (1 : ℝ) ≤ dimM ^ 4 := one_le_pow₀ (by linarith only [hdimM_ge2])
  have hBD2_ge1 : (1 : ℝ) ≤ B_D ^ 2 := one_le_pow₀ hB_D_ge1
  have hB_D_sq : B_D ≤ B_D ^ 2 := by nlinarith only [hB_D_ge1]
  have hdim_le2 : dimM ≤ dimM ^ 2 := by nlinarith only [hdimM_ge2]
  have hdim2_le3 : dimM ^ 2 ≤ dimM ^ 3 := by nlinarith only [hdimM_ge2, hdim2_ge1]
  have hdim3_le4 : dimM ^ 3 ≤ dimM ^ 4 := by nlinarith only [hdimM_ge2, hdim3_ge1]
  have hdim2_le4 : dimM ^ 2 ≤ dimM ^ 4 := hdim2_le3.trans hdim3_le4
  have hdim2_le4BD2 : dimM ^ 2 ≤ dimM ^ 4 * B_D ^ 2 :=
    hdim2_le4.trans (le_mul_of_one_le_right (by positivity) hBD2_ge1)
  have hdim2BD_le4BD2 : dimM ^ 2 * B_D ≤ dimM ^ 4 * B_D ^ 2 :=
    mul_le_mul hdim2_le4 hB_D_sq hB_D_pos.le (by positivity)
  have hC_hi : 2 ≤ C_hi := by dsimp [C_hi]; nlinarith only [hdim2_ge1, hB_D_ge1]
  have hc_lo_pos : 0 < c_lo := by positivity
  have hc_lo_le : c_lo ≤ 1 / 2 := by
    dsimp [c_lo]; rw [div_le_div_iff₀ (by positivity) (by norm_num)]
    nlinarith only [hdim2_ge1, hB_D_ge1]
  have hη_pos : 0 < η := by positivity
  have hη_le : η ≤ 1 := by
    dsimp [η]; rw [div_le_one (by positivity)]
    nlinarith only [hdim3_ge1, hB_D_ge1]
  have hδ₀_pos : 0 < δ₀ := lt_min (by linarith only [hr₁]) (by positivity)
  refine ⟨δ₀, η, c_lo, C_hi, hδ₀_pos, hη_pos, hη_le, hc_lo_pos, hc_lo_le, hC_hi,
    fun δ hδ hδle u hu => ?_⟩
  rcases hu with ⟨⟨hΔA11, hΔA12m⟩, ⟨hΔA21, hΔB2t⟩⟩
  rw [mem_matBox_iff] at hΔA11 hΔA12m hΔA21 hΔB2t
  have hδ_lt_r1 : δ < r₁ := lt_of_le_of_lt (hδle.trans (min_le_left _ _)) (by linarith)
  have hδ_bound : δ ≤ 1 / (32 * dimM ^ 4 * B_D ^ 2) := hδle.trans (min_le_right _ _)
  have hdist : dist u.1.1 0 < r₁ := by
    rw [dist_zero_right]
    refine ((pi_norm_le_iff_of_nonneg hδ.le).mpr fun i =>
      (pi_norm_le_iff_of_nonneg hδ.le).mpr fun j => by
        simpa [Real.norm_eq_abs] using hΔA11 i j).trans_lt hδ_lt_r1
  obtain ⟨⟨hSdet_lo, hSdet_hi⟩, ⟨hDPdet_lo, hDPdet_hi⟩, hSinv_b, hDP_b, hDPinv_b, hP12_b⟩ :=
    hr₁_prop hdist
  set S := Sfn u.1.1
  set DP := DPfn u.1.1
  have hSdet_ne : S.det ≠ 0 := fun h0 => by rw [h0, abs_zero] at hSdet_lo; linarith
  have hDPdet_ne : DP.det ≠ 0 := fun h0 => by rw [h0, abs_zero] at hDPdet_lo; linarith
  have hS_unit : IsUnit S.det := isUnit_iff_ne_zero.mpr hSdet_ne
  have hDP_unit : IsUnit DP.det := isUnit_iff_ne_zero.mpr hDPdet_ne
  have hr_le : (r : ℝ) ≤ dimM := by
    rw [hdimM_def]; linarith [Nat.cast_nonneg (α := ℝ) p, Nat.cast_nonneg (α := ℝ) q,
      Nat.cast_nonneg (α := ℝ) m', Nat.cast_nonneg (α := ℝ) n', Nat.cast_nonneg (α := ℝ) k']
  have hq_le : (q : ℝ) ≤ dimM := by
    rw [hdimM_def]; linarith [Nat.cast_nonneg (α := ℝ) r, Nat.cast_nonneg (α := ℝ) p,
      Nat.cast_nonneg (α := ℝ) m', Nat.cast_nonneg (α := ℝ) n', Nat.cast_nonneg (α := ℝ) k']
  have ha_le : ((r + q : ℕ) : ℝ) ≤ dimM := by
    push_cast; rw [hdimM_def]; linarith [Nat.cast_nonneg (α := ℝ) p, Nat.cast_nonneg (α := ℝ) m',
      Nat.cast_nonneg (α := ℝ) n', Nat.cast_nonneg (α := ℝ) k']
  have hpk_le : ((p + k' : ℕ) : ℝ) ≤ dimM := by
    push_cast; rw [hdimM_def]; linarith [Nat.cast_nonneg (α := ℝ) r, Nat.cast_nonneg (α := ℝ) q,
      Nat.cast_nonneg (α := ℝ) m', Nat.cast_nonneg (α := ℝ) n']
  have hn_le : (n' : ℝ) ≤ dimM := by
    rw [hdimM_def]; linarith [Nat.cast_nonneg (α := ℝ) r, Nat.cast_nonneg (α := ℝ) p,
      Nat.cast_nonneg (α := ℝ) q, Nat.cast_nonneg (α := ℝ) m', Nat.cast_nonneg (α := ℝ) k']
  have hδ_dim : 32 * dimM ^ 4 * B_D ^ 2 * δ ≤ 1 := (le_div_iff₀' (by positivity)).mp hδ_bound
  have hδ_le1 : δ ≤ 1 := by
    have h1 : (1 : ℝ) ≤ 32 * dimM ^ 4 * B_D ^ 2 := by nlinarith only [hdim4_ge1, hBD2_ge1]
    nlinarith only [hδ_dim, h1, hδ]
  have hc_lo_δ_le : c_lo * δ ≤ δ := by nlinarith only [hc_lo_le, hδ]
  have hfrobC_of : ∀ (A_top : RMat r n'), A_top ∈ matBox δ →
      frobSq (S⁻¹ * vcatMat (Matrix.of A_top) (Matrix.of u.1.2)) ≤ 1 / 4 := fun A_top hAt => by
    rw [mem_matBox_iff] at hAt
    have hA12 : ∀ i j, |(vcatMat (Matrix.of A_top) (Matrix.of u.1.2)) i j| ≤ δ := fun i j => by
      by_cases hi : i.1 < r
      · have hieq : i = Fin.castAdd q ⟨i.1, hi⟩ := by ext; rfl
        rw [hieq, vcatMat_castAdd]; exact hAt ⟨i.1, hi⟩ j
      · have hieq : i = Fin.natAdd r ⟨i.1 - r, by omega⟩ := by ext; dsimp; omega
        rw [hieq, vcatMat_natAdd]; exact hΔA12m ⟨i.1 - r, by omega⟩ j
    have hSinv_f : frobSq S⁻¹ ≤ 4 * ((r + q : ℕ) : ℝ) ^ 2 := by
      unfold frobSq
      calc ∑ i : Fin (r + q), ∑ j : Fin (r + q), (S⁻¹ i j) ^ 2
          ≤ ∑ _ : Fin (r + q), ∑ _ : Fin (r + q), (4 : ℝ) := by
            refine Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ => ?_
            have h := (hSinv_b i j).le
            have := pow_le_pow_left₀ (abs_nonneg _) h 2
            rw [sq_abs] at this; linarith
        _ = 4 * ((r + q : ℕ) : ℝ) ^ 2 := by simp [Finset.sum_const]; ring
    have hA12_f : frobSq (vcatMat (Matrix.of A_top) (Matrix.of u.1.2)) ≤
        ((r + q : ℕ) : ℝ) * n' * δ ^ 2 := by
      unfold frobSq
      calc ∑ i : Fin (r + q), ∑ j : Fin n', (vcatMat (Matrix.of A_top) (Matrix.of u.1.2) i j) ^ 2
          ≤ ∑ _ : Fin (r + q), ∑ _ : Fin n', δ ^ 2 := by
            refine Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ => ?_
            have := pow_le_pow_left₀ (abs_nonneg _) (hA12 i j) 2
            rwa [sq_abs] at this
        _ = ((r + q : ℕ) : ℝ) * n' * δ ^ 2 := by simp [Finset.sum_const]; ring
    have ha2 : ((r + q : ℕ) : ℝ) ^ 2 ≤ dimM ^ 2 := pow_le_pow_left₀ (Nat.cast_nonneg _) ha_le 2
    have han : ((r + q : ℕ) : ℝ) * n' ≤ dimM ^ 2 := by
      calc ((r + q : ℕ) : ℝ) * n' ≤ dimM * dimM := mul_le_mul ha_le hn_le (Nat.cast_nonneg _) (by linarith)
        _ = dimM ^ 2 := (sq dimM).symm
    have hdim2δ : dimM ^ 2 * δ ≤ 1 / 32 := by
      have h1 : 32 * dimM ^ 2 * δ ≤ 32 * dimM ^ 4 * B_D ^ 2 * δ := by
        nlinarith only [hdim2_le4BD2, hδ]
      linarith only [h1, hδ_dim]
    calc frobSq (S⁻¹ * vcatMat (Matrix.of A_top) (Matrix.of u.1.2))
        ≤ frobSq S⁻¹ * frobSq (vcatMat (Matrix.of A_top) (Matrix.of u.1.2)) := frobSq_mul_le _ _
      _ ≤ (4 * ((r + q : ℕ) : ℝ) ^ 2) * (((r + q : ℕ) : ℝ) * n' * δ ^ 2) :=
          mul_le_mul hSinv_f hA12_f (frobSq_nonneg _) (by positivity)
      _ ≤ (4 * dimM ^ 2) * (dimM ^ 2 * δ ^ 2) := by
          refine mul_le_mul (by linarith) (mul_le_mul_of_nonneg_right han (sq_nonneg δ))
            (by positivity) (by positivity)
      _ = 4 * (dimM ^ 2 * δ) ^ 2 := by ring
      _ ≤ 4 * (1 / 32 : ℝ) ^ 2 := by
          have := pow_le_pow_left₀ (by positivity) hdim2δ 2; linarith
      _ ≤ 1 / 4 := by norm_num
  have hB0L_entry : ∀ i j, |rrrBaseBL (p := p) (q := q) (m' := m') D i j| ≤ B_D := fun i j => by
    unfold rrrBaseBL
    by_cases hi : i.1 < r
    · have hieq : i = Fin.castAdd (p + m') ⟨i.1, hi⟩ := by ext; rfl
      rw [hieq, vcatMat_castAdd]
      by_cases hj : j.1 < r
      · have hjeq : j = Fin.castAdd q ⟨j.1, hj⟩ := by ext; rfl
        rw [hjeq, hcatMat_castAdd]; exact hD_entry _ _
      · have hjeq : j = Fin.natAdd r ⟨j.1 - r, by omega⟩ := by ext; dsimp; omega
        rw [hjeq, hcatMat_natAdd]; simp; linarith
    · have hieq : i = Fin.natAdd r ⟨i.1 - r, by omega⟩ := by ext; dsimp; omega
      rw [hieq, vcatMat_natAdd]; simp; linarith
  have hBlo_entry : ∀ i j, |rrrBaseBlo p m' k' i j| ≤ 1 := fun i j => by
    unfold rrrBaseBlo
    by_cases hi : i.1 < p
    · have hieq : i = Fin.castAdd m' ⟨i.1, hi⟩ := by ext; rfl
      rw [hieq, vcatMat_castAdd]
      by_cases hj : j.1 < p
      · have hjeq : j = Fin.castAdd k' ⟨j.1, hj⟩ := by ext; rfl
        rw [hjeq, hcatMat_castAdd]
        by_cases hij : (⟨i.1, hi⟩ : Fin p) = ⟨j.1, hj⟩ <;> simp [Matrix.one_apply, hij]
      · have hjeq : j = Fin.natAdd p ⟨j.1 - p, by omega⟩ := by ext; dsimp; omega
        rw [hjeq, hcatMat_natAdd]; simp
    · have hieq : i = Fin.natAdd p ⟨i.1 - p, by omega⟩ := by ext; dsimp; omega
      rw [hieq, vcatMat_natAdd]; simp
  refine ⟨hS_unit, hDP_unit, hSdet_lo.le, hSdet_hi.le, hDPdet_lo.le, hDPdet_hi.le, ?_, ?_⟩
  · intro ⟨B1, A12t, B2lo, A22⟩ ⟨hB1, hA12t, hB2lo, hA22⟩
    dsimp only at hB1 hA12t hB2lo hA22
    refine ⟨hfrobC_of A12t hA12t, ?_⟩
    rw [mem_matBox_iff] at hB1 hA12t hB2lo hA22
    have hA12_b : ∀ i j, |(vcatMat (Matrix.of A12t) (Matrix.of u.1.2)) i j| ≤ δ := fun i j => by
      by_cases hi : i.1 < r
      · have hieq : i = Fin.castAdd q ⟨i.1, hi⟩ := by ext; rfl
        rw [hieq, vcatMat_castAdd]; exact hA12t _ _
      · have hieq : i = Fin.natAdd r ⟨i.1 - r, by omega⟩ := by ext; dsimp; omega
        rw [hieq, vcatMat_natAdd]; exact hΔA12m _ _
    have hC_b : ∀ i j, |(S⁻¹ * vcatMat (Matrix.of A12t) (Matrix.of u.1.2)) i j| ≤ 2 * dimM * δ :=
      fun i j => by
        have h1 := rrr_matMul_entry_abs_le S⁻¹ (vcatMat (Matrix.of A12t) (Matrix.of u.1.2))
          (by norm_num) (fun a b => (hSinv_b a b).le) hA12_b i j
        have : ((r + q : ℕ) : ℝ) * 2 * δ ≤ 2 * dimM * δ := by nlinarith only [ha_le, hδ]
        exact h1.trans this
    have hA21C_b : ∀ i j,
        |(Matrix.of u.2.1 * (S⁻¹ * vcatMat (Matrix.of A12t) (Matrix.of u.1.2))) i j| ≤ δ :=
      fun i j => by
        have h1 := rrr_matMul_entry_abs_le (Matrix.of u.2.1)
          (S⁻¹ * vcatMat (Matrix.of A12t) (Matrix.of u.1.2)) hδ.le hΔA21 hC_b i j
        have h2 : ((r + q : ℕ) : ℝ) * δ * (2 * dimM * δ) ≤ 2 * dimM ^ 2 * δ ^ 2 := by
          have : 0 ≤ 2 * dimM * δ ^ 2 := by positivity
          have hha : ((r + q : ℕ) : ℝ) * (2 * dimM * δ ^ 2) ≤ dimM * (2 * dimM * δ ^ 2) :=
            mul_le_mul_of_nonneg_right ha_le this
          linarith only [hha]
        have h3 : 2 * dimM ^ 2 * δ ≤ 1 := by
          have h3a : 32 * dimM ^ 2 * δ ≤ 32 * dimM ^ 4 * B_D ^ 2 * δ := by
            nlinarith only [hdim2_le4BD2, hδ]
          linarith only [h3a, hδ_dim]
        nlinarith only [h1, h2, h3, hδ]
    have hÃ22_b : ∀ i j,
        |(Matrix.of A22 - Matrix.of u.2.1 * (S⁻¹ * vcatMat (Matrix.of A12t) (Matrix.of u.1.2))) i j| ≤
          2 * δ := fun i j => by
      rw [Matrix.sub_apply, Matrix.of_apply]
      exact (abs_sub _ _).trans (by linarith only [hA22 i j, hA21C_b i j])
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [mem_matBox_iff]; intro i j
      simp only [rrrTriMap, toRMat_apply, Matrix.add_apply]
      have hS_b : ∀ a b, |S a b| ≤ 2 := fun a b => by
        dsimp only [S, Sfn]; rw [Matrix.add_apply, Matrix.of_apply]
        have h1 : |(1 : Matrix (Fin (r + q)) (Fin (r + q)) ℝ) a b| ≤ 1 := by
          by_cases hab : a = b <;> simp [Matrix.one_apply, hab]
        exact (abs_add_le _ _).trans (by linarith only [h1, hΔA11 a b, hδ_le1])
      have ht1 : |(Matrix.of B1 * S) i j| ≤ 2 * dimM * δ := by
        have h := rrr_matMul_entry_abs_le (Matrix.of B1) S hδ.le hB1 hS_b i j
        nlinarith only [h, ha_le, hδ]
      have ht2 : |(rrrBaseBL (p := p) (q := q) (m' := m') D * Matrix.of u.1.1) i j| ≤ dimM * B_D * δ := by
        have h := rrr_matMul_entry_abs_le (rrrBaseBL (p := p) (q := q) (m' := m') D)
          (Matrix.of u.1.1) (by linarith only [hB_D_ge1]) hB0L_entry hΔA11 i j
        have : ((r + q : ℕ) : ℝ) * (B_D * δ) ≤ dimM * (B_D * δ) :=
          mul_le_mul_of_nonneg_right ha_le (by positivity)
        linarith only [h, this]
      have hB2_b : ∀ a b, |(vcatMat (Matrix.of u.2.2) (rrrBaseBlo p m' k' + Matrix.of B2lo)) a b| ≤ 2 :=
        fun a b => by
          by_cases ha : a.1 < r
          · have haeq : a = Fin.castAdd (p + m') ⟨a.1, ha⟩ := by ext; rfl
            rw [haeq, vcatMat_castAdd, Matrix.of_apply]; linarith only [hΔB2t ⟨a.1, ha⟩ b, hδ_le1]
          · have haeq : a = Fin.natAdd r ⟨a.1 - r, by omega⟩ := by ext; dsimp; omega
            rw [haeq, vcatMat_natAdd, Matrix.add_apply, Matrix.of_apply]
            exact (abs_add_le _ _).trans (by linarith only [hBlo_entry ⟨a.1 - r, by omega⟩ b, hB2lo ⟨a.1 - r, by omega⟩ b, hδ_le1])
      have ht3 : |(vcatMat (Matrix.of u.2.2) (rrrBaseBlo p m' k' + Matrix.of B2lo) *
          Matrix.of u.2.1) i j| ≤ 2 * dimM * δ := by
        have h := rrr_matMul_entry_abs_le
          (vcatMat (Matrix.of u.2.2) (rrrBaseBlo p m' k' + Matrix.of B2lo))
          (Matrix.of u.2.1) (by norm_num) hB2_b hΔA21 i j
        nlinarith only [h, hpk_le, hδ]
      have htri : |(Matrix.of B1 * S) i j + (rrrBaseBL (p := p) (q := q) (m' := m') D * Matrix.of u.1.1) i j +
          (vcatMat (Matrix.of u.2.2) (rrrBaseBlo p m' k' + Matrix.of B2lo) * Matrix.of u.2.1) i j| ≤
          (4 * dimM + dimM * B_D) * δ := by
        have h1 := abs_add_le ((Matrix.of B1 * S) i j + (rrrBaseBL (p := p) (q := q) (m' := m') D * Matrix.of u.1.1) i j)
          ((vcatMat (Matrix.of u.2.2) (rrrBaseBlo p m' k' + Matrix.of B2lo) * Matrix.of u.2.1) i j)
        have h2 := abs_add_le ((Matrix.of B1 * S) i j) ((rrrBaseBL (p := p) (q := q) (m' := m') D * Matrix.of u.1.1) i j)
        linarith only [h1, h2, ht1, ht2, ht3]
      have hcoef : 4 * dimM + dimM * B_D ≤ C_hi := by
        dsimp [C_hi]; nlinarith only [hdim_le2, hB_D_ge1, hdimM_ge2]
      exact htri.trans (mul_le_mul_of_nonneg_right hcoef hδ.le)
    · rw [mem_matBox_iff]; intro i j
      simp only [rrrTriMap, toRMat_apply, Matrix.add_apply]
      have ht1 : |(DP * Matrix.of A12t) i j| ≤ 2 * dimM * B_D * δ := by
        have h := rrr_matMul_entry_abs_le DP (Matrix.of A12t) (by positivity)
          (fun a b => (hDP_b a b).le) hA12t i j
        have : (r : ℝ) * (2 * B_D * δ) ≤ dimM * (2 * B_D * δ) :=
          mul_le_mul_of_nonneg_right hr_le (by positivity)
        linarith only [h, this]
      have hDP12_b : ∀ a b, |(D * topRightBlock S⁻¹) a b| ≤ 2 * dimM * B_D := fun a b => by
        have h := rrr_matMul_entry_abs_le D (topRightBlock S⁻¹) (by linarith only [hB_D_ge1]) hD_entry
          (fun x y => (hSinv_b _ _).le) a b
        have : (r : ℝ) * (B_D * 2) ≤ dimM * (B_D * 2) :=
          mul_le_mul_of_nonneg_right hr_le (by positivity)
        linarith only [h, this]
      have ht2 : |(D * topRightBlock S⁻¹ * Matrix.of u.1.2) i j| ≤ 2 * dimM ^ 2 * B_D * δ := by
        have h := rrr_matMul_entry_abs_le (D * topRightBlock S⁻¹) (Matrix.of u.1.2) (by positivity)
          hDP12_b hΔA12m i j
        have : (q : ℝ) * (2 * dimM * B_D * δ) ≤ dimM * (2 * dimM * B_D * δ) :=
          mul_le_mul_of_nonneg_right hq_le (by positivity)
        linarith only [h, this]
      have ht3 : |(Matrix.of u.2.2 * (Matrix.of A22 - Matrix.of u.2.1 *
          (S⁻¹ * vcatMat (Matrix.of A12t) (Matrix.of u.1.2)))) i j| ≤ 2 * dimM * δ ^ 2 := by
        have h := rrr_matMul_entry_abs_le (Matrix.of u.2.2)
          (Matrix.of A22 - Matrix.of u.2.1 * (S⁻¹ * vcatMat (Matrix.of A12t) (Matrix.of u.1.2)))
          hδ.le hΔB2t hÃ22_b i j
        have : ((p + k' : ℕ) : ℝ) * (δ * (2 * δ)) ≤ dimM * (δ * (2 * δ)) :=
          mul_le_mul_of_nonneg_right hpk_le (by positivity)
        linarith only [h, this]
      have htri : |(DP * Matrix.of A12t) i j + (D * topRightBlock S⁻¹ * Matrix.of u.1.2) i j +
          (Matrix.of u.2.2 * (Matrix.of A22 - Matrix.of u.2.1 *
            (S⁻¹ * vcatMat (Matrix.of A12t) (Matrix.of u.1.2)))) i j| ≤
          (2 * dimM * B_D + 2 * dimM ^ 2 * B_D + 2 * dimM) * δ := by
        have h1 := abs_add_le ((DP * Matrix.of A12t) i j +
            (D * topRightBlock S⁻¹ * Matrix.of u.1.2) i j)
          ((Matrix.of u.2.2 * (Matrix.of A22 - Matrix.of u.2.1 *
            (S⁻¹ * vcatMat (Matrix.of A12t) (Matrix.of u.1.2)))) i j)
        have h2 := abs_add_le ((DP * Matrix.of A12t) i j)
          ((D * topRightBlock S⁻¹ * Matrix.of u.1.2) i j)
        have hsq : 2 * dimM * δ ^ 2 ≤ 2 * dimM * δ := by
          have : δ ^ 2 ≤ δ := by nlinarith only [hδ, hδ_le1]
          nlinarith only [this, hdim_pos]
        linarith only [h1, h2, ht1, ht2, ht3, hsq]
      have hcoef : 2 * dimM * B_D + 2 * dimM ^ 2 * B_D + 2 * dimM ≤ C_hi := by
        dsimp [C_hi]; nlinarith only [hdim_le2, hB_D_ge1, hdimM_ge2]
      exact htri.trans (mul_le_mul_of_nonneg_right hcoef hδ.le)
    · exact rrr_matBox_mono (by linarith) (mem_matBox_iff.mpr hB2lo)
    · exact mem_matBox_iff.mpr hÃ22_b
  · intro hu_small ⟨W1, W2, B2lo, Ã22⟩ ⟨hW1, hW2, hB2lo, hÃ22⟩
    dsimp only at hW1 hW2 hB2lo hÃ22
    rcases hu_small with ⟨⟨hΔA11s, hΔA12ms⟩, ⟨hΔA21s, hΔB2ts⟩⟩
    rw [mem_matBox_iff] at hΔA11s hΔA12ms hΔA21s hΔB2ts hW1 hW2 hB2lo hÃ22
    let A12t_mat : Matrix (Fin r) (Fin n') ℝ :=
      DP⁻¹ * (Matrix.of W2 - D * topRightBlock S⁻¹ * Matrix.of u.1.2 -
        Matrix.of u.2.2 * Matrix.of Ã22)
    let A22_mat : Matrix (Fin (p + k')) (Fin n') ℝ :=
      Matrix.of Ã22 + Matrix.of u.2.1 * (S⁻¹ * vcatMat A12t_mat (Matrix.of u.1.2))
    let B2_mat : Matrix (Fin (r + (p + m'))) (Fin (p + k')) ℝ :=
      vcatMat (Matrix.of u.2.2) (rrrBaseBlo p m' k' + Matrix.of B2lo)
    let B1_mat : Matrix (Fin (r + (p + m'))) (Fin (r + q)) ℝ :=
      (Matrix.of W1 - rrrBaseBL (p := p) (q := q) (m' := m') D * Matrix.of u.1.1 - B2_mat * Matrix.of u.2.1) * S⁻¹
    have hA12t_in : toRMat A12t_mat ∈ matBox δ := by
      rw [mem_matBox_iff]; intro i j
      have hP12_le : ∀ a b, |topRightBlock S⁻¹ a b| ≤ 1 / (8 * dimM ^ 3 * B_D ^ 2) :=
        fun a b => (hP12_b a b).le
      have hDP12_b : ∀ a b, |(D * topRightBlock S⁻¹) a b| ≤ 1 / (8 * dimM ^ 2 * B_D) := fun a b => by
        have h := rrr_matMul_entry_abs_le D (topRightBlock S⁻¹) (by linarith only [hB_D_ge1]) hD_entry hP12_le a b
        have hdiv : (r : ℝ) * B_D * (1 / (8 * dimM ^ 3 * B_D ^ 2)) ≤ 1 / (8 * dimM ^ 2 * B_D) := by
          rw [mul_one_div, div_le_div_iff₀ (by positivity) (by positivity)]
          have : (r : ℝ) * (8 * dimM ^ 2 * B_D ^ 2) ≤ dimM * (8 * dimM ^ 2 * B_D ^ 2) :=
            mul_le_mul_of_nonneg_right hr_le (by positivity)
          linarith only [this]
        exact h.trans hdiv
      have ht2 : ∀ a b, |(D * topRightBlock S⁻¹ * Matrix.of u.1.2) a b| ≤ 1 / (8 * dimM * B_D) * δ :=
        fun a b => by
          have h := rrr_matMul_entry_abs_le (D * topRightBlock S⁻¹) (Matrix.of u.1.2) (by positivity)
            hDP12_b hΔA12m a b
          have h1 : (q : ℝ) * (1 / (8 * dimM ^ 2 * B_D)) * δ ≤ 1 / (8 * dimM * B_D) * δ := by
            refine mul_le_mul_of_nonneg_right ?_ hδ.le
            rw [mul_one_div, div_le_div_iff₀ (by positivity) (by positivity)]
            have : (q : ℝ) * (8 * dimM * B_D) ≤ dimM * (8 * dimM * B_D) :=
              mul_le_mul_of_nonneg_right hq_le (by positivity)
            linarith only [this]
          exact h.trans h1
      have ht3 : ∀ a b, |(Matrix.of u.2.2 * Matrix.of Ã22) a b| ≤ 1 / (8 * dimM * B_D) * δ :=
        fun a b => by
          have h := rrr_matMul_entry_abs_le (Matrix.of u.2.2) (Matrix.of Ã22) hδ.le hΔB2t hÃ22 a b
          have h1 : ((p + k' : ℕ) : ℝ) * δ * (c_lo * δ) ≤ 1 / (8 * dimM * B_D) * δ := by
            have hdimδ : dimM * δ ≤ 1 / (8 * dimM * B_D) := by
              rw [le_div_iff₀ (by positivity)]
              have hle1 : 8 * (dimM ^ 2 * B_D) * δ ≤ 8 * (dimM ^ 4 * B_D ^ 2) * δ :=
                mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hdim2BD_le4BD2 (by norm_num)) hδ.le
              have hle2 : 8 * (dimM ^ 4 * B_D ^ 2) * δ ≤ 32 * (dimM ^ 4 * B_D ^ 2) * δ :=
                mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right (by norm_num) (by positivity)) hδ.le
              linarith only [hle1, hle2, hδ_dim]
            have hpkδ : ((p + k' : ℕ) : ℝ) * δ ≤ dimM * δ :=
              mul_le_mul_of_nonneg_right hpk_le hδ.le
            have hstep1 : ((p + k' : ℕ) : ℝ) * δ * (c_lo * δ) ≤ (dimM * δ) * δ :=
              mul_le_mul hpkδ hc_lo_δ_le (by positivity) (by positivity)
            exact hstep1.trans (mul_le_mul_of_nonneg_right hdimδ hδ.le)
          exact h.trans h1
      have hRHS_b : ∀ a b, |(Matrix.of W2 - D * topRightBlock S⁻¹ * Matrix.of u.1.2 -
          Matrix.of u.2.2 * Matrix.of Ã22) a b| ≤ (c_lo + 1 / (4 * dimM * B_D)) * δ := fun a b => by
        simp only [Matrix.sub_apply, Matrix.of_apply]
        have h1 := abs_sub (W2 a b - (D * topRightBlock S⁻¹ * Matrix.of u.1.2) a b)
          ((Matrix.of u.2.2 * Matrix.of Ã22) a b)
        have h2 := abs_sub (W2 a b) ((D * topRightBlock S⁻¹ * Matrix.of u.1.2) a b)
        have h3 := hW2 a b
        have h4 := ht2 a b
        have h5 := ht3 a b
        have heq : (1 / (8 * dimM * B_D) : ℝ) * δ + 1 / (8 * dimM * B_D) * δ = 1 / (4 * dimM * B_D) * δ := by ring
        linarith only [h1, h2, h3, h4, h5, heq]
      have hmul := rrr_matMul_entry_abs_le DP⁻¹
        (Matrix.of W2 - D * topRightBlock S⁻¹ * Matrix.of u.1.2 - Matrix.of u.2.2 * Matrix.of Ã22)
        (by positivity) (fun a b => (hDPinv_b a b).le) hRHS_b i j
      have hcoef : (r : ℝ) * (2 * B_D) * (c_lo + 1 / (4 * dimM * B_D)) ≤ 1 := by
        have h1 : (r : ℝ) * (2 * B_D) * c_lo ≤ 1 / 6 := by
          dsimp [c_lo]; rw [mul_one_div, div_le_div_iff₀ (by positivity) (by norm_num)]
          have ha1 : (r : ℝ) * (12 * B_D) ≤ dimM * (12 * B_D) :=
            mul_le_mul_of_nonneg_right hr_le (by positivity)
          have ha2 : dimM * (12 * B_D) ≤ dimM ^ 2 * (12 * B_D) :=
            mul_le_mul_of_nonneg_right hdim_le2 (by positivity)
          linarith only [ha1, ha2]
        have h2 : (r : ℝ) * (2 * B_D) * (1 / (4 * dimM * B_D)) ≤ 1 / 2 := by
          rw [mul_one_div, div_le_div_iff₀ (by positivity) (by norm_num)]
          have ha1 : (r : ℝ) * (4 * B_D) ≤ dimM * (4 * B_D) :=
            mul_le_mul_of_nonneg_right hr_le (by positivity)
          linarith only [ha1]
        linarith only [h1, h2]
      calc |A12t_mat i j|
          ≤ (r : ℝ) * (2 * B_D) * ((c_lo + 1 / (4 * dimM * B_D)) * δ) := hmul
        _ = ((r : ℝ) * (2 * B_D) * (c_lo + 1 / (4 * dimM * B_D))) * δ := by ring
        _ ≤ 1 * δ := mul_le_mul_of_nonneg_right hcoef hδ.le
        _ = δ := one_mul δ
    have hA22_in : toRMat A22_mat ∈ matBox δ := by
      rw [mem_matBox_iff]; intro i j
      simp only [toRMat_apply, A22_mat, Matrix.add_apply]
      have hA12_b : ∀ a b, |(vcatMat A12t_mat (Matrix.of u.1.2)) a b| ≤ δ := fun a b => by
        by_cases ha : a.1 < r
        · have haeq : a = Fin.castAdd q ⟨a.1, ha⟩ := by ext; rfl
          rw [haeq, vcatMat_castAdd]; exact mem_matBox_iff.mp hA12t_in _ _
        · have haeq : a = Fin.natAdd r ⟨a.1 - r, by omega⟩ := by ext; dsimp; omega
          rw [haeq, vcatMat_natAdd]; exact hΔA12m _ _
      have hC_b : ∀ a b, |(S⁻¹ * vcatMat A12t_mat (Matrix.of u.1.2)) a b| ≤ 2 * dimM * δ :=
        fun a b => by
          have h := rrr_matMul_entry_abs_le S⁻¹ (vcatMat A12t_mat (Matrix.of u.1.2))
            (by norm_num) (fun x y => (hSinv_b x y).le) hA12_b a b
          nlinarith only [h, ha_le, hδ]
      have hshift : |(Matrix.of u.2.1 * (S⁻¹ * vcatMat A12t_mat (Matrix.of u.1.2))) i j| ≤
          δ / 2 := by
        have h := rrr_matMul_entry_abs_le (Matrix.of u.2.1) (S⁻¹ * vcatMat A12t_mat (Matrix.of u.1.2))
          hδ.le hΔA21 hC_b i j
        have h1 : 2 * dimM ^ 2 * δ ≤ 1 / 2 := by
          have hle : 4 * dimM ^ 2 * δ ≤ 32 * dimM ^ 4 * B_D ^ 2 * δ := by
            nlinarith only [hdim2_le4BD2, hδ]
          linarith only [hle, hδ_dim]
        have h2 : ((r + q : ℕ) : ℝ) * δ * (2 * dimM * δ) ≤ (2 * dimM ^ 2 * δ) * δ := by
          have := mul_le_mul_of_nonneg_right ha_le (show 0 ≤ 2 * dimM * δ ^ 2 by positivity)
          linarith only [this]
        have h3 : (2 * dimM ^ 2 * δ) * δ ≤ (1 / 2) * δ := mul_le_mul_of_nonneg_right h1 hδ.le
        linarith only [h, h2, h3]
      have hÃ_le : |Ã22 i j| ≤ δ / 2 := by
        have := hÃ22 i j
        nlinarith only [this, hc_lo_le, hδ]
      rw [Matrix.of_apply]
      exact (abs_add_le _ _).trans (by linarith only [hÃ_le, hshift])
    have hB1_in : toRMat B1_mat ∈ matBox δ := by
      rw [mem_matBox_iff]; intro i j
      have hB2_b : ∀ a b, |B2_mat a b| ≤ 2 := fun a b => by
        dsimp only [B2_mat]
        by_cases ha : a.1 < r
        · have haeq : a = Fin.castAdd (p + m') ⟨a.1, ha⟩ := by ext; rfl
          rw [haeq, vcatMat_castAdd, Matrix.of_apply]; linarith only [hΔB2t ⟨a.1, ha⟩ b, hδ_le1]
        · have haeq : a = Fin.natAdd r ⟨a.1 - r, by omega⟩ := by ext; dsimp; omega
          rw [haeq, vcatMat_natAdd, Matrix.add_apply, Matrix.of_apply]
          exact (abs_add_le _ _).trans (by linarith only [hBlo_entry ⟨a.1 - r, by omega⟩ b, hB2lo ⟨a.1 - r, by omega⟩ b, hc_lo_δ_le, hδ_le1])
      have hRHS_b : ∀ a b, |(Matrix.of W1 - rrrBaseBL (p := p) (q := q) (m' := m') D * Matrix.of u.1.1 -
          B2_mat * Matrix.of u.2.1) a b| ≤ (c_lo + dimM * B_D * η + 2 * dimM * η) * δ := fun a b => by
        simp only [Matrix.sub_apply, Matrix.of_apply]
        have h1 := abs_sub (W1 a b - (rrrBaseBL (p := p) (q := q) (m' := m') D * Matrix.of u.1.1) a b)
          ((B2_mat * Matrix.of u.2.1) a b)
        have h2 := abs_sub (W1 a b) ((rrrBaseBL (p := p) (q := q) (m' := m') D * Matrix.of u.1.1) a b)
        have ht2 := rrr_matMul_entry_abs_le (rrrBaseBL (p := p) (q := q) (m' := m') D)
          (Matrix.of u.1.1) (by linarith only [hB_D_ge1]) hB0L_entry hΔA11s a b
        have ht3 := rrr_matMul_entry_abs_le B2_mat (Matrix.of u.2.1) (by norm_num) hB2_b hΔA21s a b
        have h2' : |(rrrBaseBL (p := p) (q := q) (m' := m') D * Matrix.of u.1.1) a b| ≤ dimM * B_D * η * δ := by
          have := mul_le_mul_of_nonneg_right ha_le (show 0 ≤ B_D * (η * δ) by positivity)
          linarith only [ht2, this]
        have h3' : |(B2_mat * Matrix.of u.2.1) a b| ≤ 2 * dimM * η * δ := by
          have := mul_le_mul_of_nonneg_right hpk_le (show 0 ≤ 2 * (η * δ) by positivity)
          linarith only [ht3, this]
        linarith only [h1, h2, h2', h3', hW1 a b]
      have hmul := rrr_matMul_entry_abs_le
        (Matrix.of W1 - rrrBaseBL (p := p) (q := q) (m' := m') D * Matrix.of u.1.1 - B2_mat * Matrix.of u.2.1) S⁻¹
        (by positivity) hRHS_b (fun a b => (hSinv_b a b).le) i j
      have hcoef : ((r + q : ℕ) : ℝ) * (c_lo + dimM * B_D * η + 2 * dimM * η) * 2 ≤ 1 := by
        have h1 : 2 * dimM * c_lo ≤ 1 / 6 := by
          dsimp [c_lo]; rw [mul_one_div, div_le_div_iff₀ (by positivity) (by norm_num)]
          have : 12 * dimM ≤ 12 * dimM ^ 2 * B_D := by nlinarith only [hdim_le2, hB_D_ge1, hdim_pos]
          linarith only [this]
        have h2 : 2 * dimM ^ 2 * B_D * η ≤ 1 / 12 := by
          dsimp [η]; rw [mul_one_div, div_le_div_iff₀ (by positivity) (by norm_num)]
          have : 24 * dimM ^ 2 * B_D ≤ 24 * dimM ^ 3 * B_D := by nlinarith only [hdim2_le3, hB_D_pos]
          linarith only [this]
        have h3 : 4 * dimM ^ 2 * η ≤ 1 / 6 := by
          dsimp [η]; rw [mul_one_div, div_le_div_iff₀ (by positivity) (by norm_num)]
          have : 24 * dimM ^ 2 ≤ 24 * dimM ^ 3 * B_D := by nlinarith only [hdim2_le3, hB_D_ge1, hdim3_ge1]
          linarith only [this]
        have h4 : ((r + q : ℕ) : ℝ) * (c_lo + dimM * B_D * η + 2 * dimM * η) * 2 ≤
            dimM * (c_lo + dimM * B_D * η + 2 * dimM * η) * 2 := by
          have := mul_le_mul_of_nonneg_right ha_le (show 0 ≤ (c_lo + dimM * B_D * η + 2 * dimM * η) * 2 by positivity)
          linarith only [this]
        linarith only [h1, h2, h3, h4]
      calc |B1_mat i j|
          ≤ ((r + q : ℕ) : ℝ) * ((c_lo + dimM * B_D * η + 2 * dimM * η) * δ) * 2 := hmul
        _ = (((r + q : ℕ) : ℝ) * (c_lo + dimM * B_D * η + 2 * dimM * η) * 2) * δ := by ring
        _ ≤ 1 * δ := mul_le_mul_of_nonneg_right hcoef hδ.le
        _ = δ := one_mul δ
    have hB2lo_in : B2lo ∈ matBox δ :=
      rrr_matBox_mono hc_lo_δ_le (mem_matBox_iff.mpr hB2lo)
    refine ⟨(toRMat B1_mat, (toRMat A12t_mat, (B2lo, toRMat A22_mat))),
      ⟨hB1_in, hA12t_in, hB2lo_in, hA22_in⟩, hfrobC_of (toRMat A12t_mat) hA12t_in, ?_⟩
    unfold rrrTriMap
    have hÃ_eq : A22_mat - Matrix.of u.2.1 * (S⁻¹ * vcatMat A12t_mat (Matrix.of u.1.2)) =
        Matrix.of Ã22 := by
      dsimp only [A22_mat]; ext i j; simp only [Matrix.add_apply, Matrix.sub_apply]; ring
    have hW2_eq : DP * A12t_mat + D * topRightBlock S⁻¹ * Matrix.of u.1.2 +
        Matrix.of u.2.2 * Matrix.of Ã22 = Matrix.of W2 := by
      dsimp only [A12t_mat]
      rw [← Matrix.mul_assoc, Matrix.mul_nonsing_inv DP hDP_unit, Matrix.one_mul]
      ext i j; simp only [Matrix.add_apply, Matrix.sub_apply]; ring
    have hW1_eq : B1_mat * S + rrrBaseBL (p := p) (q := q) (m' := m') D * Matrix.of u.1.1 + B2_mat * Matrix.of u.2.1 =
        Matrix.of W1 := by
      dsimp only [B1_mat]
      rw [Matrix.mul_assoc, Matrix.nonsing_inv_mul S hS_unit, Matrix.mul_one]
      ext i j; simp only [Matrix.add_apply, Matrix.sub_apply]; ring
    refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext rfl ?_))
    · change toRMat (B1_mat * S + rrrBaseBL (p := p) (q := q) (m' := m') D * Matrix.of u.1.1 + B2_mat * Matrix.of u.2.1) = W1
      rw [hW1_eq]; rfl
    · change toRMat (DP * A12t_mat + D * topRightBlock S⁻¹ * Matrix.of u.1.2 +
        Matrix.of u.2.2 * (A22_mat - Matrix.of u.2.1 * (S⁻¹ * vcatMat A12t_mat (Matrix.of u.1.2)))) = W2
      rw [hÃ_eq, hW2_eq]; rfl
    · change toRMat (A22_mat - Matrix.of u.2.1 * (S⁻¹ * vcatMat A12t_mat (Matrix.of u.1.2))) = Ã22
      rw [hÃ_eq]; rfl

/-! ### 5. Main Theorem: Complete Unconditional `(r*, p, q)` RRR Fiber Certificate (`MAIS-O70`) -/

/-- **Complete Unconditional `(r*, p, q)` Zero-Loss Fiber Theorem (`[LC]`):**
For ALL `r, p, q, m', n', k' : ℕ` (setting `M = r + p + m'`, `N = r + q + n'`, `H = r + q + p + k'`,
`rank(B₀) = b = r + p`, `rank(A₀) = a = r + q`) and EVERY invertible `D : Matrix (Fin r) (Fin r) ℝ`,
the exact reduced-rank regression loss `‖(B₀ + ΔB)(A₀ + ΔA) - M*‖_F²` on
`[-δ, δ]^{M × H} × [-δ, δ]^{H × N}` has exact learning pair
`((M * N - m' * n') / 2 + awLambda m' n' k', awMult m' n' k')`. -/
theorem rrr_fiber_exact_pair {r p q m' n' k' : ℕ}
    (D : Matrix (Fin r) (Fin r) ℝ) (hD : IsUnit D.det) :
    ∃ δ₀ : ℝ, 0 < δ₀ ∧ ∀ δ : ℝ, 0 < δ → δ ≤ δ₀ →
      SLT.HasLaplacePair
        (SLT.Z (volume.restrict
            (matBox (a := r + (p + m')) (b := (r + q) + (p + k')) δ ×ˢ
             matBox (a := (r + q) + (p + k')) (b := (r + q) + n') δ))
          (rrrFiberLoss D))
        ((((r + (p + m') : ℕ) : ℝ) * ((r + q) + n' : ℕ) - (m' : ℝ) * n') / 2 + awLambda m' n' k')
        (awMult m' n' k') := by
  obtain ⟨δ_lo, hδ_lo, hlo_pair⟩ := rrrLoLoss_exact_pair p m' n' k'
  obtain ⟨δ₁, η, c_lo, C_hi, hδ₁, hη_pos, hη_le, hc_lo_pos, hc_lo_le, hC_hi, hgood⟩ :=
    exists_good_rrr_radius (p := p) (q := q) (m' := m') (n' := n') (k' := k') D hD
  have hD_abs_pos : 0 < |D.det| := abs_pos.mpr (isUnit_iff_ne_zero.mp hD)
  refine ⟨min δ₁ (δ_lo / 2), lt_min hδ₁ (by linarith), fun δ hδ hδle => ?_⟩
  have hδ_le1 : δ ≤ δ₁ := hδle.trans (min_le_left _ _)
  have h2δ_le : 2 * δ ≤ δ_lo := by linarith [hδle.trans (min_le_right _ _)]
  have hcδ_le : c_lo * δ ≤ δ_lo := by nlinarith
  have hlam_eq :
      ((((r + (p + m') : ℕ) : ℝ) * ((r + q) : ℕ) / 2) +
        (((r : ℝ) * n' / 2) + ((p : ℝ) * n' / 2 + awLambda m' n' k'))) =
      ((((r + (p + m') : ℕ) : ℝ) * ((r + q) + n' : ℕ) - (m' : ℝ) * n') / 2 + awLambda m' n' k') := by
    push_cast; ring
  have hmodel : ∀ (ρ₁ ρ₂ ρ₃ : ℝ), 0 < ρ₁ → 0 < ρ₂ → 0 < ρ₃ → ρ₃ ≤ δ_lo →
      SLT.HasLaplacePair
        (SLT.Z (volume.restrict (rrrActiveBox (r := r) (p := p) (q := q) (m' := m') (n' := n') (k' := k')
          ρ₁ ρ₂ ρ₃)) rrrDecoupledLoss)
        ((((r + (p + m') : ℕ) : ℝ) * ((r + q) + n' : ℕ) - (m' : ℝ) * n') / 2 + awLambda m' n' k')
        (awMult m' n' k') := fun ρ₁ ρ₂ ρ₃ hρ₁ hρ₂ hρ₃ hρ₃le => by
    have hG1 := hasLaplacePair_frobSq (m := r + (p + m')) (q := r + q) hρ₁
    have hG2 := hasLaplacePair_frobSq (m := r) (q := n') hρ₂
    have hL3 := hlo_pair ρ₃ hρ₃ hρ₃le
    have h23 := hG2.mul hL3
    have h123 := hG1.mul h23
    have hm2 : 1 + (1 + awMult m' n' k' - 1) - 1 = awMult m' n' k' := by
      have := awMult_pos m' n' k'; omega
    rw [hm2, hlam_eq] at h123
    refine h123.congr (Eventually.of_forall fun N => ?_)
    have hZ23 :
        SLT.Z (volume.restrict (matBox (a := r) (b := n') ρ₂ ×ˢ
            (matBox (a := p + m') (b := p + k') ρ₃ ×ˢ matBox (a := p + k') (b := n') ρ₃)))
          (fun y : RMat r n' × (RMat (p + m') (p + k') × RMat (p + k') n') =>
            frobSq (Matrix.of y.1) + rrrLoLoss y.2) N =
        SLT.Z (volume.restrict (matBox (a := r) (b := n') ρ₂))
            (fun X : RMat r n' => frobSq (Matrix.of X)) N *
          SLT.Z (volume.restrict
              (matBox (a := p + m') (b := p + k') ρ₃ ×ˢ matBox (a := p + k') (b := n') ρ₃))
            (rrrLoLoss (p := p) (m' := m') (n' := n') (k' := k')) N := by
      rw [Measure.volume_eq_prod, ← Measure.prod_restrict]
      exact SLT.Z_prod (ν₁ := volume.restrict (matBox (a := r) (b := n') ρ₂))
        (ν₂ := volume.restrict (matBox (a := p + m') (b := p + k') ρ₃ ×ˢ matBox (a := p + k') (b := n') ρ₃))
        measurable_frobSq measurable_rrrLoLoss N
    have hZ123 :
        SLT.Z (volume.restrict (rrrActiveBox (r := r) (p := p) (q := q) (m' := m') (n' := n') (k' := k')
            ρ₁ ρ₂ ρ₃)) rrrDecoupledLoss N =
        SLT.Z (volume.restrict (matBox (a := r + (p + m')) (b := r + q) ρ₁))
            (fun X : RMat (r + (p + m')) (r + q) => frobSq (Matrix.of X)) N *
          SLT.Z (volume.restrict (matBox (a := r) (b := n') ρ₂ ×ˢ
              (matBox (a := p + m') (b := p + k') ρ₃ ×ˢ matBox (a := p + k') (b := n') ρ₃)))
            (fun y : RMat r n' × (RMat (p + m') (p + k') × RMat (p + k') n') =>
              frobSq (Matrix.of y.1) + rrrLoLoss y.2) N := by
      unfold rrrActiveBox rrrDecoupledLoss
      rw [Measure.volume_eq_prod, ← Measure.prod_restrict]
      have hmeas2 : Measurable (fun y : RMat r n' × (RMat (p + m') (p + k') × RMat (p + k') n') =>
          frobSq (Matrix.of y.1) + rrrLoLoss y.2) :=
        (measurable_frobSq.comp measurable_fst).add (measurable_rrrLoLoss.comp measurable_snd)
      exact SLT.Z_prod
        (ν₁ := volume.restrict (matBox (a := r + (p + m')) (b := r + q) ρ₁))
        (ν₂ := volume.restrict (matBox (a := r) (b := n') ρ₂ ×ˢ
          (matBox (a := p + m') (b := p + k') ρ₃ ×ˢ matBox (a := p + k') (b := n') ρ₃)))
        measurable_frobSq hmeas2 N
    rw [hZ123, hZ23]
  have hmeas_blk : Measurable (rrrBlockLoss (r := r) (p := p) (q := q) (m' := m') (n' := n') (k' := k') D) := by
    have hA11_m : ∀ (i j : Fin (r + q)), Measurable (fun uv : RRRPassive r p q n' k' × RRRActive r p q m' n' k' => uv.1.1.1 i j) :=
      fun i j => (measurable_pi_apply j).comp ((measurable_pi_apply i).comp (measurable_fst.comp (measurable_fst.comp measurable_fst)))
    have hA12m_m : ∀ (i : Fin q) (j : Fin n'), Measurable (fun uv : RRRPassive r p q n' k' × RRRActive r p q m' n' k' => uv.1.1.2 i j) :=
      fun i j => (measurable_pi_apply j).comp ((measurable_pi_apply i).comp (measurable_snd.comp (measurable_fst.comp measurable_fst)))
    have hA21_m : ∀ (i : Fin (p + k')) (j : Fin (r + q)), Measurable (fun uv : RRRPassive r p q n' k' × RRRActive r p q m' n' k' => uv.1.2.1 i j) :=
      fun i j => (measurable_pi_apply j).comp ((measurable_pi_apply i).comp (measurable_fst.comp (measurable_snd.comp measurable_fst)))
    have hB2t_m : ∀ (i : Fin r) (j : Fin (p + k')), Measurable (fun uv : RRRPassive r p q n' k' × RRRActive r p q m' n' k' => uv.1.2.2 i j) :=
      fun i j => (measurable_pi_apply j).comp ((measurable_pi_apply i).comp (measurable_snd.comp (measurable_snd.comp measurable_fst)))
    have hB1_m : ∀ (i : Fin (r + (p + m'))) (j : Fin (r + q)), Measurable (fun uv : RRRPassive r p q n' k' × RRRActive r p q m' n' k' => uv.2.1 i j) :=
      fun i j => (measurable_pi_apply j).comp ((measurable_pi_apply i).comp (measurable_fst.comp measurable_snd))
    have hA12t_m : ∀ (i : Fin r) (j : Fin n'), Measurable (fun uv : RRRPassive r p q n' k' × RRRActive r p q m' n' k' => uv.2.2.1 i j) :=
      fun i j => (measurable_pi_apply j).comp ((measurable_pi_apply i).comp (measurable_fst.comp (measurable_snd.comp measurable_snd)))
    have hB2lo_m : ∀ (i : Fin (p + m')) (j : Fin (p + k')), Measurable (fun uv : RRRPassive r p q n' k' × RRRActive r p q m' n' k' => uv.2.2.2.1 i j) :=
      fun i j => (measurable_pi_apply j).comp ((measurable_pi_apply i).comp (measurable_fst.comp (measurable_snd.comp (measurable_snd.comp measurable_snd))))
    have hA22_m : ∀ (i : Fin (p + k')) (j : Fin n'), Measurable (fun uv : RRRPassive r p q n' k' × RRRActive r p q m' n' k' => uv.2.2.2.2 i j) :=
      fun i j => (measurable_pi_apply j).comp ((measurable_pi_apply i).comp (measurable_snd.comp (measurable_snd.comp (measurable_snd.comp measurable_snd))))
    have hvcatA_m : ∀ (i : Fin (r + q)) (j : Fin n'),
        Measurable (fun uv : RRRPassive r p q n' k' × RRRActive r p q m' n' k' =>
          vcatMat (Matrix.of uv.2.2.1) (Matrix.of uv.1.1.2) i j) := fun i j => by
      simp only [vcatMat, Matrix.of_apply]; split_ifs
      · exact hA12t_m _ _
      · exact hA12m_m _ _
    have hvcatB_m : ∀ (i : Fin (r + (p + m'))) (j : Fin (p + k')),
        Measurable (fun uv : RRRPassive r p q n' k' × RRRActive r p q m' n' k' =>
          vcatMat (Matrix.of uv.1.2.2) (rrrBaseBlo p m' k' + Matrix.of uv.2.2.2.1) i j) := fun i j => by
      simp only [vcatMat, Matrix.add_apply, Matrix.of_apply]; split_ifs
      · exact hB2t_m _ _
      · exact measurable_const.add (hB2lo_m _ _)
    unfold rrrBlockLoss frobSq
    simp only [Matrix.add_apply, Matrix.mul_apply, Matrix.of_apply]
    refine (Finset.measurable_sum _ fun i _ => Finset.measurable_sum _ fun j _ => ?_).add
      (Finset.measurable_sum _ fun i _ => Finset.measurable_sum _ fun j _ => ?_)
    · refine Measurable.pow_const ?_ 2
      exact ((Finset.measurable_sum _ fun t _ => (hB1_m i t).mul (measurable_const.add (hA11_m t j))).add
        (Finset.measurable_sum _ fun t _ => measurable_const.mul (hA11_m t j))).add
        (Finset.measurable_sum _ fun t _ => (hvcatB_m i t).mul (hA21_m t j))
    · refine Measurable.pow_const ?_ 2
      exact (Finset.measurable_sum _ fun t _ => (measurable_const.add (hB1_m i t)).mul (hvcatA_m t j)).add
        (Finset.measurable_sum _ fun t _ => (hvcatB_m i t).mul (hA22_m t j))
  have hZ_iter : ∀ N : ℝ,
      SLT.Z (volume.restrict
          (matBox (a := r + (p + m')) (b := (r + q) + (p + k')) δ ×ˢ
           matBox (a := (r + q) + (p + k')) (b := (r + q) + n') δ)) (rrrFiberLoss D) N =
      ∫⁻ u in rrrPassiveBox δ, ∫⁻ v in rrrActiveBox δ δ δ,
        ENNReal.ofReal (Real.exp (-(N * rrrBlockLoss D (u, v)))) := fun N => by
    have h1 := rrr_Z_eq_of_mp (rrrSplitEquiv r p q m' n' k') (rrr_mp_splitEquiv r p q m' n' k')
      (matBox δ ×ˢ matBox δ) (rrrPassiveBox δ ×ˢ rrrActiveBox δ δ δ)
      ((measurableSet_matBox δ).prod (measurableSet_matBox δ))
      ((measurableSet_rrrPassiveBox δ).prod (measurableSet_rrrActiveBox δ δ δ))
      (rrrSplitEquiv_mem_box (δ := δ))
      (rrrFiberLoss D) (rrrBlockLoss D) hmeas_blk (rrrFiberLoss_eq_blockLoss D) N
    rw [h1]
    unfold SLT.Z
    rw [Measure.volume_eq_prod, ← Measure.prod_restrict,
      MeasureTheory.lintegral_prod _ (by fun_prop)]
  let jLo : ℝ := (2 : ℝ)⁻¹ ^ (r + (p + m')) * (2 * |D.det|)⁻¹ ^ n'
  let jHi : ℝ := (2 : ℝ) ^ (r + (p + m')) * (|D.det| / 2)⁻¹ ^ n'
  have hjLo_pos : 0 < jLo := by positivity
  have hjHi_pos : 0 < jHi := by positivity
  let cLo : ℝ≥0∞ := ENNReal.ofReal jLo * volume (rrrPassiveBox (r := r) (p := p) (q := q)
    (n' := n') (k' := k') (η * δ))
  let cHi : ℝ≥0∞ := ENNReal.ofReal jHi * volume (rrrPassiveBox (r := r) (p := p) (q := q)
    (n' := n') (k' := k') δ)
  have hcLo_pos : cLo ≠ 0 := (ENNReal.mul_pos (ENNReal.ofReal_pos.mpr hjLo_pos).ne'
    (volume_rrrPassiveBox_pos (mul_pos hη_pos hδ)).ne').ne'
  have hcLo_top : cLo ≠ ⊤ := ENNReal.mul_ne_top ENNReal.ofReal_lt_top.ne
    (volume_rrrPassiveBox_lt_top (mul_pos hη_pos hδ).le).ne
  have hcHi_pos : cHi ≠ 0 := (ENNReal.mul_pos (ENNReal.ofReal_pos.mpr hjHi_pos).ne'
    (volume_rrrPassiveBox_pos hδ).ne').ne'
  have hcHi_top : cHi ≠ ⊤ := ENNReal.mul_ne_top ENNReal.ofReal_lt_top.ne
    (volume_rrrPassiveBox_lt_top hδ.le).ne
  have hbounds : ∀ N : ℝ, 0 ≤ N →
      cLo * SLT.Z (volume.restrict (rrrActiveBox (c_lo * δ) (c_lo * δ) (c_lo * δ)))
          rrrDecoupledLoss (2 * N) ≤
        SLT.Z (volume.restrict (matBox δ ×ˢ matBox δ)) (rrrFiberLoss D) N ∧
      SLT.Z (volume.restrict (matBox δ ×ˢ matBox δ)) (rrrFiberLoss D) N ≤
        cHi * SLT.Z (volume.restrict (rrrActiveBox (C_hi * δ) (C_hi * δ) (2 * δ)))
          rrrDecoupledLoss ((1 / 4 : ℝ) * N) := fun N hN => by
    rw [hZ_iter N]
    constructor
    · have hsub_u : rrrPassiveBox (r := r) (p := p) (q := q) (n' := n') (k' := k') (η * δ) ⊆
          rrrPassiveBox δ := by
        rintro ⟨⟨a, b⟩, ⟨c, d⟩⟩ ⟨⟨ha, hb⟩, ⟨hc, hd⟩⟩
        exact ⟨⟨rrr_matBox_mono (by nlinarith) ha, rrr_matBox_mono (by nlinarith) hb⟩,
               ⟨rrr_matBox_mono (by nlinarith) hc, rrr_matBox_mono (by nlinarith) hd⟩⟩
      have hslice : ∀ u ∈ rrrPassiveBox (r := r) (p := p) (q := q) (n' := n') (k' := k') (η * δ),
          ENNReal.ofReal jLo *
            SLT.Z (volume.restrict (rrrActiveBox (r := r) (p := p) (q := q) (m' := m') (n' := n') (k' := k')
              (c_lo * δ) (c_lo * δ) (c_lo * δ)))
              rrrDecoupledLoss (2 * N) ≤
          ∫⁻ (v : RRRActive r p q m' n' k') in rrrActiveBox δ δ δ,
            ENNReal.ofReal (Real.exp (-(N * rrrBlockLoss D (u, v)))) := fun u hu_s => by
        have hu := hsub_u hu_s
        obtain ⟨hS_u, hDP_u, _, hS_hi, _, hDP_hi, _, hsurj⟩ := hgood δ hδ hδ_le1 u hu
        have hS_ne : ((1 : Matrix (Fin (r + q)) (Fin (r + q)) ℝ) + Matrix.of u.1.1).det ≠ 0 :=
          isUnit_iff_ne_zero.mp hS_u
        have hDP_ne : (D * topLeftBlock ((1 : Matrix (Fin (r + q)) (Fin (r + q)) ℝ) +
            Matrix.of u.1.1)⁻¹).det ≠ 0 := isUnit_iff_ne_zero.mp hDP_u
        have hjac : ENNReal.ofReal jLo ≤
            ENNReal.ofReal |(((1 : Matrix (Fin (r + q)) (Fin (r + q)) ℝ) + Matrix.of u.1.1).det)⁻¹| ^
              (r + (p + m')) *
            ENNReal.ofReal |((D * topLeftBlock ((1 : Matrix (Fin (r + q)) (Fin (r + q)) ℝ) +
              Matrix.of u.1.1)⁻¹).det)⁻¹| ^ n' := by
          dsimp only [jLo]
          rw [ENNReal.ofReal_mul (by positivity), ENNReal.ofReal_pow (by positivity),
            ENNReal.ofReal_pow (by positivity)]
          refine mul_le_mul' (pow_le_pow_left' (ENNReal.ofReal_le_ofReal ?_) _)
            (pow_le_pow_left' (ENNReal.ofReal_le_ofReal ?_) _)
          · rw [abs_inv]; exact inv_anti₀ (abs_pos.mpr hS_ne) hS_hi
          · rw [abs_inv]; exact inv_anti₀ (abs_pos.mpr hDP_ne) hDP_hi
        have hg : Measurable fun w : RRRActive r p q m' n' k' =>
            ENNReal.ofReal (Real.exp (-((2 * N) * rrrDecoupledLoss w))) :=
          ENNReal.measurable_ofReal.comp (Real.measurable_exp.comp
            ((measurable_const.mul measurable_rrrDecoupledLoss).neg))
        have hmap_eq :
            (ENNReal.ofReal |(((1 : Matrix (Fin (r + q)) (Fin (r + q)) ℝ) + Matrix.of u.1.1).det)⁻¹| ^
                (r + (p + m')) *
              ENNReal.ofReal |((D * topLeftBlock ((1 : Matrix (Fin (r + q)) (Fin (r + q)) ℝ) +
                Matrix.of u.1.1)⁻¹).det)⁻¹| ^ n') *
              SLT.Z (volume.restrict (rrrActiveBox (r := r) (p := p) (q := q) (m' := m') (n' := n') (k' := k')
                (c_lo * δ) (c_lo * δ) (c_lo * δ)))
                rrrDecoupledLoss (2 * N) =
            ∫⁻ v : RRRActive r p q m' n' k',
              (rrrActiveBox (r := r) (p := p) (q := q) (m' := m') (n' := n') (k' := k')
                (c_lo * δ) (c_lo * δ) (c_lo * δ)).indicator
                (fun w => ENNReal.ofReal (Real.exp (-((2 * N) * rrrDecoupledLoss w))))
                (rrrTriMap (m' := m') D u v) := by
          rw [← lintegral_map (hg.indicator (measurableSet_rrrActiveBox _ _ _))
              (measurable_rrrTriMap D u),
            map_rrrTriMap_volume D u hS_ne hDP_ne, lintegral_smul_measure, smul_eq_mul,
            lintegral_indicator (measurableSet_rrrActiveBox _ _ _)]
          rfl
        have htri_inj : Function.Injective (rrrTriMap (m' := m') D u) := by
          intro v₁ v₂ heq
          have h3 := congrArg (fun x => x.2.2.1) heq
          have h4 := congrArg (fun x => x.2.2.2) heq
          have h2 := congrArg (fun x => x.2.1) heq
          have h1 := congrArg (fun x => x.1) heq
          have hB2lo_eq : v₁.2.2.1 = v₂.2.2.1 := h3
          set S := (1 : Matrix (Fin (r + q)) (Fin (r + q)) ℝ) + Matrix.of u.1.1
          set DP := D * topLeftBlock S⁻¹
          have hA12t_eq : v₁.2.1 = v₂.2.1 := by
            have hW2 : DP * Matrix.of v₁.2.1 + D * topRightBlock S⁻¹ * Matrix.of u.1.2 +
                Matrix.of u.2.2 * Matrix.of (rrrTriMap D u v₁).2.2.2 =
              DP * Matrix.of v₂.2.1 + D * topRightBlock S⁻¹ * Matrix.of u.1.2 +
                Matrix.of u.2.2 * Matrix.of (rrrTriMap D u v₂).2.2.2 :=
              congrArg Matrix.of h2
            rw [h4] at hW2
            have hmul : DP * Matrix.of v₁.2.1 = DP * Matrix.of v₂.2.1 := by
              have := congrArg (fun X => X - (D * topRightBlock S⁻¹ * Matrix.of u.1.2 +
                Matrix.of u.2.2 * Matrix.of (rrrTriMap D u v₂).2.2.2)) hW2
              simpa [add_assoc] using this
            have hinv := congrArg (fun X => DP⁻¹ * X) hmul
            simp only [← Matrix.mul_assoc, Matrix.nonsing_inv_mul DP hDP_u, Matrix.one_mul] at hinv
            funext i j; exact congrFun (congrFun hinv i) j
          have hA22_eq : v₁.2.2.2 = v₂.2.2.2 := by
            have hÃ : Matrix.of v₁.2.2.2 - Matrix.of u.2.1 * (S⁻¹ * vcatMat (Matrix.of v₁.2.1) (Matrix.of u.1.2)) =
                Matrix.of v₂.2.2.2 - Matrix.of u.2.1 * (S⁻¹ * vcatMat (Matrix.of v₂.2.1) (Matrix.of u.1.2)) :=
              congrArg Matrix.of h4
            rw [hA12t_eq] at hÃ
            have hsub := sub_left_inj.mp hÃ
            funext i j; exact congrFun (congrFun hsub i) j
          have hB1_eq : v₁.1 = v₂.1 := by
            have hW1 : Matrix.of v₁.1 * S + rrrBaseBL D * Matrix.of u.1.1 +
                vcatMat (Matrix.of u.2.2) (rrrBaseBlo p m' k' + Matrix.of v₁.2.2.1) * Matrix.of u.2.1 =
              Matrix.of v₂.1 * S + rrrBaseBL D * Matrix.of u.1.1 +
                vcatMat (Matrix.of u.2.2) (rrrBaseBlo p m' k' + Matrix.of v₂.2.2.1) * Matrix.of u.2.1 :=
              congrArg Matrix.of h1
            rw [hB2lo_eq] at hW1
            have hmul : Matrix.of v₁.1 * S = Matrix.of v₂.1 * S := by
              have := congrArg (fun X => X - (rrrBaseBL D * Matrix.of u.1.1 +
                vcatMat (Matrix.of u.2.2) (rrrBaseBlo p m' k' + Matrix.of v₂.2.2.1) * Matrix.of u.2.1)) hW1
              simpa [add_assoc] using this
            have hinv := congrArg (fun X => X * S⁻¹) hmul
            simp only [Matrix.mul_assoc, Matrix.mul_nonsing_inv S hS_u, Matrix.mul_one] at hinv
            funext i j; exact congrFun (congrFun hinv i) j
          rcases v₁ with ⟨_, _, _, _⟩; rcases v₂ with ⟨_, _, _, _⟩
          simp only [Prod.mk.injEq] at hB1_eq hA12t_eq hB2lo_eq hA22_eq ⊢
          exact ⟨hB1_eq, hA12t_eq, hB2lo_eq, hA22_eq⟩
        have hstep_ind :
            ∫⁻ v : RRRActive r p q m' n' k',
              (rrrActiveBox (r := r) (p := p) (q := q) (m' := m') (n' := n') (k' := k')
                (c_lo * δ) (c_lo * δ) (c_lo * δ)).indicator
                (fun w => ENNReal.ofReal (Real.exp (-((2 * N) * rrrDecoupledLoss w))))
                (rrrTriMap (m' := m') D u v) ≤
            ∫⁻ (v : RRRActive r p q m' n' k') in rrrActiveBox δ δ δ,
              ENNReal.ofReal (Real.exp (-(N * rrrBlockLoss D (u, v)))) := by
          rw [← lintegral_indicator (measurableSet_rrrActiveBox δ δ δ)]
          refine lintegral_mono fun v => ?_
          by_cases hm : rrrTriMap (m' := m') D u v ∈ rrrActiveBox (c_lo * δ) (c_lo * δ) (c_lo * δ)
          · obtain ⟨v₀, hv₀_mem, hC₀, hv₀_eq⟩ := hsurj hu_s _ hm
            have hv_eq : v = v₀ := htri_inj hv₀_eq.symm
            subst hv_eq
            rw [Set.indicator_of_mem hm, Set.indicator_of_mem hv₀_mem]
            apply ENNReal.ofReal_le_ofReal
            apply Real.exp_le_exp.mpr
            have hup := (rrr_loss_comparison D u v hS_u hC₀).2
            nlinarith
          · rw [Set.indicator_of_notMem hm]; exact bot_le
        calc ENNReal.ofReal jLo *
              SLT.Z (volume.restrict (rrrActiveBox (c_lo * δ) (c_lo * δ) (c_lo * δ)))
                rrrDecoupledLoss (2 * N)
            ≤ _ * SLT.Z (volume.restrict (rrrActiveBox (c_lo * δ) (c_lo * δ) (c_lo * δ)))
                rrrDecoupledLoss (2 * N) := by gcongr
          _ = _ := hmap_eq
          _ ≤ _ := hstep_ind
      calc cLo * SLT.Z (volume.restrict (rrrActiveBox (c_lo * δ) (c_lo * δ) (c_lo * δ)))
              rrrDecoupledLoss (2 * N)
          = (ENNReal.ofReal jLo *
              SLT.Z (volume.restrict (rrrActiveBox (c_lo * δ) (c_lo * δ) (c_lo * δ)))
                rrrDecoupledLoss (2 * N)) *
              volume (rrrPassiveBox (r := r) (p := p) (q := q) (n' := n') (k' := k') (η * δ)) := by
            dsimp only [cLo]; ring
        _ = ∫⁻ _ in rrrPassiveBox (η * δ), (ENNReal.ofReal jLo *
              SLT.Z (volume.restrict (rrrActiveBox (c_lo * δ) (c_lo * δ) (c_lo * δ)))
                rrrDecoupledLoss (2 * N)) := (setLIntegral_const _ _).symm
        _ ≤ ∫⁻ u in rrrPassiveBox (η * δ), ∫⁻ v in rrrActiveBox δ δ δ,
              ENNReal.ofReal (Real.exp (-(N * rrrBlockLoss D (u, v)))) :=
            setLIntegral_mono' (measurableSet_rrrPassiveBox (η * δ)) hslice
        _ ≤ ∫⁻ u in rrrPassiveBox δ, ∫⁻ v in rrrActiveBox δ δ δ,
              ENNReal.ofReal (Real.exp (-(N * rrrBlockLoss D (u, v)))) :=
            lintegral_mono_set hsub_u
    · have hslice : ∀ u ∈ rrrPassiveBox (r := r) (p := p) (q := q) (n' := n') (k' := k') δ,
          ∫⁻ (v : RRRActive r p q m' n' k') in rrrActiveBox δ δ δ,
            ENNReal.ofReal (Real.exp (-(N * rrrBlockLoss D (u, v)))) ≤
          ENNReal.ofReal jHi *
            SLT.Z (volume.restrict (rrrActiveBox (r := r) (p := p) (q := q) (m' := m') (n' := n') (k' := k')
              (C_hi * δ) (C_hi * δ) (2 * δ)))
              rrrDecoupledLoss ((1 / 4 : ℝ) * N) := fun u hu => by
        obtain ⟨hS_u, hDP_u, hS_lo, _, hDP_lo, _, hfwd, _⟩ := hgood δ hδ hδ_le1 u hu
        have hS_ne : ((1 : Matrix (Fin (r + q)) (Fin (r + q)) ℝ) + Matrix.of u.1.1).det ≠ 0 :=
          isUnit_iff_ne_zero.mp hS_u
        have hDP_ne : (D * topLeftBlock ((1 : Matrix (Fin (r + q)) (Fin (r + q)) ℝ) +
            Matrix.of u.1.1)⁻¹).det ≠ 0 := isUnit_iff_ne_zero.mp hDP_u
        have hjac :
            ENNReal.ofReal |(((1 : Matrix (Fin (r + q)) (Fin (r + q)) ℝ) + Matrix.of u.1.1).det)⁻¹| ^
              (r + (p + m')) *
            ENNReal.ofReal |((D * topLeftBlock ((1 : Matrix (Fin (r + q)) (Fin (r + q)) ℝ) +
              Matrix.of u.1.1)⁻¹).det)⁻¹| ^ n' ≤ ENNReal.ofReal jHi := by
          dsimp only [jHi]
          rw [ENNReal.ofReal_mul (by positivity), ENNReal.ofReal_pow (by positivity),
            ENNReal.ofReal_pow (by positivity)]
          refine mul_le_mul' (pow_le_pow_left' (ENNReal.ofReal_le_ofReal ?_) _)
            (pow_le_pow_left' (ENNReal.ofReal_le_ofReal ?_) _)
          · rw [abs_inv]
            have := inv_anti₀ (by norm_num : (0 : ℝ) < 1 / 2) hS_lo
            linarith
          · rw [abs_inv]; exact inv_anti₀ (by positivity) hDP_lo
        have hg : Measurable fun w : RRRActive r p q m' n' k' =>
            ENNReal.ofReal (Real.exp (-(((1 / 4 : ℝ) * N) * rrrDecoupledLoss w))) :=
          ENNReal.measurable_ofReal.comp (Real.measurable_exp.comp
            ((measurable_const.mul measurable_rrrDecoupledLoss).neg))
        have hstep_ind :
            ∫⁻ (v : RRRActive r p q m' n' k') in rrrActiveBox δ δ δ,
              ENNReal.ofReal (Real.exp (-(N * rrrBlockLoss D (u, v)))) ≤
            ∫⁻ v : RRRActive r p q m' n' k',
              (rrrActiveBox (r := r) (p := p) (q := q) (m' := m') (n' := n') (k' := k')
                (C_hi * δ) (C_hi * δ) (2 * δ)).indicator
                (fun w => ENNReal.ofReal (Real.exp (-(((1 / 4 : ℝ) * N) * rrrDecoupledLoss w))))
                (rrrTriMap (m' := m') D u v) := by
          rw [← lintegral_indicator (measurableSet_rrrActiveBox δ δ δ)]
          refine lintegral_mono fun v => ?_
          by_cases hv : v ∈ rrrActiveBox δ δ δ
          · obtain ⟨hC, hmem⟩ := hfwd v hv
            rw [Set.indicator_of_mem hv, Set.indicator_of_mem hmem]
            apply ENNReal.ofReal_le_ofReal
            apply Real.exp_le_exp.mpr
            have hlo := (rrr_loss_comparison D u v hS_u hC).1
            nlinarith
          · rw [Set.indicator_of_notMem hv]; exact bot_le
        have hmap_eq :
            ∫⁻ v : RRRActive r p q m' n' k',
              (rrrActiveBox (r := r) (p := p) (q := q) (m' := m') (n' := n') (k' := k')
                (C_hi * δ) (C_hi * δ) (2 * δ)).indicator
                (fun w => ENNReal.ofReal (Real.exp (-(((1 / 4 : ℝ) * N) * rrrDecoupledLoss w))))
                (rrrTriMap (m' := m') D u v) =
            (ENNReal.ofReal |(((1 : Matrix (Fin (r + q)) (Fin (r + q)) ℝ) + Matrix.of u.1.1).det)⁻¹| ^
                (r + (p + m')) *
              ENNReal.ofReal |((D * topLeftBlock ((1 : Matrix (Fin (r + q)) (Fin (r + q)) ℝ) +
                Matrix.of u.1.1)⁻¹).det)⁻¹| ^ n') *
              SLT.Z (volume.restrict (rrrActiveBox (r := r) (p := p) (q := q) (m' := m') (n' := n') (k' := k')
                (C_hi * δ) (C_hi * δ) (2 * δ)))
                rrrDecoupledLoss ((1 / 4 : ℝ) * N) := by
          rw [← lintegral_map (hg.indicator (measurableSet_rrrActiveBox _ _ _))
              (measurable_rrrTriMap D u),
            map_rrrTriMap_volume D u hS_ne hDP_ne, lintegral_smul_measure, smul_eq_mul,
            lintegral_indicator (measurableSet_rrrActiveBox _ _ _)]
          rfl
        calc ∫⁻ v in rrrActiveBox δ δ δ,
              ENNReal.ofReal (Real.exp (-(N * rrrBlockLoss D (u, v))))
            ≤ _ * SLT.Z (volume.restrict (rrrActiveBox (C_hi * δ) (C_hi * δ) (2 * δ)))
                rrrDecoupledLoss ((1 / 4 : ℝ) * N) := hstep_ind.trans hmap_eq.le
          _ ≤ ENNReal.ofReal jHi *
                SLT.Z (volume.restrict (rrrActiveBox (C_hi * δ) (C_hi * δ) (2 * δ)))
                  rrrDecoupledLoss ((1 / 4 : ℝ) * N) := by gcongr
      calc ∫⁻ u in rrrPassiveBox δ, ∫⁻ v in rrrActiveBox δ δ δ,
              ENNReal.ofReal (Real.exp (-(N * rrrBlockLoss D (u, v))))
          ≤ ∫⁻ _ in rrrPassiveBox δ, (ENNReal.ofReal jHi *
              SLT.Z (volume.restrict (rrrActiveBox (C_hi * δ) (C_hi * δ) (2 * δ)))
                rrrDecoupledLoss ((1 / 4 : ℝ) * N)) :=
            setLIntegral_mono' (measurableSet_rrrPassiveBox δ) hslice
        _ = cHi * SLT.Z (volume.restrict (rrrActiveBox (C_hi * δ) (C_hi * δ) (2 * δ)))
              rrrDecoupledLoss ((1 / 4 : ℝ) * N) := by
            rw [setLIntegral_const]; dsimp only [cHi]; ring
  have hlo_p := ((hmodel (c_lo * δ) (c_lo * δ) (c_lo * δ) (mul_pos hc_lo_pos hδ)
    (mul_pos hc_lo_pos hδ) (mul_pos hc_lo_pos hδ) hcδ_le).comp_mul (by norm_num : (0 : ℝ) < 2)).const_mul
    hcLo_pos hcLo_top
  have hhi_p := ((hmodel (C_hi * δ) (C_hi * δ) (2 * δ) (mul_pos (by linarith) hδ)
    (mul_pos (by linarith) hδ) (by linarith) h2δ_le).comp_mul (by norm_num : (0 : ℝ) < 1 / 4)).const_mul
    hcHi_pos hcHi_top
  refine ⟨hlo_p.1, ?_, ?_⟩
  · exact hlo_p.2.1.mono ((eventually_ge_atTop 0).mono fun N hN => (hbounds N hN).1)
  · exact hhi_p.2.2.mono ((eventually_ge_atTop 0).mono fun N hN => (hbounds N hN).2)

/-! ### 6. Complete 2D Phase Diagram & Irreducible-Component Codimension Characterization (`[LC]`) -/

/-- Active + excess decomposition of the exact `(r, p, q)` learning coefficient. -/
theorem rrr_lambda_active_plus_excess (r p q m' n' k' : ℕ) :
    let M := r + (p + m')
    let N := (r + q) + n'
    let m := p + m'
    let n := q + n'
    (((M : ℝ) * N - (m' : ℝ) * n') / 2) + awLambda m' n' k' =
      ((r : ℝ) * (M + N - r) / 2) +
        (((p : ℝ) * n + (q : ℝ) * m - (p : ℝ) * q) / 2 + awLambda m' n' k') := by
  intro M N m n
  dsimp only [M, N, m, n]
  push_cast
  ring

/-- **2D Subcritical Plateau (`2 * p < k + m - n` and `2 * q < k + n - m`):**
Throughout the entire 2D subcritical region of the zero-loss fiber `{B A = M*}`, the local
learning pair is **constant in `(p, q)`** and equal to the global minimum
`(r * (M + N - r) / 2 + awLambda m n k, awMult m n k)`. -/
theorem rrr_fiber_subcritical_plateau {r p q m' n' k' : ℕ}
    (D : Matrix (Fin r) (Fin r) ℝ) (hD : IsUnit D.det)
    (h1 : p + q + k' < (p + m') + (q + n'))
    (hsubB : 2 * p + (q + n') < (p + q + k') + (p + m'))
    (hsubA : 2 * q + (p + m') < (p + q + k') + (q + n')) :
    ∃ δ₀ : ℝ, 0 < δ₀ ∧ ∀ δ : ℝ, 0 < δ → δ ≤ δ₀ →
      SLT.HasLaplacePair
        (SLT.Z (volume.restrict
            (matBox (a := r + (p + m')) (b := (r + q) + (p + k')) δ ×ˢ
             matBox (a := (r + q) + (p + k')) (b := (r + q) + n') δ))
          (rrrFiberLoss D))
        (((r : ℝ) * ((r + (p + m') : ℕ) + ((r + q) + n' : ℕ) - r) / 2) +
          awLambda (p + m') (q + n') (p + q + k'))
        (awMult (p + m') (q + n') (p + q + k')) := by
  obtain ⟨δ₀, hδ₀, h⟩ := rrr_fiber_exact_pair (p := p) (q := q) (m' := m') (n' := n') (k' := k') D hD
  refine ⟨δ₀, hδ₀, fun δ hδ hδle => ?_⟩
  have hpair := h δ hδ hδle
  have hc1 : ¬ (m' + n' ≤ k') := by omega
  have hc2 : ¬ (k' + n' ≤ m') := by omega
  have hc3 : ¬ (k' + m' ≤ n') := by omega
  have hb1 : ¬ ((p + m') + (q + n') ≤ p + q + k') := by omega
  have hb2 : ¬ ((p + q + k') + (q + n') ≤ p + m') := by omega
  have hb3 : ¬ ((p + q + k') + (p + m') ≤ q + n') := by omega
  have hpar : Even (m' + n' + k') ↔ Even ((p + m') + (q + n') + (p + q + k')) := by
    have heq : (p + m') + (q + n') + (p + q + k') = (m' + n' + k') + 2 * (p + q) := by omega
    rw [heq]; simp [Nat.even_add, even_two]
  have hmult : awMult m' n' k' = awMult (p + m') (q + n') (p + q + k') := by
    unfold awMult; simp only [hc1, hc2, hc3, hb1, hb2, hb3, ↓reduceIte, hpar]
  have hlam :
      ((((r + (p + m') : ℕ) : ℝ) * ((r + q) + n' : ℕ) - (m' : ℝ) * n') / 2 + awLambda m' n' k') =
      (((r : ℝ) * ((r + (p + m') : ℕ) + ((r + q) + n' : ℕ) - r) / 2) +
        awLambda (p + m') (q + n') (p + q + k')) := by
    unfold awLambda awLambdaEven
    simp only [hc1, hc2, hc3, hb1, hb2, hb3, ↓reduceIte, hpar]
    split_ifs <;> { push_cast; ring }
  rwa [hlam, hmult] at hpair

/-- **`A`-Supercritical Parabola (`k + n ≤ m + 2 * q`, i.e., `q ≥ q_c`):**
Whenever `q ≥ q_c = (n + k - m) / 2`, the local learning pair climbs quadratically in `q`
(independently of `p`!) with exact multiplicity `1`. -/
theorem rrr_fiber_A_supercritical_parabola {r p q m' n' k' : ℕ}
    (D : Matrix (Fin r) (Fin r) ℝ) (hD : IsUnit D.det)
    (h1 : p + q + k' < (p + m') + (q + n'))
    (hsupA : (p + q + k') + (q + n') ≤ (p + m') + 2 * q) :
    ∃ δ₀ : ℝ, 0 < δ₀ ∧ ∀ δ : ℝ, 0 < δ → δ ≤ δ₀ →
      SLT.HasLaplacePair
        (SLT.Z (volume.restrict
            (matBox (a := r + (p + m')) (b := (r + q) + (p + k')) δ ×ˢ
             matBox (a := (r + q) + (p + k')) (b := (r + q) + n') δ))
          (rrrFiberLoss D))
        (((r : ℝ) * ((r + (p + m') : ℕ) + ((r + q) + n' : ℕ) - r) / 2) +
          awLambdaEven (p + m') (q + n') (p + q + k') +
          ((q : ℝ) - (((q + n' : ℕ) : ℝ) + (p + q + k' : ℕ) - (p + m' : ℕ)) / 2) ^ 2 / 2)
        1 := by
  obtain ⟨δ₀, hδ₀, h⟩ := rrr_fiber_exact_pair (p := p) (q := q) (m' := m') (n' := n') (k' := k') D hD
  refine ⟨δ₀, hδ₀, fun δ hδ hδle => ?_⟩
  have hpair := h δ hδ hδle
  have hc1 : ¬ (m' + n' ≤ k') := by omega
  have hc2 : k' + n' ≤ m' := by omega
  have hmult : awMult m' n' k' = 1 := by unfold awMult; simp only [hc1, hc2, ↓reduceIte]
  have hlam :
      ((((r + (p + m') : ℕ) : ℝ) * ((r + q) + n' : ℕ) - (m' : ℝ) * n') / 2 + awLambda m' n' k') =
      (((r : ℝ) * ((r + (p + m') : ℕ) + ((r + q) + n' : ℕ) - r) / 2) +
        awLambdaEven (p + m') (q + n') (p + q + k') +
        ((q : ℝ) - (((q + n' : ℕ) : ℝ) + (p + q + k' : ℕ) - (p + m' : ℕ)) / 2) ^ 2 / 2) := by
    unfold awLambda awLambdaEven
    simp only [hc1, hc2, ↓reduceIte]
    push_cast; ring
  rwa [hlam, hmult] at hpair

/-- **`B`-Supercritical Parabola (`k + m ≤ n + 2 * p`, i.e., `p ≥ p_c`):**
Whenever `p ≥ p_c = (m + k - n) / 2`, the local learning pair climbs quadratically in `p`
(independently of `q`!) with exact multiplicity `1`. -/
theorem rrr_fiber_B_supercritical_parabola {r p q m' n' k' : ℕ}
    (D : Matrix (Fin r) (Fin r) ℝ) (hD : IsUnit D.det)
    (h1 : p + q + k' < (p + m') + (q + n'))
    (hsupB : (p + q + k') + (p + m') ≤ (q + n') + 2 * p) :
    ∃ δ₀ : ℝ, 0 < δ₀ ∧ ∀ δ : ℝ, 0 < δ → δ ≤ δ₀ →
      SLT.HasLaplacePair
        (SLT.Z (volume.restrict
            (matBox (a := r + (p + m')) (b := (r + q) + (p + k')) δ ×ˢ
             matBox (a := (r + q) + (p + k')) (b := (r + q) + n') δ))
          (rrrFiberLoss D))
        (((r : ℝ) * ((r + (p + m') : ℕ) + ((r + q) + n' : ℕ) - r) / 2) +
          awLambdaEven (p + m') (q + n') (p + q + k') +
          ((p : ℝ) - (((p + m' : ℕ) : ℝ) + (p + q + k' : ℕ) - (q + n' : ℕ)) / 2) ^ 2 / 2)
        1 := by
  obtain ⟨δ₀, hδ₀, h⟩ := rrr_fiber_exact_pair (p := p) (q := q) (m' := m') (n' := n') (k' := k') D hD
  refine ⟨δ₀, hδ₀, fun δ hδ hδle => ?_⟩
  have hpair := h δ hδ hδle
  have hc1 : ¬ (m' + n' ≤ k') := by omega
  have hc3 : k' + m' ≤ n' := by omega
  by_cases hc2 : k' + n' ≤ m'
  · have hkm : k' = 0 ∧ m' = n' := by omega
    have hmult : awMult m' n' k' = 1 := by unfold awMult; simp only [hc1, hc2, ↓reduceIte]
    have hlam :
        ((((r + (p + m') : ℕ) : ℝ) * ((r + q) + n' : ℕ) - (m' : ℝ) * n') / 2 + awLambda m' n' k') =
        (((r : ℝ) * ((r + (p + m') : ℕ) + ((r + q) + n' : ℕ) - r) / 2) +
          awLambdaEven (p + m') (q + n') (p + q + k') +
          ((p : ℝ) - (((p + m' : ℕ) : ℝ) + (p + q + k' : ℕ) - (q + n' : ℕ)) / 2) ^ 2 / 2) := by
      unfold awLambda awLambdaEven
      simp only [hc1, hc2, ↓reduceIte]
      rw [hkm.1, hkm.2]
      push_cast; ring
    rwa [hlam, hmult] at hpair
  · have hmult : awMult m' n' k' = 1 := by unfold awMult; simp only [hc1, hc2, hc3, ↓reduceIte]
    have hlam :
        ((((r + (p + m') : ℕ) : ℝ) * ((r + q) + n' : ℕ) - (m' : ℝ) * n') / 2 + awLambda m' n' k') =
        (((r : ℝ) * ((r + (p + m') : ℕ) + ((r + q) + n' : ℕ) - r) / 2) +
          awLambdaEven (p + m') (q + n') (p + q + k') +
          ((p : ℝ) - (((p + m' : ℕ) : ℝ) + (p + q + k' : ℕ) - (q + n' : ℕ)) / 2) ^ 2 / 2) := by
      unfold awLambda awLambdaEven
      simp only [hc1, hc2, hc3, ↓reduceIte]
      push_cast; ring
    rwa [hlam, hmult] at hpair

/-- **Mutual Exclusivity of Supercritical Regimes (`[LC]`):**
Because `p_c + q_c = k ≥ p + q`, a zero-loss fiber point `(p, q)` can NEVER be strictly
supercritical on both `B` (`p > p_c`) and `A` (`q > q_c`) simultaneously. -/
theorem rrr_supercritical_mutually_exclusive {m n k p q : ℕ} (hpq : p + q ≤ k)
    (hA : k + n < m + 2 * q) (hB : k + m < n + 2 * p) : False := by
  omega

/-- Excess codimension of the irreducible fiber component `Z_{α, k - α}`:
`componentCodim m n k α = m * α + n * (k - α) - α * (k - α)`. -/
def componentCodim (m n k α : ℕ) : ℝ :=
  (m : ℝ) * α + (n : ℝ) * ((k : ℝ) - α) - (α : ℝ) * ((k : ℝ) - α)

/-- **Irreducible-Component Codimension Characterization (`[LC]`, synthesizing `arc-llc` / `MAIS-O70`):**
For ALL interior `(m, n, k)` and ALL excess ranks `(p, q)` with `p + q ≤ k`, `p ≤ m`, `q ≤ n`,
twice the excess local learning coefficient
`2 * ((p * n + q * m - p * q) / 2 + awLambda (m - p) (n - q) (k - p - q))`
equals the **exact minimum** of `componentCodim m n k α` over all integer components
`α ∈ [q, k - p]` passing through `(p, q)`. -/
theorem rrr_fiber_eq_min_component_codim {m n k p q : ℕ}
    (h1 : k < m + n) (hp : p ≤ m) (hq : q ≤ n) (hpq : p + q ≤ k) :
    let twoLam := ((p : ℝ) * n + (q : ℝ) * m - (p : ℝ) * q) +
      2 * awLambda (m - p) (n - q) (k - p - q)
    (∀ α : ℕ, q ≤ α → α ≤ k - p → twoLam ≤ componentCodim m n k α) ∧
    (∃ α : ℕ, q ≤ α ∧ α ≤ k - p ∧ twoLam = componentCodim m n k α) := by
  intro twoLam
  have hkey : ∀ α : ℕ, componentCodim m n k α =
      2 * awLambdaEven m n k + ((α : ℝ) - ((n : ℝ) + k - m) / 2) ^ 2 := fun α => by
    unfold componentCodim awLambdaEven; ring
  have hc1 : ¬ ((m - p) + (n - q) ≤ k - p - q) := by omega
  have hcast_m : ((m - p : ℕ) : ℝ) = (m : ℝ) - p := Nat.cast_sub hp
  have hcast_n : ((n - q : ℕ) : ℝ) = (n : ℝ) - q := Nat.cast_sub hq
  have hcast_k : ((k - p - q : ℕ) : ℝ) = (k : ℝ) - p - q := by
    rw [Nat.cast_sub (by omega), Nat.cast_sub (by omega)]
  by_cases hc2 : (k - p - q) + (n - q) ≤ m - p
  · have hval : twoLam = componentCodim m n k q := by
      dsimp only [twoLam]; unfold awLambda componentCodim
      simp only [hc1, hc2, ↓reduceIte, hcast_n, hcast_k]; ring
    refine ⟨fun α hqα _ => ?_, ⟨q, le_rfl, by omega, hval⟩⟩
    rw [hval, hkey q, hkey α]
    have hq_ge : ((n : ℝ) + k - m) / 2 ≤ q := by
      have : n + k ≤ m + 2 * q := by omega
      have hR : (n : ℝ) + k ≤ m + 2 * q := by exact_mod_cast this
      linarith
    have hα_ge : (q : ℝ) ≤ α := Nat.cast_le.mpr hqα
    nlinarith
  · by_cases hc3 : (k - p - q) + (m - p) ≤ n - q
    · have hval : twoLam = componentCodim m n k (k - p) := by
        dsimp only [twoLam]; unfold awLambda componentCodim
        simp only [hc1, hc2, hc3, ↓reduceIte, hcast_m, hcast_k, Nat.cast_sub (show p ≤ k by omega)]
        ring
      refine ⟨fun α _ hαkp => ?_, ⟨k - p, by omega, le_rfl, hval⟩⟩
      rw [hval, hkey (k - p), hkey α]
      have hkp_le : ((k - p : ℕ) : ℝ) ≤ ((n : ℝ) + k - m) / 2 := by
        rw [Nat.cast_sub (show p ≤ k by omega)]
        have : k + m ≤ n + 2 * p := by omega
        have hR : (k : ℝ) + m ≤ n + 2 * p := by exact_mod_cast this
        linarith
      have hα_le : (α : ℝ) ≤ ((k - p : ℕ) : ℝ) := Nat.cast_le.mpr hαkp
      nlinarith
    · have hpar : Even ((m - p) + (n - q) + (k - p - q)) ↔ Even (m + n + k) := by
        have heq : (m - p) + (n - q) + (k - p - q) + 2 * (p + q) = m + n + k := by omega
        rw [← heq]; simp [Nat.even_add, even_two]
      by_cases hev : Even (m + n + k)
      · have hval : twoLam = 2 * awLambdaEven m n k := by
          dsimp only [twoLam]; unfold awLambda awLambdaEven
          simp only [hc1, hc2, hc3, hpar.mpr hev, ↓reduceIte, hcast_m, hcast_n, hcast_k]; ring
        obtain ⟨t, ht⟩ := hev
        refine ⟨fun α _ _ => by rw [hval, hkey α]; linarith [sq_nonneg ((α : ℝ) - ((n : ℝ) + k - m) / 2)],
          ⟨t - m, by omega, by omega, ?_⟩⟩
        rw [hval, hkey (t - m)]
        have hcast_t : ((t - m : ℕ) : ℝ) = ((n : ℝ) + k - m) / 2 := by
          have : 2 * (t - m) + m = n + k := by omega
          have hR : 2 * ((t - m : ℕ) : ℝ) + m = n + k := by exact_mod_cast this
          linarith
        rw [hcast_t]; ring
      · have hval : twoLam = 2 * awLambdaEven m n k + 1 / 4 := by
          dsimp only [twoLam]; unfold awLambda awLambdaEven
          simp only [hc1, hc2, hc3, mt hpar.mp hev, ↓reduceIte, hcast_m, hcast_n, hcast_k]; ring
        obtain ⟨t, ht⟩ := Nat.not_even_iff_odd.mp hev
        refine ⟨fun α _ _ => ?_, ⟨t - m, by omega, by omega, ?_⟩⟩
        · rw [hval, hkey α]
          have hodd_diff : (2 : ℤ) * (α : ℤ) - ((n : ℤ) + k - m) ≠ 0 := by omega
          have hsq_ge : (1 : ℤ) ≤ (2 * (α : ℤ) - ((n : ℤ) + k - m)) ^ 2 := by
            have h1 : 1 ≤ |2 * (α : ℤ) - ((n : ℤ) + k - m)| := Int.one_le_abs hodd_diff
            nlinarith [sq_abs (2 * (α : ℤ) - ((n : ℤ) + k - m))]
          have hsqR : (1 : ℝ) ≤ (2 * (α : ℝ) - ((n : ℝ) + k - m)) ^ 2 := by exact_mod_cast hsq_ge
          nlinarith
        · rw [hval, hkey (t - m)]
          have hcast_t : ((t - m : ℕ) : ℝ) = ((n : ℝ) + k - m - 1) / 2 := by
            have : 2 * (t - m) + m + 1 = n + k := by omega
            have hR : 2 * ((t - m : ℕ) : ℝ) + m + 1 = n + k := by exact_mod_cast this
            linarith
          rw [hcast_t]; ring

end EatYourFiber
