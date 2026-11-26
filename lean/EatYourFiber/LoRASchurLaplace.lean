/-
Copyright (c) 2026 Bobby Faber. All rights reserved.
Released under Apache 2.0 license.
-/
module

public import EatYourFiber.LoRASchur
public import Mathlib.MeasureTheory.Measure.Lebesgue.EqHaar
public import Mathlib.MeasureTheory.Constructions.Pi
public import Mathlib.LinearAlgebra.Matrix.Determinant.Basic
public import Mathlib.LinearAlgebra.Matrix.Adjugate
public import Mathlib.Topology.Instances.Matrix

/-!
# LoRA / Reduced-Rank Fiber Phase Transition: Measure-Theoretic Laplace Reduction

This module proves the exact measure-theoretic Schur-complement reduction in `SLT.HasLaplacePair`:

1. `hasLaplacePair_frobSq`: The Gaussian Frobenius-squared loss on `matBox δ` (`δ > 0`) has exact
   learning pair `((m * q : ℝ) / 2, 1)`.
2. `rank0_hasLaplacePair_of_unit`: Radius invariance of the rank-0 excess matrix factorization
   problem `‖B₂ Δ₂₂‖_F^2` via homogeneous scaling `(B₂, Δ₂₂) ↦ (ρ • B₂, ρ • Δ₂₂)`.
3. `lora_schur_fiber_reduction`: For any `m q k' n' : ℕ`, if the rank-0 excess problem `(m, k', n')`
   has `HasLaplacePair (lam, m_log)` on the unit box, then for all sufficiently small `δ > 0`, the
   exact LoRA fiber loss around `(B₀, A₀) = (0, fromBlocks 1 0 0 0)` on the uniform box `fiberBox δ`
   has exact `HasLaplacePair ((m * q : ℝ) / 2 + lam, m_log)`.
4. `lora_standard_init_exact_pair`: Unconditional corollary at `k' = 0` (standard LoRA
   initialization `B₀ = 0`, `A₀` full row rank `q = k ≤ n`): exact learning pair `((m * q : ℝ) / 2, 1)`
   for all `m, q, n'`.
-/

@[expose] public section

namespace EatYourFiber

open MeasureTheory Set Filter Matrix Finset
open scoped ENNReal NNReal

/-! ### 1. Real matrix Euclidean space and boxes -/

/-- Real `a × b` matrices viewed as Euclidean space `Fin a → Fin b → ℝ` (carrying Mathlib's
standard product Lebesgue `volume`, `BorelSpace`, `SigmaFinite`, and `MetricSpace` with no
typeclass diamonds). -/
abbrev RMat (a b : ℕ) := Fin a → Fin b → ℝ

/-- Convert a `Matrix (Fin a) (Fin b) ℝ` back to `RMat a b`. -/
def toRMat {a b : ℕ} (X : Matrix (Fin a) (Fin b) ℝ) : RMat a b := fun i j => X i j

@[simp] theorem toRMat_apply {a b : ℕ} (X : Matrix (Fin a) (Fin b) ℝ) (i : Fin a) (j : Fin b) :
    toRMat X i j = X i j := rfl

@[simp] theorem of_toRMat {a b : ℕ} (X : Matrix (Fin a) (Fin b) ℝ) :
    Matrix.of (toRMat X) = X := rfl

@[simp] theorem toRMat_of {a b : ℕ} (X : RMat a b) :
    toRMat (Matrix.of X) = X := rfl

theorem toRMat_injective {a b : ℕ} : Function.Injective (toRMat (a := a) (b := b)) := by
  intro X Y h
  ext i j
  exact congrFun (congrFun h i) j

/-- The symmetric entrywise box `[-δ, δ]^{a × b}` in `RMat a b`. -/
def matBox {a b : ℕ} (δ : ℝ) : Set (RMat a b) :=
  Set.pi Set.univ fun _ => Set.pi Set.univ fun _ => Set.Icc (-δ) δ

theorem mem_matBox_iff {a b : ℕ} {δ : ℝ} {X : RMat a b} :
    X ∈ matBox δ ↔ ∀ i j, |X i j| ≤ δ := by
  simp only [matBox, Set.mem_pi, Set.mem_univ, forall_const, Set.mem_Icc, abs_le]

theorem measurableSet_matBox {a b : ℕ} (δ : ℝ) :
    MeasurableSet (matBox (a := a) (b := b) δ) :=
  MeasurableSet.univ_pi fun _ => MeasurableSet.univ_pi fun _ => measurableSet_Icc

theorem volume_matBox {a b : ℕ} {δ : ℝ} (_hδ : 0 ≤ δ) :
    volume (matBox (a := a) (b := b) δ) = ENNReal.ofReal (2 * δ) ^ (a * b) := by
  have h1 : ∀ _ : Fin a, volume (Set.pi Set.univ fun _ : Fin b => Set.Icc (-δ) δ) =
      ENNReal.ofReal (2 * δ) ^ b := fun _ => by
    change (Measure.pi fun _ : Fin b => (volume : Measure ℝ))
      (Set.pi Set.univ fun _ : Fin b => Set.Icc (-δ) δ) = _
    rw [Measure.pi_pi]
    simp [Real.volume_Icc, two_mul, sub_neg_eq_add]
  change (Measure.pi fun _ : Fin a => (volume : Measure (Fin b → ℝ)))
    (Set.pi Set.univ fun _ => Set.pi Set.univ fun _ => Set.Icc (-δ) δ) = _
  rw [Measure.pi_pi, Finset.prod_congr rfl fun i _ => h1 i]
  simp only [Finset.prod_const, Finset.card_univ, Fintype.card_fin]
  rw [← pow_mul, mul_comm b a]

theorem volume_matBox_pos {a b : ℕ} {δ : ℝ} (hδ : 0 < δ) :
    0 < volume (matBox (a := a) (b := b) δ) := by
  rw [volume_matBox hδ.le]
  exact ENNReal.pow_pos (ENNReal.ofReal_pos.mpr (by linarith)) _

theorem volume_matBox_lt_top {a b : ℕ} {δ : ℝ} (hδ : 0 ≤ δ) :
    volume (matBox (a := a) (b := b) δ) < ⊤ := by
  rw [volume_matBox hδ]
  exact ENNReal.pow_lt_top ENNReal.ofReal_lt_top

theorem measurable_frobSq {a b : ℕ} :
    Measurable (fun X : RMat a b => frobSq (Matrix.of X)) := by
  unfold frobSq
  fun_prop

/-- Scaling each factor of `Measure.pi` by a finite `c : ℝ≥0∞` multiplies the product measure
by `c ^ n`. -/
private theorem measure_pi_const_smul {n : ℕ} {α : Type*} [MeasurableSpace α]
    (μ : Measure α) [SigmaFinite μ] (c : ℝ≥0∞) [SigmaFinite (c • μ)] :
    Measure.pi (fun _ : Fin n => c • μ) = c ^ n • Measure.pi (fun _ : Fin n => μ) := by
  refine Measure.pi_eq fun s _ => ?_
  rw [Measure.smul_apply, smul_eq_mul, Measure.pi_pi]
  simp only [Measure.smul_apply, smul_eq_mul, Finset.prod_mul_distrib, Finset.prod_const,
    Finset.card_univ, Fintype.card_fin]

private theorem smul_prod_measure {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    (μ : Measure α) (ν : Measure β) [SigmaFinite μ] [SigmaFinite ν] (c : ℝ≥0∞)
    [SigmaFinite (c • μ)] :
    (c • μ).prod ν = c • (μ.prod ν) := by
  refine Measure.prod_eq (μ := c • μ) (ν := ν) fun s t _ _ => ?_
  rw [Measure.smul_apply, smul_eq_mul, Measure.prod_prod, Measure.smul_apply, smul_eq_mul, mul_assoc]

private theorem prod_smul_measure {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    (μ : Measure α) (ν : Measure β) [SigmaFinite μ] [SigmaFinite ν] (c : ℝ≥0∞)
    [SigmaFinite (c • ν)] :
    μ.prod (c • ν) = c • (μ.prod ν) := by
  refine Measure.prod_eq (μ := μ) (ν := c • ν) fun s t _ _ => ?_
  rw [Measure.smul_apply, smul_eq_mul, Measure.prod_prod, Measure.smul_apply, smul_eq_mul,
    mul_left_comm]

private theorem sigmaFinite_ofReal_smul {α : Type*} [MeasurableSpace α]
    (μ : Measure α) [SigmaFinite μ] (r : ℝ) :
    SigmaFinite (ENNReal.ofReal r • μ) := by
  have : ENNReal.ofReal r • μ = (r.toNNReal : ℝ≥0) • μ := rfl
  rw [this]
  infer_instance

private theorem sigmaFinite_ofReal_pow_smul {α : Type*} [MeasurableSpace α]
    (μ : Measure α) [SigmaFinite μ] (r : ℝ) (k : ℕ) :
    SigmaFinite (ENNReal.ofReal r ^ k • μ) := by
  have : ENNReal.ofReal r ^ k • μ = (r.toNNReal ^ k : ℝ≥0) • μ := by
    rw [ENNReal.smul_def, ENNReal.ofReal, ENNReal.coe_pow]
  rw [this]
  infer_instance

/-- Entrywise bound for matrix product `X * Y`. -/
private theorem matMul_entry_abs_le {a b c : ℕ}
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

/-! ### 2. Gaussian box Laplace pair for `frobSq` -/

/-- One-dimensional Gaussian Laplace pair: `x ↦ x^2` on `[-δ, δ]` has pair `(1/2, 1)`. -/
theorem hasLaplacePair_sq_Icc {δ : ℝ} (hδ : 0 < δ) :
    SLT.HasLaplacePair (SLT.Z (volume.restrict (Set.Icc (-δ) δ)) (fun x : ℝ => x ^ 2))
      (1 / 2 : ℝ) 1 := by
  have hnc := SLT.normalCrossing_box (d := 1) hδ (fun _ => 2) (fun _ => 0)
    (fun _ => by norm_num) ⟨0, by norm_num⟩
  have hact : SLT.NC.Active (d := 1) (fun _ => 2) = {0} := by
    ext i; fin_cases i; simp [SLT.NC.Active]
  have hmu0 : SLT.NC.ncMu (d := 1) (fun _ => 2) (fun _ => 0) 0 = (1 / 2 : ℝ) := by
    simp [SLT.NC.ncMu]
  have hlam : SLT.NC.ncLam (d := 1) (fun _ => 2) (fun _ => 0) = (1 / 2 : ℝ) := by
    unfold SLT.NC.ncLam
    rw [hact]
    simp [hmu0]
  have hmult : SLT.NC.ncMult (d := 1) (fun _ => 2) (fun _ => 0) = 1 := by
    unfold SLT.NC.ncMult
    rw [hlam]
    have hfilt : (Finset.univ : Finset (Fin 1)).filter
        (fun i => (2 : ℕ) ≠ 0 ∧
          SLT.NC.ncMu (fun _ : Fin 1 => 2) (fun _ => (0 : ℝ)) i = (1 / 2 : ℝ)) = {0} := by
      ext i; fin_cases i; simp [hmu0]
    rw [hfilt]; simp
  rw [hlam, hmult] at hnc
  refine hnc.congr (Eventually.of_forall fun N => ?_)
  let e : (Fin 1 → ℝ) ≃ᵐ ℝ := MeasurableEquiv.funUnique (Fin 1) ℝ
  have hmp : MeasurePreserving e volume volume := volume_preserving_funUnique (Fin 1) ℝ
  have hbm : SLT.boxMeasure (d := 1) δ (fun _ => 0) = volume.restrict (SLT.box (d := 1) δ) := by
    unfold SLT.boxMeasure SLT.absDens
    simp
  rw [hbm]
  unfold SLT.Z
  have hpre : e ⁻¹' Set.Icc (-δ) δ = SLT.box (d := 1) δ := by
    ext y
    simp only [Set.mem_preimage, SLT.box, Set.mem_pi, Set.mem_univ, forall_const, e,
      MeasurableEquiv.funUnique_apply]
    constructor
    · intro hy i; fin_cases i; exact hy
    · intro hy; exact hy 0
  rw [← hmp.map_eq, Measure.restrict_map e.measurable measurableSet_Icc, hpre,
    lintegral_map (by fun_prop) e.measurable]
  refine setLIntegral_congr_fun (SLT.measurableSet_box δ) fun y _ => ?_
  simp [SLT.absLoss, e, sq_abs]

/-- Product-space Laplace integral of a separable sum over `Fin n → α` factors as a power. -/
private theorem Z_pi_sum_eq_pow {α : Type*} [MeasureSpace α] [SigmaFinite (volume : Measure α)]
    (s : Set α) (hs : MeasurableSet s) (g : α → ℝ) (hg : Measurable g) :
    ∀ (n : ℕ) (N : ℝ),
      SLT.Z (volume.restrict (Set.pi Set.univ fun _ : Fin n => s))
        (fun x : Fin n → α => ∑ i : Fin n, g (x i)) N =
      SLT.Z (volume.restrict s) g N ^ n
  | 0, N => by
    unfold SLT.Z
    simp only [Finset.univ_eq_empty, Finset.sum_empty, mul_zero, neg_zero, Real.exp_zero,
      ENNReal.ofReal_one, setLIntegral_const, one_mul, pow_zero]
    change (Measure.pi fun _ : Fin 0 => (volume : Measure α))
      (Set.pi Set.univ fun _ : Fin 0 => s) = 1
    rw [Measure.pi_pi]; simp
  | n + 1, N => by
    let e : (Fin (n + 1) → α) ≃ᵐ α × (Fin n → α) :=
      MeasurableEquiv.piFinSuccAbove (fun _ => α) 0
    have hmp : MeasurePreserving e volume volume :=
      volume_preserving_piFinSuccAbove (fun _ => α) 0
    let box_n : Set (Fin n → α) := Set.pi Set.univ fun _ => s
    have hpre : e ⁻¹' (s ×ˢ box_n) = Set.pi Set.univ fun _ : Fin (n + 1) => s := by
      ext x
      simp only [Set.mem_preimage, Set.mem_prod, Set.mem_pi, Set.mem_univ, forall_const, e, box_n,
        MeasurableEquiv.piFinSuccAbove_apply, Fin.insertNthEquiv_symm_apply]
      constructor
      · rintro ⟨h0, htail⟩ i
        refine Fin.cases h0 (fun j => ?_) i
        simpa [Fin.tail] using htail j
      · intro hx
        exact ⟨hx 0, fun j => by simpa [Fin.tail] using hx (Fin.succ j)⟩
    have hsum : ∀ x : Fin (n + 1) → α,
        (∑ i : Fin (n + 1), g (x i)) = g (e x).1 + ∑ j : Fin n, g ((e x).2 j) := fun x => by
      rw [Fin.sum_univ_succ]
      rfl
    have hres : (volume : Measure (α × (Fin n → α))).restrict (s ×ˢ box_n) =
        Measure.map e (volume.restrict (Set.pi Set.univ fun _ : Fin (n + 1) => s)) := by
      rw [← hmp.map_eq, Measure.restrict_map e.measurable (hs.prod (MeasurableSet.univ_pi fun _ => hs)),
        hpre]
    have hsum_m : Measurable fun v : Fin n → α => ∑ j : Fin n, g (v j) :=
      Finset.measurable_sum _ fun j _ => hg.comp (measurable_pi_apply j)
    have hstep :
        SLT.Z (volume.restrict (Set.pi Set.univ fun _ : Fin (n + 1) => s))
          (fun x : Fin (n + 1) → α => ∑ i : Fin (n + 1), g (x i)) N =
        SLT.Z ((volume.restrict s).prod (volume.restrict box_n))
          (fun p : α × (Fin n → α) => g p.1 + ∑ j : Fin n, g (p.2 j)) N := by
      rw [Measure.prod_restrict]
      change _ = SLT.Z ((volume : Measure (α × (Fin n → α))).restrict (s ×ˢ box_n)) _ N
      rw [hres]
      unfold SLT.Z
      have hm2 : Measurable fun p : α × (Fin n → α) =>
          ENNReal.ofReal (Real.exp (-(N * (g p.1 + ∑ j : Fin n, g (p.2 j))))) := by
        have := hsum_m; fun_prop
      rw [lintegral_map hm2 e.measurable]
      refine setLIntegral_congr_fun (MeasurableSet.univ_pi fun _ => hs) fun x _ => ?_
      dsimp only
      rw [hsum x]
    rw [hstep, SLT.Z_prod (K₁ := g) (K₂ := fun v : Fin n → α => ∑ j : Fin n, g (v j)) hg hsum_m,
      Z_pi_sum_eq_pow s hs g hg n N, pow_succ']

/-- The Gaussian Frobenius-squared loss on `matBox δ` has exact learning pair `(m * q / 2, 1)`. -/
theorem hasLaplacePair_frobSq {m q : ℕ} {δ : ℝ} (hδ : 0 < δ) :
    SLT.HasLaplacePair
      (SLT.Z (volume.restrict (matBox (a := m) (b := q) δ)) (fun B => frobSq (Matrix.of B)))
      ((m : ℝ) * q / 2) 1 := by
  have h1 := hasLaplacePair_sq_Icc hδ
  have hrow : SLT.HasLaplacePair
      (fun N => SLT.Z (volume.restrict (Set.Icc (-δ) δ)) (fun x : ℝ => x ^ 2) N ^ q)
      ((q : ℝ) / 2) 1 := by
    have hp := SLT.HasLaplacePair.finset_prod (Finset.univ : Finset (Fin q)) (fun _ _ => h1)
    simpa [div_eq_mul_inv, mul_comm, Finset.prod_const] using hp
  have hmat : SLT.HasLaplacePair
      (fun N => (SLT.Z (volume.restrict (Set.Icc (-δ) δ)) (fun x : ℝ => x ^ 2) N ^ q) ^ m)
      ((m : ℝ) * q / 2) 1 := by
    have hp := SLT.HasLaplacePair.finset_prod (Finset.univ : Finset (Fin m)) (fun _ _ => hrow)
    simpa [mul_div_assoc, Finset.prod_const] using hp
  refine hmat.congr (Eventually.of_forall fun N => ?_)
  dsimp only
  have hinner := Z_pi_sum_eq_pow (α := ℝ) (Set.Icc (-δ) δ) measurableSet_Icc
    (fun x : ℝ => x ^ 2) (by fun_prop) q N
  have houter := Z_pi_sum_eq_pow (α := Fin q → ℝ)
    (Set.pi Set.univ fun _ : Fin q => Set.Icc (-δ) δ)
    (MeasurableSet.univ_pi fun _ => measurableSet_Icc)
    (fun v : Fin q → ℝ => ∑ j : Fin q, (v j) ^ 2) (by fun_prop) m N
  rw [← hinner]
  exact houter.symm

/-! ### 3. Volume transformation under right matrix multiplication & scaling -/

/-- Right multiplication `B ↦ B * M` on `RMat m q` scales Lebesgue measure by `|det M|^{-m}`. -/
theorem map_mul_right_volume {m q : ℕ} (M : Matrix (Fin q) (Fin q) ℝ) (hM : M.det ≠ 0) :
    Measure.map (fun B : RMat m q => toRMat (Matrix.of B * M)) volume =
      ENNReal.ofReal |(M.det)⁻¹| ^ m • volume := by
  let L : (Fin q → ℝ) →ₗ[ℝ] (Fin q → ℝ) := Matrix.toLin' Mᵀ
  have hdet : LinearMap.det L = M.det := by
    rw [LinearMap.det_toLin', Matrix.det_transpose]
  have hLdet : LinearMap.det L ≠ 0 := by rw [hdet]; exact hM
  have hrow : Measure.map (⇑L) (volume : Measure (Fin q → ℝ)) =
      ENNReal.ofReal |(M.det)⁻¹| • volume := by
    rw [← hdet]
    exact Real.map_linearMap_volume_pi_eq_smul_volume_pi hLdet
  have hfn : (fun B : RMat m q => toRMat (Matrix.of B * M)) = fun B i => L (B i) := by
    funext B i j
    simp only [toRMat_apply, Matrix.mul_apply, Matrix.of_apply, L, Matrix.toLin'_apply,
      Matrix.mulVec, dotProduct, Matrix.transpose_apply, mul_comm]
  rw [hfn]
  have hsf1 : SigmaFinite (ENNReal.ofReal |(M.det)⁻¹| • (volume : Measure (Fin q → ℝ))) :=
    sigmaFinite_ofReal_smul volume |(M.det)⁻¹|
  have hsf : ∀ _ : Fin m, SigmaFinite (Measure.map (⇑L) (volume : Measure (Fin q → ℝ))) := fun _ => by
    rw [hrow]; exact hsf1
  change Measure.map (fun B i => L (B i)) (Measure.pi fun _ : Fin m => (volume : Measure (Fin q → ℝ))) = _
  rw [@Measure.pi_map_pi _ _ _ _ _ _ _ (fun _ : Fin m => (⇑L : (Fin q → ℝ) → Fin q → ℝ)) hsf
    (fun _ => L.continuous_of_finiteDimensional.measurable.aemeasurable)]
  simp_rw [hrow]
  exact measure_pi_const_smul (volume : Measure (Fin q → ℝ)) (ENNReal.ofReal |(M.det)⁻¹|)

/-- Scaling `X ↦ ρ • X` on `RMat a b` (`ρ > 0`) multiplies `volume` by
`ENNReal.ofReal (ρ⁻¹) ^ (a * b)`. -/
theorem map_smul_matrix_volume {a b : ℕ} {ρ : ℝ} (hρ : 0 < ρ) :
    Measure.map (fun X : RMat a b => ρ • X) volume =
      ENNReal.ofReal (ρ⁻¹) ^ (a * b) • volume := by
  have h1 : Measure.map (fun x : ℝ => ρ * x) volume = ENNReal.ofReal (ρ⁻¹) • volume := by
    have h := Real.map_volume_mul_left hρ.ne'
    simpa [abs_of_pos (inv_pos.mpr hρ)] using h
  have hsf1 : SigmaFinite (ENNReal.ofReal (ρ⁻¹) • (volume : Measure ℝ)) :=
    sigmaFinite_ofReal_smul volume (ρ⁻¹)
  have hrow : Measure.map (fun v : Fin b → ℝ => ρ • v) volume =
      ENNReal.ofReal (ρ⁻¹) ^ b • volume := by
    have hsf : ∀ _ : Fin b, SigmaFinite (Measure.map (fun x : ℝ => ρ * x) volume) := fun _ => by
      rw [h1]; exact hsf1
    change Measure.map (fun (v : Fin b → ℝ) j => ρ * v j)
      (Measure.pi fun _ : Fin b => (volume : Measure ℝ)) = _
    rw [@Measure.pi_map_pi _ _ _ _ _ _ _ (fun _ : Fin b => fun x : ℝ => ρ * x) hsf
      (fun _ => (measurable_const.mul measurable_id).aemeasurable)]
    simp_rw [h1]
    exact measure_pi_const_smul (volume : Measure ℝ) (ENNReal.ofReal (ρ⁻¹))
  have hsf2 : SigmaFinite (ENNReal.ofReal (ρ⁻¹) ^ b • (volume : Measure (Fin b → ℝ))) :=
    sigmaFinite_ofReal_pow_smul volume (ρ⁻¹) b
  have hsf' : ∀ _ : Fin a, SigmaFinite (Measure.map (fun v : Fin b → ℝ => ρ • v) volume) := fun _ => by
    rw [hrow]; exact hsf2
  change Measure.map (fun (X : RMat a b) i => ρ • X i)
    (Measure.pi fun _ : Fin a => (volume : Measure (Fin b → ℝ))) = _
  rw [@Measure.pi_map_pi _ _ _ _ _ _ _ (fun _ : Fin a => fun v : Fin b → ℝ => ρ • v) hsf'
    (fun _ => (continuous_const_smul ρ).measurable.aemeasurable)]
  simp_rw [hrow]
  rw [measure_pi_const_smul (volume : Measure (Fin b → ℝ)) (ENNReal.ofReal (ρ⁻¹) ^ b), ← pow_mul,
    mul_comm b a]
  rfl

/-! ### 4. Rank-0 excess matrix factorization: scaling invariance -/

/-- The rank-0 excess matrix factorization loss `‖B₂ Δ₂₂‖_F^2`. -/
def rank0Loss {m k' n' : ℕ} (p : RMat m k' × RMat k' n') : ℝ :=
  frobSq (Matrix.of p.1 * Matrix.of p.2)

theorem measurable_rank0Loss {m k' n' : ℕ} :
    Measurable (rank0Loss (m := m) (k' := k') (n' := n')) := by
  unfold rank0Loss frobSq
  simp only [Matrix.mul_apply, Matrix.of_apply]
  fun_prop

/-- Scaling `(B₂, Δ₂₂) ↦ (ρ • B₂, ρ • Δ₂₂)` relates `Z` on `matBox ρ ×ˢ matBox ρ` to `matBox 1 ×ˢ matBox 1`. -/
theorem rank0_Z_scale {m k' n' : ℕ} {ρ : ℝ} (hρ : 0 < ρ) (N : ℝ) :
    SLT.Z (volume.restrict (matBox (a := m) (b := k') ρ ×ˢ matBox (a := k') (b := n') ρ))
      rank0Loss N =
    ENNReal.ofReal ρ ^ (m * k' + k' * n') *
      SLT.Z (volume.restrict (matBox (a := m) (b := k') 1 ×ˢ matBox (a := k') (b := n') 1))
        rank0Loss (ρ ^ 4 * N) := by
  let S : RMat m k' × RMat k' n' → RMat m k' × RMat k' n' := fun p => (ρ • p.1, ρ • p.2)
  have hS_meas : Measurable S :=
    ((continuous_const_smul ρ).comp continuous_fst).prodMk
      ((continuous_const_smul ρ).comp continuous_snd) |>.measurable
  have hmap : Measure.map S volume =
      ENNReal.ofReal (ρ⁻¹) ^ (m * k' + k' * n') • volume := by
    have h1 := map_smul_matrix_volume (a := m) (b := k') hρ
    have h2 := map_smul_matrix_volume (a := k') (b := n') hρ
    have hsf1 : SigmaFinite (ENNReal.ofReal (ρ⁻¹) ^ (m * k') • (volume : Measure (RMat m k'))) :=
      sigmaFinite_ofReal_pow_smul volume (ρ⁻¹) (m * k')
    have hsf2 : SigmaFinite (ENNReal.ofReal (ρ⁻¹) ^ (k' * n') • (volume : Measure (RMat k' n'))) :=
      sigmaFinite_ofReal_pow_smul volume (ρ⁻¹) (k' * n')
    have hsf1' : SigmaFinite (Measure.map (fun X : RMat m k' => ρ • X) volume) := by
      rw [h1]; exact hsf1
    have hsf2' : SigmaFinite (Measure.map (fun X : RMat k' n' => ρ • X) volume) := by
      rw [h2]; exact hsf2
    change Measure.map (Prod.map (fun X => ρ • X) (fun Y => ρ • Y)) (volume.prod volume) = _
    rw [← Measure.map_prod_map _ _ (continuous_const_smul ρ).measurable
      (continuous_const_smul ρ).measurable, h1, h2, smul_prod_measure, prod_smul_measure,
      smul_smul, pow_add]
    rfl
  have hpre : S ⁻¹' (matBox ρ ×ˢ matBox ρ) = matBox 1 ×ˢ matBox 1 := by
    ext ⟨B, A⟩
    simp only [Set.mem_preimage, Set.mem_prod, mem_matBox_iff, S, Pi.smul_apply, smul_eq_mul,
      abs_mul, abs_of_pos hρ]
    constructor <;> rintro ⟨hB, hA⟩
    · exact ⟨fun i j => (mul_le_iff_le_one_right hρ).mp (hB i j),
        fun i j => (mul_le_iff_le_one_right hρ).mp (hA i j)⟩
    · exact ⟨fun i j => by nlinarith [hB i j], fun i j => by nlinarith [hA i j]⟩
  have hloss : ∀ p, rank0Loss (S p) = ρ ^ 4 * rank0Loss p := fun ⟨B, A⟩ => by
    unfold rank0Loss frobSq
    simp only [S, Matrix.mul_apply, Matrix.of_apply, Pi.smul_apply, smul_eq_mul]
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    have hsum : (∑ r : Fin k', (ρ * B i r) * (ρ * A r j)) =
        ρ ^ 2 * ∑ r : Fin k', B i r * A r j := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun r _ => by ring
    rw [hsum]; ring
  have hbox_meas : MeasurableSet (matBox (a := m) (b := k') ρ ×ˢ matBox (a := k') (b := n') ρ) :=
    (measurableSet_matBox ρ).prod (measurableSet_matBox ρ)
  have hkey :
      ∫⁻ p, (matBox (a := m) (b := k') ρ ×ˢ matBox (a := k') (b := n') ρ).indicator
        (fun x => ENNReal.ofReal (Real.exp (-(N * rank0Loss x)))) (S p) =
      ENNReal.ofReal (ρ⁻¹) ^ (m * k' + k' * n') *
        SLT.Z (volume.restrict (matBox (a := m) (b := k') ρ ×ˢ matBox (a := k') (b := n') ρ))
          rank0Loss N := by
    have hf : Measurable fun x : RMat m k' × RMat k' n' =>
        ENNReal.ofReal (Real.exp (-(N * rank0Loss x))) := by
      have := measurable_rank0Loss (m := m) (k' := k') (n' := n')
      fun_prop
    rw [← lintegral_map (hf.indicator hbox_meas) hS_meas, hmap, lintegral_smul_measure,
      smul_eq_mul, lintegral_indicator hbox_meas]
    rfl
  have hLHS :
      ∫⁻ p, (matBox (a := m) (b := k') ρ ×ˢ matBox (a := k') (b := n') ρ).indicator
        (fun x => ENNReal.ofReal (Real.exp (-(N * rank0Loss x)))) (S p) =
      SLT.Z (volume.restrict (matBox (a := m) (b := k') 1 ×ˢ matBox (a := k') (b := n') 1))
        rank0Loss (ρ ^ 4 * N) := by
    have hpt : ∀ p : RMat m k' × RMat k' n',
        (matBox (a := m) (b := k') ρ ×ˢ matBox (a := k') (b := n') ρ).indicator
          (fun x => ENNReal.ofReal (Real.exp (-(N * rank0Loss x)))) (S p) =
        (matBox (a := m) (b := k') 1 ×ˢ matBox (a := k') (b := n') 1).indicator
          (fun x => ENNReal.ofReal (Real.exp (-((ρ ^ 4 * N) * rank0Loss x)))) p := fun p => by
      by_cases hp : p ∈ matBox (a := m) (b := k') 1 ×ˢ matBox (a := k') (b := n') 1
      · have hSp : S p ∈ matBox (a := m) (b := k') ρ ×ˢ matBox (a := k') (b := n') ρ := by
          rw [← hpre] at hp; exact hp
        rw [Set.indicator_of_mem hSp, Set.indicator_of_mem hp, hloss]
        congr 3; ring
      · have hSp : S p ∉ matBox (a := m) (b := k') ρ ×ˢ matBox (a := k') (b := n') ρ :=
          fun h => hp (by rw [← hpre]; exact h)
        rw [Set.indicator_of_notMem hSp, Set.indicator_of_notMem hp]
    simp_rw [hpt]
    rw [lintegral_indicator ((measurableSet_matBox (a := m) (b := k') 1).prod
      (measurableSet_matBox (a := k') (b := n') 1))]
    rfl
  rw [hLHS] at hkey
  have hinv : ENNReal.ofReal ρ ^ (m * k' + k' * n') *
      ENNReal.ofReal (ρ⁻¹) ^ (m * k' + k' * n') = 1 := by
    rw [← mul_pow, ← ENNReal.ofReal_mul hρ.le, mul_inv_cancel₀ hρ.ne', ENNReal.ofReal_one, one_pow]
  calc
    SLT.Z (volume.restrict (matBox ρ ×ˢ matBox ρ)) rank0Loss N
      = 1 * SLT.Z (volume.restrict (matBox ρ ×ˢ matBox ρ)) rank0Loss N := (one_mul _).symm
    _ = ENNReal.ofReal ρ ^ (m * k' + k' * n') *
          (ENNReal.ofReal (ρ⁻¹) ^ (m * k' + k' * n') *
            SLT.Z (volume.restrict (matBox ρ ×ˢ matBox ρ)) rank0Loss N) := by
        rw [← mul_assoc, hinv]
    _ = ENNReal.ofReal ρ ^ (m * k' + k' * n') *
          SLT.Z (volume.restrict (matBox 1 ×ˢ matBox 1)) rank0Loss (ρ ^ 4 * N) := by
        rw [← hkey]

/-- Radius invariance of `HasLaplacePair` for the rank-0 excess matrix factorization problem. -/
theorem rank0_hasLaplacePair_of_unit {m k' n' : ℕ} {lam : ℝ} {m_log : ℕ} {ρ : ℝ} (hρ : 0 < ρ)
    (h : SLT.HasLaplacePair
      (SLT.Z (volume.restrict (matBox (a := m) (b := k') 1 ×ˢ matBox (a := k') (b := n') 1))
        rank0Loss) lam m_log) :
    SLT.HasLaplacePair
      (SLT.Z (volume.restrict (matBox (a := m) (b := k') ρ ×ˢ matBox (a := k') (b := n') ρ))
        rank0Loss) lam m_log := by
  have hc_pos : ENNReal.ofReal ρ ^ (m * k' + k' * n') ≠ 0 :=
    (ENNReal.pow_pos (ENNReal.ofReal_pos.mpr hρ) _).ne'
  have hc_top : ENNReal.ofReal ρ ^ (m * k' + k' * n') ≠ ⊤ :=
    (ENNReal.pow_lt_top ENNReal.ofReal_lt_top).ne
  have h1 := (h.comp_mul (pow_pos hρ 4)).const_mul hc_pos hc_top
  exact h1.congr (Eventually.of_forall fun N => (rank0_Z_scale hρ N).symm)

/-! ### 5. Exact LoRA fiber loss & Schur-complement Laplace reduction -/

/-- The three decoupled matrix blocks `(Δ₁₁, Δ₁₂, Δ₂₁)`. -/
abbrev DeltaSide (q k' n' : ℕ) :=
  RMat q q × RMat q n' × RMat k' q

/-- The core transformed blocks `(B₁, B₂, Δ₂₂)`. -/
abbrev CoreBlock (m q k' n' : ℕ) :=
  RMat m q × (RMat m k' × RMat k' n')

/-- Uniform box `matBox δ` on `DeltaSide`. -/
def deltaBox {q k' n' : ℕ} (δ : ℝ) : Set (DeltaSide q k' n') :=
  matBox δ ×ˢ (matBox δ ×ˢ matBox δ)

/-- Uniform box `matBox δ` on `CoreBlock`. -/
def coreBox {m q k' n' : ℕ} (δ : ℝ) : Set (CoreBlock m q k' n') :=
  matBox δ ×ˢ (matBox δ ×ˢ matBox δ)

/-- Uniform box `matBox δ` on all 6 perturbation blocks around `(0, fromBlocks 1 0 0 0)`. -/
def fiberBox {m q k' n' : ℕ} (δ : ℝ) : Set (DeltaSide q k' n' × CoreBlock m q k' n') :=
  deltaBox δ ×ˢ coreBox δ

/-- The exact squared-Frobenius LoRA fiber loss `‖B (A₀ + Δ)‖_F^2` around
`(B₀, A₀) = (0, fromBlocks 1 0 0 0)`. -/
def loraFiberLoss {m q k' n' : ℕ} (p : DeltaSide q k' n' × CoreBlock m q k' n') : ℝ :=
  let Δ₁₁ := Matrix.of p.1.1
  let Δ₁₂ := Matrix.of p.1.2.1
  let Δ₂₁ := Matrix.of p.1.2.2
  let B₁ := Matrix.of p.2.1
  let B₂ := Matrix.of p.2.2.1
  let Δ₂₂ := Matrix.of p.2.2.2
  let M := (1 : Matrix (Fin q) (Fin q) ℝ) + Δ₁₁
  frobSq (B₁ * M + B₂ * Δ₂₁) + frobSq (B₁ * Δ₁₂ + B₂ * Δ₂₂)

/-- The decoupled model loss `‖B₁'‖_F^2 + ‖B₂ Δ₂₂'‖_F^2` on `CoreBlock`. -/
def decoupledCoreLoss {m q k' n' : ℕ} (c : CoreBlock m q k' n') : ℝ :=
  frobSq (Matrix.of c.1) + rank0Loss c.2

/-- The fixed-`Δ` affine Schur-complement map on `CoreBlock`:
`(B₁, B₂, Δ₂₂) ↦ (B₁ M + B₂ Δ₂₁, B₂, Δ₂₂ - Δ₂₁ M⁻¹ Δ₁₂)`. -/
noncomputable def schurMap {m q k' n' : ℕ} (d : DeltaSide q k' n') :
    CoreBlock m q k' n' → CoreBlock m q k' n' :=
  let M := (1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of d.1
  let Δ₁₂ := Matrix.of d.2.1
  let Δ₂₁ := Matrix.of d.2.2
  fun c =>
    ( toRMat (Matrix.of c.1 * M) + toRMat (Matrix.of c.2.1 * Δ₂₁),
      ( c.2.1,
        c.2.2 - toRMat (Δ₂₁ * M⁻¹ * Δ₁₂) ) )

private theorem measurable_matMul_const {a b c : ℕ} (A : Matrix (Fin b) (Fin c) ℝ) :
    Measurable (fun X : RMat a b => toRMat (Matrix.of X * A)) := by
  refine measurable_pi_iff.mpr fun i => measurable_pi_iff.mpr fun j => ?_
  have hfn : (fun X : RMat a b => toRMat (Matrix.of X * A) i j) =
      fun X => ∑ r : Fin b, X i r * A r j := by
    funext X; simp [toRMat_apply, Matrix.mul_apply]
  rw [hfn]
  fun_prop

theorem measurable_schurMap {m q k' n' : ℕ} (d : DeltaSide q k' n') :
    Measurable (schurMap (m := m) d) := by
  unfold schurMap
  refine Measurable.prodMk ?_ (Measurable.prodMk ?_ ?_)
  · exact ((measurable_matMul_const _).comp measurable_fst).add
      ((measurable_matMul_const _).comp (measurable_fst.comp measurable_snd))
  · exact measurable_fst.comp measurable_snd
  · exact (measurable_snd.comp measurable_snd).sub measurable_const

/-- Pushforward of `volume` on `CoreBlock` by `schurMap d` is `|det M|^{-m} • volume`. -/
theorem map_schurMap_volume {m q k' n' : ℕ} (d : DeltaSide q k' n')
    (hM : ((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of d.1).det ≠ 0) :
    Measure.map (schurMap (m := m) d) volume =
      ENNReal.ofReal |(((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of d.1).det)⁻¹| ^ m • volume := by
  set M := (1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of d.1
  set Δ₁₂ := Matrix.of d.2.1
  set Δ₂₁ := Matrix.of d.2.2
  set shiftA : RMat k' n' := toRMat (Δ₂₁ * M⁻¹ * Δ₁₂)
  let T₁ : CoreBlock m q k' n' → CoreBlock m q k' n' :=
    fun c => (toRMat (Matrix.of c.1 * M), c.2)
  let T₂ : CoreBlock m q k' n' → CoreBlock m q k' n' :=
    fun c => (c.1 + toRMat (Matrix.of c.2.1 * Δ₂₁), c.2)
  let T₃ : CoreBlock m q k' n' → CoreBlock m q k' n' :=
    fun c => (c.1, (c.2.1, c.2.2 - shiftA))
  have hcomp : schurMap (m := m) d = T₃ ∘ T₂ ∘ T₁ := rfl
  have hT₁_meas : Measurable T₁ :=
    ((measurable_matMul_const M).comp measurable_fst).prodMk measurable_snd
  have hT₂_meas : Measurable T₂ :=
    (measurable_fst.add ((measurable_matMul_const Δ₂₁).comp
      (measurable_fst.comp measurable_snd))).prodMk measurable_snd
  have hT₃_meas : Measurable T₃ :=
    measurable_fst.prodMk ((measurable_fst.comp measurable_snd).prodMk
      ((measurable_snd.comp measurable_snd).sub measurable_const))
  have hT₁ : Measure.map T₁ volume = ENNReal.ofReal |(M.det)⁻¹| ^ m • volume := by
    have hm := map_mul_right_volume (m := m) M hM
    have hsf1 : SigmaFinite (ENNReal.ofReal |(M.det)⁻¹| ^ m • (volume : Measure (RMat m q))) :=
      sigmaFinite_ofReal_pow_smul volume |(M.det)⁻¹| m
    have hsf1' : SigmaFinite (Measure.map (fun B : RMat m q => toRMat (Matrix.of B * M)) volume) := by
      rw [hm]; exact hsf1
    change Measure.map (Prod.map (fun B : RMat m q => toRMat (Matrix.of B * M)) id)
      (volume.prod volume) = _
    rw [← Measure.map_prod_map _ _ (measurable_matMul_const M) measurable_id, hm, Measure.map_id,
      smul_prod_measure]
    rfl
  have hT₂ : Measure.map T₂ volume = volume := by
    refine Measure.ext fun s hs => ?_
    rw [Measure.map_apply hT₂_meas hs]
    change (volume.prod volume) (T₂ ⁻¹' s) = (volume.prod volume) s
    rw [Measure.prod_apply_symm (hT₂_meas hs), Measure.prod_apply_symm hs]
    refine lintegral_congr fun y => ?_
    have hpre : (Prod.mk · y) ⁻¹' (T₂ ⁻¹' s) =
        (fun x => x + toRMat (Matrix.of y.1 * Δ₂₁)) ⁻¹' ((Prod.mk · y) ⁻¹' s) := rfl
    rw [hpre]
    have hmp := measurePreserving_add_right (volume : Measure (RMat m q))
      (toRMat (Matrix.of y.1 * Δ₂₁))
    exact hmp.measure_preimage (hs.preimage measurable_prodMk_right).nullMeasurableSet
  have hT₃ : Measure.map T₃ volume = volume := by
    have hsub := measurePreserving_sub_right (volume : Measure (RMat k' n')) shiftA
    have hp1 : MeasurePreserving (Prod.map (id : RMat m k' → _) (fun X => X - shiftA))
        volume volume :=
      (MeasurePreserving.id volume).prod hsub
    have hp2 : MeasurePreserving (Prod.map (id : RMat m q → _)
        (Prod.map id (fun X => X - shiftA))) volume volume :=
      (MeasurePreserving.id volume).prod hp1
    exact hp2.map_eq
  rw [hcomp, ← Measure.map_map hT₃_meas (hT₂_meas.comp hT₁_meas),
    ← Measure.map_map hT₂_meas hT₁_meas, hT₁, Measure.map_smul, hT₂, Measure.map_smul, hT₃]
  all_goals first | exact hT₂_meas.aemeasurable | exact hT₃_meas.aemeasurable

/-- Continuity at `0` yields `δ₀ > 0` such that on `deltaBox δ₀`, `1 + Δ₁₁` is invertible with
determinant in `[1/2, 2]`, `frobSq ((1 + Δ₁₁)⁻¹ * Δ₁₂) ≤ 1/4`, and `schurMap d` satisfies
the uniform box inclusions `coreBox (δ / 2) ⊆ schurMap d '' coreBox δ ⊆ coreBox (2 * δ)`
for every `0 < δ ≤ δ₀`. -/
theorem exists_good_schur_radius (m q k' n' : ℕ) :
    ∃ δ₀ : ℝ, 0 < δ₀ ∧ ∀ δ : ℝ, 0 < δ → δ ≤ δ₀ → ∀ d ∈ deltaBox (q := q) (k' := k') (n' := n') δ,
      let M := (1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of d.1
      IsUnit M.det ∧
      (1 / 2 : ℝ) ≤ |M.det| ∧ |M.det| ≤ 2 ∧
      frobSq (M⁻¹ * Matrix.of d.2.1) ≤ 1 / 4 ∧
      (∀ c ∈ coreBox (m := m) δ, schurMap (m := m) d c ∈ coreBox (m := m) (2 * δ)) ∧
      (∀ c' ∈ coreBox (m := m) (δ / 2), ∃ c ∈ coreBox (m := m) δ, schurMap (m := m) d c = c') := by
  have hdet_cont : Continuous (fun Δ : RMat q q =>
      ((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ).det) :=
    (continuous_const.add continuous_id).matrix_det
  have hnhds_det : ∀ᶠ Δ in nhds (0 : RMat q q),
      (1 / 2 : ℝ) < |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ).det| ∧
      |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ).det| < 2 := by
    have h1 : ContinuousAt (fun Δ : RMat q q =>
        |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ).det|) 0 := hdet_cont.continuousAt.abs
    have hval : |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of (0 : RMat q q)).det| = 1 := by
      have : Matrix.of (0 : RMat q q) = 0 := rfl
      rw [this, add_zero, Matrix.det_one, abs_one]
    have hmem : Set.Ioo (1 / 2 : ℝ) 2 ∈
        nhds |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of (0 : RMat q q)).det| := by
      rw [hval]; exact Ioo_mem_nhds (by norm_num) (by norm_num)
    exact h1.eventually hmem
  have hinv_contAt : ContinuousAt (fun Δ : RMat q q =>
      toRMat (((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ)⁻¹)) 0 := by
    have heq : (fun Δ : RMat q q => toRMat (((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ)⁻¹)) =
        fun Δ => (((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ).det)⁻¹ •
          toRMat (((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ).adjugate) := by
      funext Δ i j
      simp only [toRMat_apply, Matrix.inv_def, Matrix.smul_apply, smul_eq_mul,
        Pi.smul_apply, Ring.inverse_eq_inv]
    rw [heq]
    have hdet0 : ((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of (0 : RMat q q)).det ≠ 0 := by
      have : Matrix.of (0 : RMat q q) = 0 := rfl
      rw [this, add_zero, Matrix.det_one]; exact one_ne_zero
    have hinv_det : ContinuousAt (fun Δ : RMat q q =>
        (((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ).det)⁻¹) 0 :=
      hdet_cont.continuousAt.inv₀ hdet0
    have hadj : Continuous (fun Δ : RMat q q =>
        toRMat (((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ).adjugate)) :=
      (continuous_const.add continuous_id).matrix_adjugate
    exact hinv_det.smul hadj.continuousAt
  have hnhds_inv : ∀ᶠ Δ in nhds (0 : RMat q q),
      ∀ i j, |(((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ)⁻¹ - 1) i j| <
        (1 : ℝ) / (4 * ((q : ℝ) + 1)) := by
    let oneR : RMat q q := toRMat (1 : Matrix (Fin q) (Fin q) ℝ)
    have hsub : ContinuousAt (fun Δ : RMat q q =>
        toRMat (((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ)⁻¹) - oneR) 0 :=
      hinv_contAt.sub continuousAt_const
    have h0 : toRMat (((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of (0 : RMat q q))⁻¹) - oneR = 0 := by
      funext i j
      change (((1 : Matrix (Fin q) (Fin q) ℝ) + 0)⁻¹ - 1) i j = 0
      simp
    have hpos : (0 : ℝ) < 1 / (4 * ((q : ℝ) + 1)) := by positivity
    have hball : ∀ᶠ Δ in nhds (0 : RMat q q),
        dist (toRMat (((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ)⁻¹) - oneR) 0 <
          (1 : ℝ) / (4 * ((q : ℝ) + 1)) := by
      have hm : Metric.ball (0 : RMat q q) (1 / (4 * ((q : ℝ) + 1))) ∈
          nhds (toRMat (((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of (0 : RMat q q))⁻¹) - oneR) := by
        rw [h0]; exact Metric.ball_mem_nhds _ hpos
      exact hsub.eventually hm
    refine hball.mono fun Δ hΔ i j => ?_
    rw [dist_zero_right] at hΔ
    have hle1 : ‖(toRMat (((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ)⁻¹) - oneR) i‖ ≤
        ‖toRMat (((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ)⁻¹) - oneR‖ :=
      norm_le_pi_norm _ i
    have hle2 : ‖(toRMat (((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ)⁻¹) - oneR) i j‖ ≤
        ‖(toRMat (((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ)⁻¹) - oneR) i‖ :=
      norm_le_pi_norm _ j
    simpa [Real.norm_eq_abs, oneR, toRMat] using (hle2.trans hle1).trans_lt hΔ
  have hboth := hnhds_det.and hnhds_inv
  rw [Metric.eventually_nhds_iff] at hboth
  obtain ⟨r₁, hr₁, hr₁_prop⟩ := hboth
  let D : ℝ := (q : ℝ) + (k' : ℝ) + (n' : ℝ) + 1
  have hD : 1 ≤ D := by
    dsimp [D]; linarith [Nat.cast_nonneg (α := ℝ) q, Nat.cast_nonneg (α := ℝ) k',
      Nat.cast_nonneg (α := ℝ) n']
  let δ₀ : ℝ := min (r₁ / 2) (1 / (16 * D ^ 3))
  have hδ₀ : 0 < δ₀ := lt_min (by linarith) (by positivity)
  refine ⟨δ₀, hδ₀, fun δ hδ hδle d hd => ?_⟩
  rcases hd with ⟨hΔ₁₁, hΔ₁₂, hΔ₂₁⟩
  rw [mem_matBox_iff] at hΔ₁₁ hΔ₁₂ hΔ₂₁
  have hδ_lt_r₁ : δ < r₁ := lt_of_le_of_lt (hδle.trans (min_le_left _ _)) (by linarith)
  have hδ_D : δ ≤ 1 / (16 * D ^ 3) := hδle.trans (min_le_right _ _)
  have hΔ₁₁_dist : dist d.1 0 < r₁ := by
    rw [dist_zero_right]
    refine ((pi_norm_le_iff_of_nonneg hδ.le).mpr fun i =>
      (pi_norm_le_iff_of_nonneg hδ.le).mpr fun j => by
        simpa [Real.norm_eq_abs] using hΔ₁₁ i j).trans_lt hδ_lt_r₁
  obtain ⟨⟨hdet_lo, hdet_hi⟩, hinv_diff⟩ := hr₁_prop hΔ₁₁_dist
  clear hdet_cont hnhds_det hinv_contAt hnhds_inv hr₁_prop hΔ₁₁_dist
  set M := (1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of d.1
  set Δ₁₂ := Matrix.of d.2.1
  set Δ₂₁ := Matrix.of d.2.2
  have hdet_ne : M.det ≠ 0 := fun h0 => by rw [h0, abs_zero] at hdet_lo; linarith
  have hM_unit : IsUnit M.det := isUnit_iff_ne_zero.mpr hdet_ne
  have hMinv_entry : ∀ i j, |M⁻¹ i j| ≤ 2 := fun i j => by
    have hdij := (hinv_diff i j).le
    rw [Matrix.sub_apply] at hdij
    have h1le : |(1 : Matrix (Fin q) (Fin q) ℝ) i j| ≤ 1 := by
      by_cases hij : i = j <;> simp [Matrix.one_apply, hij]
    have hqpos : (1 : ℝ) / (4 * ((q : ℝ) + 1)) ≤ 1 := by
      rw [div_le_one (by positivity)]; linarith [Nat.cast_nonneg (α := ℝ) q]
    have htri := abs_sub_abs_le_abs_sub (M⁻¹ i j) ((1 : Matrix (Fin q) (Fin q) ℝ) i j)
    linarith
  have hq_le_D : (q : ℝ) ≤ D := by
    dsimp [D]; linarith [Nat.cast_nonneg (α := ℝ) k', Nat.cast_nonneg (α := ℝ) n']
  have hk_le_D : (k' : ℝ) ≤ D := by
    dsimp [D]; linarith [Nat.cast_nonneg (α := ℝ) q, Nat.cast_nonneg (α := ℝ) n']
  have hn_le_D : (n' : ℝ) ≤ D := by
    dsimp [D]; linarith [Nat.cast_nonneg (α := ℝ) q, Nat.cast_nonneg (α := ℝ) k']
  have hD_pos : 0 < D := by linarith
  have hδD3 : 16 * D ^ 3 * δ ≤ 1 := (le_div_iff₀' (by positivity)).mp hδ_D
  have hD2_le_D3 : D ^ 2 ≤ D ^ 3 := by nlinarith
  have hD_le_D2 : D ≤ D ^ 2 := by nlinarith
  have hδD2 : 16 * D ^ 2 * δ ≤ 1 := by
    have : 16 * D ^ 2 * δ ≤ 16 * D ^ 3 * δ :=
      mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hD2_le_D3 (by norm_num)) hδ.le
    linarith
  have hδD : 16 * D * δ ≤ 1 := by
    have : 16 * D * δ ≤ 16 * D ^ 2 * δ :=
      mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hD_le_D2 (by norm_num)) hδ.le
    linarith
  have hq2_le_D2 : (q : ℝ) ^ 2 ≤ D ^ 2 := pow_le_pow_left₀ (Nat.cast_nonneg q) hq_le_D 2
  have hfrob_C : frobSq (M⁻¹ * Δ₁₂) ≤ 1 / 4 := by
    have h1 : frobSq (M⁻¹ * Δ₁₂) ≤ frobSq M⁻¹ * frobSq Δ₁₂ := frobSq_mul_le _ _
    have hMinv_frob : frobSq M⁻¹ ≤ 4 * (q : ℝ) ^ 2 := by
      unfold frobSq
      calc ∑ i : Fin q, ∑ j : Fin q, (M⁻¹ i j) ^ 2
          ≤ ∑ _ : Fin q, ∑ _ : Fin q, (4 : ℝ) := by
            refine Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ => ?_
            have := pow_le_pow_left₀ (abs_nonneg (M⁻¹ i j)) (hMinv_entry i j) 2
            rw [sq_abs] at this; linarith
        _ = 4 * (q : ℝ) ^ 2 := by simp [Finset.sum_const]; ring
    have hΔ12_frob : frobSq Δ₁₂ ≤ (q : ℝ) * n' * δ ^ 2 := by
      unfold frobSq
      calc ∑ i : Fin q, ∑ j : Fin n', (Δ₁₂ i j) ^ 2
          ≤ ∑ _ : Fin q, ∑ _ : Fin n', δ ^ 2 := by
            refine Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ => ?_
            have := pow_le_pow_left₀ (abs_nonneg (Δ₁₂ i j)) (hΔ₁₂ i j) 2
            rw [sq_abs] at this; exact this
        _ = (q : ℝ) * n' * δ ^ 2 := by simp [Finset.sum_const]; ring
    have hprod : 4 * (q : ℝ) ^ 2 * ((q : ℝ) * n' * δ ^ 2) ≤ 1 / 4 := by
      have hqn' : (q : ℝ) * n' ≤ D ^ 2 := by
        calc (q : ℝ) * n' ≤ D * D :=
              mul_le_mul hq_le_D hn_le_D (Nat.cast_nonneg n') hD_pos.le
          _ = D ^ 2 := (sq D).symm
      have hqn : (q : ℝ) ^ 2 * ((q : ℝ) * n') ≤ D ^ 4 := by
        calc (q : ℝ) ^ 2 * ((q : ℝ) * n') ≤ D ^ 2 * D ^ 2 :=
              mul_le_mul hq2_le_D2 hqn' (by positivity) (by positivity)
          _ = D ^ 4 := by ring
      have hD2δ : D ^ 2 * δ ≤ 1 / 16 := by linarith
      have hD4δ2 : D ^ 4 * δ ^ 2 ≤ (1 / 16 : ℝ) ^ 2 := by
        calc D ^ 4 * δ ^ 2 = (D ^ 2 * δ) ^ 2 := by ring
          _ ≤ (1 / 16 : ℝ) ^ 2 := pow_le_pow_left₀ (by positivity) hD2δ 2
      calc 4 * (q : ℝ) ^ 2 * ((q : ℝ) * n' * δ ^ 2)
          = 4 * ((q : ℝ) ^ 2 * ((q : ℝ) * n')) * δ ^ 2 := by ring
        _ ≤ 4 * D ^ 4 * δ ^ 2 := by gcongr
        _ = 4 * (D ^ 4 * δ ^ 2) := by ring
        _ ≤ 4 * (1 / 16 : ℝ) ^ 2 := by gcongr
        _ ≤ 1 / 4 := by norm_num
    calc frobSq (M⁻¹ * Δ₁₂)
        ≤ frobSq M⁻¹ * frobSq Δ₁₂ := h1
      _ ≤ (4 * (q : ℝ) ^ 2) * ((q : ℝ) * n' * δ ^ 2) :=
          mul_le_mul hMinv_frob hΔ12_frob (frobSq_nonneg _) (by positivity)
      _ ≤ 1 / 4 := hprod
  have hshift_entry : ∀ i j, |(Δ₂₁ * M⁻¹ * Δ₁₂) i j| ≤ δ / 4 := fun i j => by
    have h1 : ∀ a b, |(Δ₂₁ * M⁻¹) a b| ≤ (q : ℝ) * δ * 2 := fun a b =>
      matMul_entry_abs_le Δ₂₁ M⁻¹ hδ.le hΔ₂₁ hMinv_entry a b
    have h2 : |((Δ₂₁ * M⁻¹) * Δ₁₂) i j| ≤ (q : ℝ) * ((q : ℝ) * δ * 2) * δ :=
      matMul_entry_abs_le (Δ₂₁ * M⁻¹) Δ₁₂ (by positivity) h1 hΔ₁₂ i j
    have h8 : 8 * (q : ℝ) ^ 2 * δ ≤ 1 := by
      calc 8 * (q : ℝ) ^ 2 * δ ≤ 8 * D ^ 2 * δ := by gcongr
        _ ≤ 16 * D ^ 2 * δ := by nlinarith [sq_nonneg D, hδ]
        _ ≤ 1 := hδD2
    calc |(Δ₂₁ * M⁻¹ * Δ₁₂) i j|
        ≤ (q : ℝ) * ((q : ℝ) * δ * 2) * δ := h2
      _ = (8 * (q : ℝ) ^ 2 * δ) * (δ / 4) := by ring
      _ ≤ 1 * (δ / 4) := mul_le_mul_of_nonneg_right h8 (by linarith)
      _ = δ / 4 := one_mul _
  refine ⟨hM_unit, hdet_lo.le, hdet_hi.le, hfrob_C, ?_, ?_⟩
  · rintro ⟨B₁, B₂, Δ₂₂⟩ ⟨hB₁, hB₂, hΔ₂₂⟩
    rw [mem_matBox_iff] at hB₁ hB₂ hΔ₂₂
    dsimp only at hB₁ hB₂ hΔ₂₂
    refine ⟨?_, ?_, ?_⟩
    · rw [mem_matBox_iff]
      intro i j
      change |toRMat (Matrix.of B₁ * (1 + Matrix.of d.1)) i j +
        toRMat (Matrix.of B₂ * Δ₂₁) i j| ≤ 2 * δ
      have hsplit : toRMat (Matrix.of B₁ * (1 + Matrix.of d.1)) i j +
          toRMat (Matrix.of B₂ * Δ₂₁) i j =
          B₁ i j + (Matrix.of B₁ * Matrix.of d.1) i j + (Matrix.of B₂ * Δ₂₁) i j := by
        simp only [toRMat_apply, Matrix.mul_add, Matrix.mul_one, Matrix.add_apply, Matrix.of_apply]
      rw [hsplit]
      have h1 : |(Matrix.of B₁ * Matrix.of d.1) i j| ≤ (q : ℝ) * δ * δ :=
        matMul_entry_abs_le (Matrix.of B₁) (Matrix.of d.1) hδ.le hB₁ hΔ₁₁ i j
      have h2 : |(Matrix.of B₂ * Δ₂₁) i j| ≤ (k' : ℝ) * δ * δ :=
        matMul_entry_abs_le (Matrix.of B₂) Δ₂₁ hδ.le hB₂ hΔ₂₁ i j
      have hqk : ((q : ℝ) + k') * δ ≤ 1 := by
        have hle : (q : ℝ) + k' ≤ D := by dsimp [D]; linarith [Nat.cast_nonneg (α := ℝ) n']
        calc ((q : ℝ) + k') * δ ≤ D * δ := mul_le_mul_of_nonneg_right hle hδ.le
          _ ≤ 16 * D * δ := by nlinarith
          _ ≤ 1 := hδD
      have hsum_le : (q : ℝ) * δ * δ + (k' : ℝ) * δ * δ ≤ δ := by
        calc (q : ℝ) * δ * δ + (k' : ℝ) * δ * δ = (((q : ℝ) + k') * δ) * δ := by ring
          _ ≤ 1 * δ := mul_le_mul_of_nonneg_right hqk hδ.le
          _ = δ := one_mul _
      have ht1 := abs_add_le (B₁ i j + (Matrix.of B₁ * Matrix.of d.1) i j)
        ((Matrix.of B₂ * Δ₂₁) i j)
      have ht2 := abs_add_le (B₁ i j) ((Matrix.of B₁ * Matrix.of d.1) i j)
      linarith [hB₁ i j]
    · rw [mem_matBox_iff]; intro i j; change |B₂ i j| ≤ 2 * δ; linarith [hB₂ i j]
    · rw [mem_matBox_iff]
      intro i j
      change |Δ₂₂ i j - (Δ₂₁ * M⁻¹ * Δ₁₂) i j| ≤ 2 * δ
      have htri := abs_sub (Δ₂₂ i j) ((Δ₂₁ * M⁻¹ * Δ₁₂) i j)
      linarith [hΔ₂₂ i j, hshift_entry i j]
  · rintro ⟨B₁', B₂, Δ₂₂'⟩ ⟨hB₁', hB₂, hΔ₂₂'⟩
    rw [mem_matBox_iff] at hB₁' hB₂ hΔ₂₂'
    dsimp only at hB₁' hB₂ hΔ₂₂'
    let B₁ : RMat m q := toRMat ((Matrix.of B₁' - Matrix.of B₂ * Δ₂₁) * M⁻¹)
    let Δ₂₂ : RMat k' n' := Δ₂₂' + toRMat (Δ₂₁ * M⁻¹ * Δ₁₂)
    refine ⟨(B₁, (B₂, Δ₂₂)), ⟨?_, ?_, ?_⟩, ?_⟩
    · rw [mem_matBox_iff]
      intro i j
      set W : Matrix (Fin m) (Fin q) ℝ := Matrix.of B₁' - Matrix.of B₂ * Δ₂₁
      have hW_entry : ∀ a b, |W a b| ≤ 3 * δ / 4 := fun a b => by
        change |B₁' a b - (Matrix.of B₂ * Δ₂₁) a b| ≤ 3 * δ / 4
        have h2 : |(Matrix.of B₂ * Δ₂₁) a b| ≤ (k' : ℝ) * (δ / 2) * δ :=
          matMul_entry_abs_le (Matrix.of B₂) Δ₂₁ (by linarith) hB₂ hΔ₂₁ a b
        have hkδ : (k' : ℝ) * δ ≤ 1 / 2 := by
          calc (k' : ℝ) * δ ≤ D * δ := mul_le_mul_of_nonneg_right hk_le_D hδ.le
            _ ≤ 1 / 2 := by linarith
        have h2' : (k' : ℝ) * (δ / 2) * δ ≤ δ / 4 := by
          calc (k' : ℝ) * (δ / 2) * δ = ((k' : ℝ) * δ) * (δ / 2) := by ring
            _ ≤ (1 / 2) * (δ / 2) := mul_le_mul_of_nonneg_right hkδ (by linarith)
            _ = δ / 4 := by ring
        have htri := abs_sub (B₁' a b) ((Matrix.of B₂ * Δ₂₁) a b)
        linarith [hB₁' a b]
      set E : Matrix (Fin q) (Fin q) ℝ := M⁻¹ - 1
      have hB₁_eq : B₁ i j = W i j + (W * E) i j := by
        change (W * M⁻¹) i j = _
        have : M⁻¹ = 1 + E := by dsimp [E]; abel
        conv_lhs => rw [this, Matrix.mul_add, Matrix.mul_one]
        rfl
      change |B₁ i j| ≤ δ
      rw [hB₁_eq]
      have hrem : |(W * E) i j| ≤ δ / 4 := by
        have hbound := matMul_entry_abs_le W E (by linarith) hW_entry
          (fun r s => (hinv_diff r s).le) i j
        have hfrac : (q : ℝ) / ((q : ℝ) + 1) ≤ 1 := by
          rw [div_le_one (by positivity)]; linarith
        calc |(W * E) i j|
            ≤ (q : ℝ) * (3 * δ / 4) * (1 / (4 * ((q : ℝ) + 1))) := hbound
          _ = ((q : ℝ) / ((q : ℝ) + 1)) * (3 * δ / 16) := by field_simp; ring
          _ ≤ 1 * (3 * δ / 16) := mul_le_mul_of_nonneg_right hfrac (by linarith)
          _ ≤ δ / 4 := by linarith
      have htri := abs_add_le (W i j) ((W * E) i j)
      linarith [hW_entry i j]
    · rw [mem_matBox_iff]; intro i j; change |B₂ i j| ≤ δ; linarith [hB₂ i j]
    · rw [mem_matBox_iff]
      intro i j
      change |Δ₂₂' i j + (Δ₂₁ * M⁻¹ * Δ₁₂) i j| ≤ δ
      have htri := abs_add_le (Δ₂₂' i j) ((Δ₂₁ * M⁻¹ * Δ₁₂) i j)
      linarith [hΔ₂₂' i j, hshift_entry i j]
    · unfold schurMap
      simp only [B₁, Δ₂₂, of_toRMat]
      refine Prod.ext ?_ (Prod.ext rfl ?_)
      · funext i j
        simp only [Pi.add_apply, toRMat_apply]
        change ((Matrix.of B₁' - Matrix.of B₂ * Δ₂₁) * M⁻¹ * M + Matrix.of B₂ * Δ₂₁) i j = B₁' i j
        rw [Matrix.mul_assoc, Matrix.nonsing_inv_mul M hM_unit, Matrix.mul_one, sub_add_cancel]
        rfl
      · exact add_sub_cancel_right Δ₂₂' (toRMat (Δ₂₁ * M⁻¹ * Δ₁₂))

/-- **Main Theorem (Schur-complement LoRA fiber reduction):**
If the rank-0 excess problem `(m, k', n')` has learning pair `(lam, m_log)` on the unit box, then
for all sufficiently small `δ > 0`, the exact LoRA fiber loss around the rank-`q` half-dead fiber
point `(B₀, A₀) = (0, fromBlocks 1 0 0 0)` on `fiberBox δ` has exact learning pair
`((m * q : ℝ) / 2 + lam, m_log)`. -/
theorem lora_schur_fiber_reduction {m q k' n' : ℕ} {lam : ℝ} {m_log : ℕ}
    (hrem : SLT.HasLaplacePair
      (SLT.Z (volume.restrict (matBox (a := m) (b := k') 1 ×ˢ matBox (a := k') (b := n') 1))
        rank0Loss) lam m_log) :
    ∃ δ₀ : ℝ, 0 < δ₀ ∧ ∀ δ : ℝ, 0 < δ → δ ≤ δ₀ →
      SLT.HasLaplacePair
        (SLT.Z (volume.restrict (fiberBox (m := m) (q := q) (k' := k') (n' := n') δ))
          loraFiberLoss)
        ((m : ℝ) * q / 2 + lam) m_log := by
  obtain ⟨δ₀, hδ₀, hgood⟩ := exists_good_schur_radius m q k' n'
  refine ⟨δ₀, hδ₀, fun δ hδ hδle => ?_⟩
  have hcore_meas : ∀ ρ : ℝ, MeasurableSet (coreBox (m := m) (q := q) (k' := k') (n' := n') ρ) :=
    fun ρ => (measurableSet_matBox ρ).prod ((measurableSet_matBox ρ).prod (measurableSet_matBox ρ))
  have hdelta_meas : MeasurableSet (deltaBox (q := q) (k' := k') (n' := n') δ) :=
    (measurableSet_matBox δ).prod ((measurableSet_matBox δ).prod (measurableSet_matBox δ))
  have hmodel : ∀ ρ : ℝ, 0 < ρ →
      SLT.HasLaplacePair
        (SLT.Z (volume.restrict (coreBox (m := m) (q := q) (k' := k') (n' := n') ρ))
          decoupledCoreLoss)
        ((m : ℝ) * q / 2 + lam) m_log := fun ρ hρ => by
    have hg := hasLaplacePair_frobSq (m := m) (q := q) hρ
    have hr := rank0_hasLaplacePair_of_unit hρ hrem
    have hp := hg.prod measurable_frobSq measurable_rank0Loss hr
    have hmul : 1 + m_log - 1 = m_log := by omega
    rw [hmul] at hp
    refine hp.congr (Eventually.of_forall fun N => ?_)
    have hprod_res :
        (volume.restrict (matBox (a := m) (b := q) ρ)).prod
          (volume.restrict (matBox (a := m) (b := k') ρ ×ˢ matBox (a := k') (b := n') ρ)) =
        volume.restrict (coreBox (m := m) (q := q) (k' := k') (n' := n') ρ) :=
      Measure.prod_restrict _ _
    rw [hprod_res]; rfl
  have hvol_pos : volume (deltaBox (q := q) (k' := k') (n' := n') δ) ≠ 0 := by
    have h1 := volume_matBox_pos (a := q) (b := q) hδ
    have h2 := volume_matBox_pos (a := q) (b := n') hδ
    have h3 := volume_matBox_pos (a := k') (b := q) hδ
    change (volume.prod (volume.prod volume)) (matBox δ ×ˢ (matBox δ ×ˢ matBox δ)) ≠ 0
    rw [Measure.prod_prod, Measure.prod_prod]
    exact (ENNReal.mul_pos h1.ne' (ENNReal.mul_pos h2.ne' h3.ne').ne').ne'
  have hvol_top : volume (deltaBox (q := q) (k' := k') (n' := n') δ) ≠ ⊤ := by
    have h1 := volume_matBox_lt_top (a := q) (b := q) hδ.le
    have h2 := volume_matBox_lt_top (a := q) (b := n') hδ.le
    have h3 := volume_matBox_lt_top (a := k') (b := q) hδ.le
    change (volume.prod (volume.prod volume)) (matBox δ ×ˢ (matBox δ ×ˢ matBox δ)) ≠ ⊤
    rw [Measure.prod_prod, Measure.prod_prod]
    exact ENNReal.mul_ne_top h1.ne (ENNReal.mul_ne_top h2.ne h3.ne)
  have hloss_meas : Measurable (loraFiberLoss (m := m) (q := q) (k' := k') (n' := n')) := by
    unfold loraFiberLoss frobSq
    simp only [Matrix.add_apply, Matrix.mul_apply, Matrix.of_apply, Matrix.one_apply]
    fun_prop
  have hdec_meas : Measurable (decoupledCoreLoss (m := m) (q := q) (k' := k') (n' := n')) :=
    (measurable_frobSq.comp measurable_fst).add (measurable_rank0Loss.comp measurable_snd)
  have hZ_eq : ∀ N : ℝ,
      SLT.Z (volume.restrict (fiberBox (m := m) (q := q) (k' := k') (n' := n') δ))
        loraFiberLoss N =
      ∫⁻ (d : DeltaSide q k' n') in deltaBox δ, ∫⁻ (c : CoreBlock m q k' n') in coreBox δ,
        ENNReal.ofReal (Real.exp (-(N * loraFiberLoss (d, c)))) := fun N => by
    unfold SLT.Z fiberBox
    change ∫⁻ p in deltaBox δ ×ˢ coreBox δ,
      ENNReal.ofReal (Real.exp (-(N * loraFiberLoss p))) ∂(volume.prod volume) = _
    rw [← Measure.prod_restrict]
    exact lintegral_prod _ (by fun_prop : Measurable fun p =>
      ENNReal.ofReal (Real.exp (-(N * loraFiberLoss p)))).aemeasurable
  have hdec_schur : ∀ (d : DeltaSide q k' n') (c : CoreBlock m q k' n'),
      decoupledCoreLoss (schurMap d c) =
      frobSq (Matrix.of c.1 * ((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of d.1) +
        Matrix.of c.2.1 * Matrix.of d.2.2) +
      frobSq (Matrix.of c.2.1 * (Matrix.of c.2.2 -
        Matrix.of d.2.2 * ((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of d.1)⁻¹ * Matrix.of d.2.1)) :=
    fun d c => by
      unfold decoupledCoreLoss rank0Loss schurMap
      rfl
  have hbounds : ∀ N : ℝ, 0 ≤ N →
      (ENNReal.ofReal ((2 : ℝ)⁻¹ ^ m) * volume (deltaBox (q := q) (k' := k') (n' := n') δ)) *
        SLT.Z (volume.restrict (coreBox (m := m) (q := q) (k' := k') (n' := n') (δ / 2)))
          decoupledCoreLoss (2 * N) ≤
      SLT.Z (volume.restrict (fiberBox (m := m) (q := q) (k' := k') (n' := n') δ))
        loraFiberLoss N ∧
      SLT.Z (volume.restrict (fiberBox (m := m) (q := q) (k' := k') (n' := n') δ))
        loraFiberLoss N ≤
      (ENNReal.ofReal ((2 : ℝ) ^ m) * volume (deltaBox (q := q) (k' := k') (n' := n') δ)) *
        SLT.Z (volume.restrict (coreBox (m := m) (q := q) (k' := k') (n' := n') (2 * δ)))
          decoupledCoreLoss ((1 / 4 : ℝ) * N) := fun N hN => by
    rw [hZ_eq N]
    constructor
    · have hslice : ∀ d ∈ deltaBox (q := q) (k' := k') (n' := n') δ,
          ENNReal.ofReal ((2 : ℝ)⁻¹ ^ m) *
            SLT.Z (volume.restrict (coreBox (m := m) (q := q) (k' := k') (n' := n') (δ / 2)))
              decoupledCoreLoss (2 * N) ≤
          ∫⁻ (c : CoreBlock m q k' n') in coreBox δ,
            ENNReal.ofReal (Real.exp (-(N * loraFiberLoss (d, c)))) := fun d hd => by
        obtain ⟨hM_unit, hdet_lo, hdet_hi, hC, _, hsurj⟩ := hgood δ hδ hδle d hd
        set M := (1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of d.1
        have hdet_ne : M.det ≠ 0 := isUnit_iff_ne_zero.mp hM_unit
        have hstep1 :
            ∫⁻ (c : CoreBlock m q k' n') in coreBox δ,
              ENNReal.ofReal (Real.exp (-((2 * N) * decoupledCoreLoss (schurMap d c)))) ≤
            ∫⁻ (c : CoreBlock m q k' n') in coreBox δ,
              ENNReal.ofReal (Real.exp (-(N * loraFiberLoss (d, c)))) := by
          refine setLIntegral_mono' (hcore_meas δ) fun c _ => ?_
          apply ENNReal.ofReal_le_ofReal
          apply Real.exp_le_exp.mpr
          have hup := schur_loss_upper (Matrix.of c.1) (Matrix.of c.2.1) M hM_unit
            (Matrix.of d.2.1) (Matrix.of d.2.2) (Matrix.of c.2.2) hC
          have hmul := mul_le_mul_of_nonneg_left hup hN
          rw [hdec_schur d c]
          unfold loraFiberLoss
          dsimp only
          linarith
        have hstep2 :
            ∫⁻ (c : CoreBlock m q k' n'), (coreBox (m := m) (δ / 2)).indicator
              (fun c' => ENNReal.ofReal (Real.exp (-((2 * N) * decoupledCoreLoss c'))))
              (schurMap d c) ≤
            ∫⁻ (c : CoreBlock m q k' n') in coreBox δ,
              ENNReal.ofReal (Real.exp (-((2 * N) * decoupledCoreLoss (schurMap d c)))) := by
          rw [← lintegral_indicator (hcore_meas δ)]
          refine lintegral_mono fun c => ?_
          by_cases hc' : schurMap d c ∈ coreBox (m := m) (δ / 2)
          · obtain ⟨c0, hc0_mem, hc0_eq⟩ := hsurj (schurMap d c) hc'
            have hceqc0 : c = c0 := by
              have h21 : c.2.1 = c0.2.1 := congrArg (fun x => x.2.1) hc0_eq.symm
              have h22 : c.2.2 = c0.2.2 := by
                have hsub : c.2.2 - toRMat (Matrix.of d.2.2 * M⁻¹ * Matrix.of d.2.1) =
                    c0.2.2 - toRMat (Matrix.of d.2.2 * M⁻¹ * Matrix.of d.2.1) :=
                  congrArg (fun x => x.2.2) hc0_eq.symm
                exact sub_left_inj.mp hsub
              have h1 : c.1 = c0.1 := by
                have hadd : toRMat (Matrix.of c.1 * M) + toRMat (Matrix.of c.2.1 * Matrix.of d.2.2) =
                    toRMat (Matrix.of c0.1 * M) + toRMat (Matrix.of c0.2.1 * Matrix.of d.2.2) :=
                  congrArg (fun x => x.1) hc0_eq.symm
                rw [h21] at hadd
                have hmulR : toRMat (Matrix.of c.1 * M) = toRMat (Matrix.of c0.1 * M) :=
                  add_right_cancel hadd
                have hmul : Matrix.of c.1 * M = Matrix.of c0.1 * M := toRMat_injective hmulR
                have heqM : Matrix.of c.1 = Matrix.of c0.1 := by
                  calc Matrix.of c.1 = Matrix.of c.1 * M * M⁻¹ := by
                        rw [Matrix.mul_assoc, Matrix.mul_nonsing_inv M hM_unit, Matrix.mul_one]
                    _ = Matrix.of c0.1 * M * M⁻¹ := by rw [hmul]
                    _ = Matrix.of c0.1 := by
                        rw [Matrix.mul_assoc, Matrix.mul_nonsing_inv M hM_unit, Matrix.mul_one]
                exact Matrix.of.injective heqM
              exact Prod.ext h1 (Prod.ext h21 h22)
            rw [Set.indicator_of_mem hc', Set.indicator_of_mem (hceqc0 ▸ hc0_mem)]
          · rw [Set.indicator_of_notMem hc']; exact bot_le
        have hstep3 :
            ∫⁻ (c : CoreBlock m q k' n'), (coreBox (m := m) (q := q) (k' := k') (n' := n') (δ / 2)).indicator
              (fun c' => ENNReal.ofReal (Real.exp (-((2 * N) * decoupledCoreLoss c'))))
              (schurMap d c) =
            ENNReal.ofReal |(M.det)⁻¹| ^ m *
              SLT.Z (volume.restrict (coreBox (m := m) (q := q) (k' := k') (n' := n') (δ / 2)))
                decoupledCoreLoss (2 * N) := by
          have hg : Measurable fun c' : CoreBlock m q k' n' =>
              ENNReal.ofReal (Real.exp (-((2 * N) * decoupledCoreLoss c'))) := by fun_prop
          rw [← lintegral_map (hg.indicator (hcore_meas (δ / 2))) (measurable_schurMap d),
            map_schurMap_volume d hdet_ne, lintegral_smul_measure, smul_eq_mul,
            lintegral_indicator (hcore_meas (δ / 2))]
          rfl
        have hjac : ENNReal.ofReal ((2 : ℝ)⁻¹ ^ m) ≤ ENNReal.ofReal |(M.det)⁻¹| ^ m := by
          rw [ENNReal.ofReal_pow (by positivity)]
          refine pow_le_pow_left' (ENNReal.ofReal_le_ofReal ?_) m
          rw [abs_inv]
          exact inv_anti₀ (by linarith) hdet_hi
        calc ENNReal.ofReal ((2 : ℝ)⁻¹ ^ m) *
              SLT.Z (volume.restrict (coreBox (m := m) (q := q) (k' := k') (n' := n') (δ / 2)))
                decoupledCoreLoss (2 * N)
            ≤ ENNReal.ofReal |(M.det)⁻¹| ^ m *
              SLT.Z (volume.restrict (coreBox (m := m) (q := q) (k' := k') (n' := n') (δ / 2)))
                decoupledCoreLoss (2 * N) := by
              gcongr
          _ = ∫⁻ (c : CoreBlock m q k' n'), (coreBox (m := m) (q := q) (k' := k') (n' := n') (δ / 2)).indicator
                (fun c' => ENNReal.ofReal (Real.exp (-((2 * N) * decoupledCoreLoss c'))))
                (schurMap d c) := hstep3.symm
          _ ≤ ∫⁻ (c : CoreBlock m q k' n') in coreBox δ,
                ENNReal.ofReal (Real.exp (-(N * loraFiberLoss (d, c)))) :=
              hstep2.trans hstep1
      calc (ENNReal.ofReal ((2 : ℝ)⁻¹ ^ m) * volume (deltaBox (q := q) (k' := k') (n' := n') δ)) *
            SLT.Z (volume.restrict (coreBox (m := m) (q := q) (k' := k') (n' := n') (δ / 2)))
              decoupledCoreLoss (2 * N)
          = (ENNReal.ofReal ((2 : ℝ)⁻¹ ^ m) *
              SLT.Z (volume.restrict (coreBox (m := m) (q := q) (k' := k') (n' := n') (δ / 2)))
                decoupledCoreLoss (2 * N)) *
            volume (deltaBox (q := q) (k' := k') (n' := n') δ) := by ring
        _ = ∫⁻ _ in deltaBox (q := q) (k' := k') (n' := n') δ, (ENNReal.ofReal ((2 : ℝ)⁻¹ ^ m) *
              SLT.Z (volume.restrict (coreBox (m := m) (q := q) (k' := k') (n' := n') (δ / 2)))
                decoupledCoreLoss (2 * N)) := by
            rw [setLIntegral_const]
        _ ≤ ∫⁻ (d : DeltaSide q k' n') in deltaBox δ, ∫⁻ (c : CoreBlock m q k' n') in coreBox δ,
              ENNReal.ofReal (Real.exp (-(N * loraFiberLoss (d, c)))) :=
            setLIntegral_mono' hdelta_meas hslice
    · have hslice : ∀ d ∈ deltaBox (q := q) (k' := k') (n' := n') δ,
          ∫⁻ (c : CoreBlock m q k' n') in coreBox δ,
            ENNReal.ofReal (Real.exp (-(N * loraFiberLoss (d, c)))) ≤
          ENNReal.ofReal ((2 : ℝ) ^ m) *
            SLT.Z (volume.restrict (coreBox (m := m) (q := q) (k' := k') (n' := n') (2 * δ)))
              decoupledCoreLoss ((1 / 4 : ℝ) * N) := fun d hd => by
        obtain ⟨hM_unit, hdet_lo, hdet_hi, hC, hfwd, _⟩ := hgood δ hδ hδle d hd
        set M := (1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of d.1
        have hdet_ne : M.det ≠ 0 := isUnit_iff_ne_zero.mp hM_unit
        have hstep1 :
            ∫⁻ (c : CoreBlock m q k' n') in coreBox δ,
              ENNReal.ofReal (Real.exp (-(N * loraFiberLoss (d, c)))) ≤
            ∫⁻ (c : CoreBlock m q k' n') in coreBox δ,
              ENNReal.ofReal (Real.exp (-(((1 / 4 : ℝ) * N) * decoupledCoreLoss (schurMap d c)))) := by
          refine setLIntegral_mono' (hcore_meas δ) fun c _ => ?_
          apply ENNReal.ofReal_le_ofReal
          apply Real.exp_le_exp.mpr
          have hlo := schur_loss_lower (Matrix.of c.1) (Matrix.of c.2.1) M hM_unit
            (Matrix.of d.2.1) (Matrix.of d.2.2) (Matrix.of c.2.2) hC
          have hmul := mul_le_mul_of_nonneg_left hlo hN
          rw [hdec_schur d c]
          unfold loraFiberLoss
          dsimp only
          linarith
        have hstep2 :
            ∫⁻ (c : CoreBlock m q k' n') in coreBox δ,
              ENNReal.ofReal (Real.exp (-(((1 / 4 : ℝ) * N) * decoupledCoreLoss (schurMap d c)))) ≤
            ∫⁻ (c : CoreBlock m q k' n'), (coreBox (m := m) (2 * δ)).indicator
              (fun c' => ENNReal.ofReal (Real.exp (-(((1 / 4 : ℝ) * N) * decoupledCoreLoss c'))))
              (schurMap d c) := by
          rw [← lintegral_indicator (hcore_meas δ)]
          refine lintegral_mono fun c => ?_
          by_cases hc : c ∈ coreBox (m := m) δ
          · rw [Set.indicator_of_mem hc, Set.indicator_of_mem (hfwd c hc)]
          · rw [Set.indicator_of_notMem hc]; exact bot_le
        have hstep3 :
            ∫⁻ (c : CoreBlock m q k' n'), (coreBox (m := m) (q := q) (k' := k') (n' := n') (2 * δ)).indicator
              (fun c' => ENNReal.ofReal (Real.exp (-(((1 / 4 : ℝ) * N) * decoupledCoreLoss c'))))
              (schurMap d c) =
            ENNReal.ofReal |(M.det)⁻¹| ^ m *
              SLT.Z (volume.restrict (coreBox (m := m) (q := q) (k' := k') (n' := n') (2 * δ)))
                decoupledCoreLoss ((1 / 4 : ℝ) * N) := by
          have hg : Measurable fun c' : CoreBlock m q k' n' =>
              ENNReal.ofReal (Real.exp (-(((1 / 4 : ℝ) * N) * decoupledCoreLoss c'))) := by fun_prop
          rw [← lintegral_map (hg.indicator (hcore_meas (2 * δ))) (measurable_schurMap d),
            map_schurMap_volume d hdet_ne, lintegral_smul_measure, smul_eq_mul,
            lintegral_indicator (hcore_meas (2 * δ))]
          rfl
        have hjac : ENNReal.ofReal |(M.det)⁻¹| ^ m ≤ ENNReal.ofReal ((2 : ℝ) ^ m) := by
          rw [ENNReal.ofReal_pow (by positivity)]
          refine pow_le_pow_left' (ENNReal.ofReal_le_ofReal ?_) m
          rw [abs_inv]
          have : (|M.det|)⁻¹ ≤ (1 / 2 : ℝ)⁻¹ := inv_anti₀ (by norm_num) hdet_lo
          linarith
        calc ∫⁻ (c : CoreBlock m q k' n') in coreBox δ,
              ENNReal.ofReal (Real.exp (-(N * loraFiberLoss (d, c))))
            ≤ ENNReal.ofReal |(M.det)⁻¹| ^ m *
                SLT.Z (volume.restrict (coreBox (m := m) (q := q) (k' := k') (n' := n') (2 * δ)))
                  decoupledCoreLoss ((1 / 4 : ℝ) * N) :=
              (hstep1.trans hstep2).trans hstep3.le
          _ ≤ ENNReal.ofReal ((2 : ℝ) ^ m) *
                SLT.Z (volume.restrict (coreBox (m := m) (q := q) (k' := k') (n' := n') (2 * δ)))
                  decoupledCoreLoss ((1 / 4 : ℝ) * N) := by
              gcongr
      calc ∫⁻ (d : DeltaSide q k' n') in deltaBox δ, ∫⁻ (c : CoreBlock m q k' n') in coreBox δ,
              ENNReal.ofReal (Real.exp (-(N * loraFiberLoss (d, c))))
          ≤ ∫⁻ _ in deltaBox (q := q) (k' := k') (n' := n') δ, (ENNReal.ofReal ((2 : ℝ) ^ m) *
              SLT.Z (volume.restrict (coreBox (m := m) (q := q) (k' := k') (n' := n') (2 * δ)))
                decoupledCoreLoss ((1 / 4 : ℝ) * N)) :=
            setLIntegral_mono' hdelta_meas hslice
        _ = (ENNReal.ofReal ((2 : ℝ) ^ m) * volume (deltaBox (q := q) (k' := k') (n' := n') δ)) *
              SLT.Z (volume.restrict (coreBox (m := m) (q := q) (k' := k') (n' := n') (2 * δ)))
                decoupledCoreLoss ((1 / 4 : ℝ) * N) := by
            rw [setLIntegral_const]; ring
  let cLo : ℝ≥0∞ := ENNReal.ofReal ((2 : ℝ)⁻¹ ^ m) * volume (deltaBox (q := q) (k' := k') (n' := n') δ)
  let cHi : ℝ≥0∞ := ENNReal.ofReal ((2 : ℝ) ^ m) * volume (deltaBox (q := q) (k' := k') (n' := n') δ)
  have hlo_pair := ((hmodel (δ / 2) (by linarith)).comp_mul (by norm_num : (0 : ℝ) < 2)).const_mul
    (a := cLo)
    (ENNReal.mul_pos (ENNReal.ofReal_pos.mpr (by positivity)).ne' hvol_pos).ne'
    (ENNReal.mul_ne_top ENNReal.ofReal_lt_top.ne hvol_top)
  have hhi_pair := ((hmodel (2 * δ) (by linarith)).comp_mul (by norm_num : (0 : ℝ) < 1 / 4)).const_mul
    (a := cHi)
    (ENNReal.mul_pos (ENNReal.ofReal_pos.mpr (by positivity)).ne' hvol_pos).ne'
    (ENNReal.mul_ne_top ENNReal.ofReal_lt_top.ne hvol_top)
  refine ⟨hrem.1, ?_, ?_⟩
  · exact hlo_pair.2.1.mono ((eventually_ge_atTop 0).mono fun N hN => (hbounds N hN).1)
  · exact hhi_pair.2.2.mono ((eventually_ge_atTop 0).mono fun N hN => (hbounds N hN).2)

/-! ### 6. Unconditional corollary: Standard LoRA initialization (`q = k`, i.e., `k' = 0`) -/

/-- When `k' = 0` (no remaining fully-dead units), the rank-0 excess loss is identically `0`
on a one-point space and has exact learning pair `(0, 1)`. -/
theorem rank0_empty_hasLaplacePair (m n' : ℕ) :
    SLT.HasLaplacePair
      (SLT.Z (volume.restrict (matBox (a := m) (b := 0) 1 ×ˢ matBox (a := 0) (b := n') 1))
        rank0Loss) 0 1 := by
  have hZ : ∀ N : ℝ,
      SLT.Z (volume.restrict (matBox (a := m) (b := 0) 1 ×ˢ matBox (a := 0) (b := n') 1))
        rank0Loss N = 1 := fun N => by
    unfold SLT.Z
    have h0 : ∀ p : RMat m 0 × RMat 0 n', rank0Loss p = 0 := fun p => by
      unfold rank0Loss frobSq
      simp [Matrix.mul_apply]
    simp_rw [h0, mul_zero, neg_zero, Real.exp_zero, ENNReal.ofReal_one]
    rw [setLIntegral_const, one_mul]
    change (volume.prod volume) (matBox (a := m) (b := 0) 1 ×ˢ matBox (a := 0) (b := n') 1) = 1
    rw [Measure.prod_prod, volume_matBox (by norm_num), volume_matBox (by norm_num)]
    simp
  rw [SLT.hasLaplacePair_iff_bounds]
  refine ⟨le_rfl, ⟨1, one_pos, 2, fun N hN => ?_⟩, ⟨1, 2, fun N hN => ?_⟩⟩
  · rw [hZ N]; simp [Real.rpow_zero]
  · rw [hZ N]; simp [Real.rpow_zero]

/-- **Unconditional Lean-certified theorem for standard LoRA initialization (`q = k`, `k' = 0`):**
At any full-row-rank inactive `A₀` (`B₀ = 0`, `A₀ = [I_q | 0]`), the exact local learning pair on
every sufficiently small box is `((m * q : ℝ) / 2, 1)` for all `m, q, n' : ℕ`. -/
theorem lora_standard_init_exact_pair (m q n' : ℕ) :
    ∃ δ₀ : ℝ, 0 < δ₀ ∧ ∀ δ : ℝ, 0 < δ → δ ≤ δ₀ →
      SLT.HasLaplacePair
        (SLT.Z (volume.restrict (fiberBox (m := m) (q := q) (k' := 0) (n' := n') δ))
          loraFiberLoss)
        ((m : ℝ) * q / 2) 1 := by
  obtain ⟨δ₀, hδ₀, h⟩ := lora_schur_fiber_reduction (q := q) (rank0_empty_hasLaplacePair m n')
  refine ⟨δ₀, hδ₀, fun δ hδ hδle => ?_⟩
  simpa using h δ hδ hδle

end EatYourFiber
