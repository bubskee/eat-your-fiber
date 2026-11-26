/-
Copyright (c) 2026 Bobby Faber. All rights reserved.
Released under Apache 2.0 license.
-/
module

public import EatYourFiber.AoyagiWatanabe

/-!
# Self-Attention LoRA Coupling Dichotomy: `V+O` (4× Jump) vs. `Q+K` (`GL(q)` Flat Cancellation)

This module proves, completely unconditionally in Lean 4 (`[LC]`) on top of `AoyagiWatanabe.lean`
and `LoRASchurLaplace.lean`, the exact **all-orders** local learning pairs for joint Self-Attention
LoRA adapters on `(V, O)` and `(Q, K)` around the zero-loss fiber `(B₀ = 0, rank(A₀) = q)`:

1. **Exact All-Orders Nonlinear-to-Bilinear Reduction (`hasLaplacePair_of_leftMul_reduction`):**
   - For joint `V+O`: `(I_d + B_O A_O)(I_d + B_V A_V) - I_d = B_O A_O + ((I_d + B_O A_O) B_V) A_V`.
   - For joint `Q+K`: `(I_d + B_Q A_Q)ᵀ (I_d + B_K A_K) - I_d = B_K A_K + A_Qᵀ ((I_d + B_K A_K)ᵀ B_Q)ᵀ`.
   In both cases, left-multiplication by `M = I_d + B A` has bounded Jacobian `|det M|^{-H} ∈ [2^{-H}, 2^H]`
   and maps `matBox δ` bilipschitzly onto `matBox (δ/2) ⊆ · ⊆ matBox (2δ)`, eliminating the
   higher-order product term **exactly to all orders**.
2. **Joint `V+O` LoRA — Same-Side Stacking & 4× Supercritical Parabola:**
   - `attention_VO_exact_pair`: for all `d, H, q` with `q ≤ H` and `2 * q ≤ d`,
     `HasLaplacePair Z_{VO}(d, H, q, δ) (d * q + awLambda d (d - 2*q) (2*H - 2*q)) (awMult d (d - 2*q) (2*H - 2*q))`.
   - `attention_VO_subcritical_plateau`: for `H ≤ d` and `2 * q ≤ H`, flat plateau `(d * H - H² / 2, 1)`.
   - `attention_VO_supercritical_parabola`: for `H ≤ d` and `H ≤ 2 * q ≤ d`, exact `4×` parabola
     `(d * H - H² / 2 + 2 * (q - H / 2)², 1)`.
3. **Joint `Q+K` LoRA — Exact All-Orders `GL(q)` Gauge Symmetry & Complete Flat Cancellation:**
   - `attention_QK_gauge_orbit`: explicit `q²`-dimensional all-orders zero-loss gauge family
     `(I_d + B_Q(C) A_{Q,0})ᵀ (I_d + B_K(C) A_{K,0}) = I_d` for every `C` with `IsUnit (I_q + C).det`.
   - `twoSided_schur_reduction`: exact two-sided Schur complement on balanced `(q, q)` matrix fibers,
     yielding `(q * d - q² / 2 + awLambda (d - q) (d - q) k', awMult (d - q) (d - q) k')`.
   - `attention_QK_exact_pair` & `attention_QK_flat_pair`: for ALL `q ≤ H ≤ d`, the `q`-dependence
     cancels identically and the supercritical parabola is completely annihilated:
     `HasLaplacePair Z_{QK}(d, H, q, δ) (d * H - H² / 2) 1`.
4. **Standard LoRA Initialization Coupling Dichotomy (`attention_coupling_dichotomy_standard_init`):**
   At standard LoRA init `q = H` (`2 * H ≤ d`):
   - `V+O` sits at `(d * H, 1)` (jump `+H² / 2`, `4×` the single-projection jump `+H² / 8`),
   - `Q+K` sits at `(d * H - H² / 2, 1)` (jump `0`, identically flat!).
-/

@[expose] public section

namespace EatYourFiber

open MeasureTheory Set Filter Matrix Finset Real
open scoped ENNReal NNReal BigOperators

private local instance (priority := high) instSigmaFiniteVolumeRMat (a b : ℕ) :
    SigmaFinite (volume : Measure (RMat a b)) :=
  inferInstance

private local instance (priority := high) instSFiniteVolumeRMat (a b : ℕ) :
    SFinite (volume : Measure (RMat a b)) :=
  inferInstance

private local instance (priority := high) instSigmaFiniteVolumeProd {α β : Type*} [MeasureSpace α]
    [MeasureSpace β] [SigmaFinite (volume : Measure α)] [SigmaFinite (volume : Measure β)] :
    SigmaFinite (volume : Measure (α × β)) :=
  Measure.instSigmaFiniteProdVolume

private local instance (priority := high) instSFiniteVolumeProd {α β : Type*} [MeasureSpace α]
    [MeasureSpace β] [SFinite (volume : Measure α)] [SFinite (volume : Measure β)] :
    SFinite (volume : Measure (α × β)) := by
  rw [Measure.volume_eq_prod]; infer_instance

/-! ### 1. Measure-preserving matrix block splits and left-multiplication Jacobian -/

/-- Transpose equivalence on `RMat a b ≃ᵐ RMat b a`. -/
def transposeEquiv (a b : ℕ) : RMat a b ≃ᵐ RMat b a where
  toFun M j i := M i j
  invFun M i j := M j i
  left_inv _ := rfl
  right_inv _ := rfl
  measurable_toFun := Measurable.of_eval fun j => Measurable.of_eval fun i =>
    (measurable_pi_apply j).comp (measurable_pi_apply i)
  measurable_invFun := Measurable.of_eval fun i => Measurable.of_eval fun j =>
    (measurable_pi_apply i).comp (measurable_pi_apply j)

private theorem matBox_mono {a b : ℕ} {ρ₁ ρ₂ : ℝ} (h : ρ₁ ≤ ρ₂) :
    matBox (a := a) (b := b) ρ₁ ⊆ matBox ρ₂ := fun _ hM =>
  mem_matBox_iff.mpr fun i j => (mem_matBox_iff.mp hM i j).trans h

private theorem smul_prod_measure {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    (μ : Measure α) (ν : Measure β) [SigmaFinite μ] [SigmaFinite ν] (c : ℝ≥0∞)
    [SigmaFinite (c • μ)] :
    (c • μ).prod ν = c • (μ.prod ν) := by
  refine Measure.prod_eq (μ := c • μ) (ν := ν) fun s t _ _ => ?_
  simp [Measure.smul_apply, smul_eq_mul, mul_assoc]

private theorem sigmaFinite_ofReal_pow_smul {α : Type*} [MeasurableSpace α]
    (μ : Measure α) [SigmaFinite μ] (r : ℝ) (k : ℕ) :
    SigmaFinite (ENNReal.ofReal r ^ k • μ) := by
  have : ENNReal.ofReal r ^ k • μ = (r.toNNReal ^ k : ℝ≥0) • μ := by
    rw [ENNReal.smul_def, ENNReal.ofReal, ENNReal.coe_pow]
  rw [this]
  infer_instance

theorem measurePreserving_transposeEquiv (a b : ℕ) :
    MeasurePreserving (transposeEquiv a b) volume volume :=
  measurePreserving_transpose a b

theorem transposeEquiv_mem_matBox {a b : ℕ} {δ : ℝ} {M : RMat a b} :
    transposeEquiv a b M ∈ matBox δ ↔ M ∈ matBox δ := by
  simp only [mem_matBox_iff, transposeEquiv, MeasurableEquiv.coe_mk, Equiv.coe_fn_mk]
  exact ⟨fun h i j => h j i, fun h j i => h i j⟩

/-- Left multiplication `B ↦ M * B` on `RMat d H` scales Lebesgue measure by `|det M|^{-H}`. -/
theorem map_mul_left_volume {d H : ℕ} (M : Matrix (Fin d) (Fin d) ℝ) (hM : M.det ≠ 0) :
    Measure.map (fun B : RMat d H => toRMat (M * Matrix.of B)) volume =
      ENNReal.ofReal |(M.det)⁻¹| ^ H • volume := by
  have htr : (fun B : RMat d H => toRMat (M * Matrix.of B)) =
      (transposeEquiv H d) ∘ (fun X : RMat H d => toRMat (Matrix.of X * Mᵀ)) ∘ (transposeEquiv d H) := by
    funext B i r
    simp only [Function.comp_apply, transposeEquiv, MeasurableEquiv.coe_mk, Equiv.coe_fn_mk,
      toRMat_apply, Matrix.mul_apply, Matrix.of_apply, Matrix.transpose_apply]
    refine Finset.sum_congr rfl fun j _ => mul_comm _ _
  have hdetT : Mᵀ.det ≠ 0 := by rwa [Matrix.det_transpose]
  have hmeas_mul : Measurable (fun X : RMat H d => toRMat (Matrix.of X * Mᵀ)) := by
    refine Measurable.of_eval fun r => Measurable.of_eval fun i => ?_
    simp only [toRMat_apply, Matrix.mul_apply, Matrix.of_apply]
    fun_prop
  rw [htr, ← Measure.map_map (transposeEquiv H d).measurable
      (hmeas_mul.comp (transposeEquiv d H).measurable),
    ← Measure.map_map hmeas_mul (transposeEquiv d H).measurable,
    (measurePreserving_transposeEquiv d H).map_eq,
    map_mul_right_volume (m := H) Mᵀ hdetT, Matrix.det_transpose,
    Measure.map_smul, (measurePreserving_transposeEquiv H d).map_eq]
  · exact (transposeEquiv H d).measurable.aemeasurable

/-- Split rows of `RMat (a₁ + a₂) b` into `RMat a₁ b × RMat a₂ b`. -/
def splitRowsEquiv (a₁ a₂ b : ℕ) : RMat (a₁ + a₂) b ≃ᵐ RMat a₁ b × RMat a₂ b where
  toFun M := (fun i => M (Fin.castAdd a₂ i), fun i => M (Fin.natAdd a₁ i))
  invFun p i := if h : i.1 < a₁ then p.1 ⟨i.1, h⟩ else p.2 ⟨i.1 - a₁, by omega⟩
  left_inv M := by
    funext i j
    by_cases h : i.1 < a₁
    · simp only [h, ↓reduceDIte]; rfl
    · simp only [h, ↓reduceDIte]
      congr 1; ext; dsimp; omega
  right_inv p := by
    ext i j
    · simp [Fin.castAdd, i.2]
    · have h : ¬ (a₁ + i.1 < a₁) := by omega
      simp [Fin.natAdd, h]
  measurable_toFun := by
    refine Measurable.prodMk ?_ ?_
    · exact Measurable.of_eval fun i => measurable_pi_apply (Fin.castAdd a₂ i)
    · exact Measurable.of_eval fun i => measurable_pi_apply (Fin.natAdd a₁ i)
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

theorem measurePreserving_splitRowsEquiv (a₁ a₂ b : ℕ) :
    MeasurePreserving (splitRowsEquiv a₁ a₂ b) volume volume := by
  have hsymm : MeasurePreserving (splitRowsEquiv a₁ a₂ b).symm volume volume := by
    refine ⟨(splitRowsEquiv a₁ a₂ b).symm.measurable, ?_⟩
    symm
    refine Measure.pi_eq fun s hs => ?_
    rw [Measure.map_apply (splitRowsEquiv a₁ a₂ b).symm.measurable (MeasurableSet.univ_pi hs)]
    have hpre : (splitRowsEquiv a₁ a₂ b).symm ⁻¹' (Set.univ.pi s) =
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

theorem splitRowsEquiv_mem_matBox {a₁ a₂ b : ℕ} {δ : ℝ} {M : RMat (a₁ + a₂) b} :
    splitRowsEquiv a₁ a₂ b M ∈ matBox δ ×ˢ matBox δ ↔ M ∈ matBox δ := by
  simp only [Set.mem_prod, mem_matBox_iff, splitRowsEquiv, MeasurableEquiv.coe_mk, Equiv.coe_fn_mk]
  constructor
  · rintro ⟨h₁, h₂⟩ i j
    by_cases h : i.1 < a₁
    · exact h₁ ⟨i.1, h⟩ j
    · have heq : Fin.natAdd a₁ ⟨i.1 - a₁, by omega⟩ = i := by ext; dsimp; omega
      simpa [heq] using h₂ ⟨i.1 - a₁, by omega⟩ j
  · intro h
    exact ⟨fun i j => h (Fin.castAdd a₂ i) j, fun i j => h (Fin.natAdd a₁ i) j⟩

/-- Split columns of `RMat a (b₁ + b₂)` into `RMat a b₁ × RMat a b₂`. -/
def splitColsEquiv (a b₁ b₂ : ℕ) : RMat a (b₁ + b₂) ≃ᵐ RMat a b₁ × RMat a b₂ :=
  (transposeEquiv a (b₁ + b₂)).trans
    ((splitRowsEquiv b₁ b₂ a).trans ((transposeEquiv b₁ a).prodCongr (transposeEquiv b₂ a)))

theorem measurePreserving_splitColsEquiv (a b₁ b₂ : ℕ) :
    MeasurePreserving (splitColsEquiv a b₁ b₂) volume volume :=
  ((measurePreserving_transposeEquiv b₁ a).prod (measurePreserving_transposeEquiv b₂ a)).comp
    ((measurePreserving_splitRowsEquiv b₁ b₂ a).comp (measurePreserving_transposeEquiv a (b₁ + b₂)))

theorem splitColsEquiv_mem_matBox {a b₁ b₂ : ℕ} {δ : ℝ} {M : RMat a (b₁ + b₂)} :
    splitColsEquiv a b₁ b₂ M ∈ matBox δ ×ˢ matBox δ ↔ M ∈ matBox δ := by
  change (transposeEquiv b₁ a (splitRowsEquiv b₁ b₂ a (transposeEquiv a (b₁ + b₂) M)).1,
    transposeEquiv b₂ a (splitRowsEquiv b₁ b₂ a (transposeEquiv a (b₁ + b₂) M)).2) ∈
      matBox δ ×ˢ matBox δ ↔ M ∈ matBox δ
  simp only [Set.mem_prod, transposeEquiv_mem_matBox]
  exact (splitRowsEquiv_mem_matBox (M := transposeEquiv a (b₁ + b₂) M)).trans
    transposeEquiv_mem_matBox

private theorem splitRowsEquiv_apply_fst {a₁ a₂ b : ℕ} (M : RMat (a₁ + a₂) b) (i : Fin a₁) (j : Fin b) :
    (splitRowsEquiv a₁ a₂ b M).1 i j = M (Fin.castAdd a₂ i) j := rfl

private theorem splitRowsEquiv_apply_snd {a₁ a₂ b : ℕ} (M : RMat (a₁ + a₂) b) (i : Fin a₂) (j : Fin b) :
    (splitRowsEquiv a₁ a₂ b M).2 i j = M (Fin.natAdd a₁ i) j := rfl

private theorem splitColsEquiv_apply_fst {a b₁ b₂ : ℕ} (M : RMat a (b₁ + b₂)) (i : Fin a) (j : Fin b₁) :
    (splitColsEquiv a b₁ b₂ M).1 i j = M i (Fin.castAdd b₂ j) := rfl

private theorem splitColsEquiv_apply_snd {a b₁ b₂ : ℕ} (M : RMat a (b₁ + b₂)) (i : Fin a) (j : Fin b₂) :
    (splitColsEquiv a b₁ b₂ M).2 i j = M i (Fin.natAdd b₁ j) := rfl

private theorem splitRowsEquiv_symm_apply_castAdd {a₁ a₂ b : ℕ} (M₁ : RMat a₁ b) (M₂ : RMat a₂ b)
    (i : Fin a₁) (j : Fin b) :
    (splitRowsEquiv a₁ a₂ b).symm (M₁, M₂) (Fin.castAdd a₂ i) j = M₁ i j := by
  have h := splitRowsEquiv_apply_fst ((splitRowsEquiv a₁ a₂ b).symm (M₁, M₂)) i j
  simpa only [MeasurableEquiv.apply_symm_apply] using h.symm

private theorem splitRowsEquiv_symm_apply_natAdd {a₁ a₂ b : ℕ} (M₁ : RMat a₁ b) (M₂ : RMat a₂ b)
    (i : Fin a₂) (j : Fin b) :
    (splitRowsEquiv a₁ a₂ b).symm (M₁, M₂) (Fin.natAdd a₁ i) j = M₂ i j := by
  have h := splitRowsEquiv_apply_snd ((splitRowsEquiv a₁ a₂ b).symm (M₁, M₂)) i j
  simpa only [MeasurableEquiv.apply_symm_apply] using h.symm

private theorem splitColsEquiv_symm_apply_castAdd {a b₁ b₂ : ℕ} (M₁ : RMat a b₁) (M₂ : RMat a b₂)
    (i : Fin a) (j : Fin b₁) :
    (splitColsEquiv a b₁ b₂).symm (M₁, M₂) i (Fin.castAdd b₂ j) = M₁ i j := by
  have h := splitColsEquiv_apply_fst ((splitColsEquiv a b₁ b₂).symm (M₁, M₂)) i j
  simpa only [MeasurableEquiv.apply_symm_apply] using h.symm

private theorem splitColsEquiv_symm_apply_natAdd {a b₁ b₂ : ℕ} (M₁ : RMat a b₁) (M₂ : RMat a b₂)
    (i : Fin a) (j : Fin b₂) :
    (splitColsEquiv a b₁ b₂).symm (M₁, M₂) i (Fin.natAdd b₁ j) = M₂ i j := by
  have h := splitColsEquiv_apply_snd ((splitColsEquiv a b₁ b₂).symm (M₁, M₂)) i j
  simpa only [MeasurableEquiv.apply_symm_apply] using h.symm

private theorem measurePreserving_trans {α β γ : Type*} [MeasureSpace α] [MeasureSpace β]
    [MeasureSpace γ] {e₁ : α ≃ᵐ β} {e₂ : β ≃ᵐ γ}
    (h₁ : MeasurePreserving e₁ volume volume) (h₂ : MeasurePreserving e₂ volume volume) :
    MeasurePreserving (e₁.trans e₂) volume volume :=
  h₂.comp h₁

private theorem measurePreserving_prodCongr {α₁ α₂ β₁ β₂ : Type*}
    [MeasureSpace α₁] [MeasureSpace α₂] [MeasureSpace β₁] [MeasureSpace β₂]
    [SFinite (volume : Measure α₁)] [SFinite (volume : Measure α₂)]
    [SFinite (volume : Measure β₁)] [SFinite (volume : Measure β₂)]
    {e₁ : α₁ ≃ᵐ β₁} {e₂ : α₂ ≃ᵐ β₂}
    (h₁ : MeasurePreserving e₁ volume volume) (h₂ : MeasurePreserving e₂ volume volume) :
    MeasurePreserving (e₁.prodCongr e₂) volume volume :=
  h₁.prod h₂

private theorem prodCongr_apply_mk {α₁ α₂ β₁ β₂ : Type*}
    [MeasurableSpace α₁] [MeasurableSpace α₂] [MeasurableSpace β₁] [MeasurableSpace β₂]
    (e₁ : α₁ ≃ᵐ β₁) (e₂ : α₂ ≃ᵐ β₂) (a₁ : α₁) (a₂ : α₂) :
    (e₁.prodCongr e₂) (a₁, a₂) = (e₁ a₁, e₂ a₂) := rfl

/-- Permute rows of `RMat a b` along `σ : Fin a ≃ Fin a`. -/
def permRowsEquiv {a b : ℕ} (σ : Fin a ≃ Fin a) : RMat a b ≃ᵐ RMat a b :=
  MeasurableEquiv.piCongrLeft (fun _ : Fin a => (Fin b → ℝ)) σ

theorem measurePreserving_permRowsEquiv {a b : ℕ} (σ : Fin a ≃ Fin a) :
    MeasurePreserving (permRowsEquiv (b := b) σ) volume volume :=
  MeasureTheory.measurePreserving_piCongrLeft (fun _ : Fin a => (volume : Measure (Fin b → ℝ))) σ

theorem permRowsEquiv_mem_matBox {a b : ℕ} (σ : Fin a ≃ Fin a) {δ : ℝ} {M : RMat a b} :
    permRowsEquiv (b := b) σ M ∈ matBox δ ↔ M ∈ matBox δ := by
  simp only [mem_matBox_iff, permRowsEquiv]
  constructor
  · intro h i j
    have heq := Equiv.piCongrLeft_apply_apply (fun _ : Fin a => (Fin b → ℝ)) σ M i
    have h1 := h (σ i) j
    change (Equiv.piCongrLeft (fun _ : Fin a => (Fin b → ℝ)) σ M) (σ i) j |> (|·| ≤ δ) at h1
    rwa [heq] at h1
  · intro h i j
    have heq := Equiv.piCongrLeft_apply_apply (fun _ : Fin a => (Fin b → ℝ)) σ M (σ.symm i)
    simp only [Equiv.apply_symm_apply] at heq
    change (Equiv.piCongrLeft (fun _ : Fin a => (Fin b → ℝ)) σ M) i j |> (|·| ≤ δ)
    rw [heq]
    exact h (σ.symm i) j

/-- Permute columns of `RMat a b` along `σ : Fin b ≃ Fin b`. -/
def permColsEquiv {a b : ℕ} (σ : Fin b ≃ Fin b) : RMat a b ≃ᵐ RMat a b :=
  (transposeEquiv a b).trans ((permRowsEquiv σ).trans (transposeEquiv b a))

theorem measurePreserving_permColsEquiv {a b : ℕ} (σ : Fin b ≃ Fin b) :
    MeasurePreserving (permColsEquiv (a := a) σ) volume volume :=
  (measurePreserving_transposeEquiv b a).comp
    ((measurePreserving_permRowsEquiv σ).comp (measurePreserving_transposeEquiv a b))

theorem permColsEquiv_mem_matBox {a b : ℕ} (σ : Fin b ≃ Fin b) {δ : ℝ} {M : RMat a b} :
    permColsEquiv (a := a) σ M ∈ matBox δ ↔ M ∈ matBox δ := by
  simp only [permColsEquiv, MeasurableEquiv.trans_apply, transposeEquiv_mem_matBox,
    permRowsEquiv_mem_matBox]

/-- Cast dimensions of `RMat` along equalities `a = a'`, `b = b'`. -/
def castMatEquiv {a a' b b' : ℕ} (ha : a = a') (hb : b = b') : RMat a b ≃ᵐ RMat a' b' :=
  ha ▸ hb ▸ MeasurableEquiv.refl (RMat a b)

theorem measurePreserving_castMatEquiv {a a' b b' : ℕ} (ha : a = a') (hb : b = b') :
    MeasurePreserving (castMatEquiv ha hb) volume volume := by
  subst ha; subst hb; exact MeasurePreserving.id volume

theorem castMatEquiv_mem_matBox {a a' b b' : ℕ} (ha : a = a') (hb : b = b')
    {δ : ℝ} {M : RMat a b} :
    castMatEquiv ha hb M ∈ matBox δ ↔ M ∈ matBox δ := by
  subst ha; subst hb; rfl

theorem castMatEquiv_apply {a a' b b' : ℕ} (ha : a = a') (hb : b = b')
    (M : RMat a b) (i : Fin a) (j : Fin b) :
    castMatEquiv ha hb M (finCongr ha i) (finCongr hb j) = M i j := by
  subst ha; subst hb; rfl

/-- Pushforward of `SLT.Z` along a measure-preserving `MeasurableEquiv` that maps `S₁` onto `S₂`
and pulls `L₂` back to `L₁`. -/
theorem Z_eq_of_measurePreserving {α β : Type*} [MeasureSpace α] [MeasureSpace β]
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

/-! ### 2. Generic all-orders left-multiplication Laplace reduction -/

/-- Continuity of `p ↦ (M p)⁻¹` at `0` whenever `M` is continuous and `M 0 = 1`. -/
private theorem continuousAt_matrix_inv_of_one {P : Type*} [TopologicalSpace P] [Zero P] {d : ℕ}
    (M : P → Matrix (Fin d) (Fin d) ℝ) (hM_cont : Continuous M) (hM0 : M 0 = 1) :
    ContinuousAt (fun p => (M p)⁻¹) 0 := by
  have heq : (fun p => (M p)⁻¹) = fun p => ((M p).det)⁻¹ • (M p).adjugate := by
    funext p
    ext i j
    simp only [Matrix.inv_def, Matrix.smul_apply, smul_eq_mul, Ring.inverse_eq_inv]
  rw [heq]
  have hdet0 : (M 0).det ≠ 0 := by rw [hM0, Matrix.det_one]; exact one_ne_zero
  exact (hM_cont.matrix_det.continuousAt.inv₀ hdet0).smul hM_cont.matrix_adjugate.continuousAt

/-- Uniform neighborhood bound for a continuous matrix family `M : P → Matrix (Fin d) (Fin d) ℝ`
with `M 0 = 1` on a family of sets `B_P δ` shrinking to `0`. -/
theorem exists_good_leftMul_radius {P : Type*} [TopologicalSpace P] [Zero P]
    {d H : ℕ} (B_P : ℝ → Set P)
    (hB_nhds : ∀ U ∈ nhds (0 : P), ∃ δ₀ > 0, ∀ δ, 0 < δ → δ ≤ δ₀ → B_P δ ⊆ U)
    (M : P → Matrix (Fin d) (Fin d) ℝ)
    (hM_cont : Continuous M) (hM0 : M 0 = 1) :
    ∃ δ₀ : ℝ, 0 < δ₀ ∧ ∀ δ : ℝ, 0 < δ → δ ≤ δ₀ → ∀ p ∈ B_P δ,
      IsUnit (M p).det ∧
      (1 / 2 : ℝ) ≤ |(M p).det| ∧ |(M p).det| ≤ 2 ∧
      (∀ B ∈ matBox (a := d) (b := H) δ, toRMat (M p * Matrix.of B) ∈ matBox (2 * δ)) ∧
      (∀ B' ∈ matBox (a := d) (b := H) (δ / 2),
        ∃ B ∈ matBox (a := d) (b := H) δ, toRMat (M p * Matrix.of B) = B') := by
  have hdet_nhds : ∀ᶠ p in nhds (0 : P),
      (1 / 2 : ℝ) < |(M p).det| ∧ |(M p).det| < 2 := by
    have hcont : ContinuousAt (fun p => |(M p).det|) 0 :=
      (hM_cont.matrix_det.continuousAt).abs
    have hval : |(M 0).det| = 1 := by rw [hM0, Matrix.det_one, abs_one]
    have hmem : Set.Ioo (1 / 2 : ℝ) 2 ∈ nhds |(M 0).det| := by
      rw [hval]; exact Ioo_mem_nhds (by norm_num) (by norm_num)
    exact hcont.preimage_mem_nhds hmem
  have hrow_M : ∀ᶠ p in nhds (0 : P),
      ∀ i : Fin d, ∑ j : Fin d, |M p i j| < 2 := by
    rw [Filter.eventually_all]
    intro i
    have hcont : ContinuousAt (fun p => ∑ j : Fin d, |M p i j|) 0 :=
      (continuous_finsetSum _ fun j _ => ((continuous_apply j).comp
        ((continuous_apply i).comp hM_cont)).abs).continuousAt
    have hval : ∑ j : Fin d, |M 0 i j| = 1 := by
      rw [hM0]
      simp [Matrix.one_apply, apply_ite]
    have hmem : Set.Iio (2 : ℝ) ∈ nhds (∑ j : Fin d, |M 0 i j|) := by
      rw [hval]; exact Iio_mem_nhds (by norm_num)
    exact hcont.preimage_mem_nhds hmem
  have hinv_cont : ContinuousAt (fun p => (M p)⁻¹) 0 :=
    continuousAt_matrix_inv_of_one M hM_cont hM0
  have hinv0 : (M 0)⁻¹ = 1 := by rw [hM0]; simp
  have hrow_inv : ∀ᶠ p in nhds (0 : P),
      ∀ i : Fin d, ∑ j : Fin d, |(M p)⁻¹ i j| < 2 := by
    rw [Filter.eventually_all]
    intro i
    have hcont : ContinuousAt (fun p => ∑ j : Fin d, |(M p)⁻¹ i j|) 0 := by
      have h1 : ContinuousAt (fun X : Matrix (Fin d) (Fin d) ℝ => ∑ j : Fin d, |X i j|) (M 0)⁻¹ :=
        (continuous_finsetSum _ fun j _ => ((continuous_apply j).comp
          ((continuous_apply i).comp continuous_id)).abs).continuousAt
      exact ContinuousAt.comp (f := fun p => (M p)⁻¹) h1 hinv_cont
    have hval : ∑ j : Fin d, |(M 0)⁻¹ i j| = 1 := by
      rw [hinv0]
      simp [Matrix.one_apply, apply_ite]
    have hmem : Set.Iio (2 : ℝ) ∈ nhds (∑ j : Fin d, |(M 0)⁻¹ i j|) := by
      rw [hval]; exact Iio_mem_nhds (by norm_num)
    exact hcont.preimage_mem_nhds hmem
  have hall := hdet_nhds.and (hrow_M.and hrow_inv)
  obtain ⟨δ₀, hδ₀, hsub⟩ := hB_nhds _ hall
  refine ⟨δ₀, hδ₀, fun δ hδ hδle p hp => ?_⟩
  obtain ⟨⟨hdet_lo, hdet_hi⟩, hMrow, hinvrow⟩ := hsub δ hδ hδle hp
  have hdet_ne : (M p).det ≠ 0 := by
    intro h0; rw [h0, abs_zero] at hdet_lo; linarith
  have hunit : IsUnit (M p).det := isUnit_iff_ne_zero.mpr hdet_ne
  refine ⟨hunit, hdet_lo.le, hdet_hi.le, ?_, ?_⟩
  · intro B hB
    rw [mem_matBox_iff] at hB ⊢
    intro i r
    simp only [toRMat_apply, Matrix.mul_apply, Matrix.of_apply]
    calc |∑ j : Fin d, M p i j * B j r|
        ≤ ∑ j : Fin d, |M p i j * B j r| := Finset.abs_sum_le_sum_abs _ _
      _ = ∑ j : Fin d, |M p i j| * |B j r| := by simp_rw [abs_mul]
      _ ≤ ∑ j : Fin d, |M p i j| * δ :=
          Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (hB j r) (abs_nonneg _)
      _ = (∑ j : Fin d, |M p i j|) * δ := (Finset.sum_mul _ _ _).symm
      _ ≤ 2 * δ := mul_le_mul_of_nonneg_right (hMrow i).le hδ.le
  · intro B' hB'
    refine ⟨toRMat ((M p)⁻¹ * Matrix.of B'), ?_, ?_⟩
    · rw [mem_matBox_iff] at hB' ⊢
      intro i r
      simp only [toRMat_apply, Matrix.mul_apply, Matrix.of_apply]
      calc |∑ j : Fin d, (M p)⁻¹ i j * B' j r|
          ≤ ∑ j : Fin d, |(M p)⁻¹ i j * B' j r| := Finset.abs_sum_le_sum_abs _ _
        _ = ∑ j : Fin d, |(M p)⁻¹ i j| * |B' j r| := by simp_rw [abs_mul]
        _ ≤ ∑ j : Fin d, |(M p)⁻¹ i j| * (δ / 2) :=
            Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (hB' j r) (abs_nonneg _)
        _ = (∑ j : Fin d, |(M p)⁻¹ i j|) * (δ / 2) := (Finset.sum_mul _ _ _).symm
        _ ≤ 2 * (δ / 2) := mul_le_mul_of_nonneg_right (hinvrow i).le (by linarith)
        _ = δ := by ring
    · funext i r
      have hmul : M p * ((M p)⁻¹ * Matrix.of B') = Matrix.of B' := by
        rw [← Matrix.mul_assoc, Matrix.mul_nonsing_inv (M p) hunit, Matrix.one_mul]
      exact congrFun (congrFun hmul i) r

/-- **All-Orders Left-Multiplication Reduction Lemma:**
If `L_raw (p, B) = L_lin (p, M p * B)` where `M : P → Matrix (Fin d) (Fin d) ℝ` is continuous
with `M 0 = 1`, and `L_lin` has `HasLaplacePair (lam, m_log)` on `B_P ρ ×ˢ matBox ρ` for all
sufficiently small `ρ > 0`, then `L_raw` has the **exact same** `HasLaplacePair (lam, m_log)`
on `B_P δ ×ˢ matBox δ` for all sufficiently small `δ > 0`. -/
theorem hasLaplacePair_of_leftMul_reduction
    {P : Type*} [TopologicalSpace P] [MeasureSpace P] [SigmaFinite (volume : Measure P)]
    [Zero P] {d H : ℕ} {lam : ℝ} {m_log : ℕ}
    (B_P : ℝ → Set P)
    (hB_meas : ∀ ρ, MeasurableSet (B_P ρ))
    (hB_mono : ∀ {ρ₁ ρ₂}, ρ₁ ≤ ρ₂ → B_P ρ₁ ⊆ B_P ρ₂)
    (hB_nhds : ∀ U ∈ nhds (0 : P), ∃ δ₀ > 0, ∀ δ, 0 < δ → δ ≤ δ₀ → B_P δ ⊆ U)
    (M : P → Matrix (Fin d) (Fin d) ℝ)
    (hM_cont : Continuous M) (hM0 : M 0 = 1)
    (L_raw L_lin : P × RMat d H → ℝ)
    (hL_raw_meas : Measurable L_raw) (hL_lin_meas : Measurable L_lin)
    (heq : ∀ p B, L_raw (p, B) = L_lin (p, toRMat (M p * Matrix.of B)))
    (hlin : ∃ ρ₀ > 0, ∀ ρ, 0 < ρ → ρ ≤ ρ₀ →
      SLT.HasLaplacePair (SLT.Z (volume.restrict (B_P ρ ×ˢ matBox (a := d) (b := H) ρ)) L_lin) lam m_log) :
    ∃ δ₀ > 0, ∀ δ, 0 < δ → δ ≤ δ₀ →
      SLT.HasLaplacePair (SLT.Z (volume.restrict (B_P δ ×ˢ matBox (a := d) (b := H) δ)) L_raw) lam m_log := by
  obtain ⟨δ₁, hδ₁, hgood⟩ := exists_good_leftMul_radius (H := H) B_P hB_nhds M hM_cont hM0
  obtain ⟨ρ₀, hρ₀, hlin_pair⟩ := hlin
  refine ⟨min δ₁ (ρ₀ / 2), lt_min hδ₁ (by linarith), fun δ hδ hδle => ?_⟩
  have hδ_le1 : δ ≤ δ₁ := hδle.trans (min_le_left _ _)
  have h2δ_le : 2 * δ ≤ ρ₀ := by linarith [hδle.trans (min_le_right _ _)]
  have hδ2_le : δ / 2 ≤ ρ₀ := by linarith
  have hZ_eq : ∀ (L : P × RMat d H → ℝ) (hL : Measurable L) (ρ : ℝ) (N : ℝ),
      SLT.Z (volume.restrict (B_P ρ ×ˢ matBox (a := d) (b := H) ρ)) L N =
      ∫⁻ p in B_P ρ, ∫⁻ B in matBox (a := d) (b := H) ρ,
        ENNReal.ofReal (Real.exp (-(N * L (p, B)))) := fun L hL ρ N => by
    unfold SLT.Z
    have hprod : (volume : Measure (P × RMat d H)).restrict (B_P ρ ×ˢ matBox ρ) =
        ((volume : Measure P).restrict (B_P ρ)).prod
          ((volume : Measure (RMat d H)).restrict (matBox ρ)) := by
      rw [Measure.volume_eq_prod, ← Measure.prod_restrict]
    rw [hprod, MeasureTheory.lintegral_prod _ (by fun_prop)]
  have hbounds : ∀ N : ℝ,
      ENNReal.ofReal ((2 : ℝ)⁻¹ ^ H) *
        SLT.Z (volume.restrict (B_P (δ / 2) ×ˢ matBox (a := d) (b := H) (δ / 2))) L_lin N ≤
      SLT.Z (volume.restrict (B_P δ ×ˢ matBox (a := d) (b := H) δ)) L_raw N ∧
      SLT.Z (volume.restrict (B_P δ ×ˢ matBox (a := d) (b := H) δ)) L_raw N ≤
      ENNReal.ofReal ((2 : ℝ) ^ H) *
        SLT.Z (volume.restrict (B_P (2 * δ) ×ˢ matBox (a := d) (b := H) (2 * δ))) L_lin N := fun N => by
    rw [hZ_eq L_raw hL_raw_meas δ N, hZ_eq L_lin hL_lin_meas (δ / 2) N,
      hZ_eq L_lin hL_lin_meas (2 * δ) N]
    constructor
    · have hslice : ∀ p ∈ B_P δ,
          ENNReal.ofReal ((2 : ℝ)⁻¹ ^ H) *
            ∫⁻ B' in matBox (a := d) (b := H) (δ / 2), ENNReal.ofReal (Real.exp (-(N * L_lin (p, B')))) ≤
          ∫⁻ B in matBox (a := d) (b := H) δ, ENNReal.ofReal (Real.exp (-(N * L_raw (p, B)))) := fun p hp => by
        obtain ⟨hunit, _, hdet_hi, _, hsurj⟩ := hgood δ hδ hδ_le1 p hp
        have hdet_ne : (M p).det ≠ 0 := isUnit_iff_ne_zero.mp hunit
        have hmeas_phi : Measurable (fun B : RMat d H => toRMat (M p * Matrix.of B)) := by
          refine Measurable.of_eval fun i => Measurable.of_eval fun r => ?_
          simp only [toRMat_apply, Matrix.mul_apply, Matrix.of_apply]
          fun_prop
        have hg : Measurable (fun B' : RMat d H => ENNReal.ofReal (Real.exp (-(N * L_lin (p, B'))))) :=
          ENNReal.measurable_ofReal.comp (Real.measurable_exp.comp
            ((measurable_const.mul (hL_lin_meas.comp (measurable_const.prodMk measurable_id))).neg))
        have hjac : ENNReal.ofReal ((2 : ℝ)⁻¹ ^ H) ≤ ENNReal.ofReal |((M p).det)⁻¹| ^ H := by
          rw [ENNReal.ofReal_pow (by positivity)]
          refine pow_le_pow_left' (ENNReal.ofReal_le_ofReal ?_) H
          rw [abs_inv]
          exact inv_anti₀ (abs_pos.mpr hdet_ne) hdet_hi
        have hstep_map :
            ENNReal.ofReal |((M p).det)⁻¹| ^ H *
              ∫⁻ B' in matBox (a := d) (b := H) (δ / 2), ENNReal.ofReal (Real.exp (-(N * L_lin (p, B')))) =
            ∫⁻ B : RMat d H, (matBox (a := d) (b := H) (δ / 2)).indicator
              (fun B' => ENNReal.ofReal (Real.exp (-(N * L_lin (p, B')))))
              (toRMat (M p * Matrix.of B)) := by
          rw [← lintegral_map (hg.indicator (measurableSet_matBox (δ / 2))) hmeas_phi,
            map_mul_left_volume (M p) hdet_ne, lintegral_smul_measure, smul_eq_mul,
            lintegral_indicator (measurableSet_matBox (δ / 2))]
        have hstep_ind :
            ∫⁻ B : RMat d H, (matBox (a := d) (b := H) (δ / 2)).indicator
              (fun B' => ENNReal.ofReal (Real.exp (-(N * L_lin (p, B')))))
              (toRMat (M p * Matrix.of B)) ≤
            ∫⁻ B in matBox (a := d) (b := H) δ, ENNReal.ofReal (Real.exp (-(N * L_raw (p, B)))) := by
          rw [← lintegral_indicator (measurableSet_matBox δ)]
          refine lintegral_mono fun B => ?_
          by_cases hmem : toRMat (M p * Matrix.of B) ∈ matBox (a := d) (b := H) (δ / 2)
          · obtain ⟨B₀, hB₀_mem, hB₀_eq⟩ := hsurj _ hmem
            have hBeq : B = B₀ := by
              have hM : M p * Matrix.of B = M p * Matrix.of B₀ := by
                have := congrArg Matrix.of hB₀_eq.symm
                simpa [toRMat] using this
              have hinv := congrArg (fun X => (M p)⁻¹ * X) hM
              simp only [← Matrix.mul_assoc, Matrix.nonsing_inv_mul (M p) hunit, Matrix.one_mul] at hinv
              funext i r
              exact congrFun (congrFun hinv i) r
            rw [Set.indicator_of_mem hmem, Set.indicator_of_mem (hBeq ▸ hB₀_mem), heq p B]
          · rw [Set.indicator_of_notMem hmem]; exact bot_le
        calc ENNReal.ofReal ((2 : ℝ)⁻¹ ^ H) *
              ∫⁻ B' in matBox (a := d) (b := H) (δ / 2), ENNReal.ofReal (Real.exp (-(N * L_lin (p, B'))))
            ≤ ENNReal.ofReal |((M p).det)⁻¹| ^ H *
              ∫⁻ B' in matBox (a := d) (b := H) (δ / 2), ENNReal.ofReal (Real.exp (-(N * L_lin (p, B')))) := by
              gcongr
          _ = _ := hstep_map
          _ ≤ _ := hstep_ind
      calc ENNReal.ofReal ((2 : ℝ)⁻¹ ^ H) *
            ∫⁻ p in B_P (δ / 2), ∫⁻ B' in matBox (δ / 2), ENNReal.ofReal (Real.exp (-(N * L_lin (p, B'))))
          = ∫⁻ p in B_P (δ / 2), ENNReal.ofReal ((2 : ℝ)⁻¹ ^ H) *
              ∫⁻ B' in matBox (δ / 2), ENNReal.ofReal (Real.exp (-(N * L_lin (p, B')))) :=
            (MeasureTheory.lintegral_const_mul' _ _ ENNReal.ofReal_ne_top).symm
        _ ≤ ∫⁻ p in B_P δ, ENNReal.ofReal ((2 : ℝ)⁻¹ ^ H) *
              ∫⁻ B' in matBox (δ / 2), ENNReal.ofReal (Real.exp (-(N * L_lin (p, B')))) :=
            MeasureTheory.lintegral_mono_set (hB_mono (by linarith))
        _ ≤ ∫⁻ p in B_P δ, ∫⁻ B in matBox δ, ENNReal.ofReal (Real.exp (-(N * L_raw (p, B)))) :=
            MeasureTheory.setLIntegral_mono' (hB_meas δ) hslice
    · have hslice : ∀ p ∈ B_P δ,
          ∫⁻ B in matBox (a := d) (b := H) δ, ENNReal.ofReal (Real.exp (-(N * L_raw (p, B)))) ≤
          ENNReal.ofReal ((2 : ℝ) ^ H) *
            ∫⁻ B' in matBox (a := d) (b := H) (2 * δ), ENNReal.ofReal (Real.exp (-(N * L_lin (p, B')))) := fun p hp => by
        obtain ⟨hunit, hdet_lo, _, hfwd, _⟩ := hgood δ hδ hδ_le1 p hp
        have hdet_ne : (M p).det ≠ 0 := isUnit_iff_ne_zero.mp hunit
        have hmeas_phi : Measurable (fun B : RMat d H => toRMat (M p * Matrix.of B)) := by
          refine Measurable.of_eval fun i => Measurable.of_eval fun r => ?_
          simp only [toRMat_apply, Matrix.mul_apply, Matrix.of_apply]
          fun_prop
        have hg : Measurable (fun B' : RMat d H => ENNReal.ofReal (Real.exp (-(N * L_lin (p, B'))))) :=
          ENNReal.measurable_ofReal.comp (Real.measurable_exp.comp
            ((measurable_const.mul (hL_lin_meas.comp (measurable_const.prodMk measurable_id))).neg))
        have hjac : ENNReal.ofReal |((M p).det)⁻¹| ^ H ≤ ENNReal.ofReal ((2 : ℝ) ^ H) := by
          rw [ENNReal.ofReal_pow (by positivity)]
          refine pow_le_pow_left' (ENNReal.ofReal_le_ofReal ?_) H
          rw [abs_inv]
          have : |(M p).det|⁻¹ ≤ (1 / 2 : ℝ)⁻¹ := inv_anti₀ (by norm_num) hdet_lo
          linarith
        have hstep_ind :
            ∫⁻ B in matBox (a := d) (b := H) δ, ENNReal.ofReal (Real.exp (-(N * L_raw (p, B)))) ≤
            ∫⁻ B : RMat d H, (matBox (a := d) (b := H) (2 * δ)).indicator
              (fun B' => ENNReal.ofReal (Real.exp (-(N * L_lin (p, B')))))
              (toRMat (M p * Matrix.of B)) := by
          rw [← lintegral_indicator (measurableSet_matBox δ)]
          refine lintegral_mono fun B => ?_
          by_cases hmem : B ∈ matBox (a := d) (b := H) δ
          · rw [Set.indicator_of_mem hmem, Set.indicator_of_mem (hfwd B hmem), heq p B]
          · rw [Set.indicator_of_notMem hmem]; exact bot_le
        have hstep_map :
            ∫⁻ B : RMat d H, (matBox (a := d) (b := H) (2 * δ)).indicator
              (fun B' => ENNReal.ofReal (Real.exp (-(N * L_lin (p, B')))))
              (toRMat (M p * Matrix.of B)) =
            ENNReal.ofReal |((M p).det)⁻¹| ^ H *
              ∫⁻ B' in matBox (a := d) (b := H) (2 * δ), ENNReal.ofReal (Real.exp (-(N * L_lin (p, B')))) := by
          rw [← lintegral_map (hg.indicator (measurableSet_matBox (2 * δ))) hmeas_phi,
            map_mul_left_volume (M p) hdet_ne, lintegral_smul_measure, smul_eq_mul,
            lintegral_indicator (measurableSet_matBox (2 * δ))]
        calc ∫⁻ B in matBox (a := d) (b := H) δ, ENNReal.ofReal (Real.exp (-(N * L_raw (p, B))))
            ≤ ENNReal.ofReal |((M p).det)⁻¹| ^ H *
              ∫⁻ B' in matBox (a := d) (b := H) (2 * δ), ENNReal.ofReal (Real.exp (-(N * L_lin (p, B')))) :=
              hstep_ind.trans hstep_map.le
          _ ≤ ENNReal.ofReal ((2 : ℝ) ^ H) *
              ∫⁻ B' in matBox (a := d) (b := H) (2 * δ), ENNReal.ofReal (Real.exp (-(N * L_lin (p, B')))) := by
              gcongr
      calc ∫⁻ p in B_P δ, ∫⁻ B in matBox δ, ENNReal.ofReal (Real.exp (-(N * L_raw (p, B))))
          ≤ ∫⁻ p in B_P δ, ENNReal.ofReal ((2 : ℝ) ^ H) *
              ∫⁻ B' in matBox (2 * δ), ENNReal.ofReal (Real.exp (-(N * L_lin (p, B')))) :=
            MeasureTheory.setLIntegral_mono' (hB_meas δ) hslice
        _ ≤ ∫⁻ p in B_P (2 * δ), ENNReal.ofReal ((2 : ℝ) ^ H) *
              ∫⁻ B' in matBox (2 * δ), ENNReal.ofReal (Real.exp (-(N * L_lin (p, B')))) :=
            MeasureTheory.lintegral_mono_set (hB_mono (by linarith))
        _ = ENNReal.ofReal ((2 : ℝ) ^ H) *
              ∫⁻ p in B_P (2 * δ), ∫⁻ B' in matBox (2 * δ), ENNReal.ofReal (Real.exp (-(N * L_lin (p, B')))) :=
            MeasureTheory.lintegral_const_mul' _ _ ENNReal.ofReal_ne_top
  have hlo := (hlin_pair (δ / 2) (by linarith) hδ2_le).const_mul
    (a := ENNReal.ofReal ((2 : ℝ)⁻¹ ^ H))
    (ENNReal.ofReal_pos.mpr (by positivity)).ne' ENNReal.ofReal_lt_top.ne
  have hhi := (hlin_pair (2 * δ) (by linarith) h2δ_le).const_mul
    (a := ENNReal.ofReal ((2 : ℝ) ^ H))
    (ENNReal.ofReal_pos.mpr (by positivity)).ne' ENNReal.ofReal_lt_top.ne
  refine ⟨hlo.1, ?_, ?_⟩
  · exact hlo.2.1.mono (Filter.Eventually.of_forall fun N => (hbounds N).1)
  · exact hhi.2.2.mono (Filter.Eventually.of_forall fun N => (hbounds N).2)

/-- Helper: `matBox δ` shrinks into every neighborhood of `0` on `RMat a b`. -/
private theorem matBox_nhds {a b : ℕ} {U : Set (RMat a b)} (hU : U ∈ nhds (0 : RMat a b)) :
    ∃ δ₀ > 0, ∀ δ, 0 < δ → δ ≤ δ₀ → matBox (a := a) (b := b) δ ⊆ U := by
  rw [Metric.mem_nhds_iff] at hU
  obtain ⟨r, hr, hsub⟩ := hU
  refine ⟨r / 2, by linarith, fun δ hδ hδle M hM => hsub ?_⟩
  rw [Metric.mem_ball, dist_pi_lt_iff hr]
  intro i
  rw [dist_pi_lt_iff hr]
  intro j
  rw [Real.dist_eq, Pi.zero_apply, Pi.zero_apply, sub_zero]
  have := (mem_matBox_iff.mp hM) i j
  linarith

private theorem tripleBox_nhds {a₁ b₁ a₂ b₂ a₃ b₃ : ℕ}
    {U : Set (RMat a₁ b₁ × RMat a₂ b₂ × RMat a₃ b₃)} (hU : U ∈ nhds 0) :
    ∃ δ₀ > 0, ∀ δ, 0 < δ → δ ≤ δ₀ →
      (matBox (a := a₁) (b := b₁) δ ×ˢ matBox (a := a₂) (b := b₂) δ ×ˢ
        matBox (a := a₃) (b := b₃) δ) ⊆ U := by
  rw [nhds_prod_eq, nhds_prod_eq] at hU
  obtain ⟨U₁, hU₁, U₂₃, hU₂₃, hsub1⟩ := Filter.mem_prod_iff.mp hU
  obtain ⟨U₂, hU₂, U₃, hU₃, hsub2⟩ := Filter.mem_prod_iff.mp hU₂₃
  obtain ⟨δ₁, hδ₁, h1⟩ := matBox_nhds hU₁
  obtain ⟨δ₂, hδ₂, h2⟩ := matBox_nhds hU₂
  obtain ⟨δ₃, hδ₃, h3⟩ := matBox_nhds hU₃
  refine ⟨min δ₁ (min δ₂ δ₃), lt_min hδ₁ (lt_min hδ₂ hδ₃), fun δ hδ hδle x hx => ?_⟩
  rcases x with ⟨x₁, x₂, x₃⟩
  rcases hx with ⟨hx₁, hx₂, hx₃⟩
  exact hsub1 ⟨h1 δ hδ (hδle.trans (min_le_left _ _)) hx₁,
    hsub2 ⟨h2 δ hδ (hδle.trans ((min_le_right _ _).trans (min_le_left _ _))) hx₂,
           h3 δ hδ (hδle.trans ((min_le_right _ _).trans (min_le_right _ _))) hx₃⟩⟩

/-! ### 3. Self-Attention LoRA definitions & exact all-orders `GL(q)` gauge orbit -/

/-- Parameter space for a coupled pair of rank-`H` LoRA adapters on `d × d` attention projections:
- For `V+O`: `((B_O, ΔA_O, ΔA_V), B_V)`
- For `Q+K`: `((B_K, ΔA_Q, ΔA_K), B_Q)` -/
abbrev AttnPairParam (d H : ℕ) := (RMat d H × RMat H d × RMat H d) × RMat d H

/-- Uniform coordinate cube `[-δ, δ]^{4 d H}` on `AttnPairParam d H`. -/
def attnPairBox (d H : ℕ) (δ : ℝ) : Set (AttnPairParam d H) :=
  (matBox (a := d) (b := H) δ ×ˢ matBox (a := H) (b := d) δ ×ˢ matBox (a := H) (b := d) δ) ×ˢ
    matBox (a := d) (b := H) δ

/-- Base `A_{O,0} ∈ ℝ^{H×d}` with rank `q` (`I_q` in rows `0..q-1`, cols `0..q-1`). -/
def baseAO (d H q : ℕ) : Matrix (Fin H) (Fin d) ℝ :=
  Matrix.of fun r j => if r.1 < q ∧ r.1 = j.1 then 1 else 0

/-- Base `A_{V,0} ∈ ℝ^{H×d}` with rank `q` orthogonal to `A_{O,0}` (`I_q` in rows `0..q-1`, cols `q..2q-1`). -/
def baseAV (d H q : ℕ) : Matrix (Fin H) (Fin d) ℝ :=
  Matrix.of fun r j => if r.1 < q ∧ q + r.1 = j.1 then 1 else 0

/-- Base `A_{Q,0} = A_{K,0} ∈ ℝ^{H×d}` with rank `q` (`I_q` in rows `0..q-1`, cols `0..q-1`). -/
def baseAQK (d H q : ℕ) : Matrix (Fin H) (Fin d) ℝ :=
  Matrix.of fun r j => if r.1 < q ∧ r.1 = j.1 then 1 else 0

/-- **Exact all-orders joint `V+O` Self-Attention LoRA loss** around `(B_O=0, A_{O,0}, B_V=0, A_{V,0})`:
`‖(I_d + B_O (A_{O,0} + ΔA_O)) (I_d + B_V (A_{V,0} + ΔA_V)) - I_d‖_F²`. -/
def attnVOLoss (d H q : ℕ) (θ : AttnPairParam d H) : ℝ :=
  let B_O := Matrix.of θ.1.1
  let A_O := baseAO d H q + Matrix.of θ.1.2.1
  let A_V := baseAV d H q + Matrix.of θ.1.2.2
  let B_V := Matrix.of θ.2
  frobSq ((1 + B_O * A_O) * (1 + B_V * A_V) - 1)

/-- **Exact all-orders joint `Q+K` Self-Attention LoRA loss** around `(B_Q=0, A_{Q,0}, B_K=0, A_{K,0})`:
`‖(I_d + B_Q (A_{Q,0} + ΔA_Q))ᵀ (I_d + B_K (A_{K,0} + ΔA_K)) - I_d‖_F²`,
where `θ = ((B_K, ΔA_Q, ΔA_K), B_Q)`. -/
def attnQKLoss (d H q : ℕ) (θ : AttnPairParam d H) : ℝ :=
  let B_K := Matrix.of θ.1.1
  let A_Q := baseAQK d H q + Matrix.of θ.1.2.1
  let A_K := baseAQK d H q + Matrix.of θ.1.2.2
  let B_Q := Matrix.of θ.2
  frobSq ((1 + B_Q * A_Q)ᵀ * (1 + B_K * A_K) - 1)

/-- Exact all-orders factorization of the `V+O` residual. -/
theorem attnVO_factor {d H : ℕ}
    (B_O B_V : Matrix (Fin d) (Fin H) ℝ) (A_O A_V : Matrix (Fin H) (Fin d) ℝ) :
    (1 + B_O * A_O) * (1 + B_V * A_V) - 1 =
      B_O * A_O + ((1 + B_O * A_O) * B_V) * A_V := by
  simp only [Matrix.mul_add, Matrix.add_mul, Matrix.one_mul, Matrix.mul_one, Matrix.mul_assoc]
  abel

/-- Exact all-orders factorization of the `Q+K` residual. -/
theorem attnQK_factor {d H : ℕ}
    (B_Q B_K : Matrix (Fin d) (Fin H) ℝ) (A_Q A_K : Matrix (Fin H) (Fin d) ℝ) :
    (1 + B_Q * A_Q)ᵀ * (1 + B_K * A_K) - 1 =
      B_K * A_K + A_Qᵀ * ((1 + B_K * A_K)ᵀ * B_Q)ᵀ := by
  rw [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.transpose_add, Matrix.transpose_one,
    Matrix.transpose_mul]
  simp only [Matrix.mul_add, Matrix.add_mul, Matrix.one_mul, Matrix.mul_one, Matrix.mul_assoc]
  abel

/-- **Exact All-Orders `GL(q)` Gauge Symmetry Orbit on the `Q+K` Zero-Loss Fiber (`[LC]`):**
For any `q ≤ H`, `q ≤ d`, and any `C ∈ ℝ^{q×q}` with `IsUnit (1 + C).det`, setting
`B_K(C) = [[C, 0], [0, 0]]` and `B_Q(C) = [[-((1 + Cᵀ)⁻¹ Cᵀ), 0], [0, 0]]` satisfies
`(I_d + B_Q(C) A_{Q,0})ᵀ (I_d + B_K(C) A_{K,0}) = I_d` **exactly to all orders**,
so `attnQKLoss d H q ((B_K(C), 0, 0), B_Q(C)) = 0`. -/
theorem attention_QK_gauge_orbit {d H q : ℕ} (hqH : q ≤ H) (hqd : q ≤ d)
    (C : Matrix (Fin q) (Fin q) ℝ) (hC : IsUnit (1 + C).det) :
    let B_K : RMat d H := fun i r =>
      if hi : i.1 < q then if hr : r.1 < q then C ⟨i.1, hi⟩ ⟨r.1, hr⟩ else 0 else 0
    let D : Matrix (Fin q) (Fin q) ℝ := -((1 + Cᵀ)⁻¹ * Cᵀ)
    let B_Q : RMat d H := fun i r =>
      if hi : i.1 < q then if hr : r.1 < q then D ⟨i.1, hi⟩ ⟨r.1, hr⟩ else 0 else 0
    (1 + Matrix.of B_Q * baseAQK d H q)ᵀ * (1 + Matrix.of B_K * baseAQK d H q) = 1 ∧
      attnQKLoss d H q ((B_K, 0, 0), B_Q) = 0 := by
  intro B_K D B_Q
  have hCT : IsUnit (1 + Cᵀ).det := by
    have : (1 + Cᵀ).det = (1 + C).det := by
      rw [← Matrix.det_transpose, Matrix.transpose_add, Matrix.transpose_one, Matrix.transpose_transpose]
    rwa [this]
  have hDT : (1 + Dᵀ) * (1 + C) = 1 := by
    have hD1 : (1 + Cᵀ) * (1 + D) = 1 := by
      dsimp only [D]
      rw [Matrix.mul_add, Matrix.mul_one, Matrix.mul_neg, ← Matrix.mul_assoc,
        Matrix.mul_nonsing_inv (1 + Cᵀ) hCT, Matrix.one_mul]
      abel
    have := congrArg Matrix.transpose hD1
    simpa [Matrix.transpose_mul, Matrix.transpose_add] using this
  have hprod_BK : ∀ i j : Fin d, (Matrix.of B_K * baseAQK d H q) i j =
      if hi : i.1 < q then if hj : j.1 < q then C ⟨i.1, hi⟩ ⟨j.1, hj⟩ else 0 else 0 := by
    intro i j
    simp only [Matrix.mul_apply, Matrix.of_apply, baseAQK, B_K]
    by_cases hi : i.1 < q
    · by_cases hj : j.1 < q
      · simp only [hi, hj, ↓reduceDIte]
        have hjH : j.1 < H := lt_of_lt_of_le hj hqH
        rw [Finset.sum_eq_single ⟨j.1, hjH⟩]
        · simp [hj]
        · intro r _ hr
          have hne : r.1 ≠ j.1 := fun h => hr (Fin.ext h)
          simp [hne]
        · intro h; exact absurd (Finset.mem_univ _) h
      · simp only [hi, hj, ↓reduceDIte]
        apply Finset.sum_eq_zero
        intro r _
        by_cases hr : r.1 < q
        · have hne : r.1 ≠ j.1 := by omega
          simp [hr, hne]
        · simp [hr]
    · simp [hi]
  have hprod_BQ : ∀ i j : Fin d, (Matrix.of B_Q * baseAQK d H q) i j =
      if hi : i.1 < q then if hj : j.1 < q then D ⟨i.1, hi⟩ ⟨j.1, hj⟩ else 0 else 0 := by
    intro i j
    simp only [Matrix.mul_apply, Matrix.of_apply, baseAQK, B_Q]
    by_cases hi : i.1 < q
    · by_cases hj : j.1 < q
      · simp only [hi, hj, ↓reduceDIte]
        have hjH : j.1 < H := lt_of_lt_of_le hj hqH
        rw [Finset.sum_eq_single ⟨j.1, hjH⟩]
        · simp [hj]
        · intro r _ hr
          have hne : r.1 ≠ j.1 := fun h => hr (Fin.ext h)
          simp [hne]
        · intro h; exact absurd (Finset.mem_univ _) h
      · simp only [hi, hj, ↓reduceDIte]
        apply Finset.sum_eq_zero
        intro r _
        by_cases hr : r.1 < q
        · have hne : r.1 ≠ j.1 := by omega
          simp [hr, hne]
        · simp [hr]
    · simp [hi]
  have h1 : (1 + Matrix.of B_Q * baseAQK d H q)ᵀ * (1 + Matrix.of B_K * baseAQK d H q) = 1 := by
    ext i j
    rw [Matrix.mul_apply]
    have hL : ∀ k : Fin d, ((1 : Matrix (Fin d) (Fin d) ℝ) + Matrix.of B_Q * baseAQK d H q)ᵀ i k =
        (if k = i then 1 else 0) +
        (if hk : k.1 < q then if hi : i.1 < q then D ⟨k.1, hk⟩ ⟨i.1, hi⟩ else 0 else 0) := by
      intro k
      simp only [Matrix.transpose_apply, Matrix.add_apply, Matrix.one_apply, hprod_BQ]
    have hR : ∀ k : Fin d, ((1 : Matrix (Fin d) (Fin d) ℝ) + Matrix.of B_K * baseAQK d H q) k j =
        (if k = j then 1 else 0) +
        (if hk : k.1 < q then if hj : j.1 < q then C ⟨k.1, hk⟩ ⟨j.1, hj⟩ else 0 else 0) := by
      intro k
      simp only [Matrix.add_apply, Matrix.one_apply, hprod_BK]
    simp_rw [hL, hR]
    by_cases hi : i.1 < q
    · by_cases hj : j.1 < q
      · have hentry := congrFun (congrFun hDT ⟨i.1, hi⟩) ⟨j.1, hj⟩
        simp only [Matrix.mul_apply, Matrix.add_apply, Matrix.one_apply, Matrix.transpose_apply] at hentry
        have hsum_eq :
            ∑ k : Fin d,
              ((if k = i then (1 : ℝ) else 0) + if hk : k.1 < q then D ⟨k.1, hk⟩ ⟨i.1, hi⟩ else 0) *
              ((if k = j then 1 else 0) + if hk : k.1 < q then C ⟨k.1, hk⟩ ⟨j.1, hj⟩ else 0) =
            ∑ k : Fin q,
              ((if (⟨i.1, hi⟩ : Fin q) = k then (1 : ℝ) else 0) + D k ⟨i.1, hi⟩) *
              ((if k = ⟨j.1, hj⟩ then 1 else 0) + C k ⟨j.1, hj⟩) := by
          rw [← Finset.sum_filter_add_sum_filter_not (s := Finset.univ) (p := fun k : Fin d => k.1 < q)]
          have htail : ∑ k ∈ Finset.univ.filter (fun k : Fin d => ¬ k.1 < q),
              ((if k = i then (1 : ℝ) else 0) + if hk : k.1 < q then D ⟨k.1, hk⟩ ⟨i.1, hi⟩ else 0) *
              ((if k = j then 1 else 0) + if hk : k.1 < q then C ⟨k.1, hk⟩ ⟨j.1, hj⟩ else 0) = 0 := by
            apply Finset.sum_eq_zero
            intro k hk
            have hnk : ¬ k.1 < q := (Finset.mem_filter.mp hk).2
            have hki : k ≠ i := fun h => hnk (h ▸ hi)
            simp [hnk, hki]
          rw [htail, add_zero]
          refine Finset.sum_bij
            (fun (k : Fin d) (hk : k ∈ Finset.univ.filter (fun k : Fin d => k.1 < q)) =>
              (⟨k.1, (Finset.mem_filter.mp hk).2⟩ : Fin q))
            (fun _ _ => Finset.mem_univ _)
            (fun k₁ _ k₂ _ h => Fin.ext (congrArg (Fin.val : Fin q → ℕ) h))
            (fun r _ => ⟨⟨r.1, lt_of_lt_of_le r.2 hqd⟩, by simp [r.2], by ext; rfl⟩)
            (fun k hk => by
              have hk' : k.1 < q := (Finset.mem_filter.mp hk).2
              simp only [hk', ↓reduceDIte]
              have h1' : (k = i) ↔ (⟨i.1, hi⟩ : Fin q) = ⟨k.1, hk'⟩ := by
                simp [Fin.ext_iff, eq_comm]
              have h2' : (k = j) ↔ (⟨k.1, hk'⟩ : Fin q) = ⟨j.1, hj⟩ := by
                simp [Fin.ext_iff]
              simp only [h1', h2'])
        have hgoal : ∑ k : Fin d,
            ((if k = i then (1 : ℝ) else 0) + if hk : k.1 < q then if hi' : i.1 < q then D ⟨k.1, hk⟩ ⟨i.1, hi'⟩ else 0 else 0) *
            ((if k = j then 1 else 0) + if hk : k.1 < q then if hj' : j.1 < q then C ⟨k.1, hk⟩ ⟨j.1, hj'⟩ else 0 else 0) =
            if i = j then 1 else 0 := by
          simp only [hi, hj, ↓reduceDIte]
          rw [hsum_eq, hentry]
          simp [Fin.ext_iff]
        simpa [Matrix.one_apply] using hgoal
      · have hij : i ≠ j := fun h => hj (h ▸ hi)
        simp only [Matrix.one_apply, hij, ↓reduceIte]
        apply Finset.sum_eq_zero
        intro k _
        by_cases hkj : k = j
        · subst hkj
          have hki : k ≠ i := fun h => hj (h ▸ hi)
          simp [hj, hki]
        · simp [hj, hkj]
    · rw [Finset.sum_eq_single i]
      · simp [hi, Matrix.one_apply]
      · intro k _ hki
        simp [hi, hki]
      · intro h; exact absurd (Finset.mem_univ _) h
  refine ⟨h1, ?_⟩
  unfold attnQKLoss
  have h0 : Matrix.of (0 : RMat H d) = 0 := rfl
  simp only [h0, add_zero, h1, sub_self]
  unfold frobSq
  simp

/-! ### 4. Joint `V+O` LoRA Exact Learning Pair, Plateau, and 4× Supercritical Parabola -/

/-- Exact linearized `V+O` stacked matrix-product loss on `AttnPairParam d H`. -/
private def attnVOLinLoss (d H q : ℕ) (θ : AttnPairParam d H) : ℝ :=
  frobSq (Matrix.of θ.1.1 * (baseAO d H q + Matrix.of θ.1.2.1) +
          Matrix.of θ.2 * (baseAV d H q + Matrix.of θ.1.2.2))

/-- Clean 4-tuple shuffle `((a, b, c), d) ≃ᵐ (a, d) × (b, c)` preserving product Lebesgue measure. -/
private def shuffle4Equiv (α β γ δ : Type*) [MeasurableSpace α] [MeasurableSpace β]
    [MeasurableSpace γ] [MeasurableSpace δ] :
    ((α × β × γ) × δ) ≃ᵐ ((α × δ) × (β × γ)) :=
  (MeasurableEquiv.prodAssoc (α := α) (β := β × γ) (γ := δ)).trans
    (((MeasurableEquiv.refl α).prodCongr (MeasurableEquiv.prodComm (α := β × γ) (β := δ))).trans
      (MeasurableEquiv.prodAssoc (α := α) (β := δ) (γ := β × γ)).symm)

private theorem measurePreserving_shuffle4 {α β γ δ : Type*}
    [MeasureSpace α] [MeasureSpace β] [MeasureSpace γ] [MeasureSpace δ]
    [SigmaFinite (volume : Measure α)] [SigmaFinite (volume : Measure β)]
    [SigmaFinite (volume : Measure γ)] [SigmaFinite (volume : Measure δ)] :
    MeasurePreserving (shuffle4Equiv α β γ δ) volume volume := by
  unfold shuffle4Equiv
  refine (MeasureTheory.volume_preserving_prodAssoc.symm).comp ?_
  refine ((MeasurePreserving.id volume).prod Measure.measurePreserving_swap).comp ?_
  exact MeasureTheory.volume_preserving_prodAssoc

/-- Permutation of `Fin (H + H)` placing the `q` active rows of block 0 at `0..q-1`,
the `q` active rows of block 1 at `q..2q-1`, the `H-q` inactive rows of block 0 at `2q..H+q-1`,
and the `H-q` inactive rows of block 1 at `H+q..2H-1`. -/
private def interleavePerm {H q : ℕ} (hqH : q ≤ H) : Fin (H + H) ≃ Fin (H + H) where
  toFun i :=
    if h1 : i.1 < q then ⟨i.1, by omega⟩
    else if h2 : i.1 < H then ⟨q + i.1, by omega⟩
    else if h3 : i.1 < H + q then ⟨i.1 - H + q, by omega⟩
    else ⟨i.1, by omega⟩
  invFun j :=
    if h1 : j.1 < q then ⟨j.1, by omega⟩
    else if h2 : j.1 < 2 * q then ⟨H + (j.1 - q), by omega⟩
    else if h3 : j.1 < H + q then ⟨j.1 - q, by omega⟩
    else ⟨j.1, by omega⟩
  left_inv i := by
    ext; dsimp only
    by_cases h1 : i.1 < q
    · simp [h1]
    · by_cases h2 : i.1 < H
      · have h2a : ¬ (q + i.1 < q) := by omega
        have h2b : ¬ (q + i.1 < 2 * q) := by omega
        have h2c : q + i.1 < H + q := by omega
        simp [h1, h2, h2a, h2b, h2c]
      · by_cases h3 : i.1 < H + q
        · have h3a : ¬ (i.1 - H + q < q) := by omega
          have h3b : i.1 - H + q < 2 * q := by omega
          simp [h1, h2, h3, h3a, h3b]; omega
        · have h4b : ¬ (i.1 < 2 * q) := by omega
          simp [h1, h2, h3, h4b]
  right_inv j := by
    ext; dsimp only
    by_cases h1 : j.1 < q
    · simp [h1]
    · by_cases h2 : j.1 < 2 * q
      · have h2a : ¬ (H + (j.1 - q) < q) := by omega
        have h2b : ¬ (H + (j.1 - q) < H) := by omega
        have h2c : H + (j.1 - q) < H + q := by omega
        simp [h1, h2, h2a, h2b, h2c]; omega
      · by_cases h3 : j.1 < H + q
        · have h3a : ¬ (j.1 - q < q) := by omega
          have h3b : j.1 - q < H := by omega
          simp [h1, h2, h3, h3a, h3b]; omega
        · have h4b : ¬ (j.1 < H) := by omega
          simp [h1, h2, h3, h4b]

private theorem interleavePerm_castAdd_lt {H q : ℕ} (hqH : q ≤ H) (r : Fin H) (hr : r.1 < q) :
    (interleavePerm hqH (Fin.castAdd H r)).1 = r.1 := by
  simp [interleavePerm, hr]

private theorem interleavePerm_castAdd_ge {H q : ℕ} (hqH : q ≤ H) (r : Fin H) (hr : ¬ r.1 < q) :
    (interleavePerm hqH (Fin.castAdd H r)).1 = q + r.1 := by
  simp [interleavePerm, hr, r.2]

private theorem interleavePerm_natAdd_lt {H q : ℕ} (hqH : q ≤ H) (r : Fin H) (hr : r.1 < q) :
    (interleavePerm hqH (Fin.natAdd H r)).1 = q + r.1 := by
  have hval : (Fin.natAdd H r).1 = H + r.1 := Fin.val_natAdd H r
  dsimp only [interleavePerm, Equiv.coe_fn_mk]
  split_ifs <;> dsimp only <;> omega

private theorem interleavePerm_natAdd_ge {H q : ℕ} (hqH : q ≤ H) (r : Fin H) (hr : ¬ r.1 < q) :
    (interleavePerm hqH (Fin.natAdd H r)).1 = H + r.1 := by
  have hval : (Fin.natAdd H r).1 = H + r.1 := Fin.val_natAdd H r
  dsimp only [interleavePerm, Equiv.coe_fn_mk]
  split_ifs <;> dsimp only <;> omega

/-- Measure-preserving bijection from `(RMat m (q + k') × RMat (q + k') (q + n'))`
to `DeltaSide q k' n' × CoreBlock m q k' n'`, mapping `matBox ρ ×ˢ matBox ρ` onto `fiberBox ρ`. -/
private def matPairToFiber (m q k' n' : ℕ) :
    (RMat m (q + k') × RMat (q + k') (q + n')) ≃ᵐ
      (DeltaSide q k' n' × CoreBlock m q k' n') :=
  let e1 := (splitColsEquiv m q k').prodCongr
    ((splitRowsEquiv q k' (q + n')).trans
      ((splitColsEquiv q q n').prodCongr (splitColsEquiv k' q n')))
  let eA : ((RMat q q × RMat q n') × (RMat k' q × RMat k' n')) ≃ᵐ
      (DeltaSide q k' n' × RMat k' n') :=
    MeasurableEquiv.prodAssoc.trans
      (((MeasurableEquiv.refl _).prodCongr MeasurableEquiv.prodAssoc.symm).trans
        MeasurableEquiv.prodAssoc.symm)
  let e2 : ((RMat m q × RMat m k') × ((RMat q q × RMat q n') × (RMat k' q × RMat k' n'))) ≃ᵐ
      (DeltaSide q k' n' × CoreBlock m q k' n') :=
    MeasurableEquiv.prodComm.trans
      ((eA.prodCongr (MeasurableEquiv.refl _)).trans
        (MeasurableEquiv.prodAssoc.trans
          ((MeasurableEquiv.refl _).prodCongr
            (MeasurableEquiv.prodComm.trans MeasurableEquiv.prodAssoc))))
  e1.trans e2

private theorem measurePreserving_matPairToFiber (m q k' n' : ℕ) :
    MeasurePreserving (matPairToFiber m q k' n') volume volume := by
  unfold matPairToFiber
  have he1 := (measurePreserving_splitColsEquiv m q k').prod
    (((measurePreserving_splitColsEquiv q q n').prod
      (measurePreserving_splitColsEquiv k' q n')).comp
      (measurePreserving_splitRowsEquiv q k' (q + n')))
  have heA : MeasurePreserving
      ((MeasurableEquiv.prodAssoc (α := RMat q q) (β := RMat q n') (γ := RMat k' q × RMat k' n')).trans
        (((MeasurableEquiv.refl (RMat q q)).prodCongr
          (MeasurableEquiv.prodAssoc (α := RMat q n') (β := RMat k' q) (γ := RMat k' n')).symm).trans
          (MeasurableEquiv.prodAssoc (α := RMat q q) (β := RMat q n' × RMat k' q) (γ := RMat k' n')).symm))
      volume volume :=
    MeasureTheory.volume_preserving_prodAssoc.symm.comp
      (((MeasurePreserving.id volume).prod MeasureTheory.volume_preserving_prodAssoc.symm).comp
        MeasureTheory.volume_preserving_prodAssoc)
  have he2_inner : MeasurePreserving
      ((MeasurableEquiv.refl (DeltaSide q k' n')).prodCongr
        (MeasurableEquiv.prodComm.trans MeasurableEquiv.prodAssoc))
      (volume : Measure (DeltaSide q k' n' × (RMat k' n' × (RMat m q × RMat m k'))))
      (volume : Measure (DeltaSide q k' n' × CoreBlock m q k' n')) :=
    (MeasurePreserving.id (volume : Measure (DeltaSide q k' n'))).prod
      (MeasureTheory.volume_preserving_prodAssoc.comp Measure.measurePreserving_swap)
  have he2_mid : MeasurePreserving
      (MeasurableEquiv.prodAssoc (α := DeltaSide q k' n') (β := RMat k' n')
        (γ := RMat m q × RMat m k')) volume volume :=
    MeasureTheory.volume_preserving_prodAssoc
  exact (he2_inner.comp he2_mid).comp
    ((heA.prod (MeasurePreserving.id (volume : Measure (RMat m q × RMat m k')))).comp
      (Measure.measurePreserving_swap.comp he1))

private theorem matPairToFiber_mem_box {m q k' n' : ℕ} {ρ : ℝ}
    (p : RMat m (q + k') × RMat (q + k') (q + n')) :
    matPairToFiber m q k' n' p ∈ fiberBox (m := m) (q := q) (k' := k') (n' := n') ρ ↔
      p ∈ matBox ρ ×ˢ matBox ρ := by
  rcases p with ⟨B, A⟩
  change (((splitColsEquiv q q n' (splitRowsEquiv q k' (q + n') A).1).1,
      ((splitColsEquiv q q n' (splitRowsEquiv q k' (q + n') A).1).2,
       (splitColsEquiv k' q n' (splitRowsEquiv q k' (q + n') A).2).1)),
     ((splitColsEquiv m q k' B).1,
      ((splitColsEquiv m q k' B).2,
       (splitColsEquiv k' q n' (splitRowsEquiv q k' (q + n') A).2).2))) ∈
      fiberBox (m := m) (q := q) (k' := k') (n' := n') ρ ↔ (B, A) ∈ matBox ρ ×ˢ matBox ρ
  simp only [fiberBox, deltaBox, coreBox, Set.mem_prod]
  constructor
  · rintro ⟨⟨h11, h12, h21⟩, ⟨hb1, hb2, h22⟩⟩
    refine ⟨(splitColsEquiv_mem_matBox (M := B)).mp ⟨hb1, hb2⟩, ?_⟩
    have hA1 : (splitRowsEquiv q k' (q + n') A).1 ∈ matBox ρ :=
      (splitColsEquiv_mem_matBox (M := (splitRowsEquiv q k' (q + n') A).1)).mp ⟨h11, h12⟩
    have hA2 : (splitRowsEquiv q k' (q + n') A).2 ∈ matBox ρ :=
      (splitColsEquiv_mem_matBox (M := (splitRowsEquiv q k' (q + n') A).2)).mp ⟨h21, h22⟩
    exact (splitRowsEquiv_mem_matBox (M := A)).mp ⟨hA1, hA2⟩
  · rintro ⟨hB, hA⟩
    obtain ⟨hb1, hb2⟩ := (splitColsEquiv_mem_matBox (M := B)).mpr hB
    obtain ⟨hA1, hA2⟩ := (splitRowsEquiv_mem_matBox (M := A)).mpr hA
    obtain ⟨h11, h12⟩ := (splitColsEquiv_mem_matBox (M := (splitRowsEquiv q k' (q + n') A).1)).mpr hA1
    obtain ⟨h21, h22⟩ := (splitColsEquiv_mem_matBox (M := (splitRowsEquiv q k' (q + n') A).2)).mpr hA2
    exact ⟨⟨h11, h12, h21⟩, ⟨hb1, hb2, h22⟩⟩

/-- Canonical top-left identity base matrix `I_q ⊕ 0` of shape `(q + k') × (q + n')`. -/
private def canonBaseA (q k' n' : ℕ) : Matrix (Fin (q + k')) (Fin (q + n')) ℝ :=
  Matrix.of fun r j => if r.1 < q ∧ r.1 = j.1 then 1 else 0

private theorem matPairToFiber_apply {m q k' n' : ℕ}
    (B : RMat m (q + k')) (A : RMat (q + k') (q + n')) :
    matPairToFiber m q k' n' (B, A) =
      (((splitColsEquiv q q n' (splitRowsEquiv q k' (q + n') A).1).1,
        ((splitColsEquiv q q n' (splitRowsEquiv q k' (q + n') A).1).2,
         (splitColsEquiv k' q n' (splitRowsEquiv q k' (q + n') A).2).1)),
       ((splitColsEquiv m q k' B).1,
        ((splitColsEquiv m q k' B).2,
         (splitColsEquiv k' q n' (splitRowsEquiv q k' (q + n') A).2).2))) := rfl

/-- Squared Frobenius norm of `B * (canonBaseA + A)` equals `loraFiberLoss (matPairToFiber (B, A))`. -/
private theorem matPair_loss_eq_loraFiberLoss {m q k' n' : ℕ}
    (B : RMat m (q + k')) (A : RMat (q + k') (q + n')) :
    frobSq (Matrix.of B * (canonBaseA q k' n' + Matrix.of A)) =
      loraFiberLoss (matPairToFiber m q k' n' (B, A)) := by
  rw [matPairToFiber_apply]
  unfold loraFiberLoss frobSq
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Fin.sum_univ_add]
  congr 1
  · refine Finset.sum_congr rfl fun j _ => ?_
    congr 1
    simp only [Matrix.mul_apply, Matrix.add_apply, Matrix.of_apply, Matrix.one_apply,
      canonBaseA, splitColsEquiv_apply_fst, splitColsEquiv_apply_snd,
      splitRowsEquiv_apply_fst, splitRowsEquiv_apply_snd]
    rw [Fin.sum_univ_add]
    congr 1
    · refine Finset.sum_congr rfl fun r _ => ?_
      simp only [Fin.val_castAdd, r.2, true_and, Fin.val_inj]
    · refine Finset.sum_congr rfl fun r _ => ?_
      have hnr : ¬ (Fin.natAdd q r).1 < q := by simp
      simp only [hnr, false_and, ↓reduceIte, zero_add]
  · refine Finset.sum_congr rfl fun j _ => ?_
    congr 1
    simp only [Matrix.mul_apply, Matrix.add_apply, Matrix.of_apply,
      canonBaseA, splitColsEquiv_apply_fst, splitColsEquiv_apply_snd,
      splitRowsEquiv_apply_fst, splitRowsEquiv_apply_snd]
    rw [Fin.sum_univ_add]
    congr 1
    · refine Finset.sum_congr rfl fun r _ => ?_
      have hne : ¬ ((Fin.castAdd k' r).1 = (Fin.natAdd q j).1) := by simp; omega
      simp only [hne, and_false, ↓reduceIte, zero_add]
    · refine Finset.sum_congr rfl fun r _ => ?_
      have hnr : ¬ (Fin.natAdd q r).1 < q := by simp
      simp only [hnr, false_and, ↓reduceIte, zero_add]

/-- Linearized `V+O` loss has the exact learning pair of a single rank-`2H`, right-rank-`2q` LoRA fiber. -/
private theorem attnVOLin_hasLaplacePair {d H q : ℕ} (hqH : q ≤ H) (h2qd : 2 * q ≤ d) :
    ∃ ρ₀ : ℝ, 0 < ρ₀ ∧ ∀ ρ : ℝ, 0 < ρ → ρ ≤ ρ₀ →
      SLT.HasLaplacePair
        (SLT.Z (volume.restrict (attnPairBox d H ρ)) (attnVOLinLoss d H q))
        ((d : ℝ) * q + awLambda d (d - 2 * q) (2 * H - 2 * q))
        (awMult d (d - 2 * q) (2 * H - 2 * q)) := by
  have h2H : H + H = 2 * q + (2 * H - 2 * q) := by omega
  have hd : d = 2 * q + (d - 2 * q) := by omega
  -- Equivalence `eStack : AttnPairParam d H ≃ᵐ RMat d (2*q + (2*H - 2*q)) × RMat (2*q + (2*H - 2*q)) (2*q + (d - 2*q))`
  let σ : Fin (H + H) ≃ Fin (H + H) := interleavePerm hqH
  let eB : (RMat d H × RMat d H) ≃ᵐ RMat d (2 * q + (2 * H - 2 * q)) :=
    (splitColsEquiv d H H).symm.trans ((permColsEquiv σ).trans (castMatEquiv rfl h2H))
  let eA : (RMat H d × RMat H d) ≃ᵐ RMat (2 * q + (2 * H - 2 * q)) (2 * q + (d - 2 * q)) :=
    (splitRowsEquiv H H d).symm.trans ((permRowsEquiv σ).trans (castMatEquiv h2H hd))
  let eTot : AttnPairParam d H ≃ᵐ
      DeltaSide (2 * q) (2 * H - 2 * q) (d - 2 * q) × CoreBlock d (2 * q) (2 * H - 2 * q) (d - 2 * q) :=
    (shuffle4Equiv (RMat d H) (RMat H d) (RMat H d) (RMat d H)).trans
      ((eB.prodCongr eA).trans (matPairToFiber d (2 * q) (2 * H - 2 * q) (d - 2 * q)))
  have heB_mp : MeasurePreserving eB volume volume :=
    measurePreserving_trans (measurePreserving_splitColsEquiv d H H).symm
      (measurePreserving_trans (measurePreserving_permColsEquiv σ) (measurePreserving_castMatEquiv rfl h2H))
  have heA_mp : MeasurePreserving eA volume volume :=
    measurePreserving_trans (measurePreserving_splitRowsEquiv H H d).symm
      (measurePreserving_trans (measurePreserving_permRowsEquiv σ) (measurePreserving_castMatEquiv h2H hd))
  have heTot_mp : MeasurePreserving eTot volume volume :=
    measurePreserving_trans measurePreserving_shuffle4
      (measurePreserving_trans (measurePreserving_prodCongr heB_mp heA_mp)
        (measurePreserving_matPairToFiber d (2 * q) (2 * H - 2 * q) (d - 2 * q)))
  have heTot_val : ∀ BO AO AV BV,
      eTot ((BO, AO, AV), BV) =
        matPairToFiber d (2 * q) (2 * H - 2 * q) (d - 2 * q) (eB (BO, BV), eA (AO, AV)) := by
    intro BO AO AV BV
    dsimp only [eTot]
    rw [MeasurableEquiv.trans_apply, MeasurableEquiv.trans_apply]
    have hsh : shuffle4Equiv (RMat d H) (RMat H d) (RMat H d) (RMat d H) ((BO, AO, AV), BV) =
        ((BO, BV), (AO, AV)) := rfl
    rw [hsh, prodCongr_apply_mk]
  have heTot_box : ∀ {ρ : ℝ} (θ : AttnPairParam d H),
      eTot θ ∈ fiberBox (m := d) (q := 2 * q) (k' := 2 * H - 2 * q) (n' := d - 2 * q) ρ ↔
        θ ∈ attnPairBox d H ρ := fun {ρ} θ => by
    rcases θ with ⟨⟨BO, AO, AV⟩, BV⟩
    rw [heTot_val, matPairToFiber_mem_box]
    simp only [eB, eA, MeasurableEquiv.trans_apply, Set.mem_prod, castMatEquiv_mem_matBox,
      permColsEquiv_mem_matBox, permRowsEquiv_mem_matBox]
    have hB : (splitColsEquiv d H H).symm (BO, BV) ∈ matBox ρ ↔ BO ∈ matBox ρ ∧ BV ∈ matBox ρ := by
      rw [← splitColsEquiv_mem_matBox, MeasurableEquiv.apply_symm_apply]; rfl
    have hA : (splitRowsEquiv H H d).symm (AO, AV) ∈ matBox ρ ↔ AO ∈ matBox ρ ∧ AV ∈ matBox ρ := by
      rw [← splitRowsEquiv_mem_matBox, MeasurableEquiv.apply_symm_apply]; rfl
    rw [hB, hA]
    simp only [attnPairBox, Set.mem_prod]
    tauto
  -- Loss invariance: `attnVOLinLoss d H q θ = loraFiberLoss (eTot θ)`.
  have heTot_loss : ∀ θ : AttnPairParam d H, attnVOLinLoss d H q θ = loraFiberLoss (eTot θ) := by
    intro ⟨⟨BO, AO, AV⟩, BV⟩
    have h1 : loraFiberLoss (eTot ((BO, AO, AV), BV)) =
        frobSq (Matrix.of (eB (BO, BV)) * (canonBaseA (2 * q) (2 * H - 2 * q) (d - 2 * q) +
          Matrix.of (eA (AO, AV)))) := by
      rw [heTot_val]
      exact (matPair_loss_eq_loraFiberLoss (eB (BO, BV)) (eA (AO, AV))).symm
    rw [h1]
    unfold attnVOLinLoss frobSq
    refine Finset.sum_congr rfl fun i _ => ?_
    refine (Finset.sum_equiv (finCongr hd) (fun _ => by simp) fun j _ => ?_)
    congr 1
    simp only [Matrix.add_apply, Matrix.mul_apply, Matrix.of_apply]
    have hcast_sum :
        ∑ r : Fin (2 * q + (2 * H - 2 * q)),
          eB (BO, BV) i r * (canonBaseA (2 * q) (2 * H - 2 * q) (d - 2 * q) r (finCongr hd j) +
            eA (AO, AV) r (finCongr hd j)) =
        ∑ r : Fin (H + H),
          (splitColsEquiv d H H).symm (BO, BV) i r *
            ((if (σ r).1 < 2 * q ∧ (σ r).1 = j.1 then (1 : ℝ) else 0) +
              (splitRowsEquiv H H d).symm (AO, AV) r j) := by
      symm
      refine Finset.sum_equiv (σ.trans (finCongr h2H)) (fun _ => by simp) fun r _ => ?_
      have hBval : eB (BO, BV) i ((σ.trans (finCongr h2H)) r) =
          (splitColsEquiv d H H).symm (BO, BV) i r := by
        dsimp only [eB, MeasurableEquiv.trans_apply, Equiv.trans_apply]
        have h1 : castMatEquiv rfl h2H (permColsEquiv σ ((splitColsEquiv d H H).symm (BO, BV)))
            (finCongr rfl i) (finCongr h2H (σ r)) =
            permColsEquiv σ ((splitColsEquiv d H H).symm (BO, BV)) i (σ r) :=
          castMatEquiv_apply rfl h2H _ i (σ r)
        simpa only [finCongr_refl, Equiv.refl_apply, permColsEquiv, permRowsEquiv, transposeEquiv,
          MeasurableEquiv.trans_apply, MeasurableEquiv.coe_mk, Equiv.coe_fn_mk] using
          h1.trans (congrFun (Equiv.piCongrLeft_apply_apply (fun _ : Fin (H + H) => (Fin d → ℝ))
            σ (fun c r' => (splitColsEquiv d H H).symm (BO, BV) r' c) r) i)
      have hAval : eA (AO, AV) ((σ.trans (finCongr h2H)) r) (finCongr hd j) =
          (splitRowsEquiv H H d).symm (AO, AV) r j := by
        dsimp only [eA, MeasurableEquiv.trans_apply, Equiv.trans_apply]
        have h1 : castMatEquiv h2H hd (permRowsEquiv σ ((splitRowsEquiv H H d).symm (AO, AV)))
            (finCongr h2H (σ r)) (finCongr hd j) =
            permRowsEquiv σ ((splitRowsEquiv H H d).symm (AO, AV)) (σ r) j :=
          castMatEquiv_apply h2H hd _ (σ r) j
        exact h1.trans (congrFun (Equiv.piCongrLeft_apply_apply (fun _ : Fin (H + H) => (Fin d → ℝ))
          σ ((splitRowsEquiv H H d).symm (AO, AV)) r) j)
      have hbase : canonBaseA (2 * q) (2 * H - 2 * q) (d - 2 * q) ((σ.trans (finCongr h2H)) r) (finCongr hd j) =
          if (σ r).1 < 2 * q ∧ (σ r).1 = j.1 then (1 : ℝ) else 0 := by
        simp [canonBaseA]
      rw [hBval, hAval, hbase]
    rw [hcast_sum, Fin.sum_univ_add]
    congr 1
    · refine Finset.sum_congr rfl fun r _ => ?_
      have hB0 : (splitColsEquiv d H H).symm (BO, BV) i (Fin.castAdd H r) = BO i r :=
        splitColsEquiv_symm_apply_castAdd BO BV i r
      have hA0 : (splitRowsEquiv H H d).symm (AO, AV) (Fin.castAdd H r) j = AO r j :=
        splitRowsEquiv_symm_apply_castAdd AO AV r j
      have hbase0 : (if (σ (Fin.castAdd H r)).1 < 2 * q ∧ (σ (Fin.castAdd H r)).1 = j.1 then (1 : ℝ) else 0) =
          baseAO d H q r j := by
        unfold baseAO
        simp only [Matrix.of_apply]
        by_cases h1 : r.1 < q
        · rw [interleavePerm_castAdd_lt hqH r h1]
          have hlt : r.1 < 2 * q := by omega
          simp only [hlt, true_and, h1]
        · rw [interleavePerm_castAdd_ge hqH r h1]
          have hnot : ¬ (q + r.1 < 2 * q) := by omega
          simp only [hnot, false_and, ↓reduceIte, h1]
      rw [hB0, hA0, hbase0]
    · refine Finset.sum_congr rfl fun r _ => ?_
      have hB1 : (splitColsEquiv d H H).symm (BO, BV) i (Fin.natAdd H r) = BV i r :=
        splitColsEquiv_symm_apply_natAdd BO BV i r
      have hA1 : (splitRowsEquiv H H d).symm (AO, AV) (Fin.natAdd H r) j = AV r j :=
        splitRowsEquiv_symm_apply_natAdd AO AV r j
      have hbase1 : (if (σ (Fin.natAdd H r)).1 < 2 * q ∧ (σ (Fin.natAdd H r)).1 = j.1 then (1 : ℝ) else 0) =
          baseAV d H q r j := by
        unfold baseAV
        simp only [Matrix.of_apply]
        by_cases h3 : r.1 < q
        · rw [interleavePerm_natAdd_lt hqH r h3]
          have hlt : q + r.1 < 2 * q := by omega
          simp only [hlt, true_and, h3]
        · rw [interleavePerm_natAdd_ge hqH r h3]
          have hnot : ¬ (H + r.1 < 2 * q) := by omega
          simp only [hnot, false_and, ↓reduceIte, h3]
      rw [hB1, hA1, hbase1]
  -- Apply `lora_fiber_exact_pair` directly!
  obtain ⟨ρ₀, hρ₀, hlora⟩ := lora_fiber_exact_pair (m := d) (n := d) (k := 2 * H) (q := 2 * q)
    (by omega) h2qd
  refine ⟨ρ₀, hρ₀, fun ρ hρ hρle => ?_⟩
  have hmeas_attn : MeasurableSet (attnPairBox d H ρ) :=
    ((measurableSet_matBox ρ).prod ((measurableSet_matBox ρ).prod (measurableSet_matBox ρ))).prod
      (measurableSet_matBox ρ)
  have hmeas_fib : MeasurableSet
      (fiberBox (m := d) (q := 2 * q) (k' := 2 * H - 2 * q) (n' := d - 2 * q) ρ) :=
    ((measurableSet_matBox ρ).prod ((measurableSet_matBox ρ).prod (measurableSet_matBox ρ))).prod
      ((measurableSet_matBox ρ).prod ((measurableSet_matBox ρ).prod (measurableSet_matBox ρ)))
  have hmeas_lora : Measurable
      (loraFiberLoss (m := d) (q := 2 * q) (k' := 2 * H - 2 * q) (n' := d - 2 * q)) := by
    unfold loraFiberLoss frobSq
    simp only [Matrix.mul_apply, Matrix.add_apply, Matrix.of_apply]
    fun_prop
  have hZeq : ∀ N : ℝ,
      SLT.Z (volume.restrict (attnPairBox d H ρ)) (attnVOLinLoss d H q) N =
      SLT.Z (volume.restrict (fiberBox (m := d) (q := 2 * q) (k' := 2 * H - 2 * q) (n' := d - 2 * q) ρ))
        loraFiberLoss N := fun N =>
    Z_eq_of_measurePreserving eTot heTot_mp (attnPairBox d H ρ) _ hmeas_attn hmeas_fib
      heTot_box (attnVOLinLoss d H q) loraFiberLoss hmeas_lora heTot_loss N
  have hlam_eq : (d : ℝ) * ((2 * q : ℕ) : ℝ) / 2 + awLambda d (d - 2 * q) (2 * H - 2 * q) =
      (d : ℝ) * q + awLambda d (d - 2 * q) (2 * H - 2 * q) := by
    push_cast; ring
  have hpair := hlora ρ hρ hρle
  rw [hlam_eq] at hpair
  exact hpair.congr (Filter.Eventually.of_forall fun N => (hZeq N).symm)

/-- **Joint `V+O` Self-Attention LoRA Exact Learning Pair (`[LC]`, exact to all orders):**
For ALL `d, H, q : ℕ` with `q ≤ H` and `2 * q ≤ d`, the exact all-orders `V+O` loss
`‖(I_d + B_O A_O)(I_d + B_V A_V) - I_d‖_F²` on `[-δ, δ]^{4 d H}` has exact learning pair
`(d * q + awLambda d (d - 2*q) (2*H - 2*q), awMult d (d - 2*q) (2*H - 2*q))`. -/
theorem attention_VO_exact_pair {d H q : ℕ} (hqH : q ≤ H) (h2qd : 2 * q ≤ d) :
    ∃ δ₀ : ℝ, 0 < δ₀ ∧ ∀ δ : ℝ, 0 < δ → δ ≤ δ₀ →
      SLT.HasLaplacePair
        (SLT.Z (volume.restrict (attnPairBox d H δ)) (attnVOLoss d H q))
        ((d : ℝ) * q + awLambda d (d - 2 * q) (2 * H - 2 * q))
        (awMult d (d - 2 * q) (2 * H - 2 * q)) := by
  let B_P : ℝ → Set (RMat d H × RMat H d × RMat H d) := fun ρ =>
    matBox ρ ×ˢ matBox ρ ×ˢ matBox ρ
  let M : (RMat d H × RMat H d × RMat H d) → Matrix (Fin d) (Fin d) ℝ := fun p =>
    1 + Matrix.of p.1 * (baseAO d H q + Matrix.of p.2.1)
  have hM_cont : Continuous M := by
    refine continuous_const.add ?_
    refine continuous_pi fun i => continuous_pi fun j => ?_
    simp only [Matrix.mul_apply, Matrix.add_apply, Matrix.of_apply]
    fun_prop
  have hM0 : M 0 = 1 := by
    ext i j
    simp [M, Matrix.mul_apply]
  have hL_raw_meas : Measurable (attnVOLoss d H q) := by
    unfold attnVOLoss frobSq
    simp only [Matrix.sub_apply, Matrix.mul_apply, Matrix.add_apply, Matrix.of_apply]
    fun_prop
  have hL_lin_meas : Measurable (attnVOLinLoss d H q) := by
    unfold attnVOLinLoss frobSq
    simp only [Matrix.add_apply, Matrix.mul_apply, Matrix.of_apply]
    fun_prop
  have heq : ∀ p B, attnVOLoss d H q (p, B) = attnVOLinLoss d H q (p, toRMat (M p * Matrix.of B)) := by
    intro p B
    unfold attnVOLoss attnVOLinLoss
    have hto : Matrix.of (toRMat (M p * Matrix.of B)) = M p * Matrix.of B := rfl
    rw [hto]
    exact congrArg frobSq (attnVO_factor (Matrix.of p.1) (Matrix.of B)
      (baseAO d H q + Matrix.of p.2.1) (baseAV d H q + Matrix.of p.2.2))
  exact hasLaplacePair_of_leftMul_reduction B_P
    (fun ρ => (measurableSet_matBox ρ).prod ((measurableSet_matBox ρ).prod (measurableSet_matBox ρ)))
    (fun hle => Set.prod_mono (matBox_mono hle) (Set.prod_mono (matBox_mono hle) (matBox_mono hle)))
    (fun U hU => tripleBox_nhds hU)
    M hM_cont hM0 (attnVOLoss d H q) (attnVOLinLoss d H q) hL_raw_meas hL_lin_meas heq
    (attnVOLin_hasLaplacePair hqH h2qd)

/-- Algebraic evaluation of `awLambda d d (2*H)` when `H ≤ d`: `d * H - H² / 2`, multiplicity `1`. -/
theorem awLambda_sq_even (d H : ℕ) (hHd : H ≤ d) :
    awLambda d d (2 * H) = (d : ℝ) * H - (H : ℝ) ^ 2 / 2 ∧ awMult d d (2 * H) = 1 := by
  unfold awLambda awLambdaEven awMult
  by_cases heq : d = H
  · subst heq
    have hcond : d + d ≤ 2 * d := by omega
    simp only [hcond, ↓reduceIte]
    exact ⟨by ring, trivial⟩
  · have hn1 : ¬ (d + d ≤ 2 * H) := by omega
    by_cases hH0 : H = 0
    · subst hH0
      have hn1' : ¬ (d + d ≤ 0) := by omega
      have h2 : 0 + d ≤ d := by omega
      simp only [mul_zero, hn1', h2, ↓reduceIte, Nat.cast_zero, zero_mul, zero_div,
        zero_pow (by norm_num : 2 ≠ 0), sub_zero]
      exact ⟨trivial, trivial⟩
    · have hn2 : ¬ (2 * H + d ≤ d) := by omega
      have heven : Even (d + d + 2 * H) := ⟨d + H, by omega⟩
      simp only [hn1, hn2, heven, ↓reduceIte]
      exact ⟨by push_cast; ring, trivial⟩

/-- **Joint `V+O` Subcritical Plateau (`[LC]`):**
For `H ≤ d` and `2 * q ≤ H`, the exact all-orders `V+O` learning pair is flat at `(d * H - H² / 2, 1)`. -/
theorem attention_VO_subcritical_plateau {d H q : ℕ} (hHd : H ≤ d) (hsub : 2 * q ≤ H) :
    ∃ δ₀ : ℝ, 0 < δ₀ ∧ ∀ δ : ℝ, 0 < δ → δ ≤ δ₀ →
      SLT.HasLaplacePair
        (SLT.Z (volume.restrict (attnPairBox d H δ)) (attnVOLoss d H q))
        ((d : ℝ) * H - (H : ℝ) ^ 2 / 2) 1 := by
  obtain ⟨δ₀, hδ₀, h⟩ := attention_VO_exact_pair (d := d) (H := H) (q := q) (by omega) (by omega)
  refine ⟨δ₀, hδ₀, fun δ hδ hδle => ?_⟩
  have hpair := h δ hδ hδle
  have hvals : (d : ℝ) * q + awLambda d (d - 2 * q) (2 * H - 2 * q) = (d : ℝ) * H - (H : ℝ) ^ 2 / 2 ∧
      awMult d (d - 2 * q) (2 * H - 2 * q) = 1 := by
    by_cases hH0 : H = 0
    · have hq0 : q = 0 := by omega
      subst hH0; subst hq0
      simpa using awLambda_sq_even d 0 (Nat.zero_le d)
    · by_cases hHd_eq : H = d
      · rw [hHd_eq]
        have hcond : d + (d - 2 * q) ≤ 2 * d - 2 * q := by omega
        unfold awLambda awMult
        simp only [hcond, ↓reduceIte]
        have hcast : ((d - 2 * q : ℕ) : ℝ) = (d : ℝ) - 2 * (q : ℝ) := by
          rw [Nat.cast_sub (by omega)]; push_cast; ring
        rw [hcast]
        exact ⟨by ring, trivial⟩
      · by_cases h2qH : 2 * q = H
        · have hcond : (2 * H - 2 * q) + (d - 2 * q) ≤ d := by omega
          have hn1 : ¬ (d + (d - 2 * q) ≤ 2 * H - 2 * q) := by omega
          unfold awLambda awMult
          simp only [hn1, hcond, ↓reduceIte]
          have hc1 : ((d - 2 * q : ℕ) : ℝ) = (d : ℝ) - 2 * (q : ℝ) := by
            rw [Nat.cast_sub (by omega)]; push_cast; ring
          have hc2 : ((2 * H - 2 * q : ℕ) : ℝ) = 2 * (H : ℝ) - 2 * (q : ℝ) := by
            rw [Nat.cast_sub (by omega)]; push_cast; ring
          have hqH_r : (H : ℝ) = 2 * (q : ℝ) := by exact_mod_cast h2qH.symm
          rw [hc1, hc2, hqH_r]
          exact ⟨by ring, trivial⟩
        · have hn1 : ¬ (d + (d - 2 * q) ≤ 2 * H - 2 * q) := by omega
          have hn2 : ¬ ((2 * H - 2 * q) + (d - 2 * q) ≤ d) := by omega
          have hn3 : ¬ ((2 * H - 2 * q) + d ≤ d - 2 * q) := by omega
          have heven : Even (d + (d - 2 * q) + (2 * H - 2 * q)) := ⟨d + H - 2 * q, by omega⟩
          unfold awLambda awLambdaEven awMult
          simp only [hn1, hn2, hn3, heven, ↓reduceIte]
          have hc1 : ((d - 2 * q : ℕ) : ℝ) = (d : ℝ) - 2 * (q : ℝ) := by
            rw [Nat.cast_sub (by omega)]; push_cast; ring
          have hc2 : ((2 * H - 2 * q : ℕ) : ℝ) = 2 * (H : ℝ) - 2 * (q : ℝ) := by
            rw [Nat.cast_sub (by omega)]; push_cast; ring
          rw [hc1, hc2]
          exact ⟨by ring, trivial⟩
  rw [hvals.1, hvals.2] at hpair
  exact hpair

/-- **Joint `V+O` 4× Supercritical Parabola (`[LC]`):**
For `H ≤ d` and `H ≤ 2 * q ≤ d` (with `q ≤ H`), the exact all-orders `V+O` learning pair sits on
the `4×`-amplified supercritical parabola `(d * H - H² / 2 + 2 * (q - H / 2)², 1)`. -/
theorem attention_VO_supercritical_parabola {d H q : ℕ}
    (hqH : q ≤ H) (hsup : H ≤ 2 * q) (h2qd : 2 * q ≤ d) :
    ∃ δ₀ : ℝ, 0 < δ₀ ∧ ∀ δ : ℝ, 0 < δ → δ ≤ δ₀ →
      SLT.HasLaplacePair
        (SLT.Z (volume.restrict (attnPairBox d H δ)) (attnVOLoss d H q))
        ((d : ℝ) * H - (H : ℝ) ^ 2 / 2 + 2 * ((q : ℝ) - (H : ℝ) / 2) ^ 2) 1 := by
  obtain ⟨δ₀, hδ₀, h⟩ := attention_VO_exact_pair hqH h2qd
  refine ⟨δ₀, hδ₀, fun δ hδ hδle => ?_⟩
  have hpair := h δ hδ hδle
  have hvals : (d : ℝ) * q + awLambda d (d - 2 * q) (2 * H - 2 * q) =
      (d : ℝ) * H - (H : ℝ) ^ 2 / 2 + 2 * ((q : ℝ) - (H : ℝ) / 2) ^ 2 ∧
      awMult d (d - 2 * q) (2 * H - 2 * q) = 1 := by
    have hc1 : ((d - 2 * q : ℕ) : ℝ) = (d : ℝ) - 2 * (q : ℝ) := by
      rw [Nat.cast_sub h2qd]; push_cast; ring
    have hc2 : ((2 * H - 2 * q : ℕ) : ℝ) = 2 * (H : ℝ) - 2 * (q : ℝ) := by
      rw [Nat.cast_sub (by omega)]; push_cast; ring
    by_cases h1 : d + (d - 2 * q) ≤ 2 * H - 2 * q
    · have hdeq : d = H := by omega
      have h2qH : 2 * q = H := by omega
      unfold awLambda awMult
      simp only [h1, ↓reduceIte]
      rw [hc1]
      have hd_r : (d : ℝ) = (H : ℝ) := by exact_mod_cast hdeq
      have hH_r : (H : ℝ) = 2 * (q : ℝ) := by exact_mod_cast h2qH.symm
      rw [hd_r, hH_r]
      exact ⟨by ring, trivial⟩
    · have h2 : (2 * H - 2 * q) + (d - 2 * q) ≤ d := by omega
      unfold awLambda awMult
      simp only [h1, h2, ↓reduceIte]
      rw [hc1, hc2]
      exact ⟨by ring, trivial⟩
  rw [hvals.1, hvals.2] at hpair
  exact hpair

/-! ### 5. Exact Two-Sided Schur Complement Reduction (Balanced `(q, q)` Fibers) -/

/-- Passive coordinates `(((Δ₁₁, Δ₂₁), (Γ₁₁, Γ₁₂)), (B₁₁, U₁, V₁))` of a balanced two-sided
`(q, q)` matrix fiber (note `B₁₁ ∈ ℝ^{q×q}` is the exact `q²`-dimensional `GL(q)` gauge coordinate!). -/
abbrev TwoSidedPassive (q d' k' : ℕ) :=
  ((RMat q q × RMat d' q) × (RMat q q × RMat q d')) × (RMat q q × RMat q k' × RMat k' q)

/-- Active coordinates `((C₁₁, C₁₂, B₂₁), (U₂, V₂))` of a balanced two-sided `(q, q)` fiber. -/
abbrev TwoSidedActive (q d' k' : ℕ) :=
  (RMat q q × RMat q d' × RMat d' q) × (RMat d' k' × RMat k' d')

/-- Uniform `δ`-box on `TwoSidedPassive q d' k'`. -/
def passiveBox {q d' k' : ℕ} (δ : ℝ) : Set (TwoSidedPassive q d' k') :=
  ((matBox δ ×ˢ matBox δ) ×ˢ (matBox δ ×ˢ matBox δ)) ×ˢ (matBox δ ×ˢ matBox δ ×ˢ matBox δ)

/-- Uniform `δ`-box on `TwoSidedActive q d' k'`. -/
def activeBox {q d' k' : ℕ} (δ : ℝ) : Set (TwoSidedActive q d' k') :=
  (matBox δ ×ˢ matBox δ ×ˢ matBox δ) ×ˢ (matBox δ ×ˢ matBox δ)

/-- Exact 4-block Frobenius-squared loss on `TwoSidedPassive q d' k' × TwoSidedActive q d' k'`. -/
def twoSidedLoss {q d' k' : ℕ} (x : TwoSidedPassive q d' k' × TwoSidedActive q d' k') : ℝ :=
  let Δ₁₁ := Matrix.of x.1.1.1.1
  let Δ₂₁ := Matrix.of x.1.1.1.2
  let Γ₁₁ := Matrix.of x.1.1.2.1
  let Γ₁₂ := Matrix.of x.1.1.2.2
  let B₁₁ := Matrix.of x.1.2.1
  let U₁  := Matrix.of x.1.2.2.1
  let V₁  := Matrix.of x.1.2.2.2
  let C₁₁ := Matrix.of x.2.1.1
  let C₁₂ := Matrix.of x.2.1.2.1
  let B₂₁ := Matrix.of x.2.1.2.2
  let U₂  := Matrix.of x.2.2.1
  let V₂  := Matrix.of x.2.2.2
  let M_L := (1 : Matrix (Fin q) (Fin q) ℝ) + Δ₁₁
  let M_R := (1 : Matrix (Fin q) (Fin q) ℝ) + Γ₁₁
  frobSq (B₁₁ * M_R + M_L * C₁₁ + U₁ * V₁) +
  frobSq (B₁₁ * Γ₁₂ + M_L * C₁₂ + U₁ * V₂) +
  frobSq (B₂₁ * M_R + Δ₂₁ * C₁₁ + U₂ * V₁) +
  frobSq (B₂₁ * Γ₁₂ + Δ₂₁ * C₁₂ + U₂ * V₂)

/-- Decoupled active model loss: 3 independent Gaussian blocks `(q×q, q×d', d'×q)` + rank-0 `(d', k', d')`. -/
def twoSidedDecoupledLoss {q d' k' : ℕ} (a : TwoSidedActive q d' k') : ℝ :=
  frobSq (Matrix.of a.1.1) + frobSq (Matrix.of a.1.2.1) + frobSq (Matrix.of a.1.2.2) +
    rank0Loss a.2

/-- Exact two-sided Schur diffeomorphism on `TwoSidedActive` for fixed `p : TwoSidedPassive`. -/
noncomputable def twoSidedSchurMap {q d' k' : ℕ} (p : TwoSidedPassive q d' k')
    (a : TwoSidedActive q d' k') : TwoSidedActive q d' k' :=
  let Δ₁₁ := Matrix.of p.1.1.1
  let Δ₂₁ := Matrix.of p.1.1.2
  let Γ₁₁ := Matrix.of p.1.2.1
  let Γ₁₂ := Matrix.of p.1.2.2
  let B₁₁ := Matrix.of p.2.1
  let U₁  := Matrix.of p.2.2.1
  let V₁  := Matrix.of p.2.2.2
  let M_L := (1 : Matrix (Fin q) (Fin q) ℝ) + Δ₁₁
  let M_R := (1 : Matrix (Fin q) (Fin q) ℝ) + Γ₁₁
  let C₁₁ := Matrix.of a.1.1
  let C₁₂ := Matrix.of a.1.2.1
  let B₂₁ := Matrix.of a.1.2.2
  let U₂  := Matrix.of a.2.1
  let V₂  := Matrix.of a.2.2
  let U₂' := U₂ - Δ₂₁ * M_L⁻¹ * U₁
  let V₂' := V₂ - V₁ * M_R⁻¹ * Γ₁₂
  let Z₂₁ := B₂₁ - Δ₂₁ * M_L⁻¹ * B₁₁ + U₂' * (V₁ * M_R⁻¹)
  let Z₁₂ := C₁₂ - C₁₁ * (M_R⁻¹ * Γ₁₂) + (M_L⁻¹ * U₁) * V₂'
  let Z₁₁ := C₁₁ * M_R⁻¹ + M_L⁻¹ * B₁₁ + M_L⁻¹ * U₁ * V₁ * M_R⁻¹
  ((toRMat Z₁₁, toRMat Z₁₂, toRMat Z₂₁), (toRMat U₂', toRMat V₂'))

/-- Exact algebraic 4-block two-sided Schur identities. -/
theorem twoSided_schur_identities {q d' k' : ℕ} (p : TwoSidedPassive q d' k')
    (a : TwoSidedActive q d' k')
    (hML : IsUnit ((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of p.1.1.1).det)
    (hMR : IsUnit ((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of p.1.2.1).det) :
    let Δ₁₁ := Matrix.of p.1.1.1; let Δ₂₁ := Matrix.of p.1.1.2
    let Γ₁₁ := Matrix.of p.1.2.1; let Γ₁₂ := Matrix.of p.1.2.2
    let B₁₁ := Matrix.of p.2.1; let U₁ := Matrix.of p.2.2.1; let V₁ := Matrix.of p.2.2.2
    let M_L := (1 : Matrix (Fin q) (Fin q) ℝ) + Δ₁₁
    let M_R := (1 : Matrix (Fin q) (Fin q) ℝ) + Γ₁₁
    let C₁₁ := Matrix.of a.1.1; let C₁₂ := Matrix.of a.1.2.1; let B₂₁ := Matrix.of a.1.2.2
    let U₂ := Matrix.of a.2.1; let V₂ := Matrix.of a.2.2
    let s := twoSidedSchurMap p a
    let Z₁₁ := Matrix.of s.1.1; let Z₁₂ := Matrix.of s.1.2.1; let Z₂₁ := Matrix.of s.1.2.2
    let U₂' := Matrix.of s.2.1; let V₂' := Matrix.of s.2.2
    (B₁₁ * M_R + M_L * C₁₁ + U₁ * V₁ = M_L * Z₁₁ * M_R) ∧
    (B₁₁ * Γ₁₂ + M_L * C₁₂ + U₁ * V₂ = M_L * (Z₁₁ * Γ₁₂ + Z₁₂)) ∧
    (B₂₁ * M_R + Δ₂₁ * C₁₁ + U₂ * V₁ = (Δ₂₁ * Z₁₁ + Z₂₁) * M_R) ∧
    (B₂₁ * Γ₁₂ + Δ₂₁ * C₁₂ + U₂ * V₂ = Δ₂₁ * (Z₁₁ * Γ₁₂ + Z₁₂) + Z₂₁ * Γ₁₂ + U₂' * V₂') := by
  intro Δ₁₁ Δ₂₁ Γ₁₁ Γ₁₂ B₁₁ U₁ V₁ M_L M_R C₁₁ C₁₂ B₂₁ U₂ V₂ s Z₁₁ Z₁₂ Z₂₁ U₂' V₂'
  have hML_r : M_L * M_L⁻¹ = 1 := Matrix.mul_nonsing_inv M_L hML
  have hMR_l : M_R⁻¹ * M_R = 1 := Matrix.nonsing_inv_mul M_R hMR
  have hZ11 : Z₁₁ = C₁₁ * M_R⁻¹ + M_L⁻¹ * B₁₁ + M_L⁻¹ * U₁ * V₁ * M_R⁻¹ := rfl
  have hZ12 : Z₁₂ = C₁₂ - C₁₁ * (M_R⁻¹ * Γ₁₂) + (M_L⁻¹ * U₁) * V₂' := rfl
  have hZ21 : Z₂₁ = B₂₁ - Δ₂₁ * M_L⁻¹ * B₁₁ + U₂' * (V₁ * M_R⁻¹) := rfl
  have hU2' : U₂' = U₂ - Δ₂₁ * M_L⁻¹ * U₁ := rfl
  have hV2' : V₂' = V₂ - V₁ * M_R⁻¹ * Γ₁₂ := rfl
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hZ11]
    simp only [Matrix.mul_add, Matrix.add_mul, Matrix.mul_assoc]
    have ht1 : M_L * (C₁₁ * (M_R⁻¹ * M_R)) = M_L * C₁₁ := by rw [hMR_l, Matrix.mul_one]
    have ht2 : M_L * (M_L⁻¹ * (B₁₁ * M_R)) = B₁₁ * M_R := by
      rw [← Matrix.mul_assoc M_L M_L⁻¹, hML_r, Matrix.one_mul]
    have ht3 : M_L * (M_L⁻¹ * (U₁ * (V₁ * (M_R⁻¹ * M_R)))) = U₁ * V₁ := by
      rw [hMR_l, Matrix.mul_one, ← Matrix.mul_assoc M_L M_L⁻¹, hML_r, Matrix.one_mul]
    rw [ht1, ht2, ht3]
    abel
  · rw [hZ11, hZ12, hV2']
    simp only [Matrix.mul_add, Matrix.add_mul, Matrix.mul_sub, Matrix.mul_assoc]
    have ht1 : M_L * (M_L⁻¹ * (B₁₁ * Γ₁₂)) = B₁₁ * Γ₁₂ := by
      rw [← Matrix.mul_assoc M_L M_L⁻¹, hML_r, Matrix.one_mul]
    have ht2 : M_L * (M_L⁻¹ * (U₁ * (V₁ * (M_R⁻¹ * Γ₁₂)))) = U₁ * (V₁ * (M_R⁻¹ * Γ₁₂)) := by
      rw [← Matrix.mul_assoc M_L M_L⁻¹, hML_r, Matrix.one_mul]
    have ht3 : M_L * (M_L⁻¹ * (U₁ * V₂)) = U₁ * V₂ := by
      rw [← Matrix.mul_assoc M_L M_L⁻¹, hML_r, Matrix.one_mul]
    rw [ht1, ht2, ht3]
    abel
  · rw [hZ11, hZ21, hU2']
    simp only [Matrix.mul_add, Matrix.add_mul, Matrix.sub_mul, Matrix.mul_assoc]
    have ht1 : Δ₂₁ * (C₁₁ * (M_R⁻¹ * M_R)) = Δ₂₁ * C₁₁ := by rw [hMR_l, Matrix.mul_one]
    have ht2 : Δ₂₁ * (M_L⁻¹ * (U₁ * (V₁ * (M_R⁻¹ * M_R)))) = Δ₂₁ * (M_L⁻¹ * (U₁ * V₁)) := by
      rw [hMR_l, Matrix.mul_one]
    have ht3 : U₂ * (V₁ * (M_R⁻¹ * M_R)) = U₂ * V₁ := by rw [hMR_l, Matrix.mul_one]
    rw [ht1, ht2, ht3]
    abel
  · rw [hZ11, hZ12, hZ21, hU2', hV2']
    simp only [Matrix.mul_add, Matrix.add_mul, Matrix.mul_sub, Matrix.sub_mul, Matrix.mul_assoc]
    abel

/-- Helper: multiplying by `1 + E` with `frobSq E ≤ 1/16` inflates `frobSq` by at most `3`. -/
private theorem frobSq_one_add_mul_le {a b : ℕ} (E : Matrix (Fin a) (Fin a) ℝ)
    (W : Matrix (Fin a) (Fin b) ℝ) (hE : frobSq E ≤ 1 / 16) :
    frobSq ((1 + E) * W) ≤ 3 * frobSq W := by
  rw [Matrix.add_mul, Matrix.one_mul]
  have h1 := frobSq_add_le W (E * W)
  have h2 := frobSq_mul_le E W
  have h3 : frobSq E * frobSq W ≤ (1 / 16 : ℝ) * frobSq W :=
    mul_le_mul_of_nonneg_right hE (frobSq_nonneg W)
  linarith [frobSq_nonneg W]

private theorem frobSq_mul_one_add_le {a b : ℕ} (W : Matrix (Fin a) (Fin b) ℝ)
    (E : Matrix (Fin b) (Fin b) ℝ) (hE : frobSq E ≤ 1 / 16) :
    frobSq (W * (1 + E)) ≤ 3 * frobSq W := by
  rw [Matrix.mul_add, Matrix.mul_one]
  have h1 := frobSq_add_le W (W * E)
  have h2 := frobSq_mul_le W E
  have h3 : frobSq W * frobSq E ≤ frobSq W * (1 / 16 : ℝ) :=
    mul_le_mul_of_nonneg_left hE (frobSq_nonneg W)
  linarith [frobSq_nonneg W]

/-- **Two-Sided Schur Bilipschitz Loss Comparison (`1/20` and `20`, dimension-independent):** -/
theorem twoSided_loss_bounds {q d' k' : ℕ} (p : TwoSidedPassive q d' k')
    (a : TwoSidedActive q d' k')
    (hML : IsUnit ((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of p.1.1.1).det)
    (hMR : IsUnit ((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of p.1.2.1).det)
    (hΔ11 : frobSq (Matrix.of p.1.1.1) ≤ 1 / 16)
    (hΔ21 : frobSq (Matrix.of p.1.1.2) ≤ 1 / 16)
    (hΓ11 : frobSq (Matrix.of p.1.2.1) ≤ 1 / 16)
    (hΓ12 : frobSq (Matrix.of p.1.2.2) ≤ 1 / 16)
    (hMLinv : frobSq (((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of p.1.1.1)⁻¹ - 1) ≤ 1 / 16)
    (hMRinv : frobSq (((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of p.1.2.1)⁻¹ - 1) ≤ 1 / 16) :
    (1 / 20 : ℝ) * twoSidedDecoupledLoss (twoSidedSchurMap p a) ≤ twoSidedLoss (p, a) ∧
      twoSidedLoss (p, a) ≤ 20 * twoSidedDecoupledLoss (twoSidedSchurMap p a) := by
  set M_L := (1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of p.1.1.1
  set M_R := (1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of p.1.2.1
  set Δ₂₁ := Matrix.of p.1.1.2
  set Γ₁₂ := Matrix.of p.1.2.2
  set s := twoSidedSchurMap p a
  set Z₁₁ := Matrix.of s.1.1; set Z₁₂ := Matrix.of s.1.2.1; set Z₂₁ := Matrix.of s.1.2.2
  set U₂' := Matrix.of s.2.1; set V₂' := Matrix.of s.2.2
  set P₁₁ := Matrix.of p.2.1 * M_R + M_L * Matrix.of a.1.1 + Matrix.of p.2.2.1 * Matrix.of p.2.2.2
  set P₁₂ := Matrix.of p.2.1 * Γ₁₂ + M_L * Matrix.of a.1.2.1 + Matrix.of p.2.2.1 * Matrix.of a.2.2
  set P₂₁ := Matrix.of a.1.2.2 * M_R + Δ₂₁ * Matrix.of a.1.1 + Matrix.of a.2.1 * Matrix.of p.2.2.2
  set P₂₂ := Matrix.of a.1.2.2 * Γ₁₂ + Δ₂₁ * Matrix.of a.1.2.1 + Matrix.of a.2.1 * Matrix.of a.2.2
  obtain ⟨hP11, hP12, hP21, hP22⟩ := twoSided_schur_identities p a hML hMR
  have hP11' : P₁₁ = M_L * Z₁₁ * M_R := hP11
  have hP12' : P₁₂ = M_L * (Z₁₁ * Γ₁₂ + Z₁₂) := hP12
  have hP21' : P₂₁ = (Δ₂₁ * Z₁₁ + Z₂₁) * M_R := hP21
  have hP22' : P₂₂ = Δ₂₁ * (Z₁₁ * Γ₁₂ + Z₁₂) + Z₂₁ * Γ₁₂ + U₂' * V₂' := hP22
  have hML_l : M_L⁻¹ * M_L = 1 := Matrix.nonsing_inv_mul M_L hML
  have hMR_r : M_R * M_R⁻¹ = 1 := Matrix.mul_nonsing_inv M_R hMR
  have hML_inv_mul : ∀ {b : ℕ} (W : Matrix (Fin q) (Fin b) ℝ), frobSq (M_L⁻¹ * W) ≤ 3 * frobSq W := fun W => by
    have : M_L⁻¹ = 1 + (M_L⁻¹ - 1) := by abel
    rw [this]; exact frobSq_one_add_mul_le _ W hMLinv
  have hMR_inv_mul : ∀ {b : ℕ} (W : Matrix (Fin b) (Fin q) ℝ), frobSq (W * M_R⁻¹) ≤ 3 * frobSq W := fun W => by
    have : M_R⁻¹ = 1 + (M_R⁻¹ - 1) := by abel
    rw [this]; exact frobSq_mul_one_add_le W _ hMRinv
  have hML_mul : ∀ {b : ℕ} (W : Matrix (Fin q) (Fin b) ℝ), frobSq (M_L * W) ≤ 3 * frobSq W :=
    fun W => frobSq_one_add_mul_le _ W hΔ11
  have hMR_mul : ∀ {b : ℕ} (W : Matrix (Fin b) (Fin q) ℝ), frobSq (W * M_R) ≤ 3 * frobSq W :=
    fun W => frobSq_mul_one_add_le W _ hΓ11
  -- Block 11 bounds
  have h11_up : frobSq P₁₁ ≤ 9 * frobSq Z₁₁ := by
    rw [hP11']
    exact (hMR_mul (M_L * Z₁₁)).trans (by linarith [hML_mul Z₁₁])
  have h11_lo : frobSq Z₁₁ ≤ 9 * frobSq P₁₁ := by
    have heq : Z₁₁ = M_L⁻¹ * P₁₁ * M_R⁻¹ := by
      calc Z₁₁ = 1 * Z₁₁ * 1 := by rw [Matrix.one_mul, Matrix.mul_one]
        _ = (M_L⁻¹ * M_L) * Z₁₁ * (M_R * M_R⁻¹) := by rw [hML_l, hMR_r]
        _ = M_L⁻¹ * P₁₁ * M_R⁻¹ := by rw [hP11']; simp only [Matrix.mul_assoc]
    rw [heq]
    exact (hMR_inv_mul (M_L⁻¹ * P₁₁)).trans (by linarith [hML_inv_mul P₁₁])
  -- Block 12 bounds
  have h12_sum_up : frobSq (Z₁₁ * Γ₁₂ + Z₁₂) ≤ (1 / 8 : ℝ) * frobSq Z₁₁ + 2 * frobSq Z₁₂ := by
    have h1 := frobSq_add_le (Z₁₁ * Γ₁₂) Z₁₂
    have h2 : frobSq (Z₁₁ * Γ₁₂) ≤ frobSq Z₁₁ * (1 / 16 : ℝ) :=
      (frobSq_mul_le Z₁₁ Γ₁₂).trans (mul_le_mul_of_nonneg_left hΓ12 (frobSq_nonneg _))
    linarith
  have h12_up : frobSq P₁₂ ≤ (3 / 8 : ℝ) * frobSq Z₁₁ + 6 * frobSq Z₁₂ := by
    rw [hP12']; linarith [hML_mul (Z₁₁ * Γ₁₂ + Z₁₂), h12_sum_up]
  have h12_lo : frobSq Z₁₂ ≤ 6 * frobSq P₁₂ + (9 / 8 : ℝ) * frobSq P₁₁ := by
    have heq : Z₁₂ = M_L⁻¹ * P₁₂ - Z₁₁ * Γ₁₂ := by
      rw [hP12', ← Matrix.mul_assoc, hML_l, Matrix.one_mul]; abel
    rw [heq]
    have h1 := frobSq_sub_le (M_L⁻¹ * P₁₂) (Z₁₁ * Γ₁₂)
    have h2 := hML_inv_mul P₁₂
    have h3 : frobSq (Z₁₁ * Γ₁₂) ≤ frobSq Z₁₁ * (1 / 16 : ℝ) :=
      (frobSq_mul_le Z₁₁ Γ₁₂).trans (mul_le_mul_of_nonneg_left hΓ12 (frobSq_nonneg _))
    linarith [h11_lo]
  -- Block 21 bounds
  have h21_up : frobSq P₂₁ ≤ (3 / 8 : ℝ) * frobSq Z₁₁ + 6 * frobSq Z₂₁ := by
    rw [hP21']
    have h1 := frobSq_add_le (Δ₂₁ * Z₁₁) Z₂₁
    have h2 : frobSq (Δ₂₁ * Z₁₁) ≤ (1 / 16 : ℝ) * frobSq Z₁₁ :=
      (frobSq_mul_le Δ₂₁ Z₁₁).trans (mul_le_mul_of_nonneg_right hΔ21 (frobSq_nonneg _))
    linarith [hMR_mul (Δ₂₁ * Z₁₁ + Z₂₁)]
  have h21_lo : frobSq Z₂₁ ≤ 6 * frobSq P₂₁ + (9 / 8 : ℝ) * frobSq P₁₁ := by
    have heq : Z₂₁ = P₂₁ * M_R⁻¹ - Δ₂₁ * Z₁₁ := by
      rw [hP21', Matrix.mul_assoc, hMR_r, Matrix.mul_one]; abel
    rw [heq]
    have h1 := frobSq_sub_le (P₂₁ * M_R⁻¹) (Δ₂₁ * Z₁₁)
    have h2 := hMR_inv_mul P₂₁
    have h3 : frobSq (Δ₂₁ * Z₁₁) ≤ (1 / 16 : ℝ) * frobSq Z₁₁ :=
      (frobSq_mul_le Δ₂₁ Z₁₁).trans (mul_le_mul_of_nonneg_right hΔ21 (frobSq_nonneg _))
    linarith [h11_lo]
  -- Block 22 bounds
  have h22_up : frobSq P₂₂ ≤ (1 / 2 : ℝ) * frobSq Z₁₁ + frobSq Z₁₂ + frobSq Z₂₁ + 2 * frobSq (U₂' * V₂') := by
    rw [hP22']
    have h1 := frobSq_add_le (Δ₂₁ * (Z₁₁ * Γ₁₂ + Z₁₂) + Z₂₁ * Γ₁₂) (U₂' * V₂')
    have h2 := frobSq_add_le (Δ₂₁ * (Z₁₁ * Γ₁₂ + Z₁₂)) (Z₂₁ * Γ₁₂)
    have h3 : frobSq (Δ₂₁ * (Z₁₁ * Γ₁₂ + Z₁₂)) ≤ (1 / 16 : ℝ) * frobSq (Z₁₁ * Γ₁₂ + Z₁₂) :=
      (frobSq_mul_le Δ₂₁ _).trans (mul_le_mul_of_nonneg_right hΔ21 (frobSq_nonneg _))
    have h4 : frobSq (Z₂₁ * Γ₁₂) ≤ frobSq Z₂₁ * (1 / 16 : ℝ) :=
      (frobSq_mul_le Z₂₁ Γ₁₂).trans (mul_le_mul_of_nonneg_left hΓ12 (frobSq_nonneg _))
    linarith [h12_sum_up, frobSq_nonneg Z₁₁, frobSq_nonneg Z₁₂, frobSq_nonneg Z₂₁]
  have h22_lo : frobSq (U₂' * V₂') ≤ 2 * frobSq P₂₂ + frobSq P₁₂ + 2 * frobSq P₂₁ + frobSq P₁₁ := by
    have heq : U₂' * V₂' = P₂₂ - (Δ₂₁ * (M_L⁻¹ * P₁₂) + Z₂₁ * Γ₁₂) := by
      have hsub1 : M_L⁻¹ * P₁₂ = Z₁₁ * Γ₁₂ + Z₁₂ := by
        rw [hP12', ← Matrix.mul_assoc, hML_l, Matrix.one_mul]
      rw [hsub1, hP22']; abel
    rw [heq]
    have h1 := frobSq_sub_le P₂₂ (Δ₂₁ * (M_L⁻¹ * P₁₂) + Z₂₁ * Γ₁₂)
    have h2 := frobSq_add_le (Δ₂₁ * (M_L⁻¹ * P₁₂)) (Z₂₁ * Γ₁₂)
    have h3 : frobSq (Δ₂₁ * (M_L⁻¹ * P₁₂)) ≤ (1 / 16 : ℝ) * frobSq (M_L⁻¹ * P₁₂) :=
      (frobSq_mul_le Δ₂₁ _).trans (mul_le_mul_of_nonneg_right hΔ21 (frobSq_nonneg _))
    have h4 := hML_inv_mul P₁₂
    have h5 : frobSq (Z₂₁ * Γ₁₂) ≤ frobSq Z₂₁ * (1 / 16 : ℝ) :=
      (frobSq_mul_le Z₂₁ Γ₁₂).trans (mul_le_mul_of_nonneg_left hΓ12 (frobSq_nonneg _))
    linarith [h21_lo, frobSq_nonneg P₁₁, frobSq_nonneg P₁₂, frobSq_nonneg P₂₁, frobSq_nonneg P₂₂]
  have hloss_eq : twoSidedLoss (p, a) = frobSq P₁₁ + frobSq P₁₂ + frobSq P₂₁ + frobSq P₂₂ := rfl
  have hdec_eq : twoSidedDecoupledLoss (twoSidedSchurMap p a) =
      frobSq Z₁₁ + frobSq Z₁₂ + frobSq Z₂₁ + frobSq (U₂' * V₂') := rfl
  rw [hloss_eq, hdec_eq]
  constructor <;> linarith [frobSq_nonneg Z₁₁, frobSq_nonneg Z₁₂, frobSq_nonneg Z₂₁,
    frobSq_nonneg (U₂' * V₂'), frobSq_nonneg P₁₁, frobSq_nonneg P₁₂, frobSq_nonneg P₂₁, frobSq_nonneg P₂₂]

/-- Pushforward of `volume` on `TwoSidedActive` by `twoSidedSchurMap p` is `|det M_R|^q • volume`. -/
theorem map_twoSidedSchurMap_volume {q d' k' : ℕ} (p : TwoSidedPassive q d' k')
    (hMR : ((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of p.1.2.1).det ≠ 0) :
    Measure.map (twoSidedSchurMap p) volume =
      ENNReal.ofReal |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of p.1.2.1).det| ^ q • volume := by
  set M_L := (1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of p.1.1.1
  set M_R := (1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of p.1.2.1
  set Δ₂₁ := Matrix.of p.1.1.2
  set Γ₁₂ := Matrix.of p.1.2.2
  set B₁₁ := Matrix.of p.2.1
  set U₁  := Matrix.of p.2.2.1
  set V₁  := Matrix.of p.2.2.2
  set shU : RMat d' k' := toRMat (Δ₂₁ * M_L⁻¹ * U₁)
  set shV : RMat k' d' := toRMat (V₁ * M_R⁻¹ * Γ₁₂)
  set shB : RMat d' q := toRMat (Δ₂₁ * M_L⁻¹ * B₁₁)
  set shC : RMat q q := toRMat (M_L⁻¹ * B₁₁ + M_L⁻¹ * U₁ * V₁ * M_R⁻¹)
  let T₁ : TwoSidedActive q d' k' → TwoSidedActive q d' k' :=
    fun a => (a.1, (a.2.1 - shU, a.2.2 - shV))
  let T₂ : TwoSidedActive q d' k' → TwoSidedActive q d' k' :=
    fun a => ((a.1.1, a.1.2.1, a.1.2.2 - shB + toRMat (Matrix.of a.2.1 * (V₁ * M_R⁻¹))), a.2)
  let T₃ : TwoSidedActive q d' k' → TwoSidedActive q d' k' :=
    fun a => ((a.1.1, a.1.2.1 - toRMat (Matrix.of a.1.1 * (M_R⁻¹ * Γ₁₂)) +
      toRMat ((M_L⁻¹ * U₁) * Matrix.of a.2.2), a.1.2.2), a.2)
  let T₄ : TwoSidedActive q d' k' → TwoSidedActive q d' k' :=
    fun a => ((toRMat (Matrix.of a.1.1 * M_R⁻¹), a.1.2), a.2)
  let T₅ : TwoSidedActive q d' k' → TwoSidedActive q d' k' :=
    fun a => ((a.1.1 + shC, a.1.2), a.2)
  have hcomp : twoSidedSchurMap p = T₅ ∘ T₄ ∘ T₃ ∘ T₂ ∘ T₁ := by
    have hof_sub : ∀ {a b : ℕ} (X Y : RMat a b), Matrix.of (X - Y) = Matrix.of X - Matrix.of Y :=
      fun _ _ => rfl
    have hof_to : ∀ {a b : ℕ} (X : Matrix (Fin a) (Fin b) ℝ), Matrix.of (toRMat X) = X :=
      fun _ => rfl
    funext ⟨⟨C11, C12, B21⟩, ⟨U2, V2⟩⟩
    ext i j <;> simp [twoSidedSchurMap, T₁, T₂, T₃, T₄, T₅, shU, shV, shB, shC,
      M_L, M_R, Δ₂₁, Γ₁₂, B₁₁, U₁, V₁, toRMat_apply, hof_sub, hof_to, Matrix.of_apply,
      add_assoc, Matrix.mul_assoc]
  have hmeas_mulR : ∀ {a b c : ℕ} (K : Matrix (Fin b) (Fin c) ℝ),
      Measurable (fun X : RMat a b => toRMat (Matrix.of X * K)) := fun K => by
    refine Measurable.of_eval fun i => Measurable.of_eval fun j => ?_
    simp only [toRMat_apply, Matrix.mul_apply, Matrix.of_apply]; fun_prop
  have hmeas_mulL : ∀ {a b c : ℕ} (K : Matrix (Fin a) (Fin b) ℝ),
      Measurable (fun X : RMat b c => toRMat (K * Matrix.of X)) := fun K => by
    refine Measurable.of_eval fun i => Measurable.of_eval fun j => ?_
    simp only [toRMat_apply, Matrix.mul_apply, Matrix.of_apply]; fun_prop
  have hT1_m : Measurable T₁ :=
    measurable_fst.prodMk (((measurable_fst.comp measurable_snd).sub measurable_const).prodMk
      ((measurable_snd.comp measurable_snd).sub measurable_const))
  have hT2_m : Measurable T₂ :=
    ((measurable_fst.comp measurable_fst).prodMk
      ((measurable_fst.comp (measurable_snd.comp measurable_fst)).prodMk
        (((measurable_snd.comp (measurable_snd.comp measurable_fst)).sub measurable_const).add
          ((hmeas_mulR _).comp (measurable_fst.comp measurable_snd))))).prodMk measurable_snd
  have hT3_m : Measurable T₃ :=
    ((measurable_fst.comp measurable_fst).prodMk
      ((((measurable_fst.comp (measurable_snd.comp measurable_fst)).sub
          ((hmeas_mulR _).comp (measurable_fst.comp measurable_fst))).add
          ((hmeas_mulL _).comp (measurable_snd.comp measurable_snd))).prodMk
        (measurable_snd.comp (measurable_snd.comp measurable_fst)))).prodMk measurable_snd
  have hT4_m : Measurable T₄ :=
    (((hmeas_mulR _).comp (measurable_fst.comp measurable_fst)).prodMk
      (measurable_snd.comp measurable_fst)).prodMk measurable_snd
  have hT5_m : Measurable T₅ :=
    ((((measurable_fst.comp measurable_fst).add measurable_const)).prodMk
      (measurable_snd.comp measurable_fst)).prodMk measurable_snd
  -- Step 1: `T₁` preserves volume
  have hT1 : Measure.map T₁ volume = volume :=
    ((MeasurePreserving.id volume).prod
      ((measurePreserving_sub_right volume shU).prod (measurePreserving_sub_right volume shV))).map_eq
  -- Step 2: `T₂` preserves volume (fiberwise translation of `B₂₁`)
  have hT2 : Measure.map T₂ volume = volume := by
    refine Measure.ext fun s hs => ?_
    rw [Measure.map_apply hT2_m hs]
    change (volume.prod volume) (T₂ ⁻¹' s) = (volume.prod volume) s
    rw [Measure.prod_apply_symm (hT2_m hs), Measure.prod_apply_symm hs]
    refine lintegral_congr fun uv => ?_
    set sh : RMat d' q := -shB + toRMat (Matrix.of uv.1 * (V₁ * M_R⁻¹))
    have hslice_eq : (Prod.mk · uv) ⁻¹' (T₂ ⁻¹' s) =
        (Prod.map (id : RMat q q → _) (Prod.map (id : RMat q d' → _) (fun x => x + sh))) ⁻¹'
          ((Prod.mk · uv) ⁻¹' s) := by
      ext ⟨c11, c12, b21⟩
      simp only [Set.mem_preimage, Prod.map_apply, id_eq, T₂, sh, sub_eq_add_neg, add_assoc]
    rw [hslice_eq]
    have hmp := (MeasurePreserving.id (volume : Measure (RMat q q))).prod
      ((MeasurePreserving.id (volume : Measure (RMat q d'))).prod
        (measurePreserving_add_right (volume : Measure (RMat d' q)) sh))
    exact hmp.measure_preimage (hs.preimage measurable_prodMk_right).nullMeasurableSet
  -- Step 3: `T₃` preserves volume (fiberwise translation of `C₁₂` given `(C₁₁, B₂₁, U₂, V₂)`)
  have hT3 : Measure.map T₃ volume = volume := by
    refine Measure.ext fun s hs => ?_
    rw [Measure.map_apply hT3_m hs]
    change (volume.prod volume) (T₃ ⁻¹' s) = (volume.prod volume) s
    rw [Measure.prod_apply_symm (hT3_m hs), Measure.prod_apply_symm hs]
    refine lintegral_congr fun uv => ?_
    change (volume.prod volume) ((Prod.mk · uv) ⁻¹' (T₃ ⁻¹' s)) =
      (volume.prod volume) ((Prod.mk · uv) ⁻¹' s)
    rw [Measure.prod_apply ((hT3_m hs).preimage measurable_prodMk_right),
      Measure.prod_apply (hs.preimage measurable_prodMk_right)]
    refine lintegral_congr fun c11 => ?_
    set sh : RMat q d' :=
      -toRMat (Matrix.of c11 * (M_R⁻¹ * Γ₁₂)) + toRMat ((M_L⁻¹ * U₁) * Matrix.of uv.2)
    have hslice_eq : (Prod.mk c11 ·) ⁻¹' ((Prod.mk · uv) ⁻¹' (T₃ ⁻¹' s)) =
        (Prod.map (fun x => x + sh) (id : RMat d' q → _)) ⁻¹'
          ((Prod.mk c11 ·) ⁻¹' ((Prod.mk · uv) ⁻¹' s)) := by
      ext ⟨c12, b21⟩
      simp only [Set.mem_preimage, Prod.map_apply, id_eq, T₃, sh, sub_eq_add_neg, add_assoc]
    rw [hslice_eq]
    have hmp := (measurePreserving_add_right (volume : Measure (RMat q d')) sh).prod
      (MeasurePreserving.id (volume : Measure (RMat d' q)))
    exact hmp.measure_preimage
      ((hs.preimage measurable_prodMk_right).preimage measurable_prodMk_left).nullMeasurableSet
  -- Step 4: `T₄` scales volume by `|det M_R|^q`
  have hMRinv_det : (M_R⁻¹).det ≠ 0 := by
    rw [Matrix.det_nonsing_inv, Ring.inverse_eq_inv]; exact inv_ne_zero hMR
  have hT4 : Measure.map T₄ volume = ENNReal.ofReal |M_R.det| ^ q • volume := by
    have hm := map_mul_right_volume (m := q) M_R⁻¹ hMRinv_det
    rw [Matrix.det_nonsing_inv, Ring.inverse_eq_inv, inv_inv] at hm
    have : SigmaFinite (ENNReal.ofReal |M_R.det| ^ q • (volume : Measure (RMat q q))) :=
      sigmaFinite_ofReal_pow_smul volume |M_R.det| q
    have hsf1 : SigmaFinite (Measure.map (fun X : RMat q q => toRMat (Matrix.of X * M_R⁻¹)) volume) := by
      rw [hm]; infer_instance
    have hinner : Measure.map (Prod.map (fun X : RMat q q => toRMat (Matrix.of X * M_R⁻¹))
        (id : RMat q d' × RMat d' q → _)) (volume.prod volume) =
        ENNReal.ofReal |M_R.det| ^ q • (volume.prod volume) := by
      rw [← Measure.map_prod_map _ _ (hmeas_mulR M_R⁻¹) measurable_id, hm, Measure.map_id,
        smul_prod_measure]
    have : SigmaFinite (ENNReal.ofReal |M_R.det| ^ q •
        ((volume : Measure (RMat q q)).prod (volume : Measure (RMat q d' × RMat d' q)))) :=
      sigmaFinite_ofReal_pow_smul (volume.prod volume) |M_R.det| q
    have hsf2 : SigmaFinite (Measure.map (Prod.map (fun X : RMat q q => toRMat (Matrix.of X * M_R⁻¹))
        (id : RMat q d' × RMat d' q → _)) volume) := by
      change SigmaFinite (Measure.map _ (volume.prod volume))
      rw [hinner]
      infer_instance
    change Measure.map (Prod.map (Prod.map (fun X : RMat q q => toRMat (Matrix.of X * M_R⁻¹)) id) id)
      (volume.prod volume) = _
    rw [← Measure.map_prod_map _ _ ((hmeas_mulR M_R⁻¹).prodMap measurable_id) measurable_id]
    change (Measure.map (Prod.map (fun X : RMat q q => toRMat (Matrix.of X * M_R⁻¹)) id)
      (volume.prod volume)).prod (Measure.map id volume) = _
    rw [hinner, Measure.map_id, smul_prod_measure]
    rfl
  -- Step 5: `T₅` preserves volume
  have hT5 : Measure.map T₅ volume = volume :=
    (((measurePreserving_add_right volume shC).prod (MeasurePreserving.id volume)).prod
      (MeasurePreserving.id volume)).map_eq
  rw [hcomp,
    ← Measure.map_map hT5_m (hT4_m.comp (hT3_m.comp (hT2_m.comp hT1_m))),
    ← Measure.map_map hT4_m (hT3_m.comp (hT2_m.comp hT1_m)),
    ← Measure.map_map hT3_m (hT2_m.comp hT1_m),
    ← Measure.map_map hT2_m hT1_m,
    hT1, hT2, hT3, hT4, Measure.map_smul, hT5]
  · exact hT5_m.aemeasurable

/-- Uniform good radius for the two-sided Schur complement on `TwoSidedPassive × TwoSidedActive`. -/
private theorem exists_good_twoSided_radius (q d' k' : ℕ) :
    ∃ δ₀ : ℝ, 0 < δ₀ ∧ ∀ δ : ℝ, 0 < δ → δ ≤ δ₀ →
      (∀ p ∈ passiveBox (q := q) (d' := d') (k' := k') δ,
        let M_L := (1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of p.1.1.1
        let M_R := (1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of p.1.2.1
        IsUnit M_L.det ∧ IsUnit M_R.det ∧
        (1 / 2 : ℝ) ≤ |M_R.det| ∧ |M_R.det| ≤ 2 ∧
        frobSq (Matrix.of p.1.1.1) ≤ 1 / 16 ∧ frobSq (Matrix.of p.1.1.2) ≤ 1 / 16 ∧
        frobSq (Matrix.of p.1.2.1) ≤ 1 / 16 ∧ frobSq (Matrix.of p.1.2.2) ≤ 1 / 16 ∧
        frobSq (M_L⁻¹ - 1) ≤ 1 / 16 ∧ frobSq (M_R⁻¹ - 1) ≤ 1 / 16 ∧
        (∀ a ∈ activeBox (q := q) (d' := d') (k' := k') δ,
          twoSidedSchurMap p a ∈ activeBox (4 * δ))) ∧
      (∀ p ∈ passiveBox (q := q) (d' := d') (k' := k') (δ / 8),
        ∀ a' ∈ activeBox (q := q) (d' := d') (k' := k') (δ / 4),
          ∃ a ∈ activeBox (q := q) (d' := d') (k' := k') δ, twoSidedSchurMap p a = a') := by
  -- Use `L^∞` row-sum / matrix-norm continuity at `0` on `Fin q × Fin q`.
  have hmat_inv_nhds : ∀ᶠ Δ in nhds (0 : RMat q q),
      let M := (1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ
      (1 / 2 : ℝ) < |M.det| ∧ |M.det| < 2 ∧
      frobSq (Matrix.of Δ) < 1 / 16 ∧ frobSq (M⁻¹ - 1) < 1 / 16 ∧
      (∀ i : Fin q, ∑ j : Fin q, |M i j| < 5 / 4) ∧
      (∀ j : Fin q, ∑ i : Fin q, |M i j| < 5 / 4) ∧
      (∀ i : Fin q, ∑ j : Fin q, |M⁻¹ i j| < 5 / 4) ∧
      (∀ j : Fin q, ∑ i : Fin q, |M⁻¹ i j| < 5 / 4) := by
    have hM_cont : Continuous (fun Δ : RMat q q => (1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ) :=
      continuous_const.add continuous_id
    have hM0 : (1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of (0 : RMat q q) = 1 := by
      ext i j; simp
    have hinv_cont : ContinuousAt (fun Δ : RMat q q =>
        ((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ)⁻¹) 0 :=
      continuousAt_matrix_inv_of_one
        (fun Δ : RMat q q => (1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ) hM_cont hM0
    have h1 : ∀ᶠ Δ in nhds (0 : RMat q q),
        (1 / 2 : ℝ) < |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ).det| ∧
        |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ).det| < 2 := by
      have hc : ContinuousAt (fun Δ : RMat q q => |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ).det|) 0 :=
        hM_cont.matrix_det.continuousAt.abs
      have hmem : Set.Ioo (1 / 2 : ℝ) 2 ∈ nhds |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of (0 : RMat q q)).det| := by
        rw [hM0, Matrix.det_one, abs_one]; exact Ioo_mem_nhds (by norm_num) (by norm_num)
      exact hc.preimage_mem_nhds hmem
    have h2 : ∀ᶠ Δ in nhds (0 : RMat q q), frobSq (Matrix.of Δ) < 1 / 16 := by
      have hc : ContinuousAt (fun Δ : RMat q q => frobSq (Matrix.of Δ)) 0 :=
        (continuous_finsetSum _ fun i _ => continuous_finsetSum _ fun j _ =>
          (((continuous_apply j).comp (continuous_apply i)).pow 2)).continuousAt
      have hv : frobSq (Matrix.of (0 : RMat q q)) = 0 := by unfold frobSq; simp
      have hmem : Set.Iio (1 / 16 : ℝ) ∈ nhds (frobSq (Matrix.of (0 : RMat q q))) := by
        rw [hv]; exact Iio_mem_nhds (by norm_num)
      exact hc.preimage_mem_nhds hmem
    have h3 : ∀ᶠ Δ in nhds (0 : RMat q q),
        frobSq (((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ)⁻¹ - 1) < 1 / 16 := by
      have hc1 : ContinuousAt (fun X : Matrix (Fin q) (Fin q) ℝ => frobSq (X - 1))
          ((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of (0 : RMat q q))⁻¹ := by
        have hsub : (fun X : Matrix (Fin q) (Fin q) ℝ => frobSq (X - 1)) =
            fun X => ∑ i : Fin q, ∑ j : Fin q, (X i j - (1 : Matrix (Fin q) (Fin q) ℝ) i j) ^ 2 := by
          funext X; rfl
        rw [hsub]
        refine (continuous_finsetSum _ fun i _ => continuous_finsetSum _ fun j _ => ?_).continuousAt
        have hcoord : Continuous (fun X : Matrix (Fin q) (Fin q) ℝ => X i j) :=
          (continuous_apply j).comp (continuous_apply i)
        exact (hcoord.sub continuous_const).pow 2
      have hc : ContinuousAt (fun Δ : RMat q q =>
          frobSq (((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ)⁻¹ - 1)) 0 :=
        ContinuousAt.comp (f := fun Δ : RMat q q => ((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ)⁻¹)
          hc1 hinv_cont
      have hv : frobSq (((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of (0 : RMat q q))⁻¹ - 1) = 0 := by
        rw [hM0]; simp [frobSq]
      have hmem : Set.Iio (1 / 16 : ℝ) ∈ nhds (frobSq (((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of (0 : RMat q q))⁻¹ - 1)) := by
        rw [hv]; exact Iio_mem_nhds (by norm_num)
      exact hc.preimage_mem_nhds hmem
    have h4 : ∀ᶠ Δ in nhds (0 : RMat q q),
        ∀ i : Fin q, ∑ j : Fin q, |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ) i j| < 5 / 4 := by
      rw [Filter.eventually_all]; intro i
      have hc : ContinuousAt (fun Δ : RMat q q =>
          ∑ j : Fin q, |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ) i j|) 0 :=
        (continuous_finsetSum _ fun j _ => ((continuous_apply j).comp
          ((continuous_apply i).comp hM_cont)).abs).continuousAt
      have hv : ∑ j : Fin q, |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of (0 : RMat q q)) i j| = 1 := by
        rw [hM0]; simp [Matrix.one_apply, apply_ite]
      have hmem : Set.Iio (5 / 4 : ℝ) ∈ nhds (∑ j : Fin q, |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of (0 : RMat q q)) i j|) := by
        rw [hv]; exact Iio_mem_nhds (by norm_num)
      exact hc.preimage_mem_nhds hmem
    have h5 : ∀ᶠ Δ in nhds (0 : RMat q q),
        ∀ j : Fin q, ∑ i : Fin q, |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ) i j| < 5 / 4 := by
      rw [Filter.eventually_all]; intro j
      have hc : ContinuousAt (fun Δ : RMat q q =>
          ∑ i : Fin q, |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ) i j|) 0 :=
        (continuous_finsetSum _ fun i _ => ((continuous_apply j).comp
          ((continuous_apply i).comp hM_cont)).abs).continuousAt
      have hv : ∑ i : Fin q, |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of (0 : RMat q q)) i j| = 1 := by
        rw [hM0]; simp [Matrix.one_apply, apply_ite]
      have hmem : Set.Iio (5 / 4 : ℝ) ∈ nhds (∑ i : Fin q, |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of (0 : RMat q q)) i j|) := by
        rw [hv]; exact Iio_mem_nhds (by norm_num)
      exact hc.preimage_mem_nhds hmem
    have h6 : ∀ᶠ Δ in nhds (0 : RMat q q),
        ∀ i : Fin q, ∑ j : Fin q, |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ)⁻¹ i j| < 5 / 4 := by
      rw [Filter.eventually_all]; intro i
      have hc1 : ContinuousAt (fun X : Matrix (Fin q) (Fin q) ℝ => ∑ j : Fin q, |X i j|)
          ((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of (0 : RMat q q))⁻¹ :=
        (continuous_finsetSum _ fun j _ => ((continuous_apply j).comp
          ((continuous_apply i).comp continuous_id)).abs).continuousAt
      have hc : ContinuousAt (fun Δ : RMat q q =>
          ∑ j : Fin q, |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ)⁻¹ i j|) 0 :=
        ContinuousAt.comp (f := fun Δ : RMat q q => ((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ)⁻¹)
          hc1 hinv_cont
      have hv : ∑ j : Fin q, |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of (0 : RMat q q))⁻¹ i j| = 1 := by
        rw [hM0]; simp [Matrix.one_apply, apply_ite]
      have hmem : Set.Iio (5 / 4 : ℝ) ∈ nhds (∑ j : Fin q, |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of (0 : RMat q q))⁻¹ i j|) := by
        rw [hv]; exact Iio_mem_nhds (by norm_num)
      exact hc.preimage_mem_nhds hmem
    have h7 : ∀ᶠ Δ in nhds (0 : RMat q q),
        ∀ j : Fin q, ∑ i : Fin q, |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ)⁻¹ i j| < 5 / 4 := by
      rw [Filter.eventually_all]; intro j
      have hc1 : ContinuousAt (fun X : Matrix (Fin q) (Fin q) ℝ => ∑ i : Fin q, |X i j|)
          ((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of (0 : RMat q q))⁻¹ :=
        (continuous_finsetSum _ fun i _ => ((continuous_apply j).comp
          ((continuous_apply i).comp continuous_id)).abs).continuousAt
      have hc : ContinuousAt (fun Δ : RMat q q =>
          ∑ i : Fin q, |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ)⁻¹ i j|) 0 :=
        ContinuousAt.comp (f := fun Δ : RMat q q => ((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of Δ)⁻¹)
          hc1 hinv_cont
      have hv : ∑ i : Fin q, |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of (0 : RMat q q))⁻¹ i j| = 1 := by
        rw [hM0]; simp [Matrix.one_apply, apply_ite]
      have hmem : Set.Iio (5 / 4 : ℝ) ∈ nhds (∑ i : Fin q, |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of (0 : RMat q q))⁻¹ i j|) := by
        rw [hv]; exact Iio_mem_nhds (by norm_num)
      exact hc.preimage_mem_nhds hmem
    filter_upwards [h1, h2, h3, h4, h5, h6, h7] with Δ ⟨a, b⟩ c d e f g h
    exact ⟨a, b, c, d, e, f, g, h⟩
  obtain ⟨δ₁, hδ₁, hsub1⟩ := matBox_nhds hmat_inv_nhds
  -- Choose `δ₀ ≤ δ₁` small enough that all quadratic/cubic cross terms are `≤ (1/8) * δ`.
  let D : ℝ := max (max (q : ℝ) (d' : ℝ)) (max (k' : ℝ) 1)
  have hD1 : 1 ≤ D := le_trans (le_max_right _ _) (le_max_right _ _)
  have hDq : (q : ℝ) ≤ D := le_trans (le_max_left _ _) (le_max_left _ _)
  have hDd' : (d' : ℝ) ≤ D := le_trans (le_max_right _ _) (le_max_left _ _)
  have hDk' : (k' : ℝ) ≤ D := le_trans (le_max_left _ _) (le_max_right _ _)
  let δ₀ : ℝ := min δ₁ (1 / (64 * D ^ 2))
  have hδ₀_pos : 0 < δ₀ := lt_min hδ₁ (by positivity)
  -- Helper entrywise product bounds on boxes
  have hmul_bound : ∀ {a b c : ℕ} {X : Matrix (Fin a) (Fin b) ℝ} {Y : Matrix (Fin b) (Fin c) ℝ}
      {rX rY : ℝ}, 0 ≤ rX → 0 ≤ rY →
      (∀ i r, |X i r| ≤ rX) → (∀ r j, |Y r j| ≤ rY) →
      ∀ i j, |(X * Y) i j| ≤ (b : ℝ) * rX * rY := by
    intro a b c X Y rX rY hrX hrY hX hY i j
    simp only [Matrix.mul_apply]
    calc |∑ r, X i r * Y r j| ≤ ∑ r, |X i r * Y r j| := Finset.abs_sum_le_sum_abs _ _
      _ = ∑ r, |X i r| * |Y r j| := by simp_rw [abs_mul]
      _ ≤ ∑ _r, rX * rY := Finset.sum_le_sum fun r _ =>
          mul_le_mul (hX i r) (hY r j) (abs_nonneg _) hrX
      _ = _ := by simp [mul_assoc]
  have hrow_mul_bound : ∀ {a c : ℕ} {M : Matrix (Fin a) (Fin a) ℝ} {Y : Matrix (Fin a) (Fin c) ℝ}
      {K rY : ℝ}, 0 ≤ rY → (∀ i, ∑ r, |M i r| ≤ K) → (∀ r j, |Y r j| ≤ rY) →
      ∀ i j, |(M * Y) i j| ≤ K * rY := by
    intro a c M Y K rY hrY hM hY i j
    simp only [Matrix.mul_apply]
    calc |∑ r, M i r * Y r j| ≤ ∑ r, |M i r * Y r j| := Finset.abs_sum_le_sum_abs _ _
      _ = ∑ r, |M i r| * |Y r j| := by simp_rw [abs_mul]
      _ ≤ ∑ r, |M i r| * rY := Finset.sum_le_sum fun r _ =>
          mul_le_mul_of_nonneg_left (hY r j) (abs_nonneg _)
      _ = (∑ r, |M i r|) * rY := (Finset.sum_mul _ _ _).symm
      _ ≤ K * rY := mul_le_mul_of_nonneg_right (hM i) hrY
  have hcol_mul_bound : ∀ {a c : ℕ} {X : Matrix (Fin a) (Fin c) ℝ} {M : Matrix (Fin c) (Fin c) ℝ}
      {rX K : ℝ}, 0 ≤ rX → (∀ i r, |X i r| ≤ rX) → (∀ j, ∑ r, |M r j| ≤ K) →
      ∀ i j, |(X * M) i j| ≤ rX * K := by
    intro a c X M rX K hrX hX hM i j
    simp only [Matrix.mul_apply]
    calc |∑ r, X i r * M r j| ≤ ∑ r, |X i r * M r j| := Finset.abs_sum_le_sum_abs _ _
      _ = ∑ r, |X i r| * |M r j| := by simp_rw [abs_mul]
      _ ≤ ∑ r, rX * |M r j| := Finset.sum_le_sum fun r _ =>
          mul_le_mul_of_nonneg_right (hX i r) (abs_nonneg _)
      _ = rX * ∑ r, |M r j| := (Finset.mul_sum _ _ _).symm
      _ ≤ rX * K := mul_le_mul_of_nonneg_left (hM j) hrX
  refine ⟨δ₀, hδ₀_pos, fun δ hδ hδle => ?_⟩
  have hδ_le1 : δ ≤ δ₁ := hδle.trans (min_le_left _ _)
  have hδ_small : δ ≤ 1 / (64 * D ^ 2) := hδle.trans (min_le_right _ _)
  have hDδ : D * δ ≤ 1 / 64 := by
    have hD2 : D ≤ D ^ 2 := by nlinarith
    have h1 : D ^ 2 * δ ≤ 1 / 64 := by
      have := mul_le_mul_of_nonneg_left hδ_small (by positivity : (0 : ℝ) ≤ D ^ 2)
      have heq : D ^ 2 * (1 / (64 * D ^ 2)) = 1 / 64 := by field_simp
      linarith
    nlinarith
  have hD2δ : D ^ 2 * δ ≤ 1 / 64 := by
    have := mul_le_mul_of_nonneg_left hδ_small (by positivity : (0 : ℝ) ≤ D ^ 2)
    have heq : D ^ 2 * (1 / (64 * D ^ 2)) = 1 / 64 := by field_simp
    linarith
  have hqδ : (q : ℝ) * δ ≤ 1 / 64 := (mul_le_mul_of_nonneg_right hDq hδ.le).trans hDδ
  have hkδ : (k' : ℝ) * δ ≤ 1 / 64 := (mul_le_mul_of_nonneg_right hDk' hδ.le).trans hDδ
  constructor
  · -- Part 1: forward properties for `p ∈ passiveBox δ`
    intro p hp
    rcases hp with ⟨⟨⟨hΔ11, hΔ21⟩, ⟨hΓ11, hΓ12⟩⟩, ⟨hB11, hU1, hV1⟩⟩
    obtain ⟨hML_lo, _, hΔ11_f, hMLinv_f, hML_r, hML_c, hMLinv_r, hMLinv_c⟩ :=
      hsub1 δ hδ hδ_le1 hΔ11
    obtain ⟨hMR_lo, hMR_hi, hΓ11_f, hMRinv_f, hMR_r, hMR_c, hMRinv_r, hMRinv_c⟩ :=
      hsub1 δ hδ hδ_le1 hΓ11
    have hML_unit : IsUnit ((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of p.1.1.1).det :=
      isUnit_iff_ne_zero.mpr fun h0 => by rw [h0, abs_zero] at hML_lo; linarith
    have hMR_unit : IsUnit ((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of p.1.2.1).det :=
      isUnit_iff_ne_zero.mpr fun h0 => by rw [h0, abs_zero] at hMR_lo; linarith
    have hΔ21_f : frobSq (Matrix.of p.1.1.2) ≤ 1 / 16 := by
      unfold frobSq
      calc ∑ i : Fin d', ∑ j : Fin q, (p.1.1.2 i j) ^ 2
          ≤ ∑ _i : Fin d', ∑ _j : Fin q, δ ^ 2 := Finset.sum_le_sum fun i _ =>
            Finset.sum_le_sum fun j _ => by
              rw [← sq_abs, sq, sq]; exact mul_le_mul (mem_matBox_iff.mp hΔ21 i j)
                (mem_matBox_iff.mp hΔ21 i j) (abs_nonneg _) hδ.le
        _ = (d' : ℝ) * (q : ℝ) * δ ^ 2 := by simp [mul_assoc]
        _ ≤ D ^ 2 * δ * δ := by
            have : (d' : ℝ) * (q : ℝ) ≤ D ^ 2 := by
              rw [sq]; exact mul_le_mul hDd' hDq (Nat.cast_nonneg _) (by linarith)
            nlinarith [sq_nonneg δ]
        _ ≤ 1 / 16 := by nlinarith
    have hΓ12_f : frobSq (Matrix.of p.1.2.2) ≤ 1 / 16 := by
      unfold frobSq
      calc ∑ i : Fin q, ∑ j : Fin d', (p.1.2.2 i j) ^ 2
          ≤ ∑ _i : Fin q, ∑ _j : Fin d', δ ^ 2 := Finset.sum_le_sum fun i _ =>
            Finset.sum_le_sum fun j _ => by
              rw [← sq_abs, sq, sq]; exact mul_le_mul (mem_matBox_iff.mp hΓ12 i j)
                (mem_matBox_iff.mp hΓ12 i j) (abs_nonneg _) hδ.le
        _ = (q : ℝ) * (d' : ℝ) * δ ^ 2 := by simp [mul_assoc]
        _ ≤ D ^ 2 * δ * δ := by
            have : (q : ℝ) * (d' : ℝ) ≤ D ^ 2 := by
              rw [sq]; exact mul_le_mul hDq hDd' (Nat.cast_nonneg _) (by linarith)
            nlinarith [sq_nonneg δ]
        _ ≤ 1 / 16 := by nlinarith
    refine ⟨hML_unit, hMR_unit, hMR_lo.le, hMR_hi.le, hΔ11_f.le, hΔ21_f, hΓ11_f.le, hΓ12_f,
      hMLinv_f.le, hMRinv_f.le, ?_⟩
    intro a ha
    rcases ha with ⟨⟨hC11, hC12, hB21⟩, ⟨hU2, hV2⟩⟩
    set M_L := (1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of p.1.1.1
    set M_R := (1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of p.1.2.1
    have hMLinv_U1 : ∀ i j, |(M_L⁻¹ * Matrix.of p.2.2.1) i j| ≤ (5 / 4 : ℝ) * δ :=
      hrow_mul_bound hδ.le (fun i => (hMLinv_r i).le) (mem_matBox_iff.mp hU1)
    have hMLinv_B11 : ∀ i j, |(M_L⁻¹ * Matrix.of p.2.1) i j| ≤ (5 / 4 : ℝ) * δ :=
      hrow_mul_bound hδ.le (fun i => (hMLinv_r i).le) (mem_matBox_iff.mp hB11)
    have hV1_MRinv : ∀ i j, |(Matrix.of p.2.2.2 * M_R⁻¹) i j| ≤ δ * (5 / 4 : ℝ) :=
      hcol_mul_bound hδ.le (mem_matBox_iff.mp hV1) (fun j => (hMRinv_c j).le)
    have hMRinv_Γ12 : ∀ i j, |(M_R⁻¹ * Matrix.of p.1.2.2) i j| ≤ (5 / 4 : ℝ) * δ :=
      hrow_mul_bound hδ.le (fun i => (hMRinv_r i).le) (mem_matBox_iff.mp hΓ12)
    have hU2'_b : ∀ i j, |(Matrix.of a.2.1 - Matrix.of p.1.1.2 * M_L⁻¹ * Matrix.of p.2.2.1) i j| ≤ 2 * δ := by
      intro i j
      rw [Matrix.sub_apply, Matrix.of_apply, Matrix.mul_assoc]
      have h1 := mem_matBox_iff.mp hU2 i j
      have h2 := hmul_bound (X := Matrix.of p.1.1.2) hδ.le (by positivity) (mem_matBox_iff.mp hΔ21) hMLinv_U1 i j
      have h3 : (q : ℝ) * δ * ((5 / 4 : ℝ) * δ) ≤ δ := by nlinarith only [hqδ, hδ]
      exact (abs_sub _ _).trans (by linarith only [h1, h2, h3])
    have hV2'_b : ∀ i j, |(Matrix.of a.2.2 - Matrix.of p.2.2.2 * M_R⁻¹ * Matrix.of p.1.2.2) i j| ≤ 2 * δ := by
      intro i j
      rw [Matrix.sub_apply, Matrix.of_apply, Matrix.mul_assoc]
      have h1 := mem_matBox_iff.mp hV2 i j
      have h2 := hmul_bound (X := Matrix.of p.2.2.2) hδ.le (by positivity) (mem_matBox_iff.mp hV1) hMRinv_Γ12 i j
      have h3 : (q : ℝ) * δ * ((5 / 4 : ℝ) * δ) ≤ δ := by nlinarith only [hqδ, hδ]
      exact (abs_sub _ _).trans (by linarith only [h1, h2, h3])
    refine ⟨⟨?_, ?_, ?_⟩, ?_, ?_⟩
    · -- Z11 ∈ matBox (4δ)
      rw [mem_matBox_iff]; intro i j
      change |((Matrix.of a.1.1 * M_R⁻¹) i j + (M_L⁻¹ * Matrix.of p.2.1) i j) +
          (M_L⁻¹ * Matrix.of p.2.2.1 * Matrix.of p.2.2.2 * M_R⁻¹) i j| ≤ 4 * δ
      have h1 := hcol_mul_bound (X := Matrix.of a.1.1) hδ.le (mem_matBox_iff.mp hC11) (fun j' => (hMRinv_c j').le) i j
      have h2 := hMLinv_B11 i j
      have h3 : |(M_L⁻¹ * Matrix.of p.2.2.1 * Matrix.of p.2.2.2 * M_R⁻¹) i j| ≤ δ := by
        rw [Matrix.mul_assoc (M_L⁻¹ * Matrix.of p.2.2.1)]
        have h3a := hmul_bound (by positivity) (by positivity) hMLinv_U1 hV1_MRinv i j
        have hbd : (k' : ℝ) * ((5 / 4 : ℝ) * δ) * (δ * (5 / 4 : ℝ)) ≤ δ := by nlinarith only [hkδ, hδ]
        linarith only [h3a, hbd]
      have ht1 := abs_add_le ((Matrix.of a.1.1 * M_R⁻¹) i j + (M_L⁻¹ * Matrix.of p.2.1) i j)
        ((M_L⁻¹ * Matrix.of p.2.2.1 * Matrix.of p.2.2.2 * M_R⁻¹) i j)
      have ht2 := abs_add_le ((Matrix.of a.1.1 * M_R⁻¹) i j) ((M_L⁻¹ * Matrix.of p.2.1) i j)
      linarith only [h1, h2, h3, ht1, ht2, hδ]
    · -- Z12 ∈ matBox (4δ)
      rw [mem_matBox_iff]; intro i j
      change |(a.1.2.1 i j - (Matrix.of a.1.1 * (M_R⁻¹ * Matrix.of p.1.2.2)) i j) +
          ((M_L⁻¹ * Matrix.of p.2.2.1) * (Matrix.of a.2.2 - Matrix.of p.2.2.2 * M_R⁻¹ * Matrix.of p.1.2.2)) i j| ≤ 4 * δ
      have h1 := mem_matBox_iff.mp hC12 i j
      have h2 := hmul_bound (X := Matrix.of a.1.1) hδ.le (by positivity) (mem_matBox_iff.mp hC11) hMRinv_Γ12 i j
      have h3 := hmul_bound (by positivity) (by positivity) hMLinv_U1 hV2'_b i j
      have h2' : (q : ℝ) * δ * ((5 / 4 : ℝ) * δ) ≤ δ := by nlinarith only [hqδ, hδ]
      have h3' : (k' : ℝ) * ((5 / 4 : ℝ) * δ) * (2 * δ) ≤ δ := by nlinarith only [hkδ, hδ]
      have ht1 := abs_add_le (a.1.2.1 i j - (Matrix.of a.1.1 * (M_R⁻¹ * Matrix.of p.1.2.2)) i j)
        (((M_L⁻¹ * Matrix.of p.2.2.1) * (Matrix.of a.2.2 - Matrix.of p.2.2.2 * M_R⁻¹ * Matrix.of p.1.2.2)) i j)
      have ht2 := abs_sub (a.1.2.1 i j) ((Matrix.of a.1.1 * (M_R⁻¹ * Matrix.of p.1.2.2)) i j)
      linarith only [h1, h2, h3, h2', h3', ht1, ht2, hδ]
    · -- Z21 ∈ matBox (4δ)
      rw [mem_matBox_iff]; intro i j
      change |(a.1.2.2 i j - (Matrix.of p.1.1.2 * M_L⁻¹ * Matrix.of p.2.1) i j) +
          ((Matrix.of a.2.1 - Matrix.of p.1.1.2 * M_L⁻¹ * Matrix.of p.2.2.1) * (Matrix.of p.2.2.2 * M_R⁻¹)) i j| ≤ 4 * δ
      have h1 := mem_matBox_iff.mp hB21 i j
      have h2 : |(Matrix.of p.1.1.2 * M_L⁻¹ * Matrix.of p.2.1) i j| ≤ δ := by
        rw [Matrix.mul_assoc]
        have h2a := hmul_bound (X := Matrix.of p.1.1.2) hδ.le (by positivity) (mem_matBox_iff.mp hΔ21) hMLinv_B11 i j
        have h2b : (q : ℝ) * δ * ((5 / 4 : ℝ) * δ) ≤ δ := by nlinarith only [hqδ, hδ]
        linarith only [h2a, h2b]
      have h3 := hmul_bound (by positivity) (by positivity) hU2'_b hV1_MRinv i j
      have h3' : (k' : ℝ) * (2 * δ) * (δ * (5 / 4 : ℝ)) ≤ δ := by nlinarith only [hkδ, hδ]
      have ht1 := abs_add_le (a.1.2.2 i j - (Matrix.of p.1.1.2 * M_L⁻¹ * Matrix.of p.2.1) i j)
        (((Matrix.of a.2.1 - Matrix.of p.1.1.2 * M_L⁻¹ * Matrix.of p.2.2.1) * (Matrix.of p.2.2.2 * M_R⁻¹)) i j)
      have ht2 := abs_sub (a.1.2.2 i j) ((Matrix.of p.1.1.2 * M_L⁻¹ * Matrix.of p.2.1) i j)
      linarith only [h1, h2, h3, h3', ht1, ht2, hδ]
    · rw [mem_matBox_iff]; intro i j
      simpa [twoSidedSchurMap, toRMat_apply] using (hU2'_b i j).trans (by linarith only [hδ])
    · rw [mem_matBox_iff]; intro i j
      simpa [twoSidedSchurMap, toRMat_apply] using (hV2'_b i j).trans (by linarith only [hδ])
  · -- Part 2: surjectivity onto `activeBox (δ / 4)` for `p ∈ passiveBox (δ / 8)`
    intro p hp a' ha'
    rcases hp with ⟨⟨⟨hΔ11, hΔ21⟩, ⟨hΓ11, hΓ12⟩⟩, ⟨hB11, hU1, hV1⟩⟩
    rcases ha' with ⟨⟨hZ11, hZ12, hZ21⟩, ⟨hU2', hV2'⟩⟩
    have hδ8_pos : 0 < δ / 8 := by linarith only [hδ]
    have hδ8_le1 : δ / 8 ≤ δ₁ := by linarith only [hδ, hδ_le1]
    obtain ⟨hML_lo, _, _, _, _, _, hMLinv_r, _⟩ := hsub1 (δ / 8) hδ8_pos hδ8_le1 hΔ11
    obtain ⟨hMR_lo, _, _, _, _, hMR_c, hMRinv_r, hMRinv_c⟩ := hsub1 (δ / 8) hδ8_pos hδ8_le1 hΓ11
    have hMR_unit : IsUnit ((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of p.1.2.1).det :=
      isUnit_iff_ne_zero.mpr fun h0 => by rw [h0, abs_zero] at hMR_lo; linarith only [hMR_lo]
    set M_L := (1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of p.1.1.1
    set M_R := (1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of p.1.2.1
    set U₂ := Matrix.of a'.2.1 + Matrix.of p.1.1.2 * M_L⁻¹ * Matrix.of p.2.2.1
    set V₂ := Matrix.of a'.2.2 + Matrix.of p.2.2.2 * M_R⁻¹ * Matrix.of p.1.2.2
    set B₂₁ := Matrix.of a'.1.2.2 + Matrix.of p.1.1.2 * M_L⁻¹ * Matrix.of p.2.1 -
      Matrix.of a'.2.1 * (Matrix.of p.2.2.2 * M_R⁻¹)
    set C₁₁ := (Matrix.of a'.1.1 - M_L⁻¹ * Matrix.of p.2.1 -
      M_L⁻¹ * Matrix.of p.2.2.1 * Matrix.of p.2.2.2 * M_R⁻¹) * M_R
    set C₁₂ := Matrix.of a'.1.2.1 + C₁₁ * (M_R⁻¹ * Matrix.of p.1.2.2) -
      (M_L⁻¹ * Matrix.of p.2.2.1) * Matrix.of a'.2.2
    refine ⟨((toRMat C₁₁, toRMat C₁₂, toRMat B₂₁), (toRMat U₂, toRMat V₂)), ?_, ?_⟩
    · have hMLinv_U1 : ∀ i j, |(M_L⁻¹ * Matrix.of p.2.2.1) i j| ≤ (5 / 4 : ℝ) * (δ / 8) :=
        hrow_mul_bound hδ8_pos.le (fun i => (hMLinv_r i).le) (mem_matBox_iff.mp hU1)
      have hMLinv_B11 : ∀ i j, |(M_L⁻¹ * Matrix.of p.2.1) i j| ≤ (5 / 4 : ℝ) * (δ / 8) :=
        hrow_mul_bound hδ8_pos.le (fun i => (hMLinv_r i).le) (mem_matBox_iff.mp hB11)
      have hV1_MRinv : ∀ i j, |(Matrix.of p.2.2.2 * M_R⁻¹) i j| ≤ (δ / 8) * (5 / 4 : ℝ) :=
        hcol_mul_bound hδ8_pos.le (mem_matBox_iff.mp hV1) (fun j => (hMRinv_c j).le)
      have hMRinv_Γ12 : ∀ i j, |(M_R⁻¹ * Matrix.of p.1.2.2) i j| ≤ (5 / 4 : ℝ) * (δ / 8) :=
        hrow_mul_bound hδ8_pos.le (fun i => (hMRinv_r i).le) (mem_matBox_iff.mp hΓ12)
      have hC11_b : ∀ i j, |C₁₁ i j| ≤ δ := by
        have hpre : ∀ i j, |(Matrix.of a'.1.1 - M_L⁻¹ * Matrix.of p.2.1 -
            M_L⁻¹ * Matrix.of p.2.2.1 * Matrix.of p.2.2.2 * M_R⁻¹) i j| ≤ (3 / 4 : ℝ) * δ := by
          intro i j
          simp only [Matrix.sub_apply, Matrix.of_apply]
          have h1 := mem_matBox_iff.mp hZ11 i j
          have h2 := hMLinv_B11 i j
          have h3 : |(M_L⁻¹ * Matrix.of p.2.2.1 * Matrix.of p.2.2.2 * M_R⁻¹) i j| ≤ δ / 8 := by
            rw [Matrix.mul_assoc (M_L⁻¹ * Matrix.of p.2.2.1)]
            have h3a := hmul_bound (by positivity) (by positivity) hMLinv_U1 hV1_MRinv i j
            have h3b : (k' : ℝ) * ((5 / 4 : ℝ) * (δ / 8)) * ((δ / 8) * (5 / 4 : ℝ)) ≤ δ / 8 := by
              nlinarith only [hkδ, hδ]
            linarith only [h3a, h3b]
          have ht1 := abs_sub (a'.1.1 i j - (M_L⁻¹ * Matrix.of p.2.1) i j)
            ((M_L⁻¹ * Matrix.of p.2.2.1 * Matrix.of p.2.2.2 * M_R⁻¹) i j)
          have ht2 := abs_sub (a'.1.1 i j) ((M_L⁻¹ * Matrix.of p.2.1) i j)
          linarith only [h1, h2, h3, ht1, ht2, hδ]
        intro i j
        have hcb := hcol_mul_bound (by positivity) hpre (fun j' => (hMR_c j').le) i j
        linarith only [hcb, hδ]
      refine ⟨⟨?_, ?_, ?_⟩, ?_, ?_⟩
      · rw [mem_matBox_iff]; exact hC11_b
      · rw [mem_matBox_iff]; intro i j
        simp only [toRMat_apply, C₁₂, Matrix.sub_apply, Matrix.add_apply, Matrix.of_apply]
        have h1 := mem_matBox_iff.mp hZ12 i j
        have h2 := hmul_bound hδ.le (by positivity) hC11_b hMRinv_Γ12 i j
        have h3 := hmul_bound (Y := Matrix.of a'.2.2) (by positivity) (by linarith only [hδ]) hMLinv_U1 (mem_matBox_iff.mp hV2') i j
        have h2' : (q : ℝ) * δ * ((5 / 4 : ℝ) * (δ / 8)) ≤ δ / 4 := by nlinarith only [hqδ, hδ]
        have h3' : (k' : ℝ) * ((5 / 4 : ℝ) * (δ / 8)) * (δ / 4) ≤ δ / 4 := by nlinarith only [hkδ, hδ]
        have ht1 := abs_sub (a'.1.2.1 i j + (C₁₁ * (M_R⁻¹ * Matrix.of p.1.2.2)) i j)
          ((M_L⁻¹ * Matrix.of p.2.2.1 * Matrix.of a'.2.2) i j)
        have ht2 := abs_add_le (a'.1.2.1 i j) ((C₁₁ * (M_R⁻¹ * Matrix.of p.1.2.2)) i j)
        linarith only [h1, h2, h3, h2', h3', ht1, ht2, hδ]
      · rw [mem_matBox_iff]; intro i j
        simp only [toRMat_apply, B₂₁, Matrix.sub_apply, Matrix.add_apply, Matrix.of_apply]
        have h1 := mem_matBox_iff.mp hZ21 i j
        have h2 : |(Matrix.of p.1.1.2 * M_L⁻¹ * Matrix.of p.2.1) i j| ≤ δ / 4 := by
          rw [Matrix.mul_assoc]
          have h2a := hmul_bound (X := Matrix.of p.1.1.2) hδ8_pos.le (by positivity) (mem_matBox_iff.mp hΔ21) hMLinv_B11 i j
          have h2b : (q : ℝ) * (δ / 8) * ((5 / 4 : ℝ) * (δ / 8)) ≤ δ / 4 := by nlinarith only [hqδ, hδ]
          linarith only [h2a, h2b]
        have h3 := hmul_bound (X := Matrix.of a'.2.1) (by linarith only [hδ]) (by positivity) (mem_matBox_iff.mp hU2') hV1_MRinv i j
        have h3' : (k' : ℝ) * (δ / 4) * ((δ / 8) * (5 / 4 : ℝ)) ≤ δ / 4 := by nlinarith only [hkδ, hδ]
        have ht1 := abs_sub (a'.1.2.2 i j + (Matrix.of p.1.1.2 * M_L⁻¹ * Matrix.of p.2.1) i j)
          ((Matrix.of a'.2.1 * (Matrix.of p.2.2.2 * M_R⁻¹)) i j)
        have ht2 := abs_add_le (a'.1.2.2 i j) ((Matrix.of p.1.1.2 * M_L⁻¹ * Matrix.of p.2.1) i j)
        linarith only [h1, h2, h3, h3', ht1, ht2, hδ]
      · rw [mem_matBox_iff]; intro i j
        simp only [toRMat_apply, U₂, Matrix.add_apply, Matrix.of_apply]
        have h1 := mem_matBox_iff.mp hU2' i j
        have h2 : |(Matrix.of p.1.1.2 * M_L⁻¹ * Matrix.of p.2.2.1) i j| ≤ δ / 4 := by
          rw [Matrix.mul_assoc]
          have h2a := hmul_bound (X := Matrix.of p.1.1.2) hδ8_pos.le (by positivity) (mem_matBox_iff.mp hΔ21) hMLinv_U1 i j
          have h2b : (q : ℝ) * (δ / 8) * ((5 / 4 : ℝ) * (δ / 8)) ≤ δ / 4 := by nlinarith only [hqδ, hδ]
          linarith only [h2a, h2b]
        exact (abs_add_le _ _).trans (by linarith only [h1, h2, hδ])
      · rw [mem_matBox_iff]; intro i j
        simp only [toRMat_apply, V₂, Matrix.add_apply, Matrix.of_apply]
        have h1 := mem_matBox_iff.mp hV2' i j
        have h2 : |(Matrix.of p.2.2.2 * M_R⁻¹ * Matrix.of p.1.2.2) i j| ≤ δ / 4 := by
          rw [Matrix.mul_assoc]
          have h2a := hmul_bound (X := Matrix.of p.2.2.2) hδ8_pos.le (by positivity) (mem_matBox_iff.mp hV1) hMRinv_Γ12 i j
          have h2b : (q : ℝ) * (δ / 8) * ((5 / 4 : ℝ) * (δ / 8)) ≤ δ / 4 := by nlinarith only [hqδ, hδ]
          linarith only [h2a, h2b]
        exact (abs_add_le _ _).trans (by linarith only [h1, h2, hδ])
    · -- Verify `twoSidedSchurMap p a = a'`
      have hMR_r : M_R * M_R⁻¹ = 1 := Matrix.mul_nonsing_inv M_R hMR_unit
      have hU2_eq : U₂ - Matrix.of p.1.1.2 * M_L⁻¹ * Matrix.of p.2.2.1 = Matrix.of a'.2.1 := by
        dsimp only [U₂]; abel
      have hV2_eq : V₂ - Matrix.of p.2.2.2 * M_R⁻¹ * Matrix.of p.1.2.2 = Matrix.of a'.2.2 := by
        dsimp only [V₂]; abel
      have hB21_eq : B₂₁ - Matrix.of p.1.1.2 * M_L⁻¹ * Matrix.of p.2.1 +
          Matrix.of a'.2.1 * (Matrix.of p.2.2.2 * M_R⁻¹) = Matrix.of a'.1.2.2 := by
        dsimp only [B₂₁]; abel
      have hC12_eq : C₁₂ - C₁₁ * (M_R⁻¹ * Matrix.of p.1.2.2) +
          (M_L⁻¹ * Matrix.of p.2.2.1) * Matrix.of a'.2.2 = Matrix.of a'.1.2.1 := by
        dsimp only [C₁₂]; abel
      have hC11_eq : C₁₁ * M_R⁻¹ + M_L⁻¹ * Matrix.of p.2.1 +
          M_L⁻¹ * Matrix.of p.2.2.1 * Matrix.of p.2.2.2 * M_R⁻¹ = Matrix.of a'.1.1 := by
        dsimp only [C₁₁]
        rw [Matrix.mul_assoc, hMR_r, Matrix.mul_one]
        abel
      rcases a' with ⟨⟨Z11, Z12, Z21⟩, ⟨U2', V2'⟩⟩
      unfold twoSidedSchurMap
      dsimp only
      have hof : ∀ {a b : ℕ} (X : Matrix (Fin a) (Fin b) ℝ), Matrix.of (toRMat X) = X := fun _ => rfl
      simp only [hof]
      rw [hU2_eq, hV2_eq, hB21_eq, hC12_eq, hC11_eq]
      rfl

private theorem Z_prod_add {α β : Type*}
    [MeasureSpace α] [MeasureSpace β] [SigmaFinite (volume : Measure α)] [SigmaFinite (volume : Measure β)]
    (S₁ : Set α) (S₂ : Set β) (f : α → ℝ) (g : β → ℝ) (hf : Measurable f) (hg : Measurable g) (N : ℝ) :
    SLT.Z (volume.restrict (S₁ ×ˢ S₂)) (fun p : α × β => f p.1 + g p.2) N =
      SLT.Z (volume.restrict S₁) f N * SLT.Z (volume.restrict S₂) g N := by
  unfold SLT.Z
  have hres : (volume : Measure (α × β)).restrict (S₁ ×ˢ S₂) =
      ((volume : Measure α).restrict S₁).prod ((volume : Measure β).restrict S₂) := by
    rw [Measure.volume_eq_prod, ← Measure.prod_restrict]
  have hmf : Measurable fun x : α => ENNReal.ofReal (Real.exp (-(N * f x))) :=
    ENNReal.measurable_ofReal.comp (Real.measurable_exp.comp ((measurable_const.mul hf).neg))
  have hmfg : Measurable fun p : α × β => ENNReal.ofReal (Real.exp (-(N * (f p.1 + g p.2)))) :=
    ENNReal.measurable_ofReal.comp (Real.measurable_exp.comp
      ((measurable_const.mul ((hf.comp measurable_fst).add (hg.comp measurable_snd))).neg))
  rw [hres, MeasureTheory.lintegral_prod _ hmfg.aemeasurable]
  have hsplit : ∀ x y, ENNReal.ofReal (Real.exp (-(N * (f x + g y)))) =
      ENNReal.ofReal (Real.exp (-(N * f x))) * ENNReal.ofReal (Real.exp (-(N * g y))) := by
    intro x y; rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add]; congr 2; ring
  simp_rw [hsplit, MeasureTheory.lintegral_const_mul' _ _ ENNReal.ofReal_ne_top]
  rw [MeasureTheory.lintegral_mul_const _ hmf]

private theorem twoSidedDecoupled_hasLaplacePair (q d' k' : ℕ) {ρ : ℝ} (hρ : 0 < ρ) :
    SLT.HasLaplacePair
      (SLT.Z (volume.restrict (activeBox (q := q) (d' := d') (k' := k') ρ)) twoSidedDecoupledLoss)
      (((q : ℝ) * q + 2 * q * d') / 2 + awLambda d' d' k')
      (awMult d' d' k') := by
  have hZ_fac : ∀ N : ℝ,
      SLT.Z (volume.restrict (activeBox (q := q) (d' := d') (k' := k') ρ)) twoSidedDecoupledLoss N =
      (SLT.Z (volume.restrict (matBox (a := q) (b := q) ρ)) (fun X => frobSq (Matrix.of X)) N *
       (SLT.Z (volume.restrict (matBox (a := q) (b := d') ρ)) (fun X => frobSq (Matrix.of X)) N *
        SLT.Z (volume.restrict (matBox (a := d') (b := q) ρ)) (fun X => frobSq (Matrix.of X)) N)) *
      SLT.Z (volume.restrict (matBox (a := d') (b := k') ρ ×ˢ matBox (a := k') (b := d') ρ))
        rank0Loss N := fun N => by
    have hstep1 := Z_prod_add
      (matBox (a := q) (b := q) ρ ×ˢ matBox (a := q) (b := d') ρ ×ˢ matBox (a := d') (b := q) ρ)
      (matBox (a := d') (b := k') ρ ×ˢ matBox (a := k') (b := d') ρ)
      (fun x => frobSq (Matrix.of x.1) + (frobSq (Matrix.of x.2.1) + frobSq (Matrix.of x.2.2)))
      rank0Loss
      (measurable_frobSq.comp measurable_fst |>.add
        ((measurable_frobSq.comp (measurable_fst.comp measurable_snd)).add
         (measurable_frobSq.comp (measurable_snd.comp measurable_snd))))
      measurable_rank0Loss N
    have hstep2 := Z_prod_add
      (matBox (a := q) (b := q) ρ)
      (matBox (a := q) (b := d') ρ ×ˢ matBox (a := d') (b := q) ρ)
      (fun X => frobSq (Matrix.of X))
      (fun x => frobSq (Matrix.of x.1) + frobSq (Matrix.of x.2))
      measurable_frobSq
      ((measurable_frobSq.comp measurable_fst).add (measurable_frobSq.comp measurable_snd)) N
    have hstep3 := Z_prod_add
      (matBox (a := q) (b := d') ρ)
      (matBox (a := d') (b := q) ρ)
      (fun X => frobSq (Matrix.of X))
      (fun X => frobSq (Matrix.of X))
      measurable_frobSq measurable_frobSq N
    have hloss_rew : twoSidedDecoupledLoss (q := q) (d' := d') (k' := k') =
        fun a => (frobSq (Matrix.of a.1.1) + (frobSq (Matrix.of a.1.2.1) + frobSq (Matrix.of a.1.2.2))) +
          rank0Loss a.2 := by
      funext a; unfold twoSidedDecoupledLoss; ring
    rw [hloss_rew, activeBox, hstep1, hstep2, hstep3]
  have h1 := hasLaplacePair_frobSq (m := q) (q := q) hρ
  have h2 := hasLaplacePair_frobSq (m := q) (q := d') hρ
  have h3 := hasLaplacePair_frobSq (m := d') (q := q) hρ
  have h4 := rank0_hasLaplacePair_of_unit hρ (aoyagi_watanabe_rank0 d' k' d')
  have hprod := (h1.mul (h2.mul h3)).mul h4
  have hlam : ((q : ℝ) * q / 2 + ((q : ℝ) * d' / 2 + (d' : ℝ) * q / 2)) + awLambda d' d' k' =
      ((q : ℝ) * q + 2 * q * d') / 2 + awLambda d' d' k' := by ring
  have hmult : 1 + (1 + 1 - 1) - 1 + awMult d' d' k' - 1 = awMult d' d' k' := by
    have := awMult_pos d' d' k'; omega
  rw [hlam, hmult] at hprod
  exact hprod.congr (Filter.Eventually.of_forall fun N => (hZ_fac N).symm)

/-- **Exact Two-Sided Schur Complement Reduction (`[LC]`):**
For all `q, d', k' : ℕ`, the balanced two-sided `(q, q)` block loss on `passiveBox δ ×ˢ activeBox δ`
has exact learning pair
`((q * q + 2 * q * d') / 2 + awLambda d' d' k', awMult d' d' k')`. -/
theorem twoSided_schur_reduction (q d' k' : ℕ) :
    ∃ δ₀ : ℝ, 0 < δ₀ ∧ ∀ δ : ℝ, 0 < δ → δ ≤ δ₀ →
      SLT.HasLaplacePair
        (SLT.Z (volume.restrict (passiveBox (q := q) (d' := d') (k' := k') δ ×ˢ
          activeBox (q := q) (d' := d') (k' := k') δ)) twoSidedLoss)
        (((q : ℝ) * q + 2 * q * d') / 2 + awLambda d' d' k')
        (awMult d' d' k') := by
  obtain ⟨δ₀, hδ₀, hgood⟩ := exists_good_twoSided_radius q d' k'
  have hdec_pair : ∀ ρ : ℝ, 0 < ρ →
      SLT.HasLaplacePair
        (SLT.Z (volume.restrict (activeBox (q := q) (d' := d') (k' := k') ρ)) twoSidedDecoupledLoss)
        (((q : ℝ) * q + 2 * q * d') / 2 + awLambda d' d' k')
        (awMult d' d' k') := fun _ hρ => twoSidedDecoupled_hasLaplacePair q d' k' hρ
  -- Step 2: sandwich `twoSidedLoss` integral between decoupled integrals at `δ/4` and `4*δ`.
  refine ⟨δ₀, hδ₀, fun δ hδ hδle => ?_⟩
  obtain ⟨hgood_fwd, hgood_surj⟩ := hgood δ hδ hδle
  have hP_meas : ∀ ρ, MeasurableSet (passiveBox (q := q) (d' := d') (k' := k') ρ) := fun ρ =>
    (((measurableSet_matBox ρ).prod (measurableSet_matBox ρ)).prod
      ((measurableSet_matBox ρ).prod (measurableSet_matBox ρ))).prod
    ((measurableSet_matBox ρ).prod ((measurableSet_matBox ρ).prod (measurableSet_matBox ρ)))
  have hA_meas : ∀ ρ, MeasurableSet (activeBox (q := q) (d' := d') (k' := k') ρ) := fun ρ =>
    ((measurableSet_matBox ρ).prod ((measurableSet_matBox ρ).prod (measurableSet_matBox ρ))).prod
      ((measurableSet_matBox ρ).prod (measurableSet_matBox ρ))
  have hP_vol_pos : ∀ ρ > 0, volume (passiveBox (q := q) (d' := d') (k' := k') ρ) ≠ 0 := fun ρ hρ => by
    unfold passiveBox
    simp only [Measure.volume_eq_prod, Measure.prod_prod]
    exact (ENNReal.mul_pos
      (ENNReal.mul_pos
        (ENNReal.mul_pos (volume_matBox_pos hρ).ne' (volume_matBox_pos hρ).ne').ne'
        (ENNReal.mul_pos (volume_matBox_pos hρ).ne' (volume_matBox_pos hρ).ne').ne').ne'
      (ENNReal.mul_pos (volume_matBox_pos hρ).ne'
        (ENNReal.mul_pos (volume_matBox_pos hρ).ne' (volume_matBox_pos hρ).ne').ne').ne').ne'
  have hP_vol_top : ∀ ρ ≥ 0, volume (passiveBox (q := q) (d' := d') (k' := k') ρ) ≠ ⊤ := fun ρ hρ => by
    unfold passiveBox
    simp only [Measure.volume_eq_prod, Measure.prod_prod]
    exact ENNReal.mul_ne_top
      (ENNReal.mul_ne_top (ENNReal.mul_ne_top (volume_matBox_lt_top hρ).ne (volume_matBox_lt_top hρ).ne)
        (ENNReal.mul_ne_top (volume_matBox_lt_top hρ).ne (volume_matBox_lt_top hρ).ne))
      (ENNReal.mul_ne_top (volume_matBox_lt_top hρ).ne
        (ENNReal.mul_ne_top (volume_matBox_lt_top hρ).ne (volume_matBox_lt_top hρ).ne))
  have hP_sub : passiveBox (q := q) (d' := d') (k' := k') (δ / 8) ⊆
      passiveBox (q := q) (d' := d') (k' := k') δ := by
    have hle : δ / 8 ≤ δ := by linarith
    exact Set.prod_mono
      (Set.prod_mono (Set.prod_mono (matBox_mono hle) (matBox_mono hle))
        (Set.prod_mono (matBox_mono hle) (matBox_mono hle)))
      (Set.prod_mono (matBox_mono hle) (Set.prod_mono (matBox_mono hle) (matBox_mono hle)))
  have hL_meas : Measurable (twoSidedLoss (q := q) (d' := d') (k' := k')) := by
    unfold twoSidedLoss frobSq
    simp only [Matrix.add_apply, Matrix.mul_apply, Matrix.of_apply]
    fun_prop
  have hdec_meas : Measurable (twoSidedDecoupledLoss (q := q) (d' := d') (k' := k')) := by
    unfold twoSidedDecoupledLoss
    exact ((measurable_frobSq.comp (measurable_fst.comp measurable_fst)).add
      (measurable_frobSq.comp (measurable_fst.comp (measurable_snd.comp measurable_fst)))).add
      (measurable_frobSq.comp (measurable_snd.comp (measurable_snd.comp measurable_fst))) |>.add
      (measurable_rank0Loss.comp measurable_snd)
  have hschur_meas : ∀ p : TwoSidedPassive q d' k', Measurable (twoSidedSchurMap p) := fun p => by
    unfold twoSidedSchurMap
    refine Measurable.prodMk ?_ ?_
    · refine Measurable.prodMk ?_ (Measurable.prodMk ?_ ?_)
      · refine Measurable.of_eval fun i => Measurable.of_eval fun j => ?_
        simp only [toRMat_apply, Matrix.add_apply, Matrix.mul_apply, Matrix.of_apply]; fun_prop
      · refine Measurable.of_eval fun i => Measurable.of_eval fun j => ?_
        simp only [toRMat_apply, Matrix.add_apply, Matrix.sub_apply, Matrix.mul_apply, Matrix.of_apply]; fun_prop
      · refine Measurable.of_eval fun i => Measurable.of_eval fun j => ?_
        simp only [toRMat_apply, Matrix.add_apply, Matrix.sub_apply, Matrix.mul_apply, Matrix.of_apply]; fun_prop
    · refine Measurable.prodMk ?_ ?_
      · refine Measurable.of_eval fun i => Measurable.of_eval fun j => ?_
        simp only [toRMat_apply, Matrix.sub_apply, Matrix.mul_apply, Matrix.of_apply]; fun_prop
      · refine Measurable.of_eval fun i => Measurable.of_eval fun j => ?_
        simp only [toRMat_apply, Matrix.sub_apply, Matrix.mul_apply, Matrix.of_apply]; fun_prop
  have hbounds : ∀ N : ℝ, 0 ≤ N →
      (ENNReal.ofReal ((2 : ℝ)⁻¹ ^ q) * volume (passiveBox (q := q) (d' := d') (k' := k') (δ / 8))) *
        SLT.Z (volume.restrict (activeBox (q := q) (d' := d') (k' := k') (δ / 4)))
          twoSidedDecoupledLoss (20 * N) ≤
      SLT.Z (volume.restrict (passiveBox (q := q) (d' := d') (k' := k') δ ×ˢ
        activeBox (q := q) (d' := d') (k' := k') δ)) twoSidedLoss N ∧
      SLT.Z (volume.restrict (passiveBox (q := q) (d' := d') (k' := k') δ ×ˢ
        activeBox (q := q) (d' := d') (k' := k') δ)) twoSidedLoss N ≤
      (ENNReal.ofReal ((2 : ℝ) ^ q) * volume (passiveBox (q := q) (d' := d') (k' := k') δ)) *
        SLT.Z (volume.restrict (activeBox (q := q) (d' := d') (k' := k') (4 * δ)))
          twoSidedDecoupledLoss ((1 / 20 : ℝ) * N) := by
    intro N hN
    have hZ_iter :
        SLT.Z (volume.restrict (passiveBox (q := q) (d' := d') (k' := k') δ ×ˢ
          activeBox (q := q) (d' := d') (k' := k') δ)) twoSidedLoss N =
        ∫⁻ (p : TwoSidedPassive q d' k') in passiveBox δ,
          ∫⁻ (a : TwoSidedActive q d' k') in activeBox δ,
            ENNReal.ofReal (Real.exp (-(N * twoSidedLoss (p, a)))) := by
      unfold SLT.Z
      have hres : (volume : Measure (TwoSidedPassive q d' k' × TwoSidedActive q d' k')).restrict
          (passiveBox δ ×ˢ activeBox δ) =
          ((volume : Measure (TwoSidedPassive q d' k')).restrict (passiveBox δ)).prod
            ((volume : Measure (TwoSidedActive q d' k')).restrict (activeBox δ)) := by
        rw [Measure.volume_eq_prod, ← Measure.prod_restrict]
      have hm : Measurable fun x : TwoSidedPassive q d' k' × TwoSidedActive q d' k' =>
          ENNReal.ofReal (Real.exp (-(N * twoSidedLoss x))) :=
        ENNReal.measurable_ofReal.comp (Real.measurable_exp.comp ((measurable_const.mul hL_meas).neg))
      rw [hres, MeasureTheory.lintegral_prod _ hm.aemeasurable]
    rw [hZ_iter]
    constructor
    · -- Lower bound: restrict passive to `passiveBox (δ/8)`
      have hslice : ∀ p ∈ passiveBox (q := q) (d' := d') (k' := k') (δ / 8),
          ENNReal.ofReal ((2 : ℝ)⁻¹ ^ q) *
            SLT.Z (volume.restrict (activeBox (q := q) (d' := d') (k' := k') (δ / 4)))
              twoSidedDecoupledLoss (20 * N) ≤
          ∫⁻ a in activeBox δ, ENNReal.ofReal (Real.exp (-(N * twoSidedLoss (p, a)))) := by
        intro p hp8
        have hp := hP_sub hp8
        obtain ⟨hML, hMR, hMR_lo, _, hΔ11, hΔ21, hΓ11, hΓ12, hMLinv, hMRinv, _⟩ := hgood_fwd p hp
        have hMR_ne : ((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of p.1.2.1).det ≠ 0 :=
          isUnit_iff_ne_zero.mp hMR
        have hjac : ENNReal.ofReal ((2 : ℝ)⁻¹ ^ q) ≤
            ENNReal.ofReal |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of p.1.2.1).det| ^ q := by
          rw [ENNReal.ofReal_pow (by positivity)]
          refine pow_le_pow_left' (ENNReal.ofReal_le_ofReal ?_) q
          linarith
        have hstep1 :
            ∫⁻ a in activeBox δ, ENNReal.ofReal (Real.exp (-((20 * N) * twoSidedDecoupledLoss (twoSidedSchurMap p a)))) ≤
            ∫⁻ a in activeBox δ, ENNReal.ofReal (Real.exp (-(N * twoSidedLoss (p, a)))) := by
          refine setLIntegral_mono' (hA_meas δ) fun a _ => ?_
          apply ENNReal.ofReal_le_ofReal
          apply Real.exp_le_exp.mpr
          have hup := (twoSided_loss_bounds p a hML hMR hΔ11 hΔ21 hΓ11 hΓ12 hMLinv hMRinv).2
          have := mul_le_mul_of_nonneg_left hup hN
          linarith
        have hstep2 :
            ∫⁻ a : TwoSidedActive q d' k', (activeBox (δ / 4)).indicator
              (fun a' => ENNReal.ofReal (Real.exp (-((20 * N) * twoSidedDecoupledLoss a'))))
              (twoSidedSchurMap p a) ≤
            ∫⁻ a in activeBox δ,
              ENNReal.ofReal (Real.exp (-((20 * N) * twoSidedDecoupledLoss (twoSidedSchurMap p a)))) := by
          rw [← lintegral_indicator (hA_meas δ)]
          refine lintegral_mono fun a => ?_
          by_cases hmem : twoSidedSchurMap p a ∈ activeBox (δ / 4)
          · obtain ⟨a₀, ha₀_mem, ha₀_eq⟩ := hgood_surj p hp8 _ hmem
            -- Injectivity of `twoSidedSchurMap p` when `M_R` is invertible!
            have ha_eq : a = a₀ := by
              have hMR_l : ((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of p.1.2.1)⁻¹ *
                  ((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of p.1.2.1) = 1 :=
                Matrix.nonsing_inv_mul _ hMR
              rcases a with ⟨⟨c11, c12, b21⟩, ⟨u2, v2⟩⟩
              rcases a₀ with ⟨⟨c11', c12', b21'⟩, ⟨u2', v2'⟩⟩
              have hof_to : ∀ {a b : ℕ} (X : Matrix (Fin a) (Fin b) ℝ), Matrix.of (toRMat X) = X :=
                fun _ => rfl
              have h1 := congrArg (fun x => Matrix.of x.2.1) ha₀_eq
              have h2 := congrArg (fun x => Matrix.of x.2.2) ha₀_eq
              have h3 := congrArg (fun x => Matrix.of x.1.2.2) ha₀_eq
              have h4 := congrArg (fun x => Matrix.of x.1.2.1) ha₀_eq
              have h5 := congrArg (fun x => Matrix.of x.1.1) ha₀_eq
              simp only [twoSidedSchurMap, hof_to] at h1 h2 h3 h4 h5
              have hu : Matrix.of u2 = Matrix.of u2' := sub_left_inj.mp h1.symm
              have hv : Matrix.of v2 = Matrix.of v2' := sub_left_inj.mp h2.symm
              have hc11_m : Matrix.of c11 * ((1 + Matrix.of p.1.2.1)⁻¹) =
                  Matrix.of c11' * ((1 + Matrix.of p.1.2.1)⁻¹) :=
                add_right_cancel (add_right_cancel h5.symm)
              have hc11 : Matrix.of c11 = Matrix.of c11' := by
                have := congrArg (fun X => X * (1 + Matrix.of p.1.2.1)) hc11_m
                simpa only [Matrix.mul_assoc, hMR_l, Matrix.mul_one] using this
              have hb21 : Matrix.of b21 = Matrix.of b21' := by
                rw [hu] at h3
                exact sub_left_inj.mp (add_right_cancel h3.symm)
              have hc12 : Matrix.of c12 = Matrix.of c12' := by
                rw [hc11, hv] at h4
                exact sub_left_inj.mp (add_right_cancel h4.symm)
              have hE : ∀ {a b : ℕ} {X Y : RMat a b}, Matrix.of X = Matrix.of Y → X = Y :=
                fun h => funext fun i => funext fun j => congrFun (congrFun h i) j
              rw [hE hc11, hE hc12, hE hb21, hE hu, hE hv]
            rw [Set.indicator_of_mem hmem, Set.indicator_of_mem (ha_eq ▸ ha₀_mem)]
          · rw [Set.indicator_of_notMem hmem]; exact bot_le
        have hstep3 :
            ∫⁻ a : TwoSidedActive q d' k', (activeBox (δ / 4)).indicator
              (fun a' => ENNReal.ofReal (Real.exp (-((20 * N) * twoSidedDecoupledLoss a'))))
              (twoSidedSchurMap p a) =
            ENNReal.ofReal |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of p.1.2.1).det| ^ q *
              SLT.Z (volume.restrict (activeBox (q := q) (d' := d') (k' := k') (δ / 4)))
                twoSidedDecoupledLoss (20 * N) := by
          have hg : Measurable fun a' : TwoSidedActive q d' k' =>
              ENNReal.ofReal (Real.exp (-((20 * N) * twoSidedDecoupledLoss a'))) :=
            ENNReal.measurable_ofReal.comp (Real.measurable_exp.comp ((measurable_const.mul hdec_meas).neg))
          rw [← lintegral_map (hg.indicator (hA_meas (δ / 4))) (hschur_meas p),
            map_twoSidedSchurMap_volume p hMR_ne, lintegral_smul_measure, smul_eq_mul,
            lintegral_indicator (hA_meas (δ / 4))]
          rfl
        calc ENNReal.ofReal ((2 : ℝ)⁻¹ ^ q) *
              SLT.Z (volume.restrict (activeBox (q := q) (d' := d') (k' := k') (δ / 4)))
                twoSidedDecoupledLoss (20 * N)
            ≤ ENNReal.ofReal |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of p.1.2.1).det| ^ q *
              SLT.Z (volume.restrict (activeBox (q := q) (d' := d') (k' := k') (δ / 4)))
                twoSidedDecoupledLoss (20 * N) := by gcongr
          _ = _ := hstep3.symm
          _ ≤ _ := hstep2.trans hstep1
      calc (ENNReal.ofReal ((2 : ℝ)⁻¹ ^ q) * volume (passiveBox (q := q) (d' := d') (k' := k') (δ / 8))) *
            SLT.Z (volume.restrict (activeBox (q := q) (d' := d') (k' := k') (δ / 4)))
              twoSidedDecoupledLoss (20 * N)
          = (ENNReal.ofReal ((2 : ℝ)⁻¹ ^ q) *
              SLT.Z (volume.restrict (activeBox (q := q) (d' := d') (k' := k') (δ / 4)))
                twoSidedDecoupledLoss (20 * N)) *
            volume (passiveBox (q := q) (d' := d') (k' := k') (δ / 8)) := by ring
        _ = ∫⁻ _ in passiveBox (q := q) (d' := d') (k' := k') (δ / 8), (ENNReal.ofReal ((2 : ℝ)⁻¹ ^ q) *
              SLT.Z (volume.restrict (activeBox (q := q) (d' := d') (k' := k') (δ / 4)))
                twoSidedDecoupledLoss (20 * N)) :=
            (setLIntegral_const _ _).symm
        _ ≤ ∫⁻ (p : TwoSidedPassive q d' k') in passiveBox (δ / 8),
              ∫⁻ (a : TwoSidedActive q d' k') in activeBox δ,
                ENNReal.ofReal (Real.exp (-(N * twoSidedLoss (p, a)))) :=
            setLIntegral_mono' (hP_meas (δ / 8)) hslice
        _ ≤ ∫⁻ (p : TwoSidedPassive q d' k') in passiveBox δ,
              ∫⁻ (a : TwoSidedActive q d' k') in activeBox δ,
                ENNReal.ofReal (Real.exp (-(N * twoSidedLoss (p, a)))) :=
            lintegral_mono_set hP_sub
    · -- Upper bound
      have hslice : ∀ p ∈ passiveBox (q := q) (d' := d') (k' := k') δ,
          ∫⁻ a in activeBox δ, ENNReal.ofReal (Real.exp (-(N * twoSidedLoss (p, a)))) ≤
          ENNReal.ofReal ((2 : ℝ) ^ q) *
            SLT.Z (volume.restrict (activeBox (q := q) (d' := d') (k' := k') (4 * δ)))
              twoSidedDecoupledLoss ((1 / 20 : ℝ) * N) := by
        intro p hp
        obtain ⟨hML, hMR, _, hMR_hi, hΔ11, hΔ21, hΓ11, hΓ12, hMLinv, hMRinv, hfwd⟩ := hgood_fwd p hp
        have hMR_ne : ((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of p.1.2.1).det ≠ 0 :=
          isUnit_iff_ne_zero.mp hMR
        have hjac : ENNReal.ofReal |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of p.1.2.1).det| ^ q ≤
            ENNReal.ofReal ((2 : ℝ) ^ q) := by
          rw [ENNReal.ofReal_pow (by positivity)]
          exact pow_le_pow_left' (ENNReal.ofReal_le_ofReal hMR_hi) q
        have hstep1 :
            ∫⁻ a in activeBox δ, ENNReal.ofReal (Real.exp (-(N * twoSidedLoss (p, a)))) ≤
            ∫⁻ a in activeBox δ,
              ENNReal.ofReal (Real.exp (-(((1 / 20 : ℝ) * N) * twoSidedDecoupledLoss (twoSidedSchurMap p a)))) := by
          refine setLIntegral_mono' (hA_meas δ) fun a _ => ?_
          apply ENNReal.ofReal_le_ofReal
          apply Real.exp_le_exp.mpr
          have hlo := (twoSided_loss_bounds p a hML hMR hΔ11 hΔ21 hΓ11 hΓ12 hMLinv hMRinv).1
          have := mul_le_mul_of_nonneg_left hlo hN
          linarith
        have hstep2 :
            ∫⁻ a in activeBox δ,
              ENNReal.ofReal (Real.exp (-(((1 / 20 : ℝ) * N) * twoSidedDecoupledLoss (twoSidedSchurMap p a)))) ≤
            ∫⁻ a : TwoSidedActive q d' k', (activeBox (4 * δ)).indicator
              (fun a' => ENNReal.ofReal (Real.exp (-(((1 / 20 : ℝ) * N) * twoSidedDecoupledLoss a'))))
              (twoSidedSchurMap p a) := by
          rw [← lintegral_indicator (hA_meas δ)]
          refine lintegral_mono fun a => ?_
          by_cases ha : a ∈ activeBox δ
          · rw [Set.indicator_of_mem ha, Set.indicator_of_mem (hfwd a ha)]
          · rw [Set.indicator_of_notMem ha]; exact bot_le
        have hstep3 :
            ∫⁻ a : TwoSidedActive q d' k', (activeBox (4 * δ)).indicator
              (fun a' => ENNReal.ofReal (Real.exp (-(((1 / 20 : ℝ) * N) * twoSidedDecoupledLoss a'))))
              (twoSidedSchurMap p a) =
            ENNReal.ofReal |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of p.1.2.1).det| ^ q *
              SLT.Z (volume.restrict (activeBox (q := q) (d' := d') (k' := k') (4 * δ)))
                twoSidedDecoupledLoss ((1 / 20 : ℝ) * N) := by
          have hg : Measurable fun a' : TwoSidedActive q d' k' =>
              ENNReal.ofReal (Real.exp (-(((1 / 20 : ℝ) * N) * twoSidedDecoupledLoss a'))) :=
            ENNReal.measurable_ofReal.comp (Real.measurable_exp.comp ((measurable_const.mul hdec_meas).neg))
          rw [← lintegral_map (hg.indicator (hA_meas (4 * δ))) (hschur_meas p),
            map_twoSidedSchurMap_volume p hMR_ne, lintegral_smul_measure, smul_eq_mul,
            lintegral_indicator (hA_meas (4 * δ))]
          rfl
        calc ∫⁻ (a : TwoSidedActive q d' k') in activeBox δ, ENNReal.ofReal (Real.exp (-(N * twoSidedLoss (p, a))))
            ≤ ENNReal.ofReal |((1 : Matrix (Fin q) (Fin q) ℝ) + Matrix.of p.1.2.1).det| ^ q *
              SLT.Z (volume.restrict (activeBox (q := q) (d' := d') (k' := k') (4 * δ)))
                twoSidedDecoupledLoss ((1 / 20 : ℝ) * N) :=
              (hstep1.trans hstep2).trans hstep3.le
          _ ≤ ENNReal.ofReal ((2 : ℝ) ^ q) *
              SLT.Z (volume.restrict (activeBox (q := q) (d' := d') (k' := k') (4 * δ)))
                twoSidedDecoupledLoss ((1 / 20 : ℝ) * N) := by
              gcongr
      calc ∫⁻ (p : TwoSidedPassive q d' k') in passiveBox δ,
            ∫⁻ (a : TwoSidedActive q d' k') in activeBox δ,
              ENNReal.ofReal (Real.exp (-(N * twoSidedLoss (p, a))))
          ≤ ∫⁻ _ in passiveBox (q := q) (d' := d') (k' := k') δ, (ENNReal.ofReal ((2 : ℝ) ^ q) *
              SLT.Z (volume.restrict (activeBox (q := q) (d' := d') (k' := k') (4 * δ)))
                twoSidedDecoupledLoss ((1 / 20 : ℝ) * N)) :=
            setLIntegral_mono' (hP_meas δ) hslice
        _ = (ENNReal.ofReal ((2 : ℝ) ^ q) * volume (passiveBox (q := q) (d' := d') (k' := k') δ)) *
              SLT.Z (volume.restrict (activeBox (q := q) (d' := d') (k' := k') (4 * δ)))
                twoSidedDecoupledLoss ((1 / 20 : ℝ) * N) := by
            rw [setLIntegral_const]; ring
  let cLo := ENNReal.ofReal ((2 : ℝ)⁻¹ ^ q) * volume (passiveBox (q := q) (d' := d') (k' := k') (δ / 8))
  let cHi := ENNReal.ofReal ((2 : ℝ) ^ q) * volume (passiveBox (q := q) (d' := d') (k' := k') δ)
  have hlo_pair := ((hdec_pair (δ / 4) (by linarith)).comp_mul (by norm_num : (0 : ℝ) < 20)).const_mul
    (a := cLo)
    (ENNReal.mul_pos (ENNReal.ofReal_pos.mpr (by positivity)).ne' (hP_vol_pos (δ / 8) (by linarith))).ne'
    (ENNReal.mul_ne_top ENNReal.ofReal_lt_top.ne (hP_vol_top (δ / 8) (by linarith)))
  have hhi_pair := ((hdec_pair (4 * δ) (by linarith)).comp_mul (by norm_num : (0 : ℝ) < 1 / 20)).const_mul
    (a := cHi)
    (ENNReal.mul_pos (ENNReal.ofReal_pos.mpr (by positivity)).ne' (hP_vol_pos δ hδ)).ne'
    (ENNReal.mul_ne_top ENNReal.ofReal_lt_top.ne (hP_vol_top δ hδ.le))
  refine ⟨hlo_pair.1, ?_, ?_⟩
  · exact hlo_pair.2.1.mono ((eventually_ge_atTop 0).mono fun N hN => (hbounds N hN).1)
  · exact hhi_pair.2.2.mono ((eventually_ge_atTop 0).mono fun N hN => (hbounds N hN).2)

/-! ### 6. Joint `Q+K` LoRA Exact Learning Pair & Complete `GL(q)` Flat Cancellation -/

/-- Exact linearized `Q+K` two-sided matrix-product loss on `AttnPairParam d H`. -/
private def attnQKLinLoss (d H q : ℕ) (θ : AttnPairParam d H) : ℝ :=
  frobSq (Matrix.of θ.1.1 * (baseAQK d H q + Matrix.of θ.1.2.2) +
          (baseAQK d H q + Matrix.of θ.1.2.1)ᵀ * (Matrix.of θ.2)ᵀ)

/-- Clean 4-block transposition `(A × B) × (C × D) ≃ᵐ (A × C) × (B × D)` preserving volume. -/
private def fourSwap (A B C D : Type*) [MeasurableSpace A] [MeasurableSpace B]
    [MeasurableSpace C] [MeasurableSpace D] :
    ((A × B) × (C × D)) ≃ᵐ ((A × C) × (B × D)) :=
  (MeasurableEquiv.prodAssoc (α := A) (β := B) (γ := C × D)).trans
    (((MeasurableEquiv.refl A).prodCongr
      ((MeasurableEquiv.prodAssoc (α := B) (β := C) (γ := D)).symm.trans
        (((MeasurableEquiv.prodComm (α := B) (β := C)).prodCongr (MeasurableEquiv.refl D)).trans
          (MeasurableEquiv.prodAssoc (α := C) (β := B) (γ := D))))).trans
     (MeasurableEquiv.prodAssoc (α := A) (β := C) (γ := B × D)).symm)

private theorem measurePreserving_fourSwap {A B C D : Type*}
    [MeasureSpace A] [MeasureSpace B] [MeasureSpace C] [MeasureSpace D]
    [SigmaFinite (volume : Measure A)] [SigmaFinite (volume : Measure B)]
    [SigmaFinite (volume : Measure C)] [SigmaFinite (volume : Measure D)] :
    MeasurePreserving (fourSwap A B C D) volume volume := by
  unfold fourSwap
  have h1 : MeasurePreserving
      ((MeasurableEquiv.prodComm (α := B) (β := C)).prodCongr (MeasurableEquiv.refl D))
      (volume : Measure ((B × C) × D)) (volume : Measure ((C × B) × D)) :=
    (Measure.measurePreserving_swap (μ := (volume : Measure B)) (ν := (volume : Measure C))).prod
      (MeasurePreserving.id (volume : Measure D))
  have hinner : MeasurePreserving
      ((MeasurableEquiv.prodAssoc (α := B) (β := C) (γ := D)).symm.trans
        (((MeasurableEquiv.prodComm (α := B) (β := C)).prodCongr (MeasurableEquiv.refl D)).trans
          (MeasurableEquiv.prodAssoc (α := C) (β := B) (γ := D))))
      (volume : Measure (B × (C × D))) (volume : Measure (C × (B × D))) :=
    MeasureTheory.volume_preserving_prodAssoc.comp
      (h1.comp MeasureTheory.volume_preserving_prodAssoc.symm)
  exact MeasureTheory.volume_preserving_prodAssoc.symm.comp
    (((MeasurePreserving.id (volume : Measure A)).prod hinner).comp
      MeasureTheory.volume_preserving_prodAssoc)

/-- Measure-preserving bijection from `RMat (q + d') (q + q + k') × RMat (q + q + k') (q + d')`
to `TwoSidedPassive q d' k' × TwoSidedActive q d' k'`. -/
private def twoSidedMatToBlocks (q d' k' : ℕ) :
    (RMat (q + d') ((q + q) + k') × RMat ((q + q) + k') (q + d')) ≃ᵐ
      (TwoSidedPassive q d' k' × TwoSidedActive q d' k') :=
  let eX : RMat (q + d') ((q + q) + k') ≃ᵐ
      ((RMat q q × RMat q q) × RMat q k') × ((RMat d' q × RMat d' q) × RMat d' k') :=
    (splitRowsEquiv q d' ((q + q) + k')).trans
      (((splitColsEquiv q (q + q) k').trans ((splitColsEquiv q q q).prodCongr (MeasurableEquiv.refl _))).prodCongr
       ((splitColsEquiv d' (q + q) k').trans ((splitColsEquiv d' q q).prodCongr (MeasurableEquiv.refl _))))
  let eY : RMat ((q + q) + k') (q + d') ≃ᵐ
      ((RMat q q × RMat q d') × (RMat q q × RMat q d')) × (RMat k' q × RMat k' d') :=
    (splitRowsEquiv (q + q) k' (q + d')).trans
      (((splitRowsEquiv q q (q + d')).trans ((splitColsEquiv q q d').prodCongr (splitColsEquiv q q d'))).prodCongr
       (splitColsEquiv k' q d'))
  -- `X = (((B11, Δ11), U1), ((B21, Δ21), U2))` -> `(((Δ11, Δ21), (B11, U1)), (B21, U2))`
  let eX_sep : (((RMat q q × RMat q q) × RMat q k') × ((RMat d' q × RMat d' q) × RMat d' k')) ≃ᵐ
      (((RMat q q × RMat d' q) × (RMat q q × RMat q k')) × (RMat d' q × RMat d' k')) :=
    (((MeasurableEquiv.prodComm.prodCongr (MeasurableEquiv.refl _)).trans
        MeasurableEquiv.prodAssoc).prodCongr
      ((MeasurableEquiv.prodComm.prodCongr (MeasurableEquiv.refl _)).trans
        MeasurableEquiv.prodAssoc)).trans
    ((fourSwap (RMat q q) (RMat q q × RMat q k') (RMat d' q) (RMat d' q × RMat d' k')).trans
      MeasurableEquiv.prodAssoc.symm)
  -- `Y = (((Γ11, Γ12), (C11, C12)), (V1, V2))` -> `(((Γ11, Γ12), V1), ((C11, C12), V2))`
  let eY_sep : (((RMat q q × RMat q d') × (RMat q q × RMat q d')) × (RMat k' q × RMat k' d')) ≃ᵐ
      (((RMat q q × RMat q d') × RMat k' q) × ((RMat q q × RMat q d') × RMat k' d')) :=
    fourSwap (RMat q q × RMat q d') (RMat q q × RMat q d') (RMat k' q) (RMat k' d')
  -- Combine `(P_X, A_X) × (P_Y, A_Y) → (P_X × P_Y) × (A_X × A_Y)`
  let eInter := fourSwap
    ((RMat q q × RMat d' q) × (RMat q q × RMat q k'))
    (RMat d' q × RMat d' k')
    ((RMat q q × RMat q d') × RMat k' q)
    ((RMat q q × RMat q d') × RMat k' d')
  -- Polish `P_X × P_Y → TwoSidedPassive` and `A_X × A_Y → TwoSidedActive`
  let eP : (((RMat q q × RMat d' q) × (RMat q q × RMat q k')) × ((RMat q q × RMat q d') × RMat k' q)) ≃ᵐ
      TwoSidedPassive q d' k' :=
    (fourSwap (RMat q q × RMat d' q) (RMat q q × RMat q k') (RMat q q × RMat q d') (RMat k' q)).trans
      ((MeasurableEquiv.refl _).prodCongr MeasurableEquiv.prodAssoc)
  let eA : ((RMat d' q × RMat d' k') × ((RMat q q × RMat q d') × RMat k' d')) ≃ᵐ
      TwoSidedActive q d' k' :=
    MeasurableEquiv.prodComm.trans
      ((fourSwap (RMat q q × RMat q d') (RMat k' d') (RMat d' q) (RMat d' k')).trans
        (MeasurableEquiv.prodAssoc.prodCongr MeasurableEquiv.prodComm))
  (eX.prodCongr eY).trans ((eX_sep.prodCongr eY_sep).trans (eInter.trans (eP.prodCongr eA)))

private theorem measurePreserving_twoSidedMatToBlocks (q d' k' : ℕ) :
    MeasurePreserving (twoSidedMatToBlocks q d' k') volume volume := by
  unfold twoSidedMatToBlocks
  have heX : MeasurePreserving
      ((splitRowsEquiv q d' ((q + q) + k')).trans
        (((splitColsEquiv q (q + q) k').trans ((splitColsEquiv q q q).prodCongr (MeasurableEquiv.refl _))).prodCongr
         ((splitColsEquiv d' (q + q) k').trans ((splitColsEquiv d' q q).prodCongr (MeasurableEquiv.refl _)))))
      volume volume :=
    ((((measurePreserving_splitColsEquiv q q q).prod (MeasurePreserving.id (volume : Measure (RMat q k')))).comp
      (measurePreserving_splitColsEquiv q (q + q) k')).prod
     (((measurePreserving_splitColsEquiv d' q q).prod (MeasurePreserving.id (volume : Measure (RMat d' k')))).comp
      (measurePreserving_splitColsEquiv d' (q + q) k'))).comp
      (measurePreserving_splitRowsEquiv q d' ((q + q) + k'))
  have heY : MeasurePreserving
      ((splitRowsEquiv (q + q) k' (q + d')).trans
        (((splitRowsEquiv q q (q + d')).trans ((splitColsEquiv q q d').prodCongr (splitColsEquiv q q d'))).prodCongr
         (splitColsEquiv k' q d')))
      volume volume :=
    ((((measurePreserving_splitColsEquiv q q d').prod (measurePreserving_splitColsEquiv q q d')).comp
      (measurePreserving_splitRowsEquiv q q (q + d'))).prod
      (measurePreserving_splitColsEquiv k' q d')).comp
      (measurePreserving_splitRowsEquiv (q + q) k' (q + d'))
  have hXpre1 : MeasurePreserving
      (((MeasurableEquiv.prodComm (α := RMat q q) (β := RMat q q)).prodCongr
          (MeasurableEquiv.refl (RMat q k'))).trans
        (MeasurableEquiv.prodAssoc (α := RMat q q) (β := RMat q q) (γ := RMat q k')))
      (volume : Measure ((RMat q q × RMat q q) × RMat q k'))
      (volume : Measure (RMat q q × (RMat q q × RMat q k'))) :=
    MeasureTheory.volume_preserving_prodAssoc.comp
      ((Measure.measurePreserving_swap (μ := (volume : Measure (RMat q q)))
        (ν := (volume : Measure (RMat q q)))).prod
        (MeasurePreserving.id (volume : Measure (RMat q k'))))
  have hXpre2 : MeasurePreserving
      (((MeasurableEquiv.prodComm (α := RMat d' q) (β := RMat d' q)).prodCongr
          (MeasurableEquiv.refl (RMat d' k'))).trans
        (MeasurableEquiv.prodAssoc (α := RMat d' q) (β := RMat d' q) (γ := RMat d' k')))
      (volume : Measure ((RMat d' q × RMat d' q) × RMat d' k'))
      (volume : Measure (RMat d' q × (RMat d' q × RMat d' k'))) :=
    MeasureTheory.volume_preserving_prodAssoc.comp
      ((Measure.measurePreserving_swap (μ := (volume : Measure (RMat d' q)))
        (ν := (volume : Measure (RMat d' q)))).prod
        (MeasurePreserving.id (volume : Measure (RMat d' k'))))
  have heX_sep := (MeasureTheory.volume_preserving_prodAssoc.symm.comp
    (measurePreserving_fourSwap (A := RMat q q) (B := RMat q q × RMat q k')
      (C := RMat d' q) (D := RMat d' q × RMat d' k'))).comp (hXpre1.prod hXpre2)
  have heY_sep := measurePreserving_fourSwap (A := RMat q q × RMat q d')
    (B := RMat q q × RMat q d') (C := RMat k' q) (D := RMat k' d')
  have heInter := measurePreserving_fourSwap
    (A := (RMat q q × RMat d' q) × (RMat q q × RMat q k'))
    (B := RMat d' q × RMat d' k')
    (C := (RMat q q × RMat q d') × RMat k' q)
    (D := (RMat q q × RMat q d') × RMat k' d')
  have heP := ((MeasurePreserving.id (volume : Measure ((RMat q q × RMat d' q) × (RMat q q × RMat q d')))).prod
      MeasureTheory.volume_preserving_prodAssoc).comp
    (measurePreserving_fourSwap (A := RMat q q × RMat d' q) (B := RMat q q × RMat q k')
      (C := RMat q q × RMat q d') (D := RMat k' q))
  have heA := (MeasureTheory.volume_preserving_prodAssoc.prod
      (Measure.measurePreserving_swap (μ := (volume : Measure (RMat k' d')))
        (ν := (volume : Measure (RMat d' k'))))).comp
    ((measurePreserving_fourSwap (A := RMat q q × RMat q d') (B := RMat k' d')
      (C := RMat d' q) (D := RMat d' k')).comp
      (Measure.measurePreserving_swap (μ := (volume : Measure (RMat d' q × RMat d' k')))
        (ν := (volume : Measure ((RMat q q × RMat q d') × RMat k' d')))))
  exact (heP.prod heA).comp (heInter.comp ((heX_sep.prod heY_sep).comp (heX.prod heY)))

private theorem twoSidedMatToBlocks_mem_box {q d' k' : ℕ} {ρ : ℝ}
    (p : RMat (q + d') ((q + q) + k') × RMat ((q + q) + k') (q + d')) :
    twoSidedMatToBlocks q d' k' p ∈ passiveBox ρ ×ˢ activeBox ρ ↔ p ∈ matBox ρ ×ˢ matBox ρ := by
  rcases p with ⟨X, Y⟩
  have hX_split : ∀ M : RMat (q + d') ((q + q) + k'),
      M ∈ matBox ρ ↔
      let r := splitRowsEquiv q d' ((q + q) + k') M
      let c1 := splitColsEquiv q (q + q) k' r.1
      let c2 := splitColsEquiv d' (q + q) k' r.2
      (splitColsEquiv q q q c1.1).1 ∈ matBox ρ ∧ (splitColsEquiv q q q c1.1).2 ∈ matBox ρ ∧
      c1.2 ∈ matBox ρ ∧
      (splitColsEquiv d' q q c2.1).1 ∈ matBox ρ ∧ (splitColsEquiv d' q q c2.1).2 ∈ matBox ρ ∧
      c2.2 ∈ matBox ρ := fun M => by
    rw [← splitRowsEquiv_mem_matBox]
    simp only [Set.mem_prod]
    rw [← splitColsEquiv_mem_matBox (M := (splitRowsEquiv q d' ((q + q) + k') M).1),
      ← splitColsEquiv_mem_matBox (M := (splitRowsEquiv q d' ((q + q) + k') M).2)]
    simp only [Set.mem_prod]
    rw [← splitColsEquiv_mem_matBox (M := (splitColsEquiv q (q + q) k' (splitRowsEquiv q d' ((q + q) + k') M).1).1),
      ← splitColsEquiv_mem_matBox (M := (splitColsEquiv d' (q + q) k' (splitRowsEquiv q d' ((q + q) + k') M).2).1)]
    simp only [Set.mem_prod]; tauto
  have hY_split : ∀ M : RMat ((q + q) + k') (q + d'),
      M ∈ matBox ρ ↔
      let r := splitRowsEquiv (q + q) k' (q + d') M
      let r1 := splitRowsEquiv q q (q + d') r.1
      (splitColsEquiv q q d' r1.1).1 ∈ matBox ρ ∧ (splitColsEquiv q q d' r1.1).2 ∈ matBox ρ ∧
      (splitColsEquiv q q d' r1.2).1 ∈ matBox ρ ∧ (splitColsEquiv q q d' r1.2).2 ∈ matBox ρ ∧
      (splitColsEquiv k' q d' r.2).1 ∈ matBox ρ ∧ (splitColsEquiv k' q d' r.2).2 ∈ matBox ρ := fun M => by
    rw [← splitRowsEquiv_mem_matBox]
    simp only [Set.mem_prod]
    rw [← splitRowsEquiv_mem_matBox (M := (splitRowsEquiv (q + q) k' (q + d') M).1),
      ← splitColsEquiv_mem_matBox (M := (splitRowsEquiv (q + q) k' (q + d') M).2)]
    simp only [Set.mem_prod]
    rw [← splitColsEquiv_mem_matBox (M := (splitRowsEquiv q q (q + d') (splitRowsEquiv (q + q) k' (q + d') M).1).1),
      ← splitColsEquiv_mem_matBox (M := (splitRowsEquiv q q (q + d') (splitRowsEquiv (q + q) k' (q + d') M).1).2)]
    simp only [Set.mem_prod]; tauto
  change (((((splitColsEquiv q q q (splitColsEquiv q (q + q) k' (splitRowsEquiv q d' ((q + q) + k') X).1).1).2,
            (splitColsEquiv d' q q (splitColsEquiv d' (q + q) k' (splitRowsEquiv q d' ((q + q) + k') X).2).1).2),
           ((splitColsEquiv q q d' (splitRowsEquiv q q (q + d') (splitRowsEquiv (q + q) k' (q + d') Y).1).1).1,
            (splitColsEquiv q q d' (splitRowsEquiv q q (q + d') (splitRowsEquiv (q + q) k' (q + d') Y).1).1).2)),
          ((splitColsEquiv q q q (splitColsEquiv q (q + q) k' (splitRowsEquiv q d' ((q + q) + k') X).1).1).1,
           ((splitColsEquiv q (q + q) k' (splitRowsEquiv q d' ((q + q) + k') X).1).2,
            (splitColsEquiv k' q d' (splitRowsEquiv (q + q) k' (q + d') Y).2).1))),
         (((splitColsEquiv q q d' (splitRowsEquiv q q (q + d') (splitRowsEquiv (q + q) k' (q + d') Y).1).2).1,
           ((splitColsEquiv q q d' (splitRowsEquiv q q (q + d') (splitRowsEquiv (q + q) k' (q + d') Y).1).2).2,
            (splitColsEquiv d' q q (splitColsEquiv d' (q + q) k' (splitRowsEquiv q d' ((q + q) + k') X).2).1).1)),
          ((splitColsEquiv d' (q + q) k' (splitRowsEquiv q d' ((q + q) + k') X).2).2,
           (splitColsEquiv k' q d' (splitRowsEquiv (q + q) k' (q + d') Y).2).2))) ∈
      passiveBox ρ ×ˢ activeBox ρ ↔ (X, Y) ∈ matBox ρ ×ˢ matBox ρ
  simp only [Set.mem_prod, hX_split X, hY_split Y, passiveBox, activeBox]
  tauto

/-- Canonical base matrices for the balanced two-sided `(q, q)` product `(X₀ + X) (Y₀ + Y)`:
`X₀` has `I_q` in rows `0..q-1`, cols `q..2q-1`; `Y₀` has `I_q` in rows `0..q-1`, cols `0..q-1`. -/
private def canonTwoSidedX0 (q d' k' : ℕ) : Matrix (Fin (q + d')) (Fin ((q + q) + k')) ℝ :=
  Matrix.of fun i r => if i.1 < q ∧ r.1 = q + i.1 then 1 else 0

private def canonTwoSidedY0 (q d' k' : ℕ) : Matrix (Fin ((q + q) + k')) (Fin (q + d')) ℝ :=
  Matrix.of fun r j => if r.1 < q ∧ r.1 = j.1 then 1 else 0

private theorem twoSidedMatToBlocks_apply {q d' k' : ℕ}
    (X : RMat (q + d') ((q + q) + k')) (Y : RMat ((q + q) + k') (q + d')) :
    twoSidedMatToBlocks q d' k' (X, Y) =
      (((((splitColsEquiv q q q (splitColsEquiv q (q + q) k' (splitRowsEquiv q d' ((q + q) + k') X).1).1).2,
          (splitColsEquiv d' q q (splitColsEquiv d' (q + q) k' (splitRowsEquiv q d' ((q + q) + k') X).2).1).2),
         ((splitColsEquiv q q d' (splitRowsEquiv q q (q + d') (splitRowsEquiv (q + q) k' (q + d') Y).1).1).1,
          (splitColsEquiv q q d' (splitRowsEquiv q q (q + d') (splitRowsEquiv (q + q) k' (q + d') Y).1).1).2)),
        ((splitColsEquiv q q q (splitColsEquiv q (q + q) k' (splitRowsEquiv q d' ((q + q) + k') X).1).1).1,
         ((splitColsEquiv q (q + q) k' (splitRowsEquiv q d' ((q + q) + k') X).1).2,
          (splitColsEquiv k' q d' (splitRowsEquiv (q + q) k' (q + d') Y).2).1))),
       (((splitColsEquiv q q d' (splitRowsEquiv q q (q + d') (splitRowsEquiv (q + q) k' (q + d') Y).1).2).1,
         ((splitColsEquiv q q d' (splitRowsEquiv q q (q + d') (splitRowsEquiv (q + q) k' (q + d') Y).1).2).2,
          (splitColsEquiv d' q q (splitColsEquiv d' (q + q) k' (splitRowsEquiv q d' ((q + q) + k') X).2).1).1)),
        ((splitColsEquiv d' (q + q) k' (splitRowsEquiv q d' ((q + q) + k') X).2).2,
         (splitColsEquiv k' q d' (splitRowsEquiv (q + q) k' (q + d') Y).2).2))) := rfl

private theorem twoSidedMat_loss_eq {q d' k' : ℕ}
    (X : RMat (q + d') ((q + q) + k')) (Y : RMat ((q + q) + k') (q + d')) :
    frobSq ((canonTwoSidedX0 q d' k' + Matrix.of X) * (canonTwoSidedY0 q d' k' + Matrix.of Y)) =
      twoSidedLoss (twoSidedMatToBlocks q d' k' (X, Y)) := by
  rw [twoSidedMatToBlocks_apply]
  unfold frobSq twoSidedLoss
  rw [Fin.sum_univ_add]
  simp_rw [Fin.sum_univ_add]
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib]
  have hsum3 : ∀ (f : Fin ((q + q) + k') → ℝ),
      ∑ r : Fin ((q + q) + k'), f r =
      (∑ r : Fin q, f (Fin.castAdd k' (Fin.castAdd q r))) +
      (∑ r : Fin q, f (Fin.castAdd k' (Fin.natAdd q r))) +
      (∑ r : Fin k', f (Fin.natAdd (q + q) r)) := fun f => by
    rw [Fin.sum_univ_add, Fin.sum_univ_add]
  have h4 : ∀ a b c d a' b' c' d' : ℝ,
      a = a' → b = b' → c = c' → d = d' → (a + b) + (c + d) = ((a' + b') + c') + d' := by
    intro _ _ _ _ _ _ _ _ h1 h2 h3 h4; rw [h1, h2, h3, h4]; ring
  refine h4 _ _ _ _ _ _ _ _ ?_ ?_ ?_ ?_
  · -- Block 11: `i ∈ Fin q`, `j ∈ Fin q`
    refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
    congr 1
    simp only [Matrix.mul_apply, Matrix.add_apply, Matrix.of_apply, Matrix.one_apply,
      canonTwoSidedX0, canonTwoSidedY0, splitRowsEquiv_apply_fst, splitRowsEquiv_apply_snd,
      splitColsEquiv_apply_fst, splitColsEquiv_apply_snd]
    rw [hsum3]
    congr 1
    · congr 1
      · refine Finset.sum_congr rfl fun r _ => ?_
        have h1 : ¬ (r.1 = q + i.1) := by omega
        simp only [Fin.val_castAdd, h1, and_false, ↓reduceIte, zero_add, r.2, true_and, Fin.val_inj]
      · refine Finset.sum_congr rfl fun r _ => ?_
        have h1 : (Fin.castAdd d' i).1 < q := i.2
        have h2 : ((Fin.castAdd k' (Fin.natAdd q r)).1 = q + (Fin.castAdd d' i).1) ↔ i = r := by
          simp [Fin.ext_iff]; omega
        have h3 : ¬ (Fin.castAdd k' (Fin.natAdd q r)).1 < q := by simp
        simp only [h1, true_and, h2, h3, false_and, ↓reduceIte, zero_add]
    · refine Finset.sum_congr rfl fun r _ => ?_
      have h1 : ¬ ((Fin.natAdd (q + q) r).1 = q + (Fin.castAdd d' i).1) := by simp; omega
      have h2 : ¬ (Fin.natAdd (q + q) r).1 < q := by simp; omega
      simp only [h1, and_false, ↓reduceIte, zero_add, h2, false_and]
  · -- Block 12: `i ∈ Fin q`, `j ∈ Fin d'`
    refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
    congr 1
    simp only [Matrix.mul_apply, Matrix.add_apply, Matrix.of_apply, Matrix.one_apply,
      canonTwoSidedX0, canonTwoSidedY0, splitRowsEquiv_apply_fst, splitRowsEquiv_apply_snd,
      splitColsEquiv_apply_fst, splitColsEquiv_apply_snd]
    rw [hsum3]
    congr 1
    · congr 1
      · refine Finset.sum_congr rfl fun r _ => ?_
        have h1 : ¬ ((Fin.castAdd k' (Fin.castAdd q r)).1 = q + (Fin.castAdd d' i).1) := by
          simp; omega
        have h2 : ¬ ((Fin.castAdd k' (Fin.castAdd q r)).1 = (Fin.natAdd q j).1) := by
          simp; omega
        simp only [h1, h2, and_false, ↓reduceIte, zero_add]
      · refine Finset.sum_congr rfl fun r _ => ?_
        have h1 : (Fin.castAdd d' i).1 < q := i.2
        have h2 : ((Fin.castAdd k' (Fin.natAdd q r)).1 = q + (Fin.castAdd d' i).1) ↔ i = r := by
          simp [Fin.ext_iff]; omega
        have h3 : ¬ (Fin.castAdd k' (Fin.natAdd q r)).1 < q := by simp
        simp only [h1, true_and, h2, h3, false_and, ↓reduceIte, zero_add]
    · refine Finset.sum_congr rfl fun r _ => ?_
      have h1 : ¬ ((Fin.natAdd (q + q) r).1 = q + (Fin.castAdd d' i).1) := by simp; omega
      have h2 : ¬ (Fin.natAdd (q + q) r).1 < q := by simp; omega
      simp only [h1, and_false, ↓reduceIte, zero_add, h2, false_and]
  · -- Block 21: `i ∈ Fin d'`, `j ∈ Fin q`
    refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
    congr 1
    simp only [Matrix.mul_apply, Matrix.add_apply, Matrix.of_apply, Matrix.one_apply,
      canonTwoSidedX0, canonTwoSidedY0, splitRowsEquiv_apply_fst, splitRowsEquiv_apply_snd,
      splitColsEquiv_apply_fst, splitColsEquiv_apply_snd]
    rw [hsum3]
    congr 1
    · congr 1
      · refine Finset.sum_congr rfl fun r _ => ?_
        have h1 : ¬ (Fin.natAdd q i).1 < q := by simp
        simp only [h1, false_and, ↓reduceIte, zero_add, Fin.val_castAdd, r.2, true_and, Fin.val_inj]
      · refine Finset.sum_congr rfl fun r _ => ?_
        have h1 : ¬ (Fin.natAdd q i).1 < q := by simp
        have h2 : ¬ (Fin.castAdd k' (Fin.natAdd q r)).1 < q := by simp
        simp only [h1, h2, false_and, ↓reduceIte, zero_add]
    · refine Finset.sum_congr rfl fun r _ => ?_
      have h1 : ¬ (Fin.natAdd q i).1 < q := by simp
      have h2 : ¬ (Fin.natAdd (q + q) r).1 < q := by simp; omega
      simp only [h1, h2, false_and, ↓reduceIte, zero_add]
  · -- Block 22: `i ∈ Fin d'`, `j ∈ Fin d'`
    refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
    congr 1
    simp only [Matrix.mul_apply, Matrix.add_apply, Matrix.of_apply,
      canonTwoSidedX0, canonTwoSidedY0, splitRowsEquiv_apply_fst, splitRowsEquiv_apply_snd,
      splitColsEquiv_apply_fst, splitColsEquiv_apply_snd]
    rw [hsum3]
    congr 1
    · congr 1
      · refine Finset.sum_congr rfl fun r _ => ?_
        have h1 : ¬ (Fin.natAdd q i).1 < q := by simp
        have h2 : ¬ ((Fin.castAdd k' (Fin.castAdd q r)).1 = (Fin.natAdd q j).1) := by
          simp; omega
        simp only [h1, false_and, h2, and_false, ↓reduceIte, zero_add]
      · refine Finset.sum_congr rfl fun r _ => ?_
        have h1 : ¬ (Fin.natAdd q i).1 < q := by simp
        have h2 : ¬ (Fin.castAdd k' (Fin.natAdd q r)).1 < q := by simp
        simp only [h1, h2, false_and, ↓reduceIte, zero_add]
    · refine Finset.sum_congr rfl fun r _ => ?_
      have h1 : ¬ (Fin.natAdd q i).1 < q := by simp
      have h2 : ¬ (Fin.natAdd (q + q) r).1 < q := by simp; omega
      simp only [h1, h2, false_and, ↓reduceIte, zero_add]

/-- Linearized `Q+K` loss has the exact learning pair of a balanced two-sided `(q, q)` fiber. -/
private theorem attnQKLin_hasLaplacePair {d H q : ℕ} (hqH : q ≤ H) (hqd : q ≤ d) :
    ∃ ρ₀ : ℝ, 0 < ρ₀ ∧ ∀ ρ : ℝ, 0 < ρ → ρ ≤ ρ₀ →
      SLT.HasLaplacePair
        (SLT.Z (volume.restrict (attnPairBox d H ρ)) (attnQKLinLoss d H q))
        (((q : ℝ) * q + 2 * q * (d - q)) / 2 + awLambda (d - q) (d - q) (2 * H - 2 * q))
        (awMult (d - q) (d - q) (2 * H - 2 * q)) := by
  have h2H : H + H = (q + q) + (2 * H - 2 * q) := by omega
  have hd : d = q + (d - q) := by omega
  let σ : Fin (H + H) ≃ Fin (H + H) := interleavePerm hqH
  -- Map `((B_K, ΔA_Q, ΔA_K), B_Q) → ((B_K, ΔA_Qᵀ), (ΔA_K, B_Qᵀ)) → (X, Y) → TwoSidedPassive × TwoSidedActive`.
  let ePre : AttnPairParam d H ≃ᵐ (RMat d H × RMat d H) × (RMat H d × RMat H d) :=
    -- `((BK, AQ, AK), BQ) → ((BK, AQ), (AK, BQ)) → ((BK, AQᵀ), (AK, BQᵀ))`
    (MeasurableEquiv.prodAssoc.symm.prodCongr (MeasurableEquiv.refl _) |>.trans
      MeasurableEquiv.prodAssoc).trans
    (((MeasurableEquiv.refl _).prodCongr (transposeEquiv H d)).prodCongr
     ((MeasurableEquiv.refl _).prodCongr (transposeEquiv d H)))
  let eX : (RMat d H × RMat d H) ≃ᵐ RMat (q + (d - q)) ((q + q) + (2 * H - 2 * q)) :=
    (splitColsEquiv d H H).symm.trans ((permColsEquiv σ).trans (castMatEquiv hd h2H))
  let eY : (RMat H d × RMat H d) ≃ᵐ RMat ((q + q) + (2 * H - 2 * q)) (q + (d - q)) :=
    (splitRowsEquiv H H d).symm.trans ((permRowsEquiv σ).trans (castMatEquiv h2H hd))
  let eTot : AttnPairParam d H ≃ᵐ
      TwoSidedPassive q (d - q) (2 * H - 2 * q) × TwoSidedActive q (d - q) (2 * H - 2 * q) :=
    ePre.trans ((eX.prodCongr eY).trans (twoSidedMatToBlocks q (d - q) (2 * H - 2 * q)))
  have heTot_mp : MeasurePreserving eTot volume volume := by
    have hePre : MeasurePreserving ePre volume volume :=
      measurePreserving_trans
        (measurePreserving_trans
          (measurePreserving_prodCongr MeasureTheory.volume_preserving_prodAssoc.symm
            (MeasurePreserving.id (volume : Measure (RMat d H))))
          MeasureTheory.volume_preserving_prodAssoc)
        (measurePreserving_prodCongr
          (measurePreserving_prodCongr (MeasurePreserving.id (volume : Measure (RMat d H)))
            (measurePreserving_transposeEquiv H d))
          (measurePreserving_prodCongr (MeasurePreserving.id (volume : Measure (RMat H d)))
            (measurePreserving_transposeEquiv d H)))
    have heX_mp : MeasurePreserving eX volume volume :=
      measurePreserving_trans (measurePreserving_splitColsEquiv d H H).symm
        (measurePreserving_trans (measurePreserving_permColsEquiv σ) (measurePreserving_castMatEquiv hd h2H))
    have heY_mp : MeasurePreserving eY volume volume :=
      measurePreserving_trans (measurePreserving_splitRowsEquiv H H d).symm
        (measurePreserving_trans (measurePreserving_permRowsEquiv σ) (measurePreserving_castMatEquiv h2H hd))
    exact measurePreserving_trans hePre
      (measurePreserving_trans (measurePreserving_prodCongr heX_mp heY_mp)
        (measurePreserving_twoSidedMatToBlocks q (d - q) (2 * H - 2 * q)))
  have heTot_val : ∀ BK AQ AK BQ,
      eTot ((BK, AQ, AK), BQ) =
        twoSidedMatToBlocks q (d - q) (2 * H - 2 * q)
          (eX (BK, transposeEquiv H d AQ), eY (AK, transposeEquiv d H BQ)) := by
    intro BK AQ AK BQ
    dsimp only [eTot]
    rw [MeasurableEquiv.trans_apply, MeasurableEquiv.trans_apply]
    have hpre : ePre ((BK, AQ, AK), BQ) =
        ((BK, transposeEquiv H d AQ), (AK, transposeEquiv d H BQ)) := rfl
    rw [hpre, prodCongr_apply_mk]
  have heTot_box : ∀ {ρ : ℝ} (θ : AttnPairParam d H),
      eTot θ ∈ passiveBox (q := q) (d' := d - q) (k' := 2 * H - 2 * q) ρ ×ˢ
        activeBox (q := q) (d' := d - q) (k' := 2 * H - 2 * q) ρ ↔
      θ ∈ attnPairBox d H ρ := fun {ρ} θ => by
    rcases θ with ⟨⟨BK, AQ, AK⟩, BQ⟩
    rw [heTot_val, twoSidedMatToBlocks_mem_box]
    simp only [eX, eY, MeasurableEquiv.trans_apply, Set.mem_prod,
      castMatEquiv_mem_matBox, permColsEquiv_mem_matBox, permRowsEquiv_mem_matBox]
    have hX : (splitColsEquiv d H H).symm (BK, transposeEquiv H d AQ) ∈ matBox ρ ↔
        BK ∈ matBox ρ ∧ AQ ∈ matBox ρ := by
      rw [← splitColsEquiv_mem_matBox, MeasurableEquiv.apply_symm_apply]
      simp only [Set.mem_prod, transposeEquiv_mem_matBox]
    have hY : (splitRowsEquiv H H d).symm (AK, transposeEquiv d H BQ) ∈ matBox ρ ↔
        AK ∈ matBox ρ ∧ BQ ∈ matBox ρ := by
      rw [← splitRowsEquiv_mem_matBox, MeasurableEquiv.apply_symm_apply]
      simp only [Set.mem_prod, transposeEquiv_mem_matBox]
    rw [hX, hY]
    simp only [attnPairBox, Set.mem_prod]
    tauto
  have heTot_loss : ∀ θ : AttnPairParam d H, attnQKLinLoss d H q θ = twoSidedLoss (eTot θ) := by
    intro ⟨⟨BK, AQ, AK⟩, BQ⟩
    have h1 : twoSidedLoss (eTot ((BK, AQ, AK), BQ)) =
        frobSq ((canonTwoSidedX0 q (d - q) (2 * H - 2 * q) +
          Matrix.of (eX (BK, transposeEquiv H d AQ))) *
         (canonTwoSidedY0 q (d - q) (2 * H - 2 * q) +
          Matrix.of (eY (AK, transposeEquiv d H BQ)))) := by
      rw [heTot_val]
      exact (twoSidedMat_loss_eq (eX (BK, transposeEquiv H d AQ)) (eY (AK, transposeEquiv d H BQ))).symm
    rw [h1]
    unfold attnQKLinLoss frobSq
    refine Finset.sum_equiv (finCongr hd) (fun _ => by simp) fun i _ => ?_
    refine Finset.sum_equiv (finCongr hd) (fun _ => by simp) fun j _ => ?_
    congr 1
    simp only [Matrix.add_apply, Matrix.mul_apply, Matrix.of_apply, Matrix.transpose_apply]
    have hcast_sum :
        ∑ r : Fin ((q + q) + (2 * H - 2 * q)),
          (canonTwoSidedX0 q (d - q) (2 * H - 2 * q) (finCongr hd i) r +
            eX (BK, transposeEquiv H d AQ) (finCongr hd i) r) *
          (canonTwoSidedY0 q (d - q) (2 * H - 2 * q) r (finCongr hd j) +
            eY (AK, transposeEquiv d H BQ) r (finCongr hd j)) =
        ∑ r : Fin (H + H),
          ((if i.1 < q ∧ (σ r).1 = q + i.1 then (1 : ℝ) else 0) +
            (splitColsEquiv d H H).symm (BK, transposeEquiv H d AQ) i r) *
          ((if (σ r).1 < q ∧ (σ r).1 = j.1 then (1 : ℝ) else 0) +
            (splitRowsEquiv H H d).symm (AK, transposeEquiv d H BQ) r j) := by
      symm
      refine Finset.sum_equiv (σ.trans (finCongr h2H)) (fun _ => by simp) fun r _ => ?_
      have hXval : eX (BK, transposeEquiv H d AQ) (finCongr hd i) ((σ.trans (finCongr h2H)) r) =
          (splitColsEquiv d H H).symm (BK, transposeEquiv H d AQ) i r := by
        dsimp only [eX, MeasurableEquiv.trans_apply, Equiv.trans_apply]
        have hca := castMatEquiv_apply hd h2H
          (permColsEquiv σ ((splitColsEquiv d H H).symm (BK, transposeEquiv H d AQ))) i (σ r)
        simpa only [permColsEquiv, permRowsEquiv, transposeEquiv, MeasurableEquiv.trans_apply,
          MeasurableEquiv.coe_mk, Equiv.coe_fn_mk] using
          hca.trans (congrFun (Equiv.piCongrLeft_apply_apply (fun _ : Fin (H + H) => (Fin d → ℝ))
            σ (fun c r' => (splitColsEquiv d H H).symm (BK, transposeEquiv H d AQ) r' c) r) i)
      have hYval : eY (AK, transposeEquiv d H BQ) ((σ.trans (finCongr h2H)) r) (finCongr hd j) =
          (splitRowsEquiv H H d).symm (AK, transposeEquiv d H BQ) r j := by
        dsimp only [eY, MeasurableEquiv.trans_apply, Equiv.trans_apply]
        have hca := castMatEquiv_apply h2H hd
          (permRowsEquiv σ ((splitRowsEquiv H H d).symm (AK, transposeEquiv d H BQ))) (σ r) j
        exact hca.trans (congrFun (Equiv.piCongrLeft_apply_apply (fun _ : Fin (H + H) => (Fin d → ℝ))
          σ ((splitRowsEquiv H H d).symm (AK, transposeEquiv d H BQ)) r) j)
      have hX0b : canonTwoSidedX0 q (d - q) (2 * H - 2 * q) (finCongr hd i) ((σ.trans (finCongr h2H)) r) =
          if i.1 < q ∧ (σ r).1 = q + i.1 then (1 : ℝ) else 0 := by
        simp [canonTwoSidedX0]
      have hY0b : canonTwoSidedY0 q (d - q) (2 * H - 2 * q) ((σ.trans (finCongr h2H)) r) (finCongr hd j) =
          if (σ r).1 < q ∧ (σ r).1 = j.1 then (1 : ℝ) else 0 := by
        simp [canonTwoSidedY0]
      rw [hXval, hYval, hX0b, hY0b]
    rw [hcast_sum, Fin.sum_univ_add]
    congr 1
    · refine Finset.sum_congr rfl fun r _ => ?_
      have hX0 : (splitColsEquiv d H H).symm (BK, transposeEquiv H d AQ) i (Fin.castAdd H r) = BK i r :=
        splitColsEquiv_symm_apply_castAdd BK (transposeEquiv H d AQ) i r
      have hY0 : (splitRowsEquiv H H d).symm (AK, transposeEquiv d H BQ) (Fin.castAdd H r) j = AK r j :=
        splitRowsEquiv_symm_apply_castAdd AK (transposeEquiv d H BQ) r j
      have hbX : (if i.1 < q ∧ (σ (Fin.castAdd H r)).1 = q + i.1 then (1 : ℝ) else 0) = 0 := by
        have hnot : ¬ (i.1 < q ∧ (σ (Fin.castAdd H r)).1 = q + i.1) := by
          by_cases h1 : r.1 < q
          · rw [interleavePerm_castAdd_lt hqH r h1]; omega
          · rw [interleavePerm_castAdd_ge hqH r h1]; omega
        simp only [hnot, ↓reduceIte]
      have hbY : (if (σ (Fin.castAdd H r)).1 < q ∧ (σ (Fin.castAdd H r)).1 = j.1 then (1 : ℝ) else 0) =
          baseAQK d H q r j := by
        unfold baseAQK
        simp only [Matrix.of_apply]
        by_cases h1 : r.1 < q
        · rw [interleavePerm_castAdd_lt hqH r h1]
        · rw [interleavePerm_castAdd_ge hqH r h1]
          have hnot : ¬ (q + r.1 < q) := by omega
          simp only [hnot, false_and, ↓reduceIte, h1]
      rw [hX0, hY0, hbX, hbY, zero_add]
    · refine Finset.sum_congr rfl fun r _ => ?_
      have hX1 : (splitColsEquiv d H H).symm (BK, transposeEquiv H d AQ) i (Fin.natAdd H r) = AQ r i :=
        splitColsEquiv_symm_apply_natAdd BK (transposeEquiv H d AQ) i r
      have hY1 : (splitRowsEquiv H H d).symm (AK, transposeEquiv d H BQ) (Fin.natAdd H r) j = BQ j r :=
        splitRowsEquiv_symm_apply_natAdd AK (transposeEquiv d H BQ) r j
      have hbX : (if i.1 < q ∧ (σ (Fin.natAdd H r)).1 = q + i.1 then (1 : ℝ) else 0) =
          baseAQK d H q r i := by
        unfold baseAQK
        simp only [Matrix.of_apply]
        by_cases h3 : r.1 < q
        · rw [interleavePerm_natAdd_lt hqH r h3]
          have hiff : (i.1 < q ∧ q + r.1 = q + i.1) ↔ r.1 = i.1 := by omega
          simp only [hiff, h3, true_and]
        · rw [interleavePerm_natAdd_ge hqH r h3]
          have hnot : ¬ (i.1 < q ∧ H + r.1 = q + i.1) := by omega
          simp only [hnot, ↓reduceIte, h3, false_and]
      have hbY : (if (σ (Fin.natAdd H r)).1 < q ∧ (σ (Fin.natAdd H r)).1 = j.1 then (1 : ℝ) else 0) = 0 := by
        have hnot : ¬ ((σ (Fin.natAdd H r)).1 < q) := by
          by_cases h3 : r.1 < q
          · rw [interleavePerm_natAdd_lt hqH r h3]; omega
          · rw [interleavePerm_natAdd_ge hqH r h3]; omega
        simp only [hnot, false_and, ↓reduceIte]
      rw [hX1, hY1, hbX, hbY, zero_add]
  -- Apply `twoSided_schur_reduction`!
  obtain ⟨ρ₀, hρ₀, hts⟩ := twoSided_schur_reduction q (d - q) (2 * H - 2 * q)
  refine ⟨ρ₀, hρ₀, fun ρ hρ hρle => ?_⟩
  have hmeas_attn : MeasurableSet (attnPairBox d H ρ) :=
    ((measurableSet_matBox ρ).prod ((measurableSet_matBox ρ).prod (measurableSet_matBox ρ))).prod
      (measurableSet_matBox ρ)
  have hmeas_ts : MeasurableSet
      (passiveBox (q := q) (d' := d - q) (k' := 2 * H - 2 * q) ρ ×ˢ
        activeBox (q := q) (d' := d - q) (k' := 2 * H - 2 * q) ρ) :=
    ((((measurableSet_matBox ρ).prod (measurableSet_matBox ρ)).prod
      ((measurableSet_matBox ρ).prod (measurableSet_matBox ρ))).prod
     ((measurableSet_matBox ρ).prod ((measurableSet_matBox ρ).prod (measurableSet_matBox ρ)))).prod
    (((measurableSet_matBox ρ).prod ((measurableSet_matBox ρ).prod (measurableSet_matBox ρ))).prod
      ((measurableSet_matBox ρ).prod (measurableSet_matBox ρ)))
  have hmeas_L : Measurable (twoSidedLoss (q := q) (d' := d - q) (k' := 2 * H - 2 * q)) := by
    unfold twoSidedLoss frobSq
    simp only [Matrix.add_apply, Matrix.mul_apply, Matrix.of_apply]
    fun_prop
  have hZeq : ∀ N : ℝ,
      SLT.Z (volume.restrict (attnPairBox d H ρ)) (attnQKLinLoss d H q) N =
      SLT.Z (volume.restrict (passiveBox (q := q) (d' := d - q) (k' := 2 * H - 2 * q) ρ ×ˢ
        activeBox (q := q) (d' := d - q) (k' := 2 * H - 2 * q) ρ)) twoSidedLoss N := fun N =>
    Z_eq_of_measurePreserving eTot heTot_mp (attnPairBox d H ρ) _ hmeas_attn hmeas_ts
      heTot_box (attnQKLinLoss d H q) twoSidedLoss hmeas_L heTot_loss N
  have hpair := (hts ρ hρ hρle).congr (Filter.Eventually.of_forall fun N => (hZeq N).symm)
  rwa [Nat.cast_sub hqd] at hpair

/-- **Joint `Q+K` Self-Attention LoRA Exact Learning Pair (`[LC]`, exact to all orders):**
For ALL `d, H, q : ℕ` with `q ≤ H` and `q ≤ d`, the exact all-orders `Q+K` loss
`‖(I_d + B_Q A_Q)ᵀ (I_d + B_K A_K) - I_d‖_F²` on `[-δ, δ]^{4 d H}` has exact learning pair
`((q² + 2 q (d - q)) / 2 + awLambda (d - q) (d - q) (2*H - 2*q), awMult (d - q) (d - q) (2*H - 2*q))`. -/
theorem attention_QK_exact_pair {d H q : ℕ} (hqH : q ≤ H) (hqd : q ≤ d) :
    ∃ δ₀ : ℝ, 0 < δ₀ ∧ ∀ δ : ℝ, 0 < δ → δ ≤ δ₀ →
      SLT.HasLaplacePair
        (SLT.Z (volume.restrict (attnPairBox d H δ)) (attnQKLoss d H q))
        (((q : ℝ) * q + 2 * q * (d - q)) / 2 + awLambda (d - q) (d - q) (2 * H - 2 * q))
        (awMult (d - q) (d - q) (2 * H - 2 * q)) := by
  let B_P : ℝ → Set (RMat d H × RMat H d × RMat H d) := fun ρ =>
    matBox ρ ×ˢ matBox ρ ×ˢ matBox ρ
  let M : (RMat d H × RMat H d × RMat H d) → Matrix (Fin d) (Fin d) ℝ := fun p =>
    (1 + Matrix.of p.1 * (baseAQK d H q + Matrix.of p.2.2))ᵀ
  have hM_cont : Continuous M := by
    refine continuous_pi fun i => continuous_pi fun j => ?_
    simp only [M, Matrix.transpose_apply, Matrix.add_apply, Matrix.mul_apply, Matrix.of_apply]
    fun_prop
  have hM0 : M 0 = 1 := by
    ext i j
    simp [M, Matrix.transpose_apply, Matrix.mul_apply, Matrix.one_apply]
  have hL_raw_meas : Measurable (attnQKLoss d H q) := by
    unfold attnQKLoss frobSq
    simp only [Matrix.sub_apply, Matrix.mul_apply, Matrix.transpose_apply, Matrix.add_apply, Matrix.of_apply]
    fun_prop
  have hL_lin_meas : Measurable (attnQKLinLoss d H q) := by
    unfold attnQKLinLoss frobSq
    simp only [Matrix.add_apply, Matrix.mul_apply, Matrix.transpose_apply, Matrix.of_apply]
    fun_prop
  have heq : ∀ p B, attnQKLoss d H q (p, B) = attnQKLinLoss d H q (p, toRMat (M p * Matrix.of B)) := by
    intro p B
    unfold attnQKLoss attnQKLinLoss
    have hto : Matrix.of (toRMat (M p * Matrix.of B)) = M p * Matrix.of B := rfl
    rw [hto]
    exact congrArg frobSq (attnQK_factor (Matrix.of B) (Matrix.of p.1)
      (baseAQK d H q + Matrix.of p.2.1) (baseAQK d H q + Matrix.of p.2.2))
  exact hasLaplacePair_of_leftMul_reduction B_P
    (fun ρ => (measurableSet_matBox ρ).prod ((measurableSet_matBox ρ).prod (measurableSet_matBox ρ)))
    (fun hle => Set.prod_mono (matBox_mono hle) (Set.prod_mono (matBox_mono hle) (matBox_mono hle)))
    (fun U hU => tripleBox_nhds hU)
    M hM_cont hM0 (attnQKLoss d H q) (attnQKLinLoss d H q) hL_raw_meas hL_lin_meas heq
    (attnQKLin_hasLaplacePair hqH hqd)

/-- **Complete `Q+K` Flat Cancellation Theorem (`[LC]`):**
For ALL `d, H, q : ℕ` with `q ≤ H ≤ d`, the exact all-orders `Q+K` learning pair is **identically
flat** (independent of `q`!) across the entire fiber `q ∈ {0, …, H}`:
`HasLaplacePair Z_{QK}(d, H, q, δ) (d * H - H² / 2) 1`.
The supercritical parabola is completely annihilated by the exact `GL(q)` gauge symmetry! -/
theorem attention_QK_flat_pair {d H q : ℕ} (hqH : q ≤ H) (hHd : H ≤ d) :
    ∃ δ₀ : ℝ, 0 < δ₀ ∧ ∀ δ : ℝ, 0 < δ → δ ≤ δ₀ →
      SLT.HasLaplacePair
        (SLT.Z (volume.restrict (attnPairBox d H δ)) (attnQKLoss d H q))
        ((d : ℝ) * H - (H : ℝ) ^ 2 / 2) 1 := by
  obtain ⟨δ₀, hδ₀, h⟩ := attention_QK_exact_pair (d := d) hqH (hqH.trans hHd)
  refine ⟨δ₀, hδ₀, fun δ hδ hδle => ?_⟩
  have hpair := h δ hδ hδle
  have hsq := awLambda_sq_even (d - q) (H - q) (by omega)
  have h2H : 2 * (H - q) = 2 * H - 2 * q := by omega
  rw [h2H] at hsq
  have hc1 : ((d - q : ℕ) : ℝ) = (d : ℝ) - (q : ℝ) := Nat.cast_sub (hqH.trans hHd)
  have hc2 : ((H - q : ℕ) : ℝ) = (H : ℝ) - (q : ℝ) := Nat.cast_sub hqH
  rw [hsq.1, hsq.2, hc1, hc2] at hpair
  have hlam : ((q : ℝ) * q + 2 * q * ((d : ℝ) - q)) / 2 +
      (((d : ℝ) - q) * ((H : ℝ) - q) - ((H : ℝ) - q) ^ 2 / 2) =
      (d : ℝ) * H - (H : ℝ) ^ 2 / 2 := by ring
  rw [hlam] at hpair
  exact hpair

/-! ### 7. Side-by-side Standard LoRA Initialization Coupling Dichotomy -/

/-- **Self-Attention Coupling Dichotomy at Standard LoRA Initialization (`q = H`, `2 * H ≤ d`) (`[LC]`):**
At standard LoRA initialization (`B₀ = 0`, `rank(A₀) = H`) on a single-head self-attention block:
- Joint **`V+O`** sits at `(d * H, 1)` — a macroscopic jump `+H² / 2` (`4×` the single-projection jump)
  above the subcritical plateau `d * H - H² / 2`;
- Joint **`Q+K`** sits at `(d * H - H² / 2, 1)` — jump `0`, completely flat! -/
theorem attention_coupling_dichotomy_standard_init {d H : ℕ} (h2Hd : 2 * H ≤ d) :
    (∃ δ₀ : ℝ, 0 < δ₀ ∧ ∀ δ : ℝ, 0 < δ → δ ≤ δ₀ →
      SLT.HasLaplacePair
        (SLT.Z (volume.restrict (attnPairBox d H δ)) (attnVOLoss d H H))
        ((d : ℝ) * H) 1) ∧
    (∃ δ₀ : ℝ, 0 < δ₀ ∧ ∀ δ : ℝ, 0 < δ → δ ≤ δ₀ →
      SLT.HasLaplacePair
        (SLT.Z (volume.restrict (attnPairBox d H δ)) (attnQKLoss d H H))
        ((d : ℝ) * H - (H : ℝ) ^ 2 / 2) 1) := by
  constructor
  · obtain ⟨δ₀, hδ₀, h⟩ := attention_VO_supercritical_parabola (d := d) (H := H) (q := H)
      le_rfl (by omega) h2Hd
    refine ⟨δ₀, hδ₀, fun δ hδ hδle => ?_⟩
    have hlam : (d : ℝ) * H - (H : ℝ) ^ 2 / 2 + 2 * ((H : ℝ) - (H : ℝ) / 2) ^ 2 = (d : ℝ) * H := by
      ring
    rw [hlam] at h
    exact h δ hδ hδle
  · exact attention_QK_flat_pair le_rfl (by omega)

end EatYourFiber

