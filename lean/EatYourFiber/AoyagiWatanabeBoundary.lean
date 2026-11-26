/-
Copyright (c) 2026 Bobby Faber. All rights reserved.
Released under Apache 2.0 license.
-/
module

public import EatYourFiber.LoRASchurLaplace
public import Mathlib.Analysis.SpecialFunctions.Gaussian.GaussianIntegral
public import Mathlib.Analysis.InnerProductSpace.Projection.Basic
public import Mathlib.Analysis.InnerProductSpace.PiL2
public import Mathlib.Analysis.InnerProductSpace.Orthonormal
public import Mathlib.MeasureTheory.Measure.Lebesgue.VolumeOfBalls

/-!
# Aoyagi–Watanabe (2005) Boundary Regimes and the Unconditional Supercritical LoRA Parabola

This module proves, completely unconditionally in Lean 4 on top of `SLT` (zero resolution of
singularities):

1. `rank0_laplaceLower_kn`: Universal lower bound `Z(N) ≳ N^{-kn/2}` for `‖B A‖_F^2` for all
   `m, k, n : ℕ`.
2. `lintegral_gaussian_cols_le`: Exact inductive Gaussian integral bound over `a : Fin k → ℝ`
   using orthogonal projection (`Submodule.starProjection`) and translation invariance, giving
   `∫ exp(-N ‖∑ a_r b_r‖²) da ≤ (π / N)^{k/2} · orthHeight(B)⁻¹`.
3. `lintegral_orthHeight_inv_pow_lt_top`: Elementary finiteness of
   `∫_{[-1,1]^{k × m}} orthHeight(B)^{-n} dB < ∞` whenever `k + n ≤ m`, proved via orthogonal
   invariance (`LinearIsometryEquiv.measurePreserving`) and the AM–GM cube bound
   `‖v‖₂ ≥ (∏ |v_i|)^{1/d}` (no polar coordinates needed).
4. `rank0_hasLaplacePair_of_le_m`: Unconditional Aoyagi–Watanabe boundary theorem for `k + n ≤ m`:
   `HasLaplacePair (Z rank0Loss) (k * n / 2) 1`.
5. `rank0_hasLaplacePair_of_le_n`: Symmetric boundary theorem for `k + m ≤ n` by matrix transpose:
   `HasLaplacePair (Z rank0Loss) (k * m / 2) 1`.
6. `lora_supercritical_parabola_exact_pair`: **Unconditional Lean-certified supercritical LoRA
   parabola** for all `m, n, k, q : ℕ` with `q ≤ k`, `q ≤ n`, `k < m + n`, `n + k ≤ m + 2 * q`:
   exact local learning pair `(awLambdaEven m n k + (q - q_c)^2 / 2, 1)`.
-/

@[expose] public section

namespace EatYourFiber

open MeasureTheory Set Filter Matrix Finset Real
open scoped ENNReal NNReal

/-! ### 1. Universal lower bound `λ ≤ k * n / 2` for `rank0Loss` -/

/-- Universal Laplace lower bound: for ALL `m, k, n : ℕ`, the rank-0 matrix factorization integral
satisfies `Z(N) ≥ c · N^{-k*n/2}` eventually. -/
theorem rank0_laplaceLower_kn (m k n : ℕ) :
    SLT.LaplaceLower
      (SLT.Z (volume.restrict (matBox (a := m) (b := k) 1 ×ˢ matBox (a := k) (b := n) 1))
        rank0Loss)
      ((k : ℝ) * n / 2) 1 := by
  rw [SLT.laplaceLower_iff]
  let Cexp : ℝ := Real.exp (-((m : ℝ) * (k : ℝ) ^ 2 * (n : ℝ)))
  let c : ℝ := Cexp * (2 : ℝ) ^ (m * k + k * n)
  refine ⟨c, by positivity, 1, fun N hN => ?_⟩
  have hNpos : 0 < N := by linarith
  let ρ : ℝ := N ^ (-(1 / 2 : ℝ))
  have hρpos : 0 < ρ := Real.rpow_pos_of_pos hNpos _
  have hρle1 : ρ ≤ 1 := Real.rpow_le_one_of_one_le_of_nonpos hN (by norm_num)
  have hρsq : ρ ^ 2 = N⁻¹ := by
    dsimp [ρ]
    rw [← Real.rpow_natCast, ← Real.rpow_mul hNpos.le]
    norm_num
    exact Real.rpow_neg_one N
  have hsub :
      matBox (a := m) (b := k) 1 ×ˢ matBox (a := k) (b := n) ρ ⊆
      matBox (a := m) (b := k) 1 ×ˢ matBox (a := k) (b := n) 1 := by
    rintro ⟨B, A⟩ ⟨hB, hA⟩
    refine ⟨hB, ?_⟩
    rw [mem_matBox_iff] at hA ⊢
    intro i j; exact (hA i j).trans hρle1
  have hbound : ∀ p ∈ matBox (a := m) (b := k) 1 ×ˢ matBox (a := k) (b := n) ρ,
      ENNReal.ofReal Cexp ≤ ENNReal.ofReal (Real.exp (-(N * rank0Loss p))) := by
    rintro ⟨B, A⟩ ⟨hB, hA⟩
    rw [mem_matBox_iff] at hB hA
    apply ENNReal.ofReal_le_ofReal
    apply Real.exp_le_exp.mpr
    have hfB : frobSq (Matrix.of B) ≤ (m : ℝ) * k := by
      unfold frobSq
      calc ∑ i : Fin m, ∑ j : Fin k, (B i j) ^ 2
          ≤ ∑ _ : Fin m, ∑ _ : Fin k, (1 : ℝ) := by
            refine Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ => ?_
            have := pow_le_pow_left₀ (abs_nonneg (B i j)) (hB i j) 2
            rw [sq_abs] at this; linarith
        _ = (m : ℝ) * k := by simp [Finset.sum_const]
    have hfA : frobSq (Matrix.of A) ≤ (k : ℝ) * n * N⁻¹ := by
      unfold frobSq
      calc ∑ i : Fin k, ∑ j : Fin n, (A i j) ^ 2
          ≤ ∑ _ : Fin k, ∑ _ : Fin n, ρ ^ 2 := by
            refine Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ => ?_
            have := pow_le_pow_left₀ (abs_nonneg (A i j)) (hA i j) 2
            rw [sq_abs] at this; exact this
        _ = (k : ℝ) * n * N⁻¹ := by simp [Finset.sum_const, hρsq, mul_assoc]
    have hloss : rank0Loss (B, A) ≤ (m : ℝ) * k ^ 2 * n * N⁻¹ := by
      unfold rank0Loss
      calc frobSq (Matrix.of B * Matrix.of A)
          ≤ frobSq (Matrix.of B) * frobSq (Matrix.of A) := frobSq_mul_le _ _
        _ ≤ ((m : ℝ) * k) * ((k : ℝ) * n * N⁻¹) :=
            mul_le_mul hfB hfA (frobSq_nonneg _) (by positivity)
        _ = (m : ℝ) * k ^ 2 * n * N⁻¹ := by ring
    have hNloss : N * rank0Loss (B, A) ≤ (m : ℝ) * k ^ 2 * n := by
      calc N * rank0Loss (B, A)
          ≤ N * ((m : ℝ) * k ^ 2 * n * N⁻¹) := mul_le_mul_of_nonneg_left hloss hNpos.le
        _ = (m : ℝ) * k ^ 2 * n := by field_simp
    linarith
  have hvol :
      volume (matBox (a := m) (b := k) 1 ×ˢ matBox (a := k) (b := n) ρ) =
      ENNReal.ofReal ((2 : ℝ) ^ (m * k + k * n) * N ^ (-((k : ℝ) * n / 2))) := by
    change (volume.prod volume) (matBox (a := m) (b := k) 1 ×ˢ matBox (a := k) (b := n) ρ) = _
    rw [Measure.prod_prod, volume_matBox (by norm_num), volume_matBox hρpos.le]
    have hpow : (2 * ρ) ^ (k * n) = (2 : ℝ) ^ (k * n) * N ^ (-((k : ℝ) * n / 2)) := by
      rw [mul_pow]
      congr 1
      dsimp [ρ]
      rw [← Real.rpow_natCast, ← Real.rpow_mul hNpos.le]
      congr 1
      push_cast; ring
    rw [← ENNReal.ofReal_pow (by positivity), ← ENNReal.ofReal_pow (by positivity),
      ← ENNReal.ofReal_mul (by positivity), mul_one, hpow]
    congr 1; rw [pow_add]; ring
  calc ENNReal.ofReal (c * N ^ (-((k : ℝ) * n / 2)) * Real.log N ^ (1 - 1))
      = ENNReal.ofReal Cexp *
          volume (matBox (a := m) (b := k) 1 ×ˢ matBox (a := k) (b := n) ρ) := by
        rw [hvol, ← ENNReal.ofReal_mul (by positivity)]
        congr 1; dsimp [c]; simp [mul_assoc]
    _ = ∫⁻ _ in matBox (a := m) (b := k) 1 ×ˢ matBox (a := k) (b := n) ρ, ENNReal.ofReal Cexp := by
        rw [setLIntegral_const]
    _ ≤ ∫⁻ p in matBox (a := m) (b := k) 1 ×ˢ matBox (a := k) (b := n) ρ,
          ENNReal.ofReal (Real.exp (-(N * rank0Loss p))) :=
        setLIntegral_mono' ((measurableSet_matBox 1).prod (measurableSet_matBox ρ)) hbound
    _ ≤ SLT.Z (volume.restrict (matBox (a := m) (b := k) 1 ×ˢ matBox (a := k) (b := n) 1))
          rank0Loss N :=
        lintegral_mono_set hsub

/-! ### 2. Matrix transpose & column decoupling -/

/-- Uncurrying `RMat a b → (Fin a × Fin b → ℝ)` preserves Lebesgue measure. -/
theorem measurePreserving_uncurry (a b : ℕ) :
    MeasurePreserving (fun (X : RMat a b) (p : Fin a × Fin b) => X p.1 p.2) volume volume := by
  have hm : Measurable (fun (X : RMat a b) (p : Fin a × Fin b) => X p.1 p.2) := by
    refine measurable_pi_iff.mpr fun p => ?_
    exact (measurable_pi_apply p.2).comp (measurable_pi_apply p.1)
  refine ⟨hm, ?_⟩
  symm
  refine Measure.pi_eq fun s hs => ?_
  rw [Measure.map_apply hm (MeasurableSet.univ_pi hs)]
  have hpre : (fun (X : RMat a b) (p : Fin a × Fin b) => X p.1 p.2) ⁻¹' Set.pi Set.univ s =
      Set.pi Set.univ fun i => Set.pi Set.univ fun j => s (i, j) := by
    ext X; simp only [Set.mem_preimage, Set.mem_pi, Set.mem_univ, forall_const, Prod.forall]
  rw [hpre]
  change (Measure.pi fun _ : Fin a => (Measure.pi fun _ : Fin b => (volume : Measure ℝ))) _ = _
  rw [Measure.pi_pi]
  simp_rw [Measure.pi_pi]
  exact (Fintype.prod_prod_type (fun p : Fin a × Fin b => (volume : Measure ℝ) (s p))).symm

/-- Matrix transpose `X ↦ Xᵀ` on `RMat a b → RMat b a` preserves Lebesgue measure. -/
theorem measurePreserving_transpose (a b : ℕ) :
    MeasurePreserving (fun X : RMat a b => (fun (j : Fin b) (i : Fin a) => X i j : RMat b a))
      volume volume := by
  let e1 : RMat a b ≃ᵐ (Fin a × Fin b → ℝ) := (MeasurableEquiv.curry (Fin a) (Fin b) ℝ).symm
  let e2 : (Fin a × Fin b → ℝ) ≃ᵐ (Fin b × Fin a → ℝ) :=
    MeasurableEquiv.piCongrLeft (fun _ => ℝ) (Equiv.prodComm (Fin a) (Fin b))
  let e3 : RMat b a ≃ᵐ (Fin b × Fin a → ℝ) := (MeasurableEquiv.curry (Fin b) (Fin a) ℝ).symm
  have h1 : MeasurePreserving e1 volume volume := measurePreserving_uncurry a b
  have h2 : MeasurePreserving e2 volume volume :=
    measurePreserving_piCongrLeft (fun _ : Fin b × Fin a => (volume : Measure ℝ))
      (Equiv.prodComm (Fin a) (Fin b))
  have h3 : MeasurePreserving e3.symm volume volume := (measurePreserving_uncurry b a).symm e3
  have hcomp : ((⇑(e1.trans (e2.trans e3.symm))) : RMat a b → RMat b a) =
      fun X j i => X i j := by
    funext X j i; rfl
  rw [← hcomp]
  exact h3.comp (h2.comp h1)

/-- Product-space integral of a separable product over `Fin d → α` factors as a power. -/
private theorem lintegral_pi_prod_eq_pow {α : Type*} [MeasureSpace α]
    [SigmaFinite (volume : Measure α)] (s : Set α) (hs : MeasurableSet s)
    (f : α → ℝ≥0∞) (hf : Measurable f) :
    ∀ d : ℕ,
      ∫⁻ x in Set.pi Set.univ (fun _ : Fin d => s), ∏ i : Fin d, f (x i) =
      (∫⁻ t in s, f t) ^ d
  | 0 => by
    simp only [Finset.univ_eq_empty, Finset.prod_empty, setLIntegral_const, one_mul, pow_zero]
    change (Measure.pi fun _ : Fin 0 => (volume : Measure α))
      (Set.pi Set.univ fun _ : Fin 0 => s) = 1
    rw [Measure.pi_pi]; simp
  | d + 1 => by
    let e : (Fin (d + 1) → α) ≃ᵐ α × (Fin d → α) :=
      MeasurableEquiv.piFinSuccAbove (fun _ => α) 0
    have hmp : MeasurePreserving e volume volume :=
      volume_preserving_piFinSuccAbove (fun _ => α) 0
    let box_d : Set (Fin d → α) := Set.pi Set.univ fun _ => s
    have hpre : e ⁻¹' (s ×ˢ box_d) = Set.pi Set.univ fun _ : Fin (d + 1) => s := by
      ext x
      simp only [Set.mem_preimage, Set.mem_prod, Set.mem_pi, Set.mem_univ, forall_const, e, box_d,
        MeasurableEquiv.piFinSuccAbove_apply, Fin.insertNthEquiv_symm_apply]
      constructor
      · rintro ⟨h0, htail⟩ i
        refine Fin.cases h0 (fun j => ?_) i
        simpa [Fin.tail] using htail j
      · intro hx
        exact ⟨hx 0, fun j => by simpa [Fin.tail] using hx (Fin.succ j)⟩
    have hprod_pt : ∀ x : Fin (d + 1) → α,
        (∏ i : Fin (d + 1), f (x i)) = f (e x).1 * ∏ j : Fin d, f ((e x).2 j) := fun x => by
      rw [Fin.prod_univ_succ]; rfl
    have hres : (volume : Measure (α × (Fin d → α))).restrict (s ×ˢ box_d) =
        Measure.map e (volume.restrict (Set.pi Set.univ fun _ : Fin (d + 1) => s)) := by
      rw [← hmp.map_eq, Measure.restrict_map e.measurable (hs.prod (MeasurableSet.univ_pi fun _ => hs)),
        hpre]
    have hm_tail : Measurable fun v : Fin d → α => ∏ j : Fin d, f (v j) :=
      Finset.measurable_prod _ fun j _ => hf.comp (measurable_pi_apply j)
    have hm_prod : Measurable fun p : α × (Fin d → α) => f p.1 * ∏ j : Fin d, f (p.2 j) :=
      (hf.comp measurable_fst).mul (hm_tail.comp measurable_snd)
    have hstep :
        ∫⁻ x in Set.pi Set.univ (fun _ : Fin (d + 1) => s), ∏ i : Fin (d + 1), f (x i) =
        ∫⁻ p : α × (Fin d → α), f p.1 * ∏ j : Fin d, f (p.2 j)
          ∂((volume.restrict s).prod (volume.restrict box_d)) := by
      rw [Measure.prod_restrict]
      change _ = ∫⁻ p in s ×ˢ box_d, f p.1 * ∏ j : Fin d, f (p.2 j) ∂volume
      rw [hres, lintegral_map hm_prod e.measurable]
      refine setLIntegral_congr_fun (MeasurableSet.univ_pi fun _ => hs) fun x _ => ?_
      exact hprod_pt x
    rw [hstep, lintegral_prod_mul hf.aemeasurable hm_tail.aemeasurable,
      lintegral_pi_prod_eq_pow s hs f hf d, pow_succ']

/-- Column decoupling: for fixed `B : RMat m k`, integrating `exp(-N ‖B A‖_F²)` over
`A ∈ [-1,1]^{k × n}` equals the `n`-th power of the single-column integral over `a ∈ [-1,1]^k`. -/
theorem col_decouple {m k n : ℕ} (B : RMat m k) (N : ℝ) :
    ∫⁻ A in matBox (a := k) (b := n) 1,
      ENNReal.ofReal (Real.exp (-(N * rank0Loss (B, A)))) =
    (∫⁻ a in Set.pi Set.univ (fun _ : Fin k => Set.Icc (-1 : ℝ) 1),
      ENNReal.ofReal (Real.exp (-(N * ∑ i : Fin m, (∑ r : Fin k, B i r * a r) ^ 2)))) ^ n := by
  let tr : RMat k n → RMat n k := fun A j r => A r j
  have hmp : MeasurePreserving tr volume volume := measurePreserving_transpose k n
  have htr_meas : Measurable tr := hmp.measurable
  have hpre : tr ⁻¹' matBox (a := n) (b := k) 1 = matBox (a := k) (b := n) 1 := by
    ext A; simp only [Set.mem_preimage, mem_matBox_iff, tr]
    exact ⟨fun h r j => h j r, fun h j r => h r j⟩
  let g : (Fin k → ℝ) → ℝ≥0∞ := fun a =>
    ENNReal.ofReal (Real.exp (-(N * ∑ i : Fin m, (∑ r : Fin k, B i r * a r) ^ 2)))
  have hg : Measurable g := by unfold g; fun_prop
  have hpt : ∀ A : RMat k n,
      ENNReal.ofReal (Real.exp (-(N * rank0Loss (B, A)))) = ∏ j : Fin n, g (tr A j) := fun A => by
    unfold rank0Loss frobSq g
    simp only [Matrix.mul_apply, Matrix.of_apply, tr]
    rw [Finset.sum_comm, Finset.mul_sum, ← Finset.sum_neg_distrib, Real.exp_sum,
      ENNReal.ofReal_prod_of_nonneg (fun _ _ => (Real.exp_pos _).le)]
  simp_rw [hpt]
  have hres : (volume : Measure (RMat n k)).restrict (matBox (a := n) (b := k) 1) =
      Measure.map tr (volume.restrict (matBox (a := k) (b := n) 1)) := by
    rw [← hmp.map_eq, Measure.restrict_map htr_meas (measurableSet_matBox 1), hpre]
  have hprod_meas : Measurable fun X : RMat n k => ∏ j : Fin n, g (X j) :=
    Finset.measurable_prod _ fun j _ => hg.comp (measurable_pi_apply j)
  have hstep :
      ∫⁻ A in matBox (a := k) (b := n) 1, ∏ j : Fin n, g (tr A j) =
      ∫⁻ X in matBox (a := n) (b := k) 1, ∏ j : Fin n, g (X j) := by
    rw [hres, lintegral_map hprod_meas htr_meas]
  rw [hstep]
  exact lintegral_pi_prod_eq_pow (Set.pi Set.univ fun _ : Fin k => Set.Icc (-1 : ℝ) 1)
    (MeasurableSet.univ_pi fun _ => measurableSet_Icc) g hg n

/-! ### 3. Elementary cube power-singularity integrability `∫_{[-R,R]^d} (∑ v_j^2)^{-n/2} < ∞` -/

/-- One-dimensional symmetric power integral on `[-R, R]` for `0 ≤ α < 1`. -/
theorem lintegral_Icc_abs_rpow_neg {R α : ℝ} (hR : 0 ≤ R) (_hα0 : 0 ≤ α) (hα1 : α < 1) :
    ∫⁻ x in Set.Icc (-R) R, ENNReal.ofReal (|x| ^ (-α)) ≤
      ENNReal.ofReal (2 * R ^ (1 - α) / (1 - α)) := by
  have h1 : ∫⁻ x in Set.Ioc 0 R, ENNReal.ofReal (x ^ (-α)) =
      ENNReal.ofReal (R ^ (1 - α) / (1 - α)) := by
    have h := SLT.NC.lintegral_Ioc_zero_rpow hR (show (-1 : ℝ) < -α by linarith)
    simpa [sub_eq_add_neg, add_comm] using h
  have hpos : ∫⁻ x in Set.Ioc 0 R, ENNReal.ofReal (|x| ^ (-α)) =
      ENNReal.ofReal (R ^ (1 - α) / (1 - α)) := by
    rw [← h1]
    refine setLIntegral_congr_fun measurableSet_Ioc fun x hx => ?_
    rw [abs_of_pos hx.1]
  have hneg : ∫⁻ x in Set.Ico (-R) 0, ENNReal.ofReal (|x| ^ (-α)) =
      ENNReal.ofReal (R ^ (1 - α) / (1 - α)) := by
    have hmp := Measure.measurePreserving_neg (volume : Measure ℝ)
    have hpre : (fun x : ℝ => -x) ⁻¹' Set.Ioc 0 R = Set.Ico (-R) 0 := by
      ext x; simp only [Set.mem_preimage, Set.mem_Ioc, Set.mem_Ico]
      constructor <;> rintro ⟨a, b⟩ <;> exact ⟨by linarith, by linarith⟩
    have hres : (volume : Measure ℝ).restrict (Set.Ioc 0 R) =
        Measure.map (fun x : ℝ => -x) (volume.restrict (Set.Ico (-R) 0)) := by
      conv_lhs => rw [← hmp.map_eq]
      rw [Measure.restrict_map measurable_neg measurableSet_Ioc, hpre]
    rw [← hpos, hres, lintegral_map (by fun_prop) measurable_neg]
    refine setLIntegral_congr_fun measurableSet_Ico fun x _ => ?_
    simp [abs_neg]
  have hcov : Set.Icc (-R) R ⊆ Set.Ico (-R) 0 ∪ ({0} ∪ Set.Ioc 0 R) := by
    intro x ⟨hx1, hx2⟩
    rcases lt_trichotomy x 0 with hlt | rfl | hgt
    · exact Or.inl ⟨hx1, hlt⟩
    · exact Or.inr (Or.inl rfl)
    · exact Or.inr (Or.inr ⟨hgt, hx2⟩)
  have hzero0 : ∫⁻ x in ({0} : Set ℝ), ENNReal.ofReal (|x| ^ (-α)) = 0 :=
    setLIntegral_measure_zero _ _ (by simp)
  calc ∫⁻ x in Set.Icc (-R) R, ENNReal.ofReal (|x| ^ (-α))
      ≤ ∫⁻ x in Set.Ico (-R) 0 ∪ ({0} ∪ Set.Ioc 0 R), ENNReal.ofReal (|x| ^ (-α)) :=
        lintegral_mono_set hcov
    _ ≤ (∫⁻ x in Set.Ico (-R) 0, ENNReal.ofReal (|x| ^ (-α))) +
          ((∫⁻ x in ({0} : Set ℝ), ENNReal.ofReal (|x| ^ (-α))) +
            ∫⁻ x in Set.Ioc 0 R, ENNReal.ofReal (|x| ^ (-α))) :=
        (lintegral_union_le _ _ _).trans (add_le_add le_rfl (lintegral_union_le _ _ _))
    _ = ENNReal.ofReal (R ^ (1 - α) / (1 - α)) +
          (0 + ENNReal.ofReal (R ^ (1 - α) / (1 - α))) := by
        rw [hneg, hpos, hzero0]
    _ = ENNReal.ofReal (2 * R ^ (1 - α) / (1 - α)) := by
        rw [zero_add, ← ENNReal.ofReal_add (by positivity) (by positivity)]
        congr 1; ring

/-- Almost every vector in `Fin d → ℝ` has all coordinates nonzero. -/
private theorem ae_all_ne_zero (d : ℕ) :
    ∀ᵐ v : Fin d → ℝ ∂volume, ∀ j : Fin d, v j ≠ 0 := by
  rw [ae_all_iff]
  intro j
  have hzero : volume {v : Fin d → ℝ | v j = 0} = 0 := by
    have hset : {v : Fin d → ℝ | v j = 0} =
        Set.pi Set.univ (fun i : Fin d => if i = j then ({0} : Set ℝ) else Set.univ) := by
      ext v
      simp only [Set.mem_ofPred_eq, Set.mem_pi, Set.mem_univ, forall_const]
      constructor
      · intro hv i; split_ifs with hij
        · subst hij; exact hv
        · trivial
      · intro h; simpa using h j
    rw [hset]
    change (Measure.pi fun _ : Fin d => (volume : Measure ℝ)) _ = 0
    rw [Measure.pi_pi]
    exact Finset.prod_eq_zero (Finset.mem_univ j) (by simp)
  exact measure_mono_null (fun v hv => by simpa using hv) hzero

/-- Elementary AM–GM cube bound: for `d > n` and `R ≥ 0`,
`∫_{[-R, R]^d} (√(∑ v_j^2))^{-n} dv ≤ (2 * R^{1 - n/d} / (1 - n/d))^d`. -/
theorem lintegral_cube_sum_sq_inv_pow_le {d n : ℕ} {R : ℝ} (hR : 0 ≤ R) (hdn : n < d) :
    ∫⁻ v in Set.pi Set.univ (fun _ : Fin d => Set.Icc (-R) R),
      (ENNReal.ofReal (Real.sqrt (∑ j : Fin d, (v j) ^ 2)))⁻¹ ^ n ≤
    ENNReal.ofReal (2 * R ^ (1 - (n : ℝ) / d) / (1 - (n : ℝ) / d)) ^ d := by
  have hdpos : (0 : ℝ) < d := Nat.cast_pos.mpr (by omega)
  have hα0 : (0 : ℝ) ≤ (n : ℝ) / d := by positivity
  have hα1 : (n : ℝ) / d < 1 := (div_lt_one hdpos).mpr (Nat.cast_lt.mpr hdn)
  have hae : ∀ᵐ v : Fin d → ℝ ∂(volume.restrict (Set.pi Set.univ fun _ : Fin d => Set.Icc (-R) R)),
      (ENNReal.ofReal (Real.sqrt (∑ j : Fin d, (v j) ^ 2)))⁻¹ ^ n ≤
      ∏ j : Fin d, ENNReal.ofReal (|v j| ^ (-((n : ℝ) / d))) := by
    filter_upwards [ae_restrict_of_ae (ae_all_ne_zero d)] with v hv
    have hsum_pos : 0 < ∑ j : Fin d, (v j) ^ 2 := by
      have h0 : Fin d := ⟨0, by omega⟩
      have hle : (v h0) ^ 2 ≤ ∑ j : Fin d, (v j) ^ 2 :=
        Finset.single_le_sum (fun j _ => sq_nonneg (v j)) (Finset.mem_univ h0)
      have hpos0 : 0 < (v h0) ^ 2 := sq_pos_of_ne_zero (hv h0)
      linarith
    have hsqrt_pos : 0 < Real.sqrt (∑ j : Fin d, (v j) ^ 2) := Real.sqrt_pos.mpr hsum_pos
    have hprod_le : (∏ j : Fin d, |v j|) ≤ (Real.sqrt (∑ j : Fin d, (v j) ^ 2)) ^ d := by
      calc ∏ j : Fin d, |v j|
          ≤ ∏ _ : Fin d, Real.sqrt (∑ j : Fin d, (v j) ^ 2) := by
            gcongr with j _
            rw [← Real.sqrt_sq_eq_abs]
            exact Real.sqrt_le_sqrt
              (Finset.single_le_sum (fun r _ => sq_nonneg (v r)) (Finset.mem_univ j))
        _ = (Real.sqrt (∑ j : Fin d, (v j) ^ 2)) ^ d := by simp [Finset.prod_const]
    have hprod_pos : 0 < ∏ j : Fin d, |v j| :=
      Finset.prod_pos fun j _ => abs_pos.mpr (hv j)
    have hreal : (Real.sqrt (∑ j : Fin d, (v j) ^ 2))⁻¹ ^ n ≤
        ∏ j : Fin d, |v j| ^ (-((n : ℝ) / d)) := by
      have hrhs : (∏ j : Fin d, |v j| ^ (-((n : ℝ) / d))) =
          (∏ j : Fin d, |v j|) ^ (-((n : ℝ) / d)) := by
        rw [Real.finsetProd_rpow _ _ (fun j _ => abs_nonneg _)]
      rw [hrhs]
      have hlhs : (Real.sqrt (∑ j : Fin d, (v j) ^ 2))⁻¹ ^ n =
          ((Real.sqrt (∑ j : Fin d, (v j) ^ 2)) ^ d) ^ (-((n : ℝ) / d)) := by
        rw [← Real.rpow_natCast (Real.sqrt _), ← Real.rpow_mul hsqrt_pos.le]
        have hexp : (d : ℝ) * (-((n : ℝ) / d)) = -(n : ℝ) := by field_simp
        rw [hexp, Real.rpow_neg hsqrt_pos.le, Real.rpow_natCast, inv_pow]
      rw [hlhs]
      exact Real.rpow_le_rpow_of_nonpos hprod_pos hprod_le (by linarith)
    rw [← ENNReal.ofReal_inv_of_pos hsqrt_pos, ← ENNReal.ofReal_pow (by positivity),
      ← ENNReal.ofReal_prod_of_nonneg (fun j _ => Real.rpow_nonneg (abs_nonneg _) _)]
    exact ENNReal.ofReal_le_ofReal hreal
  calc ∫⁻ v in Set.pi Set.univ (fun _ : Fin d => Set.Icc (-R) R),
        (ENNReal.ofReal (Real.sqrt (∑ j : Fin d, (v j) ^ 2)))⁻¹ ^ n
      ≤ ∫⁻ v in Set.pi Set.univ (fun _ : Fin d => Set.Icc (-R) R),
          ∏ j : Fin d, ENNReal.ofReal (|v j| ^ (-((n : ℝ) / d))) :=
        lintegral_mono_ae hae
    _ = (∫⁻ t in Set.Icc (-R) R, ENNReal.ofReal (|t| ^ (-((n : ℝ) / d)))) ^ d :=
        lintegral_pi_prod_eq_pow (Set.Icc (-R) R) measurableSet_Icc
          (fun t : ℝ => ENNReal.ofReal (|t| ^ (-((n : ℝ) / d)))) (by fun_prop) d
    _ ≤ ENNReal.ofReal (2 * R ^ (1 - (n : ℝ) / d) / (1 - (n : ℝ) / d)) ^ d :=
        pow_le_pow_left' (lintegral_Icc_abs_rpow_neg hR hα0 hα1) d

/-! ### 4. Orthogonal height & inductive Gaussian integral bound -/

/-- Orthogonal distance of `b : Fin m → ℝ` from `span (range C)` for `C : Fin k → Fin m → ℝ`,
defined as the norm of the orthogonal projection onto `Vᗮ` in `EuclideanSpace ℝ (Fin m)`. -/
noncomputable def orthDist {m k : ℕ} (C : Fin k → (Fin m → ℝ)) (b : Fin m → ℝ) : ℝ :=
  let toE : (Fin m → ℝ) → EuclideanSpace ℝ (Fin m) := WithLp.toLp 2
  let V : Submodule ℝ (EuclideanSpace ℝ (Fin m)) := Submodule.span ℝ (Set.range (fun i => toE (C i)))
  ‖Vᗮ.starProjection (toE b)‖

theorem orthDist_nonneg {m k : ℕ} (C : Fin k → (Fin m → ℝ)) (b : Fin m → ℝ) :
    0 ≤ orthDist C b := norm_nonneg _

/-- Countable-rational infimum characterization of `orthDist` (giving immediate measurability!). -/
theorem orthDist_eq_iInf_rat {m k : ℕ} (C : Fin k → (Fin m → ℝ)) (b : Fin m → ℝ) :
    ENNReal.ofReal (orthDist C b) =
    ⨅ (q : Fin k → ℚ), ENNReal.ofReal
      ‖WithLp.toLp 2 b - ∑ i : Fin k, ((q i : ℝ) • WithLp.toLp 2 (C i))‖ := by
  let toE : (Fin m → ℝ) → EuclideanSpace ℝ (Fin m) := WithLp.toLp 2
  let V : Submodule ℝ (EuclideanSpace ℝ (Fin m)) := Submodule.span ℝ (Set.range (fun i => toE (C i)))
  have hle : ∀ q : Fin k → ℚ,
      orthDist C b ≤ ‖toE b - ∑ i : Fin k, ((q i : ℝ) • toE (C i))‖ := fun q => by
    have hmem : (∑ i : Fin k, ((q i : ℝ) • toE (C i))) ∈ V :=
      Submodule.sum_mem V fun i _ => Submodule.smul_mem V _ (Submodule.subset_span ⟨i, rfl⟩)
    -- `Vᗮ.starProjection` annihilates `V` and is 1-Lipschitz (norm ≤ 1)!
    have hproj : Vᗮ.starProjection (toE b - ∑ i : Fin k, ((q i : ℝ) • toE (C i))) =
        Vᗮ.starProjection (toE b) := by
      simp only [map_sub, Submodule.starProjection_apply]
      rw [Submodule.orthogonalProjectionOnto_apply_of_mem_orthogonal (V.le_orthogonal_orthogonal hmem),
        Submodule.coe_zero, sub_zero]
    unfold orthDist
    change ‖Vᗮ.starProjection (toE b)‖ ≤ _
    rw [← hproj]
    exact Submodule.norm_starProjection_apply_le Vᗮ _
  refine le_antisymm (le_iInf fun q => ENNReal.ofReal_le_ofReal (hle q)) ?_
  -- Reverse inequality: approximate `V.starProjection (toE b) ∈ V` by rational combinations
  have hmemV : V.starProjection (toE b) ∈ V := Submodule.starProjection_apply_mem V (toE b)
  rw [Submodule.mem_span_range_iff_exists_fun] at hmemV
  obtain ⟨c, hc⟩ := hmemV
  have hdiff : toE b - ∑ i : Fin k, c i • toE (C i) = Vᗮ.starProjection (toE b) := by
    rw [hc]
    have hsum := Submodule.starProjection_add_starProjection_orthogonal (K := V) (toE b)
    exact sub_eq_of_eq_add' hsum.symm
  -- Use `ENNReal.le_of_forall_pos_le_add` via rational approximation of `c`
  refine ENNReal.le_of_forall_pos_le_add fun ε hε _ => ?_
  have hεR : 0 < (ε : ℝ) := NNReal.coe_pos.mpr hε
  let M : ℝ := 1 + ∑ i : Fin k, ‖toE (C i)‖
  have hMpos : 0 < M := by dsimp [M]; positivity
  have hq_ex : ∀ i : Fin k, ∃ qi : ℚ, |(qi : ℝ) - c i| < (ε : ℝ) / ((k + 1 : ℝ) * M) := fun i => by
    obtain ⟨qi, h1, h2⟩ := exists_rat_btwn (show c i - (ε : ℝ) / ((k + 1 : ℝ) * M) <
      c i + (ε : ℝ) / ((k + 1 : ℝ) * M) by linarith [show 0 < (ε : ℝ) / ((k + 1 : ℝ) * M) by positivity])
    exact ⟨qi, abs_sub_lt_iff.mpr ⟨by linarith, by linarith⟩⟩
  choose q hq using hq_ex
  refine (iInf_le _ q).trans ?_
  have htri :
      ‖toE b - ∑ i : Fin k, ((q i : ℝ) • toE (C i))‖ ≤ orthDist C b + (ε : ℝ) := by
    have hsplit : toE b - ∑ i : Fin k, ((q i : ℝ) • toE (C i)) =
        Vᗮ.starProjection (toE b) + ∑ i : Fin k, ((c i - (q i : ℝ)) • toE (C i)) := by
      rw [← hdiff]
      simp only [sub_smul, Finset.sum_sub_distrib]
      abel
    rw [hsplit]
    have hsum_le : ‖∑ i : Fin k, ((c i - (q i : ℝ)) • toE (C i))‖ ≤ (ε : ℝ) := by
      calc ‖∑ i : Fin k, ((c i - (q i : ℝ)) • toE (C i))‖
          ≤ ∑ i : Fin k, ‖(c i - (q i : ℝ)) • toE (C i)‖ := norm_sum_le _ _
        _ = ∑ i : Fin k, |c i - (q i : ℝ)| * ‖toE (C i)‖ := by
            simp_rw [norm_smul, Real.norm_eq_abs]
        _ ≤ ∑ i : Fin k, ((ε : ℝ) / ((k + 1 : ℝ) * M)) * ‖toE (C i)‖ := by
            refine Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_right ?_ (norm_nonneg _)
            rw [abs_sub_comm]; exact (hq i).le
        _ = ((ε : ℝ) / ((k + 1 : ℝ) * M)) * ∑ i : Fin k, ‖toE (C i)‖ := by
            rw [← Finset.mul_sum]
        _ ≤ ((ε : ℝ) / ((k + 1 : ℝ) * M)) * M := by
            refine mul_le_mul_of_nonneg_left ?_ (by positivity)
            dsimp [M]; linarith
        _ = (ε : ℝ) / (k + 1 : ℝ) := by field_simp
        _ ≤ (ε : ℝ) := div_le_self hεR.le (by linarith [Nat.cast_nonneg (α := ℝ) k])
    calc ‖Vᗮ.starProjection (toE b) + ∑ i : Fin k, ((c i - (q i : ℝ)) • toE (C i))‖
        ≤ ‖Vᗮ.starProjection (toE b)‖ + ‖∑ i : Fin k, ((c i - (q i : ℝ)) • toE (C i))‖ :=
          norm_add_le _ _
      _ ≤ orthDist C b + (ε : ℝ) := add_le_add le_rfl hsum_le
  calc ENNReal.ofReal ‖toE b - ∑ i : Fin k, ((q i : ℝ) • toE (C i))‖
      ≤ ENNReal.ofReal (orthDist C b + (ε : ℝ)) := ENNReal.ofReal_le_ofReal htri
    _ = ENNReal.ofReal (orthDist C b) + (ε : ℝ≥0∞) := by
        rw [ENNReal.ofReal_add (orthDist_nonneg C b) hεR.le, ENNReal.ofReal_coe_nnreal]

/-- Joint measurability of `(C, b) ↦ ENNReal.ofReal (orthDist C b)` via countable rational infimum. -/
theorem measurable_ofReal_orthDist (m k : ℕ) :
    Measurable (fun p : (Fin k → (Fin m → ℝ)) × (Fin m → ℝ) =>
      ENNReal.ofReal (orthDist p.1 p.2)) := by
  have heq : (fun p : (Fin k → (Fin m → ℝ)) × (Fin m → ℝ) => ENNReal.ofReal (orthDist p.1 p.2)) =
      fun p => ⨅ (q : Fin k → ℚ), ENNReal.ofReal
        ‖WithLp.toLp 2 p.2 - ∑ i : Fin k, ((q i : ℝ) • WithLp.toLp 2 (p.1 i))‖ := by
    funext p; exact orthDist_eq_iInf_rat p.1 p.2
  rw [heq]
  refine Measurable.iInf fun q => ?_
  have hcont : Continuous (fun p : (Fin k → (Fin m → ℝ)) × (Fin m → ℝ) =>
      ‖WithLp.toLp 2 p.2 - ∑ i : Fin k, ((q i : ℝ) • WithLp.toLp 2 (p.1 i))‖) := by
    have htoE : Continuous (WithLp.toLp 2 : (Fin m → ℝ) → EuclideanSpace ℝ (Fin m)) :=
      (WithLp.linearEquiv 2 ℝ (Fin m → ℝ)).symm.continuous_of_finiteDimensional
    refine Continuous.norm ?_
    refine (htoE.comp continuous_snd).sub ?_
    refine continuous_finsetSum _ fun i _ => ?_
    have hi : Continuous (fun p : (Fin k → (Fin m → ℝ)) × (Fin m → ℝ) => p.1 i) :=
      (continuous_apply i).comp continuous_fst
    exact (continuous_const (y := (q i : ℝ))).smul (htoE.comp hi)
  exact ENNReal.continuous_ofReal.measurable.comp hcont.measurable

/-- Recursive Gram–Schmidt orthogonal height `∏_{j=0}^{k-1} dist(b_j, span(b_0, ..., b_{j-1}))`
in `ℝ≥0∞`, for `k` column vectors `B : Fin k → Fin m → ℝ`. -/
noncomputable def orthHeight {m : ℕ} : ∀ {k : ℕ}, (Fin k → (Fin m → ℝ)) → ℝ≥0∞
  | 0, _ => 1
  | k + 1, B => orthHeight (Fin.init B) * ENNReal.ofReal (orthDist (Fin.init B) (B (Fin.last k)))

theorem orthHeight_ne_top {m : ℕ} : ∀ {k : ℕ} (B : Fin k → (Fin m → ℝ)), orthHeight B ≠ ⊤
  | 0, _ => by simp [orthHeight]
  | k + 1, B => ENNReal.mul_ne_top (orthHeight_ne_top (Fin.init B)) ENNReal.ofReal_ne_top

theorem measurable_orthHeight (m k : ℕ) :
    Measurable (fun B : Fin k → (Fin m → ℝ) => orthHeight B) := by
  induction k with
  | zero =>
    have heq : (fun B : Fin 0 → (Fin m → ℝ) => orthHeight B) = fun _ => 1 := funext fun _ => rfl
    rw [heq]
    exact measurable_const
  | succ k ih =>
    have hinit : Measurable (Fin.init : (Fin (k + 1) → (Fin m → ℝ)) → (Fin k → (Fin m → ℝ))) :=
      measurable_pi_iff.mpr fun i => measurable_pi_apply (Fin.castSucc i)
    have hlast : Measurable (fun B : Fin (k + 1) → (Fin m → ℝ) => B (Fin.last k)) :=
      measurable_pi_apply (Fin.last k)
    have h1 := ih.comp hinit
    have hpair : Measurable (fun B : Fin (k + 1) → (Fin m → ℝ) =>
        (Fin.init B, B (Fin.last k))) := hinit.prodMk hlast
    have h2 := (measurable_ofReal_orthDist m k).comp hpair
    have hmul := h1.mul h2
    have heq : (fun B : Fin (k + 1) → (Fin m → ℝ) => orthHeight B) =
        ((fun B : Fin k → (Fin m → ℝ) => orthHeight B) ∘ Fin.init) *
        ((fun p : (Fin k → (Fin m → ℝ)) × (Fin m → ℝ) => ENNReal.ofReal (orthDist p.1 p.2)) ∘
          fun B : Fin (k + 1) → (Fin m → ℝ) => (Fin.init B, B (Fin.last k))) :=
      funext fun _ => rfl
    rw [heq]
    exact hmul

/-- 1D Gaussian `lintegral`: `∫⁻ x : ℝ, exp(-(b * x^2)) = √(π / b)` for `b > 0`. -/
theorem lintegral_gaussian_1d {b : ℝ} (hb : 0 < b) :
    ∫⁻ x : ℝ, ENNReal.ofReal (Real.exp (-(b * x ^ 2))) = ENNReal.ofReal (Real.sqrt (Real.pi / b)) := by
  have h1 := integral_gaussian b
  have hint : Integrable (fun x : ℝ => Real.exp (-b * x ^ 2)) := integrable_exp_neg_mul_sq hb
  have h2 : ∀ x : ℝ, -(b * x ^ 2) = -b * x ^ 2 := fun x => by ring
  simp_rw [h2]
  rw [← ofReal_integral_eq_lintegral_ofReal hint
    (Eventually.of_forall fun x => (Real.exp_pos _).le), h1]

/-- **Inductive Gaussian column bound:** for every `m, k : ℕ`, `B : Fin k → Fin m → ℝ`, and `N > 0`:
`∫⁻ a : Fin k → ℝ, exp(-N ∑_i (∑_r a_r B_r i)^2) ≤ (√(π / N))^k · (orthHeight B)⁻¹`. -/
theorem lintegral_gaussian_cols_le {m : ℕ} :
    ∀ {k : ℕ} (B : Fin k → (Fin m → ℝ)) {N : ℝ}, 0 < N →
      ∫⁻ a : Fin k → ℝ,
        ENNReal.ofReal (Real.exp (-(N * ∑ i : Fin m, (∑ r : Fin k, B r i * a r) ^ 2))) ≤
      ENNReal.ofReal (Real.sqrt (Real.pi / N)) ^ k * (orthHeight B)⁻¹
  | 0, B, N, hN => by
    simp only [Finset.univ_eq_empty, Finset.sum_empty, sq, mul_zero, Finset.sum_const_zero,
      neg_zero, Real.exp_zero, ENNReal.ofReal_one, pow_zero, orthHeight, inv_one, mul_one,
      lintegral_const]
    have hvol : (volume : Measure (Fin 0 → ℝ)) Set.univ = 1 := by
      have : (Set.univ : Set (Fin 0 → ℝ)) = Set.pi Set.univ (fun _ : Fin 0 => Set.univ) := by ext; simp
      rw [this]
      change (Measure.pi fun _ : Fin 0 => (volume : Measure ℝ)) _ = 1
      rw [Measure.pi_pi]; simp
    rw [hvol, one_mul]
  | k + 1, B, N, hN => by
    let B0 : Fin k → (Fin m → ℝ) := Fin.init B
    let bLast : Fin m → ℝ := B (Fin.last k)
    let toE : (Fin m → ℝ) → EuclideanSpace ℝ (Fin m) := WithLp.toLp 2
    let V : Submodule ℝ (EuclideanSpace ℝ (Fin m)) :=
      Submodule.span ℝ (Set.range (fun i => toE (B0 i)))
    have hdist_nn : 0 ≤ orthDist B0 bLast := orthDist_nonneg B0 bLast
    -- If either factor of `orthHeight B` is 0, then `(orthHeight B)⁻¹ = ⊤` and the bound is trivial
    by_cases h0 : orthHeight B0 = 0 ∨ orthDist B0 bLast = 0
    · have hzero : orthHeight B = 0 := by
        show orthHeight B0 * ENNReal.ofReal (orthDist B0 bLast) = 0
        rcases h0 with h | h
        · rw [h, zero_mul]
        · rw [h, ENNReal.ofReal_zero, mul_zero]
      rw [hzero, ENNReal.inv_zero, ENNReal.mul_top]
      · exact le_top
      · exact (ENNReal.pow_pos (ENNReal.ofReal_pos.mpr (Real.sqrt_pos.mpr
          (div_pos Real.pi_pos hN))) _).ne'
    · push Not at h0
      rcases h0 with ⟨hH0_ne, hd_ne⟩
      have hd_pos : 0 < orthDist B0 bLast := lt_of_le_of_ne hdist_nn hd_ne.symm
      -- Decompose `toE bLast = v_par + v_perp` with `v_par = ∑ i, c i • toE (B0 i) ∈ V`
      have hmemV : V.starProjection (toE bLast) ∈ V := Submodule.starProjection_apply_mem V _
      rw [Submodule.mem_span_range_iff_exists_fun] at hmemV
      obtain ⟨c, hc⟩ := hmemV
      set vPerp : EuclideanSpace ℝ (Fin m) := Vᗮ.starProjection (toE bLast)
      have hvPerp_norm : ‖vPerp‖ = orthDist B0 bLast := rfl
      -- Split `a : Fin (k + 1) → ℝ` into `(t, a')` via `piFinSuccAbove (Fin.last k)`
      let e : (Fin (k + 1) → ℝ) ≃ᵐ ℝ × (Fin k → ℝ) :=
        MeasurableEquiv.piFinSuccAbove (fun _ => ℝ) (Fin.last k)
      have hmp : MeasurePreserving e volume volume :=
        volume_preserving_piFinSuccAbove (fun _ => ℝ) (Fin.last k)
      have hsum_eq : ∀ a : Fin (k + 1) → ℝ,
          ∑ i : Fin m, (∑ r : Fin (k + 1), B r i * a r) ^ 2 =
          (∑ i : Fin m, (∑ r : Fin k, B0 r i * ((e a).2 r + (e a).1 * c r)) ^ 2) +
            (e a).1 ^ 2 * (orthDist B0 bLast) ^ 2 := fun a => by
        let t : ℝ := (e a).1
        let a' : Fin k → ℝ := (e a).2
        have ha_last : a (Fin.last k) = t := rfl
        have ha_init : ∀ r : Fin k, a (Fin.castSucc r) = a' r := fun r => by
          simp only [a', e, MeasurableEquiv.piFinSuccAbove_apply, Fin.insertNthEquiv_symm_apply,
            Fin.removeNth, Fin.succAbove_last]
        -- Express both sides as `‖·‖^2` in `EuclideanSpace ℝ (Fin m)`
        have hLHS_norm : ∑ i : Fin m, (∑ r : Fin (k + 1), B r i * a r) ^ 2 =
            ‖∑ r : Fin (k + 1), a r • toE (B r)‖ ^ 2 := by
          rw [EuclideanSpace.norm_sq_eq]
          refine Finset.sum_congr rfl fun i _ => ?_
          rw [Real.norm_eq_abs, sq_abs]
          congr 1
          simp only [WithLp.ofLp_sum, WithLp.ofLp_smul, Finset.sum_apply, Pi.smul_apply,
            smul_eq_mul, toE, mul_comm]
        have hRHS1_norm : ∑ i : Fin m, (∑ r : Fin k, B0 r i * (a' r + t * c r)) ^ 2 =
            ‖∑ r : Fin k, (a' r + t * c r) • toE (B0 r)‖ ^ 2 := by
          rw [EuclideanSpace.norm_sq_eq]
          refine Finset.sum_congr rfl fun i _ => ?_
          rw [Real.norm_eq_abs, sq_abs]
          congr 1
          simp only [WithLp.ofLp_sum, WithLp.ofLp_smul, Finset.sum_apply, Pi.smul_apply,
            smul_eq_mul, toE, mul_comm]
        rw [hLHS_norm, hRHS1_norm]
        have hdecomp : (∑ r : Fin (k + 1), a r • toE (B r)) =
            (∑ r : Fin k, (a' r + t * c r) • toE (B0 r)) + t • vPerp := by
          rw [Fin.sum_univ_castSucc, ha_last]
          have h1 : (∑ r : Fin k, a (Fin.castSucc r) • toE (B (Fin.castSucc r))) =
              ∑ r : Fin k, a' r • toE (B0 r) :=
            Finset.sum_congr rfl fun r _ => by rw [ha_init r]; rfl
          rw [h1]
          have h2 : toE (B (Fin.last k)) = (∑ r : Fin k, c r • toE (B0 r)) + vPerp := by
            change toE bLast = _
            rw [hc]
            exact (Submodule.starProjection_add_starProjection_orthogonal (K := V) (toE bLast)).symm
          rw [h2, smul_add, Finset.smul_sum]
          simp only [add_smul, Finset.sum_add_distrib, smul_smul]
          abel
        rw [hdecomp]
        have hmem1 : (∑ r : Fin k, (a' r + t * c r) • toE (B0 r)) ∈ V :=
          Submodule.sum_mem V fun r _ => Submodule.smul_mem V _ (Submodule.subset_span ⟨r, rfl⟩)
        have hmem2 : t • vPerp ∈ Vᗮ :=
          Submodule.smul_mem Vᗮ t (Submodule.starProjection_apply_mem Vᗮ _)
        have horth : inner ℝ (∑ r : Fin k, (a' r + t * c r) • toE (B0 r)) (t • vPerp) = 0 :=
          Submodule.inner_right_of_mem_orthogonal hmem1 hmem2
        rw [norm_add_sq_real, horth, mul_zero, add_zero, norm_smul, Real.norm_eq_abs,
          mul_pow, sq_abs, hvPerp_norm]
      -- Push integral forward through `e` and apply Fubini
      let F : ℝ × (Fin k → ℝ) → ℝ≥0∞ := fun p =>
        ENNReal.ofReal (Real.exp (-(N * (p.1 ^ 2 * (orthDist B0 bLast) ^ 2)))) *
        ENNReal.ofReal (Real.exp (-(N * ∑ i : Fin m, (∑ r : Fin k, B0 r i * (p.2 r + p.1 * c r)) ^ 2)))
      have hF_meas : Measurable F := by unfold F; fun_prop
      have hpt : ∀ a : Fin (k + 1) → ℝ,
          ENNReal.ofReal (Real.exp (-(N * ∑ i : Fin m, (∑ r : Fin (k + 1), B r i * a r) ^ 2))) =
          F (e a) := fun a => by
        rw [hsum_eq a]
        unfold F
        rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add]
        congr 2; ring
      simp_rw [hpt]
      rw [← lintegral_map hF_meas e.measurable, hmp.map_eq]
      change ∫⁻ p : ℝ × (Fin k → ℝ), F p ∂(volume.prod volume) ≤ _
      rw [lintegral_prod _ hF_meas.aemeasurable]
      have hinner : ∀ t : ℝ,
          ∫⁻ a' : Fin k → ℝ, F (t, a') ≤
          ENNReal.ofReal (Real.exp (-((N * (orthDist B0 bLast) ^ 2) * t ^ 2))) *
            (ENNReal.ofReal (Real.sqrt (Real.pi / N)) ^ k * (orthHeight B0)⁻¹) := fun t => by
        unfold F
        dsimp only
        rw [lintegral_const_mul _ (by fun_prop)]
        have hshift :
            ∫⁻ a' : Fin k → ℝ,
              ENNReal.ofReal (Real.exp (-(N * ∑ i : Fin m, (∑ r : Fin k, B0 r i * (a' r + t * c r)) ^ 2))) =
            ∫⁻ a' : Fin k → ℝ,
              ENNReal.ofReal (Real.exp (-(N * ∑ i : Fin m, (∑ r : Fin k, B0 r i * a' r) ^ 2))) := by
          have htr := measurePreserving_add_right (volume : Measure (Fin k → ℝ)) (t • c)
          have hm_g : Measurable fun a' : Fin k → ℝ =>
              ENNReal.ofReal (Real.exp (-(N * ∑ i : Fin m, (∑ r : Fin k, B0 r i * a' r) ^ 2))) := by
            fun_prop
          conv_rhs => rw [← htr.map_eq]
          rw [lintegral_map hm_g htr.measurable]
          rfl
        rw [hshift]
        have hexp_eq : -(N * (t ^ 2 * (orthDist B0 bLast) ^ 2)) =
            -((N * (orthDist B0 bLast) ^ 2) * t ^ 2) := by ring
        rw [hexp_eq]
        exact mul_le_mul' le_rfl (lintegral_gaussian_cols_le B0 hN)
      calc ∫⁻ t : ℝ, ∫⁻ a' : Fin k → ℝ, F (t, a')
          ≤ ∫⁻ t : ℝ,
              ENNReal.ofReal (Real.exp (-((N * (orthDist B0 bLast) ^ 2) * t ^ 2))) *
                (ENNReal.ofReal (Real.sqrt (Real.pi / N)) ^ k * (orthHeight B0)⁻¹) :=
            lintegral_mono hinner
        _ = (∫⁻ t : ℝ, ENNReal.ofReal (Real.exp (-((N * (orthDist B0 bLast) ^ 2) * t ^ 2)))) *
              (ENNReal.ofReal (Real.sqrt (Real.pi / N)) ^ k * (orthHeight B0)⁻¹) :=
            lintegral_mul_const _ (by fun_prop)
        _ = ENNReal.ofReal (Real.sqrt (Real.pi / (N * (orthDist B0 bLast) ^ 2))) *
              (ENNReal.ofReal (Real.sqrt (Real.pi / N)) ^ k * (orthHeight B0)⁻¹) := by
            rw [lintegral_gaussian_1d (by positivity)]
        _ = ENNReal.ofReal (Real.sqrt (Real.pi / N)) ^ (k + 1) * (orthHeight B)⁻¹ := by
            have hsqrt_split : Real.sqrt (Real.pi / (N * (orthDist B0 bLast) ^ 2)) =
                Real.sqrt (Real.pi / N) * (orthDist B0 bLast)⁻¹ := by
              rw [show Real.pi / (N * (orthDist B0 bLast) ^ 2) =
                  (Real.pi / N) * ((orthDist B0 bLast)⁻¹) ^ 2 by field_simp,
                Real.sqrt_mul (by positivity), Real.sqrt_sq (by positivity)]
            rw [hsqrt_split, ENNReal.ofReal_mul (Real.sqrt_nonneg _),
              ENNReal.ofReal_inv_of_pos hd_pos]
            show (ENNReal.ofReal (Real.sqrt (Real.pi / N)) * (ENNReal.ofReal (orthDist B0 bLast))⁻¹) *
                (ENNReal.ofReal (Real.sqrt (Real.pi / N)) ^ k * (orthHeight B0)⁻¹) =
              ENNReal.ofReal (Real.sqrt (Real.pi / N)) ^ (k + 1) *
                (orthHeight B0 * ENNReal.ofReal (orthDist B0 bLast))⁻¹
            rw [ENNReal.mul_inv (Or.inl hH0_ne) (Or.inl (orthHeight_ne_top B0)), pow_succ']
            ring

/-! ### 5. Uniform integrability of `orthHeight(B)^{-n}` when `k + n ≤ m` -/

/-- For any subspace `V ⊆ EuclideanSpace ℝ (Fin m)` with `finrank V + n < m`, the integral of
`‖Vᗮ.starProjection x‖^{-n}` over `x ∈ [-1,1]^m` is bounded by the explicit `V`-independent
constant `ENNReal.ofReal (2 * (Real.sqrt m + 1) * m) ^ m`. -/
theorem lintegral_orthDist_inv_pow_le {m k n : ℕ} (C : Fin k → (Fin m → ℝ)) (hkm : k + n < m) :
    ∫⁻ b in Set.pi Set.univ (fun _ : Fin m => Set.Icc (-1 : ℝ) 1),
      (ENNReal.ofReal (orthDist C b))⁻¹ ^ n ≤
    ENNReal.ofReal (2 * (Real.sqrt m + 1) * m) ^ m := by
  let toE : (Fin m → ℝ) → EuclideanSpace ℝ (Fin m) := WithLp.toLp 2
  let V : Submodule ℝ (EuclideanSpace ℝ (Fin m)) :=
    Submodule.span ℝ (Set.range (fun i => toE (C i)))
  let r : ℕ := Module.finrank ℝ V
  let d : ℕ := Module.finrank ℝ Vᗮ
  have hrd : r + d = m := by
    have h := Submodule.finrank_add_finrank_orthogonal V
    simpa [r, d, finrank_euclideanSpace_fin] using h
  have hr_le : r ≤ k := by
    let L : (Fin k → ℝ) →ₗ[ℝ] EuclideanSpace ℝ (Fin m) :=
      Fintype.linearCombination ℝ (fun i => toE (C i))
    have hrange : LinearMap.range L = V := by
      simp [L, V]
    change Module.finrank ℝ V ≤ k
    rw [← hrange]
    calc Module.finrank ℝ (LinearMap.range L)
        ≤ Module.finrank ℝ (Fin k → ℝ) := LinearMap.finrank_range_le L
      _ = k := Module.finrank_fin_fun ℝ
  have hdn : n < d := by omega
  have hm_pos : (0 : ℝ) < m := Nat.cast_pos.mpr (by omega)
  have hd_pos : (0 : ℝ) < d := Nat.cast_pos.mpr (by omega)
  -- Build an explicit `LinearIsometryEquiv` `EuclideanSpace ℝ (Fin m) ≃ₗᵢ EuclideanSpace ℝ (Fin r ⊕ Fin d)`
  let b1 : OrthonormalBasis (Fin r) ℝ V := stdOrthonormalBasis ℝ V
  let b2 : OrthonormalBasis (Fin d) ℝ Vᗮ := stdOrthonormalBasis ℝ Vᗮ
  let fwdLin : EuclideanSpace ℝ (Fin m) →ₗ[ℝ] EuclideanSpace ℝ (Fin r ⊕ Fin d) :=
    { toFun := fun x => WithLp.toLp 2 (Sum.elim (b1.repr (V.orthogonalProjectionOnto x))
        (b2.repr (Vᗮ.orthogonalProjectionOnto x)))
      map_add' := fun x y => by
        ext i; cases i <;> simp [map_add]
      map_smul' := fun t x => by
        ext i; cases i <;> simp [map_smul] }
  let invLin : EuclideanSpace ℝ (Fin r ⊕ Fin d) →ₗ[ℝ] EuclideanSpace ℝ (Fin m) :=
    { toFun := fun y =>
        (b1.repr.symm (WithLp.toLp 2 (fun i => y (Sum.inl i))) : EuclideanSpace ℝ (Fin m)) +
        (b2.repr.symm (WithLp.toLp 2 (fun j => y (Sum.inr j))) : EuclideanSpace ℝ (Fin m))
      map_add' := fun y z => by
        have h1 : WithLp.toLp 2 (fun i : Fin r => (y + z) (Sum.inl i)) =
            WithLp.toLp 2 (fun i : Fin r => y (Sum.inl i)) +
            WithLp.toLp 2 (fun i : Fin r => z (Sum.inl i)) := by ext; rfl
        have h2 : WithLp.toLp 2 (fun j : Fin d => (y + z) (Sum.inr j)) =
            WithLp.toLp 2 (fun j : Fin d => y (Sum.inr j)) +
            WithLp.toLp 2 (fun j : Fin d => z (Sum.inr j)) := by ext; rfl
        rw [h1, h2, map_add, map_add]
        simp only [Submodule.coe_add]; abel
      map_smul' := fun t y => by
        have h1 : WithLp.toLp 2 (fun i : Fin r => (t • y) (Sum.inl i)) =
            t • WithLp.toLp 2 (fun i : Fin r => y (Sum.inl i)) := by ext; rfl
        have h2 : WithLp.toLp 2 (fun j : Fin d => (t • y) (Sum.inr j)) =
            t • WithLp.toLp 2 (fun j : Fin d => y (Sum.inr j)) := by ext; rfl
        rw [h1, h2, map_smul, map_smul]
        simp only [RingHom.id_apply, Submodule.coe_smul_of_tower, smul_add] }
  have hleft : ∀ x, invLin (fwdLin x) = x := fun x => by
    dsimp [invLin, fwdLin]
    rw [LinearIsometryEquiv.symm_apply_apply, LinearIsometryEquiv.symm_apply_apply]
    exact Submodule.starProjection_add_starProjection_orthogonal (K := V) x
  have hright : ∀ y, fwdLin (invLin y) = y := fun y => by
    let u : V := b1.repr.symm (WithLp.toLp 2 (fun i => y (Sum.inl i)))
    let v : Vᗮ := b2.repr.symm (WithLp.toLp 2 (fun j => y (Sum.inr j)))
    have hproj1 : V.orthogonalProjectionOnto ((u : EuclideanSpace ℝ (Fin m)) + v) = u := by
      rw [map_add, Submodule.orthogonalProjectionOnto_mem_subspace_eq_self u,
        Submodule.orthogonalProjectionOnto_apply_of_mem_orthogonal v.2, add_zero]
    have hproj2 : Vᗮ.orthogonalProjectionOnto ((u : EuclideanSpace ℝ (Fin m)) + v) = v := by
      rw [map_add,
        Submodule.orthogonalProjectionOnto_apply_of_mem_orthogonal
          (Submodule.le_orthogonal_orthogonal V u.2),
        zero_add, Submodule.orthogonalProjectionOnto_mem_subspace_eq_self v]
    ext s
    cases s with
    | inl i =>
      change b1.repr (V.orthogonalProjectionOnto ((u : EuclideanSpace ℝ (Fin m)) + v)) i = y (Sum.inl i)
      rw [hproj1, LinearIsometryEquiv.apply_symm_apply]
    | inr j =>
      change b2.repr (Vᗮ.orthogonalProjectionOnto ((u : EuclideanSpace ℝ (Fin m)) + v)) j = y (Sum.inr j)
      rw [hproj2, LinearIsometryEquiv.apply_symm_apply]
  have hnorm : ∀ x, ‖fwdLin x‖ = ‖x‖ := fun x => by
    have hsq : ‖fwdLin x‖ ^ 2 = ‖x‖ ^ 2 := by
      rw [EuclideanSpace.norm_sq_eq]
      rw [Fintype.sum_sum_type]
      have h1 : (∑ i : Fin r, ‖fwdLin x (Sum.inl i)‖ ^ 2) = ‖V.orthogonalProjectionOnto x‖ ^ 2 := by
        rw [← b1.repr.norm_map, EuclideanSpace.norm_sq_eq]; rfl
      have h2 : (∑ j : Fin d, ‖fwdLin x (Sum.inr j)‖ ^ 2) = ‖Vᗮ.orthogonalProjectionOnto x‖ ^ 2 := by
        rw [← b2.repr.norm_map, EuclideanSpace.norm_sq_eq]; rfl
      rw [h1, h2]
      have hsum := Submodule.starProjection_add_starProjection_orthogonal (K := V) x
      have horth : inner ℝ (V.starProjection x) (Vᗮ.starProjection x) = 0 :=
        Submodule.inner_right_of_mem_orthogonal (Submodule.starProjection_apply_mem V x)
          (Submodule.starProjection_apply_mem Vᗮ x)
      conv_rhs => rw [← hsum]
      rw [norm_add_sq_real, horth, mul_zero, add_zero]
      rfl
    nlinarith [norm_nonneg (fwdLin x), norm_nonneg x]
  let iso : EuclideanSpace ℝ (Fin m) ≃ₗᵢ[ℝ] EuclideanSpace ℝ (Fin r ⊕ Fin d) :=
    { toLinearEquiv := LinearEquiv.ofLinearMap fwdLin invLin (LinearMap.ext hright) (LinearMap.ext hleft)
      norm_map' := hnorm }
  -- Compose with `toLp` and split `Fin r ⊕ Fin d → ℝ` into `(Fin r → ℝ) × (Fin d → ℝ)`
  let e_toLp : (Fin m → ℝ) ≃ᵐ EuclideanSpace ℝ (Fin m) := MeasurableEquiv.toLp 2 (Fin m → ℝ)
  let e_ofLp : EuclideanSpace ℝ (Fin r ⊕ Fin d) ≃ᵐ (Fin r ⊕ Fin d → ℝ) :=
    (MeasurableEquiv.toLp 2 (Fin r ⊕ Fin d → ℝ)).symm
  let e_split : (Fin r ⊕ Fin d → ℝ) ≃ᵐ (Fin r → ℝ) × (Fin d → ℝ) :=
    { toFun := fun f => (fun i => f (Sum.inl i), fun j => f (Sum.inr j))
      invFun := fun p => Sum.elim p.1 p.2
      left_inv := fun f => by funext x; cases x <;> rfl
      right_inv := fun ⟨_, _⟩ => rfl
      measurable_toFun := (measurable_pi_iff.mpr fun i => measurable_pi_apply (Sum.inl i)).prodMk
        (measurable_pi_iff.mpr fun j => measurable_pi_apply (Sum.inr j))
      measurable_invFun := measurable_pi_iff.mpr fun x => by
        cases x with
        | inl i => exact (measurable_pi_apply i).comp measurable_fst
        | inr j => exact (measurable_pi_apply j).comp measurable_snd }
  let φ : (Fin m → ℝ) ≃ᵐ (Fin r → ℝ) × (Fin d → ℝ) :=
    e_toLp.trans (iso.toHomeomorph.toMeasurableEquiv.trans (e_ofLp.trans e_split))
  have hmp_toLp : MeasurePreserving e_toLp volume volume :=
    (EuclideanSpace.volume_preserving_symm_measurableEquiv_toLp (Fin m)).symm
      (MeasurableEquiv.toLp 2 (Fin m → ℝ)).symm
  have hmp_iso : MeasurePreserving iso volume volume := iso.measurePreserving
  have hmp_ofLp : MeasurePreserving e_ofLp volume volume :=
    EuclideanSpace.volume_preserving_symm_measurableEquiv_toLp (Fin r ⊕ Fin d)
  have hmp_split : MeasurePreserving e_split volume volume := by
    let P : Fin r ⊕ Fin d → Prop := fun x => x.isLeft = true
    let eqL : Fin r ≃ {x : Fin r ⊕ Fin d // P x} :=
      { toFun := fun i => ⟨Sum.inl i, rfl⟩
        invFun := fun x => match x with | ⟨Sum.inl i, _⟩ => i
        left_inv := fun _ => rfl
        right_inv := fun ⟨x, hx⟩ => by cases x with | inl _ => rfl | inr _ => simp [P] at hx }
    let eqR : Fin d ≃ {x : Fin r ⊕ Fin d // ¬ P x} :=
      { toFun := fun j => ⟨Sum.inr j, by simp [P]⟩
        invFun := fun x => match x with | ⟨Sum.inr j, _⟩ => j
        left_inv := fun _ => rfl
        right_inv := fun ⟨x, hx⟩ => by cases x with | inl _ => simp [P] at hx | inr _ => rfl }
    let eA := MeasurableEquiv.piEquivPiSubtypeProd (fun _ : Fin r ⊕ Fin d => ℝ) P
    let eL := (MeasurableEquiv.piCongrLeft (fun _ : {x : Fin r ⊕ Fin d // P x} => ℝ) eqL).symm
    let eR := (MeasurableEquiv.piCongrLeft (fun _ : {x : Fin r ⊕ Fin d // ¬ P x} => ℝ) eqR).symm
    have hA : MeasurePreserving eA volume volume :=
      measurePreserving_piEquivPiSubtypeProd (fun _ : Fin r ⊕ Fin d => (volume : Measure ℝ)) P
    have hL : MeasurePreserving eL volume volume :=
      (measurePreserving_piCongrLeft (fun _ : {x : Fin r ⊕ Fin d // P x} => (volume : Measure ℝ)) eqL).symm
        (MeasurableEquiv.piCongrLeft (fun _ => ℝ) eqL)
    have hR : MeasurePreserving eR volume volume :=
      (measurePreserving_piCongrLeft (fun _ : {x : Fin r ⊕ Fin d // ¬ P x} => (volume : Measure ℝ)) eqR).symm
        (MeasurableEquiv.piCongrLeft (fun _ => ℝ) eqR)
    have hcomp : (⇑e_split : (Fin r ⊕ Fin d → ℝ) → (Fin r → ℝ) × (Fin d → ℝ)) =
        (Prod.map eL eR) ∘ eA := by
      funext f
      refine Prod.ext (funext fun i => ?_) (funext fun j => ?_) <;> rfl
    rw [hcomp]
    exact (hL.prod hR).comp hA
  have hmp_φ : MeasurePreserving φ volume volume :=
    hmp_split.comp (hmp_ofLp.comp (hmp_iso.comp hmp_toLp))
  -- Under `φ`, `orthDist C b = √(∑ j, ((φ b).2 j)^2)` and `b ∈ [-1,1]^m → φ b ∈ [-√m, √m]^r × [-√m, √m]^d`
  have hdist_φ : ∀ b : Fin m → ℝ,
      orthDist C b = Real.sqrt (∑ j : Fin d, ((φ b).2 j) ^ 2) := fun b => by
    unfold orthDist
    have hsq : ‖Vᗮ.starProjection (toE b)‖ ^ 2 = ∑ j : Fin d, ((φ b).2 j) ^ 2 := by
      have h1 : ‖Vᗮ.orthogonalProjectionOnto (toE b)‖ ^ 2 =
          ‖b2.repr (Vᗮ.orthogonalProjectionOnto (toE b))‖ ^ 2 := by
        rw [b2.repr.norm_map]
      change ‖Vᗮ.orthogonalProjectionOnto (toE b)‖ ^ 2 = _
      rw [h1, EuclideanSpace.norm_sq_eq]
      refine Finset.sum_congr rfl fun j _ => ?_
      rw [Real.norm_eq_abs, sq_abs]; rfl
    rw [← Real.sqrt_sq (norm_nonneg _), hsq]
  let R : ℝ := Real.sqrt m
  have hRnn : 0 ≤ R := Real.sqrt_nonneg m
  let boxR : Set (Fin r → ℝ) := Set.pi Set.univ fun _ => Set.Icc (-R) R
  let boxD : Set (Fin d → ℝ) := Set.pi Set.univ fun _ => Set.Icc (-R) R
  have hsub : ∀ b ∈ Set.pi Set.univ (fun _ : Fin m => Set.Icc (-1 : ℝ) 1),
      φ b ∈ boxR ×ˢ boxD := fun b hb => by
    have hnorm_b : ‖toE b‖ ≤ R := by
      rw [← Real.sqrt_sq (norm_nonneg _), EuclideanSpace.norm_sq_eq]
      refine Real.sqrt_le_sqrt ?_
      calc ∑ i : Fin m, ‖toE b i‖ ^ 2
          ≤ ∑ _ : Fin m, (1 : ℝ) := by
            refine Finset.sum_le_sum fun i _ => ?_
            have hbi : |b i| ≤ 1 := abs_le.mpr (hb i (Set.mem_univ i))
            rw [Real.norm_eq_abs]
            have := pow_le_pow_left₀ (abs_nonneg (b i)) hbi 2
            linarith
        _ = m := by simp
    have hnorm_iso : ‖iso (toE b)‖ ≤ R := by rw [iso.norm_map]; exact hnorm_b
    refine ⟨fun i _ => abs_le.mp ?_, fun j _ => abs_le.mp ?_⟩
    · have h1 : |(φ b).1 i| = ‖iso (toE b) (Sum.inl i)‖ := by rw [Real.norm_eq_abs]; rfl
      rw [h1]
      exact (PiLp.norm_apply_le (iso (toE b)) (Sum.inl i)).trans hnorm_iso
    · have h2 : |(φ b).2 j| = ‖iso (toE b) (Sum.inr j)‖ := by rw [Real.norm_eq_abs]; rfl
      rw [h2]
      exact (PiLp.norm_apply_le (iso (toE b)) (Sum.inr j)).trans hnorm_iso
  -- Change variables through `φ`
  have hmeas_int : Measurable fun p : (Fin r → ℝ) × (Fin d → ℝ) =>
      (ENNReal.ofReal (Real.sqrt (∑ j : Fin d, (p.2 j) ^ 2)))⁻¹ ^ n := by fun_prop
  have hstep1 :
      ∫⁻ b in Set.pi Set.univ (fun _ : Fin m => Set.Icc (-1 : ℝ) 1),
        (ENNReal.ofReal (orthDist C b))⁻¹ ^ n ≤
      ∫⁻ p in boxR ×ˢ boxD,
        (ENNReal.ofReal (Real.sqrt (∑ j : Fin d, (p.2 j) ^ 2)))⁻¹ ^ n := by
    simp_rw [hdist_φ]
    rw [← lintegral_indicator (MeasurableSet.univ_pi fun _ => measurableSet_Icc)]
    calc ∫⁻ b, (Set.pi Set.univ fun _ : Fin m => Set.Icc (-1 : ℝ) 1).indicator
            (fun x => (ENNReal.ofReal (Real.sqrt (∑ j : Fin d, ((φ x).2 j) ^ 2)))⁻¹ ^ n) b
        ≤ ∫⁻ b, (boxR ×ˢ boxD).indicator
            (fun p => (ENNReal.ofReal (Real.sqrt (∑ j : Fin d, (p.2 j) ^ 2)))⁻¹ ^ n) (φ b) := by
          refine lintegral_mono fun b => ?_
          by_cases hb : b ∈ Set.pi Set.univ (fun _ : Fin m => Set.Icc (-1 : ℝ) 1)
          · rw [Set.indicator_of_mem hb, Set.indicator_of_mem (hsub b hb)]
          · rw [Set.indicator_of_notMem hb]; exact bot_le
      _ = ∫⁻ p in boxR ×ˢ boxD,
            (ENNReal.ofReal (Real.sqrt (∑ j : Fin d, (p.2 j) ^ 2)))⁻¹ ^ n := by
          rw [← lintegral_map (hmeas_int.indicator ((MeasurableSet.univ_pi fun _ => measurableSet_Icc).prod
            (MeasurableSet.univ_pi fun _ => measurableSet_Icc))) φ.measurable,
            hmp_φ.map_eq, lintegral_indicator ((MeasurableSet.univ_pi fun _ => measurableSet_Icc).prod
            (MeasurableSet.univ_pi fun _ => measurableSet_Icc))]
  -- Split `boxR ×ˢ boxD` by Fubini and bound both factors by `(2 * (R + 1) * m)^m`
  have hvolR : volume boxR = ENNReal.ofReal (2 * R) ^ r := by
    change (Measure.pi fun _ : Fin r => (volume : Measure ℝ)) _ = _
    rw [Measure.pi_pi]
    simp [Real.volume_Icc, two_mul, sub_neg_eq_add]
  have hstep2 :
      ∫⁻ p in boxR ×ˢ boxD,
        (ENNReal.ofReal (Real.sqrt (∑ j : Fin d, (p.2 j) ^ 2)))⁻¹ ^ n ≤
      ENNReal.ofReal (2 * R) ^ r *
        ENNReal.ofReal (2 * R ^ (1 - (n : ℝ) / d) / (1 - (n : ℝ) / d)) ^ d := by
    change ∫⁻ p in boxR ×ˢ boxD, _ ∂(volume.prod volume) ≤ _
    rw [← Measure.prod_restrict, lintegral_prod _ hmeas_int.aemeasurable]
    simp_rw [setLIntegral_const, hvolR]
    rw [mul_comm]
    exact mul_le_mul' le_rfl (lintegral_cube_sum_sq_inv_pow_le hRnn hdn)
  -- Uniformly dominate both bases by `2 * (R + 1) * m`
  have hbase1 : 2 * R ≤ 2 * (R + 1) * m := by
    have hm1 : (1 : ℝ) ≤ m := by exact_mod_cast (show 1 ≤ m by omega)
    nlinarith
  have hbase2 : 2 * R ^ (1 - (n : ℝ) / d) / (1 - (n : ℝ) / d) ≤ 2 * (R + 1) * m := by
    have hα_pos : 0 < 1 - (n : ℝ) / d := by
      have : (n : ℝ) / d < 1 := (div_lt_one hd_pos).mpr (Nat.cast_lt.mpr hdn)
      linarith
    have hα_le1 : 1 - (n : ℝ) / d ≤ 1 := by
      have : 0 ≤ (n : ℝ) / d := by positivity
      linarith
    have hRpow : R ^ (1 - (n : ℝ) / d) ≤ R + 1 := by
      by_cases hR1 : R ≤ 1
      · calc R ^ (1 - (n : ℝ) / d) ≤ 1 ^ (1 - (n : ℝ) / d) :=
              Real.rpow_le_rpow hRnn hR1 hα_pos.le
          _ = 1 := Real.one_rpow _
          _ ≤ R + 1 := by linarith
      · push Not at hR1
        calc R ^ (1 - (n : ℝ) / d) ≤ R ^ (1 : ℝ) := Real.rpow_le_rpow_of_exponent_le hR1.le hα_le1
          _ = R := Real.rpow_one R
          _ ≤ R + 1 := by linarith
    have hdenom : 1 / (m : ℝ) ≤ 1 - (n : ℝ) / d := by
      have hd_le_m : (d : ℝ) ≤ m := by exact_mod_cast (show d ≤ m by omega)
      have hdn1 : (1 : ℝ) ≤ (d : ℝ) - n := by
        have : (n : ℝ) + 1 ≤ d := by exact_mod_cast (show n + 1 ≤ d by omega)
        linarith
      rw [show 1 - (n : ℝ) / d = ((d : ℝ) - n) / d by field_simp]
      rw [div_le_div_iff₀ hm_pos hd_pos]
      nlinarith
    rw [div_le_iff₀ hα_pos]
    calc 2 * R ^ (1 - (n : ℝ) / d)
        ≤ 2 * (R + 1) := by linarith
      _ = 2 * (R + 1) * m * (1 / (m : ℝ)) := by field_simp
      _ ≤ 2 * (R + 1) * m * (1 - (n : ℝ) / d) :=
          mul_le_mul_of_nonneg_left hdenom (by positivity)
  calc ∫⁻ b in Set.pi Set.univ (fun _ : Fin m => Set.Icc (-1 : ℝ) 1),
        (ENNReal.ofReal (orthDist C b))⁻¹ ^ n
      ≤ ENNReal.ofReal (2 * R) ^ r *
          ENNReal.ofReal (2 * R ^ (1 - (n : ℝ) / d) / (1 - (n : ℝ) / d)) ^ d :=
        hstep1.trans hstep2
    _ ≤ ENNReal.ofReal (2 * (R + 1) * m) ^ r * ENNReal.ofReal (2 * (R + 1) * m) ^ d :=
        mul_le_mul' (pow_le_pow_left' (ENNReal.ofReal_le_ofReal hbase1) r)
          (pow_le_pow_left' (ENNReal.ofReal_le_ofReal hbase2) d)
    _ = ENNReal.ofReal (2 * (R + 1) * m) ^ m := by rw [← pow_add, hrd]

/-- **Inductive finiteness of `∫_{[-1,1]^{k × m}} orthHeight(B)^{-n} dB` when `k + n ≤ m`.** -/
theorem lintegral_orthHeight_inv_pow_lt_top {m n : ℕ} :
    ∀ {k : ℕ}, k + n ≤ m →
      ∫⁻ B in matBox (a := k) (b := m) 1, (orthHeight B)⁻¹ ^ n < ⊤
  | 0, _ => by
    simp only [orthHeight, inv_one, one_pow, setLIntegral_const, one_mul]
    exact volume_matBox_lt_top (by norm_num)
  | k + 1, hkn => by
    let e : RMat (k + 1) m ≃ᵐ (Fin m → ℝ) × RMat k m :=
      MeasurableEquiv.piFinSuccAbove (fun _ => Fin m → ℝ) (Fin.last k)
    have hmp : MeasurePreserving e volume volume :=
      volume_preserving_piFinSuccAbove (fun _ => Fin m → ℝ) (Fin.last k)
    let box_m : Set (Fin m → ℝ) := Set.pi Set.univ fun _ => Set.Icc (-1 : ℝ) 1
    have hpre : e ⁻¹' (box_m ×ˢ matBox (a := k) (b := m) 1) = matBox (a := k + 1) (b := m) 1 := by
      ext B
      simp only [Set.mem_preimage, Set.mem_prod, box_m, matBox, Set.mem_pi, Set.mem_univ,
        forall_const, e, MeasurableEquiv.piFinSuccAbove_apply, Fin.insertNthEquiv_symm_apply]
      constructor
      · rintro ⟨hlast, hinit⟩ i j
        refine Fin.lastCases (hlast j) (fun r => ?_) i
        have := hinit r j
        simpa [Fin.removeNth, Fin.succAbove_last] using this
      · intro hB
        exact ⟨fun j => hB (Fin.last k) j,
          fun r j => by simpa [Fin.removeNth, Fin.succAbove_last] using hB (Fin.castSucc r) j⟩
    have hsplit : ∀ B : RMat (k + 1) m,
        (orthHeight B)⁻¹ ^ n =
        (orthHeight (e B).2)⁻¹ ^ n * (ENNReal.ofReal (orthDist (e B).2 (e B).1))⁻¹ ^ n := fun B => by
      have he1 : (e B).1 = B (Fin.last k) := rfl
      have he2 : (e B).2 = Fin.init B := by
        funext r j
        simp only [e, MeasurableEquiv.piFinSuccAbove_apply, Fin.insertNthEquiv_symm_apply,
          Fin.removeNth, Fin.succAbove_last, Fin.init]
      rw [he1, he2]
      show (orthHeight (Fin.init B) * ENNReal.ofReal (orthDist (Fin.init B) (B (Fin.last k))))⁻¹ ^ n = _
      by_cases h1 : orthHeight (Fin.init B) = 0
      · rw [h1, zero_mul, ENNReal.inv_zero]
        cases n with
        | zero => simp
        | succ n' =>
          rw [ENNReal.top_pow (Nat.succ_ne_zero n'), ENNReal.top_mul]
          exact (ENNReal.pow_pos (ENNReal.inv_pos.mpr ENNReal.ofReal_ne_top) _).ne'
      · rw [ENNReal.mul_inv (Or.inl h1) (Or.inl (orthHeight_ne_top _)), mul_pow]
    have hF_meas : Measurable fun p : (Fin m → ℝ) × RMat k m =>
        (orthHeight p.2)⁻¹ ^ n * (ENNReal.ofReal (orthDist p.2 p.1))⁻¹ ^ n := by
      have hm1 : Measurable fun p : (Fin m → ℝ) × RMat k m => (orthHeight p.2)⁻¹ ^ n :=
        (((measurable_orthHeight m k).comp measurable_snd).inv).pow_const n
      have hm2 : Measurable fun p : (Fin m → ℝ) × RMat k m =>
          (ENNReal.ofReal (orthDist p.2 p.1))⁻¹ ^ n :=
        (((measurable_ofReal_orthDist m k).comp (measurable_snd.prodMk measurable_fst)).inv).pow_const n
      exact hm1.mul hm2
    have hres : (volume : Measure ((Fin m → ℝ) × RMat k m)).restrict
        (box_m ×ˢ matBox (a := k) (b := m) 1) =
        Measure.map e (volume.restrict (matBox (a := k + 1) (b := m) 1)) := by
      rw [← hmp.map_eq, Measure.restrict_map e.measurable
        ((MeasurableSet.univ_pi fun _ => measurableSet_Icc).prod (measurableSet_matBox 1)), hpre]
    have hstep :
        ∫⁻ B in matBox (a := k + 1) (b := m) 1, (orthHeight B)⁻¹ ^ n =
        ∫⁻ B0 in matBox (a := k) (b := m) 1,
          (orthHeight B0)⁻¹ ^ n *
            ∫⁻ b in box_m, (ENNReal.ofReal (orthDist B0 b))⁻¹ ^ n := by
      simp_rw [hsplit]
      rw [← lintegral_map hF_meas e.measurable, ← hres]
      change ∫⁻ p in box_m ×ˢ matBox (a := k) (b := m) 1, _ ∂(volume.prod volume) = _
      rw [← Measure.prod_restrict, lintegral_prod_symm _ hF_meas.aemeasurable]
      refine setLIntegral_congr_fun (measurableSet_matBox 1) fun B0 _ => ?_
      dsimp only
      have hm_inner : Measurable fun b : Fin m → ℝ => (ENNReal.ofReal (orthDist B0 b))⁻¹ ^ n :=
        (((measurable_ofReal_orthDist m k).comp (measurable_const.prodMk measurable_id)).inv).pow_const n
      exact lintegral_const_mul _ hm_inner
    rw [hstep]
    let C : ℝ≥0∞ := ENNReal.ofReal (2 * (Real.sqrt m + 1) * m) ^ m
    have hC_lt : C < ⊤ := ENNReal.pow_lt_top ENNReal.ofReal_lt_top
    have hm_h : Measurable fun B0 : RMat k m => (orthHeight B0)⁻¹ ^ n :=
      ((measurable_orthHeight m k).inv).pow_const n
    have hle :
        ∫⁻ B0 in matBox (a := k) (b := m) 1,
          (orthHeight B0)⁻¹ ^ n * ∫⁻ b in box_m, (ENNReal.ofReal (orthDist B0 b))⁻¹ ^ n ≤
        (∫⁻ B0 in matBox (a := k) (b := m) 1, (orthHeight B0)⁻¹ ^ n) * C := by
      rw [← lintegral_mul_const _ hm_h]
      refine setLIntegral_mono' (measurableSet_matBox 1) fun B0 _ => ?_
      exact mul_le_mul' le_rfl (lintegral_orthDist_inv_pow_le B0 (by omega))
    exact lt_of_le_of_lt hle (ENNReal.mul_lt_top (lintegral_orthHeight_inv_pow_lt_top (by omega)) hC_lt)

/-! ### 6. Aoyagi–Watanabe (2005) Boundary Regimes (`k + n ≤ m` and `k + m ≤ n`) -/

/-- **Aoyagi–Watanabe (2005) Boundary Regime 1 (`k + n ≤ m`):**
For `rank0Loss (B, A) = ‖B A‖_F^2` on `[-1,1]^{m×k} × [-1,1]^{k×n}`, if `k + n ≤ m`, then
the exact learning pair is `(k * n / 2, 1)` unconditionally. -/
theorem rank0_hasLaplacePair_of_le_m {m k n : ℕ} (h : k + n ≤ m) :
    SLT.HasLaplacePair
      (SLT.Z (volume.restrict (matBox (a := m) (b := k) 1 ×ˢ matBox (a := k) (b := n) 1))
        rank0Loss)
      ((k : ℝ) * n / 2) 1 := by
  refine ⟨le_rfl, rank0_laplaceLower_kn m k n, ?_⟩
  rw [SLT.laplaceUpper_iff]
  let I_B : ℝ≥0∞ := ∫⁻ B in matBox (a := k) (b := m) 1, (orthHeight B)⁻¹ ^ n
  have hI_lt : I_B < ⊤ := lintegral_orthHeight_inv_pow_lt_top h
  let C : ℝ := Real.sqrt Real.pi ^ (k * n) * I_B.toReal
  refine ⟨C, 1, fun N hN => ?_⟩
  have hNpos : 0 < N := by linarith
  -- Fubini over `B` then `A`
  have hZ :
      SLT.Z (volume.restrict (matBox (a := m) (b := k) 1 ×ˢ matBox (a := k) (b := n) 1))
        rank0Loss N =
      ∫⁻ B in matBox (a := m) (b := k) 1,
        ∫⁻ A in matBox (a := k) (b := n) 1,
          ENNReal.ofReal (Real.exp (-(N * rank0Loss (B, A)))) := by
    unfold SLT.Z
    change ∫⁻ p in matBox (a := m) (b := k) 1 ×ˢ matBox (a := k) (b := n) 1, _ ∂(volume.prod volume) = _
    rw [← Measure.prod_restrict]
    have hm_exp : Measurable fun p : RMat m k × RMat k n =>
        ENNReal.ofReal (Real.exp (-(N * rank0Loss p))) := by
      have := measurable_rank0Loss (m := m) (k' := k) (n' := n); fun_prop
    exact lintegral_prod _ hm_exp.aemeasurable
  rw [hZ]
  let tr : RMat m k → RMat k m := fun B r i => B i r
  have hmp : MeasurePreserving tr volume volume := measurePreserving_transpose m k
  have hpre : tr ⁻¹' matBox (a := k) (b := m) 1 = matBox (a := m) (b := k) 1 := by
    ext B; simp only [Set.mem_preimage, mem_matBox_iff, tr]
    exact ⟨fun h i r => h r i, fun h r i => h i r⟩
  have hslice : ∀ B : RMat m k,
      ∫⁻ A in matBox (a := k) (b := n) 1,
        ENNReal.ofReal (Real.exp (-(N * rank0Loss (B, A)))) ≤
      ENNReal.ofReal (Real.sqrt (Real.pi / N)) ^ (k * n) * (orthHeight (tr B))⁻¹ ^ n := fun B => by
    rw [col_decouple B N]
    have h1 :
        ∫⁻ a in Set.pi Set.univ (fun _ : Fin k => Set.Icc (-1 : ℝ) 1),
          ENNReal.ofReal (Real.exp (-(N * ∑ i : Fin m, (∑ r : Fin k, B i r * a r) ^ 2))) ≤
        ENNReal.ofReal (Real.sqrt (Real.pi / N)) ^ k * (orthHeight (tr B))⁻¹ :=
      (setLIntegral_le_lintegral _ _).trans (lintegral_gaussian_cols_le (tr B) hNpos)
    calc (∫⁻ a in Set.pi Set.univ (fun _ : Fin k => Set.Icc (-1 : ℝ) 1),
            ENNReal.ofReal (Real.exp (-(N * ∑ i : Fin m, (∑ r : Fin k, B i r * a r) ^ 2)))) ^ n
        ≤ (ENNReal.ofReal (Real.sqrt (Real.pi / N)) ^ k * (orthHeight (tr B))⁻¹) ^ n :=
          pow_le_pow_left' h1 n
      _ = ENNReal.ofReal (Real.sqrt (Real.pi / N)) ^ (k * n) * (orthHeight (tr B))⁻¹ ^ n := by
          rw [mul_pow, ← pow_mul]
  have hres : (volume : Measure (RMat k m)).restrict (matBox (a := k) (b := m) 1) =
      Measure.map tr (volume.restrict (matBox (a := m) (b := k) 1)) := by
    rw [← hmp.map_eq, Measure.restrict_map hmp.measurable (measurableSet_matBox 1), hpre]
  have hm_h : Measurable fun B : RMat k m => (orthHeight B)⁻¹ ^ n :=
    ((measurable_orthHeight m k).inv).pow_const n
  calc ∫⁻ B in matBox (a := m) (b := k) 1,
        ∫⁻ A in matBox (a := k) (b := n) 1,
          ENNReal.ofReal (Real.exp (-(N * rank0Loss (B, A))))
      ≤ ∫⁻ B in matBox (a := m) (b := k) 1,
          ENNReal.ofReal (Real.sqrt (Real.pi / N)) ^ (k * n) * (orthHeight (tr B))⁻¹ ^ n :=
        setLIntegral_mono' (measurableSet_matBox 1) fun B _ => hslice B
    _ = ENNReal.ofReal (Real.sqrt (Real.pi / N)) ^ (k * n) *
          ∫⁻ B in matBox (a := m) (b := k) 1, (orthHeight (tr B))⁻¹ ^ n :=
        lintegral_const_mul _ (hm_h.comp hmp.measurable)
    _ = ENNReal.ofReal (Real.sqrt (Real.pi / N)) ^ (k * n) * I_B := by
        congr 1
        dsimp [I_B]
        rw [hres, lintegral_map hm_h hmp.measurable]
    _ = ENNReal.ofReal (C * N ^ (-((k : ℝ) * n / 2)) * Real.log N ^ (1 - 1)) := by
        have hsqrt : Real.sqrt (Real.pi / N) ^ (k * n) =
            Real.sqrt Real.pi ^ (k * n) * N ^ (-((k : ℝ) * n / 2)) := by
          rw [Real.sqrt_div Real.pi_pos.le, div_pow]
          have hNsq : (Real.sqrt N) ^ (k * n) = N ^ (((k : ℝ) * n) / 2) := by
            rw [Real.sqrt_eq_rpow, ← Real.rpow_natCast, ← Real.rpow_mul hNpos.le]
            congr 1; push_cast; ring
          rw [hNsq, div_eq_mul_inv, ← Real.rpow_neg hNpos.le]
        rw [← ENNReal.ofReal_pow (Real.sqrt_nonneg _), hsqrt,
          ← ENNReal.ofReal_toReal hI_lt.ne, ← ENNReal.ofReal_mul (by positivity)]
        congr 1; dsimp [C]; simp; ring

/-- **Aoyagi–Watanabe (2005) Boundary Regime 2 (`k + m ≤ n`):**
By transpose symmetry `(B * A)ᵀ = Aᵀ * Bᵀ`, if `k + m ≤ n`, then the exact learning pair is
`(k * m / 2, 1)` unconditionally. -/
theorem rank0_hasLaplacePair_of_le_n {m k n : ℕ} (h : k + m ≤ n) :
    SLT.HasLaplacePair
      (SLT.Z (volume.restrict (matBox (a := m) (b := k) 1 ×ˢ matBox (a := k) (b := n) 1))
        rank0Loss)
      ((k : ℝ) * m / 2) 1 := by
  have hsym := rank0_hasLaplacePair_of_le_m (m := n) (k := k) (n := m) h
  refine hsym.congr (Eventually.of_forall fun N => ?_)
  let tr1 : RMat m k → RMat k m := fun B r i => B i r
  let tr2 : RMat k n → RMat n k := fun A j r => A r j
  let Φ : RMat m k × RMat k n → RMat n k × RMat k m := fun p => (tr2 p.2, tr1 p.1)
  have hmp : MeasurePreserving Φ volume volume :=
    ((measurePreserving_transpose k n).prod (measurePreserving_transpose m k)).comp
      Measure.measurePreserving_swap
  have hpre : Φ ⁻¹' (matBox (a := n) (b := k) 1 ×ˢ matBox (a := k) (b := m) 1) =
      matBox (a := m) (b := k) 1 ×ˢ matBox (a := k) (b := n) 1 := by
    ext ⟨B, A⟩
    simp only [Set.mem_preimage, Set.mem_prod, mem_matBox_iff, Φ, tr1, tr2]
    exact ⟨fun ⟨hA, hB⟩ => ⟨fun i r => hB r i, fun r j => hA j r⟩,
      fun ⟨hB, hA⟩ => ⟨fun j r => hA r j, fun r i => hB i r⟩⟩
  have hloss : ∀ p : RMat m k × RMat k n, rank0Loss (Φ p) = rank0Loss p := fun ⟨B, A⟩ => by
    unfold rank0Loss frobSq
    simp only [Φ, tr1, tr2, Matrix.mul_apply, Matrix.of_apply]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
    congr 1
    exact Finset.sum_congr rfl fun r _ => mul_comm _ _
  have hres : (volume : Measure (RMat n k × RMat k m)).restrict
      (matBox (a := n) (b := k) 1 ×ˢ matBox (a := k) (b := m) 1) =
      Measure.map Φ (volume.restrict (matBox (a := m) (b := k) 1 ×ˢ matBox (a := k) (b := n) 1)) := by
    rw [← hmp.map_eq, Measure.restrict_map hmp.measurable
      ((measurableSet_matBox 1).prod (measurableSet_matBox 1)), hpre]
  unfold SLT.Z
  rw [hres, lintegral_map (by have := measurable_rank0Loss (m := n) (k' := k) (n' := m); fun_prop)
    hmp.measurable]
  refine setLIntegral_congr_fun ((measurableSet_matBox 1).prod (measurableSet_matBox 1)) fun p _ => ?_
  rw [hloss p]

/-! ### 7. Crown Jewel: Unconditional Supercritical LoRA Fiber Parabola -/

/-- **Unconditional Supercritical LoRA Fiber Phase Transition (`n + k ≤ m + 2 * q`):**
For ALL `m, n, k, q : ℕ` with `q ≤ k`, `q ≤ n`, `k < m + n`, and `n + k ≤ m + 2 * q`,
the exact LoRA fiber loss around `(B₀, A₀) = (0, fromBlocks 1 0 0 0)` (rank-`q` half-dead `A₀`)
on every sufficiently small uniform box `fiberBox δ` has **unconditionally Lean-certified**
exact learning pair
`(awLambdaEven m n k + (q - (n + k - m)/2)^2 / 2, 1)`. -/
theorem lora_supercritical_parabola_exact_pair {m n k q : ℕ}
    (h1 : k < m + n) (hqk : q ≤ k) (hqn : q ≤ n) (hsup : n + k ≤ m + 2 * q) :
    ∃ δ₀ : ℝ, 0 < δ₀ ∧ ∀ δ : ℝ, 0 < δ → δ ≤ δ₀ →
      SLT.HasLaplacePair
        (SLT.Z (volume.restrict (fiberBox (m := m) (q := q) (k' := k - q) (n' := n - q) δ))
          loraFiberLoss)
        (awLambdaEven m n k + ((q : ℝ) - ((n : ℝ) + k - m) / 2) ^ 2 / 2)
        1 := by
  have hle : (k - q) + (n - q) ≤ m := by omega
  have hrem := rank0_hasLaplacePair_of_le_m (m := m) (k := k - q) (n := n - q) hle
  obtain ⟨δ₀, hδ₀, hred⟩ := lora_schur_fiber_reduction (q := q) hrem
  have hlam_eq :
      (m : ℝ) * q / 2 + ((k - q : ℕ) : ℝ) * ((n - q : ℕ) : ℝ) / 2 =
      awLambdaEven m n k + ((q : ℝ) - ((n : ℝ) + k - m) / 2) ^ 2 / 2 := by
    have hpar := lora_parabola_lambda h1 hqk hqn hsup
    have h1' : ¬ (m + (n - q) ≤ k - q) := by omega
    have h2' : (k - q) + (n - q) ≤ m := by omega
    unfold awLambda at hpar
    simp only [h1', h2', ↓reduceIte] at hpar
    exact hpar
  refine ⟨δ₀, hδ₀, fun δ hδ hδle => ?_⟩
  rw [← hlam_eq]
  exact hred δ hδ hδle

end EatYourFiber
