/-
Copyright (c) 2026 Bobby Faber. All rights reserved.
Released under Apache 2.0 license.
-/
module

public import EatYourFiber.AoyagiWatanabeBoundary
public import SLT.NormalCrossing.Cube

/-!
# Complete Unconditional Proof of Aoyagi–Watanabe (2005) Theorem 1 and the LoRA Fiber Phase Transition

This module proves, completely unconditionally in Lean 4 on top of `SLT` (zero Hironaka
resolution of singularities, zero SVD/Weyl Jacobian formula, zero polar coordinates):

1. `rank0Z_ncBound`: For **all** `m, k, n : ℕ`, the rank-0 matrix factorization partition function
   `Z_{m,k,n}(N) = ∫_{[-1,1]^{m×k} × [-1,1]^{k×n}} exp(-N ‖B A‖_F²) dB dA` satisfies
   `SLT.NC.NCBound (rank0Z m k n) (awLambda m n k) (awMult m n k)`.
   The proof is a single structural induction on `(m, k)` via the elementary projective blowup of
   `B` at its maximum-magnitude entry `t = B_{i₀, r₀}`:
   - `B = t · [[1, uᵀ], [v, B̃]]` (diagonal scaling Jacobian `t^{(m+1)(k+1) - 1}`),
   - two unit-Jacobian affine translations `â = a + uᵀ A₂` and `B̂ = B̃ - v uᵀ`,
   - exact two-sided loss equivalence `‖B A‖_F² ≍ t² (‖â‖₂² + ‖B̂ A₂‖_F²)`,
   - reducing directly to `SLT.NC.stepG` (the 1D normal-crossing integration step) on
     `gaussBoxZ n M * rank0Z m k n M`.
2. `aoyagi_watanabe_rank0`: **Full Aoyagi–Watanabe (2005) Theorem 1** across all four regimes
   (`m + n ≤ k`, `k + n ≤ m`, `k + m ≤ n`, and the interior triangle for both even and odd parity,
   with exact multiplicity `m_log ∈ {1, 2}`):
   `SLT.HasLaplacePair (rank0Z m k n) (awLambda m n k) (awMult m n k)`.
3. `lora_fiber_exact_pair`: **Complete unconditional LoRA zero-loss fiber theorem** for all
   `m, n, k, q : ℕ` with `q ≤ k` and `q ≤ n`, combining `lora_schur_fiber_reduction` with
   `aoyagi_watanabe_rank0`.
4. `lora_subcritical_plateau_exact_pair`: **Unconditional subcritical LoRA plateau** (`m + 2*q < n + k`),
   completing the entire LoRA fiber phase transition (flat plateau for `q ≤ q_c` + quadratic rise
   `(q - q_c)² / 2` for `q > q_c`) unconditionally in Lean 4.
-/

@[expose] public section

namespace EatYourFiber

open MeasureTheory Set Filter Matrix Finset Real
open scoped ENNReal NNReal

/-- Symmetric coordinate cube `[-ρ, ρ]^d ⊆ ℝ^d`. -/
def vecBox {d : ℕ} (ρ : ℝ) : Set (Fin d → ℝ) :=
  Set.pi Set.univ fun _ => Set.Icc (-ρ) ρ

theorem measurableSet_vecBox {d : ℕ} (ρ : ℝ) : MeasurableSet (vecBox (d := d) ρ) :=
  MeasurableSet.univ_pi fun _ => measurableSet_Icc

theorem mem_vecBox_iff {d : ℕ} {ρ : ℝ} {x : Fin d → ℝ} :
    x ∈ vecBox ρ ↔ ∀ i : Fin d, |x i| ≤ ρ := by
  simp only [vecBox, Set.mem_pi, Set.mem_univ, forall_const, Set.mem_Icc, abs_le]

theorem volume_vecBox {d : ℕ} {ρ : ℝ} (_hρ : 0 ≤ ρ) :
    volume (vecBox (d := d) ρ) = ENNReal.ofReal (2 * ρ) ^ d := by
  unfold vecBox
  change (Measure.pi fun _ : Fin d => (volume : Measure ℝ)) _ = _
  rw [Measure.pi_pi]
  simp [Real.volume_Icc, two_mul, sub_neg_eq_add]

/-- The rank-0 matrix factorization partition function on `[-1, 1]^{m×k} × [-1, 1]^{k×n}`. -/
noncomputable def rank0Z (m k n : ℕ) (N : ℝ) : ℝ≥0∞ :=
  SLT.Z (volume.restrict (matBox (a := m) (b := k) 1 ×ˢ matBox (a := k) (b := n) 1))
    rank0Loss N

/-- The `n`-dimensional unit-box Gaussian partition function `∫_{[-1,1]^n} exp(-N ‖a‖₂²) da`. -/
noncomputable def gaussBoxZ (n : ℕ) (N : ℝ) : ℝ≥0∞ :=
  ∫⁻ a in vecBox (d := n) 1, ENNReal.ofReal (Real.exp (-(N * ∑ j : Fin n, (a j) ^ 2)))

/-! ### 1. `SLT.NC.NCBound` combinators -/

/-- `NCBound` for a positive constant function has `(λ, m) = (0, 1)`. -/
theorem ncBound_const {c : ℝ} (hc : 0 < c) :
    SLT.NC.NCBound (fun _ => ENNReal.ofReal c) 0 1 := by
  refine ⟨fun _ _ _ _ => le_rfl, ⟨c, le_rfl⟩, ⟨c, hc, le_rfl⟩,
    ⟨c, fun M hM => by simp⟩, ⟨c, hc, 1, fun M hM => by simp⟩⟩

/-- `NCBound` is preserved under rescaling `M ↦ b * M` (`b > 0`, `λ ≥ 0`). -/
theorem ncBound_comp_mul {F : ℝ → ℝ≥0∞} {lam b : ℝ} {m : ℕ}
    (hF : SLT.NC.NCBound F lam m) (hlam : 0 ≤ lam) (hb : 0 < b) :
    SLT.NC.NCBound (fun M => F (b * M)) lam m := by
  obtain ⟨hmono, ⟨B, hB⟩, ⟨b0, hb0, hb1⟩, ⟨C, hC⟩, ⟨c, hc, M₀, hM₀⟩⟩ := hF
  have hB' : 0 ≤ max B 0 := le_max_right _ _
  have hC' : 0 ≤ max C 0 := le_max_right _ _
  let Mref : ℝ := max b (max M₀ (Real.exp 1))
  have hMref_ge_e : Real.exp 1 ≤ Mref := le_trans (le_max_right _ _) (le_max_right _ _)
  have hMref_ge_M0 : M₀ ≤ Mref := le_trans (le_max_left _ _) (le_max_right _ _)
  have hMref_ge_b : b ≤ Mref := le_max_left _ _
  have hMref_pos : 0 < Mref := lt_of_lt_of_le (Real.exp_pos 1) hMref_ge_e
  have hlog_Mref : 1 ≤ Real.log Mref := by
    rw [← Real.log_exp 1]
    exact Real.log_le_log (Real.exp_pos 1) hMref_ge_e
  have hFb_pos : 0 < c * Mref ^ (-lam) * Real.log Mref ^ (m - 1) := by
    have : 0 < Real.log Mref := by linarith
    positivity
  have hFb_le : ENNReal.ofReal (c * Mref ^ (-lam) * Real.log Mref ^ (m - 1)) ≤ F (b * 1) := by
    rw [mul_one]
    exact (hM₀ Mref hMref_ge_M0).trans (hmono b Mref hb.le hMref_ge_b)
  let Klog : ℝ := 1 + max 0 (Real.log b)
  have hKlog_ge1 : 1 ≤ Klog := by dsimp [Klog]; linarith [le_max_left (0 : ℝ) (Real.log b)]
  let Cup : ℝ := max (max C 0 * b ^ (-lam) * Klog ^ (m - 1)) (max B 0 * b ^ (-lam))
  let Mlo : ℝ := max (max (M₀ / b) (Real.exp 1 / b)) (max (Real.exp 1) (b⁻¹ ^ 2))
  refine ⟨fun M M' hM hMM => hmono (b * M) (b * M') (by positivity) (by nlinarith),
    ⟨B, by simpa using hB⟩,
    ⟨c * Mref ^ (-lam) * Real.log Mref ^ (m - 1), hFb_pos, hFb_le⟩,
    ⟨Cup, fun M hM => ?_⟩,
    ⟨c * b ^ (-lam) * (1 / 2) ^ (m - 1), by positivity, Mlo, fun M hM => ?_⟩⟩
  · have hMpos : 0 < M := by linarith
    have hlogM_nn : 0 ≤ Real.log M := Real.log_nonneg hM
    by_cases hbM : 1 ≤ b * M
    · have h1 := hC (b * M) hbM
      refine h1.trans (ENNReal.ofReal_le_ofReal ?_)
      have hlog_le : 1 + Real.log (b * M) ≤ Klog * (1 + Real.log M) := by
        rw [Real.log_mul hb.ne' hMpos.ne']
        have hlb : Real.log b ≤ max 0 (Real.log b) := le_max_right _ _
        dsimp [Klog]
        nlinarith [le_max_left (0 : ℝ) (Real.log b)]
      have hpow_le : (1 + Real.log (b * M)) ^ (m - 1) ≤
          Klog ^ (m - 1) * (1 + Real.log M) ^ (m - 1) := by
        rw [← mul_pow]
        exact pow_le_pow_left₀ (by linarith [Real.log_nonneg hbM]) hlog_le _
      have hlog1 : 0 ≤ 1 + Real.log M := by linarith
      have hlog_bM : 0 ≤ 1 + Real.log (b * M) := by linarith [Real.log_nonneg hbM]
      have hKlog_nn : 0 ≤ Klog := by linarith
      have hrpow : (b * M) ^ (-lam) = b ^ (-lam) * M ^ (-lam) :=
        Real.mul_rpow hb.le hMpos.le
      calc C * (b * M) ^ (-lam) * (1 + Real.log (b * M)) ^ (m - 1)
          ≤ max C 0 * (b ^ (-lam) * M ^ (-lam)) * (Klog ^ (m - 1) * (1 + Real.log M) ^ (m - 1)) := by
            rw [hrpow]
            have hC_le : C ≤ max C 0 := le_max_left _ _
            have h1' : C * (b ^ (-lam) * M ^ (-lam)) ≤ max C 0 * (b ^ (-lam) * M ^ (-lam)) :=
              mul_le_mul_of_nonneg_right hC_le (by positivity)
            exact mul_le_mul h1' hpow_le (by positivity) (by positivity)
        _ = (max C 0 * b ^ (-lam) * Klog ^ (m - 1)) * M ^ (-lam) * (1 + Real.log M) ^ (m - 1) := by
            ring
        _ ≤ Cup * M ^ (-lam) * (1 + Real.log M) ^ (m - 1) := by
            have hCup : max C 0 * b ^ (-lam) * Klog ^ (m - 1) ≤ Cup := le_max_left _ _
            exact mul_le_mul_of_nonneg_right
              (mul_le_mul_of_nonneg_right hCup (by positivity)) (by positivity)
    · push Not at hbM
      have hFM : F (b * M) ≤ ENNReal.ofReal B := (hmono 0 (b * M) le_rfl (by positivity)).trans hB
      refine hFM.trans (ENNReal.ofReal_le_ofReal ?_)
      have hM_le_inv : M ≤ b⁻¹ := by
        rw [inv_eq_one_div, le_div_iff₀' hb]
        linarith
      have hpow_ge : b ^ lam ≤ M ^ (-lam) := by
        have h1 : (b⁻¹) ^ (-lam) ≤ M ^ (-lam) :=
          Real.rpow_le_rpow_of_nonpos hMpos hM_le_inv (by linarith)
        rw [Real.inv_rpow hb.le, Real.rpow_neg hb.le, inv_inv] at h1
        exact h1
      have hlog_ge1 : (1 : ℝ) ≤ (1 + Real.log M) ^ (m - 1) :=
        one_le_pow₀ (by linarith)
      have hB_le : B ≤ max B 0 := le_max_left _ _
      have hCup2 : max B 0 * b ^ (-lam) ≤ Cup := le_max_right _ _
      calc B ≤ max B 0 := hB_le
        _ = max B 0 * b ^ (-lam) * b ^ lam * 1 := by
            rw [mul_assoc (max B 0), ← Real.rpow_add hb, neg_add_cancel, Real.rpow_zero]; ring
        _ ≤ Cup * M ^ (-lam) * (1 + Real.log M) ^ (m - 1) := by
            have h1 : max B 0 * b ^ (-lam) * b ^ lam ≤ Cup * M ^ (-lam) :=
              mul_le_mul hCup2 hpow_ge (by positivity) (by positivity)
            exact mul_le_mul h1 hlog_ge1 (by norm_num) (by positivity)
  · have hM_ge_M0b : M₀ / b ≤ M := le_trans (le_max_left _ _) (le_trans (le_max_left _ _) hM)
    have hM_ge_eb : Real.exp 1 / b ≤ M := le_trans (le_max_right _ _) (le_trans (le_max_left _ _) hM)
    have hM_ge_e : Real.exp 1 ≤ M := le_trans (le_max_left _ _) (le_trans (le_max_right _ _) hM)
    have hM_ge_inv2 : b⁻¹ ^ 2 ≤ M := le_trans (le_max_right _ _) (le_trans (le_max_right _ _) hM)
    have hMpos : 0 < M := lt_of_lt_of_le (Real.exp_pos 1) hM_ge_e
    have hbM_ge_M0 : M₀ ≤ b * M := by
      have := (div_le_iff₀ hb).mp hM_ge_M0b; linarith
    have hbM_ge_e : Real.exp 1 ≤ b * M := by
      have := (div_le_iff₀ hb).mp hM_ge_eb; linarith
    have hlogM_ge1 : 1 ≤ Real.log M := by
      rw [← Real.log_exp 1]; exact Real.log_le_log (Real.exp_pos 1) hM_ge_e
    have hlog_half : (1 / 2 : ℝ) * Real.log M ≤ Real.log (b * M) := by
      have h1 : Real.log (b⁻¹ ^ 2) ≤ Real.log M := Real.log_le_log (by positivity) hM_ge_inv2
      rw [Real.log_pow, Real.log_inv] at h1
      rw [Real.log_mul hb.ne' hMpos.ne']
      push_cast at h1
      linarith
    have hpow_log : ((1 / 2 : ℝ) ^ (m - 1)) * Real.log M ^ (m - 1) ≤ Real.log (b * M) ^ (m - 1) := by
      rw [← mul_pow]
      exact pow_le_pow_left₀ (by linarith) hlog_half _
    have hrpow : (b * M) ^ (-lam) = b ^ (-lam) * M ^ (-lam) := Real.mul_rpow hb.le hMpos.le
    refine (ENNReal.ofReal_le_ofReal ?_).trans (hM₀ (b * M) hbM_ge_M0)
    rw [hrpow]
    calc c * b ^ (-lam) * (1 / 2) ^ (m - 1) * M ^ (-lam) * Real.log M ^ (m - 1)
        = (c * (b ^ (-lam) * M ^ (-lam))) * ((1 / 2) ^ (m - 1) * Real.log M ^ (m - 1)) := by ring
      _ ≤ (c * (b ^ (-lam) * M ^ (-lam))) * Real.log (b * M) ^ (m - 1) :=
          mul_le_mul_of_nonneg_left hpow_log (by positivity)
      _ = c * (b ^ (-lam) * M ^ (-lam)) * Real.log (b * M) ^ (m - 1) := by ring

/-- Pointwise product of two `NCBound` functions adds `λ` and combines multiplicities `m₁ + m₂ - 1`. -/
theorem ncBound_mul {F₁ F₂ : ℝ → ℝ≥0∞} {lam₁ lam₂ : ℝ} {m₁ m₂ : ℕ}
    (h₁ : SLT.NC.NCBound F₁ lam₁ m₁) (h₂ : SLT.NC.NCBound F₂ lam₂ m₂)
    (hm₁ : 1 ≤ m₁) (hm₂ : 1 ≤ m₂) :
    SLT.NC.NCBound (fun M => F₁ M * F₂ M) (lam₁ + lam₂) (m₁ + m₂ - 1) := by
  obtain ⟨hmono₁, ⟨B₁, hB₁⟩, ⟨b₁, hb₁, hb1₁⟩, ⟨C₁, hC₁⟩, ⟨c₁, hc₁, M₀₁, hM₀₁⟩⟩ := h₁
  obtain ⟨hmono₂, ⟨B₂, hB₂⟩, ⟨b₂, hb₂, hb1₂⟩, ⟨C₂, hC₂⟩, ⟨c₂, hc₂, M₀₂, hM₀₂⟩⟩ := h₂
  have hexp : m₁ + m₂ - 1 - 1 = (m₁ - 1) + (m₂ - 1) := by omega
  have hB₁' : F₁ 0 ≤ ENNReal.ofReal (max B₁ 0) :=
    hB₁.trans (ENNReal.ofReal_le_ofReal (le_max_left _ _))
  have hB₂' : F₂ 0 ≤ ENNReal.ofReal (max B₂ 0) :=
    hB₂.trans (ENNReal.ofReal_le_ofReal (le_max_left _ _))
  refine ⟨fun M M' hM hMM => mul_le_mul' (hmono₁ M M' hM hMM) (hmono₂ M M' hM hMM),
    ⟨max B₁ 0 * max B₂ 0, by
      rw [ENNReal.ofReal_mul (le_max_right _ _)]
      exact mul_le_mul' hB₁' hB₂'⟩,
    ⟨b₁ * b₂, by positivity, by
      rw [ENNReal.ofReal_mul hb₁.le]
      exact mul_le_mul' hb1₁ hb1₂⟩,
    ⟨max C₁ 0 * max C₂ 0, fun M hM => ?_⟩,
    ⟨c₁ * c₂, by positivity, max (max M₀₁ M₀₂) 1, fun M hM => ?_⟩⟩
  · have hMpos : 0 < M := by linarith
    have hlog_nn : 0 ≤ 1 + Real.log M := by linarith [Real.log_nonneg hM]
    have h1' : F₁ M ≤ ENNReal.ofReal (max C₁ 0 * M ^ (-lam₁) * (1 + Real.log M) ^ (m₁ - 1)) :=
      (hC₁ M hM).trans (ENNReal.ofReal_le_ofReal
        (mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right (le_max_left _ _) (by positivity))
          (by positivity)))
    have h2' : F₂ M ≤ ENNReal.ofReal (max C₂ 0 * M ^ (-lam₂) * (1 + Real.log M) ^ (m₂ - 1)) :=
      (hC₂ M hM).trans (ENNReal.ofReal_le_ofReal
        (mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right (le_max_left _ _) (by positivity))
          (by positivity)))
    calc F₁ M * F₂ M
        ≤ ENNReal.ofReal (max C₁ 0 * M ^ (-lam₁) * (1 + Real.log M) ^ (m₁ - 1)) *
            ENNReal.ofReal (max C₂ 0 * M ^ (-lam₂) * (1 + Real.log M) ^ (m₂ - 1)) :=
          mul_le_mul' h1' h2'
      _ = ENNReal.ofReal (max C₁ 0 * max C₂ 0 * M ^ (-(lam₁ + lam₂)) *
            (1 + Real.log M) ^ (m₁ + m₂ - 1 - 1)) := by
          rw [← ENNReal.ofReal_mul (by positivity)]
          congr 1
          rw [hexp, pow_add, neg_add, Real.rpow_add hMpos]
          ring
  · have hM1 : M₀₁ ≤ M := le_trans (le_max_left _ _) (le_trans (le_max_left _ _) hM)
    have hM2 : M₀₂ ≤ M := le_trans (le_max_right _ _) (le_trans (le_max_left _ _) hM)
    have hM_ge1 : 1 ≤ M := le_trans (le_max_right _ _) hM
    have hMpos : 0 < M := by linarith
    have hlog_nn : 0 ≤ Real.log M := Real.log_nonneg hM_ge1
    calc ENNReal.ofReal (c₁ * c₂ * M ^ (-(lam₁ + lam₂)) * Real.log M ^ (m₁ + m₂ - 1 - 1))
        = ENNReal.ofReal (c₁ * M ^ (-lam₁) * Real.log M ^ (m₁ - 1)) *
            ENNReal.ofReal (c₂ * M ^ (-lam₂) * Real.log M ^ (m₂ - 1)) := by
          rw [← ENNReal.ofReal_mul (by positivity)]
          congr 1
          rw [hexp, pow_add, neg_add, Real.rpow_add hMpos]
          ring
      _ ≤ F₁ M * F₂ M := mul_le_mul' (hM₀₁ M hM1) (hM₀₂ M hM2)

/-- Sandwiching an antitone function `F` between two `NCBound` functions with the same `(λ, m)`
gives `NCBound F lam m`. -/
theorem ncBound_of_sandwich {F G₁ G₂ : ℝ → ℝ≥0∞} {lam : ℝ} {m : ℕ}
    (hmono : ∀ M M', 0 ≤ M → M ≤ M' → F M' ≤ F M)
    (h1 : ∀ M, 0 ≤ M → G₁ M ≤ F M)
    (h2 : ∀ M, 0 ≤ M → F M ≤ G₂ M)
    (hG₁ : SLT.NC.NCBound G₁ lam m)
    (hG₂ : SLT.NC.NCBound G₂ lam m) :
    SLT.NC.NCBound F lam m := by
  obtain ⟨-, -, ⟨b, hb, hb1⟩, -, ⟨c, hc, M₀, hM₀⟩⟩ := hG₁
  obtain ⟨-, ⟨B, hB⟩, -, ⟨C, hC⟩, -⟩ := hG₂
  refine ⟨hmono,
    ⟨B, (h2 0 le_rfl).trans hB⟩,
    ⟨b, hb, hb1.trans (h1 1 zero_le_one)⟩,
    ⟨C, fun M hM => (h2 M (by linarith)).trans (hC M hM)⟩,
    ⟨c, hc, max M₀ 0, fun M hM =>
      (hM₀ M (le_trans (le_max_left _ _) hM)).trans (h1 M (le_trans (le_max_right _ _) hM))⟩⟩

/-- 1D symmetric Gaussian box integral equals `2 * stepG (exp(-·)) 0 2`. -/
theorem gaussBoxZ_one (N : ℝ) :
    ∫⁻ x in Set.Icc (-1 : ℝ) 1, ENNReal.ofReal (Real.exp (-(N * x ^ 2))) =
    ENNReal.ofReal 2 * SLT.NC.stepG (fun M => ENNReal.ofReal (Real.exp (-M))) 0 2 N := by
  have hstep : SLT.NC.stepG (fun M => ENNReal.ofReal (Real.exp (-M))) 0 2 N =
      ∫⁻ x in Set.Ioc (0 : ℝ) 1, ENNReal.ofReal (Real.exp (-(N * x ^ 2))) := by
    unfold SLT.NC.stepG
    refine setLIntegral_congr_fun measurableSet_Ioc fun x _ => ?_
    rw [Real.rpow_zero, ENNReal.ofReal_one, one_mul]
  rw [hstep]
  have hneg : ∫⁻ x in Set.Ico (-1 : ℝ) 0, ENNReal.ofReal (Real.exp (-(N * x ^ 2))) =
      ∫⁻ x in Set.Ioc (0 : ℝ) 1, ENNReal.ofReal (Real.exp (-(N * x ^ 2))) := by
    have hmp := Measure.measurePreserving_neg (volume : Measure ℝ)
    have hpre : (fun x : ℝ => -x) ⁻¹' Set.Ioc (0 : ℝ) 1 = Set.Ico (-1 : ℝ) 0 := by
      ext x; simp only [Set.mem_preimage, Set.mem_Ioc, Set.mem_Ico]
      constructor <;> rintro ⟨a, b⟩ <;> exact ⟨by linarith, by linarith⟩
    have hres : (volume : Measure ℝ).restrict (Set.Ioc (0 : ℝ) 1) =
        Measure.map (fun x : ℝ => -x) (volume.restrict (Set.Ico (-1 : ℝ) 0)) := by
      conv_lhs => rw [← hmp.map_eq]
      rw [Measure.restrict_map measurable_neg measurableSet_Ioc, hpre]
    rw [hres, lintegral_map (by fun_prop) measurable_neg]
    refine setLIntegral_congr_fun measurableSet_Ico fun x _ => by simp
  have hae : (Set.Icc (-1 : ℝ) 1 : Set ℝ) =ᵐ[volume] Set.Ico (-1 : ℝ) 0 ∪ Set.Ioc (0 : ℝ) 1 := by
    have hdiff : Set.Icc (-1 : ℝ) 1 \ (Set.Ico (-1 : ℝ) 0 ∪ Set.Ioc (0 : ℝ) 1) ⊆ ({0} : Set ℝ) := by
      intro x ⟨⟨h1, h2⟩, hnot⟩
      simp only [Set.mem_union, Set.mem_Ico, Set.mem_Ioc, not_or] at hnot
      rcases lt_trichotomy x 0 with hlt | rfl | hgt
      · exact absurd ⟨h1, hlt⟩ hnot.1
      · rfl
      · exact absurd ⟨hgt, h2⟩ hnot.2
    have hsub : Set.Ico (-1 : ℝ) 0 ∪ Set.Ioc (0 : ℝ) 1 ⊆ Set.Icc (-1 : ℝ) 1 := by
      rintro x (⟨h1, h2⟩ | ⟨h1, h2⟩) <;> exact ⟨by linarith, by linarith⟩
    rw [ae_eq_set]
    exact ⟨measure_mono_null hdiff (by simp),
      by rw [Set.sdiff_eq_empty.mpr hsub, measure_empty]⟩
  rw [Measure.restrict_congr_set hae, lintegral_union measurableSet_Ioc (by
    rw [Set.disjoint_iff]; intro x ⟨h1, h2⟩; linarith [h1.2, h2.1]), hneg]
  rw [show ENNReal.ofReal 2 = 1 + 1 by norm_num, add_mul, one_mul]

/-- The `n`-dimensional unit-box Gaussian integral has `NCBound` with `(n / 2, 1)`. -/
theorem ncBound_gaussBoxZ : ∀ n : ℕ, SLT.NC.NCBound (gaussBoxZ n) ((n : ℝ) / 2) 1
  | 0 => by
    have heq : gaussBoxZ 0 = fun _ => ENNReal.ofReal 1 := by
      funext N
      unfold gaussBoxZ
      simp only [Finset.univ_eq_empty, Finset.sum_empty, mul_zero, neg_zero, Real.exp_zero,
        ENNReal.ofReal_one, setLIntegral_const, one_mul, volume_vecBox zero_le_one, pow_zero]
    rw [heq]
    simpa using ncBound_const one_pos
  | n + 1 => by
    have h1 : SLT.NC.NCBound
        (fun N => ∫⁻ x in Set.Icc (-1 : ℝ) 1, ENNReal.ofReal (Real.exp (-(N * x ^ 2))))
        (1 / 2) 1 := by
      have heq : (fun N => ∫⁻ x in Set.Icc (-1 : ℝ) 1, ENNReal.ofReal (Real.exp (-(N * x ^ 2)))) =
          fun N => ENNReal.ofReal 2 * SLT.NC.stepG (fun M => ENNReal.ofReal (Real.exp (-M))) 0 2 N :=
        funext gaussBoxZ_one
      rw [heq]
      have hbase := SLT.NC.NCBound.base (h := 0) (k := 2) (by norm_num) (by norm_num)
      norm_num at hbase
      exact hbase.const_mul (by norm_num)
    have hsplit : gaussBoxZ (n + 1) = fun N =>
        (∫⁻ x in Set.Icc (-1 : ℝ) 1, ENNReal.ofReal (Real.exp (-(N * x ^ 2)))) * gaussBoxZ n N := by
      funext N
      let e : (Fin (n + 1) → ℝ) ≃ᵐ ℝ × (Fin n → ℝ) :=
        MeasurableEquiv.piFinSuccAbove (fun _ => ℝ) 0
      have hmp : MeasurePreserving e volume volume :=
        volume_preserving_piFinSuccAbove (fun _ => ℝ) 0
      have hpre : e ⁻¹' (Set.Icc (-1 : ℝ) 1 ×ˢ vecBox (d := n) 1) = vecBox (d := n + 1) 1 := by
        ext x
        simp only [Set.mem_preimage, Set.mem_prod, vecBox, Set.mem_pi, Set.mem_univ, forall_const,
          e, MeasurableEquiv.piFinSuccAbove_apply, Fin.insertNthEquiv_symm_apply]
        constructor
        · rintro ⟨h0, htail⟩ i
          refine Fin.cases h0 (fun j => ?_) i
          simpa [Fin.tail] using htail j
        · intro hx
          exact ⟨hx 0, fun j => by simpa [Fin.tail] using hx (Fin.succ j)⟩
      have hres : (volume : Measure (ℝ × (Fin n → ℝ))).restrict
          (Set.Icc (-1 : ℝ) 1 ×ˢ vecBox (d := n) 1) =
          Measure.map e (volume.restrict (vecBox (d := n + 1) 1)) := by
        rw [← hmp.map_eq, Measure.restrict_map e.measurable
          (measurableSet_Icc.prod (measurableSet_vecBox 1)), hpre]
      let f1 : ℝ → ℝ≥0∞ := fun x => ENNReal.ofReal (Real.exp (-(N * x ^ 2)))
      let f2 : (Fin n → ℝ) → ℝ≥0∞ := fun v =>
        ENNReal.ofReal (Real.exp (-(N * ∑ j : Fin n, (v j) ^ 2)))
      have hf1 : Measurable f1 := by unfold f1; fun_prop
      have hf2 : Measurable f2 := by unfold f2; fun_prop
      have hprod_m : Measurable fun p : ℝ × (Fin n → ℝ) => f1 p.1 * f2 p.2 :=
        (hf1.comp measurable_fst).mul (hf2.comp measurable_snd)
      have hpt : ∀ x : Fin (n + 1) → ℝ,
          ENNReal.ofReal (Real.exp (-(N * ∑ j : Fin (n + 1), (x j) ^ 2))) =
          f1 (e x).1 * f2 (e x).2 := fun x => by
        unfold f1 f2
        rw [Fin.sum_univ_succ, mul_add, neg_add, Real.exp_add,
          ENNReal.ofReal_mul (Real.exp_pos _).le]
        rfl
      unfold gaussBoxZ
      simp_rw [hpt]
      rw [← lintegral_map hprod_m e.measurable, ← hres]
      change ∫⁻ p in Set.Icc (-1 : ℝ) 1 ×ˢ vecBox (d := n) 1, f1 p.1 * f2 p.2 ∂(volume.prod volume) = _
      rw [← Measure.prod_restrict, lintegral_prod_mul hf1.aemeasurable hf2.aemeasurable]
    rw [hsplit]
    have hmul := ncBound_mul h1 (ncBound_gaussBoxZ n) le_rfl le_rfl
    have hlam_eq : (1 / 2 : ℝ) + (n : ℝ) / 2 = ((n + 1 : ℕ) : ℝ) / 2 := by push_cast; ring
    have hm_eq : (1 + 1 - 1 : ℕ) = 1 := rfl
    rw [hlam_eq, hm_eq] at hmul
    exact hmul

/-! ### 2. Pure algebra of the Aoyagi–Watanabe recursion `(m, k, n) ↦ (m + 1, k + 1, n)` -/

theorem awLambda_nonneg (m n k : ℕ) : 0 ≤ awLambda m n k := by
  unfold awLambda awLambdaEven
  split_ifs with h1 h2 h3 h4
  · positivity
  · positivity
  · positivity
  · have hsum : (m : ℝ) + n > k := by exact_mod_cast (show k < m + n by omega)
    have hsum2 : (k : ℝ) + n > m := by exact_mod_cast (show m < k + n by omega)
    have hsum3 : (k : ℝ) + m > n := by exact_mod_cast (show n < k + m by omega)
    have hquad : 2 * ((m : ℝ) * n + n * k + k * m) - (m : ℝ) ^ 2 - (n : ℝ) ^ 2 - (k : ℝ) ^ 2 =
        4 * (m : ℝ) * k - ((m : ℝ) + k - n) ^ 2 := by ring
    rw [hquad]
    have hlt : (m : ℝ) + k - n < 2 * min (m : ℝ) k := by
      rcases le_total (m : ℝ) k with hmk | hkm
      · rw [min_eq_left hmk]; linarith
      · rw [min_eq_right hkm]; linarith
    have hsq : ((m : ℝ) + k - n) ^ 2 ≤ 4 * (m : ℝ) * k := by
      nlinarith [sq_nonneg ((m : ℝ) - k), min_le_left (m : ℝ) k, min_le_right (m : ℝ) k]
    linarith
  · have hsum : (m : ℝ) + n > k := by exact_mod_cast (show k < m + n by omega)
    have hsum2 : (k : ℝ) + n > m := by exact_mod_cast (show m < k + n by omega)
    have hsum3 : (k : ℝ) + m > n := by exact_mod_cast (show n < k + m by omega)
    have hquad : 2 * ((m : ℝ) * n + n * k + k * m) - (m : ℝ) ^ 2 - (n : ℝ) ^ 2 - (k : ℝ) ^ 2 =
        4 * (m : ℝ) * k - ((m : ℝ) + k - n) ^ 2 := by ring
    rw [hquad]
    have hlt : (m : ℝ) + k - n < 2 * min (m : ℝ) k := by
      rcases le_total (m : ℝ) k with hmk | hkm
      · rw [min_eq_left hmk]; linarith
      · rw [min_eq_right hkm]; linarith
    have hsq : ((m : ℝ) + k - n) ^ 2 ≤ 4 * (m : ℝ) * k := by
      nlinarith [sq_nonneg ((m : ℝ) - k), min_le_left (m : ℝ) k, min_le_right (m : ℝ) k]
    linarith

theorem awMult_pos (m n k : ℕ) : 1 ≤ awMult m n k := by
  unfold awMult
  split_ifs <;> omega

theorem awLambda_zero_left (n k : ℕ) : awLambda 0 n k = 0 := by
  unfold awLambda
  by_cases h : 0 + n ≤ k
  · simp only [h, ↓reduceIte]; norm_num
  · have h2 : ¬ (k + n ≤ 0) := by omega
    have h3 : k + 0 ≤ n := by omega
    simp only [h, h2, h3, ↓reduceIte]; norm_num

theorem awMult_zero_left (n k : ℕ) : awMult 0 n k = 1 := by
  unfold awMult
  by_cases h : 0 + n ≤ k
  · simp only [h, ↓reduceIte]
  · have h2 : ¬ (k + n ≤ 0) := by omega
    have h3 : k + 0 ≤ n := by omega
    simp only [h, h2, h3, ↓reduceIte]

theorem awLambda_zero_right (m n : ℕ) : awLambda m n 0 = 0 := by
  unfold awLambda
  by_cases h1 : m + n ≤ 0
  · have hm : m = 0 := by omega
    simp [hm]
  · by_cases h2 : 0 + n ≤ m
    · simp only [h1, h2, ↓reduceIte]; norm_num
    · have h3 : 0 + m ≤ n := by omega
      simp only [h1, h2, h3, ↓reduceIte]; norm_num

theorem awMult_zero_right (m n : ℕ) : awMult m n 0 = 1 := by
  unfold awMult
  by_cases h1 : m + n ≤ 0
  · simp only [h1, ↓reduceIte]
  · by_cases h2 : 0 + n ≤ m
    · simp only [h1, h2, ↓reduceIte]
    · have h3 : 0 + m ≤ n := by omega
      simp only [h1, h2, h3, ↓reduceIte]

/-- **Core algebraic recursion of Aoyagi–Watanabe (2005):**
Comparing `lam' := n / 2 + awLambda m n k` against `mu := (m + 1)(k + 1) / 2` reproduces
`(awLambda (m + 1) n (k + 1), awMult (m + 1) n (k + 1))` in all 3 cases (`<`, `=`, `>`). -/
theorem aw_recursion (m k n : ℕ) :
    let lam' := (n : ℝ) / 2 + awLambda m n k
    let mu := (((m + 1) * (k + 1) : ℕ) : ℝ) / 2
    (lam' < mu ∧ awLambda (m + 1) n (k + 1) = lam' ∧ awMult (m + 1) n (k + 1) = awMult m n k) ∨
    (lam' = mu ∧ awLambda (m + 1) n (k + 1) = mu ∧ awMult (m + 1) n (k + 1) = awMult m n k + 1) ∨
    (mu < lam' ∧ awLambda (m + 1) n (k + 1) = mu ∧ awMult (m + 1) n (k + 1) = 1) := by
  intro lam' mu
  dsimp only [lam', mu]
  by_cases h1 : (m + 1) + n ≤ k + 1
  · left
    have h1' : m + n ≤ k := by omega
    have hn_lt_K : (n : ℝ) < (k + 1 : ℕ) := by exact_mod_cast (show n < k + 1 by omega)
    unfold awLambda awMult
    simp only [h1, h1', ↓reduceIte]
    refine ⟨?_, by push_cast; ring, trivial⟩
    push_cast at hn_lt_K ⊢
    nlinarith [Nat.cast_nonneg (α := ℝ) m]
  · by_cases h2 : (k + 1) + n ≤ m + 1
    · left
      have h1' : ¬ (m + n ≤ k) := by omega
      have h2' : k + n ≤ m := by omega
      have hn_lt_M : (n : ℝ) < (m + 1 : ℕ) := by exact_mod_cast (show n < m + 1 by omega)
      unfold awLambda awMult
      simp only [h1, h2, h1', h2', ↓reduceIte]
      refine ⟨?_, by push_cast; ring, trivial⟩
      push_cast at hn_lt_M ⊢
      nlinarith [Nat.cast_nonneg (α := ℝ) k]
    · by_cases h3 : (k + 1) + (m + 1) ≤ n
      · right; right
        have h1' : ¬ (m + n ≤ k) := by omega
        have h2' : ¬ (k + n ≤ m) := by omega
        have h3' : k + m ≤ n := by omega
        have hsum_le : ((m : ℝ) + k + 2) ≤ n := by exact_mod_cast (show m + k + 2 ≤ n by omega)
        unfold awLambda awMult
        simp only [h1, h2, h3, h1', h2', h3', ↓reduceIte]
        refine ⟨?_, by push_cast; ring, trivial⟩
        push_cast
        linarith
      · have h1' : ¬ (m + n ≤ k) := by omega
        have h2' : ¬ (k + n ≤ m) := by omega
        by_cases hs1 : (m + 1) + (k + 1) = n + 1
        · right; left
          have h3' : k + m ≤ n := by omega
          have hodd : ¬ Even ((m + 1) + n + (k + 1)) := by
            have heq : (m + 1) + n + (k + 1) = 2 * n + 1 := by omega
            rw [heq]; exact Nat.not_even_iff_odd.mpr ⟨n, rfl⟩
          have hcast : (n : ℝ) = (m : ℝ) + k + 1 := by exact_mod_cast (show n = m + k + 1 by omega)
          unfold awLambda awMult awLambdaEven
          simp only [h1, h2, h3, h1', h2', h3', hodd, ↓reduceIte]
          refine ⟨?_, ?_, trivial⟩
          · push_cast; rw [hcast]; ring
          · push_cast; rw [hcast]; ring
        · by_cases hs2 : (m + 1) + (k + 1) = n + 2
          · left
            have h3' : k + m ≤ n := by omega
            have heven : Even ((m + 1) + n + (k + 1)) := by
              have heq : (m + 1) + n + (k + 1) = 2 * (n + 1) := by omega
              rw [heq]; exact even_two_mul _
            have hcast : (n : ℝ) = (m : ℝ) + k := by exact_mod_cast (show n = m + k by omega)
            unfold awLambda awMult awLambdaEven
            simp only [h1, h2, h3, h1', h2', h3', heven, ↓reduceIte]
            refine ⟨?_, ?_, trivial⟩
            · push_cast; rw [hcast]; linarith
            · push_cast; rw [hcast]; ring
          · left
            have h3' : ¬ (k + m ≤ n) := by omega
            have hpar : Even ((m + 1) + n + (k + 1)) ↔ Even (m + n + k) := by
              have heq : (m + 1) + n + (k + 1) = (m + n + k) + 2 := by omega
              rw [heq]; simp [Nat.even_add, even_two]
            have hs3 : (3 : ℝ) ≤ ((m : ℝ) + 1) + ((k : ℝ) + 1) - (n : ℝ) := by
              have : n + 3 ≤ (m + 1) + (k + 1) := by omega
              have hR : (n : ℝ) + 3 ≤ ((m + 1 : ℕ) : ℝ) + ((k + 1 : ℕ) : ℝ) := by exact_mod_cast this
              push_cast at hR; linarith
            unfold awLambda awMult awLambdaEven
            simp only [h1, h2, h3, h1', h2', h3', hpar, ↓reduceIte]
            split_ifs
            · refine ⟨?_, by push_cast; ring, trivial⟩
              push_cast; nlinarith
            · refine ⟨?_, by push_cast; ring, trivial⟩
              push_cast; nlinarith

/-! ### 3. Scaling helpers, block splitting, and blowup loss equivalence -/

/-- Scaling a `vecBox ρ` integral by `ρ > 0` pulls out `ρ^d`. -/
theorem lintegral_vecBox_smul {d : ℕ} {ρ : ℝ} (hρ : 0 < ρ)
    (f : (Fin d → ℝ) → ℝ≥0∞) (hf : Measurable f) :
    ∫⁻ x in vecBox (d := d) ρ, f x =
    ENNReal.ofReal ρ ^ d * ∫⁻ u in vecBox (d := d) 1, f (ρ • u) := by
  have h1 : Measure.map (fun x : ℝ => ρ * x) volume = ENNReal.ofReal (ρ⁻¹) • volume := by
    have h := Real.map_volume_mul_left hρ.ne'
    simpa [abs_of_pos (inv_pos.mpr hρ)] using h
  have hsf1 : SigmaFinite (ENNReal.ofReal (ρ⁻¹) • (volume : Measure ℝ)) := by
    have : ENNReal.ofReal (ρ⁻¹) • (volume : Measure ℝ) =
        ((ρ⁻¹).toNNReal : ℝ≥0) • (volume : Measure ℝ) := rfl
    rw [this]; infer_instance
  have hmap : Measure.map (fun v : Fin d → ℝ => ρ • v) volume =
      ENNReal.ofReal (ρ⁻¹) ^ d • volume := by
    have hsf : ∀ _ : Fin d, SigmaFinite (Measure.map (fun x : ℝ => ρ * x) volume) := fun _ => by
      rw [h1]; exact hsf1
    change Measure.map (fun (v : Fin d → ℝ) j => ρ * v j)
      (Measure.pi fun _ : Fin d => (volume : Measure ℝ)) = _
    rw [@Measure.pi_map_pi _ _ _ _ _ _ _ (fun _ : Fin d => fun x : ℝ => ρ * x) hsf
      (fun _ => (measurable_const.mul measurable_id).aemeasurable)]
    simp_rw [h1]
    have hsf_pi : ∀ _ : Fin d, SigmaFinite (ENNReal.ofReal (ρ⁻¹) • (volume : Measure ℝ)) :=
      fun _ => hsf1
    have hpi :
        Measure.pi (fun _ : Fin d => ENNReal.ofReal (ρ⁻¹) • (volume : Measure ℝ)) =
        ENNReal.ofReal (ρ⁻¹) ^ d • (volume : Measure (Fin d → ℝ)) := by
      refine Measure.pi_eq (μ := fun _ : Fin d => ENNReal.ofReal (ρ⁻¹) • (volume : Measure ℝ))
        (μ' := ENNReal.ofReal (ρ⁻¹) ^ d • (volume : Measure (Fin d → ℝ))) fun s hs => ?_
      rw [Measure.smul_apply, smul_eq_mul]
      change ENNReal.ofReal (ρ⁻¹) ^ d *
          (Measure.pi fun _ : Fin d => (volume : Measure ℝ)) (Set.univ.pi s) = _
      rw [Measure.pi_pi]
      simp [Finset.prod_mul_distrib, Finset.prod_const]
    exact hpi
  have hpre : (fun u : Fin d → ℝ => ρ • u) ⁻¹' vecBox ρ = vecBox 1 := by
    ext u
    simp only [Set.mem_preimage, mem_vecBox_iff, Pi.smul_apply, smul_eq_mul, abs_mul, abs_of_pos hρ]
    exact ⟨fun h i => (mul_le_iff_le_one_right hρ).mp (h i), fun h i => by nlinarith [h i]⟩
  have hkey :
      ∫⁻ u in vecBox (d := d) 1, f (ρ • u) =
      ENNReal.ofReal (ρ⁻¹) ^ d * ∫⁻ x in vecBox (d := d) ρ, f x := by
    rw [← lintegral_indicator (measurableSet_vecBox 1)]
    have hind : ∀ u : Fin d → ℝ,
        (vecBox (d := d) 1).indicator (fun v => f (ρ • v)) u =
        (vecBox (d := d) ρ).indicator f (ρ • u) := fun u => by
      by_cases hu : u ∈ vecBox 1
      · have hu' : ρ • u ∈ vecBox ρ := by rwa [← hpre] at hu
        rw [Set.indicator_of_mem hu, Set.indicator_of_mem hu']
      · have hu' : ρ • u ∉ vecBox ρ := by rwa [← hpre] at hu
        rw [Set.indicator_of_notMem hu, Set.indicator_of_notMem hu']
    simp_rw [hind]
    rw [← lintegral_map (hf.indicator (measurableSet_vecBox ρ)) (continuous_const_smul ρ).measurable,
      hmap, lintegral_smul_measure, smul_eq_mul, lintegral_indicator (measurableSet_vecBox ρ)]
  rw [hkey, ← mul_assoc, ← mul_pow, ← ENNReal.ofReal_mul hρ.le, mul_inv_cancel₀ hρ.ne',
    ENNReal.ofReal_one, one_pow, one_mul]

/-- Scaling a `matBox ρ` integral by `ρ > 0` pulls out `ρ^(a * b)`. -/
theorem lintegral_matBox_smul {a b : ℕ} {ρ : ℝ} (hρ : 0 < ρ)
    (f : RMat a b → ℝ≥0∞) (hf : Measurable f) :
    ∫⁻ X in matBox (a := a) (b := b) ρ, f X =
    ENNReal.ofReal ρ ^ (a * b) * ∫⁻ U in matBox (a := a) (b := b) 1, f (ρ • U) := by
  have hmap := map_smul_matrix_volume (a := a) (b := b) hρ
  have hpre : (fun U : RMat a b => ρ • U) ⁻¹' matBox ρ = matBox 1 := by
    ext U
    simp only [Set.mem_preimage, mem_matBox_iff, Pi.smul_apply, smul_eq_mul, abs_mul, abs_of_pos hρ]
    exact ⟨fun h i j => (mul_le_iff_le_one_right hρ).mp (h i j),
      fun h i j => by nlinarith [h i j]⟩
  have hkey :
      ∫⁻ U in matBox (a := a) (b := b) 1, f (ρ • U) =
      ENNReal.ofReal (ρ⁻¹) ^ (a * b) * ∫⁻ X in matBox (a := a) (b := b) ρ, f X := by
    rw [← lintegral_indicator (measurableSet_matBox 1)]
    have hind : ∀ U : RMat a b,
        (matBox (a := a) (b := b) 1).indicator (fun V => f (ρ • V)) U =
        (matBox (a := a) (b := b) ρ).indicator f (ρ • U) := fun U => by
      by_cases hU : U ∈ matBox 1
      · have hU' : ρ • U ∈ matBox ρ := by rwa [← hpre] at hU
        rw [Set.indicator_of_mem hU, Set.indicator_of_mem hU']
      · have hU' : ρ • U ∉ matBox ρ := by rwa [← hpre] at hU
        rw [Set.indicator_of_notMem hU, Set.indicator_of_notMem hU']
    simp_rw [hind]
    rw [← lintegral_map (hf.indicator (measurableSet_matBox ρ)) (continuous_const_smul ρ).measurable,
      hmap, lintegral_smul_measure, smul_eq_mul, lintegral_indicator (measurableSet_matBox ρ)]
  rw [hkey, ← mul_assoc, ← mul_pow, ← ENNReal.ofReal_mul hρ.le, mul_inv_cancel₀ hρ.ne',
    ENNReal.ofReal_one, one_pow, one_mul]

/-- Scaling `gaussBoxZ` from `vecBox ρ` to `vecBox 1`. -/
theorem gaussBoxZ_scale {n : ℕ} {ρ : ℝ} (hρ : 0 < ρ) (N : ℝ) :
    ∫⁻ a in vecBox (d := n) ρ, ENNReal.ofReal (Real.exp (-(N * ∑ j : Fin n, (a j) ^ 2))) =
    ENNReal.ofReal ρ ^ n * gaussBoxZ n (ρ ^ 2 * N) := by
  rw [lintegral_vecBox_smul hρ _ (by fun_prop)]
  congr 1
  unfold gaussBoxZ
  refine setLIntegral_congr_fun (measurableSet_vecBox 1) fun u _ => ?_
  congr 3
  simp only [Pi.smul_apply, smul_eq_mul, mul_pow, ← Finset.mul_sum]
  ring

/-- `gaussBoxZ n` is antitone on `[0, ∞)`. -/
theorem gaussBoxZ_mono (n : ℕ) {M M' : ℝ} (hMM : M ≤ M') :
    gaussBoxZ n M' ≤ gaussBoxZ n M := by
  unfold gaussBoxZ
  refine setLIntegral_mono' (measurableSet_vecBox 1) fun a _ => ?_
  apply ENNReal.ofReal_le_ofReal
  apply Real.exp_le_exp.mpr
  have hnn : 0 ≤ ∑ j : Fin n, (a j) ^ 2 := Finset.sum_nonneg fun _ _ => sq_nonneg _
  nlinarith

/-- `rank0Z m k n` is antitone on `[0, ∞)`. -/
theorem rank0Z_mono (m k n : ℕ) {M M' : ℝ} (hMM : M ≤ M') :
    rank0Z m k n M' ≤ rank0Z m k n M := by
  unfold rank0Z SLT.Z
  refine setLIntegral_mono' ((measurableSet_matBox 1).prod (measurableSet_matBox 1)) fun p _ => ?_
  apply ENNReal.ofReal_le_ofReal
  apply Real.exp_le_exp.mpr
  have hnn : 0 ≤ rank0Loss p := frobSq_nonneg _
  nlinarith

/-- Split `B : RMat (m + 1) (k + 1)` into `(B 0 0, top row tail, left col tail, bottom-right block)`. -/
def splitB (m k : ℕ) :
    RMat (m + 1) (k + 1) ≃ᵐ ℝ × (Fin k → ℝ) × (Fin m → ℝ) × RMat m k where
  toFun B := (B 0 0, fun r => B 0 r.succ, fun i => B i.succ 0, fun i r => B i.succ r.succ)
  invFun p := Fin.cons (Fin.cons p.1 p.2.1) (fun i => Fin.cons (p.2.2.1 i) (p.2.2.2 i))
  left_inv B := by
    funext i j
    refine Fin.cases ?_ (fun i' => ?_) i
    · refine Fin.cases rfl (fun r' => rfl) j
    · refine Fin.cases rfl (fun r' => rfl) j
  right_inv := fun ⟨_, _, _, _⟩ => rfl
  measurable_toFun :=
    (((measurable_pi_apply 0).comp (measurable_pi_apply 0)).prodMk
      ((measurable_pi_iff.mpr fun r => (measurable_pi_apply r.succ).comp (measurable_pi_apply 0)).prodMk
        ((measurable_pi_iff.mpr fun i => (measurable_pi_apply 0).comp (measurable_pi_apply i.succ)).prodMk
          (measurable_pi_iff.mpr fun i => measurable_pi_iff.mpr fun r =>
            (measurable_pi_apply r.succ).comp (measurable_pi_apply i.succ)))))
  measurable_invFun := measurable_pi_iff.mpr fun i => measurable_pi_iff.mpr fun j => by
    refine Fin.cases ?_ (fun i' => ?_) i
    · refine Fin.cases measurable_fst (fun r' => ?_) j
      exact (measurable_pi_apply r').comp (measurable_fst.comp measurable_snd)
    · refine Fin.cases ((measurable_pi_apply i').comp (measurable_fst.comp (measurable_snd.comp measurable_snd)))
        (fun r' => ?_) j
      exact (measurable_pi_apply r').comp
        ((measurable_pi_apply i').comp (measurable_snd.comp (measurable_snd.comp measurable_snd)))

theorem measurePreserving_splitB (m k : ℕ) :
    MeasurePreserving (splitB m k) volume volume := by
  have hRow : MeasurePreserving
      (MeasurableEquiv.piFinSuccAbove (fun _ : Fin (m + 1) => Fin (k + 1) → ℝ) 0)
      volume volume :=
    volume_preserving_piFinSuccAbove (fun _ => Fin (k + 1) → ℝ) 0
  have hTop : MeasurePreserving
      (MeasurableEquiv.piFinSuccAbove (fun _ : Fin (k + 1) => ℝ) 0)
      volume volume :=
    volume_preserving_piFinSuccAbove (fun _ => ℝ) 0
  have hBot : MeasurePreserving
      (fun X : RMat m (k + 1) => ((fun i => X i 0 : Fin m → ℝ), (fun i r => X i r.succ : RMat m k)))
      volume volume := by
    have htr1 := measurePreserving_transpose m (k + 1)
    have hpeel := volume_preserving_piFinSuccAbove (fun _ : Fin (k + 1) => Fin m → ℝ) 0
    have htr2 := (MeasurePreserving.id (volume : Measure (Fin m → ℝ))).prod
      (measurePreserving_transpose k m)
    have hcomp :
        ((Prod.map id (fun Y : RMat k m => (fun i r => Y r i : RMat m k))) ∘
          (⇑(MeasurableEquiv.piFinSuccAbove (fun _ : Fin (k + 1) => Fin m → ℝ) 0)) ∘
          (fun X : RMat m (k + 1) => (fun j i => X i j : RMat (k + 1) m))) =
        fun X : RMat m (k + 1) => ((fun i => X i 0 : Fin m → ℝ), (fun i r => X i r.succ : RMat m k)) := by
      funext X; rfl
    rw [← hcomp]
    exact htr2.comp (hpeel.comp htr1)
  have hAssoc : MeasurePreserving
      (MeasurableEquiv.prodAssoc :
        (ℝ × (Fin k → ℝ)) × ((Fin m → ℝ) × RMat m k) ≃ᵐ ℝ × (Fin k → ℝ) × (Fin m → ℝ) × RMat m k)
      volume volume :=
    volume_preserving_prodAssoc
  have hcomp_all :
      (⇑(splitB m k)) =
      (⇑(MeasurableEquiv.prodAssoc :
        (ℝ × (Fin k → ℝ)) × ((Fin m → ℝ) × RMat m k) ≃ᵐ ℝ × (Fin k → ℝ) × (Fin m → ℝ) × RMat m k)) ∘
      (Prod.map (⇑(MeasurableEquiv.piFinSuccAbove (fun _ : Fin (k + 1) => ℝ) 0))
        (fun X : RMat m (k + 1) => ((fun i => X i 0 : Fin m → ℝ), (fun i r => X i r.succ : RMat m k)))) ∘
      (⇑(MeasurableEquiv.piFinSuccAbove (fun _ : Fin (m + 1) => Fin (k + 1) → ℝ) 0)) := by
    funext B; rfl
  rw [hcomp_all]
  exact hAssoc.comp ((hTop.prod hBot).comp hRow)

/-- Split `A : RMat (k + 1) n` into `(top row, remaining rows)`. -/
def splitA (k n : ℕ) : RMat (k + 1) n ≃ᵐ (Fin n → ℝ) × RMat k n :=
  MeasurableEquiv.piFinSuccAbove (fun _ => Fin n → ℝ) 0

theorem measurePreserving_splitA (k n : ℕ) :
    MeasurePreserving (splitA k n) volume volume :=
  volume_preserving_piFinSuccAbove (fun _ => Fin n → ℝ) 0

/-- Assemble `(t, u, v, B₂)` into `RMat (m + 1) (k + 1)`. -/
def mkB {m k : ℕ} (t : ℝ) (u : Fin k → ℝ) (v : Fin m → ℝ) (B₂ : RMat m k) :
    RMat (m + 1) (k + 1) :=
  (splitB m k).symm (t, u, v, B₂)

/-- Assemble `(a, A₂)` into `RMat (k + 1) n`. -/
def mkA {k n : ℕ} (a : Fin n → ℝ) (A₂ : RMat k n) : RMat (k + 1) n :=
  (splitA k n).symm (a, A₂)

/-- Exact rank-1 shear identity for the projective blowup of `B`:
if `B = t · [[1, uᵀ], [v, B̃]]` and `A = [aᵀ; A₂]`, then writing
`â = a + uᵀ A₂` and `B̂ = B̃ - v uᵀ` gives
`‖B A‖_F² = t² (‖â‖₂² + ‖v âᵀ + B̂ A₂‖_F²)`. -/
theorem rank0Loss_blowup_eq {m k n : ℕ}
    (t : ℝ) (u : Fin k → ℝ) (v : Fin m → ℝ) (Btilde : RMat m k)
    (a : Fin n → ℝ) (A₂ : RMat k n) :
    let ahat : Fin n → ℝ := fun j => a j + ∑ r : Fin k, u r * A₂ r j
    let Bhat : RMat m k := fun i r => Btilde i r - v i * u r
    rank0Loss (mkB t (t • u) (t • v) (t • Btilde), mkA a A₂) =
      t ^ 2 * ((∑ j : Fin n, (ahat j) ^ 2) +
        ∑ i : Fin m, ∑ j : Fin n, (v i * ahat j + ∑ r : Fin k, Bhat i r * A₂ r j) ^ 2) := by
  intro ahat Bhat
  unfold rank0Loss frobSq
  simp only [Matrix.mul_apply, Matrix.of_apply]
  rw [Fin.sum_univ_succ]
  have htop : ∀ j : Fin n,
      (∑ r : Fin (k + 1), mkB t (t • u) (t • v) (t • Btilde) 0 r * mkA a A₂ r j) =
      t * ahat j := fun j => by
    rw [Fin.sum_univ_succ]
    have h0 : mkB t (t • u) (t • v) (t • Btilde) 0 0 = t := rfl
    have ha0 : mkA a A₂ 0 j = a j := rfl
    have hsucc : ∀ r : Fin k,
        mkB t (t • u) (t • v) (t • Btilde) 0 r.succ * mkA a A₂ r.succ j =
        t * (u r * A₂ r j) := fun r => by
      change (t * u r) * A₂ r j = _
      ring
    rw [h0, ha0]
    simp_rw [hsucc]
    rw [← Finset.mul_sum]
    dsimp [ahat]; ring
  have hbot : ∀ (i : Fin m) (j : Fin n),
      (∑ r : Fin (k + 1), mkB t (t • u) (t • v) (t • Btilde) i.succ r * mkA a A₂ r j) =
      t * (v i * ahat j + ∑ r : Fin k, Bhat i r * A₂ r j) := fun i j => by
    rw [Fin.sum_univ_succ]
    have h0 : mkB t (t • u) (t • v) (t • Btilde) i.succ 0 = t * v i := rfl
    have ha0 : mkA a A₂ 0 j = a j := rfl
    have hsucc : ∀ r : Fin k,
        mkB t (t • u) (t • v) (t • Btilde) i.succ r.succ * mkA a A₂ r.succ j =
        t * (Btilde i r * A₂ r j) := fun r => by
      change (t * Btilde i r) * A₂ r j = _
      ring
    rw [h0, ha0]
    simp_rw [hsucc]
    rw [← Finset.mul_sum]
    have hsplit : ∑ r : Fin k, Bhat i r * A₂ r j =
        (∑ r : Fin k, Btilde i r * A₂ r j) - v i * ∑ r : Fin k, u r * A₂ r j := by
      dsimp [Bhat]
      simp_rw [sub_mul, Finset.sum_sub_distrib, mul_assoc, ← Finset.mul_sum]
    rw [hsplit]
    dsimp [ahat]; ring
  simp_rw [htop, hbot, mul_pow, ← Finset.mul_sum]
  ring

/-- Two-sided loss comparison for the projective blowup when `v ∈ [-1, 1]^m`. -/
theorem rank0Loss_blowup_bounds {m k n : ℕ}
    (t : ℝ) (u : Fin k → ℝ) {v : Fin m → ℝ} (hv : v ∈ vecBox 1) (Btilde : RMat m k)
    (a : Fin n → ℝ) (A₂ : RMat k n) :
    let ahat : Fin n → ℝ := fun j => a j + ∑ r : Fin k, u r * A₂ r j
    let Bhat : RMat m k := fun i r => Btilde i r - v i * u r
    let S := (∑ j : Fin n, (ahat j) ^ 2) + rank0Loss (Bhat, A₂)
    (1 / (2 * (m : ℝ) + 2)) * (t ^ 2 * S) ≤
      rank0Loss (mkB t (t • u) (t • v) (t • Btilde), mkA a A₂) ∧
    rank0Loss (mkB t (t • u) (t • v) (t • Btilde), mkA a A₂) ≤
      (2 * (m : ℝ) + 2) * (t ^ 2 * S) := by
  intro ahat Bhat S
  rw [rank0Loss_blowup_eq]
  rw [mem_vecBox_iff] at hv
  set Sa : ℝ := ∑ j : Fin n, (ahat j) ^ 2
  set SBA : ℝ := rank0Loss (Bhat, A₂)
  set Smix : ℝ := ∑ i : Fin m, ∑ j : Fin n, (v i * ahat j + ∑ r : Fin k, Bhat i r * A₂ r j) ^ 2
  have hSa_nn : 0 ≤ Sa := Finset.sum_nonneg fun _ _ => sq_nonneg _
  have hSBA_nn : 0 ≤ SBA := frobSq_nonneg _
  have hSmix_nn : 0 ≤ Smix := Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _
  have hv_sq : ∀ i : Fin m, (v i) ^ 2 ≤ 1 := fun i => by
    have := pow_le_pow_left₀ (abs_nonneg (v i)) (hv i) 2
    rw [sq_abs] at this; linarith
  have hcross : ∑ i : Fin m, ∑ j : Fin n, (v i * ahat j) ^ 2 ≤ (m : ℝ) * Sa := by
    calc ∑ i : Fin m, ∑ j : Fin n, (v i * ahat j) ^ 2
        = ∑ i : Fin m, (v i) ^ 2 * Sa := by
          refine Finset.sum_congr rfl fun i _ => ?_
          simp_rw [mul_pow, ← Finset.mul_sum]; rfl
      _ ≤ ∑ _ : Fin m, 1 * Sa :=
          Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_right (hv_sq i) hSa_nn
      _ = (m : ℝ) * Sa := by simp [Finset.sum_const]
  have hSmix_le : Smix ≤ 2 * ((m : ℝ) * Sa) + 2 * SBA := by
    have h1 : Smix ≤ 2 * (∑ i : Fin m, ∑ j : Fin n, (v i * ahat j) ^ 2) + 2 * SBA := by
      unfold Smix SBA rank0Loss frobSq
      simp only [Matrix.mul_apply, Matrix.of_apply]
      rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
      refine Finset.sum_le_sum fun i _ => ?_
      rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
      refine Finset.sum_le_sum fun j _ => ?_
      nlinarith [sq_nonneg (v i * ahat j - ∑ r : Fin k, Bhat i r * A₂ r j)]
    linarith
  have hSBA_le : SBA ≤ 2 * Smix + 2 * ((m : ℝ) * Sa) := by
    have h1 : SBA ≤ 2 * Smix + 2 * (∑ i : Fin m, ∑ j : Fin n, (v i * ahat j) ^ 2) := by
      unfold Smix SBA rank0Loss frobSq
      simp only [Matrix.mul_apply, Matrix.of_apply]
      rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
      refine Finset.sum_le_sum fun i _ => ?_
      rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
      refine Finset.sum_le_sum fun j _ => ?_
      set y := ∑ r : Fin k, Bhat i r * A₂ r j
      nlinarith [sq_nonneg ((v i * ahat j + y) + v i * ahat j)]
    linarith
  have hm_pos : (0 : ℝ) < 2 * (m : ℝ) + 2 := by positivity
  have hm_nn : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg m
  constructor
  · have hlow : Sa + SBA ≤ (2 * (m : ℝ) + 2) * (Sa + Smix) := by
      nlinarith [mul_nonneg hm_nn hSmix_nn]
    have hlow' : (1 / (2 * (m : ℝ) + 2)) * (Sa + SBA) ≤ Sa + Smix := by
      rw [one_div, inv_mul_le_iff₀ hm_pos]; exact hlow
    have ht2 : 0 ≤ t ^ 2 := sq_nonneg t
    calc (1 / (2 * (m : ℝ) + 2)) * (t ^ 2 * (Sa + SBA))
        = t ^ 2 * ((1 / (2 * (m : ℝ) + 2)) * (Sa + SBA)) := by ring
      _ ≤ t ^ 2 * (Sa + Smix) := mul_le_mul_of_nonneg_left hlow' ht2
  · have hup : Sa + Smix ≤ (2 * (m : ℝ) + 2) * (Sa + SBA) := by
      nlinarith [mul_nonneg hm_nn hSBA_nn]
    have ht2 : 0 ≤ t ^ 2 := sq_nonneg t
    calc t ^ 2 * (Sa + Smix)
        ≤ t ^ 2 * ((2 * (m : ℝ) + 2) * (Sa + SBA)) := mul_le_mul_of_nonneg_left hup ht2
      _ = (2 * (m : ℝ) + 2) * (t ^ 2 * (Sa + SBA)) := by ring

/-! ### 4. Positive max-entry chart reduction and two-sided `stepG` sandwich -/

/-- The positive `(0, 0)`-dominant chart `{B ∈ [-1,1]^{(m+1)×(k+1)} | 0 < B 0 0 ∧ ∀ i r, |B i r| ≤ B 0 0}`. -/
def chartPos (m k : ℕ) : Set (RMat (m + 1) (k + 1)) :=
  {B | B ∈ matBox 1 ∧ 0 < B 0 0 ∧ ∀ i r, |B i r| ≤ B 0 0}

theorem measurableSet_chartPos (m k : ℕ) : MeasurableSet (chartPos m k) := by
  have heq : chartPos m k =
      matBox 1 ∩ {B : RMat (m + 1) (k + 1) | 0 < B 0 0} ∩
      ⋂ i : Fin (m + 1), ⋂ r : Fin (k + 1), {B : RMat (m + 1) (k + 1) | |B i r| ≤ B 0 0} := by
    ext B; simp only [chartPos, Set.mem_ofPred_eq, Set.mem_inter_iff, Set.mem_iInter]; tauto
  rw [heq]
  refine ((measurableSet_matBox 1).inter (measurableSet_lt measurable_const
    ((measurable_pi_apply 0).comp (measurable_pi_apply 0)))).inter ?_
  refine MeasurableSet.iInter fun i => MeasurableSet.iInter fun r => ?_
  exact measurableSet_le (((measurable_pi_apply r).comp (measurable_pi_apply i)).abs)
    ((measurable_pi_apply 0).comp (measurable_pi_apply 0))

/-- Partition function restricted to `chartPos m k ×ˢ matBox 1`. -/
noncomputable def chartPosZ (m k n : ℕ) (N : ℝ) : ℝ≥0∞ :=
  ∫⁻ p in chartPos m k ×ˢ matBox (a := k + 1) (b := n) 1,
    ENNReal.ofReal (Real.exp (-(N * rank0Loss p)))

/-- Symmetry reduction: `chartPosZ ≤ rank0Z ≤ 2 * (m + 1) * (k + 1) * chartPosZ`. -/
theorem rank0Z_sandwich_chartPos (m k n : ℕ) (N : ℝ) :
    chartPosZ m k n N ≤ rank0Z (m + 1) (k + 1) n N ∧
    rank0Z (m + 1) (k + 1) n N ≤
      ENNReal.ofReal (2 * ((m + 1 : ℕ) : ℝ) * ((k + 1 : ℕ) : ℝ)) * chartPosZ m k n N := by
  constructor
  · unfold chartPosZ rank0Z SLT.Z
    refine lintegral_mono_set ?_
    rintro ⟨B, A⟩ ⟨hB, hA⟩
    exact ⟨hB.1, hA⟩
  · -- Cover `matBox 1` by the `(m + 1) * (k + 1)` max-entry charts
    let chart : Fin (m + 1) → Fin (k + 1) → Set (RMat (m + 1) (k + 1)) :=
      fun i₀ r₀ => {B | B ∈ matBox 1 ∧ ∀ i r, |B i r| ≤ |B i₀ r₀|}
    have hcov : matBox (a := m + 1) (b := k + 1) 1 ⊆
        ⋃ p : Fin (m + 1) × Fin (k + 1), chart p.1 p.2 := by
      intro B hB
      have hne : (Finset.univ : Finset (Fin (m + 1) × Fin (k + 1))).Nonempty := ⟨(0, 0), Finset.mem_univ _⟩
      obtain ⟨⟨i₀, r₀⟩, _, hmax⟩ := Finset.exists_max_image Finset.univ
        (fun p : Fin (m + 1) × Fin (k + 1) => |B p.1 p.2|) hne
      exact Set.mem_iUnion.mpr ⟨(i₀, r₀), hB, fun i r => hmax (i, r) (Finset.mem_univ _)⟩
    have hcov_prod :
        matBox (a := m + 1) (b := k + 1) 1 ×ˢ matBox (a := k + 1) (b := n) 1 ⊆
        ⋃ p : Fin (m + 1) × Fin (k + 1), chart p.1 p.2 ×ˢ matBox (a := k + 1) (b := n) 1 := by
      rintro ⟨B, A⟩ ⟨hB, hA⟩
      obtain ⟨p, hp⟩ := Set.mem_iUnion.mp (hcov hB)
      exact Set.mem_iUnion.mpr ⟨p, hp, hA⟩
    -- Every `(i₀, r₀)` chart has the same integral as `(0, 0)` by row/col swap symmetry
    have hswap_eq : ∀ p : Fin (m + 1) × Fin (k + 1),
        ∫⁻ x in chart p.1 p.2 ×ˢ matBox (a := k + 1) (b := n) 1,
          ENNReal.ofReal (Real.exp (-(N * rank0Loss x))) =
        ∫⁻ x in chart 0 0 ×ˢ matBox (a := k + 1) (b := n) 1,
          ENNReal.ofReal (Real.exp (-(N * rank0Loss x))) := fun ⟨i₀, r₀⟩ => by
      let σ : Equiv.Perm (Fin (m + 1)) := Equiv.swap i₀ 0
      let τ : Equiv.Perm (Fin (k + 1)) := Equiv.swap r₀ 0
      let swapB : RMat (m + 1) (k + 1) → RMat (m + 1) (k + 1) := fun B i r => B (σ i) (τ r)
      let swapA : RMat (k + 1) n → RMat (k + 1) n := fun A r j => A (τ r) j
      let Φ : RMat (m + 1) (k + 1) × RMat (k + 1) n → RMat (m + 1) (k + 1) × RMat (k + 1) n :=
        Prod.map swapB swapA
      have hmpB : MeasurePreserving swapB volume volume := by
        have hrow : MeasurePreserving (fun B : RMat (m + 1) (k + 1) => fun i => B (σ i))
            volume volume := by
          have h := measurePreserving_piCongrLeft
            (fun _ : Fin (m + 1) => (volume : Measure (Fin (k + 1) → ℝ))) σ.symm
          have heq : (⇑(MeasurableEquiv.piCongrLeft (fun _ : Fin (m + 1) => Fin (k + 1) → ℝ) σ.symm)) =
              fun B i => B (σ i) := by
            funext B i
            have h1 := Equiv.piCongrLeft_apply_apply
              (fun _ : Fin (m + 1) => Fin (k + 1) → ℝ) σ.symm B (σ i)
            simpa [MeasurableEquiv.piCongrLeft] using h1
          rw [← heq]; exact h
        have hcol : MeasurePreserving (fun B : RMat (m + 1) (k + 1) => fun i r => B i (τ r))
            volume volume := by
          have h1 : MeasurePreserving (fun v : Fin (k + 1) → ℝ => fun r => v (τ r)) volume volume := by
            have h := measurePreserving_piCongrLeft (fun _ : Fin (k + 1) => (volume : Measure ℝ)) τ.symm
            have heq : (⇑(MeasurableEquiv.piCongrLeft (fun _ : Fin (k + 1) => ℝ) τ.symm)) =
                fun v r => v (τ r) := by
              funext v r
              have h2 := Equiv.piCongrLeft_apply_apply (fun _ : Fin (k + 1) => ℝ) τ.symm v (τ r)
              simpa [MeasurableEquiv.piCongrLeft] using h2
            rw [← heq]; exact h
          exact measurePreserving_pi (fun _ : Fin (m + 1) => (volume : Measure (Fin (k + 1) → ℝ)))
            (fun _ => volume) (fun _ => h1)
        exact hcol.comp hrow
      have hmpA : MeasurePreserving swapA volume volume := by
        have h := measurePreserving_piCongrLeft (fun _ : Fin (k + 1) => (volume : Measure (Fin n → ℝ))) τ.symm
        have heq : (⇑(MeasurableEquiv.piCongrLeft (fun _ : Fin (k + 1) => Fin n → ℝ) τ.symm)) = swapA := by
          funext A r
          have h2 := Equiv.piCongrLeft_apply_apply (fun _ : Fin (k + 1) => Fin n → ℝ) τ.symm A (τ r)
          simpa [MeasurableEquiv.piCongrLeft, swapA] using h2
        rw [← heq]; exact h
      have hmpΦ : MeasurePreserving Φ volume volume := hmpB.prod hmpA
      have hpre : Φ ⁻¹' (chart 0 0 ×ˢ matBox (a := k + 1) (b := n) 1) =
          chart i₀ r₀ ×ˢ matBox (a := k + 1) (b := n) 1 := by
        ext ⟨B, A⟩
        simp only [Set.mem_preimage, Set.mem_prod, Φ, Prod.map_apply, chart, Set.mem_ofPred_eq,
          mem_matBox_iff, swapB, swapA, σ, τ, Equiv.swap_apply_right]
        constructor
        · rintro ⟨⟨hB, hmax⟩, hA⟩
          refine ⟨⟨fun i r => by simpa [Equiv.swap_apply_self] using hB (Equiv.swap i₀ 0 i) (Equiv.swap r₀ 0 r),
            fun i r => by simpa [Equiv.swap_apply_self] using hmax (Equiv.swap i₀ 0 i) (Equiv.swap r₀ 0 r)⟩,
            fun r j => by simpa [Equiv.swap_apply_self] using hA (Equiv.swap r₀ 0 r) j⟩
        · rintro ⟨⟨hB, hmax⟩, hA⟩
          exact ⟨⟨fun i r => hB _ _, fun i r => hmax _ _⟩, fun r j => hA _ _⟩
      have hloss : ∀ x, rank0Loss (Φ x) = rank0Loss x := fun ⟨B, A⟩ => by
        unfold rank0Loss frobSq
        simp only [Φ, Prod.map_apply, swapB, swapA, Matrix.mul_apply, Matrix.of_apply]
        rw [Equiv.sum_comp σ (fun i => ∑ j : Fin n, (∑ r : Fin (k + 1), B i (τ r) * A (τ r) j) ^ 2)]
        refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
        congr 1
        exact Equiv.sum_comp τ (fun r => B i r * A r j)
      have hchart0_meas : MeasurableSet (chart 0 0) := by
        have heq : chart 0 0 =
            matBox 1 ∩ ⋂ i : Fin (m + 1), ⋂ r : Fin (k + 1),
              {B : RMat (m + 1) (k + 1) | |B i r| ≤ |B 0 0|} := by
          ext B; simp [chart]
        rw [heq]
        refine (measurableSet_matBox 1).inter (MeasurableSet.iInter fun i => MeasurableSet.iInter fun r => ?_)
        exact measurableSet_le (((measurable_pi_apply r).comp (measurable_pi_apply i)).abs)
          (((measurable_pi_apply 0).comp (measurable_pi_apply 0)).abs)
      have hres : (volume : Measure (RMat (m + 1) (k + 1) × RMat (k + 1) n)).restrict
          (chart 0 0 ×ˢ matBox 1) =
          Measure.map Φ (volume.restrict (chart i₀ r₀ ×ˢ matBox 1)) := by
        conv_lhs => rw [← hmpΦ.map_eq]
        rw [Measure.restrict_map hmpΦ.measurable (hchart0_meas.prod (measurableSet_matBox 1)), hpre]
      have hm_exp : Measurable fun x : RMat (m + 1) (k + 1) × RMat (k + 1) n =>
          ENNReal.ofReal (Real.exp (-(N * rank0Loss x))) := by
        have := measurable_rank0Loss (m := m + 1) (k' := k + 1) (n' := n); fun_prop
      rw [hres, lintegral_map hm_exp hmpΦ.measurable]
      refine setLIntegral_congr_fun ?_ fun x _ => by rw [hloss x]
      rw [← hpre]; exact hmpΦ.measurable (hchart0_meas.prod (measurableSet_matBox 1))
    -- On `chart 0 0`, split into `B 0 0 > 0`, `B 0 0 < 0` (equal by `B ↦ -B`), and `B 0 0 = 0` (null)
    let chartNeg : Set (RMat (m + 1) (k + 1)) :=
      {B | B ∈ matBox 1 ∧ B 0 0 < 0 ∧ ∀ i r, |B i r| ≤ -B 0 0}
    let chartZero : Set (RMat (m + 1) (k + 1)) := {B | B 0 0 = 0}
    have hchart0_cov : chart 0 0 ⊆ chartPos m k ∪ (chartNeg ∪ chartZero) := by
      intro B ⟨hB, hmax⟩
      rcases lt_trichotomy (B 0 0) 0 with hlt | heq | hgt
      · refine Or.inr (Or.inl ⟨hB, hlt, fun i r => ?_⟩)
        simpa [abs_of_neg hlt] using hmax i r
      · exact Or.inr (Or.inr heq)
      · refine Or.inl ⟨hB, hgt, fun i r => ?_⟩
        simpa [abs_of_pos hgt] using hmax i r
    have hzero_null : volume (chartZero ×ˢ matBox (a := k + 1) (b := n) 1) = 0 := by
      have hvol0 : (volume : Measure (RMat (m + 1) (k + 1))) chartZero = 0 := by
        have hpre : (splitB m k) ⁻¹' (({0} : Set ℝ) ×ˢ Set.univ) = chartZero := by
          ext B; simp [chartZero, splitB]
        rw [← hpre, (measurePreserving_splitB m k).measure_preimage
          (MeasurableSet.nullMeasurableSet ((measurableSet_singleton 0).prod MeasurableSet.univ))]
        change (volume.prod volume) (({0} : Set ℝ) ×ˢ Set.univ) = 0
        rw [Measure.prod_prod, Real.volume_singleton, zero_mul]
      change (volume.prod volume) (chartZero ×ˢ matBox 1) = 0
      rw [Measure.prod_prod, hvol0, zero_mul]
    have hneg_eq :
        ∫⁻ x in chartNeg ×ˢ matBox (a := k + 1) (b := n) 1,
          ENNReal.ofReal (Real.exp (-(N * rank0Loss x))) =
        chartPosZ m k n N := by
      let negB : RMat (m + 1) (k + 1) → RMat (m + 1) (k + 1) := fun B => -B
      have hmpNeg : MeasurePreserving negB volume volume := Measure.measurePreserving_neg volume
      have hmpΦ : MeasurePreserving (Prod.map negB (id : RMat (k + 1) n → RMat (k + 1) n))
          volume volume := by
        change MeasurePreserving (Prod.map negB (id : RMat (k + 1) n → RMat (k + 1) n))
          (volume.prod volume) (volume.prod volume)
        exact hmpNeg.prod (MeasurePreserving.id volume)
      have hpre : (Prod.map negB id) ⁻¹' (chartPos m k ×ˢ matBox (a := k + 1) (b := n) 1) =
          chartNeg ×ˢ matBox (a := k + 1) (b := n) 1 := by
        ext ⟨B, A⟩
        simp only [Set.mem_preimage, Set.mem_prod, Prod.map_apply, id_eq, chartPos, chartNeg,
          Set.mem_ofPred_eq, mem_matBox_iff, negB, Pi.neg_apply, abs_neg, Left.neg_pos_iff]
      have hres : (volume : Measure (RMat (m + 1) (k + 1) × RMat (k + 1) n)).restrict
          (chartPos m k ×ˢ matBox 1) =
          Measure.map (Prod.map negB id) (volume.restrict (chartNeg ×ˢ matBox 1)) := by
        conv_lhs => rw [← hmpΦ.map_eq]
        rw [Measure.restrict_map hmpΦ.measurable ((measurableSet_chartPos m k).prod (measurableSet_matBox 1)), hpre]
      have hm_exp : Measurable fun x : RMat (m + 1) (k + 1) × RMat (k + 1) n =>
          ENNReal.ofReal (Real.exp (-(N * rank0Loss x))) := by
        have := measurable_rank0Loss (m := m + 1) (k' := k + 1) (n' := n); fun_prop
      unfold chartPosZ
      rw [hres, lintegral_map hm_exp hmpΦ.measurable]
      refine setLIntegral_congr_fun ?_ fun ⟨B, A⟩ _ => ?_
      · rw [← hpre]; exact hmpΦ.measurable ((measurableSet_chartPos m k).prod (measurableSet_matBox 1))
      · congr 4
        unfold rank0Loss frobSq
        simp only [Prod.map_apply, id_eq, negB, Matrix.mul_apply, Matrix.of_apply, Pi.neg_apply]
        refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
        have hsum : (∑ r : Fin (k + 1), -B i r * A r j) = -(∑ r : Fin (k + 1), B i r * A r j) := by
          simp_rw [neg_mul, Finset.sum_neg_distrib]
        rw [hsum, neg_sq]
    have hchart0_le :
        ∫⁻ x in chart 0 0 ×ˢ matBox (a := k + 1) (b := n) 1,
          ENNReal.ofReal (Real.exp (-(N * rank0Loss x))) ≤
        ENNReal.ofReal 2 * chartPosZ m k n N := by
      have hsub : chart 0 0 ×ˢ matBox (a := k + 1) (b := n) 1 ⊆
          (chartPos m k ×ˢ matBox 1) ∪ ((chartNeg ×ˢ matBox 1) ∪ (chartZero ×ˢ matBox 1)) := by
        rintro ⟨B, A⟩ ⟨hB, hA⟩
        rcases hchart0_cov hB with h1 | h2 | h3
        · exact Or.inl ⟨h1, hA⟩
        · exact Or.inr (Or.inl ⟨h2, hA⟩)
        · exact Or.inr (Or.inr ⟨h3, hA⟩)
      calc ∫⁻ x in chart 0 0 ×ˢ matBox 1, ENNReal.ofReal (Real.exp (-(N * rank0Loss x)))
          ≤ ∫⁻ x in (chartPos m k ×ˢ matBox 1) ∪ ((chartNeg ×ˢ matBox 1) ∪ (chartZero ×ˢ matBox 1)),
              ENNReal.ofReal (Real.exp (-(N * rank0Loss x))) :=
            lintegral_mono_set hsub
        _ ≤ chartPosZ m k n N +
              ((∫⁻ x in chartNeg ×ˢ matBox 1, ENNReal.ofReal (Real.exp (-(N * rank0Loss x)))) +
                ∫⁻ x in chartZero ×ˢ matBox 1, ENNReal.ofReal (Real.exp (-(N * rank0Loss x)))) :=
            (lintegral_union_le _ _ _).trans (add_le_add le_rfl (lintegral_union_le _ _ _))
        _ = chartPosZ m k n N + (chartPosZ m k n N + 0) := by
            rw [hneg_eq, setLIntegral_measure_zero _ _ hzero_null]
        _ = ENNReal.ofReal 2 * chartPosZ m k n N := by
            rw [add_zero, show ENNReal.ofReal 2 = 1 + 1 by norm_num, add_mul, one_mul]
    calc rank0Z (m + 1) (k + 1) n N
        ≤ ∫⁻ x in ⋃ p : Fin (m + 1) × Fin (k + 1), chart p.1 p.2 ×ˢ matBox 1,
            ENNReal.ofReal (Real.exp (-(N * rank0Loss x))) :=
          lintegral_mono_set hcov_prod
      _ ≤ ∑' p : Fin (m + 1) × Fin (k + 1),
            ∫⁻ x in chart p.1 p.2 ×ˢ matBox 1, ENNReal.ofReal (Real.exp (-(N * rank0Loss x))) :=
          lintegral_iUnion_le _ _
      _ = ∑ _ : Fin (m + 1) × Fin (k + 1),
            ∫⁻ x in chart 0 0 ×ˢ matBox 1, ENNReal.ofReal (Real.exp (-(N * rank0Loss x))) := by
          rw [tsum_fintype]
          exact Finset.sum_congr rfl fun p _ => hswap_eq p
      _ = ((m + 1) * (k + 1) : ℕ) *
            ∫⁻ x in chart 0 0 ×ˢ matBox 1, ENNReal.ofReal (Real.exp (-(N * rank0Loss x))) := by
          simp [Finset.sum_const]
      _ ≤ ((m + 1) * (k + 1) : ℕ) * (ENNReal.ofReal 2 * chartPosZ m k n N) :=
          mul_le_mul' le_rfl hchart0_le
      _ = ENNReal.ofReal (2 * ((m + 1 : ℕ) : ℝ) * ((k + 1 : ℕ) : ℝ)) * chartPosZ m k n N := by
          rw [← mul_assoc]
          congr 1
          rw [show (((m + 1) * (k + 1) : ℕ) : ℝ≥0∞) =
              ENNReal.ofReal (((m + 1 : ℕ) : ℝ) * ((k + 1 : ℕ) : ℝ)) by
                rw [← ENNReal.ofReal_natCast, Nat.cast_mul],
            ← ENNReal.ofReal_mul (by positivity)]
          congr 1; ring

/-- Iterated Fubini representation of `chartPosZ` before rescaling by `t`. -/
private theorem chartPosZ_eq_iterated (m k n : ℕ) (N : ℝ) :
    chartPosZ m k n N =
    ∫⁻ t in Set.Ioc (0 : ℝ) 1,
      ∫⁻ u' in vecBox (d := k) t,
      ∫⁻ v' in vecBox (d := m) t,
      ∫⁻ B' in matBox (a := m) (b := k) t,
      ∫⁻ a in vecBox (d := n) 1,
      ∫⁻ A₂ in matBox (a := k) (b := n) 1,
        ENNReal.ofReal (Real.exp (-(N * rank0Loss (mkB t u' v' B', mkA a A₂)))) := by
  let Φ : RMat (m + 1) (k + 1) × RMat (k + 1) n ≃ᵐ
      (ℝ × (Fin k → ℝ) × (Fin m → ℝ) × RMat m k) × ((Fin n → ℝ) × RMat k n) :=
    (splitB m k).prodCongr (splitA k n)
  have hmpΦ : MeasurePreserving Φ volume volume :=
    (measurePreserving_splitB m k).prod (measurePreserving_splitA k n)
  let S : Set ((ℝ × (Fin k → ℝ) × (Fin m → ℝ) × RMat m k) × ((Fin n → ℝ) × RMat k n)) :=
    {q | q.1.1 ∈ Set.Ioc (0 : ℝ) 1 ∧ q.1.2.1 ∈ vecBox q.1.1 ∧ q.1.2.2.1 ∈ vecBox q.1.1 ∧
      q.1.2.2.2 ∈ matBox q.1.1 ∧ q.2.1 ∈ vecBox 1 ∧ q.2.2 ∈ matBox 1}
  have hpre : Φ ⁻¹' S = chartPos m k ×ˢ matBox (a := k + 1) (b := n) 1 := by
    ext ⟨B, A⟩
    have hΦ : Φ (B, A) =
        ((B 0 0, fun r => B 0 r.succ, fun i => B i.succ 0, fun i r => B i.succ r.succ),
          (A 0, fun r => A r.succ)) := rfl
    simp only [Set.mem_preimage, Set.mem_prod, hΦ, S, Set.mem_ofPred_eq, chartPos,
      mem_matBox_iff, mem_vecBox_iff, Set.mem_Ioc]
    constructor
    · rintro ⟨⟨ht0, ht1⟩, hu, hv, hB2, ha, hA2⟩
      refine ⟨⟨?_, ht0, ?_⟩, ?_⟩
      · intro i r
        refine Fin.cases (Fin.cases ?_ (fun r' => (hu r').trans ht1) r)
          (fun i' => Fin.cases ((hv i').trans ht1) (fun r' => (hB2 i' r').trans ht1) r) i
        simpa [abs_of_pos ht0] using ht1
      · intro i r
        refine Fin.cases (Fin.cases ?_ (fun r' => hu r') r)
          (fun i' => Fin.cases (hv i') (fun r' => hB2 i' r') r) i
        simp [abs_of_pos ht0]
      · intro r j
        exact Fin.cases (ha j) (fun r' => hA2 r' j) r
    · rintro ⟨⟨hB, hB0, hmax⟩, hA⟩
      have hB0_le1 : B 0 0 ≤ 1 := by simpa [abs_of_pos hB0] using hB 0 0
      exact ⟨⟨hB0, hB0_le1⟩, fun r => hmax 0 r.succ, fun i => hmax i.succ 0,
        fun i r => hmax i.succ r.succ, fun j => hA 0 j, fun r j => hA r.succ j⟩
  have hS_meas : MeasurableSet S := by
    have hS_eq : S = Φ.symm ⁻¹' (chartPos m k ×ˢ matBox (a := k + 1) (b := n) 1) := by
      rw [← hpre]; ext q; simp
    rw [hS_eq]
    exact Φ.symm.measurable ((measurableSet_chartPos m k).prod (measurableSet_matBox 1))
  let F : (ℝ × (Fin k → ℝ) × (Fin m → ℝ) × RMat m k) × ((Fin n → ℝ) × RMat k n) → ℝ≥0∞ :=
    fun q => ENNReal.ofReal (Real.exp (-(N * rank0Loss (Φ.symm q))))
  have hF_meas : Measurable F := by
    unfold F
    have := (measurable_rank0Loss (m := m + 1) (k' := k + 1) (n' := n)).comp Φ.symm.measurable
    fun_prop
  have hres : volume.restrict S =
      Measure.map Φ (volume.restrict (chartPos m k ×ˢ matBox (a := k + 1) (b := n) 1)) := by
    conv_lhs => rw [← hmpΦ.map_eq]
    rw [Measure.restrict_map hmpΦ.measurable hS_meas, hpre]
  have hstep1 : chartPosZ m k n N = ∫⁻ q in S, F q := by
    unfold chartPosZ
    rw [hres, lintegral_map hF_meas hmpΦ.measurable]
    refine setLIntegral_congr_fun ((measurableSet_chartPos m k).prod (measurableSet_matBox 1)) fun p _ => ?_
    simp [F]
  rw [hstep1, ← lintegral_indicator hS_meas]
  change ∫⁻ q : (ℝ × (Fin k → ℝ) × (Fin m → ℝ) × RMat m k) × ((Fin n → ℝ) × RMat k n),
      S.indicator F q ∂(volume.prod volume) = _
  rw [lintegral_prod _ (hF_meas.indicator hS_meas).aemeasurable]
  have hm1 : Measurable fun x : ℝ × (Fin k → ℝ) × (Fin m → ℝ) × RMat m k =>
      ∫⁻ y : (Fin n → ℝ) × RMat k n, S.indicator F (x, y) :=
    (hF_meas.indicator hS_meas).lintegral_prod_right'
  change ∫⁻ x : ℝ × ((Fin k → ℝ) × (Fin m → ℝ) × RMat m k), _ ∂(volume.prod volume) = _
  rw [lintegral_prod _ hm1.aemeasurable]
  rw [← lintegral_indicator measurableSet_Ioc]
  refine lintegral_congr fun t => ?_
  by_cases ht : t ∈ Set.Ioc (0 : ℝ) 1
  · rw [Set.indicator_of_mem ht]
    let S_t : Set ((Fin k → ℝ) × (Fin m → ℝ) × RMat m k) :=
      vecBox t ×ˢ (vecBox t ×ˢ matBox t)
    have hSt_meas : MeasurableSet S_t :=
      (measurableSet_vecBox t).prod ((measurableSet_vecBox t).prod (measurableSet_matBox t))
    have hinner_eq : ∀ p : (Fin k → ℝ) × (Fin m → ℝ) × RMat m k,
        (∫⁻ y : (Fin n → ℝ) × RMat k n, S.indicator F ((t, p), y)) =
        S_t.indicator (fun p' =>
          ∫⁻ a in vecBox (d := n) 1, ∫⁻ A₂ in matBox (a := k) (b := n) 1,
            F ((t, p'), (a, A₂))) p := fun ⟨u', v', B'⟩ => by
      by_cases hp : (u', v', B') ∈ S_t
      · rw [Set.indicator_of_mem hp]
        have hmem : ∀ y : (Fin n → ℝ) × RMat k n,
            ((t, u', v', B'), y) ∈ S ↔ y ∈ vecBox (d := n) 1 ×ˢ matBox (a := k) (b := n) 1 := fun ⟨a, A₂⟩ => by
          simp only [S, S_t, Set.mem_ofPred_eq, Set.mem_prod] at hp ⊢
          exact ⟨fun h => ⟨h.2.2.2.2.1, h.2.2.2.2.2⟩,
            fun h => ⟨ht, hp.1, hp.2.1, hp.2.2, h.1, h.2⟩⟩
        have hind_y : ∀ y, S.indicator F ((t, u', v', B'), y) =
            (vecBox (d := n) 1 ×ˢ matBox (a := k) (b := n) 1).indicator
              (fun y' => F ((t, u', v', B'), y')) y := fun y => by
          by_cases hy : y ∈ vecBox 1 ×ˢ matBox 1
          · rw [Set.indicator_of_mem ((hmem y).mpr hy), Set.indicator_of_mem hy]
          · rw [Set.indicator_of_notMem (mt (hmem y).mp hy), Set.indicator_of_notMem hy]
        simp_rw [hind_y]
        rw [lintegral_indicator ((measurableSet_vecBox 1).prod (measurableSet_matBox 1))]
        change ∫⁻ y in vecBox 1 ×ˢ matBox 1, _ ∂(volume.prod volume) = _
        rw [← Measure.prod_restrict]
        exact lintegral_prod _ (hF_meas.comp (measurable_const.prodMk measurable_id)).aemeasurable
      · rw [Set.indicator_of_notMem hp]
        have h0 : ∀ y, S.indicator F ((t, u', v', B'), y) = 0 := fun y => by
          apply Set.indicator_of_notMem
          intro hmem
          exact hp ⟨hmem.2.1, hmem.2.2.1, hmem.2.2.2.1⟩
        simp_rw [h0, lintegral_zero]
    simp_rw [hinner_eq]
    rw [lintegral_indicator hSt_meas]
    have hm_inner : Measurable fun p : (Fin k → ℝ) × (Fin m → ℝ) × RMat m k =>
        ∫⁻ a in vecBox (d := n) 1, ∫⁻ A₂ in matBox (a := k) (b := n) 1,
          F ((t, p), (a, A₂)) := by
      have hmA : Measurable fun q : ((Fin k → ℝ) × (Fin m → ℝ) × RMat m k) × (Fin n → ℝ) =>
          ∫⁻ A₂ in matBox (a := k) (b := n) 1, F ((t, q.1), (q.2, A₂)) := by
        have hm0 : Measurable fun z : (((Fin k → ℝ) × (Fin m → ℝ) × RMat m k) × (Fin n → ℝ)) × RMat k n =>
            F ((t, z.1.1), (z.1.2, z.2)) :=
          hF_meas.comp ((measurable_const.prodMk (measurable_fst.comp measurable_fst)).prodMk
            ((measurable_snd.comp measurable_fst).prodMk measurable_snd))
        exact hm0.lintegral_prod_right'
      exact hmA.lintegral_prod_right'
    change ∫⁻ p in vecBox t ×ˢ (vecBox t ×ˢ matBox t), _ ∂(volume.prod volume) = _
    rw [← Measure.prod_restrict, lintegral_prod _ hm_inner.aemeasurable]
    refine setLIntegral_congr_fun (measurableSet_vecBox t) fun u' _ => ?_
    change ∫⁻ p2 in vecBox t ×ˢ matBox t, _ ∂(volume.prod volume) = _
    rw [← Measure.prod_restrict]
    exact lintegral_prod _ (hm_inner.comp (measurable_const.prodMk measurable_id)).aemeasurable
  · rw [Set.indicator_of_notMem ht]
    have h0 : ∀ x2, ∫⁻ y : (Fin n → ℝ) × RMat k n, S.indicator F ((t, x2), y) = 0 := fun x2 => by
      have : ∀ y, S.indicator F ((t, x2), y) = 0 := fun y =>
        Set.indicator_of_notMem (f := F) (fun (hmem : ((t, x2), y) ∈ S) => ht hmem.1)
      simp_rw [this, lintegral_zero]
    simp_rw [h0, lintegral_zero]

/-- Generic 3-block rescaling by `t > 0` pulling out `t^((m+1)(k+1) - 1)`. -/
private theorem lintegral_blowup_scale {m k : ℕ} {t : ℝ} (ht0 : 0 < t)
    (G : (Fin k → ℝ) → (Fin m → ℝ) → RMat m k → ℝ≥0∞)
    (hG : Measurable fun p : ((Fin k → ℝ) × (Fin m → ℝ)) × RMat m k => G p.1.1 p.1.2 p.2) :
    ∫⁻ u' in vecBox (d := k) t, ∫⁻ v' in vecBox (d := m) t, ∫⁻ B' in matBox (a := m) (b := k) t,
      G u' v' B' =
    ENNReal.ofReal (t ^ (((m + 1) * (k + 1) - 1 : ℕ) : ℝ)) *
      ∫⁻ u in vecBox (d := k) 1, ∫⁻ v in vecBox (d := m) 1, ∫⁻ Btilde in matBox (a := m) (b := k) 1,
        G (t • u) (t • v) (t • Btilde) := by
  have hm_v : Measurable fun p : (Fin k → ℝ) × (Fin m → ℝ) =>
      ∫⁻ B' in matBox (a := m) (b := k) t, G p.1 p.2 B' :=
    hG.lintegral_prod_right'
  have hm_u : Measurable fun u' : Fin k → ℝ =>
      ∫⁻ v' in vecBox (d := m) t, ∫⁻ B' in matBox (a := m) (b := k) t, G u' v' B' :=
    hm_v.lintegral_prod_right'
  rw [lintegral_vecBox_smul ht0 _ hm_u]
  have hstep_v : ∀ u : Fin k → ℝ,
      ∫⁻ v' in vecBox (d := m) t, ∫⁻ B' in matBox (a := m) (b := k) t, G (t • u) v' B' =
      ENNReal.ofReal t ^ m * ENNReal.ofReal t ^ (m * k) *
        ∫⁻ v in vecBox (d := m) 1, ∫⁻ Btilde in matBox (a := m) (b := k) 1,
          G (t • u) (t • v) (t • Btilde) := fun u => by
    have hm_v_u : Measurable fun v' : Fin m → ℝ =>
        ∫⁻ B' in matBox (a := m) (b := k) t, G (t • u) v' B' :=
      hm_v.comp (measurable_const.prodMk measurable_id)
    rw [lintegral_vecBox_smul ht0 _ hm_v_u]
    have hB_scale : ∀ v : Fin m → ℝ,
        ∫⁻ B' in matBox (a := m) (b := k) t, G (t • u) (t • v) B' =
        ENNReal.ofReal t ^ (m * k) *
          ∫⁻ Btilde in matBox (a := m) (b := k) 1, G (t • u) (t • v) (t • Btilde) := fun v =>
      lintegral_matBox_smul ht0 _ (hG.comp ((measurable_const (a := (t • u, t • v))).prodMk measurable_id))
    simp_rw [hB_scale]
    rw [lintegral_const_mul' _ _ (ENNReal.pow_ne_top ENNReal.ofReal_ne_top), ← mul_assoc]
  simp_rw [hstep_v]
  rw [lintegral_const_mul' _ _ (ENNReal.mul_ne_top (ENNReal.pow_ne_top ENNReal.ofReal_ne_top)
    (ENNReal.pow_ne_top ENNReal.ofReal_ne_top)), ← mul_assoc]
  have hpow : ENNReal.ofReal t ^ k * (ENNReal.ofReal t ^ m * ENNReal.ofReal t ^ (m * k)) =
      ENNReal.ofReal (t ^ (((m + 1) * (k + 1) - 1 : ℕ) : ℝ)) := by
    rw [← pow_add, ← pow_add, ← ENNReal.ofReal_pow ht0.le, Real.rpow_natCast]
    congr 2
    have hmul : (m + 1) * (k + 1) = m * k + m + k + 1 := by ring
    omega
  rw [hpow]

/-- Exact representation of `chartPosZ` after blowing up `B` by `t = B 0 0 ∈ (0, 1]`. -/
theorem chartPosZ_eq_blowup (m k n : ℕ) (N : ℝ) :
    chartPosZ m k n N =
    ∫⁻ t in Set.Ioc (0 : ℝ) 1,
      ENNReal.ofReal (t ^ (((m + 1) * (k + 1) - 1 : ℕ) : ℝ)) *
        ∫⁻ u in vecBox (d := k) 1,
        ∫⁻ v in vecBox (d := m) 1,
        ∫⁻ Btilde in matBox (a := m) (b := k) 1,
        ∫⁻ a in vecBox (d := n) 1,
        ∫⁻ A₂ in matBox (a := k) (b := n) 1,
          ENNReal.ofReal (Real.exp (-(N * rank0Loss (mkB t (t • u) (t • v) (t • Btilde), mkA a A₂)))) := by
  rw [chartPosZ_eq_iterated]
  refine setLIntegral_congr_fun measurableSet_Ioc fun t ht => ?_
  have hmkB : Measurable fun p : ((Fin k → ℝ) × (Fin m → ℝ)) × RMat m k =>
      mkB t p.1.1 p.1.2 p.2 :=
    (splitB m k).symm.measurable.comp
      (measurable_const.prodMk ((measurable_fst.comp measurable_fst).prodMk
        ((measurable_snd.comp measurable_fst).prodMk measurable_snd)))
  have hmkA : Measurable fun p : (Fin n → ℝ) × RMat k n => mkA p.1 p.2 :=
    (splitA k n).symm.measurable
  have hm0 : Measurable fun w : ((((Fin k → ℝ) × (Fin m → ℝ)) × RMat m k) × (Fin n → ℝ)) × RMat k n =>
      ENNReal.ofReal (Real.exp (-(N * rank0Loss (mkB t w.1.1.1.1 w.1.1.1.2 w.1.1.2, mkA w.1.2 w.2)))) := by
    have hpair : Measurable fun w : ((((Fin k → ℝ) × (Fin m → ℝ)) × RMat m k) × (Fin n → ℝ)) × RMat k n =>
        (mkB t w.1.1.1.1 w.1.1.1.2 w.1.1.2, mkA w.1.2 w.2) :=
      (hmkB.comp (measurable_fst.comp measurable_fst)).prodMk
        (hmkA.comp ((measurable_snd.comp measurable_fst).prodMk measurable_snd))
    have hloss := (measurable_rank0Loss (m := m + 1) (k' := k + 1) (n' := n)).comp hpair
    exact ENNReal.continuous_ofReal.measurable.comp
      (Real.continuous_exp.measurable.comp (measurable_const.mul hloss).neg)
  have hG : Measurable fun p : ((Fin k → ℝ) × (Fin m → ℝ)) × RMat m k =>
      ∫⁻ a in vecBox (d := n) 1, ∫⁻ A₂ in matBox (a := k) (b := n) 1,
        ENNReal.ofReal (Real.exp (-(N * rank0Loss (mkB t p.1.1 p.1.2 p.2, mkA a A₂)))) :=
    hm0.lintegral_prod_right'.lintegral_prod_right'
  exact lintegral_blowup_scale ht.1
    (fun u' v' B' => ∫⁻ a in vecBox (d := n) 1, ∫⁻ A₂ in matBox (a := k) (b := n) 1,
      ENNReal.ofReal (Real.exp (-(N * rank0Loss (mkB t u' v' B', mkA a A₂))))) hG

/-- Upper bound for `chartPosZ` in terms of `SLT.NC.stepG (gaussBoxZ n * rank0Z m k n)`. -/
theorem chartPosZ_le_stepG (m k n : ℕ) {N : ℝ} (hN : 0 ≤ N) :
    let c₁ : ℝ := 1 / (2 * (m : ℝ) + 2)
    let Cup : ℝ := 2 ^ (k + m + m * k + k * n) * ((k + 1 : ℕ) : ℝ) ^ n
    chartPosZ m k n N ≤
      ENNReal.ofReal Cup *
        SLT.NC.stepG (fun M => gaussBoxZ n M * rank0Z m k n M)
          (((m + 1) * (k + 1) - 1 : ℕ) : ℝ) 2 (c₁ * N) := by
  intro c₁ Cup
  rw [chartPosZ_eq_blowup]
  unfold SLT.NC.stepG
  rw [← lintegral_const_mul' _ _ ENNReal.ofReal_ne_top]
  refine setLIntegral_mono' measurableSet_Ioc fun t ht => ?_
  have ht0 : 0 < t := ht.1
  rw [← mul_assoc, mul_comm (ENNReal.ofReal Cup), mul_assoc]
  refine mul_le_mul' le_rfl ?_
  -- For fixed `t ∈ Ioc 0 1`, bound the `(u, v, Btilde, a, A₂)` integral
  have hslice : ∀ (u : Fin k → ℝ), u ∈ vecBox 1 → ∀ (v : Fin m → ℝ), v ∈ vecBox 1 →
      ∫⁻ Btilde in matBox (a := m) (b := k) 1,
      ∫⁻ a in vecBox (d := n) 1,
      ∫⁻ A₂ in matBox (a := k) (b := n) 1,
        ENNReal.ofReal (Real.exp (-(N * rank0Loss (mkB t (t • u) (t • v) (t • Btilde), mkA a A₂)))) ≤
      ENNReal.ofReal (((k + 1 : ℕ) : ℝ) ^ n * 2 ^ (m * k + k * n)) *
        (gaussBoxZ n (c₁ * N * t ^ 2) * rank0Z m k n (c₁ * N * t ^ 2)) := fun u hu v hv => by
    -- Pointwise lower loss bound `rank0Loss ≥ c₁ * t² * (‖â‖₂² + rank0Loss (B̂, A₂))`
    have hpt : ∀ (Btilde : RMat m k) (a : Fin n → ℝ) (A₂ : RMat k n),
        ENNReal.ofReal (Real.exp (-(N * rank0Loss (mkB t (t • u) (t • v) (t • Btilde), mkA a A₂)))) ≤
        ENNReal.ofReal (Real.exp (-(c₁ * N * t ^ 2 *
          ∑ j : Fin n, (a j + ∑ r : Fin k, u r * A₂ r j) ^ 2))) *
        ENNReal.ofReal (Real.exp (-(c₁ * N * t ^ 2 *
          rank0Loss (fun i r => Btilde i r - v i * u r, A₂)))) := fun Btilde a A₂ => by
      rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add]
      apply ENNReal.ofReal_le_ofReal
      apply Real.exp_le_exp.mpr
      have hlo := (rank0Loss_blowup_bounds t u hv Btilde a A₂).1
      have h1 : N * (c₁ * (t ^ 2 * ((∑ j : Fin n, (a j + ∑ r : Fin k, u r * A₂ r j) ^ 2) +
          rank0Loss (fun i r => Btilde i r - v i * u r, A₂)))) ≤
          N * rank0Loss (mkB t (t • u) (t • v) (t • Btilde), mkA a A₂) :=
        mul_le_mul_of_nonneg_left hlo hN
      linarith
    -- Apply `hpt` and swap `a` and `A₂` by Fubini, then translate `a ↦ a + uᵀ A₂`
    have ha_bound : ∀ A₂ : RMat k n, A₂ ∈ matBox 1 →
        ∫⁻ a in vecBox (d := n) 1,
          ENNReal.ofReal (Real.exp (-(c₁ * N * t ^ 2 *
            ∑ j : Fin n, (a j + ∑ r : Fin k, u r * A₂ r j) ^ 2))) ≤
        ENNReal.ofReal (((k + 1 : ℕ) : ℝ) ^ n) * gaussBoxZ n (c₁ * N * t ^ 2) := fun A₂ hA₂ => by
      let shift : Fin n → ℝ := fun j => ∑ r : Fin k, u r * A₂ r j
      have htr := measurePreserving_add_right (volume : Measure (Fin n → ℝ)) shift
      have hsub : ∀ a ∈ vecBox (d := n) 1, a + shift ∈ vecBox (d := n) ((k + 1 : ℕ) : ℝ) := fun a ha => by
        rw [mem_vecBox_iff] at ha hu ⊢
        rw [mem_matBox_iff] at hA₂
        intro j
        have hsum_le : |∑ r : Fin k, u r * A₂ r j| ≤ (k : ℝ) := by
          calc |∑ r : Fin k, u r * A₂ r j|
              ≤ ∑ r : Fin k, |u r * A₂ r j| := Finset.abs_sum_le_sum_abs _ _
            _ ≤ ∑ _ : Fin k, (1 : ℝ) := by
                refine Finset.sum_le_sum fun r _ => ?_
                rw [abs_mul]
                have := mul_le_mul (hu r) (hA₂ r j) (abs_nonneg _) zero_le_one
                linarith
            _ = (k : ℝ) := by simp
        calc |a j + shift j| ≤ |a j| + |shift j| := abs_add_le _ _
          _ ≤ 1 + (k : ℝ) := add_le_add (ha j) hsum_le
          _ = ((k + 1 : ℕ) : ℝ) := by push_cast; ring
      let g : (Fin n → ℝ) → ℝ≥0∞ := fun x =>
        ENNReal.ofReal (Real.exp (-(c₁ * N * t ^ 2 * ∑ j : Fin n, (x j) ^ 2)))
      have hg : Measurable g := by unfold g; fun_prop
      have hk1 : (0 : ℝ) < ((k + 1 : ℕ) : ℝ) := by positivity
      calc ∫⁻ a in vecBox (d := n) 1, g (a + shift)
          = ∫⁻ a, (vecBox (d := n) 1).indicator (fun x => g (x + shift)) a :=
            (lintegral_indicator (measurableSet_vecBox 1) _).symm
        _ ≤ ∫⁻ a, (vecBox (d := n) ((k + 1 : ℕ) : ℝ)).indicator g (a + shift) := by
            refine lintegral_mono fun a => ?_
            by_cases ha : a ∈ vecBox 1
            · rw [Set.indicator_of_mem ha, Set.indicator_of_mem (hsub a ha)]
            · rw [Set.indicator_of_notMem ha]; exact bot_le
        _ = ∫⁻ x in vecBox (d := n) ((k + 1 : ℕ) : ℝ), g x := by
            rw [← lintegral_map (hg.indicator (measurableSet_vecBox _)) htr.measurable,
              htr.map_eq, lintegral_indicator (measurableSet_vecBox _)]
        _ = ENNReal.ofReal (((k + 1 : ℕ) : ℝ)) ^ n *
              gaussBoxZ n ((((k + 1 : ℕ) : ℝ) ^ 2) * (c₁ * N * t ^ 2)) :=
            gaussBoxZ_scale hk1 (c₁ * N * t ^ 2)
        _ ≤ ENNReal.ofReal (((k + 1 : ℕ) : ℝ) ^ n) * gaussBoxZ n (c₁ * N * t ^ 2) := by
            rw [ENNReal.ofReal_pow hk1.le]
            refine mul_le_mul' le_rfl (gaussBoxZ_mono n ?_)
            have hge1 : (1 : ℝ) ≤ ((k + 1 : ℕ) : ℝ) ^ 2 := by
              have : (1 : ℝ) ≤ ((k + 1 : ℕ) : ℝ) := by exact_mod_cast (show 1 ≤ k + 1 by omega)
              nlinarith
            have hnn : 0 ≤ c₁ * N * t ^ 2 := by positivity
            nlinarith
    -- Now integrate over `Btilde`, `a`, `A₂`
    have hB_bound : ∀ A₂ : RMat k n,
        ∫⁻ Btilde in matBox (a := m) (b := k) 1,
          ENNReal.ofReal (Real.exp (-(c₁ * N * t ^ 2 *
            rank0Loss (fun i r => Btilde i r - v i * u r, A₂)))) ≤
        ∫⁻ Bhat in matBox (a := m) (b := k) 2,
          ENNReal.ofReal (Real.exp (-(c₁ * N * t ^ 2 * rank0Loss (Bhat, A₂)))) := fun A₂ => by
      let vu : RMat m k := fun i r => v i * u r
      have htr := measurePreserving_sub_right (volume : Measure (RMat m k)) vu
      have hsub : ∀ Btilde ∈ matBox (a := m) (b := k) 1, Btilde - vu ∈ matBox (a := m) (b := k) 2 := fun Btilde hB => by
        rw [mem_matBox_iff] at hB ⊢
        rw [mem_vecBox_iff] at hu hv
        intro i r
        have hvu : |v i * u r| ≤ 1 := by
          rw [abs_mul]
          have := mul_le_mul (hv i) (hu r) (abs_nonneg _) zero_le_one
          linarith
        calc |Btilde i r - v i * u r| ≤ |Btilde i r| + |v i * u r| := abs_sub _ _
          _ ≤ 1 + 1 := add_le_add (hB i r) hvu
          _ = 2 := by norm_num
      let h : RMat m k → ℝ≥0∞ := fun Bhat =>
        ENNReal.ofReal (Real.exp (-(c₁ * N * t ^ 2 * rank0Loss (Bhat, A₂))))
      have hh : Measurable h := by
        unfold h
        have := (measurable_rank0Loss (m := m) (k' := k) (n' := n)).comp
          (measurable_id.prodMk (measurable_const (a := A₂)))
        fun_prop
      calc ∫⁻ Btilde in matBox (a := m) (b := k) 1, h (Btilde - vu)
          = ∫⁻ Btilde, (matBox (a := m) (b := k) 1).indicator (fun X => h (X - vu)) Btilde :=
            (lintegral_indicator (measurableSet_matBox 1) _).symm
        _ ≤ ∫⁻ Btilde, (matBox (a := m) (b := k) 2).indicator h (Btilde - vu) := by
            refine lintegral_mono fun X => ?_
            by_cases hX : X ∈ matBox 1
            · rw [Set.indicator_of_mem hX, Set.indicator_of_mem (hsub X hX)]
            · rw [Set.indicator_of_notMem hX]; exact bot_le
        _ = ∫⁻ Bhat in matBox (a := m) (b := k) 2, h Bhat := by
            rw [← lintegral_map (hh.indicator (measurableSet_matBox 2)) htr.measurable,
              htr.map_eq, lintegral_indicator (measurableSet_matBox 2)]
    -- Assemble via Fubini
    have hstepA : ∀ Btilde : RMat m k,
        ∫⁻ a in vecBox (d := n) 1, ∫⁻ A₂ in matBox (a := k) (b := n) 1,
          ENNReal.ofReal (Real.exp (-(N * rank0Loss (mkB t (t • u) (t • v) (t • Btilde), mkA a A₂)))) ≤
        ENNReal.ofReal (((k + 1 : ℕ) : ℝ) ^ n) * gaussBoxZ n (c₁ * N * t ^ 2) *
          ∫⁻ A₂ in matBox (a := k) (b := n) 1,
            ENNReal.ofReal (Real.exp (-(c₁ * N * t ^ 2 *
              rank0Loss (fun i r => Btilde i r - v i * u r, A₂)))) := fun Btilde => by
      let G1 : (Fin n → ℝ) × RMat k n → ℝ≥0∞ := fun p =>
        ENNReal.ofReal (Real.exp (-(c₁ * N * t ^ 2 * ∑ j : Fin n, (p.1 j + ∑ r : Fin k, u r * p.2 r j) ^ 2))) *
        ENNReal.ofReal (Real.exp (-(c₁ * N * t ^ 2 * rank0Loss (fun i r => Btilde i r - v i * u r, p.2))))
      have hG1_m : Measurable G1 := by
        unfold G1
        have hmr : Measurable fun p : (Fin n → ℝ) × RMat k n =>
            rank0Loss (fun i r => Btilde i r - v i * u r, p.2) :=
          (measurable_rank0Loss (m := m) (k' := k) (n' := n)).comp
            (measurable_const.prodMk measurable_snd)
        fun_prop
      calc ∫⁻ a in vecBox (d := n) 1, ∫⁻ A₂ in matBox (a := k) (b := n) 1,
              ENNReal.ofReal (Real.exp (-(N * rank0Loss (mkB t (t • u) (t • v) (t • Btilde), mkA a A₂))))
          ≤ ∫⁻ a in vecBox (d := n) 1, ∫⁻ A₂ in matBox (a := k) (b := n) 1, G1 (a, A₂) :=
            setLIntegral_mono' (measurableSet_vecBox 1) fun a _ =>
              setLIntegral_mono' (measurableSet_matBox 1) fun A₂ _ => hpt Btilde a A₂
        _ = ∫⁻ A₂ in matBox (a := k) (b := n) 1, ∫⁻ a in vecBox (d := n) 1, G1 (a, A₂) :=
            lintegral_lintegral_swap hG1_m.aemeasurable
        _ ≤ ∫⁻ A₂ in matBox (a := k) (b := n) 1,
              (ENNReal.ofReal (((k + 1 : ℕ) : ℝ) ^ n) * gaussBoxZ n (c₁ * N * t ^ 2)) *
              ENNReal.ofReal (Real.exp (-(c₁ * N * t ^ 2 *
                rank0Loss (fun i r => Btilde i r - v i * u r, A₂)))) := by
            refine setLIntegral_mono' (measurableSet_matBox 1) fun A₂ hA₂ => ?_
            unfold G1
            dsimp only
            rw [lintegral_mul_const' _ _ ENNReal.ofReal_ne_top]
            exact mul_le_mul' (ha_bound A₂ hA₂) le_rfl
        _ = ENNReal.ofReal (((k + 1 : ℕ) : ℝ) ^ n) * gaussBoxZ n (c₁ * N * t ^ 2) *
              ∫⁻ A₂ in matBox (a := k) (b := n) 1,
                ENNReal.ofReal (Real.exp (-(c₁ * N * t ^ 2 *
                  rank0Loss (fun i r => Btilde i r - v i * u r, A₂)))) := by
            have hm_A2 : Measurable fun A₂ : RMat k n =>
                ENNReal.ofReal (Real.exp (-(c₁ * N * t ^ 2 *
                  rank0Loss (fun i r => Btilde i r - v i * u r, A₂)))) := by
              have hmr : Measurable fun A₂ : RMat k n =>
                  rank0Loss (fun i r => Btilde i r - v i * u r, A₂) :=
                (measurable_rank0Loss (m := m) (k' := k) (n' := n)).comp
                  (measurable_const.prodMk measurable_id)
              fun_prop
            exact lintegral_const_mul _ hm_A2
    have hBA_meas : Measurable fun p : RMat m k × RMat k n =>
        ENNReal.ofReal (Real.exp (-(c₁ * N * t ^ 2 *
          rank0Loss (fun i r => p.1 i r - v i * u r, p.2)))) := by
      have hm_sub : Measurable fun Btilde : RMat m k => (fun i r => Btilde i r - v i * u r : RMat m k) := by
        fun_prop
      have := (measurable_rank0Loss (m := m) (k' := k) (n' := n)).comp
        ((hm_sub.comp measurable_fst).prodMk measurable_snd)
      fun_prop
    have hBA_m2 : Measurable fun p : RMat m k × RMat k n =>
        ENNReal.ofReal (Real.exp (-(c₁ * N * t ^ 2 * rank0Loss p))) := by
      have := measurable_rank0Loss (m := m) (k' := k) (n' := n); fun_prop
    have hsubA2 : matBox (a := k) (b := n) 1 ⊆ matBox (a := k) (b := n) 2 := fun A₂ hA₂ => by
      rw [mem_matBox_iff] at hA₂ ⊢; intro r j; linarith [hA₂ r j]
    calc ∫⁻ Btilde in matBox (a := m) (b := k) 1,
          ∫⁻ a in vecBox (d := n) 1, ∫⁻ A₂ in matBox (a := k) (b := n) 1,
            ENNReal.ofReal (Real.exp (-(N * rank0Loss (mkB t (t • u) (t • v) (t • Btilde), mkA a A₂))))
        ≤ ∫⁻ Btilde in matBox (a := m) (b := k) 1,
            (ENNReal.ofReal (((k + 1 : ℕ) : ℝ) ^ n) * gaussBoxZ n (c₁ * N * t ^ 2)) *
              ∫⁻ A₂ in matBox (a := k) (b := n) 1,
                ENNReal.ofReal (Real.exp (-(c₁ * N * t ^ 2 *
                  rank0Loss (fun i r => Btilde i r - v i * u r, A₂)))) :=
          setLIntegral_mono' (measurableSet_matBox 1) fun Btilde _ => hstepA Btilde
      _ = (ENNReal.ofReal (((k + 1 : ℕ) : ℝ) ^ n) * gaussBoxZ n (c₁ * N * t ^ 2)) *
            ∫⁻ Btilde in matBox (a := m) (b := k) 1, ∫⁻ A₂ in matBox (a := k) (b := n) 1,
              ENNReal.ofReal (Real.exp (-(c₁ * N * t ^ 2 *
                rank0Loss (fun i r => Btilde i r - v i * u r, A₂)))) :=
          lintegral_const_mul _ hBA_meas.lintegral_prod_right'
      _ = (ENNReal.ofReal (((k + 1 : ℕ) : ℝ) ^ n) * gaussBoxZ n (c₁ * N * t ^ 2)) *
            ∫⁻ A₂ in matBox (a := k) (b := n) 1, ∫⁻ Btilde in matBox (a := m) (b := k) 1,
              ENNReal.ofReal (Real.exp (-(c₁ * N * t ^ 2 *
                rank0Loss (fun i r => Btilde i r - v i * u r, A₂)))) := by
          rw [lintegral_lintegral_swap hBA_meas.aemeasurable]
      _ ≤ (ENNReal.ofReal (((k + 1 : ℕ) : ℝ) ^ n) * gaussBoxZ n (c₁ * N * t ^ 2)) *
            ∫⁻ A₂ in matBox (a := k) (b := n) 2, ∫⁻ Bhat in matBox (a := m) (b := k) 2,
              ENNReal.ofReal (Real.exp (-(c₁ * N * t ^ 2 * rank0Loss (Bhat, A₂)))) := by
          refine mul_le_mul' le_rfl ?_
          refine (setLIntegral_mono' (measurableSet_matBox 1) fun A₂ _ => hB_bound A₂).trans
            (lintegral_mono_set hsubA2)
      _ = (ENNReal.ofReal (((k + 1 : ℕ) : ℝ) ^ n) * gaussBoxZ n (c₁ * N * t ^ 2)) *
            (ENNReal.ofReal 2 ^ (m * k + k * n) * rank0Z m k n (2 ^ 4 * (c₁ * N * t ^ 2))) := by
          congr 1
          rw [← lintegral_lintegral_swap hBA_m2.aemeasurable]
          have hprod :
              ∫⁻ Bhat in matBox (a := m) (b := k) 2, ∫⁻ A₂ in matBox (a := k) (b := n) 2,
                ENNReal.ofReal (Real.exp (-(c₁ * N * t ^ 2 * rank0Loss (Bhat, A₂)))) =
              SLT.Z (volume.restrict (matBox (a := m) (b := k) 2 ×ˢ matBox (a := k) (b := n) 2))
                rank0Loss (c₁ * N * t ^ 2) := by
            unfold SLT.Z
            change _ = ∫⁻ p in matBox 2 ×ˢ matBox 2, _ ∂(volume.prod volume)
            rw [← Measure.prod_restrict, lintegral_prod _ hBA_m2.aemeasurable]
          rw [hprod, rank0_Z_scale (by norm_num)]
          rfl
      _ ≤ ENNReal.ofReal (((k + 1 : ℕ) : ℝ) ^ n * 2 ^ (m * k + k * n)) *
            (gaussBoxZ n (c₁ * N * t ^ 2) * rank0Z m k n (c₁ * N * t ^ 2)) := by
          have hnn : 0 ≤ c₁ * N * t ^ 2 := by positivity
          have hmono_Z : rank0Z m k n (2 ^ 4 * (c₁ * N * t ^ 2)) ≤ rank0Z m k n (c₁ * N * t ^ 2) :=
            rank0Z_mono m k n (by linarith)
          rw [ENNReal.ofReal_mul (by positivity), ENNReal.ofReal_pow (p := (2 : ℝ)) (by norm_num)]
          calc (ENNReal.ofReal (((k + 1 : ℕ) : ℝ) ^ n) * gaussBoxZ n (c₁ * N * t ^ 2)) *
                  (ENNReal.ofReal 2 ^ (m * k + k * n) * rank0Z m k n (2 ^ 4 * (c₁ * N * t ^ 2)))
              = (ENNReal.ofReal (((k + 1 : ℕ) : ℝ) ^ n) * ENNReal.ofReal 2 ^ (m * k + k * n)) *
                  (gaussBoxZ n (c₁ * N * t ^ 2) * rank0Z m k n (2 ^ 4 * (c₁ * N * t ^ 2))) := by
                ring
            _ ≤ _ := mul_le_mul' le_rfl (mul_le_mul' le_rfl hmono_Z)
  -- Integrate the constant bound over `u ∈ [-1,1]^k` and `v ∈ [-1,1]^m`
  calc ∫⁻ u in vecBox (d := k) 1, ∫⁻ v in vecBox (d := m) 1,
        ∫⁻ Btilde in matBox (a := m) (b := k) 1,
        ∫⁻ a in vecBox (d := n) 1, ∫⁻ A₂ in matBox (a := k) (b := n) 1,
          ENNReal.ofReal (Real.exp (-(N * rank0Loss (mkB t (t • u) (t • v) (t • Btilde), mkA a A₂))))
      ≤ ∫⁻ _ in vecBox (d := k) 1, ∫⁻ _ in vecBox (d := m) 1,
          ENNReal.ofReal (((k + 1 : ℕ) : ℝ) ^ n * 2 ^ (m * k + k * n)) *
            (gaussBoxZ n (c₁ * N * t ^ 2) * rank0Z m k n (c₁ * N * t ^ 2)) :=
        setLIntegral_mono' (measurableSet_vecBox 1) fun u hu =>
          setLIntegral_mono' (measurableSet_vecBox 1) fun v hv => hslice u hu v hv
    _ = ENNReal.ofReal Cup * (gaussBoxZ n (c₁ * N * t ^ 2) * rank0Z m k n (c₁ * N * t ^ 2)) := by
        simp only [setLIntegral_const, volume_vecBox zero_le_one, mul_one]
        have hC : ENNReal.ofReal (((k + 1 : ℕ) : ℝ) ^ n * 2 ^ (m * k + k * n)) *
            ENNReal.ofReal 2 ^ m * ENNReal.ofReal 2 ^ k = ENNReal.ofReal Cup := by
          dsimp [Cup]
          rw [← ENNReal.ofReal_pow (by norm_num), ← ENNReal.ofReal_pow (by norm_num),
            ← ENNReal.ofReal_mul (by positivity), ← ENNReal.ofReal_mul (by positivity)]
          congr 1
          rw [pow_add, pow_add, pow_add]
          ring
        rw [← hC]; ring

/-- Lower bound for `chartPosZ` in terms of `SLT.NC.stepG (gaussBoxZ n * rank0Z m k n)`. -/
theorem stepG_le_chartPosZ (m k n : ℕ) {N : ℝ} (hN : 0 ≤ N) :
    let c₂ : ℝ := 2 * (m : ℝ) + 2
    let r : ℝ := 1 / (2 * ((k + 1 : ℕ) : ℝ))
    let clo : ℝ := (1 / 2 : ℝ) ^ n * r ^ (m * k + k * n)
    ENNReal.ofReal clo *
        SLT.NC.stepG (fun M => gaussBoxZ n M * rank0Z m k n M)
          (((m + 1) * (k + 1) - 1 : ℕ) : ℝ) 2 (c₂ * N) ≤
      chartPosZ m k n N := by
  intro c₂ r clo
  have hr_pos : 0 < r := by dsimp [r]; positivity
  have hr_le_half : r ≤ 1 / 2 := by
    dsimp [r]
    have hk1 : (1 : ℝ) ≤ ((k + 1 : ℕ) : ℝ) := by exact_mod_cast (show 1 ≤ k + 1 by omega)
    rw [div_le_div_iff₀ (by positivity) (by norm_num)]
    linarith
  have hr_le_one : r ≤ 1 := by linarith
  rw [chartPosZ_eq_blowup]
  unfold SLT.NC.stepG
  rw [← lintegral_const_mul' _ _ ENNReal.ofReal_ne_top]
  refine setLIntegral_mono' measurableSet_Ioc fun t ht => ?_
  have ht0 : 0 < t := ht.1
  rw [← mul_assoc, mul_comm (ENNReal.ofReal clo), mul_assoc]
  refine mul_le_mul' le_rfl ?_
  -- Restrict `u ∈ [-1/2, 1/2]^k`, `v ∈ [-1/2, 1/2]^m` (each has volume 1!)
  have hsub_u : vecBox (d := k) (1 / 2) ⊆ vecBox (d := k) 1 := fun u hu => by
    rw [mem_vecBox_iff] at hu ⊢; intro r'; linarith [hu r']
  have hsub_v : vecBox (d := m) (1 / 2) ⊆ vecBox (d := m) 1 := fun v hv => by
    rw [mem_vecBox_iff] at hv ⊢; intro i; linarith [hv i]
  have hslice : ∀ (u : Fin k → ℝ), u ∈ vecBox (1 / 2) → ∀ (v : Fin m → ℝ), v ∈ vecBox (1 / 2) →
      ENNReal.ofReal clo * (gaussBoxZ n (c₂ * N * t ^ 2) * rank0Z m k n (c₂ * N * t ^ 2)) ≤
      ∫⁻ Btilde in matBox (a := m) (b := k) 1,
      ∫⁻ a in vecBox (d := n) 1,
      ∫⁻ A₂ in matBox (a := k) (b := n) 1,
        ENNReal.ofReal (Real.exp (-(N * rank0Loss (mkB t (t • u) (t • v) (t • Btilde), mkA a A₂)))) :=
    fun u hu v hv => by
    have hv1 : v ∈ vecBox 1 := hsub_v hv
    -- Pointwise upper loss bound `rank0Loss ≤ c₂ * t² * (‖â‖₂² + rank0Loss (B̂, A₂))`
    have hpt : ∀ (Btilde : RMat m k) (a : Fin n → ℝ) (A₂ : RMat k n),
        ENNReal.ofReal (Real.exp (-(c₂ * N * t ^ 2 *
          ∑ j : Fin n, (a j + ∑ r' : Fin k, u r' * A₂ r' j) ^ 2))) *
        ENNReal.ofReal (Real.exp (-(c₂ * N * t ^ 2 *
          rank0Loss (fun i r' => Btilde i r' - v i * u r', A₂)))) ≤
        ENNReal.ofReal (Real.exp (-(N * rank0Loss (mkB t (t • u) (t • v) (t • Btilde), mkA a A₂)))) :=
      fun Btilde a A₂ => by
      rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add]
      apply ENNReal.ofReal_le_ofReal
      apply Real.exp_le_exp.mpr
      have hup := (rank0Loss_blowup_bounds t u hv1 Btilde a A₂).2
      have h1 : N * rank0Loss (mkB t (t • u) (t • v) (t • Btilde), mkA a A₂) ≤
          N * (c₂ * (t ^ 2 * ((∑ j : Fin n, (a j + ∑ r' : Fin k, u r' * A₂ r' j) ^ 2) +
            rank0Loss (fun i r' => Btilde i r' - v i * u r', A₂)))) :=
        mul_le_mul_of_nonneg_left hup hN
      linarith
    -- For `A₂ ∈ matBox r`, shifting `a ∈ vecBox 1` covers `ahat ∈ vecBox (1/2)`!
    have ha_lower : ∀ A₂ : RMat k n, A₂ ∈ matBox r →
        ENNReal.ofReal ((1 / 2 : ℝ) ^ n) * gaussBoxZ n (c₂ * N * t ^ 2) ≤
        ∫⁻ a in vecBox (d := n) 1,
          ENNReal.ofReal (Real.exp (-(c₂ * N * t ^ 2 *
            ∑ j : Fin n, (a j + ∑ r' : Fin k, u r' * A₂ r' j) ^ 2))) := fun A₂ hA₂ => by
      let shift : Fin n → ℝ := fun j => ∑ r' : Fin k, u r' * A₂ r' j
      have htr := measurePreserving_sub_right (volume : Measure (Fin n → ℝ)) shift
      have hsub : ∀ ahat ∈ vecBox (d := n) (1 / 2), ahat - shift ∈ vecBox (d := n) 1 := fun ahat ha => by
        rw [mem_vecBox_iff] at ha hu ⊢
        rw [mem_matBox_iff] at hA₂
        intro j
        have hsum_le : |shift j| ≤ 1 / 2 := by
          dsimp [shift]
          calc |∑ r' : Fin k, u r' * A₂ r' j|
              ≤ ∑ r' : Fin k, |u r' * A₂ r' j| := Finset.abs_sum_le_sum_abs _ _
            _ ≤ ∑ _ : Fin k, (1 / 2) * r := by
                refine Finset.sum_le_sum fun r' _ => ?_
                rw [abs_mul]
                exact mul_le_mul (hu r') (hA₂ r' j) (abs_nonneg _) (by norm_num)
            _ = (k : ℝ) * ((1 / 2) * r) := by simp
            _ ≤ 1 / 2 := by
                dsimp [r]
                have hk_le : (k : ℝ) ≤ ((k + 1 : ℕ) : ℝ) := by exact_mod_cast (show k ≤ k + 1 by omega)
                have hk_pos : (0 : ℝ) < ((k + 1 : ℕ) : ℝ) := by positivity
                rw [show (k : ℝ) * (1 / 2 * (1 / (2 * ((k + 1 : ℕ) : ℝ)))) =
                    (k : ℝ) / (4 * ((k + 1 : ℕ) : ℝ)) by ring,
                  div_le_div_iff₀ (by positivity) (by norm_num)]
                linarith
        calc |ahat j - shift j| ≤ |ahat j| + |shift j| := abs_sub _ _
          _ ≤ 1 / 2 + 1 / 2 := add_le_add (ha j) hsum_le
          _ = 1 := by norm_num
      let g : (Fin n → ℝ) → ℝ≥0∞ := fun x =>
        ENNReal.ofReal (Real.exp (-(c₂ * N * t ^ 2 * ∑ j : Fin n, (x j) ^ 2)))
      have hg : Measurable g := by unfold g; fun_prop
      calc ENNReal.ofReal ((1 / 2 : ℝ) ^ n) * gaussBoxZ n (c₂ * N * t ^ 2)
          ≤ ENNReal.ofReal (1 / 2 : ℝ) ^ n * gaussBoxZ n ((1 / 2 : ℝ) ^ 2 * (c₂ * N * t ^ 2)) := by
            rw [ENNReal.ofReal_pow (by norm_num)]
            refine mul_le_mul' le_rfl (gaussBoxZ_mono n ?_)
            have : 0 ≤ c₂ * N * t ^ 2 := by positivity
            nlinarith
        _ = ∫⁻ ahat in vecBox (d := n) (1 / 2), g ahat :=
            (gaussBoxZ_scale (by norm_num) (c₂ * N * t ^ 2)).symm
        _ = ∫⁻ ahat, (vecBox (d := n) (1 / 2)).indicator g ahat :=
            (lintegral_indicator (measurableSet_vecBox (1 / 2)) _).symm
        _ ≤ ∫⁻ ahat, (vecBox (d := n) 1).indicator (fun a => g (a + shift)) (ahat - shift) := by
            refine lintegral_mono fun ahat => ?_
            by_cases ha : ahat ∈ vecBox (1 / 2)
            · rw [Set.indicator_of_mem ha, Set.indicator_of_mem (hsub ahat ha), sub_add_cancel]
            · rw [Set.indicator_of_notMem ha]; exact bot_le
        _ = ∫⁻ a in vecBox (d := n) 1, g (a + shift) := by
            have hm_gs : Measurable fun a : Fin n → ℝ => g (a + shift) :=
              hg.comp (measurable_id.add measurable_const)
            rw [← lintegral_map (hm_gs.indicator (measurableSet_vecBox 1)) htr.measurable,
              htr.map_eq, lintegral_indicator (measurableSet_vecBox 1)]
    -- Shifting `Btilde ∈ matBox 1` covers `Bhat ∈ matBox r`!
    have hB_lower : ∀ A₂ : RMat k n,
        ∫⁻ Bhat in matBox (a := m) (b := k) r,
          ENNReal.ofReal (Real.exp (-(c₂ * N * t ^ 2 * rank0Loss (Bhat, A₂)))) ≤
        ∫⁻ Btilde in matBox (a := m) (b := k) 1,
          ENNReal.ofReal (Real.exp (-(c₂ * N * t ^ 2 *
            rank0Loss (fun i r' => Btilde i r' - v i * u r', A₂)))) := fun A₂ => by
      let vu : RMat m k := fun i r' => v i * u r'
      have htr := measurePreserving_add_right (volume : Measure (RMat m k)) vu
      have hsub : ∀ Bhat ∈ matBox (a := m) (b := k) r, Bhat + vu ∈ matBox (a := m) (b := k) 1 :=
        fun Bhat hB => by
        rw [mem_matBox_iff] at hB ⊢
        rw [mem_vecBox_iff] at hu hv
        intro i r'
        have hvu : |v i * u r'| ≤ 1 / 4 := by
          rw [abs_mul]
          have := mul_le_mul (hv i) (hu r') (abs_nonneg _) (by norm_num)
          linarith
        calc |Bhat i r' + v i * u r'| ≤ |Bhat i r'| + |v i * u r'| := abs_add_le _ _
          _ ≤ r + 1 / 4 := add_le_add (hB i r') hvu
          _ ≤ 1 := by linarith
      let h : RMat m k → ℝ≥0∞ := fun Bhat =>
        ENNReal.ofReal (Real.exp (-(c₂ * N * t ^ 2 * rank0Loss (Bhat, A₂))))
      have hh : Measurable h := by
        unfold h
        have := (measurable_rank0Loss (m := m) (k' := k) (n' := n)).comp
          (measurable_id.prodMk (measurable_const (a := A₂)))
        fun_prop
      calc ∫⁻ Bhat in matBox (a := m) (b := k) r, h Bhat
          = ∫⁻ Bhat, (matBox (a := m) (b := k) r).indicator h Bhat :=
            (lintegral_indicator (measurableSet_matBox r) _).symm
        _ ≤ ∫⁻ Bhat, (matBox (a := m) (b := k) 1).indicator (fun X => h (X - vu)) (Bhat + vu) := by
            refine lintegral_mono fun Bhat => ?_
            by_cases hB : Bhat ∈ matBox r
            · rw [Set.indicator_of_mem hB, Set.indicator_of_mem (hsub Bhat hB), add_sub_cancel_right]
            · rw [Set.indicator_of_notMem hB]; exact bot_le
        _ = ∫⁻ Btilde in matBox (a := m) (b := k) 1, h (Btilde - vu) := by
            have hm_hs : Measurable fun X : RMat m k => h (X - vu) :=
              hh.comp (measurable_id.sub measurable_const)
            rw [← lintegral_map (hm_hs.indicator (measurableSet_matBox 1)) htr.measurable,
              htr.map_eq, lintegral_indicator (measurableSet_matBox 1)]
    -- Assemble via Fubini
    have hBA_meas : Measurable fun p : RMat m k × RMat k n =>
        ENNReal.ofReal (Real.exp (-(c₂ * N * t ^ 2 *
          rank0Loss (fun i r' => p.1 i r' - v i * u r', p.2)))) := by
      have hm_sub : Measurable fun Btilde : RMat m k => (fun i r' => Btilde i r' - v i * u r' : RMat m k) := by
        fun_prop
      have := (measurable_rank0Loss (m := m) (k' := k) (n' := n)).comp
        ((hm_sub.comp measurable_fst).prodMk measurable_snd)
      fun_prop
    have hBA_m2 : Measurable fun p : RMat m k × RMat k n =>
        ENNReal.ofReal (Real.exp (-(c₂ * N * t ^ 2 * rank0Loss p))) := by
      have := measurable_rank0Loss (m := m) (k' := k) (n' := n); fun_prop
    have hsubA2 : matBox (a := k) (b := n) r ⊆ matBox (a := k) (b := n) 1 := fun A₂ hA₂ => by
      rw [mem_matBox_iff] at hA₂ ⊢; intro r' j; linarith [hA₂ r' j]
    have hstepA : ∀ Btilde : RMat m k,
        ENNReal.ofReal ((1 / 2 : ℝ) ^ n) * gaussBoxZ n (c₂ * N * t ^ 2) *
          ∫⁻ A₂ in matBox (a := k) (b := n) r,
            ENNReal.ofReal (Real.exp (-(c₂ * N * t ^ 2 *
              rank0Loss (fun i r' => Btilde i r' - v i * u r', A₂)))) ≤
        ∫⁻ a in vecBox (d := n) 1, ∫⁻ A₂ in matBox (a := k) (b := n) 1,
          ENNReal.ofReal (Real.exp (-(N * rank0Loss (mkB t (t • u) (t • v) (t • Btilde), mkA a A₂)))) :=
      fun Btilde => by
      let G1 : (Fin n → ℝ) × RMat k n → ℝ≥0∞ := fun p =>
        ENNReal.ofReal (Real.exp (-(c₂ * N * t ^ 2 * ∑ j : Fin n, (p.1 j + ∑ r' : Fin k, u r' * p.2 r' j) ^ 2))) *
        ENNReal.ofReal (Real.exp (-(c₂ * N * t ^ 2 * rank0Loss (fun i r' => Btilde i r' - v i * u r', p.2))))
      have hG1_m : Measurable G1 := by
        unfold G1
        have hmr : Measurable fun p : (Fin n → ℝ) × RMat k n =>
            rank0Loss (fun i r => Btilde i r - v i * u r, p.2) :=
          (measurable_rank0Loss (m := m) (k' := k) (n' := n)).comp
            (measurable_const.prodMk measurable_snd)
        fun_prop
      have hm_A2 : Measurable fun A₂ : RMat k n =>
          ENNReal.ofReal (Real.exp (-(c₂ * N * t ^ 2 *
            rank0Loss (fun i r' => Btilde i r' - v i * u r', A₂)))) := by
        have hmr : Measurable fun A₂ : RMat k n =>
            rank0Loss (fun i r' => Btilde i r' - v i * u r', A₂) :=
          (measurable_rank0Loss (m := m) (k' := k) (n' := n)).comp
            (measurable_const.prodMk measurable_id)
        fun_prop
      calc ENNReal.ofReal ((1 / 2 : ℝ) ^ n) * gaussBoxZ n (c₂ * N * t ^ 2) *
              ∫⁻ A₂ in matBox (a := k) (b := n) r,
                ENNReal.ofReal (Real.exp (-(c₂ * N * t ^ 2 *
                  rank0Loss (fun i r' => Btilde i r' - v i * u r', A₂))))
          = ∫⁻ A₂ in matBox (a := k) (b := n) r,
              (ENNReal.ofReal ((1 / 2 : ℝ) ^ n) * gaussBoxZ n (c₂ * N * t ^ 2)) *
              ENNReal.ofReal (Real.exp (-(c₂ * N * t ^ 2 *
                rank0Loss (fun i r' => Btilde i r' - v i * u r', A₂)))) :=
            (lintegral_const_mul _ hm_A2).symm
        _ ≤ ∫⁻ A₂ in matBox (a := k) (b := n) r, ∫⁻ a in vecBox (d := n) 1, G1 (a, A₂) := by
            refine setLIntegral_mono' (measurableSet_matBox r) fun A₂ hA₂ => ?_
            unfold G1
            dsimp only
            rw [lintegral_mul_const' _ _ ENNReal.ofReal_ne_top]
            exact mul_le_mul' (ha_lower A₂ hA₂) le_rfl
        _ ≤ ∫⁻ A₂ in matBox (a := k) (b := n) 1, ∫⁻ a in vecBox (d := n) 1, G1 (a, A₂) :=
            lintegral_mono_set hsubA2
        _ = ∫⁻ a in vecBox (d := n) 1, ∫⁻ A₂ in matBox (a := k) (b := n) 1, G1 (a, A₂) :=
            (lintegral_lintegral_swap hG1_m.aemeasurable).symm
        _ ≤ ∫⁻ a in vecBox (d := n) 1, ∫⁻ A₂ in matBox (a := k) (b := n) 1,
              ENNReal.ofReal (Real.exp (-(N * rank0Loss (mkB t (t • u) (t • v) (t • Btilde), mkA a A₂)))) :=
            setLIntegral_mono' (measurableSet_vecBox 1) fun a _ =>
              setLIntegral_mono' (measurableSet_matBox 1) fun A₂ _ => hpt Btilde a A₂
    calc ENNReal.ofReal clo * (gaussBoxZ n (c₂ * N * t ^ 2) * rank0Z m k n (c₂ * N * t ^ 2))
        ≤ (ENNReal.ofReal ((1 / 2 : ℝ) ^ n) * gaussBoxZ n (c₂ * N * t ^ 2)) *
            (ENNReal.ofReal r ^ (m * k + k * n) * rank0Z m k n (r ^ 4 * (c₂ * N * t ^ 2))) := by
          have hr4_le : r ^ 4 ≤ 1 := by
            have := pow_le_pow_left₀ hr_pos.le hr_le_one 4; simpa using this
          have hnn : 0 ≤ c₂ * N * t ^ 2 := by positivity
          have hmono_Z : rank0Z m k n (c₂ * N * t ^ 2) ≤ rank0Z m k n (r ^ 4 * (c₂ * N * t ^ 2)) :=
            rank0Z_mono m k n (by nlinarith)
          dsimp [clo]
          rw [ENNReal.ofReal_mul (by positivity), ENNReal.ofReal_pow (p := r) hr_pos.le]
          calc (ENNReal.ofReal ((1 / 2 : ℝ) ^ n) * ENNReal.ofReal r ^ (m * k + k * n)) *
                  (gaussBoxZ n (c₂ * N * t ^ 2) * rank0Z m k n (c₂ * N * t ^ 2))
              ≤ (ENNReal.ofReal ((1 / 2 : ℝ) ^ n) * ENNReal.ofReal r ^ (m * k + k * n)) *
                  (gaussBoxZ n (c₂ * N * t ^ 2) * rank0Z m k n (r ^ 4 * (c₂ * N * t ^ 2))) :=
                mul_le_mul' le_rfl (mul_le_mul' le_rfl hmono_Z)
            _ = _ := by ring
      _ = (ENNReal.ofReal ((1 / 2 : ℝ) ^ n) * gaussBoxZ n (c₂ * N * t ^ 2)) *
            ∫⁻ A₂ in matBox (a := k) (b := n) r, ∫⁻ Bhat in matBox (a := m) (b := k) r,
              ENNReal.ofReal (Real.exp (-(c₂ * N * t ^ 2 * rank0Loss (Bhat, A₂)))) := by
          congr 1
          have hprod :
              SLT.Z (volume.restrict (matBox (a := m) (b := k) r ×ˢ matBox (a := k) (b := n) r))
                rank0Loss (c₂ * N * t ^ 2) =
              ∫⁻ Bhat in matBox (a := m) (b := k) r, ∫⁻ A₂ in matBox (a := k) (b := n) r,
                ENNReal.ofReal (Real.exp (-(c₂ * N * t ^ 2 * rank0Loss (Bhat, A₂)))) := by
            unfold SLT.Z
            change ∫⁻ p in matBox r ×ˢ matBox r, _ ∂(volume.prod volume) = _
            rw [← Measure.prod_restrict, lintegral_prod _ hBA_m2.aemeasurable]
          unfold rank0Z
          rw [← rank0_Z_scale hr_pos, hprod, lintegral_lintegral_swap hBA_m2.aemeasurable]
      _ ≤ (ENNReal.ofReal ((1 / 2 : ℝ) ^ n) * gaussBoxZ n (c₂ * N * t ^ 2)) *
            ∫⁻ A₂ in matBox (a := k) (b := n) r, ∫⁻ Btilde in matBox (a := m) (b := k) 1,
              ENNReal.ofReal (Real.exp (-(c₂ * N * t ^ 2 *
                rank0Loss (fun i r' => Btilde i r' - v i * u r', A₂)))) :=
          mul_le_mul' le_rfl (setLIntegral_mono' (measurableSet_matBox r) fun A₂ _ => hB_lower A₂)
      _ = (ENNReal.ofReal ((1 / 2 : ℝ) ^ n) * gaussBoxZ n (c₂ * N * t ^ 2)) *
            ∫⁻ Btilde in matBox (a := m) (b := k) 1, ∫⁻ A₂ in matBox (a := k) (b := n) r,
              ENNReal.ofReal (Real.exp (-(c₂ * N * t ^ 2 *
                rank0Loss (fun i r' => Btilde i r' - v i * u r', A₂)))) := by
          rw [← lintegral_lintegral_swap hBA_meas.aemeasurable]
      _ = ∫⁻ Btilde in matBox (a := m) (b := k) 1,
            (ENNReal.ofReal ((1 / 2 : ℝ) ^ n) * gaussBoxZ n (c₂ * N * t ^ 2)) *
              ∫⁻ A₂ in matBox (a := k) (b := n) r,
                ENNReal.ofReal (Real.exp (-(c₂ * N * t ^ 2 *
                  rank0Loss (fun i r' => Btilde i r' - v i * u r', A₂)))) :=
          (lintegral_const_mul _ hBA_meas.lintegral_prod_right').symm
      _ ≤ ∫⁻ Btilde in matBox (a := m) (b := k) 1,
          ∫⁻ a in vecBox (d := n) 1, ∫⁻ A₂ in matBox (a := k) (b := n) 1,
            ENNReal.ofReal (Real.exp (-(N * rank0Loss (mkB t (t • u) (t • v) (t • Btilde), mkA a A₂)))) :=
          setLIntegral_mono' (measurableSet_matBox 1) fun Btilde _ => hstepA Btilde
  calc ENNReal.ofReal clo * (gaussBoxZ n (c₂ * N * t ^ 2) * rank0Z m k n (c₂ * N * t ^ 2))
      = ∫⁻ _ in vecBox (d := k) (1 / 2), ∫⁻ _ in vecBox (d := m) (1 / 2),
          ENNReal.ofReal clo * (gaussBoxZ n (c₂ * N * t ^ 2) * rank0Z m k n (c₂ * N * t ^ 2)) := by
        rw [setLIntegral_const, setLIntegral_const,
          volume_vecBox (show (0 : ℝ) ≤ 1 / 2 by norm_num),
          volume_vecBox (show (0 : ℝ) ≤ 1 / 2 by norm_num)]
        norm_num
    _ ≤ ∫⁻ u in vecBox (d := k) (1 / 2), ∫⁻ v in vecBox (d := m) (1 / 2),
          ∫⁻ Btilde in matBox (a := m) (b := k) 1,
          ∫⁻ a in vecBox (d := n) 1, ∫⁻ A₂ in matBox (a := k) (b := n) 1,
            ENNReal.ofReal (Real.exp (-(N * rank0Loss (mkB t (t • u) (t • v) (t • Btilde), mkA a A₂)))) :=
        setLIntegral_mono' (measurableSet_vecBox (1 / 2)) fun u hu =>
          setLIntegral_mono' (measurableSet_vecBox (1 / 2)) fun v hv => hslice u hu v hv
    _ ≤ ∫⁻ u in vecBox (d := k) 1, ∫⁻ v in vecBox (d := m) 1,
          ∫⁻ Btilde in matBox (a := m) (b := k) 1,
          ∫⁻ a in vecBox (d := n) 1, ∫⁻ A₂ in matBox (a := k) (b := n) 1,
            ENNReal.ofReal (Real.exp (-(N * rank0Loss (mkB t (t • u) (t • v) (t • Btilde), mkA a A₂)))) :=
        (setLIntegral_mono' (measurableSet_vecBox (1 / 2)) fun _ _ =>
          lintegral_mono_set hsub_v).trans (lintegral_mono_set hsub_u)

/-! ### 5. Main Theorems: Full Aoyagi–Watanabe (2005) & Complete LoRA Fiber Phase Transition -/

/-- **Full inductive `NCBound` for `rank0Z m k n` across ALL `(m, k, n) : ℕ × ℕ × ℕ`.** -/
theorem rank0Z_ncBound : ∀ (m k n : ℕ),
    SLT.NC.NCBound (rank0Z m k n) (awLambda m n k) (awMult m n k)
  | 0, k, n => by
    rw [awLambda_zero_left, awMult_zero_left]
    have heq : rank0Z 0 k n = fun _ => ENNReal.ofReal ((2 : ℝ) ^ (k * n)) := by
      funext N
      unfold rank0Z SLT.Z rank0Loss frobSq
      simp only [Finset.univ_eq_empty, Finset.sum_empty, mul_zero, neg_zero, Real.exp_zero,
        ENNReal.ofReal_one, setLIntegral_const, one_mul]
      change (volume.prod volume) (matBox (a := 0) (b := k) 1 ×ˢ matBox (a := k) (b := n) 1) = _
      rw [Measure.prod_prod, volume_matBox zero_le_one, volume_matBox zero_le_one]
      simp [ENNReal.ofReal_pow]
    rw [heq]
    exact ncBound_const (by positivity)
  | m + 1, 0, n => by
    rw [awLambda_zero_right, awMult_zero_right]
    have heq : rank0Z (m + 1) 0 n = fun _ => ENNReal.ofReal 1 := by
      funext N
      unfold rank0Z SLT.Z rank0Loss frobSq
      simp only [Matrix.mul_apply, Finset.univ_eq_empty, Finset.sum_empty, sq, mul_zero,
        Finset.sum_const_zero, neg_zero, Real.exp_zero, ENNReal.ofReal_one, setLIntegral_const,
        one_mul]
      change (volume.prod volume) (matBox (a := m + 1) (b := 0) 1 ×ˢ matBox (a := 0) (b := n) 1) = _
      rw [Measure.prod_prod, volume_matBox zero_le_one, volume_matBox zero_le_one]
      simp
    rw [heq]
    exact ncBound_const one_pos
  | m + 1, k + 1, n => by
    have ih := rank0Z_ncBound m k n
    have hG := ncBound_gaussBoxZ n
    have hprod_raw := ncBound_mul hG ih le_rfl (awMult_pos m n k)
    have hmult_simp : 1 + awMult m n k - 1 = awMult m n k := by omega
    rw [hmult_simp] at hprod_raw
    -- Apply `SLT.NC.stepG` with `h = (m+1)(k+1) - 1` and `k_step = 2`
    let h : ℝ := (((m + 1) * (k + 1) - 1 : ℕ) : ℝ)
    have hh : (-1 : ℝ) < h := by
      dsimp [h]; linarith [Nat.cast_nonneg (α := ℝ) ((m + 1) * (k + 1) - 1)]
    have hmu_eq : (h + 1) / (2 : ℕ) = (((m + 1) * (k + 1) : ℕ) : ℝ) / 2 := by
      dsimp [h]
      have h1 : ((m + 1) * (k + 1) - 1) + 1 = (m + 1) * (k + 1) := by
        have : 1 ≤ (m + 1) * (k + 1) := Nat.mul_pos (Nat.succ_pos m) (Nat.succ_pos k)
        omega
      have hcast : (((m + 1) * (k + 1) - 1 : ℕ) : ℝ) + 1 = (((m + 1) * (k + 1) : ℕ) : ℝ) := by
        exact_mod_cast h1
      rw [hcast]
    let H : ℝ → ℝ≥0∞ :=
      SLT.NC.stepG (fun M => gaussBoxZ n M * rank0Z m k n M) h 2
    have hH : SLT.NC.NCBound H (awLambda (m + 1) n (k + 1)) (awMult (m + 1) n (k + 1)) := by
      rcases aw_recursion m k n with ⟨hlt, hL, hM⟩ | ⟨heq, hL, hM⟩ | ⟨hgt, hL, hM⟩
      · rw [hL, hM]
        have hlam_nn : 0 ≤ (n : ℝ) / 2 + awLambda m n k := by
          linarith [awLambda_nonneg m n k, show (0 : ℝ) ≤ (n : ℝ) / 2 by positivity]
        exact hprod_raw.step_lt hlam_nn hh (by norm_num) (by rw [hmu_eq]; exact hlt)
      · rw [hL, hM, ← heq]
        exact hprod_raw.step_eq (awMult_pos m n k) hh (by norm_num) (by rw [hmu_eq]; exact heq)
      · rw [hL, hM, ← hmu_eq]
        exact hprod_raw.step_gt hh (by norm_num) (by rw [hmu_eq]; exact hgt)
    -- Sandwich `rank0Z (m + 1) (k + 1) n` between `clo * H (c₂ * N)` and `Cup_tot * H (c₁ * N)`
    let c₁ : ℝ := 1 / (2 * (m : ℝ) + 2)
    let c₂ : ℝ := 2 * (m : ℝ) + 2
    let r : ℝ := 1 / (2 * ((k + 1 : ℕ) : ℝ))
    let clo : ℝ := (1 / 2 : ℝ) ^ n * r ^ (m * k + k * n)
    let Cup_tot : ℝ :=
      (2 * ((m + 1 : ℕ) : ℝ) * ((k + 1 : ℕ) : ℝ)) *
        (2 ^ (k + m + m * k + k * n) * ((k + 1 : ℕ) : ℝ) ^ n)
    have hlo_bound : SLT.NC.NCBound (fun N => ENNReal.ofReal clo * H (c₂ * N))
        (awLambda (m + 1) n (k + 1)) (awMult (m + 1) n (k + 1)) :=
      (ncBound_comp_mul hH (awLambda_nonneg _ _ _) (by dsimp [c₂]; positivity)).const_mul
        (by dsimp [clo, r]; positivity)
    have hup_bound : SLT.NC.NCBound (fun N => ENNReal.ofReal Cup_tot * H (c₁ * N))
        (awLambda (m + 1) n (k + 1)) (awMult (m + 1) n (k + 1)) :=
      (ncBound_comp_mul hH (awLambda_nonneg _ _ _) (by dsimp [c₁]; positivity)).const_mul
        (by dsimp [Cup_tot]; positivity)
    refine ncBound_of_sandwich
      (fun M M' _ hMM => rank0Z_mono (m + 1) (k + 1) n hMM)
      (fun N hN => (stepG_le_chartPosZ m k n hN).trans (rank0Z_sandwich_chartPos m k n N).1)
      (fun N hN => ?_)
      hlo_bound hup_bound
    calc rank0Z (m + 1) (k + 1) n N
        ≤ ENNReal.ofReal (2 * ((m + 1 : ℕ) : ℝ) * ((k + 1 : ℕ) : ℝ)) * chartPosZ m k n N :=
          (rank0Z_sandwich_chartPos m k n N).2
      _ ≤ ENNReal.ofReal (2 * ((m + 1 : ℕ) : ℝ) * ((k + 1 : ℕ) : ℝ)) *
            (ENNReal.ofReal (2 ^ (k + m + m * k + k * n) * ((k + 1 : ℕ) : ℝ) ^ n) * H (c₁ * N)) :=
          mul_le_mul' le_rfl (chartPosZ_le_stepG m k n hN)
      _ = ENNReal.ofReal Cup_tot * H (c₁ * N) := by
          rw [← mul_assoc, ← ENNReal.ofReal_mul (by positivity)]

/-- **Aoyagi–Watanabe (2005) Theorem 1 (Full, Unconditional):**
For **all** `m, k, n : ℕ`, the rank-0 reduced-rank matrix factorization loss `‖B A‖_F²` on
`[-1, 1]^{m×k} × [-1, 1]^{k×n}` has exact learning pair `(awLambda m n k, awMult m n k)`. -/
theorem aoyagi_watanabe_rank0 (m k n : ℕ) :
    SLT.HasLaplacePair (rank0Z m k n) (awLambda m n k) (awMult m n k) :=
  (rank0Z_ncBound m k n).hasLaplacePair (awMult_pos m n k)

/-- **Complete Unconditional LoRA Zero-Loss Fiber Theorem (All `m, n, k, q`):**
For ALL `m, n, k, q : ℕ` with `q ≤ k` and `q ≤ n`, the exact LoRA fiber loss around
`(B₀, A₀) = (0, fromBlocks 1 0 0 0)` (rank-`q` half-dead `A₀`) on every sufficiently small
uniform box `fiberBox δ` has **unconditionally Lean-certified** exact learning pair
`(m * q / 2 + awLambda m (n - q) (k - q), awMult m (n - q) (k - q))`. -/
theorem lora_fiber_exact_pair {m n k q : ℕ} (_hqk : q ≤ k) (_hqn : q ≤ n) :
    ∃ δ₀ : ℝ, 0 < δ₀ ∧ ∀ δ : ℝ, 0 < δ → δ ≤ δ₀ →
      SLT.HasLaplacePair
        (SLT.Z (volume.restrict (fiberBox (m := m) (q := q) (k' := k - q) (n' := n - q) δ))
          loraFiberLoss)
        ((m : ℝ) * q / 2 + awLambda m (n - q) (k - q))
        (awMult m (n - q) (k - q)) :=
  lora_schur_fiber_reduction (q := q) (aoyagi_watanabe_rank0 m (k - q) (n - q))

/-- **Unconditional Subcritical LoRA Fiber Plateau (`m + 2 * q < n + k`):**
For ALL interior `(m, n, k)` and ALL `q` below the critical rank `q_c = (n + k - m) / 2`,
the exact local learning pair is **flat (independent of `q`)** and equal to the global minimum
`(awLambda m n k, awMult m n k)`. -/
theorem lora_subcritical_plateau_exact_pair {m n k q : ℕ}
    (h1 : k < m + n) (h2 : n < m + k) (h3 : m < n + k)
    (hqk : q ≤ k) (hqn : q ≤ n) (hsub : m + 2 * q < n + k) :
    ∃ δ₀ : ℝ, 0 < δ₀ ∧ ∀ δ : ℝ, 0 < δ → δ ≤ δ₀ →
      SLT.HasLaplacePair
        (SLT.Z (volume.restrict (fiberBox (m := m) (q := q) (k' := k - q) (n' := n - q) δ))
          loraFiberLoss)
        (awLambda m n k)
        (awMult m n k) := by
  obtain ⟨δ₀, hδ₀, hred⟩ := lora_fiber_exact_pair (m := m) hqk hqn
  refine ⟨δ₀, hδ₀, fun δ hδ hδle => ?_⟩
  rw [← lora_plateau_lambda h1 h2 h3 hqk hqn hsub, ← lora_plateau_mult h1 h2 h3 hqk hqn hsub]
  exact hred δ hδ hδle

end EatYourFiber
