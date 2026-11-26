/-
Copyright (c) 2026 Bobby Faber. All rights reserved.
Released under Apache 2.0 license.
-/
module

public import Mathlib.Data.Matrix.Basic
public import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Algebra.Order.BigOperators.Ring.Finset
public import Mathlib.Tactic.Linarith
public import Mathlib.Tactic.Ring
public import Mathlib.Tactic.Positivity
public import SLT

/-!
# LoRA / Reduced-Rank Fiber Phase Transition: Algebraic Core & Schur-Complement Reduction

This module establishes:
1. Elementary Frobenius norm-squared bounds (`frobSq_nonneg`, `frobSq_add_le`, `frobSq_mul_le`).
2. The exact block Schur-complement identity (`schur_block_right`) and two-sided loss comparison
   (`schur_loss_lower`, `schur_loss_upper`) around any rank-`q` half-dead fiber point
   `(B₀, A₀) = (0, fromBlocks 1 0 0 0)` of `{B A = 0}`.
3. The exact algebraic Aoyagi–Watanabe fiber phase transition identity (`lora_plateau_lambda`,
   `lora_plateau_mult`, `lora_parabola_lambda`, `lora_parabola_mult`):
   - Subcritical plateau (`2 * q ≤ n + k - m`): `m*q/2 + awLambda m (n-q) (k-q) = awLambda m n k`,
     with unchanged multiplicity.
   - Supercritical parabola (`n + k - m ≤ 2 * q`):
     `m*q/2 + awLambda m (n-q) (k-q) = awLambdaEven m n k + (q - q_c)^2 / 2`, multiplicity `1`.
-/

@[expose] public section

namespace EatYourFiber

open Finset Matrix

/-! ### 1. Frobenius norm squared -/

/-- Squared Frobenius norm of a real matrix: `∑ i, ∑ j, (X i j)^2`. -/
def frobSq {a b : ℕ} (X : Matrix (Fin a) (Fin b) ℝ) : ℝ :=
  ∑ i : Fin a, ∑ j : Fin b, (X i j) ^ 2

theorem frobSq_nonneg {a b : ℕ} (X : Matrix (Fin a) (Fin b) ℝ) : 0 ≤ frobSq X :=
  Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _

theorem frobSq_add_le {a b : ℕ} (X Y : Matrix (Fin a) (Fin b) ℝ) :
    frobSq (X + Y) ≤ 2 * frobSq X + 2 * frobSq Y := by
  unfold frobSq
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun i _ => ?_
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun j _ => ?_
  simp only [Matrix.add_apply]
  nlinarith [sq_nonneg (X i j - Y i j)]

theorem frobSq_sub_le {a b : ℕ} (X Y : Matrix (Fin a) (Fin b) ℝ) :
    frobSq (X - Y) ≤ 2 * frobSq X + 2 * frobSq Y := by
  unfold frobSq
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun i _ => ?_
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun j _ => ?_
  simp only [Matrix.sub_apply]
  nlinarith [sq_nonneg (X i j + Y i j)]

/-- Submultiplicativity of squared Frobenius norm: `‖X Y‖_F^2 ≤ ‖X‖_F^2 ‖Y‖_F^2`. -/
theorem frobSq_mul_le {a b c : ℕ}
    (X : Matrix (Fin a) (Fin b) ℝ) (Y : Matrix (Fin b) (Fin c) ℝ) :
    frobSq (X * Y) ≤ frobSq X * frobSq Y := by
  unfold frobSq
  calc
    ∑ i : Fin a, ∑ j : Fin c, ((X * Y) i j) ^ 2
      = ∑ i : Fin a, ∑ j : Fin c, (∑ r : Fin b, X i r * Y r j) ^ 2 := by
          simp only [Matrix.mul_apply]
    _ ≤ ∑ i : Fin a, ∑ j : Fin c,
          (∑ r : Fin b, (X i r) ^ 2) * (∑ r : Fin b, (Y r j) ^ 2) := by
          refine Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ => ?_
          exact Finset.sum_mul_sq_le_sq_mul_sq Finset.univ (fun r => X i r) (fun r => Y r j)
    _ = (∑ i : Fin a, ∑ r : Fin b, (X i r) ^ 2) *
          (∑ j : Fin c, ∑ r : Fin b, (Y r j) ^ 2) := by
          simp_rw [← Finset.mul_sum, ← Finset.sum_mul]
    _ = (∑ i : Fin a, ∑ r : Fin b, (X i r) ^ 2) *
          (∑ r : Fin b, ∑ j : Fin c, (Y r j) ^ 2) := by
          congr 1
          exact Finset.sum_comm

/-! ### 2. Exact Schur-complement block identity and two-sided comparison -/

/-- Exact Schur-complement rewrite of the right output block `B₁ Δ₁₂ + B₂ Δ₂₂`. -/
theorem schur_block_right {m q k' n' : ℕ}
    (B₁ : Matrix (Fin m) (Fin q) ℝ)
    (B₂ : Matrix (Fin m) (Fin k') ℝ)
    (M : Matrix (Fin q) (Fin q) ℝ)
    (hM : IsUnit M.det)
    (Δ₁₂ : Matrix (Fin q) (Fin n') ℝ)
    (Δ₂₁ : Matrix (Fin k') (Fin q) ℝ)
    (Δ₂₂ : Matrix (Fin k') (Fin n') ℝ) :
    B₁ * Δ₁₂ + B₂ * Δ₂₂ =
      (B₁ * M + B₂ * Δ₂₁) * (M⁻¹ * Δ₁₂) + B₂ * (Δ₂₂ - Δ₂₁ * M⁻¹ * Δ₁₂) := by
  have hMM : M * M⁻¹ = 1 := Matrix.mul_nonsing_inv M hM
  symm
  calc
    (B₁ * M + B₂ * Δ₂₁) * (M⁻¹ * Δ₁₂) + B₂ * (Δ₂₂ - Δ₂₁ * M⁻¹ * Δ₁₂)
      = (B₁ * M * (M⁻¹ * Δ₁₂) + B₂ * Δ₂₁ * (M⁻¹ * Δ₁₂)) +
          (B₂ * Δ₂₂ - B₂ * (Δ₂₁ * M⁻¹ * Δ₁₂)) := by
          rw [Matrix.add_mul, Matrix.mul_sub]
    _ = (B₁ * (M * M⁻¹) * Δ₁₂ + B₂ * (Δ₂₁ * M⁻¹ * Δ₁₂)) +
          (B₂ * Δ₂₂ - B₂ * (Δ₂₁ * M⁻¹ * Δ₁₂)) := by
          simp only [Matrix.mul_assoc]
    _ = (B₁ * Δ₁₂ + B₂ * (Δ₂₁ * M⁻¹ * Δ₁₂)) +
          (B₂ * Δ₂₂ - B₂ * (Δ₂₁ * M⁻¹ * Δ₁₂)) := by
          rw [hMM, Matrix.mul_one]
    _ = B₁ * Δ₁₂ + B₂ * Δ₂₂ := by
          abel

/-- Upper loss comparison after Schur complement when `‖M⁻¹ Δ₁₂‖_F^2 ≤ 1/4`. -/
theorem schur_loss_upper {m q k' n' : ℕ}
    (B₁ : Matrix (Fin m) (Fin q) ℝ)
    (B₂ : Matrix (Fin m) (Fin k') ℝ)
    (M : Matrix (Fin q) (Fin q) ℝ)
    (hM : IsUnit M.det)
    (Δ₁₂ : Matrix (Fin q) (Fin n') ℝ)
    (Δ₂₁ : Matrix (Fin k') (Fin q) ℝ)
    (Δ₂₂ : Matrix (Fin k') (Fin n') ℝ)
    (hC : frobSq (M⁻¹ * Δ₁₂) ≤ 1 / 4) :
    frobSq (B₁ * M + B₂ * Δ₂₁) + frobSq (B₁ * Δ₁₂ + B₂ * Δ₂₂) ≤
      2 * (frobSq (B₁ * M + B₂ * Δ₂₁) + frobSq (B₂ * (Δ₂₂ - Δ₂₁ * M⁻¹ * Δ₁₂))) := by
  set B₁' := B₁ * M + B₂ * Δ₂₁
  set C := M⁻¹ * Δ₁₂
  set R := B₂ * (Δ₂₂ - Δ₂₁ * M⁻¹ * Δ₁₂)
  rw [schur_block_right B₁ B₂ M hM Δ₁₂ Δ₂₁ Δ₂₂]
  have h1 : frobSq (B₁' * C + R) ≤ 2 * frobSq (B₁' * C) + 2 * frobSq R :=
    frobSq_add_le _ _
  have h2 : frobSq (B₁' * C) ≤ frobSq B₁' * frobSq C := frobSq_mul_le _ _
  have h3 : frobSq B₁' * frobSq C ≤ frobSq B₁' * (1 / 4) :=
    mul_le_mul_of_nonneg_left hC (frobSq_nonneg B₁')
  linarith [frobSq_nonneg B₁', frobSq_nonneg R]

/-- Lower loss comparison after Schur complement when `‖M⁻¹ Δ₁₂‖_F^2 ≤ 1/4`. -/
theorem schur_loss_lower {m q k' n' : ℕ}
    (B₁ : Matrix (Fin m) (Fin q) ℝ)
    (B₂ : Matrix (Fin m) (Fin k') ℝ)
    (M : Matrix (Fin q) (Fin q) ℝ)
    (hM : IsUnit M.det)
    (Δ₁₂ : Matrix (Fin q) (Fin n') ℝ)
    (Δ₂₁ : Matrix (Fin k') (Fin q) ℝ)
    (Δ₂₂ : Matrix (Fin k') (Fin n') ℝ)
    (hC : frobSq (M⁻¹ * Δ₁₂) ≤ 1 / 4) :
    (1 / 4 : ℝ) * (frobSq (B₁ * M + B₂ * Δ₂₁) + frobSq (B₂ * (Δ₂₂ - Δ₂₁ * M⁻¹ * Δ₁₂))) ≤
      frobSq (B₁ * M + B₂ * Δ₂₁) + frobSq (B₁ * Δ₁₂ + B₂ * Δ₂₂) := by
  set B₁' := B₁ * M + B₂ * Δ₂₁
  set C := M⁻¹ * Δ₁₂
  set R := B₂ * (Δ₂₂ - Δ₂₁ * M⁻¹ * Δ₁₂)
  set MR := B₁ * Δ₁₂ + B₂ * Δ₂₂
  have hR_eq : R = MR - B₁' * C := by
    have h := schur_block_right B₁ B₂ M hM Δ₁₂ Δ₂₁ Δ₂₂
    exact eq_sub_of_add_eq' h.symm
  have h1 : frobSq R ≤ 2 * frobSq MR + 2 * frobSq (B₁' * C) := by
    rw [hR_eq]
    exact frobSq_sub_le MR (B₁' * C)
  have h2 : frobSq (B₁' * C) ≤ frobSq B₁' * frobSq C := frobSq_mul_le _ _
  have h3 : frobSq B₁' * frobSq C ≤ frobSq B₁' * (1 / 4) :=
    mul_le_mul_of_nonneg_left hC (frobSq_nonneg B₁')
  linarith [frobSq_nonneg B₁', frobSq_nonneg MR]

/-! ### 3. Exact Aoyagi–Watanabe (2005) formula and the LoRA Fiber Phase Transition -/

/-- The interior even-parity Aoyagi–Watanabe quadratic form
`(2(mn + nk + km) - m^2 - n^2 - k^2) / 8`. -/
noncomputable def awLambdaEven (m n k : ℕ) : ℝ :=
  (2 * ((m : ℝ) * n + n * k + k * m) - (m : ℝ) ^ 2 - (n : ℝ) ^ 2 - (k : ℝ) ^ 2) / 8

/-- Exact Aoyagi–Watanabe (2005) Theorem 1 RLCT `λ(m, n, k)` for `‖B A‖_F^2`
with `B : ℝ^{m × k}`, `A : ℝ^{k × n}`. -/
noncomputable def awLambda (m n k : ℕ) : ℝ :=
  if m + n ≤ k then (m : ℝ) * n / 2
  else if k + n ≤ m then (k : ℝ) * n / 2
  else if k + m ≤ n then (k : ℝ) * m / 2
  else if Even (m + n + k) then awLambdaEven m n k
  else awLambdaEven m n k + 1 / 8

/-- Exact Aoyagi–Watanabe (2005) Theorem 1 multiplicity `m_log(m, n, k)`. -/
def awMult (m n k : ℕ) : ℕ :=
  if m + n ≤ k then 1
  else if k + n ≤ m then 1
  else if k + m ≤ n then 1
  else if Even (m + n + k) then 1
  else 2

/-- Subcritical plateau (`2 * q < n + k - m`, interior regime):
the local learning coefficient `m*q/2 + awLambda m (n-q) (k-q)` is **constant in `q`**
and equal to the global minimum `awLambda m n k`. -/
theorem lora_plateau_lambda {m n k q : ℕ}
    (h1 : k < m + n) (h2 : n < m + k) (h3 : m < n + k)
    (hqk : q ≤ k) (hqn : q ≤ n) (hsub : m + 2 * q < n + k) :
    (m : ℝ) * q / 2 + awLambda m (n - q) (k - q) = awLambda m n k := by
  have h1' : ¬ (m + (n - q) ≤ k - q) := by omega
  have h2' : ¬ ((k - q) + (n - q) ≤ m) := by omega
  have h3' : ¬ ((k - q) + m ≤ n - q) := by omega
  have h1₀ : ¬ (m + n ≤ k) := by omega
  have h2₀ : ¬ (k + n ≤ m) := by omega
  have h3₀ : ¬ (k + m ≤ n) := by omega
  have hpar : Even (m + (n - q) + (k - q)) ↔ Even (m + n + k) := by
    have heq : m + (n - q) + (k - q) + 2 * q = m + n + k := by omega
    rw [← heq]
    simp [Nat.even_add, even_two]
  unfold awLambda
  simp only [h1', h2', h3', h1₀, h2₀, h3₀, ↓reduceIte, hpar]
  have hsub_n : ((n - q : ℕ) : ℝ) = (n : ℝ) - (q : ℝ) := Nat.cast_sub hqn
  have hsub_k : ((k - q : ℕ) : ℝ) = (k : ℝ) - (q : ℝ) := Nat.cast_sub hqk
  split_ifs <;> { unfold awLambdaEven; rw [hsub_n, hsub_k]; ring }

/-- Subcritical plateau multiplicity: unchanged across `m + 2*q < n + k`. -/
theorem lora_plateau_mult {m n k q : ℕ}
    (h1 : k < m + n) (h2 : n < m + k) (h3 : m < n + k)
    (hqk : q ≤ k) (hqn : q ≤ n) (hsub : m + 2 * q < n + k) :
    awMult m (n - q) (k - q) = awMult m n k := by
  have h1' : ¬ (m + (n - q) ≤ k - q) := by omega
  have h2' : ¬ ((k - q) + (n - q) ≤ m) := by omega
  have h3' : ¬ ((k - q) + m ≤ n - q) := by omega
  have h1₀ : ¬ (m + n ≤ k) := by omega
  have h2₀ : ¬ (k + n ≤ m) := by omega
  have h3₀ : ¬ (k + m ≤ n) := by omega
  have hpar : Even (m + (n - q) + (k - q)) ↔ Even (m + n + k) := by
    have heq : m + (n - q) + (k - q) + 2 * q = m + n + k := by omega
    rw [← heq]
    simp [Nat.even_add, even_two]
  unfold awMult
  simp only [h1', h2', h3', h1₀, h2₀, h3₀, ↓reduceIte, hpar]

/-- Supercritical parabola (`n + k ≤ m + 2*q`):
the local learning coefficient climbs **quadratically** as `awLambdaEven m n k + (q - q_c)^2 / 2`
where `q_c = (n + k - m) / 2`. -/
theorem lora_parabola_lambda {m n k q : ℕ}
    (h1 : k < m + n) (hqk : q ≤ k) (hqn : q ≤ n) (hsup : n + k ≤ m + 2 * q) :
    (m : ℝ) * q / 2 + awLambda m (n - q) (k - q) =
      awLambdaEven m n k + ((q : ℝ) - ((n : ℝ) + k - m) / 2) ^ 2 / 2 := by
  have h1' : ¬ (m + (n - q) ≤ k - q) := by omega
  have h2' : (k - q) + (n - q) ≤ m := by omega
  unfold awLambda awLambdaEven
  simp only [h1', h2', ↓reduceIte]
  have hsub_n : ((n - q : ℕ) : ℝ) = (n : ℝ) - (q : ℝ) := Nat.cast_sub hqn
  have hsub_k : ((k - q : ℕ) : ℝ) = (k : ℝ) - (q : ℝ) := Nat.cast_sub hqk
  rw [hsub_n, hsub_k]
  ring

/-- Supercritical multiplicity: always `1` (no logarithm) for `n + k ≤ m + 2*q`. -/
theorem lora_parabola_mult {m n k q : ℕ}
    (h1 : k < m + n) (hqk : q ≤ k) (hqn : q ≤ n) (hsup : n + k ≤ m + 2 * q) :
    awMult m (n - q) (k - q) = 1 := by
  have h1' : ¬ (m + (n - q) ≤ k - q) := by omega
  have h2' : (k - q) + (n - q) ≤ m := by omega
  unfold awMult
  simp only [h1', h2', ↓reduceIte]

end EatYourFiber
