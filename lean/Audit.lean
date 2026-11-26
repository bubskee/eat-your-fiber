import EatYourFiber

-- Algebraic LoRA Schur-complement & Aoyagi–Watanabe phase transition identities
#print axioms EatYourFiber.schur_block_right
#print axioms EatYourFiber.schur_loss_lower
#print axioms EatYourFiber.schur_loss_upper
#print axioms EatYourFiber.lora_plateau_lambda
#print axioms EatYourFiber.lora_plateau_mult
#print axioms EatYourFiber.lora_parabola_lambda
#print axioms EatYourFiber.lora_parabola_mult

-- Measure-theoretic Laplace reduction & unconditional standard LoRA initialization theorem
#print axioms EatYourFiber.hasLaplacePair_frobSq
#print axioms EatYourFiber.rank0_hasLaplacePair_of_unit
#print axioms EatYourFiber.lora_schur_fiber_reduction
#print axioms EatYourFiber.lora_standard_init_exact_pair

-- Unconditional Aoyagi–Watanabe (2005) boundary regimes & full supercritical LoRA parabola
#print axioms EatYourFiber.rank0_laplaceLower_kn
#print axioms EatYourFiber.lintegral_gaussian_cols_le
#print axioms EatYourFiber.lintegral_orthHeight_inv_pow_lt_top
#print axioms EatYourFiber.rank0_hasLaplacePair_of_le_m
#print axioms EatYourFiber.rank0_hasLaplacePair_of_le_n
#print axioms EatYourFiber.lora_supercritical_parabola_exact_pair

-- Full Aoyagi–Watanabe (2005) Theorem 1 & Complete Unconditional LoRA Fiber Phase Transition
#print axioms EatYourFiber.rank0Z_ncBound
#print axioms EatYourFiber.aoyagi_watanabe_rank0
#print axioms EatYourFiber.lora_fiber_exact_pair
#print axioms EatYourFiber.lora_subcritical_plateau_exact_pair

-- Self-Attention LoRA Coupling Dichotomy: V+O (4x jump) vs Q+K (GL(q) flat cancellation)
#print axioms EatYourFiber.attnVO_factor
#print axioms EatYourFiber.attnQK_factor
#print axioms EatYourFiber.attention_QK_gauge_orbit
#print axioms EatYourFiber.attention_VO_exact_pair
#print axioms EatYourFiber.attention_VO_subcritical_plateau
#print axioms EatYourFiber.attention_VO_supercritical_parabola
#print axioms EatYourFiber.twoSided_schur_identities
#print axioms EatYourFiber.twoSided_loss_bounds
#print axioms EatYourFiber.twoSided_schur_reduction
#print axioms EatYourFiber.attention_QK_exact_pair
#print axioms EatYourFiber.attention_QK_flat_pair
#print axioms EatYourFiber.attention_coupling_dichotomy_standard_init

-- Order-3 Bilinear MLPs, Border-Rank Gap, and Stretched-Exponential SLT (Track 1)
#print axioms EatYourFiber.exists_zero_loss_iff_tensorRank_le
#print axioms EatYourFiber.geometricBorderRank_le_of_approx_loss
#print axioms EatYourFiber.wTensor_geometricBorderRank
#print axioms EatYourFiber.wTensor_tensorRank
#print axioms EatYourFiber.wTensor_borderRank_lt_tensorRank
#print axioms EatYourFiber.wTensor_loss_ge_of_paramNormSq
#print axioms EatYourFiber.wTensor_escape_upper
#print axioms EatYourFiber.wTensor_weightDecay_lower
#print axioms EatYourFiber.wTensor_weightDecay_upper
#print axioms EatYourFiber.wPartitionZ_two_le_stretchedExp
#print axioms EatYourFiber.wPartitionZ_two_ge_stretchedExp
#print axioms EatYourFiber.wPartitionZ_three_ge_poly

-- Vendored OAI Tensor & BorderRank highlights (for Track 1: Bilinear MLPs)
#print axioms OAI.LinearAlgebra.SpectralBooks.geometricBorderRank_le_tensorRank
#print axioms OAI.LinearAlgebra.SpectralBooks.card_le_geometricBorderRank_of_minor
#print axioms OAI.LinearAlgebra.SpectralBooks.rank_fromBlocks_diagonal
