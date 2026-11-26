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

-- Vendored OAI Tensor & BorderRank highlights (for Track 1: Bilinear MLPs)
#print axioms OAI.LinearAlgebra.SpectralBooks.geometricBorderRank_le_tensorRank
#print axioms OAI.LinearAlgebra.SpectralBooks.card_le_geometricBorderRank_of_minor
#print axioms OAI.LinearAlgebra.SpectralBooks.rank_fromBlocks_diagonal
