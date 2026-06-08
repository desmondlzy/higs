import torch

def even_split_opacities(opacities: torch.Tensor):
	"""
	solution to:
	(alpha_2(1 - alpha_1) + alpha_1) = alpha_orig
	when alpha_1 = alpha_2 
	"""
	assert torch.all(opacities >= 0) and torch.all(opacities <= 1)
	split_opacities = -torch.sqrt(1 - opacities) + 1

	return split_opacities
	