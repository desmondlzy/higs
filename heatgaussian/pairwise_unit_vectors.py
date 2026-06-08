import torch

def pairwise_unit_vectors(froms, tos):
	"""
	return the unit vectors from froms to tos
	vectors: (n_tos, n_shapes, 3)

	"""
	n_froms = froms.shape[0]
	n_tos = tos.shape[0]

	assert froms.shape == (n_froms, 3), f"froms shape {froms.shape}"
	assert tos.shape == (n_tos, 3), f"tos shape {tos.shape}"

	# (n_froms, n_tos, 3)
	vectors = torch.nn.functional.normalize(
		tos.view(n_tos, 1, 3) - froms.view(1, n_froms, 3)
	)

	return vectors