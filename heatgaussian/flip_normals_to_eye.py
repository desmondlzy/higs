import torch

def flip_normals_to_eye(positions: torch.Tensor, normals: torch.Tensor, eye: torch.Tensor):
	assert eye.shape == (3, ) or eye.shape == (1, 3), f"eye shape {eye.shape} != expected (3, ) or (1, 3)"

	pos_to_eye = eye.view(1, 3) - positions
	
	return torch.where(
		torch.sum(pos_to_eye * normals, axis=-1, keepdim=True) < 0,
		-normals,
		normals,
	)
