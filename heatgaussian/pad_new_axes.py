import torch

def pad_new_axes(tensor: torch.Tensor, num: int) -> torch.Tensor:
	if num > 0:
		return tensor.view(*[1 for _ in range(num)], *tensor.shape)
	else:
		return tensor
