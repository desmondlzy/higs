import torch

def reflect(win, normal): 
	return 2 * torch.sum(win * normal, axis=-1, keepdim=True) * normal - win