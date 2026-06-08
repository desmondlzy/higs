import torch


def invsoftplus(x):
	"""Numerically stable inverse of softplus, 
	https://github.com/pytorch/pytorch/issues/72759#issuecomment-1236496693
	"""

	return x + torch.log(-torch.expm1(-x))	
