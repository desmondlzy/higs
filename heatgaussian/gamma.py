import torch
import numpy as np

def gamma(x):
	if isinstance(x, torch.Tensor):
		return torch.clamp(x, 0.0, 1.0) ** (1 / 2.2)
	elif isinstance(x, np.ndarray):
		return np.clip(x, 0.0, 1.0) ** (1 / 2.2)
	elif isinstance(x, float) or isinstance(x, int):
		return np.clip(x, 0.0, 1.0) ** (1 / 2.2)
