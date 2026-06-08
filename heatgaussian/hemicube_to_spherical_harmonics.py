import numpy as np
import torch
from gsplat.cuda._torch_impl import _eval_sh_bases_fast

from .hemicube_longlats import hemicube_longlats
from .hemicube_solid_angles import hemicube_solid_angles
from .hemicube_xyzs import hemicube_xyzs
from .hemicube_keys import hemicube_keys


def _get_hemicube_xyzs(reso: int, pad: bool):
	xyzs = torch.from_numpy(np.vstack([
		hemicube_xyzs(reso=reso, pad=pad)[key].reshape(-1, 3)
		for key in hemicube_keys]))
	return xyzs


def hemicube_to_spherical_harmonics(hemicube_vals, num_sh_bases: int):
	assert "front" in hemicube_vals, "front hemicube is required."
	reso = hemicube_vals["front"].shape[0]
	assert hemicube_vals["front"].shape[1] == reso

	pad = "right" in hemicube_vals and hemicube_vals["right"].shape[0] == hemicube_vals["right"].shape[1]

	num_channels = hemicube_vals["front"].shape[2]

	solidangles = torch.from_numpy(
		np.vstack([
			hemicube_solid_angles(reso=reso, pad=pad)[key].reshape(-1, 1)
			for key in hemicube_keys])).reshape(-1, 1).cuda() # (n_samples, 1)	

	dirs = _get_hemicube_xyzs(reso, pad).cuda()  # (n_samples, 3)

	fvals = torch.vstack(
		[hemicube_vals[key].reshape(-1, num_channels) for key in hemicube_keys]).cuda()  # (n_samples, num_channels)

	bases =  _eval_sh_bases_fast(num_sh_bases, dirs)   # (n_samples, 25)

	# coeffs: (n_samples, num_channels)
	coeffs = torch.sum(
		fvals.unsqueeze(-2) * 
		solidangles.unsqueeze(-1) * 
		bases.unsqueeze(-1), dim=tuple(range(0, bases.ndim - 1)))
	
	# coeffs = torch.empty((bases.shape[-1], num_channels), device=fvals.device, dtype=fvals.dtype)
	# for channel in range(num_channels):
	# 	coeffs[:, channel] = torch.sum(fvals[..., channel:channel+1] * solidangles * bases, dim=tuple(range(0, bases.ndim - 1))).cuda()

	return coeffs
