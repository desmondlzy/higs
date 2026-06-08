import torch
import numpy as np

from heatgaussian.hemicube_xyzs import hemicube_xyzs
from gsplat.cuda._torch_impl import _eval_sh_bases_fast

def spherical_harmonics_to_hemicube(shs: torch.Tensor, resolution: int):
	assert shs.ndim == 2, f"{shs.ndim = }"
	num_sh_bases, num_channels = shs.shape

	xyzs = torch.from_numpy(
		np.stack(list(hemicube_xyzs(resolution, pad=True).values()), axis=0)).double().cuda()   # (n_samples, 3)
	bases = _eval_sh_bases_fast(num_sh_bases, xyzs)

	images = torch.einsum("sc,...s->...c", shs, bases)
	assert images.shape == (5, resolution, resolution, num_channels)

	images[1, :, :resolution // 2, :] = 0.0
	images[2, :resolution // 2, :, :] = 0.0
	images[3, :, resolution // 2:, :] = 0.0
	images[4, resolution // 2:, :, :] = 0.0

	return images
