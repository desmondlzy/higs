import torch
import numpy as np

from functools import cache

from .hemicube_keys import hemicube_keys


def hemicube_cosines_left(reso, pad):
	"""
	compute cosine terms (w dot n) for each pixel in a hemicube
	using the center of each pixel as the direction vector
	"""
	reso_x, reso_z = reso, reso // 2
	xaxis, dx = np.linspace(-1, 1, reso_x, endpoint=False, retstep=True)
	zaxis, dz = np.linspace(0, 1, reso_z, endpoint=False, retstep=True)
	xaxis = xaxis + dx / 2
	zaxis = zaxis + dz / 2

	xx, zz = np.meshgrid(xaxis, zaxis, indexing='ij')

	norms = np.sqrt(xx ** 2 + 1 ** 2 + zz ** 2)

	nn = np.array([0.0, 0.0, 1.0])

	dots = (
		# xx * nn[0] + 
		# yy * nn[1] + 
		zz * nn[2]) / norms
	
	if pad:
		# pad zeros to the left of dots, make it a square
		dots = np.pad(dots, ((0, 0), (reso_x - reso_z, 0)))
	
	return dots

def hemicube_cosines_front(reso):
	reso_x = reso_y = reso
	xaxis, dx = np.linspace(-1, 1, reso_x, retstep=True, endpoint=False)
	yaxis, dy = np.linspace(-1, 1, reso_y, retstep=True, endpoint=False)
	xaxis = xaxis + dx / 2
	yaxis = yaxis + dy / 2

	xx, yy = np.meshgrid(xaxis, yaxis, indexing='ij')

	norms = np.sqrt(xx ** 2 + yy ** 2 + 1 ** 2)

	# nn = [0.0, 0.0, 1.0]
	# zz = 1.0
	# dots = (xx * nn[0] + yy * nn[1] + zz * nn[2]) / norms
	# simplified to:
	dots = np.ones_like(xx) / norms

	return dots

@cache
def hemicube_cosines(reso, pad=False, return_torch=False, return_array=False):
	"""
	"""
	if return_array and not pad:
		raise ValueError("pad must be True when return_array is True")

	front = hemicube_cosines_front(reso)
	left = hemicube_cosines_left(reso, pad)

	ret = {
		"front": front,
		"left": left,
		"right": np.fliplr(left),
		"up": np.rot90(left, k=3),
		"down": np.rot90(left, k=1),
	}

	if return_array:
		return np.stack([ret[key] for key in hemicube_keys], axis=0)

	if return_torch:
		return {k: torch.from_numpy(v.copy()) for k, v in ret.items()}
	else:
		return ret
