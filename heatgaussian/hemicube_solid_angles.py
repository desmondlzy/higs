from functools import cache
import numpy as np
import torch

from .desbox.solid_angle import solid_angle

from .hemicube_keys import hemicube_keys


def hemicube_solid_angles_front(reso):
	reso_x = reso_y = reso
	xaxis, dx = np.linspace(-1, 1, reso_x, retstep=True, endpoint=False)
	yaxis, dy = np.linspace(-1, 1, reso_y, retstep=True, endpoint=False)

	xx, yy = np.meshgrid(xaxis, yaxis, indexing='ij')
	xdx, ydy = xx + dx, yy + dy

	zz = np.ones_like(xx)

	p1s = np.stack([xx, yy, zz], axis=-1)
	p2s = np.stack([xdx, yy, zz], axis=-1)
	p3s = np.stack([xdx, ydy, zz], axis=-1)
	p4s = np.stack([xx, ydy, zz], axis=-1)

	# compute solid angles
	ori = np.array([[[0.0, 0.0, 0.0]]])
	angles = solid_angle(p1s, p2s, p3s, ori) + solid_angle(p3s, p4s, p1s, ori)
	
	return angles


def hemicube_solid_angles_left(reso, pad):
	"""
	"""
	reso_x, reso_z = reso, reso // 2
	xaxis, dx = np.linspace(-1, 1, reso_x, endpoint=False, retstep=True)
	zaxis, dz = np.linspace(0, 1, reso_z, endpoint=False, retstep=True)
	xaxis = xaxis + dx / 2
	zaxis = zaxis + dz / 2

	xx, zz = np.meshgrid(xaxis, zaxis, indexing='ij')
	xdx, zdz = xx + dx, zz + dz

	yy = np.full_like(xx, -1.0)

	p1s = np.stack([xx, yy, zz], axis=-1)
	p2s = np.stack([xdx, yy, zz], axis=-1)
	p3s = np.stack([xdx, yy, zdz], axis=-1)
	p4s = np.stack([xx, yy, zdz], axis=-1)

	# compute solid angles
	ori = np.array([[[0.0, 0.0, 0.0]]])
	angles = solid_angle(p1s, p2s, p3s, ori) + solid_angle(p3s, p4s, p1s, ori)

	if pad:
		# pad zeros to the left of dots, make it a square
		angles = np.pad(angles, ((0, 0), (reso_x - reso_z, 0)))

	return angles
	

@cache
def hemicube_solid_angles(reso, pad=False, return_torch=False, return_array=False):
	if return_array and not pad:
		raise ValueError("pad must be True when return_array is True")

	front = hemicube_solid_angles_front(reso)
	left = hemicube_solid_angles_left(reso, pad)

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