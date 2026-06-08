from functools import cache
import numpy as np
import torch

_indexing = "xy"

def longlat_from_xyz(x, y, z):
	"""
	range of longs: [-pi/2, pi/2]
	range of lats: [-pi, pi]
	"""
	norms = np.sqrt(x ** 2 + y ** 2 + z ** 2)
	lats = np.arctan2(y, x)
	longs = np.arcsin(z / norms)
	return longs, lats


def hemicube_longlats_front(reso):
	reso_x = reso_y = reso
	xaxis, dx = np.linspace(-1, 1, reso_x, retstep=True, endpoint=False)
	yaxis, dy = np.linspace(-1, 1, reso_y, retstep=True, endpoint=False)

	xaxis = xaxis + dx / 2
	yaxis = yaxis + dy / 2

	xx, yy = np.meshgrid(xaxis, yaxis, indexing=_indexing)
	zz = np.ones_like(xx)

	longs, lats = longlat_from_xyz(xx, yy, zz)

	return np.stack((longs, lats), axis=-1)


def hemicube_longlats_left(reso, pad=False):
	reso_y = reso
	reso_z = reso if pad else reso // 2
	z_start = -1 if pad else 0

	yaxis, dy = np.linspace(-1, 1, reso_y, endpoint=False, retstep=True)
	zaxis, dz = np.linspace(z_start, 1, reso_z, endpoint=False, retstep=True)

	yaxis = yaxis + dy / 2
	zaxis = zaxis + dz / 2

	zz, yy = np.meshgrid(zaxis, yaxis, indexing=_indexing)

	xx = np.full_like(yy, -1.0)

	longs, lats = longlat_from_xyz(xx, yy, zz)

	return np.stack((longs, lats), axis=-1)


def hemicube_longlats_up(reso, pad=False):
	reso_x = reso
	reso_z = reso if pad else reso // 2
	z_start = -1 if pad else 0

	xaxis, dx = np.linspace(-1, 1, reso_x, retstep=True, endpoint=False)
	zaxis, dz = np.linspace(z_start, 1, reso_z, endpoint=False, retstep=True)

	xaxis = xaxis + dx / 2
	zaxis = zaxis + dz / 2

	xx, zz = np.meshgrid(xaxis, zaxis, indexing=_indexing)
	yy = np.full_like(xx, -1.0)

	longs, lats = longlat_from_xyz(xx, yy, zz)
	return np.stack((longs, lats), axis=-1)

def hemicube_longlats_right(reso, pad=False):
	reso_y = reso
	reso_z = reso if pad else reso // 2 
	z_start = -1 if pad else 0

	yaxis, dy = np.linspace(-1, 1, reso_y, endpoint=False, retstep=True)
	zaxis, dz = np.linspace(1, z_start, reso_z, endpoint=False, retstep=True)

	yaxis = yaxis + dy / 2
	zaxis = zaxis + dz / 2

	zz, yy = np.meshgrid(zaxis, yaxis, indexing=_indexing)

	xx = np.full_like(yy, 1.0)

	longs, lats = longlat_from_xyz(xx, yy, zz)

	return np.stack((longs, lats), axis=-1)


def hemicube_longlats_down(reso, pad=False):
	reso_x = reso
	reso_z = reso if pad else reso // 2
	z_start = -1 if pad else 0

	xaxis, dx = np.linspace(-1, 1, reso_x, retstep=True, endpoint=False)
	zaxis, dz = np.linspace(1, z_start, reso_z, endpoint=False, retstep=True)

	xaxis = xaxis + dx / 2
	zaxis = zaxis + dz / 2

	xx, zz = np.meshgrid(xaxis, zaxis, indexing=_indexing)
	yy = np.full_like(xx, 1.0)

	longs, lats = longlat_from_xyz(xx, yy, zz)

	return np.stack((longs, lats), axis=-1)


def hemicube_longlats(reso, pad=False, return_torch=False):
	front = hemicube_longlats_front(reso)
	left = hemicube_longlats_left(reso, pad)
	right = hemicube_longlats_right(reso, pad)
	up = hemicube_longlats_up(reso, pad)
	down = hemicube_longlats_down(reso, pad)

	ret = {
		"front": front,
		"left": left,
		"right": right,
		"up": up,
		"down": down,
	}

	if return_torch:
		return {k: torch.from_numpy(v.copy()) for k, v in ret.items()}
	else:
		return ret
