import numpy as np

from .hemicube_longlats import hemicube_longlats
from .hemicube_keys import hemicube_keys

def hemicube_xyzs(reso: int, pad: bool = True):
	hc_longlats = hemicube_longlats(reso=reso, pad=pad)

	res = {}
	for key in hemicube_keys:
		longlats = hc_longlats[key]
		longs, lats = longlats[..., 0] + np.pi / 2, longlats[..., 1]

		xs = np.sin(longs) * np.cos(lats)
		ys = np.sin(longs) * np.sin(lats)
		zs = -np.cos(longs)

		xyzs = np.stack([xs, ys, zs], axis=-1)
		res[key] = xyzs

	return res
