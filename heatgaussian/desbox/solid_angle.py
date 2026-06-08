import numpy as np

def solid_angle(x, y, z, q):
	"""
	vectorizedly solid angle of triangle formed by x, y, z, view from query
	x: (..., 3)
	y: (..., 3)
	z: (..., 3)
	q: (..., 3)

	reference implementation: 
	https://github.com/libigl/libigl/blob/main/include/igl/solid_angle.cpp
	
	difference: return values is signed area on unit sphere: between [-4pi, 4pi] (or [-2pi, 2pi])
	"""
	n = x.shape[:-1]
	V = np.zeros((*n, 3, 3))
	for d in range(3):
		V[..., 0, d] = x[..., d] - q[..., d]
		V[..., 1, d] = y[..., d] - q[..., d]
		V[..., 2, d] = z[..., d] - q[..., d]
	
	vl = np.linalg.norm(V, axis=-1)

	# compute the determinant of V
	detf = np.linalg.det(V)

	# compute pairwise dot products
	dp0 = np.sum(V[..., 1, :] * V[..., 2, :], axis=-1)
	dp1 = np.sum(V[..., 2, :] * V[..., 0, :], axis=-1)
	dp2 = np.sum(V[..., 0, :] * V[..., 1, :], axis=-1)

	vl0, vl1, vl2 = vl[..., 0], vl[..., 1], vl[..., 2]
	angles = np.arctan2(
		detf,
		vl0 * vl1 * vl2 + vl0 * dp0 + vl1 * dp1 + vl2 * dp2
	) * 2

	return angles
