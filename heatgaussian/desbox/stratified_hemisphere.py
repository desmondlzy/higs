import numpy as np

def stratified_hemisphere(num):
	num_x = int(np.sqrt(num))
	num_y = int(np.ceil(num / num_x))

	grid_x = np.linspace(0, 1, endpoint=False, num=num_x)
	grid_y = np.linspace(0, 1, endpoint=False, num=num_y)

	grid_pts = np.array(np.meshgrid(grid_x, grid_y)).reshape(2, -1).T
	uu = grid_pts[:, 0]
	vv = grid_pts[:, 1]
	zz = uu
	rr = np.sqrt(np.maximum(1 - zz ** 2, 0))
	phis = 2 * np.pi * vv
	cam_pos = np.stack([rr * np.cos(phis), rr * np.sin(phis), zz], axis=1)

	return cam_pos
