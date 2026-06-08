import numpy as np
from .utils import quaternion_from_between_two_vectors

from .utils import set_seed; set_seed(0)
# radius = 1
# t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
# rr = (np.cos(t * 6) + 1) / 2 * 0.95
# xx = rr * np.cos(t)
# yy = rr * np.sin(t)
# zz = np.sqrt(np.maximum(radius ** 2 - xx ** 2 - yy ** 2, 0.0))
# cam_pos = np.stack([xx, yy, zz], axis=1)

def stratified_hemisphere(num, jitter=0.0):
	num_x = int(np.sqrt(num))
	num_y = int(np.ceil(num / num_x))

	grid_x, step_x = np.linspace(0, 1, endpoint=False, num=num_x, retstep=True)
	jitter_x = np.random.uniform(-jitter, jitter, num_x)
	grid_y, step_y = np.linspace(0, 1, endpoint=False, num=num_y, retstep=True)
	jitter_y = np.random.uniform(-jitter, jitter, num_y)

	grid_x += step_x * jitter_x
	grid_y += step_y * jitter_y

	grid_pts = np.array(np.meshgrid(grid_x, grid_y)).reshape(2, -1).T
	uu = grid_pts[:, 0]
	vv = grid_pts[:, 1]
	zz = uu
	rr = np.sqrt(np.maximum(1 - zz ** 2, 0))
	phis = 2 * np.pi * vv
	cam_pos = np.stack([rr * np.cos(phis), rr * np.sin(phis), zz], axis=1)

	return cam_pos



def make_subset(name, heat_data, jitter):
	assert jitter <= 1
	cam_pos = stratified_hemisphere(100, jitter)
	if np.isnan(cam_pos).any():
		raise ValueError("NaN in cam_pos")

	return {
		'name': name,
		'cam_radius': 5,
		'pose_dist_config': {
			'module': 'distribution.Constant',
			'constants': cam_pos.tolist()
		},
		'parameter_dist_config': {
			'module': 'distribution.Constant',
			'constants': [heat_data for _ in range(len(cam_pos))]
		},
	}

n = 3
using_first_n = 3
assert using_first_n <= n

# data_range = np.linspace(260, 330, n)
data_range = [
	[400],
	[450],
	[500],
]

config = {
	'compute_device': 'CUDA',                              # Device used for ray tracing, can be 'OPTIX', 'CUDA' or 'CPU'
	'seed': 0,
	'subsets': [                                            # List of dataset subsets to create
		make_subset(
			f"heat-{i:03d}", 
			[*data_range[i]],
			jitter=0.25
		) for i in range(using_first_n)
	],
	'resolution': 256,
	'samples': 256,
	'stardis_samples': 64,
	'range_low': 250,
	'range_high': 500,
	'light': 'Directional',                                 # Light source
	'template': 'templates/teapot2_bunny_table_ro',
	'pose_file_prefix': 'transforms_',                      # Prefix of the pose file
	'pose_file_save_interval': 10,                          # Number of generated samples after whicht to save poses
	'target_path': f'datasets/teapot2_bunny_table_ro_spp64'              # Path where the generated files are stored to
}
