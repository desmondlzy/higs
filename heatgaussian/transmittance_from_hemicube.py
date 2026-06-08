import torch 
import numpy as np
from scipy.spatial.transform import Rotation as R

from nerfstudio.models.splatfacto import get_viewmat

from .transmittance_from_view import transmittance_from_view
from .lookat_cameras import lookat_cameras

def transmittance_from_hemicube(
	query_pos: torch.Tensor,
	query_dir: torch.Tensor,
	means: torch.Tensor,
	quats: torch.Tensor,
	scales: torch.Tensor,
	opacities: torch.Tensor,
	resolution: int = 128,	
):
	r1 = R.from_rotvec([0.0, np.pi / 2, 0.0])
	r2 = R.from_rotvec([np.pi / 2, 0.0, 0.0])
	zero = R.from_rotvec([0.0, 0.0, 0.0])

	eye = query_pos.cpu()
	target = eye + query_dir.cpu()
	up = torch.linalg.cross(query_dir.cpu(), torch.tensor([0.0, 0.0, 1.0]))

	camera = lookat_cameras(
		eye=eye,
		target=target,
		up=up,
		camera_angle_x=np.deg2rad(90),
		img_h=resolution,
		img_w=resolution,
	)

	Ks = camera.get_intrinsics_matrices().cuda()

	camera_rotations = {
		'left': r1,
		'up': r2,
		'right': r1.inv(),
		'down': r2.inv(),
		'front': zero,
	}


	transmittances = torch.zeros_like(means[:, 0])

	for key, r in camera_rotations.items():
		rot = torch.tensor(r.as_matrix()).float().cuda()
		# rot_r = viewmat[:, :3, :3] @ rot
		c2w = camera.camera_to_worlds.clone().cuda()
		c2w_r = c2w[:, :3, :3] @ rot
		c2w[:, :3, :3] = c2w_r

		rotated_left = c2w[0, :3, 0]
		rotated_up   = c2w[0, :3, 1]
		# rotated_look = c2w[0, :3, 2]
		eye = c2w[0, :3, 3].cuda()

		rotated_viewmat = get_viewmat(c2w)

		match key:
			case 'front':
				within = torch.ones_like(means[:, 0], dtype=torch.bool)
			case 'left':
				within = (means - eye) @ rotated_left > 0
			case 'up':
				within = (means - eye) @ rotated_up < 0
			case 'right':
				within = (means - eye) @ rotated_left < 0
			case 'down':
				within = (means - eye) @ rotated_up > 0


		trans, gaussian_ids = transmittance_from_view(
			means[within],
			quats[within],
			scales[within],
			opacities[within],
			rotated_viewmat,
			Ks,
			resolution,
		)

		transmittances[gaussian_ids] = torch.maximum(trans, transmittances[gaussian_ids])
	
	return transmittances
