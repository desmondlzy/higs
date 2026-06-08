import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import gsplat
from nerfstudio.models.splatfacto import get_viewmat

from .lookat_cameras import lookat_cameras
from .transmittance_from_view import transmittance_from_view

def transmittance_from_cube(
	query_pos: torch.Tensor,
	query_dir: torch.Tensor,
	means: torch.Tensor,
	quats: torch.Tensor,
	scales: torch.Tensor,
	opacities: torch.Tensor,
	resolution: int = 128,
):
	camera = lookat_cameras(
		eye=query_pos.cpu(),
		target=query_pos.cpu() + query_dir.cpu(),
		up=torch.tensor([0.0, 1.0, 0.0]),
		camera_angle_x=np.deg2rad(90),
		img_h=resolution,
		img_w=resolution,
	)

	rotations = {
		"left": R.from_rotvec([0.0, np.pi / 2, 0.0]),
		"right": R.from_rotvec([0.0, -np.pi / 2, 0.0]),
		"top": R.from_rotvec([np.pi / 2, 0.0, 0.0]),
		"bottom": R.from_rotvec([-np.pi / 2, 0.0, 0.0]),
		"front": R.from_rotvec([0.0, 0.0, 0.0]),
		"back": R.from_rotvec([np.pi, 0.0, 0.0]),
	}

	transmittances = torch.zeros_like(opacities)

	for key, r in rotations.items():
		rot = torch.tensor(r.as_matrix()).float().cuda()
		c2w = camera.camera_to_worlds.clone().cuda()
		c2w_r = c2w[:, :3, :3] @ rot
		c2w[:, :3, :3] = c2w_r
		Ks = camera.get_intrinsics_matrices().cuda()

		rotated_viewmat = get_viewmat(c2w)

		trans, gaussian_ids = transmittance_from_view(
			means,
			quats,
			scales,
			opacities,
			rotated_viewmat,
			Ks,
			resolution,
		)

		transmittances[gaussian_ids] = torch.maximum(trans, transmittances[gaussian_ids])

	return transmittances
