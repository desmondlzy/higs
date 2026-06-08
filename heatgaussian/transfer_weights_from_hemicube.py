import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from nerfstudio.models.splatfacto import get_viewmat

from .lookat_cameras import lookat_cameras
from .hemicube_cosines import hemicube_cosines
from .hemicube_solid_angles import hemicube_solid_angles
from .visibility_per_gaussian_from_cameras import visibility_per_gaussian_from_cameras


def transfer_weights_from_hemicube(
	query_point: torch.Tensor,
	query_dir: torch.Tensor,
	up: torch.Tensor,
	means: torch.Tensor,
	quats: torch.Tensor,
	scales: torch.Tensor,
	opacities: torch.Tensor,
	colors: torch.Tensor,
	normals: torch.Tensor,
	resolution: int,
	normal_filtering: bool = True,
	max_hits: int = 20,
):
	assert torch.min(opacities) > 0.0, f"forgot to sigmoid? {opacities.min() = }, {opacities.max() = }"
	assert torch.max(opacities) <= 1.0, f"forgot to sigmoid? {opacities.min() = }, {opacities.max() = }"
	assert torch.min(scales) > 0.0, f"forgot to exp? {scales.min() = }, {scales.max() = }"

	r1 = R.from_rotvec([0.0, np.pi / 2, 0.0])
	r2 = R.from_rotvec([np.pi / 2, 0.0, 0.0])
	zero = R.from_rotvec([0.0, 0.0, 0.0])

	if up is None:
		up = torch.linalg.cross(query_dir, torch.tensor([0.0, 0.0, 1.0], device=query_dir.device))
		if torch.linalg.norm(up) == 0:
			up = torch.linalg.cross(query_dir, torch.tensor([1.0, 1.0, 0.0], device=query_dir.device))
		
	camera = lookat_cameras(
		eye=query_point,
		target=query_point + query_dir,
		up=up,
		camera_angle_x=np.deg2rad(90),
		img_h=resolution,
		img_w=resolution,
	)

	Ks = camera.get_intrinsics_matrices().cuda()

	camera_rotations = {
		'front': zero,
		'left': r1,
		'up': r2,
		'right': r1.inv(),
		'down': r2.inv(),
	}

	num_cameras = len(camera_rotations)

	c2ws = {}
	for key, r in camera_rotations.items():
		rot = torch.tensor(r.as_matrix()).float().cuda()
		# rot_r = viewmat[:, :3, :3] @ rot
		c2w = camera.camera_to_worlds.clone().cuda()
		c2w_r = c2w[:, :3, :3] @ rot
		c2w[:, :3, :3] = c2w_r
		c2ws[key] = c2w

	c2ws_tensor = torch.cat([c2w for c2w in c2ws.values()], dim=0)
	assert c2ws_tensor.shape == (num_cameras, 3, 4), c2ws_tensor.shape
	hemicube_viewmats = get_viewmat(c2ws_tensor)
	hemicube_Ks = torch.tile(Ks, (num_cameras, 1, 1))

	assert hemicube_viewmats.shape == (num_cameras, 4, 4), hemicube_viewmats.shape
	assert hemicube_Ks.shape == (num_cameras, 3, 3), hemicube_Ks.shape
	visible_gauss_indices, visible_values = visibility_per_gaussian_from_cameras(
		means3d=means, 
		quats=quats, 
		scales=scales, 
		opacities=opacities,
		colors=colors,
		normals_world=normals,
		viewmats=hemicube_viewmats, 
		Ks=hemicube_Ks, 
		resolution=resolution, 
		is_hemicube=True,
		normal_filtering=normal_filtering, 
		max_hits=max_hits,
	)

	hc_cosines = hemicube_cosines(resolution, pad=True)
	hc_solid_angles = hemicube_solid_angles(resolution, pad=True)
	hc_cosines_solid_angles = torch.stack(
		[torch.tensor(hc_cosines[k] * hc_solid_angles[k], device="cuda") for k in camera_rotations.keys()],
		dim=0,
	)

	gauss_weights = hc_cosines_solid_angles.unsqueeze(-1) * visible_values

	weights_per_gauss = torch.zeros(len(means), dtype=gauss_weights.dtype, device=gauss_weights.device)
	visible_gauss_indices_flat = visible_gauss_indices.flatten()  # some values here are -1
	valid_indices = visible_gauss_indices_flat != -1
	visible_gauss_indices_flat = visible_gauss_indices_flat[valid_indices]
	gauss_weights_flat = gauss_weights.flatten()[valid_indices]

	weights_per_gauss.scatter_add_(0, visible_gauss_indices_flat, gauss_weights_flat)

	return weights_per_gauss
