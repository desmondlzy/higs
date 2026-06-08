import torch
import gsplat
import numpy as np

def prepare_intersections_and_alphas_from_camera(
	means: torch.Tensor,
	quats: torch.Tensor,
	scales: torch.Tensor,
	opacities: torch.Tensor,
	viewmats: torch.Tensor,
	Ks: torch.Tensor,
	resolution: int,
):
	"""
	batch on camera. return alphas, flatten_ids (intersection ids), camera_ids, gaussian_ids, tile_ids, total_tiles

	to bring per-gaussian values to per-intersection values:
	per_intersection_quant = per_gaussian_quant[guassian_ids][flatten_ids]
	"""
	image_height = image_width = resolution
	tile_width, tile_height = image_width, image_height
	camera_ids, gaussian_ids, radii, means2d, depths, ray_transforms, normals = gsplat.fully_fused_projection_2dgs(
		means,
		quats,
		scales,
		viewmats,
		Ks, 
		width=image_width,
		height=image_height,
		packed=True,
	)

	n_cameras = viewmats.shape[0]
	assert n_cameras == Ks.shape[0]

	tiles_per_gauss, isect_ids, flatten_ids = gsplat.isect_tiles(
		means2d,
		radii,
		depths,
		tile_size=1,
		tile_width=tile_width,
		tile_height=tile_height,
		sort=True,
		packed=True,
		n_cameras=n_cameras,
		camera_ids=camera_ids,
		gaussian_ids=gaussian_ids,
	)

	tile_n_bits = int(np.floor(np.log2(tile_width * tile_height)) + 1);
	# cam_n_bits = int(np.floor(np.log2(n_cameras)) + 1);

	tile_ids = (isect_ids >> 32) & ((1 << tile_n_bits) - 1)
	# cam_ids = isect_ids >> (32 + tile_n_bits)
	tile_xs, tile_ys = tile_ids % tile_width + 0.5, tile_ids // tile_width + 0.5
	tile_coords = torch.stack([tile_xs.float(), tile_ys.float()], dim=-1)

	deltas = tile_coords - means2d[flatten_ids]

	M = ray_transforms[flatten_ids]

	h_u = -M[..., 0, :3] + M[..., 2, :3] * tile_xs[..., None]  # [M, 3]
	h_v = -M[..., 1, :3] + M[..., 2, :3] * tile_ys[..., None]  # [M, 3]
	tmp = torch.cross(h_u, h_v, dim=-1)
	us = tmp[..., 0] / tmp[..., 2]
	vs = tmp[..., 1] / tmp[..., 2]
	sigmas_3d = us**2 + vs**2  # [M]
	sigmas_2d = 2 * (deltas[..., 0] ** 2 + deltas[..., 1] ** 2)
	sigmas = 0.5 * torch.minimum(sigmas_3d, sigmas_2d)  # [M]

	opacities_per_isect = opacities[gaussian_ids][flatten_ids]

	alphas = torch.clamp_max(
		opacities_per_isect * torch.exp(-sigmas), 0.999
	)

	total_tiles = tile_width * tile_height

	return alphas, flatten_ids, camera_ids, gaussian_ids, tile_ids, total_tiles
