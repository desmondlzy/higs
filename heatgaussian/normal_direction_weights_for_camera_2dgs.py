import math
import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splatfacto import get_viewmat
import nerfacc
import gsplat
from tqdm import tqdm


@torch.no_grad()
def normal_direction_weights_for_cameras_2dgs(
	means: torch.Tensor,
	quats: torch.Tensor,
	scales: torch.Tensor,
	opacities: torch.Tensor,
	cameras: Cameras,
	aggregation_method: str = "amax",
):
	"""
	return the two normals and transmittances for each camera. 
	Transmittances are aggregated by all the intersections' transmittance for each Gaussians/camera pair
	"""
	assert torch.min(opacities) > 0.0, f"forgot to sigmoid? {opacities.min() = }, {opacities.max() = }"
	assert torch.max(opacities) <= 1.0, f"forgot to sigmoid? {opacities.min() = }, {opacities.max() = }"
	assert torch.min(scales) > 0.0, f"forgot to exp? {scales.min() = }, {scales.max() = }"

	primary_normals = torch.zeros_like(means)
	n_cameras = len(cameras)
	n_gaussians = means.shape[0]
	primary_transmittances = torch.zeros((n_cameras, n_gaussians), device="cuda")
	secondary_transmittances = torch.zeros_like(primary_transmittances, device=primary_transmittances.device)
	
	for i, cam in tqdm(enumerate(cameras), total=n_cameras, desc="orienting normals"):
		viewmats = get_viewmat(cam.camera_to_worlds.view(-1, 3, 4)).cuda()
		Ks = cam.get_intrinsics_matrices().view(-1, 3, 3).cuda()
		image_width = cam.width
		image_height = cam.height

		tile_size = 16
		tile_width = math.ceil(image_width / float(tile_size))
		tile_height = math.ceil(image_height / float(tile_size))

		camera_ids, gaussian_ids, radii, means2d, depths, ray_transforms, normals = gsplat.fully_fused_projection_2dgs(
			means,
			quats,
			scales,
			viewmats,
			Ks, 
			width=image_width,
			height=image_height,
			packed=True,
			near_plane=0.0001,
			far_plane=1000
		)

		tiles_per_gauss, isect_ids, flatten_ids = gsplat.isect_tiles(
			means2d,
			radii,
			depths,
			tile_size=tile_size,
			tile_width=tile_width,
			tile_height=tile_height,
			sort=True,
			packed=True,
			n_cameras=1,
			camera_ids=camera_ids,
			gaussian_ids=gaussian_ids,
		)

		tile_n_bits = int(np.floor(np.log2(tile_width * tile_height)) + 1);
		cam_n_bits = int(np.floor(np.log2(1)) + 1);
		# print(tile_n_bits, cam_n_bits)
		# print(tiles_per_gauss, len(isect_ids), len(flatten_ids))
		tile_ids = (isect_ids >> 32) & ((1 << tile_n_bits) - 1)
		cam_ids = isect_ids >> (32 + tile_n_bits)
		tile_xs, tile_ys = tile_ids % tile_width + 0.5, tile_ids // tile_width + 0.5
		tile_coords = torch.stack([tile_xs.float(), tile_ys.float()], dim=-1) 

		# deltas = tile_coords - means2d[flatten_ids]

		M = ray_transforms[flatten_ids]

		h_u = -M[..., 0, :3] + M[..., 2, :3] * tile_xs[..., None]  # [M, 3]
		h_v = -M[..., 1, :3] + M[..., 2, :3] * tile_ys[..., None]  # [M, 3]
		tmp = torch.cross(h_u, h_v, dim=-1)
		us = tmp[..., 0] / tmp[..., 2]
		vs = tmp[..., 1] / tmp[..., 2]
		sigmas_3d = us**2 + vs**2  # [M]
		# sigmas_2d = 2 * (deltas[..., 0] ** 2 + deltas[..., 1] ** 2)
		sigmas_2d = torch.full_like(sigmas_3d, torch.inf)
		sigmas = 0.5 * torch.minimum(sigmas_3d, sigmas_2d)  # [M]

		alphas = torch.clamp_max(
			opacities[flatten_ids] * torch.exp(-sigmas), 0.999
		)

		indices = tile_ids
		total_tiles = tile_width * tile_height

		assert torch.all(indices < total_tiles)
		weights, transmittances = nerfacc.render_weight_from_alpha(
			alphas, ray_indices=indices, n_rays=total_tiles)

		assert gaussian_ids[flatten_ids].max() < n_gaussians, f"{gaussian_ids[flatten_ids].max() = }, {n_gaussians = }"

		weighted_transmittances = torch.scatter_reduce(
			torch.zeros((n_gaussians, ), device=weights.device),
			0,
			gaussian_ids[flatten_ids].long(),
			transmittances,
			reduce=aggregation_method,
		).to(primary_transmittances.device)

		assert weighted_transmittances.shape == (n_gaussians, ), f"{weighted_transmittances.shape = }"

		# very important... can't use camera.camera_to_worlds here! different convention!!
		c2w_rotation = torch.inverse(viewmats)[0, :3, :3]

		assert c2w_rotation.shape == (3, 3), f"{c2w_rotation.shape = }"

		render_normals = torch.zeros_like(means)
		# row-wise matrix vector product
		render_normals[gaussian_ids] = (c2w_rotation @ normals.T).T

		# fill unfilled normals with render normals
		unfilled_normals = torch.sum(primary_normals.abs(), dim=-1, keepdim=True) < 1e-4
		if unfilled_normals.any():
			primary_normals = torch.where(
				unfilled_normals,
				render_normals,
				primary_normals,
			)

		# print(f"{world_normals.shape = }, {oriented_normals.shape = }, {bigger_transmittances.shape = }")
		cosines_with_primary = torch.sum(render_normals * primary_normals, dim=-1) 
		is_primary = (cosines_with_primary > 0.0).to(primary_transmittances.device)
		reversed_primary = torch.logical_not(is_primary)

		primary_transmittances[i, is_primary] += weighted_transmittances[is_primary]
		secondary_transmittances[i, reversed_primary] += weighted_transmittances[reversed_primary]
	
	return (
		primary_normals,
		primary_transmittances, 
		-primary_normals,
		secondary_transmittances,
	)
