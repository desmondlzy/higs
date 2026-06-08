import torch
import numpy as np
import gsplat
import nerfacc

def transmittance_from_view(
	means: torch.Tensor,
	quats: torch.Tensor,
	scales: torch.Tensor,
	opacities: torch.Tensor,
	viewmats: torch.Tensor,
	Ks: torch.Tensor,
	resolution: int,
):
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

	tiles_per_gauss, isect_ids, flatten_ids = gsplat.isect_tiles(
		means2d,
		radii,
		depths,
		tile_size=1,
		tile_width=tile_width,
		tile_height=tile_height,
		sort=True,
		packed=True,
		n_cameras=1,
		camera_ids=camera_ids,
		gaussian_ids=gaussian_ids,
	)

	if flatten_ids.numel() == 0:
		empty_indices = torch.zeros((0, ), dtype=torch.long, device=means.device)
		empty_values = torch.zeros((0, ), dtype=torch.float, device=means.device)
		return empty_values, empty_indices

	tile_n_bits = int(np.floor(np.log2(tile_width * tile_height)) + 1);
	cam_n_bits = int(np.floor(np.log2(1)) + 1)
	# print(tile_n_bits, cam_n_bits)
	# print(tiles_per_gauss, len(isect_ids), len(flatten_ids))
	tile_ids = (isect_ids >> 32) & ((1 << tile_n_bits) - 1)
	cam_ids = isect_ids >> (32 + tile_n_bits)
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

	alphas = torch.clamp_max(
		opacities[flatten_ids] * torch.exp(-sigmas), 0.999
	)

	indices = tile_ids
	total_tiles = tile_width * tile_height

	assert torch.all(indices < total_tiles)
	weights, transmittances = nerfacc.render_weight_from_alpha(
		alphas, ray_indices=indices, n_rays=total_tiles)

	# visibility = nerfacc.render_visibility_from_alpha(
	# 	alphas, ray_indices=indices, n_rays=total_tiles, 
	# 	early_stop_eps=0.001, alpha_thre=0.001)

	# visible_alphas = alphas[visibility]
	# visible_weights, visible_transmittances = nerfacc.render_weight_from_alpha(
	# 	visible_alphas, ray_indices=indices[visibility], n_rays=total_tiles)


	normalization_factors = torch.scatter_add(
		torch.zeros((torch.max(flatten_ids) + 1, ), device=weights.device),
		0,
		flatten_ids.long(),
		alphas,
	)

	# normalized_weights = weights / normalization_factors
	normalization_factors = torch.clamp_min(normalization_factors, 0.001)
	normalized_weights = weights / normalization_factors[flatten_ids]

	weighted_transmittances = torch.scatter_add(
		torch.zeros((torch.max(flatten_ids) + 1, ), device=weights.device),
		0,
		flatten_ids.long(),
		transmittances * normalized_weights,
	)

	return weighted_transmittances, gaussian_ids

