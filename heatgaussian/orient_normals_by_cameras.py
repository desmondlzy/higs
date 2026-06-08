import math

import torch
import numpy as np
import gsplat
import nerfacc

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splatfacto import get_viewmat


@torch.no_grad()
def orient_normals_by_cameras(
	means: torch.Tensor,
	quats: torch.Tensor,
	scales: torch.Tensor,
	opacities: torch.Tensor,
	cameras: Cameras,
) -> torch.Tensor:

	assert torch.min(opacities) > 0.0, f"forgot to sigmoid? {opacities.min() = }, {opacities.max() = }"
	assert torch.max(opacities) <= 1.0, f"forgot to sigmoid? {opacities.min() = }, {opacities.max() = }"
	assert torch.min(scales) > 0.0, f"forgot to exp? {scales.min() = }, {scales.max() = }"

	oriented_normals = torch.zeros_like(means)
	max_transmittances = torch.zeros_like(opacities)
	
	n_gaussians = means.shape[0]

	for i, cam in enumerate(cameras):
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

		assert gaussian_ids[flatten_ids].max() < n_gaussians, f"{gaussian_ids[flatten_ids].max() = }, {n_gaussians = }"

		weighted_transmittances = torch.scatter_add(
			torch.zeros((n_gaussians, ), device=weights.device),
			0,
			gaussian_ids[flatten_ids].long(),
			transmittances,
		)

		bigger_transmittances = weighted_transmittances > max_transmittances
		assert bigger_transmittances.shape == (n_gaussians, ), f"{bigger_transmittances.shape = }"

		# print("cuda memory consumption in GB", torch.cuda.memory_allocated() / 1e9)

		if torch.any(bigger_transmittances):
			# very important... can't use camera.camera_to_worlds here! different convention!!
			c2w_rotation = torch.inverse(viewmats)[0, :3, :3]

			assert c2w_rotation.shape == (3, 3), f"{c2w_rotation.shape = }"

			world_normals = torch.zeros_like(means)

			# row-wise matrix vector product
			world_normals[gaussian_ids] = (c2w_rotation @ normals.T).T

			# print(f"{world_normals.shape = }, {oriented_normals.shape = }, {bigger_transmittances.shape = }")

			oriented_normals = torch.where(
				bigger_transmittances[:, None],
				world_normals,
				oriented_normals,
			)
			max_transmittances[bigger_transmittances] = weighted_transmittances[bigger_transmittances]

	
	return oriented_normals, max_transmittances


@torch.no_grad()
def orient_model_normals_using_rgb_opacities_by_cameras(model, cameras: Cameras):
	return orient_normals_by_cameras(
		model.means,
		model.quats,
		torch.exp(model.scales),
		torch.sigmoid(model.opacities),
		cameras,
	)


@torch.no_grad()
def orient_model_normals_using_thermal_opacities_by_cameras(model, cameras: Cameras):
	return orient_normals_by_cameras(
		model.means,
		model.quats,
		torch.exp(model.scales),
		torch.sigmoid(model.thermal_opacities),
		cameras,
	)
