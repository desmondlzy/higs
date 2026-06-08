import torch
import gsplat
import math

import warnings

from heatgaussian.cpp import _heatgaussian_impl

@torch.no_grad()
def visibility_per_gaussian_from_cameras(
	means3d: torch.Tensor,
	quats: torch.Tensor,
	scales: torch.Tensor,
	colors: torch.Tensor,
	opacities: torch.Tensor,
	normals_world: torch.Tensor,
	viewmats: torch.Tensor,  # (C, 4, 4)
	Ks: torch.Tensor,
	resolution: int,
	max_hits: int = 20,
	near_plane: float = 0.02,
	is_hemicube: bool = False,
	normal_filtering: bool = True,
	returns_rgb: bool = False,
):
	image_width = resolution
	image_height = resolution
	tile_size = 16 if  resolution >= 32 else resolution // 2
	assert tile_size <= resolution // 2, f"{tile_size = }, {resolution = }"
	tile_width = math.ceil(image_width / float(tile_size))
	tile_height = math.ceil(image_height / float(tile_size))
	C = Ks.shape[0]
	N = means3d.shape[0]

	assert torch.min(opacities) > 0.0, f"forgot to sigmoid? {opacities.min() = }, {opacities.max() = }"
	assert torch.max(opacities) <= 1.0, f"forgot to sigmoid? {opacities.min() = }, {opacities.max() = }"
	assert torch.min(scales) > 0.0, f"forgot to exp? {scales.min() = }, {scales.max() = }"

	if near_plane < 0.02:
		warnings.warn(f"near_plane is set to {near_plane}, which is less than 0.02. This may cause numerical instability.")

	(
		camera_ids, 
		gaussian_ids,  
		radii, 
		means2d, 
		depths, 
		ray_transforms, 
		normals_camera_unoriented,
	) = gsplat.fully_fused_projection_2dgs(
		means3d, 
		quats, 
		scales, 
		viewmats, 
		Ks, 
		width=image_width, 
		height=image_height, 
		near_plane=near_plane,
		far_plane=1e8,
		packed=True,
	)

	_, isect_ids, flatten_ids = gsplat.isect_tiles(
		means2d, 
		radii,
		depths,
		tile_size,
		tile_width,
		tile_height,
		sort=True,
		packed=True,
		camera_ids=camera_ids,
		gaussian_ids=gaussian_ids,
		n_cameras=C,
	);

	# normals_camera_CN (C, N, 3)
	# rots = viewmats[:, None, :3, :3] # (C, 1, 3, 3)
	# assert rots.shape == (C, 1, 3, 3), rots.shape
	# normals_world = normals_world[:, :, None]  
	# assert normals_world.shape == (N, 3, 1), normals_world.shape
	# _normals_camera = torch.matmul(rots, normals_world)  # (C, N, 3, 1)
	# normals_camera_oriented = _normals_camera[camera_ids, gaussian_ids, :, 0]
	if normal_filtering:
		eyes = torch.linalg.inv(viewmats)[:, :3, 3]  # (C, 3)
		means3d_to_eyes = eyes.view(C, 1, 3) - means3d.view(1, N, 3)  # (C, N, 3)
		# assert (normals_world[None, :, :] * means3d_to_eyes).shape == (C, N, 3)
		# normal_cosines_per_cam = torch.sum(normals_world[None, :, :] * means3d_to_eyes, dim=-1)
		# assert normal_cosines_per_cam.shape == (C, N), normal_cosines_per_cam.shape

		means_3d_to_eyes_nnz = means3d_to_eyes[camera_ids, gaussian_ids, :]
		normals_world_nnz = normals_world[gaussian_ids]
		normal_cosines_per_cam_nnz = torch.sum(means_3d_to_eyes_nnz * normals_world_nnz, dim=-1)
		# assert torch.allclose(
		# 	normal_cosines_per_cam[camera_ids, gaussian_ids], 
		# 	normal_cosines_per_cam_nnz), f"{normal_cosines_per_cam[camera_ids, gaussian_ids] = }\n{normals_cosines_per_cam_nnz = }"

		gaussian_masks = normal_cosines_per_cam_nnz > 0
	else:
		gaussian_masks = None

	isect_offsets = gsplat.isect_offset_encode(
		isect_ids, 
		C,
		tile_width, 
		tile_height,
	);

	background = None

	if is_hemicube:
		tile_masks = torch.ones((C, tile_height, tile_width), dtype=torch.bool).cuda()
		tile_masks[1, :, :tile_width // 2] = False
		tile_masks[2, :tile_height // 2, :] = False
		tile_masks[3, :, tile_width // 2:] = False
		tile_masks[4, tile_height // 2:, :] = False

	if returns_rgb:
		render_results = _heatgaussian_impl.intersects_visibility_fwd_2dgs(
			means2d,
			ray_transforms,
			colors[gaussian_ids].contiguous(),
			opacities[gaussian_ids].contiguous(),
			normals_camera_unoriented,
			background,
			tile_masks,
			gaussian_masks,
			image_width,
			image_height,
			tile_size,
			isect_offsets,
			flatten_ids,
			max_hits,
		)
		assert len(render_results) == 9, f"{len(render_results) = }; if binding is changed, need to change the indexing as well"

		render_visibility_indices = render_results[-2]
		render_visibility_values = render_results[-1]
		render_rgb = render_results[0]

	else:
		render_results = _heatgaussian_impl.intersects_visibility_fwd_2dgs_skinny_outputs(
			means2d,
			ray_transforms,
			colors[gaussian_ids].contiguous(),
			opacities[gaussian_ids].contiguous(),
			normals_camera_unoriented,
			background,
			tile_masks,
			gaussian_masks,
			image_width,
			image_height,
			tile_size,
			isect_offsets,
			flatten_ids,
			max_hits,
		)

		assert len(render_results) == 2, f"{len(render_results) = }; if binding is changed, need to change the indexing as well"

		render_visibility_indices = render_results[0]
		render_visibility_values = render_results[1]
	

	assert render_visibility_indices.shape == (C, image_height, image_width, max_hits), render_visibility_indices.shape
	assert render_visibility_values.shape == (C, image_height, image_width, max_hits), render_visibility_indices.shape

	# if is_hemicube:
	# 	image_masks = torch.ones((C, image_height, image_width), dtype=torch.bool).cuda()
	# 	image_masks[1, :, :image_width // 2] = False
	# 	image_masks[2, :image_height // 2, :] = False
	# 	image_masks[3, :, image_width // 2:] = False
	# 	image_masks[4, image_height // 2:, :] = False

	# 	render_rgb[~image_masks] = 0.0
	# 	render_alpha[~image_masks] = 0.0

	render_visibility_gaussian_indices = torch.where(
		render_visibility_indices != -1,
		gaussian_ids[render_visibility_indices],
		-1,
	)

	if returns_rgb:
		return render_visibility_gaussian_indices, render_visibility_values, render_rgb

	return render_visibility_gaussian_indices, render_visibility_values
