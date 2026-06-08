from torch import Tensor
from typing import Dict, Literal, Optional, Tuple

import torch
import math

from gsplat.utils import depth_to_normal, get_projection_matrix
from gsplat.cuda._wrapper import (
	fully_fused_projection_2dgs,
	isect_offset_encode,
	isect_tiles,
	spherical_harmonics, 
	rasterize_to_pixels_2dgs,
)


def rasterize_2dgs_normal_filtering(
	means: Tensor,
	quats: Tensor,
	scales: Tensor,
	opacities: Tensor,
	colors: Tensor,
	normals_world: Tensor | None,
	viewmats: Tensor,
	Ks: Tensor,
	width: int,
	height: int,
	near_plane: float = 1e-5,
	far_plane: float = 1e10,
	radius_clip: float = 0.0,
	eps2d: float = 0.3,
	sh_degree: Optional[int] = None,
	packed: bool = False,
	tile_size: int = 16,
	backgrounds: Optional[Tensor] = None,
	render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
	sparse_grad: bool = False,
	absgrad: bool = False,
	distloss: bool = False,
	depth_mode: Literal["expected", "median"] = "expected",
	normal_occluding: bool = False,
	normal_transparent: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
	"""
	rasterize 2dgs. the same as the gsplat func other than no 2dgs gradient (which seems causing memory leak)
	"""
	N = means.shape[0]
	C = viewmats.shape[0]
	assert means.shape == (N, 3), means.shape
	assert quats.shape == (N, 4), quats.shape
	assert scales.shape == (N, 3), scales.shape
	assert opacities.shape == (N,), opacities.shape
	assert normals_world is None or normals_world.shape == (N, 3), normals_world.shape
	assert viewmats.shape == (C, 4, 4), viewmats.shape
	assert Ks.shape == (C, 3, 3), Ks.shape
	assert render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED"], render_mode
	if distloss:
		assert render_mode in [
			"D",
			"ED",
			"RGB+D",
			"RGB+ED",
		], f"distloss requires depth rendering, render_mode should be D, ED, RGB+D, RGB+ED, but got {render_mode}"

	if sh_degree is None:
		# treat colors as post-activation values
		# colors should be in shape [N, D] or (C, N, D) (silently support)
		assert (colors.dim() == 2 and colors.shape[0] == N) or (
			colors.dim() == 3 and colors.shape[:2] == (C, N)
		), colors.shape
	else:
		# treat colors as SH coefficients. Allowing for activating partial SH bands
		assert (
			colors.dim() == 3 and colors.shape[0] == N and colors.shape[2] == 3
		), colors.shape
		assert (sh_degree + 1) ** 2 <= colors.shape[1], colors.shape

	# Compute Ray-Splat intersection transformation.
	proj_results = fully_fused_projection_2dgs(
		means,
		quats,
		scales,
		viewmats,
		Ks,
		width,
		height,
		eps2d,
		near_plane,
		far_plane,
		radius_clip,
		packed,
		sparse_grad,
	)

	if packed:
		(
			camera_ids,
			gaussian_ids,
			radii,
			means2d,
			depths,
			ray_transforms,
			normals_camera,
		) = proj_results
		opacities = opacities[gaussian_ids]
	else:
		radii, means2d, depths, ray_transforms, normals_camera = proj_results
		opacities = opacities.repeat(C, 1)
		camera_ids, gaussian_ids = None, None

	# requires grad = False here, otherwise causing cuda memory leak
	densify = torch.zeros_like(
		means2d, dtype=means.dtype, requires_grad=False, device="cuda"
	)
	# Identify intersecting tiles
	tile_width = math.ceil(width / float(tile_size))
	tile_height = math.ceil(height / float(tile_size))
	tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
		means2d,
		radii,
		depths,
		tile_size,
		tile_width,
		tile_height,
		packed=packed,
		n_cameras=C,
		camera_ids=camera_ids,
		gaussian_ids=gaussian_ids,
	)
	isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

	camtoworlds = torch.linalg.inv(viewmats)
	camera_eyes = camtoworlds[:, :3, 3]  # [C, 3]
	means3d_to_eyes = camera_eyes.view(C, 1, 3) - means.view(1, N, 3)  # [C, N, 3]

	if not (
		colors.dim() == 3 and sh_degree is None
	):  # silently support [C, N, D] color.
		colors = (
			colors[gaussian_ids] if packed else colors.expand(C, *([-1] * colors.dim()))
		)  # [nnz, D] or [C, N, 3]
	else:
		if packed:
			colors = colors[camera_ids, gaussian_ids, :]
	if sh_degree is not None:  # SH coefficients
		if packed:
			dirs = means[gaussian_ids, :] - camtoworlds[camera_ids, :3, 3]
		else:
			dirs = means[None, :, :] - camtoworlds[:, None, :3, 3]
		colors = spherical_harmonics(
			sh_degree, dirs, colors, masks=radii > 0
		)  # [nnz, D] or [C, N, 3]
		# make it apple-to-apple with Inria's CUDA Backend.
		colors = torch.clamp_min(colors + 0.5, 0.0)
	
	if normal_occluding or normal_transparent:
		assert normals_world is not None, "normals_world should be provided for normal filtering"
		assert normals_world.shape == (N, 3), normals_world.shape
		# if gaussians facing away from the camera, mask them out
		# cosines > 0, facing camera
		if packed:
			means3d_to_eyes_nnz = means3d_to_eyes[camera_ids, gaussian_ids, :]
			normals_nnz = normals_world[gaussian_ids]
			normal_cosines_per_cam_nnz = torch.sum(means3d_to_eyes_nnz * normals_nnz, dim=-1)
			gaussian_masks = normal_cosines_per_cam_nnz > 0
		else:
			normal_cosines = torch.sum(means3d_to_eyes * normals_world.view(1, N, 3), dim=-1)
			gaussian_masks = normal_cosines > 0

		# mask out the color contribution -- so will get darker image
		if normal_occluding:
			colors = torch.where( 
				gaussian_masks.unsqueeze(-1), 
				colors, 
				torch.zeros_like(colors))

		# prevent the gaussians from being rendered, zero 
		if normal_transparent:
			assert C == 1, f"normal_transparent only supports one camera at a time, got {C}, otherwise might conflict"
			opacities = opacities * gaussian_masks.float()
		


	# Rasterize to pixels
	if render_mode in ["RGB+D", "RGB+ED"]:
		colors = torch.cat((colors, depths[..., None]), dim=-1)
		# backgrounds = torch.cat((backgrounds, torch.zeros((C, 1), device="cuda")), dim=-1)
	elif render_mode in ["D", "ED"]:
		colors = depths[..., None]
	else:  # RGB
		pass
	
	(
		render_colors,
		render_alphas,
		render_normals,
		render_distort,
		render_median,
	) = rasterize_to_pixels_2dgs(
		means2d,
		ray_transforms,
		colors,
		opacities,
		normals_camera,
		densify,
		width,
		height,
		tile_size,
		isect_offsets,
		flatten_ids,
		backgrounds=backgrounds,
		packed=packed,
		absgrad=absgrad,
		distloss=distloss,
	)
	render_normals_from_depth = None
	if render_mode in ["ED", "RGB+ED"]:
		# normalize the accumulated depth to get the expected depth
		render_colors = torch.cat(
			[
				render_colors[..., :-1],
				render_colors[..., -1:] / render_alphas.clamp(min=1e-10),
			],
			dim=-1,
		)
	if render_mode in ["RGB+ED", "RGB+D"]:
		# render_depths = render_colors[..., -1:]
		if depth_mode == "expected":
			depth_for_normal = render_colors[..., -1:]
		elif depth_mode == "median":
			depth_for_normal = render_median

		render_normals_from_depth = depth_to_normal(
			depth_for_normal, torch.linalg.inv(viewmats), Ks
		).squeeze(0)

	meta = {
		"camera_ids": camera_ids,
		"gaussian_ids": gaussian_ids,
		"radii": radii,
		"means2d": means2d,
		"depths": depths,
		"ray_transforms": ray_transforms,
		"opacities": opacities,
		"normals": normals_camera,
		"tile_width": tile_width,
		"tile_height": tile_height,
		"tiles_per_gauss": tiles_per_gauss,
		"isect_ids": isect_ids,
		"flatten_ids": flatten_ids,
		"isect_offsets": isect_offsets,
		"width": width,
		"height": height,
		"tile_size": tile_size,
		"n_cameras": C,
		"render_distort": render_distort,
	}

	# render_normals = torch.einsum("...ij,...hwj->...hwi", torch.linalg.inv(viewmats)[..., :3, :3], render_normals)

	return (
		render_colors,
		render_alphas,
		render_normals,
		render_normals_from_depth,
		render_distort,
		render_median,
		meta,
	)