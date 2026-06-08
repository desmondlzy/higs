from typing import Dict, Literal, Optional, Tuple

import torch
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from torch import Tensor

from gsplat.utils import depth_to_normal, get_projection_matrix
from gsplat.cuda._wrapper import (
	fully_fused_projection_2dgs,
	isect_offset_encode,
	isect_tiles,
	rasterize_to_pixels_2dgs,
	spherical_harmonics,
)

from nerfstudio.models.splatfacto import get_viewmat

from .lookat_cameras import lookat_cameras


def _rasterization_2dgs(
	means: Tensor,
	quats: Tensor,
	scales: Tensor,
	opacities: Tensor,
	colors: Tensor,
	viewmats: Tensor,
	Ks: Tensor,
	width: int,
	height: int,
	near_plane: float = 0.01,
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
			normals,
		) = proj_results
		opacities = opacities[gaussian_ids]
	else:
		radii, means2d, depths, ray_transforms, normals = proj_results
		opacities = opacities.repeat(C, 1)
		camera_ids, gaussian_ids = None, None

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
		camtoworlds = torch.inverse(viewmats)
		if packed:
			dirs = means[gaussian_ids, :] - camtoworlds[camera_ids, :3, 3]
		else:
			dirs = means[None, :, :] - camtoworlds[:, None, :3, 3]
		colors = spherical_harmonics(
			sh_degree, dirs, colors, masks=radii > 0
		)  # [nnz, D] or [C, N, 3]
		# make it apple-to-apple with Inria's CUDA Backend.
		colors = torch.clamp_min(colors + 0.5, 0.0)

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
		normals,
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
		"normals": normals,
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

	return (
		render_colors,
		render_alphas,
		# render_normals,
		# render_normals_from_depth,
		render_distort,
		render_median,
		meta,
	)


_z = torch.tensor([0.0, 0.0, 1.0], device="cuda")
_xy = torch.tensor([1.0, 1.0, 0.0], device="cuda")
@torch.compile
def _get_up_vector(query_dir: torch.Tensor) -> torch.Tensor:
	up = torch.linalg.cross(query_dir, _z)
	if torch.linalg.norm(up) == 0:
		up = torch.linalg.cross(query_dir, _xy)
	return up


def rasterize_2dgs_hemicube(
	query_pos: torch.Tensor,
	query_dir: torch.Tensor,
	means: torch.Tensor,
	quats: torch.Tensor,
	scales: torch.Tensor,
	opacities: torch.Tensor,
	colors: torch.Tensor,
	resolution: int = 128,	
	near_plane: float = 0.01,
):
	r1 = R.from_rotvec([0.0, np.pi / 2, 0.0])
	r2 = R.from_rotvec([np.pi / 2, 0.0, 0.0])
	zero = R.from_rotvec([0.0, 0.0, 0.0])

	eye = query_pos
	target = eye + query_dir
	up = _get_up_vector(query_dir)

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

	renders = {}


	viewmats = []
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
		viewmats.append(rotated_viewmat[0])
	

	color_channels = colors.shape[-1]
	render_rgbs, *_, meta = _rasterization_2dgs(
		means,
		quats,
		scales,
		opacities,
		colors,
		torch.stack(viewmats, dim=0),
		torch.tile(Ks, (len(camera_rotations), 1, 1)),
		width=resolution,
		height=resolution,
		near_plane=near_plane,
		packed=True
	)
	assert render_rgbs.shape == (len(camera_rotations), resolution, resolution, color_channels)

	for key, render_rgb in zip(camera_rotations.keys(), render_rgbs):
		assert render_rgb.shape == (resolution, resolution, color_channels)
		renders[key] = render_rgb
		half = resolution // 2
		match key:
			case 'front':
				renders[key] = render_rgb
			case 'left':
				renders[key] = render_rgb[:, half:]
			case 'up':
				renders[key] = render_rgb[half:, :]
			case 'right':
				renders[key] = render_rgb[:, :half]
			case 'down':
				renders[key] = render_rgb[:half, :]
			case _:
				raise ValueError(f"Invalid key {key} for hemicube")
		
	return renders
