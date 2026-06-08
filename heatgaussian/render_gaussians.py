"""Render Gaussians with normal filtering."""

import torch
from nerfstudio.models.splatfacto import get_viewmat
from nerfstudio.cameras.cameras import Cameras

from heatgaussian.rasterize_2dgs_normal_filtering import rasterize_2dgs_normal_filtering


def render_gaussians(
	gaussians,
	camera: Cameras,
	colors: torch.Tensor,
	near_plane: float,
	normal_occluding: bool = False,
	normal_transparent: bool = False,
):
	"""
	Render Gaussians using 2DGS with normal filtering.

	Args:
		gaussians: Gaussian primitives with means, quats, scales, thermal_opacities, normals
		camera: Camera to render from
		colors: Per-Gaussian colors to render
		near_plane: Near clipping plane distance
		normal_occluding: Whether to use normal-based occlusion
		normal_transparent: Whether to use normal-based transparency

	Returns:
		Rendered image tensor of shape (H, W, C)
	"""
	# Handle both single camera and batched camera
	if camera.camera_to_worlds.ndim == 2:
		# Single camera: shape (3, 4) -> add batch dim -> (1, 3, 4)
		viewmats = get_viewmat(camera.camera_to_worlds.unsqueeze(0).cuda())
		Ks = camera.get_intrinsics_matrices().unsqueeze(0).cuda()
	else:
		# Batched camera: shape (N, 3, 4)
		viewmats = get_viewmat(camera.camera_to_worlds.cuda())
		Ks = camera.get_intrinsics_matrices().cuda()

	rendering, *_ = rasterize_2dgs_normal_filtering(
		means=gaussians.means,
		quats=gaussians.quats,
		scales=torch.exp(gaussians.scales),
		opacities=torch.sigmoid(gaussians.thermal_opacities).squeeze(),
		colors=colors,
		normals_world=gaussians.normals,
		viewmats=viewmats,
		Ks=Ks,
		width=camera.width,
		height=camera.height,
		normal_occluding=normal_occluding,
		normal_transparent=normal_transparent,
		near_plane=near_plane,
	)

	# Return single image if single camera, otherwise return batch
	if rendering.shape[0] == 1:
		return rendering[0]
	return rendering
