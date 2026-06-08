import torch
import nerfacc

from .prepare_intersections_and_alphas_from_cameras import prepare_intersections_and_alphas_from_cameras

def visibility_from_cameras(
	means: torch.Tensor,
	quats: torch.Tensor,
	scales: torch.Tensor,
	opacities: torch.Tensor,
	viewmats: torch.Tensor,
	Ks: torch.Tensor,
	resolution: int,
):
	"""
	"""

	assert torch.min(opacities) > 0.0, f"forgot to sigmoid? {opacities.min() = }, {opacities.max() = }"
	assert torch.max(opacities) <= 1.0, f"forgot to sigmoid? {opacities.min() = }, {opacities.max() = }"
	assert torch.min(scales) > 0.0, f"forgot to exp? {scales.min() = }, {scales.max() = }"

	alphas, flatten_ids, indices, total_tiles = prepare_intersections_and_alphas_from_cameras(
		means, quats, scales, opacities, viewmats, Ks, resolution)
	
	visibility = nerfacc.render_visibility_from_alpha(
		alphas, ray_indices=indices, n_rays=total_tiles,
		early_stop_eps=0.001, alpha_thre=0.001)

	per_gaussian_visibility = torch.scatter_add(
		torch.zeros((torch.max(flatten_ids) + 1, ), device=visibility.device, dtype=torch.bool),
		0,
		flatten_ids.long(),
		visibility,
	)

	return per_gaussian_visibility
	