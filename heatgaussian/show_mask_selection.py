from pathlib import Path

import torch
from torch.nn.functional import softplus
from matplotlib import pyplot as plt

from heatgaussian.render_gaussians import render_gaussians
from heatgaussian.gamma import gamma


def show_mask_selection(
	gaussians,
	selected_indices,
	mask_camera,
	mask,
	near_plane: float,
	output_path: Path,
	use_thermal: bool = False,
):
	"""Render and save a verification image showing the mask-selected Gaussians.

	Args:
		gaussians: Full Gaussian object.
		selected_indices: Boolean tensor selecting which Gaussians to show.
		mask_camera: Camera to render from.
		mask: Boolean mask image tensor (H x W) used to show the complement region.
		near_plane: Near clipping plane distance.
		output_path: File path to save the verification image (.png or .svg).
		use_thermal: If True, renders temperature (gamma-corrected) instead of RGB colors.
	"""
	from heatgaussian.heat_gaussians import HeatGaussianWithDiffuse, HeatGaussians

	# Slice all fields to the selected Gaussians
	base_kwargs = dict(
		means=gaussians.means[selected_indices].detach(),
		quats=gaussians.quats[selected_indices].detach(),
		scales=gaussians.scales[selected_indices].detach(),
		thermal_opacities=gaussians.thermal_opacities[selected_indices].detach(),
		normals=gaussians.normals[selected_indices].detach(),
		rgb_colors=gaussians.rgb_colors[selected_indices].detach(),
		specularities=gaussians.specularities[selected_indices].detach(),
		emissivities=gaussians.emissivities[selected_indices].detach(),
		temperatures=gaussians.temperatures[selected_indices].detach(),
	)
	if hasattr(gaussians, "diffuses"):
		masked_gaussians = HeatGaussianWithDiffuse(
			**base_kwargs, diffuses=gaussians.diffuses[selected_indices].detach()
		)
	else:
		masked_gaussians = HeatGaussians(**base_kwargs)

	with torch.no_grad():
		if use_thermal:
			mask_rendering = gamma(render_gaussians(
				gaussians=masked_gaussians,
				camera=mask_camera,
				colors=softplus(masked_gaussians.temperatures[:, 0:1]),
				near_plane=near_plane,
			))
		else:
			mask_rendering = render_gaussians(
				gaussians=masked_gaussians,
				camera=mask_camera,
				colors=masked_gaussians.rgb_colors,
				near_plane=near_plane,
			)

		plt.figure(figsize=(10, 5))

		plt.subplot(1, 2, 1)
		plt.title("Selected Gaussians")
		plt.imshow(mask_rendering.detach().cpu().numpy(), vmin=0)
		plt.axis("off")

		plt.subplot(1, 2, 2)
		plt.title("Complement mask")
		complement = mask_rendering * (~mask).unsqueeze(-1).to(mask_rendering.device)
		plt.imshow(complement.detach().cpu().numpy(), vmin=0, vmax=1)
		plt.axis("off")

		plt.tight_layout()
		output_path = Path(output_path)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(output_path, dpi=150)
		plt.close()
