from dataclasses import dataclass
from pathlib import Path

import torch
import gsplat
import tyro
from nerfstudio.models.splatfacto import SH2RGB, RGB2SH

from heatgaussian.load_2dgs_checkpoint import load_2dgs_checkpoint


def repeat_last_channel_thrice(values):
	assert values.shape[-1] == 1, f"{values.shape = }"
	repeated_values = values.repeat_interleave(3, dim=-1)
	assert repeated_values.shape == (*values.shape[:-1], 3)
	return repeated_values


def export_ply(model_2dgs, output_dir):
	for ext in ["ply", "splat"]:
		gsplat.export_splats(
			means=model_2dgs.means,
			quats=model_2dgs.quats,
			scales=model_2dgs.scales,
			opacities=model_2dgs.opacities.squeeze(),
			sh0=model_2dgs.features_dc,
			shN=model_2dgs.features_rest,
			format=ext,
			save_to=output_dir / f"geometry-rgb.{ext}",
		)

		num_heat_channels = model_2dgs.thermal_radiances.shape[-1]
		for heat_ind in range(num_heat_channels):
			gsplat.export_splats(
				means=model_2dgs.means,
				quats=model_2dgs.quats,
				scales=model_2dgs.scales,
				opacities=model_2dgs.opacities.squeeze(),
				sh0=repeat_last_channel_thrice(
					RGB2SH(torch.sigmoid(model_2dgs.thermal_radiances[..., heat_ind: heat_ind + 1]).clamp(0.0, 1.0))
				).unsqueeze(-2),
				shN=torch.zeros((len(model_2dgs.means), 3, 3), device=model_2dgs.means.device),
				format=ext,
				save_to=output_dir / f"geometry-thermal-{heat_ind:02d}.{ext}",
			)


@dataclass
class ExportPlyArgs:
	geometry_output_dir: Path
	"""Path to geometry reconstruction output directory (contains config.yml)"""

	output_dir: Path | None = None
	"""Directory to save PLY/splat files. Defaults to <geometry_output_path>/export"""


def main():
	args = tyro.cli(ExportPlyArgs)

	config_path = args.geometry_output_dir / "config.yml"
	assert config_path.exists(), f"config.yml not found in {args.geometry_output_dir}"

	output_dir = args.output_dir or args.geometry_output_dir / "export"
	output_dir.mkdir(parents=True, exist_ok=True)

	print(f"Loading checkpoint from {config_path}")
	_, pipeline, _, step = load_2dgs_checkpoint(config_path)
	model = pipeline.model
	print("Number of Gaussians:", len(model.means))

	print(f"Exporting PLY/splat files to {output_dir} (step {step})")
	export_ply(model, output_dir)
	print("Done.")


if __name__ == "__main__":
	main()