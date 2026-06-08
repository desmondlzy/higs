#!/usr/bin/env python3
"""
Evaluation script for novel temperature conditions.

This script loads a trained HeatGaussian model and evaluates it on
novel temperature conditions by modifying selected Gaussians' temperatures.
"""

import os
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
import tyro

from torch.nn.functional import softplus
from torchmetrics.image import PeakSignalNoiseRatio, LearnedPerceptualImagePatchSimilarity
from pytorch_msssim import SSIM

# HeatGaussian imports
from heatgaussian.double_spherical_gaussian_radiance_cache import DoubleSphericalGaussianRadianceCache
from heatgaussian.select_gaussians_from_masks import select_gaussians_from_masks, load_masks_from_dataset
from heatgaussian.hemicube_scattering import hemicube_scattering
from heatgaussian.invsoftplus import invsoftplus
from heatgaussian.prepare_output_dir_getter import prepare_output_dir_getter

# Utility functions
from heatgaussian.load_2dgs_checkpoint import load_2dgs_checkpoint
from heatgaussian.render_gaussians import render_gaussians
from heatgaussian.gamma import gamma
from heatgaussian.show_mask_selection import show_mask_selection


@dataclass
class EvalTemperatureArgs:
	"""Evaluation configuration arguments."""

	thermal_output_dir: str
	"""Directory containing thermal reconstruction outputs and checkpoints"""

	checkpoint_filename: str = "checkpoint-latest.pt"
	"""Checkpoint file to load"""

	experiment_name: str | None = None
	"""Optional experiment name for output directory"""

	target_temperature: float | None = None
	"""Temperature to prescribe to mask-selected Gaussians.
	Defaults to the boundary_temperature stored in the dataset metadata.
	Required if masks are found and the dataset has no boundary_temperature."""

	save_detailed_plots: bool = False
	"""Whether to save detailed visualization plots"""


def eval_temperature(args: EvalTemperatureArgs | None = None):
	"""Main evaluation function."""

	# Parse arguments
	if args is None:
		args = tyro.cli(EvalTemperatureArgs)

	project_root = Path(__file__).parent.parent
	os.chdir(project_root)

	thermal_output_dir = Path(args.thermal_output_dir)
	assert thermal_output_dir.exists(), f"thermal_output_dir {thermal_output_dir} does not exist"

	# Setup output directory
	get_output_dir = prepare_output_dir_getter(__file__, args.experiment_name)

	# Load thermal training configuration
	args_json_path = thermal_output_dir / "args.json"
	assert args_json_path.exists(), f"args.json not found at {args_json_path}"

	with open(args_json_path) as f:
		thermal_config = json.load(f)

	geometry_output_path = Path(thermal_config["geometry_output_path"])
	near_plane = thermal_config["near_plane"]

	# Load geometry reconstruction config to get the 2DGS checkpoint
	geometry_config_path = geometry_output_path / "config.yml"
	assert geometry_config_path.exists(), f"config.yml not found at {geometry_config_path}"

	print(f"Loading geometry checkpoint from: {geometry_config_path}")
	_, pipeline, _, _ = load_2dgs_checkpoint(geometry_config_path)

	eval_cameras = pipeline.datamanager.eval_dataset.cameras
	eval_dataset = pipeline.datamanager.eval_dataset

	dataset_name = pipeline.datamanager.dataset_name

	# Load thermal checkpoint
	thermal_checkpoint_path = thermal_output_dir / "checkpoints" / args.checkpoint_filename
	print(f"Loading thermal checkpoint from {thermal_checkpoint_path}")
	assert thermal_checkpoint_path.exists(), f"Checkpoint not found at {thermal_checkpoint_path}"

	checkpoint = torch.load(thermal_checkpoint_path, weights_only=False)

	_gaussians = checkpoint["gaussians"]
	print(f"Loaded {_gaussians.means.shape[0]} Gaussians")

	# Load or initialize radiance cache
	if "radiance_cache_object" in checkpoint:
		radiance_cache = checkpoint["radiance_cache_object"]
	else:
		radiance_cache = DoubleSphericalGaussianRadianceCache(
			num_primitives=_gaussians.means.shape[0],
			num_lobes=3,
			num_channels=3,
		).cuda()
		radiance_cache.init_with_only_diffuse(0.001)
		radiance_cache.orient_lobes_to_hemisphere(normals=_gaussians.normals)

	# Load masks from dataset and prescribe novel temperature to selected Gaussians
	masks, mask_cameras = load_masks_from_dataset(
		pipeline.datamanager.get_datapath(),
		pipeline.datamanager.train_dataset.cameras,
		pipeline.datamanager.train_dataset,
	)

	print(f"Found {len(masks)} masks in dataset")

	if len(masks) == 0:
		raise ValueError("No masks found in dataset. Cannot prescribe novel temperature without masks. ")

	# Resolve target temperature: CLI arg > dataset metadata > error
	if args.target_temperature is not None:
		target_temperature = args.target_temperature
	else:
		bt = pipeline.datamanager.eval_dataset._dataparser_outputs.metadata.get("boundary_temperature")
		if bt is not None:
			unique_temps = bt[~torch.isnan(bt)].unique()
			assert len(unique_temps) == 1, (
				f"Multiple boundary_temperature values in eval split: {unique_temps.tolist()}. "
				"Specify --target-temperature explicitly."
			)
			target_temperature = unique_temps[0].item()
		else:
			raise ValueError(
				"Masks found but no boundary_temperature in dataset metadata. "
				"Specify --target-temperature explicitly."
			)
	print(f"Prescribing target_temperature={target_temperature} to selected Gaussians")

	selected_indices = select_gaussians_from_masks(
		_gaussians.means,
		_gaussians.quats,
		torch.exp(_gaussians.scales),
		mask_cameras,
		masks,
		dilation_radius=5,
	)
	print(f"Selected {selected_indices.sum()} Gaussians for temperature prescription")
	_gaussians.temperatures[selected_indices] = invsoftplus(
		torch.tensor(args.target_temperature, device=_gaussians.temperatures.device)
	)

	print(f"Evaluating {_gaussians.means.shape[0]} Gaussians")

	# Run evaluation on evaluation dataset (novel temperature images)
	print(f"\nEvaluating on {len(eval_cameras)} evaluation cameras...")
	evaluate_on_dataset(
		trained_gaussians=_gaussians,
		radiance_cache=radiance_cache,
		eval_cameras=eval_cameras,
		eval_dataset=eval_dataset,
		dataset_name=dataset_name,
		near_plane=near_plane,
		args=args,
		get_output_dir=get_output_dir,
	)

	print("\nEvaluation completed.")
	print(f"Results saved to: {get_output_dir()}")



def get_target_temperature(dataset_name: str, max_temperature: float) -> float:
	"""Get target temperature based on dataset."""
	temperature_map = {
		"bunny_box_ro_spp512": 475,
		"heater_bunny_box_ro": 475,
		"heater_bunny_box_ro_spp1024": 475,
		"torous_bunny_box_ro": 475,
		"heater_human_desk_ro": 475,
		"teapot_v14_15_17": max_temperature,
	}

	if dataset_name not in temperature_map:
		raise NotImplementedError(f"Dataset {dataset_name} not supported")

	return temperature_map[dataset_name]


def evaluate_on_dataset(
	trained_gaussians,
	radiance_cache,
	eval_cameras,
	eval_dataset,
	dataset_name,
	near_plane,
	args,
	get_output_dir,
):
	"""
	Evaluate trained gaussians on the evaluation dataset.

	Computes metrics (PSNR, SSIM, LPIPS) on novel temperature images and saves results.
	"""
	# Initialize metrics
	psnr = PeakSignalNoiseRatio().to("cuda")
	ssim = SSIM(data_range=1.0, size_average=True, channel=1).to("cuda")
	lpips = LearnedPerceptualImagePatchSimilarity().to("cuda")

	psnr_scores = []
	ssim_scores = []
	lpips_scores = []

	# Evaluate on all cameras
	for cam_i, eval_cam in enumerate(tqdm(eval_cameras, desc=f"Eval {dataset_name}")):
		with torch.no_grad():
			eval_cam_eye = eval_cam.camera_to_worlds[:3, 3].view(1, 3).cuda()

			# Compute scattering using hemicube
			computed_scattering = hemicube_scattering(
				gaussians=trained_gaussians,
				radiance_cache=radiance_cache,
				eye=eval_cam_eye,
				max_batch_size=256,
				near_plane=near_plane,
				hemicube_resolution=64,
				brdf_model="cook_torrance_ggx",
			)

			# Compute emissions
			self_emissions = softplus(trained_gaussians.temperatures) * torch.sigmoid(trained_gaussians.emissivities)
			total_radiances = computed_scattering + self_emissions

			# Get datapoint
			datapoint = eval_dataset[cam_i]
			thermal_idx = datapoint.get("subset_idx", 0)  # Get the thermal subset index
			ground_truth = datapoint["heat_image"].cuda()

			# Render
			renderings = render_gaussians(
				gaussians=trained_gaussians,
				camera=eval_cam,
				colors=total_radiances,
				near_plane=near_plane,
			)[..., thermal_idx:thermal_idx + 1]

			computed_scattering_rendering = render_gaussians(
				gaussians=trained_gaussians,
				camera=eval_cam,
				colors=computed_scattering,
				near_plane=near_plane,
			)[..., thermal_idx:thermal_idx + 1]

			emission_rendering = render_gaussians(
				gaussians=trained_gaussians,
				camera=eval_cam,
				colors=self_emissions,
				near_plane=near_plane,
			)[..., thermal_idx:thermal_idx + 1]

			# Apply gamma correction
			gamma_gt = gamma(ground_truth)
			gamma_renderings = gamma(renderings)

			H, W = eval_cam.height, eval_cam.width

			# Compute metrics
			psnr_scores.append(psnr(
				gamma_renderings.view(1, 1, H, W),
				gamma_gt.view(1, 1, H, W),
			).item())

			ssim_scores.append(ssim(
				gamma_renderings.view(1, 1, H, W),
				gamma_gt.view(1, 1, H, W),
			).item())

			lpips_scores.append(lpips(
				gamma_renderings.view(1, 1, H, W).expand(1, 3, H, W),
				gamma_gt.view(1, 1, H, W).expand(1, 3, H, W),
			).item())

			# Save visualizations if requested
			if args.save_detailed_plots:
				save_detailed_visualization(
					cam_i=cam_i,
					ground_truth=ground_truth,
					renderings=renderings,
					computed_scattering_rendering=computed_scattering_rendering,
					emission_rendering=emission_rendering,
					gamma_gt=gamma_gt,
					gamma_renderings=gamma_renderings,
					H=H,
					W=W,
					get_output_dir=get_output_dir,
				)

	# Compute summary metrics
	metrics = {
		"num_images": len(psnr_scores),
		"psnr_mean": float(np.mean(psnr_scores)),
		"psnr_std": float(np.std(psnr_scores)),
		"psnr_min": float(np.min(psnr_scores)),
		"psnr_max": float(np.max(psnr_scores)),
		"ssim_mean": float(np.mean(ssim_scores)),
		"ssim_std": float(np.std(ssim_scores)),
		"ssim_min": float(np.min(ssim_scores)),
		"ssim_max": float(np.max(ssim_scores)),
		"lpips_mean": float(np.mean(lpips_scores)),
		"lpips_std": float(np.std(lpips_scores)),
		"lpips_min": float(np.min(lpips_scores)),
		"lpips_max": float(np.max(lpips_scores)),
	}

	# Save metrics
	metrics_file = get_output_dir() / "eval_metrics.json"
	with open(metrics_file, "w") as f:
		json.dump(metrics, f, indent=4)

	print(f"\nEvaluation Metrics:")
	print(json.dumps(metrics, indent=2))
	print(f"Saved to {metrics_file}")


def get_eval_camera_indices(dataset_name: str):
	"""Get camera indices and reflection max value for evaluation."""
	config_map = {
		"bunny_box_ro_spp512": ([1], 0.25),
		"heater_bunny_box_ro": ([1], 0.25),
		"heater_bunny_box_ro_spp1024": ([1], 0.25),
		"torous_bunny_box_ro": ([4], 0.5),
		"teapot_v14_15_17": ([], None),
		"heater_human_desk_ro": ([82], 0.1),
	}

	if dataset_name not in config_map:
		return ([], None)

	return config_map[dataset_name]


def save_detailed_visualization(
	cam_i, ground_truth, renderings, computed_scattering_rendering,
	emission_rendering, gamma_gt, gamma_renderings,
	H, W, get_output_dir
):
	"""Save detailed visualization plots for a camera view."""
	_r, _c = 7, 3
	plt.figure(figsize=(_c * 5, _r * 5))
	plt.suptitle(f"Camera {cam_i}")

	ground_truth_render_max = max(ground_truth.max().item(), renderings.max().item())

	# Ground truth linear
	plt.subplot(_r, _c, 1)
	plt.title(f"Reference cam {cam_i}")
	plt.imshow(ground_truth.reshape(H, W).detach().cpu().numpy(), cmap="inferno", vmax=ground_truth_render_max)
	plt.colorbar()
	plt.axis("off")

	# Full render linear
	plt.subplot(_r, _c, 2)
	plt.title("Full render")
	plt.imshow(renderings.reshape(H, W).detach().cpu().numpy(), cmap="inferno", vmax=ground_truth_render_max)
	plt.colorbar()
	plt.axis("off")

	# Difference
	plt.subplot(_r, _c, 3)
	plt.title("GT - render")
	max_diff = torch.abs(ground_truth - renderings).max().item()
	plt.imshow((ground_truth - renderings).detach().cpu().numpy(), vmin=-max_diff, vmax=max_diff, cmap="seismic")
	plt.colorbar()
	plt.axis("off")

	# Gamma GT
	plt.subplot(_r, _c, 4)
	plt.title("Gamma GT")
	plt.imshow(gamma_gt.reshape(H, W).detach().cpu().numpy(), vmin=0, vmax=1, cmap="inferno")
	plt.axis("off")

	# Gamma rendering
	plt.subplot(_r, _c, 5)
	plt.title("Gamma rendering")
	plt.imshow(gamma_renderings.reshape(H, W).detach().cpu().numpy(), vmin=0, vmax=1, cmap="inferno")
	plt.axis("off")

	# Gamma difference
	plt.subplot(_r, _c, 6)
	plt.title("Gamma GT - render")
	max_diff = torch.abs(gamma_gt - gamma_renderings).max().item()
	plt.imshow((gamma_gt - gamma_renderings).detach().cpu().numpy(), vmin=-max_diff, vmax=max_diff, cmap="seismic")
	plt.colorbar()
	plt.axis("off")

	# Computed scattering
	plt.subplot(_r, _c, 7)
	plt.title("Computed scattering")
	plt.imshow(computed_scattering_rendering.reshape(H, W).detach().cpu().numpy(), cmap="inferno", vmin=0, vmax=1)
	plt.colorbar()
	plt.axis("off")

	# Emission only
	plt.subplot(_r, _c, 10)
	plt.title("Emission only")
	plt.imshow(emission_rendering.reshape(H, W).detach().cpu().numpy(), vmin=0, vmax=1, cmap="inferno")
	plt.axis("off")

	plt.tight_layout()
	plt.savefig(get_output_dir("plots") / f"cam_{cam_i:06d}.png", dpi=150)
	plt.close()

	# Save key images separately for easy viewing
	plt.imsave(get_output_dir("gamma_ref") / f"{cam_i:06d}.png",
	          gamma_gt.reshape(H, W).detach().cpu().numpy(), cmap="inferno", vmin=0, vmax=1)
	plt.imsave(get_output_dir("gamma_render") / f"{cam_i:06d}.png",
	          gamma_renderings.reshape(H, W).detach().cpu().numpy(), cmap="inferno", vmin=0, vmax=1)


if __name__ == "__main__":
	eval_temperature()
