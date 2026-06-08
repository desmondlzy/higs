#!/usr/bin/env python3
"""
Training script for HeatGaussian thermal-RGB reconstruction.

This script trains thermal material properties (temperature, emissivity, specularity)
using a radiosity-based optimization with spherical Gaussian radiance caching.
"""

import os
import json
import math
from collections import defaultdict
from dataclasses import dataclass
import dataclasses
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import imageio
from tqdm import tqdm
from matplotlib import pyplot as plt
from IPython import get_ipython
import tyro

from torch.nn.functional import softplus
from nerfstudio.models.splatfacto import get_viewmat
from pytorch_msssim import SSIM

# HeatGaussian imports
from heatgaussian.hemicube_scattering import hemicube_scattering_for_indices
from heatgaussian.nsext.heat_2dgs import Heat2DGSModel
from heatgaussian.lookat_cameras import lookat_cameras
from heatgaussian.normal_direction_weights_for_camera_2dgs import normal_direction_weights_for_cameras_2dgs
from heatgaussian.double_spherical_gaussian_radiance_cache import DoubleSphericalGaussianRadianceCache
from heatgaussian.heat_gaussians import HeatGaussianWithDiffuse
from heatgaussian.visibility_from_cameras_from_width_height import visibility_from_cameras_width_height
from heatgaussian.local_to_worlds_from_normalized_quats import local_to_worlds_from_normalized_quats
from heatgaussian.canonical_camera_up_local import canonical_camera_up_local
from heatgaussian.invsoftplus import invsoftplus
from heatgaussian.evaluate_metrics_over_dataset import evaluate_metrics_over_dataset, get_metric_functions
from heatgaussian.prepare_output_dir_getter import prepare_output_dir_getter
from heatgaussian.hemicube_scattering import hemicube_scattering

# New utility functions
from heatgaussian.load_2dgs_checkpoint import load_2dgs_checkpoint
from heatgaussian.temperature_to_emission import temperature_to_emission
from heatgaussian.render_gaussians import render_gaussians
from heatgaussian.gamma import gamma


@dataclass
class ThermalReconstructionArgs:
	"""Training configuration arguments."""

	geometry_output_path: str
	"""Path to geometry reconstruction output folder (e.g., outputs/synthetic-exr/heat-2dgs/run_name)"""

	near_plane: float
	"""Near clipping plane for hemicube rendering; colmap data: 0.01, synthetic data: 0.3"""

	experiment_name: str | None = None
	"""Optional experiment name for output directory"""

	num_iters: int = 300000
	"""Total number of training iterations"""

	emissivity_inits: float = 0.5
	"""Initial emissivity value"""

	specularity_inits: float = 0.4
	"""Initial specularity value"""

	temperature_lr: float = 1e-3
	"""Learning rate for temperature"""

	specularity_lr: float = 1e-2
	"""Learning rate for specularity"""

	emissivities_lr: float = 0
	"""Learning rate for emissivities """

	residue_weight: float = 1e1
	"""Weight for radiosity residue loss"""

	save_checkpoint_every: int = 25000
	"""Save checkpoint every N iterations"""

	brdf_model: Literal["cook_torrance_ggx", "cook_torrance_sg"] = "cook_torrance_ggx"
	"""BRDF model: 'cook_torrance_ggx' or 'cook_torrance_sg'"""

	batch_size: int = 64
	"""Batch size for radiosity computation"""

	save_scene_video: bool = False
	"""Whether to save scene setup visualization video"""


def thermal_reconstruction(args: ThermalReconstructionArgs | None = None):
	"""Main training function.

	Args:
		args: Optional pre-configured arguments. If None, will parse from command line.
	"""
	# Parse arguments from CLI if not provided
	if args is None:
		args = tyro.cli(ThermalReconstructionArgs)

	# Setup output directory
	project_root = Path(__file__).parent.parent
	get_output_dir = prepare_output_dir_getter(__file__, args.experiment_name)
	output_dir = get_output_dir()

	# Save configuration
	with open(output_dir / "thermal-args.json", "w") as f:
		json.dump(dataclasses.asdict(args), f, indent=4)

	# Load 2DGS checkpoint from geometry reconstruction output
	os.chdir(project_root)

	geometry_output_path = Path(args.geometry_output_path)

	# Look for config.yml in the geometry output directory
	config_path = geometry_output_path / "config.yml"

	if not config_path.exists():
		raise FileNotFoundError(f"Config file not found at {config_path}")

	print(f"Loading checkpoint from: {geometry_output_path}")
	config, pipeline, _, _ = load_2dgs_checkpoint(config_path)

	train_cameras = pipeline.datamanager.train_dataset.cameras
	train_dataset = pipeline.datamanager.train_dataset
	model_2dgs = pipeline.model

	# Extract metadata
	min_radiance = model_2dgs.metadata["min_radiance"]
	max_radiance = model_2dgs.metadata["max_radiance"]
	min_temperature = model_2dgs.metadata["min_temperature"]
	max_temperature = model_2dgs.metadata["max_temperature"]

	print(f"Loaded checkpoint from: {geometry_output_path}")
	print(f"Radiance range: [{min_radiance:.4f}, {max_radiance:.4f}]")
	print(f"Temperature range: [{min_temperature:.2f}, {max_temperature:.2f}]")

	# Orient normals and select Gaussians
	print("Orienting normals and selecting Gaussians...")
	primary_ns, primary_ts, secondary_ns, secondary_ts = normal_direction_weights_for_cameras_2dgs(
		means=model_2dgs.means,
		quats=model_2dgs.quats,
		scales=torch.exp(model_2dgs.scales),
		opacities=torch.sigmoid(model_2dgs.thermal_opacities).squeeze(),
		cameras=train_cameras,
		aggregation_method="amax",
	)

	# Filter Gaussians based on normal visibility
	zero_normals = (torch.sum(primary_ns.abs(), axis=-1) < 1e-4) & (torch.sum(secondary_ns.abs(), axis=-1) < 1e-4)
	keep_indices = ~zero_normals

	# Select the better-oriented normal
	bigger_ts = primary_ts.sum(axis=0) > secondary_ts.sum(axis=0)
	bigger_ns = torch.where(bigger_ts[:, None], primary_ns, secondary_ns)

	print(f"Kept {keep_indices.sum()}/{len(keep_indices)} Gaussians after normal filtering")

	# Initialize material properties
	_init_emissivities = torch.full_like(model_2dgs.thermal_opacities, args.emissivity_inits).unsqueeze(-1)
	_init_temperatures = torch.sigmoid(model_2dgs.thermal_radiances) / _init_emissivities
	_init_specularities = torch.full_like(model_2dgs.thermal_opacities, args.specularity_inits)
	_init_diffuses = torch.full_like(model_2dgs.thermal_opacities, 0.5)

	# Create Gaussian model (always uses HeatGaussianWithDiffuse)
	gaussians = HeatGaussianWithDiffuse(
		means=torch.nn.Parameter(model_2dgs.means[keep_indices].detach().clone()),
		quats=torch.nn.Parameter(model_2dgs.quats[keep_indices].detach().clone()),
		scales=torch.nn.Parameter(model_2dgs.scales[keep_indices].detach().clone()),
		thermal_opacities=torch.nn.Parameter(model_2dgs.thermal_opacities[keep_indices].detach().clone()),
		normals=torch.nn.Parameter(bigger_ns[keep_indices].detach().clone()),
		rgb_colors=torch.nn.Parameter((model_2dgs.colors.squeeze(1).clamp_min(1e-5)[keep_indices]).detach().clone()),
		specularities=torch.nn.Parameter(torch.logit(_init_specularities).unsqueeze(-1)[keep_indices].detach().clone()),
		emissivities=torch.nn.Parameter(torch.logit(_init_emissivities)[keep_indices].detach().clone()),
		temperatures=torch.nn.Parameter(invsoftplus(_init_temperatures)[keep_indices].detach().clone()),
		diffuses=torch.nn.Parameter(
			torch.logit(_init_diffuses)[keep_indices].detach().clone() if args.temperature_lr > 1e-8
			else _init_diffuses[keep_indices].detach().clone()
		),
	)

	torch.cuda.empty_cache()
	n_gaussians = gaussians.means.shape[0]

	# Initialize radiance cache
	print("Initializing radiance cache...")
	radiance_cache = DoubleSphericalGaussianRadianceCache(
		num_primitives=n_gaussians,
		num_lobes=3,
		num_channels=3,
	).cuda()

	with torch.no_grad():
		original_thermal_emissions = softplus(gaussians.temperatures.clone()) * torch.sigmoid(gaussians.emissivities.clone())
		radiance_cache.orient_lobes_to_hemisphere(gaussians.normals)
		radiance_cache.init_with_only_diffuse(0.01)
		radiance_cache.compile()

	print(f"Training {n_gaussians} Gaussians")
	print(f"Near plane: {args.near_plane}")

	# Optional: Save scene setup video
	if args.save_scene_video:
		print("Generating scene setup video...")
		save_scene_visualization(gaussians, original_thermal_emissions, radiance_cache, output_dir, args.near_plane)

	# Setup training
	setup_training_parameters(gaussians, args)

	# Setup optimizers
	gaussian_optimizer = torch.optim.Adam([
		{"params": gaussians.temperatures, "lr": args.temperature_lr},
		{"params": gaussians.specularities, "lr": args.specularity_lr},
		{"params": gaussians.diffuses, "lr": 1e-3},
	])
	if args.emissivities_lr > 1e-8:
		gaussian_optimizer.add_param_group({
			"params": gaussians.emissivities,
			"lr": args.emissivities_lr,
		})

	diffuse_strengths_optimizer = torch.optim.Adam([
		{"params": radiance_cache.primary_cache.diffuse_strengths, "lr": 1e-3},
		{"params": radiance_cache.secondary_cache.diffuse_strengths, "lr": 1e-3},
	])

	specular_optimizer = torch.optim.Adam([
		{"params": radiance_cache.primary_cache.lobes, "lr": 1e-3},
		{"params": radiance_cache.primary_cache.sharpness, "lr": 1e-3},
		{"params": radiance_cache.primary_cache.amplitudes, "lr": 5e-3},
		{"params": radiance_cache.secondary_cache.lobes, "lr": 1e-3},
		{"params": radiance_cache.secondary_cache.sharpness, "lr": 1e-3},
		{"params": radiance_cache.secondary_cache.amplitudes, "lr": 5e-3},
	])

	train_loop(
		gaussians=gaussians,
		radiance_cache=radiance_cache,
		original_thermal_emissions=original_thermal_emissions,
		train_cameras=train_cameras,
		train_dataset=train_dataset,
		gaussian_optimizer=gaussian_optimizer,
		diffuse_strengths_optimizer=diffuse_strengths_optimizer,
		specular_optimizer=specular_optimizer,
		args=args,
		get_output_dir=get_output_dir,
		config_path=config_path,
	)

	print("Training completed.")
	return str(output_dir)


def setup_training_parameters(gaussians, args):
	"""Set requires_grad flags for Gaussian parameters."""
	gaussians.means.requires_grad_(False)
	gaussians.quats.requires_grad_(False)
	gaussians.scales.requires_grad_(False)
	gaussians.thermal_opacities.requires_grad_(False)
	gaussians.specularities.requires_grad_(args.specularity_lr > 1e-8)
	gaussians.emissivities.requires_grad_(args.emissivities_lr > 1e-8)
	gaussians.temperatures.requires_grad_(args.temperature_lr > 1e-8)
	gaussians.normals.requires_grad_(False)
	gaussians.rgb_colors.requires_grad_(False)


def train_loop(
	gaussians,
	radiance_cache,
	original_thermal_emissions,
	train_cameras,
	train_dataset,
	gaussian_optimizer,
	diffuse_strengths_optimizer,
	specular_optimizer,
	args: ThermalReconstructionArgs,
	get_output_dir,
	config_path,
):
	"""
	Main training loop.

	Optimizes Gaussian material properties and radiance cache to match
	observed thermal images using radiosity-based rendering.
	"""
	n_gaussians = gaussians.means.shape[0]
	reso = 64  # Hemicube resolution
	specular_start_at = 15000

	# Compute transformation matrices for hemicubes
	gaussian_to_worlds = local_to_worlds_from_normalized_quats(
		gaussians.quats / torch.norm(gaussians.quats, dim=1, keepdim=True)
	)
	ups_world = gaussian_to_worlds @ canonical_camera_up_local.to(gaussian_to_worlds.device)

	psnr, ssim, lpips, mae = get_metric_functions()
	history = defaultdict(list)

	total_residue = 0.0
	total_image_loss = 0.0

	# Shuffle camera order
	all_camera_indices = torch.randperm(len(train_cameras), device="cuda")

	# Training loop
	for it in tqdm(range(args.num_iters + 1)):
		cam_index = all_camera_indices[it % len(all_camera_indices)]
		train_cam = train_cameras[cam_index: cam_index + 1]
		datapoint = train_dataset[cam_index]
		thermal_idx = datapoint["subset_idx"]

		# Validate cache
		assert torch.isfinite(radiance_cache.primary_cache.diffuse_strengths).all()
		assert torch.isfinite(radiance_cache.secondary_cache.diffuse_strengths).all()

		# Sample Gaussians based on visibility
		with torch.no_grad():
			visible_indices, visible_weights = visibility_from_cameras_width_height(
				means3d=gaussians.means,
				quats=gaussians.quats,
				scales=torch.exp(gaussians.scales),
				colors=gaussians.rgb_colors,
				opacities=torch.sigmoid(gaussians.thermal_opacities).squeeze(),
				normals_world=gaussians.normals,
				viewmats=get_viewmat(train_cam.camera_to_worlds.cuda()),
				Ks=train_cam.get_intrinsics_matrices().cuda(),
				image_width=train_cam.width,
				image_height=train_cam.height,
				max_hits=25,
				near_plane=args.near_plane,
				normal_filtering=False,
				returns_rgb=False,
			)

			valid_mask = visible_indices >= 0
			valid_indices = visible_indices[valid_mask]
			valid_weights = visible_weights[valid_mask]

			weights_per_gaussian = torch.scatter_add(
				torch.zeros((n_gaussians,), device="cuda"),
				0,
				valid_indices,
				valid_weights,
			)

			radiosity_gaus_indices = torch.multinomial(
				weights_per_gaussian, num_samples=args.batch_size, replacement=False
			)

		train_cam_eyes = train_cam.camera_to_worlds[0, :3, 3].view(1, 3).expand(args.batch_size, 3).cuda()

		# Compute self-emission
		self_emissions = temperature_to_emission(
			softplus(gaussians.temperatures),
			torch.sigmoid(gaussians.emissivities),
		)

		# Compute scattering using hemicubes
		computed_scattering_to_train_cam, info = hemicube_scattering_for_indices(
			gaussians=gaussians,
			radiance_cache=radiance_cache,
			gaus_indices=radiosity_gaus_indices,
			eyes=train_cam_eyes,
			near_plane=args.near_plane,
			hemicube_resolution=reso,
			hemicube_ups=ups_world[radiosity_gaus_indices, :],
			brdf_model=args.brdf_model,
		)

		computed_radiance = computed_scattering_to_train_cam + self_emissions[radiosity_gaus_indices, :]

		# Cached radiance (from spherical Gaussian cache)
		gaussians_to_cam = info["gaussians_to_cam"]
		cached_scattering_to_train_cam = radiance_cache.partial_evaluate(
			gaussians_to_cam,
			gaussians.normals,
			radiosity_gaus_indices,
		)

		cached_radiance = cached_scattering_to_train_cam + self_emissions[radiosity_gaus_indices, :]

		# Radiosity residue loss
		diff = cached_scattering_to_train_cam - computed_scattering_to_train_cam
		residue = torch.square(diff).mean()

		# Image reconstruction loss
		gt_image = datapoint["heat_image"].cuda()
		gaussians_to_train_cam = torch.nn.functional.normalize(
			train_cam.camera_to_worlds[0, :3, 3].view(1, 3).cuda() - gaussians.means
		)
		cached_radiances = radiance_cache(gaussians_to_train_cam, gaussians.normals) + self_emissions

		render_from_camera = render_gaussians(
			gaussians=gaussians,
			camera=train_cam,
			colors=cached_radiances,
			near_plane=args.near_plane,
		)[..., thermal_idx: thermal_idx + 1]

		# Apply gamma correction for perceptual loss
		gamma_render = gamma(render_from_camera)
		gamma_gt = gamma(gt_image)

		# Combined L1 + SSIM loss
		image_diff = gamma_render - gamma_gt
		ssim_lambda = 0.2
		ssim_value = ssim(
			gamma_gt.permute(2, 0, 1)[None, ...],
			gamma_render.permute(2, 0, 1)[None, ...]
		)
		ssimloss = 1 - ssim_value
		image_loss = (1 - ssim_lambda) * torch.abs(image_diff).mean() + ssim_lambda * ssimloss

		# Total loss
		loss = args.residue_weight * residue + image_loss

		# Optimization step
		diffuse_strengths_optimizer.zero_grad()
		gaussian_optimizer.zero_grad()
		if it > specular_start_at:
			specular_optimizer.zero_grad()

		loss.backward()

		diffuse_strengths_optimizer.step()
		gaussian_optimizer.step()
		if it > specular_start_at:
			specular_optimizer.step()

		total_residue += residue
		total_image_loss += image_loss

		# Evaluation and logging
		is_new_epoch = 0 < (it + 1) * args.batch_size % n_gaussians <= args.batch_size
		num_epoch = (it + 1) * args.batch_size // n_gaussians

		if (is_new_epoch and num_epoch % 2 == 0) or (it >= args.num_iters):
			log_training_progress(
				it=it,
				num_epoch=num_epoch,
				gaussians=gaussians,
				radiance_cache=radiance_cache,
				original_thermal_emissions=original_thermal_emissions,
				train_cam=train_cam,
				train_dataset=train_dataset,
				cam_index=cam_index,
				thermal_idx=thermal_idx,
				total_residue=total_residue,
				image_loss=image_loss,
				near_plane=args.near_plane,
				brdf_model=args.brdf_model,
				get_output_dir=get_output_dir,
				history=history,
			)

		# Save checkpoint
		if it % args.save_checkpoint_every == 0:
			checkpoint = {
				"it": it,
				"gaussians": gaussians,
				"radiance_cache_object": radiance_cache,
				"gaussian_optimizer": gaussian_optimizer.state_dict(),
				"diffuse_strengths_optimizer": diffuse_strengths_optimizer.state_dict(),
				"specular_optimizer": specular_optimizer.state_dict(),
				"history": history,
				"nerfstudio_config_path": config_path,
			}
			torch.save(checkpoint, get_output_dir("checkpoints") / f"checkpoint-{it:06d}.pt")
			torch.save(checkpoint, get_output_dir("checkpoints") / f"checkpoint-latest.pt")
			print(f"Checkpoint saved: iteration {it}")

		# Evaluate on train dataset
		if (is_new_epoch and num_epoch % 2 == 0) or (it >= args.num_iters):
			metrics = evaluate_metrics_over_dataset(
				gaussians=gaussians,
				radiance_cache=radiance_cache,
				dataset=train_dataset,
				cameras=train_cameras,
				near_plane=args.near_plane,
				save_to=get_output_dir("metrics") / f"metrics-train-{it:06d}.json",
			)
			print(f"Iteration {it}, train metrics:\n{json.dumps(metrics, indent=4)}")

		# Reset accumulators at epoch boundary
		if is_new_epoch:
			total_residue = 0.0
			total_image_loss = 0.0
			radiance_cache.normalize_lobes()


def log_training_progress(
	it, num_epoch, gaussians, radiance_cache, original_thermal_emissions,
	train_cam, train_dataset, cam_index, thermal_idx, total_residue, image_loss,
	near_plane, brdf_model, get_output_dir, history
):
	"""Log training progress with visualizations."""
	n_gaussians = gaussians.means.shape[0]
	reso = 64

	with torch.no_grad():
		print(f"Iteration {it}: residue={total_residue:.4f}, image_loss={image_loss:.4f}")

		# Visualization camera
		vis_camera = lookat_cameras(
			target=torch.tensor([0.0, 0.0, -0.10], dtype=torch.float32).cuda(),
			up=torch.tensor([0.0, 1.0, 0.1], dtype=torch.float32).cuda(),
			eye=torch.tensor([-0.0, -0.2, 0.02], dtype=torch.float32).cuda(),
			camera_angle_x=np.deg2rad(60.0),
			flip=True,
			img_h=512,
			img_w=512,
		)

		vis_eye = vis_camera.camera_to_worlds[:, :3, 3].cuda()
		train_eye = train_cam.camera_to_worlds[:, :3, 3].cuda()

		vis_wout = torch.nn.functional.normalize(vis_eye - gaussians.means)
		train_wout = torch.nn.functional.normalize(train_eye - gaussians.means)

		self_emissions = temperature_to_emission(
			softplus(gaussians.temperatures),
			torch.sigmoid(gaussians.emissivities),
		)

		# Render with cache
		cached_scatterings_train = radiance_cache(train_wout, gaussians.normals).reshape(n_gaussians, 3)
		cached_radiances_train = cached_scatterings_train + self_emissions

		# Compute actual scattering
		computed_scatterings_train = hemicube_scattering(
			gaussians=gaussians,
			radiance_cache=radiance_cache,
			eye=train_eye,
			max_batch_size=128,
			near_plane=near_plane,
			hemicube_resolution=reso,
			brdf_model=brdf_model,
		)

		# Render various components
		gt_image = train_dataset[cam_index.item()]["heat_image"].cuda()

		render_from_camera = render_gaussians(
			gaussians=gaussians,
			camera=train_cam,
			colors=cached_radiances_train,
			near_plane=near_plane,
		)[..., thermal_idx: thermal_idx + 1]

		original_rendering = gamma(render_gaussians(
			gaussians=gaussians,
			camera=train_cam,
			colors=original_thermal_emissions,
			near_plane=near_plane,
		)[..., thermal_idx: thermal_idx + 1])

		cached_scatter_render = render_gaussians(
			gaussians=gaussians,
			camera=train_cam,
			colors=cached_scatterings_train,
			near_plane=near_plane,
		)[..., thermal_idx: thermal_idx + 1]

		computed_scatter_render = render_gaussians(
			gaussians=gaussians,
			camera=train_cam,
			colors=computed_scatterings_train,
			near_plane=near_plane,
		)[..., thermal_idx: thermal_idx + 1]

		self_emission_render = render_gaussians(
			gaussians=gaussians,
			camera=train_cam,
			colors=self_emissions,
			near_plane=near_plane,
		)[..., thermal_idx: thermal_idx + 1]

		# Create visualization
		image_diff = gamma(render_from_camera) - gamma(gt_image)

		_r, _c = 6, 3
		plt.figure(figsize=(5 * _c, 5 * _r))

		plt.subplot(_r, _c, 1)
		plt.title(f"GT (cam {cam_index.item()})")
		plt.imshow(gamma(gt_image).detach().cpu().numpy(), vmin=0.0, vmax=1.0, cmap="inferno")
		plt.axis("off")

		plt.subplot(_r, _c, 2)
		plt.title(f"Render (epoch={num_epoch})")
		plt.imshow(gamma(render_from_camera).detach().cpu().numpy(), vmin=0.0, vmax=1.0, cmap="inferno")
		plt.axis("off")

		plt.subplot(_r, _c, 3)
		plt.title(f"Render - GT ({image_diff.abs().mean().item():.3e})")
		plt.imshow(image_diff.detach().cpu().numpy(), vmin=-image_diff.abs().max().item(),
		          vmax=image_diff.abs().max().item(), cmap="seismic")
		plt.axis("off")

		plt.subplot(_r, _c, 4)
		plt.title("Cached scattering (gamma)")
		plt.imshow(gamma(cached_scatter_render).detach().cpu().numpy(), vmin=0.0, vmax=1.0, cmap="inferno")
		plt.axis("off")

		plt.subplot(_r, _c, 5)
		plt.title("Computed scattering (gamma)")
		plt.imshow(gamma(computed_scatter_render).detach().cpu().numpy(), vmin=0.0, vmax=1.0, cmap="inferno")
		plt.axis("off")

		plt.subplot(_r, _c, 6)
		difference = computed_scatter_render - cached_scatter_render
		diff_extrema = max(difference.min().abs().item(), difference.max().abs().item())
		plt.title(f"Difference ({difference.abs().mean().item():.3e})")
		plt.imshow(difference.detach().cpu().numpy(), vmin=-diff_extrema, vmax=diff_extrema, cmap="seismic")
		plt.axis("off")

		# Additional rows showing linear space, reflections, etc.
		plt.subplot(_r, _c, 7)
		plt.title("GT linear")
		plt.imshow(gt_image.detach().cpu().numpy(), vmin=0.0, vmax=1.0, cmap="inferno")
		plt.axis("off")

		plt.subplot(_r, _c, 8)
		plt.title("Render linear")
		plt.imshow(render_from_camera.detach().cpu().numpy(), vmin=0.0, vmax=1.0, cmap="inferno")
		plt.axis("off")

		plt.subplot(_r, _c, 16)
		plt.title("Self emissions")
		plt.imshow(gamma(self_emission_render).detach().cpu().numpy(), vmin=0.0, vmax=1.0, cmap="inferno")
		plt.axis("off")

		plt.subplot(_r, _c, 17)
		difference_original_gt = original_rendering - gamma(gt_image)
		plt.title(f"Original - GT ({difference_original_gt.abs().mean().item():.3e})")
		plt.imshow(difference_original_gt.detach().cpu().numpy(),
		          vmin=-difference_original_gt.abs().max(), vmax=difference_original_gt.abs().max(), cmap="seismic")
		plt.axis("off")

		plt.tight_layout()
		plt.savefig(get_output_dir("combines") / f"{it:06d}_train_plot.svg")
		plt.clf()
		plt.close()

		# Update history
		if it > 0:
			history["it"].append(it)
			history["residue"].append(total_residue.item())
			history["image_loss"].append(image_loss.item())

			# Plot training curves
			plt.figure(figsize=(8, 4))
			plt.subplot(1, 2, 1)
			plt.title("Residue")
			plt.plot(history["it"], history["residue"])
			plt.subplot(1, 2, 2)
			plt.title("Image Loss")
			plt.plot(history["it"], history["image_loss"])
			plt.tight_layout()
			plt.savefig(get_output_dir() / "history.svg")
			plt.close()


def save_scene_visualization(gaussians, original_thermal_emissions, radiance_cache, output_dir, near_plane):
	"""Generate and save a video showing the scene setup from different viewpoints."""
	n_gaussians = gaussians.means.shape[0]
	video_writer = imageio.get_writer(output_dir / "scene-setup.mp4", fps=15)

	from heatgaussian.rasterize_2dgs_normal_filtering import rasterize_2dgs_normal_filtering

	with torch.no_grad():
		for theta in torch.linspace(0, 2 * math.pi, 80):
			camera_eye = torch.tensor([
				3 * math.sin(theta),
				3 * math.cos(theta),
				0.8
			], dtype=torch.float32).cuda()

			vis_camera = lookat_cameras(
				target=torch.tensor([0.0, 0.0, -0.10], dtype=torch.float32).cuda(),
				up=torch.tensor([0.0, 0.0, 1], dtype=torch.float32).cuda(),
				eye=camera_eye,
				camera_angle_x=np.deg2rad(60.0),
				flip=True,
				img_h=512,
				img_w=512,
			)

			normals_to_camera = gaussians.normals * torch.sign(
				torch.sum(gaussians.normals * camera_eye.view(1, 3), dim=-1, keepdim=True)
			)

			wout = torch.nn.functional.normalize(camera_eye - gaussians.means)
			emissivities_sigmoid = torch.sigmoid(gaussians.emissivities)

			self_emissions = softplus(gaussians.temperatures) * emissivities_sigmoid
			cache_emissions = radiance_cache(wout, gaussians.normals).reshape(n_gaussians, 3)

			orig_renders = render_gaussians(
				gaussians=gaussians,
				camera=vis_camera,
				colors=original_thermal_emissions,
				near_plane=near_plane,
				normal_occluding=False,
				normal_transparent=False,
			)

			cache_renders = render_gaussians(
				gaussians=gaussians,
				camera=vis_camera,
				colors=cache_emissions,
				near_plane=near_plane,
				normal_occluding=False,
				normal_transparent=False,
			)

			normal_renders = render_gaussians(
				gaussians=gaussians,
				camera=vis_camera,
				colors=normals_to_camera * 0.5 + 0.5,
				near_plane=near_plane,
				normal_occluding=False,
				normal_transparent=False,
			)

			# Combine renderings side by side
			from heatgaussian.radiation_to_intensity import radiation_to_intensity
			inferno_cmap = plt.get_cmap("inferno")
			combine_renders = torch.cat([
				torch.tensor(inferno_cmap(gamma(orig_renders[..., 0]).detach().cpu().numpy())).to(normal_renders)[..., :3],
				cache_renders,
				normal_renders,
			], dim=1)

			video_writer.append_data((combine_renders.clamp_max(1.0).detach().cpu().numpy() * 255).astype(np.uint8))

		video_writer.close()
		print(f"Scene setup video saved to {output_dir / 'scene-setup.mp4'}")


if __name__ == "__main__":
	thermal_reconstruction()
