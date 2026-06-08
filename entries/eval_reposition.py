import os
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import imageio
from tqdm import tqdm
from matplotlib import pyplot as plt
from IPython import get_ipython
import tyro

from torch.nn.functional import softplus
from nerfstudio.models.splatfacto import get_viewmat
from torchmetrics.image import PeakSignalNoiseRatio, LearnedPerceptualImagePatchSimilarity
from torchmetrics.regression import MeanAbsoluteError
from pytorch_msssim import SSIM

# HeatGaussian imports
from heatgaussian.double_spherical_gaussian_radiance_cache import DoubleSphericalGaussianRadianceCache
from heatgaussian.heat_gaussians import HeatGaussianWithDiffuse
from heatgaussian.select_gaussians_from_masks import select_gaussians_from_masks, load_masks_from_dataset
from heatgaussian.hemicube_scattering import hemicube_scattering
from heatgaussian.invsoftplus import invsoftplus
from heatgaussian.prepare_output_dir_getter import prepare_output_dir_getter

# Utility functions
from heatgaussian.load_2dgs_checkpoint import load_2dgs_checkpoint
from heatgaussian.temperature_to_emission import temperature_to_emission
from heatgaussian.render_gaussians import render_gaussians
from heatgaussian.gamma import gamma
from heatgaussian.show_mask_selection import show_mask_selection
from heatgaussian.circle_path import circle_path
from heatgaussian.rotate_around_axis import rotate_around_axis
from heatgaussian.camera_set_aspect_ratio_by_cropping import camera_set_aspect_ratio_by_cropping
from heatgaussian.translate_camera_by_local_displacement import translate_camera_by_local_displacement

from named_checkpoints import named_checkpoints


@dataclass
class EvalRepositionArgs:
	"""Evaluation configuration arguments."""

	thermal_output_dir: str
	"""Directory containing thermal reconstruction outputs and checkpoints"""

	checkpoint_filename: str = "checkpoint-latest.pt"
	"""Checkpoint file to load"""

	experiment_name: str | None = None
	"""Optional experiment name for output directory"""

	seconds: int = 3
	"""Duration of generated videos in seconds"""

	fps: int = 24
	"""Frames per second for generated videos"""

	generate_rgb_video: bool = False
	"""Whether to generate RGB renderings video"""

	generate_thermal_video: bool = False
	"""Whether to generate thermal renderings video"""

	evaluate_metrics: bool = True
	"""Whether to evaluate metrics against ground truth"""

	save_metric_plots: bool = False
	"""Whether to save visualization plots for each camera during metric evaluation"""

	displacement: list[float] | None = None
	"""Final displacement vector [x, y, z] to move selected Gaussians to.
	Defaults to the displacement stored in the dataset metadata."""


def eval_reposition(args: EvalRepositionArgs | None = None):
	"""Main evaluation function."""

	# Parse arguments
	if args is None:
		args = tyro.cli(EvalRepositionArgs)

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
	config, pipeline, _, _ = load_2dgs_checkpoint(geometry_config_path)

	model_2dgs = pipeline.model

	eval_cameras = pipeline.datamanager.novel_position_dataset.cameras
	eval_dataset = pipeline.datamanager.novel_position_dataset

	dataset_name = pipeline.datamanager.get_datapath().stem

	# Load thermal checkpoint
	thermal_checkpoint_path = thermal_output_dir / "checkpoints" / args.checkpoint_filename
	print(f"Loading thermal checkpoint from {thermal_checkpoint_path}")
	assert thermal_checkpoint_path.exists(), f"Checkpoint not found at {thermal_checkpoint_path}"

	checkpoint = torch.load(thermal_checkpoint_path, weights_only=False)

	_gaussians = checkpoint["gaussians"]
	print(f"Loaded {_gaussians.means.shape[0]} Gaussians")

	# Load radiance cache
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

	# Create empty radiance cache for novel view synthesis
	empty_radiance_cache = DoubleSphericalGaussianRadianceCache(
		num_primitives=_gaussians.means.shape[0],
		num_lobes=3,
		num_channels=3,
	).cuda()
	empty_radiance_cache.init_with_only_diffuse(0.001)
	empty_radiance_cache.orient_lobes_to_hemisphere(normals=_gaussians.normals)

	# Load masks for gaussian selection
	masks, mask_cameras = load_masks_from_dataset(
		pipeline.datamanager.get_datapath(),
		pipeline.datamanager.train_dataset.cameras,
		pipeline.datamanager.train_dataset,
	)

	print(f"Loaded {len(masks)} masks")

	# Select Gaussians to reposition
	selected_indices = select_gaussians_by_dataset(dataset_name, _gaussians, mask_cameras, masks)

	print(f"Selected {selected_indices.sum()} Gaussians for repositioning")

	# Verify selection
	show_mask_selection(
		gaussians=_gaussians,
		selected_indices=selected_indices,
		mask_camera=mask_cameras[0],
		mask=masks[0],
		near_plane=near_plane,
		output_path=get_output_dir() / "masked_gaussians_rendering.svg",
		use_thermal=(dataset_name == "teapot_v14_15_17"),
	)

	# Resolve displacement: CLI arg > dataset metadata > error
	if args.displacement is not None:
		displacement = torch.tensor(args.displacement, dtype=torch.float32)
	else:
		disp_meta = pipeline.datamanager.novel_position_dataset._dataparser_outputs.metadata.get("displacement")
		if disp_meta is not None:
			unique_disps = disp_meta.unique(dim=0)
			assert len(unique_disps) == 1, (
				f"Multiple displacement values in novel-position split: {unique_disps.tolist()}. "
				"Specify --displacement explicitly."
			)
			displacement = unique_disps[0]
		else:
			displacement = None  # get_displacement_trajectory will use dataset-name fallback

	# Get displacement trajectory
	displacements, rotate_angles = get_displacement_trajectory(dataset_name, args.seconds, args.fps, _gaussians.means.device, displacement)

	print(f"Generated {len(displacements)} displacement points")

	# Setup visualization camera
	if args.generate_rgb_video or args.generate_thermal_video:
		vis_camera, thermal_idx, scattering_vmin, scattering_vmax = setup_visualization_camera(
			dataset_name, eval_cameras
		)

	# Generate RGB video
	if args.generate_rgb_video:
		print("Generating RGB video...")
		generate_rgb_video(
			gaussians=_gaussians,
			selected_indices=selected_indices,
			displacements=displacements,
			rotate_angles=rotate_angles,
			vis_camera=vis_camera,
			near_plane=near_plane,
			get_output_dir=get_output_dir,
			fps=args.fps,
		)

	# Generate thermal video
	if args.generate_thermal_video:
		print("Generating thermal video...")
		generate_thermal_video(
			gaussians=_gaussians,
			selected_indices=selected_indices,
			displacements=displacements,
			rotate_angles=rotate_angles,
			vis_camera=vis_camera,
			empty_radiance_cache=empty_radiance_cache,
			thermal_idx=thermal_idx,
			scattering_vmin=scattering_vmin,
			scattering_vmax=scattering_vmax,
			near_plane=near_plane,
			get_output_dir=get_output_dir,
			fps=args.fps,
		)

	# Evaluate metrics against ground truth
	if args.evaluate_metrics:
		print("Evaluating metrics against ground truth...")

		# Get the final displaced gaussians
		center_of_selected = _gaussians.means[selected_indices].mean(dim=0)
		final_disp = displacements[-1]
		final_angle = rotate_angles[-1]

		heated_means = torch.clone(_gaussians.means)
		heated_quats = torch.clone(_gaussians.quats)
		heated_temperatures = torch.clone(_gaussians.temperatures)
		heated_emissivities = torch.clone(_gaussians.emissivities)
		heated_specularities = torch.clone(_gaussians.specularities)

		# Modify selected gaussians
		heated_emissivities[selected_indices] = torch.logit(
			torch.full_like(heated_emissivities[selected_indices], 0.9)
		)

		# Rotate and translate to final position
		rotated_means, rotated_quats = rotate_around_axis(
			axis=torch.tensor([0.0, 0.0, 1.0]).to(_gaussians.means.device),
			origin=center_of_selected,
			angle=final_angle,
			points=_gaussians.means[selected_indices],
			quats=_gaussians.quats[selected_indices],
		)

		heated_means[selected_indices] = rotated_means + final_disp.view(1, 3)
		heated_quats[selected_indices] = rotated_quats

		heated_gaussians = HeatGaussianWithDiffuse(
			means=heated_means,
			quats=heated_quats,
			scales=_gaussians.scales,
			thermal_opacities=_gaussians.thermal_opacities,
			normals=_gaussians.normals,
			rgb_colors=_gaussians.rgb_colors,
			specularities=heated_specularities,
			emissivities=heated_emissivities,
			temperatures=heated_temperatures,
			diffuses=_gaussians.diffuses,
		)

		evaluate_novel_position_metrics(
			heated_gaussians=heated_gaussians,
			radiance_cache=radiance_cache,
			eval_cameras=eval_cameras,
			eval_dataset=eval_dataset,
			dataset_name=dataset_name,
			near_plane=near_plane,
			get_output_dir=get_output_dir,
			save_plots=args.save_metric_plots,
		)

	print("Evaluation completed.")


def select_gaussians_by_dataset(dataset_name, _gaussians, mask_cameras, masks):
	"""Select Gaussians based on dataset-specific criteria."""
	if dataset_name == "torous_bunny_box_ro":
		# Use bounding box selection
		inner_bbox = torch.tensor([
			[-1.0, -1.0, -0.750],
			[1.0, 1.0, 1.0],
		]).to(_gaussians.means)
		outer_bbox = torch.tensor([
			[-3.5, -3.5, -0.710],
			[3.0, 3.0, 1.0],
		]).to(_gaussians.means)

		inner_selection = torch.all(
			(_gaussians.means > inner_bbox[0]) & (_gaussians.means < inner_bbox[1]), dim=1
		)
		outer_selection = torch.all(
			(_gaussians.means > outer_bbox[0]) & (_gaussians.means < outer_bbox[1]), dim=1
		)
		return (outer_selection & ~inner_selection)
	else:
		# Use mask-based selection
		return select_gaussians_from_masks(
			_gaussians.means,
			_gaussians.quats,
			torch.exp(_gaussians.scales),
			mask_cameras,
			masks,
			dilation_radius=5,
		)



def get_displacement_trajectory(dataset_name, seconds, fps, device, displacement=None):
	"""Get displacement trajectory and rotation angles for the selected dataset.

	For known datasets, uses a dataset-specific trajectory shape.
	For unknown datasets, falls back to a linear interpolation to `displacement`
	(which must be provided, either from metadata or the CLI).
	"""
	# Cubic easing for smooth motion
	t = torch.linspace(0, 1, fps * seconds)
	t_eased = 3 * t**2 - 2 * t**3

	spin_max = 2 * math.pi
	rotate_angles = t_eased * spin_max

	if dataset_name == "teapot_v14_15_17":
		displacements = circle_path(
			start=torch.tensor([0.0, 0.0, 0.0]),
			passing=torch.tensor([-0.01, -0.11, 0.0]),
			end=torch.tensor([-0.11, -0.08, 0.0]),
			num_points=fps * seconds,
			interp_order=3,
		).to(device)
	elif dataset_name in ["heater_bunny_box_ro", "heater_bunny_box_ro_spp1024", "heater_human_desk_ro", "bunny_box_ro_spp512"]:
		# Spiral trajectory
		theta = t_eased * 4 * math.pi
		r = 1.0 * torch.clamp(2 * t_eased, max=1.0)
		displacements = torch.column_stack([
			torch.cos(theta) * r,
			torch.sin(theta) * r,
			torch.full_like(theta, 0.0),
		]).to(device)
	else:
		assert displacement is not None, (
			f"No built-in trajectory for dataset '{dataset_name}'. "
			"Add a displacement to the dataset metadata or pass --displacement."
		)
		# Linear interpolation from zero to the target displacement
		end = displacement.to(device)
		displacements = t_eased.unsqueeze(1) * end.unsqueeze(0)

	return displacements, rotate_angles


def setup_visualization_camera(dataset_name, eval_cameras):
	"""Setup visualization camera and parameters for the dataset."""
	camera_config = {
		"bunny": (41, 2, 0, 0.3, 8/9),
		"human": (41, 2, 0, 0.3, 8/9),
		"radiator": (12, 2, 0, 0.3, 8/9),
		"teapot": (18, 0, 0, 0.4, 16/9),
	}

	if dataset_name not in camera_config:
		raise NotImplementedError(f"Dataset {dataset_name} not supported")

	vis_index, thermal_idx, svmin, svmax, aspect_ratio = camera_config[dataset_name]

	vis_camera = eval_cameras[vis_index]
	vis_camera.rescale_output_resolution(scaling_factor=10)
	camera_set_aspect_ratio_by_cropping(vis_camera, target_aspect_ratio=aspect_ratio)

	# Special camera adjustment for teapot
	if dataset_name == "teapot_v14_15_17":
		vis_camera = translate_camera_by_local_displacement(
			vis_camera, torch.tensor([0.02, -0.05, -0.24]), copy=True
		)

	return vis_camera, thermal_idx, svmin, svmax


def generate_rgb_video(gaussians, selected_indices, displacements, rotate_angles, vis_camera, near_plane, get_output_dir, fps):
	"""Generate RGB rendering video along the trajectory."""
	center_of_selected = gaussians.means[selected_indices].mean(dim=0)

	with torch.no_grad():
		for i, disp in enumerate(tqdm(displacements, desc="RGB frames")):
			heated_means = torch.clone(gaussians.means)
			heated_quats = torch.clone(gaussians.quats)

			# Rotate and translate
			rotated_means, rotated_quats = rotate_around_axis(
				axis=torch.tensor([0.0, 0.0, 1.0]).to(gaussians.means.device),
				origin=center_of_selected,
				angle=rotate_angles[i],
				points=gaussians.means[selected_indices],
				quats=gaussians.quats[selected_indices],
			)

			heated_means[selected_indices] = rotated_means + disp.view(1, 3)
			heated_quats[selected_indices] = rotated_quats

			heated_gaussians = HeatGaussianWithDiffuse(
				means=heated_means,
				quats=heated_quats,
				scales=gaussians.scales,
				thermal_opacities=gaussians.thermal_opacities,
				normals=gaussians.normals,
				rgb_colors=gaussians.rgb_colors,
				specularities=gaussians.specularities,
				emissivities=gaussians.emissivities,
				temperatures=gaussians.temperatures,
				diffuses=gaussians.diffuses,
			)

			rgb_renderings = render_gaussians(
				gaussians=heated_gaussians,
				camera=vis_camera,
				colors=heated_gaussians.rgb_colors[..., :3],
				near_plane=near_plane,
			)

			plt.imsave(
				get_output_dir("rgb_renderings") / f"frame_{i:04d}.png",
				rgb_renderings.detach().cpu().numpy(),
				vmin=0, vmax=1
			)

	# Create video
	create_video_from_frames(get_output_dir("rgb_renderings"), fps)


def generate_thermal_video(
	gaussians, selected_indices, displacements, rotate_angles, vis_camera,
	empty_radiance_cache, thermal_idx, scattering_vmin, scattering_vmax,
	near_plane, get_output_dir, fps
):
	"""Generate thermal rendering video along the trajectory."""
	center_of_selected = gaussians.means[selected_indices].mean(dim=0)

	with torch.no_grad():
		for i, disp in enumerate(tqdm(displacements, desc="Thermal frames")):
			heated_means = torch.clone(gaussians.means)
			heated_quats = torch.clone(gaussians.quats)
			heated_temperatures = torch.clone(gaussians.temperatures)
			heated_emissivities = torch.clone(gaussians.emissivities)
			heated_specularities = torch.clone(gaussians.specularities)

			# Modify selected gaussians
			heated_emissivities[selected_indices] = torch.logit(
				torch.full_like(heated_emissivities[selected_indices], 0.9)
			)

			# Rotate and translate
			rotated_means, rotated_quats = rotate_around_axis(
				axis=torch.tensor([0.0, 0.0, 1.0]).to(gaussians.means.device),
				origin=center_of_selected,
				angle=rotate_angles[i],
				points=gaussians.means[selected_indices],
				quats=gaussians.quats[selected_indices],
			)

			heated_means[selected_indices] = rotated_means + disp.view(1, 3)
			heated_quats[selected_indices] = rotated_quats

			heated_gaussians = HeatGaussianWithDiffuse(
				means=heated_means,
				quats=heated_quats,
				scales=gaussians.scales,
				thermal_opacities=gaussians.thermal_opacities,
				normals=gaussians.normals,
				rgb_colors=gaussians.rgb_colors,
				specularities=heated_specularities,
				emissivities=heated_emissivities,
				temperatures=heated_temperatures,
				diffuses=gaussians.diffuses,
			)

			_cam_eye = vis_camera.camera_to_worlds[:3, 3].view(1, 3).cuda()

			# Compute scattering
			computed_scattering = hemicube_scattering(
				gaussians=heated_gaussians,
				radiance_cache=empty_radiance_cache,
				eye=_cam_eye,
				max_batch_size=128,
				near_plane=near_plane,
				brdf_model="cook_torrance_ggx",
			)

			self_emissions = softplus(heated_gaussians.temperatures) * torch.sigmoid(heated_gaussians.emissivities)
			total_radiances = computed_scattering + self_emissions

			# Render
			renderings = render_gaussians(
				gaussians=heated_gaussians,
				camera=vis_camera,
				colors=total_radiances[..., thermal_idx:thermal_idx + 1],
				near_plane=near_plane,
			)

			computed_scattering_rendering = render_gaussians(
				gaussians=heated_gaussians,
				camera=vis_camera,
				colors=computed_scattering[..., thermal_idx:thermal_idx + 1],
				near_plane=near_plane,
			)

			emission_rendering = render_gaussians(
				gaussians=heated_gaussians,
				camera=vis_camera,
				colors=self_emissions[..., thermal_idx:thermal_idx + 1],
				near_plane=near_plane,
			)

			# Save frames
			gamma_rendering = gamma(renderings)
			gamma_scattering = gamma(computed_scattering_rendering)
			gamma_emission = gamma(emission_rendering)

			plt.imsave(
				get_output_dir("gamma_scattering") / f"frame_{i:04d}.png",
				gamma_scattering.squeeze(-1).detach().cpu().numpy(),
				vmin=0, vmax=scattering_vmax, cmap="inferno"
			)
			plt.imsave(
				get_output_dir("gamma_renderings") / f"frame_{i:04d}.png",
				gamma_rendering.squeeze(-1).detach().cpu().numpy(),
				vmin=0, vmax=1, cmap="inferno"
			)
			plt.imsave(
				get_output_dir("gamma_emission_renderings") / f"frame_{i:04d}.png",
				gamma_emission.squeeze(-1).detach().cpu().numpy(),
				vmin=0, vmax=1, cmap="inferno"
			)

	# Create videos
	for folder_name in ["gamma_scattering", "gamma_renderings", "gamma_emission_renderings"]:
		create_video_from_frames(get_output_dir(folder_name), fps)


def create_video_from_frames(folder: Path, fps: int):
	"""Create MP4 video from PNG frames in a folder."""
	video_filename = folder / f"{folder.stem}.mp4"
	video_writer = imageio.get_writer(video_filename, fps=fps)

	image_filenames = sorted(folder.glob("frame_*.png"))
	for filename in image_filenames:
		image = imageio.imread(filename)
		video_writer.append_data(image)

	video_writer.close()
	print(f"Saved video: {video_filename}")


def evaluate_novel_position_metrics(
	heated_gaussians,
	radiance_cache,
	eval_cameras,
	eval_dataset,
	dataset_name,
	near_plane,
	get_output_dir,
	save_plots=False,
):
	"""Evaluate PSNR/SSIM/LPIPS/MAE metrics against ground truth images."""
	# Initialize metrics
	psnr = PeakSignalNoiseRatio().to("cuda")
	ssim = SSIM(data_range=1.0, size_average=True, channel=1).to("cuda")
	lpips = LearnedPerceptualImagePatchSimilarity().to("cuda")
	mae = MeanAbsoluteError().to("cuda")

	psnr_scores = []
	ssim_scores = []
	lpips_scores = []
	mae_scores = []

	for cam_i, eval_cam in enumerate(tqdm(eval_cameras, desc="Evaluating metrics")):
		with torch.no_grad():
			datapoint = eval_dataset[cam_i]
			thermal_index = datapoint["subset_idx"]

			eval_cam_eye = eval_cam.camera_to_worlds[:3, 3].view(1, 3).cuda()

			# Compute scattering
			computed_scattering = hemicube_scattering(
				gaussians=heated_gaussians,
				radiance_cache=radiance_cache,
				eye=eval_cam_eye,
				max_batch_size=256,
				near_plane=near_plane,
				brdf_model="cook_torrance_ggx",
			)

			# Compute total radiance
			self_emissions = softplus(heated_gaussians.temperatures) * torch.sigmoid(heated_gaussians.emissivities)
			total_radiances = computed_scattering + self_emissions

			# Render
			renderings = render_gaussians(
				gaussians=heated_gaussians,
				camera=eval_cam,
				colors=total_radiances[..., thermal_index:thermal_index + 1],
				near_plane=near_plane,
			)

			# Load ground truth
			ground_truth_path = eval_dataset.heatimage_filenames[cam_i]
			ground_truth = torch.tensor(
				cv2.imread(str(ground_truth_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
			).reshape_as(renderings).to(renderings)

			# Apply gamma correction
			gamma_gt = gamma(ground_truth)
			gamma_renderings = gamma(renderings)

			H, W = eval_cam.height.item(), eval_cam.width.item()

			# Compute metrics (in NCHW format)
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

			mae_scores.append(mae(
				gamma_renderings.view(1, 1, H, W),
				gamma_gt.view(1, 1, H, W),
			).item())

			# Create visualization if requested
			if save_plots:
				emission_rendering = render_gaussians(
					gaussians=heated_gaussians,
					camera=eval_cam,
					colors=self_emissions[..., thermal_index:thermal_index + 1],
					near_plane=near_plane,
				)
				gamma_emission_rendering = gamma(emission_rendering)

				computed_scattering_rendering = render_gaussians(
					gaussians=heated_gaussians,
					camera=eval_cam,
					colors=computed_scattering[..., thermal_index:thermal_index + 1],
					near_plane=near_plane,
				)
				gamma_scattering = gamma(computed_scattering_rendering)

				_r, _c = 3, 3
				plt.figure(figsize=(_c * 5, _r * 5))
				plt.suptitle(f"Camera {cam_i}")

				# Row 1: Ground truth, rendering, difference (gamma)
				plt.subplot(_r, _c, 1)
				plt.title(f"GT gamma (cam {cam_i})")
				plt.imshow(gamma_gt.detach().cpu().numpy(), vmin=0, vmax=1, cmap="inferno")
				plt.axis("off")

				plt.subplot(_r, _c, 2)
				plt.title("Rendering gamma")
				plt.imshow(gamma_renderings.detach().cpu().numpy(), vmin=0, vmax=1, cmap="inferno")
				plt.axis("off")

				plt.subplot(_r, _c, 3)
				plt.title("GT - Rendering")
				max_diff = torch.abs(gamma_gt - gamma_renderings).max().item()
				plt.imshow((gamma_gt - gamma_renderings).detach().cpu().numpy(), vmin=-max_diff, vmax=max_diff, cmap="seismic")
				plt.colorbar()
				plt.axis("off")

				# Row 2: Linear space comparisons
				plt.subplot(_r, _c, 4)
				plt.title("GT linear")
				plt.imshow(ground_truth.detach().cpu().numpy(), vmin=0, vmax=1, cmap="inferno")
				plt.axis("off")

				plt.subplot(_r, _c, 5)
				plt.title("Rendering linear")
				plt.imshow(renderings.detach().cpu().numpy(), vmin=0, vmax=1, cmap="inferno")
				plt.axis("off")

				plt.subplot(_r, _c, 6)
				plt.title("GT - Rendering (linear)")
				max_diff_linear = torch.abs(ground_truth - renderings).max().item()
				plt.imshow((ground_truth - renderings).detach().cpu().numpy(), vmin=-max_diff_linear, vmax=max_diff_linear, cmap="seismic")
				plt.colorbar()
				plt.axis("off")

				# Row 3: Emission and scattering
				plt.subplot(_r, _c, 7)
				plt.title("Emission only gamma")
				plt.imshow(gamma_emission_rendering.detach().cpu().numpy(), vmin=0, vmax=1, cmap="inferno")
				plt.axis("off")

				plt.subplot(_r, _c, 8)
				plt.title("Scattering gamma")
				plt.imshow(gamma_scattering.detach().cpu().numpy(), vmin=0, vmax=1, cmap="inferno")
				plt.colorbar()
				plt.axis("off")

				plt.subplot(_r, _c, 9)
				plt.title("Metrics")
				plt.text(0.1, 0.85, f"PSNR: {psnr_scores[-1]:.2f}", transform=plt.gca().transAxes, fontsize=14)
				plt.text(0.1, 0.65, f"SSIM: {ssim_scores[-1]:.4f}", transform=plt.gca().transAxes, fontsize=14)
				plt.text(0.1, 0.45, f"LPIPS: {lpips_scores[-1]:.4f}", transform=plt.gca().transAxes, fontsize=14)
				plt.text(0.1, 0.25, f"MAE: {mae_scores[-1]:.4f}", transform=plt.gca().transAxes, fontsize=14)
				plt.axis("off")

				plt.tight_layout()
				plt.savefig(get_output_dir("metrics_eval") / f"cam_{cam_i:06d}.svg")
				plt.close()

	# Compute summary statistics
	metrics = {
		"psnr_mean": float(np.mean(psnr_scores)),
		"psnr_std": float(np.std(psnr_scores)),
		"ssim_mean": float(np.mean(ssim_scores)),
		"ssim_std": float(np.std(ssim_scores)),
		"lpips_mean": float(np.mean(lpips_scores)),
		"lpips_std": float(np.std(lpips_scores)),
		"mae_mean": float(np.mean(mae_scores)),
		"mae_std": float(np.std(mae_scores)),
	}

	# Save metrics to JSON
	metrics_path = get_output_dir("metrics_eval") / "metrics.json"
	with open(metrics_path, "w") as f:
		json.dump(metrics, f, indent=4)

	print(f"\nMetrics Summary:")
	print(f"  PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f}")
	print(f"  SSIM: {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
	print(f"  LPIPS: {metrics['lpips_mean']:.4f} ± {metrics['lpips_std']:.4f}")
	print(f"  MAE: {metrics['mae_mean']:.4f} ± {metrics['mae_std']:.4f}")
	print(f"  MAPE: {metrics['mape_mean']:.4f} ± {metrics['mape_std']:.4f}")
	print(f"\nSaved metrics to: {metrics_path}")


if __name__ == "__main__":
	eval_reposition()
