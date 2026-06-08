#!/usr/bin/env python3
"""
Geometry reconstruction training script for thermal 2DGS.

This script programmatically trains a thermal 2D Gaussian Splatting model
without needing to use the ns-train command line interface.
"""

from pathlib import Path
from dataclasses import dataclass
import tyro

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

from heatgaussian.nsext.heat_2dgs import Heat2DGSModelConfig
from heatgaussian.nsext.heat_dataset import HeatDataset
from heatgaussian.nsext.multi_full_image_datamanager import (
	MultiFullImageDatamanagerConfig,
	MultiFullImageDatamanager,
)
from heatgaussian.nsext.multi_heat_dataparser import MultiHeatDataparserConfig
from heatgaussian.nsext.higs_heat_dataparser import HigsHeatDataparserConfig


@dataclass
class GeometryReconstructionArgs:
	"""Configuration for geometry reconstruction training."""

	# Data parameters
	data_path: str = "data/higs/radiator"
	"""Path to the dataset"""

	output_dir: str | None = None
	"""Output folder name (if None: outputs/{experiment_name}/heat-2dgs/{timestamp})"""

	# Experiment parameters
	experiment_name: str = "unnamed"
	"""Name of the experiment"""

	# Model parameters
	max_gaussian_num: int = 100000
	"""Maximum number of Gaussians"""

	dist_loss: bool = True
	"""Enable distance loss"""

	random_scale: float = 5.0
	"""Random initialization scale for Gaussians"""

	prune_opa: float = 0.01
	"""Opacity threshold for pruning RGB Gaussians"""

	prune_thermal_opa: float = 0.01
	"""Opacity threshold for pruning thermal Gaussians"""

	grow_scale3d: float = 0.02
	"""3D scale threshold for growing Gaussians"""

	grow_grad2d: float = 0.0012
	"""2D gradient threshold for growing Gaussians"""

	use_tonemapping: bool = True
	"""Enable tonemapping for thermal rendering"""

	# Dataparser parameters
	auto_scale_poses: bool = False
	"""Auto-scale camera poses"""

	center_method: str = "none"
	"""Method for centering poses: 'focus', 'poses', 'none'"""

	orientation_method: str = "none"
	"""Method for orienting poses: 'pca', 'up', 'vertical', 'none'"""

	# Memory parameters
	image_scale_factor: float = 1.0
	"""Downsample images before caching (e.g. 0.5 = half res, 0.25 = quarter res). Reduces RAM ~4x per halving."""

	# Training parameters
	num_iters: int = 30000
	"""Maximum number of training iterations"""

	steps_per_save: int = 2000
	"""Save checkpoint every N steps"""

	steps_per_eval_image: int = 100
	"""Evaluate single image every N steps"""

	steps_per_eval_all_images: int = 1000
	"""Evaluate all images every N steps"""

	viewer: bool = False
	"""Enable the viser viewer during training"""


def get_optimizers_config():
	"""Get optimizer configuration for training."""
	return {
		"means": {
			"optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
			"scheduler": ExponentialDecaySchedulerConfig(
				lr_final=1.6e-6,
				max_steps=30000,
			),
		},
		"features_dc": {
			"optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
			"scheduler": None,
		},
		"features_rest": {
			"optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
			"scheduler": None,
		},
		"opacities": {
			"optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
			"scheduler": None,
		},
		"thermal_opacities": {
			"optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
			"scheduler": None,
		},
		"scales": {
			"optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
			"scheduler": None,
		},
		"thermal_radiances": {
			"optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
			"scheduler": None,
		},
		"quats": {
			"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
			"scheduler": None,
		},
		"camera_opt": {
			"optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
			"scheduler": ExponentialDecaySchedulerConfig(
				lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
			),
		},
		"bilateral_grid": {
			"optimizer": AdamOptimizerConfig(lr=5e-3, eps=1e-15),
			"scheduler": ExponentialDecaySchedulerConfig(
				lr_final=1e-4, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
			),
		},
	}


def create_trainer_config(args: GeometryReconstructionArgs) -> TrainerConfig:
	"""Create a trainer configuration from arguments.

	Args:
		args: Training arguments

	Returns:
		TrainerConfig ready for training
	"""
	dataparser_config = HigsHeatDataparserConfig(
		data=Path(args.data_path),
		auto_scale_poses=args.auto_scale_poses,
		center_method=args.center_method,
		orientation_method=args.orientation_method,
	)

	# Create model config
	model_config = Heat2DGSModelConfig(
		max_gaussian_num=args.max_gaussian_num,
		dist_loss=args.dist_loss,
		random_scale=args.random_scale,
		prune_opa=args.prune_opa,
		prune_thermal_opa=args.prune_thermal_opa,
		grow_scale3d=args.grow_scale3d,
		grow_grad2d=args.grow_grad2d,
		use_tonemapping=args.use_tonemapping,
		use_separate_opacities=False,
		random_init=True,  # Typically True for synthetic data
	)

	# Create datamanager config
	MultiFullImageHeatDatamanager = MultiFullImageDatamanager[HeatDataset]
	datamanager_config = MultiFullImageDatamanagerConfig(
		_target=MultiFullImageHeatDatamanager,
		dataparser=dataparser_config,
		image_keys=["image", "heat_image"],
		allow_none=False,
		cache_images_type="float32",
		camera_res_scale_factor=args.image_scale_factor,
	)

	# Create pipeline config
	pipeline_config = VanillaPipelineConfig(
		datamanager=datamanager_config,
		model=model_config,
	)

	# Create trainer config
	trainer_config = TrainerConfig(
		method_name="heat-2dgs",
		experiment_name=args.experiment_name,
		steps_per_eval_image=args.steps_per_eval_image,
		steps_per_eval_batch=0,
		steps_per_save=args.steps_per_save,
		steps_per_eval_all_images=args.steps_per_eval_all_images,
		max_num_iterations=args.num_iters,
		mixed_precision=False,
		pipeline=pipeline_config,
		optimizers=get_optimizers_config(),
		viewer=ViewerConfig(
			num_rays_per_chunk=1 << 15,
			quit_on_train_completion=True,
		),
		vis="viewer" if args.viewer else "tensorboard",
	)
	trainer_config.set_timestamp()

	return trainer_config


def geometry_reconstruction(args: GeometryReconstructionArgs | None = None):
	"""Main training function.

	Args:
		args: Optional pre-configured arguments. If None, will parse from command line.

	Returns:
		Path to the output directory
	"""
	# Parse command line arguments if not provided
	if args is None:
		args = tyro.cli(GeometryReconstructionArgs)


	print("=" * 80)
	print("Geometry Reconstruction Training")
	print("=" * 80)
	print(f"Experiment name: {args.experiment_name}")
	print(f"Output folder: {args.output_dir}")
	print(f"Data path: {args.data_path}")
	print(f"Max Gaussians: {args.max_gaussian_num}")
	print(f"Random scale: {args.random_scale}")
	print(f"Prune opacity: {args.prune_opa}")
	print(f"Prune thermal opacity: {args.prune_thermal_opa}")
	print(f"Grow scale3d: {args.grow_scale3d}")
	print(f"Grow grad2d: {args.grow_grad2d}")
	print(f"Use tonemapping: {args.use_tonemapping}")
	print(f"Max iterations: {args.num_iters}")
	print("=" * 80)

	# Create trainer configuration
	config = create_trainer_config(args)

	# Initialize trainer
	print("\nInitializing trainer...")
	trainer = config.setup(local_rank=0, world_size=1)
	if args.output_dir is not None:
		trainer.base_dir = Path(args.output_dir).expanduser()
	trainer.base_dir.mkdir(parents=True, exist_ok=True)

	# Setup the trainer (this initializes pipeline, datamanager, etc.)
	trainer.setup()

	print("Starting training...")
	trainer.train()

	# Get actual output directory from trainer
	actual_output_dir = trainer.checkpoint_dir.parent

	# Save config.yml to output directory
	config.save_config()

	print("\nTraining completed!")
	print(f"Outputs saved to: {actual_output_dir}")

	return str(actual_output_dir)


if __name__ == "__main__":
	geometry_reconstruction()
