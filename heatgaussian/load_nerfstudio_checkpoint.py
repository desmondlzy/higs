"""Utility for loading nerfstudio 2DGS checkpoints."""

import os
from pathlib import Path

import torch


def load_2dgs_checkpoint(config_path: Path, eval_mode: str = 'test'):
	"""
	Load a nerfstudio 2DGS model checkpoint.

	Args:
		config_path: Path to the nerfstudio config.yml file
		eval_mode: Evaluation mode, either 'test' or 'train'

	Returns:
		tuple: (config, pipeline, checkpoint_path, step)
			- config: Nerfstudio config object
			- pipeline: Nerfstudio pipeline with loaded model
			- checkpoint_path: Path to the loaded checkpoint
			- step: Training step number

	Example:
		>>> from pathlib import Path
		>>> config, pipeline, _, step = load_2dgs_checkpoint(
		...     Path("outputs/heat-2dgs/config.yml")
		... )
		>>> model = pipeline.model
		>>> train_cameras = pipeline.datamanager.train_dataset.cameras
	"""
	from nerfstudio.utils.eval_utils import eval_setup

	# Store current directory and restore after loading
	curr_dir = os.getcwd()

	try:
		config, pipeline, checkpoint_path, step = eval_setup(config_path, eval_mode)
		model = pipeline.model

		print(f"Loaded model from {config_path}")

		trainset = pipeline.datamanager.train_dataset
		train_cams = trainset.cameras

		print(f"Number of cameras: {len(train_cams)}")
		if hasattr(model, 'means'):
			print(f"Number of gaussians: {model.means.shape[0]}")

		torch.cuda.empty_cache()

		return config, pipeline, checkpoint_path, step

	finally:
		# Restore original directory
		os.chdir(curr_dir)
