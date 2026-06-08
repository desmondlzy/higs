"""Load nerfstudio 2DGS checkpoint."""

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
	"""
	from nerfstudio.utils.eval_utils import eval_setup

	curr_dir = os.getcwd()
	try:
		config, pipeline, checkpoint_path, step = eval_setup(config_path, eval_mode)
		model = pipeline.model

		print(f"Loaded model from config: {config_path}; checkpoint: {checkpoint_path}; step: {step}")

		trainset = pipeline.datamanager.train_dataset
		train_cams = trainset.cameras

		print(f"Number of cameras: {len(train_cams)}")
		if hasattr(model, 'means'):
			print(f"Number of gaussians: {model.means.shape[0]}")

		torch.cuda.empty_cache()

		return config, pipeline, checkpoint_path, step

	finally:
		os.chdir(curr_dir)
