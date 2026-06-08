from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, ForwardRef, Generic, List, Literal, Optional, Tuple, Type, Union, cast, get_args, get_origin
from functools import cached_property

import numpy as np
import torch
from rich.progress import track
from typing_extensions import assert_never

from nerfstudio.utils.misc import get_orig_class
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.cameras.cameras import Cameras

import torch

from nerfstudio.data.datamanagers.full_images_datamanager import (
	FullImageDatamanager,
	FullImageDatamanagerConfig,
	_undistort_image,
	TDataset,
)

@dataclass
class MultiFullImageDatamanagerConfig(FullImageDatamanagerConfig):
	_target: Type = field(default_factory=lambda: MultiFullImageDatamanager)
	image_keys: list[str] = field(default_factory=lambda: ["image"])
	allow_none: bool = False


class MultiFullImageDatamanager(FullImageDatamanager, Generic[TDataset]):
	config: MultiFullImageDatamanagerConfig
	is_synthetic: bool
	dataset_name: str

	train_dataset: TDataset
	eval_dataset: TDataset
	novel_temperature_dataset: Optional[TDataset] = None
	novel_position_dataset: Optional[TDataset] = None

	def __init__(
		self,
		config: MultiFullImageDatamanagerConfig,
		device: Union[torch.device, str] = "cpu",
		test_mode: Literal["test", "val", "inference"] = "val",
		world_size: int = 1,
		local_rank: int = 0,
		**kwargs,
	):
		# Call parent __init__ first to set up train and eval datasets
		super().__init__(
			config=config,
			device=device,
			test_mode=test_mode,
			world_size=world_size,
			local_rank=local_rank,
			**kwargs,
		)

		self.dataset_name = self.train_dataset._dataparser_outputs.metadata["dataset_name"]
		self.is_synthetic = self.train_dataset._dataparser_outputs.metadata["synthetic"]

		# Initialize novel datasets if available
		self.novel_temperature_dataset = self._try_create_dataset("novel-temperature")
		self.novel_position_dataset = self._try_create_dataset("novel-position")

	def _try_create_dataset(self, split: str) -> Optional[TDataset]:
		"""Try to create a dataset for the given split, return None if not available."""
		try:
			outputs = self.dataparser.get_dataparser_outputs(split=split)
			if len(outputs.image_filenames) > 0:
				dataset = self.dataset_type(
					dataparser_outputs=outputs,
					scale_factor=self.config.camera_res_scale_factor,
				)
				CONSOLE.log(f"Loaded {split} dataset with {len(dataset)} images")
				return dataset
		except Exception as e:
			CONSOLE.log(f"Could not load {split} dataset: {e}")
		return None

	def _load_images(
		self, 
		split: Literal["train", "eval", "novel_temperature", "novel_position"], 
		cache_images_device: Literal["cpu", "gpu"]
	) -> List[Dict[str, torch.Tensor]]:
		undistorted_images: List[Dict[str, torch.Tensor]] = []

		# Which dataset?
		match split:
			case "train":
				dataset = self.train_dataset
			case "eval":
				dataset = self.eval_dataset
			case "novel_temperature":
				dataset = self.novel_temperature_dataset
			case "novel_position":
				dataset = self.novel_position_dataset
			case _:
				assert_never(split)

		
		def undistort_idx(idx: int) -> Dict[str, torch.Tensor]:
			data = dataset.get_data(idx, image_type=self.config.cache_images_type)
			camera = dataset.cameras[idx].reshape(())
			assert data["image"].shape[1] == camera.width.item() and data["image"].shape[0] == camera.height.item(), (
				f'The size of image ({data["image"].shape[1]}, {data["image"].shape[0]}) loaded '
				f'does not match the camera parameters ({camera.width.item(), camera.height.item()})'
			)
			if camera.distortion_params is None or torch.all(camera.distortion_params == 0):
				return data
			K = camera.get_intrinsics_matrices().numpy()
			distortion_params = camera.distortion_params.numpy()

			widths, heights = {}, {}
			for key in self.config.image_keys:
				image = data[key].numpy()
				K, image, mask = _undistort_image(camera, distortion_params, data, image, K)
				data[key] = torch.from_numpy(image)

				widths[key] = image.shape[1]
				heights[key] = image.shape[0]

				if mask is not None:
					data["mask"] = mask
				
			# all the image widths and heights should be the same
			if not all([width == widths[self.config.image_keys[0]] for width in widths.values()]):
				raise ValueError(f"All the image widths for the same camera should be the same, but got {widths}")

			if not all([height == heights[self.config.image_keys[0]] for height in heights.values()]):
				raise ValueError(f"All the image heights for the same camera should be the same, but got {heights}")


			dataset.cameras.fx[idx] = float(K[0, 0])
			dataset.cameras.fy[idx] = float(K[1, 1])
			dataset.cameras.cx[idx] = float(K[0, 2])
			dataset.cameras.cy[idx] = float(K[1, 2])
			dataset.cameras.width[idx] = widths[self.config.image_keys[0]]
			dataset.cameras.height[idx] = heights[self.config.image_keys[0]]
			return data

		CONSOLE.log(f"Caching / undistorting {split} images")
		with ThreadPoolExecutor(max_workers=2) as executor:
			undistorted_images = list(
				track(
					executor.map(
						undistort_idx,
						range(len(dataset)),
					),
					description=f"Caching / undistorting {split} images",
					transient=True,
					total=len(dataset),
				)
			)

		# Move to device.
		if cache_images_device == "gpu":
			for cache in undistorted_images:
				for key in self.config.image_keys:
					cache[key] = cache[key].to(self.device)
				if "mask" in cache:
					cache["mask"] = cache["mask"].to(self.device)
				if "depth" in cache:
					cache["depth"] = cache["depth"].to(self.device)
				self.train_cameras = self.train_dataset.cameras.to(self.device)
		elif cache_images_device == "cpu":
			for cache in undistorted_images:
				for key in self.config.image_keys:
					cache[key] = cache[key].pin_memory()
				if "mask" in cache:
					cache["mask"] = cache["mask"].pin_memory()
				self.train_cameras = self.train_dataset.cameras
		else:
			assert_never(cache_images_device)

		return undistorted_images


	@cached_property
	def dataset_type(self) -> Type[TDataset]:
		"""Returns the dataset type passed as the generic argument"""
		orig_class: Type[FullImageDatamanager] = get_orig_class(self, default=None)  # type: ignore
		if orig_class is not None and get_origin(orig_class) is MultiFullImageDatamanager:
			return get_args(orig_class)[0]

		return TDataset
	
	def next_train(self, step) -> Tuple[Cameras, Dict]:
		camera, data = super().next_train(step)
		camera.metadata["subset_idx"] = data["subset_idx"]
		camera.metadata["num_subsets"] = data["num_subsets"]
		return camera, data
	
	def next_eval(self, step) -> Tuple[Cameras, Dict]:
		camera, data = super().next_eval(step)
		camera.metadata["subset_idx"] = data["subset_idx"]
		camera.metadata["num_subsets"] = data["num_subsets"]
		return camera, data
