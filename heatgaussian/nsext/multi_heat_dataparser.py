from dataclasses import dataclass, field
from typing import Type, Literal
from pathlib import Path
import json

import imageio
import numpy as np
import torch

from functools import reduce

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.utils.io import load_from_json

from .heatblender_dataparser import HeatDataparserOutputs, HeatBlenderDataParserConfig, HeatBlender
from .heatcolmap_dataparser import HeatColmapDataParserConfig, HeatColmapDataParser


def concat_cameras(cameras: list[Cameras]):
	merged_cameras = Cameras(
		camera_to_worlds=torch.concatenate([c.camera_to_worlds for c in cameras], dim=0),
		fx=torch.concatenate([c.fx for c in cameras], dim=0),
		fy=torch.concatenate([c.fy for c in cameras], dim=0),
		cx=torch.concatenate([c.cx for c in cameras], dim=0),
		cy=torch.concatenate([c.cy for c in cameras], dim=0),
		width=torch.concatenate([c.width for c in cameras], dim=0),
		height=torch.concatenate([c.height for c in cameras], dim=0),
	)

	return merged_cameras

def concat_lists_or_none(lists_or_none: list[list] | None):
	if lists_or_none[0] is None:
		if all(item is None for item in lists_or_none):
			return None
		else:
			raise ValueError(f"Lists are not all None, {lists_or_none}")

	return reduce(lambda x, y: x + y, lists_or_none)


def take_first_if_all_same(items, compare=None):
	if compare == None:
		compare = lambda x, y: x == y
	
	# check all types are the same
	if not all(type(x) == type(items[0]) for x in items):
		raise TypeError(f"Items are not all the same type, {items}")
	
	match type(items[0]):
		case torch.Tensor:
			if all(torch.allclose(x, items[0]) for x in items):
				return items[0]
		case _:
			if all(compare(x, items[0]) for x in items):
				return items[0]

	raise ValueError(f"Items are not all the same, {items}")


@dataclass
class MultiHeatDataparserConfig(DataParserConfig):
	_target: Type = field(default_factory=lambda: MultiHeatDataparser)
	subset_map_filename: str = "subsets.json"

	auto_scale_poses: bool = False

	center_method: Literal["poses", "focus", "none"] = "none"

	orientation_method: Literal["pca", "up", "vertical", "none"] = "up"


class MultiHeatDataparser(DataParser):
	config: MultiHeatDataparserConfig

	def __init__(self, config: MultiHeatDataparserConfig):
		super().__init__(config)

	
	def _generate_dataparser_outputs(self, split="train") -> HeatDataparserOutputs:
		if split == "val": 
			return self._generate_dataparser_outputs(split="test")

		# read the subset map
		subset_map = load_from_json(self.config.data / self.config.subset_map_filename)

		num_train_subsets = len([s for s in subset_map["subsets"].values() if s["split"] == "train"])

		image_filenames_per_subset = []
		heatimage_filenames_per_subset = []
		reflection_filenames_per_subset = []
		cameras_per_subset = []
		alpha_color_per_subset = []
		scene_box_per_subset = []
		mask_filenames_per_subset = []
		metadata_per_subset = []
		dataparser_transform_per_subset = []
		dataparser_scale_per_subset = []

		valid_subsets = [s for s in subset_map["subsets"].values() if s["split"] == split]

		for index, subset_params in enumerate(valid_subsets):
			# get the subset data
			subset_type = subset_params["type"]

			# prefix with the root of dir to this dataset
			subset_data_root = self.config.data / subset_params["data"]

			if subset_type == "heat-blender":
				subparser_config = HeatBlenderDataParserConfig(
					data=subset_data_root,
					auto_scale_poses=self.config.auto_scale_poses,
					center_method=self.config.center_method,
					orientation_method=self.config.orientation_method,
				)
				subparser = HeatBlender(subparser_config)
			elif subset_type == "heat-colmap":
				subparser_config = HeatColmapDataParserConfig(
					data=subset_data_root,
				)
				subparser = HeatColmapDataParser(subparser_config)
			else:
				raise ValueError(f"Unknown subset type {subset_type}")
			
			subparser_outputs = subparser._generate_dataparser_outputs(split=split)

			n_data = len(subparser_outputs.image_filenames)

			if n_data > 0:
				image_filenames_per_subset.append(subparser_outputs.image_filenames)
				heatimage_filenames_per_subset.append(subparser_outputs.heatimage_filenames)
				reflection_filenames_per_subset.append(subparser_outputs.reflection_filenames)
				cameras_per_subset.append(subparser_outputs.cameras)
				alpha_color_per_subset.append(subparser_outputs.alpha_color)
				scene_box_per_subset.append(subparser_outputs.scene_box)
				mask_filenames_per_subset.append(subparser_outputs.mask_filenames)
				metadata_per_subset.append({
					**subparser_outputs.metadata,
					"subset_idx": torch.full((n_data,), index, dtype=torch.int32),
				})
				dataparser_transform_per_subset.append(subparser_outputs.dataparser_transform)
				dataparser_scale_per_subset.append(subparser_outputs.dataparser_scale)
			else:
				print(f"Reading {split} in subset {subset_params['data']}, has no data, skipping...")


		merged_metadata = {
			key: torch.concatenate([metadata[key] for metadata in metadata_per_subset]) for key in metadata_per_subset[0].keys()
		}
		merged_metadata["num_subsets"] = num_train_subsets

		heat_ranges = load_from_json(self.config.data / "heat_ranges.json")
		merged_metadata.update(heat_ranges)


		concated_image_filenames = concat_lists_or_none(image_filenames_per_subset)
		concated_heatimage_filenames = concat_lists_or_none(heatimage_filenames_per_subset)
		concated_reflection_filenames = concat_lists_or_none(reflection_filenames_per_subset)

		assert len(concated_image_filenames) == len(concated_heatimage_filenames) == len(concated_reflection_filenames)

		return HeatDataparserOutputs(
			image_filenames=concated_image_filenames,
			heatimage_filenames=concated_heatimage_filenames,
			reflection_filenames=concated_reflection_filenames,
			cameras=concat_cameras(cameras_per_subset),
			alpha_color=take_first_if_all_same(alpha_color_per_subset),
			scene_box=take_first_if_all_same(scene_box_per_subset, compare=lambda x, y: torch.allclose(x.aabb, y.aabb)),
			mask_filenames=concat_lists_or_none(mask_filenames_per_subset),
			dataparser_scale=take_first_if_all_same(dataparser_scale_per_subset),
			metadata=merged_metadata,
			dataparser_transform=take_first_if_all_same(dataparser_transform_per_subset, compare=lambda x, y: torch.allclose(x, y)),
		)

		
