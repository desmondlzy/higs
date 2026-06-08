from dataclasses import dataclass, field
from typing import Type
from pathlib import Path
import json
import warnings

import imageio
import numpy as np
import torch

from functools import reduce

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.utils.io import load_from_json

from .heatblender_dataparser import HeatDataparserOutputs, HeatBlenderDataParserConfig, HeatBlender
from .heatcolmap_dataparser import HeatColmapDataParserConfig, HeatColmapDataParser
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig, ColmapDataParser, MAX_AUTO_RESOLUTION

@dataclass
class MultiHeatColmapDataParserConfig(ColmapDataParserConfig):
	_target: Type = field(default_factory=lambda: MultiHeatColmapDataParser)
	heatimage_path_pattern: str = "ir*"
	heatimage_suffix: str = ""


class MultiHeatColmapDataParser(ColmapDataParser):
	def __init__(self, config: MultiHeatColmapDataParserConfig):
		super().__init__(config=config)


	def _generate_dataparser_outputs(self, split="train") -> HeatDataparserOutputs | DataparserOutputs:
		# this parse the rgb part of the dataset
		dataparser_outputs: DataparserOutputs = super()._generate_dataparser_outputs(split=split)
		
		# get the heat images path
		heatimage_filenames = []
		subset_indices = []


		warnings.warn("This is a hacky way to force .exr images to reuse the old checkpoints")
		self.config.heatimage_suffix = ".exr"

		# match folders using the pattern
		heatimage_folders = list(self.config.data.glob(self.config.heatimage_path_pattern))

		for ii, image_filename in enumerate(dataparser_outputs.image_filenames):
			for hi, heat_folder in enumerate(heatimage_folders):
				heat_filename = heat_folder / Path(f"{image_filename.name}")
				if self.config.heatimage_suffix != "":
					heat_filename = heat_filename.with_suffix(self.config.heatimage_suffix)

				if heat_filename.exists():
					heatimage_filenames.append(heat_filename)
					subset_indices.append(hi)
					break
			else:
				raise FileNotFoundError(f"Heat image file {heat_filename} does not exist, while {image_filename} exists; {self.config.heatimage_suffix}.")
		
		flir_params_file = heatimage_folders[0] / "parameters.json"
		with open(flir_params_file) as f:
			flir_params = json.load(f)

		metadata = {
			**dataparser_outputs.metadata,
			"subset_idx": torch.tensor(subset_indices, dtype=torch.int32),
			"num_subsets": len(heatimage_folders),
			**flir_params,
		}

		assert len(heatimage_filenames) == len(dataparser_outputs.image_filenames), f"length mismatch: {len(heatimage_filenames)} != {len(dataparser_outputs.image_filenames)}"

		return HeatDataparserOutputs(
			image_filenames=dataparser_outputs.image_filenames,
			cameras=dataparser_outputs.cameras,
			alpha_color=dataparser_outputs.alpha_color,
			scene_box=dataparser_outputs.scene_box,
			mask_filenames=dataparser_outputs.mask_filenames,
			metadata=metadata,
			dataparser_transform=dataparser_outputs.dataparser_transform,
			dataparser_scale=dataparser_outputs.dataparser_scale,
			heatimage_filenames=heatimage_filenames,
			reflection_filenames=[None for _ in heatimage_filenames],
		)