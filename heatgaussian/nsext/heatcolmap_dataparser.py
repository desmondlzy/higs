from dataclasses import dataclass, field
from typing import Type
from pathlib import Path
from functools import partial
import sys

import imageio
from PIL import Image
import numpy as np
from rich.prompt import Confirm

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.rich_utils import CONSOLE, status
from nerfstudio.utils.io import load_from_json
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig, ColmapDataParser, MAX_AUTO_RESOLUTION

from .heatblender_dataparser import HeatDataparserOutputs


@dataclass
class HeatColmapDataParserConfig(ColmapDataParserConfig):
	_target: Type = field(default_factory=lambda: HeatColmapDataParser)
	heatimage_key: Path = Path("heat_img_path")
	heatimage_path: Path = Path("input")
	filrcsv_path: Path = Path("csv")


class HeatColmapDataParser(ColmapDataParser):
	config: HeatColmapDataParserConfig

	def __init__(self, config: HeatColmapDataParserConfig):
		super().__init__(config=config)


	def _generate_dataparser_outputs(self, split="train") -> HeatDataparserOutputs:
		dataparser_outputs: DataparserOutputs = super()._generate_dataparser_outputs(split=split)

		if self._downscale_factor is not None and self._downscale_factor != 1:
			raise NotImplementedError(f"Downscaling is not supported for heat images. but get {self._downscale_factor = }")
		
	
		if self.config.downscale_factor is not None and self.config.downscale_factor != 1:
			raise NotImplementedError(f"Downscaling is not supported for heat images. but get {self.config.downscale_factor = }.")

		# get the heat images path
		heatimage_filenames = []
		reflection_filenames = []
		for i, image_filename in enumerate(dataparser_outputs.image_filenames):
			heat_filename = self.config.data / self.config.heatimage_path / Path(f"{image_filename.name}")

			if not heat_filename.exists():
				raise FileNotFoundError(f"Heat image file {heat_filename} does not exist, while {image_filename} exists.")

			heatimage_filenames.append(heat_filename)
			reflection_filenames.append(Path(""))  # no reflection images for colmap datasets
		
		
		return HeatDataparserOutputs(
			image_filenames=dataparser_outputs.image_filenames,
			cameras=dataparser_outputs.cameras,
			alpha_color=dataparser_outputs.alpha_color,
			scene_box=dataparser_outputs.scene_box,
			mask_filenames=dataparser_outputs.mask_filenames,
			metadata=dataparser_outputs.metadata,
			dataparser_transform=dataparser_outputs.dataparser_transform,
			dataparser_scale=dataparser_outputs.dataparser_scale,
			heatimage_filenames=heatimage_filenames,
			reflection_filenames=reflection_filenames,
		)


	def downscale_heatimages(
		self, 
		heatimage_filenames: list[Path],
	):
		"""
		Setup the downscale factor for the dataset. This is used to downscale the images and cameras.
		"""

		def get_fname(parent: Path, filepath: Path) -> Path:
			"""Returns transformed file name when downscale factor is applied"""
			rel_part = filepath.relative_to(parent)
			base_part = parent.parent / (str(parent.name) + f"_{self._downscale_factor}")
			return base_part / rel_part

		filepath = next(iter(heatimage_filenames))
		if self._downscale_factor is None:
			if self.config.downscale_factor is None:
				test_img = Image.open(filepath)
				w, h = test_img.size
				max_res = max(h, w)
				df = 0
				while True:
					if (max_res / 2 ** (df)) <= MAX_AUTO_RESOLUTION:
						break
					df += 1

				self._downscale_factor = 2**df
				CONSOLE.log(f"Using image downscale factor of {self._downscale_factor}")
			else:
				self._downscale_factor = self.config.downscale_factor
			if self._downscale_factor > 1 and not all(
				get_fname(self.config.data / self.config.heatimage_path, fp).parent.exists() for fp in heatimage_filenames
			):
				# Downscaled images not found
				# Ask if user wants to downscale the images automatically here
				CONSOLE.print(
					f"[bold red]Downscaled images do not exist for factor of {self._downscale_factor}.[/bold red]"
				)
				if Confirm.ask(
					f"\nWould you like to downscale the images using '{self.config.downscale_rounding_mode}' rounding mode now?",
					default=False,
					console=CONSOLE,
				):
					# Install the method
					self._downscale_images(
						heatimage_filenames,
						partial(get_fname, self.config.data / self.config.heatimages_path),
						self._downscale_factor,
						self.config.downscale_rounding_mode,
						nearest_neighbor=False,
					)
				else:
					sys.exit(1)

		# Return transformed filenames
		if self._downscale_factor > 1:
			heatimage_filenames = [get_fname(self.config.data / self.config.heatimages_path, fp) for fp in heatimage_filenames]

		assert isinstance(self._downscale_factor, int)
		return heatimage_filenames
