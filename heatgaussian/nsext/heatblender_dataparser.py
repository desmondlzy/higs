from dataclasses import dataclass, field
from typing import Type, Literal
from pathlib import Path

import imageio
import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras import camera_utils
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig, Blender



@dataclass
class HeatDataparserOutputs(DataparserOutputs):
	heatimage_filenames: list[Path] = field(default_factory=list)
	"""Filenames for the heat images."""

	reflection_filenames: list[Path | None] = field(default_factory=list)
	"""Filenames for the reflection images."""


@dataclass
class HeatBlenderDataParserConfig(BlenderDataParserConfig):
	_target: Type = field(default_factory=lambda: HeatBlender)

	load_heatimages: bool = True

	thermal_image_key: str = "heat_img_path"

	reflection_image_key: str = "reflection"

	auto_scale_poses: bool = True

	center_method: Literal["poses", "focus", "none"] = "poses"

	orientation_method: Literal["pca", "up", "vertical", "none"] = "up"


@dataclass
class HeatBlender(Blender):
	config: HeatBlenderDataParserConfig

	def __init__(self, config: HeatBlenderDataParserConfig):
		super().__init__(config=config)


	def _generate_dataparser_outputs(self, split="train", test_every=None):
		meta = load_from_json(self.data / f"transforms_{split}.json")
		image_filenames = []
		heatimage_filenames = []
		reflection_filenames = []

		poses = []

		if len(meta["frames"]) > 0:
			for i, frame in enumerate(meta["frames"]):
				img_fname = self.data / Path(frame["file_path"].replace("./", ""))
				heat_fname = self.data / Path(frame[self.config.thermal_image_key].replace("./", ""))

				image_filenames.append(img_fname)
				heatimage_filenames.append(heat_fname)

				# real capture datasets may not have reflection images
				if self.config.reflection_image_key in frame:
					refl_fname = self.data / Path(frame[self.config.reflection_image_key].replace("./", ""))
					reflection_filenames.append(refl_fname)
				else:
					# try look in the "reflection" dir using the same filename as the rgb
					refl_fname = heat_fname.parent.parent / "reflection" / heat_fname.name
					if refl_fname.exists():
						reflection_filenames.append(refl_fname)
					else:
						reflection_filenames.append(None)

				poses.append(np.array(frame["transform_matrix"]))


			poses = torch.from_numpy(np.array(poses).astype(np.float32))
			poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
				poses,
				method=self.config.orientation_method,
				center_method=self.config.center_method,
			)

			# Scale poses
			scale_factor = 1.0
			if self.config.auto_scale_poses:
				scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))

			scale_factor *= self.config.scale_factor

			poses[:, :3, 3] *= scale_factor

			img_0 = imageio.v2.imread(image_filenames[0])
			image_height, image_width = img_0.shape[:2]
			camera_angle_x = float(meta["camera_angle_x"])
			focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

			cx = image_width / 2.0
			cy = image_height / 2.0

			camera_to_world = poses[:, :3, :4]  # camera to world transform

			# in x,y,z order
			# camera_to_world[..., 3] *= self.scale_factor
			scene_box = SceneBox(aabb=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32))

			cameras = Cameras(
				camera_to_worlds=camera_to_world,
				fx=focal_length,
				fy=focal_length,
				cx=cx,
				cy=cy,
				camera_type=CameraType.PERSPECTIVE,
			)

			metadata = {}
			if self.config.ply_path is not None:
				metadata.update(self._load_3D_points(self.config.data / self.config.ply_path))

			
			assert len(image_filenames) == len(heatimage_filenames) == len(reflection_filenames) 

			dataparser_outputs = HeatDataparserOutputs(
				image_filenames=image_filenames,
				heatimage_filenames=heatimage_filenames,
				reflection_filenames=reflection_filenames,
				cameras=cameras,
				alpha_color=self.alpha_color_tensor,
				scene_box=scene_box,
				dataparser_scale=self.scale_factor,
				metadata=metadata,
			)


			return dataparser_outputs
		
		else:
			return HeatDataparserOutputs(
				image_filenames=[],
				heatimage_filenames=[],
				reflection_filenames=[],
				cameras=Cameras(
					camera_to_worlds=torch.empty((0, 3, 4)),
					fx=torch.empty((0,)),
					fy=torch.empty((0,)),
					cx=torch.empty((0,)),
					cy=torch.empty((0,)),
					camera_type=CameraType.PERSPECTIVE,
				),
				alpha_color=self.alpha_color_tensor,
				scene_box=SceneBox(aabb=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32)),
				dataparser_scale=1.0,
				metadata={},
			)
