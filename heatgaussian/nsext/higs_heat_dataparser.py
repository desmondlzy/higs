"""
HIGS heat dataparser for datasets with unified metadata format.

This dataparser handles datasets organized as:
- Root metadata.json defining subsets and their splits
- Each subset is a subdirectory with heat-blender format data
- Supports custom split types: train, test, novel-temperature, novel-position, novel-temperature-position
"""

from dataclasses import dataclass, field
from typing import Type, Literal
from pathlib import Path

import torch
import numpy as np
from functools import reduce

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig
from nerfstudio.data.dataparsers.colmap_dataparser import colmap_utils, parse_colmap_camera_params, CAMERA_MODEL_TO_TYPE
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataparsers_utils import (
	get_train_eval_split_fraction,
	get_train_eval_split_all,
)
from nerfstudio.utils.io import load_from_json

from .heatblender_dataparser import HeatDataparserOutputs, HeatBlenderDataParserConfig, HeatBlender


def concat_cameras(cameras: list[Cameras]):
	"""Concatenate multiple camera objects into one."""
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
	"""Concatenate lists, or return None if all are None."""
	if lists_or_none[0] is None:
		if all(item is None for item in lists_or_none):
			return None
		else:
			raise ValueError(f"Lists are not all None, {lists_or_none}")
	return reduce(lambda x, y: x + y, lists_or_none)


def take_first_if_all_same(items, compare=None):
	"""Take the first item if all items are the same."""
	if compare is None:
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
class HigsHeatDataparserConfig(DataParserConfig):
	"""Configuration for HIGS heat dataparser."""

	_target: Type = field(default_factory=lambda: HigsHeatDataparser)

	metadata_filename: str = "metadata.json"
	"""Name of the metadata file containing subset definitions"""

	auto_scale_poses: bool = False
	"""Auto-scale camera poses"""

	center_method: Literal["poses", "focus", "none"] = "none"
	"""Method for centering poses: 'focus', 'poses', 'none'"""

	orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
	"""Method for orienting poses: 'pca', 'up', 'vertical', 'none'"""

	eval_mode: Literal["fraction", "all"] = "all"
	"""How to generate eval/test set when 'test' split is not defined:
	- 'all': Use entire train set as test set (default)
	- 'fraction': Split train set based on train_split_fraction"""

	train_split_fraction: float = 0.9
	"""Fraction of train data to use for training when eval_mode='fraction'"""


class HigsHeatDataparser(DataParser):
	"""Dataparser for HIGS heat datasets with unified metadata format."""

	config: HigsHeatDataparserConfig

	def __init__(self, config: HigsHeatDataparserConfig):
		super().__init__(config)

	def _load_heat_colmap_subset(self, subset_data_root: Path) -> HeatDataparserOutputs:
		"""Load data from a heat-colmap format subset.

		Camera poses and RGB images are shared at the dataset root (sparse/ and rgb/).
		The subset directory contains thermal .exr files named identically to the RGB images.
		Only frames that have a matching thermal image in the subset are included.

		Args:
			subset_data_root: Path to the subset directory (contains .exr files)

		Returns:
			HeatDataparserOutputs for this subset
		"""
		import imageio

		colmap_path = self.config.data / "sparse" / "0"
		assert colmap_path.exists(), f"COLMAP path not found: {colmap_path}"

		# Read COLMAP reconstruction
		if (colmap_path / "cameras.bin").exists():
			cam_id_to_camera = colmap_utils.read_cameras_binary(colmap_path / "cameras.bin")
			im_id_to_image = colmap_utils.read_images_binary(colmap_path / "images.bin")
		else:
			cam_id_to_camera = colmap_utils.read_cameras_text(colmap_path / "cameras.txt")
			im_id_to_image = colmap_utils.read_images_text(colmap_path / "images.txt")

		# Parse camera intrinsics per camera id
		cameras_params = {cam_id: parse_colmap_camera_params(cam) for cam_id, cam in cam_id_to_camera.items()}

		# Build set of thermal stems present in this subset
		thermal_stems = {p.stem for p in subset_data_root.glob("*.exr")}

		rgb_dir = self.config.data / "rgb"

		image_filenames = []
		heatimage_filenames = []
		poses = []
		fx_list, fy_list, cx_list, cy_list, h_list, w_list = [], [], [], [], [], []

		for im_id in sorted(im_id_to_image.keys()):
			im_data = im_id_to_image[im_id]
			stem = Path(im_data.name).stem

			# Skip frames with no thermal image in this subset
			if stem not in thermal_stems:
				continue

			# Pose: COLMAP w2c → c2w, OpenCV → OpenGL
			rotation = colmap_utils.qvec2rotmat(im_data.qvec)
			translation = im_data.tvec.reshape(3, 1)
			w2c = np.concatenate([rotation, translation], 1)
			w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
			c2w = np.linalg.inv(w2c)
			c2w[0:3, 1:3] *= -1  # OpenCV → OpenGL

			cam_params = cameras_params[im_data.camera_id]
			camera_type = CAMERA_MODEL_TO_TYPE[cam_params["camera_model"]]

			image_filenames.append(rgb_dir / im_data.name)
			heatimage_filenames.append(subset_data_root / f"{stem}.exr")
			poses.append(c2w)

			fx_list.append(float(cam_params["fl_x"]))
			fy_list.append(float(cam_params["fl_y"]))
			cx_list.append(float(cam_params["cx"]))
			cy_list.append(float(cam_params["cy"]))
			h_list.append(int(cam_params["h"]))
			w_list.append(int(cam_params["w"]))

		assert len(image_filenames) > 0, f"No frames found in subset {subset_data_root}"

		poses = torch.from_numpy(np.array(poses).astype(np.float32))
		poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
			poses,
			method=self.config.orientation_method,
			center_method=self.config.center_method,
		)

		scale_factor = 1.0
		if self.config.auto_scale_poses:
			scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
		poses[:, :3, 3] *= scale_factor

		scene_box = SceneBox(
			aabb=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32)
		)

		cameras = Cameras(
			camera_to_worlds=poses[:, :3, :4],
			fx=torch.tensor(fx_list, dtype=torch.float32),
			fy=torch.tensor(fy_list, dtype=torch.float32),
			cx=torch.tensor(cx_list, dtype=torch.float32),
			cy=torch.tensor(cy_list, dtype=torch.float32),
			width=torch.tensor(w_list, dtype=torch.int32),
			height=torch.tensor(h_list, dtype=torch.int32),
			camera_type=camera_type,
		)

		return HeatDataparserOutputs(
			image_filenames=image_filenames,
			heatimage_filenames=heatimage_filenames,
			reflection_filenames=[None] * len(image_filenames),
			cameras=cameras,
			alpha_color=None,
			scene_box=scene_box,
			mask_filenames=None,
			dataparser_scale=scale_factor,
			metadata={},
			dataparser_transform=transform_matrix,
		)

	def _load_heat_blender_subset(self, subset_data_root: Path) -> HeatDataparserOutputs:
		"""Load data from a heat-blender format subset with transforms.json.

		Args:
			subset_data_root: Path to the subset directory

		Returns:
			HeatDataparserOutputs for this subset
		"""
		import imageio

		# Load transforms.json
		transforms_path = subset_data_root / "transforms.json"
		if not transforms_path.exists():
			raise FileNotFoundError(f"transforms.json not found in {subset_data_root}")

		meta = load_from_json(transforms_path)

		image_filenames = []
		heatimage_filenames = []
		reflection_filenames = []
		poses = []

		# Process each frame
		for frame in meta["frames"]:
			img_fname = subset_data_root / Path(frame["file_path"].replace("./", ""))
			heat_fname = subset_data_root / Path(frame.get("heat_img_path", "").replace("./", ""))

			image_filenames.append(img_fname)
			heatimage_filenames.append(heat_fname)

			# Handle reflection images
			if "reflection" in frame:
				refl_fname = subset_data_root / Path(frame["reflection"].replace("./", ""))
				reflection_filenames.append(refl_fname)
			else:
				# Try to find reflection in the reflection directory
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

		poses[:, :3, 3] *= scale_factor

		# Get camera parameters from first image
		img_0 = imageio.v2.imread(image_filenames[0])
		image_height, image_width = img_0.shape[:2]
		camera_angle_x = float(meta["camera_angle_x"])
		focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)
		cx = image_width / 2.0
		cy = image_height / 2.0

		camera_to_world = poses[:, :3, :4]

		scene_box = SceneBox(
			aabb=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32)
		)

		cameras = Cameras(
			camera_to_worlds=camera_to_world,
			fx=focal_length,
			fy=focal_length,
			cx=cx,
			cy=cy,
			camera_type=CameraType.PERSPECTIVE,
		)

		return HeatDataparserOutputs(
			image_filenames=image_filenames,
			heatimage_filenames=heatimage_filenames,
			reflection_filenames=reflection_filenames,
			cameras=cameras,
			alpha_color=None,
			scene_box=scene_box,
			dataparser_scale=scale_factor,
			metadata={},
			dataparser_transform=transform_matrix,
		)

	def _apply_train_test_split(self, outputs: HeatDataparserOutputs, split: str) -> HeatDataparserOutputs:
		"""Apply train/test split to dataparser outputs using nerfstudio utilities.

		Args:
			outputs: Full dataparser outputs
			split: Either "train" or "test" to determine which portion to return

		Returns:
			Subset of outputs for train or test
		"""
		num_images = len(outputs.image_filenames)

		# Use nerfstudio's built-in split utilities
		if self.config.eval_mode == "fraction":
			i_train, i_eval = get_train_eval_split_fraction(
				outputs.image_filenames,
				self.config.train_split_fraction
			)
		elif self.config.eval_mode == "all":
			i_train, i_eval = get_train_eval_split_all(outputs.image_filenames)
		else:
			raise ValueError(f"Unknown eval_mode: {self.config.eval_mode}")

		# Select indices based on split
		if split == "train":
			indices = i_train
		else:  # split == "test"
			indices = i_eval

		print(f"[{split}] Using {len(indices)} / {num_images} images")

		# Select subset of data
		indices_list = indices.tolist()

		# Handle metadata tensors
		new_metadata = {}
		for k, v in outputs.metadata.items():
			if isinstance(v, torch.Tensor) and len(v) == num_images:
				new_metadata[k] = v[torch.tensor(indices)]
			else:
				new_metadata[k] = v

		return HeatDataparserOutputs(
			image_filenames=[outputs.image_filenames[i] for i in indices_list],
			heatimage_filenames=[outputs.heatimage_filenames[i] for i in indices_list],
			reflection_filenames=[outputs.reflection_filenames[i] for i in indices_list],
			cameras=outputs.cameras[torch.tensor(indices)],
			alpha_color=outputs.alpha_color,
			scene_box=outputs.scene_box,
			mask_filenames=[outputs.mask_filenames[i] for i in indices_list] if outputs.mask_filenames else None,
			dataparser_scale=outputs.dataparser_scale,
			metadata=new_metadata,
			dataparser_transform=outputs.dataparser_transform,
		)

	def _generate_dataparser_outputs(self, split="train") -> HeatDataparserOutputs:
		"""Generate dataparser outputs for the specified split.

		Args:
			split: One of "train", "test", "val", "novel-temperature",
				   "novel-position", "novel-temperature-position"

		Returns:
			HeatDataparserOutputs containing all data for the split
		"""
		# Map "val" to "test" for compatibility
		if split == "val":
			split = "test"

		# Load metadata
		metadata_path = self.config.data / self.config.metadata_filename
		assert metadata_path.exists(), f"Metadata file not found at {metadata_path}"

		metadata = load_from_json(metadata_path)

		# Check if test split exists in metadata
		has_test_split = any(
			subset_params["split"] == "test"
			for subset_params in metadata["subsets"].values()
		)

		# Handle test split generation when not defined in metadata
		if split == "test" and not has_test_split:
			if self.config.eval_mode == "all":
				# Use entire train set as test set
				print("No 'test' split found in metadata. Using entire train set as test set (eval_mode='all').")
				return self._generate_dataparser_outputs(split="train")
			elif self.config.eval_mode == "fraction":
				# Generate train/test split from train data
				print(f"No 'test' split found in metadata. Splitting train set with fraction={self.config.train_split_fraction} (eval_mode='fraction').")
				train_outputs = self._load_split_data(split="train")
				return self._apply_train_test_split(train_outputs, split="test")

		# For train split with fraction mode, apply the split
		if split == "train" and self.config.eval_mode == "fraction" and not has_test_split:
			train_outputs = self._load_split_data(split="train")
			return self._apply_train_test_split(train_outputs, split="train")

		# Load data normally for other splits or when test split is defined
		return self._load_split_data(split=split)

	def _load_split_data(self, split="train") -> HeatDataparserOutputs:
		"""Load data for a specific split from metadata.

		Args:
			split: Split name to load

		Returns:
			HeatDataparserOutputs containing all data for the split
		"""
		# Load metadata
		root_metadata_path = self.config.data / self.config.metadata_filename
		root_metadata = load_from_json(root_metadata_path)

		# Get subset type (e.g., "heat-blender")
		subset_type = root_metadata.get("type", "heat-blender")

		# Collect data from all subsets matching the requested split
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

		# Filter subsets by split
		valid_subsets = [
			(subset_name, subset_params)
			for subset_name, subset_params in root_metadata["subsets"].items()
			if subset_params["split"] == split
		]

		if len(valid_subsets) == 0:
			print(f"Warning: No subsets found for split '{split}'")
			# Return empty dataparser outputs
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
					width=torch.empty((0,)),
					height=torch.empty((0,)),
				),
				scene_box=None,
				metadata={},
			)

		# Count training subsets for metadata
		num_train_subsets = len([
			s for s in root_metadata["subsets"].values()
			if s["split"] == "train"
		])

		# Process each subset
		for subset_idx, (subset_name, subset_params) in enumerate(valid_subsets):
			# Get subset data directory
			subset_data_root = self.config.data / subset_params["data"]

			# Create appropriate subparser based on subset type
			if subset_type == "heat-blender":
				subparser_outputs = self._load_heat_blender_subset(subset_data_root)
			elif subset_type == "heat-colmap":
				subparser_outputs = self._load_heat_colmap_subset(subset_data_root)
			else:
				raise ValueError(f"Unknown subset type: {subset_type}")

			n_data = len(subparser_outputs.image_filenames)

			if n_data > 0:
				image_filenames_per_subset.append(subparser_outputs.image_filenames)
				heatimage_filenames_per_subset.append(subparser_outputs.heatimage_filenames)
				reflection_filenames_per_subset.append(subparser_outputs.reflection_filenames)
				cameras_per_subset.append(subparser_outputs.cameras)
				alpha_color_per_subset.append(subparser_outputs.alpha_color)
				scene_box_per_subset.append(subparser_outputs.scene_box)
				mask_filenames_per_subset.append(subparser_outputs.mask_filenames)

				# Add subset metadata
				boundary_temperature = subset_params.get("boundary_temperature", float("nan"))
				displacement = subset_params.get("displacement", [0.0, 0.0, 0.0])

				subset_metadata = {
					**subparser_outputs.metadata,
					"subset_idx": torch.full((n_data,), subset_idx, dtype=torch.int32),
					"subset_name": subset_name,
					"boundary_temperature": torch.full((n_data,), boundary_temperature, dtype=torch.float32),
					"displacement": torch.tensor(displacement, dtype=torch.float32).unsqueeze(0).expand(n_data, 3).contiguous(),
				}
				metadata_per_subset.append(subset_metadata)

				dataparser_transform_per_subset.append(subparser_outputs.dataparser_transform)
				dataparser_scale_per_subset.append(subparser_outputs.dataparser_scale)
			else:
				print(f"Warning: Subset '{subset_name}' has no data, skipping...")

		# Check if we have any data
		if len(metadata_per_subset) == 0:
			print(f"Warning: No data found for split '{split}'")
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
					width=torch.empty((0,)),
					height=torch.empty((0,)),
				),
				scene_box=None,
				metadata={},
			)

		# Merge metadata from all subsets
		merged_metadata = {
			key: torch.concatenate([m[key] for m in metadata_per_subset])
			for key in metadata_per_subset[0].keys()
			if key != "subset_name"  # Don't concatenate string values
		}
		merged_metadata["num_subsets"] = num_train_subsets

		# Add subset names as a list
		merged_metadata["subset_names"] = [m["subset_name"] for m in metadata_per_subset]

		# Load heat ranges if available
		heat_ranges_path = self.config.data / "heat_ranges.json"
		if not heat_ranges_path.exists():
			# Try loading from first subset
			find_heat_ranges = False
			if len(valid_subsets) > 0:
				subset_heat_ranges = self.config.data / valid_subsets[0][1]["data"] / "heat_ranges.json"
				if subset_heat_ranges.exists():
					find_heat_ranges = True
					heat_ranges = load_from_json(subset_heat_ranges)
					merged_metadata.update(heat_ranges)
			if not find_heat_ranges:
				print(f"Warning: heat_ranges.json not found at {heat_ranges_path} or in any subset. Heat ranges will not be included in metadata.")
				raise FileNotFoundError(f"heat_ranges.json not found at {heat_ranges_path} or in any subset.")
		else:
			heat_ranges = load_from_json(heat_ranges_path)
			merged_metadata.update(heat_ranges)

		# Concatenate all data
		concatenated_image_filenames = concat_lists_or_none(image_filenames_per_subset)
		concatenated_heatimage_filenames = concat_lists_or_none(heatimage_filenames_per_subset)
		concatenated_reflection_filenames = concat_lists_or_none(reflection_filenames_per_subset)

		assert len(concatenated_image_filenames) == len(concatenated_heatimage_filenames) == len(concatenated_reflection_filenames)

		# Note: For HIGS datasets, each subset may have different dataparser transforms
		# since they might represent different thermal conditions. We'll use the first one
		# or create an identity transform if they differ.
		try:
			dataparser_transform = take_first_if_all_same(
				dataparser_transform_per_subset,
				compare=lambda x, y: torch.allclose(x, y, atol=1e-3)
			)
		except ValueError:
			# Transforms differ across subsets, use identity
			dataparser_transform = torch.eye(4, dtype=torch.float32)[:3, :]
		
		assert "name" in root_metadata, "dataset_name should be defined in root metadata"
		assert "synthetic" in root_metadata, "synthetic flag should be defined in root metadata"

		combined_metadata = {
			**merged_metadata,
			"dataset_name": root_metadata["name"],
			"synthetic": root_metadata["synthetic"],
		}

		return HeatDataparserOutputs(
			image_filenames=concatenated_image_filenames,
			heatimage_filenames=concatenated_heatimage_filenames,
			reflection_filenames=concatenated_reflection_filenames,
			cameras=concat_cameras(cameras_per_subset),
			alpha_color=take_first_if_all_same(alpha_color_per_subset) if alpha_color_per_subset[0] is not None else None,
			scene_box=take_first_if_all_same(scene_box_per_subset, compare=lambda x, y: torch.allclose(x.aabb, y.aabb)),
			mask_filenames=concat_lists_or_none(mask_filenames_per_subset),
			dataparser_scale=take_first_if_all_same(dataparser_scale_per_subset),
			metadata=combined_metadata,
			dataparser_transform=dataparser_transform,
		)
