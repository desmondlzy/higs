"""Adjust camera aspect ratio by cropping."""

import torch
from typing import Literal
from nerfstudio.cameras.cameras import Cameras


def camera_set_aspect_ratio_by_cropping(
	camera: Cameras,
	target_aspect_ratio: float,
	crop_mode: Literal["center", "top", "bottom", "left", "right"] = "center"
):
	"""
	Adjust camera aspect ratio by cropping the viewport.

	Args:
		camera: Camera to modify (modified in-place)
		target_aspect_ratio: Target aspect ratio (width / height)
		crop_mode: Where to align the crop ("center", "top", "bottom", "left", "right")

	Returns:
		Modified camera

	Note:
		aspect_ratio = width / height
	"""
	original_width = camera.width[0].item() if hasattr(camera.width, 'item') else camera.width
	original_height = camera.height[0].item() if hasattr(camera.height, 'item') else camera.height
	original_aspect_ratio = original_width / original_height

	if original_aspect_ratio > target_aspect_ratio:
		# Crop width
		new_width = int(original_height * target_aspect_ratio)
		camera.width = torch.tensor([new_width])

		if crop_mode == "center":
			cx_offset = (original_width - new_width) / 2
			camera.cx = camera.cx - cx_offset
		elif crop_mode == "right":
			cx_offset = (original_width - new_width)
			camera.cx = camera.cx - cx_offset
		elif crop_mode == "left":
			# no change needed
			pass
		else:
			raise ValueError(
				f"crop_mode '{crop_mode}' not supported for ratio "
				f"{original_aspect_ratio:.3f} -> {target_aspect_ratio:.3f}"
			)
	else:
		# Crop height
		new_height = int(original_width / target_aspect_ratio)
		camera.height = torch.tensor([new_height])

		if crop_mode == "center":
			cy_offset = (original_height - new_height) / 2
			camera.cy = camera.cy - cy_offset
		elif crop_mode == "bottom":
			cy_offset = (original_height - new_height)
			camera.cy = camera.cy - cy_offset
		elif crop_mode == "top":
			# no change needed
			pass
		else:
			raise ValueError(
				f"crop_mode '{crop_mode}' not supported for ratio "
				f"{original_aspect_ratio:.3f} -> {target_aspect_ratio:.3f}"
			)

	return camera
