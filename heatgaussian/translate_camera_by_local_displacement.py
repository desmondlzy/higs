"""Translate camera in local coordinates."""

import torch
from nerfstudio.cameras.cameras import Cameras


def translate_camera_by_local_displacement(
	camera: Cameras,
	translate: torch.Tensor,
	copy: bool = False,
):
	"""
	Translate the camera by a displacement in camera-local coordinates.

	In camera-local coordinates:
	- X: right
	- Y: up
	- Z: forward (towards where camera is looking)

	For example: translate = (0, 0, 1) moves the camera forward by 1 unit
	along its viewing direction.

	Args:
		camera: The camera to translate
		translate: Local displacement vector (3,)
		copy: If True, creates a copy before modifying. If False, modifies in-place.

	Returns:
		The translated camera (either the same object or a copy)
	"""
	assert translate.shape == (3,)

	# Make a copy if requested
	if copy:
		import copy as copy_module
		camera = copy_module.deepcopy(camera)

	# Extract the rotation matrix from camera_to_worlds (upper-left 3x3)
	c2w = camera.camera_to_worlds
	rotation = c2w[:3, :3]

	# In OpenGL convention, camera looks down -Z axis, so negate Z to match
	# intuitive forward direction
	translate_adjusted = translate.clone()
	translate_adjusted[2] = -translate_adjusted[2]

	# Transform local displacement to world space
	world_displacement = rotation @ translate_adjusted

	# Apply translation to camera position
	camera.camera_to_worlds[:3, 3] += world_displacement

	return camera
