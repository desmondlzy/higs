import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras


def make_nerfstudio_cameras(c2w: torch.Tensor, camera_angle_x: float, img_w: int, img_h: int) -> Cameras:
	camera_to_world = c2w[:3].unsqueeze(0)
	focal_length = 0.5 * img_w / np.tan(0.5 * camera_angle_x)
	cx = img_w / 2
	cy = img_h / 2

	camera = Cameras(
		camera_to_worlds=camera_to_world.float(),
		fx=focal_length,
		fy=focal_length,
		cx=cx,
		cy=cy,
	)

	return camera
