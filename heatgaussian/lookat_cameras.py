import numpy as np
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.camera_utils import viewmatrix

def lookat_cameras(target, up, eye, img_w=128, img_h=128, camera_angle_x=0.5, focal=None, flip=False):
	"""
	setup a nerfstudio camera from lookat parameters; viewmatrix convention opengl; 
	camera_angle_x in radians
	"""
	lookat = target - eye
	if flip:
		lookat = -lookat
	c2w = viewmatrix(lookat=lookat, up=up, pos=eye)

	focal = 0.5 * img_w / np.tan(0.5 * camera_angle_x) if focal is None else focal
	cx = img_w / 2
	cy = img_h / 2

	cam = Cameras(
		camera_to_worlds=c2w[:3].unsqueeze(0).float(),
		fx=focal,
		fy=focal,
		cx=cx,
		cy=cy,
	)

	return cam