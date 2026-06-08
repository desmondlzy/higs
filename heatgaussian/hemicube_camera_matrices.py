import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from nerfstudio.models.splatfacto import get_viewmat
from nerfstudio.cameras.camera_utils import viewmatrix

from heatgaussian.lookat_cameras import lookat_cameras
from heatgaussian.hemicube_keys import hemicube_keys

def _default_up(look_dir: torch.Tensor):
	up = torch.linalg.cross(look_dir, torch.tensor([0.0, 0.0, 1.0], device=look_dir.device))
	if torch.linalg.norm(up) == 0:
		up = torch.linalg.cross(look_dir, torch.tensor([1.0, 1.0, 0.0], device=look_dir.device))

	return up


def viewmatrix(lookat: torch.Tensor, up: torch.Tensor, pos: torch.Tensor):
	"""Returns a camera transformation matrix.

	Args:
		lookat: The direction the camera is looking.
		up: The upward direction of the camera.
		pos: The position of the camera.

	Returns:
		A camera transformation matrix.
	"""
	from torch.nn.functional import normalize

	vec2 = normalize(lookat, dim=-1) # (..., 3)
	vec1_avg = normalize(up, dim=-1) # (..., 3)
	vec0 = normalize(torch.linalg.cross(vec1_avg, vec2), dim=-1) # (..., 3)
	vec1 = normalize(torch.linalg.cross(vec2, vec0), dim=-1) # (..., 3)

	# (..., 3, 4)
	m = torch.stack([vec0, vec1, vec2, pos], -1)
	return m


def hemicube_camera_matrices(
	eye: torch.Tensor,
	look_dir: torch.Tensor,
	up: torch.Tensor,
	resolution: int,
):
	"""
	return the viewmats and Ks for the hemicube cameras per gsplat convention
	"""
	assert up is not None, "need to specify `up` when making hemicube cameras, use conanical_camera_local_up and transform to world space"
	assert eye.shape == (3,), eye.shape
	assert look_dir.shape == (3,), look_dir.shape
	assert torch.abs(torch.linalg.norm(look_dir) - 1.0) < 1e-5, f"{torch.linalg.norm(look_dir)=}, not normalized, {look_dir=}"
	assert torch.abs(torch.linalg.norm(up) - 1.0) < 1e-5, f"{torch.linalg.norm(up)=}, not normalized, {up=}"
	assert torch.dot(look_dir, up).abs() < 1e-5, f"look_dir and up must be orthogonal, got {look_dir} and {up}, {torch.dot(look_dir, up)=}"

	camera = lookat_cameras(
		eye=eye,
		target=eye + look_dir,
		up=up,
		camera_angle_x=np.deg2rad(90),
		img_h=resolution,
		img_w=resolution,
	)

	# z axis is the look direction
	# about y-axis (up): left and right
	r1 = R.from_rotvec([0.0, np.pi / 2, 0.0])
	# r1 = R.from_rotvec(up.detach().cpu().numpy() * np.pi / 2)

	# about x-axis: up and down
	r2 = R.from_rotvec([np.pi / 2, 0.0, 0.0])
	# left = torch.linalg.cross(up, look_dir)
	# r2 = R.from_rotvec(left.detach().cpu().numpy() * np.pi / 2)

	zero = R.from_rotvec([0.0, 0.0, 0.0])

	Ks = camera.get_intrinsics_matrices().cuda()

	camera_rotations = {
		'front': zero,
		'left': r1,
		'up': r2,
		'right': r1.inv(),
		'down': r2.inv(),
	}

	num_cameras = len(camera_rotations)

	c2ws = {}
	for key in hemicube_keys:
		r = camera_rotations[key]
		rot = torch.tensor(r.as_matrix()).float().cuda()
		# rot_r = viewmat[:, :3, :3] @ rot
		c2w = camera.camera_to_worlds.clone().cuda()
		c2w_r = c2w[:, :3, :3] @ rot
		c2w[:, :3, :3] = c2w_r
		c2ws[key] = c2w

	c2ws_tensor = torch.cat([c2w for c2w in c2ws.values()], dim=0)
	assert c2ws_tensor.shape == (num_cameras, 3, 4), c2ws_tensor.shape


	hemicube_viewmats = get_viewmat(c2ws_tensor)
	hemicube_Ks = torch.tile(Ks, (num_cameras, 1, 1))


	assert hemicube_viewmats.shape == (num_cameras, 4, 4), hemicube_viewmats.shape
	assert hemicube_Ks.shape == (num_cameras, 3, 3), hemicube_Ks.shape

	return hemicube_viewmats, hemicube_Ks


# @torch.compile
def hemicube_camera_matrices_lightweight(
	eye: torch.Tensor,
	look_dir: torch.Tensor,
	up: torch.Tensor,
	resolution: int,
):
	assert up is not None, "need to specify `up` when making hemicube cameras, use conanical_camera_local_up and transform to world space"
	assert torch.all(torch.abs(torch.linalg.norm(look_dir, dim=-1) - 1.0) < 1e-5), f"{torch.linalg.norm(look_dir)=}, not normalized, {look_dir=}"
	assert torch.all(torch.abs(torch.linalg.norm(up, dim=-1) - 1.0) < 1e-5), f"{torch.linalg.norm(up)=}, not normalized, {up=}"

	camera_c2w = viewmatrix(
		lookat=look_dir, 
		up=up, 
		pos=eye)
	assert camera_c2w.shape == (*eye.shape[:-1], 3, 4), camera_c2w.shape

	# z axis is the look direction
	# about y-axis (up): left and right
	# r1 = torch.tensor(R.from_rotvec([0.0, np.pi / 2, 0.0]).as_matrix(), device=camera_c2w.device, dtype=camera_c2w.dtype)
	r1 = torch.tensor(
		[[0, 0, 1],
		[ 0, 1, 0],
		[-1, 0, 0]],
		device=camera_c2w.device, dtype=camera_c2w.dtype
	)
	# r1_inv = torch.tensor(R.from_rotvec([0.0, -np.pi / 2, 0.0]).as_matrix(), device=r1.device, dtype=r1.dtype)
	r1_inv = torch.tensor(
		[[0, 0, -1],
		[ 0, 1, 0],
		[ 1, 0, 0]],
		device=r1.device, dtype=r1.dtype
	)

	# about x-axis: up and down
	# r2 = torch.tensor(R.from_rotvec([np.pi / 2, 0.0, 0.0]).as_matrix(), device=r1.device, dtype=r1.dtype)
	r2 = torch.tensor(
		[[1, 0, 0],
		[0, 0, -1],
		[0, 1, 0]],
		device=r1.device, dtype=r1.dtype
	)
	# r2_inv = torch.tensor(R.from_rotvec([-np.pi / 2, 0.0, 0.0]).as_matrix(), device=r1.device, dtype=r1.dtype)
	r2_inv = torch.tensor(
		[[1, 0, 0],
		[0, 0, 1],
		[0, -1, 0]],	
		device=r1.device, dtype=r1.dtype
	)
	# left = torch.linalg.cross(up, look_dir)
	# r2 = R.from_rotvec(left.detach().cpu().numpy() * np.pi / 2)

	camera_dims = eye.shape[:-1]
	num_views = 5
	c2ws_tensor = torch.tile(camera_c2w.view(*camera_dims, 1, 3, 4), (*[1 for _ in range(len(camera_dims))], num_views, 1, 1))
	assert c2ws_tensor.shape == (*camera_dims, num_views, 3, 4), c2ws_tensor.shape

	c2ws_tensor[..., 1, :3, :3] = camera_c2w[..., :3, :3] @ r1
	c2ws_tensor[..., 2, :3, :3] = camera_c2w[..., :3, :3] @ r2
	c2ws_tensor[..., 3, :3, :3] = camera_c2w[..., :3, :3] @ r1_inv
	c2ws_tensor[..., 4, :3, :3] = camera_c2w[..., :3, :3] @ r2_inv

	assert c2ws_tensor.shape == (*camera_dims, num_views, 3, 4), c2ws_tensor.shape

	hemicube_viewmats = get_viewmat(c2ws_tensor.reshape(-1, 3, 4)).reshape(*camera_dims, num_views, 4, 4)


	img_w = img_h = resolution
	camera_angle_x = np.deg2rad(90)
	fx = fy = 0.5 * img_w / np.tan(0.5 * camera_angle_x)
	cx = img_w / 2
	cy = img_h / 2

	hemicube_Ks = torch.zeros((*camera_dims, num_views, 3, 3), device=camera_c2w.device, dtype=torch.float32)
	hemicube_Ks[..., 0, 0] = fx
	hemicube_Ks[..., 1, 1] = fy
	hemicube_Ks[..., 0, 2] = cx
	hemicube_Ks[..., 1, 2] = cy
	hemicube_Ks[..., 2, 2] = 1.0

	assert hemicube_viewmats.shape == (*camera_dims, num_views, 4, 4), hemicube_viewmats.shape

	return hemicube_viewmats, hemicube_Ks
