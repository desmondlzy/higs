"""Rotate points and quaternions around an axis."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def rotate_around_axis(axis, origin, angle, points, quats):
	"""
	Rotate points and quaternions around an axis by a given angle.

	Args:
		axis: Rotation axis (3,)
		origin: Center of rotation (3,)
		angle: Rotation angle in radians (scalar)
		points: Points to rotate (N, 3)
		quats: Quaternions (w, x, y, z) to rotate (N, 4)

	Returns:
		tuple: (rotated_points, rotated_quats)
			- rotated_points: (N, 3) rotated points
			- rotated_quats: (N, 4) rotated quaternions
	"""
	assert axis.shape == (3,)

	axis = axis / torch.linalg.norm(axis)
	cos_theta = torch.cos(angle)
	sin_theta = torch.sin(angle)
	ux, uy, uz = axis

	# Rotation matrix
	R_mat = torch.tensor([
		[cos_theta + ux * ux * (1 - cos_theta),       ux * uy * (1 - cos_theta) - uz * sin_theta, ux * uz * (1 - cos_theta) + uy * sin_theta],
		[uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy * uy * (1 - cos_theta),       uy * uz * (1 - cos_theta) - ux * sin_theta],
		[uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta, cos_theta + uz * uz * (1 - cos_theta)]
	], device=axis.device, dtype=axis.dtype)

	# Rotate points
	points_shifted = points - origin
	rotated_points = (R_mat @ points_shifted.T).T + origin

	# Convert axis-angle to quaternion using scipy
	axis_np = axis.cpu().numpy()
	angle_np = angle.cpu().numpy() if torch.is_tensor(angle) else angle
	rotation = R.from_rotvec(axis_np * angle_np)
	rotation_quat = rotation.as_quat()  # Returns [x, y, z, w]

	# Convert to [w, x, y, z] format (PyTorch convention)
	rotation_quat_wxyz = np.array([rotation_quat[3], rotation_quat[0], rotation_quat[1], rotation_quat[2]])
	rotation_quat_tensor = torch.tensor(rotation_quat_wxyz, device=quats.device, dtype=quats.dtype)

	# Multiply quaternions: q_rotated = q_rotation * q_original
	def quat_multiply(q1, q2):
		"""
		Multiply two quaternions in [w, x, y, z] format.

		Args:
			q1: (4,) or (N, 4)
			q2: (N, 4)

		Returns:
			(N, 4) multiplied quaternions
		"""
		if q1.ndim == 1:
			q1 = q1.unsqueeze(0)  # (1, 4)

		w1, x1, y1, z1 = q1[..., 0:1], q1[..., 1:2], q1[..., 2:3], q1[..., 3:4]
		w2, x2, y2, z2 = q2[..., 0:1], q2[..., 1:2], q2[..., 2:3], q2[..., 3:4]

		w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
		x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
		y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
		z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

		return torch.cat([w, x, y, z], dim=-1)

	rotated_quats = quat_multiply(rotation_quat_tensor, quats)

	return rotated_points, rotated_quats
