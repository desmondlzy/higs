"""Generate circular path between three points."""

import numpy as np
import torch


def circle_path(start, passing, end, num_points, interp_order):
	"""
	Generate a circular path from start to end, passing through a point.

	Args:
		start: Starting point (3,)
		passing: Intermediate point the path passes through (3,)
		end: End point (3,)
		num_points: Number of points to generate along the path
		interp_order: Interpolation order (1=linear, 2=quadratic, 3=cubic easing)

	Returns:
		Tensor of shape (num_points, 3) containing points along the circular path
	"""
	assert start.shape == (3,)
	assert passing.shape == (3,)
	assert end.shape == (3,)
	assert interp_order in [1, 2, 3], "interp_order must be 1, 2, or 3"

	# Convert to numpy for easier calculation
	is_tensor = torch.is_tensor(start)
	device = start.device if is_tensor else None
	dtype = start.dtype if is_tensor else None

	if is_tensor:
		start_np = start.cpu().numpy()
		passing_np = passing.cpu().numpy()
		end_np = end.cpu().numpy()
	else:
		start_np = np.array(start)
		passing_np = np.array(passing)
		end_np = np.array(end)

	# Find the center of the circle passing through the three points
	# The center is equidistant from all three points:
	# |center - start|^2 = |center - passing|^2 = |center - end|^2

	# This gives us two equations:
	# 2*(passing - start) · center = |passing|^2 - |start|^2
	# 2*(end - start) · center = |end|^2 - |start|^2

	A = np.vstack([
		2 * (passing_np - start_np),
		2 * (end_np - start_np)
	])
	b = np.array([
		np.dot(passing_np, passing_np) - np.dot(start_np, start_np),
		np.dot(end_np, end_np) - np.dot(start_np, start_np)
	])

	# Solve for center using least squares (system is underdetermined in 3D)
	center, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

	# Calculate radius
	radius = np.linalg.norm(center - start_np)

	# Create orthonormal basis for the circle plane
	# v1 points from center to start
	v1 = (start_np - center) / radius

	# v2 is perpendicular to v1 in the plane of the circle
	# We find it by taking the component of (passing - center) perpendicular to v1
	v_passing = passing_np - center
	v_passing_norm = v_passing / np.linalg.norm(v_passing)
	v2 = v_passing_norm - np.dot(v_passing_norm, v1) * v1
	v2 = v2 / np.linalg.norm(v2)

	# Calculate angles for each point
	angle_start = 0.0  # by definition

	v_passing_unit = (passing_np - center) / np.linalg.norm(passing_np - center)
	angle_passing = np.arctan2(np.dot(v_passing_unit, v2), np.dot(v_passing_unit, v1))

	v_end_unit = (end_np - center) / np.linalg.norm(end_np - center)
	angle_end = np.arctan2(np.dot(v_end_unit, v2), np.dot(v_end_unit, v1))

	# Ensure we go through passing point in the correct direction
	if angle_passing < 0:
		angle_passing += 2 * np.pi
	if angle_end < 0:
		angle_end += 2 * np.pi
	if angle_end < angle_passing:
		angle_end += 2 * np.pi

	# Generate parameter t based on interp_order
	t = np.linspace(0, 1, num_points)
	if interp_order == 1:
		# Linear spacing
		pass
	elif interp_order == 2:
		# Quadratic easing (ease-in)
		t = t ** 2
	elif interp_order == 3:
		# Cubic easing (ease-in-out)
		t = 3 * t**2 - 2 * t**3

	# Interpolate angles
	angles = t * angle_end

	# Generate points on the circle
	points = center[np.newaxis, :] + radius * (
		np.cos(angles)[:, np.newaxis] * v1[np.newaxis, :] +
		np.sin(angles)[:, np.newaxis] * v2[np.newaxis, :]
	)

	# Convert back to tensor if needed
	if is_tensor:
		return torch.tensor(points, dtype=dtype, device=device)
	else:
		return points
