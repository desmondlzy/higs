import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


def gaussians_on_sphere(
		radius=1.2, 
		scale=0.025, 
		opacity=0.9999,
		azimuth_count=128, 
		elevation_count=64, 
		radius_count=1, 
		disk=False,
		radii=None,
		center=None,
		hemisphere=False,
		**kwargs):
	"""
	create a bunch of gaussians on the surface of a sphere of radius 1.2, center at 0.0
	"""
	if center is None: 
		center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32).cuda()
	azimuths = torch.linspace(
		0, 
		2 * ((azimuth_count - 1) / azimuth_count) * torch.pi, 
		azimuth_count)
	elevations = torch.linspace(
		-torch.pi / 2 * (1 - 0.5 / elevation_count),
		torch.pi / 2 * (1 - 0.5 / elevation_count), 
		elevation_count)
	
	if radii == None:
		if radius_count == 1:
			radii = [radius]
		else:
			start = radius * (radius_count - 1) / radius_count
			radii = torch.linspace(start, radius, radius_count)
	else:
		assert radius_count == 1, "radius_count must be 1 if radii is not None"

	us, vs = torch.meshgrid(azimuths, elevations)

	r = radius
	xyz_rs = []
	for r in radii:
		xyz_r = torch.stack([
			r * torch.cos(vs) * torch.cos(us),
			r * torch.cos(vs) * torch.sin(us),
			r * torch.sin(vs),
		], axis=-1).reshape(-1, 3)

		if hemisphere:
			positive_z = xyz_r[:, 2] > 0.0
			xyz_r = xyz_r[positive_z]

		xyz_rs.append(xyz_r)

	xyz = torch.row_stack(xyz_rs)

	n = xyz.shape[0]

	if not disk:
		scales = torch.ones((n, 3), dtype=torch.float32) * torch.tensor(scale, dtype=torch.float32)
	else:
		scales = torch.column_stack([
			torch.ones((n, 2), dtype=torch.float32) * torch.tensor(scale, dtype=torch.float32),
			torch.ones((n, 1), dtype=torch.float32) * torch.tensor(scale * 0.000001, dtype=torch.float32),
		])

	scales = scales.cuda()


	if not disk:
		quats = torch.ones((n, 4), dtype=torch.float32, device="cuda")
	else:
		start_frame = np.array((0.0, 0.0, 1.0))
		scales[:, 2] = 0.000001

		quats = []
		for pos in tqdm(xyz):
			# get the rotation matrix that aligns the start frame to the current point
			# then convert the rotation matrix to quaternion
			target_normal = pos / torch.linalg.norm(pos)
			rot, rssd = R.align_vectors(target_normal.numpy(), start_frame)

			quat = rot.as_quat()

			# change from xyzw (scipy) to wxyz (torch)
			quats.append([quat[3], quat[0], quat[1], quat[2]])
		
		quats = torch.tensor(quats, dtype=torch.float32, device="cuda")


	return dict(
		means=xyz.cuda() + center.cuda(),
		normals=xyz.cuda() / torch.linalg.norm(xyz.cuda(), dim=-1, keepdim=True),
		scales=scales,
		quats=quats,
		shs=torch.ones((n, 16, 3), dtype=torch.float32, device="cuda"),
		opacities=torch.full((n, 1), opacity, dtype=torch.float32, device="cuda"),
		**kwargs,
	)
