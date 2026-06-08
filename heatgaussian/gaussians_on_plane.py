import torch
import numpy as np
from scipy.spatial.transform import Rotation as R


def gaussians_on_plane(
	width=10.0,
	height=10.0,
	plate_normal=None,
	scale=0.025,
	opacity=0.9999,	
	width_count=30,
	height_count=30,
	center=None,
	disk=False,
):
	if center is None: 
		center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32).cuda()
	if plate_normal is None:
		plate_normal = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32).cuda()

	widths = torch.linspace(-width / 2, width / 2, width_count)
	heights = torch.linspace(-height / 2, height / 2, height_count)

	us, vs = torch.meshgrid(widths, heights)

	xyz = torch.stack([
		us.reshape(-1),
		vs.reshape(-1),
		torch.zeros_like(us).reshape(-1),
	], axis=-1)

	n = xyz.shape[0]

	scales = torch.ones((n, 3), dtype=torch.float32) * torch.tensor(scale, dtype=torch.float32)

	normals = plate_normal.repeat(n, 1)

	if not disk:
		quats = torch.ones((n, 4), dtype=torch.float32, device="cuda")
	else:
		start_frame = np.array((0.0, 0.0, 1.0))
		scales[:, 2] = 0.000001

		quats = []
		rot, rssd = R.align_vectors(plate_normal.cpu().numpy(), start_frame)

		quat = rot.as_quat()

		# change from xyzw (scipy) to wxyz (torch)
		quat = torch.tensor([quat[3], quat[0], quat[1], quat[2]], dtype=torch.float32)
		
		quats = torch.tile(quat, (n, 1)).cuda()


	return dict(
		means=xyz.cuda() + center.cuda(),
		normals=normals.cuda(),
		scales=scales.cuda(),
		quats=quats.cuda(),
		opacities=torch.full((n, 1), opacity, dtype=torch.float32, device="cuda"),
	)
