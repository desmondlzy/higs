from pathlib import Path

import torch
from nerfstudio.models.splatfacto import get_viewmat
import gsplat
import skimage	
import imageio

def select_gaussians_from_masks(
	means,
	quats,
	scales,
	cameras,
	masks,
	dilation_radius=3,
) -> torch.Tensor:
	"""
	given masks and corresponding camera objects, select gaussians that project into all the masks. return the boolean tensor
	"""

	selected_indices = torch.ones((len(means), ), dtype=torch.bool).cuda()
	for cam, mask in zip(cameras, masks):
		viewmats = get_viewmat(cam.camera_to_worlds.reshape(1, 3, 4)).cuda()
		Ks = cam.get_intrinsics_matrices().reshape(1, 3, 3).cuda()

		mask = torch.tensor(
			skimage.morphology.dilation(
				mask.detach().cpu().numpy(), 
				skimage.morphology.disk(dilation_radius)), 
		dtype=torch.bool, device=means.device)

		proj_results = gsplat.fully_fused_projection_2dgs(
			means=means,
			quats=quats,
			scales=scales,
			viewmats=viewmats,
			Ks=Ks,
			width=cam.width,
			height=cam.height,
			packed=False,
		)

		radii, means2d, depths, ray_transforms, normals_camera = proj_results

		# [0] is for camera axis
		xs = means2d[0][:, 0]
		ys = means2d[0][:, 1]
		cam_w = cam.width.item()
		cam_h = cam.height.item()
		in_camera_frame = (xs >= 0) & (xs < cam_w) & (ys >= 0) & (ys < cam_h)
		mask_values = mask.to(means2d.device)[
			torch.clamp(ys, 0, cam_h - 1).long(), 
			torch.clamp(xs, 0, cam_w - 1).long(),
		]
		camera_gaussian_choice = in_camera_frame.cuda() & mask_values.cuda()
		selected_indices &= camera_gaussian_choice

		# for gid, (x, y) in enumerate(means2d[0]):
		# 	x = int(x)
		# 	y = int(y)
			
		# 	if 0 <= x < cam.width and 0 <= y < cam.height:
		# 		if mask[y, x]:
		# 			...
		# 		else:
		# 			selected_indices[gid] = False
		# 	else:
		# 		selected_indices[gid] = False
	
	return selected_indices


def load_masks_from_directory(mask_dir, candidate_cameras, candidate_dataset):
	mask_files = sorted(mask_dir.glob("*.png"))
	mask_maps = {
		name.stem: imageio.imread(str(mask_dir / name))
		for name in mask_files
	}

	masks = []
	cameras = []

	for it, (camera, datapoint) in enumerate(zip(candidate_cameras, candidate_dataset)):
		image_filename = Path(datapoint["image_filename"])
		if image_filename.stem not in mask_maps:
			continue

		mask = torch.tensor(mask_maps[image_filename.stem], dtype=torch.bool)

		masks.append(mask)
		cameras.append(camera)
	
	return masks, cameras


def load_masks_from_dataset(
		data_path,
		candidate_cameras,
		candidate_dataset,
		masks_subdir="masks",
		mask_suffix=".png",
	):
	"""Load masks stored in the dataset directory and pair them with their cameras.

	Looks for masks at {data_path}/{masks_subdir}/{stem}{mask_suffix} where stem
	matches the image filename stem.  Only frames that have a mask are returned.

	Args:
		data_path: Root path of the dataset (e.g. Path("data/higs/teapot"))
		candidate_cameras: Cameras object for the training split
		candidate_dataset: Dataset object (iterable of datapoints with "image_filename")
		masks_subdir: Subdirectory under data_path that contains mask files
		mask_suffix: File extension for mask images

	Returns:
		(masks, cameras): Lists of matched boolean mask tensors and Camera objects
	"""
	masks_dir = Path(data_path) / masks_subdir

	masks = []
	cameras = []

	for camera, datapoint in zip(candidate_cameras, candidate_dataset):
		stem = Path(datapoint["image_filename"]).stem
		mask_path = masks_dir / f"{stem}{mask_suffix}"
		if not mask_path.exists():
			continue
		mask = torch.tensor(imageio.imread(str(mask_path)), dtype=torch.bool)
		masks.append(mask)
		cameras.append(camera)

	return masks, cameras


def load_masks_from_directory_data_dir(
		mask_dir, 
		data_dir,
		candidate_cameras, 
		candidate_dataset,
		mask_suffix=".png",
	):
	mask_files = sorted(mask_dir.glob("**/*.png"))
	mask_maps = {
		mask_path.relative_to(mask_dir): imageio.imread(mask_path)
		for mask_path in mask_files
	}

	masks = []
	cameras = []

	for it, (camera, datapoint) in enumerate(zip(candidate_cameras, candidate_dataset)):
		image_filename = Path(datapoint["image_filename"]).relative_to(data_dir)
		mask_filename = image_filename.parents[1] / "masks" / image_filename.with_suffix(mask_suffix).name
		if mask_filename not in mask_maps:
			continue

		# mask = torch.tensor(mask_maps[image_filename.stem], dtype=torch.bool)
		mask = torch.tensor(mask_maps[mask_filename], dtype=torch.bool)

	
		masks.append(mask)
		cameras.append(camera)
	
	return masks, cameras
