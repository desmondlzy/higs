import json 
from functools import cache
from pathlib import Path

import torch
import numpy as np
from nerfstudio.cameras.cameras import Cameras
from torchmetrics.image import PeakSignalNoiseRatio, LearnedPerceptualImagePatchSimilarity
from torchmetrics.regression import MeanAbsolutePercentageError, MeanAbsoluteError
from pytorch_msssim import SSIM
from nerfstudio.models.splatfacto import get_viewmat, SH2RGB, RGB2SH

from heatgaussian.rasterize_2dgs_normal_filtering import rasterize_2dgs_normal_filtering
from heatgaussian.nsext.heat_dataset import HeatDataset


@cache
def get_metric_functions():
	psnr = PeakSignalNoiseRatio(data_range=1.0).to("cuda")
	ssim = SSIM(data_range=1.0, size_average=True, channel=1).to("cuda")
	lpips = LearnedPerceptualImagePatchSimilarity().to("cuda")
	mae = MeanAbsoluteError().to("cuda")

	return psnr, ssim, lpips, mae

def gamma(x):
	return (torch.clamp(x, 0.0, 1.0) ** (1 / 2.2))


@torch.no_grad()
def evaluate_metrics_over_dataset(
	gaussians,
	radiance_cache,
	dataset,
	cameras: list[Cameras],
	near_plane: float,
	save_to: Path | None = None,
):
	psnr_scores = []
	ssim_scores = []
	lpips_scores = []

	mae_refl_scores = []

	psnr, ssim, lpips, mae = get_metric_functions()

	self_emissions = torch.nn.functional.softplus(gaussians.temperatures) * torch.sigmoid(gaussians.emissivities)

	for cam_idx in range(len(cameras)):
		cam = cameras[cam_idx]
		datapoint = dataset[cam_idx]

		thermal_idx = datapoint["subset_idx"]
		gaussians_to_train_cam = torch.nn.functional.normalize(cam.camera_to_worlds[:3, 3].view(1, 3).cuda() - gaussians.means)
		cached_radiances = radiance_cache(gaussians_to_train_cam, gaussians.normals)
		total_radiances = cached_radiances + self_emissions
		render_from_camera = rasterize_2dgs_normal_filtering(
			means=gaussians.means,
			quats=gaussians.quats,
			scales=torch.exp(gaussians.scales),
			opacities=torch.sigmoid(gaussians.thermal_opacities).squeeze(),
			colors=total_radiances,
			normals_world=gaussians.normals,
			viewmats=get_viewmat(cam.camera_to_worlds.reshape(1, 3, 4).cuda()),
			Ks=cam.get_intrinsics_matrices().reshape(1, 3, 3).cuda(),
			width=cam.width,
			height=cam.height,
			near_plane=near_plane,
		)[0][0, ..., thermal_idx: thermal_idx + 1]

		render_reflection = rasterize_2dgs_normal_filtering(
			means=gaussians.means,
			quats=gaussians.quats,
			scales=torch.exp(gaussians.scales),
			opacities=torch.sigmoid(gaussians.thermal_opacities).squeeze(),
			colors=cached_radiances,
			normals_world=gaussians.normals,
			viewmats=get_viewmat(cam.camera_to_worlds.reshape(1, 3, 4).cuda()),
			Ks=cam.get_intrinsics_matrices().reshape(1, 3, 3).cuda(),
			width=cam.width,
			height=cam.height,
			near_plane=near_plane,
		)[0][0, ..., thermal_idx: thermal_idx + 1]

		gamma_render_from_camera = gamma(render_from_camera)
		gamma_gt = gamma(datapoint["heat_image"].cuda())
		assert gamma_render_from_camera.shape == (cam.height, cam.width, 1), f"gamma_render_from_camera shape {gamma_render_from_camera.shape}"
		assert gamma_gt.shape == (cam.height, cam.width, 1), f"gamma_gt shape {gamma_gt.shape}"

		H, W = cam.height, cam.width
		ssim_value = ssim(gamma_gt.permute(2, 0, 1)[None, ...], gamma_render_from_camera.permute(2, 0, 1)[None, ...])
		ssim_scores.append(ssim_value.item())
		psnr_value = psnr(gamma_gt.permute(2, 0, 1)[None, ...], gamma_render_from_camera.permute(2, 0, 1)[None, ...])
		psnr_scores.append(psnr_value.item())
		lpips_scores.append(lpips(
			gamma_render_from_camera.view(1, 1, H, W).expand(1, 3, H, W),
			gamma_gt.view(1, 1, H, W).expand(1, 3, H, W),
		).item())

		gt_reflection = dataset.get_reflection(cam_idx)
		if gt_reflection is not None:
			mae_refl_scores.append(mae(
					render_reflection.permute(2, 0, 1)[None, ...],
					gt_reflection.permute(2, 0, 1)[None, ...],
				).item()
			)
			# print(f"cam {cam_idx} reflection mae: {mae_refl_scores[-1]}")


	metrics = {
		"full/psnr_mean": np.mean(psnr_scores),
		"full/psnr_std": np.std(psnr_scores),
		"full/ssim_mean": np.mean(ssim_scores),
		"full/ssim_std": np.std(ssim_scores),
		"full/lpips_mean": np.mean(lpips_scores),
		"full/lpips_std": np.std(lpips_scores),
		"full/count": len(psnr_scores),
	}

	# if there are reflection images
	if len(mae_refl_scores) > 0:
		metrics["refl/mae_refl_mean"] = np.mean(mae_refl_scores)
		metrics["refl/mae_refl_std"] = np.std(mae_refl_scores)
		metrics["refl/count"] = len(mae_refl_scores)

	if save_to is not None:
		with open(save_to, "w") as f:
			json.dump(metrics, f, indent=4)
			print("metrics saved to", save_to)

	return metrics
