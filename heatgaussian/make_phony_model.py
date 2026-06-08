from dataclasses import dataclass, field
from typing import Type

import torch
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
from nerfstudio.cameras.cameras import Cameras
from torchmetrics.image import (
	PeakSignalNoiseRatio, 
	StructuralSimilarityIndexMeasure, 
	LearnedPerceptualImagePatchSimilarity
)

def make_phony_model(
	means,
	quats,
	scales,
	opacities,
	thermal_opacities,
	emissivities,
	temperatures,
	specularities,
	normals,
	out_shs,
	rasterization_func,
):

	@dataclass
	class _PhonyGSConfig(SplatfactoModelConfig):
		_target: Type = field(default_factory=lambda: _PhonyGS)
		pass

	psnr_func = PeakSignalNoiseRatio().cuda()
	ssim_func = StructuralSimilarityIndexMeasure().cuda()
	lpips_func = LearnedPerceptualImagePatchSimilarity().cuda()


	class _PhonyGS(SplatfactoModel):
		"""
		the viewer will call `get_outputs_for_camera`, potentially set a obb crop box

		the logic of using crop box should be handled in the `get_outputs` 
		"""
		device: str = "cuda"
		gauss_params = torch.nn.ParameterDict(
			{
				"emissivities": torch.nn.Parameter(emissivities),
				"temperatures": torch.nn.Parameter(temperatures),
				"means": torch.nn.Parameter(means),
				"scales": torch.nn.Parameter(scales),
				"quats": torch.nn.Parameter(quats),
				"opacities": torch.nn.Parameter(opacities),
				"thermal_opacities": torch.nn.Parameter(thermal_opacities),
				"emissivities": torch.nn.Parameter(emissivities),
				"temperatures": torch.nn.Parameter(temperatures),
			}
		)
		def get_outputs(self, camera: Cameras):
			results = {
				"depth": torch.zeros((camera.height, camera.width, 1), device=self.device),
			}


			thermal_results = {
				key: value[0] for key, value in rasterization_func(
					means=means,
					quats=quats,
					scales=torch.exp(scales),
					thermal_opacities=torch.sigmoid(thermal_opacities),
					normals_world=normals,
					emissivities=torch.sigmoid(emissivities),
					# emissivities=emissivities_temporary,
					temperatures=torch.sigmoid(temperatures),
					specularities=torch.sigmoid(specularities),
					out_shs=out_shs,
					out_shs_purely_diffuse=None,
					out_shs_purely_specular=None,
					camera=camera,
					normal_filtering=False,
				).items()
			}

			return {**thermal_results, **results}
		

		def get_image_metrics_and_images(self, outputs, batch):
			subset_idx = batch["subset_idx"]
			render = outputs[f"thermals_{subset_idx}"]
			gt = batch["heat_image"].cuda().unsqueeze(-1)

			render = torch.concat((render, render, render), dim=-1)
			gt = torch.concat((gt, gt, gt), dim=-1)

			render_tonemapped = torch.pow(render.clip(0.0, 1.0), 1 / 2.2)
			gt_tonemapped = torch.pow(gt.clip(0.0, 1.0), 1 / 2.2)

			combined_img = torch.cat([render, gt], dim=1)
			combined_img_tonemapped = torch.cat([render_tonemapped, gt_tonemapped], dim=1)

			# reshape from (H, W, C) to (1, C, H, W) for metrics computation
			render_tonemapped = torch.moveaxis(render_tonemapped, -1, 0).unsqueeze(0)
			gt_tonemapped = torch.moveaxis(gt_tonemapped, -1, 0).unsqueeze(0)

			images_dict = {
				"img": combined_img,
				"img_tonemapped": combined_img_tonemapped,
			}

			metrics_dict = {
				"psnr": float(psnr_func(render_tonemapped, gt_tonemapped).item()),
				"ssim": float(ssim_func(render_tonemapped, gt_tonemapped).item()),
				"lpips": float(lpips_func(render_tonemapped, gt_tonemapped).item()),
			}

			return metrics_dict, images_dict
		

		@property
		def colors(self):
			return torch.sigmoid(self.gauss_params["emissivities"]) * torch.sigmoid(self.gauss_params["temperatures"]) ** 4

		@property
		def features_dc(self):
			return torch.zeros((self.gauss_params["means"].shape[0], 1, 3), device=self.device)

		@property
		def features_rest(self):
			return torch.zeros((self.gauss_params["means"].shape[0], 3, 3), device=self.device)


	phony_model = _PhonyGS(
		_PhonyGSConfig(),
		scene_box=None,
		num_train_data=0,
	)	

	return phony_model