from dataclasses import dataclass, field

from typing import Dict, List, Literal, Optional, Tuple, Type, Union
from sklearn.neighbors import NearestNeighbors
import numpy as np
import math
import torch
from torch import Tensor
from gsplat import rasterization_2dgs
from pytorch_msssim import SSIM
from torch.nn import Parameter
from gsplat.strategy import MCMCStrategy

import torch.nn.functional as F

from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig, Model, DefaultStrategy
from nerfstudio.models.splatfacto import num_sh_bases, quat_to_rotmat, get_viewmat, random_quat_tensor
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.model_components.lib_bilagrid import BilateralGrid, color_correct, slice, total_variation_loss
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation 

from .default_heat_strategy import DefaultHeatStrategy

def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
	"""
	Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
	using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
	Args:
		d6: 6D rotation representation, of size (*, 6)

	Returns:
		batch of rotation matrices of size (*, 3, 3)

	[1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
	On the Continuity of Rotation Representations in Neural Networks.
	IEEE Conference on Computer Vision and Pattern Recognition, 2019.
	Retrieved from http://arxiv.org/abs/1812.07035
	"""

	a1, a2 = d6[..., :3], d6[..., 3:]
	b1 = F.normalize(a1, dim=-1)
	b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
	b2 = F.normalize(b2, dim=-1)
	b3 = torch.cross(b1, b2, dim=-1)
	return torch.stack((b1, b2, b3), dim=-2)


def knn(x: Tensor, K: int = 4) -> Tensor:
	x_np = x.detach().cpu().numpy()
	model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
	distances, _ = model.kneighbors(x_np)
	return torch.from_numpy(distances).to(x)

def rgb_to_sh(rgb: Tensor) -> Tensor:
	C0 = 0.28209479177387814
	return (rgb - 0.5) / C0


@dataclass
class Heat2DGSModelConfig(ModelConfig):
	"""Heat2DGS Model Config"""

	_target: Type = field(default_factory=lambda: Heat2DGSModel)

	random_init: bool = False

	num_random: int = 100000

	random_scale: float = 10.0

	ssim_lambda: float = 0.2

	sh_degree = 3

	sh_degree_interval: int = 1000
	"""every n intervals turn on another sh degree"""

	background_color: Literal["random", "black", "white"] = "random"
	"""Whether to randomize the background color."""
	num_downscales: int = 2
	"""at the beginning, resolution is 1/2^d, where d is this number"""
	resolution_schedule: int = 3000
	"""training starts at 1/d resolution, every n steps this is doubled"""

	rasterize_mode: Literal["classic", "antialiased"] = "classic"

	# Random crop size for training  (experimental)
	patch_size: Optional[int] = None

	# Batch size for training. Learning rates are scaled automatically
	batch_size: int = 1
	# A global factor to scale the number of training steps
	steps_scaler: float = 1.0

	# Near plane clipping distance
	near_plane: float = 0.01
	# Far plane clipping distance
	far_plane: float = 1e10

	# GSs with opacity below this value will be pruned
	prune_opa: float = 0.01

	prune_thermal_opa: float = 0.01

	grow_grad2d: float = 0.0002
	# GSs with scale below this value will be duplicated. Above will be split
	grow_scale3d: float = 0.01
	# GSs with scale above this value will be pruned.
	prune_scale3d: float = 0.9

	# Start refining GSs after this iteration
	refine_start_iter: int = 500
	# Stop refining GSs after this iteration
	refine_stop_iter: int = 15_000
	# Refine GSs every this steps
	refine_every: int = 100
	# Reset opacities every this steps
	reset_every: int = 3000

	# maximum number of gaussians for MCMC strategy, if not using MCMC, this is ignored
	max_gaussian_num: int = 100_000

	# Use packed mode for rasterization, this leads to less memory usage but slightly slower.
	packed: bool = False
	# Use sparse gradients for optimization. (experimental)
	sparse_grad: bool = False
	# Use absolute gradient for pruning. This typically requires larger --grow_grad2d, e.g., 0.0008 or 0.0006
	use_absgrad: bool = False
	# Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
	antialiased: bool = False
	# Whether to use revised opacity heuristic from arXiv:2404.06109 (experimental)
	revised_opacity: bool = False

	# Enable camera optimization.
	pose_opt: bool = False
	# Learning rate for camera optimization
	pose_opt_lr: float = 1e-5
	# Regularization for camera optimization as weight decay
	pose_opt_reg: float = 1e-6
	# Add noise to camera extrinsics. This is only to test the camera pose optimization.
	pose_noise: float = 0.0

	heat_loss: bool = True

	heat_lambda: float = 1.0

	heat_start_iter: int = 10_000
	heat_full_iter: int = 20_000

	# Enable depth loss. (experimental)
	depth_loss: bool = False
	# Weight for depth loss
	depth_lambda: float = 1e-2

	# Enable normal consistency loss. (Currently for 2DGS only)
	normal_loss: bool = True
	# Weight for normal loss
	normal_lambda: float = 5e-2
	# Iteration to start normal consistency regulerization
	normal_start_iter: int = 7_000

	output_depth_during_training: bool = True

	entropy_lambda: float = 0.0

	entropy_start_iter: int = 10_000

	# Distortion loss. (experimental)
	dist_loss: bool = False
	# Weight for distortion loss
	dist_lambda: float = 1e-2
	# Iteration to start distortion loss regulerization
	dist_start_iter: int = 3_000

	use_scale_regularization: bool = True

	use_tonemapping: bool = True
	
	use_separate_opacities: bool = True

	max_gauss_ratio: float = 10.0
	"""threshold of ratio of gaussian max to min scale before applying regularization
	loss from the PhysGaussian paper
	"""

	camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="off"))
	"""Config of the camera optimizer to use"""
	use_bilateral_grid: bool = False
	"""If True, use bilateral grid to handle the ISP changes in the image space. This technique was introduced in the paper 'Bilateral Guided Radiance Field Processing' (https://bilarfpro.github.io/)."""
	grid_shape: Tuple[int, int, int] = (16, 16, 8)
	"""Shape of the bilateral grid (X, Y, W)"""
	color_corrected_metrics: bool = False
	"""If True, apply color correction to the rendered images before computing the metrics."""

	# Model for splatting.
	model_type: Literal["2dgs", "2dgs-inria"] = "2dgs"

	# Dump information to tensorboard every this steps
	tb_every: int = 100
	# Save training images to tensorboard
	tb_save_image: bool = False


	def adjust_steps(self, factor: float):
		self.eval_steps = [int(i * factor) for i in self.eval_steps]
		self.save_steps = [int(i * factor) for i in self.save_steps]
		self.sh_degree_interval = int(self.sh_degree_interval * factor)
		self.refine_start_iter = int(self.refine_start_iter * factor)
		self.refine_stop_iter = int(self.refine_stop_iter * factor)
		self.reset_every = int(self.reset_every * factor)
		self.refine_every = int(self.refine_every * factor)



class Heat2DGSModel(SplatfactoModel):
	"""Nerfstudio's implementation of Gaussian Splatting

	Args:
		config: Splatfacto configuration to instantiate model
	"""

	config: Heat2DGSModelConfig

	def __init__(
		self,
		*args,
		seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
		metadata: dict,
		**kwargs,
	):
		assert "num_subsets" in metadata, "num_subsets should be in metadata, using multi-heat-colmap?"

		self.thermal_channels = metadata["num_subsets"]
		self.metadata = metadata

		super().__init__(*args, seed_points=seed_points, **kwargs)

	def populate_modules(self):
		init_type = "sfm" if self.seed_points is not None and not self.config.random_init else "random"
		init_scale = 1.0
		if init_type == "sfm":
			points = torch.nn.Parameter(self.seed_points[0].cuda())  # (Location, Color)
			point_colors = self.seed_points[1].cuda()
			rgbs = torch.nn.Parameter(point_colors / 255)

		elif init_type == "random":
			points = torch.nn.Parameter((torch.rand((self.config.max_gaussian_num, 3)) - 0.5) * self.config.random_scale)
			rgbs = torch.rand((self.config.max_gaussian_num, 3))
			print(f"Randomly initialized {points.shape[0]} points")
		else:
			raise ValueError("Please specify a correct init_type: sfm or random")

		N = points.shape[0]
		# Initialize the GS size to be the average dist of the 3 nearest neighbors
		dist_avg = (knn(points, 4)[:, 1:]).mean(dim=-1)  # [N,]
		scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
		quats = random_quat_tensor(N)  # [N, 4]
		opacities = torch.logit(torch.full((N,), 0.1))  # [N,]

		thermal_opacities = torch.logit(torch.full((N,), 0.1))  # [N,]
		# thermal_radiances = torch.logit(torch.rand((N, self.thermal_channels)))  # [N,]
		thermal_radiances = torch.logit(torch.full((N, self.thermal_channels), 0.3))  # [N,]

		params = [
			# name, value, lr
			("means", torch.nn.Parameter(points), 1.6e-4),
			("scales", torch.nn.Parameter(scales), 5e-3),
			("quats", torch.nn.Parameter(quats), 1e-3),
			("opacities", torch.nn.Parameter(opacities), 5e-2),
			("thermal_radiances", torch.nn.Parameter(thermal_radiances), 2.5e-3),
		]
		if self.config.use_separate_opacities:
			params.append(
				("thermal_opacities", torch.nn.Parameter(thermal_opacities), 5e-2),
			)
		# else:
		# 	# placeholder: thermal opacities access thru .thermal_opacities will redirected to .opacities
		# 	params.append(
		# 		("thermal_opacities", torch.nn.Parameter(torch.empty((N, 0))), 5e-2),
		# 	)

		# color is SH coefficients.
		colors = torch.zeros((N, (self.config.sh_degree + 1) ** 2, 3))  # [N, K, 3]
		colors[:, 0, :] = rgb_to_sh(rgbs)
		params.append(("features_dc", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
		params.append(("features_rest", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
		# params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

		self.gauss_params = torch.nn.ParameterDict({n: v for n, v, _ in params})
		# Scale learning rate based on batch size, reference:
		# https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
		# Note that this would not make the training exactly equivalent, see
		# https://arxiv.org/pdf/2402.18824v1
		# optimizers = {
		# 	name: (torch.optim.SparseAdam if self.config.sparse_grad else torch.optim.Adam)(
		# 		[{"params": splats[name], "lr": lr * math.sqrt(self.config.batch_size)}],
		# 		eps=1e-15 / math.sqrt(self.config.batch_size),
		# 		betas=(1 - self.config.batch_size * (1 - 0.9), 1 - self.config.batch_size * (1 - 0.999)),
		# 	)
		# 	for name, _, lr in params
		# }

		self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
			num_cameras=self.num_train_data, device="cpu"
		)

		# metrics
		from torchmetrics.image import PeakSignalNoiseRatio
		from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

		self.psnr = PeakSignalNoiseRatio(data_range=1.0)
		self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
		self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
		self.step = 0

		self.crop_box: Optional[OrientedBox] = None
		if self.config.background_color == "random":
			self.background_color = torch.tensor(
				[0.1490, 0.1647, 0.2157],
			)  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
		else:
			self.background_color = get_color(self.config.background_color)
		if self.config.use_bilateral_grid:
			self.bil_grids = BilateralGrid(
				num=self.num_train_data,
				grid_X=self.config.grid_shape[0],
				grid_Y=self.config.grid_shape[1],
				grid_W=self.config.grid_shape[2],
			)

		# Strategy for GS densification
		# the parameters from the example code in gsplat repo
		StrategyClass = DefaultHeatStrategy if self.config.use_separate_opacities else DefaultStrategy 
		if self.config.use_separate_opacities:
			self.strategy = DefaultHeatStrategy(
				verbose=True,
				prune_opa=self.config.prune_opa,
				prune_thermal_opa=self.config.prune_thermal_opa,
				grow_grad2d=self.config.grow_grad2d,
				grow_scale3d=self.config.grow_scale3d,
				prune_scale3d=self.config.prune_scale3d,
				refine_scale2d_stop_iter=4000, # splatfacto behavior
				refine_start_iter=self.config.refine_start_iter,
				refine_stop_iter=self.config.refine_stop_iter,
				reset_every=self.config.reset_every,
				refine_every=self.config.refine_every,
				absgrad=self.config.use_absgrad,
				revised_opacity=self.config.revised_opacity,
				key_for_gradient="gradient_2dgs",
			)
		else:
			# self.strategy = DefaultStrategy(
			# 	verbose=True,
			# 	prune_opa=self.config.prune_opa,
			# 	grow_grad2d=self.config.grow_grad2d,
			# 	grow_scale3d=self.config.grow_scale3d,
			# 	prune_scale3d=self.config.prune_scale3d,
			# 	refine_scale2d_stop_iter=4000, # splatfacto behavior
			# 	refine_start_iter=self.config.refine_start_iter,
			# 	refine_stop_iter=self.config.refine_stop_iter,
			# 	reset_every=self.config.reset_every,
			# 	refine_every=self.config.refine_every,
			# 	absgrad=self.config.use_absgrad,
			# 	revised_opacity=self.config.revised_opacity,
			# 	key_for_gradient="gradient_2dgs",
			# )

			self.strategy = MCMCStrategy(
				cap_max=self.config.max_gaussian_num,
				noise_lr=500000,
				refine_start_iter=self.config.refine_start_iter,
				refine_stop_iter=self.config.refine_stop_iter,
				refine_every=self.config.refine_every,
				min_opacity=self.config.prune_opa,
				verbose=True,
			)


		if isinstance(self.strategy, (DefaultHeatStrategy, DefaultStrategy)):
			self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)
		elif isinstance(self.strategy, MCMCStrategy):
			self.strategy_state = self.strategy.initialize_state()


	def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
		names = ["means", "scales", "quats", "features_dc", "features_rest", "opacities", "thermal_radiances"]
		if self.config.use_separate_opacities:
			names.append("thermal_opacities")
		res = {
			name: [self.gauss_params[name]] for name in names
		}
		
		return res


	@staticmethod
	def get_empty_outputs(width: int, height: int, background: torch.Tensor) -> Dict[str, Union[torch.Tensor, List]]:
		rgb = background.repeat(height, width, 1)
		depth = background.new_ones(*rgb.shape[:2], 1) * 10
		accumulation = background.new_zeros(*rgb.shape[:2], 1)
		normals = background.new_zeros(*rgb.shape[:2], 3)
		normals_from_depth = background.new_zeros(*rgb.shape[:2], 3)
		thermals = background.new_ones(*rgb.shape[:2], 1)

		res = {
			"rgb": rgb, 
			"depth": depth, 
			"accumulation": accumulation, 
			"normals": normals,
			"normals_from_depth": normals_from_depth,
			"thermal_0": thermals,  # type: ignore
			"thermal_1": thermals,  # type: ignore
			"background": background
		}

		return res


	def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
		"""Takes in a camera and returns a dictionary of outputs.

		Args:
			camera: The camera(s) for which output images are rendered. It should have
			all the needed information to compute the outputs.

		Returns:
			Outputs of model. (ie. rendered colors)
		"""
		if not isinstance(camera, Cameras):
			print("Called get_outputs with not a camera")
			return {}

		if self.training:
			assert camera.shape[0] == 1, "Only one camera at a time"
			optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
		else:
			optimized_camera_to_world = camera.camera_to_worlds
		
		if torch.isnan(optimized_camera_to_world).any():
			print(camera.metadata)

		# cropping
		if self.crop_box is not None and not self.training:
			crop_ids = self.crop_box.within(self.means).squeeze()
			if crop_ids.sum() == 0:
				return self.get_empty_outputs(
					int(camera.width.item()), int(camera.height.item()), self.background_color
				)
		else:
			crop_ids = None

		if crop_ids is not None:
			opacities_crop = self.opacities[crop_ids]
			thermal_opacities_crop = self.thermal_opacities[crop_ids]
			means_crop = self.means[crop_ids]
			features_dc_crop = self.features_dc[crop_ids]
			features_rest_crop = self.features_rest[crop_ids]
			scales_crop = self.scales[crop_ids]
			quats_crop = self.quats[crop_ids]
			thermal_radiances_crop = self.thermal_radiances[crop_ids]
		else:
			opacities_crop = self.opacities
			thermal_opacities_crop = self.thermal_opacities
			means_crop = self.means
			features_dc_crop = self.features_dc
			features_rest_crop = self.features_rest
			scales_crop = self.scales
			quats_crop = self.quats
			thermal_radiances_crop = self.thermal_radiances

		colors_crop = torch.cat((features_dc_crop, features_rest_crop), dim=1)

		camera_scale_fac = self._get_downscale_factor()
		camera.rescale_output_resolution(1 / camera_scale_fac)
		viewmat = get_viewmat(optimized_camera_to_world)

		K = camera.get_intrinsics_matrices().cuda()
		W, H = int(camera.width.item()), int(camera.height.item())
		self.last_size = (H, W)
		camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

		# apply the compensation of screen space blurring to gaussians
		if self.config.rasterize_mode not in ["antialiased", "classic"]:
			raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

		if self.config.output_depth_during_training or not self.training:
			render_mode = "RGB+ED"
		else:
			render_mode = "RGB"

		if self.config.sh_degree > 0:
			sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
		else:
			colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
			sh_degree_to_use = None
		
		(
			render_colors_depths,
			render_alphas,
			render_normals,
			normals_from_depth,
			render_distort,
			render_median,
			self.info,
		) = rasterization_2dgs(
			means=means_crop,
			quats=quats_crop,  # rasterization does normalization internally
			scales=torch.exp(scales_crop),
			opacities=torch.sigmoid(opacities_crop).squeeze(-1),
			colors=colors_crop,
			viewmats=viewmat,  # [1, 4, 4]
			Ks=K,  # [1, 3, 3]
			width=W,
			height=H,
			packed=self.config.packed,
			near_plane=self.config.near_plane,
			far_plane=self.config.far_plane,
			render_mode=render_mode,
			sh_degree=sh_degree_to_use,
			sparse_grad=False,
			absgrad=self.config.use_absgrad,
			# set some threshold to disregrad small gaussians for faster rendering.
			# radius_clip=3.0,
		)

		if self.training:
			if isinstance(self.strategy, (DefaultHeatStrategy, DefaultStrategy)):
				self.strategy.step_pre_backward(
					self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
				)

		render_alphas = render_alphas[:, ...]

		background = self._get_background_color()
		render_colors = render_colors_depths[:, ..., :3] + (1 - render_alphas) * background
		render_colors = torch.clamp(render_colors, 0.0, 1.0)

		(
			render_thermal_rad_depth,
			render_thermal_alpha,
			render_thermal_normals,
			thermal_normals_from_depth,
			render_distort,
			_,
			_,
		) = rasterization_2dgs(
			means=means_crop,
			quats=quats_crop,  # rasterization does normalization internally
			scales=torch.exp(scales_crop),
			opacities=torch.sigmoid(thermal_opacities_crop).squeeze(-1),
			colors=torch.sigmoid(thermal_radiances_crop), 
			viewmats=viewmat, 
			Ks=K,  # [1, 3, 3]
			width=W,
			height=H,
			packed=self.config.packed,
			near_plane=self.config.near_plane,
			far_plane=self.config.far_plane,
			render_mode="RGB+ED",  # don't need depth for thermal
			sh_degree=None, # scalar, no SH
			sparse_grad=False,
			absgrad=self.config.use_absgrad,
		)

		# throw away the depth channel
		render_thermal_rad = render_thermal_rad_depth[:, ..., :thermal_radiances_crop.shape[-1]]

		# apply bilateral grid
		if self.config.use_bilateral_grid and self.training:
			if camera.metadata is not None and "cam_idx" in camera.metadata:
				render_colors = self._apply_bilateral_grid(render_colors, camera.metadata["cam_idx"], H, W)

		if render_mode == "RGB+ED":
			depth_im = render_colors_depths[:, ..., 3:4]
			depth_im = torch.where(render_alphas > 0, depth_im, depth_im.detach().max()).squeeze(0)
		else:
			depth_im = None

		if background.shape[0] == 3 and not self.training:
			background = background.expand(H, W, 3)


		outputs = {
			"rgb": render_colors.squeeze(0),  # type: ignore
			"depth": depth_im,  # type: ignore
			"accumulation": render_alphas.squeeze(0),  # type: ignore
			"thermal_accumulation": render_thermal_alpha.squeeze(0),  # type: ignore
			"background": background,  # type: ignore
			"normals": render_normals.squeeze(0),  # type: ignore
			"thermal_normals": render_thermal_normals.squeeze(0),  # type: ignore
			"normals_from_depth": normals_from_depth.squeeze(0) if normals_from_depth is not None else None,  # type: ignore
			"thermal_normals_from_depth": thermal_normals_from_depth.squeeze(0) if thermal_normals_from_depth is not None else None,  # type: ignore
			"distort": render_distort.squeeze(0) if render_distort is not None else None,  # type: ignore
		}  # type: ignore

		if self.config.heat_loss:
			heat_outputs = {}
			for i in range(self.thermal_channels):
				thermal_img_channel = render_thermal_rad[..., i:i+1].squeeze(0)
				heat_outputs[f"thermal_{i}"] = thermal_img_channel
				heat_outputs[f"thermal_tonemapped_{i}"] = thermal_img_channel ** (1 / 2.2)

			outputs.update(heat_outputs)

		return outputs


	def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
		"""Computes and returns the losses dict.

		Args:
			outputs: the output to compute loss dict to
			batch: ground truth batch corresponding to outputs
			metrics_dict: dictionary of metrics, some of which we can use for loss
		"""

		# RGB loss
		gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
		pred_img = outputs["rgb"]
		alphas = outputs["accumulation"]

		# Set masked part of both ground-truth and rendered image to black.
		# This is a little bit sketchy for the SSIM loss.
		if "mask" in batch:
			# batch["mask"] : [H, W, 1]
			mask = self._downscale_if_required(batch["mask"])
			mask = mask.to(self.device)
			assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
			gt_img = gt_img * mask
			pred_img = pred_img * mask

		Ll1 = torch.abs(gt_img - pred_img).mean()
		simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
		if self.config.use_scale_regularization and self.step % 10 == 0:
			scale_exp = torch.exp(self.scales)
			scale_reg = (
				torch.maximum(
					scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
					torch.tensor(self.config.max_gauss_ratio),
				)
				- self.config.max_gauss_ratio
			)
			scale_reg = 0.1 * scale_reg.mean()
		else:
			scale_reg = torch.tensor(0.0).to(self.device)

	
		main_loss = (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss
		loss_dict = {
			"main_loss": (2 - self.config.heat_lambda) * main_loss,
			"scale_reg": scale_reg,
		}

		if self.config.heat_loss and self.step > self.config.heat_start_iter:
			if "heat_image" in batch:
				if len(batch["heat_image"].shape) == 3:
					heat_img = self.get_gt_img(batch["heat_image"])[..., 0:1]
				elif len(batch["heat_image"].shape) == 2:
					heat_img = self.get_gt_img(batch["heat_image"].unsqueeze(-1))
				
				if self.config.use_tonemapping:
					heat_img = heat_img ** (1 / 2.2)
				
				subset_idx = batch["subset_idx"]

				if self.config.use_tonemapping:
					heat_pred = outputs[f"thermal_tonemapped_{subset_idx}"]
				else:
					heat_pred = outputs[f"thermal_{subset_idx}"]

				heat_progress = (self.step - self.config.heat_start_iter) / (self.config.heat_full_iter - self.config.heat_start_iter)
				heat_lambda = self.config.heat_lambda * min(heat_progress, 1.0)
				heat_l1loss = heat_lambda * torch.abs(heat_img - heat_pred).mean()
				loss_dict["heat_loss"] = heat_l1loss

				thermal_reg = 1e-5 * torch.sigmoid(self.thermal_radiances).mean()
				loss_dict["thermal_reg"] = thermal_reg
			else:
				print("No heat_image in batch")

		# Entropy loss to encourage the opacities to be binary
		if self.step > self.config.entropy_start_iter and self.config.entropy_lambda > 0:
			opacities_sigmoid = torch.sigmoid(self.thermal_opacities)
			entropy = -(
				opacities_sigmoid * torch.log(opacities_sigmoid + 1e-6) 
				+ (1 - opacities_sigmoid) * torch.log(1 - opacities_sigmoid + 1e-6)
			).mean()
		else:
			entropy = torch.tensor(0.0).to(self.device)
		
		loss_dict["entropy_loss"] = self.config.entropy_lambda * entropy


		if self.config.depth_loss:
			raise NotImplementedError

		if self.config.normal_loss:
			if self.step > self.config.normal_start_iter:
				curr_normal_lambda = self.config.normal_lambda
			else:
				curr_normal_lambda = 0.0

			def _normal_loss(alphas, normals, normals_from_depth):
				normals = normals.squeeze(0).permute((2, 0, 1))
				normals_from_depth *= alphas.squeeze(0).detach()
				if len(normals_from_depth.shape) == 4:
					normals_from_depth = normals_from_depth.squeeze(0)
				normals_from_depth = normals_from_depth.permute((2, 0, 1))
				normal_error = (1 - (normals * normals_from_depth).sum(dim=0))[None]
				normalloss = normal_error.mean()
				return normalloss

			# normal consistency loss
			assert "accumulation" in outputs
			assert "normals" in outputs
			assert "normals_from_depth" in outputs
			assert "thermal_accumulation" in outputs
			assert "thermal_normals" in outputs
			assert "thermal_normals_from_depth" in outputs

			rgb_normal_loss = _normal_loss(outputs["accumulation"], outputs["normals"], outputs["normals_from_depth"])
			thermal_normal_loss = _normal_loss(outputs["thermal_accumulation"], outputs["thermal_normals"], outputs["thermal_normals_from_depth"])

			# normals = normals.squeeze(0).permute((2, 0, 1))
			# normals_from_depth *= alphas.squeeze(0).detach()
			# if len(normals_from_depth.shape) == 4:
			# 	normals_from_depth = normals_from_depth.squeeze(0)
			# normals_from_depth = normals_from_depth.permute((2, 0, 1))
			# normal_error = (1 - (normals * normals_from_depth).sum(dim=0))[None]
			# normalloss = curr_normal_lambda * normal_error.mean()

			normalloss = curr_normal_lambda * (rgb_normal_loss + thermal_normal_loss) / 2

		else:
			normalloss = torch.tensor(0.0).to(self.device)
		loss_dict["normal_loss"] = normalloss

		if self.config.dist_loss:
			if self.step > self.config.dist_start_iter:
				curr_dist_lambda = self.config.dist_lambda
			else:
				curr_dist_lambda = 0.0
			
			distloss = curr_dist_lambda * torch.mean(outputs["distort"])

			loss_dict["dist_loss"] = distloss


		if self.training:
			# Add loss from camera optimizer
			self.camera_optimizer.get_loss_dict(loss_dict)
			if self.config.use_bilateral_grid:
				loss_dict["tv_loss"] = 10 * total_variation_loss(self.bil_grids.grids)

		return loss_dict

	
	def get_image_metrics_and_images(self, outputs, batch):
		# Compute metrics from 3dgs (psnr, ssim, lpips)
		metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)

		# Compute heat metrics from 2dgs (heat loss)
		if self.config.heat_loss:
			heat_gt = self.get_gt_img(batch["heat_image"])
			if self.config.use_tonemapping:
				heat_gt = heat_gt ** (1 / 2.2)

			if len(heat_gt.shape) == 3:
				heat_gt = heat_gt[..., 0:1]
			elif len(batch["heat_image"].shape) == 2:
				heat_gt = heat_gt.unsqueeze(-1)
			else:
				raise ValueError(f"Invalid heat_image shape, got {heat_gt.shape}, expected [H, W, 1] or [H, W]")

			subset_idx = batch["subset_idx"]

			if self.config.use_tonemapping:
				heat_pred = outputs[f"thermal_tonemapped_{subset_idx}"]
			else:
				heat_pred = outputs[f"thermal_{subset_idx}"]

			heat_combined = torch.cat([heat_gt, heat_pred], dim=1)

			# Switch images from [H, W, C] to [1, C, H, W] for metrics computations
			heat_gt = torch.moveaxis(heat_gt, -1, 0)[None, ...]
			heat_pred = torch.moveaxis(heat_pred, -1, 0)[None, ...]

			# print(heat_gt.shape, heat_pred.shape)

			metrics_dict.update({
				"heat_psnr": self.psnr(heat_pred, heat_gt),
				"num_gaussians": torch.tensor(self.means.shape[0], dtype=torch.float),  # cast to float because it has to be float or complex
				# "heat_ssim": self.ssim(heat_pred, heat_gt),
				# "heat_lpips": self.lpips(heat_pred, heat_gt),
			})
			images_dict.update({
				"heat": heat_combined,
			})

		return metrics_dict, images_dict
	
	
	@property
	def thermal_opacities(self):
		if self.config.use_separate_opacities:
			return self.gauss_params["thermal_opacities"]
		else:
			return self.gauss_params["opacities"]

	@property
	def thermal_radiances(self):
		return self.gauss_params["thermal_radiances"]


	def get_training_callbacks(self, training_callback_attributes):
		# callbacks = super().get_training_callbacks(training_callback_attributes)
		# we handle the post step call backs ourselves
		callbacks = []

		def _mcmc_post_step(step: int):
			if isinstance(self.strategy, MCMCStrategy):
				self.strategy.step_post_backward(
					self.gauss_params, 
					self.optimizers, 
					self.strategy_state,
					step, 
					self.info,
					lr=1e-3)

		callbacks.append(
			TrainingCallback(
				[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
				self.step_cb,
				args=[training_callback_attributes.optimizers],
			)
		)

		callbacks.append(
			TrainingCallback(
				where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
				func=_mcmc_post_step,
			)
		)

		return callbacks