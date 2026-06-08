from dataclasses import dataclass

from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from heatgaussian.nsext.heatblender_dataparser import HeatBlenderDataParserConfig

HeatDataparser = DataParserSpecification(
	config=HeatBlenderDataParserConfig(),
)


def _common_params():
	from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
	from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig, FullImageDatamanager
	from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
	optimizers = {
		"means": {
			"optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
			"scheduler": ExponentialDecaySchedulerConfig(
				lr_final=1.6e-6,
				max_steps=30000,
			),
		},
		"features_dc": {
			"optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
			"scheduler": None,
		},
		"features_rest": {
			"optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
			"scheduler": None,
		},
		"opacities": {
			"optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
			"scheduler": None,
		},
		"thermal_opacities": {
			"optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
			"scheduler": None,
		},
		"scales": {
			"optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
			"scheduler": None,
		},
		"thermal_radiances": {
			"optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
			"scheduler": None,
		},
		"quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
		"camera_opt": {
			"optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
			"scheduler": ExponentialDecaySchedulerConfig(
				lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
			),
		},
		"bilateral_grid": {
			"optimizer": AdamOptimizerConfig(lr=5e-3, eps=1e-15),
			"scheduler": ExponentialDecaySchedulerConfig(
				lr_final=1e-4, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
			),
		},
	}

	return optimizers


def get_heat2dgs_config():
	from nerfstudio.plugins.types import MethodSpecification
	from nerfstudio.engine.trainer import TrainerConfig
	from nerfstudio.configs.base_config import ViewerConfig
	from nerfstudio.engine.trainer import TrainerConfig
	from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

	from heatgaussian.nsext.heat_2dgs import Heat2DGSModelConfig
	from heatgaussian.nsext.heat_dataset import HeatDataset
	from heatgaussian.nsext.multi_full_image_datamanager import (
		MultiFullImageDatamanagerConfig, 
		MultiFullImageDatamanager,
	)

	MultiFullImageHeatDatamanager = MultiFullImageDatamanager[HeatDataset]

	return MethodSpecification(
		config=TrainerConfig(
			method_name="heat-2dgs",
			steps_per_eval_image=100,
			steps_per_eval_batch=0,
			steps_per_save=2000,
			steps_per_eval_all_images=1000,
			max_num_iterations=30000,
			mixed_precision=False,
			pipeline=VanillaPipelineConfig(
				datamanager=MultiFullImageDatamanagerConfig(
					_target=MultiFullImageHeatDatamanager,
					dataparser=HeatBlenderDataParserConfig(
						load_heatimages=False),
					image_keys=["image", "heat_image"],
					allow_none=False,
					cache_images_type="float32",
				),
				model=Heat2DGSModelConfig(),
			),
			optimizers=_common_params(),
			viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
			vis="viewer",
		),
		description="Heat2DGS"
	)

def get_single_heat2dgs_config():
	from nerfstudio.plugins.types import MethodSpecification
	from nerfstudio.engine.trainer import TrainerConfig
	from nerfstudio.configs.base_config import ViewerConfig
	from nerfstudio.engine.trainer import TrainerConfig
	from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

	from heatgaussian.nsext.heat_2dgs import Heat2DGSModelConfig
	from heatgaussian.nsext.heat_dataset import HeatDataset
	from heatgaussian.nsext.multi_full_image_datamanager import (
		MultiFullImageDatamanagerConfig, 
		MultiFullImageDatamanager,
	)

	MultiFullImageHeatDatamanager = MultiFullImageDatamanager[HeatDataset]

	return MethodSpecification(
		config=TrainerConfig(
			method_name="single-heat-2dgs",
			steps_per_eval_image=100,
			steps_per_eval_batch=0,
			steps_per_save=2000,
			steps_per_eval_all_images=1000,
			max_num_iterations=30000,
			mixed_precision=False,
			pipeline=VanillaPipelineConfig(
				datamanager=MultiFullImageDatamanagerConfig(
					_target=MultiFullImageHeatDatamanager,
					dataparser=HeatBlenderDataParserConfig(
						load_heatimages=False),
					image_keys=["image", "heat_image"],
					allow_none=False,
					cache_images_type="float32",
				),
				model=Heat2DGSModelConfig(),
			),
			optimizers=_common_params(),
			viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
			vis="viewer",
		),
		description="Single Heat2DGS"
	)

heat2dgs_config = get_heat2dgs_config()
single_heat2dgs_config = get_single_heat2dgs_config()
