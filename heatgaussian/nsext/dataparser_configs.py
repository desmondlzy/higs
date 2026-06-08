from dataclasses import dataclass
from pathlib import Path

from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from heatgaussian.nsext.heatblender_dataparser import HeatBlenderDataParserConfig
from heatgaussian.nsext.heatcolmap_dataparser import HeatColmapDataParserConfig
from heatgaussian.nsext.multi_heat_dataparser import MultiHeatDataparserConfig
from heatgaussian.nsext.multi_heatcolmap_dataparser import MultiHeatColmapDataParserConfig
from heatgaussian.nsext.higs_heat_dataparser import HigsHeatDataparserConfig

HeatBlenderDataparser = DataParserSpecification(
	config=HeatBlenderDataParserConfig(),
)

HeatColmapDataparser = DataParserSpecification(
	config=HeatColmapDataParserConfig(
		images_path=Path("input_rgb"),
		heatimage_path=Path("input"),
		colmap_path=Path("sparse/0"),
	),
)

MultiHeatDataparser = DataParserSpecification(
	config=MultiHeatDataparserConfig(),
)

MultiHeatColmapDataparser = DataParserSpecification(
	config=MultiHeatColmapDataParserConfig(
		images_path=Path("input_rgb"),
		heatimage_path_pattern="ir*",
		colmap_path=Path("sparse/0"),
	),
)

HigsHeatDataparser = DataParserSpecification(
	config=HigsHeatDataparserConfig(),
)
