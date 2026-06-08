#%%
from pathlib import Path
import numpy as np
import imageio.v3 as iio
import json
import cv2

from heatgaussian.read_flir_temperature_csv import read_flir_temperature_csv
from heatgaussian.temperature_to_raw import temperature_to_raw
from heatgaussian.raw_to_temperature import raw_to_temperature
# %%
project_root = Path(__file__).parent.parent
splited = False

### head
# source_dirs = [
# 	project_root / "data/cdl/head_v02/head_v02_csv",
# 	project_root / "data/cdl/head_v03/head_v03_csv",
# 	project_root / "data/cdl/head_v04/head_v04_csv",
# ]
# target_dirs = [
# 	project_root / "data/cdl/head_v02_03_04/linear_v02",
# 	project_root / "data/cdl/head_v02_03_04/linear_v03",
# 	project_root / "data/cdl/head_v02_03_04/linear_v04",
# ]

### Statue
# source_dirs = [
# 	project_root / "data/cdl/statue_v02/statue_v02_csv",
# 	project_root / "data/cdl/statue_v03/statue_v03_csv",
# 	project_root / "data/cdl/statue_v04/statue_v04_csv",
# ]
# target_dirs = [
# 	project_root / "data/cdl/statue_v02_03_04/linear_v02",
# 	project_root / "data/cdl/statue_v02_03_04/linear_v03",
# 	project_root / "data/cdl/statue_v02_03_04/linear_v04",
# ]

### Vase
# source_dirs = [
# 	project_root / "data/cdl/vase_v02/vase_v02_csv",
# 	project_root / "data/cdl/vase_v03/vase_v03_csv",
# 	project_root / "data/cdl/vase_v04/vase_v04_csv",
# ]
# target_dirs = [
# 	project_root / "data/cdl/vase_v02_03_04/linear_v02",
# 	project_root / "data/cdl/vase_v02_03_04/linear_v03",
# 	project_root / "data/cdl/vase_v02_03_04/linear_v04",
# ]

### Teapot
source_dirs = [
	project_root / "data/cdl/teapot_v14/teapot_v14_csv",
	project_root / "data/cdl/teapot_v15/teapot_v15_csv",
	project_root / "data/cdl/teapot_v17/teapot_v17_csv",
]
target_dirs = [
	project_root / "data/cdl/teapot_v14_15_17/linear_v14",
	project_root / "data/cdl/teapot_v14_15_17/linear_v15",
	project_root / "data/cdl/teapot_v14_15_17/linear_v17",
]

### Kettle
# source_dirs = [
# 	project_root / "data/cdl/kettle_v01/kettle_v01_csv",
# 	project_root / "data/cdl/kettle_v02/kettle_v02_csv",
# 	project_root / "data/cdl/kettle_v03/kettle_v03_csv",
# ]
# target_dirs = [
# 	project_root / "data/cdl/kettle_v01_02_03/linear_v01",
# 	project_root / "data/cdl/kettle_v01_02_03/linear_v02",
# 	project_root / "data/cdl/kettle_v01_02_03/linear_v03",
# ]

csvs = []
for source_dir in source_dirs:
	assert source_dir.exists(), f"{source_dir} does not exist"
	csvs.extend(sorted(source_dir.glob("*.csv")))

for target_dir in target_dirs:
	target_dir.mkdir(parents=True, exist_ok=True)

parameters = {}
radiances = {}
temperature_imgs = {}
img_target_dir_map = {}

for csv in csvs:
	temperature_img, params = read_flir_temperature_csv(csv)
	filename = csv.stem

	assert filename not in radiances, f"{filename} already in radiances, repeated entry?"

	for key, value in params.items():
		if key not in parameters:
			parameters[key] = []
		parameters[key].append(value)
	
	rad = temperature_to_raw(temperature_img, **params)

	radiances[filename] = rad
	temperature_imgs[filename] = temperature_img
	img_target_dir_map[filename] = target_dirs[source_dirs.index(csv.parent)]


max_radiance = max([rad.max() for rad in radiances.values()])
min_radiance = min([rad.min() for rad in radiances.values()])
max_temp = max([temp.max() for temp in temperature_imgs.values()])
min_temp = min([temp.min() for temp in temperature_imgs.values()])

print(f"Max radiance: {max_radiance}, Max temperature: {max_temp}")
print(f"Min radiance: {min_radiance}, Min temperature: {min_temp}")

#%%
for filename, rad in radiances.items():
	normalized_rad = ((rad - min_radiance) / (max_radiance - min_radiance)).astype(np.float32)
	assert normalized_rad.dtype == np.float32, f"dtype not float32: {filename}, got {normalized_rad.dtype}"
	assert np.all(normalized_rad >= 0) and np.all(normalized_rad <= 1), f"Radiance not normalized: {filename}"

	# tonemapped_rad = normalized_rad ** (1 / 2.2)
	# normalized_rad_int = (normalized_rad * 255).astype(np.uint8)
	# tonemapped_rad_int = (tonemapped_rad * 255).astype(np.uint8)

	target_dir = img_target_dir_map[filename]

	# tonemapped_dir = target_dir.parent / target_dir.name.replace("linear", "tonemapped")
	# tonemapped_dir.mkdir(exist_ok=True)

	# save
	# iio.imwrite(target_dir / f"{filename}.png", normalized_rad_int)
	# iio.imwrite(tonemapped_dir / f"{filename}.png", tonemapped_rad_int)

	iio.imwrite(
		str(target_dir / f"{filename}.exr"), 
		normalized_rad[..., np.newaxis],
		extension=".exr")

# save parameters mean values
mean_parameters = {key: np.mean(value) for key, value in parameters.items()}
mean_parameters["max_radiance"] = max_radiance
mean_parameters["min_radiance"] = min_radiance
mean_parameters["max_temperature"] = max_temp
mean_parameters["min_temperature"] = min_temp

for target_dir in target_dirs:
	with open(target_dir / "heat_ranges.json", "w") as f:
		json.dump(mean_parameters, f, indent=4)
		print(f"Parameters saved to {target_dir / 'heat_ranges.json'}")
	
