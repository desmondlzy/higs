#%%
"""
process the stardis output ht files into exr files
"""

import os
from pathlib import Path
import numpy as np
import imageio.v3 as iio
import json
import shutil
import subprocess
import cv2

from heatgaussian.read_stardis_ht import read_stardis_ht

# %%
# name = "bunny_table_ro_spp32"
# name = "bunny_box_ro_spp512"
# name = "heater_bunny_box_moved2_ro"
# name = "bunny_box_ro_moved2"
name = "bunny_box_moved_ro_spp512"
# name = "teapot2_bunny_table_ro_spp512"
# name = "torous_bunny_box_ro"	
# name = "heater_bunny_box_ro_spp1024"
# name = "heater_human_box_ro"
# name = "heater_human_desk_ro"
# name = "torous_bunny_box_jitterless_ro"	


data_root = Path(__file__).parents[1] / f"devtools/multiview-datagen/datasets/{name}"
new_root = Path(__file__).parents[1] / f"data/cdl/{name}"

# data_root = Path(__file__).parents[1] / "devtools/multiview-datagen/datasets/bunny_table_v4_spp512"
# new_root = Path(__file__).parents[1] / "data/cdl/bunny_table_v4_spp512"
reflection_dirname = "reflection"

metas = [
	{
		"name": "heat-000",
		"source_dir": data_root / "heat-000" / "heat",
		"target_dir": new_root / "heat-000" / "linear",
		"split": "train",
	},
	{
		"name": "heat-001",
		"source_dir": data_root / "heat-001" / "heat",
		"target_dir": new_root / "heat-001" / "linear",
		"split": "train",
	},	
	{
		"name": "heat-002",
		"source_dir": data_root / "heat-002" / "heat",
		"target_dir": new_root / "heat-002" / "linear",
		"split": "train",
	},
	# {
	# 	"name": "heat-003",
	# 	"source_dir": data_root / "heat-003" / "heat",
	# 	"target_dir": new_root / "heat-003" / "linear",
	# 	"split": "test",
	# },

]

source_dirs = [m["source_dir"] for m in metas]
target_dirs = [m["target_dir"] for m in metas]
splits = [m["split"] for m in metas]

subset_map_file = data_root / "subsets.json"

hts = []
for source_dir in source_dirs:
	assert source_dir.exists(), f"{source_dir} does not exist"
	hts_in_dir = sorted(source_dir.glob("*.ht"))
	print(f"found {len(hts_in_dir)} ht files in {source_dir}")
	hts.extend(hts_in_dir)

for target_dir in target_dirs:
	target_dir.mkdir(parents=True, exist_ok=True)

	reflection_dir = target_dir.parent / reflection_dirname
	reflection_dir.mkdir(parents=True, exist_ok=True)

print(f"in total, read {len(hts)} ht files from {len(source_dirs)} source directories")

parameters = {}
radiances = {}
temperature_imgs = {}
img_target_dir_map = {}


def temperature_to_radiance(temperature):
	boltzmann = 1
	return temperature ** 4 * boltzmann / np.pi

# boltzmann = 1.380649e-23
for i, ht in enumerate(hts):
	try:
		temperature_img = read_stardis_ht(ht)
	except ValueError as e:
		print(f"Error reading {ht}, skipping... Error: {e}")
		continue

	radiance_img = temperature_to_radiance(temperature_img)

	filename = ht

	assert filename not in radiances, f"{filename} already in radiances, repeated entry?"

	# for key, value in params.items():
	# 	if key not in parameters:
	# 		parameters[key] = []
	# 	parameters[key].append(value)
	
	# rad = temperature_to_raw(temperature_img, **params)

	radiances[filename] = radiance_img
	temperature_imgs[filename] = temperature_img
	img_target_dir_map[filename] = target_dirs[source_dirs.index(ht.parent)]


max_radiance = max([rad.max() for rad in radiances.values()])
min_radiance = min([rad.min() for rad in radiances.values()])
max_temp = max([temp.max() for temp in temperature_imgs.values()])
min_temp = min([temp.min() for temp in temperature_imgs.values()])

print(f"Successfully parse all the ht files")
print(f"Max radiance: {max_radiance}, Max temperature: {max_temp}")
print(f"Min radiance: {min_radiance}, Min temperature: {min_temp}")

#%%
has_reflection = True

for filename, rad in radiances.items():
	normalized_rad = ((rad - min_radiance) / (max_radiance - min_radiance)).astype(np.float32)
	assert normalized_rad.dtype == np.float32, f"dtype not float32: {filename}, got {normalized_rad.dtype}"
	assert np.all(normalized_rad >= 0) and np.all(normalized_rad <= 1), f"Radiance not normalized: {filename}"

	target_dir = img_target_dir_map[filename]

	if has_reflection:
		reflection_dir = target_dir.parent / reflection_dirname
		source_refl_filename = filename.parent.parent / reflection_dirname / filename.name
		reflection_temperature = read_stardis_ht(source_refl_filename)
		reflection_radiance = temperature_to_radiance(reflection_temperature)
		normalized_reflection_rad = ((reflection_radiance - min_radiance) / (max_radiance - min_radiance)).astype(np.float32)
		assert np.all(normalized_reflection_rad >= 0) and np.all(normalized_reflection_rad <= 1), f"Reflection radiance not normalized: {filename}"

	exr_stem = filename.stem

	assert normalized_rad.dtype == np.float32, f"dtype not float32: {filename}, got {normalized_rad.dtype}"
	iio.imwrite(
		str(target_dir / f"{exr_stem}.exr"), 
		normalized_rad[..., np.newaxis],
		extension=".exr")

	if has_reflection:
		iio.imwrite(
			str(reflection_dir / f"{exr_stem}.exr"),
			normalized_reflection_rad[..., np.newaxis],
			extension=".exr")
	

# save parameters mean values
mean_parameters = {key: np.mean(value) for key, value in parameters.items()}
mean_parameters["max_radiance"] = max_radiance
mean_parameters["min_radiance"] = min_radiance
mean_parameters["max_temperature"] = max_temp
mean_parameters["min_temperature"] = min_temp

for target_dir in target_dirs:
	with open(target_dir.parent / "heat_ranges.json", "w") as f:
		json.dump(mean_parameters, f, indent=4)
		print(f"Parameters saved to {target_dir / 'heat_ranges.json'}")

with open(target_dir.parent.parent / "heat_ranges.json", "w") as f:
	json.dump(mean_parameters, f, indent=4)
	print(f"Parameters saved to {target_dir.parent.parent / 'heat_ranges.json'}")

#%%
# copy the heatmodels folder to the new root
for meta in metas:
	heatmodels_source = meta["source_dir"].parent / "heatmodels"
	heatmodels_target = meta["target_dir"].parent / "heatmodels"

	if heatmodels_source.exists():
		shutil.copytree(heatmodels_source, heatmodels_target, dirs_exist_ok=True)
		print(f"Copied heatmodels from {heatmodels_source} to {heatmodels_target}")
	else:
		print(f"Heatmodels source directory {heatmodels_source} does not exist, skipping copy.")

#%%
# copy all the RGB images by reading the transforms
for m in metas:
	src_dir = m["source_dir"]
	tar_dir = m["target_dir"]

	transforms_json = src_dir.parent / f"transforms_train.json"
	# copy the original transforms json as a backup
	with open(transforms_json, "r") as f:
		transforms = json.load(f)

	for frame in transforms["frames"]:
		# replace the heat image path with the new exr path
		orig_img_path = src_dir.parent / frame["file_path"]
		new_img_path = tar_dir.parent / frame["file_path"]

		new_img_path.parent.mkdir(parents=True, exist_ok=True)

		if not orig_img_path.exists():
			print(f"Original image path does not exist! {orig_img_path}")
			continue

		if not new_img_path.exists():
			shutil.copy(orig_img_path, new_img_path)
		else:
			...

#%%
# # move the subsets.json file to the new root
# shutil.copy(subset_map_file, new_root / "subsets.json")
with open(subset_map_file, "r") as f:
	subset_map = json.load(f)

for m in metas:
	split = m["split"]
	m_name = m["name"]
	subset_map["subsets"][m_name]["split"] = split

with open(new_root / "subsets.json", "w") as f:
	json.dump(subset_map, f, indent=4)
	print(f"Saved subset map to {new_root / 'subsets.json'}")
	
#%%
for m in metas:
	src_dir = m["source_dir"]
	tar_dir = m["target_dir"]
	split = m["split"]

	transforms_json = src_dir.parent / f"transforms_{split}.json"
	# copy the original transforms json as a backup
	shutil.copy(transforms_json, transforms_json.with_suffix(".bak.json"))

	with open(transforms_json, "r") as f:
		transforms = json.load(f)
		print(f"Loaded transforms from {transforms_json}")

	for frame in transforms["frames"]:
		# replace the heat image path with the new exr path
		orig_heat_img_path = Path(frame["heat_img_path"])
		img_stem = orig_heat_img_path.stem
		frame["heat_img_path"] = f"./linear/{img_stem}.exr"
	
	target_transforms_file = tar_dir.parent / f"transforms_{split}.json"
	with open(target_transforms_file, "w") as f:
		json.dump(transforms, f, indent=4)
		print(f"Saved transforms to {target_transforms_file}")
	

	for s in ("train", "val", "test"):
		if s == split:
			continue

		shutil.copy(
			target_transforms_file, 
			tar_dir.parent / f"transforms_{s}.json",
		)
		
	# 	empty_json_file = tar_dir.parent / f"transforms_{s}.json"
	# 	with open(empty_json_file, "w") as f:
	# 		json.dump({"frames": []}, f, indent=4)
	# 		print(f"Saved empty transforms to {empty_json_file}")

#%%
blender_exec = os.environ.get("BLENDER_EXECUTABLE", "blender")
for m in metas:
	target_dir = m["target_dir"]
	config_file = json.load(open(data_root / "config.json"))
	template_name = config_file["template"]
	if "spp" in name:
		blend_name = "_".join(name.split("_")[:-1])
	else:
		blend_name = template_name.split("/")[1]

	blend_file = str(Path(__file__).parents[1] / f"devtools/multiview-datagen/configs/{blend_name}.blend")
	emissivity_gen_cmd = [
		blender_exec, "-b", 
		blend_file,
		"--python", 
		str(Path(__file__).parents[1] / "devtools/multiview-datagen/render_emissivities.py"),
		"--",
		str(target_dir.parent / "transforms_train.json"),
	]

	print(f"Running emissivity generation command: {' '.join(emissivity_gen_cmd)}")

	subprocess.run(emissivity_gen_cmd, check=True)


# %%
