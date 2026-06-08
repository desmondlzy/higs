"""
Convert wolfenschiessen thermal CSVs to EXR and structure into higs dataset format.

Input:  data/cdl/wolfenschiessen/
          thermal_csv/   - 509 FLIR CSV files
          rgb_aligned/
            input/       - RGB images
            sparse/0/    - COLMAP sparse reconstruction

Output: data/higs/wolfenschiessen/
          heat-000/      - normalized EXR files + parameters.json
          heat_ranges.json
          metadata.json
          rgb/           - symlink to rgb_aligned/input
          sparse/        - symlink to rgb_aligned/sparse
"""

from pathlib import Path
import numpy as np
import imageio.v3 as iio
import json
import shutil

from heatgaussian.read_flir_temperature_csv import read_flir_temperature_csv
from heatgaussian.temperature_to_raw import temperature_to_raw

project_root = Path(__file__).parent.parent

source_dir = project_root / "data/cdl/wolfenschiessen"
csv_dir = source_dir / "thermal_csv"
rgb_input_dir = source_dir / "rgb_aligned/input"
sparse_dir = source_dir / "rgb_aligned/sparse"

output_dir = project_root / "data/higs/wolfenschiessen"
heat_dir = output_dir / "heat-000"

assert csv_dir.exists(), f"{csv_dir} does not exist"
assert rgb_input_dir.exists(), f"{rgb_input_dir} does not exist"
assert sparse_dir.exists(), f"{sparse_dir} does not exist"

heat_dir.mkdir(parents=True, exist_ok=True)

# ── Step 1: Read all CSVs and convert to radiance ────────────────────────────

csvs = sorted(csv_dir.glob("*.csv"))
print(f"Found {len(csvs)} CSV files")

parameters_list = []
radiances = {}
temperature_imgs = {}

for csv in csvs:
    temperature_img, params = read_flir_temperature_csv(csv)
    filename = csv.stem
    assert filename not in radiances, f"Duplicate filename: {filename}"

    parameters_list.append(params)
    radiances[filename] = temperature_to_raw(temperature_img, **params)
    temperature_imgs[filename] = temperature_img

# ── Step 2: Compute global ranges ────────────────────────────────────────────

max_radiance = max(r.max() for r in radiances.values())
min_radiance = min(r.min() for r in radiances.values())
max_temp = max(t.max() for t in temperature_imgs.values())
min_temp = min(t.min() for t in temperature_imgs.values())

print(f"Max radiance: {max_radiance:.2f}, Max temperature: {max_temp:.3f}")
print(f"Min radiance: {min_radiance:.2f}, Min temperature: {min_temp:.3f}")

# ── Step 3: Save normalized EXR files ────────────────────────────────────────

for filename, rad in radiances.items():
    normalized = ((rad - min_radiance) / (max_radiance - min_radiance)).astype(np.float32)
    assert normalized.dtype == np.float32
    assert np.all(normalized >= 0) and np.all(normalized <= 1), f"Not normalized: {filename}"

    iio.imwrite(
        str(heat_dir / f"{filename}.exr"),
        normalized[..., np.newaxis],
        extension=".exr",
    )

print(f"Saved {len(radiances)} EXR files to {heat_dir}")

# ── Step 4: Compute mean camera parameters ───────────────────────────────────

mean_params = {}
for params in parameters_list:
    for key, value in params.items():
        mean_params.setdefault(key, []).append(value)
mean_params = {k: float(np.mean(v)) for k, v in mean_params.items()}

# ── Step 5: Write parameters.json into heat-000 ──────────────────────────────

subset_params = {
    **mean_params,
    "max_radiance": float(max_radiance),
    "min_radiance": float(min_radiance),
    "max_temperature": float(max_temp),
    "min_temperature": float(min_temp),
}
with open(heat_dir / "parameters.json", "w") as f:
    json.dump(subset_params, f, indent=4)
print(f"Saved parameters.json")

# ── Step 6: Write heat_ranges.json at root ───────────────────────────────────

heat_ranges = {
    "max_radiance": float(max_radiance),
    "min_radiance": float(min_radiance),
    "max_temperature": float(max_temp),
    "min_temperature": float(min_temp),
}
with open(output_dir / "heat_ranges.json", "w") as f:
    json.dump(heat_ranges, f, indent=4)
print(f"Saved heat_ranges.json")

# ── Step 7: Write metadata.json at root ──────────────────────────────────────

metadata = {
    "synthetic": False,
    "type": "heat-colmap",
    "name": "wolfenschiessen",
    "subsets": {
        "heat-000": {
            "data": "./heat-000",
            "split": "train",
        }
    },
}
with open(output_dir / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)
print(f"Saved metadata.json")

# ── Step 8: Copy rgb and sparse ──────────────────────────────────────────────

rgb_dest = output_dir / "rgb"
if rgb_dest.is_symlink():
    rgb_dest.unlink()
elif rgb_dest.exists():
    shutil.rmtree(rgb_dest)
shutil.copytree(rgb_input_dir, rgb_dest)
print(f"Copied rgb from {rgb_input_dir}")

sparse_dest = output_dir / "sparse"
if sparse_dest.is_symlink():
    sparse_dest.unlink()
elif sparse_dest.exists():
    shutil.rmtree(sparse_dest)
shutil.copytree(sparse_dir, sparse_dest)
print(f"Copied sparse from {sparse_dir}")

print("\nDone. Dataset structure:")
for p in sorted(output_dir.iterdir()):
    print(f"  {p.name}/  ({len(list(p.iterdir()))} items)" if p.is_dir() else f"  {p.name}")
