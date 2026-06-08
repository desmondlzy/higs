#%%
from pathlib import Path
import numpy as np
import argparse


import matplotlib.pyplot as plt

#%%
def read_ht(ht_file: Path):
	"""
	ht file format:
	https://www.meso-star.com/projects/stardis/man/man5/stardis-output.5.html#INFRARED_IMAGE
	https://www.meso-star.com/projects/htrdr/man/man5/htrdr-image.5.html
	"""
	with open(ht_file, "r") as f:
		width, height = map(int, f.readline().split())
		n_pixels = width * height

		lines = f.readlines()


	img = np.zeros((height, width), dtype=float)

	curr_pixel = 0
	for l in lines:
		if l.startswith("#") or l.strip() == "":
			continue
		
		# the middle 4 values are always 0
		(
			temperature_expected, temperature_error,
			_, _, _, _, 
			time_expected, time_error,
		) = map(float, l.split())

		r, c = divmod(curr_pixel, width)
		img[r, c] = temperature_expected

		curr_pixel += 1
	
	return img


def convert_ht_to_png(
	ht_file: Path,
	png_file: Path,
	temperature_lower: float,
	temperature_upper: float,
	cmap: str = "inferno",
):
	temp_img = read_ht(ht_file)
	rad_img = temp_img ** 4
	vmin = temperature_lower ** 4
	vmax = temperature_upper ** 4
	plt.imsave(png_file, rad_img, cmap=cmap, vmin=vmin, vmax=vmax)

	return rad_img

#%%
if __name__ == "__main__":
	# ht_file = Path(__file__).parent / "datasets/bunny_teapot_v1_spp32/heat-000/heat/cam_03.ht"
	# png_file = ht_file.with_suffix(".png")
	# assert ht_file.exists(), f"File {ht_file} does not exist"
	# img = read_ht(ht_file)
	parser = argparse.ArgumentParser()
	parser.add_argument("ht_file", type=Path, help="Path to the ht file")
	parser.add_argument("png_file", type=Path, help="Path to the png file")
	parser.add_argument("--low-temp", type=float, required=True, help="Lower temperature limit")
	parser.add_argument("--high-temp", type=float, required=True, help="Upper temperature limit")
	parser.add_argument("--cmap", type=str, default="inferno", help="Colormap to use")

	args = parser.parse_args()

	convert_ht_to_png(args.ht_file, args.png_file, args.low_temp, args.high_temp, args.cmap)
	print("write to", args.png_file)
