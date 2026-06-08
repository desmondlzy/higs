from pathlib import Path
import numpy as np

def read_stardis_ht(ht_file: Path):
	"""
	Read a .ht file and return the temperature image.
	spec: 
		https://www.meso-star.com/projects/stardis/man/man5/stardis-output.5.html#INFRARED_IMAGE
		https://www.meso-star.com/projects/htrdr/man/man5/htrdr-image.5.html
	"""
	with open(ht_file, "r") as f:
		width, height = map(int, f.readline().split())

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
