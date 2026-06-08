from pathlib import Path
import csv

import numpy as np


def read_flir_temperature_csv(filepath: Path):
	"""
	return temperature image, and parameters from the csv file
	"""
	with open(filepath) as csvfile:
		reader = csv.reader(csvfile)

		# skip the first 4 lines of uselessness
		for _ in range(4):
			next(reader)

		# regular expression for matching the first number with dot
		emissivity = float(next(reader)[2].strip())
		refl_temp = float(next(reader)[2].strip().split()[0])
		distance = float(next(reader)[2].strip().split()[0])
		atmos_temp = float(next(reader)[2].strip().split()[0])
		ext_optics_temp = float(next(reader)[2].strip().split()[0])
		ext_optics_trans = float(next(reader)[2].strip())
		relative_humidity = float(next(reader)[2].strip().split()[0])

		# skip empty line
		next(reader)

		temps = []
		for row in reader:
			if len(row) == 0:
				break

			readings = [float(i) for i in row[1:]]
			temps.append(readings)
	
	params = {
		'E': emissivity,
		'OD': distance,
		'RTemp': refl_temp,
		'ATemp': atmos_temp,
		'IRWTemp': ext_optics_temp,
		'IRT': ext_optics_trans,
		'RH': relative_humidity
	}

	return np.array(temps), params
