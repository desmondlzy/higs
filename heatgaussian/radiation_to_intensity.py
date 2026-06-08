"""Convert radiation to pixel intensity."""


def radiation_to_intensity(radiation, min_radiance: float, max_radiance: float):
	"""
	Linear map from radiation to pixel intensity (0, 1).

	Args:
		radiation: Radiation values in range [min_radiance, max_radiance]
		min_radiance: Minimum radiance value
		max_radiance: Maximum radiance value

	Returns:
		Pixel intensity values in range [0, 1]
	"""
	return (radiation - min_radiance) / (max_radiance - min_radiance)
