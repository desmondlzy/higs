"""Convert pixel intensity to radiation."""


def intensity_to_radiation(intensity, min_radiance: float, max_radiance: float):
	"""
	Linear map from pixel intensity (0, 1) to radiation.

	Args:
		intensity: Pixel intensity values in range [0, 1]
		min_radiance: Minimum radiance value
		max_radiance: Maximum radiance value

	Returns:
		Radiation values in range [min_radiance, max_radiance]
	"""
	return intensity * (max_radiance - min_radiance) + min_radiance
