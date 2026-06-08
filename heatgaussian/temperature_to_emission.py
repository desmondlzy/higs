"""Convert temperature to thermal emission."""

import torch


def temperature_to_emission(temperature: torch.Tensor, emissivity: torch.Tensor) -> torch.Tensor:
	"""
	Convert temperature to thermal emission.

	Args:
		temperature: Temperature values (in radiometric units)
		emissivity: Emissivity values in range [0, 1]

	Returns:
		Thermal emission values
	"""
	return temperature * emissivity
