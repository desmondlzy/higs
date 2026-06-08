import torch

from heatgaussian.spherical_gaussian_radiance_cache import (
	SphericalGaussianRadianceCache,
)
from heatgaussian.pad_new_axes import pad_new_axes

class DoubleSphericalGaussianRadianceCache(torch.nn.Module):
	def __init__(self, 
			num_primitives: int, 
			num_lobes: int, 
			num_channels: int, 
		):

		super().__init__()
		self.num_primitives = num_primitives
		self.num_lobes = num_lobes
		self.num_channels = num_channels

		self.primary_cache = SphericalGaussianRadianceCache(num_primitives, num_lobes, num_channels)
		self.secondary_cache = SphericalGaussianRadianceCache(num_primitives, num_lobes, num_channels)
	

	@staticmethod
	def from_parameters(
			primary_lobes, 
			primary_amplitudes, 
			primary_sharpness, 
			primary_diffuse_strengths,
			secondary_lobes, 
			secondary_amplitudes, 
			secondary_sharpness, 
			secondary_diffuse_strengths,
		):
		assert primary_lobes.shape == secondary_lobes.shape, f"primary_lobes shape {primary_lobes.shape} != secondary_lobes shape {secondary_lobes.shape}"
		assert primary_amplitudes.shape == secondary_amplitudes.shape, f"primary_amplitudes shape {primary_amplitudes.shape} != secondary_amplitudes shape {secondary_amplitudes.shape}"
		assert primary_sharpness.shape == secondary_sharpness.shape, f"primary_sharpness shape {primary_sharpness.shape} != secondary_sharpness shape {secondary_sharpness.shape}"
		assert primary_diffuse_strengths.shape == secondary_diffuse_strengths.shape, f"primary_diffuse_strengths shape {primary_diffuse_strengths.shape} != secondary_diffuse_strengths shape {secondary_diffuse_strengths.shape}"

		cache = DoubleSphericalGaussianRadianceCache(
			num_primitives=primary_lobes.shape[0], 
			num_lobes=primary_lobes.shape[1], 
			num_channels=primary_amplitudes.shape[-1], 
		)

		cache.primary_cache = SphericalGaussianRadianceCache.from_parameters(
			lobes=primary_lobes,
			amplitudes=primary_amplitudes,
			sharpness=primary_sharpness,
			diffuse_strengths=primary_diffuse_strengths,
		)

		cache.secondary_cache = SphericalGaussianRadianceCache.from_parameters(
			lobes=secondary_lobes,
			amplitudes=secondary_amplitudes,
			sharpness=secondary_sharpness,
			diffuse_strengths=secondary_diffuse_strengths,
		)

		return cache
	

	@torch.no_grad()
	def init_with_only_diffuse(self, diffuse_strengths):
		self.primary_cache.init_with_only_diffuse(diffuse_strengths)
		self.secondary_cache.init_with_only_diffuse(diffuse_strengths)
	
	
	@torch.no_grad()
	def orient_lobes_to_hemisphere(self, normals):
		self.primary_cache.orient_lobes_to_hemisphere(normals)
		self.secondary_cache.orient_lobes_to_hemisphere(-normals)

	@torch.no_grad()
	def normalize_lobes(self):
		self.primary_cache.normalize_lobes()
		self.secondary_cache.normalize_lobes()


	def forward(self, w, normals, diffuse_only=False, specular_only=False):
		assert w.shape[-2:] == (self.num_primitives, 3), f"w shape {w.shape} != expected {(self.num_primitives, 3)}"
		batch_shapes = w.shape[:-2]
		bdim = len(batch_shapes)

		# opposite_primary.shape = (..., num_primitives)
		opposite_primary = torch.sum(w * pad_new_axes(normals, bdim), dim=-1) < 0.0

		# opposite_primary.shape = (..., num_primitives)
		eval_secondary = opposite_primary
		eval_primary = torch.logical_not(eval_secondary)

		primary_results = self.primary_cache.forward(
			w, 
			diffuse_only, 
			specular_only)

		secondary_results = self.secondary_cache.forward(
			w, 
			diffuse_only, 
			specular_only)

		assert primary_results.shape[-2:] == (self.num_primitives, self.num_channels), f"primary_results shape {primary_results.shape}"
		assert secondary_results.shape[-2:] == (self.num_primitives, self.num_channels), f"secondary_results shape {secondary_results.shape}"

		results = torch.where(
			eval_primary.unsqueeze(-1),
			primary_results,
			secondary_results,
		)
		assert results.shape == (*batch_shapes, self.num_primitives, self.num_channels), f"results shape {results.shape}, {batch_shapes = }"

		return results


	def partial_evaluate(self, w, normals, indices, diffuse_only=False, specular_only=False):
		assert indices.ndim == 1, f"indices shape {indices.shape} != expected (num_indices, )"
		assert normals.dtype in (torch.float32, torch.float64), f"normals dtype {normals.dtype} not supported"

		batch_shapes = w.shape[:-2]
		bdim = len(batch_shapes)

		# primary_results.shape = (..., num_primitives, 3)
		primary_results = self.primary_cache.partial_evaluate(
			w, indices, diffuse_only, specular_only)
		secondary_results = self.secondary_cache.partial_evaluate(
			w, indices, diffuse_only, specular_only)

		if indices.dtype == torch.bool:
			opposite_primary = torch.sum(w[..., indices, :] * pad_new_axes(normals, bdim), dim=-1) < 0.0
		else:
			opposite_primary = torch.sum(w * pad_new_axes(normals[indices], bdim), dim=-1) < 0.0

		# opposite_primary.shape = (..., num_primitives)
		eval_secondary = opposite_primary
		eval_primary = torch.logical_not(eval_secondary)

		
		results = torch.where(
			eval_primary.unsqueeze(-1),
			primary_results,
			secondary_results,
		)

		assert results.shape == (*w.shape[:-1], self.num_channels), f"results shape {results.shape}"

		return results
