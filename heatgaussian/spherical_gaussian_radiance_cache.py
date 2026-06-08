import torch

from heatgaussian.invsoftplus import invsoftplus

class SphericalGaussianRadianceCache(torch.nn.Module):
	def __init__(self, num_primitives, num_lobes, num_channels):
		super().__init__()
		self.num_primitives = num_primitives
		self.num_lobes = num_lobes
		self.num_channels = num_channels

		# (num_primitives, num_channels)
		self.diffuse_strengths = torch.nn.Parameter(torch.ones((num_primitives, num_channels)))

		# (num_primitives, n_lobes, 3)
		lobe_inits = torch.nn.functional.normalize(torch.randn((num_primitives, num_lobes, 3)))
		self.lobes = torch.nn.Parameter(
			lobe_inits
		)

		# (num_primitives, n_lobes, num_channels)
		amplitude_inits = torch.ones((1, num_lobes, num_channels))
		self.amplitudes = torch.nn.Parameter(
			torch.tile(amplitude_inits, (num_primitives, 1, 1))
		)

		# (num_primitives, n_lobes)
		sharpness_inits = torch.linspace(0.1, 2.0, num_lobes)
		self.sharpness = torch.nn.Parameter(
			torch.tile(sharpness_inits, (num_primitives, 1))
		)
	

	@staticmethod
	def from_parameters(lobes, amplitudes, sharpness, diffuse_strengths):
		"""
		lobes: (num_primitives, num_lobes, 3)
		amplitudes: (num_primitives, num_lobes, num_channels)
		sharpness: (num_primitives, num_lobes)
		diffuse_strengths: (num_primitives, num_channels)
		"""

		num_primitives = lobes.shape[0]
		num_lobes = lobes.shape[1]
		num_channels = amplitudes.shape[-1]

		assert amplitudes.shape == (num_primitives, num_lobes, num_channels), f"amplitudes shape {amplitudes.shape} != expected {(num_primitives, num_lobes, num_channels)}"
		assert sharpness.shape == (num_primitives, num_lobes), f"sharpness shape {sharpness.shape} != expected {(num_primitives, num_lobes)}"
		assert diffuse_strengths.shape == (num_primitives, num_channels), f"diffuse_strengths shape {diffuse_strengths.shape} != expected {(num_primitives, num_channels)}"
		assert lobes.shape == (num_primitives, num_lobes, 3), f"lobes shape {lobes.shape} != expected {(num_primitives, num_lobes, 3)}"

		assert torch.isfinite(lobes).all(), f"lobes {lobes} not finite"
		assert torch.isfinite(amplitudes).all(), f"amplitudes {amplitudes} not finite"
		assert torch.isfinite(sharpness).all(), f"sharpness {sharpness} not finite"
		assert torch.isfinite(diffuse_strengths).all(), f"diffuse_strengths {diffuse_strengths} not finite"


		cache = SphericalGaussianRadianceCache(
			num_primitives=num_primitives,
			num_lobes=num_lobes,
			num_channels=num_channels,
		)
		cache.lobes.data = lobes
		cache.amplitudes.data = amplitudes
		cache.sharpness.data = sharpness
		cache.diffuse_strengths.data = diffuse_strengths

		return cache
	

	def partial_evaluate(self, w, indices, diffuse_only=False, specular_only=False):
		"""
		indices: a tensor of indices where the lobes are evaluated 
				or a boolean mask of the same size as the number of primitives
		"""
		assert indices.ndim == 1, f"indices shape {indices.shape} != expected (num_indices, )"
		if indices.dtype == torch.bool:
			assert indices.shape == (self.num_primitives, ), f"indices shape {indices.shape} != expected {(self.num_primitives, )}"
			assert w.shape[-2:] == (self.num_primitives, 3), f"w shape {w.shape} != expected {(self.num_primitives, 3)}"

			w_indiced = w[..., indices, :]  # (batch..., num_indices, 3)

		else:
			assert w.shape[-2:] == (indices.shape[0], 3)
			w_indiced = w


		sg_values = evaluate_spherical_gaussians(
			self.lobes[indices, ...],
			torch.nn.functional.softplus(self.amplitudes[indices, ...]),
			self.sharpness[indices, ...],
			w_indiced,
		)

		if specular_only:
			return sg_values
		
		diffuse_values = torch.nn.functional.softplus(self.diffuse_strengths[indices, ...].contiguous())  # (num_indices, num_channels)

		if diffuse_only:
			assert not specular_only, "diffuse_only and specular_only cannot be both True"
			return diffuse_values
		
		return diffuse_values + sg_values


	def forward(self, w, diffuse_only=False, specular_only=False):
		assert w.shape[-2:] == (self.num_primitives, 3), f"w shape {w.shape}"
		# assert torch.linalg.norm(w, dim=-1).allclose(torch.ones((self.num_primitives, ), device=w.device), atol=1e-5), f"not normalized input (max {torch.max(torch.linalg.norm(w, dim=-1))}, min {torch.min(torch.linalg.norm(w, dim=-1))})"
		assert torch.isfinite(self.lobes).all(), f"lobes {self.lobes} not finite, num of nonfinite: {torch.sum(~torch.isfinite(self.lobes))}"
		assert torch.isfinite(self.amplitudes).all(), f"amplitudes {self.amplitudes} not finite"

		sg_values = evaluate_spherical_gaussians(
			self.lobes,
			torch.nn.functional.softplus(self.amplitudes),
			self.sharpness,
			w,
		)

		assert sg_values.shape == (w.shape[:-2] + (self.num_primitives, self.num_channels)), f"sg_values shape {sg_values.shape} != expected {(*w.shape[:-1], self.num_primitives, self.num_channels)}"

		if specular_only:
			return sg_values

		diffuse_values = torch.nn.functional.softplus(self.diffuse_strengths)  # (num_primitives, num_channels)

		if diffuse_only:
			assert not specular_only, "diffuse_only and specular_only cannot be both True"
			return diffuse_values

		return diffuse_values + sg_values
	

	@torch.no_grad()
	def init_with_only_diffuse(self, diffuse_strengths):
		if isinstance(diffuse_strengths, (float, int)):
			diffuse_strengths = torch.full_like(self.diffuse_strengths.data, diffuse_strengths)
		
		assert diffuse_strengths.shape == self.diffuse_strengths.shape, f"diffuse_strengths shape {diffuse_strengths.shape} != expected {self.diffuse_strengths.shape}"

		self.amplitudes.data = invsoftplus(torch.full_like(self.amplitudes.data, 1e-4))
		assert torch.isfinite(self.amplitudes).all(), f"amplitudes not finite: {self.amplitudes}"

		self.diffuse_strengths.data = invsoftplus(diffuse_strengths)
	
	
	@torch.no_grad()
	def orient_lobes_to_hemisphere(self, normals):
		assert normals.shape == (self.num_primitives, 3), f"normals shape {normals.shape} != expected {(self.num_primitives, 3)}"
		normals = normals / torch.linalg.norm(normals, dim=-1, keepdim=True)

		normals = normals.view(self.num_primitives, 1, 3)

		dots = torch.sum(normals * self.lobes, dim=-1, keepdim=True)  # (num_primitives, num_lobes, 3)
		signs = torch.where(dots > 0, 1.0, -1.0)  # (num_primitives, num_lobes, 3)
		print(f"num of negative lobes: {torch.sum(signs < 0)}")
		self.lobes.data *= signs  # (num_primitives, num_lobes, 3)


	@torch.no_grad()
	def normalize_lobes(self):
		lobe_norms = torch.linalg.norm(self.lobes.data, dim=-1, keepdim=True)
		self.lobes.data /= lobe_norms
		assert torch.isfinite(self.lobes).all(), f"lobes not finite: {self.lobes}"



def pad_new_axes(tensor, num):
	if num > 0:
		return tensor.view(*[1 for _ in range(num)], *tensor.shape)
	else:
		return tensor

def evaluate_spherical_gaussians(
	lobes,
	amplitudes,
	sharpness,
	dirs,
):
	"""
	lobes: (num_primitives, num_lobes, 3)
	amplitudes: (num_primitives, num_lobes, num_channels)
	sharpness: (num_primitives, num_lobes)
	dirs: (batch..., num_primitives, 3)
	return: (batch..., num_primitives, num_channels)
	"""
	n_prim = lobes.shape[0]
	n_lobes = lobes.shape[1]
	n_channels = amplitudes.shape[-1]

	assert dirs.shape[-2:] == (n_prim, 3), f"w shape {dirs.shape}"
	assert lobes.shape == (n_prim, n_lobes, 3), f"lobes shape {lobes.shape} != expected {(n_prim, n_lobes, 3)}"
	assert sharpness.shape == (n_prim, n_lobes), f"sharpness shape {sharpness.shape} != expected {(n_prim, n_lobes)}"
	assert amplitudes.shape == (n_prim, n_lobes, n_channels), f"amplitudes shape {amplitudes.shape} != expected {(n_prim, n_lobes, n_channels)}"

	# assert torch.linalg.norm(w, dim=-1).allclose(torch.ones((self.num_primitives, ), device=w.device), atol=1e-5), f"not normalized input (max {torch.max(torch.linalg.norm(w, dim=-1))}, min {torch.min(torch.linalg.norm(w, dim=-1))})"
	assert torch.isfinite(lobes).all(), f"lobes {lobes} not finite"
	assert torch.isfinite(amplitudes).all(), f"amplitudes {amplitudes} not finite"

	w_batch_dims = dirs.shape[:-2]

	lobes_normalized = torch.nn.functional.normalize(lobes, dim=-1)  # (..., num_primitives, num_lobes, 3)

	cosines = torch.sum(
		pad_new_axes(lobes_normalized, len(w_batch_dims)) * 
		# lobes_normalized.unsqueeze(0) * 
		dirs.view(*w_batch_dims, n_prim, 1, 3), dim=-1, keepdim=True)  # (..., num_primitives, num_lobes, 1 channels)

	exponents = torch.exp(pad_new_axes(sharpness.view(n_prim, n_lobes, 1), len(w_batch_dims)) * (cosines - 1.0))  # (..., num_primitives, num_lobes, num_channels)

	values = pad_new_axes(amplitudes.view(n_prim, n_lobes, n_channels), len(w_batch_dims)) * exponents  # (..., num_primitives, num_lobes, num_channels)
	sg_values = torch.sum(values, dim=-2)  # (..., num_primitives, num_channels)

	return sg_values



def evaluate_caches_by_hemisphere(w, normals, cache_positive, cache_negative, **kwargs):
	"""
	evaluate either cache_positive or cache_negative based on the dot product between w and normals
	"""
	padded_normals = normals.view(*[1 for _ in w.shape[:-2]], normals.shape[-2], 3)
	dots = torch.sum(w * padded_normals, axis=-1, keepdim=True)
	mask = dots > 0

	res_pos = cache_positive.forward(w, **kwargs)
	res_neg= cache_negative.forward(w, **kwargs)

	res_full_combined = torch.where(mask, res_pos, res_neg)

	return res_full_combined	
