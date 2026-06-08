import torch
from gsplat import spherical_harmonics

from heatgaussian.invsoftplus import invsoftplus
from heatgaussian.pad_new_axes import pad_new_axes

class SphericalHarmonicRadianceCache(torch.nn.Module):
	def __init__(self, num_primitives, num_degrees, num_channels):
		super().__init__()
		self.num_primitives = num_primitives
		self.num_degrees = num_degrees
		self.num_channels = num_channels

		num_coeffs = (num_degrees + 1) * (num_degrees + 1)

		# (num_primitives, n_coeffs, n_channels)
		coeffs = torch.zeros((num_primitives, num_coeffs, num_channels), dtype=torch.float32)
		self.coeffs = torch.nn.Parameter(coeffs)


	def partial_evaluate(self, w, normals, indices, diffuse_only=False, specular_only=False):
		"""
		indices: a tensor of indices where the basis functions should be evaluated.
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


		values = evaluate_spherical_harmonics(
			coeffs=self.coeffs[indices, ...],
			dirs=w_indiced,
			degree_to_use=self.num_degrees,
		)


		return torch.nn.functional.softplus(values)


	def forward(self, w, normals, diffuse_only=False, specular_only=False):
		assert w.shape[-2:] == (self.num_primitives, 3), f"w shape {w.shape}"
		# assert torch.linalg.norm(w, dim=-1).allclose(torch.ones((self.num_primitives, ), device=w.device), atol=1e-5), f"not normalized input (max {torch.max(torch.linalg.norm(w, dim=-1))}, min {torch.min(torch.linalg.norm(w, dim=-1))})"

		values = evaluate_spherical_harmonics(
			coeffs=self.coeffs,
			dirs=w,
			degree_to_use=self.num_degrees,
		)

		assert values.shape == (w.shape[:-2] + (self.num_primitives, self.num_channels)), f"sg_values shape {values.shape} != expected {(*w.shape[:-1], self.num_primitives, self.num_channels)}"

		return torch.nn.functional.softplus(values)
	


def evaluate_spherical_harmonics(
	coeffs,
	dirs,
	degree_to_use,
):
	"""
	evaluate the same set of spherical harmonics from potentially multiple dirs

	coeffs: (num_primitives, num_coeffs, num_channels)
	dirs: (batch..., num_primitives, 3)
	return: (batch..., num_primitives, num_channels)
	"""
	n_prim, n_coeffs, n_channels = coeffs.shape
	batch_dims = dirs.shape[:-2]

	if len(batch_dims) > 0:
		# expand the coeffs to match the batch dimensions
		coeffs = coeffs.unsqueeze(0).expand((*batch_dims, n_prim, n_coeffs, n_channels))

	values = spherical_harmonics(
		degrees_to_use=degree_to_use,
		dirs=dirs,  # flatten batch dims
		coeffs=coeffs,  # (num_primitives, num_coeffs, num_channels)
	)

	assert values.shape == (*batch_dims, n_prim, n_channels), f"values shape {values.shape}. coeffs shape {coeffs.shape}, dirs shape {dirs.shape}, batch_dims {batch_dims}"

	return values

