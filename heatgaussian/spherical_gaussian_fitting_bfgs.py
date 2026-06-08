import torch
from torch.nn.functional import softplus


def spherical_gaussians_reconstruction_loss(lobes, sharpness, amplitudes, obser_dirs, obser_vals):
	"""
	given the SG parameters, evaluate at each observation direction and compute the L2 loss with the observed values.
	"""
	# sg_params: (lobes (..., n, 3), sharpness (..., n, 1), amplitudes (..., n, c))
	# observations: dirs (..., o, 3), vals (..., o, c)
	batch_dims = lobes.shape[:-2]
	n_lobes = lobes.shape[-2]
	n_channels = amplitudes.shape[-1]
	assert lobes.shape == (*batch_dims, n_lobes, 3), f"lobes shape {lobes.shape} != expected {(*batch_dims, n_lobes, 3)}"
	assert sharpness.shape == (*batch_dims, n_lobes, 1), f"sharpness shape {sharpness.shape} != expected {(*batch_dims, n_lobes, 1)}"
	assert amplitudes.shape == (*batch_dims, n_lobes, n_channels), f"amplitudes shape {amplitudes.shape} != expected {(*batch_dims, n_lobes, 1)}"

	assert torch.isfinite(lobes).all(), f"lobes is not finite: {lobes}"
	assert torch.isfinite(sharpness).all(), f"sharpness is not finite: {sharpness}"
	assert torch.isfinite(amplitudes).all(), f"amplitudes is not finite: {amplitudes}"
	assert torch.isfinite(obser_dirs).all(), f"obser_dirs is not finite: {obser_dirs}"
	assert torch.isfinite(obser_vals).all(), f"obser_vals is not finite: {obser_vals}"

	lobes_normalized = torch.nn.functional.normalize(lobes, dim=-1)  # (..., n_lobes, 3)

	assert torch.isfinite(lobes_normalized).all(), f"lobes_normalized is not finite: {lobes_normalized}"

	n_obsers = obser_dirs.shape[-2]
	assert obser_dirs.shape == (*batch_dims, n_obsers, 3), f"obser_dirs shape {obser_dirs.shape} != expected {(*batch_dims, n_obsers, 3)}"
	assert obser_vals.shape == (*batch_dims, n_obsers, n_channels), f"obser_vals shape {obser_vals.shape} != expected {(*batch_dims, n_obsers, n_channels)}"

	dirs = torch.nn.functional.normalize(obser_dirs, dim=-1)  # (..., n_obsers, 3)
	vals = obser_vals  # (n_obsers, n_channels)

	# (..., 1, n_lobes, 3) * (..., n_obsers, 1, 3) -> (..., n_obsers, n_lobes, 1)
	cosines = torch.sum(lobes_normalized.unsqueeze(-3) * dirs.unsqueeze(-2), dim=-1, keepdim=True)  

	# (..., 1, n_lobes, 1) * (..., n_obsers, n_lobes, 1) -> (..., n_obsers, n_lobes, 1)
	exponents = torch.exp(sharpness.unsqueeze(-3) * (cosines - 1.0))  

	# (..., 1, n_lobes, n_channels) * (..., n_obsers, n_lobes, 1) -> (..., n_obsers, n_chan)
	sg_values = torch.sum(amplitudes.unsqueeze(-3) * exponents, dim=-2)  # (n_observations, n_channels)

	loss = torch.pow(sg_values - vals, 2).sum()

	assert torch.isfinite(loss).all(), f"Loss is not finite: {loss}"

	return loss

def spherical_gaussian_fitting_bfgs(
	lobes_init, 
	sharpness_init,
	amplitudes_init,
	obser_dirs,
	obser_vals,
	inplace=False,
):
	"""
	main function to call
	"""
	if not inplace:
		lobes = lobes_init.detach().clone().requires_grad_(lobes_init.requires_grad)
		sharpness = sharpness_init.detach().clone().requires_grad_(sharpness_init.requires_grad)
		amplitudes = amplitudes_init.detach().clone().requires_grad_(amplitudes_init.requires_grad)
	else:
		lobes = lobes_init
		sharpness = sharpness_init
		amplitudes = amplitudes_init
	
	assert torch.isfinite(lobes).all(), f"lobes is not finite: {lobes}"
	assert torch.isfinite(sharpness).all(), f"sharpness is not finite: {sharpness}"
	assert torch.isfinite(amplitudes).all(), f"amplitudes is not finite: {amplitudes}"
	assert torch.isfinite(obser_dirs).all(), f"obser_dirs is not finite: {obser_dirs}"
	assert torch.isfinite(obser_vals).all(), f"obser_vals is not finite: {obser_vals}"

	lbfgs = torch.optim.LBFGS([lobes, sharpness, amplitudes],
						lr=0.5,
						history_size=5, 
						max_iter=4, 
						tolerance_grad=1e-3,
						tolerance_change=1e-3,
						line_search_fn="strong_wolfe")

	# L-BFGS
	def closure():
		lbfgs.zero_grad()
		objective = spherical_gaussians_reconstruction_loss(
			lobes, 
			softplus(sharpness), 
			softplus(amplitudes), 
			obser_dirs, 
			obser_vals)
		objective.backward()
		return objective

	history_lbfgs = []
	for i in range(20):
		history_lbfgs.append(spherical_gaussians_reconstruction_loss(
			lobes, 
			softplus(sharpness), 
			softplus(amplitudes), 
			obser_dirs, 
			obser_vals).detach().cpu().item())
		lbfgs.step(closure)

	return (
		lobes, 
		softplus(sharpness), 
		softplus(amplitudes), 
		history_lbfgs
	)
