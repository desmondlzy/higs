import torch

from heatgaussian.spherical_gaussian_radiance_cache import (
	SphericalGaussianRadianceCache,
	evaluate_spherical_gaussians,
	evaluate_caches_by_hemisphere,
)


def test_partial_evaluate_with_mask():
	num_channels = 3
	cache = SphericalGaussianRadianceCache(
		num_primitives=100, 
		num_lobes=4, 
		num_channels=num_channels)
	
	batch_size = (2, )

	mask = torch.randint(low=0, high=2, size=(cache.num_primitives, ), device=cache.lobes.device, dtype=torch.bool)
	num_of_true = torch.sum(mask).item()

	w = torch.nn.functional.normalize(torch.randn((*batch_size, cache.num_primitives, 3), device=cache.lobes.device))

	partial_result = cache.partial_evaluate(
		w=w,
		indices=mask,
	)
	assert partial_result.shape == (*batch_size, num_of_true, num_channels)

	full_eval_results = cache.forward(w)
	assert full_eval_results.shape == (*batch_size, cache.num_primitives, num_channels)

	assert torch.allclose(
		partial_result,
		full_eval_results[..., mask, :],
		rtol=1e-5, atol=1e-5,
	)


def test_partial_evaluate_with_indices():
	num_channels = 3
	cache = SphericalGaussianRadianceCache(
		num_primitives=100, 
		num_lobes=4, 
		num_channels=num_channels)
	
	batch_size = (2, )

	indices = torch.tensor([10, 12, 99], device=cache.lobes.device, dtype=torch.long)
	n_indices = indices.shape[0]

	w_full = torch.nn.functional.normalize(torch.randn((*batch_size, cache.num_primitives, 3), device=cache.lobes.device))
	w_partial = w_full[..., indices, :]

	partial_result = cache.partial_evaluate(
		w=w_partial,
		indices=indices,
	)
	assert partial_result.shape == (*batch_size, n_indices, num_channels)

	full_eval_results = cache.forward(w_full)
	assert full_eval_results.shape == (*batch_size, cache.num_primitives, num_channels)

	assert torch.allclose(
		partial_result,
		full_eval_results[..., indices, :],
		rtol=1e-5, atol=1e-5,
	)


def test_partial_evaluate_with_mask():
	num_channels = 3
	cache = SphericalGaussianRadianceCache(
		num_primitives=100, 
		num_lobes=4, 
		num_channels=num_channels)
	
	batch_size = (2, )

	mask = torch.randint(low=0, high=2, size=(cache.num_primitives, ), device=cache.lobes.device, dtype=torch.bool)
	num_of_true = torch.sum(mask).item()

	w = torch.nn.functional.normalize(torch.randn((*batch_size, cache.num_primitives, 3), device=cache.lobes.device))

	partial_result = cache.partial_evaluate(
		w=w,
		indices=mask,
	)
	assert partial_result.shape == (*batch_size, num_of_true, num_channels)

	full_eval_results = cache.forward(w)
	assert full_eval_results.shape == (*batch_size, cache.num_primitives, num_channels)

	assert torch.allclose(
		partial_result,
		full_eval_results[..., mask, :],
		rtol=1e-5, atol=1e-5,
	)



def test_evaluate_caches_by_hemisphere():
	num_prim = 100
	num_lobes = 4
	num_channels = 3

	normals_pos = torch.nn.functional.normalize(torch.randn((num_prim, 3), device='cuda'))

	cache_pos = SphericalGaussianRadianceCache(
		num_primitives=num_prim,
		num_lobes=num_lobes,
		num_channels=num_channels,
	).cuda()

	cache_neg = SphericalGaussianRadianceCache(
		num_primitives=num_prim,
		num_lobes=num_lobes,
		num_channels=num_channels,
	).cuda()

	cache_pos.orient_lobes_to_hemisphere(normals_pos)
	cache_neg.orient_lobes_to_hemisphere(-normals_pos)

	batch_size = (19, 2, )
	w = torch.nn.functional.normalize(torch.randn((*batch_size, num_prim, 3), device='cuda'))

	res = evaluate_caches_by_hemisphere(
		w=w,
		normals=normals_pos,
		cache_positive=cache_pos,
		cache_negative=cache_neg,
	)

	assert res.shape == (*batch_size, num_prim, num_channels)


def test_orient_lobes_to_hemisphere():
	n_channels = 3
	n_lobes = 4
	n_prim = 100
	cache = SphericalGaussianRadianceCache(
		num_primitives=n_prim, 
		num_lobes=n_lobes, 
		num_channels=n_channels)

	normals = torch.randn((cache.num_primitives, 3), device=cache.lobes.device)
	cache.orient_lobes_to_hemisphere(normals)

	assert cache.lobes.shape == (cache.num_primitives, cache.num_lobes, 3), f"lobes shape {cache.lobes.shape} != expected {(cache.num_primitives, cache.num_lobes, 3)}"

	dots = torch.sum(normals.view(n_prim, 1, 3) * cache.lobes, dim=-1)  # (num_primitives, num_lobes, 3)

	assert torch.all(dots > 0), f"not all lobes are oriented to the hemisphere: {dots}"
