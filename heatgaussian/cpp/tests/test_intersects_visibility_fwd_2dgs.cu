#include <iostream>
#include <torch/extension.h>

#include <gsplat/cuda/include/bindings.h>

#include "intersects_visibility_fwd_2dgs.cuh"

bool test_intersects_visibility_fwd_2dgs() {
	// create tensor on cuda
	auto means3d = torch::tensor({
		{0.0, 0.0, 0.0},
		{0.0, 0.0, 1.0},
		{0.0, 1.0, 0.0},
		{1.0, 0.0, 0.0},
	}).cuda();

	int N = means3d.size(0);

	auto quats = torch::tile(torch::tensor({{1.0, 0.0, 0.0, 0.0}}), {N, 1}).cuda();
	auto scales = torch::ones({N, 3}).cuda() * 0.08;
	auto opacities = torch::ones({N}).cuda();
	auto normals = -means3d / torch::norm(means3d, torch::nullopt, {1}, true);

	auto masks = torch::nullopt;
	
	uint32_t image_width = 128;
	uint32_t image_height = 128;
	uint32_t tile_size = 1;
	uint32_t tile_width = image_width / (float) tile_size;
	uint32_t tile_height = image_height / (float) tile_size;

	auto viewmat = torch::tensor({{
		{1.0, 0.0, 0.0, 0.0},
		{0.0, -1.0, 0.0, 0.0},
		{0.0, 0.0, -1.0, 5.0},
		{0.0, 0.0, 0.0, 1.0},
	}}).cuda();

	auto Ks = torch::tensor({{
		{110.8512, 0.0, 64.0},
		{0.0, 110.8512, 64.0},
		{0.0, 0.0, 1.0}
	}}).cuda();

	int max_hits = 4;
	int C = 1;

	float radius_clip = 0.0;
	auto [_indptr, camera_ids, gaussian_ids, radii, means2d, depths, ray_transforms, normals_camera] = gsplat::fully_fused_projection_packed_fwd_2dgs_tensor(
		means3d, 
		quats, 
		scales, 
		viewmat, 
		Ks, 
		image_width, 
		image_height, 
		0.0001,
		1e8,
		radius_clip
	);

	auto [tiles_per_gauss, isect_ids, flatten_ids] = gsplat::isect_tiles_tensor(
		means2d, 
		radii,
		depths,
		camera_ids,
		gaussian_ids,
		tile_size,
		C,
		tile_width,
		tile_height,
		true,
		false
	);

	auto isect_offsets = gsplat::isect_offset_encode_tensor(
		isect_ids, 
		C, 
		tile_width, 
		tile_height
	);

	auto [alphas, last_ids, visibility_indices, visibility_values] = heatgaussian::intersects_visibility_fwd_2dgs_tensor(
		means2d, 
		ray_transforms, 
		opacities, 
		normals, 
		masks,	
		image_width,
		image_height,
		tile_size,
		isect_offsets,
		flatten_ids,
		max_hits
	);

	auto visibility_indices_shape = visibility_indices.sizes();
	auto visibility_values_shape = visibility_values.sizes();

	std::cout << "visibility_indices_shape: " << visibility_indices_shape << std::endl;
	std::cout << "visibility_values_shape: " << visibility_values_shape << std::endl;
	std::cout << "visibility_values max & average && min values: " << visibility_values.max().item<float>() << " " << visibility_values.mean().item<float>() << " " << visibility_values.min().item<float>() << std::endl;

	return true;

}

int main(int argc, char **argv) {
	if (!test_intersects_visibility_fwd_2dgs()) {
		throw std::runtime_error("Test failed!");
	}

	return 0;
}
