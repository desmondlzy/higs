#include <gsplat/cuda/include/bindings.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <glm/gtc/type_ptr.hpp>

#include "diffuse_specular_shs_fwd.cuh"
#include "spherical_harmonics.cuh"

namespace heatgaussian {

namespace cg = cooperative_groups;

template <typename S>
using vec3 = glm::vec<3, S>;

#define PROCESS_SHS(current_guass, crow_indices, col_indices, values, shs_output, base_offset, output_offset) \
{																						 		\
    for (int ind = crow_indices[base_offset]; ind < crow_indices[base_offset + 1]; ind++) {     \
        int another_gauss = col_indices[ind];                                                  \
        S weight = values[ind];                                                               \
                                                                                              \
        vec3<S> dir = means[another_gauss] - current_gauss;                                    \
        S dist = sqrt(max(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z, 0.00001f));                         \
                                                                                              \
        float3 dir_hemicube { (-dir.x) / dist, dir.y / dist, dir.z / dist };                 \
                                                                                              \
        int degree = spherical_harmonics::num_sh_degree(n_bases);                                    \
        float radiances[100];                                                         \
        spherical_harmonics::eval_sh(                                                               \
            degree, n_channels, dir_hemicube,                                                \
            out_shs + another_gauss * n_bases * n_channels,                                   \
            radiances                                                                        \
        );                                                                                   \
                                                                                              \
		shs_output[output_offset + c] += weight * radiances[c];                          \
    } 																					  	\
}																						 		

template <typename S>
__global__ void diffuse_specular_shs_fwd_kernel(
	const uint32_t n_gauss,
	const uint32_t n_bases,
	const uint32_t n_channels,
	const vec3<S> *__restrict__ means,
	const S *__restrict__ out_shs,

	const int *__restrict__ T_diffuse_crow_indices,
	const int *__restrict__ T_diffuse_col_indices,
	const S *__restrict__ T_diffuse_values,

	const int *__restrict__ T_specular_crow_indices,
	const int *__restrict__ T_specular_col_indices,
	const S *__restrict__ T_specular_values,
	
	S *__restrict__ diffuse_shs,  // [n_gauss, n_channels]
	S *__restrict__ specular_shs  // [n_gauss, n_bases, n_channels]
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    // uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t c = blockIdx.z * blockDim.z + threadIdx.z;

	if (i >= n_gauss) { return; }
	if (c >= n_channels) { return; }
	// if (j >= n_bases) { return; }

	// T_diffuse_row[i]
	// nnz_row = crow_indices[i + 1] - crow_indices[i]
	// col_start = crow_indices[i]
	// col_end = crow_indices[i + 1]
	vec3<S> current_gauss = means[i];

	// Process diffuse SHs
    PROCESS_SHS(current_gauss, 
		T_diffuse_crow_indices, T_diffuse_col_indices, T_diffuse_values, 
		diffuse_shs, 
		i, i * n_channels)

    // Process specular SHs
	for (int j = 0; j < n_bases; j++) {
		int specular_row = i * n_bases + j;
		PROCESS_SHS(current_gauss, 
			T_specular_crow_indices, T_specular_col_indices, T_specular_values, 
			specular_shs, specular_row, specular_row * n_channels)
	}
	
}


std::tuple<
	torch::Tensor,  // diffuse_shs
    torch::Tensor>   // specular_shs
diffuse_specular_shs_cuda_fwd_tensor(
    const torch::Tensor &means, 
	const torch::Tensor &out_shs,
	const torch::Tensor &T_diffuse,
	const torch::Tensor &T_specular
) {
    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(means);
    GSPLAT_CHECK_INPUT(out_shs);

	uint32_t n_gauss = means.size(0);
	uint32_t n_bases = out_shs.size(1);
	uint32_t n_channels = out_shs.size(2);

	torch::Tensor diffuse_shs = torch::zeros({n_gauss, n_channels}, means.options());
	torch::Tensor specular_shs = torch::zeros({n_gauss, n_bases, n_channels}, means.options());

    // Extract sparse matrix structures
    auto T_diffuse_crow_indices = T_diffuse.crow_indices().contiguous();
    auto T_diffuse_col_indices = T_diffuse.col_indices().contiguous();
    auto T_diffuse_values = T_diffuse.values().contiguous();

    auto T_specular_crow_indices = T_specular.crow_indices().contiguous();
    auto T_specular_col_indices = T_specular.col_indices().contiguous();
    auto T_specular_values = T_specular.values().contiguous();

    // Define grid and block sizes
    dim3 threadsPerBlock(
		16, 
		1, 
		min(4, n_channels)
	);  // Example: adjust based on workload
    dim3 numBlocks(
        (n_gauss + threadsPerBlock.x - 1) / threadsPerBlock.x,
		1,
        (n_channels + threadsPerBlock.z - 1) / threadsPerBlock.z
    );

    // Launch the kernel
    // AT_DISPATCH_FLOATING_TYPES_AND_HALF(means.scalar_type(), "diffuse_specular_shs_cuda", ([&] {
        diffuse_specular_shs_fwd_kernel<float><<<numBlocks, threadsPerBlock>>>(
            n_gauss,
            n_bases,
            n_channels,
            reinterpret_cast<vec3<float> *>(means.data_ptr<float>()),
            out_shs.data_ptr<float>(),

            T_diffuse_crow_indices.data_ptr<int>(),
            T_diffuse_col_indices.data_ptr<int>(),
            T_diffuse_values.data_ptr<float>(),

            T_specular_crow_indices.data_ptr<int>(),
            T_specular_col_indices.data_ptr<int>(),
            T_specular_values.data_ptr<float>(),

            diffuse_shs.data_ptr<float>(),
            specular_shs.data_ptr<float>()
        );

        // Synchronize to ensure the kernel has finished executing
        // cudaDeviceSynchronize();
    // }));


	return std::make_tuple(diffuse_shs, specular_shs);
}


std::tuple<
	torch::Tensor,  // diffuse_shs
    torch::Tensor>   // specular_shs
diffuse_specular_shs_cpp_tensor(
    const torch::Tensor &means, 
	const torch::Tensor &out_shs,
	const torch::Tensor &T_diffuse,
	const torch::Tensor &T_specular
) {
	uint32_t n_gauss = means.size(0);
	uint32_t n_bases = out_shs.size(1);
	uint32_t n_channels = out_shs.size(2);

	// n_channels need to be smaller than 10;
	if (n_channels > 10) {
		throw std::invalid_argument("n_channels should be smaller than 10");
	}

	torch::Tensor diffuse_shs = torch::zeros({n_gauss, n_channels}, means.options());
	torch::Tensor specular_shs = torch::zeros({n_gauss, n_bases, n_channels}, means.options());

	for (int g = 0; g < n_gauss; g++) {
		auto T_diffuse_row = T_diffuse[g];
		auto dir_g_to_means = means - means[g];
		dir_g_to_means[g] = 1.0;
		auto norms = torch::norm(dir_g_to_means, 2, {1}, true);
		auto dir_g_to_means_unit = dir_g_to_means / norms;

		// negate the x axis
		dir_g_to_means_unit.select(1, 0) *= -1.0;

		auto out_radiances_to_g = gsplat::compute_sh_fwd_tensor(
			4, dir_g_to_means_unit, out_shs, at::nullopt
		);

		diffuse_shs[g] = torch::matmul(T_diffuse_row, out_radiances_to_g);

		for (int s = 0; s < n_bases; s++) {
			auto T_specular_row = T_specular[g * 25 + s];
			specular_shs[g][s] = torch::matmul(T_specular_row, out_radiances_to_g);
		}
	}

	return std::make_tuple(diffuse_shs, specular_shs);
}


	
} // namespace heatgaussian
