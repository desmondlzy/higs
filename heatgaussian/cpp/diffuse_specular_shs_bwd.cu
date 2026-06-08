#include <gsplat/cuda/include/bindings.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <glm/gtc/type_ptr.hpp>

#include "diffuse_specular_shs_bwd.cuh"
#include "spherical_harmonics.cuh"

namespace heatgaussian {

namespace cg = cooperative_groups;

template <typename S>
using vec3 = glm::vec<3, S>;


template <typename S>
__global__ void diffuse_specular_shs_bwd_kernel(
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
	
	const S *__restrict__ v_diffuse_shs,  // [n_gauss, n_channels]
	const S *__restrict__ v_specular_shs,  // [n_gauss, n_bases, n_channels]

	S *__restrict__ v_out_shs // [n_gauss, n_bases, n_channels]

) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    // uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    // uint32_t k = blockIdx.z * blockDim.z + threadIdx.z;

	if (i >= n_gauss) { return; }
	// if (j >= n_bases) { return; }

	// T_diffuse_row[i]
	// nnz_row = crow_indices[i + 1] - crow_indices[i]
	// col_start = crow_indices[i]
	// col_end = crow_indices[i + 1]
	vec3<S> current_gauss = means[i];

    {
        int base_offset = i;
        for (int ind = T_diffuse_crow_indices[base_offset]; ind < T_diffuse_crow_indices[base_offset + 1]; ind++) { 
            int another_gauss = T_diffuse_col_indices[ind];                                               
            S weight = T_diffuse_values[ind];                                                             
                                                                                                
            vec3<S> dir = means[another_gauss] - current_gauss;                                 

            S dist = sqrt(max(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z, 0.00001f));                         
            float3 dir_hemicube { (-dir.x) / dist, dir.y / dist, dir.z / dist };                 

            float v_radiance[100];  // TODO: make this template
            for (int c = 0; c < n_channels; c++) {
                v_radiance[c] = weight * v_diffuse_shs[i * n_channels + c];
            }
                                                                                            
            float v_shs[100];
            int degree = spherical_harmonics::num_sh_degree(n_bases);
            spherical_harmonics::eval_sh_grad(                                                  
                degree, n_channels, dir_hemicube,                                        
                v_radiance,
                v_shs 
            );                                                                           

            for (int s = 0; s < n_bases; s++) {                                      
                for (int c = 0; c < n_channels; c++) {
                    int out_index = another_gauss * n_bases * n_channels + s * n_channels + c;
                    S val = v_shs[s * n_channels + c];
                    // v_out_shs[out_index] += val;
                    atomicAdd(
                        v_out_shs + out_index, 
                        val);
                }
            }
        } 
    }
    {
        for (int so = 0; so < n_bases; so++) {
            int base_offset = i * n_bases + so;
            for (int ind = T_specular_crow_indices[base_offset]; ind < T_specular_crow_indices[base_offset + 1]; ind++) { 
                int another_gauss = T_specular_col_indices[ind];                                               
                S weight = T_specular_values[ind];                                                             
                                                                                                    
                vec3<S> dir = means[another_gauss] - current_gauss;                                 

                S dist = sqrt(max(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z, 0.00001f));                         
                float3 dir_hemicube { (-dir.x) / dist, dir.y / dist, dir.z / dist };                 

                float v_radiance[100];  // TODO: make this template
                for (int s = 0; s < n_bases; s++) {
                    for (int c = 0; c < n_channels; c++) {
                        v_radiance[s * n_channels + c] = weight * v_specular_shs[i * n_bases * n_channels + s * n_channels + c];
                    }
                }

                float v_shs[100];
                int degree = spherical_harmonics::num_sh_degree(n_bases);
                spherical_harmonics::eval_sh_grad(                                                  
                    degree, n_channels, dir_hemicube,                                        
                    v_radiance,
                    v_shs 
                );                                                                           

                for (int s = 0; s < n_bases; s++) {                                      
                    for (int c = 0; c < n_channels; c++) {
                        int out_index = another_gauss * n_bases * n_channels + s * n_channels + c;
                        S val = v_shs[s * n_channels + c];
                        // v_out_shs[out_index] += val;
                        atomicAdd(
                            v_out_shs + out_index, 
                            val);
                    }
                }
            } 
        }
    }

}


torch::Tensor
diffuse_specular_shs_bwd_tensor(
    const torch::Tensor &means, 
	const torch::Tensor &out_shs,
	const torch::Tensor &T_diffuse,
	const torch::Tensor &T_specular,
	const torch::Tensor &v_diffuse_shs,
	const torch::Tensor &v_specular_shs
) {
    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(means);
    GSPLAT_CHECK_INPUT(out_shs);
    GSPLAT_CHECK_INPUT(v_diffuse_shs);
    GSPLAT_CHECK_INPUT(v_specular_shs);

	uint32_t n_gauss = means.size(0);
	uint32_t n_bases = out_shs.size(1);
	uint32_t n_channels = out_shs.size(2);

	auto v_out_shs = torch::zeros({n_gauss, n_bases, n_channels}, out_shs.options());

    // Extract sparse matrix structures
    auto T_diffuse_crow_indices = T_diffuse.crow_indices().contiguous();
    auto T_diffuse_col_indices = T_diffuse.col_indices().contiguous();
    auto T_diffuse_values = T_diffuse.values().contiguous();

    auto T_specular_crow_indices = T_specular.crow_indices().contiguous();
    auto T_specular_col_indices = T_specular.col_indices().contiguous();
    auto T_specular_values = T_specular.values().contiguous();

    // Define grid and block sizes
    dim3 threadsPerBlock(16, 1, 1);  // Example: adjust based on workload
    dim3 numBlocks(
        (n_gauss + threadsPerBlock.x - 1) / threadsPerBlock.x,
		1,
        1
    );

    // Launch the kernel
    // AT_DISPATCH_FLOATING_TYPES_AND_HALF(means.scalar_type(), "diffuse_specular_shs_cuda", ([&] {
        diffuse_specular_shs_bwd_kernel<float><<<numBlocks, threadsPerBlock>>>(
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

            v_diffuse_shs.data_ptr<float>(),
            v_specular_shs.data_ptr<float>(),

			v_out_shs.data_ptr<float>()
        );

        // Synchronize to ensure the kernel has finished executing
        // cudaDeviceSynchronize();
    // }));


	return v_out_shs;
}

} // namespace heatgaussian
