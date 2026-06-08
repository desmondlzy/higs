#ifndef DIFFUSE_SPECULAR_SHS_FWD_CUH
#define DIFFUSE_SPECULAR_SHS_FWD_CUH

#include <torch/extension.h>

namespace heatgaussian {

std::tuple<torch::Tensor, torch::Tensor> 
diffuse_specular_shs_cpp_tensor(
    // Gaussian parameters
    const torch::Tensor &means3d, 
	const torch::Tensor &out_shs,
	const torch::Tensor &T_diffuse,
	const torch::Tensor &T_specular
);

std::tuple<torch::Tensor, torch::Tensor> 
diffuse_specular_shs_cuda_fwd_tensor(
    // Gaussian parameters
    const torch::Tensor &means3d, 
	const torch::Tensor &out_shs,
	const torch::Tensor &T_diffuse,
	const torch::Tensor &T_specular
);

}

#endif // DIFFUSE_SPECULAR_SHS_FWD_CUH