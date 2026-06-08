#ifndef DIFFUSE_SPECULAR_SHS_BWD_CUH
#define DIFFUSE_SPECULAR_SHS_BWD_CUH

#include <torch/extension.h>

namespace heatgaussian {

torch::Tensor
diffuse_specular_shs_bwd_tensor(
    // Gaussian parameters
    const torch::Tensor &means3d, 
	const torch::Tensor &out_shs,
	const torch::Tensor &T_diffuse,
	const torch::Tensor &T_specular,

	// Gradients
	const torch::Tensor &v_diffuse_shs,
	const torch::Tensor &v_specular_shs
);

}

#endif // DIFFUSE_SPECULAR_SHS_BWD_CUH