#include <cuda_runtime.h>

#include <torch/extension.h>

__global__ void add_kernel(
	float *a,
	float *b,
	float *res,
	int n
) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		res[i] = a[i] + b[i];
	}
}

torch::Tensor add_tensor(
	torch::Tensor a,
	torch::Tensor b
) {
	const int n = a.size(0);
	const int threads = 256;
	const int blocks = (n + threads - 1) / threads;
	auto res = torch::empty_like(a);
	add_kernel<<<blocks, threads>>>(
		a.data_ptr<float>(),
		b.data_ptr<float>(),
		res.data_ptr<float>(),
		n
	);
	return res;
}