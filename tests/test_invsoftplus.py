from heatgaussian.invsoftplus import invsoftplus
import torch

def test_invsoftplus():
	x = torch.tensor([-torch.inf, -50.0, -20, 0.001, 1.0, 2.0, 3.0, 100.0], dtype=torch.float32, device='cuda')

	assert torch.allclose(
		x, 
		invsoftplus(torch.nn.functional.softplus(x)),
		rtol=1e-5, atol=1e-5,
	), f"invsoftplus failed for x={x}, got {invsoftplus(torch.nn.functional.softplus(x))}"

