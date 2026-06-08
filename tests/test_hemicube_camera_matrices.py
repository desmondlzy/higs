#%%
"""
pytest -q tests/test_hemicube_camera_matrices.py 
"""

from heatgaussian.hemicube_camera_matrices import hemicube_camera_matrices_lightweight, viewmatrix

import torch

def test_viewmatrix_1d_input():
	"""
	same convention as nerfstudio
	"""
	eyes = torch.tensor([0.0, 0.0, 2.0])
	look_dirs = torch.tensor([0.0, 0.0, 1.0])
	ups = torch.tensor([0.0, 1.0, 0.0])

	mat = viewmatrix(look_dirs, ups, eyes)

	assert mat.shape == (3, 4), mat.shape


def test_viewmatrix_2d_input():
	"""
	same convention as nerfstudio
	"""
	eyes = torch.tensor([[0.0, 0.0, 0.0]])
	look_dirs = torch.tensor([[0.0, 0.0, 1.0]])
	ups = torch.tensor([[0.0, 1.0, 0.0]])

	mat2d = viewmatrix(look_dirs, ups, eyes)

	mat1d = viewmatrix(look_dirs[0], ups[0], eyes[0])

	assert mat2d.shape == (1, 3, 4), mat2d.shape
	assert torch.allclose(mat2d[0], mat1d), f"{mat2d[0]} != {mat1d}"


def test_viewmatrix_2d_input_2cameras():
	"""
	same convention as nerfstudio
	"""
	eyes = torch.tensor([
		[0.0, 0.0, 0.0],
		[1.0, 1.0, 1.0],
	])
	look_dirs = torch.tensor([
		[0.0, 1.0, 1.0],
		[1.0, 0.0, 0.0],
	])
	ups = torch.tensor([
		[1.0, 0.0, 0.0],
		[0.0, 1.0, 0.0],
	])

	mats2d = viewmatrix(look_dirs, ups, eyes)
	mats1d = viewmatrix(look_dirs[0], ups[0], eyes[0])

	assert mats2d.shape == (2, 3, 4), mats2d.shape
	assert torch.allclose(mats2d[0], mats1d), f"{mats2d[0]} != {mats1d}"

# # def test_hemicube_camera_matrices():
# 	eyes = torch.tensor([[0.0, 0.0, 0.0]])
# 	look_dirs = torch.tensor([[0.0, 0.0, 1.0]])
# 	ups = torch.tensor([[0.0, 1.0, 0.0]])

# 	resolution = 512
# 	hemicube_viewmats, hemicube_Ks = hemicube_camera_matrices_lightweight(
# 		eyes,
# 		look_dirs,
# 		ups,
# 		resolution
# 	)

# 	# Check the shapes of the output tensors
# 	assert hemicube_viewmats.shape == (1, 4, 4), hemicube_viewmats.shape
# 	assert hemicube_Ks.shape == (1, 3, 3), hemicube_Ks.shape


def test_hemicube_camera_matrices_1d():
	eyes = torch.tensor([0.0, 0.0, 0.0])
	look_dirs = torch.tensor([0.0, 0.0, 1.0])
	ups = torch.tensor([0.0, 1.0, 0.0])

	resolution = 512
	hemicube_viewmats, hemicube_Ks = hemicube_camera_matrices_lightweight(
		eyes,
		look_dirs,
		ups,
		resolution
	)

	# Check the shapes of the output tensors
	assert hemicube_viewmats.shape == (5, 4, 4), hemicube_viewmats.shape
	assert hemicube_Ks.shape == (5, 3, 3), hemicube_Ks.shape


def test_hemicube_camera_matrices_2d():
	eyes = torch.tensor([[0.0, 0.0, 0.0]])
	look_dirs = torch.tensor([[0.0, 0.0, 1.0]])
	ups = torch.tensor([[0.0, 1.0, 0.0]])

	resolution = 512
	hemicube_viewmats, hemicube_Ks = hemicube_camera_matrices_lightweight(
		eyes,
		look_dirs,
		ups,
		resolution
	)

	# Check the shapes of the output tensors
	assert hemicube_viewmats.shape == (1, 5, 4, 4), hemicube_viewmats.shape
	assert hemicube_Ks.shape == (1, 5, 3, 3), hemicube_Ks.shape


def test_hemicube_camera_matrices_2d_2cam():
	eyes = torch.tensor([
		[0.0, 0.0, 0.0],
		[0.0, 0.0, 0.0],
	])
	look_dirs = torch.tensor([
		[0.0, 0.0, 1.0],
		[0.0, 0.0, 1.0],
	])
	ups = torch.tensor([
		[0.0, 1.0, 0.0],
		[0.0, 1.0, 0.0],
	])

	resolution = 512
	hemicube_viewmats, hemicube_Ks = hemicube_camera_matrices_lightweight(
		eyes,
		look_dirs,
		ups,
		resolution
	)

	# Check the shapes of the output tensors
	assert hemicube_viewmats.shape == (2, 5, 4, 4), hemicube_viewmats.shape
	assert hemicube_Ks.shape == (2, 5, 3, 3), hemicube_Ks.shape

	for i in range(2):
		assert torch.allclose(hemicube_viewmats[i], hemicube_viewmats[0]), f"{hemicube_viewmats[i]} != {hemicube_viewmats[0]}"
		assert torch.allclose(hemicube_Ks[i], hemicube_Ks[0]), f"{hemicube_Ks[i]} != {hemicube_Ks[0]}"

#%%
test_hemicube_camera_matrices_1d()