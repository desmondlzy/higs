import torch
import os

from torch.utils.cpp_extension import load
dirname = os.path.join(os.path.dirname(__file__))
build_dir = os.path.join(os.path.dirname(__file__), "build-torchbindings")

if not os.path.exists(build_dir):
	os.makedirs(build_dir)

# match all the .cu files in the directory
cuda_sources = [
	os.path.join(dirname, f) for f in os.listdir(dirname) if f.endswith(".cu")
]
			
sources = [
	os.path.join(dirname, "ext.cpp"),
	# os.path.join(dirname, "spherical_harmonics.cuh"),
	# os.path.join(dirname, "tensors.cuh"),
	os.path.join(dirname, "deps/gsplat/gsplat/cuda/csrc/compute_sh_fwd.cu"),
	*cuda_sources,
]

include_dirs = [
	os.path.join(dirname, "deps/gsplat"),
	os.path.join(dirname, "deps/gsplat/gsplat/cuda/include"),
	os.path.join(dirname, "deps/gsplat/gsplat/cuda/csrc/third_party/glm"),
]

_heatgaussian_impl = load(
	name="_heatgaussian_impl", 
	sources=sources,  
	build_directory=build_dir, 
    extra_include_paths=include_dirs,
	verbose=True)
