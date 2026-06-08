#include <pybind11/iostream.h>

#include "intersects_visibility_fwd_2dgs.cuh"
#include "intersects_visibility_fwd_2dgs_skinny_outputs.cuh"
#include "diffuse_specular_shs_fwd.cuh"
#include "diffuse_specular_shs_bwd.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("intersects_visibility_fwd_2dgs", &heatgaussian::intersects_visibility_fwd_2dgs_tensor);
    m.def(
        "intersects_visibility_fwd_2dgs",
        &heatgaussian::intersects_visibility_fwd_2dgs_tensor,
         py::call_guard<py::scoped_ostream_redirect,
                     py::scoped_estream_redirect>()
    );
    m.def(
        "intersects_visibility_fwd_2dgs_skinny_outputs",
        &heatgaussian::intersects_visibility_fwd_2dgs_skinny_outputs_tensor,
         py::call_guard<py::scoped_ostream_redirect,
                     py::scoped_estream_redirect>()
    );
    m.def(
        "diffuse_specular_shs_fwd_cpp",
        &heatgaussian::diffuse_specular_shs_cpp_tensor
    );
    m.def(
        "diffuse_specular_shs_fwd",
        &heatgaussian::diffuse_specular_shs_cuda_fwd_tensor
    );
    m.def(
        "diffuse_specular_shs_bwd",
        &heatgaussian::diffuse_specular_shs_bwd_tensor
    );
}