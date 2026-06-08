#pragma once

#include <torch/extension.h>
#include <glm/gtc/type_ptr.hpp>

namespace heatgaussian {

template <typename T> using vec2 = glm::vec<2, T>;
template <typename T> using vec3 = glm::vec<3, T>;

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
intersects_visibility_fwd_2dgs_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &ray_transforms,    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,    // [C, N, channels] or [nnz, channels]
    const torch::Tensor &opacities, // [C, N]  or [nnz]
    const torch::Tensor &normals,   // [C, N, 3] or [nnz, 3]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    const at::optional<torch::Tensor> &masks, // [C, tile_height, tile_width]
    const at::optional<torch::Tensor> &gaussian_masks, // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,   // [n_isects]
    const uint32_t max_hits
);

// std::tuple<
//     torch::Tensor,
//     torch::Tensor,
//     torch::Tensor,
//     torch::Tensor>
// intersects_visibility_fwd_2dgs_tensor(
//     // Gaussian parameters
//     const torch::Tensor &means2d,   // [C, N, 2] or [nnz, 2]
//     const torch::Tensor &ray_transforms,    // [C, N, 3] or [nnz, 3]
//     const torch::Tensor &opacities, // [C, N]  or [nnz]
//     const torch::Tensor &normals,   // [C, N, 3] or [nnz, 3]
//     // const at::optional<torch::Tensor> &backgrounds, // [C, channels]
//     const at::optional<torch::Tensor> &masks, // [C, tile_height, tile_width]
//     // image size
//     const int64_t image_width,
//     const int64_t image_height,
//     const int64_t tile_size,
//     // intersections
//     const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
//     const torch::Tensor &flatten_ids,   // [n_isects]
//     const int64_t max_hits
// );

}
