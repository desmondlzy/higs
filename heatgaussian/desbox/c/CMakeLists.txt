cmake_minimum_required(VERSION 3.12)  # You can adjust the minimum required version
set(CMAKE_CUDA_ARCHITECTURES 70 75 89)  # Ti 2080 uses 75. V100 uses 70. RTX 4090 uses 89.

project(relight_cuda CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

# Ensure `-fPIC` is used
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Define CUDA flags
set(NVCC_FLAGS "-O3 --use_fast_math")

find_package(Python3 REQUIRED)

# Get the Torch path from Python
execute_process(
    COMMAND python -c "import torch.utils; print(torch.utils.cmake_prefix_path)"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _python_prefix_path
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Check if the command was successful
if (_result EQUAL 0)
    # Set the CMAKE_PREFIX_PATH
    set(CMAKE_PREFIX_PATH "${_python_prefix_path}" CACHE PATH "Path to Torch")
    message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
else()
    message(FATAL_ERROR "Failed to get CMAKE_PREFIX_PATH from Python")
endif()

set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_HOME})
find_package(CUDA)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


# Set up CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${NVCC_FLAGS}")

# our library library
add_library(${PROJECT_NAME} 
    src/spherical_harmonics.cuh
    src/bindings.cpp
    src/tensors.cuh
    src/tensors.cu
)
target_link_libraries(${PROJECT_NAME} PUBLIC cuda)
target_link_libraries(${PROJECT_NAME} PUBLIC "${TORCH_LIBRARIES}")

include_directories(${PYTHON_INCLUDE_DIRS})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/glm
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "70;75;86")

# # To add an executable that uses the gsplat library,
# # follow example in the comments for a script `run_forward.cpp`
# # Add the executable
# add_executable(run_forward run_forward.cpp)

# # Link against CUDA runtime library
# target_link_libraries(run_forward PUBLIC cuda gsplat)

# # Include directories for the header-only library
# target_include_directories(run_forward PRIVATE
#     ${PROJECT_SOURCE_DIR}/third_party/glm
# )
