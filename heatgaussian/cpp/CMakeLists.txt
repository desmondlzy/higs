cmake_minimum_required(VERSION 3.12)  # You can adjust the minimum required version
set(CMAKE_CUDA_ARCHITECTURES 70 75 89)  # Ti 2080 uses 75. V100 uses 70. RTX 4090 uses 89.

project(heatgaussian_impl CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

# Ensure `-fPIC` is used
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Define CUDA flags
set(NVCC_FLAGS "-O3 --use_fast_math")

# execute_process(
#     COMMAND python -c "import sys; print(sys.executable)"
#     RESULT_VARIABLE _result
#     OUTPUT_VARIABLE _python_executable_path
#     OUTPUT_STRIP_TRAILING_WHITESPACE
# )
# if (_result EQUAL 0)
#     message(STATUS "Python executable path: ${_python_executable_path}")
#     set(PYTHON_EXECUTABLE ${_python_executable_path})
# else()
#     message(FATAL_ERROR "Failed to get Python executable path")
# endif()
find_package(Python3 COMPONENTS Development)
if (Python3_FOUND)
    message(STATUS "Python version: ${Python3_VERSION}")
    message(STATUS "Python executable found: ${Python3_EXECUTABLE}")
    message(STATUS "Python dir found: ${Python3_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "Python not found")
endif()
include_directories(${Python3_INCLUDE_DIRS})

# Get the Torch path from Python
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch.utils; print(torch.utils.cmake_prefix_path)"
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

# help cmake find the CUDA toolkit, and nvtx
set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_HOME})
find_package(CUDA)

# Set up CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${NVCC_FLAGS}")

find_package(Torch REQUIRED)
if (Torch_FOUND)
    message(STATUS "Torch found")
    message(STATUS "Torch include path: ${TORCH_INCLUDE_DIRS}")
    message(STATUS "Torch lib path: ${TORCH_LIBRARIES}")
else()
    message(FATAL_ERROR "Torch not found")
endif()

include_directories(${TORCH_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

FILE(GLOB SOURCE_FILES *.cu)
message(STATUS "Source files: ${SOURCE_FILES}")

# our library library
add_executable(tests
    tests/test_intersects_visibility_fwd_2dgs.cu
    ext.cpp
    ${SOURCE_FILES}
)


include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(deps/gsplat/gsplat/cuda/csrc)

target_link_libraries(tests PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(tests PRIVATE ${PYTHON_LIBRARIES})
target_link_libraries(tests PRIVATE cuda gsplat)
target_include_directories(tests PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/deps/gsplat)
target_include_directories(tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/deps/gsplat/gsplat/cuda/csrc/third_party/glm)

# add_library(heatgaussian_impl SHARED ext.cu ${SOURCE_FILES})
# target_link_libraries(heatgaussian_impl PUBLIC ${TORCH_LIBRARIES})
# target_link_libraries(heatgaussian_impl PUBLIC ${PYTHON_LIBRARIES})
# target_link_libraries(heatgaussian_impl PUBLIC cuda gsplat)
# target_include_directories(heatgaussian_impl PUBLIC ${TORCH_INCLUDE_DIRS})
# target_include_directories(heatgaussian_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/deps/gsplat)
# target_include_directories(heatgaussian_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/deps/gsplat/gsplat/cuda/csrc/third_party/glm)

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
