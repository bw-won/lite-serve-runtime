cmake_minimum_required(VERSION 3.18)
project(lite_gpu_runtime LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 查找 CUDA Toolkit (包含 Driver API)
find_package(CUDAToolkit REQUIRED)

# 编译 Runtime 静态库
add_library(lite_gpu STATIC
    src/context.cpp
    src/stream.cpp
    src/memory.cpp
    src/launcher.cpp
    src/graph.cpp
    src/profiler.cpp
)
target_include_directories(lite_gpu PUBLIC include PRIVATE src)
target_link_libraries(lite_gpu PRIVATE CUDA::cuda_driver)

# 编译测试可执行文件
add_executable(test_runtime tests/test_runtime.cpp)
target_link_libraries(test_runtime PRIVATE lite_gpu CUDA::cuda_driver)