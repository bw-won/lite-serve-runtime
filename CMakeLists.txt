cmake_minimum_required(VERSION 3.18)
project(lite_gpu_runtime LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

option(LITE_WITH_NCCL "Enable NCCL multi-GPU communicator (cloud TP)" OFF)

add_library(lite_gpu STATIC
    src/context.cpp
    src/stream.cpp
    src/memory.cpp
    src/launcher.cpp
    src/dummy_kernel.cu
    src/graph.cpp
    src/profiler.cpp
    src/kv_pool.cpp
    src/scheduler.cpp
    src/paged_attention.cpp
    src/comm.cpp
)

# kernels/ 放 AOT 生成的 paged_attn_meta.h（cubin 字节 + 符号名）。
# 该头由 tools/aot_paged_attention.py 在开发机生成，是强制构建产物：缺失时
# paged_attention.cpp 直接 #include 失败（fail-fast），不提供无 cubin 的降级桩。
target_include_directories(lite_gpu PUBLIC include PRIVATE src ${CMAKE_CURRENT_SOURCE_DIR}/kernels)
target_link_libraries(lite_gpu PRIVATE cupti CUDA::cuda_driver CUDA::cudart)

add_executable(test_runtime tests/test_runtime.cpp)
target_link_libraries(test_runtime PRIVATE lite_gpu CUDA::cuda_driver cublas)

if(LITE_WITH_NCCL)
    find_path(NCCL_INCLUDE_DIR nccl.h
        HINTS ENV NCCL_INCLUDE_DIR ENV CONDA_PREFIX
        PATH_SUFFIXES include)
    find_library(NCCL_LIBRARY
        NAMES nccl libnccl.so.2 nccl.so.2
        HINTS ENV NCCL_LIBRARY ENV NCCL_LIBDIR ENV CONDA_PREFIX
        PATH_SUFFIXES lib lib64)
    if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
        message(FATAL_ERROR "LITE_WITH_NCCL=ON but NCCL not found. "
                            "Install libnccl-dev or `conda install -c nvidia nccl`, "
                            "or set NCCL_INCLUDE_DIR / NCCL_LIBRARY.")
    endif()
    message(STATUS "NCCL include: ${NCCL_INCLUDE_DIR}")
    message(STATUS "NCCL library: ${NCCL_LIBRARY}")

    target_sources(lite_gpu PRIVATE src/comm_nccl.cpp)
    target_include_directories(lite_gpu PRIVATE ${NCCL_INCLUDE_DIR})
    target_compile_definitions(lite_gpu PUBLIC LITE_WITH_NCCL)
    target_link_libraries(lite_gpu PRIVATE ${NCCL_LIBRARY})

    add_executable(test_tp_multigpu tests/test_tp_multigpu.cpp)
    target_link_libraries(test_tp_multigpu PRIVATE lite_gpu CUDA::cuda_driver CUDA::cudart ${NCCL_LIBRARY})
endif()
