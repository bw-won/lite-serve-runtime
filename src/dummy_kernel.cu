#include <cuda_runtime.h>
#include <cuda.h>

extern "C" __global__ void lite_dummy_kernel(int* buf, int fill) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        buf[0] = fill;
    }
}

extern "C" CUfunction lite_get_dummy_kernel_func(void) {
    cudaFunction_t f = nullptr;
    cudaError_t err = cudaGetFuncBySymbol(&f, (const void*)lite_dummy_kernel);
    return (err == cudaSuccess) ? (CUfunction)f : nullptr;
}

extern "C" __global__ void lite_dummy_kernel_v2(int* buf, int fill) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        buf[0] = fill + 100;
    }
}

extern "C" CUfunction lite_get_dummy_kernel_v2_func(void) {
    cudaFunction_t f = nullptr;
    cudaError_t err = cudaGetFuncBySymbol(&f, (const void*)lite_dummy_kernel_v2);
    return (err == cudaSuccess) ? (CUfunction)f : nullptr;
}

extern "C" __global__ void lite_empty_kernel() {
    // 仅执行 launch 机制
}

extern "C" CUfunction lite_get_empty_kernel_func(void) {
    cudaFunction_t f = nullptr;
    cudaError_t err = cudaGetFuncBySymbol(&f, (const void*)lite_empty_kernel);
    return (err == cudaSuccess) ? (CUfunction)f : nullptr;
}

extern "C" __global__ void lite_dummy_kernel_heavy(int* buf, int loops, int fill) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < loops; ++i) {
            sum += fill;
        }
        buf[0] = sum;
    }
}

extern "C" CUfunction lite_get_dummy_kernel_heavy_func(void) {
    cudaFunction_t f = nullptr;
    cudaError_t err = cudaGetFuncBySymbol(&f, (const void*)lite_dummy_kernel_heavy);
    return (err == cudaSuccess) ? (CUfunction)f : nullptr;
}
