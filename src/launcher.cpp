#include "lite_gpu.h"
#include "lite_gpu_internal.h"
#include <cuda.h>
#include <cstdio>

LiteError liteMemsetAsync(void* ptr, size_t size_bytes, LiteStream stream) {
    if (!ptr || !stream || size_bytes == 0) {
        return LITE_ERROR_INVALID_VALUE;
    }

    CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(ptr);
    CUresult    err = cuMemsetD8Async(dptr, 0xFF, size_bytes, stream->handle);
    return (err == CUDA_SUCCESS) ? LITE_SUCCESS : LITE_ERROR_DRIVER_API;
}

extern "C" CUfunction lite_get_dummy_kernel_func(void);
extern "C" CUfunction lite_get_dummy_kernel_v2_func(void);
extern "C" CUfunction lite_get_dummy_kernel_heavy_func(void);
extern "C" CUfunction lite_get_empty_kernel_func(void);

static CUfunction g_dummy_func = nullptr;
static CUfunction g_empty_func = nullptr;
static CUfunction g_heavy_func = nullptr;

void liteInvalidateCachedFuncs(void) {
    g_dummy_func = nullptr;
    g_empty_func = nullptr;
    g_heavy_func = nullptr;
}

LiteError liteLaunchTinyKernel(void* ptr, int fill, LiteStream stream) {
    if (!ptr || !stream) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if (!g_dummy_func) {
        g_dummy_func = lite_get_dummy_kernel_func();
        if (!g_dummy_func) {
            return LITE_ERROR_DRIVER_API;
        }
    }

    int* p   = reinterpret_cast<int*>(ptr);
    int  val = fill;
    void* args[2] = { &p, &val };

    CUresult err = cuLaunchKernel(
        g_dummy_func,
        1, 1, 1,
        32, 1, 1,
        0,
        stream->handle, args, nullptr);

    return (err == CUDA_SUCCESS) ? LITE_SUCCESS : LITE_ERROR_DRIVER_API;
}

LiteError liteLaunchEmptyKernel(LiteStream stream) {
    if (!stream) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if (!g_empty_func) {
        g_empty_func = lite_get_empty_kernel_func();
        if (!g_empty_func) {
            return LITE_ERROR_DRIVER_API;
        }
    }

    CUresult err = cuLaunchKernel(
        g_empty_func,
        1, 1, 1,
        1, 1, 1,
        0,
        stream->handle,
        nullptr,
        nullptr
    );

    return (err == CUDA_SUCCESS) ? LITE_SUCCESS : LITE_ERROR_DRIVER_API;
}

LiteError liteLaunchDummyKernelHeavy(void* ptr, int loops, int fill, LiteStream stream) {
    if (!ptr || !stream || loops <= 0) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if (!g_heavy_func) {
        g_heavy_func = lite_get_dummy_kernel_heavy_func();
        if (!g_heavy_func) {
            return LITE_ERROR_DRIVER_API;
        }
    }

    int* p   = reinterpret_cast<int*>(ptr);
    void* args[3] = { &p, &loops, &fill };

    CUresult err = cuLaunchKernel(
        g_heavy_func,
        1, 1, 1,
        1, 1, 1,
        0,
        stream->handle, args, nullptr);

    return (err == CUDA_SUCCESS) ? LITE_SUCCESS : LITE_ERROR_DRIVER_API;
}
