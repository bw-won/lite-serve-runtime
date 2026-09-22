#include "lite_gpu.h"
#include <cuda.h>
#include <cstdio>

// 🔥 P0修复：模块级保存 Context 句柄
static CUcontext g_ctx = nullptr;

LiteError liteInit() {
    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) return LITE_ERROR_DRIVER_API;

    CUdevice dev;
    err = cuDeviceGet(&dev, 0);
    if (err != CUDA_SUCCESS) return LITE_ERROR_DRIVER_API;

    err = cuCtxCreate(&g_ctx, 0, dev);
    if (err != CUDA_SUCCESS) return LITE_ERROR_DRIVER_API;

    return LITE_SUCCESS;
}

LiteError liteShutdown() {
    // 🔥 P0修复：正确销毁已保存的 Context 并置空
    if (g_ctx) {
        CUresult err = cuCtxDestroy(g_ctx);
        g_ctx = nullptr;
        if (err != CUDA_SUCCESS) return LITE_ERROR_DRIVER_API;
    }
    return LITE_SUCCESS;
}
