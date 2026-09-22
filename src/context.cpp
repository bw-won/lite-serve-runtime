#include "lite_gpu.h"
#include <cuda.h>
#include <cstdio>
#include <atomic>
#include <mutex>

static CUcontext g_ctx = nullptr;
static std::mutex g_init_mtx;
std::atomic<bool> g_runtime_inited{false};

bool liteIsRuntimeInited() {
    return g_runtime_inited.load(std::memory_order_acquire);
}

LiteError liteInit() {
    if (g_runtime_inited.load()) {
        return LITE_SUCCESS;
    }

    std::lock_guard<std::mutex> lock(g_init_mtx);
    if (g_runtime_inited.load()) {
        return LITE_SUCCESS;
    }

    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
        return LITE_ERROR_DRIVER_API;
    }

    CUdevice dev;
    err = cuDeviceGet(&dev, 0);
    if (err != CUDA_SUCCESS) {
        return LITE_ERROR_DRIVER_API;
    }

    err = cuCtxCreate(&g_ctx, 0, dev);
    if (err != CUDA_SUCCESS) {
        return LITE_ERROR_DRIVER_API;
    }

    g_runtime_inited.store(true, std::memory_order_release);
    return LITE_SUCCESS;
}

LiteError liteShutdown() {
    if (!g_runtime_inited.load()) {
        return LITE_SUCCESS;
    }

    std::lock_guard<std::mutex> lock(g_init_mtx);
    if (!g_runtime_inited.load()) {
        return LITE_SUCCESS;
    }

    liteInvalidateCachedFuncs();

    if (g_ctx) {
        CUresult err = cuCtxDestroy(g_ctx);
        g_ctx = nullptr;
        if (err != CUDA_SUCCESS) {
            return LITE_ERROR_DRIVER_API;
        }
    }

    g_runtime_inited.store(false, std::memory_order_release);
    return LITE_SUCCESS;
}

LiteError liteGetContext(LiteContext* ctx) {
    if (!ctx) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if (!g_runtime_inited.load()) {
        return LITE_ERROR_NOT_INITIALIZED;
    }

    *ctx = g_ctx;
    return LITE_SUCCESS;
}

LiteError liteSetCurrentContext(LiteContext ctx) {
    if (!ctx) {
        return LITE_ERROR_INVALID_VALUE;
    }

    CUresult err = cuCtxSetCurrent((CUcontext)ctx);
    return (err == CUDA_SUCCESS) ? LITE_SUCCESS : LITE_ERROR_DRIVER_API;
}

LiteError liteDeviceGetCount(int* count) {
    if (!count) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if (!g_runtime_inited.load()) {
        return LITE_ERROR_NOT_INITIALIZED;
    }

    int n = 0;
    CUresult err = cuDeviceGetCount(&n);
    if (err != CUDA_SUCCESS) {
        return LITE_ERROR_DRIVER_API;
    }

    *count = n;
    return LITE_SUCCESS;
}
