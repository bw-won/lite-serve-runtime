#include "lite_gpu.h"
#include "lite_gpu_internal.h"
#include <cuda.h>

LiteError liteLaunchDummyKernel(void* ptr, size_t size_bytes, LiteStream stream) {
    if (!ptr || !stream || size_bytes == 0) return LITE_ERROR_INVALID_VALUE;
    CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(ptr);

    // 🔥 P0修复：使用调用方传入的 size，零隐式契约
    CUresult err = cuMemsetD8Async(dptr, 0xFF, size_bytes, stream->handle);
    return (err == CUDA_SUCCESS) ? LITE_SUCCESS : LITE_ERROR_DRIVER_API;
}
