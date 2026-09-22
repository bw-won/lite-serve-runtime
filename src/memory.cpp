#include "lite_gpu.h"
#include "lite_gpu_internal.h"
#include <vector>
#include <unordered_map>
#include <mutex>
#include <cstdint>

struct PageBlock {
    size_t     offset;
    bool       in_use;
    LiteStream last_stream;
    LiteEvent  sync_event;
    PageBlock* next_free;
};

static struct {
    CUdeviceptr base_ptr;
    std::vector<PageBlock> blocks;
    std::unordered_map<CUdeviceptr, size_t> ptr_to_idx;
    PageBlock* free_head;
    size_t page_size;
    size_t max_pages;
    std::mutex mtx;
    bool initialized;
} g_pool = {};

LiteError liteMemPoolInit(size_t page_size, size_t max_pages) {
    std::lock_guard<std::mutex> lock(g_pool.mtx);
    if (g_pool.initialized) {
        return LITE_ERROR_INVALID_VALUE;
    }

    g_pool.page_size = page_size;
    g_pool.max_pages = max_pages;

    size_t total_bytes = page_size * max_pages;
    CUresult err = cuMemAlloc(&g_pool.base_ptr, total_bytes);
    if (err != CUDA_SUCCESS) {
        return LITE_ERROR_OUT_OF_MEMORY;
    }

    g_pool.blocks.resize(max_pages);
    g_pool.ptr_to_idx.reserve(max_pages);
    g_pool.free_head = nullptr;

    for (size_t i = 0; i < max_pages; ++i) {
        LiteEvent evt;
        liteEventCreate(&evt);

        size_t offset = i * page_size;
        CUdeviceptr page_ptr = g_pool.base_ptr + offset;

        g_pool.blocks[i] = {offset, false, nullptr, evt, g_pool.free_head};
        g_pool.ptr_to_idx[page_ptr] = i;
        g_pool.free_head = &g_pool.blocks[i];
    }

    g_pool.initialized = true;
    return LITE_SUCCESS;
}

LiteError liteMemPoolDestroy() {
    std::lock_guard<std::mutex> lock(g_pool.mtx);
    if (!g_pool.initialized) {
        return LITE_SUCCESS;
    }

    for (auto& blk : g_pool.blocks) {
        if (blk.sync_event) {
            liteEventDestroy(blk.sync_event);
        }
    }

    if (g_pool.base_ptr) {
        cuMemFree(g_pool.base_ptr);
    }

    g_pool.blocks.clear();
    g_pool.ptr_to_idx.clear();
    g_pool.free_head = nullptr;
    g_pool.base_ptr = 0;
    g_pool.initialized = false;
    return LITE_SUCCESS;
}

LiteError liteMemAllocPage(void** ptr, LiteStream stream) {
    if (!ptr || !stream) {
        return LITE_ERROR_INVALID_VALUE;
    }

    std::lock_guard<std::mutex> lock(g_pool.mtx);
    if (!g_pool.free_head) {
        return LITE_ERROR_OUT_OF_MEMORY;
    }

    PageBlock* blk = g_pool.free_head;
    g_pool.free_head = blk->next_free;

    if (blk->last_stream && blk->last_stream != stream) {
        liteStreamWaitEvent(stream, blk->sync_event);
    }

    blk->in_use = true;
    blk->last_stream = stream;
    blk->next_free = nullptr;
    *ptr = reinterpret_cast<void*>(g_pool.base_ptr + blk->offset);
    return LITE_SUCCESS;
}

LiteError liteMemFreePage(void* ptr, LiteStream stream) {
    if (!ptr || !stream) {
        return LITE_ERROR_INVALID_VALUE;
    }

    std::lock_guard<std::mutex> lock(g_pool.mtx);

    CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(ptr);
    auto        it = g_pool.ptr_to_idx.find(dptr);
    if (it == g_pool.ptr_to_idx.end()) {
        return LITE_ERROR_INVALID_VALUE;
    }

    size_t     idx = it->second;
    PageBlock* blk = &g_pool.blocks[idx];
    if (!blk->in_use) {
        return LITE_ERROR_INVALID_VALUE;
    }

    liteEventRecord(blk->sync_event, stream);
    blk->last_stream = stream;
    blk->in_use = false;
    blk->next_free = g_pool.free_head;
    g_pool.free_head = blk;
    return LITE_SUCCESS;
}

LiteError liteMemcpyHtoDAsync(void* dst, const void* src, size_t size_bytes, LiteStream stream) {
    if (!dst || !src || !stream || size_bytes == 0) {
        return LITE_ERROR_INVALID_VALUE;
    }

    CUdeviceptr d_dst = reinterpret_cast<CUdeviceptr>(dst);
    CUresult err = cuMemcpyHtoDAsync(d_dst, src, size_bytes, stream->handle);
    return (err == CUDA_SUCCESS) ? LITE_SUCCESS : LITE_ERROR_DRIVER_API;
}

LiteError liteMemcpyDtoHAsync(void* dst, const void* src, size_t size_bytes, LiteStream stream) {
    if (!dst || !src || !stream || size_bytes == 0) {
        return LITE_ERROR_INVALID_VALUE;
    }

    CUdeviceptr d_src = reinterpret_cast<CUdeviceptr>(src);
    CUresult err = cuMemcpyDtoHAsync(dst, d_src, size_bytes, stream->handle);
    return (err == CUDA_SUCCESS) ? LITE_SUCCESS : LITE_ERROR_DRIVER_API;
}

LiteError liteMemcpyDtoDAsync(void* dst, const void* src, size_t size, LiteStream stream) {
    if (!dst || !src || !stream) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if (size == 0) {
        return LITE_SUCCESS;
    }

    LiteStream_s* s = static_cast<LiteStream_s*>(stream);
    CUresult err = cuMemcpyDtoDAsync((CUdeviceptr)dst, (CUdeviceptr)src, size, s->handle);
    if (err != CUDA_SUCCESS) {
        return LITE_ERROR_DRIVER_API;
    }

    return LITE_SUCCESS;
}
