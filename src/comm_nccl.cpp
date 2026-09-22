// comm_nccl.cpp — 真机多卡张量并行（TP）通信域，基于 NCCL。仅在 -DLITE_WITH_NCCL=ON 编译。
//
// 与单卡 SimComm（comm.cpp）的区别：
//   - 每个 rank 对应一张物理 device，使用其【primary context】（cuDevicePrimaryCtxRetain，
//     与 runtime API cudaSetDevice 看到的是同一个 context，NCCL 官方支持最稳；不用
//     cuCtxCreate 自建 context，避免 NCCL 内部 cuCtxGetCurrent/流归属检查的兼容问题）。
//   - 跨卡指针不可互相解引用，半头输出的汇聚走真实 ncclAllGather（PCIe/NVLink）。
//   - ncclCommInitRankConfig / ncclAllGather 都是【集合操作】，world 个 rank 必须并发
//     进入，故 bootstrap 用每 rank 一个线程完成；运行期 all-gather 也由测试的 worker
//     线程在各自 rank 上并发调用。
//
// 线程模型：cuCtxSetCurrent 是线程局部状态。任何 rank 的 Driver/NCCL 调用前，该线程
// 必须先 liteNcclRankCtx() + liteSetCurrentContext() 切到本 rank 的 primary context。
#include "lite_gpu.h"
#include "lite_gpu_internal.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <new>
#include <vector>
#include <thread>
#include <cstdio>

struct LiteNcclGroup_s {
    int world = 0;
    int base_device = 0;
    std::vector<CUcontext> ctx;                 // 每 rank 的 primary context（retain）
    std::vector<CUdevice>  dev;
    std::vector<LiteStream> streams;            // 每 rank 一条 NON_BLOCKING 计算/通信流
    std::vector<ncclComm_t> comm;               // 每 rank 一个 NCCL 通信域
};

static LiteError nccl_fail(const char* what, ncclResult_t r) {
    fprintf(stderr, "[comm_nccl] %s failed: %s\n", what, ncclGetErrorString(r));
    return LITE_ERROR_DRIVER_API;
}

LiteError liteNcclGroupCreate(const LiteNcclGroupConfig* cfg, LiteNcclGroup* out) {
    if (!cfg || !out) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if (cfg->world_size < 1) {
        return LITE_ERROR_INVALID_VALUE;
    }

    int ndev = 0;
    LiteError le0 = liteDeviceGetCount(&ndev);   // 内部已检查 runtime init
    if (le0 != LITE_SUCCESS) {
        return le0;
    }

    if (ndev < cfg->base_device + cfg->world_size) {
        fprintf(stderr, "[comm_nccl] need devices [%d,%d), only %d visible\n",
                cfg->base_device, cfg->base_device + cfg->world_size, ndev);
        return LITE_ERROR_INVALID_VALUE;
    }

    LiteNcclGroup_s* g = new (std::nothrow) LiteNcclGroup_s();
    if (!g) {
        return LITE_ERROR_OUT_OF_MEMORY;
    }

    g->world = cfg->world_size;
    g->base_device = cfg->base_device;
    g->ctx.resize(g->world, nullptr);
    g->dev.resize(g->world, 0);
    g->streams.resize(g->world, nullptr);
    g->comm.resize(g->world, nullptr);

    // 1) 主线程顺序 retain 每 rank 的 primary context（与线程无关，引用计数）。
    for (int r = 0; r < g->world; ++r) {
        int devid = g->base_device + r;
        CUdevice dv;
        if (cuDeviceGet(&dv, devid) != CUDA_SUCCESS ||
            cuDevicePrimaryCtxRetain(&g->ctx[r], dv) != CUDA_SUCCESS) {
            fprintf(stderr, "[comm_nccl] primary ctx retain failed for device %d\n", devid);
            liteNcclGroupDestroy(g);
            return LITE_ERROR_DRIVER_API;
        }

        g->dev[r] = dv;
    }

    // 2) 单进程内由 rank0 生成 uniqueId，经 host 内存共享给各 bootstrap 线程。
    ncclUniqueId uid;
    ncclResult_t nr = ncclGetUniqueId(&uid);
    if (nr != ncclSuccess) {
        liteNcclGroupDestroy(g);
        return nccl_fail("ncclGetUniqueId", nr);
    }

    // 3) 每 rank 一个线程：set current primary ctx -> cudaSetDevice（双保险）->
    //    在该 context 建流 -> ncclCommInitRankConfig（集合调用，world 线程并发握手）。
    std::vector<LiteError> rerr(g->world, LITE_SUCCESS);
    std::vector<std::thread> th;
    th.reserve(g->world);
    for (int r = 0; r < g->world; ++r) {
        th.emplace_back([&, r]() {
            cuCtxSetCurrent(g->ctx[r]);
            cudaSetDevice(g->base_device + r);
            LiteError le = liteStreamCreate(&g->streams[r]);   // 建在 current(primary) ctx
            if (le != LITE_SUCCESS) {
                rerr[r] = le;
                return;
            }

            ncclConfig_t nc = NCCL_CONFIG_INITIALIZER;
            nc.blocking = 1;   // 阻塞模式：init 内部错误直接返回而非异步
            nr = ncclCommInitRankConfig(&g->comm[r], g->world, uid, r, &nc);
            if (nr != ncclSuccess) {
                rerr[r] = LITE_ERROR_DRIVER_API;
            }
        });
    }

    for (auto& t : th) {
        t.join();
    }

    for (int r = 0; r < g->world; ++r) {
        if (rerr[r] != LITE_SUCCESS) {
            fprintf(stderr, "[comm_nccl] rank %d bootstrap failed\n", r);
            liteNcclGroupDestroy(g);
            return rerr[r];
        }
    }

    *out = g;
    return LITE_SUCCESS;
}

void liteNcclGroupDestroy(LiteNcclGroup g) {
    if (!g) {
        return;
    }

    // NCCL 销毁也是集合操作：每 rank 线程并发 ncclCommDestroy，再销毁流。
    if (g->world > 0 && !g->comm.empty()) {
        std::vector<std::thread> th;
        for (int r = 0; r < g->world; ++r) {
            if (!g->comm[r] && !g->streams[r] && !g->ctx[r]) {
                continue;
            }

            th.emplace_back([&, r]() {
                if (g->ctx[r]) {
                    cuCtxSetCurrent(g->ctx[r]);
                }

                if (g->comm[r]) {
                    ncclCommDestroy(g->comm[r]);
                }

                if (g->streams[r]) {
                    liteStreamDestroy(g->streams[r]);
                }
            });
        }

        for (auto& t : th) {
            t.join();
        }
    }

    for (int r = 0; r < g->world; ++r) {
        if (g->ctx[r]) {
            cuDevicePrimaryCtxRelease(g->dev[r]);
        }
    }

    delete g;
}

LiteError liteNcclRankCtx(LiteNcclGroup g, int rank, LiteContext* ctx) {
    if (!g || !ctx || rank < 0 || rank >= g->world) {
        return LITE_ERROR_INVALID_VALUE;
    }

    *ctx = g->ctx[rank];
    return LITE_SUCCESS;
}

LiteError liteNcclRankStream(LiteNcclGroup g, int rank, LiteStream* out) {
    if (!g || !out || rank < 0 || rank >= g->world) {
        return LITE_ERROR_INVALID_VALUE;
    }

    *out = g->streams[rank];
    return LITE_SUCCESS;
}

LiteError liteNcclAllGatherFp16(LiteNcclGroup g, int rank,
                                const void* d_send, void* d_recv, size_t send_count) {
    if (!g || !d_send || !d_recv || send_count == 0) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if (rank < 0 || rank >= g->world) {
        return LITE_ERROR_INVALID_VALUE;
    }

    // 调用方在 rank 自己的 worker 线程里调用；这里再保险切一次 current context。
    cuCtxSetCurrent(g->ctx[rank]);
    cudaSetDevice(g->base_device + rank);
    // send_count = 本 rank 发送的 fp16 元素数；d_recv 容量须 >= world*send_count。
    // NCCL 输出按 rank 块排列：[rank][send_count]。
    ncclResult_t nr = ncclAllGather(d_send, d_recv, send_count, ncclFloat16,
                                    g->comm[rank], (cudaStream_t)g->streams[rank]->handle);
    return nr == ncclSuccess ? LITE_SUCCESS : nccl_fail("ncclAllGather", nr);
}

LiteError liteNcclRankSync(LiteNcclGroup g, int rank) {
    if (!g || rank < 0 || rank >= g->world) {
        return LITE_ERROR_INVALID_VALUE;
    }

    return liteStreamSynchronize(g->streams[rank]);
}
