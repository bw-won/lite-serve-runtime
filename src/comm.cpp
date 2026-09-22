// comm.cpp — 张量并行（TP）通信抽象（本地单卡 SimComm）。
//
// liteCommCreateSim 提供 SimComm —— world_size 个 rank 共享同一 CUDA context，每 rank
// 一条独立 stream；all-gather 用同 context 内的 D2D 拷贝把各 rank 的半头输出按 rank 顺序
// 拼进一块完整输出，再用 event 在 rank0 流上 join。它在单张卡上真实地复现了 TP 的
// "切分 -> 各 rank 并行计算 -> all-gather 汇聚 -> join 同步"数据流与依赖，仅把跨卡链路
// （PCIe/NVLink + NCCL）替换为同卡 D2D。真机多卡 NCCL 通信域由 comm_nccl.cpp 的
// liteNcclGroup* API 提供（-DLITE_WITH_NCCL=ON），与本模拟层相互独立。
#include "lite_gpu.h"
#include <new>
#include <vector>

struct LiteComm_s {
    int world = 0;
    std::vector<LiteStream> streams;     // 每 rank 一条计算流
    std::vector<LiteEvent>  seg_done;    // 每 rank 自己那段 gather 拷贝完成事件
};

LiteError liteCommCreateSim(const LiteCommConfig* cfg, LiteComm* out) {
    if (!cfg || !out) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if (cfg->world_size < 1) {
        return LITE_ERROR_INVALID_VALUE;
    }

    LiteComm_s* c = new (std::nothrow) LiteComm_s();
    if (!c) {
        return LITE_ERROR_OUT_OF_MEMORY;
    }

    c->world = cfg->world_size;
    c->streams.resize(c->world, nullptr);
    c->seg_done.resize(c->world, nullptr);

    for (int r = 0; r < c->world; ++r) {
        LiteError le = liteStreamCreate(&c->streams[r]);
        if (le != LITE_SUCCESS) {
            liteCommDestroy(c);
            return le;
        }

        // fork/join 同步事件用 DEFAULT（与多流 capture 一致），不可用 DISABLE_TIMING。
        le = liteEventCreateForSync(&c->seg_done[r]);
        if (le != LITE_SUCCESS) {
            liteCommDestroy(c);
            return le;
        }
    }

    *out = c;
    return LITE_SUCCESS;
}

LiteError liteCommDestroy(LiteComm c) {
    if (!c) {
        return LITE_ERROR_INVALID_VALUE;
    }

    for (int r = 0; r < c->world; ++r) {
        if (c->seg_done[r]) {
            liteEventDestroy(c->seg_done[r]);
        }

        if (c->streams[r]) {
            liteStreamDestroy(c->streams[r]);
        }
    }

    delete c;
    return LITE_SUCCESS;
}

int liteCommWorldSize(LiteComm c) {
    return c ? c->world : 0;
}

LiteError liteCommRankStream(LiteComm c, int rank, LiteStream* out) {
    if (!c || !out) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if (rank < 0 || rank >= c->world) {
        return LITE_ERROR_INVALID_VALUE;
    }

    *out = c->streams[rank];
    return LITE_SUCCESS;
}

LiteError liteCommAllGatherSim(LiteComm c, void* d_out, const void* const* d_in,
                               int rows, size_t row_bytes) {
    if (!c || !d_out || !d_in || rows < 1 || row_bytes == 0) {
        return LITE_ERROR_INVALID_VALUE;
    }

    // 沿 head(shard) 维汇聚：rank r 的第 row 行放到 d_out 的 [row][r] 位置，
    // 直接摆成 [rows][world][row_bytes]（rank 轴并入 head 轴），无需后续 transpose。
    // 每个 rank 的全部行拷贝都在它自己的流上、排在该 rank 半头 kernel 之后（同流有序），
    // 故天然等待计算完成，无需额外 event。
    for (int r = 0; r < c->world; ++r) {
        if (!d_in[r]) {
            return LITE_ERROR_INVALID_VALUE;
        }

        char* dstBase = reinterpret_cast<char*>(d_out) + (size_t)r * row_bytes;   // [*][r] 起始
        const char* srcBase = static_cast<const char*>(d_in[r]);

        for (int row = 0; row < rows; ++row) {
            char* dst = dstBase + (size_t)row * c->world * row_bytes;
            const char* src = srcBase + (size_t)row * row_bytes;
            LiteError le = liteMemcpyDtoDAsync(dst, src, row_bytes, c->streams[r]);
            if (le != LITE_SUCCESS) {
                return le;
            }
        }

        LiteError le = liteEventRecord(c->seg_done[r], c->streams[r]);
        if (le != LITE_SUCCESS) {
            return le;
        }
    }

    // join：rank0 流等待其余 rank 的行拷贝。rank0 自己的行与后续消费操作同在 rank0 流，
    // 流内有序，无需自等。返回后调用方在 rank0 流上 enqueue 的回读即可见完整 d_out。
    for (int r = 1; r < c->world; ++r) {
        LiteError le = liteStreamWaitEvent(c->streams[0], c->seg_done[r]);
        if (le != LITE_SUCCESS) {
            return le;
        }
    }

    return LITE_SUCCESS;
}

LiteError liteCommBarrier(LiteComm c) {
    if (!c) {
        return LITE_ERROR_INVALID_VALUE;
    }

    for (int r = 0; r < c->world; ++r) {
        LiteError le = liteStreamSynchronize(c->streams[r]);
        if (le != LITE_SUCCESS) {
            return le;
        }
    }

    return LITE_SUCCESS;
}
