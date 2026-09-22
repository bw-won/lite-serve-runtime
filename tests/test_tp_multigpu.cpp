// test_tp_multigpu.cpp — 真机多卡张量并行（TP=2）验证，仅 -DLITE_WITH_NCCL=ON 构建。
//
// 在两张物理 GPU 上各跑一个【半头】split-KV PagedAttention（H=8 -> 4+4，HEADS 是运行时
// 参数，sm_86 cubin 无需重编），用 NCCL all-gather 沿 head 维拼回全头，host reshape 成
// [B][H][D] 后对 host fp32 oracle 校验（期望 max_diff ~ 1e-4，仅 fp16 舍入误差）。
// 随后做两档 all-gather 通信微基准：decode 真实小消息 + 1MB/rank 链路吞吐。
//
// 线程模型：单进程、每 rank 一个 worker 线程；线程内 cuCtxSetCurrent(本 rank primary ctx)
// 后再建 LitePagedAttn（cuMemAlloc/cuModuleLoadData 作用于 current context，故天然落到本卡）。
// 集合调用（ncclAllGather）必须 world 个 rank 并发、count 一致，用可中止 Barrier 对齐；
// 任一线程 setup 失败置 abort，其余线程在 barrier 立即退出，避免集合调用永久挂死。
//
#include "lite_gpu.h"
#include <cuda.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <algorithm>

#define CHECK(cond) do { LiteError _e = (cond); if (_e != LITE_SUCCESS) { \
    fprintf(stderr, "CHECK failed: %s -> err=%d at %s:%d\n", #cond, _e, __FILE__, __LINE__); \
    std::exit(1); } } while (0)
#define CHECK_TRUE(cond, msg) do { if (!(cond)) { \
    fprintf(stderr, "CHECK failed: %s at %s:%d\n", msg, __FILE__, __LINE__); std::exit(1); } } while (0)

// ---------------- 最小确定性 fp16 / mock 数据 / fp32 oracle（与 test_runtime 同源）----------------
static uint16_t f32_to_f16(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x & 0x80000000u) >> 16;
    int32_t  e = (int32_t)((x >> 23) & 255) - 127 + 15;
    uint32_t m = x & 0x007fffffu;
    if (e >= 143) {  // inf/nan
        return (uint16_t)(sign | 0x7c00u | (m ? 0x200u : 0u));
    }

    if (e > 30) {  // overflow
        return (uint16_t)(sign | 0x7c00u);
    }

    if (e <= 0) {  // subnormal/zero
        if (e < -10) {
            return (uint16_t)sign;
        }

        m |= 0x00800000u;
        int sh = 14 - e;
        uint32_t r = m >> sh, rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1))) {
            r++;
        }

        return (uint16_t)(sign | r);
    }

    uint32_t r = ((uint32_t)e << 10) | (m >> 13);
    uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) {
        r++;
    }

    return (uint16_t)(sign | r);
}
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ff, f;
    if (e == 0) {
        if (m == 0) {
            f = sign;
        } else {
            while (!(m & 0x400)) {
                m <<= 1;
                e--;
            }

            m &= 0x3ff; e++;
            f = sign | ((e + 127 - 15) << 23) | (m << 13);
        }
    } else if (e == 31) {
        f = sign | 0x7f800000u | (m << 13);
    } else {
        f = sign | ((e + 127 - 15) << 23) | (m << 13);
    }

    float out; std::memcpy(&out, &f, 4); return out;
}
static uint32_t pa_rng(uint32_t& s) { s = s * 1664525u + 1013904223u; return s; }
static float pa_rand(uint32_t& s) {
    return (float)((pa_rng(s) >> 8) & 0xFFFFFFu) / (float)(1u << 24) - 0.5f;
}
static void pa_fill_token(int rid, int pos, uint16_t* hk, uint16_t* hv, uint16_t* hq, int H, int D) {
    uint32_t s = (uint32_t)(0x9e3779b9u ^ (uint32_t)(rid + 1) * 1000003u
                            ^ (uint32_t)(pos + 1) * 9173u);
    if (!s) {
        s = 1;
    }

    for (int i = 0; i < H * D; ++i) {
        hk[i] = f32_to_f16(pa_rand(s));
    }

    for (int i = 0; i < H * D; ++i) {
        hv[i] = f32_to_f16(pa_rand(s));
    }

    for (int i = 0; i < H * D; ++i) {
        hq[i] = f32_to_f16(pa_rand(s));
    }
}
// host fp32 标准答案：KV 含 [0,pos] 共 pos+1 个 token，Q 取 pos；softmax(QK/√D)·V。
static double pa_oracle_slot(int rid, int pos, const uint16_t* gpu_out, int H, int D) {
    const int T = pos + 1, HD = H * D;
    std::vector<uint16_t> Kv((size_t)T * HD), Vv((size_t)T * HD), hk(HD), hv(HD), hq(HD);
    for (int t = 0; t < T; ++t) {
        pa_fill_token(rid, t, hk.data(), hv.data(), hq.data(), H, D);
        std::memcpy(&Kv[(size_t)t * HD], hk.data(), HD * sizeof(uint16_t));
        std::memcpy(&Vv[(size_t)t * HD], hv.data(), HD * sizeof(uint16_t));
    }

    const float scale = 1.0f / std::sqrt((float)D);
    std::vector<float> sc(T);
    double md = 0.0;
    for (int h = 0; h < H; ++h) {
        float mx = -1e30f;
        for (int t = 0; t < T; ++t) {
            double a = 0.0;
            for (int d = 0; d < D; ++d) {
                a += (double)f16_to_f32(hq[h * D + d]) * f16_to_f32(Kv[(size_t)t * HD + h * D + d]);
            }

            sc[t] = (float)a * scale;
            if (sc[t] > mx) {
                mx = sc[t];
            }
        }

        double sum = 0.0;
        for (int t = 0; t < T; ++t) {
            sc[t] = std::exp(sc[t] - mx);
            sum += sc[t];
        }

        for (int d = 0; d < D; ++d) {
            double w = 0.0;
            for (int t = 0; t < T; ++t) {
                w += (double)sc[t] * f16_to_f32(Vv[(size_t)t * HD + h * D + d]);
            }

            md = std::max(md, (double)std::fabs(f16_to_f32(gpu_out[h * D + d]) - (float)(w / sum)));
        }
    }

    return md;
}

// 可中止的复用屏障：任一 rank setup 失败调 abort()，其余 rank 不再阻塞，直接走清理退出。
struct Barrier {
    std::mutex m; std::condition_variable cv;
    size_t n = 0, gen = 0, total; bool aborted = false;
    explicit Barrier(size_t t) : total(t) {}
    void wait() {
        std::unique_lock<std::mutex> lk(m);
        if (aborted) {
            return;
        }

        size_t g = gen;
        if (++n == total) {
            n = 0;
            ++gen;
            cv.notify_all();
        } else {
            while (!aborted && gen == g) {
                cv.wait(lk);
            }
        }
    }
    void abort() { std::lock_guard<std::mutex> lk(m); aborted = true; cv.notify_all(); }
};

int main() {
    printf("==== Lite-GPU Multi-GPU TP Test (NCCL, CUDA 12.6) ====\n");
    CHECK(liteInit());
    int ndev = 0;
    CHECK(liteDeviceGetCount(&ndev));
    if (ndev < 2) {
        printf("[SKIP] need >=2 GPUs for TP test, only %d visible.\n", ndev);
        return 0;
    }

    const int WORLD = 2, BASE_DEV = 0;
    const int B = 4, H = 8, Hh = H / 2, D = 128, TPB = 16, MBPS = 8, PHYS = 40, NSPLIT = 4;
    const int slen[B] = {17, 3, 64, 40};
    const int HD = H * D, HDh = Hh * D;

    // ---------- host 权威全头 paged 数据（未写槽位填 5.0 反污染）----------
    std::vector<uint16_t> kPool((size_t)PHYS * TPB * HD, f32_to_f16(5.0f));
    std::vector<uint16_t> vPool((size_t)PHYS * TPB * HD, f32_to_f16(5.0f));
    std::vector<uint16_t> q((size_t)B * HD, 0);
    std::vector<int> bt(B * MBPS, 0), sl(B, 0);
    std::vector<uint16_t> hk(HD), hv(HD), hq(HD);
    int next = 0;
    for (int b = 0; b < B; ++b) {
        int nb = (slen[b] + TPB - 1) / TPB;
        for (int k = 0; k < nb; ++k) {
            bt[b * MBPS + k] = next++;
        }

        for (int t = 0; t < slen[b]; ++t) {
            int phys = bt[b * MBPS + (t / TPB)], slotw = t % TPB;
            pa_fill_token(b, t, hk.data(), hv.data(), hq.data(), H, D);
            size_t base = ((size_t)phys * TPB + slotw) * HD;
            std::memcpy(&kPool[base], hk.data(), HD * sizeof(uint16_t));
            std::memcpy(&vPool[base], hv.data(), HD * sizeof(uint16_t));
            if (t == slen[b] - 1) {
                std::memcpy(&q[(size_t)b * HD], hq.data(), HD * sizeof(uint16_t));
            }
        }

        sl[b] = slen[b];
    }
    int usedBlocks = next;

    // ---------- 按 head 维切两半（rank r 取 head [r*Hh, r*Hh+Hh)）----------
    std::vector<uint16_t> kH[WORLD], vH[WORLD], qH[WORLD];
    for (int r = 0; r < WORLD; ++r) {
        kH[r].assign((size_t)PHYS * TPB * HDh, f32_to_f16(5.0f));
        vH[r].assign((size_t)PHYS * TPB * HDh, f32_to_f16(5.0f));
        qH[r].assign((size_t)B * HDh, 0);
        for (int phys = 0; phys < usedBlocks; ++phys) {
            for (int slotw = 0; slotw < TPB; ++slotw) {
                for (int hh = 0; hh < Hh; ++hh) {
                    int h = r * Hh + hh;
                    size_t sFull = ((size_t)(phys * TPB + slotw) * H + h) * D;
                    size_t sHalf = ((size_t)(phys * TPB + slotw) * Hh + hh) * D;
                    std::memcpy(&kH[r][sHalf], &kPool[sFull], D * sizeof(uint16_t));
                    std::memcpy(&vH[r][sHalf], &vPool[sFull], D * sizeof(uint16_t));
                }
            }
        }

        for (int b = 0; b < B; ++b) {
            for (int hh = 0; hh < Hh; ++hh) {
                int h = r * Hh + hh;
                std::memcpy(&qH[r][((size_t)b * Hh + hh) * D],
                            &q[((size_t)b * H + h) * D], D * sizeof(uint16_t));
            }
        }
    }

    // ---------- 建立 NCCL 通信域（内部 primary ctx + 每 rank bootstrap 线程）----------
    LiteNcclGroupConfig gc{WORLD, BASE_DEV};
    LiteNcclGroup group{};
    CHECK(liteNcclGroupCreate(&gc, &group));
    printf("[TP] world=%d on devices [%d,%d], head-shard H=%d -> %d+%d\n",
           WORLD, BASE_DEV, BASE_DEV + 1, H, Hh, Hh);

    Barrier bar(WORLD);
    std::atomic<int> nerr{0};
    std::vector<uint16_t> tpOut((size_t)B * HD, 0);
    std::atomic<double> mdOracle{0.0};
    std::atomic<double> bwDecode{0.0}, bwBulk{0.0};

    auto fail = [&](const char* what) {
        fprintf(stderr, "[TP rank] FAIL: %s\n", what);
        nerr.fetch_add(1);
        bar.abort();
    };

    auto worker = [&](int rank) {
        LiteContext ctx;
        if (liteNcclRankCtx(group, rank, &ctx) != LITE_SUCCESS) {
            fail("rank ctx");
            return;
        }

        liteSetCurrentContext(ctx);
        LiteStream rs;
        if (liteNcclRankStream(group, rank, &rs) != LITE_SUCCESS) {
            fail("rank stream");
            return;
        }

        // 半头算子：所有资源在 current(本卡 primary) context 上分配/加载。
        LitePagedAttnConfig acfg{Hh, D, TPB, B, MBPS, PHYS, NSPLIT};
        LitePagedAttn attn = nullptr;
        CUdeviceptr dGather = 0, dBSend = 0, dBRecv = 0;
        const size_t rowBytes = (size_t)B * Hh * D * sizeof(uint16_t); // 每 rank 半头总字节
        const size_t bulkElems = (1u << 20) / sizeof(uint16_t);      // 524288 fp16 = 1MB
        auto cleanup_resources = [&]() {
            if (dBRecv) {
                cuMemFree(dBRecv);
            }

            if (dBSend) {
                cuMemFree(dBSend);
            }

            if (dGather) {
                cuMemFree(dGather);
            }

            if (attn) {
                litePagedAttnDestroy(attn);
            }
        };
        auto bail = [&](const char* w) { fail(w); cleanup_resources(); };

        if (litePagedAttnCreate(&acfg, &attn) != LITE_SUCCESS) {
            bail("PagedAttn create");
            return;
        }

        if (litePagedAttnCopyKV(attn, kH[rank].data(), vH[rank].data()) != LITE_SUCCESS ||
            litePagedAttnCopyInput(attn, qH[rank].data(), bt.data(), sl.data()) != LITE_SUCCESS) {
            bail("upload half KV/Q");
            return;
        }

        if (cuMemAlloc(&dGather, WORLD * rowBytes) != CUDA_SUCCESS) {
            bail("alloc gather");
            return;
        }

        if (cuMemAlloc(&dBSend, bulkElems * sizeof(uint16_t)) != CUDA_SUCCESS ||
            cuMemAlloc(&dBRecv, WORLD * bulkElems * sizeof(uint16_t)) != CUDA_SUCCESS) {
            bail("alloc bench buf");
            return;
        }

        bar.wait();                       // B0：两 rank setup 完成
        if (nerr.load()) {
            cleanup_resources();
            return;
        }

        // ---- 正确性：半头 kernel（与 all-gather 同 rank 流，流内有序，无需额外 event）----
        if (litePagedAttnLaunchB(attn, B, rs) != LITE_SUCCESS) {
            bail("half launch");
            return;
        }

        bar.wait();                       // B1：两 rank 半头都已提交
        if (nerr.load()) {
            cleanup_resources();
            return;
        }

        if (liteNcclAllGatherFp16(group, rank, litePagedAttnOutPtr(attn),
                                  (void*)dGather, (size_t)B * Hh * D) != LITE_SUCCESS) {
            bail("all-gather correctness");
            return;
        }

        if (liteNcclRankSync(group, rank) != LITE_SUCCESS) {
            bail("rank sync");
            return;
        }

        bar.wait();                       // B2：两 rank gather 完成
        if (nerr.load()) {
            cleanup_resources();
            return;
        }

        if (rank == 0) {
            std::vector<uint16_t> g(WORLD * B * Hh * D);
            if (cuMemcpyDtoH(g.data(), dGather, WORLD * rowBytes) != CUDA_SUCCESS) {
                bail("DtoH gather");
                return;
            }

            // NCCL 输出 [rank][B][Hh][D] -> reshape 为 [B][H][D]（rank 轴并入 head 轴）。
            for (int b = 0; b < B; ++b) {
                for (int rr = 0; rr < WORLD; ++rr) {
                    for (int hh = 0; hh < Hh; ++hh) {
                        for (int d = 0; d < D; ++d) {
                            size_t gi = ((size_t)(rr * B + b) * Hh + hh) * D + d;
                            size_t oi = ((size_t)b * H + rr * Hh + hh) * D + d;
                            tpOut[oi] = g[gi];
                        }
                    }
                }
            }

            double md = 0.0;
            for (int b = 0; b < B; ++b) {
                md = std::max(md, pa_oracle_slot(b, slen[b] - 1, &tpOut[(size_t)b * HD], H, D));
            }

            mdOracle.store(md);
        }

        bar.wait();                       // B3：rank0 校验完成
        if (nerr.load()) {
            cleanup_resources();
            return;
        }

        // ---- 通信微基准：两档，计时仅 rank0 用 timing event ----
        LiteEvent e0 = nullptr, e1 = nullptr;
        if (rank == 0 &&
            (liteEventCreateWithTiming(&e0) != LITE_SUCCESS ||
             liteEventCreateWithTiming(&e1) != LITE_SUCCESS)) {
            bail("timing event");
            return;
        }

        auto bench = [&](size_t sendElems, const void* sendb, void* recvb,
                         int iters, std::atomic<double>& outBw) -> LiteError {
            bar.wait();                   // 对齐开始
            if (rank == 0) {
                liteEventRecord(e0, rs);
            }

            for (int it = 0; it < iters; ++it) {
                if (liteNcclAllGatherFp16(group, rank, sendb, recvb, sendElems) != LITE_SUCCESS) {
                    fail("all-gather bench");
                    return LITE_ERROR_DRIVER_API;
                }
            }

            if (rank == 0) {
                liteEventRecord(e1, rs);
                if (liteStreamSynchronize(rs) != LITE_SUCCESS) {
                    fail("sync bench");
                    return LITE_ERROR_DRIVER_API;
                }

                float ms = 0.f; liteEventElapsedTime(&ms, e0, e1);
                double bytes = (double)WORLD * sendElems * sizeof(uint16_t) * iters;
                outBw.store(bytes / (ms / 1e3) / 1e9);   // 逻辑汇聚吞吐 GB/s
            } else if (liteNcclRankSync(group, rank) != LITE_SUCCESS) {
                fail("sync bench2");
                return LITE_ERROR_DRIVER_API;
            }

            bar.wait();                   // 对齐结束
            return LITE_SUCCESS;
        };
        LiteError e = bench((size_t)B * Hh * D, litePagedAttnOutPtr(attn), (void*)dGather,
                            200, bwDecode);   // decode 真实形状（~4KB/rank，延迟主导）
        if (e == LITE_SUCCESS) {
            e = bench(bulkElems, (void*)dBSend, (void*)dBRecv, 50, bwBulk);  // 1MB/rank 链路档
        }

        if (rank == 0) {
            liteEventDestroy(e0);
            liteEventDestroy(e1);
        }

        cleanup_resources();
        if (e != LITE_SUCCESS) {
            return;
        }
    };

    std::vector<std::thread> th;
    for (int r = 0; r < WORLD; ++r) {
        th.emplace_back(worker, r);
    }

    for (auto& t : th) {
        t.join();
    }

    int failures = nerr.load();
    double md = mdOracle.load();
    printf("[TP] all-gather head vs fp32 oracle : max_diff=%.2e\n", md);
    printf("[TP] all-gather bandwidth (logical aggregate):\n");
    printf("       decode-shape %d fp16/rank : %.3f GB/s (small msg, latency-bound)\n",
           B * Hh * D, bwDecode.load());
    printf("       bulk 1MB/rank             : %.3f GB/s (near-link, PCIe/no-NVLink expected)\n",
           bwBulk.load());
    fflush(stdout);

    liteNcclGroupDestroy(group);

    CHECK_TRUE(failures == 0, "no rank failure during multi-GPU TP test");
    CHECK_TRUE(md < 2e-2, "multi-GPU TP all-gather output matches fp32 oracle");
    printf("[PASS] Tensor-Parallel head-shard over 2 physical GPUs + NCCL all-gather\n");
    return 0;
}
