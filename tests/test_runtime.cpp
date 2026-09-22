#include "lite_gpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <vector>
#include <thread>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// =====================================================================
// Lite-GPU Runtime 测试套件（单文件；main 按序执行，全部通过打 ALL TESTS COMPLETE）
//
// 结构总览（按 main() 中的执行顺序）：
//   Unit3   : MemoryPool 页级 alloc/free，占满后触发 OOM
//   Unit5   : 多线程 THREAD_LOCAL capture + context 跨线程绑定
//   Unit7   : MEMCPY 图节点 get/set params（含热更新；MEMSET 节点同构，留本代表）
//   Unit10  : 多流 fork/join capture（单图跨流 event 依赖 + UNJOINED 负对照）
//   Unit10b : 图内多流扇出真并行（SM ∥ CE overlap，parallel/serial 严格对照）
//   Bench1  : launch overhead（eager vs graph，含背压的 wall/host 双口径）
//   Bench1b : native 纯提交成本（无背压，性能数字来源）
//   Bench2  : 热更新机制对比（node param / func-swap / ExecUpdate / 重捕获）
//   Bench5  : 真实 cublas GEMM 谱系（decode host-bound -> prefill GPU-bound）
//   Unit15  : scheduler 抢占/重算（vLLM 式 KV 超卖，纯 host 状态机）
//   Unit16  : prefix caching（块哈希共享）+ COW + refcnt（host 机制 + 真数据 COW）
//   Unit17  : INT8 KV 量化：正确性/精度代价/显存收益
//   Bench8  : 端到端 PagedAttention continuous batching（对 fp32 oracle 对拍）
//   Unit13  : batch 分桶多图 + 运行时选桶（vLLM capture sizes）
//   Unit14  : TP 半头切分 + SimComm all-gather 拼头（单卡模拟双 rank）
//   Profiler: CUPTI 归因段（SM∩CE 交叠 + hotspot），不用于性能结论
//
// =====================================================================
extern "C" __global__ void lite_dummy_kernel(int*, int);
extern "C" __global__ void lite_dummy_kernel_v2(int*, int);
extern "C" CUfunction lite_get_dummy_kernel_func(void);
extern "C" CUfunction lite_get_dummy_kernel_v2_func(void);

static constexpr size_t MEM_PAGE_SIZE      = 8U * 1024U * 1024U;
static constexpr size_t MEM_POOL_PAGE_CNT  = 12U;
static constexpr size_t COPY_BYTES         = 1U * 1024U * 1024U;
static constexpr int    HOT_UPDATE_NODES   = 200;

#define CHECK(call)                                                     \
    do {                                                                \
        LiteError err_ = (call);                                        \
        if (err_ != LITE_SUCCESS) {                                     \
            fprintf(stderr, "CHECK FAILED at %s:%d err=%d\n",           \
                    __FILE__, __LINE__, (int)err_);                     \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

static void CHECK_EQ(bool cond, const char* msg) {
    if (!cond) {
        fprintf(stderr, "CHECK_EQ FAILED: %s\n", msg);
        exit(EXIT_FAILURE);
    }
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

struct Stats {
    double mean, p50, p90, p99, min, max, sd;
};

static Stats compute_stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    double sum = 0;
    for (double x : v) {
        sum += x;
    }

    double mean = n ? sum / n : 0;
    double var = 0;
    for (double x : v) {
        var += (x - mean) * (x - mean);
    }

    double sd = n ? std::sqrt(var / n) : 0;
    auto pct = [&](double p) -> double {
        if (n == 0) {
            return 0;
        }

        return v[(size_t)(p / 100.0 * (n - 1))];
    };
    return { mean, pct(50), pct(90), pct(99), v.front(), v.back(), sd };
}

static void print_stats(const char* label, const std::vector<double>& v, const char* unit = "ms") {
    Stats s = compute_stats(v);
    printf("  %-38s mean=%8.4f  p50=%8.4f  p90=%8.4f  p99=%8.4f  sd=%6.4f %s\n",
           label, s.mean, s.p50, s.p90, s.p99, s.sd, unit);
}

// ---------- 读写辅助 ----------
static void read_bytes(LiteStream s, void* dptr, void* host, size_t n) {
    CHECK(liteMemcpyDtoHAsync(host, dptr, n, s));
    CHECK(liteStreamSynchronize(s));
}

static void run_unit_tests() {
    printf("\n========== Run Unit Tests ==========\n");

    CHECK_EQ(liteInit() == LITE_SUCCESS, "liteInit idempotent");
    CHECK(liteMemPoolInit(MEM_PAGE_SIZE, MEM_POOL_PAGE_CNT));

    LiteStream stream{};
    CHECK(liteStreamCreate(&stream));

    // -------- Unit3: Memory pool alloc / free / OOM --------
    // 把池中全部页占满后再分配，必须返回 LITE_ERROR_OUT_OF_MEMORY（不允许裸越界）；
    // 随后逐页释放。验完池的容量记账与 fail-fast 行为。
    {
        void* pages[MEM_POOL_PAGE_CNT] = {nullptr};
        for (size_t i = 0; i < MEM_POOL_PAGE_CNT; ++i) {
            CHECK(liteMemAllocPage(&pages[i], stream));
        }

        void* extra{}; LiteError err = liteMemAllocPage(&extra, stream);
        CHECK_EQ(err == LITE_ERROR_OUT_OF_MEMORY, "memory pool oom");

        for (size_t i = 0; i < MEM_POOL_PAGE_CNT; ++i) {
            CHECK(liteMemFreePage(pages[i], stream));
        }
        printf("[Unit3] MemoryPool alloc/free/OOM test pass\n");
    }

    // -------- Unit5: 多线程 THREAD_LOCAL capture + Context Binding --------
    // 子线程 liteSetCurrentContext 绑回主 context，独立建流并 THREAD_LOCAL 模式
    // capture/instantiate/launch；主线程同时做 memset——验证 capture 状态按线程隔离，
    // 两线程互不串扰、全链路成功即 pass。
    {
        void* ptrA{}, *ptrT{};
        CHECK(liteMemAllocPage(&ptrA, stream));
        CHECK(liteMemAllocPage(&ptrT, stream));

        LiteContext main_ctx{};
        CHECK(liteGetContext(&main_ctx));

        std::atomic<int> thread_ok{0};
        std::thread th([&]() {
            CHECK(liteSetCurrentContext(main_ctx));
            LiteStream s_th{}; CHECK(liteStreamCreate(&s_th));

            LiteGraph g_th{};
            CHECK(liteStreamBeginCaptureEx(s_th, LITE_CAPTURE_THREAD_LOCAL));
            for (int k = 0; k < 10; ++k) {
                CHECK(liteMemsetAsync(ptrT, COPY_BYTES, s_th));
            }
            CHECK(liteStreamEndCapture(s_th, &g_th));

            LiteGraphExec ge_th{};
            CHECK(liteGraphInstantiate(g_th, &ge_th));
            CHECK(liteGraphLaunch(ge_th, s_th));
            CHECK(liteStreamSynchronize(s_th));

            CHECK(liteGraphExecDestroy(ge_th));
            CHECK(liteGraphDestroy(g_th));
            CHECK(liteStreamDestroy(s_th));
            thread_ok.store(1);
        });

        CHECK(liteMemsetAsync(ptrA, COPY_BYTES, stream));
        CHECK(liteStreamSynchronize(stream));
        th.join();
        CHECK_EQ(thread_ok.load() == 1, "Multi-thread CUDA context binding failed");
        CHECK(liteMemFreePage(ptrA, stream));
        CHECK(liteMemFreePage(ptrT, stream));
        printf("[Unit5] Multi-thread THREAD_LOCAL capture test pass\n");
    }

    // -------- Unit7: MEMCPY 节点 --------
    // capture 一次 D2D 得到单 MEMCPY 节点图：校验 node count / 节点类型 / dst-src-width
    // 参数回读；launch 后数据正确。再热更新节点参数（dst/src 换新缓冲），重放后
    // 新 dst 写入 0x55 且旧 dst 不被触碰 —— 覆盖 memcpy 节点的 get/set params 全套。
    {
        void* ptrA{}, *ptrB{}, *ptrA2{}, *ptrB2{};
        CHECK(liteMemAllocPage(&ptrA, stream));
        CHECK(liteMemAllocPage(&ptrB, stream));
        CHECK(liteMemAllocPage(&ptrA2, stream));
        CHECK(liteMemAllocPage(&ptrB2, stream));

        std::vector<unsigned char> hA(COPY_BYTES, 0xAB), hB(COPY_BYTES, 0x00);
        CHECK(liteMemcpyHtoDAsync(ptrA, hA.data(), COPY_BYTES, stream));
        CHECK(liteStreamSynchronize(stream));

        LiteGraph graph{};
        CHECK(liteStreamBeginCapture(stream));
        CHECK(liteMemcpyDtoDAsync(ptrB, ptrA, COPY_BYTES, stream));
        CHECK(liteStreamEndCapture(stream, &graph));

        size_t node_cnt = 0; CHECK(liteGraphGetNodeCount(graph, &node_cnt));
        CHECK_EQ(node_cnt == 1U, "memcpy graph has 1 node");

        LiteGraphNode node{}; CHECK(liteGraphGetNodeByIndex(graph, 0, &node));
        LiteGraphNodeType nt;
        CHECK(liteGraphNodeGetType(node, &nt));
        CHECK_EQ(nt == LITE_GRAPH_NODE_MEMCPY, "node type is MEMCPY");

        uint64_t dst=0, src=0; size_t width=0; uint32_t hgt=0, dep=0;
        CHECK(liteGraphMemcpyNodeGetInfo(node, &dst, &src, &width, &hgt, &dep));
        CHECK_EQ(dst == (uint64_t)ptrB && src == (uint64_t)ptrA, "memcpy dst/src");
        CHECK_EQ(width == COPY_BYTES && hgt == 1 && dep == 1, "memcpy dims");

        LiteGraphExec exec{};
        CHECK(liteGraphInstantiate(graph, &exec));
        CHECK(liteGraphLaunch(exec, stream));
        CHECK(liteStreamSynchronize(stream));
        read_bytes(stream, ptrB, hB.data(), COPY_BYTES);
        CHECK_EQ(std::memcmp(hB.data(), hA.data(), COPY_BYTES) == 0, "memcpy result correct");

        // 热更新节点参数：改 dst/src 指向新缓冲
        std::vector<unsigned char> hA2(COPY_BYTES, 0x55), hB2(COPY_BYTES, 0x00), hB_old(COPY_BYTES, 0x00);
        CHECK(liteMemcpyHtoDAsync(ptrA2, hA2.data(), COPY_BYTES, stream));
        CHECK(liteStreamSynchronize(stream));
        CHECK(liteGraphExecMemcpyNodeSetParams(exec, node, ptrB2, ptrA2));
        CHECK(liteGraphLaunch(exec, stream));
        CHECK(liteStreamSynchronize(stream));
        read_bytes(stream, ptrB2, hB2.data(), COPY_BYTES);
        CHECK_EQ(std::memcmp(hB2.data(), hA2.data(), COPY_BYTES) == 0, "updated memcpy dst correct");
        read_bytes(stream, ptrB, hB_old.data(), COPY_BYTES);
        CHECK_EQ(std::memcmp(hB_old.data(), hA.data(), COPY_BYTES) == 0, "old dst untouched");

        CHECK(liteGraphNodeDestroy(node));
        CHECK(liteGraphExecDestroy(exec));
        CHECK(liteGraphDestroy(graph));
        CHECK(liteMemFreePage(ptrA, stream));
        CHECK(liteMemFreePage(ptrB, stream));
        CHECK(liteMemFreePage(ptrA2, stream));
        CHECK(liteMemFreePage(ptrB2, stream));
        printf("[Unit7] MEMCPY node get/set params test pass\n");
    }

    // -------- Unit10: 多流 Fork/Join Capture（RELAXED，单图含跨流 event 依赖）--------
    // 考点：用 event record/wait 在一次 stream capture 内把两条流 fork 后再 join 回 root，
    //   endCapture 到 root 流得到【一张图】。本用例不向 runtime 库新增任何 API：
    //   图内省（cuGraphGetNodes/cuGraphGetEdges）与负对照收尾直接用 Driver API，
    //   与本文件其它 benchmark 直接用 cuMemAlloc/cublas 的风格一致。
    // 跨流成立的稳定证据（不依赖驱动是否保留 event 节点，CUDA12 常把图内 event
    //   融合成边/信号量节点）：
    //     - worker 流的 MEMCPY 节点在 DAG 中同时有入边(fork)与出边(join)；
    //     - replay 后数据顺序被强制（dst=111 而非 222）；
    //     - 负对照：不 join 时 endCapture 返回 CUDA_ERROR_STREAM_CAPTURE_UNJOINED。
    // 依赖链（DAG）：
    //   sCompute: k(写 src=111) --rec forkEv            wait joinEv--> k(写 src=222)
    //                                        \                     /
    //   sCopy:                                 wait forkEv-> memcpy(src->dst) --rec joinEv
    {
        // 不透明句柄首成员即原生 CUstream / CUgraph（见 lite_gpu_internal.h），
        // 与 Bench5/6 取 raw stream 的写法一致，仅在测试侧使用。
        auto raw_stream = [](LiteStream s) -> CUstream {
            return *reinterpret_cast<CUstream*>(s);
        };

        LiteStream sCompute{}, sCopy{};
        CHECK(liteStreamCreate(&sCompute));
        CHECK(liteStreamCreate(&sCopy));
        CUstream rComp = raw_stream(sCompute);
        CUstream rCopy = raw_stream(sCopy);

        CUdeviceptr dSrc = 0, dDst = 0;
        CHECK_EQ(cuMemAlloc(&dSrc, sizeof(int)) == CUDA_SUCCESS, "fj alloc src");
        CHECK_EQ(cuMemAlloc(&dDst, sizeof(int)) == CUDA_SUCCESS, "fj alloc dst");
        void* pSrc = reinterpret_cast<void*>(dSrc);
        void* pDst = reinterpret_cast<void*>(dDst);

        LiteEvent evFork{}, evJoin{};
        CHECK(liteEventCreateForSync(&evFork));
        CHECK(liteEventCreateForSync(&evJoin));

        // ===== 主路径：fork -> worker(copy) -> join，构成合法单图 =====
        LiteGraph graph{};
        CHECK(liteStreamBeginCaptureEx(sCompute, LITE_CAPTURE_RELAXED));
        CHECK(liteLaunchTinyKernel(pSrc, 111, sCompute));
        CHECK(liteEventRecord(evFork, sCompute));
        CHECK(liteStreamWaitEvent(sCopy, evFork));
        CHECK(liteMemcpyDtoDAsync(pDst, pSrc, sizeof(int), sCopy));
        CHECK(liteEventRecord(evJoin, sCopy));
        CHECK(liteStreamWaitEvent(sCompute, evJoin));
        CHECK(liteLaunchTinyKernel(pSrc, 222, sCompute));
        CHECK(liteStreamEndCapture(sCompute, &graph));

        // ---- 图结构证据（直接 Driver API）：节点类型 + 依赖边 + memcpy 节点入/出边 ----
        CUgraph cg = *reinterpret_cast<CUgraph*>(graph);
        size_t nN = 0;
        CHECK_EQ(cuGraphGetNodes(cg, nullptr, &nN) == CUDA_SUCCESS, "fj cuGraphGetNodes count");
        std::vector<CUgraphNode> ns(nN);
        if (nN) {
            CHECK_EQ(cuGraphGetNodes(cg, ns.data(), &nN) == CUDA_SUCCESS, "fj cuGraphGetNodes");
        }
        size_t gk = 0, gm = 0, ge = 0, go = 0;
        std::vector<char> is_cp(nN, 0);
        for (size_t i = 0; i < nN; ++i) {
            CUgraphNodeType ty;
            CHECK_EQ(cuGraphNodeGetType(ns[i], &ty) == CUDA_SUCCESS, "fj node type");
            if (ty == CU_GRAPH_NODE_TYPE_KERNEL) {
                ++gk;
            } else if (ty == CU_GRAPH_NODE_TYPE_MEMCPY) {
                ++gm;
                is_cp[i] = 1;
            } else if (ty == CU_GRAPH_NODE_TYPE_EVENT_RECORD) {
                ++ge;
            } else {
                ++go;
            }
        }
        size_t ne = 0;
        CHECK_EQ(cuGraphGetEdges(cg, nullptr, nullptr, &ne) == CUDA_SUCCESS, "fj cuGraphGetEdges count");
        int cp_in = 0, cp_out = 0;
        if (ne) {
            std::vector<CUgraphNode> ef(ne), et(ne);
            CHECK_EQ(cuGraphGetEdges(cg, ef.data(), et.data(), &ne) == CUDA_SUCCESS, "fj cuGraphGetEdges");
            for (size_t e = 0; e < ne; ++e) {
                for (size_t i = 0; i < nN; ++i) {
                    if (is_cp[i]) {
                        if (et[e] == ns[i]) {
                            cp_in  = 1;  // 有边指向 memcpy（fork 依赖）
                        }

                        if (ef[e] == ns[i]) {
                            cp_out = 1;  // 有边从 memcpy 出发（join 依赖）
                        }
                    }
                }
            }
        }
        printf("[Unit10] nodes: kernel=%zu memcpy=%zu event_record=%zu other=%zu (V=%zu), edges=%zu\n",
               gk, gm, ge, go, nN, ne);
        printf("[Unit10] worker MEMCPY node in-edge=%d out-edge=%d (expect 1/1)\n", cp_in, cp_out);
        CHECK_EQ(gk >= 2, "fj: >=2 kernel nodes (pre-fork & post-join)");
        CHECK_EQ(gm >= 1, "fj: >=1 memcpy node captured on WORKER stream");
        CHECK_EQ(cp_in == 1 && cp_out == 1,
                 "fj: worker memcpy sits BETWEEN fork(in-edge) and join(out-edge) in DAG");
        CHECK_EQ(ne >= nN - 1, "fj: graph is a connected DAG (edges >= V-1)");
        printf("[Unit10] note: visible EVENT_RECORD=%zu (may be 0 if fused into edges/semaphore nodes)\n", ge);

        LiteGraphExec exec{};
        CHECK(liteGraphInstantiate(graph, &exec));
        CHECK(liteGraphLaunch(exec, sCompute));
        CHECK(liteStreamSynchronize(sCompute));

        int hSrc = -1, hDst = -1;
        read_bytes(sCompute, pSrc, &hSrc, sizeof(int));
        read_bytes(sCompute, pDst, &hDst, sizeof(int));
        printf("[Unit10] after replay: src=%d (expect 222), dst=%d (expect 111)\n", hSrc, hDst);
        // src 最终=222：join 后的 kernel 最后执行；dst=111（不是 222）：copy 严格排在
        //   post-join kernel 之前；若 fork/join 依赖失效，copy 可能读到 222 -> 断言失败。
        CHECK_EQ(hSrc == 222, "fj: src final == 222 (post-join kernel ran last)");
        CHECK_EQ(hDst == 111, "fj: dst == 111 (copy saw pre-join value => ordering enforced)");

        CHECK(liteGraphExecDestroy(exec));
        CHECK(liteGraphDestroy(graph));

        // ===== 负对照：fork 到 worker 但【不 join】，endCapture 必须返回 UNJOINED =====
        // 直接用 Driver API 调 cuStreamEndCapture 以拿到精确错误码（库封装会压成通用码）。
        LiteEvent evFork2{};
        CHECK(liteEventCreateForSync(&evFork2));
        CHECK(liteStreamBeginCaptureEx(sCompute, LITE_CAPTURE_RELAXED));
        CHECK(liteLaunchTinyKernel(pSrc, 1, sCompute));
        CHECK(liteEventRecord(evFork2, sCompute));
        CHECK(liteStreamWaitEvent(sCopy, evFork2));
        CHECK(liteMemcpyDtoDAsync(pDst, pSrc, sizeof(int), sCopy));  // worker 分叉后未 join
        CUgraph cgBad = nullptr;
        CUresult rj = cuStreamEndCapture(rComp, &cgBad);
        const char* rj_name = nullptr; cuGetErrorName(rj, &rj_name);
        printf("[Unit10] unjoined cuStreamEndCapture -> %d (%s)\n", (int)rj,
               rj_name ? rj_name : "?");
        CHECK_EQ(rj == CUDA_ERROR_STREAM_CAPTURE_UNJOINED, "fj: unjoined fork MUST yield STREAM_CAPTURE_UNJOINED");
        CHECK_EQ(cgBad == nullptr, "fj: failed endCapture must not produce a graph");

        cuStreamSynchronize(rComp);
        CUgraph cgDrop = nullptr;
        CUresult rd = cuStreamEndCapture(rComp, &cgDrop);
        (void)rd;
        if (cgDrop) {
            cuGraphDestroy(cgDrop);
        }

        CHECK(liteEventDestroy(evFork2));
        CHECK(liteEventDestroy(evFork));
        CHECK(liteEventDestroy(evJoin));
        cuMemFree(dSrc);
        cuMemFree(dDst);
        CHECK(liteStreamDestroy(sCopy));
        CHECK(liteStreamDestroy(sCompute));
        printf("[Unit10] Multi-Stream Fork/Join capture (single graph, cross-stream event edges) pass\n");
    }

    // -------- Unit10b: Graph 内多流【扇出并行】（fan-out -> overlap -> join）--------
    // Unit10 证明了多流 fork/join 的"依赖正确性"，但那张图是 k1->copy->k2 串行链、
    //   没有无依赖的并发分支。本用例证明 graph 内多流【真并行】：
    //   root(sCompute,SM 长 kernel) 与 worker(sCopy, H2D pinned 走 CE) 两条分支
    //   fork 后互不依赖、同时执行，再 join 回 root。
    // 严格控制变量：并行图与串行图节点集合完全相同，只移动 fork event 的 record 位置：
    //   - parallel : forkEv 在长 compute【之前】record -> copy 分支与 compute 无依赖，并行；
    //   - serial   : forkEv 在长 compute【之后】record -> copy 必须等 compute，纯串行。
    // 判据：T_parallel ≈ max(Tc,Tm)，T_serial ≈ Tc+Tm，二者之差即 graph 内 SM∩CE overlap。
    {
        LiteStream sCompute{}, sCopy{};
        CHECK(liteStreamCreate(&sCompute));
        CHECK(liteStreamCreate(&sCopy));

        const int        LOOPS = 10000000;      // ~15ms 级 SM 长 kernel（与 profiler 段负载同量级）
        const size_t     CHUNK = 16ull << 20;   // 单次 H2D 16MB（走 copy engine）
        const int        KCOPY = 8;             // worker 分支连续 8 次 H2D（约 128MB，CE 长任务）

        CUdeviceptr dWork = 0, dCopyDst = 0;
        CHECK_EQ(cuMemAlloc(&dWork, 1 << 20) == CUDA_SUCCESS, "fj10b alloc work");
        CHECK_EQ(cuMemAlloc(&dCopyDst, CHUNK) == CUDA_SUCCESS, "fj10b alloc copy dst");
        void* hPinned = nullptr;
        CHECK_EQ(cudaMallocHost(&hPinned, CHUNK) == cudaSuccess, "fj10b pinned alloc");
        memset(hPinned, 0x7, CHUNK);
        void* pWork = reinterpret_cast<void*>(dWork);
        void* pDst  = reinterpret_cast<void*>(dCopyDst);

        LiteEvent evTa{}, evTb{};
        CHECK(liteEventCreateWithTiming(&evTa));
        CHECK(liteEventCreateWithTiming(&evTb));

        const int WARM = 5, REP = 30;
        auto time_ms = [&]() -> float {
            float ms = 0.f;
            CHECK(liteEventElapsedTime(&ms, evTa, evTb));
            return ms;
        };

        // ---- 单分支基线（eager）：纯 SM compute 时长 Tc、纯 CE copy ----
        std::vector<double> vc, vm;
        for (int w = 0; w < WARM; ++w) {
            liteLaunchDummyKernelHeavy(pWork, LOOPS, 1, sCompute);
            CHECK(liteStreamSynchronize(sCompute));
        }
        for (int r = 0; r < REP; ++r) {
            CHECK(liteEventRecord(evTa, sCompute));
            liteLaunchDummyKernelHeavy(pWork, LOOPS, 1, sCompute);
            CHECK(liteEventRecord(evTb, sCompute));
            CHECK(liteStreamSynchronize(sCompute));
            vc.push_back(time_ms());
        }
        for (int w = 0; w < WARM; ++w) {
            for (int k = 0; k < KCOPY; ++k) {
                CHECK(liteMemcpyHtoDAsync(pDst, hPinned, CHUNK, sCopy));
            }

            CHECK(liteStreamSynchronize(sCopy));
        }
        for (int r = 0; r < REP; ++r) {
            CHECK(liteEventRecord(evTa, sCopy));
            for (int k = 0; k < KCOPY; ++k) {
                CHECK(liteMemcpyHtoDAsync(pDst, hPinned, CHUNK, sCopy));
            }

            CHECK(liteEventRecord(evTb, sCopy));
            CHECK(liteStreamSynchronize(sCopy));
            vm.push_back(time_ms());
        }
        double Tc = compute_stats(vc).p50;
        double Tm = compute_stats(vm).p50;

        // ---- 构图：parallel=true 扇出并行；false 强制串行（节点集合相同）----
        auto build_graph = [&](bool parallel, LiteEvent evFork, LiteEvent evJoin,
                               LiteGraph* gOut, LiteGraphExec* eOut) {
            CHECK(liteStreamBeginCaptureEx(sCompute, LITE_CAPTURE_RELAXED));
            if (parallel) {
                // fork 点在 compute 之前：copy 分支只需等 fork，即可与 compute 并行
                CHECK(liteEventRecord(evFork, sCompute));
                CHECK(liteStreamWaitEvent(sCopy, evFork));
                for (int k = 0; k < KCOPY; ++k) {
                    CHECK(liteMemcpyHtoDAsync(pDst, hPinned, CHUNK, sCopy));
                }

                CHECK(liteEventRecord(evJoin, sCopy));
                liteLaunchDummyKernelHeavy(pWork, LOOPS, 1, sCompute);  // root 长 kernel
                CHECK(liteStreamWaitEvent(sCompute, evJoin));           // join：root 等 copy 完
            } else {
                // fork 点在 compute 之后：copy 必须等 compute，纯串行
                liteLaunchDummyKernelHeavy(pWork, LOOPS, 1, sCompute);
                CHECK(liteEventRecord(evFork, sCompute));
                CHECK(liteStreamWaitEvent(sCopy, evFork));
                for (int k = 0; k < KCOPY; ++k) {
                    CHECK(liteMemcpyHtoDAsync(pDst, hPinned, CHUNK, sCopy));
                }

                CHECK(liteEventRecord(evJoin, sCopy));
                CHECK(liteStreamWaitEvent(sCompute, evJoin));
            }
            CHECK(liteStreamEndCapture(sCompute, gOut));
            CHECK(liteGraphInstantiate(*gOut, eOut));
        };

        auto time_graph = [&](LiteGraphExec exec) -> double {
            for (int w = 0; w < WARM; ++w) {
                CHECK(liteGraphLaunch(exec, sCompute));
                CHECK(liteStreamSynchronize(sCompute));
            }
            std::vector<double> v;
            for (int r = 0; r < REP; ++r) {
                CHECK(liteEventRecord(evTa, sCompute));
                CHECK(liteGraphLaunch(exec, sCompute));
                CHECK(liteEventRecord(evTb, sCompute));
                CHECK(liteStreamSynchronize(sCompute));
                v.push_back(time_ms());
            }
            return compute_stats(v).p50;
        };

        LiteEvent evFP{}, evJP{}, evFS{}, evJS{};
        CHECK(liteEventCreateForSync(&evFP));
        CHECK(liteEventCreateForSync(&evJP));
        CHECK(liteEventCreateForSync(&evFS));
        CHECK(liteEventCreateForSync(&evJS));

        LiteGraph gP{}; LiteGraphExec eP{};
        build_graph(true, evFP, evJP, &gP, &eP);
        double Tpar = time_graph(eP);

        LiteGraph gS{}; LiteGraphExec eS{};
        build_graph(false, evFS, evJS, &gS, &eS);
        double Tser = time_graph(eS);

        double overlap = Tser - Tpar;
        double ratio = (Tc + Tm) > 0 ? 100.0 * overlap / (Tc + Tm) : 0.0;
        printf("[Unit10b] eager p50: compute(SM)=%.3f ms  copy-branch(CE)=%.3f ms  sum=%.3f ms\n",
               Tc, Tm, Tc + Tm);
        printf("[Unit10b] graph fan-out PARALLEL p50=%.3f ms | forced SERIAL p50=%.3f ms\n", Tpar, Tser);
        printf("[Unit10b] in-graph overlap = %.3f ms (%.1f%% of Tc+Tm); parallel≈max=%.3f, serial≈sum=%.3f\n",
               overlap, ratio, std::max(Tc, Tm), Tc + Tm);

        // 并行必须严格快于串行，且重叠量至少藏起"较短分支"的 1/3（留 WSL/噪声裕度）
        CHECK_EQ(Tpar < Tser, "fj10b: fan-out graph must be faster than forced-serial graph");
        CHECK_EQ(overlap >= 0.34 * std::min(Tc, Tm),
                 "fj10b: SM/CE branches overlap inside one graph (overlap >= 1/3 of shorter branch)");
        if (std::min(Tc, Tm) < 1.0) {
            printf("[Unit10b] WARN: shorter branch <1ms, overlap 证据偏弱，可调大 LOOPS/CHUNK/KCOPY\n");
        }

        CHECK(liteGraphExecDestroy(eP));
        CHECK(liteGraphDestroy(gP));
        CHECK(liteGraphExecDestroy(eS));
        CHECK(liteGraphDestroy(gS));
        CHECK(liteEventDestroy(evFP));
        CHECK(liteEventDestroy(evJP));
        CHECK(liteEventDestroy(evFS));
        CHECK(liteEventDestroy(evJS));
        CHECK(liteEventDestroy(evTa));
        CHECK(liteEventDestroy(evTb));
        cudaFreeHost(hPinned);
        cuMemFree(dCopyDst);
        cuMemFree(dWork);
        CHECK(liteStreamDestroy(sCopy));
        CHECK(liteStreamDestroy(sCompute));
        printf("[Unit10b] In-Graph multi-stream fan-out parallelism (SM ∥ CE) pass\n");
    }

    CHECK(liteStreamDestroy(stream));
    CHECK(liteMemPoolDestroy());
    printf("========== All Unit Tests DONE ==========\n\n");
}

// ============================ Benchmark：Launch Overhead ============================
// 口径说明：以"单次 decode-step 墙钟"为目标。对 eager 与 graph 都用 GPU 事件测
//   per-step 墙钟（含 launch + GPU 执行 + sync），另单独测 host 提交耗时，
//   两者分开展示，避免把队列背压误标成"launch overhead"。
static void run_bench_launch_overhead() {
    printf("\n========== Bench1: Launch Overhead (Host-Bound Decode Simulation) ==========\n");
    LiteStream stream{}; CHECK(liteStreamCreate(&stream));

    const int NUM_KERNELS = 5000;
    const int WARMUP = 10, REPEAT = 30;

    LiteEvent ev_t0{}, ev_t1{};
    CHECK(liteEventCreateWithTiming(&ev_t0));
    CHECK(liteEventCreateWithTiming(&ev_t1));

    std::vector<double> eager_wall, eager_host, graph_wall, graph_host;

    // ---------- Eager ----------
    for (int w = 0; w < WARMUP; ++w) {
        for (int k = 0; k < NUM_KERNELS; ++k) {
            CHECK(liteLaunchEmptyKernel(stream));
        }
        CHECK(liteStreamSynchronize(stream));
    }
    for (int r = 0; r < REPEAT; ++r) {
        double h0 = now_ms();
        CHECK(liteEventRecord(ev_t0, stream));
        for (int k = 0; k < NUM_KERNELS; ++k) {
            CHECK(liteLaunchEmptyKernel(stream));
        }
        CHECK(liteEventRecord(ev_t1, stream));
        double h1 = now_ms();
        CHECK(liteStreamSynchronize(stream));
        float ms = 0; CHECK(liteEventElapsedTime(&ms, ev_t0, ev_t1));
        eager_wall.push_back(ms); eager_host.push_back(h1 - h0);
    }

    // ---------- Graph Replay ----------
    LiteGraph graph{};
    CHECK(liteStreamBeginCaptureEx(stream, LITE_CAPTURE_RELAXED));
    for (int k = 0; k < NUM_KERNELS; ++k) {
        CHECK(liteLaunchEmptyKernel(stream));
    }

    CHECK(liteStreamEndCapture(stream, &graph));
    LiteGraphExec exec{};
    CHECK(liteGraphInstantiate(graph, &exec));

    for (int w = 0; w < WARMUP; ++w) {
        CHECK(liteGraphLaunch(exec, stream));
        CHECK(liteStreamSynchronize(stream));
    }
    for (int r = 0; r < REPEAT; ++r) {
        double h0 = now_ms();
        CHECK(liteEventRecord(ev_t0, stream));
        CHECK(liteGraphLaunch(exec, stream));
        CHECK(liteEventRecord(ev_t1, stream));
        double h1 = now_ms();
        CHECK(liteStreamSynchronize(stream));
        float ms = 0; CHECK(liteEventElapsedTime(&ms, ev_t0, ev_t1));
        graph_wall.push_back(ms); graph_host.push_back(h1 - h0);
    }

    double ew = compute_stats(eager_wall).p50;
    double eh = compute_stats(eager_host).p50;
    double gw = compute_stats(graph_wall).p50;
    double gh = compute_stats(graph_host).p50;

    printf("\n[Bench1] %d empty kernels/iter (Decode-phase simulation)\n", NUM_KERNELS);
    print_stats("Eager per-step wall (GPU event)", eager_wall, "ms");
    print_stats("Eager host submit",            eager_host, "ms");
    print_stats("Graph per-step wall (GPU event)", graph_wall, "ms");
    print_stats("Graph host submit",            graph_host, "ms");
    printf("\n  Per-step wall speedup (GPU event): %.2fx   (%.4f -> %.4f ms)\n", ew / gw, ew, gw);
    printf("  Host submit speedup           : %.2fx   (%.4f -> %.4f ms)\n", eh / gh, eh, gh);
    printf("  Note: eager host submit 包含队列背压阻塞，不等同于纯 launch overhead；\n"
           "        两者分开报告，不做混同。\n");

    CHECK(liteGraphExecDestroy(exec));
    CHECK(liteGraphDestroy(graph));
    CHECK(liteEventDestroy(ev_t0));
    CHECK(liteEventDestroy(ev_t1));
    CHECK(liteStreamDestroy(stream));
    printf("========== Bench1 Finished ==========\n");
}

// ============================ Bench1b: Native 纯提交成本（无背压） ============================
// 每次迭代先 sync 排空队列，计时区只含 driver 调用本身
static void run_bench_native_submit() {
    printf("\n========== Bench1b: Native Pure Submit Cost (idle queue, no backpressure) ==========\n");
    LiteStream stream{}; CHECK(liteStreamCreate(&stream));

    const int NUM_KERNELS = 5000;
    const int WARMUP = 30, REPEAT = 300;

    for (int w = 0; w < WARMUP; ++w) {
        CHECK(liteStreamSynchronize(stream));
        for (int k = 0; k < NUM_KERNELS; ++k) {
            CHECK(liteLaunchEmptyKernel(stream));
        }
        CHECK(liteStreamSynchronize(stream));
    }
    std::vector<double> eager_us;
    for (int r = 0; r < REPEAT; ++r) {
        CHECK(liteStreamSynchronize(stream));  // 每次采样前：排空队列
        double t0 = now_ms();
        for (int k = 0; k < NUM_KERNELS; ++k) {
            CHECK(liteLaunchEmptyKernel(stream));
        }
        double t1 = now_ms();
        eager_us.push_back((t1 - t0) * 1000.0 / NUM_KERNELS);
        CHECK(liteStreamSynchronize(stream));
    }

    LiteGraph graph{};
    CHECK(liteStreamBeginCaptureEx(stream, LITE_CAPTURE_RELAXED));
    for (int k = 0; k < NUM_KERNELS; ++k) {
        CHECK(liteLaunchEmptyKernel(stream));
    }

    CHECK(liteStreamEndCapture(stream, &graph));
    LiteGraphExec exec{};
    CHECK(liteGraphInstantiate(graph, &exec));
    for (int w = 0; w < WARMUP; ++w) {
        CHECK(liteStreamSynchronize(stream));
        CHECK(liteGraphLaunch(exec, stream));
        CHECK(liteStreamSynchronize(stream));
    }
    std::vector<double> graph_us;
    for (int r = 0; r < REPEAT; ++r) {
        CHECK(liteStreamSynchronize(stream));
        double t0 = now_ms();
        CHECK(liteGraphLaunch(exec, stream));
        double t1 = now_ms();
        graph_us.push_back((t1 - t0) * 1000.0);
        CHECK(liteStreamSynchronize(stream));
    }

    Stats se = compute_stats(eager_us), sg = compute_stats(graph_us);
    double eager_step_ms = se.p50 * NUM_KERNELS / 1000.0;
    double graph_step_ms = sg.p50 / 1000.0;
    printf("\n[Bench1b] native 纯提交成本（%d 次采样）\n", REPEAT);
    printf("  eager cuLaunchKernel : p50=%.3f  mean=%.3f  p99=%.3f us/call\n", se.p50, se.mean, se.p99);
    printf("  cuGraphLaunch(整图)  : p50=%.3f  mean=%.3f  p99=%.3f us/call (一次提交 %d 节点)\n",
           sg.p50, sg.mean, sg.p99, NUM_KERNELS);
    printf("  => %d-kernel step 纯 host 提交: eager ≈ %.4f ms, graph ≈ %.4f ms, 纯提交加速 ≈ %.1fx\n",
           NUM_KERNELS, eager_step_ms, graph_step_ms, eager_step_ms / graph_step_ms);
    printf("  Note: 不含背压/GPU执行/sync；Bench1 host submit 含背压，两口径不要互替。\n");

    CHECK(liteGraphExecDestroy(exec));
    CHECK(liteGraphDestroy(graph));
    CHECK(liteStreamDestroy(stream));
    printf("========== Bench1b Finished ==========\n");
}

// ============================ Bench2: 热更新三种机制对比 ============================
// 先做单节点图正确性（param-only 与 func-swap 必须真的改变 GPU 输出），再在 200 节点图上
// 更新中间节点，端到端（更新+launch+sync）对比四种口径：
//   A node-level param-only / B node-level func-swap / C graph-level ExecUpdate /
//   D 全量重捕获（baseline）—— 看清"改少数节点"与"改整图"各自该用哪种热更新。
static void run_bench_hot_update_comparison() {
    printf("\n========== Bench2+: Hot Update 三机制对比 (kernel-node vs cuGraphExecUpdate) ==========\n");
    LiteStream stream{}; CHECK(liteStreamCreate(&stream));

    CUdeviceptr d_buf = 0; CUresult cr = cuMemAlloc(&d_buf, sizeof(int)); CHECK_EQ(cr == CUDA_SUCCESS, "cuMemAlloc buf");
    void* buf = reinterpret_cast<void*>(d_buf);

    CUfunction cf_v1 = lite_get_dummy_kernel_func();   // lite_dummy_kernel : buf[0]=fill
    CUfunction cf_v2 = lite_get_dummy_kernel_v2_func(); // lite_dummy_kernel_v2 : buf[0]=fill+100
    CHECK_EQ(cf_v1 && cf_v2, "get v1/v2 funcs");

    const int N = HOT_UPDATE_NODES;      // 200
    const size_t UP_IDX = N / 2;         // 任意内部节点（证明不是只热更"最后一个"）

    // ---- [Correctness] 单节点图：验证 param-only 与 func-swap 都真的改了 GPU 行为 ----
    {
        LiteGraph g; CHECK(liteStreamBeginCaptureEx(stream, LITE_CAPTURE_THREAD_LOCAL));
        CHECK(liteLaunchTinyKernel(buf, 42, stream));   // v1: buf[0]=42
        CHECK(liteStreamEndCapture(stream, &g));
        LiteGraphExec e; CHECK(liteGraphInstantiate(g, &e));
        LiteGraphNode nd; CHECK(liteGraphGetNodeByIndex(g, 0, &nd));

        CHECK(liteGraphLaunch(e, stream)); CHECK(liteStreamSynchronize(stream));
        int h = 0; read_bytes(stream, buf, &h, sizeof(int)); CHECK_EQ(h == 42, "v1 fill=42");

        // A) param-only：fill 42 -> 77
        int nf = 77; void* a[2] = { &buf, &nf };
        CHECK(liteGraphExecKernelNodeSetParams(e, nd, a));
        CHECK(liteGraphLaunch(e, stream)); CHECK(liteStreamSynchronize(stream));
        h = 0; read_bytes(stream, buf, &h, sizeof(int)); CHECK_EQ(h == 77, "param-only 77");

        // B) func-swap：切到 v2 -> buf[0]=fill+100=177
        int nf2 = 77; void* a2[2] = { &buf, &nf2 };
        CHECK(liteGraphExecKernelNodeSetParamsEx(e, nd, (void*)cf_v2, a2));
        CHECK(liteGraphLaunch(e, stream)); CHECK(liteStreamSynchronize(stream));
        h = 0; read_bytes(stream, buf, &h, sizeof(int)); CHECK_EQ(h == 177, "func-swap v2: 77+100=177");

        CHECK(liteGraphNodeDestroy(nd));
        CHECK(liteGraphExecDestroy(e)); CHECK(liteGraphDestroy(g));
        printf("  [Correctness] param-only(42->77) + func-swap(v2:77+100=177) PASS\n");
    }

    // ---- 性能：N 节点图，更新内部节点 idx=UP_IDX ----
    LiteGraph graph; CHECK(liteStreamBeginCapture(stream));
    for (int i = 0; i < N; ++i) {
        CHECK(liteLaunchTinyKernel(buf, 1, stream));
    }

    CHECK(liteStreamEndCapture(stream, &graph));
    LiteGraphExec exec; CHECK(liteGraphInstantiate(graph, &exec));
    LiteGraphNode node; CHECK(liteGraphGetNodeByIndex(graph, UP_IDX, &node));
    CHECK(liteGraphLaunch(exec, stream)); CHECK(liteStreamSynchronize(stream));

    int nf = 7; void* args[2] = { &buf, &nf };
    const int WARM = 50, REP = 200;

    // A) node-level param-only
    for (int w = 0; w < WARM; ++w) {
        CHECK(liteGraphExecKernelNodeSetParams(exec, node, args));
        CHECK(liteGraphLaunch(exec, stream));
        CHECK(liteStreamSynchronize(stream));
    }
    std::vector<double> t_param;
    for (int i = 0; i < REP; ++i) {
        double t0 = now_ms();
        CHECK(liteGraphExecKernelNodeSetParams(exec, node, args));
        CHECK(liteGraphLaunch(exec, stream));
        CHECK(liteStreamSynchronize(stream));
        t_param.push_back(now_ms() - t0);
    }

    // B) node-level func+param swap
    for (int w = 0; w < WARM; ++w) {
        CHECK(liteGraphExecKernelNodeSetParamsEx(exec, node, (void*)cf_v2, args));
        CHECK(liteGraphLaunch(exec, stream));
        CHECK(liteStreamSynchronize(stream));
    }
    std::vector<double> t_swap;
    for (int i = 0; i < REP; ++i) {
        double t0 = now_ms();
        CHECK(liteGraphExecKernelNodeSetParamsEx(exec, node, (void*)cf_v2, args));
        CHECK(liteGraphLaunch(exec, stream));
        CHECK(liteStreamSynchronize(stream));
        t_swap.push_back(now_ms() - t0);
    }
    // 把内部节点切回 v1，使 exec 拓扑与新捕获的 v1 图一致，graph-level update 才能成功
    CHECK(liteGraphExecKernelNodeSetParamsEx(exec, node, (void*)cf_v1, args));

    // C) graph-level ExecUpdate：捕获同拓扑新图(fill=8) -> 整图更新 -> launch
    for (int w = 0; w < WARM; ++w) {
        LiteGraph ng; CHECK(liteStreamBeginCaptureEx(stream, LITE_CAPTURE_THREAD_LOCAL));
        for (int k = 0; k < N; ++k) {
            CHECK(liteLaunchTinyKernel(buf, 8, stream));
        }

        CHECK(liteStreamEndCapture(stream, &ng));
        LiteGraphExecUpdateResult ur; CHECK(liteGraphExecUpdate(exec, ng, &ur));
        CHECK_EQ(ur == LITE_GRAPH_UPDATE_SUCCESS, "graph update success");
        CHECK(liteGraphLaunch(exec, stream));
        CHECK(liteStreamSynchronize(stream));
        CHECK(liteGraphDestroy(ng));
    }
    std::vector<double> t_grp;
    for (int i = 0; i < REP; ++i) {
        LiteGraph ng; double t0 = now_ms();
        CHECK(liteStreamBeginCaptureEx(stream, LITE_CAPTURE_THREAD_LOCAL));
        for (int k = 0; k < N; ++k) {
            CHECK(liteLaunchTinyKernel(buf, 8, stream));
        }

        CHECK(liteStreamEndCapture(stream, &ng));
        LiteGraphExecUpdateResult ur; CHECK(liteGraphExecUpdate(exec, ng, &ur));
        CHECK_EQ(ur == LITE_GRAPH_UPDATE_SUCCESS, "graph update success");
        CHECK(liteGraphLaunch(exec, stream));
        CHECK(liteStreamSynchronize(stream));
        t_grp.push_back(now_ms() - t0);
        CHECK(liteGraphDestroy(ng));
    }

    // D) full recapture baseline
    std::vector<double> t_recap;
    for (int i = 0; i < REP / 2; ++i) {
        LiteGraph ng; LiteGraphExec in; double t0 = now_ms();
        CHECK(liteStreamBeginCaptureEx(stream, LITE_CAPTURE_THREAD_LOCAL));
        for (int k = 0; k < N; ++k) {
            CHECK(liteLaunchTinyKernel(buf, 9, stream));
        }

        CHECK(liteStreamEndCapture(stream, &ng));
        CHECK(liteGraphInstantiate(ng, &in));
        CHECK(liteGraphLaunch(in, stream));
        CHECK(liteStreamSynchronize(stream));
        t_recap.push_back(now_ms() - t0);
        CHECK(liteGraphExecDestroy(in));
        CHECK(liteGraphDestroy(ng));
    }

    Stats sp = compute_stats(t_param), ss = compute_stats(t_swap),
          sg = compute_stats(t_grp),  sr = compute_stats(t_recap);
    printf("\n[Hot Update 效率对比] %d kernel nodes, 更新内部节点 idx=%zu（端到端=更新+launch+sync）\n", N, UP_IDX);
    printf("  A) node-level param-only    p50=%.4f ms  p99=%.4f\n", sp.p50, sp.p99);
    printf("  B) node-level func-swap     p50=%.4f ms  p99=%.4f\n", ss.p50, ss.p99);
    printf("  C) graph-level ExecUpdate   p50=%.4f ms  p99=%.4f\n", sg.p50, sg.p99);
    printf("  D) full recapture (baseline) p50=%.4f ms  p99=%.4f\n", sr.p50, sr.p99);
    printf("\n  对比（越小越快；相对 D 的加速 = D.p50/该.p50）:\n");
    printf("    A vs D : node param-only     %.2fx faster than recapture\n", sr.p50 / sp.p50);
    printf("    B vs D : node func-swap      %.2fx faster than recapture\n", sr.p50 / ss.p50);
    printf("    C vs D : graph ExecUpdate    %.2fx faster than recapture\n", sr.p50 / sg.p50);
    printf("    A vs C : node(param-only) vs graph(ExecUpdate)  %.2fx\n", sg.p50 / sp.p50);
    printf("    B vs C : node(func-swap)  vs graph(ExecUpdate)  %.2fx\n", sg.p50 / ss.p50);
    printf("  解读: node-level 只改 1 个节点的参数/函数（亚微秒 setParams），graph-level 需\n"
           "        重新捕获整图 + 整图校验更新，代价与图规模/节点属性变更相关；\n"
           "        仅少数节点变化时 node-level 更省，节点变化多/拓扑变化时 graph-level 更稳。\n");

    CHECK(liteGraphNodeDestroy(node));
    CHECK(liteGraphExecDestroy(exec));
    CHECK(liteGraphDestroy(graph));
    cuMemFree(d_buf);
    CHECK(liteStreamDestroy(stream));
    printf("========== Bench2+ Finished ==========\n");
}

// ============================ Bench5: 真实 GEMM（cublas）—— decode vs prefill 真负载 ============================
// 全真实 cuBLAS SGEMM，沿"单算子小而碎 -> 大而重"扫一条谱（tiny/small/gemv/prefill），
// 给出 graph 收益在真实负载下的落点：decode 小算子 host-bound 收益大，prefill 大 GEMM GPU-bound ≈1x。
// 与 Bench1（空 kernel 隔离 launch 开销，机制上限）构成上下界。
static void run_bench_real_gemm() {
    printf("\n========== Bench5: Real GEMM (cublas) decode vs prefill ==========\n");
    cublasHandle_t h; CHECK_EQ(cublasCreate(&h) == CUBLAS_STATUS_SUCCESS, "cublasCreate");
    LiteStream s{}; CHECK(liteStreamCreate(&s));
    cublasSetStream(h, (cudaStream_t)*reinterpret_cast<CUstream*>(s));

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    const float alpha = 1.0f, beta = 0.0f;

    // 全部用真实 cuBLAS SGEMM；通过 (M,K,N,每步个数) 扫描 host-bound -> GPU-bound 的连续谱。
    // 真实 transformer 一层有 QKV/out/MLP up/down 多次 GEMM，decode 还含大量 head_dim 级小算子。
    auto run_shape = [&](const char* tag, int M, int K, int N, int gemms_per_step) {
        cudaMalloc(&dA, (size_t)M * K * sizeof(float));
        cudaMalloc(&dB, (size_t)K * N * sizeof(float));
        cudaMalloc(&dC, (size_t)M * N * sizeof(float));
        // warmup + eager
        for (int w = 0; w < 3; ++w) {
            for (int g = 0; g < gemms_per_step; ++g) {
                cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                            &alpha, dB, N, dA, K, &beta, dC, N);
            }
        }

        CHECK(liteStreamSynchronize(s));
        std::vector<double> t_e;
        for (int i = 0; i < 15; ++i) {
            double t0 = now_ms();
            for (int g = 0; g < gemms_per_step; ++g) {
                cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                            &alpha, dB, N, dA, K, &beta, dC, N);
            }

            CHECK(liteStreamSynchronize(s));
            t_e.push_back(now_ms() - t0);
        }

        LiteGraph g; CHECK(liteStreamBeginCaptureEx(s, LITE_CAPTURE_THREAD_LOCAL));
        for (int g2 = 0; g2 < gemms_per_step; ++g2) {
            cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                        &alpha, dB, N, dA, K, &beta, dC, N);
        }

        CHECK(liteStreamEndCapture(s, &g));
        LiteGraphExec e; CHECK(liteGraphInstantiate(g, &e));
        for (int w = 0; w < 3; ++w) {
            CHECK(liteGraphLaunch(e, s));
            CHECK(liteStreamSynchronize(s));
        }
        std::vector<double> t_g;
        for (int i = 0; i < 15; ++i) {
            double t0 = now_ms();
            CHECK(liteGraphLaunch(e, s));
            CHECK(liteStreamSynchronize(s));
            t_g.push_back(now_ms() - t0);
        }
        Stats se = compute_stats(t_e), sg = compute_stats(t_g);
        double spd = se.p50 / sg.p50;
        const char* regime = spd >= 1.15 ? "host-bound, graph 有效"
                              : spd >= 1.05 ? "临界 host-bound, graph 收益很小"
                              : "GPU-bound/临界, graph ≈1x";
        printf("  [%-12s M=%d K=N=%d x%d GEMMs/step] eager p50=%.3f ms  graph p50=%.3f ms  speedup=%.2fx  (%s)\n",
               tag, M, K, gemms_per_step, se.p50, sg.p50, spd, regime);
        CHECK(liteGraphExecDestroy(e)); CHECK(liteGraphDestroy(g));
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        dA = dB = dC = nullptr;
    };

    // 全真实 cuBLAS，沿"单算子越小越碎 -> 越大越重"扫一条谱：
    // 1) tiny  : head_dim 级极小 GEMV(128)、每步 256 个，单算子 GPU 时间极低，launch 占比最高；
    // 2) small  : 256 维 GEMV、每步 128 个（过渡区）；
    // 3) gemv   : 1024 维 GEMV、每步 50 个（典型 decode 权重投影，临界 host-bound）；
    // 4) prefill: M=2048 大 GEMM、每步 1 个（GPU-bound）。
    run_shape("decode-tiny",  1,  128,  128, 256);
    run_shape("decode-small", 1,  256,  256, 128);
    run_shape("decode-gemv",  1, 1024, 1024, 50);
    run_shape("prefill",     2048, 1024, 1024, 1);
    printf("  解读: 全部为真实 cuBLAS。单算子越小、每步数量越碎，host launch 在单步耗时里占比越高、\n"
           "        graph 收益越大（tiny 档）；随权重矩阵变大、单算子 GPU 访存时间上升，收益连续回落，\n"
           "        prefill 大 GEMM 为 GPU-bound≈1x。Bench1 的空 kernel ~15x 是隔离 launch 开销的机制上限，\n"
           "        本处是真实算子落点（decode 通常 1.x~数 x，与 vLLM decode graph 量级一致），两者为上下界关系。\n");

    cublasDestroy(h);
    CHECK(liteStreamDestroy(s));
    printf("========== Bench5 Finished ==========\n\n");
}

// ============================ CUPTI 插桩归因段（全程唯一开启 profiler 的区段） ============================
// 所有性能 bench 都在 native（无 CUPTI）下测量：activity 记账对每次 launch 收税，对 eager
// （5000 次独立 launch）惩罚远大于 graph（1 次 launch），会虚高加速比。因此 profiler 只在
// 所有 bench 结束后开启一次，且只产出"归因"证据、不产出性能结论：
//   A) SM/CE 时间区间交叠——双引擎并行的引擎级证据（CUPTI 实测区间求交，不依赖 wall-clock）；
//   B) 按 kernel 名聚合的 GPU 时间分解 + Top-N hotspot（含 cublas 对 M=1 派 gemv、大矩阵派 sgemm）。
static void run_profiler_attribution() {
    printf("\n========== Profiler Attribution (CUPTI instrumented; NOT for speedup) ==========\n");
    CHECK(liteProfilerStart());

    LiteStream sComp{}, sCopy{};
    CHECK(liteStreamCreate(&sComp));
    CHECK(liteStreamCreate(&sCopy));

    // ---- A) 双引擎区间交叠：H2D concurrent（compute 流 + copy 流）----
    const size_t CP = 32ull << 20;  // 32 MB
    CUdeviceptr  dWork = 0, dSrc = 0, dDst = 0;
    CHECK_EQ(cuMemAlloc(&dWork, 1 << 20) == CUDA_SUCCESS, "alloc work");
    CHECK_EQ(cuMemAlloc(&dSrc, CP) == CUDA_SUCCESS, "alloc src");
    CHECK_EQ(cuMemAlloc(&dDst, CP) == CUDA_SUCCESS, "alloc dst");
    void* hPinned = nullptr;
    CHECK_EQ(cudaMallocHost(&hPinned, CP) == cudaSuccess, "pinned alloc");
    const int LOOPS = 10000000, THREADS = 256;

    liteProfilerResetRegions();
    for (int i = 0; i < 20; ++i) {
        CHECK(liteLaunchDummyKernelHeavy(reinterpret_cast<void*>(dWork), LOOPS, THREADS, sComp));
        cuMemcpyHtoDAsync(dDst, hPinned, CP, *reinterpret_cast<CUstream*>(sCopy));  // H2D 走 CE
        CHECK(liteStreamSynchronize(sComp));
        CHECK(liteStreamSynchronize(sCopy));
    }
    liteProfilerReportOverlap();

    // ---- B) 按 kernel 名聚合的 GPU 时间分解 + Top-N hotspot：cublas 对 M=1 / 大矩阵派不同 kernel ----
    // 先建 handle、分配、warmup（cublas 首次初始化自身也会发 kernel），再 liteProfilerReset() 清掉
    // A 段 heavy kernel 与 cublas 初始化记录，使 Top-N 只统计正式两类 GEMM，便于读出派发路径差异。
    {
        cublasHandle_t ch;
        CHECK_EQ(cublasCreate(&ch) == CUBLAS_STATUS_SUCCESS, "cublasCreate in profiler");
        cublasSetStream(ch, (cudaStream_t)*reinterpret_cast<CUstream*>(sComp));
        const int GK = 1024, GN = 1024, MMAX = 2048;
        float *gA = nullptr, *gB = nullptr, *gC = nullptr;
        cudaMalloc(&gA, (size_t)MMAX * GK * sizeof(float));  // 按最大 M 分配，M=1/2048 共用
        cudaMalloc(&gB, (size_t)GK * GN * sizeof(float));
        cudaMalloc(&gC, (size_t)MMAX * GN * sizeof(float));
        const float galpha = 1.0f, gbeta = 0.0f;
        auto sgemm = [&](int M) {
            cublasSgemm(ch, CUBLAS_OP_N, CUBLAS_OP_N, GN, M, GK,
                        &galpha, gB, GN, gA, GK, &gbeta, gC, GN);
        };
        for (int w = 0; w < 3; ++w) {
            sgemm(1);
            sgemm(MMAX);
        }

        CHECK(liteStreamSynchronize(sComp));

        liteProfilerReset();  // 清累计计数/kernel map/区间，只保留下面正式 GEMM
        for (int i = 0; i < 200; ++i) {
            sgemm(1);      // decode 态 M=1 GEMV 派发路径
        }

        for (int i = 0; i < 5; ++i) {
            sgemm(MMAX);   // prefill 态大矩阵 SGEMM 派发路径
        }
        CHECK(liteStreamSynchronize(sComp));
        liteProfilerReport();  // flush + disable + kernel/memcpy 分解 + Top-10 by kernel name

        cublasDestroy(ch);
        cudaFree(gA); cudaFree(gB); cudaFree(gC);
    }

    printf("========== Profiler Attribution Finished ==========\n\n");
}

// ===================== Unit15: Scheduler Preemption & Recompute (vLLM‑style KV over‑subscription) =====================
// 测试点：
// 1. 物理块预算紧张，触发抢占；被抢占请求保留已生成token，释放KV物理块退回等待队列头部；
// 2. 恢复时重新分配块，需要上层重填全部历史KV（meta.resumed标记该槽需要重放prefill）；
// 3. 所有请求全部跑完，块无泄漏；统计抢占次数；纯host状态机确定性测试。
static void run_test_preemption_recompute() {
    printf("\n========== Unit15: Scheduler Preemption / Recompute (KV over‑subscription) ==========\n");
    const int TPB=16, MBPS=8, MAX_SEQS=8;
    // 负载：2个短请求16token，2个长请求128token；物理块故意压到紧张，触发抢占
    const int lens[] = {16,16,128,128};
    const int NREQ = (int)(sizeof(lens)/sizeof(lens[0]));
    const int PHYS = 10; // 故意调低预算，制造抢占条件

    LiteKvPoolConfig pcfg{TPB, MBPS, MAX_SEQS, PHYS};
    LiteKvPool pool{};
    LiteScheduler sch{};
    CHECK(liteKvPoolCreate(&pcfg, &pool));
    CHECK(liteSchedulerCreate(pool, MAX_SEQS, TPB, MBPS, 1, &sch));

    for (int i = 0; i < NREQ; ++i) {
        CHECK_EQ(liteSchedulerAddRequest(sch, lens[i]) == i, "add req return rid");
    }

    LiteStepMeta meta;
    int steps=0;
    for (;;) {
        if (liteSchedulerStep(sch, &meta)) {
            break;
        }

        steps++;
        // host侧mock：resumed标记只做断言，不需要真实KV
        std::vector<int> dummy_tok(meta.n_active, 1);
        liteSchedulerCommit(sch, &meta, dummy_tok.data());
    }

    int finished = liteSchedulerNumFinished(sch);
    int preempt_cnt = liteSchedulerNumPreemptions(sch);
    int free_blk = liteKvPoolFreeBlocks(pool);
    printf("[Unit15] steps=%d finished=%d preemptions=%d free_blocks=%d (expect %d)\n",
           steps, finished, preempt_cnt, free_blk, PHYS);

    CHECK_EQ(finished, "all requests complete after preemption‑recompute");
    CHECK_EQ(preempt_cnt > 0, "test load shall trigger at least one preemption");
    CHECK_EQ(free_blk == PHYS, "no physical‑block leak after drain");

    // 校验每个请求产出token等于目标长度
    for (int i = 0; i < NREQ; ++i) {
        std::vector<int> out_buf(256);
        int got = liteSchedulerGetResult(sch, i, out_buf.data(), (int)out_buf.size());
        CHECK_EQ(got == lens[i], "per‑request output length matches target_len");
    }

    printf("[Unit15] Preemption & Recompute host‑logic pass\n");

    liteSchedulerDestroy(sch);
    CHECK(liteKvPoolDestroy(pool));
}

// ===================== helper =====================
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
    return (float)((pa_rng(s) >> 8) & 0xFFFFFFu) / (float)(1u << 24) - 0.5f;  // [-0.5,0.5)
}

// ===================== Bench8: 端到端 continuous batching + 真 PagedAttention =====================
// 整套 runtime 的集成测试：scheduler 连续 batching（pack/commit）+ 真实 fp16 PagedAttention
// split/reduce 两 kernel（Triton AOT cubin，与 scheduler 共用同一套 KV 物理块 id），
// 捕获成定形 graph 后连续 128 步重放（长短请求混合、自然完成退出）；
// 每步 attention 输出对 host fp32 oracle 对拍，并断言 KV 物理块跨请求复用、跑完无泄漏。
// 确定性 mock 数据：同一 (rid,pos) 每次从同一 seed 复现该 token 的 K/V/Q（各 H*D 个 fp16）。
static void pa_fill_token(int rid, int pos, uint16_t* hk, uint16_t* hv, uint16_t* hq,
                          int H, int D) {
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

// host fp32 oracle：请求 rid 当前 decode 到 pos（KV 含 [0,pos] 共 pos+1 个 token），
// K/V/Q 都用与 GPU 位级一致的 fp16（f16->f32 参与计算），返回与 GPU 输出的最大绝对差。
static double pa_oracle_slot(int rid, int pos, const uint16_t* gpu_out, int H, int D) {
    const int T = pos + 1, HD = H * D;
    std::vector<uint16_t> Kv((size_t)T * HD), Vv((size_t)T * HD), hk(HD), hv(HD), hq(HD);
    for (int t = 0; t < T; ++t) {
        pa_fill_token(rid, t, hk.data(), hv.data(), hq.data(), H, D);  // 结束时 hq = Q@pos
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

// 从 attention 输出确定性"采样"一个 token（FNV 混合 fp16 位），仅用于推进 scheduler。
static int pa_sample_token(const uint16_t* slot_out, int HD) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < HD; ++i) {
        h ^= slot_out[i];
        h *= 16777619u;
    }

    return (int)(h % 10000u) + 1;
}

static void run_bench_paged_cb() {
    printf("\n========== Bench8: End-to-End Continuous Batching with real PagedAttention ==========\n");
    const int TPB = 16, MBPS = 8, MAX_SEQS = 8, H = 8, D = 128, PHYS = 40, NSPLIT = 4;
    const int lens[] = {1, 3, 16, 20, 47, 64, 128, 128, 8, 24, 40, 96};
    const int NREQ = (int)(sizeof(lens) / sizeof(lens[0]));
    const int HD = H * D;

    LiteKvPool pool{};
    LiteScheduler sch{};
    LiteStream stream{};
    LiteKvPoolConfig pcfg{TPB, MBPS, MAX_SEQS, PHYS};
    CHECK(liteKvPoolCreate(&pcfg, &pool));
    CHECK(liteSchedulerCreate(pool, MAX_SEQS, TPB, MBPS, 1, &sch));
    CHECK(liteStreamCreate(&stream));
    for (int i = 0; i < NREQ; ++i) {
        CHECK_EQ(liteSchedulerAddRequest(sch, lens[i]) == i, "b8 add request returns req_id");
    }

    LitePagedAttnConfig acfg{H, D, TPB, MAX_SEQS, MBPS, PHYS, NSPLIT};
    LitePagedAttn attn{};
    CHECK(litePagedAttnCreate(&acfg, &attn));
    printf("[Bench8] num_splits=%d split_size=%d; fp16 paged K/V 与 scheduler 共用同一套物理块 id\n",
           litePagedAttnNumSplits(attn), litePagedAttnSplitSize(attn));

    std::vector<uint16_t> kPool((size_t)PHYS * TPB * HD, f32_to_f16(5.0f));  // 未写=噪声(反污染)
    std::vector<uint16_t> vPool((size_t)PHYS * TPB * HD, f32_to_f16(5.0f));
    std::vector<uint16_t> q(MAX_SEQS * HD, 0), out(MAX_SEQS * HD, 0), hk(HD), hv(HD), hq(HD);
    std::vector<int> bt(MAX_SEQS * MBPS, 0), sl(MAX_SEQS, 0), tok(MAX_SEQS, 0);

    LiteGraph g{};
    LiteGraphExec e{};
    CHECK(liteStreamBeginCapture(stream));
    CHECK(litePagedAttnLaunch(attn, stream));
    CHECK(liteStreamEndCapture(stream, &g));
    CHECK(liteGraphInstantiate(g, &e));
    size_t nc = 0; CHECK(liteGraphGetNodeCount(g, &nc));
    int nk = 0;
    for (size_t i = 0; i < nc; ++i) {
        LiteGraphNode nd{}; LiteGraphNodeType t;
        CHECK(liteGraphGetNodeByIndex(g, i, &nd));
        CHECK(liteGraphNodeGetType(nd, &t));
        if (t == LITE_GRAPH_NODE_KERNEL) {
            ++nk;
        }
    }
    printf("[Bench8] captured graph: total=%zu kernel_nodes=%d (expect 2: split+reduce)\n", nc, nk);
    CHECK_EQ(nk == 2, "b8 graph contains exactly split+reduce 2 kernel nodes");

    std::vector<LiteStepMeta> metas_log;
    double worst = 0.0;
    int steps = 0;
    LiteStepMeta meta;
    for (;;) {
        if (liteSchedulerStep(sch, &meta)) {
            break;
        }

        ++steps;
        std::fill(q.begin(), q.end(), (uint16_t)0);
        std::fill(bt.begin(), bt.end(), 0);
        std::fill(sl.begin(), sl.end(), 0);
        bool dirty[PHYS];
        for (int i = 0; i < PHYS; ++i) {
            dirty[i] = false;
        }

        for (int s = 0; s < meta.n_active; ++s) {
            int rid = meta.slot_req[s], pos = meta.positions[s];
            bool need_replay = (meta.resumed[s] != 0);

            if (!need_replay) {
                // 普通decode step：仅新增当前pos的token
                int phys = meta.btable[s * MBPS + (pos / TPB)], slotw = pos % TPB;
                pa_fill_token(rid, pos, hk.data(), hv.data(), hq.data(), H, D);
                size_t base = ((size_t)phys * TPB + slotw) * HD;
                std::memcpy(&kPool[base], hk.data(), HD * sizeof(uint16_t));
                std::memcpy(&vPool[base], hv.data(), HD * sizeof(uint16_t));
                dirty[phys] = true;
                std::memcpy(&q[(size_t)s * HD], hq.data(), HD * sizeof(uint16_t));
            } else {
                // ========== 抢占后恢复：重放全部历史token 0~pos ==========
                for (int t = 0; t <= pos; ++t) {
                    pa_fill_token(rid, t, hk.data(), hv.data(), hq.data(), H, D);
                    int phys_t = meta.btable[s * MBPS + (t / TPB)];
                    int slotw_t = t % TPB;
                    size_t base = ((size_t)phys_t * TPB + slotw_t) * HD;
                    std::memcpy(&kPool[base], hk.data(), HD * sizeof(uint16_t));
                    std::memcpy(&vPool[base], hv.data(), HD * sizeof(uint16_t));
                    dirty[phys_t] = true;
                    if (t == pos) {
                        std::memcpy(&q[(size_t)s * HD], hq.data(), HD * sizeof(uint16_t));
                    }
                }
            }

            for (int k = 0; k < MBPS; ++k) {
                bt[s * MBPS + k] = meta.btable[s * MBPS + k];
            }
            sl[s] = pos + 1;
        }

        // 仅增量上传本步变脏的物理块（decode 每槽至多 1 块），而非整池 H2D。
        for (int phys = 0; phys < PHYS; ++phys) {
            if (dirty[phys]) {
                CHECK(litePagedAttnCopyBlock(attn, phys,
                        &kPool[(size_t)phys * TPB * HD], &vPool[(size_t)phys * TPB * HD]));
            }
        }
        CHECK(litePagedAttnCopyInput(attn, q.data(), bt.data(), sl.data()));
        CHECK(liteGraphLaunch(e, stream));
        // 必须先等 NON_BLOCKING 流上的 graph replay 写完 d_out，再用同步 D2H 回读：
        // 同步 cuMemcpyDtoH 走 legacy 流、不与非阻塞流保序，若先回读会抢在 graph 前
        // 拷到上一 step 的旧 out（间歇 max_diff 尖峰的根因，与块复用/TP 无关）。
        CHECK(liteStreamSynchronize(stream));
        CHECK(litePagedAttnCopyOutput(attn, out.data()));

        for (int s = 0; s < meta.n_active; ++s) {
            int rid = meta.slot_req[s], pos = meta.positions[s];
            double md = pa_oracle_slot(rid, pos, &out[(size_t)s * HD], H, D);
            if (md > worst) {
                worst = md;
            }
            tok[s] = pa_sample_token(&out[(size_t)s * HD], HD);
        }
        liteSchedulerCommit(sch, &meta, tok.data());
        metas_log.push_back(meta);
    }
    printf("[Bench8] steps=%d  worst max_diff vs fp32 oracle = %.2e\n", steps, worst);
    fflush(stdout);
    CHECK_EQ(worst < 2e-2, "b8 real paged-attention matches oracle across admit/retire/reuse");
    CHECK_EQ(liteSchedulerNumFinished(sch) == NREQ, "b8 all requests finished");
    for (int i = 0; i < NREQ; ++i) {
        std::vector<int> ores(256);
        CHECK_EQ(liteSchedulerGetResult(sch, i, ores.data(), (int)ores.size()) == lens[i],
                 "b8 per-request token count == target_len");
    }

    // 跨请求物理块复用：从每步 block-table 重建 物理块 -> 先后归属 req_id。
    std::vector<std::vector<int>> owner(PHYS);
    for (const LiteStepMeta& m : metas_log) {
        for (int s = 0; s < m.n_active; ++s) {
            int rid = m.slot_req[s], nb = (lens[rid] + TPB - 1) / TPB;
            for (int k = 0; k < nb; ++k) {
                int phys = m.btable[s * MBPS + k];
                bool seen = false;
                for (int r : owner[phys]) {
                    if (r == rid) {
                        seen = true;
                    }
                }

                if (!seen) {
                    owner[phys].push_back(rid);
                }
            }
        }
    }

    int reused = 0;
    for (auto& v : owner) {
        if (v.size() >= 2) {
            ++reused;
        }
    }
    CHECK_EQ(reused >= 1, "b8 physical fp16 KV block freed then reused by another request");
    CHECK_EQ(liteKvPoolFreeBlocks(pool) == PHYS, "b8 all physical blocks returned after drain");
    printf("[Bench8] fp16 KV blocks reused across requests: %d; short#0 finish@step%d, long#6(len128) finish@step%d\n",
           reused, liteSchedulerFinishStep(sch, 0), liteSchedulerFinishStep(sch, 6));

    CHECK(liteGraphExecDestroy(e)); CHECK(liteGraphDestroy(g));
    CHECK(litePagedAttnDestroy(attn));
    CHECK(liteStreamDestroy(stream));
    liteSchedulerDestroy(sch);
    CHECK(liteKvPoolDestroy(pool));
    printf("========== Bench8 Finished ==========\n");
}

// ===================== Unit16: Prefix Caching（block hash 共享）+ COW + refcnt（P0）=====================
// 两个验证面（A2 回归场景已删，bug 修复叙事保留在 docs/INTERVIEW.md）：
//  A) host 机制：同 prompt 前缀的第二请求经内容哈希查找命中共享块（refs=2）→ 分叉写触发
//     COW（换私有块、refs 归 1、计数 +1）→ 全部完成后块无泄漏；对空闲块的双重释放被
//     显式拒绝（返回 -1），不再静默重复入空闲表。
//  B) 真实数据 COW：PagedAttn 上把共享前缀块 D2D 搬运到 COW 新块，attention 输出对
//     fp32 oracle 对拍 —— 证明 D2D 拷贝忠实、共享/分叉后两请求各读各的块、互不污染。
//
// 确定性数据约定：pos < prefix_len 的 token 内容与 rid 无关（模拟共享 prompt 前缀），
// 之后按 rid 分叉 —— 与 LiteKvPool/scheduler 的哈希链（按 token 内容）天然一致。
static uint32_t pc_seed(int rid, int pos, int prefix_len) {
    return (pos < prefix_len)
        ? (uint32_t)(0x9e3779b9u ^ (uint32_t)(pos + 1) * 9173u)
        : (uint32_t)(0x9e3779b9u ^ (uint32_t)(rid + 1) * 1000003u
                     ^ (uint32_t)(pos + 1) * 9173u);
}
static void pc_fill_token(int rid, int pos, int prefix_len,
                          uint16_t* hk, uint16_t* hv, uint16_t* hq, int H, int D) {
    uint32_t s = pc_seed(rid, pos, prefix_len);
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
// host fp32 oracle（与 pa_oracle_slot 同构，token 生成器换成 pc_fill_token）
static double pc_oracle(int rid, int seq, int prefix_len, const uint16_t* gpu_out,
                        int H, int D) {
    const int HD = H * D;
    std::vector<uint16_t> Kv((size_t)seq * HD), Vv((size_t)seq * HD),
                          hk(HD), hv(HD), hq(HD);
    for (int t = 0; t < seq; ++t) {
        pc_fill_token(rid, t, prefix_len, hk.data(), hv.data(), hq.data(), H, D);
        std::memcpy(&Kv[(size_t)t * HD], hk.data(), HD * sizeof(uint16_t));
        std::memcpy(&Vv[(size_t)t * HD], hv.data(), HD * sizeof(uint16_t));
    }
    const float scale = 1.0f / std::sqrt((float)D);
    std::vector<float> sc(seq);
    double md = 0.0;
    for (int h = 0; h < H; ++h) {
        float mx = -1e30f;
        for (int t = 0; t < seq; ++t) {
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
        for (int t = 0; t < seq; ++t) {
            sc[t] = std::exp(sc[t] - mx);
            sum += sc[t];
        }

        for (int d = 0; d < D; ++d) {
            double w = 0.0;
            for (int t = 0; t < seq; ++t) {
                w += (double)sc[t] * f16_to_f32(Vv[(size_t)t * HD + h * D + d]);
            }

            md = std::max(md, (double)std::fabs(f16_to_f32(gpu_out[h * D + d]) - (float)(w / sum)));
        }
    }
    return md;
}

static void run_test_prefix_caching() {
    printf("\n========== Unit16: Prefix Caching (block hash) + COW + refcnt ==========\n");
    const int TPB = 16, MBPS = 8, MAX_SEQS = 4, PHYS = 10;
    const int PREFIX = 5, TARGET = 40;  // prompt 5 token（未满块）→ 必共享 + 必 COW

    // ---- A) host 机制：共享 -> 分叉 COW -> 无泄漏 -> double-free 拒绝 ----
    {
        LiteKvPoolConfig pcfg{TPB, MBPS, MAX_SEQS, PHYS};
        LiteKvPool pool{}; LiteScheduler sch{};
        CHECK(liteKvPoolCreate(&pcfg, &pool));
        CHECK(liteSchedulerCreate(pool, MAX_SEQS, TPB, MBPS, /*continuous=*/1, &sch));

        int prompt[PREFIX];
        for (int i = 0; i < PREFIX; ++i) prompt[i] = 7000 + i;  // 同一 prompt
        CHECK_EQ(liteSchedulerAddRequestEx(sch, TARGET, prompt, PREFIX) == LITE_SUCCESS,
                 "u16 add req0 with prompt");
        CHECK_EQ(liteSchedulerAddRequestEx(sch, TARGET, prompt, PREFIX) == LITE_SUCCESS,
                 "u16 add req1 with same prompt");

        LiteStepMeta meta;
        int steps = 0, saw_cow_src = 0;
        for (;;) {
            if (liteSchedulerStep(sch, &meta)) {
                break;
            }
            ++steps;
            for (int i = 0; i < meta.n_active; ++i) {
                if (meta.cow_src[i] != 0) {
                    ++saw_cow_src;
                }
            }
            // 分叉 token：rid*1000+pos，两请求自 position=PREFIX 起内容不同
            std::vector<int> tok(meta.n_active);
            for (int i = 0; i < meta.n_active; ++i) {
                tok[i] = meta.slot_req[i] * 1000 + meta.positions[i];
            }

            liteSchedulerCommit(sch, &meta, tok.data());
        }

        int hits = liteSchedulerNumCacheHits(sch);
        int cows = liteSchedulerNumCow(sch);
        int freeb = liteKvPoolFreeBlocks(pool);
        printf("[Unit16-A] steps=%d cache_hits=%d cow=%d cow_src_rows=%d free_blocks=%d/%d\n",
               steps, hits, cows, saw_cow_src, freeb, PHYS);
        CHECK_EQ(hits == 1, "u16 req1 prompt block hit prefix cache exactly once");
        CHECK_EQ(cows == 1 && saw_cow_src == 1, "u16 exactly one COW at the fork write");
        CHECK_EQ(liteSchedulerNumFinished(sch) == 2, "u16 both requests finished");
        CHECK_EQ(freeb == PHYS, "u16 no physical-block leak after drain");
        for (int i = 0; i < 2; ++i) {
            std::vector<int> ores(64);
            CHECK_EQ(liteSchedulerGetResult(sch, i, ores.data(), (int)ores.size()) == TARGET,
                     "u16 per-request output length");
        }

        // double-free 探针：空闲块二次释放必须被拒绝
        int pb = liteKvPoolAllocBlock(pool);
        CHECK_EQ(pb >= 0, "u16 alloc probe");
        CHECK_EQ(liteKvPoolFreeBlock(pool, pb) == 0, "u16 first free returns refs=0");
        CHECK_EQ(liteKvPoolFreeBlock(pool, pb) == -1, "u16 double free rejected (-1)");
        CHECK_EQ(liteKvPoolFreeBlocks(pool) == PHYS, "u16 no phantom page after double free");

        liteSchedulerDestroy(sch);
        CHECK(liteKvPoolDestroy(pool));
        printf("[Unit16-A] Prefix sharing + COW + refcnt (host) pass\n");
    }

    // ---- B) 真实数据 COW：D2D 搬运前缀块 + 分叉写，attention 对拍 fp32 oracle ----
    {
        const int B = 2, H = 2, D = 128, PHYS2 = 8, MB2 = 2, SEQ0 = 5, SEQ1 = 6;
        const int HD = H * D;
        LiteKvPoolConfig pcfg{TPB, MB2, B, PHYS2};
        LiteKvPool pool{}; CHECK(liteKvPoolCreate(&pcfg, &pool));
        LitePagedAttnConfig acfg{H, D, TPB, B, MB2, PHYS2, /*num_splits=*/2};
        LitePagedAttn attn{}; CHECK(litePagedAttnCreate(&acfg, &attn));
        LiteStream stream{}; CHECK(liteStreamCreate(&stream));

        std::vector<uint16_t> bk((size_t)TPB * HD), bv((size_t)TPB * HD), hq(HD);
        std::vector<uint16_t> q((size_t)B * HD, 0), out((size_t)B * HD, 0);
        std::vector<int> bt((size_t)B * MB2, 0), sl(B, 0);

        // b0 = 共享前缀块（slot0..4，与 rid 无关）；req0 只用它（COW 后的 D2D 拷贝目标）
        for (int t = 0; t < PREFIX; ++t) {
            pc_fill_token(0, t, PREFIX, &bk[(size_t)t * HD], &bv[(size_t)t * HD], hq.data(), H, D);
        }
        int b0 = liteKvPoolAllocBlock(pool);
        int b1 = liteKvPoolAllocBlock(pool);
        CHECK(litePagedAttnCopyBlock(attn, b0, bk.data(), bv.data()));

        // COW 数据搬运：b0 -> b1 整块 D2D；b1 上只有前缀内容（seq=5 可完整对拍 D2D 忠实性）
        CHECK(litePagedAttnCopyBlockD2D(attn, b1, b0, stream));
        CHECK(liteStreamSynchronize(stream));
        CHECK_EQ(liteKvPoolBlockRefs(pool, b0) == 1 && liteKvPoolBlockRefs(pool, b1) == 1,
                 "u16-B COW blocks privately owned after copy");

        // req1 在 b0 原地分叉写 slot5（refs=1，私有后可直接写）：整块镜像 = 前缀 + rid1 token
        pc_fill_token(1, PREFIX, PREFIX, &bk[(size_t)PREFIX * HD], &bv[(size_t)PREFIX * HD],
                      hq.data(), H, D);
        CHECK(litePagedAttnCopyBlock(attn, b0, bk.data(), bv.data()));

        // attention：行0 = req0（seq=5, btable[b1]）；行1 = req1（seq=6, btable[b0]）
        // block-table 布局为 [B][MB2]：行 b 的槽位从 b*MB2 开始
        bt[0] = b1; bt[1 * MB2] = b0;
        sl[0] = SEQ0; sl[1] = SEQ1;
        // Q：req0 取 pos4，req1 取 pos5
        {
            std::vector<uint16_t> hk2(HD), hv2(HD);
            pc_fill_token(0, SEQ0 - 1, PREFIX, hk2.data(), hv2.data(), &q[0], H, D);
            pc_fill_token(1, SEQ1 - 1, PREFIX, hk2.data(), hv2.data(), &q[(size_t)HD], H, D);
        }
        CHECK(litePagedAttnCopyInput(attn, q.data(), bt.data(), sl.data()));
        CHECK(litePagedAttnLaunchB(attn, B, stream));
        CHECK(liteStreamSynchronize(stream));
        CHECK(litePagedAttnCopyOutput(attn, out.data()));

        double md0 = pc_oracle(0, SEQ0, PREFIX, &out[0], H, D);
        double md1 = pc_oracle(1, SEQ1, PREFIX, &out[(size_t)HD], H, D);
        printf("[Unit16-B] COW-D2D row max_diff=%.2e  forked-inplace row max_diff=%.2e\n", md0, md1);
        CHECK_EQ(md0 < 2e-2, "u16-B D2D-copied prefix block matches fp32 oracle");
        CHECK_EQ(md1 < 2e-2, "u16-B forked in-place write matches fp32 oracle (no cross-pollution)");

        CHECK_EQ(liteKvPoolFreeBlock(pool, b0) == 0, "u16-B free b0");
        CHECK_EQ(liteKvPoolFreeBlock(pool, b1) == 0, "u16-B free b1");
        CHECK_EQ(liteKvPoolFreeBlocks(pool) == PHYS2, "u16-B pool drained");
        CHECK(liteStreamDestroy(stream));
        CHECK(litePagedAttnDestroy(attn));
        CHECK(liteKvPoolDestroy(pool));
        printf("[Unit16-B] Real-data COW (D2D block copy) + forked write pass\n");
    }

    printf("[Unit16] Prefix Caching + COW + refcnt pass\n");
}

// ===================== Unit17: INT8 KV 量化（P1）=====================
// KV 常驻显存从 fp16 换成 int8（per-(block,head) 对称量化，K/V 各 [P][H] 个 fp32 scale），
// kernel load 时反量化。两个验证面（B int8 COW 已删，结论保留在 docs/INTERVIEW.md）：
//  A) 正确性 + 精度代价：int8 实例逐块量化上传，attention 对【量化感知】fp32 oracle 对拍
//     （oracle 用 int8*scale 反量化值，模型 kernel 眼中的真实数据）；同数据的 fp16 实例
//     作基线，报告量化代价。seq=45/NSPLIT=2 使 SPLIT_SIZE=23 不对齐 16，tile 跨物理块，
//     检验 per-token scale 向量寻址（一个 tile 读两个物理块的 scale）。
//  C) 显存：litePagedAttnKvBytes 断言 int8（含 scale）< fp16 的 55%；数据通路 fail-fast
//     （fp16 实例拒收 int8 块、反之亦然）。
//
// 量化口径（与 liteQuantBlockFp16ToI8 / Triton 侧 quantize_kv_per_head 三方一致）：
//   scale = amax/127（每块每 head，K/V 独立），q = round(x/scale) 截 [-127,127]，
//   反量化 x' = q*scale —— 即 kernel 内 per-token 乘 scale 的精确语义。
static double qz_attn_oracle_deq(const uint16_t* hq, const float* deq_k, const float* deq_v,
                                 int seq, const uint16_t* gpu_out, int H, int D) {
    const int HD = H * D;
    const float scale = 1.0f / std::sqrt((float)D);
    std::vector<float> sc(seq);
    double md = 0.0;
    for (int h = 0; h < H; ++h) {
        float mx = -1e30f;
        for (int t = 0; t < seq; ++t) {
            double a = 0.0;
            for (int d = 0; d < D; ++d) {
                a += (double)f16_to_f32(hq[h * D + d]) * deq_k[(size_t)t * HD + h * D + d];
            }

            sc[t] = (float)a * scale;
            if (sc[t] > mx) {
                mx = sc[t];
            }
        }

        double sum = 0.0;
        for (int t = 0; t < seq; ++t) {
            sc[t] = std::exp(sc[t] - mx);
            sum += sc[t];
        }

        for (int d = 0; d < D; ++d) {
            double w = 0.0;
            for (int t = 0; t < seq; ++t) {
                w += (double)sc[t] * deq_v[(size_t)t * HD + h * D + d];
            }

            md = std::max(md, (double)std::fabs(f16_to_f32(gpu_out[h * D + d]) - (float)(w / sum)));
        }
    }
    return md;
}
// 把一个请求的 int8 块列表反量化成 [seq][HD] fp32（oracle 数据源）
static void qz_row_deq(const std::vector<int8_t>* blk_k, const std::vector<int8_t>* blk_v,
                       const std::vector<float>* blk_sk, const std::vector<float>* blk_sv,
                       const int* phys_ids, int seq, int TPB, int H, int D,
                       std::vector<float>& deq_k, std::vector<float>& deq_v) {
    const int HD = H * D;
    deq_k.assign((size_t)seq * HD, 0.0f);
    deq_v.assign((size_t)seq * HD, 0.0f);
    for (int t = 0; t < seq; ++t) {
        int p = phys_ids[t / TPB], slot = t % TPB;
        for (int i = 0; i < HD; ++i) {
            int h = i / D;
            deq_k[(size_t)t * HD + i] = (float)blk_k[p][(size_t)slot * HD + i] * blk_sk[p][h];
            deq_v[(size_t)t * HD + i] = (float)blk_v[p][(size_t)slot * HD + i] * blk_sv[p][h];
        }
    }
}
static void run_test_int8_kv() {
    printf("\n========== Unit17: INT8 KV Cache Quantization (per-block-head scale) ==========\n");
    const int TPB = 16;

    // ---- C) 显存断言 + 数据通路 fail-fast（独立实例，先跑） ----
    {
        LitePagedAttnConfig cfg{2, 128, 16, 2, 4, 8, /*num_splits=*/2, /*int8_kv=*/0};
        LitePagedAttn f16{}; CHECK(litePagedAttnCreate(&cfg, &f16));
        cfg.int8_kv = 1;
        LitePagedAttn i8{}; CHECK(litePagedAttnCreate(&cfg, &i8));
        size_t bf = litePagedAttnKvBytes(f16), bi = litePagedAttnKvBytes(i8);
        // 数据部分精确减半；加 scale 后总量 ≈50%+ε（H*D/BS 越大 ε 越小），上限 55%
        CHECK_EQ(bi * 20 < bf * 11, "u17-C int8 KV bytes (incl. scales) < 55% of fp16");
        printf("[Unit17-C] KV resident bytes: fp16=%zu  int8=%zu  (%.1f%%, 含 scale)\n",
               bf, bi, 100.0 * (double)bi / (double)bf);
        // 交叉数据通路必须被拒绝（fp16/int8 的块字节数与语义都不同）
        std::vector<uint16_t> tmp((size_t)16 * 2 * 128, 0);
        std::vector<float> sc(2, 1.0f);
        CHECK_EQ(litePagedAttnCopyBlock(i8, 0, tmp.data(), tmp.data()) != LITE_SUCCESS,
                 "u17-C CopyBlock(fp16 path) rejected on int8 instance");
        CHECK_EQ(litePagedAttnCopyBlockI8(f16, 0, tmp.data(), tmp.data(),
                                          sc.data(), sc.data()) != LITE_SUCCESS,
                 "u17-C CopyBlockI8 rejected on fp16 instance");
        CHECK_EQ(litePagedAttnCopyKV(i8, tmp.data(), tmp.data()) != LITE_SUCCESS,
                 "u17-C CopyKV(full-pool fp16) rejected on int8 instance");
        litePagedAttnDestroy(f16);
        litePagedAttnDestroy(i8);
    }

    // ---- A) 正确性 + 精度代价（int8 vs 量化感知 oracle；fp16 基线） ----
    {
        const int B = 2, H = 2, D = 128, MB = 4, PHYS = 8, NS = 2;
        const int HD = H * D;
        const int seqs[B] = {20, 45};   // 45/NS=2 → SPLIT_SIZE=23 不对齐 16，tile 跨块
        LitePagedAttnConfig ac{H, D, TPB, B, MB, PHYS, NS, /*int8_kv=*/0};
        LitePagedAttn f16{}; CHECK(litePagedAttnCreate(&ac, &f16));
        ac.int8_kv = 1;
        LitePagedAttn i8{}; CHECK(litePagedAttnCreate(&ac, &i8));
        LiteStream stream{}; CHECK(liteStreamCreate(&stream));

        std::vector<int> row_phys[B];
        row_phys[0] = {0, 1};           // 20 tok = 2 块
        row_phys[1] = {2, 3, 4};        // 45 tok = 3 块

        // host 侧：每物理块的 fp16 原文 / int8 量化版 / scale
        std::vector<std::vector<uint16_t>> fp16_k(PHYS, std::vector<uint16_t>((size_t)TPB * HD)),
                                           fp16_v(PHYS, std::vector<uint16_t>((size_t)TPB * HD));
        std::vector<std::vector<int8_t>> i8k(PHYS, std::vector<int8_t>((size_t)TPB * HD)),
                                         i8v(PHYS, std::vector<int8_t>((size_t)TPB * HD));
        std::vector<std::vector<float>> sk(PHYS, std::vector<float>(H)),
                                        sv(PHYS, std::vector<float>(H));
        std::vector<uint16_t> hk((size_t)TPB * HD), hv((size_t)TPB * HD), hq(HD);

        for (int b = 0; b < B; ++b) {
            int nblk = (seqs[b] + TPB - 1) / TPB;
            for (int bi_ = 0; bi_ < nblk; ++bi_) {
                int p = row_phys[b][bi_];
                for (int t = 0; t < TPB; ++t) {
                    int pos = bi_ * TPB + t;
                    if (pos >= seqs[b]) {
                        break;
                    }

                    pc_fill_token(b, pos, 0, &hk[(size_t)t * HD], &hv[(size_t)t * HD],
                                  hq.data(), H, D);
                }
                std::memcpy(fp16_k[p].data(), hk.data(), fp16_k[p].size() * 2);
                std::memcpy(fp16_v[p].data(), hv.data(), fp16_v[p].size() * 2);
                CHECK(liteQuantBlockFp16ToI8(fp16_k[p].data(), fp16_v[p].data(),
                                             H, D, TPB, i8k[p].data(), i8v[p].data(),
                                             sk[p].data(), sv[p].data()));
                CHECK(litePagedAttnCopyBlockI8(i8, p, i8k[p].data(), i8v[p].data(),
                                               sk[p].data(), sv[p].data()));
                CHECK(litePagedAttnCopyBlock(f16, p, fp16_k[p].data(), fp16_v[p].data()));
            }
        }

        // block-table + seq_lens + Q（取各自最后 token 的 q）
        std::vector<int> bt((size_t)B * MB, 0), sl(B, 0);
        std::vector<uint16_t> q((size_t)B * HD, 0), out((size_t)B * HD, 0);
        for (int b = 0; b < B; ++b) {
            sl[b] = seqs[b];
            for (int i = 0; i < (int)row_phys[b].size(); ++i) {
                bt[(size_t)b * MB + i] = row_phys[b][i];
            }
            pc_fill_token(b, seqs[b] - 1, 0, hk.data(), hv.data(), &q[(size_t)b * HD], H, D);
        }

        // 未量化 oracle 数据源（fp16 原文反成 fp32，两个实例共用）；量化感知数据源按需现算
        std::vector<std::vector<float>> uq_k(B), uq_v(B);
        for (int b = 0; b < B; ++b) {
            uq_k[b].resize((size_t)seqs[b] * HD);
            uq_v[b].resize((size_t)seqs[b] * HD);
            for (int t = 0; t < seqs[b]; ++t) {
                int p = row_phys[b][t / TPB], slot = t % TPB;
                for (int i = 0; i < HD; ++i) {
                    uq_k[b][(size_t)t * HD + i] = f16_to_f32(fp16_k[p][(size_t)slot * HD + i]);
                    uq_v[b][(size_t)t * HD + i] = f16_to_f32(fp16_v[p][(size_t)slot * HD + i]);
                }
            }
        }
        double md_i8 = 0.0, md_f16 = 0.0, md_i8_cost = 0.0;
        for (int pass = 0; pass < 2; ++pass) {
            LitePagedAttn a = pass == 0 ? i8 : f16;
            CHECK(litePagedAttnCopyInput(a, q.data(), bt.data(), sl.data()));
            CHECK(litePagedAttnLaunchB(a, B, stream));
            CHECK(liteStreamSynchronize(stream));
            CHECK(litePagedAttnCopyOutput(a, out.data()));
            for (int b = 0; b < B; ++b) {
                pc_fill_token(b, seqs[b] - 1, 0, hk.data(), hv.data(), hq.data(), H, D);
                if (pass == 0) {
                    // kernel 正确性：int8 输出 vs 量化感知 oracle（int8*scale 反量化）
                    std::vector<float> dq_k, dq_v;
                    qz_row_deq(i8k.data(), i8v.data(), sk.data(), sv.data(),
                               row_phys[b].data(), seqs[b], TPB, H, D, dq_k, dq_v);
                    md_i8 = std::max(md_i8,
                                     qz_attn_oracle_deq(hq.data(), dq_k.data(), dq_v.data(),
                                                        seqs[b], &out[(size_t)b * HD], H, D));
                    // 量化代价：int8 输出 vs 未量化 fp32 oracle
                    md_i8_cost = std::max(md_i8_cost,
                                          qz_attn_oracle_deq(hq.data(), uq_k[b].data(),
                                                             uq_v[b].data(), seqs[b],
                                                             &out[(size_t)b * HD], H, D));
                } else {
                    // fp16 基线：fp16 输出 vs 同一未量化 fp32 oracle
                    md_f16 = std::max(md_f16,
                                      qz_attn_oracle_deq(hq.data(), uq_k[b].data(),
                                                         uq_v[b].data(), seqs[b],
                                                         &out[(size_t)b * HD], H, D));
                }
            }
        }
        printf("[Unit17-A] kernel 正确性(int8 vs 量化感知 oracle)=%.2e  fp16 基线=%.2e  "
               "量化代价(int8 vs 未量化 oracle)=%.2e (%.0fx, 常驻省 ~50%%)\n",
               md_i8, md_f16, md_i8_cost, md_i8_cost / std::max(md_f16, 1e-9));
        CHECK_EQ(md_i8 < 2e-2, "u17-A int8 attention matches quant-aware fp32 oracle");
        CHECK_EQ(md_f16 < 2e-2, "u17-A fp16 baseline matches fp32 oracle");
        CHECK_EQ(md_i8_cost > md_f16 * 2,
                 "u17-A quant cost should exceed fp16 baseline by >2x (sanity)");

        CHECK(liteStreamDestroy(stream));
        litePagedAttnDestroy(i8);
        litePagedAttnDestroy(f16);
    }

    printf("[Unit17] INT8 KV quantization pass\n");
}

// ===================== Unit13: batch 分桶多图 + 运行时选桶（vLLM capture sizes）=====================
// 同一套按 max_batch 预分配的固定 buffer（各桶图烘焙同一批指针），按 capture sizes {1,2,4,8} 各捕
// 一张定长图，split/reduce 的 grid 按桶 batch 固定；运行时按 n_active 选 ceil2pow 最小桶图 launch，
// 零 recapture / 零 setParams。padding 行 seq_len=0，变长 split kernel 空 split 自写 LSE 哨兵、
// reduce 输出精确 0，天然兼容。对应 vLLM 的 capture sizes + padding 机制。
static void run_test_bucket_graph() {
    printf("\n========== Unit13: Batch-Bucketed Graphs + Runtime Selection (capture sizes) ==========\n");
    const int TPB = 16, MBPS = 8, MAX_SEQS = 8, H = 8, D = 128, PHYS = 40, NSPLIT = 4;
    const int HD = H * D;
    const int slen[MAX_SEQS] = {17, 3, 64, 40, 5, 128, 33, 96};
    const int SIZES[4] = {1, 2, 4, 8}, NB = 4;

    LiteStream stream{}; CHECK(liteStreamCreate(&stream));
    LitePagedAttnConfig acfg{H, D, TPB, MAX_SEQS, MBPS, PHYS, NSPLIT};
    LitePagedAttn attn{}; CHECK(litePagedAttnCreate(&acfg, &attn));

    std::vector<uint16_t> kPool((size_t)PHYS * TPB * HD, f32_to_f16(5.0f));  // 未用块=噪声(反污染)
    std::vector<uint16_t> vPool((size_t)PHYS * TPB * HD, f32_to_f16(5.0f));
    std::vector<uint16_t> qfull(MAX_SEQS * HD, 0), q(MAX_SEQS * HD, 0), out(MAX_SEQS * HD, 0),
                         hk(HD), hv(HD), hq(HD);
    std::vector<int> btfull(MAX_SEQS * MBPS, 0), bt(MAX_SEQS * MBPS, 0), sl(MAX_SEQS, 0);

    // 静态 paged 数据：顺序物理块分配（跨请求 btable 天然非连续），decode Q=每请求末 token。
    int next_phys = 0;
    for (int s = 0; s < MAX_SEQS; ++s) {
        int nb = (slen[s] + TPB - 1) / TPB;
        for (int k = 0; k < nb; ++k) {
            btfull[s * MBPS + k] = next_phys++;
        }
        for (int t = 0; t < slen[s]; ++t) {
            int phys = btfull[s * MBPS + (t / TPB)], slotw = t % TPB;
            pa_fill_token(s, t, hk.data(), hv.data(), hq.data(), H, D);
            std::memcpy(&kPool[((size_t)phys * TPB + slotw) * HD], hk.data(), HD * 2);
            std::memcpy(&vPool[((size_t)phys * TPB + slotw) * HD], hv.data(), HD * 2);
            if (t == slen[s] - 1) {
                std::memcpy(&qfull[(size_t)s * HD], hq.data(), HD * 2);
            }
        }
    }
    CHECK_EQ(next_phys <= PHYS, "fail");
    CHECK(litePagedAttnCopyKV(attn, kPool.data(), vPool.data()));

    // 4 张桶图：共享同一套 max_batch 固定 buffer（烘焙相同指针），仅 grid 有效行数不同。
    LiteGraph g[4] = {}; LiteGraphExec e[4] = {};
    for (int i = 0; i < NB; ++i) {
        CHECK(liteStreamBeginCapture(stream));
        CHECK(litePagedAttnLaunchB(attn, SIZES[i], stream));
        CHECK(liteStreamEndCapture(stream, &g[i]));
        CHECK(liteGraphInstantiate(g[i], &e[i]));
        size_t nc = 0; CHECK(liteGraphGetNodeCount(g[i], &nc));
        int nk = 0;
        for (size_t j = 0; j < nc; ++j) {
            LiteGraphNode nd{}; LiteGraphNodeType t;
            CHECK(liteGraphGetNodeByIndex(g[i], j, &nd));
            CHECK(liteGraphNodeGetType(nd, &t));
            if (t == LITE_GRAPH_NODE_KERNEL) {
                ++nk;
            }
        }
        CHECK_EQ(nk == 2, "bucket graph has exactly split+reduce 2 kernel nodes");
    }
    printf("[Unit13] captured %d bucket graphs {1,2,4,8}, all sharing one max_batch=%d fixed buffer set\n",
           NB, MAX_SEQS);

    auto bucket_idx = [&](int n) {
        for (int i = 0; i < NB; ++i) {
            if (SIZES[i] >= n) {
                return i;
            }
        }

        return NB - 1;
    };
    // 每步只在捕获外改固定 buffer 内容：前 n 行有效，其余 padding(sl=0/bt=0/q=0)。
    auto prepare = [&](int n) {
        for (int s = 0; s < MAX_SEQS; ++s) {
            sl[s] = (s < n) ? slen[s] : 0;
            for (int k = 0; k < MBPS; ++k) {
                bt[s * MBPS + k] = (s < n) ? btfull[s * MBPS + k] : 0;
            }

            if (s < n) {
                std::memcpy(&q[(size_t)s * HD], &qfull[(size_t)s * HD], HD * 2);
            } else {
                std::memset(&q[(size_t)s * HD], 0, HD * 2);
            }
        }

        CHECK(litePagedAttnCopyInput(attn, q.data(), bt.data(), sl.data()));
    };
    auto replay_bucket = [&](int n) {
        int bi = bucket_idx(n);
        CHECK(liteGraphLaunch(e[bi], stream));
        CHECK(liteStreamSynchronize(stream));
        CHECK(litePagedAttnCopyOutput(attn, out.data()));
        return SIZES[bi];
    };

    // 正确性：active 行对 fp32 oracle；padding 行（n..bucket-1）reduce 必须输出精确 0。
    const int test_n[] = {1, 2, 3, 4, 7, 8};
    double worst = 0.0;
    for (int n : test_n) {
        prepare(n);
        int bk = replay_bucket(n);
        double md = 0.0; bool pad_zero = true;
        for (int s = 0; s < n; ++s) {
            md = std::max(md, pa_oracle_slot(s, slen[s] - 1, &out[(size_t)s * HD], H, D));
        }

        for (int s = n; s < bk; ++s) {
            for (int j = 0; j < HD; ++j) {
                if (f16_to_f32(out[(size_t)s * HD + j]) != 0.0f) {
                    pad_zero = false;
                }
            }
        }
        worst = std::max(worst, md);
        CHECK_EQ(pad_zero, "padding rows (seq_len=0) reduce to exact 0");
        printf("[Unit13] n_active=%d -> bucket %d: worst max_diff=%.2e, padding rows zero\n", n, bk, md);
    }
    CHECK_EQ(worst < 2e-2, "bucketed graph outputs match fp32 oracle across all bucket selections");

    // 运行时选桶：连续 replay 不同桶图（仅图外 memcpy 改 sl/q），全程零 recapture。
    {
        const int seq[] = {8, 1, 4, 2, 8}; int picked[5];
        for (int i = 0; i < 5; ++i) {
            prepare(seq[i]);
            picked[i] = replay_bucket(seq[i]);
        }

        double md = 0.0;
        for (int s = 0; s < 8; ++s) {
            md = std::max(md, pa_oracle_slot(s, slen[s] - 1, &out[(size_t)s * HD], H, D));
        }
        CHECK_EQ(md < 2e-2, "outputs stay correct after runtime bucket switching without recapture");
        printf("[Unit13] runtime select 8->1->4->2->8 (buckets %d->%d->%d->%d->%d), zero recapture, max_diff=%.2e\n",
               picked[0], picked[1], picked[2], picked[3], picked[4], md);
    }

    // 收益（克制口径）：n_active=1 时桶(1) 40 个 program vs 满批图 320 个，省掉 padding 空 split/reduce。
    {
        prepare(1);
        const int W = 5, R = 50;
        auto time_exec = [&](int bi) {
            for (int w = 0; w < W; ++w) {
                CHECK(liteGraphLaunch(e[bi], stream));
                CHECK(liteStreamSynchronize(stream));
            }

            std::vector<double> v;
            for (int i = 0; i < R; ++i) {
                double t0 = now_ms();
                CHECK(liteGraphLaunch(e[bi], stream));
                CHECK(liteStreamSynchronize(stream));
                v.push_back(now_ms() - t0);
            }

            return compute_stats(v).p50;
        };
        double t1 = time_exec(bucket_idx(1)), t8 = time_exec(bucket_idx(8));
        printf("[Unit13] n_active=1: bucket(1) p50=%.4f ms (programs=%d) vs fixed-max(8) p50=%.4f ms (programs=%d), %.2fx\n",
               t1, 1 * H * NSPLIT + 1 * H, t8, MAX_SEQS * H * NSPLIT + MAX_SEQS * H, t8 / t1);
        printf("        注: 分桶省的是 padding 行空 split/reduce program 的固定发射/调度，属利用率叙事；\n"
               "            split-KV 单步极轻，3050 上差异可能接近噪声，非数量级加速。\n");
    }

    for (int i = 0; i < NB; ++i) {
        CHECK(liteGraphExecDestroy(e[i]));
        CHECK(liteGraphDestroy(g[i]));
    }
    CHECK(litePagedAttnDestroy(attn));
    CHECK(liteStreamDestroy(stream));
    printf("[Unit13] Batch-bucketed capture sizes + runtime bucket selection pass\n");
}

// ============================ Unit14: TP 半头切分 + SimComm all-gather ============================
// 张量并行（按 attention head 切）的本地单卡验证：
//   参考：一个全头(H=8) PagedAttn 跑出 ref_out；
//   TP=2：两个半头(H=4) PagedAttn 各算一半 head（rank0=head0..3, rank1=head4..7），投到
//         SimComm 的两条 rank 流上并行，all-gather 按 head 维拼回全头 tp_out；
//   断言 tp_out 与全头 GPU 参考逐元素一致（同 cubin、每 head 独立、输入位级一致，差异应极小），
//   并对 host fp32 oracle 校验绝对正确性。
// 单卡上两 rank 共享主 context（device 指针互通），跨卡链路用同卡 D2D+event 模拟；
// 真机多卡只需把 SimComm 换成 NCCL、每 rank 一个 device/context（L1，云上）。
static void run_test_tp_halfhead() {
    printf("\n========== Unit14: Tensor-Parallel (head-shard) + SimComm all-gather ==========\n");
    const int B = 4, H = 8, Hh = H / 2, D = 128, TPB = 16, MBPS = 8, PHYS = 40, NSPLIT = 4;
    const int slen[B] = {17, 3, 64, 40};
    const int HD = H * D, HDh = Hh * D;

    // ---- 全头 host paged 数据（权威输入，未写槽位填 5.0 反污染）----
    std::vector<uint16_t> kPool((size_t)PHYS * TPB * HD, f32_to_f16(5.0f));
    std::vector<uint16_t> vPool((size_t)PHYS * TPB * HD, f32_to_f16(5.0f));
    std::vector<uint16_t> q((size_t)B * HD, 0), refOut((size_t)B * HD, 0);
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
            if (t == slen[b] - 1) {  // decode：Q 仅当前 token
                std::memcpy(&q[(size_t)b * HD], hq.data(), HD * sizeof(uint16_t));
            }
        }
        sl[b] = slen[b];
    }
    const int usedBlocks = next;

    // ---- 参考：全头 PagedAttn（独立流，eager）----
    LiteStream sref{}; CHECK(liteStreamCreate(&sref));
    LitePagedAttnConfig rcfg{H, D, TPB, B, MBPS, PHYS, NSPLIT};
    LitePagedAttn ref{}; CHECK(litePagedAttnCreate(&rcfg, &ref));
    CHECK(litePagedAttnCopyKV(ref, kPool.data(), vPool.data()));
    CHECK(litePagedAttnCopyInput(ref, q.data(), bt.data(), sl.data()));
    CHECK(litePagedAttnLaunchB(ref, B, sref));
    CHECK(liteStreamSynchronize(sref));
    CHECK(litePagedAttnCopyOutput(ref, refOut.data()));

    // ---- 两半头 host 数据：按 head 维从全头切（rank r 取 head [r*Hh, r*Hh+Hh)）----
    std::vector<uint16_t> kH[2], vH[2], qH[2];
    for (int r = 0; r < 2; ++r) {
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

    // ---- 两半头 PagedAttn（H=4，HEADS 是运行时参数，无需重 AOT）----
    LitePagedAttn tp[2];
    LitePagedAttnConfig tcfg{Hh, D, TPB, B, MBPS, PHYS, NSPLIT};
    for (int r = 0; r < 2; ++r) {
        tp[r] = nullptr;
        CHECK(litePagedAttnCreate(&tcfg, &tp[r]));
        CHECK(litePagedAttnCopyKV(tp[r], kH[r].data(), vH[r].data()));
        CHECK(litePagedAttnCopyInput(tp[r], qH[r].data(), bt.data(), sl.data()));
    }

    // ---- SimComm：world=2，两 rank 各一条流，半头 kernel 并行后 all-gather 拼头 ----
    LiteCommConfig cc{/*world_size=*/2, /*device=*/0};
    LiteComm comm{}; CHECK(liteCommCreateSim(&cc, &comm));
    CHECK_EQ(liteCommWorldSize(comm) == 2, "u14 world=2");
    LiteStream rs0{}, rs1{};
    CHECK(liteCommRankStream(comm, 0, &rs0));
    CHECK(liteCommRankStream(comm, 1, &rs1));
    CHECK(litePagedAttnLaunchB(tp[0], B, rs0));
    CHECK(litePagedAttnLaunchB(tp[1], B, rs1));

    // all-gather 沿 head 维：每 rank 每个 b 贡献 Hh 个头（rowBytes），逐 b 行摆位，
    // 直接拼成完整 [B][H][D]（rank 轴并入 head 轴），与全头参考布局一致。
    const size_t rowBytes = (size_t)Hh * D * sizeof(uint16_t);   // 每 rank 每 b 半头行
    const size_t outBytes = (size_t)B * H * D * sizeof(uint16_t); // 完整全头 [B][H][D]
    CUdeviceptr dGather = 0;
    CHECK_EQ(cuMemAlloc(&dGather, outBytes) == CUDA_SUCCESS, "fail");
    const void* dIn[2] = {litePagedAttnOutPtr(tp[0]), litePagedAttnOutPtr(tp[1])};
    CHECK(liteCommAllGatherSim(comm, (void*)dGather, dIn, B, rowBytes));
    std::vector<uint16_t> tpOut((size_t)B * HD, 0);
    CHECK(liteMemcpyDtoHAsync(tpOut.data(), (void*)dGather, outBytes, rs0));  // join 在 rank0 流
    CHECK(liteStreamSynchronize(rs0));
    CHECK_EQ(cuMemFree(dGather) == CUDA_SUCCESS, "fail");

    // ---- 对拍：TP 拼头 vs 全头 GPU 参考；再对 host fp32 oracle ----
    double mdGpu = 0.0, mdOracle = 0.0;
    for (size_t i = 0; i < tpOut.size(); ++i) {
        mdGpu = std::max(mdGpu, (double)std::fabs(f16_to_f32(tpOut[i]) - f16_to_f32(refOut[i])));
    }

    for (int b = 0; b < B; ++b) {
        mdOracle = std::max(mdOracle, pa_oracle_slot(b, slen[b] - 1, &tpOut[(size_t)b * HD], H, D));
    }
    printf("[Unit14] world=2 head-shard H=%d->%d+%d, gather 拼头(r0:h0..%d, r1:h%d..%d)\n",
           H, Hh, Hh, Hh - 1, Hh, H - 1);
    printf("[Unit14] tp-gather vs full-head GPU ref : max_diff=%.2e (期望极小, 同算子 per-head 独立)\n", mdGpu);
    printf("[Unit14] tp-gather vs fp32 oracle       : max_diff=%.2e\n", mdOracle);
    fflush(stdout);
    CHECK_EQ(mdGpu < 5e-3, "u14 TP all-gather output == single-card full-head output");
    CHECK_EQ(mdOracle < 2e-2, "u14 TP output matches fp32 oracle");

    for (int r = 0; r < 2; ++r) {
        litePagedAttnDestroy(tp[r]);
    }
    litePagedAttnDestroy(ref);
    liteCommDestroy(comm);
    liteStreamDestroy(sref);
    printf("[Unit14] Tensor-Parallel head-shard + SimComm all-gather pass\n");
}

// ============================ main ============================
int main() {
    printf("==== Lite-GPU Runtime Test Suite (CUDA 12.6) ====\n");
    CHECK(liteInit());
    run_unit_tests();
    // 计时类 bench 全程不开 CUPTI：activity 记账对每次 launch 收税，
    // 对 eager（5000 次独立 launch）惩罚远大于 graph（1 次 launch），会虚高加速比。
    run_bench_launch_overhead();
    run_bench_native_submit();        // native 纯提交微基准（无背压），简历性能数字出自这里
    run_bench_hot_update_comparison();
    run_bench_real_gemm();
    run_test_preemption_recompute();
    run_test_prefix_caching();           // Unit16：prefix caching（block hash）+ COW + refcnt
    run_test_int8_kv();                  // Unit17：INT8 KV 量化（per-block-head scale）
    run_bench_paged_cb();                // Bench8：真 PagedAttention 端到端 continuous batching
    run_test_bucket_graph();             // Unit13：batch 分桶多图 + 运行时选桶（capture sizes）
    run_test_tp_halfhead();              // Unit14：TP 按 head 切分 + SimComm all-gather 拼头（单卡模拟多卡）
    run_profiler_attribution();       // 唯一插桩段：SM∩CE 交叠 + hotspot 归因，不用于性能结论
    CHECK(liteShutdown());
    printf("\n==== ALL TESTS COMPLETE ====\n");
    return EXIT_SUCCESS;
}
