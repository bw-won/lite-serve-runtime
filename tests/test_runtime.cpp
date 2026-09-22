#include "lite_gpu.h"
#include <cstdio>
#include <cstdlib>

static constexpr size_t PAGE_SIZE = 4 * 1024 * 1024; // 4MB

#define CHECK(call)                                                     \
    do {                                                                \
        LiteError err = call;                                           \
        if (err != LITE_SUCCESS) {                                      \
            fprintf(stderr, "Error at %s:%d code=%d\n", __FILE__, __LINE__, err); \
            exit(1);                                                    \
        }                                                               \
    } while (0)

int main() {
    printf("=== Lite GPU Runtime: Graph Benchmark (Fixed) ===\n");
    CHECK(liteInit());

    // 🔥 Page Size 设为 8MB，防止 launcher 中 4MB memset 越界
    CHECK(liteMemPoolInit(8 * 1024 * 1024, 10)); 

    LiteStream stream;
    CHECK(liteStreamCreate(&stream));

    void* ptr = nullptr;
    CHECK(liteMemAllocPage(&ptr, stream));

    // 🔥 使用带计时功能的 Event
    LiteEvent ev_start, ev_end;
    CHECK(liteEventCreateWithTiming(&ev_start));
    CHECK(liteEventCreateWithTiming(&ev_end));

    const int NUM_KERNELS = 100; 
    const int WARMUP_ITERS = 20;
    const int BENCH_ITERS = 100;

    // ==========================================
    // 1. Eager Mode Benchmark
    // ==========================================
    printf("[Benchmark 1] Eager Mode (%d kernels x %d iters)...\n", NUM_KERNELS, BENCH_ITERS);
    
    // Warmup
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        for (int k = 0; k < NUM_KERNELS; ++k) {
            liteLaunchDummyKernel(ptr, PAGE_SIZE, stream);
        }
    }
    CHECK(liteStreamSynchronize(stream)); 

    // 正式测时 (仅测量 GPU 侧执行时间)
    CHECK(liteEventRecord(ev_start, stream));
    for (int i = 0; i < BENCH_ITERS; ++i) {
        for (int k = 0; k < NUM_KERNELS; ++k) {
            liteLaunchDummyKernel(ptr, PAGE_SIZE, stream);
        }
    }
    CHECK(liteEventRecord(ev_end, stream));
    CHECK(liteStreamSynchronize(stream)); 

    float eager_gpu_ms = 0;
    // 🔥 修复：使用封装好的 API 替代直接访问 ->handle
    CHECK(liteEventElapsedTime(&eager_gpu_ms, ev_start, ev_end));
    printf("  Eager GPU Time: %.2f ms | Per-Iter: %.3f ms\n", eager_gpu_ms, eager_gpu_ms / BENCH_ITERS);

    // ==========================================
    // 2. CUDA Graph Mode Benchmark
    // ==========================================
    printf("[Benchmark 2] CUDA Graph Mode (%d kernels x %d iters)...\n", NUM_KERNELS, BENCH_ITERS);

    // Step A: Capture
    LiteGraph graph;
    CHECK(liteStreamBeginCapture(stream));
    for (int k = 0; k < NUM_KERNELS; ++k) {
        liteLaunchDummyKernel(ptr, PAGE_SIZE, stream); 
    }
    CHECK(liteStreamEndCapture(stream, &graph));

    // Step B: Instantiate
    LiteGraphExec exec;
    CHECK(liteGraphInstantiate(graph, &exec));

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        liteGraphLaunch(exec, stream);
    }
    CHECK(liteStreamSynchronize(stream)); 

    // Step C: Launch & 测时
    CHECK(liteEventRecord(ev_start, stream));
    for (int i = 0; i < BENCH_ITERS; ++i) {
        liteGraphLaunch(exec, stream); 
    }
    CHECK(liteEventRecord(ev_end, stream));
    CHECK(liteStreamSynchronize(stream));

    float graph_gpu_ms = 0;
    // 🔥 修复：使用封装好的 API 替代直接访问 ->handle
    CHECK(liteEventElapsedTime(&graph_gpu_ms, ev_start, ev_end));
    printf("  Graph GPU Time: %.2f ms | Per-Iter: %.3f ms\n", graph_gpu_ms, graph_gpu_ms / BENCH_ITERS);

    // ==========================================
    // 3. 结果对比
    // ==========================================
    printf("\n=== Result ===\n");
    if (graph_gpu_ms > 0) {
        printf("Speedup: %.2fx\n", eager_gpu_ms / graph_gpu_ms);
    }

    // Cleanup
    CHECK(liteEventDestroy(ev_start));
    CHECK(liteEventDestroy(ev_end));
    CHECK(liteGraphExecDestroy(exec));
    CHECK(liteGraphDestroy(graph));
    CHECK(liteMemFreePage(ptr, stream));
    CHECK(liteStreamDestroy(stream));
    CHECK(liteMemPoolDestroy());
    CHECK(liteShutdown());

    printf("=== Test Finished ===\n");
    return 0;
}
