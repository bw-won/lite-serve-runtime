#include "lite_gpu.h"

#include <cupti.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

namespace {
uint64_t g_kernel_ns  = 0;
uint64_t g_memcpy_ns = 0;
uint64_t g_kernel_cnt = 0;
uint64_t g_memcpy_cnt = 0;
bool     g_profiling  = false;

struct KernelStat { uint64_t total_ns = 0; uint64_t calls = 0; };
std::map<std::string, KernelStat> g_kernel_map;

std::vector<std::pair<uint64_t,uint64_t>> g_sm_iv;
std::vector<std::pair<uint64_t,uint64_t>> g_ce_iv;
}

static void CUPTIAPI prof_buffer_requested(uint8_t** buffer, size_t* size,
                                           size_t* maxNumRecords) {
    const size_t kBuf = 1u << 20;
    *buffer = reinterpret_cast<uint8_t*>(malloc(kBuf));
    *size   = kBuf;
    *maxNumRecords = 0;
}

static void CUPTIAPI prof_buffer_completed(CUcontext /*ctx*/, uint32_t /*flag*/,
                                           uint8_t* buffer, size_t size,
                                           size_t validSize) {
    CUpti_Activity* rec = nullptr;
    while (cuptiActivityGetNextRecord(buffer, validSize, &rec) == CUPTI_SUCCESS) {
        if (rec->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
            const CUpti_ActivityKernel4* k =
                reinterpret_cast<const CUpti_ActivityKernel4*>(rec);
            uint64_t dur = (uint64_t)(k->end - k->start);
            g_kernel_ns += dur;
            g_kernel_cnt++;
            const char* kname = (k->name != nullptr) ? k->name : "(unnamed)";
            g_kernel_map[kname].total_ns += dur;
            g_kernel_map[kname].calls += 1;
            g_sm_iv.push_back({(uint64_t)k->start, (uint64_t)k->end});
        } else if (rec->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
            const CUpti_ActivityMemcpy* m =
                reinterpret_cast<const CUpti_ActivityMemcpy*>(rec);
            uint64_t dur = (uint64_t)(m->end - m->start);
            g_memcpy_ns += dur;
            g_memcpy_cnt++;
            g_ce_iv.push_back({(uint64_t)m->start, (uint64_t)m->end});
        }
    }

    free(buffer);
}

LiteError liteProfilerStart() {
    if (g_profiling) {
        return LITE_SUCCESS;
    }

    CUptiResult r1 = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
    CUptiResult r2 = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    cuptiActivityRegisterCallbacks(prof_buffer_requested, prof_buffer_completed);
    g_profiling = true;
    printf("[lite_profiler] started (kernel + memcpy activity, hotspot + overlap on)\n");
    return (r1 == CUPTI_SUCCESS && r2 == CUPTI_SUCCESS) ? LITE_SUCCESS : LITE_ERROR_DRIVER_API;
}

LiteError liteProfilerStop() {
    if (!g_profiling) {
        return LITE_SUCCESS;
    }

    cuptiActivityFlushAll(0);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);
    g_profiling = false;
    return LITE_SUCCESS;
}

LiteError liteProfilerReset() {
    if (g_profiling) {
        cuptiActivityFlushAll(0);
    }

    g_kernel_ns = g_memcpy_ns = 0;
    g_kernel_cnt = g_memcpy_cnt = 0;
    g_kernel_map.clear();
    g_sm_iv.clear();
    g_ce_iv.clear();
    return LITE_SUCCESS;
}

LiteError liteProfilerResetRegions() {
    g_sm_iv.clear();
    g_ce_iv.clear();
    if (g_profiling) {
        cuptiActivityFlushAll(0);
    }

    g_sm_iv.clear();
    g_ce_iv.clear();
    return LITE_SUCCESS;
}

static uint64_t union_len(std::vector<std::pair<uint64_t,uint64_t>> v) {
    if (v.empty()) {
        return 0;
    }

    std::sort(v.begin(), v.end());
    uint64_t total = 0, cs = v[0].first, ce = v[0].second;
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i].first <= ce) {
            if (v[i].second > ce) {
                ce = v[i].second;
            }
        } else {
            total += ce - cs;
            cs = v[i].first;
            ce = v[i].second;
        }
    }

    total += ce - cs;
    return total;
}

static uint64_t intersection_len(std::vector<std::pair<uint64_t,uint64_t>> a,
                                 std::vector<std::pair<uint64_t,uint64_t>> b) {
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    uint64_t ov = 0;
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        uint64_t lo = std::max(a[i].first, b[j].first);
        uint64_t hi = std::min(a[i].second, b[j].second);
        if (lo < hi) {
            ov += hi - lo;
        }

        if (a[i].second < b[j].second) {
            ++i;
        } else {
            ++j;
        }
    }

    return ov;
}

void liteProfilerReportOverlap() {
    if (!g_profiling) {
        return;
    }

    cuptiActivityFlushAll(0);  // 确保区间到齐
    double sm_ms = union_len(g_sm_iv) / 1e6;
    double ce_ms = union_len(g_ce_iv) / 1e6;
    double ov_ms = intersection_len(g_sm_iv, g_ce_iv) / 1e6;
    printf("\n  --- Dual-Engine Overlap (CUPTI 时间区间实测) ---\n");
    printf("    SM(kernel) 忙碌并集 : %8.3f ms  (%zu intervals)\n", sm_ms, g_sm_iv.size());
    printf("    CE(memcpy)  忙碌并集 : %8.3f ms  (%zu intervals)\n", ce_ms, g_ce_iv.size());
    printf("    SM∩CE 交叠时间       : %8.3f ms\n", ov_ms);
    if (std::max(sm_ms, ce_ms) > 0) {
        printf("    重叠利用率(交叠/max) : %6.1f%%\n",
               100.0 * ov_ms / std::max(sm_ms, ce_ms));
    }

    printf("    -> 交叠>0 即证明 SM 与 copy engine 真在并行，不依赖 wall-clock 推断。\n");
}

void liteProfilerReport() {
    if (!g_profiling) {
        return;
    }

    cuptiActivityFlushAll(0);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);
    g_profiling = false;

    double kernel_ms = g_kernel_ns / 1e6;
    double memcpy_ms = g_memcpy_ns / 1e6;
    printf("\n===== lite_profiler: GPU-side breakdown =====\n");
    printf("  GPU kernel time   : %10.3f ms  (%llu launches)\n", kernel_ms,
           (unsigned long long)g_kernel_cnt);
    printf("  GPU memcpy time   : %10.3f ms  (%llu ops)\n", memcpy_ms,
           (unsigned long long)g_memcpy_cnt);
    double total = kernel_ms + memcpy_ms;
    if (total > 0) {
        printf("  compute fraction  : %6.1f%%\n", 100.0 * kernel_ms / total);
        printf("  copy    fraction  : %6.1f%%\n", 100.0 * memcpy_ms / total);
    }

    if (!g_kernel_map.empty()) {
        std::vector<std::pair<std::string, KernelStat>> v(g_kernel_map.begin(), g_kernel_map.end());
        std::sort(v.begin(), v.end(),
                  [](const std::pair<std::string, KernelStat>& a,
                     const std::pair<std::string, KernelStat>& b) {
                      return a.second.total_ns > b.second.total_ns;
                  });
        const int TOP = 10;
        printf("\n  --- Top %d kernels by GPU time ---\n", TOP);
        printf("  %-46s %10s %8s %8s\n", "kernel", "tot_ms", "calls", "avg_us");
        int n = 0;
        for (auto& e : v) {
            if (n++ >= TOP) {
                break;
            }

            double tot_ms = e.second.total_ns / 1e6;
            double avg_us = (e.second.calls > 0) ? (e.second.total_ns / 1e3 / e.second.calls) : 0.0;
            const char* nm = e.first.c_str();
            char shortname[47];
            size_t len = strlen(nm);
            if (len >= sizeof(shortname)) {
                memcpy(shortname, nm, sizeof(shortname) - 4);
                shortname[sizeof(shortname) - 4] = '.';
                shortname[sizeof(shortname) - 3] = '.';
                shortname[sizeof(shortname) - 2] = '.';
                shortname[sizeof(shortname) - 1] = '\0';
                nm = shortname;
            }

            printf("  %-46s %10.3f %8llu %8.1f\n", nm, tot_ms,
                   (unsigned long long)e.second.calls, avg_us);
        }

        printf("  (distinct kernels: %zu)\n", g_kernel_map.size());
    }

    printf("=============================================\n\n");
}
