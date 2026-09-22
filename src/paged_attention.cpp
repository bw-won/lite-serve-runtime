// paged_attention.cpp — Triton AOT cubin 算子加载 + split-KV PagedAttention 两段执行。
//
// 构建期必须先由 tools/aot_paged_attention.py 把两个 Triton kernel AOT 编成 cubin 并生成
// kernels/paged_attn_meta.h（cubin 字节数组 + 符号名 + shared/线程元数据）；该头是强制构建
// 产物，缺失则本文件直接编译失败（fail-fast，不提供"无 cubin 也能编过"的降级路径）。
//
// 运行时纯 Driver API：cuModuleLoadData 加载 -> cuModuleGetFunction 取函数 ->
// cuFuncSetAttribute(动态 shared) -> cuLaunchKernel(split) -> cuLaunchKernel(reduce)。
//
// capture-safe：Q / paged K/V / block-table / seq_lens / O_partial / LSE / O 全部在
// Create 时一次性 cuMemAlloc，地址恒定；每步只在捕获外经 cuMemcpyHtoD 改【内容】。
//
// 参数 ABI：必须与 tools/aot_paged_attention.py 的 warmup 实参严格同序、同类型：
//   split  = 7 指针(8B) + 18 stride(int32) + 3 标量(int32) = 28
//   reduce = 3 指针(8B) + 10 stride(int32) + 1 标量(int32) = 14
// （stride 用 int32 与 Triton JIT 对 python int 的默认推导一致；本项目最大 stride≈13 万元素，不溢出）
#include "lite_gpu.h"
#include "lite_gpu_internal.h"
#include "paged_attn_meta.h"
#include <cuda.h>
#include <cstring>
#include <new>

struct LitePagedAttn_s {
    LitePagedAttnConfig cfg;
    int num_splits;
    int split_size;
    int is_i8;                       // int8 KV 模式（决定 cubin / 数据通路 / D2D 字节数）

    CUmodule mod_split, mod_reduce;
    CUfunction f_split, f_reduce;
    int split_shared, reduce_shared;

    // 固定地址 device buffer
    CUdeviceptr d_q, d_k, d_v, d_bt, d_seq;
    CUdeviceptr d_sk, d_sv;          // int8 模式：per-(block,head) scale，[P][H] fp32
    CUdeviceptr d_opart, d_lse, d_out;

    size_t kv_bytes, q_bytes, bt_bytes, seq_bytes, out_bytes;
    size_t sk_bytes;                 // int8 模式下单侧 scale 总字节数（P*H*4），fp16 模式 0
};

static LiteError load_func(CUmodule* mod, const unsigned char* cubin,
                           CUfunction* fn, const char* symbol) {
    CUresult e = cuModuleLoadData((CUmodule*)mod, cubin);
    if (e != CUDA_SUCCESS) {
        return LITE_ERROR_DRIVER_API;
    }

    e = cuModuleGetFunction(fn, *mod, symbol);
    if (e != CUDA_SUCCESS) {
        return LITE_ERROR_DRIVER_API;
    }

    return LITE_SUCCESS;
}

static LiteError optin_shared(CUfunction fn, int shared_bytes) {
    // Triton pipeline 常需 >48KB 动态 shared；sm_86 单 block 上限约 99KB，需显式 opt-in。
    if (shared_bytes > 0) {
        CUresult e = cuFuncSetAttribute(fn,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_bytes);
        if (e != CUDA_SUCCESS) {
            return LITE_ERROR_DRIVER_API;
        }
    }
    return LITE_SUCCESS;
}

LiteError litePagedAttnCreate(const LitePagedAttnConfig* cfg, LitePagedAttn* out) {
    if (!cfg || !out) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if (cfg->head_dim != LITE_PA_DIM || cfg->block_size != LITE_PA_BLOCK_SIZE) {
        return LITE_ERROR_INVALID_VALUE;  // AOT constexpr 固化，改形状需重跑 AOT
    }

    if (cfg->num_heads <= 0 || cfg->num_heads > LITE_PA_MAX_HEADS ||
        cfg->max_batch <= 0 || cfg->max_batch > LITE_PA_MAX_BATCH ||
        cfg->max_logical_blocks <= 0 || cfg->num_phys_blocks <= 0) {
        return LITE_ERROR_INVALID_VALUE;
    }

    LitePagedAttn_s* a = new (std::nothrow) LitePagedAttn_s();
    if (!a) {
        return LITE_ERROR_OUT_OF_MEMORY;
    }

    a->cfg = *cfg;

    const int B = cfg->max_batch, H = cfg->num_heads, D = cfg->head_dim;
    const int BS = cfg->block_size, MLB = cfg->max_logical_blocks, P = cfg->num_phys_blocks;

    // num_splits：显式指定优先；否则按 SM 数 *4 的目标程序数/(B*H)，clamp 到 [1, MLB]。
    int ns = cfg->num_splits;
    if (ns <= 0) {
        int sm = 1;
        CUdevice dev;
        if (cuCtxGetDevice(&dev) == CUDA_SUCCESS) {
            cuDeviceGetAttribute(&sm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
        }

        int target = sm * 4;
        ns = target / (B * H);
        if (ns < 1) {
            ns = 1;
        }
        if (ns > MLB) {
            ns = MLB;
        }
    }

    if (ns > LITE_PA_MAX_SPLITS) {
        ns = LITE_PA_MAX_SPLITS;
    }

    a->num_splits = ns;
    // SPLIT_SIZE 按【最大可能长度】固定，使 launch 形状/图恒定；短请求的多余 split 由
    // kernel 自己写 LSE sentinel（per-request seq_lens 截断）。
    a->split_size = (MLB * BS + ns - 1) / ns;

    // int8 KV 模式：加载 i8 变体 split cubin（多 scale 指针/stride 的不同 ABI），
    // K/V 数据字节数减半，另配 per-(block,head) scale 缓冲。reduce 不变（输入是 fp16 O_partial）。
    a->is_i8 = cfg->int8_kv ? 1 : 0;
    const unsigned char* split_cubin =
        a->is_i8 ? kPagedAttnSplitI8Cubin : kPagedAttnSplitCubin;
    const char* split_symbol = a->is_i8 ? LITE_PA_SPLIT_I8_SYMBOL : LITE_PA_SPLIT_SYMBOL;

    LiteError le = load_func(&a->mod_split, split_cubin, &a->f_split, split_symbol);
    if (le != LITE_SUCCESS) {
        delete a;
        return le;
    }

    le = load_func(&a->mod_reduce, kPagedAttnReduceCubin,
                   &a->f_reduce, LITE_PA_REDUCE_SYMBOL);
    if (le != LITE_SUCCESS) {
        delete a;
        return le;
    }

    a->split_shared = a->is_i8 ? LITE_PA_SPLIT_I8_SHARED : LITE_PA_SPLIT_SHARED;
    a->reduce_shared = LITE_PA_REDUCE_SHARED;
    if (optin_shared(a->f_split, a->split_shared) != LITE_SUCCESS ||
        optin_shared(a->f_reduce, a->reduce_shared) != LITE_SUCCESS) {
        delete a;
        return LITE_ERROR_DRIVER_API;
    }

    a->q_bytes   = (size_t)B * H * D * 2;
    // KV 数据：fp16 每 elem 2B；int8 每 elem 1B（per-phys 布局不变，反量化交给 kernel）
    a->kv_bytes  = (size_t)P * BS * H * D * (a->is_i8 ? 1 : 2);
    a->sk_bytes  = a->is_i8 ? (size_t)P * H * sizeof(float) : 0;
    a->bt_bytes  = (size_t)B * MLB * sizeof(int);
    a->seq_bytes = (size_t)B * sizeof(int);
    a->out_bytes = (size_t)B * H * D * 2;
    size_t opart_bytes = (size_t)B * H * ns * D * 2;
    size_t lse_bytes   = (size_t)B * H * ns * sizeof(float);

    bool ok = true;
    ok &= cuMemAlloc(&a->d_q,     a->q_bytes)   == CUDA_SUCCESS;
    ok &= cuMemAlloc(&a->d_k,     a->kv_bytes)  == CUDA_SUCCESS;
    ok &= cuMemAlloc(&a->d_v,     a->kv_bytes)  == CUDA_SUCCESS;
    ok &= cuMemAlloc(&a->d_bt,    a->bt_bytes)  == CUDA_SUCCESS;
    ok &= cuMemAlloc(&a->d_seq,   a->seq_bytes) == CUDA_SUCCESS;
    ok &= cuMemAlloc(&a->d_opart, opart_bytes)  == CUDA_SUCCESS;
    ok &= cuMemAlloc(&a->d_lse,   lse_bytes)    == CUDA_SUCCESS;
    ok &= cuMemAlloc(&a->d_out,   a->out_bytes) == CUDA_SUCCESS;
    if (a->is_i8) {
        ok &= cuMemAlloc(&a->d_sk, a->sk_bytes) == CUDA_SUCCESS;
        ok &= cuMemAlloc(&a->d_sv, a->sk_bytes) == CUDA_SUCCESS;
    } else {
        a->d_sk = a->d_sv = 0;
    }

    if (!ok) {
        litePagedAttnDestroy(a);
        return LITE_ERROR_OUT_OF_MEMORY;
    }

    // 未用槽位 seq_lens=0（reduce 对全 sentinel 输出 0）；block-table 默认清零。
    int zero = 0;
    cuMemcpyHtoD(a->d_seq, &zero, sizeof(int));
    cuMemsetD8(a->d_bt, 0, a->bt_bytes);

    *out = a;
    return LITE_SUCCESS;
}

LiteError litePagedAttnLaunchB(LitePagedAttn a, int batch, LiteStream stream) {
    if (!a || !stream) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if (batch < 1 || batch > a->cfg.max_batch) {
        return LITE_ERROR_INVALID_VALUE;
    }

    const int H = a->cfg.num_heads, D = a->cfg.head_dim, BS = a->cfg.block_size;
    const int MLB = a->cfg.max_logical_blocks, B = batch, S = a->num_splits;

    // ---- 设备指针参数（CUdeviceptr，8 字节），args 元素指向其地址 ----
    CUdeviceptr pQ = a->d_q, pK = a->d_k, pV = a->d_v, pBT = a->d_bt, pSeq = a->d_seq;
    CUdeviceptr pOpart = a->d_opart, pLse = a->d_lse;

    // ---- stride（元素单位，与 torch .stride() 一致；i32，与 Triton JIT 默认相同）----
    int sqb = H * D, sqh = D, sqd = 1;
    int skb = BS * H * D, skbs = H * D, skh = D, skd = 1;
    int svb = skb, svbs = skbs, svh = D, svd = 1;
    int sob = H * S * D, soh = S * D, sos = D, sod = 1;
    int slb = H * S, slh = S, sls = 1;
    int soutb = H * D, south = D, soutd = 1;
    int heads = H, split_size = a->split_size, max_lb = MLB, num_splits = S;

    CUresult e;
    if (a->is_i8) {
        // int8 split kernel：顺序严格对齐 SPLIT_I8_SIG（9 ptr + 20 stride + 3 i32 = 32）
        CUdeviceptr pSk = a->d_sk, pSv = a->d_sv;
        int sskb = H, ssvb = H;   // scale [P][H] 布局的 per-phys stride
        void* args_split_i8[32] = {
            &pQ, &pK, &pV, &pBT, &pSeq, &pSk, &pSv, &pOpart, &pLse,
            &sqb, &sqh, &sqd,
            &skb, &skbs, &skh, &skd,
            &svb, &svbs, &svh, &svd,
            &sskb, &ssvb,
            &sob, &soh, &sos, &sod,
            &slb, &slh, &sls,
            &heads, &split_size, &max_lb,
        };
        e = cuLaunchKernel(
            a->f_split,
            (unsigned)(B * H), (unsigned)S, 1,
            LITE_PA_SPLIT_I8_THREADS, 1, 1,
            (unsigned)a->split_shared, stream->handle, args_split_i8, nullptr);
    } else {
        // fp16 split kernel：顺序严格对齐 SPLIT_SIG（7 ptr + 18 stride + 3 i32 = 28）
        void* args_split[28] = {
            &pQ, &pK, &pV, &pBT, &pSeq, &pOpart, &pLse,
            &sqb, &sqh, &sqd,
            &skb, &skbs, &skh, &skd,
            &svb, &svbs, &svh, &svd,
            &sob, &soh, &sos, &sod,
            &slb, &slh, &sls,
            &heads, &split_size, &max_lb,
        };
        e = cuLaunchKernel(
            a->f_split,
            (unsigned)(B * H), (unsigned)S, 1,
            LITE_PA_SPLIT_THREADS, 1, 1,
            (unsigned)a->split_shared, stream->handle, args_split, nullptr);
    }

    if (e != CUDA_SUCCESS) {
        return LITE_ERROR_DRIVER_API;
    }

    // reduce kernel：顺序严格对齐 REDUCE_SIG（3 ptr + 10 stride + 1 i32 = 14）
    CUdeviceptr pOut = a->d_out;
    void* args_reduce[14] = {
        &pOpart, &pLse, &pOut,
        &sob, &soh, &sos, &sod,
        &slb, &slh, &sls,
        &soutb, &south, &soutd,
        &num_splits,
    };
    e = cuLaunchKernel(
        a->f_reduce,
        (unsigned)B, (unsigned)H, 1,
        LITE_PA_REDUCE_THREADS, 1, 1,
        (unsigned)a->reduce_shared, stream->handle, args_reduce, nullptr);
    return (e == CUDA_SUCCESS) ? LITE_SUCCESS : LITE_ERROR_DRIVER_API;
}

LiteError litePagedAttnLaunch(LitePagedAttn a, LiteStream stream) {
    if (!a || !stream) {
        return LITE_ERROR_INVALID_VALUE;
    }

    // 默认按 max_batch 启动（定长满批图）；分桶场景改用 litePagedAttnLaunchB 指定有效行数。
    return litePagedAttnLaunchB(a, a->cfg.max_batch, stream);
}

LiteError litePagedAttnDestroy(LitePagedAttn a) {
    if (!a) {
        return LITE_SUCCESS;
    }

    if (a->d_out) {
        cuMemFree(a->d_out);
    }

    if (a->d_lse) {
        cuMemFree(a->d_lse);
    }

    if (a->d_opart) {
        cuMemFree(a->d_opart);
    }

    if (a->d_sk) {
        cuMemFree(a->d_sk);
    }

    if (a->d_sv) {
        cuMemFree(a->d_sv);
    }

    if (a->d_seq) {
        cuMemFree(a->d_seq);
    }

    if (a->d_bt) {
        cuMemFree(a->d_bt);
    }

    if (a->d_v) {
        cuMemFree(a->d_v);
    }

    if (a->d_k) {
        cuMemFree(a->d_k);
    }

    if (a->d_q) {
        cuMemFree(a->d_q);
    }

    if (a->mod_reduce) {
        cuModuleUnload(a->mod_reduce);
    }

    if (a->mod_split) {
        cuModuleUnload(a->mod_split);
    }

    delete a;
    return LITE_SUCCESS;
}

LiteError litePagedAttnCopyKV(LitePagedAttn a, const void* h_k, const void* h_v) {
    if (!a || !h_k || !h_v) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if (a->is_i8) {
        return LITE_ERROR_INVALID_VALUE;   // int8 实例请走 CopyBlockI8 通路
    }

    if (cuMemcpyHtoD(a->d_k, h_k, a->kv_bytes) != CUDA_SUCCESS ||
        cuMemcpyHtoD(a->d_v, h_v, a->kv_bytes) != CUDA_SUCCESS) {
        return LITE_ERROR_DRIVER_API;
    }

    return LITE_SUCCESS;
}

// 当前实例的单侧物理块字节数（fp16 / int8 由模式决定）
static size_t pa_blk_bytes(const LitePagedAttn_s* a) {
    return (size_t)a->cfg.block_size * a->cfg.num_heads * a->cfg.head_dim
           * (a->is_i8 ? 1 : 2);
}

LiteError litePagedAttnCopyBlock(LitePagedAttn a, int phys,
                                 const void* h_k_blk, const void* h_v_blk) {
    if (!a) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if (a->is_i8) {
        return LITE_ERROR_INVALID_VALUE;   // 数据通路类型不同，fail-fast
    }

    if (phys < 0 || phys >= a->cfg.num_phys_blocks) {
        return LITE_ERROR_INVALID_VALUE;
    }

    // 一个物理块 = block_size*num_heads*head_dim 个 fp16（K/V 各一块连续缓冲）。
    const size_t blk_bytes = pa_blk_bytes(a);
    const size_t off = (size_t)phys * blk_bytes;
    if (h_k_blk && cuMemcpyHtoD(a->d_k + off, h_k_blk, blk_bytes) != CUDA_SUCCESS) {
        return LITE_ERROR_DRIVER_API;
    }

    if (h_v_blk && cuMemcpyHtoD(a->d_v + off, h_v_blk, blk_bytes) != CUDA_SUCCESS) {
        return LITE_ERROR_DRIVER_API;
    }

    return LITE_SUCCESS;
}

LiteError litePagedAttnCopyBlockI8(LitePagedAttn a, int phys,
                                   const void* h_k_blk, const void* h_v_blk,
                                   const float* h_scale_k, const float* h_scale_v) {
    if (!a) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if (!a->is_i8) {
        return LITE_ERROR_INVALID_VALUE;  // fp16 实例请走 CopyBlock
    }

    if (phys < 0 || phys >= a->cfg.num_phys_blocks) {
        return LITE_ERROR_INVALID_VALUE;
    }

    const size_t blk_bytes = pa_blk_bytes(a);
    const size_t off = (size_t)phys * blk_bytes;
    const size_t sk_off = (size_t)phys * a->cfg.num_heads * sizeof(float);
    if (h_k_blk && cuMemcpyHtoD(a->d_k + off, h_k_blk, blk_bytes) != CUDA_SUCCESS) {
        return LITE_ERROR_DRIVER_API;
    }

    if (h_v_blk && cuMemcpyHtoD(a->d_v + off, h_v_blk, blk_bytes) != CUDA_SUCCESS) {
        return LITE_ERROR_DRIVER_API;
    }

    if (h_scale_k && cuMemcpyHtoD(a->d_sk + sk_off, h_scale_k,
                                  (size_t)a->cfg.num_heads * sizeof(float)) != CUDA_SUCCESS) {
        return LITE_ERROR_DRIVER_API;
    }

    if (h_scale_v && cuMemcpyHtoD(a->d_sv + sk_off, h_scale_v,
                                  (size_t)a->cfg.num_heads * sizeof(float)) != CUDA_SUCCESS) {
        return LITE_ERROR_DRIVER_API;
    }

    return LITE_SUCCESS;
}

LiteError litePagedAttnCopyBlockD2D(LitePagedAttn a, int dst_phys, int src_phys,
                                    LiteStream stream) {
    if (!a || !stream) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if (dst_phys < 0 || dst_phys >= a->cfg.num_phys_blocks ||
        src_phys < 0 || src_phys >= a->cfg.num_phys_blocks) {
        return LITE_ERROR_INVALID_VALUE;
    }

    const size_t blk_bytes = pa_blk_bytes(a);
    const size_t dst_off = (size_t)dst_phys * blk_bytes;
    const size_t src_off = (size_t)src_phys * blk_bytes;
    // K/V 各一块整块拷贝（COW：把共享旧块内容复制进新私有块）；int8 模式还要连
    // per-(block,head) scale 一起拷——否则 COW 新块反量化出的数值就是错的。
    if (cuMemcpyDtoDAsync(a->d_k + dst_off, a->d_k + src_off, blk_bytes,
                          stream->handle) != CUDA_SUCCESS ||
        cuMemcpyDtoDAsync(a->d_v + dst_off, a->d_v + src_off, blk_bytes,
                          stream->handle) != CUDA_SUCCESS) {
        return LITE_ERROR_DRIVER_API;
    }

    if (a->is_i8) {
        const size_t sk_blk = (size_t)a->cfg.num_heads * sizeof(float);
        const size_t sk_dst = (size_t)dst_phys * sk_blk;
        const size_t sk_src = (size_t)src_phys * sk_blk;
        if (cuMemcpyDtoDAsync(a->d_sk + sk_dst, a->d_sk + sk_src, sk_blk,
                              stream->handle) != CUDA_SUCCESS ||
            cuMemcpyDtoDAsync(a->d_sv + sk_dst, a->d_sv + sk_src, sk_blk,
                              stream->handle) != CUDA_SUCCESS) {
            return LITE_ERROR_DRIVER_API;
        }
    }

    return LITE_SUCCESS;
}

LiteError litePagedAttnCopyInput(LitePagedAttn a, const void* h_q, const int* h_bt, const int* h_sl) {
    if (!a || !h_q || !h_bt || !h_sl) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if (cuMemcpyHtoD(a->d_q, h_q, a->q_bytes) != CUDA_SUCCESS ||
        cuMemcpyHtoD(a->d_bt, h_bt, a->bt_bytes) != CUDA_SUCCESS ||
        cuMemcpyHtoD(a->d_seq, h_sl, a->seq_bytes) != CUDA_SUCCESS) {
        return LITE_ERROR_DRIVER_API;
    }

    return LITE_SUCCESS;
}

LiteError litePagedAttnCopyOutput(LitePagedAttn a, void* h_out) {
    if (!a || !h_out) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if (cuMemcpyDtoH(h_out, a->d_out, a->out_bytes) != CUDA_SUCCESS) {
        return LITE_ERROR_DRIVER_API;
    }

    return LITE_SUCCESS;
}

int litePagedAttnNumSplits(LitePagedAttn a) {
    return a ? a->num_splits : 0;
}

int litePagedAttnSplitSize(LitePagedAttn a) {
    return a ? a->split_size : 0;
}

void* litePagedAttnOutPtr(LitePagedAttn a) {
    return a ? reinterpret_cast<void*>(a->d_out) : nullptr;
}

size_t litePagedAttnKvBytes(LitePagedAttn a) {
    if (!a) {
        return 0;
    }

    return 2 * a->kv_bytes + 2 * a->sk_bytes;   // K+V 数据，int8 模式含 K/V 两份 scale
}

LiteError liteQuantBlockFp16ToI8(const uint16_t* h_k_fp16, const uint16_t* h_v_fp16,
                                 int num_heads, int head_dim, int block_size,
                                 void* out_k_i8, void* out_v_i8,
                                 float* out_scale_k, float* out_scale_v) {
    if (!h_k_fp16 || !h_v_fp16 || !out_k_i8 || !out_v_i8 || !out_scale_k || !out_scale_v) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if (num_heads <= 0 || head_dim <= 0 || block_size <= 0) {
        return LITE_ERROR_INVALID_VALUE;
    }

    struct F16 {
        static float to(uint16_t h) {
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
                    m &= 0x3ff;
                    e++;
                    f = sign | ((e + 127 - 15) << 23) | (m << 13);
                }
            } else if (e == 31) {
                f = sign | 0x7f800000u | (m << 13);
            } else {
                f = sign | ((e + 127 - 15) << 23) | (m << 13);
            }
            float o;
            std::memcpy(&o, &f, 4);
            return o;
        }
    };

    const int HD = num_heads * head_dim;
    for (int side = 0; side < 2; ++side) {
        const uint16_t* src = side == 0 ? h_k_fp16 : h_v_fp16;
        int8_t* dst = side == 0 ? (int8_t*)out_k_i8 : (int8_t*)out_v_i8;
        float* scales = side == 0 ? out_scale_k : out_scale_v;

        for (int h = 0; h < num_heads; ++h) {
            float amax = 0.0f;
            for (int t = 0; t < block_size; ++t) {
                for (int d = 0; d < head_dim; ++d) {
                    float x = F16::to(src[(size_t)t * HD + (size_t)h * head_dim + d]);
                    float ax = x > 0 ? x : -x;
                    if (ax > amax) {
                        amax = ax;
                    }
                }
            }

            const float scale = amax > 0.0f ? amax / 127.0f : 1e-8f;
            scales[h] = scale;

            for (int t = 0; t < block_size; ++t) {
                for (int d = 0; d < head_dim; ++d) {
                    float x = F16::to(src[(size_t)t * HD + (size_t)h * head_dim + d]);
                    float q = x / scale;
                    int qi = (int)(q >= 0 ? q + 0.5f : q - 0.5f);
                    if (qi > 127) {
                        qi = 127;
                    }
                    if (qi < -127) {
                        qi = -127;
                    }
                    dst[(size_t)t * HD + (size_t)h * head_dim + d] = (int8_t)qi;
                }
            }
        }
    }

    return LITE_SUCCESS;
}
