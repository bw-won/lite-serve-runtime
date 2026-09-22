"""
split_kv_attention.py
Split-KV PagedAttention（per-request 变长版）—— Triton 算子源码 + 正确性自测。

本文件只保留被 C++ runtime 经 AOT 使用的两个 kernel：
  1. paged_attention_split_kernel_vl : 每个 (batch,head,split) 算一段 KV 的局部 attention，
                                       写 O_partial(fp16) + LSE_partial(fp32)；空 split 自写哨兵。
  2. paged_attention_reduce_kernel   : 跨 split 用 online softmax 归约出最终 O。

构建期由 tools/aot_paged_attention.py import 这两个 kernel 并 warmup 编成 cubin；
运行期 C++ 纯 Driver API 加载，不依赖 Python/Triton。

直接运行本脚本做 AOT 前的正确性自测（非连续物理块 + per-request 变长）：
  python3 tools/split_kv_attention.py
"""

import math

import torch
import triton
import triton.language as tl

# LSE 哨兵值：替代 -inf，避免 FP32 存储/运算产生 NaN 导致 reduce 错误
LSE_SENTINEL = -1e30
LSE_VALID_THRESHOLD = -1e29


# ==================== Kernel 1: Split-KV Attention（per-request 变长） ====================
# continuous batching 必需的两点：
#   1) Seq_Lens[b]（int32 指针）替代全局 SEQ_LEN 标量：一个 batch 内每个请求可有不同
#      上下文长度，valid mask 用 token_pos < seq_lens[b]（vLLM 标准做法）。
#   2) 空 split（split_start >= 该请求 seq_len）在返回前【自己写 LSE sentinel】，
#      不依赖 Python 端 LSE_partial.fill_(-1e30)，从而整图可被 CUDA Graph 反复 replay，
#      reduce kernel 用 LSE_THRESHOLD 跳过这些 split。
@triton.jit
def paged_attention_split_kernel_vl(
    Q, K_Cache, V_Cache, Block_Table, Seq_Lens,
    O_partial, LSE_partial,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kbs, stride_kh, stride_kd,
    stride_vb, stride_vbs, stride_vh, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    HEADS, SPLIT_SIZE, MAX_LOGICAL_BLOCKS,
    DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)

    b = pid_bh // HEADS
    h = pid_bh % HEADS

    seq_len = tl.load(Seq_Lens + b)  # per-request 变长，替代全局 SEQ_LEN
    split_start = pid_s * SPLIT_SIZE
    split_end = tl.minimum(split_start + SPLIT_SIZE, seq_len)

    if split_start >= seq_len:
        # 空 split：写 sentinel 后返回，保证 reduce 端阈值过滤、图 replay 自洽。
        lse_offs = b * stride_lb + h * stride_lh + pid_s * stride_ls
        tl.store(LSE_partial + lse_offs, -1.0e30)
        return

    d_offs = tl.arange(0, DIM)
    q = tl.load(Q + b * stride_qb + h * stride_qh + d_offs * stride_qd).to(tl.float32)
    scale = 1.0 / tl.sqrt(tl.cast(DIM, tl.float32))

    m_i = tl.full([], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([], dtype=tl.float32)
    acc = tl.zeros([DIM], dtype=tl.float32)

    num_tiles = (split_end - split_start + BLOCK_N - 1) // BLOCK_N
    for tile_idx in range(num_tiles):
        tile_start = split_start + tile_idx * BLOCK_N
        token_pos = tile_start + tl.arange(0, BLOCK_N)
        valid = token_pos < split_end  # split_end 已按该请求 seq_len 截断

        logical_idx = token_pos // BLOCK_SIZE
        slot = token_pos % BLOCK_SIZE

        phys_block = tl.load(
            Block_Table + b * MAX_LOGICAL_BLOCKS + logical_idx,
            mask=valid, other=0
        )

        k_offs = (phys_block[:, None] * stride_kb + slot[:, None] * stride_kbs
                  + h * stride_kh + d_offs[None, :] * stride_kd)
        v_offs = (phys_block[:, None] * stride_vb + slot[:, None] * stride_vbs
                  + h * stride_vh + d_offs[None, :] * stride_vd)

        k = tl.load(K_Cache + k_offs, mask=valid[:, None], other=0.0).to(tl.float32)
        v = tl.load(V_Cache + v_offs, mask=valid[:, None], other=0.0).to(tl.float32)

        scores = tl.sum(k * q[None, :], axis=1) * scale
        scores = tl.where(valid, scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)

        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    out_offs = b * stride_ob + h * stride_oh + pid_s * stride_os + d_offs * stride_od
    lse_offs = b * stride_lb + h * stride_lh + pid_s * stride_ls

    l_i_safe = tl.maximum(l_i, 1e-20)
    tl.store(O_partial + out_offs, (acc / l_i_safe).to(tl.float16))
    tl.store(LSE_partial + lse_offs, m_i + tl.log(l_i_safe))


# ==================== Kernel 1b: Split-KV Attention（INT8 KV 量化版） ====================
# KV 常驻显存为 int8（对称量化，per-(block, head) scale，K/V 各一份 [P,H] fp32）：
# 常驻减半，代价是 load 后多一次乘法。量化发生在写入侧（本项目的 K/V 生产者在 host；
# 真实 serving 中是 GEMM 输出处的 device 端量化，机制相同），kernel 只负责反量化：
#   x_deq = int8 * scale[phys, h]
# 反量化在 fp32 域进行（与 fp16 版的 .to(tl.float32) 对齐）。scale 必须按 token 所在
# 物理块逐 token 取——split_start 未对齐 BLOCK_N 时一个 tile 会跨两个物理块，
# scale 是 [BLOCK_N] 向量而非标量。
@triton.jit
def paged_attention_split_kernel_i8(
    Q, K_Cache, V_Cache, Block_Table, Seq_Lens, Scale_K, Scale_V,
    O_partial, LSE_partial,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kbs, stride_kh, stride_kd,
    stride_vb, stride_vbs, stride_vh, stride_vd,
    stride_skb, stride_svb,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    HEADS, SPLIT_SIZE, MAX_LOGICAL_BLOCKS,
    DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)

    b = pid_bh // HEADS
    h = pid_bh % HEADS

    seq_len = tl.load(Seq_Lens + b)
    split_start = pid_s * SPLIT_SIZE
    split_end = tl.minimum(split_start + SPLIT_SIZE, seq_len)

    if split_start >= seq_len:
        lse_offs = b * stride_lb + h * stride_lh + pid_s * stride_ls
        tl.store(LSE_partial + lse_offs, -1.0e30)
        return

    d_offs = tl.arange(0, DIM)
    q = tl.load(Q + b * stride_qb + h * stride_qh + d_offs * stride_qd).to(tl.float32)
    scale = 1.0 / tl.sqrt(tl.cast(DIM, tl.float32))

    m_i = tl.full([], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([], dtype=tl.float32)
    acc = tl.zeros([DIM], dtype=tl.float32)

    num_tiles = (split_end - split_start + BLOCK_N - 1) // BLOCK_N
    for tile_idx in range(num_tiles):
        tile_start = split_start + tile_idx * BLOCK_N
        token_pos = tile_start + tl.arange(0, BLOCK_N)
        valid = token_pos < split_end

        logical_idx = token_pos // BLOCK_SIZE
        slot = token_pos % BLOCK_SIZE

        phys_block = tl.load(
            Block_Table + b * MAX_LOGICAL_BLOCKS + logical_idx,
            mask=valid, other=0
        )

        # per-token 反量化 scale：该 token 所在物理块 × 当前 head
        sk = tl.load(Scale_K + phys_block * stride_skb + h, mask=valid, other=0.0)
        sv = tl.load(Scale_V + phys_block * stride_svb + h, mask=valid, other=0.0)

        k_offs = (phys_block[:, None] * stride_kb + slot[:, None] * stride_kbs
                  + h * stride_kh + d_offs[None, :] * stride_kd)
        v_offs = (phys_block[:, None] * stride_vb + slot[:, None] * stride_vbs
                  + h * stride_vh + d_offs[None, :] * stride_vd)

        k = tl.load(K_Cache + k_offs, mask=valid[:, None], other=0).to(tl.float32) * sk[:, None]
        v = tl.load(V_Cache + v_offs, mask=valid[:, None], other=0).to(tl.float32) * sv[:, None]

        scores = tl.sum(k * q[None, :], axis=1) * scale
        scores = tl.where(valid, scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)

        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    out_offs = b * stride_ob + h * stride_oh + pid_s * stride_os + d_offs * stride_od
    lse_offs = b * stride_lb + h * stride_lh + pid_s * stride_ls

    l_i_safe = tl.maximum(l_i, 1e-20)
    tl.store(O_partial + out_offs, (acc / l_i_safe).to(tl.float16))
    tl.store(LSE_partial + lse_offs, m_i + tl.log(l_i_safe))


# ==================== Kernel 2: Reduce Across Splits ====================
@triton.jit
def paged_attention_reduce_kernel(
    O_partial, LSE_partial, O,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    stride_out_b, stride_out_h, stride_out_d,
    NUM_SPLITS,
    DIM: tl.constexpr,
    LSE_THRESHOLD: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    d_offs = tl.arange(0, DIM)

    g_max = tl.full([], LSE_THRESHOLD, dtype=tl.float32)
    g_sum = tl.zeros([], dtype=tl.float32)
    g_acc = tl.zeros([DIM], dtype=tl.float32)

    for s in range(NUM_SPLITS):
        lse = tl.load(LSE_partial + pid_b * stride_lb + pid_h * stride_lh + s * stride_ls)

        if lse > LSE_THRESHOLD:
            new_max = tl.maximum(g_max, lse)
            alpha_old = tl.exp(g_max - new_max)
            alpha_new = tl.exp(lse - new_max)

            o_part = tl.load(O_partial + pid_b * stride_ob + pid_h * stride_oh
                             + s * stride_os + d_offs * stride_od).to(tl.float32)

            g_acc = g_acc * alpha_old + alpha_new * o_part
            g_sum = g_sum * alpha_old + alpha_new
            g_max = new_max

    out = g_acc / tl.maximum(g_sum, 1e-20)
    out_offs = pid_b * stride_out_b + pid_h * stride_out_h + d_offs * stride_out_d
    tl.store(O + out_offs, out.to(tl.float16))


def paged_attention_vl(Q, K_Cache, V_Cache, Block_Table, Seq_Lens,
                       block_size, num_splits, block_n=None):
    """per-request 变长 eager 入口（与 C++ AOT launch 严格同构，供正确性自测）。

    Q:[B,H,D] fp16；K/V_Cache:[P,block_size,H,D] fp16；Block_Table:[B,max_lb] int32；
    Seq_Lens:[B] int32（每请求各自长度）；SPLIT_SIZE 按批内【最大】长度固定，
    短请求的多余 split 由 kernel 自己写 LSE sentinel（图 replay 自洽）。
    """
    BATCH, HEADS, DIM = Q.shape
    if block_n is None:
        block_n = block_size
    max_seq = int(Seq_Lens.max().item())
    max_logical_blocks = Block_Table.shape[1]
    split_size = (max_seq + num_splits - 1) // num_splits

    O_partial = torch.empty(BATCH, HEADS, num_splits, DIM, device=Q.device, dtype=torch.float16)
    LSE_partial = torch.empty(BATCH, HEADS, num_splits, device=Q.device, dtype=torch.float32)
    LSE_partial.fill_(LSE_SENTINEL)  # eager 自测保留 fill；AOT/C++ 路径由 kernel 自写 sentinel
    O = torch.empty_like(Q)

    paged_attention_split_kernel_vl[(BATCH * HEADS, num_splits)](
        Q, K_Cache, V_Cache, Block_Table, Seq_Lens,
        O_partial, LSE_partial,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K_Cache.stride(0), K_Cache.stride(1), K_Cache.stride(2), K_Cache.stride(3),
        V_Cache.stride(0), V_Cache.stride(1), V_Cache.stride(2), V_Cache.stride(3),
        O_partial.stride(0), O_partial.stride(1), O_partial.stride(2), O_partial.stride(3),
        LSE_partial.stride(0), LSE_partial.stride(1), LSE_partial.stride(2),
        HEADS, split_size, max_logical_blocks,
        DIM, block_size, block_n,
        num_warps=4, num_stages=3,
    )
    paged_attention_reduce_kernel[(BATCH, HEADS)](
        O_partial, LSE_partial, O,
        O_partial.stride(0), O_partial.stride(1), O_partial.stride(2), O_partial.stride(3),
        LSE_partial.stride(0), LSE_partial.stride(1), LSE_partial.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        num_splits, DIM, LSE_VALID_THRESHOLD,
        num_warps=4,
    )
    return O


def quantize_kv_per_head(k_fp16, v_fp16):
    """per-(block, head) 对称量化（C++ 侧 liteQuantBlockFp16ToI8 同一方案）：
    scale = amax / 127（fp32），q = clamp(round(x/scale), -127, 127)（int8）。
    输入 [P, BS, H, D] fp16，返回 (k_i8, scale_k[P,H], v_i8, scale_v[P,H])。"""
    def q(x):
        amax = x.float().abs().amax(dim=(1, 3))              # [P, H]
        scale = (amax / 127.0).clamp_min(1e-8)
        xi = torch.clamp(torch.round(x.float() / scale[:, None, :, None]),
                         -127, 127).to(torch.int8)
        return xi, scale
    ki, sk = q(k_fp16)
    vi, sv = q(v_fp16)
    return ki, sk, vi, sv


def paged_attention_i8(Q, K_i8, V_i8, Scale_K, Scale_V, Block_Table, Seq_Lens,
                       block_size, num_splits, block_n=None):
    """INT8 KV eager 入口（与 C++ AOT i8 launch 严格同构，供正确性自测）。"""
    BATCH, HEADS, DIM = Q.shape
    if block_n is None:
        block_n = block_size
    max_seq = int(Seq_Lens.max().item())
    max_logical_blocks = Block_Table.shape[1]
    split_size = (max_seq + num_splits - 1) // num_splits

    O_partial = torch.empty(BATCH, HEADS, num_splits, DIM, device=Q.device, dtype=torch.float16)
    LSE_partial = torch.empty(BATCH, HEADS, num_splits, device=Q.device, dtype=torch.float32)
    LSE_partial.fill_(LSE_SENTINEL)
    O = torch.empty_like(Q)

    paged_attention_split_kernel_i8[(BATCH * HEADS, num_splits)](
        Q, K_i8, V_i8, Block_Table, Seq_Lens, Scale_K, Scale_V,
        O_partial, LSE_partial,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K_i8.stride(0), K_i8.stride(1), K_i8.stride(2), K_i8.stride(3),
        V_i8.stride(0), V_i8.stride(1), V_i8.stride(2), V_i8.stride(3),
        Scale_K.stride(0), Scale_V.stride(0),
        O_partial.stride(0), O_partial.stride(1), O_partial.stride(2), O_partial.stride(3),
        LSE_partial.stride(0), LSE_partial.stride(1), LSE_partial.stride(2),
        HEADS, split_size, max_logical_blocks,
        DIM, block_size, block_n,
        num_warps=4, num_stages=3,
    )
    paged_attention_reduce_kernel[(BATCH, HEADS)](
        O_partial, LSE_partial, O,
        O_partial.stride(0), O_partial.stride(1), O_partial.stride(2), O_partial.stride(3),
        LSE_partial.stride(0), LSE_partial.stride(1), LSE_partial.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        num_splits, DIM, LSE_VALID_THRESHOLD,
        num_warps=4,
    )
    return O


def test_int8_kv(block_size=16, num_splits=4):
    """INT8 KV 两层对拍：
    (1) vs 反量化 fp32 oracle（量化感知）：验证 int8 kernel 的调度/mask/反量化本身正确；
    (2) vs fp16 kernel 输出：量化精度代价（报告值，非 pass/fail）。"""
    DEVICE, DTYPE, DIM, H = "cuda", torch.float16, 128, 8
    B, MAX_LB = 3, 8
    seq_lens = torch.tensor([128, 40, 97], dtype=torch.int32, device=DEVICE)
    max_seq = int(seq_lens.max().item())
    num_phys = 64

    torch.manual_seed(0)
    Q = (torch.randn(B, H, DIM, device=DEVICE, dtype=torch.float32) * 0.5).to(DTYPE)
    K_full = (torch.randn(B, H, max_seq, DIM, device=DEVICE, dtype=torch.float32) * 0.5).to(DTYPE)
    V_full = (torch.randn(B, H, max_seq, DIM, device=DEVICE, dtype=torch.float32) * 0.5).to(DTYPE)

    K_cache = torch.full((num_phys, block_size, H, DIM), 5.0, device=DEVICE, dtype=DTYPE)
    V_cache = torch.full((num_phys, block_size, H, DIM), 5.0, device=DEVICE, dtype=DTYPE)
    bt = torch.zeros(B, MAX_LB, device=DEVICE, dtype=torch.int32)

    perm = torch.randperm(num_phys, device=DEVICE).to(torch.int32)
    pid = 0
    block_phys = []  # (phys, b, s, e)
    for b in range(B):
        nb = (int(seq_lens[b].item()) + block_size - 1) // block_size
        for i in range(nb):
            phys = int(perm[pid].item()); pid += 1
            bt[b, i] = phys
            s, e = i * block_size, (i + 1) * block_size
            K_cache[phys] = K_full[b, :, s:e].permute(1, 0, 2)
            V_cache[phys] = V_full[b, :, s:e].permute(1, 0, 2)
            block_phys.append((phys, b, s, e))
    # 只量化被占用的物理块（未占用块保持填充值，验证 mask 不读它们）
    ki = torch.zeros_like(K_cache, dtype=torch.int8)
    vi = torch.zeros_like(V_cache, dtype=torch.int8)
    sk = torch.ones(num_phys, H, device=DEVICE, dtype=torch.float32)
    sv = torch.ones(num_phys, device=DEVICE, dtype=torch.float32).unsqueeze(1).expand(num_phys, H).contiguous()
    sv = sv.clone()
    for phys, b, s, e in block_phys:
        kb = K_cache[phys:phys + 1]
        vb = V_cache[phys:phys + 1]
        kbi, ski, vbi, svi = quantize_kv_per_head(kb, vb)
        ki[phys] = kbi[0]; vi[phys] = vbi[0]
        sk[phys] = ski[0]; sv[phys] = svi[0]

    # 反量化缓存 = kernel 眼中的真实 K/V（量化感知 oracle 的数据源）
    K_deq = (ki.float() * sk[:, None, :, None]).to(DTYPE)
    V_deq = (vi.float() * sv[:, None, :, None]).to(DTYPE)

    O_i8 = paged_attention_i8(Q, ki, vi, sk, sv, bt, seq_lens, block_size, num_splits)
    O_f16 = paged_attention_vl(Q, K_cache, V_cache, bt, seq_lens, block_size, num_splits)

    scale_attn = 1.0 / math.sqrt(DIM)
    md_i8 = md_f16 = md_i8_total = 0.0
    for b in range(B):
        sl = int(seq_lens[b].item())
        q = Q[b].float().unsqueeze(1)
        for Kref, Vref in [(K_deq, V_deq), (K_cache, V_cache)]:
            kk = torch.empty(H, sl, DIM, device=DEVICE)
            vv = torch.empty(H, sl, DIM, device=DEVICE)
            for i in range((sl + block_size - 1) // block_size):
                s0 = i * block_size
                n = min(block_size, sl - s0)
                kk[:, s0:s0 + n] = Kref[bt[b, i]][:n].float().permute(1, 0, 2)
                vv[:, s0:s0 + n] = Vref[bt[b, i]][:n].float().permute(1, 0, 2)
            s_ = torch.matmul(q, kk.transpose(-2, -1)) * scale_attn
            o = torch.matmul(torch.softmax(s_, dim=-1), vv).squeeze(1)
            d_i8 = (o - O_i8[b].float()).abs().max().item()
            d_f16 = (o - O_f16[b].float()).abs().max().item()
            if Kref is K_deq:
                md_i8 = max(md_i8, d_i8)        # kernel 正确性（量化感知 oracle）
            else:
                md_f16 = max(md_f16, d_f16)     # fp16 基线（同源 fp32 oracle）
                md_i8_total = max(md_i8_total, d_i8)  # int8 端到端精度代价
    ok = md_i8 < 2e-2
    print(f"[Correctness] int8-KV vs dequant-fp32 oracle: max_diff={md_i8:.2e} "
          f"{'✅' if ok else '❌'};  fp16 vs fp32 oracle: {md_f16:.2e};  "
          f"int8 端到端精度代价: {md_i8_total:.2e} (vs fp16 基线, 常驻显存省 ~50%)")
    return ok


def test_variable_length(block_size=16, num_splits=4):
    """per-request 变长 + 非连续物理块正确性自测（AOT 改造的两个点都在这里被检验）。

    短请求未占用的物理块填入大噪声：若 kernel 误用全局 max 长度去读，输出会被污染，
    从而坐实 per-request seq_lens mask 真的生效。
    """
    DEVICE, DTYPE, DIM, H = "cuda", torch.float16, 128, 8
    B, MAX_LB = 3, 8
    seq_lens = torch.tensor([128, 40, 97], dtype=torch.int32, device=DEVICE)
    max_seq = int(seq_lens.max().item())
    num_phys = 64

    torch.manual_seed(0)
    Q = (torch.randn(B, H, DIM, device=DEVICE, dtype=torch.float32) * 0.5).to(DTYPE)
    K_full = (torch.randn(B, H, max_seq, DIM, device=DEVICE, dtype=torch.float32) * 0.5).to(DTYPE)
    V_full = (torch.randn(B, H, max_seq, DIM, device=DEVICE, dtype=torch.float32) * 0.5).to(DTYPE)

    K_cache = torch.full((num_phys, block_size, H, DIM), 5.0, device=DEVICE, dtype=DTYPE)
    V_cache = torch.full((num_phys, block_size, H, DIM), 5.0, device=DEVICE, dtype=DTYPE)
    bt = torch.zeros(B, MAX_LB, device=DEVICE, dtype=torch.int32)

    # 非连续分配物理块（随机置换，保证不重复），逻辑块->物理块完全打乱。
    perm = torch.randperm(num_phys, device=DEVICE).to(torch.int32)
    pid = 0
    for b in range(B):
        nb = (int(seq_lens[b].item()) + block_size - 1) // block_size
        for i in range(nb):
            phys = int(perm[pid].item()); pid += 1
            bt[b, i] = phys
            s, e = i * block_size, (i + 1) * block_size
            K_cache[phys] = K_full[b, :, s:e].permute(1, 0, 2)
            V_cache[phys] = V_full[b, :, s:e].permute(1, 0, 2)

    O_vl = paged_attention_vl(Q, K_cache, V_cache, bt, seq_lens, block_size, num_splits)

    # host/torch fp32 reference：每个请求只对 [0, seq_lens[b]) 的 K/V 做注意力。
    scale = 1.0 / math.sqrt(DIM)
    max_diff = 0.0
    for b in range(B):
        sl = int(seq_lens[b].item())
        q = Q[b].float().unsqueeze(1)                      # [H,1,D]
        k = K_full[b, :, :sl].float()                     # [H,sl,D]
        v = V_full[b, :, :sl].float()
        s = torch.matmul(q, k.transpose(-2, -1)) * scale  # [H,1,sl]
        o = torch.matmul(torch.softmax(s, dim=-1), v).squeeze(1)  # [H,D]
        d = (o - O_vl[b].float()).abs().max().item()
        max_diff = max(max_diff, d)
    ok = max_diff < 2e-2
    print(f"[Correctness] variable-length(split-vl) vs Ref: max_diff={max_diff:.2e} "
          f"{'✅' if ok else '❌'}  (B={B}, seq_lens={seq_lens.tolist()}, non-contiguous paged)")
    return ok


if __name__ == "__main__":
    ok1 = test_variable_length(block_size=16, num_splits=4)
    ok2 = test_int8_kv(block_size=16, num_splits=4)
    print(f"[Self-test] {'ALL PASS' if (ok1 and ok2) else 'FAILED'}")
