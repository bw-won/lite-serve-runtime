#!/usr/bin/env python3
"""
aot_paged_attention.py — 把 split-KV PagedAttention 的两个 Triton kernel 离线 AOT
编译成【当前 GPU 架构】的 cubin，并生成可被 C++ 直接 #include 的 meta 头。

为什么用 JITFunction.warmup 而不是 triton.compile：
  Triton 3.x 各小版本 triton.compile 的关键字签名不统一（3.2 已无 signature= 关键字）。
  warmup 是跨 2.x/3.x 稳定的公共 API：喂一组示例【实参】（tensor 决定指针类型、np.int64
  决定 i64、python int 决定 i32），它只编译不 launch，返回 CompiledKernel（.asm['cubin']、
  .metadata.name/shared/num_warps）。cubin 架构 = 运行本脚本的 GPU（你的 RTX 3050 即 sm_86）。

用法（在 WSL、装好 triton/torch 的环境里，且能看到 GPU）：
  python3 tools/aot_paged_attention.py

产物（写入 ../kernels）：
  paged_attn_split.cubin / paged_attn_reduce.cubin   裸 cubin（备查/反汇编）
  paged_attn_meta.h                                  C++ 嵌入：cubin 字节 + 符号名 + launch 元数据

ABI（与 src/paged_attention.cpp 严格对齐）：
  split  = 7 指针 + 18 stride(i32) + 3 标量(i32) = 28
  reduce = 3 指针 + 10 stride(i32) + 1 标量(i32) = 14
stride/标量一律传普通 python int（Triton 推导为 int32，与真实 JIT 的 tensor.stride() 一致）。
切勿传 numpy.int64：warmup 的 MockTensor 会把它误判成 pointer<int64> 而非标量。
本项目最大 stride ≈ H*S*D = 16*64*128 ≈ 13 万元素，int32 足够。
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split_kv_attention import (  # noqa: E402
    paged_attention_split_kernel_vl,
    paged_attention_split_kernel_i8,
    paged_attention_reduce_kernel,
)

# AOT 固化的 constexpr（改这些值需重跑本脚本，并与 C++ 侧 LitePagedAttnConfig 一致）
DIM = 128
BLOCK_SIZE = 16
BLOCK_N = 16
LSE_THRESHOLD = -1e29

DEV = "cuda"
# 仅用于让 warmup 推导指针类型的 dummy 形状（值不进 cubin，编译不 launch）。
B, H, MLB, P, S = 1, 8, 8, 1, 4
# stride 的 dummy 值必须【非 1】：Triton 会把值==1 的非 constexpr int 编译期特化内联、
# 从 cubin 形参里删掉（equal-to-1 specialization），导致运行时 ABI 形参数 < C++ 打包数而错位。
# 取一个非 1、不被 16 整除的质数，只用于保留形参槽；真实 stride 由 C++ 运行时传入。
SD = 101


def dump_param_abi(tag, compiled, symbol, n_expect):
    """打印 cubin 每个运行时参数的真实位宽/类型（constexpr 已内联，不在其中）。

    Triton 参数名固定为 `<symbol>_param_N`，N 即形参位置。参数列表以 `)` 后跟 `{` 结束
    （不能用第一个 ')' 截断，那会漏掉大部分参数）。
    """
    import re
    asm = getattr(compiled, "asm", {}) or {}
    ptx = asm.get("ptx")
    if not (isinstance(ptx, str) and ".entry" in ptx):
        print(f"        [{tag}] asm 中无 PTX（keys={list(asm.keys())}），可用 cuobjdump -ptx 兜底")
        return
    m = re.search(r"\.entry\s+" + re.escape(symbol) + r"\b", ptx)
    seg = ptx[m.start():] if m else ptx
    po = seg.find("(")
    # 参数声明内部不含括号，第一个 ')' 即参数列表结束（其后到 '{' 之间还有 .maxntid 等指令）。
    pc = seg.find(")", po)
    if pc < 0:
        print(f"        [{tag}] 未找到参数列表结束符 ')'")
        return 0
    header = seg[po + 1:pc]
    found = re.findall(r"\.param\s+(\.u\d+|\.b\d+|\.s\d+)[^,]*?" + re.escape(symbol)
                       + r"_param_(\d+)", header)
    params = sorted(((int(n), t) for t, n in found), key=lambda x: x[0])
    print(f"        ---- PTX .param ABI: cubin 实际形参 {len(params)} 个（传入实参 {n_expect} 个）----")
    for i, (n, t) in enumerate(params):
        print(f"          [{i:2d}] param_{n:<3} {t}")
    if len(params) != n_expect:
        print(f"        *** 警告: cubin 形参数({len(params)}) != 传入实参数({n_expect})；")
        print(f"            部分 int 标量被 Triton 编译期【特化内联】(常见于 dummy 值=1)，需保持非 1 dummy。")
    else:
        n_u64 = sum(1 for _, t in params if t == ".u64")
        n_u32 = sum(1 for _, t in params if t == ".u32")
        print(f"        汇总: .u64(8B)={n_u64}  .u32(4B)={n_u32}  其它={len(params) - n_u64 - n_u32}")
    return len(params)


def aot_compile(kernel, tag, positional_args, consts):
    print(f"[AOT] compiling {tag} ...")
    # warmup：跨版本稳定的编译入口；grid 仅用于元数据，不会真正 launch。
    compiled = kernel.warmup(*positional_args, grid=(1, 1, 1), **consts)
    cubin = compiled.asm["cubin"]
    if isinstance(cubin, str):
        raise RuntimeError(f"{tag}: asm['cubin'] 不是 bytes")
    meta = compiled.metadata
    print(f"        symbol      = {meta.name}")
    print(f"        cubin bytes = {len(cubin)}")
    print(f"        num_warps   = {meta.num_warps}  threads/block = {meta.num_warps * 32}")
    print(f"        shared(B)   = {meta.shared}")
    print(f"        rt params   = {len(positional_args)}")
    n_decl = dump_param_abi(tag, compiled, meta.name, len(positional_args))
    return {
        "tag": tag,
        "cubin": cubin,
        "name": meta.name,
        "num_warps": int(meta.num_warps),
        "shared": int(meta.shared),
        "n_params": len(positional_args),
        "n_decl": n_decl,
    }


def build_split_args():
    q = torch.empty(B, H, DIM, device=DEV, dtype=torch.float16)
    k = torch.empty(P, BLOCK_SIZE, H, DIM, device=DEV, dtype=torch.float16)
    v = torch.empty(P, BLOCK_SIZE, H, DIM, device=DEV, dtype=torch.float16)
    bt = torch.empty(B, MLB, device=DEV, dtype=torch.int32)
    seq = torch.empty(B, device=DEV, dtype=torch.int32)
    op = torch.empty(B, H, S, DIM, device=DEV, dtype=torch.float16)
    lse = torch.empty(B, H, S, device=DEV, dtype=torch.float32)
    args = [
        q, k, v, bt, seq, op, lse,            # 7 指针: *fp16,*fp16,*fp16,*i32,*i32,*fp16,*fp32
        SD, SD, SD,                            # stride_qb,qh,qd (i32, 非1 以防被特化内联)
        SD, SD, SD, SD,                        # stride_kb,kbs,kh,kd
        SD, SD, SD, SD,                        # stride_vb,vbs,vh,vd
        SD, SD, SD, SD,                        # stride_ob,oh,os,od
        SD, SD, SD,                            # stride_lb,lh,ls
        8, 32, MLB,                            # HEADS, SPLIT_SIZE, MAX_LOGICAL_BLOCKS (i32)
    ]
    consts = {"DIM": DIM, "BLOCK_SIZE": BLOCK_SIZE, "BLOCK_N": BLOCK_N,
              "num_warps": 4, "num_stages": 3}
    return args, consts


def build_split_i8_args():
    # INT8 KV 变体：K/V 为 int8 指针，多 Scale_K/Scale_V 两个 fp32 指针 + 两个 stride。
    # ABI：9 指针 + 20 stride(i32) + 3 标量(i32) = 32
    q = torch.empty(B, H, DIM, device=DEV, dtype=torch.float16)
    k = torch.empty(P, BLOCK_SIZE, H, DIM, device=DEV, dtype=torch.int8)
    v = torch.empty(P, BLOCK_SIZE, H, DIM, device=DEV, dtype=torch.int8)
    sk = torch.empty(P, H, device=DEV, dtype=torch.float32)
    sv = torch.empty(P, H, device=DEV, dtype=torch.float32)
    bt = torch.empty(B, MLB, device=DEV, dtype=torch.int32)
    seq = torch.empty(B, device=DEV, dtype=torch.int32)
    op = torch.empty(B, H, S, DIM, device=DEV, dtype=torch.float16)
    lse = torch.empty(B, H, S, device=DEV, dtype=torch.float32)
    args = [
        q, k, v, bt, seq, sk, sv, op, lse,     # 9 指针
        SD, SD, SD,                            # stride_qb,qh,qd
        SD, SD, SD, SD,                        # stride_kb,kbs,kh,kd
        SD, SD, SD, SD,                        # stride_vb,vbs,vh,vd
        SD, SD,                                # stride_skb,svb（per-phys，[P,H] 布局）
        SD, SD, SD, SD,                        # stride_ob,oh,os,od
        SD, SD, SD,                            # stride_lb,lh,ls
        8, 32, MLB,                            # HEADS, SPLIT_SIZE, MAX_LOGICAL_BLOCKS
    ]
    consts = {"DIM": DIM, "BLOCK_SIZE": BLOCK_SIZE, "BLOCK_N": BLOCK_N,
              "num_warps": 4, "num_stages": 3}
    return args, consts


def build_reduce_args():
    op = torch.empty(B, H, S, DIM, device=DEV, dtype=torch.float16)
    lse = torch.empty(B, H, S, device=DEV, dtype=torch.float32)
    o = torch.empty(B, H, DIM, device=DEV, dtype=torch.float16)
    args = [
        op, lse, o,                           # 3 指针: *fp16,*fp32,*fp16
        SD, SD, SD, SD,                       # stride_ob,oh,os,od (O_partial, i32, 非1)
        SD, SD, SD,                           # stride_lb,lh,ls (LSE, i32)
        SD, SD, SD,                           # stride_out_b,h,d (O, i32)
        S,                                     # NUM_SPLITS (i32)
    ]
    consts = {"DIM": DIM, "LSE_THRESHOLD": LSE_THRESHOLD, "num_warps": 4}
    return args, consts


def c_array(name, data):
    rows = []
    for i in range(0, len(data), 16):
        rows.append("    " + ", ".join(f"0x{b:02x}" for b in data[i:i + 16]) + ",")
    body = "\n".join(rows)
    return (f"static const unsigned char {name}[] = {{\n{body}\n}};\n"
            f"static const size_t {name}_SIZE = sizeof({name});\n")


def write_meta(out_dir, sp, rd, sp_i8, cc):
    os.makedirs(out_dir, exist_ok=True)
    for info in (sp, rd, sp_i8):
        with open(os.path.join(out_dir, f"paged_attn_{info['tag']}.cubin"), "wb") as f:
            f.write(info["cubin"])
    header = f"""// AUTO-GENERATED by tools/aot_paged_attention.py — 请勿手改。
// 目标架构 sm_{cc}（由运行脚本的 GPU 决定）。重新生成：python3 tools/aot_paged_attention.py
#pragma once
#include <stddef.h>
#include <stdint.h>

// AOT 固化的 constexpr 形状（必须与 C++ 侧 LitePagedAttnConfig 使用的形状一致）
#define LITE_PA_DIM {DIM}
#define LITE_PA_BLOCK_SIZE {BLOCK_SIZE}
#define LITE_PA_BLOCK_N {BLOCK_N}

// cubin 内 mangled 符号名（cuModuleGetFunction 必须用它，不能手写）
#define LITE_PA_SPLIT_SYMBOL  "{sp['name']}"
#define LITE_PA_SPLIT_I8_SYMBOL "{sp_i8['name']}"
#define LITE_PA_REDUCE_SYMBOL "{rd['name']}"

// launch 元数据
#define LITE_PA_SPLIT_THREADS  ({sp['num_warps'] * 32})
#define LITE_PA_SPLIT_I8_THREADS  ({sp_i8['num_warps'] * 32})
#define LITE_PA_REDUCE_THREADS ({rd['num_warps'] * 32})
#define LITE_PA_SPLIT_SHARED   ({sp['shared']})
#define LITE_PA_SPLIT_I8_SHARED   ({sp_i8['shared']})
#define LITE_PA_REDUCE_SHARED  ({rd['shared']})

{c_array("kPagedAttnSplitCubin", sp['cubin'])}
{c_array("kPagedAttnSplitI8Cubin", sp_i8['cubin'])}
{c_array("kPagedAttnReduceCubin", rd['cubin'])}
"""
    with open(os.path.join(out_dir, "paged_attn_meta.h"), "w") as f:
        f.write(header)
    print(f"[AOT] wrote {os.path.join(out_dir, 'paged_attn_meta.h')}")


def main():
    if not torch.cuda.is_available():
        print("ERROR: 看不到 CUDA GPU，AOT 需要在 GPU 机器上运行（warmup 要推导指针类型）。")
        sys.exit(1)
    major, minor = torch.cuda.get_device_capability(0)
    cc = major * 10 + minor
    import triton
    print(f"[AOT] device={torch.cuda.get_device_name(0)} sm_{cc}  triton={triton.__version__}")

    sp_args, sp_consts = build_split_args()
    spi8_args, spi8_consts = build_split_i8_args()
    rd_args, rd_consts = build_reduce_args()
    sp = aot_compile(paged_attention_split_kernel_vl, "split", sp_args, sp_consts)
    spi8 = aot_compile(paged_attention_split_kernel_i8, "split_i8", spi8_args, spi8_consts)
    rd = aot_compile(paged_attention_reduce_kernel, "reduce", rd_args, rd_consts)

    assert sp["n_params"] == 28, f"split rt args = {sp['n_params']}, expect 28"
    assert spi8["n_params"] == 32, f"split_i8 rt args = {spi8['n_params']}, expect 32"
    assert rd["n_params"] == 14, f"reduce rt args = {rd['n_params']}, expect 14"
    # 构建期 ABI 自检：cubin 真实形参数必须等于 C++ 打包数，防止 int 标量被特化内联而错位。
    assert sp["n_decl"] == 28, f"split cubin 形参 = {sp['n_decl']}（被特化？），expect 28"
    assert spi8["n_decl"] == 32, f"split_i8 cubin 形参 = {spi8['n_decl']}（被特化？），expect 32"
    assert rd["n_decl"] == 14, f"reduce cubin 形参 = {rd['n_decl']}（被特化？），expect 14"

    out_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"))
    write_meta(out_dir, sp, rd, spi8, cc)
    print("[AOT] DONE.")


if __name__ == "__main__":
    main()
