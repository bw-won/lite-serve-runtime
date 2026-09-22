# Lite-GPU-Runtime

基于 **CUDA Driver API** 从零实现的轻量 LLM 推理运行时（无任何推理框架依赖），面向单 GPU，
覆盖 CUDA Graph、连续 batching、PagedAttention、KV 显存管理的完整链路，并配 CUPTI 性能归因。

目标平台：Linux（WSL2 可用）、sm_86（RTX 3050 实测），CUDA 12.x，C++17。
多卡张量并行（NCCL）为可选编译项，单卡可用 SimComm 模拟验证同一套逻辑。

## 核心特性

- **CUDA Graph 全封装**：stream capture（RELAXED / THREAD_LOCAL）、instantiate/replay、
  节点级热更新（param-only / func-swap）、整图 ExecUpdate、多流 fork/join 跨流依赖。
- **vLLM 风格 Scheduler**：continuous batching（pack/commit 两阶段）、抢占-重算（recompute）、
  按 token 内容哈希的 prefix caching（块共享 + refcnt + 写时复制 COW）、batch 分桶多图 +
  运行时选桶（capture sizes，零 recapture）。
- **PagedAttention（Triton AOT）**：split-KV 两 kernel 设计（split 网格 B·H×S + reduce 用 LSE
  merge），离线 AOT 编译成 cubin 由 C++ `cuModuleLoadData` 加载；支持 fp16 与 INT8 KV
  （per-(block,head) 对称量化，KV 常驻显存 < fp16 的 50%）。
- **TP 半头切分**：按 attention head 切分 H→H/2+H/2，all-gather 拼回全头；
  单卡用 SimComm（同卡 D2D + event）模拟，多卡可切 NCCL，代码路径一致。
- **CUPTI 性能归因**：kernel/memcpy 时长聚合、按 kernel 名 Top-N hotspot、
  SM∩CE 时间区间交叠（双引擎并行的引擎级证据）。

## 目录结构

```
include/lite_gpu.h        对外 C API（唯一公共头）
src/                      运行时实现（静态库 lite_gpu）
  context/stream/memory/launcher   初始化、流/事件、页级显存池、kernel launch
  graph                   CUDA Graph capture / exec / 热更新 / memset 等节点
  profiler                CUPTI activity 聚合与双引擎交叠分析
  kv_pool / scheduler     KV 物理块池（哈希链/COW/refcnt）+ 连续 batching 调度器
  paged_attention         PagedAttention host 侧（加载 AOT cubin、量化、COW 拷贝）
  comm / comm_nccl        SimComm（单卡模拟）/ NCCL 通信域（可选）
  dummy_kernel.cu         测试用空/重负载 kernel
kernels/                  AOT 产物：paged_attn_meta.h（cubin 字节嵌入）+ 裸 cubin 备查
tools/                    Triton kernel 源码与 AOT 编译脚本（Python）
tests/test_runtime.cpp    单卡测试套件：单元测试 + 性能基准 + 集成对拍（详见文件头总览）
tests/test_tp_multigpu.cpp 真机双卡 TP 验证（仅 -DLITE_WITH_NCCL=ON 构建，设备<2 自动 SKIP）
```

### 分层说明

```
                    ┌────────────────────────────────────────┐
                    │  include/lite_gpu.h  唯一公共 C API     │
                    └────────────────────────────────────────┘
   基础层: context(主context) / stream·event / memory(页级池,stream-ordered)
           launcher(dummy kernel 缓存) / profiler(CUPTI activity)
   Graph 层: graph.cpp —— capture(GLOBAL/THREAD_LOCAL/RELAXED) → instantiate → launch
             热更新四机制: node setParams / node 换函数 / ExecUpdate / 重捕获
             多流 fork/join（图内跨流依赖）
   调度层(纯 host,不碰 CUDA): kv_pool.cpp(物理块池: refcnt+内容哈希注册表)
             scheduler.cpp(continuous batching: 准入/增长/抢占/重算/commit/re-key/COW)
   算子层: paged_attention.cpp —— 加载 Triton AOT cubin(cuModuleLoadData),
             split-KV 两 kernel + LSE 归约, fp16 与 INT8 双通路
   通信层: comm.cpp(SimComm: 单卡用 D2D+event 模拟 all-gather)
             comm_nccl.cpp(真机: 每 rank primary context + NCCL, 可选编译)
   测试: tests/test_runtime.cpp(单卡全套,10用例+4bench+profiler段, 2200行)
             tests/test_tp_multigpu.cpp(双卡真机, 仅 NCCL 构建)
```

## 构建

```bash
cmake -B build                       # 默认 Release（计时基准依赖 -O3，已在 CMakeLists 固定）
cmake --build build -j4

# 可选：启用 NCCL 多卡（需 libnccl-dev 或 conda 安装 nccl）
cmake -B build -DLITE_WITH_NCCL=ON && cmake --build build -j4
```

## 运行

```bash
./build/test_runtime                 # 单卡全套：单元测试 + 基准 + 集成对拍
./build/test_tp_multigpu             # 双卡 TP（NCCL 构建；单卡环境打印 SKIP）
```

`test_runtime` 按序执行：基础单元测试（Graph/流/显存池/多线程）→ launch overhead 与
热更新基准 → 真实 GEMM 谱系 → 调度器抢占/前缀缓存/INT8 KV → 端到端 PagedAttention
continuous batching（对 fp32 oracle 对拍）→ 分桶多图 / TP 拼头 → CUPTI 归因段。
每个模块的功能与判据见 `tests/test_runtime.cpp` 文件头与各模块注释。

## 重新生成 AOT cubin

`kernels/paged_attn_meta.h` 由开发机预生成并入库（缺失时构建直接失败，不提供降级桩）。
换 GPU 架构或改 constexpr（HEAD_DIM=128、BLOCK_SIZE=16 等）需重跑：

```bash
python3 tools/aot_paged_attention.py   # 需 triton + torch 且能看到 GPU；产物写入 kernels/
```

注意：cubin 架构跟随运行脚本的 GPU（RTX 3050 → sm_86），跨架构需重新生成。
