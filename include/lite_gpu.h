#pragma once
#include <cstddef>
#include <cstdint>

enum LiteError {
    LITE_SUCCESS = 0,
    LITE_ERROR_INVALID_VALUE = 1,
    LITE_ERROR_OUT_OF_MEMORY = 2,
    LITE_ERROR_DRIVER_API = 3,
    LITE_ERROR_STREAM_SYNC = 4,
    LITE_ERROR_NOT_INITIALIZED = 5,
    LITE_ERROR_GRAPH_UPDATE = 6,
    LITE_ERROR_GRAPH_NODE_TYPE = 7,
    LITE_ERROR_GRAPH_NODE_INVALID = 8,
    LITE_ERROR_NOT_SUPPORTED = 9,
};

typedef struct LiteStream_s* LiteStream;
typedef struct LiteEvent_s*  LiteEvent;
typedef struct LiteGraph_s* LiteGraph;
typedef struct LiteGraphExec_s* LiteGraphExec;
typedef struct LiteGraphNode_s* LiteGraphNode;

// ================= Memory Copy API =================
LiteError liteMemcpyHtoDAsync(void* dst, const void* src, size_t size_bytes, LiteStream stream);
LiteError liteMemcpyDtoHAsync(void* dst, const void* src, size_t size_bytes, LiteStream stream);
LiteError liteMemcpyDtoDAsync(void* dst, const void* src, size_t size, LiteStream stream);

// ================= Context API =================
LiteError liteInit();
LiteError liteShutdown();

typedef struct CUctx_st* LiteContext;
LiteError liteGetContext(LiteContext* ctx);
LiteError liteSetCurrentContext(LiteContext ctx);

LiteError liteDeviceGetCount(int* count);

void liteInvalidateCachedFuncs(void);

// ================= Stream & Event API =================
LiteError liteStreamCreate(LiteStream* stream);
LiteError liteStreamDestroy(LiteStream stream);
LiteError liteStreamSynchronize(LiteStream stream);
LiteError liteEventCreate(LiteEvent* event);
LiteError liteEventDestroy(LiteEvent event);
LiteError liteEventRecord(LiteEvent event, LiteStream stream);
LiteError liteStreamWaitEvent(LiteStream stream, LiteEvent event);
LiteError liteEventCreateWithTiming(LiteEvent* event);
LiteError liteEventElapsedTime(float* ms, LiteEvent start, LiteEvent end);
LiteError liteEventCreateForSync(LiteEvent* event);

// ================= Memory Pool API =================
LiteError liteMemPoolInit(size_t page_size, size_t max_pages);
LiteError liteMemPoolDestroy();
LiteError liteMemAllocPage(void** ptr, LiteStream stream);
LiteError liteMemFreePage(void* ptr, LiteStream stream);

// ================= Launcher API =================
LiteError liteMemsetAsync(void* ptr, size_t size_bytes, LiteStream stream);
LiteError liteLaunchTinyKernel(void* ptr, int fill, LiteStream stream);
LiteError liteLaunchEmptyKernel(LiteStream stream);
LiteError liteLaunchDummyKernelHeavy(void* ptr, int loops, int fill, LiteStream stream);

// ================= Profiler API =================
LiteError liteProfilerReset();
LiteError liteProfilerStart();
LiteError liteProfilerStop();
LiteError liteProfilerResetRegions();
void liteProfilerReport();
void liteProfilerReportOverlap();

// ================= Graph API =================
enum LiteCaptureMode {
    LITE_CAPTURE_GLOBAL,
    LITE_CAPTURE_THREAD_LOCAL,
    LITE_CAPTURE_RELAXED
};

enum LiteGraphNodeType {
    LITE_GRAPH_NODE_KERNEL,
    LITE_GRAPH_NODE_MEMCPY,
    LITE_GRAPH_NODE_MEMSET,
    LITE_GRAPH_NODE_EVENT,
    LITE_GRAPH_NODE_OTHER
};

LiteError liteStreamBeginCapture(LiteStream stream);
LiteError liteStreamBeginCaptureEx(LiteStream stream, LiteCaptureMode mode);
LiteError liteStreamEndCapture(LiteStream stream, LiteGraph* graph);
LiteError liteGraphInstantiate(LiteGraph graph, LiteGraphExec* exec);
LiteError liteGraphLaunch(LiteGraphExec exec, LiteStream stream);
LiteError liteGraphDestroy(LiteGraph graph);
LiteError liteGraphExecDestroy(LiteGraphExec exec);

enum LiteGraphExecUpdateResult {
    LITE_GRAPH_UPDATE_SUCCESS,
    LITE_GRAPH_UPDATE_ERROR_TOPOLOGY_MISMATCH,
    LITE_GRAPH_UPDATE_ERROR_NODE_FAILED,
    LITE_GRAPH_UPDATE_ERROR_UNKNOWN
};
LiteError liteGraphExecUpdate(LiteGraphExec exec, LiteGraph new_param_graph, LiteGraphExecUpdateResult* out_result);

LiteError liteGraphGetNodeCount(LiteGraph graph, size_t* out_node_cnt);
LiteError liteGraphGetNodeByIndex(LiteGraph graph, size_t idx, LiteGraphNode* out_node);
LiteError liteGraphNodeGetType(LiteGraphNode node, LiteGraphNodeType* out_type);
LiteError liteGraphMemcpyNodeGetInfo(LiteGraphNode node, uint64_t* dst_out, uint64_t* src_out, size_t* width_out, uint32_t* height_out, uint32_t* depth_out);
LiteError liteGraphExecMemcpyNodeSetParams(LiteGraphExec exec, LiteGraphNode node, void* dst, void* src);
LiteError liteGraphExecKernelNodeSetParams(LiteGraphExec exec, LiteGraphNode node, void** kernel_args);
LiteError liteGraphExecKernelNodeSetParamsEx(LiteGraphExec exec, LiteGraphNode node, void* func, void** kernel_args);
LiteError liteGraphMemsetNodeGetInfo(LiteGraphNode node, uint64_t* dst_out, uint32_t* fill_value_out, size_t* byte_count_out);
LiteError liteGraphExecMemsetNodeSetParams(LiteGraphExec exec, LiteGraphNode node, void* dst, uint32_t fill_val, size_t byte_count);
LiteError liteGraphNodeDestroy(LiteGraphNode node);

#define LITE_CB_MAX_SEQS 16
#define LITE_CB_MAX_BLOCKS_PER_SEQ 16

typedef struct LiteKvPool_s*    LiteKvPool;
typedef struct LiteScheduler_s* LiteScheduler;

typedef struct {
    int tokens_per_block;    // 一个物理 KV 块承载多少 token（block size）
    int max_blocks_per_seq;  // 每请求最大逻辑块数（<= LITE_CB_MAX_BLOCKS_PER_SEQ）
    int max_seqs;            // 最大并发请求（<= LITE_CB_MAX_SEQS）
    int num_phys_blocks;     // 物理块池容量（KV 容量预算，约束准入）
} LiteKvPoolConfig;

// 一个 decode step 的 host 侧元数据（紧凑打包前 n_active 行），由 scheduler 产出；
// 调用方把其中 btable/positions 等拷入算子的固定 device 缓冲后 replay 固定图。
typedef struct {
    int n_active;
    int slot_req[LITE_CB_MAX_SEQS];                                   // 批内第 s 行 -> req_id
    int positions[LITE_CB_MAX_SEQS];                                  // 批内第 s 行 -> 当前 token 位置
    // resumed[i]=1：槽 i 本步是"被抢占后重算恢复"的第一步，调用方需把该请求已生成 token
    // 的历史 K/V 重新填入新分配的物理块（recompute）；仅恢复第一步置 1，其余步为 0。
    int resumed[LITE_CB_MAX_SEQS];
    int btable[LITE_CB_MAX_SEQS * LITE_CB_MAX_BLOCKS_PER_SEQ];        // 行主序 block-table
    // cow_src[s] != 0：行 s 本步发生 COW（下一步要写的槽位落在共享块上），调度器已把
    // btable 对应项换成新分配的私有块；调用方须在 replay 前把旧块（cow_src[s]）的 K/V
    // 内容复制进新块（如 litePagedAttnCopyBlockD2D），再写入新 token。0=无 COW。
    int cow_src[LITE_CB_MAX_SEQS];
} LiteStepMeta;

LiteError liteKvPoolCreate(const LiteKvPoolConfig* cfg, LiteKvPool* out);
LiteError liteKvPoolDestroy(LiteKvPool pool);
int       liteKvPoolAllocBlock(LiteKvPool pool);
int       liteKvPoolFreeBlock(LiteKvPool pool, int phys_id);
void      liteKvPoolRetainBlock(LiteKvPool pool, int phys_id);
int       liteKvPoolBlockRefs(LiteKvPool pool, int phys_id);
int       liteKvPoolFreeBlocks(LiteKvPool pool);
int       liteKvPoolLookupHash(LiteKvPool pool, uint64_t hash);
LiteError liteKvPoolRegisterHash(LiteKvPool pool, int phys_id, uint64_t hash);
LiteError liteKvPoolUnregisterHash(LiteKvPool pool, int phys_id);
int       liteKvPoolCacheHits(LiteKvPool pool);

LiteError liteSchedulerCreate(LiteKvPool pool, int max_seqs, int tokens_per_block,
                              int max_blocks_per_seq, int continuous, LiteScheduler* out);
void      liteSchedulerDestroy(LiteScheduler s);
int       liteSchedulerAddRequest(LiteScheduler s, int target_len);
LiteError liteSchedulerAddRequestEx(LiteScheduler s, int target_len,
                                    const int* prompt_tokens, int prompt_len);

int  liteSchedulerStep(LiteScheduler s, LiteStepMeta* meta);
void liteSchedulerCommit(LiteScheduler s, const LiteStepMeta* meta, const int* out_tokens);
int  liteSchedulerNumFinished(LiteScheduler s);
int  liteSchedulerNumSteps(LiteScheduler s);
int  liteSchedulerGetResult(LiteScheduler s, int req_id, int* out_tokens, int cap);
int  liteSchedulerFinishStep(LiteScheduler s, int req_id);

// -------- Preemption / Recompute（vLLM 式显存超分 + 抢占重算，调度器内置）--------
// KV 块按需增量分配（准入只给当前块、跨 block 边界再增长）；容量不足时按 FCFS 抢占最年轻
// running 请求，释放其块给更老/新请求，被抢者退回 waiting 队头重算。抢占只丢 KV（可重算）、
// 不丢已生成 token；恢复后从原进度继续，LiteStepMeta.resumed 在恢复第一步置 1，调用方据此
// 把历史 K/V 重新填入新分配的物理块。
int  liteSchedulerNumPreemptions(LiteScheduler s);
int  liteSchedulerRequestPreempted(LiteScheduler s, int req_id);
int  liteSchedulerNumCacheHits(LiteScheduler s);
int  liteSchedulerNumCow(LiteScheduler s);

// =====================================================================
// PagedAttention（Triton AOT cubin 算子加载 + split-KV 两段执行）
//
// 把 tools/split_kv_attention.py 的 split-KV PagedAttention 经 tools/aot_paged_attention.py
// 在构建期 AOT 编成 cubin（kernels/paged_attn_meta.h），运行时用 Driver API
// cuModuleLoadData 加载、cuLaunchKernel 执行 split+reduce 两个 kernel —— 运行期无 Python/Triton。
//
// 与 continuous batching 一致的 capture-safe 设计：所有 device 缓冲（Q / paged K/V /
// block-table / seq_lens / O_partial / LSE / O）在 Create 时一次性 cuMemAlloc，地址恒定；
// 每步变化的只是这些固定地址上的【内容】（捕获外 Copy*），图只烘焙固定指针，可任意 replay。
//
// AOT 固化的 constexpr：head_dim=128、block_size=16、BLOCK_N=16（改形状需重跑 AOT）。
// 支持 per-request 变长：seq_lens[b] 给出每请求长度，未用槽位填 0（输出 0）。
// =====================================================================

#define LITE_PA_MAX_BATCH 16           // 固定行宽上限（图烘焙的 grid 上限）
#define LITE_PA_MAX_HEADS 16
#define LITE_PA_MAX_SPLITS 64

typedef struct LitePagedAttn_s* LitePagedAttn;

typedef struct {
    int num_heads;          // H（AOT kernel 支持任意 H，stride 运行时传）
    int head_dim;           // 必须 = 128（AOT constexpr 固化）
    int block_size;         // 必须 = 16（AOT constexpr 固化）
    int max_batch;          // 固定 batch 行宽 B（<= LITE_PA_MAX_BATCH）
    int max_logical_blocks; // 每请求最大逻辑块数（block-table 行宽）
    int num_phys_blocks;    // paged K/V 物理块池容量
    int num_splits;         // split-KV 切分数；<=0 则按 SM 数/(B*H) 自动并 clamp
    int int8_kv;            // 1 = KV 常驻 int8（per-(block,head) 对称量化），显存省 ~50%
} LitePagedAttnConfig;


LiteError litePagedAttnCreate(const LitePagedAttnConfig* cfg, LitePagedAttn* out);
LiteError litePagedAttnDestroy(LitePagedAttn a);
LiteError litePagedAttnCopyKV(LitePagedAttn a, const void* h_k_fp16, const void* h_v_fp16);
LiteError litePagedAttnCopyBlock(LitePagedAttn a, int phys,
                                 const void* h_k_block_fp16, const void* h_v_block_fp16);
LiteError litePagedAttnCopyBlockI8(LitePagedAttn a, int phys,
                                   const void* h_k_block_i8, const void* h_v_block_i8,
                                   const float* h_scale_k, const float* h_scale_v);
LiteError litePagedAttnCopyBlockD2D(LitePagedAttn a, int dst_phys, int src_phys,
                                    LiteStream stream);
LiteError litePagedAttnCopyInput(LitePagedAttn a, const void* h_q_fp16,
                                 const int* h_bt, const int* h_seq_lens);
LiteError litePagedAttnCopyOutput(LitePagedAttn a, void* h_out_fp16);


LiteError liteQuantBlockFp16ToI8(const uint16_t* h_k_fp16, const uint16_t* h_v_fp16,
                                 int num_heads, int head_dim, int block_size,
                                 void* out_k_i8, void* out_v_i8,
                                 float* out_scale_k, float* out_scale_v);

size_t litePagedAttnKvBytes(LitePagedAttn a);

LiteError litePagedAttnLaunch(LitePagedAttn a, LiteStream stream);
LiteError litePagedAttnLaunchB(LitePagedAttn a, int batch, LiteStream stream);

int   litePagedAttnNumSplits(LitePagedAttn a);  // 实际生效的 split 数
int   litePagedAttnSplitSize(LitePagedAttn a);  // 固定的每 split token 数（按最大长度预算）
// 输出缓冲的【设备地址】（Create 时固定）。TP all-gather 需要把每个 rank 半头输出的
// device 指针交给通信层汇聚，故显式暴露；该地址生命周期随 LitePagedAttn，调用方不释放。
void* litePagedAttnOutPtr(LitePagedAttn a);

// =====================================================================
// Communicator（张量并行 TP 通信抽象）
//
// 两层实现：
//  - SimComm（本地单卡）：world_size 个 rank 共享主 context，每 rank 一条独立 stream；
//    all-gather 用同 context 内 D2D 拷贝 + event join 模拟，单张卡即可验证
//    "按 head 切分 -> 各 rank 半头 attention -> all-gather 拼头" 的数据流正确性。
// 真机多卡 NCCL 通信域见下方 liteNcclGroup* API（-DLITE_WITH_NCCL=ON 时启用），
// 与 SimComm 是两个独立层次；SimComm 只是单进程内可见全部 rank 指针的协调器（host 传入
// d_in[]），用于单卡验证 TP 切分/汇聚数据流的正确性。
// =====================================================================
typedef struct LiteComm_s* LiteComm;

typedef struct {
    int world_size;   // rank 数（TP 并行度），必须 >=1
    int device;       // 预留：真机起始 device（rank r -> device+r）；SimComm 忽略
} LiteCommConfig;

LiteError liteCommCreateSim(const LiteCommConfig* cfg, LiteComm* out);
LiteError liteCommDestroy(LiteComm c);
int       liteCommWorldSize(LiteComm c);
// 取 rank r 的计算流：半头 PagedAttn 用 litePagedAttnLaunchB(..., rankStream) 投到该流。
LiteError liteCommRankStream(LiteComm c, int rank, LiteStream* out);
// 单进程模拟 all-gather（沿 shard/head 维）：每个 rank 输入 rows 行、每行 row_bytes
// （共 rows*row_bytes，行在 rank 内连续）；输出按 [row][rank][row_bytes] 摆位，即把 rank
// 轴并入 shard 轴 —— 当 world*行宽=完整 head 时，输出直接是 [rows][world*Hh][D]=[B][H][D]。
// 这对应真实 TP：ncclAllGather 原始结果为 [rank][B][Hh][D]，需 reshape/把 rank 轴并入 head
// 才能得到 [B][H][D]；SimComm 用逐行 D2D 直接摆好，省掉独立 transpose kernel。
// 各 rank 的行拷贝在其自己的流上（排在半头 kernel 之后），最后在 rank0 流上 wait 全部段
// event 做 join；返回后在 rank0 流上继续/同步即可见完整 d_out。
LiteError liteCommAllGatherSim(LiteComm c, void* d_out, const void* const* d_in,
                               int rows, size_t row_bytes);
LiteError liteCommBarrier(LiteComm c);  // 同步所有 rank 流（host join）

// =====================================================================
// NCCL 真机多卡通信域（云端 TP，仅 -DLITE_WITH_NCCL=ON 编译；声明不引 nccl.h）
//
// 每 rank 一张物理 device，使用其 primary context（cuDevicePrimaryCtxRetain，与 runtime
// API 同一上下文，NCCL 兼容性最好）；liteNcclGroupCreate 内部以每 rank 一个线程并发
// ncclCommInitRankConfig 建域。运行期在各 rank 的 worker 线程内先 liteSetCurrentContext
// (liteNcclRankCtx) 再建半头 PagedAttn / 调 all-gather（cuCtxSetCurrent 为线程局部）。
// all-gather 为集合操作，world 个 rank 必须并发、send_count 一致。
// =====================================================================
#ifdef LITE_WITH_NCCL
typedef struct LiteNcclGroup_s* LiteNcclGroup;
typedef struct {
    int world_size;   // rank 数（TP 并行度）
    int base_device;  // rank r -> device (base_device + r)
} LiteNcclGroupConfig;
LiteError liteNcclGroupCreate(const LiteNcclGroupConfig* cfg, LiteNcclGroup* out);
void      liteNcclGroupDestroy(LiteNcclGroup g);
LiteError liteNcclRankCtx(LiteNcclGroup g, int rank, LiteContext* ctx);    // 本 rank primary ctx
LiteError liteNcclRankStream(LiteNcclGroup g, int rank, LiteStream* out);  // 本 rank 计算/通信流
// fp16 all-gather：d_send 为本 rank send_count 个 fp16；d_recv 容量须 >= world*send_count，
// NCCL 输出按 [rank][send_count] 排列，调用方 reshape 把 rank 轴并入 head 轴。
LiteError liteNcclAllGatherFp16(LiteNcclGroup g, int rank,
                                const void* d_send, void* d_recv, size_t send_count);
LiteError liteNcclRankSync(LiteNcclGroup g, int rank);
#endif
