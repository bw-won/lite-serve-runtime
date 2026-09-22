// scheduler.cpp — Continuous batching 的 iteration-level 调度器（纯 host，不调 CUDA）。
//
// 每个 decode step：回收已完成请求（整组释放其 KV 物理块）-> 为在跑请求做块【增量增长】
// （容量不足时按 FCFS 抢占最年轻 running 请求）-> 按策略准入新请求 -> 把 running 请求紧凑
// 打包成本步 LiteStepMeta（前 n_active 行）。
//
// KV 块【增量分配 + 抢占重算（vLLM recompute 策略）】：
//   - 准入只给当前所需块（新请求 1 块、重算请求给覆盖已生成 token 的块），随后每跨一个
//     block 边界才再 alloc 1 块（grow-as-you-go），不一次性锁死整个生命周期的块。
//   - KV 容量不足时抢占【最年轻（最后进入）】的 running 请求（FCFS：最老请求优先完成、
//     无活锁）：释放其全部物理块给更老/新请求，被抢者退回 waiting 队头优先重算。
//   - 抢占只丢 KV（可重算的中间状态）、不丢已生成 token；恢复后从原进度继续，meta.resumed
//     在恢复第一步置 1，提示调用方把历史 K/V 重新填入【新分配】的物理块（recompute）。
//
// continuous=1：每个 step 都允许新请求补位 / 完成请求退出（in-flight batching）；
// continuous=0：一批凑满后必须整批全部完成，才放下一批（static batching，对照基准）；
//               抢占是批内 KV 重算，不改变 static 的准入时机。
//
// Prefix caching / COW（vLLM v1 Automatic Prefix Cache 语义）：
//  - 每个逻辑块按【内容 + 父块哈希】链式计算 block hash（block k 的哈希覆盖 token
//    0..end(k)，等价于链式混入），prompt 块在准入时经池的哈希注册表查找：命中即共享
//    （refcnt+1），未命中分配并注册；
//  - 共享块不可变：某请求要写入共享块（分叉点，必是最后一个未满块）时触发 COW ——
//    换新私有块、释放共享份额、meta.cow_src 上报旧块让调用方搬运数据（D2D）；
//  - 每步 commit 后对最后写入块 re-key（内容变了旧哈希失效，vLLM 同款语义）；
//  - 被抢占请求恢复时逐块重建哈希并查缓存：其他请求还在用的前缀块直接共享回来，
//    重算只需重填真正新建的块。
#include "lite_gpu.h"
#include <vector>
#include <deque>
#include <cstring>
#include <cstdint>
#include <new>
#include <utility>

namespace {

// ---- block hash（vLLM 式内容哈希链）----
// 顺序混入即等价于"父块哈希作种子、逐块续混"：h_k = mix(h_{k-1}, tokens(k)) 与对
// token 0..end(k) 从固定根开始顺序混入结果一致（混入是顺序复合的）。
constexpr uint64_t kHashRoot = 0x243f6a8885a308d3ull;   // block0 的父哈希（固定种子）
static inline uint64_t hash_mix(uint64_t hash, uint64_t val) {
    hash ^= val + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
    return hash;
}

// block k（含 0..end(k) 的 token，end(k)=min((k+1)*tpb, tokens.size())-1）的内容哈希。
// 返回值保证非 0（0 是池的"未注册"哨兵）。
static uint64_t chain_hash_at(const std::vector<int>& tokens, int k, int tpb) {
    uint64_t hash = kHashRoot;
    int tok_end = (k + 1) * tpb;
    if (tok_end > (int)tokens.size()) {
        tok_end = (int)tokens.size();
    }

    for (int i = 0; i < tok_end; ++i) {
        hash = hash_mix(hash, (uint32_t)tokens[i]);
    }

    return hash ? hash : 1;
}

struct CBRequest {
    int  req_id;
    int  target_len;
    int  generated;
    bool running;
    bool done;
    int  finish_step;
    int  preempted;     // 累计被抢占次数
    bool resume_flag;   // 本步是否为抢占后重算恢复的第一步（打包进 meta.resumed 后清零）
    bool preempted_pending;  // 已被抢占、待恢复（再准入的第一步上报 resume_flag）
    std::vector<int> phys;  // 该请求持有的物理 KV 块 id（逻辑块顺序 -> 物理 id，即 block table 行）
    std::vector<uint64_t> blk_hash;  // 与 phys 平行：各逻辑块当前注册哈希（0=未注册/私有）
    std::vector<int> out;   // 已产出 token（prompt + 生成的确定性 mock；抢占不丢 token）
};

}  // namespace

struct LiteScheduler_s {
    LiteKvPool pool;
    int max_seqs, tpb, mbps, continuous;
    int total_preempt;    // 累计抢占次数
    int total_cow;        // 累计 COW 次数（共享块分叉写）
    long cache_hits;      // 累计前缀缓存命中
    std::vector<CBRequest> reqs;
    std::deque<int> waiting;  // req_id FIFO（被抢占者插队头，优先重算）
    int steps;
};

LiteError liteSchedulerCreate(LiteKvPool pool, int max_seqs, int tokens_per_block,
                              int max_blocks_per_seq, int continuous, LiteScheduler* out) {
    if (!pool || !out || max_seqs <= 0 || tokens_per_block <= 0 || max_blocks_per_seq <= 0) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if (max_seqs > LITE_CB_MAX_SEQS || max_blocks_per_seq > LITE_CB_MAX_BLOCKS_PER_SEQ) {
        return LITE_ERROR_INVALID_VALUE;
    }

    LiteScheduler_s* s = new (std::nothrow) LiteScheduler_s();
    if (!s) {
        return LITE_ERROR_OUT_OF_MEMORY;
    }

    s->pool = pool;
    s->max_seqs = max_seqs;
    s->tpb = tokens_per_block;
    s->mbps = max_blocks_per_seq;
    s->continuous = continuous ? 1 : 0;
    s->total_preempt = 0;
    s->total_cow = 0;
    s->cache_hits = 0;
    s->steps = 0;
    *out = s;
    return LITE_SUCCESS;
}

void liteSchedulerDestroy(LiteScheduler s) {
    if (!s) {
        return;
    }

    for (auto& req : s->reqs) {
        for (int phys_id : req.phys) {
            liteKvPoolFreeBlock(s->pool, phys_id);
        }
    }

    delete s;
}

static LiteError add_request_impl(LiteScheduler_s* s, int target_len,
                                  const int* prompt_tokens, int prompt_len) {
    if (!s || target_len <= 0) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if (prompt_len < 0 || (prompt_len > 0 && !prompt_tokens)) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if (prompt_len > target_len) {
        return LITE_ERROR_INVALID_VALUE;
    }

    int n_needed_blks = (target_len + s->tpb - 1) / s->tpb;
    if (n_needed_blks > s->mbps) {
        return LITE_ERROR_INVALID_VALUE;
    }

    CBRequest req{};
    req.req_id = (int)s->reqs.size();
    req.target_len = target_len;
    req.generated = prompt_len;          // prompt 视为已就绪内容（KV 由调用方确认）
    req.running = false;
    req.done = false;
    req.finish_step = -1;
    req.preempted = 0;
    req.resume_flag = false;
    req.preempted_pending = false;
    for (int i = 0; i < prompt_len; ++i) {
        req.out.push_back(prompt_tokens[i]);
    }

    s->reqs.push_back(std::move(req));
    s->waiting.push_back(req.req_id);
    return LITE_SUCCESS;
}

int liteSchedulerAddRequest(LiteScheduler s, int target_len) {
    if (add_request_impl(s, target_len, nullptr, 0) != LITE_SUCCESS) {
        return -1;
    }

    return (int)s->reqs.size() - 1;
}

LiteError liteSchedulerAddRequestEx(LiteScheduler s, int target_len,
                                    const int* prompt_tokens, int prompt_len) {
    return add_request_impl(s, target_len, prompt_tokens, prompt_len);
}

static int running_count(const LiteScheduler_s* s) {
    int n_running = 0;
    for (const auto& req : s->reqs) {
        if (req.running) {
            ++n_running;
        }
    }

    return n_running;
}

// 已生成 generated 个 token、下一步要算第 generated 个（0-index）时，存放 generated+1
// 个 token 需要的块数。
static int getBlksNumForGenerated(int generated, int tpb) {
    return (generated + 1 + tpb - 1) / tpb;
}

static void doPreempt(LiteScheduler_s* s, int victim_rid) {
    CBRequest& victim = s->reqs[victim_rid];
    for (int phys_id : victim.phys) {
        liteKvPoolFreeBlock(s->pool, phys_id);
    }

    victim.phys.clear();
    victim.blk_hash.clear();
    victim.running = false;
    victim.preempted++;
    victim.preempted_pending = true;
    s->total_preempt++;
    s->waiting.push_front(victim_rid);
}

static int getVictimReqId(LiteScheduler_s* s, int min_victim_rid) {
    int best_victim = -1;
    for (const auto& cand : s->reqs) {
        if (cand.running && cand.req_id > min_victim_rid &&
            (best_victim < 0 || cand.req_id > best_victim)) {
            best_victim = cand.req_id;
        }
    }

    return best_victim;
}

// 尝试准入队头请求；成功返回 true。
// prefix caching：有内容的逻辑块（prompt 块 / 重算恢复请求的历史块）先按内容哈希查池
// 注册表——命中即共享（refcnt+1，不占空闲块），未命中才分配并注册。因此准入对空闲
// 块的真实需求 = 未命中的块数，抢占只在分配真正失败时发生（不做超量预留）。
static bool try_admit_one(LiteScheduler_s* s) {
    if (s->waiting.empty()) {
        return false;
    }

    int req_id = s->waiting.front();
    s->waiting.pop_front();
    CBRequest& req = s->reqs[req_id];

    // 新请求+重算请求
    int n_needed_blks = getBlksNumForGenerated(req.generated, s->tpb);
    int min_victim_rid = (req.generated > 0) ? req.req_id : -1;

    std::vector<int> granted_phys;
    std::vector<uint64_t> granted_hash;
    granted_phys.reserve(n_needed_blks);
    granted_hash.reserve(n_needed_blks);

    for (int k = 0; k < n_needed_blks; ++k) {
        int blk_first_tok = k * s->tpb;
        int n_exist_toks = (int)req.out.size() - blk_first_tok;  // 该逻辑块已有的 token 数（0=空块）
        if (n_exist_toks > s->tpb) {
            n_exist_toks = s->tpb;
        }

        int phys_id = -1;
        uint64_t reg_hash = 0;   // 本逻辑块最终登记的哈希（0=未登记：空块/注册被拒）
        if (n_exist_toks > 0) {
            uint64_t content_hash = chain_hash_at(req.out, k, s->tpb);
            int hit_phys = liteKvPoolLookupHash(s->pool, content_hash);
            if (hit_phys >= 0) {
                phys_id = hit_phys;
                reg_hash = content_hash;
                ++s->cache_hits;
            }
        }

        if (phys_id < 0) {
            phys_id = liteKvPoolAllocBlock(s->pool);
            while (phys_id < 0) {
                int victim_rid = getVictimReqId(s, min_victim_rid);
                if (victim_rid < 0) {  // 无可抢占放回队头
                    for (int granted_pid : granted_phys) {
                        liteKvPoolFreeBlock(s->pool, granted_pid);
                    }

                    s->waiting.push_front(req_id);
                    return false;
                }

                doPreempt(s, victim_rid);
                phys_id = liteKvPoolAllocBlock(s->pool);
            }

            if (n_exist_toks > 0) {  // 空块无内容不注册；有内容块注册哈希
                uint64_t content_hash = chain_hash_at(req.out, k, s->tpb);
                if (liteKvPoolRegisterHash(s->pool, phys_id, content_hash) == LITE_SUCCESS) {
                    reg_hash = content_hash;
                }
            }
        }

        granted_phys.push_back(phys_id);
        granted_hash.push_back(reg_hash);
    }

    req.phys = std::move(granted_phys);
    req.blk_hash = std::move(granted_hash);
    req.running = true;
    req.resume_flag = req.preempted_pending;  // 仅抢占恢复第一步上报重算；prompt 准入不算恢复
    req.preempted_pending = false;
    return true;
}

static void growRunning(LiteScheduler_s* s) {
    std::vector<int> runningReqIds;
    runningReqIds.reserve(s->reqs.size());
    for (const auto& req : s->reqs) {
        if (req.running) {
            runningReqIds.push_back(req.req_id);
        }
    }

    for (int req_id : runningReqIds) {
        CBRequest& req = s->reqs[req_id];
        if (!req.running) {
            continue;  // 本轮可能已被更老的请求抢占
        }

        int numNeedBlks = getBlksNumForGenerated(req.generated, s->tpb);
        while ((int)req.phys.size() < numNeedBlks) {
            int phys_id = liteKvPoolAllocBlock(s->pool);
            if (phys_id < 0) {
                int victimRid = getVictimReqId(s, req_id);
                if (victimRid < 0) {
                    // 最新进入的请求
                    doPreempt(s, req_id);
                    break;
                }

                doPreempt(s, victimRid);
                phys_id = liteKvPoolAllocBlock(s->pool);
            }

            if (phys_id >= 0) {
                req.phys.push_back(phys_id);
                req.blk_hash.push_back(0);
            }
        }
    }
}

int liteSchedulerStep(LiteScheduler s, LiteStepMeta* meta) {
    if (!s || !meta) {
        return 1;
    }

    // 1) 回收：上一轮已生成完的 running 请求
    for (auto& req : s->reqs) {
        if (req.running && req.generated >= req.target_len) {
            for (int phys_id : req.phys) {
                liteKvPoolFreeBlock(s->pool, phys_id);
            }

            req.phys.clear();
            req.running = false;
            req.done = true;
            req.finish_step = s->steps;  // 在第 steps 个 step 结束时完成
        }
    }

    // 2) 块增量增长：优先满足在跑请求（在跑的不被新准入饿死），容量不足时抢占。
    growRunning(s);

    // 3) 准入：continuous 每步补位；static 仅在一批全部结束（running 清空）时凑满下一批。
    int n_running = running_count(s);
    bool admit_open = s->continuous ? true : (n_running == 0);
    if (admit_open) {
        while (n_running < s->max_seqs && try_admit_one(s)) {
            ++n_running;
        }
    }

    // 4) 组装本步元数据：running 请求紧凑打包到前 n_active 行（图固定按 max_seqs 行捕获）。
    std::memset(meta, 0, sizeof(*meta));
    int row = 0;
    for (auto& req : s->reqs) {
        if (!req.running) {
            continue;
        }

        int k = req.generated / s->tpb;
        if (k < (int)req.phys.size() && liteKvPoolBlockRefs(s->pool, req.phys[k]) > 1) {
            int shared_phys = req.phys[k];        // 被分叉的共享旧块
            int new_priv_phys = liteKvPoolAllocBlock(s->pool);
            if (new_priv_phys < 0) {  // 腾不出私有块：自我换出（把块让给更老请求）
                doPreempt(s, req.req_id);
                continue;
            }

            liteKvPoolFreeBlock(s->pool, shared_phys);
            req.phys[k] = new_priv_phys;
            req.blk_hash[k] = 0;          // 新块未注册；commit 后按新内容重键
            meta->cow_src[row] = shared_phys;
            ++s->total_cow;
        }

        meta->slot_req[row] = req.req_id;
        meta->positions[row] = req.generated;
        meta->resumed[row] = req.resume_flag ? 1 : 0;
        req.resume_flag = false;  // 仅恢复第一步上报
        for (int blk = 0; blk < (int)req.phys.size() && blk < s->mbps; ++blk) {
            meta->btable[row * s->mbps + blk] = req.phys[blk];
        }
        ++row;
    }

    meta->n_active = row;
    if (row > 0) {
        ++s->steps;
    }

    // 全部完成：无 running 且 waiting 为空。
    return (row == 0 && s->waiting.empty()) ? 1 : 0;
}

void liteSchedulerCommit(LiteScheduler s, const LiteStepMeta* meta, const int* out_tokens) {
    if (!s || !meta || !out_tokens) {
        return;
    }

    for (int i = 0; i < meta->n_active; ++i) {
        int req_id = meta->slot_req[i];
        if (req_id < 0 || req_id >= (int)s->reqs.size()) {
            continue;
        }

        CBRequest& req = s->reqs[req_id];
        req.out.push_back(out_tokens[i]);
        req.generated++;

        // re-key 最后写入的逻辑块：内容变了，旧哈希注册失效（vLLM 每步 re-key 语义）。
        // COW 新块旧值本就是 0（未注册），直接按新内容注册。
        int k = (req.generated - 1) / s->tpb;
        if (k < (int)req.blk_hash.size()) {
            if (req.blk_hash[k] != 0) {
                liteKvPoolUnregisterHash(s->pool, req.phys[k]);
            }

            uint64_t content_hash = chain_hash_at(req.out, k, s->tpb);
            req.blk_hash[k] = (liteKvPoolRegisterHash(s->pool, req.phys[k], content_hash) == LITE_SUCCESS)
                                  ? content_hash : 0;
        }
    }
}

int liteSchedulerNumFinished(LiteScheduler s) {
    if (!s) {
        return 0;
    }

    int n_done = 0;
    for (const auto& req : s->reqs) {
        if (req.done) {
            ++n_done;
        }
    }

    return n_done;
}

int liteSchedulerNumSteps(LiteScheduler s) {
    return s ? s->steps : 0;
}

int liteSchedulerGetResult(LiteScheduler s, int req_id, int* out_tokens, int cap) {
    if (!s || req_id < 0 || req_id >= (int)s->reqs.size()) {
        return -1;
    }

    const CBRequest& req = s->reqs[req_id];
    int n_tokens = (int)req.out.size();
    if (out_tokens && cap > 0) {
        int n_copy = n_tokens < cap ? n_tokens : cap;
        for (int i = 0; i < n_copy; ++i) {
            out_tokens[i] = req.out[i];
        }
    }

    return n_tokens;
}

int liteSchedulerFinishStep(LiteScheduler s, int req_id) {
    if (!s || req_id < 0 || req_id >= (int)s->reqs.size()) {
        return -1;
    }

    return s->reqs[req_id].finish_step;
}

int liteSchedulerNumPreemptions(LiteScheduler s) {
    return s ? s->total_preempt : 0;
}

int liteSchedulerRequestPreempted(LiteScheduler s, int req_id) {
    if (!s || req_id < 0 || req_id >= (int)s->reqs.size()) {
        return 0;
    }

    return s->reqs[req_id].preempted;
}

int liteSchedulerNumCacheHits(LiteScheduler s) {
    return s ? (int)s->cache_hits : 0;
}

int liteSchedulerNumCow(LiteScheduler s) {
    return s ? s->total_cow : 0;
}
