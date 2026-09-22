#include "lite_gpu.h"
#include <vector>
#include <unordered_map>
#include <new>

struct LiteKvPool_s {
    LiteKvPoolConfig cfg;
    std::vector<int> refcnt;                      // 每物理块引用计数（0=空闲）
    std::vector<uint64_t> block_hash;             // 每物理块当前注册哈希（0=未注册）
    std::vector<int> free_ids;                    // 空闲物理块 id
    std::unordered_map<uint64_t, int> hash_index; // 内容哈希 -> 物理块 id
    long cache_hits;
};

static bool pool_valid_id(const LiteKvPool_s* pool, int phys_id) {
    return phys_id >= 0 && phys_id < pool->cfg.num_phys_blocks;
}

LiteError liteKvPoolCreate(const LiteKvPoolConfig* cfg, LiteKvPool* out) {
    if (!cfg || !out) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if (cfg->tokens_per_block <= 0 || cfg->max_blocks_per_seq <= 0 ||
        cfg->max_seqs <= 0 || cfg->num_phys_blocks <= 0) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if (cfg->max_seqs > LITE_CB_MAX_SEQS ||
        cfg->max_blocks_per_seq > LITE_CB_MAX_BLOCKS_PER_SEQ) {
        return LITE_ERROR_INVALID_VALUE;
    }

    LiteKvPool_s* pool = new (std::nothrow) LiteKvPool_s();
    if (!pool) {
        return LITE_ERROR_OUT_OF_MEMORY;
    }

    pool->cfg = *cfg;
    pool->refcnt.assign((size_t)cfg->num_phys_blocks, 0);
    pool->block_hash.assign((size_t)cfg->num_phys_blocks, 0);
    pool->cache_hits = 0;
    pool->free_ids.reserve(cfg->num_phys_blocks);

    for (int i = 0; i < cfg->num_phys_blocks; ++i) {
        pool->free_ids.push_back(i);
    }

    *out = pool;
    return LITE_SUCCESS;
}

LiteError liteKvPoolDestroy(LiteKvPool pool) {
    delete pool;
    return LITE_SUCCESS;
}

int liteKvPoolAllocBlock(LiteKvPool pool) {
    if (!pool || pool->free_ids.empty()) {
        return -1;
    }

    int id = pool->free_ids.back();
    pool->free_ids.pop_back();
    pool->refcnt[id] = 1;
    pool->block_hash[id] = 0;
    return id;
}

static void pool_unregister(LiteKvPool_s* pool, int phys_id) {
    uint64_t h = pool->block_hash[phys_id];
    if (!h) {
        return;
    }

    auto it = pool->hash_index.find(h);
    if (it != pool->hash_index.end() && it->second == phys_id) {
        pool->hash_index.erase(it);
    }

    pool->block_hash[phys_id] = 0;
}

int liteKvPoolFreeBlock(LiteKvPool pool, int phys_id) {
    if (!pool || !pool_valid_id(pool, phys_id)) {
        return -1;
    }

    if (pool->refcnt[phys_id] <= 0) {
        return -1;
    }

    if (--pool->refcnt[phys_id] == 0) {
        pool_unregister(pool, phys_id);
        pool->free_ids.push_back(phys_id);
        return 0;
    }

    return pool->refcnt[phys_id];
}

void liteKvPoolRetainBlock(LiteKvPool pool, int phys_id) {
    if (!pool || !pool_valid_id(pool, phys_id)) {
        return;
    }

    if (pool->refcnt[phys_id] <= 0) {
        return;
    }

    ++pool->refcnt[phys_id];
}

int liteKvPoolBlockRefs(LiteKvPool pool, int phys_id) {
    if (!pool || !pool_valid_id(pool, phys_id)) {
        return -1;
    }

    return pool->refcnt[phys_id];
}

int liteKvPoolFreeBlocks(LiteKvPool pool) {
    return pool ? (int)pool->free_ids.size() : 0;
}

int liteKvPoolLookupHash(LiteKvPool pool, uint64_t hash) {
    if (!pool || hash == 0) {
        return -1;
    }

    auto it = pool->hash_index.find(hash);
    if (it == pool->hash_index.end()) {
        return -1;
    }

    ++pool->cache_hits;
    ++pool->refcnt[it->second];
    return it->second;
}

LiteError liteKvPoolRegisterHash(LiteKvPool pool, int phys_id, uint64_t hash) {
    if (!pool || hash == 0 || !pool_valid_id(pool, phys_id)) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if (pool->refcnt[phys_id] <= 0) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if (pool->hash_index.count(hash)) {
        return LITE_ERROR_INVALID_VALUE;
    }

    pool->block_hash[phys_id] = hash;
    pool->hash_index.emplace(hash, phys_id);
    return LITE_SUCCESS;
}

LiteError liteKvPoolUnregisterHash(LiteKvPool pool, int phys_id) {
    if (!pool || !pool_valid_id(pool, phys_id)) {
        return LITE_ERROR_INVALID_VALUE;
    }

    pool_unregister(pool, phys_id);
    return LITE_SUCCESS;
}

int liteKvPoolCacheHits(LiteKvPool pool) {
    return pool ? (int)pool->cache_hits : 0;
}
