#pragma once
#include "lite_gpu.h"
#include <cuda.h>
#include <vector>

struct LiteStream_s {
    CUstream handle;
};

struct LiteEvent_s {
    CUevent handle;
};

struct LiteGraph_s {
    CUgraph handle;
    std::vector<LiteGraphNode_s*> tracked_nodes;
};

struct LiteGraphExec_s {
    CUgraphExec handle;
};

struct LiteGraphNode_s {
    CUgraphNode handle;
    bool        invalid;
    LiteGraph_s* parent_graph;
};
