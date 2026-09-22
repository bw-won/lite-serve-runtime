#pragma once
#include "lite_gpu.h"
#include <cuda.h>

struct LiteStream_s {
    CUstream handle;
};

struct LiteEvent_s {
    CUevent handle;
};

struct LiteGraph_s {
    CUgraph handle;
};

struct LiteGraphExec_s {
    CUgraphExec handle;
};
