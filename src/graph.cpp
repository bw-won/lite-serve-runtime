#include "lite_gpu.h"
#include "lite_gpu_internal.h"
#include <cstdio>

LiteError liteStreamBeginCapture(LiteStream stream) {
    if (!stream) return LITE_ERROR_INVALID_VALUE;
    // 使用 CU_STREAM_CAPTURE_MODE_GLOBAL 确保捕获所有相关流的操作
    CUresult err = cuStreamBeginCapture(stream->handle, CU_STREAM_CAPTURE_MODE_GLOBAL);
    return (err == CUDA_SUCCESS) ? LITE_SUCCESS : LITE_ERROR_DRIVER_API;
}

LiteError liteStreamEndCapture(LiteStream stream, LiteGraph* graph) {
    if (!stream || !graph) return LITE_ERROR_INVALID_VALUE;
    
    LiteGraph_s* g = new LiteGraph_s();
    CUresult err = cuStreamEndCapture(stream->handle, &g->handle);
    if (err != CUDA_SUCCESS) {
        delete g;
        return LITE_ERROR_DRIVER_API;
    }
    *graph = g;
    return LITE_SUCCESS;
}

LiteError liteGraphInstantiate(LiteGraph graph, LiteGraphExec* exec) {
    if (!graph || !exec) return LITE_ERROR_INVALID_VALUE;
    
    LiteGraphExec_s* e = new LiteGraphExec_s();
    // CUDA 12.0+ 推荐使用 cuGraphInstantiate，旧版本用 cuGraphInstantiateWithFlags
    CUresult err = cuGraphInstantiate(&e->handle, graph->handle, 0);
    if (err != CUDA_SUCCESS) {
        delete e;
        return LITE_ERROR_DRIVER_API;
    }
    *exec = e;
    return LITE_SUCCESS;
}

LiteError liteGraphLaunch(LiteGraphExec exec, LiteStream stream) {
    if (!exec || !stream) return LITE_ERROR_INVALID_VALUE;
    CUresult err = cuGraphLaunch(exec->handle, stream->handle);
    return (err == CUDA_SUCCESS) ? LITE_SUCCESS : LITE_ERROR_DRIVER_API;
}

LiteError liteGraphDestroy(LiteGraph graph) {
    if (!graph) return LITE_ERROR_INVALID_VALUE;
    cuGraphDestroy(graph->handle);
    delete graph;
    return LITE_SUCCESS;
}

LiteError liteGraphExecDestroy(LiteGraphExec exec) {
    if (!exec) return LITE_ERROR_INVALID_VALUE;
    cuGraphExecDestroy(exec->handle);
    delete exec;
    return LITE_SUCCESS;
}
