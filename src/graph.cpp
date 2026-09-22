#include "lite_gpu.h"
#include "lite_gpu_internal.h"
#include <cstdio>
#include <cstdint>
#include <type_traits>

static_assert(sizeof(CUstream) == sizeof(void*), "CUstream size mismatch");
static_assert(sizeof(CUgraphNode) == sizeof(void*), "CUgraphNode size mismatch");

LiteError liteStreamBeginCaptureEx(LiteStream stream, LiteCaptureMode mode) {
    if (!stream) {
        return LITE_ERROR_INVALID_VALUE;
    }

    CUstreamCaptureMode cu_mode;
    switch (mode) {
        case LITE_CAPTURE_GLOBAL:
            cu_mode = CU_STREAM_CAPTURE_MODE_GLOBAL;
            break;
        case LITE_CAPTURE_THREAD_LOCAL:
            cu_mode = CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
            break;
        case LITE_CAPTURE_RELAXED:
            cu_mode = CU_STREAM_CAPTURE_MODE_RELAXED;
            break;
        default:
            return LITE_ERROR_INVALID_VALUE;
    }

    CUresult err = cuStreamBeginCapture(stream->handle, cu_mode);
    if (err != CUDA_SUCCESS) {
        return LITE_ERROR_DRIVER_API;
    }

    return LITE_SUCCESS;
}

LiteError liteStreamBeginCapture(LiteStream stream) {
    return liteStreamBeginCaptureEx(stream, LITE_CAPTURE_THREAD_LOCAL);
}

LiteError liteStreamEndCapture(LiteStream stream, LiteGraph* graph) {
    if (!stream || !graph) {
        return LITE_ERROR_INVALID_VALUE;
    }

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
    if (!graph || !exec) {
        return LITE_ERROR_INVALID_VALUE;
    }

    LiteGraphExec_s* e = new LiteGraphExec_s();
    CUresult err = cuGraphInstantiate(&e->handle, graph->handle, 0);
    if (err != CUDA_SUCCESS) {
        delete e;
        return LITE_ERROR_DRIVER_API;
    }

    *exec = e;
    return LITE_SUCCESS;
}

LiteError liteGraphExecUpdate(LiteGraphExec exec, LiteGraph new_param_graph, LiteGraphExecUpdateResult* out_result) {
    if (!exec || !new_param_graph || !out_result) {
        return LITE_ERROR_INVALID_VALUE;
    }

    CUgraphExecUpdateResultInfo result_info{};
    CUresult cu_err = cuGraphExecUpdate(exec->handle, new_param_graph->handle, &result_info);
    // cuGraphExecUpdate 返回 CUDA_SUCCESS 后仍要看 result_info.result
    if (cu_err != CUDA_SUCCESS) {
        fprintf(stderr, "cuGraphExecUpdate driver err=%d\n", (int)cu_err);
        return LITE_ERROR_GRAPH_UPDATE;
    }

    switch (result_info.result) {
        case CU_GRAPH_EXEC_UPDATE_SUCCESS:
            *out_result = LITE_GRAPH_UPDATE_SUCCESS;
            break;
        case CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED:
            *out_result = LITE_GRAPH_UPDATE_ERROR_TOPOLOGY_MISMATCH;
            break;
        case CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED:
        case CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED:
        case CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED:
        case CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED:
        case CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE:
        case CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED:
        case CU_GRAPH_EXEC_UPDATE_ERROR:
            *out_result = LITE_GRAPH_UPDATE_ERROR_NODE_FAILED;
            break;
        default:
            *out_result = LITE_GRAPH_UPDATE_ERROR_UNKNOWN;
            break;
    }

    if (result_info.errorNode != nullptr) {
        fprintf(stderr, "[GraphExecUpdate] errorNode present, update result=%d\n", (int)result_info.result);
    }

    if (*out_result != LITE_GRAPH_UPDATE_SUCCESS) {
        return LITE_ERROR_GRAPH_UPDATE;
    }

    return LITE_SUCCESS;
}

LiteError liteGraphLaunch(LiteGraphExec exec, LiteStream stream) {
    if (!exec || !stream) {
        return LITE_ERROR_INVALID_VALUE;
    }

    CUresult err = cuGraphLaunch(exec->handle, stream->handle);
    return (err == CUDA_SUCCESS) ? LITE_SUCCESS : LITE_ERROR_DRIVER_API;
}

LiteError liteGraphDestroy(LiteGraph graph) {
    if (!graph) {
        return LITE_ERROR_INVALID_VALUE;
    }

    for (auto& node : graph->tracked_nodes) {
        node->invalid = true;
    }

    for (auto* node : graph->tracked_nodes) {
        delete node;
    }

    graph->tracked_nodes.clear();
    cuGraphDestroy(graph->handle);
    delete graph;
    return LITE_SUCCESS;
}

LiteError liteGraphExecDestroy(LiteGraphExec exec) {
    if (!exec) {
        return LITE_ERROR_INVALID_VALUE;
    }

    cuGraphExecDestroy(exec->handle);
    delete exec;
    return LITE_SUCCESS;
}

LiteError liteGraphGetNodeCount(LiteGraph graph, size_t* out_node_cnt) {
    if (!graph || !out_node_cnt) {
        return LITE_ERROR_INVALID_VALUE;
    }

    size_t cnt = 0;
    CUresult err = cuGraphGetNodes(graph->handle, nullptr, &cnt);
    if (err != CUDA_SUCCESS) {
        return LITE_ERROR_DRIVER_API;
    }

    *out_node_cnt = cnt;
    return LITE_SUCCESS;
}

LiteError liteGraphGetNodeByIndex(LiteGraph graph, size_t idx, LiteGraphNode* out_node) {
    if (!graph || !out_node) {
        return LITE_ERROR_INVALID_VALUE;
    }

    size_t total_nodes = 0;
    CUresult err = cuGraphGetNodes(graph->handle, nullptr, &total_nodes);
    if (err != CUDA_SUCCESS) {
        return LITE_ERROR_DRIVER_API;
    }

    if (idx >= total_nodes) {
        return LITE_ERROR_INVALID_VALUE;
    }

    CUgraphNode* tmp_buf = new CUgraphNode[total_nodes];
    err = cuGraphGetNodes(graph->handle, tmp_buf, &total_nodes);
    if (err != CUDA_SUCCESS) {
        delete[] tmp_buf;
        return LITE_ERROR_DRIVER_API;
    }

    LiteGraphNode_s* node_wrap = new LiteGraphNode_s{};
    node_wrap->handle = tmp_buf[idx];
    node_wrap->invalid = false;
    node_wrap->parent_graph = graph;
    graph->tracked_nodes.push_back(node_wrap);

    *out_node = node_wrap;
    delete[] tmp_buf;
    return LITE_SUCCESS;
}

LiteError liteGraphNodeGetType(LiteGraphNode node, LiteGraphNodeType* out_type) {
    if (!node || !out_type) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if (node->invalid) {
        return LITE_ERROR_GRAPH_NODE_INVALID;
    }

    CUgraphNodeType cu_type;
    CUresult err = cuGraphNodeGetType(node->handle, &cu_type);
    if (err != CUDA_SUCCESS) {
        return LITE_ERROR_DRIVER_API;
    }

    switch (cu_type) {
        case CU_GRAPH_NODE_TYPE_KERNEL:
            *out_type = LITE_GRAPH_NODE_KERNEL;
            break;
        case CU_GRAPH_NODE_TYPE_MEMCPY:
            *out_type = LITE_GRAPH_NODE_MEMCPY;
            break;
        case CU_GRAPH_NODE_TYPE_MEMSET:
            *out_type = LITE_GRAPH_NODE_MEMSET;
            break;
        case CU_GRAPH_NODE_TYPE_EVENT_RECORD:
            *out_type = LITE_GRAPH_NODE_EVENT;
            break;
        default:
            *out_type = LITE_GRAPH_NODE_OTHER;
            break;
    }

    return LITE_SUCCESS;
}

LiteError liteGraphMemcpyNodeGetInfo(LiteGraphNode node, uint64_t* dst_out, uint64_t* src_out, size_t* width_out, uint32_t* height_out, uint32_t* depth_out) {
    if (!node || !dst_out || !src_out || !width_out || !height_out || !depth_out) {
        return LITE_ERROR_INVALID_VALUE;
    }

    LiteGraphNodeType nt;
    auto ret = liteGraphNodeGetType(node, &nt);
    if (ret != LITE_SUCCESS) {
        return ret;
    }

    if (nt != LITE_GRAPH_NODE_MEMCPY) {
        return LITE_ERROR_GRAPH_NODE_TYPE;
    }

    CUDA_MEMCPY3D copy3d{};
    CUresult err = cuGraphMemcpyNodeGetParams(node->handle, &copy3d);
    if (err != CUDA_SUCCESS) {
        return LITE_ERROR_DRIVER_API;
    }

    *dst_out = static_cast<uint64_t>(copy3d.dstDevice);
    *src_out = static_cast<uint64_t>(copy3d.srcDevice);
    *width_out = copy3d.WidthInBytes;
    *height_out = copy3d.Height;
    *depth_out = copy3d.Depth;
    return LITE_SUCCESS;
}

LiteError liteGraphExecMemcpyNodeSetParams(LiteGraphExec exec, LiteGraphNode node, void* dst, void* src) {
    if (!exec || !node || !dst || !src) {
        return LITE_ERROR_INVALID_VALUE;
    }

    LiteGraphNodeType nt;
    LiteError ret = liteGraphNodeGetType(node, &nt);
    if (ret != LITE_SUCCESS) {
        return ret;
    }

    if (nt != LITE_GRAPH_NODE_MEMCPY) {
        return LITE_ERROR_GRAPH_NODE_TYPE;
    }

    CUDA_MEMCPY3D copy3d{};
    CUresult err = cuGraphMemcpyNodeGetParams(node->handle, &copy3d);
    if (err != CUDA_SUCCESS) {
        return LITE_ERROR_DRIVER_API;
    }

    CUdeviceptr dst_dev = reinterpret_cast<CUdeviceptr>(dst);
    CUdeviceptr src_dev = reinterpret_cast<CUdeviceptr>(src);

    copy3d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy3d.srcDevice    = src_dev;
    copy3d.srcHost       = nullptr;
    copy3d.srcXInBytes   = 0;
    copy3d.srcY          = 0;
    copy3d.srcZ          = 0;

    copy3d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy3d.dstDevice    = dst_dev;
    copy3d.dstHost      = nullptr;
    copy3d.dstXInBytes  = 0;
    copy3d.dstY         = 0;
    copy3d.dstZ         = 0;

    err = cuGraphExecMemcpyNodeSetParams(exec->handle, node->handle, &copy3d, nullptr);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[liteGraphExecMemcpyNodeSetParams] driver err=%d\n", (int)err);
        return LITE_ERROR_DRIVER_API;
    }

    return LITE_SUCCESS;
}

LiteError liteGraphExecKernelNodeSetParams(LiteGraphExec exec, LiteGraphNode node, void** kernel_args) {
    if (!exec || !node || !kernel_args) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if (node->invalid) {
        return LITE_ERROR_GRAPH_NODE_INVALID;
    }

    LiteGraphNodeType nt;
    LiteError ret = liteGraphNodeGetType(node, &nt);
    if (ret != LITE_SUCCESS) {
        return ret;
    }

    if (nt != LITE_GRAPH_NODE_KERNEL) {
        return LITE_ERROR_GRAPH_NODE_TYPE;
    }

    CUDA_KERNEL_NODE_PARAMS node_params{};
    CUresult cu_err = cuGraphKernelNodeGetParams(node->handle, &node_params);
    if (cu_err != CUDA_SUCCESS) {
        fprintf(stderr, "cuGraphKernelNodeGetParams driver err=%d\n", (int)cu_err);
        return LITE_ERROR_DRIVER_API;
    }

    node_params.kernelParams = kernel_args;
    node_params.extra = nullptr;

    cu_err = cuGraphExecKernelNodeSetParams(exec->handle, node->handle, &node_params);
    if (cu_err != CUDA_SUCCESS) {
        fprintf(stderr, "liteGraphExecKernelNodeSetParams driver err=%d\n", (int)cu_err);
        return LITE_ERROR_DRIVER_API;
    }

    return LITE_SUCCESS;
}

LiteError liteGraphExecKernelNodeSetParamsEx(LiteGraphExec exec, LiteGraphNode node,
                                             void* func, void** kernel_args) {
    if (!exec || !node || !func || !kernel_args) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if (node->invalid) {
        return LITE_ERROR_GRAPH_NODE_INVALID;
    }

    LiteGraphNodeType nt;
    LiteError ret = liteGraphNodeGetType(node, &nt);
    if (ret != LITE_SUCCESS) {
        return ret;
    }

    if (nt != LITE_GRAPH_NODE_KERNEL) {
        return LITE_ERROR_GRAPH_NODE_TYPE;
    }

    CUDA_KERNEL_NODE_PARAMS node_params{};
    CUresult cu_err = cuGraphKernelNodeGetParams(node->handle, &node_params);
    if (cu_err != CUDA_SUCCESS) {
        fprintf(stderr, "cuGraphKernelNodeGetParams driver err=%d\n", (int)cu_err);
        return LITE_ERROR_DRIVER_API;
    }

    // 换 kernel 本体：func 由调用方以 CUfunction 传入（void* 承载）。
    // 调用方需保证新 kernel 与旧 kernel 形参签名一致（实参数组长度/类型匹配）。
    node_params.func         = (CUfunction)func;
    node_params.kernelParams = kernel_args;
    node_params.extra        = nullptr;  // 与 kernelParams 互斥

    cu_err = cuGraphExecKernelNodeSetParams(exec->handle, node->handle, &node_params);
    if (cu_err != CUDA_SUCCESS) {
        fprintf(stderr, "liteGraphExecKernelNodeSetParamsEx driver err=%d\n", (int)cu_err);
        return LITE_ERROR_DRIVER_API;
    }

    return LITE_SUCCESS;
}

LiteError liteGraphMemsetNodeGetInfo(LiteGraphNode node, uint64_t* dst_out, uint32_t* fill_value_out, size_t* byte_count_out) {
    if (!node || !dst_out || !fill_value_out || !byte_count_out) {
        return LITE_ERROR_INVALID_VALUE;
    }

    LiteGraphNodeType nt;
    auto ret = liteGraphNodeGetType(node, &nt);
    if (ret != LITE_SUCCESS) {
        return ret;
    }

    if (nt != LITE_GRAPH_NODE_MEMSET) {
        return LITE_ERROR_GRAPH_NODE_TYPE;
    }

    CUDA_MEMSET_NODE_PARAMS params{};
    CUresult err = cuGraphMemsetNodeGetParams(node->handle, &params);
    if (err != CUDA_SUCCESS) {
        return LITE_ERROR_DRIVER_API;
    }

    *dst_out = static_cast<uint64_t>(params.dst);
    *fill_value_out = params.value;
    *byte_count_out = params.elementSize * params.width * params.height;
    return LITE_SUCCESS;
}

LiteError liteGraphExecMemsetNodeSetParams(LiteGraphExec exec, LiteGraphNode node, void* dst, uint32_t fill_val, size_t byte_count) {
    if (!exec || !node || !dst) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if (node->invalid) {
        return LITE_ERROR_GRAPH_NODE_INVALID;
    }

    LiteGraphNodeType nt;
    LiteError ret = liteGraphNodeGetType(node, &nt);
    if (ret != LITE_SUCCESS) {
        return ret;
    }

    if (nt != LITE_GRAPH_NODE_MEMSET) {
        return LITE_ERROR_GRAPH_NODE_TYPE;
    }

    CUDA_MEMSET_NODE_PARAMS params{};
    CUresult err = cuGraphMemsetNodeGetParams(node->handle, &params);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[liteGraphExecMemsetNodeSetParams] get params failed err=%d\n", (int)err);
        return LITE_ERROR_DRIVER_API;
    }

    params.dst = reinterpret_cast<CUdeviceptr>(dst);
    params.value = fill_val;

    if (params.elementSize == 0) {
        return LITE_ERROR_INVALID_VALUE;
    }

    size_t elem_total = byte_count / params.elementSize;
    if (elem_total * params.elementSize != byte_count) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if (params.height == 0) {
        return LITE_ERROR_INVALID_VALUE;
    }

    if ((elem_total % params.height) != 0) {
        return LITE_ERROR_INVALID_VALUE;
    }

    params.width = elem_total / params.height;

    err = cuGraphExecMemsetNodeSetParams(exec->handle, node->handle, &params, nullptr);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[liteGraphExecMemsetNodeSetParams] driver err=%d\n", (int)err);
        return LITE_ERROR_DRIVER_API;
    }

    return LITE_SUCCESS;
}

LiteError liteGraphNodeDestroy(LiteGraphNode node) {
    if (!node) {
        return LITE_ERROR_INVALID_VALUE;
    }

    LiteGraph_s* parent_graph = node->parent_graph;
    if (parent_graph != nullptr) {
        auto& vec = parent_graph->tracked_nodes;
        for (auto it = vec.begin(); it != vec.end(); ) {
            if (*it == node) {
                it = vec.erase(it);
            } else {
                ++it;
            }
        }
    }

    node->invalid = true;
    delete node;
    return LITE_SUCCESS;
}
