#pragma once
#include <cstddef>
#include <cstdint>

// 错误码定义
enum LiteError {
    LITE_SUCCESS = 0,
    LITE_ERROR_INVALID_VALUE = 1,
    LITE_ERROR_OUT_OF_MEMORY = 2,
    LITE_ERROR_DRIVER_API = 3,
    LITE_ERROR_STREAM_SYNC = 4,
    LITE_ERROR_NOT_INITIALIZED = 5
};

// 不透明句柄 (Opaque Handles)
typedef struct LiteStream_s* LiteStream;
typedef struct LiteEvent_s*  LiteEvent;

// ================= Context API =================
LiteError liteInit();
LiteError liteShutdown();

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

// ================= Memory Pool API =================
LiteError liteMemPoolInit(size_t page_size, size_t max_pages);
LiteError liteMemPoolDestroy();
LiteError liteMemAllocPage(void** ptr, LiteStream stream);
LiteError liteMemFreePage(void* ptr, LiteStream stream);

// ================= Launcher API =================
LiteError liteLaunchDummyKernel(void* ptr, size_t size_bytes, LiteStream stream);

// ================= Profiler API =================
LiteError liteProfilerReset();
void liteProfilerReport();

// ================= Graph API =================
typedef struct LiteGraph_s* LiteGraph;
typedef struct LiteGraphExec_s* LiteGraphExec;

LiteError liteStreamBeginCapture(LiteStream stream);
LiteError liteStreamEndCapture(LiteStream stream, LiteGraph* graph);
LiteError liteGraphInstantiate(LiteGraph graph, LiteGraphExec* exec);
LiteError liteGraphLaunch(LiteGraphExec exec, LiteStream stream);
LiteError liteGraphDestroy(LiteGraph graph);
LiteError liteGraphExecDestroy(LiteGraphExec exec);
