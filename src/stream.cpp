#include "lite_gpu.h"
#include "lite_gpu_internal.h"
#include <cstdio>

LiteError liteStreamCreate(LiteStream* stream) {
    if (!stream) {
        return LITE_ERROR_INVALID_VALUE;
    }

    LiteStream_s* s = new LiteStream_s();
    CUresult err = cuStreamCreate(&s->handle, CU_STREAM_NON_BLOCKING);
    if (err != CUDA_SUCCESS) {
        delete s;
        fprintf(stderr, "cuStreamCreate cu_ret = %d\n", err);
        return LITE_ERROR_DRIVER_API;
    }

    *stream = s;
    return LITE_SUCCESS;
}

LiteError liteStreamDestroy(LiteStream stream) {
    if (!stream) {
        return LITE_ERROR_INVALID_VALUE;
    }

    cuStreamDestroy(stream->handle);
    delete stream;
    return LITE_SUCCESS;
}

LiteError liteStreamSynchronize(LiteStream stream) {
    if (!stream) {
        return LITE_ERROR_INVALID_VALUE;
    }

    CUresult err = cuStreamSynchronize(stream->handle);
    return (err == CUDA_SUCCESS) ? LITE_SUCCESS : LITE_ERROR_STREAM_SYNC;
}

LiteError liteEventCreate(LiteEvent* event) {
    if (!event) {
        return LITE_ERROR_INVALID_VALUE;
    }

    LiteEvent_s* e = new LiteEvent_s();
    CUresult err = cuEventCreate(&e->handle, CU_EVENT_DISABLE_TIMING | CU_EVENT_DEFAULT);
    if (err != CUDA_SUCCESS) {
        delete e;
        return LITE_ERROR_DRIVER_API;
    }

    *event = e;
    return LITE_SUCCESS;
}

LiteError liteEventDestroy(LiteEvent event) {
    if (!event) {
        return LITE_ERROR_INVALID_VALUE;
    }

    cuEventDestroy(event->handle);
    delete event;
    return LITE_SUCCESS;
}

LiteError liteEventRecord(LiteEvent event, LiteStream stream) {
    if (!event || !stream) {
        return LITE_ERROR_INVALID_VALUE;
    }

    CUresult err = cuEventRecord(event->handle, stream->handle);
    return (err == CUDA_SUCCESS) ? LITE_SUCCESS : LITE_ERROR_DRIVER_API;
}

LiteError liteStreamWaitEvent(LiteStream stream, LiteEvent event) {
    if (!stream || !event) {
        return LITE_ERROR_INVALID_VALUE;
    }

    CUresult err = cuStreamWaitEvent(stream->handle, event->handle, 0);
    return (err == CUDA_SUCCESS) ? LITE_SUCCESS : LITE_ERROR_STREAM_SYNC;
}

LiteError liteEventCreateWithTiming(LiteEvent* event) {
    if (!event) {
        return LITE_ERROR_INVALID_VALUE;
    }

    LiteEvent_s* e = new LiteEvent_s();
    CUresult err = cuEventCreate(&e->handle, CU_EVENT_DEFAULT);
    if (err != CUDA_SUCCESS) {
        delete e;
        return LITE_ERROR_DRIVER_API;
    }

    *event = e;
    return LITE_SUCCESS;
}

LiteError liteEventElapsedTime(float* ms, LiteEvent start, LiteEvent end) {
    if (!ms || !start || !end) {
        return LITE_ERROR_INVALID_VALUE;
    }

    CUresult err = cuEventElapsedTime(ms, start->handle, end->handle);
    return (err == CUDA_SUCCESS) ? LITE_SUCCESS : LITE_ERROR_DRIVER_API;
}

LiteError liteEventCreateForSync(LiteEvent* event) {
    if (!event) {
        return LITE_ERROR_INVALID_VALUE;
    }

    LiteEvent_s* e = new LiteEvent_s();
    CUresult err = cuEventCreate(&e->handle, CU_EVENT_DEFAULT);
    if (err != CUDA_SUCCESS) {
        delete e;
        return LITE_ERROR_DRIVER_API;
    }

    *event = e;
    return LITE_SUCCESS;
}
