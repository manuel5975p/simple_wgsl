/*
 * cuvk_stream.c - Stream, event, and one-shot command buffer management
 *
 * Implements CUDA driver API functions: cuStreamCreate, cuStreamDestroy_v2,
 * cuStreamSynchronize, cuStreamQuery, cuStreamCreateWithPriority,
 * cuEventCreate, cuEventDestroy_v2, cuEventRecord, cuEventSynchronize,
 * cuEventElapsedTime_v2.
 *
 * Also provides internal helpers: cuvk_oneshot_begin, cuvk_oneshot_end,
 * cuvk_stream_submit_and_wait.
 */

#include "cuvk_internal.h"

#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Internal helper: one-shot command buffer begin
 * ============================================================================ */

CUresult cuvk_oneshot_begin(struct CUctx_st *ctx, VkCommandBuffer *out_cb)
{
    if (!ctx || !out_cb)
        return CUDA_ERROR_INVALID_VALUE;

    VkCommandBufferAllocateInfo alloc_info = {0};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = ctx->cmd_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkResult vr = vkAllocateCommandBuffers(ctx->device, &alloc_info, &cb);
    if (vr != VK_SUCCESS)
        return cuvk_vk_to_cu(vr);

    VkCommandBufferBeginInfo begin_info = {0};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vr = vkBeginCommandBuffer(cb, &begin_info);
    if (vr != VK_SUCCESS) {
        vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cb);
        return cuvk_vk_to_cu(vr);
    }

    *out_cb = cb;
    return CUDA_SUCCESS;
}

/* ============================================================================
 * Internal helper: one-shot command buffer end + submit + wait
 * ============================================================================ */

CUresult cuvk_oneshot_end(struct CUctx_st *ctx, VkCommandBuffer cb)
{
    if (!ctx)
        return CUDA_ERROR_INVALID_VALUE;

    VkResult vr = vkEndCommandBuffer(cb);
    if (vr != VK_SUCCESS) {
        vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cb);
        return cuvk_vk_to_cu(vr);
    }

    /* Create an unsignaled fence */
    VkFenceCreateInfo fence_ci = {0};
    fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    VkFence fence = VK_NULL_HANDLE;
    vr = vkCreateFence(ctx->device, &fence_ci, NULL, &fence);
    if (vr != VK_SUCCESS) {
        vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cb);
        return cuvk_vk_to_cu(vr);
    }

    /* Submit */
    VkSubmitInfo submit_info = {0};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cb;

    vr = vkQueueSubmit(ctx->compute_queue, 1, &submit_info, fence);
    if (vr != VK_SUCCESS) {
        vkDestroyFence(ctx->device, fence, NULL);
        vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cb);
        return cuvk_vk_to_cu(vr);
    }

    /* Wait for completion */
    vr = vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
    CUresult result = cuvk_vk_to_cu(vr);

    /* Cleanup */
    vkDestroyFence(ctx->device, fence, NULL);
    vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cb);

    return result;
}

/* ============================================================================
 * Internal helper: submit and wait on a stream's recorded commands
 * ============================================================================ */

CUresult cuvk_stream_submit_and_wait(struct CUstream_st *stream)
{
    if (!stream)
        return CUDA_ERROR_INVALID_VALUE;

    /* Nothing to do if not recording */
    if (!stream->recording)
        return CUDA_SUCCESS;

    struct CUctx_st *ctx = stream->ctx;

    VkResult vr = vkEndCommandBuffer(stream->cmd_buf);
    if (vr != VK_SUCCESS)
        return cuvk_vk_to_cu(vr);

    /* Reset fence before submission */
    vr = vkResetFences(ctx->device, 1, &stream->fence);
    if (vr != VK_SUCCESS)
        return cuvk_vk_to_cu(vr);

    /* Submit */
    VkSubmitInfo submit_info = {0};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &stream->cmd_buf;

    vr = vkQueueSubmit(ctx->compute_queue, 1, &submit_info, stream->fence);
    if (vr != VK_SUCCESS)
        return cuvk_vk_to_cu(vr);

    /* Wait for completion */
    vr = vkWaitForFences(ctx->device, 1, &stream->fence, VK_TRUE, UINT64_MAX);
    if (vr != VK_SUCCESS)
        return cuvk_vk_to_cu(vr);

    /* Reset command buffer for reuse */
    vr = vkResetCommandBuffer(stream->cmd_buf,
                              VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
    if (vr != VK_SUCCESS)
        return cuvk_vk_to_cu(vr);

    stream->recording = false;
    return CUDA_SUCCESS;
}

/* ============================================================================
 * cuStreamCreate
 * ============================================================================ */

CUresult CUDAAPI cuStreamCreate(CUstream *phStream, unsigned int Flags)
{
    (void)Flags;

    if (!phStream)
        return CUDA_ERROR_INVALID_VALUE;

    struct CUctx_st *ctx = g_cuvk.current_ctx;
    if (!ctx)
        return CUDA_ERROR_INVALID_CONTEXT;

    struct CUstream_st *stream = (struct CUstream_st *)calloc(1, sizeof(*stream));
    if (!stream)
        return CUDA_ERROR_OUT_OF_MEMORY;

    stream->ctx = ctx;

    /* Allocate command buffer */
    VkCommandBufferAllocateInfo alloc_info = {0};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = ctx->cmd_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VkResult vr = vkAllocateCommandBuffers(ctx->device, &alloc_info,
                                           &stream->cmd_buf);
    if (vr != VK_SUCCESS) {
        free(stream);
        return cuvk_vk_to_cu(vr);
    }

    /* Create fence (signaled initially so first wait doesn't hang) */
    VkFenceCreateInfo fence_ci = {0};
    fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_ci.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    vr = vkCreateFence(ctx->device, &fence_ci, NULL, &stream->fence);
    if (vr != VK_SUCCESS) {
        vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &stream->cmd_buf);
        free(stream);
        return cuvk_vk_to_cu(vr);
    }

    stream->recording = false;

    *phStream = stream;
    return CUDA_SUCCESS;
}

/* ============================================================================
 * cuStreamCreateWithPriority
 * ============================================================================ */

CUresult CUDAAPI cuStreamCreateWithPriority(CUstream *phStream,
                                            unsigned int flags, int priority)
{
    (void)priority; /* Vulkan compute queues don't expose priority this way */
    return cuStreamCreate(phStream, flags);
}

/* ============================================================================
 * cuStreamDestroy_v2
 * ============================================================================ */

CUresult CUDAAPI cuStreamDestroy_v2(CUstream hStream)
{
    if (!hStream)
        return CUDA_ERROR_INVALID_VALUE;

    struct CUctx_st *ctx = hStream->ctx;
    if (!ctx)
        return CUDA_ERROR_INVALID_CONTEXT;

    /* Wait for any pending work */
    vkWaitForFences(ctx->device, 1, &hStream->fence, VK_TRUE, UINT64_MAX);

    /* Free command buffer */
    if (hStream->cmd_buf)
        vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &hStream->cmd_buf);

    /* Destroy fence */
    if (hStream->fence)
        vkDestroyFence(ctx->device, hStream->fence, NULL);

    free(hStream);
    return CUDA_SUCCESS;
}

/* ============================================================================
 * cuStreamSynchronize
 * ============================================================================ */

CUresult CUDAAPI cuStreamSynchronize(CUstream hStream)
{
    if (!hStream) {
        /* NULL stream = wait for everything on the device */
        struct CUctx_st *ctx = g_cuvk.current_ctx;
        if (!ctx)
            return CUDA_ERROR_INVALID_CONTEXT;
        VkResult vr = vkDeviceWaitIdle(ctx->device);
        return cuvk_vk_to_cu(vr);
    }

    /* If stream has pending recording, submit and wait */
    if (hStream->recording)
        return cuvk_stream_submit_and_wait(hStream);

    return CUDA_SUCCESS;
}

/* ============================================================================
 * cuStreamQuery
 * ============================================================================ */

CUresult CUDAAPI cuStreamQuery(CUstream hStream)
{
    if (!hStream)
        return CUDA_SUCCESS;

    /* If not recording, the stream is idle */
    if (!hStream->recording)
        return CUDA_SUCCESS;

    /* Check if the fence is signaled */
    VkResult vr = vkGetFenceStatus(hStream->ctx->device, hStream->fence);
    if (vr == VK_SUCCESS)
        return CUDA_SUCCESS;
    if (vr == VK_NOT_READY)
        return CUDA_ERROR_NOT_READY;

    return cuvk_vk_to_cu(vr);
}

/* ============================================================================
 * cuEventCreate
 * ============================================================================ */

CUresult CUDAAPI cuEventCreate(CUevent *phEvent, unsigned int Flags)
{
    (void)Flags;

    if (!phEvent)
        return CUDA_ERROR_INVALID_VALUE;

    struct CUctx_st *ctx = g_cuvk.current_ctx;
    if (!ctx)
        return CUDA_ERROR_INVALID_CONTEXT;

    struct CUevent_st *event = (struct CUevent_st *)calloc(1, sizeof(*event));
    if (!event)
        return CUDA_ERROR_OUT_OF_MEMORY;

    event->ctx = ctx;
    event->recorded = false;

    /* Create a query pool with 1 timestamp query */
    VkQueryPoolCreateInfo pool_ci = {0};
    pool_ci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    pool_ci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    pool_ci.queryCount = 1;

    VkResult vr = vkCreateQueryPool(ctx->device, &pool_ci, NULL,
                                    &event->query_pool);
    if (vr != VK_SUCCESS) {
        free(event);
        return cuvk_vk_to_cu(vr);
    }

    *phEvent = event;
    return CUDA_SUCCESS;
}

/* ============================================================================
 * cuEventDestroy_v2
 * ============================================================================ */

CUresult CUDAAPI cuEventDestroy_v2(CUevent hEvent)
{
    if (!hEvent)
        return CUDA_ERROR_INVALID_VALUE;

    struct CUctx_st *ctx = hEvent->ctx;
    if (!ctx)
        return CUDA_ERROR_INVALID_CONTEXT;

    if (hEvent->query_pool)
        vkDestroyQueryPool(ctx->device, hEvent->query_pool, NULL);

    free(hEvent);
    return CUDA_SUCCESS;
}

/* ============================================================================
 * cuEventRecord
 * ============================================================================ */

CUresult CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream)
{
    if (!hEvent)
        return CUDA_ERROR_INVALID_VALUE;

    /* Get context: from stream if provided, else from global current */
    struct CUctx_st *ctx = hStream ? hStream->ctx : g_cuvk.current_ctx;
    if (!ctx)
        return CUDA_ERROR_INVALID_CONTEXT;

    /* Use a one-shot command buffer to record the timestamp */
    VkCommandBuffer cb = VK_NULL_HANDLE;
    CUresult res = cuvk_oneshot_begin(ctx, &cb);
    if (res != CUDA_SUCCESS)
        return res;

    /* Reset query pool and write timestamp */
    vkCmdResetQueryPool(cb, hEvent->query_pool, 0, 1);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                        hEvent->query_pool, 0);

    res = cuvk_oneshot_end(ctx, cb);
    if (res != CUDA_SUCCESS)
        return res;

    hEvent->recorded = true;
    return CUDA_SUCCESS;
}

/* ============================================================================
 * cuEventSynchronize
 * ============================================================================ */

CUresult CUDAAPI cuEventSynchronize(CUevent hEvent)
{
    if (!hEvent)
        return CUDA_ERROR_INVALID_VALUE;

    /*
     * In our implementation, cuvk_oneshot_end already waits synchronously,
     * so if the event was recorded, it has already completed.
     */
    if (!hEvent->recorded)
        return CUDA_ERROR_INVALID_VALUE;

    return CUDA_SUCCESS;
}

/* ============================================================================
 * cuEventElapsedTime_v2
 * ============================================================================ */

CUresult CUDAAPI cuEventElapsedTime_v2(float *pMilliseconds,
                                       CUevent hStart, CUevent hEnd)
{
    if (!pMilliseconds || !hStart || !hEnd)
        return CUDA_ERROR_INVALID_VALUE;

    if (!hStart->recorded || !hEnd->recorded)
        return CUDA_ERROR_INVALID_HANDLE;

    struct CUctx_st *ctx = hStart->ctx;
    if (!ctx)
        return CUDA_ERROR_INVALID_CONTEXT;

    /* Retrieve timestamp values from both query pools */
    uint64_t start_ts = 0;
    VkResult vr = vkGetQueryPoolResults(
        ctx->device, hStart->query_pool,
        0, 1,
        sizeof(start_ts), &start_ts,
        sizeof(start_ts),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (vr != VK_SUCCESS)
        return cuvk_vk_to_cu(vr);

    uint64_t end_ts = 0;
    vr = vkGetQueryPoolResults(
        ctx->device, hEnd->query_pool,
        0, 1,
        sizeof(end_ts), &end_ts,
        sizeof(end_ts),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (vr != VK_SUCCESS)
        return cuvk_vk_to_cu(vr);

    /* timestampPeriod is in nanoseconds per tick */
    float period = ctx->dev_props.limits.timestampPeriod;
    double delta_ns = (double)(end_ts - start_ts) * (double)period;
    *pMilliseconds = (float)(delta_ns / 1000000.0);

    return CUDA_SUCCESS;
}
