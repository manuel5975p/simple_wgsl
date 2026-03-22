/*
 * cuvk_launch.c - Kernel launch (compute dispatch via Vulkan)
 *
 * Implements the CUDA driver API function: cuLaunchKernel.
 */

#include "cuvk_internal.h"

#include <stdio.h>

#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Helper: get or create a compute pipeline for the given block size
 * ============================================================================ */

static CUresult cuvk_get_or_create_pipeline(struct CUctx_st *ctx,
                                             struct CUfunc_st *f,
                                             uint32_t block_x,
                                             uint32_t block_y,
                                             uint32_t block_z,
                                             VkPipeline *out_pipeline)
{
    /* Search the cache */
    for (uint32_t i = 0; i < f->pipeline_cache_count; i++) {
        CuvkPipelineEntry *e = &f->pipeline_cache[i];
        if (e->block_x == block_x &&
            e->block_y == block_y &&
            e->block_z == block_z) {
            *out_pipeline = e->pipeline;
            return CUDA_SUCCESS;
        }
    }

    /* Cache miss: create a new pipeline.
     *
     * Patch the SPIR-V LocalSize execution mode to match the requested block
     * dimensions, then create a new VkShaderModule and pipeline.
     * OpExecutionMode is: [wordcount<<16|opcode=16, func_id, mode=17, x, y, z]
     */
    uint32_t *spirv = f->module->spirv_words;
    size_t count = f->module->spirv_count;

    /* Make a copy to patch */
    uint32_t *patched = (uint32_t *)malloc(count * sizeof(uint32_t));
    if (!patched)
        return CUDA_ERROR_OUT_OF_MEMORY;
    memcpy(patched, spirv, count * sizeof(uint32_t));

    /* Scan for OpExecutionMode LocalSize (opcode 16, mode 17) */
    for (size_t i = 5; i < count; ) { /* skip SPIR-V header (5 words) */
        uint32_t word = patched[i];
        uint16_t opcode = word & 0xFFFF;
        uint16_t wc = word >> 16;
        if (wc == 0) break; /* safety */
        if (opcode == 16 /* OpExecutionMode */ && wc == 6 && i + 5 < count &&
            patched[i + 2] == 17 /* LocalSize */) {
            patched[i + 3] = block_x;
            patched[i + 4] = block_y;
            patched[i + 5] = block_z;
        }
        i += wc;
    }

    /* Create a temporary shader module with patched SPIR-V */
    VkShaderModuleCreateInfo sm_ci = {0};
    sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm_ci.codeSize = count * sizeof(uint32_t);
    sm_ci.pCode = patched;

    VkShaderModule patched_shader = VK_NULL_HANDLE;
    VkResult vr = g_cuvk.vk.vkCreateShaderModule(ctx->device, &sm_ci, NULL, &patched_shader);
    free(patched);
    if (vr != VK_SUCCESS)
        return cuvk_vk_to_cu(vr);

    VkComputePipelineCreateInfo pipe_ci = {0};
    pipe_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipe_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipe_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipe_ci.stage.module = patched_shader;
    pipe_ci.stage.pName = f->name;
    pipe_ci.layout = f->pipeline_layout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    vr = g_cuvk.vk.vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE,
                                            1, &pipe_ci, NULL, &pipeline);
    g_cuvk.vk.vkDestroyShaderModule(ctx->device, patched_shader, NULL);
    if (vr != VK_SUCCESS)
        return cuvk_vk_to_cu(vr);

    /* Add to cache */
    if (f->pipeline_cache_count >= f->pipeline_cache_capacity) {
        uint32_t new_cap = f->pipeline_cache_capacity == 0 ? 4 :
                           f->pipeline_cache_capacity * 2;
        CuvkPipelineEntry *new_cache = (CuvkPipelineEntry *)realloc(
            f->pipeline_cache, new_cap * sizeof(CuvkPipelineEntry));
        if (!new_cache) {
            g_cuvk.vk.vkDestroyPipeline(ctx->device, pipeline, NULL);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        f->pipeline_cache = new_cache;
        f->pipeline_cache_capacity = new_cap;
    }

    CuvkPipelineEntry *entry = &f->pipeline_cache[f->pipeline_cache_count++];
    entry->block_x = block_x;
    entry->block_y = block_y;
    entry->block_z = block_z;
    entry->pipeline = pipeline;

    *out_pipeline = pipeline;
    return CUDA_SUCCESS;
}

/* ============================================================================
 * cuLaunchKernel
 * ============================================================================ */

CUresult CUDAAPI cuLaunchKernel(CUfunction f,
                                 unsigned int gridDimX,
                                 unsigned int gridDimY,
                                 unsigned int gridDimZ,
                                 unsigned int blockDimX,
                                 unsigned int blockDimY,
                                 unsigned int blockDimZ,
                                 unsigned int sharedMemBytes,
                                 CUstream hStream,
                                 void **kernelParams,
                                 void **extra)
{
    (void)sharedMemBytes;
    CUVK_LOG("[cuvk] cuLaunchKernel: f=%p grid=(%u,%u,%u) block=(%u,%u,%u)\n",
            (void *)f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ);

    if (!f)
        return CUDA_ERROR_INVALID_VALUE;

    struct CUctx_st *ctx = g_cuvk.current_ctx;
    if (!ctx)
        return CUDA_ERROR_INVALID_CONTEXT;

    /* Flush any deferred cuFFT work before recording new dispatches,
     * so that FFT results are visible to subsequent kernel reads. */
    cuvk_fft_flush(ctx);

    /* Resolve NULL/sentinel stream to default stream */
    struct CUstream_st *stream = cuvk_resolve_stream(hStream);

    /* 1. Get or create the compute pipeline */
    VkPipeline pipeline = VK_NULL_HANDLE;
    CUresult res = cuvk_get_or_create_pipeline(ctx, f,
                                                blockDimX, blockDimY, blockDimZ,
                                                &pipeline);
    if (res != CUDA_SUCCESS)
        return res;

    /* Ensure the stream's command buffer is recording */
    res = cuvk_stream_ensure_recording(stream);
    if (res != CUDA_SUCCESS)
        return res;

    VkCommandBuffer cmd = stream->cmd_buf;

    if (f->use_bda) {
        /* ===== BDA mode: push constants ===== */

        /* Pack kernel parameters into push constant data buffer.
         * Layout mirrors the push constant struct built by the PTX parser:
         * each parameter is at its natural alignment. */
        uint8_t pc_data[256]; /* max push constant size on most GPUs is 128-256 */
        memset(pc_data, 0, sizeof(pc_data));
        uint32_t pc_offset = 0;

        /* Parse 'extra' parameter if kernelParams is NULL.
         * extra format: { CU_LAUNCH_PARAM_BUFFER_POINTER, ptr,
         *                 CU_LAUNCH_PARAM_BUFFER_SIZE, &sz,
         *                 CU_LAUNCH_PARAM_END } */
        void *extra_buf = NULL;
        size_t extra_buf_sz = 0;
        if (!kernelParams && extra) {
            for (int ei = 0; extra[ei]; ei++) {
                if (extra[ei] == (void *)(uintptr_t)1) { /* CU_LAUNCH_PARAM_BUFFER_POINTER */
                    extra_buf = extra[ei + 1];
                    ei++;
                } else if (extra[ei] == (void *)(uintptr_t)2) { /* CU_LAUNCH_PARAM_BUFFER_SIZE */
                    extra_buf_sz = *(size_t *)extra[ei + 1];
                    ei++;
                }
            }
        }

        if (extra_buf && f->param_count > 0) {
            /* Bulk copy from extra buffer (byte array / struct params) */
            uint32_t copy_sz = f->params[0].size;
            if (copy_sz > extra_buf_sz && extra_buf_sz > 0)
                copy_sz = (uint32_t)extra_buf_sz;
            if (copy_sz > sizeof(pc_data))
                copy_sz = sizeof(pc_data);
            memcpy(pc_data, extra_buf, copy_sz);
            pc_offset = copy_sz;
        }

        for (uint32_t i = 0; i < f->param_count && kernelParams; i++) {
            uint32_t sz = f->params[i].size;
            uint32_t align = sz <= 8 ? sz : 4; /* cap alignment for large params */
            if (align == 0) align = 4;
            /* Align offset up */
            pc_offset = (pc_offset + align - 1) & ~(align - 1);

            if (f->params[i].is_pointer) {
                /* Pointer param: write the BDA (device address) as u64 */
                if (pc_offset + 8 > sizeof(pc_data))
                    return CUDA_ERROR_INVALID_VALUE;
                CUdeviceptr dptr = *(CUdeviceptr *)kernelParams[i];
                CuvkAlloc *alloc = cuvk_alloc_lookup(ctx, dptr);
                if (!alloc)
                    return CUDA_ERROR_INVALID_VALUE;
                uint64_t bda = (uint64_t)alloc->device_addr +
                               ((uint64_t)dptr - (uint64_t)alloc->device_addr);
                memcpy(pc_data + pc_offset, &bda, 8);
            } else {
                /* Scalar param: copy raw bytes */
                if (pc_offset + sz > sizeof(pc_data))
                    return CUDA_ERROR_INVALID_VALUE;
                memcpy(pc_data + pc_offset, kernelParams[i], sz);
            }
            pc_offset += sz;
        }

        /* Append hidden block dimensions (ntid.x/y/z) */
        {
            uint32_t align = 4;
            pc_offset = (pc_offset + align - 1) & ~(align - 1);
            if (pc_offset + 12 > sizeof(pc_data))
                return CUDA_ERROR_INVALID_VALUE;
            memcpy(pc_data + pc_offset, &blockDimX, 4); pc_offset += 4;
            memcpy(pc_data + pc_offset, &blockDimY, 4); pc_offset += 4;
            memcpy(pc_data + pc_offset, &blockDimZ, 4); pc_offset += 4;
        }

        /* Memory barrier: ensure previous dispatches' writes are visible */
        {
            VkMemoryBarrier bar = {0};
            bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            g_cuvk.vk.vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &bar, 0, NULL, 0, NULL);
        }

        g_cuvk.vk.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

        if (f->module->global_count > 0) {
            g_cuvk.vk.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    f->pipeline_layout, 0, 1,
                                    &f->module->globals_desc_set, 0, NULL);
        }

        if (f->push_constant_size > 0) {
            g_cuvk.vk.vkCmdPushConstants(cmd, f->pipeline_layout,
                               VK_SHADER_STAGE_COMPUTE_BIT,
                               0, f->push_constant_size, pc_data);
        }

        g_cuvk.vk.vkCmdDispatch(cmd, gridDimX, gridDimY, gridDimZ);
        return CUDA_SUCCESS;

    } else {
        /* ===== Descriptor mode (original) ===== */

        /* 2. Allocate a descriptor set */
        VkDescriptorSet desc_set = VK_NULL_HANDLE;

        if (f->param_count > 0) {
            VkDescriptorSetAllocateInfo ds_ai = {0};
            ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ds_ai.descriptorPool = ctx->desc_pool;
            ds_ai.descriptorSetCount = 1;
            ds_ai.pSetLayouts = &f->desc_layout;

            VkResult vr = g_cuvk.vk.vkAllocateDescriptorSets(ctx->device, &ds_ai, &desc_set);
            if (vr != VK_SUCCESS)
                return cuvk_vk_to_cu(vr);

            /* 3. Write descriptor bindings for each pointer parameter */
            VkWriteDescriptorSet *writes = (VkWriteDescriptorSet *)calloc(
                f->param_count, sizeof(VkWriteDescriptorSet));
            VkDescriptorBufferInfo *buf_infos = (VkDescriptorBufferInfo *)calloc(
                f->param_count, sizeof(VkDescriptorBufferInfo));

            if (!writes || !buf_infos) {
                free(writes);
                free(buf_infos);
                g_cuvk.vk.vkFreeDescriptorSets(ctx->device, ctx->desc_pool, 1, &desc_set);
                return CUDA_ERROR_OUT_OF_MEMORY;
            }

            for (uint32_t i = 0; i < f->param_count; i++) {
                if (!f->params[i].is_pointer)
                    continue;

                /* kernelParams[i] points to the actual parameter value (a CUdeviceptr) */
                CUdeviceptr dptr = *(CUdeviceptr *)kernelParams[i];
                CuvkAlloc *alloc = cuvk_alloc_lookup(ctx, dptr);
                if (!alloc) {
                    free(writes);
                    free(buf_infos);
                    g_cuvk.vk.vkFreeDescriptorSets(ctx->device, ctx->desc_pool, 1, &desc_set);
                    return CUDA_ERROR_INVALID_VALUE;
                }

                /* Compute offset within the buffer (for sub-buffer pointers) */
                VkDeviceSize offset = (VkDeviceSize)((uint64_t)dptr -
                                                      (uint64_t)alloc->device_addr);

                buf_infos[i].buffer = alloc->buffer;
                buf_infos[i].offset = offset;
                buf_infos[i].range = alloc->size - offset;

                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstSet = desc_set;
                writes[i].dstBinding = i;
                writes[i].dstArrayElement = 0;
                writes[i].descriptorCount = 1;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[i].pBufferInfo = &buf_infos[i];
            }

            g_cuvk.vk.vkUpdateDescriptorSets(ctx->device, f->param_count, writes, 0, NULL);
            free(writes);
            free(buf_infos);
        }

        /* Memory barrier: ensure previous dispatches' writes are visible */
        {
            VkMemoryBarrier bar = {0};
            bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            g_cuvk.vk.vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &bar, 0, NULL, 0, NULL);
        }

        /* 4. Record dispatch into the stream */
        g_cuvk.vk.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

        if (desc_set) {
            g_cuvk.vk.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    f->pipeline_layout, 0, 1, &desc_set,
                                    0, NULL);
        }

        g_cuvk.vk.vkCmdDispatch(cmd, gridDimX, gridDimY, gridDimZ);

        return CUDA_SUCCESS;
    }
}
