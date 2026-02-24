/*
 * cuvk_launch.c - Kernel launch (compute dispatch via Vulkan)
 *
 * Implements the CUDA driver API function: cuLaunchKernel.
 */

#include "cuvk_internal.h"

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
    VkResult vr = vkCreateShaderModule(ctx->device, &sm_ci, NULL, &patched_shader);
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
    vr = vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE,
                                            1, &pipe_ci, NULL, &pipeline);
    vkDestroyShaderModule(ctx->device, patched_shader, NULL);
    if (vr != VK_SUCCESS)
        return cuvk_vk_to_cu(vr);

    /* Add to cache */
    if (f->pipeline_cache_count >= f->pipeline_cache_capacity) {
        uint32_t new_cap = f->pipeline_cache_capacity == 0 ? 4 :
                           f->pipeline_cache_capacity * 2;
        CuvkPipelineEntry *new_cache = (CuvkPipelineEntry *)realloc(
            f->pipeline_cache, new_cap * sizeof(CuvkPipelineEntry));
        if (!new_cache) {
            vkDestroyPipeline(ctx->device, pipeline, NULL);
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
    (void)hStream;
    (void)extra;

    if (!f)
        return CUDA_ERROR_INVALID_VALUE;

    struct CUctx_st *ctx = g_cuvk.current_ctx;
    if (!ctx)
        return CUDA_ERROR_INVALID_CONTEXT;

    /* 1. Get or create the compute pipeline */
    VkPipeline pipeline = VK_NULL_HANDLE;
    CUresult res = cuvk_get_or_create_pipeline(ctx, f,
                                                blockDimX, blockDimY, blockDimZ,
                                                &pipeline);
    if (res != CUDA_SUCCESS)
        return res;

    if (f->use_bda) {
        /* ===== BDA mode: push constants ===== */

        /* Pack kernel parameters into push constant data buffer.
         * Layout mirrors the push constant struct built by the PTX parser:
         * each parameter is at its natural alignment. */
        uint8_t pc_data[256]; /* max push constant size on most GPUs is 128-256 */
        memset(pc_data, 0, sizeof(pc_data));
        uint32_t pc_offset = 0;

        for (uint32_t i = 0; i < f->param_count && kernelParams; i++) {
            uint32_t sz = f->params[i].size;
            uint32_t align = sz;
            /* Align offset up */
            pc_offset = (pc_offset + align - 1) & ~(align - 1);

            if (f->params[i].is_pointer) {
                /* Pointer param: write the BDA (device address) as u64 */
                CUdeviceptr dptr = *(CUdeviceptr *)kernelParams[i];
                CuvkAlloc *alloc = cuvk_alloc_lookup(ctx, dptr);
                if (!alloc)
                    return CUDA_ERROR_INVALID_VALUE;
                uint64_t bda = (uint64_t)alloc->device_addr +
                               ((uint64_t)dptr - (uint64_t)alloc->device_addr);
                memcpy(pc_data + pc_offset, &bda, 8);
            } else {
                /* Scalar param: copy raw bytes */
                memcpy(pc_data + pc_offset, kernelParams[i], sz);
            }
            pc_offset += sz;
        }

        /* Append hidden block dimensions (ntid.x/y/z) */
        {
            uint32_t align = 4;
            pc_offset = (pc_offset + align - 1) & ~(align - 1);
            memcpy(pc_data + pc_offset, &blockDimX, 4); pc_offset += 4;
            memcpy(pc_data + pc_offset, &blockDimY, 4); pc_offset += 4;
            memcpy(pc_data + pc_offset, &blockDimZ, 4); pc_offset += 4;
        }

        /* Record and dispatch */
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        res = cuvk_oneshot_begin(ctx, &cmd);
        if (res != CUDA_SUCCESS)
            return res;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

        if (f->push_constant_size > 0) {
            vkCmdPushConstants(cmd, f->pipeline_layout,
                               VK_SHADER_STAGE_COMPUTE_BIT,
                               0, f->push_constant_size, pc_data);
        }

        vkCmdDispatch(cmd, gridDimX, gridDimY, gridDimZ);

        res = cuvk_oneshot_end(ctx, cmd);
        return res;

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

            VkResult vr = vkAllocateDescriptorSets(ctx->device, &ds_ai, &desc_set);
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
                vkFreeDescriptorSets(ctx->device, ctx->desc_pool, 1, &desc_set);
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
                    vkFreeDescriptorSets(ctx->device, ctx->desc_pool, 1, &desc_set);
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

            vkUpdateDescriptorSets(ctx->device, f->param_count, writes, 0, NULL);
            free(writes);
            free(buf_infos);
        }

        /* 4. Record and dispatch */
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        res = cuvk_oneshot_begin(ctx, &cmd);
        if (res != CUDA_SUCCESS) {
            if (desc_set)
                vkFreeDescriptorSets(ctx->device, ctx->desc_pool, 1, &desc_set);
            return res;
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

        if (desc_set) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    f->pipeline_layout, 0, 1, &desc_set,
                                    0, NULL);
        }

        vkCmdDispatch(cmd, gridDimX, gridDimY, gridDimZ);

        res = cuvk_oneshot_end(ctx, cmd);

        /* Free the descriptor set after execution */
        if (desc_set)
            vkFreeDescriptorSets(ctx->device, ctx->desc_pool, 1, &desc_set);

        return res;
    }
}
