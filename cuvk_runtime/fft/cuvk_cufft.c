/*
 * cuvk_cufft.c - cuFFT implementation for the CUDA-on-Vulkan runtime
 *
 * GPU-accelerated 1D FFT via Vulkan compute shaders.  Shaders are generated
 * by fft_stockham_gen (one per stage/direction), compiled from WGSL to SPIR-V,
 * and recorded into a single command buffer that ping-pongs between the user
 * buffer and a scratch buffer.
 */

#include "cuvk_internal.h"
#include "fft_stockham_gen.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * cuFFT types (must match cufft.h)
 * ============================================================================ */

typedef enum {
    CUFFT_SUCCESS            = 0,
    CUFFT_INVALID_PLAN       = 1,
    CUFFT_ALLOC_FAILED       = 2,
    CUFFT_INVALID_TYPE       = 3,
    CUFFT_INVALID_VALUE      = 4,
    CUFFT_INTERNAL_ERROR     = 5,
    CUFFT_EXEC_FAILED        = 6,
    CUFFT_SETUP_FAILED       = 7,
    CUFFT_INVALID_SIZE       = 8,
    CUFFT_UNALIGNED_DATA     = 9,
    CUFFT_INCOMPLETE_PARAMETER_LIST = 10,
    CUFFT_INVALID_DEVICE     = 11,
    CUFFT_PARSE_ERROR        = 12,
    CUFFT_NO_WORKSPACE       = 13,
    CUFFT_NOT_IMPLEMENTED    = 14,
    CUFFT_LICENSE_ERROR      = 15,
    CUFFT_NOT_SUPPORTED      = 16,
} cufftResult;

typedef enum {
    CUFFT_R2C = 0x2a,
    CUFFT_C2R = 0x2c,
    CUFFT_C2C = 0x29,
    CUFFT_D2Z = 0x6a,
    CUFFT_Z2D = 0x6c,
    CUFFT_Z2Z = 0x69,
} cufftType;

#define CUFFT_FORWARD (-1)
#define CUFFT_INVERSE  (1)

typedef int cufftHandle;
typedef float cufftReal;
typedef struct { float x, y; } cufftComplex;
typedef double cufftDoubleReal;
typedef struct { double x, y; } cufftDoubleComplex;

typedef enum {
    CUFFT_COMPATIBILITY_FFTW_PADDING = 0x01,
} cufftCompatibility;

/* ============================================================================
 * Internal plan structure — Stockham codelet-based, multi-axis
 * ============================================================================ */

#define MAX_CUFFT_PLANS    64
#define MAX_FFT_STAGES     20
#define MAX_RADIX          32
#define MAX_AXES           3
#define FFT_WORKGROUP_SIZE 256

/* Per-axis FFT plan (one dimension of a multi-dimensional transform) */
typedef struct {
    int n;                                              /* FFT size */
    int n_stages;
    int radices[MAX_FFT_STAGES];
    uint32_t dispatch_x[MAX_FFT_STAGES];
    int element_stride;                                 /* physical stride */
    int batch_stride;                                   /* stride between batches (gid.y) */
    int batch_count;                                    /* gid.y dispatch count */
    int batch_stride2;                                  /* outer batch stride (gid.z), 0=unused */
    int batch_count2;                                   /* gid.z dispatch count, 0 or 1=unused */

    VkShaderModule   shaders[2][MAX_FFT_STAGES];        /* [fwd/inv][stage] */
    VkPipeline       pipelines[2][MAX_FFT_STAGES];      /* [fwd/inv][stage] */
} FftAxis;

typedef struct {
    struct CUctx_st *ctx;
    int rank;                               /* 1, 2, or 3 */
    int dims[3];                            /* nx, ny, nz */
    int total_elements;                     /* product of all dims */
    int batch;
    cufftType type;
    int valid;

    int n_axes;
    FftAxis axes[MAX_AXES];

    /* R2C/C2R specific pipelines */
    VkShaderModule   r2c_post_shader;
    VkPipeline       r2c_post_pipeline;
    VkShaderModule   c2r_pre_shader;
    VkPipeline       c2r_pre_pipeline;
    uint32_t         r2c_dispatch_x;
    int              r2c_n;                 /* original N for R2C (axes store N/2) */

    /* Shared Vulkan resources */
    VkDescriptorSetLayout desc_layout;
    VkPipelineLayout      pipe_layout;

    VkBuffer       scratch_buf;
    VkDeviceMemory scratch_mem;

    VkDescriptorPool desc_pool;

    VkCommandBuffer cb_fwd;
    VkCommandBuffer cb_inv;
    VkFence         fence;

    VkBuffer bound_src;
    VkBuffer bound_dst;
    int      bound_inplace;
    int      cb_fwd_valid;
    int      cb_inv_valid;
} CufftPlan;

static CufftPlan g_cufft_plans[MAX_CUFFT_PLANS];
static int g_cufft_next_handle = 0;

/* ============================================================================
 * Factorize N into radices [2..MAX_RADIX], greedy largest-first
 * ============================================================================ */

static int factorize_fft(int n, int *radices, int max_stages)
{
    int count = 0;
    int rem = n;
    while (rem > 1 && count < max_stages) {
        int best = 0;
        for (int r = MAX_RADIX; r >= 2; r--) {
            if (rem % r == 0) { best = r; break; }
        }
        if (best == 0) return 0; /* cannot factorize */
        radices[count++] = best;
        rem /= best;
    }
    return (rem == 1) ? count : 0;
}

/* ============================================================================
 * Helper: compile WGSL source to SPIR-V
 * ============================================================================ */

static cufftResult compile_wgsl(const char *wgsl_source,
                                uint32_t **out_words,
                                size_t *out_count)
{
    WgslAstNode *ast = wgsl_parse(wgsl_source);
    if (!ast) {
        CUVK_LOG("[cufft] wgsl_parse failed\n");
        return CUFFT_INTERNAL_ERROR;
    }

    WgslResolver *resolver = wgsl_resolver_build(ast);
    if (!resolver) {
        CUVK_LOG("[cufft] wgsl_resolver_build failed\n");
        wgsl_free_ast(ast);
        return CUFFT_INTERNAL_ERROR;
    }

    WgslLowerOptions opts = {0};
    opts.spirv_version = 0x00010300;
    opts.env = WGSL_LOWER_ENV_VULKAN_1_1;
    opts.packing = WGSL_LOWER_PACK_STD430;
    opts.enable_debug_names = 0;

    WgslLowerResult lr = wgsl_lower_emit_spirv(ast, resolver, &opts,
                                                out_words, out_count);
    wgsl_resolver_free(resolver);
    wgsl_free_ast(ast);

    if (lr != WGSL_LOWER_OK) {
        CUVK_LOG("[cufft] wgsl_lower_emit_spirv failed: %d\n", lr);
        return CUFFT_INTERNAL_ERROR;
    }

    return CUFFT_SUCCESS;
}

/* ============================================================================
 * Helper: create a Vulkan buffer with device-local memory
 * ============================================================================ */

static cufftResult create_device_buffer(struct CUctx_st *ctx,
                                        VkDeviceSize size,
                                        VkBufferUsageFlags usage,
                                        VkBuffer *out_buf,
                                        VkDeviceMemory *out_mem)
{
    VkBufferCreateInfo buf_ci = {0};
    buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_ci.size = size;
    buf_ci.usage = usage;
    buf_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult vr = g_cuvk.vk.vkCreateBuffer(ctx->device, &buf_ci, NULL, out_buf);
    if (vr != VK_SUCCESS) return CUFFT_ALLOC_FAILED;

    VkMemoryRequirements mem_reqs;
    g_cuvk.vk.vkGetBufferMemoryRequirements(ctx->device, *out_buf, &mem_reqs);

    int32_t mem_type = cuvk_find_memory_type(
        &ctx->mem_props, mem_reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mem_type < 0) {
        g_cuvk.vk.vkDestroyBuffer(ctx->device, *out_buf, NULL);
        return CUFFT_ALLOC_FAILED;
    }

    VkMemoryAllocateInfo alloc_info = {0};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = (uint32_t)mem_type;

    vr = g_cuvk.vk.vkAllocateMemory(ctx->device, &alloc_info, NULL, out_mem);
    if (vr != VK_SUCCESS) {
        g_cuvk.vk.vkDestroyBuffer(ctx->device, *out_buf, NULL);
        return CUFFT_ALLOC_FAILED;
    }

    vr = g_cuvk.vk.vkBindBufferMemory(ctx->device, *out_buf, *out_mem, 0);
    if (vr != VK_SUCCESS) {
        g_cuvk.vk.vkFreeMemory(ctx->device, *out_mem, NULL);
        g_cuvk.vk.vkDestroyBuffer(ctx->device, *out_buf, NULL);
        return CUFFT_ALLOC_FAILED;
    }

    return CUFFT_SUCCESS;
}


/* ============================================================================
 * Stockham pipeline helpers
 * ============================================================================ */

/* Create the shared descriptor set layout and pipeline layout.
 * 2 bindings: binding 0 = storage buffer (src, read), binding 1 = storage
 * buffer (dst, read_write). */
static cufftResult create_stockham_layouts(struct CUctx_st *ctx,
                                           VkDescriptorSetLayout *out_desc_layout,
                                           VkPipelineLayout *out_pipe_layout)
{
    VkDescriptorSetLayoutBinding bindings[2] = {{0}};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dsl_ci = {0};
    dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_ci.bindingCount = 2;
    dsl_ci.pBindings = bindings;

    VkResult vr = g_cuvk.vk.vkCreateDescriptorSetLayout(ctx->device, &dsl_ci, NULL,
                                               out_desc_layout);
    if (vr != VK_SUCCESS) return CUFFT_INTERNAL_ERROR;

    VkPipelineLayoutCreateInfo pl_ci = {0};
    pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_ci.setLayoutCount = 1;
    pl_ci.pSetLayouts = out_desc_layout;

    vr = g_cuvk.vk.vkCreatePipelineLayout(ctx->device, &pl_ci, NULL, out_pipe_layout);
    if (vr != VK_SUCCESS) {
        g_cuvk.vk.vkDestroyDescriptorSetLayout(ctx->device, *out_desc_layout, NULL);
        return CUFFT_INTERNAL_ERROR;
    }

    return CUFFT_SUCCESS;
}

/* Create a VkShaderModule + VkPipeline from SPIR-V words using the shared
 * pipeline layout. */
static cufftResult create_stage_pipeline(struct CUctx_st *ctx,
                                         uint32_t *spirv_words,
                                         size_t spirv_count,
                                         VkPipelineLayout pipe_layout,
                                         VkShaderModule *out_shader,
                                         VkPipeline *out_pipeline)
{
    VkShaderModuleCreateInfo sm_ci = {0};
    sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm_ci.codeSize = spirv_count * sizeof(uint32_t);
    sm_ci.pCode = spirv_words;

    VkResult vr = g_cuvk.vk.vkCreateShaderModule(ctx->device, &sm_ci, NULL, out_shader);
    if (vr != VK_SUCCESS) return CUFFT_INTERNAL_ERROR;

    VkComputePipelineCreateInfo pipe_ci = {0};
    pipe_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipe_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipe_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipe_ci.stage.module = *out_shader;
    pipe_ci.stage.pName = "main";
    pipe_ci.layout = pipe_layout;

    vr = g_cuvk.vk.vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE,
                                  1, &pipe_ci, NULL, out_pipeline);
    if (vr != VK_SUCCESS) {
        g_cuvk.vk.vkDestroyShaderModule(ctx->device, *out_shader, NULL);
        return CUFFT_INTERNAL_ERROR;
    }

    return CUFFT_SUCCESS;
}

/* ============================================================================
 * Record all FFT stages (all axes) into a single command buffer
 * ============================================================================ */

static void emit_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier mem_bar = {0};
    mem_bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mem_bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mem_bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                            VK_ACCESS_SHADER_WRITE_BIT;
    g_cuvk.vk.vkCmdPipelineBarrier(cb,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mem_bar, 0, NULL, 0, NULL);
}

static cufftResult record_fft_cb_range(CufftPlan *p, int dir_idx,
                                       VkCommandBuffer cb,
                                       VkBuffer src_buf, VkBuffer dst_buf,
                                       int start_axis, int end_axis)
{
    struct CUctx_st *ctx = p->ctx;
    int inplace = (src_buf == dst_buf);

    VkDeviceSize buf_size = (VkDeviceSize)p->total_elements * 2 * sizeof(float)
                          * (VkDeviceSize)p->batch;

    /* Count total stages across selected axes for descriptor set allocation */
    int total_stages = 0;
    for (int a = start_axis; a < end_axis; a++)
        total_stages += p->axes[a].n_stages;

    /* Determine per-stage read/write buffer assignments across ALL axes.
     * All stages across all axes form one linear sequence for ping-pong.
     * The result must end up in dst_buf. */
    VkBuffer *read_bufs = (VkBuffer *)malloc(sizeof(VkBuffer) * (size_t)total_stages);
    VkBuffer *write_bufs = (VkBuffer *)malloc(sizeof(VkBuffer) * (size_t)total_stages);
    int need_final_copy = 0;

    if (total_stages == 1 && inplace) {
        read_bufs[0] = dst_buf;
        write_bufs[0] = p->scratch_buf;
        need_final_copy = 1;
    } else if (inplace) {
        read_bufs[0] = dst_buf;
        write_bufs[0] = p->scratch_buf;
        for (int i = 1; i < total_stages; i++) {
            read_bufs[i] = write_bufs[i - 1];
            write_bufs[i] = (read_bufs[i] == p->scratch_buf)
                          ? dst_buf : p->scratch_buf;
        }
        if (write_bufs[total_stages - 1] != dst_buf)
            need_final_copy = 1;
    } else {
        write_bufs[total_stages - 1] = dst_buf;
        for (int i = total_stages - 2; i >= 0; i--) {
            write_bufs[i] = (write_bufs[i + 1] == dst_buf)
                          ? p->scratch_buf : dst_buf;
        }
        read_bufs[0] = src_buf;
        for (int i = 1; i < total_stages; i++)
            read_bufs[i] = write_bufs[i - 1];
    }

    /* Allocate descriptor sets for all stages */
    VkDescriptorSetLayout *layouts = (VkDescriptorSetLayout *)
        malloc(sizeof(VkDescriptorSetLayout) * (size_t)total_stages);
    for (int i = 0; i < total_stages; i++)
        layouts[i] = p->desc_layout;

    VkDescriptorSet *desc_sets = (VkDescriptorSet *)
        malloc(sizeof(VkDescriptorSet) * (size_t)total_stages);
    VkDescriptorSetAllocateInfo ds_ai = {0};
    ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ds_ai.descriptorPool = p->desc_pool;
    ds_ai.descriptorSetCount = (uint32_t)total_stages;
    ds_ai.pSetLayouts = layouts;

    VkResult vr = g_cuvk.vk.vkAllocateDescriptorSets(ctx->device, &ds_ai, desc_sets);
    free(layouts);
    if (vr != VK_SUCCESS) {
        free(read_bufs); free(write_bufs); free(desc_sets);
        return CUFFT_INTERNAL_ERROR;
    }

    /* Update descriptor sets with per-stage buffer bindings */
    for (int i = 0; i < total_stages; i++) {
        VkDescriptorBufferInfo buf_infos[2] = {{0}};
        buf_infos[0].buffer = read_bufs[i];
        buf_infos[0].offset = 0;
        buf_infos[0].range = buf_size;
        buf_infos[1].buffer = write_bufs[i];
        buf_infos[1].offset = 0;
        buf_infos[1].range = buf_size;

        VkWriteDescriptorSet writes[2] = {{0}};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = desc_sets[i];
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &buf_infos[0];

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = desc_sets[i];
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &buf_infos[1];

        g_cuvk.vk.vkUpdateDescriptorSets(ctx->device, 2, writes, 0, NULL);
    }

    /* Begin command buffer */
    VkCommandBufferBeginInfo begin_info = {0};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vr = g_cuvk.vk.vkBeginCommandBuffer(cb, &begin_info);
    if (vr != VK_SUCCESS) {
        free(read_bufs); free(write_bufs); free(desc_sets);
        return CUFFT_INTERNAL_ERROR;
    }

    /* Dispatch all stages across selected axes */
    int global_stage = 0;
    for (int a = start_axis; a < end_axis; a++) {
        FftAxis *axis = &p->axes[a];
        for (int s = 0; s < axis->n_stages; s++) {
            if (global_stage > 0)
                emit_barrier(cb);

            g_cuvk.vk.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                              axis->pipelines[dir_idx][s]);
            g_cuvk.vk.vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    p->pipe_layout, 0, 1,
                                    &desc_sets[global_stage], 0, NULL);
            g_cuvk.vk.vkCmdDispatch(cb, axis->dispatch_x[s],
                          (uint32_t)axis->batch_count,
                          axis->batch_count2 > 0 ?
                              (uint32_t)axis->batch_count2 : 1);
            global_stage++;
        }
    }

    /* If result landed in scratch, copy back to dst */
    if (need_final_copy) {
        VkMemoryBarrier copy_bar = {0};
        copy_bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        copy_bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        copy_bar.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        g_cuvk.vk.vkCmdPipelineBarrier(cb,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &copy_bar, 0, NULL, 0, NULL);

        VkBufferCopy region = {0};
        region.size = buf_size;
        g_cuvk.vk.vkCmdCopyBuffer(cb, p->scratch_buf, dst_buf, 1, &region);
    }

    vr = g_cuvk.vk.vkEndCommandBuffer(cb);
    free(read_bufs); free(write_bufs); free(desc_sets);
    if (vr != VK_SUCCESS) return CUFFT_INTERNAL_ERROR;

    return CUFFT_SUCCESS;
}

static cufftResult record_fft_cb(CufftPlan *p, int dir_idx,
                                 VkCommandBuffer cb,
                                 VkBuffer src_buf, VkBuffer dst_buf)
{
    return record_fft_cb_range(p, dir_idx, cb, src_buf, dst_buf, 0, p->n_axes);
}

/* ============================================================================
 * Helper: build one FftAxis (factorize + compile stages)
 * ============================================================================ */

static cufftResult build_axis2(struct CUctx_st *ctx, FftAxis *axis,
                               int n, int element_stride, int batch_stride,
                               int batch_count, int batch_stride2,
                               int batch_count2, VkPipelineLayout pipe_layout)
{
    memset(axis, 0, sizeof(*axis));
    axis->n = n;
    axis->element_stride = element_stride;
    axis->batch_stride = batch_stride;
    axis->batch_count = batch_count;
    axis->batch_stride2 = batch_stride2;
    axis->batch_count2 = batch_count2;

    if (n == 1) {
        /* 1-point FFT is identity: 0 stages */
        axis->n_stages = 0;
    } else {
        axis->n_stages = factorize_fft(n, axis->radices, MAX_FFT_STAGES);
        if (axis->n_stages == 0) {
            CUVK_LOG("[cufft] cannot factorize n=%d into radices 2..%d\n",
                     n, MAX_RADIX);
            return CUFFT_INVALID_SIZE;
        }
    }

    int gen_directions[2] = {1, -1};  /* fwd, inv */
    int stride = 1;
    for (int s = 0; s < axis->n_stages; s++) {
        int radix = axis->radices[s];
        uint32_t n_bf = (uint32_t)(n / radix);
        axis->dispatch_x[s] = (n_bf + FFT_WORKGROUP_SIZE - 1) /
                               FFT_WORKGROUP_SIZE;

        for (int d = 0; d < 2; d++) {
            char *wgsl = gen_fft_stockham_strided2(
                radix, stride, n, gen_directions[d], FFT_WORKGROUP_SIZE,
                element_stride, batch_stride, batch_stride2);
            if (!wgsl) return CUFFT_INTERNAL_ERROR;

            uint32_t *spirv = NULL;
            size_t spirv_count = 0;
            cufftResult cr = compile_wgsl(wgsl, &spirv, &spirv_count);
            free(wgsl);
            if (cr != CUFFT_SUCCESS) return cr;

            cr = create_stage_pipeline(ctx, spirv, spirv_count, pipe_layout,
                                       &axis->shaders[d][s],
                                       &axis->pipelines[d][s]);
            wgsl_lower_free(spirv);
            if (cr != CUFFT_SUCCESS) return cr;
        }

        stride *= radix;
    }

    return CUFFT_SUCCESS;
}

static cufftResult build_axis(struct CUctx_st *ctx, FftAxis *axis,
                              int n, int element_stride, int batch_stride,
                              int batch_count, VkPipelineLayout pipe_layout)
{
    return build_axis2(ctx, axis, n, element_stride, batch_stride,
                       batch_count, 0, 0, pipe_layout);
}

/* ============================================================================
 * Helper: allocate plan resources (scratch, desc pool, CBs, fence)
 * ============================================================================ */

static cufftResult alloc_plan_resources(CufftPlan *p)
{
    struct CUctx_st *ctx = p->ctx;

    /* Total stages across all axes + R2C/C2R extra stages */
    int total_stages = 0;
    for (int a = 0; a < p->n_axes; a++)
        total_stages += p->axes[a].n_stages;
    int extra = (p->r2c_post_pipeline ? 1 : 0) + (p->c2r_pre_pipeline ? 1 : 0);
    total_stages += extra;

    /* Scratch buffer */
    VkDeviceSize scratch_size = (VkDeviceSize)p->total_elements * 2 *
                                sizeof(float) * (VkDeviceSize)p->batch;
    cufftResult cr = create_device_buffer(ctx, scratch_size,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              &p->scratch_buf, &p->scratch_mem);
    if (cr != CUFFT_SUCCESS) return cr;

    /* Descriptor pool */
    VkDescriptorPoolSize pool_size = {0};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = (uint32_t)(total_stages * 2 * 2);

    VkDescriptorPoolCreateInfo dp_ci = {0};
    dp_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp_ci.maxSets = (uint32_t)(total_stages * 2);
    dp_ci.poolSizeCount = 1;
    dp_ci.pPoolSizes = &pool_size;

    VkResult vr = g_cuvk.vk.vkCreateDescriptorPool(ctx->device, &dp_ci, NULL,
                                          &p->desc_pool);
    if (vr != VK_SUCCESS) return CUFFT_ALLOC_FAILED;

    /* Command buffers */
    VkCommandBufferAllocateInfo cb_ai = {0};
    cb_ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cb_ai.commandPool = ctx->cmd_pool;
    cb_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cb_ai.commandBufferCount = 1;

    vr = g_cuvk.vk.vkAllocateCommandBuffers(ctx->device, &cb_ai, &p->cb_fwd);
    if (vr != VK_SUCCESS) return CUFFT_ALLOC_FAILED;
    vr = g_cuvk.vk.vkAllocateCommandBuffers(ctx->device, &cb_ai, &p->cb_inv);
    if (vr != VK_SUCCESS) return CUFFT_ALLOC_FAILED;

    /* Fence */
    VkFenceCreateInfo fence_ci = {0};
    fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vr = g_cuvk.vk.vkCreateFence(ctx->device, &fence_ci, NULL, &p->fence);
    if (vr != VK_SUCCESS) return CUFFT_ALLOC_FAILED;

    p->cb_fwd_valid = 0;
    p->cb_inv_valid = 0;
    p->bound_src = VK_NULL_HANDLE;
    p->bound_dst = VK_NULL_HANDLE;
    p->bound_inplace = 0;

    return CUFFT_SUCCESS;
}

/* ============================================================================
 * Helper: init plan common fields (ctx, type, etc.)
 * ============================================================================ */

static cufftResult plan_init(CufftPlan **out_plan, int *out_handle,
                             cufftType type, int batch)
{
    CUresult r = cuInit(0);
    if (r != CUDA_SUCCESS) return CUFFT_SETUP_FAILED;

    if (!g_cuvk.current_ctx) {
        CUcontext ctx;
        r = cuDevicePrimaryCtxRetain(&ctx, 0);
        if (r != CUDA_SUCCESS) return CUFFT_SETUP_FAILED;
    }

    struct CUctx_st *ctx = g_cuvk.current_ctx;
    if (!ctx) return CUFFT_SETUP_FAILED;

    int handle = -1;
    if (g_cufft_next_handle < MAX_CUFFT_PLANS) {
        handle = g_cufft_next_handle++;
    } else {
        for (int i = 0; i < MAX_CUFFT_PLANS; i++) {
            if (!g_cufft_plans[i].valid) { handle = i; break; }
        }
    }
    if (handle < 0) return CUFFT_ALLOC_FAILED;
    CufftPlan *p = &g_cufft_plans[handle];
    memset(p, 0, sizeof(*p));
    p->ctx = ctx;
    p->type = type;
    p->batch = (batch < 1) ? 1 : batch;

    cufftResult cr = create_stockham_layouts(ctx, &p->desc_layout,
                                             &p->pipe_layout);
    if (cr != CUFFT_SUCCESS) return cr;

    *out_plan = p;
    *out_handle = handle;
    return CUFFT_SUCCESS;
}

/* ============================================================================
 * cufftPlan1d
 * ============================================================================ */

cufftResult cufftPlan1d(cufftHandle *plan, int nx, cufftType type, int batch)
{
    CUVK_LOG("[cufft] cufftPlan1d: nx=%d type=0x%x batch=%d\n", nx, type, batch);

    if (!plan || nx <= 0)
        return CUFFT_INVALID_VALUE;

    if (type != CUFFT_C2C && type != CUFFT_R2C && type != CUFFT_C2R)
        return CUFFT_INVALID_TYPE;

    CufftPlan *p;
    int handle;
    cufftResult cr = plan_init(&p, &handle, type, batch);
    if (cr != CUFFT_SUCCESS) return cr;

    struct CUctx_st *ctx = p->ctx;
    p->rank = 1;
    p->dims[0] = nx;

    if (type == CUFFT_C2C) {
        p->total_elements = nx;
        p->n_axes = 1;
        cr = build_axis(ctx, &p->axes[0], nx, 1, nx,
                         p->batch, p->pipe_layout);
        if (cr != CUFFT_SUCCESS) return cr;
    } else if (type == CUFFT_R2C || type == CUFFT_C2R) {
        /* R2C: N/2-point C2C forward + post-processing
         * C2R: pre-processing + N/2-point C2C inverse */
        if (nx < 2 || nx % 2 != 0) return CUFFT_INVALID_SIZE;
        int half = nx / 2;
        p->r2c_n = nx;
        p->total_elements = half + 1;  /* output size for R2C */
        p->n_axes = 1;
        cr = build_axis(ctx, &p->axes[0], half, 1, half,
                         p->batch, p->pipe_layout);
        if (cr != CUFFT_SUCCESS) return cr;

        /* R2C post-processing pipeline */
        {
            char *wgsl = gen_fft_r2c_postprocess(nx, FFT_WORKGROUP_SIZE);
            if (!wgsl) return CUFFT_INTERNAL_ERROR;
            uint32_t *spirv = NULL;
            size_t spirv_count = 0;
            cr = compile_wgsl(wgsl, &spirv, &spirv_count);
            free(wgsl);
            if (cr != CUFFT_SUCCESS) return cr;
            cr = create_stage_pipeline(ctx, spirv, spirv_count,
                                       p->pipe_layout,
                                       &p->r2c_post_shader,
                                       &p->r2c_post_pipeline);
            wgsl_lower_free(spirv);
            if (cr != CUFFT_SUCCESS) return cr;
            p->r2c_dispatch_x = ((uint32_t)(half + 1) + FFT_WORKGROUP_SIZE - 1) /
                                 FFT_WORKGROUP_SIZE;
        }

        /* C2R pre-processing pipeline */
        {
            char *wgsl = gen_fft_c2r_preprocess(nx, FFT_WORKGROUP_SIZE);
            if (!wgsl) return CUFFT_INTERNAL_ERROR;
            uint32_t *spirv = NULL;
            size_t spirv_count = 0;
            cr = compile_wgsl(wgsl, &spirv, &spirv_count);
            free(wgsl);
            if (cr != CUFFT_SUCCESS) return cr;
            cr = create_stage_pipeline(ctx, spirv, spirv_count,
                                       p->pipe_layout,
                                       &p->c2r_pre_shader,
                                       &p->c2r_pre_pipeline);
            wgsl_lower_free(spirv);
            if (cr != CUFFT_SUCCESS) return cr;
        }
    }

    CUVK_LOG("[cufft] axis 0: n=%d, %d stages [", p->axes[0].n,
             p->axes[0].n_stages);
    for (int i = 0; i < p->axes[0].n_stages; i++)
        CUVK_LOG(" %d", p->axes[0].radices[i]);
    CUVK_LOG(" ]\n");

    cr = alloc_plan_resources(p);
    if (cr != CUFFT_SUCCESS) return cr;

    p->valid = 1;
    *plan = handle;
    CUVK_LOG("[cufft] cufftPlan1d: SUCCESS handle=%d (%d stages)\n",
             handle, p->axes[0].n_stages);
    return CUFFT_SUCCESS;
}

/* ============================================================================
 * cufftExecC2C
 * ============================================================================ */

cufftResult cufftExecC2C(cufftHandle plan_handle,
                          cufftComplex *idata,
                          cufftComplex *odata,
                          int direction)
{
    CUVK_LOG("[cufft] cufftExecC2C: plan=%d dir=%d\n", plan_handle, direction);

    if (plan_handle < 0 || plan_handle >= MAX_CUFFT_PLANS)
        return CUFFT_INVALID_PLAN;
    CufftPlan *p = &g_cufft_plans[plan_handle];
    if (!p->valid) return CUFFT_INVALID_PLAN;

    struct CUctx_st *ctx = g_cuvk.current_ctx;
    if (!ctx) ctx = p->ctx;

    CUdeviceptr iptr = (CUdeviceptr)idata;
    CUdeviceptr optr = (CUdeviceptr)odata;

    CuvkAlloc *alloc_in = cuvk_alloc_lookup(ctx, iptr);
    if (!alloc_in) return CUFFT_INVALID_VALUE;

    CuvkAlloc *alloc_out;
    int inplace = (iptr == optr);
    if (inplace) {
        alloc_out = alloc_in;
    } else {
        alloc_out = cuvk_alloc_lookup(ctx, optr);
        if (!alloc_out) return CUFFT_INVALID_VALUE;
    }

    VkBuffer src_buf = alloc_in->buffer;
    VkBuffer dst_buf = alloc_out->buffer;

    /* Map direction: CUFFT_FORWARD(-1) -> dir_idx=0, CUFFT_INVERSE(1) -> dir_idx=1 */
    int dir_idx = (direction == CUFFT_FORWARD) ? 0 : 1;

    /* Check if we can reuse the cached command buffer */
    int cache_hit = 0;
    if (dir_idx == 0 && p->cb_fwd_valid &&
        p->bound_src == src_buf && p->bound_dst == dst_buf &&
        p->bound_inplace == inplace) {
        cache_hit = 1;
    }
    if (dir_idx == 1 && p->cb_inv_valid &&
        p->bound_src == src_buf && p->bound_dst == dst_buf &&
        p->bound_inplace == inplace) {
        cache_hit = 1;
    }

    VkCommandBuffer cb = (dir_idx == 0) ? p->cb_fwd : p->cb_inv;

    if (!cache_hit) {
        /* Invalidate both CBs, re-record */
        g_cuvk.vk.vkDeviceWaitIdle(ctx->device);
        g_cuvk.vk.vkResetDescriptorPool(ctx->device, p->desc_pool, 0);
        g_cuvk.vk.vkResetCommandBuffer(p->cb_fwd, 0);
        g_cuvk.vk.vkResetCommandBuffer(p->cb_inv, 0);
        p->cb_fwd_valid = 0;
        p->cb_inv_valid = 0;

        /* For in-place: first stage reads from dst_buf (same as src_buf). */
        VkBuffer effective_src = inplace ? dst_buf : src_buf;

        /* Record forward */
        cufftResult cr = record_fft_cb(p, 0, p->cb_fwd, effective_src, dst_buf);
        if (cr != CUFFT_SUCCESS) return cr;
        p->cb_fwd_valid = 1;

        /* Need to re-allocate descriptor sets from the pool for inverse.
         * But we already used some from the pool for forward.
         * Reset the pool and re-record both. */
        /* Actually, we need a bigger pool or separate recording.
         * Let's reset and only record what we need, then record the other
         * direction on demand. */

        /* Better approach: reset pool, record requested direction only.
         * Record the other direction later if needed. */
        g_cuvk.vk.vkResetDescriptorPool(ctx->device, p->desc_pool, 0);
        g_cuvk.vk.vkResetCommandBuffer(p->cb_fwd, 0);
        p->cb_fwd_valid = 0;

        cr = record_fft_cb(p, dir_idx, cb, effective_src, dst_buf);
        if (cr != CUFFT_SUCCESS) return cr;

        if (dir_idx == 0)
            p->cb_fwd_valid = 1;
        else
            p->cb_inv_valid = 1;

        p->bound_src = src_buf;
        p->bound_dst = dst_buf;
        p->bound_inplace = inplace;
    }

    /* Submit */
    g_cuvk.vk.vkResetFences(ctx->device, 1, &p->fence);

    VkSubmitInfo submit = {0};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cb;

    VkResult vr = g_cuvk.vk.vkQueueSubmit(ctx->compute_queue, 1, &submit, p->fence);
    if (vr != VK_SUCCESS) return CUFFT_EXEC_FAILED;

    vr = g_cuvk.vk.vkWaitForFences(ctx->device, 1, &p->fence, VK_TRUE, UINT64_MAX);
    if (vr != VK_SUCCESS) return CUFFT_EXEC_FAILED;

    return CUFFT_SUCCESS;
}

/* ============================================================================
 * R2C/C2R exec helper: record CB with C2C stages + post/pre-process stage
 * ============================================================================ */

static cufftResult record_r2c_cb(CufftPlan *p, VkCommandBuffer cb,
                                  VkBuffer src_buf, VkBuffer dst_buf)
{
    /* R2C pipeline: N/2-point forward C2C on src, then post-process to dst.
     * Input: N reals = N/2 complex pairs (reinterpret src buffer).
     * src_buf: N reals (N/2 complex), dst_buf: N/2+1 complex output.
     *
     * Stage flow:
     *   src → [C2C stages, ping-pong with scratch] → intermediate
     *   intermediate → [R2C post-process] → dst
     *
     * C2C result lands in scratch or src (depending on stage count).
     * Post-process reads from that and writes to dst. */

    struct CUctx_st *ctx = p->ctx;
    int half = p->r2c_n / 2;
    FftAxis *axis = &p->axes[0];
    int ns = axis->n_stages;

    /* Total batch count for R2C (axis 0) */
    int total_batches = axis->batch_count *
                        (axis->batch_count2 > 0 ? axis->batch_count2 : 1);
    VkDeviceSize c2c_buf_size = (VkDeviceSize)half * 2 * sizeof(float) *
                                (VkDeviceSize)total_batches;
    VkDeviceSize out_buf_size = (VkDeviceSize)(half + 1) * 2 * sizeof(float) *
                                (VkDeviceSize)total_batches;

    /* Ping-pong for C2C stages: forward from src through scratch.
     * Post-process reads from wherever the last C2C stage wrote. */
    VkBuffer read_bufs[MAX_FFT_STAGES + 1];
    VkBuffer write_bufs[MAX_FFT_STAGES + 1];

    if (ns == 0) {
        /* No C2C stages (half=1): post-process reads src directly */
        read_bufs[0] = src_buf;
        write_bufs[0] = dst_buf;
    } else {
        read_bufs[0] = src_buf;
        write_bufs[0] = p->scratch_buf;
        for (int i = 1; i < ns; i++) {
            read_bufs[i] = write_bufs[i - 1];
            write_bufs[i] = (read_bufs[i] == p->scratch_buf)
                           ? src_buf : p->scratch_buf;
        }
        /* Post-process: read from last C2C output, write to dst */
        read_bufs[ns] = write_bufs[ns - 1];
        write_bufs[ns] = dst_buf;
    }

    int total = ns + 1;

    /* Allocate descriptor sets */
    VkDescriptorSetLayout *layouts = (VkDescriptorSetLayout *)
        malloc(sizeof(VkDescriptorSetLayout) * (size_t)total);
    for (int i = 0; i < total; i++) layouts[i] = p->desc_layout;

    VkDescriptorSet *desc_sets = (VkDescriptorSet *)
        malloc(sizeof(VkDescriptorSet) * (size_t)total);
    VkDescriptorSetAllocateInfo ds_ai = {0};
    ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ds_ai.descriptorPool = p->desc_pool;
    ds_ai.descriptorSetCount = (uint32_t)total;
    ds_ai.pSetLayouts = layouts;

    VkResult vr = g_cuvk.vk.vkAllocateDescriptorSets(ctx->device, &ds_ai, desc_sets);
    free(layouts);
    if (vr != VK_SUCCESS) { free(desc_sets); return CUFFT_INTERNAL_ERROR; }

    /* Update descriptor sets */
    for (int i = 0; i < total; i++) {
        VkDeviceSize rd_size = c2c_buf_size;
        VkDeviceSize wr_size = (i < ns) ? c2c_buf_size : out_buf_size;
        VkDescriptorBufferInfo bi[2] = {{0}};
        bi[0].buffer = read_bufs[i]; bi[0].range = rd_size;
        bi[1].buffer = write_bufs[i]; bi[1].range = wr_size;
        VkWriteDescriptorSet ws[2] = {{0}};
        for (int b = 0; b < 2; b++) {
            ws[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            ws[b].dstSet = desc_sets[i]; ws[b].dstBinding = (uint32_t)b;
            ws[b].descriptorCount = 1;
            ws[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            ws[b].pBufferInfo = &bi[b];
        }
        g_cuvk.vk.vkUpdateDescriptorSets(ctx->device, 2, ws, 0, NULL);
    }

    /* Begin CB */
    VkCommandBufferBeginInfo begin = {0};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vr = g_cuvk.vk.vkBeginCommandBuffer(cb, &begin);
    if (vr != VK_SUCCESS) { free(desc_sets); return CUFFT_INTERNAL_ERROR; }

    /* C2C stages */
    for (int s = 0; s < ns; s++) {
        if (s > 0) emit_barrier(cb);
        g_cuvk.vk.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                          axis->pipelines[0][s]);
        g_cuvk.vk.vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                p->pipe_layout, 0, 1, &desc_sets[s], 0, NULL);
        g_cuvk.vk.vkCmdDispatch(cb, axis->dispatch_x[s],
                      (uint32_t)axis->batch_count,
                      axis->batch_count2 > 0 ?
                          (uint32_t)axis->batch_count2 : 1);
    }

    /* Post-process stage */
    emit_barrier(cb);
    g_cuvk.vk.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                      p->r2c_post_pipeline);
    g_cuvk.vk.vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                            p->pipe_layout, 0, 1, &desc_sets[ns], 0, NULL);
    {
        FftAxis *a0 = &p->axes[0];
        uint32_t total_batches = (uint32_t)a0->batch_count *
                                 (uint32_t)(a0->batch_count2 > 0 ?
                                            a0->batch_count2 : 1);
        g_cuvk.vk.vkCmdDispatch(cb, p->r2c_dispatch_x, total_batches, 1);
    }

    vr = g_cuvk.vk.vkEndCommandBuffer(cb);
    free(desc_sets);
    return (vr == VK_SUCCESS) ? CUFFT_SUCCESS : CUFFT_INTERNAL_ERROR;
}

static cufftResult record_c2r_cb(CufftPlan *p, VkCommandBuffer cb,
                                  VkBuffer src_buf, VkBuffer dst_buf)
{
    /* C2R pipeline: pre-process src → scratch, then N/2-point inverse C2C.
     * Input: N/2+1 complex bins, Output: N reals (= N/2 complex pairs). */

    struct CUctx_st *ctx = p->ctx;
    int half = p->r2c_n / 2;
    FftAxis *axis = &p->axes[0];
    int ns = axis->n_stages;

    int total_batches = axis->batch_count *
                        (axis->batch_count2 > 0 ? axis->batch_count2 : 1);
    VkDeviceSize in_buf_size = (VkDeviceSize)(half + 1) * 2 * sizeof(float) *
                               (VkDeviceSize)total_batches;
    VkDeviceSize c2c_buf_size = (VkDeviceSize)half * 2 * sizeof(float) *
                                (VkDeviceSize)total_batches;

    /* Stage 0: pre-process (src → scratch or dst)
     * Stages 1..ns: C2C inverse, ping-pong scratch/dst */
    int total = 1 + ns;
    VkBuffer read_bufs[MAX_FFT_STAGES + 1];
    VkBuffer write_bufs[MAX_FFT_STAGES + 1];
    int need_final_copy = 0;

    if (ns == 0) {
        /* No C2C stages (half=1): pre-process writes directly to dst */
        read_bufs[0] = src_buf;
        write_bufs[0] = dst_buf;
    } else {
        /* Pre-process: read src, write scratch */
        read_bufs[0] = src_buf;
        write_bufs[0] = p->scratch_buf;

        /* C2C stages: ping-pong between scratch and dst_buf */
        read_bufs[1] = p->scratch_buf;
        write_bufs[1] = dst_buf;
        for (int i = 2; i < total; i++) {
            read_bufs[i] = write_bufs[i - 1];
            write_bufs[i] = (read_bufs[i] == dst_buf)
                           ? p->scratch_buf : dst_buf;
        }
        need_final_copy = (write_bufs[total - 1] != dst_buf);
    }

    /* Allocate descriptor sets */
    VkDescriptorSetLayout *layouts = (VkDescriptorSetLayout *)
        malloc(sizeof(VkDescriptorSetLayout) * (size_t)total);
    for (int i = 0; i < total; i++) layouts[i] = p->desc_layout;

    VkDescriptorSet *desc_sets = (VkDescriptorSet *)
        malloc(sizeof(VkDescriptorSet) * (size_t)total);
    VkDescriptorSetAllocateInfo ds_ai = {0};
    ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ds_ai.descriptorPool = p->desc_pool;
    ds_ai.descriptorSetCount = (uint32_t)total;
    ds_ai.pSetLayouts = layouts;

    VkResult vr = g_cuvk.vk.vkAllocateDescriptorSets(ctx->device, &ds_ai, desc_sets);
    free(layouts);
    if (vr != VK_SUCCESS) { free(desc_sets); return CUFFT_INTERNAL_ERROR; }

    /* Update descriptor sets */
    for (int i = 0; i < total; i++) {
        VkDeviceSize rd_size = (i == 0) ? in_buf_size : c2c_buf_size;
        VkDeviceSize wr_size = c2c_buf_size;
        VkDescriptorBufferInfo bi[2] = {{0}};
        bi[0].buffer = read_bufs[i]; bi[0].range = rd_size;
        bi[1].buffer = write_bufs[i]; bi[1].range = wr_size;
        VkWriteDescriptorSet ws[2] = {{0}};
        for (int b = 0; b < 2; b++) {
            ws[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            ws[b].dstSet = desc_sets[i]; ws[b].dstBinding = (uint32_t)b;
            ws[b].descriptorCount = 1;
            ws[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            ws[b].pBufferInfo = &bi[b];
        }
        g_cuvk.vk.vkUpdateDescriptorSets(ctx->device, 2, ws, 0, NULL);
    }

    /* Begin CB */
    VkCommandBufferBeginInfo begin = {0};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vr = g_cuvk.vk.vkBeginCommandBuffer(cb, &begin);
    if (vr != VK_SUCCESS) { free(desc_sets); return CUFFT_INTERNAL_ERROR; }

    /* Pre-process stage */
    g_cuvk.vk.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                      p->c2r_pre_pipeline);
    g_cuvk.vk.vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                            p->pipe_layout, 0, 1, &desc_sets[0], 0, NULL);
    {
        FftAxis *a0 = &p->axes[0];
        uint32_t total_batches = (uint32_t)a0->batch_count *
                                 (uint32_t)(a0->batch_count2 > 0 ?
                                            a0->batch_count2 : 1);
        g_cuvk.vk.vkCmdDispatch(cb, ((uint32_t)half + FFT_WORKGROUP_SIZE - 1) /
                      FFT_WORKGROUP_SIZE, total_batches, 1);
    }

    /* C2C inverse stages */
    for (int s = 0; s < ns; s++) {
        emit_barrier(cb);
        g_cuvk.vk.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                          axis->pipelines[1][s]);  /* dir_idx=1 = inverse */
        g_cuvk.vk.vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                p->pipe_layout, 0, 1,
                                &desc_sets[1 + s], 0, NULL);
        g_cuvk.vk.vkCmdDispatch(cb, axis->dispatch_x[s],
                      (uint32_t)axis->batch_count,
                      axis->batch_count2 > 0 ?
                          (uint32_t)axis->batch_count2 : 1);
    }

    /* If last C2C stage landed in scratch, copy to dst */
    if (need_final_copy) {
        VkMemoryBarrier copy_bar = {0};
        copy_bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        copy_bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        copy_bar.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        g_cuvk.vk.vkCmdPipelineBarrier(cb,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &copy_bar, 0, NULL, 0, NULL);
        VkBufferCopy region = {0};
        region.size = c2c_buf_size;
        g_cuvk.vk.vkCmdCopyBuffer(cb, p->scratch_buf, dst_buf, 1, &region);
    }

    vr = g_cuvk.vk.vkEndCommandBuffer(cb);
    free(desc_sets);
    return (vr == VK_SUCCESS) ? CUFFT_SUCCESS : CUFFT_INTERNAL_ERROR;
}

/* ============================================================================
 * cufftExecR2C
 * ============================================================================ */

cufftResult cufftExecR2C(cufftHandle plan_handle,
                          cufftReal *idata,
                          cufftComplex *odata)
{
    CUVK_LOG("[cufft] cufftExecR2C: plan=%d\n", plan_handle);

    if (plan_handle < 0 || plan_handle >= MAX_CUFFT_PLANS)
        return CUFFT_INVALID_PLAN;
    CufftPlan *p = &g_cufft_plans[plan_handle];
    if (!p->valid || !p->r2c_post_pipeline) return CUFFT_INVALID_PLAN;

    struct CUctx_st *ctx = g_cuvk.current_ctx;
    if (!ctx) ctx = p->ctx;

    CUdeviceptr iptr = (CUdeviceptr)idata;
    CUdeviceptr optr = (CUdeviceptr)odata;
    CuvkAlloc *alloc_in = cuvk_alloc_lookup(ctx, iptr);
    if (!alloc_in) return CUFFT_INVALID_VALUE;
    CuvkAlloc *alloc_out = cuvk_alloc_lookup(ctx, optr);
    if (!alloc_out) return CUFFT_INVALID_VALUE;

    VkBuffer src_buf = alloc_in->buffer;
    VkBuffer dst_buf = alloc_out->buffer;

    /* Always re-record for R2C (different CB structure from C2C) */
    g_cuvk.vk.vkDeviceWaitIdle(ctx->device);
    g_cuvk.vk.vkResetDescriptorPool(ctx->device, p->desc_pool, 0);
    g_cuvk.vk.vkResetCommandBuffer(p->cb_fwd, 0);
    p->cb_fwd_valid = 0;

    /* Phase 1: R2C on axis 0 (C2C stages + post-process) → dst_buf */
    cufftResult cr = record_r2c_cb(p, p->cb_fwd, src_buf, dst_buf);
    if (cr != CUFFT_SUCCESS) return cr;

    g_cuvk.vk.vkResetFences(ctx->device, 1, &p->fence);
    VkSubmitInfo submit = {0};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &p->cb_fwd;

    VkResult vr = g_cuvk.vk.vkQueueSubmit(ctx->compute_queue, 1, &submit, p->fence);
    if (vr != VK_SUCCESS) return CUFFT_EXEC_FAILED;
    vr = g_cuvk.vk.vkWaitForFences(ctx->device, 1, &p->fence, VK_TRUE, UINT64_MAX);
    if (vr != VK_SUCCESS) return CUFFT_EXEC_FAILED;

    /* Phase 2: C2C forward on remaining axes (in-place on dst_buf) */
    if (p->n_axes > 1) {
        g_cuvk.vk.vkResetDescriptorPool(ctx->device, p->desc_pool, 0);
        g_cuvk.vk.vkResetCommandBuffer(p->cb_fwd, 0);

        cr = record_fft_cb_range(p, 0, p->cb_fwd, dst_buf, dst_buf,
                                 1, p->n_axes);
        if (cr != CUFFT_SUCCESS) return cr;

        g_cuvk.vk.vkResetFences(ctx->device, 1, &p->fence);
        submit.pCommandBuffers = &p->cb_fwd;
        vr = g_cuvk.vk.vkQueueSubmit(ctx->compute_queue, 1, &submit, p->fence);
        if (vr != VK_SUCCESS) return CUFFT_EXEC_FAILED;
        vr = g_cuvk.vk.vkWaitForFences(ctx->device, 1, &p->fence, VK_TRUE, UINT64_MAX);
        if (vr != VK_SUCCESS) return CUFFT_EXEC_FAILED;
    }

    return CUFFT_SUCCESS;
}

/* ============================================================================
 * cufftDestroy
 * ============================================================================ */

cufftResult cufftDestroy(cufftHandle plan_handle)
{
    CUVK_LOG("[cufft] cufftDestroy: plan=%d\n", plan_handle);

    if (plan_handle < 0 || plan_handle >= MAX_CUFFT_PLANS)
        return CUFFT_INVALID_PLAN;
    CufftPlan *p = &g_cufft_plans[plan_handle];
    if (!p->valid) return CUFFT_INVALID_PLAN;

    struct CUctx_st *ctx = p->ctx;
    g_cuvk.vk.vkDeviceWaitIdle(ctx->device);

    /* Destroy per-axis per-stage pipelines and shader modules */
    for (int a = 0; a < p->n_axes; a++) {
        FftAxis *axis = &p->axes[a];
        for (int d = 0; d < 2; d++) {
            for (int s = 0; s < axis->n_stages; s++) {
                if (axis->pipelines[d][s])
                    g_cuvk.vk.vkDestroyPipeline(ctx->device, axis->pipelines[d][s], NULL);
                if (axis->shaders[d][s])
                    g_cuvk.vk.vkDestroyShaderModule(ctx->device, axis->shaders[d][s], NULL);
            }
        }
    }

    /* Destroy R2C/C2R pipelines */
    if (p->r2c_post_pipeline)
        g_cuvk.vk.vkDestroyPipeline(ctx->device, p->r2c_post_pipeline, NULL);
    if (p->r2c_post_shader)
        g_cuvk.vk.vkDestroyShaderModule(ctx->device, p->r2c_post_shader, NULL);
    if (p->c2r_pre_pipeline)
        g_cuvk.vk.vkDestroyPipeline(ctx->device, p->c2r_pre_pipeline, NULL);
    if (p->c2r_pre_shader)
        g_cuvk.vk.vkDestroyShaderModule(ctx->device, p->c2r_pre_shader, NULL);

    /* Destroy shared layouts */
    if (p->pipe_layout)
        g_cuvk.vk.vkDestroyPipelineLayout(ctx->device, p->pipe_layout, NULL);
    if (p->desc_layout)
        g_cuvk.vk.vkDestroyDescriptorSetLayout(ctx->device, p->desc_layout, NULL);

    /* Destroy scratch buffer */
    if (p->scratch_buf)
        g_cuvk.vk.vkDestroyBuffer(ctx->device, p->scratch_buf, NULL);
    if (p->scratch_mem)
        g_cuvk.vk.vkFreeMemory(ctx->device, p->scratch_mem, NULL);

    /* Destroy fence */
    if (p->fence)
        g_cuvk.vk.vkDestroyFence(ctx->device, p->fence, NULL);

    /* Free command buffers */
    if (p->cb_fwd)
        g_cuvk.vk.vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &p->cb_fwd);
    if (p->cb_inv)
        g_cuvk.vk.vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &p->cb_inv);

    /* Destroy descriptor pool (frees all descriptor sets) */
    if (p->desc_pool)
        g_cuvk.vk.vkDestroyDescriptorPool(ctx->device, p->desc_pool, NULL);

    p->valid = 0;
    return CUFFT_SUCCESS;
}

/* ============================================================================
 * Stubs: no-op / return success
 * ============================================================================ */

cufftResult cufftSetStream(cufftHandle plan, CUstream stream) {
    (void)plan; (void)stream;
    return CUFFT_SUCCESS;
}

cufftResult cufftSetAutoAllocation(cufftHandle plan, int autoAllocate) {
    (void)plan; (void)autoAllocate;
    return CUFFT_SUCCESS;
}

cufftResult cufftSetWorkArea(cufftHandle plan, void *workArea) {
    (void)plan; (void)workArea;
    return CUFFT_SUCCESS;
}

cufftResult cufftGetVersion(int *version) {
    if (version) *version = 12100;
    return CUFFT_SUCCESS;
}

cufftResult cufftGetProperty(int type, int *value) {
    (void)type;
    if (value) *value = 0;
    return CUFFT_SUCCESS;
}

cufftResult cufftSetCompatibilityMode(cufftHandle plan,
                                       cufftCompatibility mode) {
    (void)plan; (void)mode;
    return CUFFT_SUCCESS;
}

/* ============================================================================
 * Stubs: not supported
 * ============================================================================ */

/* ============================================================================
 * cufftPlan2d
 * ============================================================================ */

cufftResult cufftPlan2d(cufftHandle *plan, int nx, int ny, cufftType type)
{
    CUVK_LOG("[cufft] cufftPlan2d: nx=%d ny=%d type=0x%x\n", nx, ny, type);

    if (!plan || nx <= 0 || ny <= 0)
        return CUFFT_INVALID_VALUE;
    if (type != CUFFT_C2C)
        return CUFFT_INVALID_TYPE;

    CufftPlan *p;
    int handle;
    cufftResult cr = plan_init(&p, &handle, type, 1);
    if (cr != CUFFT_SUCCESS) return cr;

    struct CUctx_st *ctx = p->ctx;
    p->rank = 2;
    p->dims[0] = nx; p->dims[1] = ny;
    p->total_elements = nx * ny;
    p->n_axes = 2;

    /* Axis 0: row-wise FFTs — ny-point FFTs on each of nx rows.
     * Data is row-major: element (r,c) = r*ny + c.
     * Row FFT: element_stride=1, batch_stride=ny, batch_count=nx */
    cr = build_axis(ctx, &p->axes[0], ny, 1, ny, nx, p->pipe_layout);
    if (cr != CUFFT_SUCCESS) return cr;

    /* Axis 1: column-wise FFTs — nx-point FFTs on each of ny columns.
     * Column c: elements at c, c+ny, c+2*ny, ...
     * element_stride=ny, batch_stride=1, batch_count=ny */
    cr = build_axis(ctx, &p->axes[1], nx, ny, 1, ny, p->pipe_layout);
    if (cr != CUFFT_SUCCESS) return cr;

    cr = alloc_plan_resources(p);
    if (cr != CUFFT_SUCCESS) return cr;

    p->valid = 1;
    *plan = handle;
    CUVK_LOG("[cufft] cufftPlan2d: SUCCESS handle=%d\n", handle);
    return CUFFT_SUCCESS;
}

/* ============================================================================
 * cufftPlan3d
 * ============================================================================ */

cufftResult cufftPlan3d(cufftHandle *plan, int nx, int ny, int nz,
                         cufftType type)
{
    CUVK_LOG("[cufft] cufftPlan3d: nx=%d ny=%d nz=%d type=0x%x\n",
             nx, ny, nz, type);

    if (!plan || nx <= 0 || ny <= 0 || nz <= 0)
        return CUFFT_INVALID_VALUE;
    if (type != CUFFT_C2C && type != CUFFT_R2C && type != CUFFT_C2R)
        return CUFFT_INVALID_TYPE;

    CufftPlan *p;
    int handle;
    cufftResult cr = plan_init(&p, &handle, type, 1);
    if (cr != CUFFT_SUCCESS) return cr;

    struct CUctx_st *ctx = p->ctx;
    p->rank = 3;
    p->dims[0] = nx; p->dims[1] = ny; p->dims[2] = nz;

    if (type == CUFFT_C2C) {
        p->total_elements = nx * ny * nz;
        p->n_axes = 3;
        int slice = ny * nz;

        /* Axis 0: z-direction FFTs — nz-point, contiguous.
         * gid.y = iy (0..ny-1), gid.z = ix (0..nx-1)
         * batch_offset = iy * nz + ix * ny*nz */
        cr = build_axis2(ctx, &p->axes[0], nz, 1, nz, ny, slice, nx,
                          p->pipe_layout);
        if (cr != CUFFT_SUCCESS) return cr;

        /* Axis 1: y-direction FFTs — ny-point, strided.
         * gid.y = iz (0..nz-1), gid.z = ix (0..nx-1)
         * batch_offset = iz * 1 + ix * ny*nz
         * element j: batch_offset + j * nz */
        cr = build_axis2(ctx, &p->axes[1], ny, nz, 1, nz, slice, nx,
                          p->pipe_layout);
        if (cr != CUFFT_SUCCESS) return cr;

        /* Axis 2: x-direction FFTs — nx-point.
         * gid.y = iz (0..nz-1), gid.z = iy (0..ny-1)
         * batch_offset = iz * 1 + iy * nz
         * element i: batch_offset + i * ny*nz */
        cr = build_axis2(ctx, &p->axes[2], nx, slice, 1, nz, nz, ny,
                          p->pipe_layout);
        if (cr != CUFFT_SUCCESS) return cr;
    } else {
        /* R2C / C2R: R2C/C2R along z (innermost), C2C along y and x.
         * R2C output layout: [nx][ny][nz/2+1] complex.
         * C2R input layout:  [nx][ny][nz/2+1] complex. */
        if (nz < 2 || nz % 2 != 0) return CUFFT_INVALID_SIZE;
        int half_z = nz / 2;
        int padded_z = half_z + 1;  /* complex elements per row in R2C output */

        p->r2c_n = nz;
        p->total_elements = nx * ny * padded_z;
        p->n_axes = 3;

        /* Axis 0: z R2C/C2R — half_z-point C2C on contiguous rows.
         * Input rows are nz reals = half_z complex pairs.
         * gid.y = iy (0..ny-1), gid.z = ix (0..nx-1)
         * batch_offset = iy * half_z + ix * ny*half_z */
        cr = build_axis2(ctx, &p->axes[0], half_z, 1, half_z, ny,
                          ny * half_z, nx, p->pipe_layout);
        if (cr != CUFFT_SUCCESS) return cr;

        /* R2C post-processing pipeline */
        {
            char *wgsl = gen_fft_r2c_postprocess(nz, FFT_WORKGROUP_SIZE);
            if (!wgsl) return CUFFT_INTERNAL_ERROR;
            uint32_t *spirv = NULL;
            size_t spirv_count = 0;
            cr = compile_wgsl(wgsl, &spirv, &spirv_count);
            free(wgsl);
            if (cr != CUFFT_SUCCESS) return cr;
            cr = create_stage_pipeline(ctx, spirv, spirv_count,
                                       p->pipe_layout,
                                       &p->r2c_post_shader,
                                       &p->r2c_post_pipeline);
            wgsl_lower_free(spirv);
            if (cr != CUFFT_SUCCESS) return cr;
            p->r2c_dispatch_x = ((uint32_t)(half_z + 1) +
                                  FFT_WORKGROUP_SIZE - 1) / FFT_WORKGROUP_SIZE;
        }

        /* C2R pre-processing pipeline */
        {
            char *wgsl = gen_fft_c2r_preprocess(nz, FFT_WORKGROUP_SIZE);
            if (!wgsl) return CUFFT_INTERNAL_ERROR;
            uint32_t *spirv = NULL;
            size_t spirv_count = 0;
            cr = compile_wgsl(wgsl, &spirv, &spirv_count);
            free(wgsl);
            if (cr != CUFFT_SUCCESS) return cr;
            cr = create_stage_pipeline(ctx, spirv, spirv_count,
                                       p->pipe_layout,
                                       &p->c2r_pre_shader,
                                       &p->c2r_pre_pipeline);
            wgsl_lower_free(spirv);
            if (cr != CUFFT_SUCCESS) return cr;
        }

        /* Axis 1: y-direction C2C — ny-point on padded complex layout.
         * gid.y = iz_c (0..padded_z-1), gid.z = ix (0..nx-1)
         * batch_offset = iz_c * 1 + ix * ny*padded_z
         * element j: batch_offset + j * padded_z */
        cr = build_axis2(ctx, &p->axes[1], ny, padded_z, 1, padded_z,
                          ny * padded_z, nx, p->pipe_layout);
        if (cr != CUFFT_SUCCESS) return cr;

        /* Axis 2: x-direction C2C — nx-point on padded complex layout.
         * gid.y = iz_c (0..padded_z-1), gid.z = iy (0..ny-1)
         * batch_offset = iz_c * 1 + iy * padded_z
         * element i: batch_offset + i * ny*padded_z */
        cr = build_axis2(ctx, &p->axes[2], nx, ny * padded_z, 1, padded_z,
                          padded_z, ny, p->pipe_layout);
        if (cr != CUFFT_SUCCESS) return cr;
    }

    cr = alloc_plan_resources(p);
    if (cr != CUFFT_SUCCESS) return cr;

    p->valid = 1;
    *plan = handle;
    CUVK_LOG("[cufft] cufftPlan3d: SUCCESS handle=%d\n", handle);
    return CUFFT_SUCCESS;
}

cufftResult cufftPlanMany(cufftHandle *plan, int rank, int *n,
                           int *inembed, int istride, int idist,
                           int *onembed, int ostride, int odist,
                           cufftType type, int batch) {
    (void)plan; (void)rank; (void)n;
    (void)inembed; (void)istride; (void)idist;
    (void)onembed; (void)ostride; (void)odist;
    (void)type; (void)batch;
    return CUFFT_NOT_SUPPORTED;
}

cufftResult cufftExecC2R(cufftHandle plan_handle, cufftComplex *idata,
                          cufftReal *odata)
{
    CUVK_LOG("[cufft] cufftExecC2R: plan=%d\n", plan_handle);

    if (plan_handle < 0 || plan_handle >= MAX_CUFFT_PLANS)
        return CUFFT_INVALID_PLAN;
    CufftPlan *p = &g_cufft_plans[plan_handle];
    if (!p->valid || !p->c2r_pre_pipeline) return CUFFT_INVALID_PLAN;

    struct CUctx_st *ctx = g_cuvk.current_ctx;
    if (!ctx) ctx = p->ctx;

    CUdeviceptr iptr = (CUdeviceptr)idata;
    CUdeviceptr optr = (CUdeviceptr)odata;
    CuvkAlloc *alloc_in = cuvk_alloc_lookup(ctx, iptr);
    if (!alloc_in) return CUFFT_INVALID_VALUE;
    CuvkAlloc *alloc_out = cuvk_alloc_lookup(ctx, optr);
    if (!alloc_out) return CUFFT_INVALID_VALUE;

    VkBuffer src_buf = alloc_in->buffer;
    VkBuffer dst_buf = alloc_out->buffer;

    g_cuvk.vk.vkDeviceWaitIdle(ctx->device);
    g_cuvk.vk.vkResetDescriptorPool(ctx->device, p->desc_pool, 0);
    g_cuvk.vk.vkResetCommandBuffer(p->cb_inv, 0);
    p->cb_inv_valid = 0;

    VkSubmitInfo submit = {0};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &p->cb_inv;
    cufftResult cr;
    VkResult vr;

    /* Phase 1: C2C inverse on remaining axes (in-place on src_buf) */
    if (p->n_axes > 1) {
        cr = record_fft_cb_range(p, 1, p->cb_inv, src_buf, src_buf,
                                 1, p->n_axes);
        if (cr != CUFFT_SUCCESS) return cr;

        g_cuvk.vk.vkResetFences(ctx->device, 1, &p->fence);
        vr = g_cuvk.vk.vkQueueSubmit(ctx->compute_queue, 1, &submit, p->fence);
        if (vr != VK_SUCCESS) return CUFFT_EXEC_FAILED;
        vr = g_cuvk.vk.vkWaitForFences(ctx->device, 1, &p->fence, VK_TRUE, UINT64_MAX);
        if (vr != VK_SUCCESS) return CUFFT_EXEC_FAILED;

        g_cuvk.vk.vkResetDescriptorPool(ctx->device, p->desc_pool, 0);
        g_cuvk.vk.vkResetCommandBuffer(p->cb_inv, 0);
    }

    /* Phase 2: C2R on axis 0 (pre-process + C2C inverse stages) → dst_buf */
    cr = record_c2r_cb(p, p->cb_inv, src_buf, dst_buf);
    if (cr != CUFFT_SUCCESS) return cr;

    g_cuvk.vk.vkResetFences(ctx->device, 1, &p->fence);
    vr = g_cuvk.vk.vkQueueSubmit(ctx->compute_queue, 1, &submit, p->fence);
    if (vr != VK_SUCCESS) return CUFFT_EXEC_FAILED;
    vr = g_cuvk.vk.vkWaitForFences(ctx->device, 1, &p->fence, VK_TRUE, UINT64_MAX);
    if (vr != VK_SUCCESS) return CUFFT_EXEC_FAILED;

    return CUFFT_SUCCESS;
}

cufftResult cufftExecZ2Z(cufftHandle plan, cufftDoubleComplex *idata,
                          cufftDoubleComplex *odata, int direction) {
    (void)plan; (void)idata; (void)odata; (void)direction;
    return CUFFT_NOT_SUPPORTED;
}

cufftResult cufftExecD2Z(cufftHandle plan, cufftDoubleReal *idata,
                          cufftDoubleComplex *odata) {
    (void)plan; (void)idata; (void)odata;
    return CUFFT_NOT_SUPPORTED;
}

cufftResult cufftExecZ2D(cufftHandle plan, cufftDoubleComplex *idata,
                          cufftDoubleReal *odata) {
    (void)plan; (void)idata; (void)odata;
    return CUFFT_NOT_SUPPORTED;
}

cufftResult cufftCreate(cufftHandle *handle) {
    if (!handle) return CUFFT_INVALID_VALUE;
    *handle = -1;
    return CUFFT_SUCCESS;
}

cufftResult cufftMakePlan1d(cufftHandle plan, int nx, cufftType type,
                             int batch, size_t *workSize) {
    (void)plan; (void)nx; (void)type; (void)batch; (void)workSize;
    return CUFFT_NOT_SUPPORTED;
}

cufftResult cufftGetSize(cufftHandle handle, size_t *workSize) {
    (void)handle;
    if (workSize) *workSize = 0;
    return CUFFT_SUCCESS;
}

cufftResult cufftEstimate1d(int nx, cufftType type, int batch,
                             size_t *workSize) {
    (void)nx; (void)type; (void)batch;
    if (workSize) *workSize = 0;
    return CUFFT_SUCCESS;
}
