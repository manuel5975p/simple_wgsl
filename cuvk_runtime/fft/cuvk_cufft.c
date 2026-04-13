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
#include "fft_fused_gen.h"
#include "fft_fourstep_gen.h"
#include "fft_2d_gen.h"
#include "fft_bda.h"
#include "fft_butterfly.h"

#include "cufft.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Internal plan structure — Stockham codelet-based, multi-axis
 * ============================================================================ */

#define MAX_CUFFT_PLANS    64
#define MAX_FFT_STAGES     20
#define MAX_RADIX          32
#define MAX_AXES           3
#define FFT_WORKGROUP_SIZE 256

/* Stage types for hybrid four-step FFT */
typedef enum {
    FFT_STAGE_STOCKHAM,           /* existing per-radix Stockham stage */
    FFT_STAGE_FUSED,              /* single-dispatch fused sub-FFT */
    FFT_STAGE_TWIDDLE_TRANSPOSE,  /* twiddle multiply + matrix transpose */
    FFT_STAGE_TRANSPOSE,          /* plain matrix transpose */
    FFT_STAGE_2D_FUSED,           /* single-dispatch 2D FFT (row + col in shared) */
    FFT_STAGE_TILED_TRANSPOSE,    /* shared-memory tiled transpose */
} FftStageType;

typedef struct {
    FftStageType type;
    VkBuffer     lut_buf[2];      /* [fwd/inv] fused stage LUT (VK_NULL_HANDLE if none) */
    VkDeviceMemory lut_mem[2];
    VkDeviceAddress lut_bda[2];   /* [fwd/inv] device address for LUT */
    uint32_t     dispatch_y;      /* per-stage gid.y dispatch count */
} FftStageInfo;

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
    FftStageInfo     stage_info[MAX_FFT_STAGES];        /* per-stage metadata */
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
    VkPipelineLayout      pipe_layout;

    VkBuffer       scratch_buf;
    VkDeviceMemory scratch_mem;
    VkDeviceAddress scratch_bda;

    VkCommandBuffer cb_fwd;
    VkCommandBuffer cb_inv;
    VkFence         fence;

    VkBuffer bound_src;
    VkBuffer bound_dst;
    int      bound_inplace;
    VkDeviceAddress bound_src_bda;
    VkDeviceAddress bound_dst_bda;
    int      cb_fwd_valid;
    int      cb_inv_valid;

    /* Deferred replay: accumulate repeated execs, record a single mega-CB
     * with N dispatches at flush time.  Avoids per-exec vkQueueSubmit. */
    int      replay_count;      /* number of deferred iterations */
    int      replay_dir_idx;    /* direction index */

    /* Looped pipeline: single-dispatch shader with control-buffer repeat
     * count for in-place single-workgroup 2D fused FFTs. */
    VkShaderModule        loop_shaders[2];    /* [fwd, inv] */
    VkPipeline            loop_pipelines[2];  /* [fwd, inv] */
    int                   has_loop_pipeline;
    VkBuffer              ctl_buf;            /* 4-byte host-coherent control buffer */
    VkDeviceMemory        ctl_mem;
    void                 *ctl_mapped;         /* persistently mapped */
    VkDeviceAddress       ctl_bda;

    /* Fused 2-dispatch R2C (forward only) */
    int              r2c_fused;
    FftAxis          r2c_fused_axis;
} CufftPlan;

static CufftPlan g_cufft_plans[MAX_CUFFT_PLANS];
static int g_cufft_next_handle = 0;

/* factorize_fft: thin wrapper around fft_stockham_factorize with max_radix=32 */
static int factorize_fft(int n, int *radices, int max_stages) {
    (void)max_stages;
    return fft_stockham_factorize(n, MAX_RADIX, radices);
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
    buf_ci.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
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

    VkMemoryAllocateFlagsInfo flags_info = {0};
    flags_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    alloc_info.pNext = &flags_info;

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
 * Helper: create a host-visible buffer for LUT data
 * ============================================================================ */

static cufftResult create_lut_buffer(struct CUctx_st *ctx,
                                     const float *data, int float_count,
                                     VkBuffer *out_buf, VkDeviceMemory *out_mem)
{
    VkDeviceSize size = (VkDeviceSize)float_count * sizeof(float);

    VkBufferCreateInfo bci = {0};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult vr = g_cuvk.vk.vkCreateBuffer(ctx->device, &bci, NULL, out_buf);
    if (vr != VK_SUCCESS) return CUFFT_ALLOC_FAILED;

    VkMemoryRequirements reqs;
    g_cuvk.vk.vkGetBufferMemoryRequirements(ctx->device, *out_buf, &reqs);

    /* Find host-visible, host-coherent memory type (use cached mem_props) */
    int32_t mem_type = cuvk_find_memory_type(
        &ctx->mem_props, reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mem_type < 0) {
        g_cuvk.vk.vkDestroyBuffer(ctx->device, *out_buf, NULL);
        return CUFFT_ALLOC_FAILED;
    }

    VkMemoryAllocateInfo mai = {0};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = reqs.size;
    mai.memoryTypeIndex = (uint32_t)mem_type;

    VkMemoryAllocateFlagsInfo lut_flags_info = {0};
    lut_flags_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    lut_flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    mai.pNext = &lut_flags_info;

    vr = g_cuvk.vk.vkAllocateMemory(ctx->device, &mai, NULL, out_mem);
    if (vr != VK_SUCCESS) {
        g_cuvk.vk.vkDestroyBuffer(ctx->device, *out_buf, NULL);
        return CUFFT_ALLOC_FAILED;
    }

    g_cuvk.vk.vkBindBufferMemory(ctx->device, *out_buf, *out_mem, 0);

    /* Map and copy (leave mapped — memory is host-coherent) */
    void *mapped;
    vr = g_cuvk.vk.vkMapMemory(ctx->device, *out_mem, 0, size, 0, &mapped);
    if (vr != VK_SUCCESS) {
        g_cuvk.vk.vkDestroyBuffer(ctx->device, *out_buf, NULL);
        g_cuvk.vk.vkFreeMemory(ctx->device, *out_mem, NULL);
        *out_buf = VK_NULL_HANDLE;
        *out_mem = VK_NULL_HANDLE;
        return CUFFT_ALLOC_FAILED;
    }
    memcpy(mapped, data, (size_t)size);

    return CUFFT_SUCCESS;
}


/* ============================================================================
 * Stockham pipeline helpers
 * ============================================================================ */

/* Create a pipeline layout with push constants for BDA addresses.
 * Max 24 bytes (3 x u64): src, dst, lut/ctl. */
static cufftResult create_bda_layouts(struct CUctx_st *ctx,
                                      VkPipelineLayout *out_pipe_layout)
{
    VkPushConstantRange pc_range = {0};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset = 0;
    pc_range.size = FFT_BDA_PC_SIZE_3BUF;  /* 24 bytes max (3 x u64) */

    VkPipelineLayoutCreateInfo pl_ci = {0};
    pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_ci.setLayoutCount = 0;
    pl_ci.pSetLayouts = NULL;
    pl_ci.pushConstantRangeCount = 1;
    pl_ci.pPushConstantRanges = &pc_range;

    VkResult vr = g_cuvk.vk.vkCreatePipelineLayout(ctx->device, &pl_ci, NULL,
                                          out_pipe_layout);
    if (vr != VK_SUCCESS) return CUFFT_INTERNAL_ERROR;
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
 * Deferred replay: record a mega-CB with N iterations of the same FFT
 * ============================================================================ */

static void flush_plan_replay(CufftPlan *p, struct CUctx_st *ctx)
{
    int n = p->replay_count;
    if (n == 0) return;

    int dir_idx = p->replay_dir_idx;
    int inplace = p->bound_inplace;

    /* Source CB provides the execution template */
    VkCommandBuffer cached_cb = (dir_idx == 0) ? p->cb_fwd : p->cb_inv;
    (void)cached_cb;

    /* Use cb_inv for replay if exec is forward, or cb_fwd otherwise,
     * so we don't clobber the cached CB.  For single-direction usage
     * (the common benchmark case), the other CB is free. */
    VkCommandBuffer replay_cb = (dir_idx == 0) ? p->cb_inv : p->cb_fwd;

    g_cuvk.vk.vkResetCommandBuffer(replay_cb, 0);

    /* Figure out buffer assignments for the single stage */
    VkBuffer dst_buf = p->bound_dst;
    VkDeviceAddress dst_bda = p->bound_dst_bda;
    VkDeviceAddress read_bda, write_bda;
    int need_copy = 0;

    FftAxis *axis = &p->axes[0];
    int total_stages = 0;
    for (int a = 0; a < p->n_axes; a++)
        total_stages += p->axes[a].n_stages;

    /* Single-dispatch check: all loads happen before all stores (workgroup
     * barrier separates them), so src == dst is safe — skip scratch copy. */
    uint32_t dy0 = axis->stage_info[0].dispatch_y;
    if (dy0 == 0) dy0 = (uint32_t)axis->batch_count;
    uint32_t dz0 = axis->batch_count2 > 0 ? (uint32_t)axis->batch_count2 : 1;
    int single_dispatch = (axis->dispatch_x[0] <= 1 && dy0 <= 1 && dz0 <= 1);

    /* Fast path: looped pipeline — single dispatch, N iterations inside shader.
     * Only worth it for n > 1 since the overhead is the same for n=1. */
    if (p->has_loop_pipeline && n > 1 && total_stages == 1 && inplace && single_dispatch) {
        /* Write repeat count to control buffer */
        uint32_t repeat_count = (uint32_t)n;
        memcpy(p->ctl_mapped, &repeat_count, sizeof(repeat_count));

        /* Record: one barrier + one dispatch */
        VkCommandBufferBeginInfo begin = {0};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        g_cuvk.vk.vkBeginCommandBuffer(replay_cb, &begin);

        VkMemoryBarrier bar = {0};
        bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT |
                            VK_ACCESS_TRANSFER_WRITE_BIT;
        bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                            VK_ACCESS_SHADER_WRITE_BIT;
        g_cuvk.vk.vkCmdPipelineBarrier(replay_cb,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &bar, 0, NULL, 0, NULL);

        g_cuvk.vk.vkCmdBindPipeline(replay_cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                     p->loop_pipelines[dir_idx]);

        /* Push data + ctl addresses */
        uint64_t pc[2];
        pc[0] = dst_bda;       /* data buffer */
        pc[1] = p->ctl_bda;   /* control buffer */
        g_cuvk.vk.vkCmdPushConstants(replay_cb, p->pipe_layout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);

        g_cuvk.vk.vkCmdDispatch(replay_cb, axis->dispatch_x[0], dy0, dz0);

        g_cuvk.vk.vkEndCommandBuffer(replay_cb);

        g_cuvk.vk.vkResetFences(ctx->device, 1, &p->fence);
        VkSubmitInfo submit = {0};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &replay_cb;
        g_cuvk.vk.vkQueueSubmit(ctx->compute_queue, 1, &submit, p->fence);
        g_cuvk.vk.vkWaitForFences(ctx->device, 1, &p->fence, VK_TRUE, UINT64_MAX);

        p->replay_count = 0;
        return;
    }

    /* For fused stages, each workgroup processes a disjoint set of FFTs
     * (no cross-workgroup data dependencies), so self-aliasing is always
     * safe regardless of dispatch count. */
    int fused_stage = (total_stages == 1 &&
                       axis->stage_info[0].type == FFT_STAGE_FUSED);

    if (total_stages == 1 && inplace && (single_dispatch || fused_stage)) {
        /* Self-aliasing: both bindings point to dst_buf, no copy needed */
        read_bda = dst_bda;
        write_bda = dst_bda;
        need_copy = 0;
    } else if (total_stages == 1 && inplace) {
        read_bda = dst_bda;
        write_bda = p->scratch_bda;
        need_copy = 1;
    } else {
        /* Multi-stage or out-of-place: fall back to single submit */
        VkCommandBuffer cb = (dir_idx == 0) ? p->cb_fwd : p->cb_inv;
        for (int i = 0; i < n; i++) {
            g_cuvk.vk.vkResetFences(ctx->device, 1, &p->fence);
            VkSubmitInfo sub = {0};
            sub.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            sub.commandBufferCount = 1;
            sub.pCommandBuffers = &cb;
            g_cuvk.vk.vkQueueSubmit(ctx->compute_queue, 1, &sub, p->fence);
            g_cuvk.vk.vkWaitForFences(ctx->device, 1, &p->fence, VK_TRUE, UINT64_MAX);
        }
        p->replay_count = 0;
        return;
    }

    VkDeviceSize buf_size = (VkDeviceSize)p->total_elements * 2 * sizeof(float)
                          * (VkDeviceSize)p->batch;

    /* Record mega-CB */
    VkCommandBufferBeginInfo begin = {0};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    g_cuvk.vk.vkBeginCommandBuffer(replay_cb, &begin);

    g_cuvk.vk.vkCmdBindPipeline(replay_cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 axis->pipelines[dir_idx][0]);

    /* Push BDA addresses */
    {
        uint64_t pc[3];
        pc[0] = read_bda;
        pc[1] = write_bda;
        VkDeviceAddress lut_bda_val = axis->stage_info[0].lut_bda[dir_idx];
        pc[2] = lut_bda_val;
        int npc = lut_bda_val ? 3 : 2;
        g_cuvk.vk.vkCmdPushConstants(replay_cb, p->pipe_layout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)(npc * 8), pc);
    }

    uint32_t dy = axis->stage_info[0].dispatch_y;
    if (dy == 0) dy = (uint32_t)axis->batch_count;
    uint32_t dz = axis->batch_count2 > 0 ? (uint32_t)axis->batch_count2 : 1;

    for (int i = 0; i < n; i++) {
        if (!need_copy) {
            /* Self-aliasing path: only barrier before first dispatch
             * (to make prior memcpy visible).  Between self-aliasing
             * dispatches of a single workgroup, NVIDIA L2 coherence
             * ensures visibility without explicit barriers. */
            if (i == 0) {
                VkMemoryBarrier bar = {0};
                bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT |
                                    VK_ACCESS_TRANSFER_WRITE_BIT;
                bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                    VK_ACCESS_SHADER_WRITE_BIT;
                g_cuvk.vk.vkCmdPipelineBarrier(replay_cb,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0, 1, &bar, 0, NULL, 0, NULL);
            }
        } else {
            /* Standard path with copy: need barrier before each dispatch */
            VkMemoryBarrier bar = {0};
            bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT |
                                VK_ACCESS_TRANSFER_WRITE_BIT;
            bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                VK_ACCESS_SHADER_WRITE_BIT;
            g_cuvk.vk.vkCmdPipelineBarrier(replay_cb,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &bar, 0, NULL, 0, NULL);
        }

        g_cuvk.vk.vkCmdDispatch(replay_cb, axis->dispatch_x[0], dy, dz);

        if (need_copy) {
            VkMemoryBarrier copy_bar = {0};
            copy_bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            copy_bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            copy_bar.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            g_cuvk.vk.vkCmdPipelineBarrier(replay_cb,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 1, &copy_bar, 0, NULL, 0, NULL);

            VkBufferCopy region = {0};
            region.size = buf_size;
            g_cuvk.vk.vkCmdCopyBuffer(replay_cb, p->scratch_buf, dst_buf,
                                        1, &region);
        }
    }

    g_cuvk.vk.vkEndCommandBuffer(replay_cb);

    /* Submit + wait */
    g_cuvk.vk.vkResetFences(ctx->device, 1, &p->fence);
    VkSubmitInfo submit = {0};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &replay_cb;
    g_cuvk.vk.vkQueueSubmit(ctx->compute_queue, 1, &submit, p->fence);
    g_cuvk.vk.vkWaitForFences(ctx->device, 1, &p->fence, VK_TRUE, UINT64_MAX);

    p->replay_count = 0;
}

static void cuvk_fft_flush_impl(struct CUctx_st *ctx)
{
    for (int i = 0; i < MAX_CUFFT_PLANS; i++) {
        if (g_cufft_plans[i].valid && g_cufft_plans[i].replay_count > 0)
            flush_plan_replay(&g_cufft_plans[i], ctx);
    }
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
                                       VkDeviceAddress src_bda, VkDeviceAddress dst_bda,
                                       int start_axis, int end_axis)
{
    int inplace = (src_buf == dst_buf);

    VkDeviceSize buf_size = (VkDeviceSize)p->total_elements * 2 * sizeof(float)
                          * (VkDeviceSize)p->batch;

    /* Count total stages across selected axes */
    int total_stages = 0;
    for (int a = start_axis; a < end_axis; a++)
        total_stages += p->axes[a].n_stages;

    /* Determine per-stage read/write buffer assignments across ALL axes.
     * All stages across all axes form one linear sequence for ping-pong.
     * The result must end up in dst_buf. */
    VkBuffer read_bufs[MAX_FFT_STAGES * MAX_AXES];
    VkBuffer write_bufs[MAX_FFT_STAGES * MAX_AXES];
    VkDeviceAddress read_bdas[MAX_FFT_STAGES * MAX_AXES];
    VkDeviceAddress write_bdas[MAX_FFT_STAGES * MAX_AXES];
    int need_final_copy = 0;

    if (total_stages == 1 && inplace) {
        read_bufs[0] = dst_buf;        read_bdas[0] = dst_bda;
        write_bufs[0] = p->scratch_buf; write_bdas[0] = p->scratch_bda;
        need_final_copy = 1;
    } else if (inplace) {
        read_bufs[0] = dst_buf;        read_bdas[0] = dst_bda;
        write_bufs[0] = p->scratch_buf; write_bdas[0] = p->scratch_bda;
        for (int i = 1; i < total_stages; i++) {
            read_bufs[i] = write_bufs[i - 1];
            read_bdas[i] = write_bdas[i - 1];
            if (read_bufs[i] == p->scratch_buf) {
                write_bufs[i] = dst_buf; write_bdas[i] = dst_bda;
            } else {
                write_bufs[i] = p->scratch_buf; write_bdas[i] = p->scratch_bda;
            }
        }
        if (write_bufs[total_stages - 1] != dst_buf)
            need_final_copy = 1;
    } else {
        write_bufs[total_stages - 1] = dst_buf;
        write_bdas[total_stages - 1] = dst_bda;
        for (int i = total_stages - 2; i >= 0; i--) {
            if (write_bufs[i + 1] == dst_buf) {
                write_bufs[i] = p->scratch_buf; write_bdas[i] = p->scratch_bda;
            } else {
                write_bufs[i] = dst_buf; write_bdas[i] = dst_bda;
            }
        }
        read_bufs[0] = src_buf;  read_bdas[0] = src_bda;
        for (int i = 1; i < total_stages; i++) {
            read_bufs[i] = write_bufs[i - 1];
            read_bdas[i] = write_bdas[i - 1];
        }
    }

    /* Begin command buffer — SIMULTANEOUS_USE allows resubmission while
     * a previous submission is still in flight (async exec path). */
    VkCommandBufferBeginInfo begin_info = {0};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    VkResult vr = g_cuvk.vk.vkBeginCommandBuffer(cb, &begin_info);
    if (vr != VK_SUCCESS)
        return CUFFT_INTERNAL_ERROR;

    /* Cross-submission barrier: ensure writes from a previous async
     * submission (compute + transfer) are visible before we read. */
    {
        VkMemoryBarrier start_bar = {0};
        start_bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        start_bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT |
                                  VK_ACCESS_TRANSFER_WRITE_BIT;
        start_bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                  VK_ACCESS_SHADER_WRITE_BIT;
        g_cuvk.vk.vkCmdPipelineBarrier(cb,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &start_bar, 0, NULL, 0, NULL);
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

            /* Push BDA addresses */
            uint64_t pc[3];
            pc[0] = read_bdas[global_stage];
            pc[1] = write_bdas[global_stage];
            VkDeviceAddress stage_lut = axis->stage_info[s].lut_bda[dir_idx];
            pc[2] = stage_lut;
            int npc = stage_lut ? 3 : 2;
            g_cuvk.vk.vkCmdPushConstants(cb, p->pipe_layout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)(npc * 8), pc);

            uint32_t dy = axis->stage_info[s].dispatch_y;
            if (dy == 0) dy = (uint32_t)axis->batch_count;
            uint32_t dz = axis->batch_count2 > 0 ?
                              (uint32_t)axis->batch_count2 : 1;
            g_cuvk.vk.vkCmdDispatch(cb, axis->dispatch_x[s], dy, dz);
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
    if (vr != VK_SUCCESS) return CUFFT_INTERNAL_ERROR;

    return CUFFT_SUCCESS;
}

static cufftResult record_fft_cb(CufftPlan *p, int dir_idx,
                                 VkCommandBuffer cb,
                                 VkBuffer src_buf, VkBuffer dst_buf,
                                 VkDeviceAddress src_bda, VkDeviceAddress dst_bda)
{
    return record_fft_cb_range(p, dir_idx, cb, src_buf, dst_buf,
                               src_bda, dst_bda, 0, p->n_axes);
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

    /* Initialize stage_info for Stockham stages */
    for (int s = 0; s < axis->n_stages; s++) {
        axis->stage_info[s].type = FFT_STAGE_STOCKHAM;
        axis->stage_info[s].lut_buf[0] = VK_NULL_HANDLE;
        axis->stage_info[s].lut_buf[1] = VK_NULL_HANDLE;
        axis->stage_info[s].lut_mem[0] = VK_NULL_HANDLE;
        axis->stage_info[s].lut_mem[1] = VK_NULL_HANDLE;
        axis->stage_info[s].dispatch_y = (uint32_t)axis->batch_count;
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

static int aligned_wg_limit(int sub_n, int actual_batch);
static void destroy_axis_resources(struct CUctx_st *ctx, FftAxis *axis);
static double bench_axis_gpu(struct CUctx_st *ctx, FftAxis *axis,
                              VkPipelineLayout pipe_layout, int total_elements);

/* ============================================================================
 * Fused single-dispatch axis builder (1D batched FFT)
 *
 * Compiles a single fused FFT shader that executes all radix stages in one
 * dispatch using workgroup shared memory / registers.  This gives n_stages=1,
 * enabling mega-CB replay instead of the synchronous fallback.
 * ============================================================================ */

static cufftResult build_fused_axis_mr(struct CUctx_st *ctx, FftAxis *axis,
                                       int n, int batch_count, int max_radix,
                                       VkPipelineLayout pipe_layout)
{
    memset(axis, 0, sizeof(*axis));
    axis->n = n;
    axis->element_stride = 1;
    axis->batch_stride = n;
    axis->batch_count = batch_count;
    axis->n_stages = 1;

    int gen_dirs[2] = {1, -1};
    cufftResult cr;

    /* Check that the fused generator supports this max_radix for n */
    int wg_size = fft_fused_workgroup_size(n, max_radix, 0);
    if (wg_size <= 0) return CUFFT_INTERNAL_ERROR;

    /* Find wg_limit such that batch_per_wg divides batch_count */
    int wpf = fft_fused_workgroup_size(n, max_radix, 1);
    if (wpf <= 0) return CUFFT_INTERNAL_ERROR;
    int max_bpw = fft_fused_batch_per_wg(n, max_radix, 0);
    int best_b = 1;
    for (int b = max_bpw; b >= 1; b--)
        if (batch_count % b == 0) { best_b = b; break; }
    int wg_limit = best_b * wpf;

    int bpw = fft_fused_batch_per_wg(n, max_radix, wg_limit);
    if (bpw <= 0) return CUFFT_INTERNAL_ERROR;

    axis->stage_info[0].type = FFT_STAGE_FUSED;
    axis->stage_info[0].dispatch_y = 1;
    axis->dispatch_x[0] = (uint32_t)(batch_count / bpw);

    CUVK_LOG("[cufft] fused axis: n=%d mr=%d batch=%d bpw=%d wg=%d dispatches=%u\n",
             n, max_radix, batch_count, bpw, wg_limit, axis->dispatch_x[0]);

    /* Compile fused shader for each direction */
    for (int d = 0; d < 2; d++) {
        char *wgsl = gen_fft_fused(n, gen_dirs[d], max_radix, wg_limit);
        if (!wgsl) return CUFFT_INTERNAL_ERROR;
        uint32_t *spirv = NULL; size_t sc = 0;
        cr = compile_wgsl(wgsl, &spirv, &sc);
        free(wgsl);
        if (cr != CUFFT_SUCCESS) return cr;
        cr = create_stage_pipeline(ctx, spirv, sc, pipe_layout,
                                    &axis->shaders[d][0], &axis->pipelines[d][0]);
        wgsl_lower_free(spirv);
        if (cr != CUFFT_SUCCESS) return cr;
    }

    /* Build LUT buffers if needed */
    for (int d = 0; d < 2; d++) {
        int lut_count = fft_fused_lut_size(n, gen_dirs[d], max_radix);
        if (lut_count > 0) {
            float *lut_data = fft_fused_compute_lut(n, gen_dirs[d], max_radix);
            if (!lut_data) return CUFFT_INTERNAL_ERROR;
            cr = create_lut_buffer(ctx, lut_data, lut_count * 2,
                                    &axis->stage_info[0].lut_buf[d],
                                    &axis->stage_info[0].lut_mem[d]);
            free(lut_data);
            if (cr != CUFFT_SUCCESS) return cr;
            VkBufferDeviceAddressInfo ai = {0};
            ai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
            ai.buffer = axis->stage_info[0].lut_buf[d];
            axis->stage_info[0].lut_bda[d] = ctx->pfn_get_bda(ctx->device, &ai);
        }
    }

    /* Store radices for logging */
    {
        int radices[MAX_FFT_STAGES];
        int ns = factorize_mr(n, radices, max_radix);
        for (int i = 0; i < ns && i < MAX_FFT_STAGES; i++)
            axis->radices[i] = radices[i];
    }

    return CUFFT_SUCCESS;
}

/* Build a fused axis with auto-tuning: sweep max_radix candidates, GPU-benchmark
 * each, and pick the fastest.  Falls back to max_radix=0 if benchmarking fails. */
static cufftResult build_fused_axis(struct CUctx_st *ctx, FftAxis *axis,
                                    int n, int batch_count,
                                    VkPipelineLayout pipe_layout)
{
    /* Candidate max_radix values to try.  0 = auto (radix_cap heuristic). */
    static const int candidates[] = {0, 4, 8, 16};
    static const int n_candidates = sizeof(candidates) / sizeof(candidates[0]);

    FftAxis best_axis;
    memset(&best_axis, 0, sizeof(best_axis));
    double best_time = -1.0;
    int best_mr = -1;

    for (int c = 0; c < n_candidates; c++) {
        FftAxis trial;
        cufftResult cr = build_fused_axis_mr(ctx, &trial, n, batch_count,
                                              candidates[c], pipe_layout);
        if (cr != CUFFT_SUCCESS) continue;

        double t = bench_axis_gpu(ctx, &trial, pipe_layout, n * batch_count);
        CUVK_LOG("[cufft] 1D tune n=%d mr=%d: %.0f ns\n",
                 n, candidates[c], t);

        if (t >= 0 && (best_time < 0 || t < best_time)) {
            /* New winner — destroy previous best */
            if (best_mr >= 0)
                destroy_axis_resources(ctx, &best_axis);
            best_axis = trial;
            best_time = t;
            best_mr = candidates[c];
        } else {
            /* Loser — destroy this trial */
            destroy_axis_resources(ctx, &trial);
        }
    }

    if (best_mr < 0)
        return CUFFT_INTERNAL_ERROR;

    CUVK_LOG("[cufft] 1D tune n=%d: winner mr=%d (%.0f ns)\n",
             n, best_mr, best_time);
    *axis = best_axis;
    return CUFFT_SUCCESS;
}

/* ============================================================================
 * Four-step hybrid FFT builder
 * ============================================================================ */

#define FOURSTEP_THRESHOLD 4096  /* max N for single fused dispatch */

/*
 * Find wg_limit for fused sub-FFT such that batch_per_wg divides actual_batch.
 * This eliminates excess threads in the last workgroup, preventing OOB access.
 */
static int aligned_wg_limit(int sub_n, int actual_batch) {
    /* wpf = threads per FFT (B=1 gives workgroup_size = wpf) */
    int wpf = fft_fused_workgroup_size(sub_n, 0, 1);
    if (wpf <= 0) return 0;

    /* Default max batch-per-wg */
    int max_bpw = fft_fused_batch_per_wg(sub_n, 0, 0);

    /* Find largest B <= max_bpw that divides actual_batch */
    int best_b = 1;
    for (int b = max_bpw; b >= 1; b--) {
        if (actual_batch % b == 0) {
            best_b = b;
            break;
        }
    }

    return best_b * wpf;
}

/*
 * Two-pass four-step FFT for batch=1: strided fused kernels eliminate
 * all 3 transpose stages (5 stages → 2 stages).
 *
 * Stage 0: Column DFTs — read columns (stride N2), write contiguous blocks
 * Stage 1: Twiddle + Row DFTs — read/write with stride N1, inline twiddle
 */
static cufftResult build_fourstep_2pass(struct CUctx_st *ctx, FftAxis *axis,
                                         int n, int n1, int n2,
                                         VkPipelineLayout pipe_layout)
{
    memset(axis, 0, sizeof(*axis));
    axis->n = n;
    axis->element_stride = 1;
    axis->batch_stride = n;
    axis->batch_count = 1;
    axis->n_stages = 2;

    int gen_dirs[2] = {1, -1};
    cufftResult cr;

    CUVK_LOG("[cufft] fourstep 2-pass: N=%d = %d x %d\n", n, n1, n2);

    /* --- Stage 0: Column DFTs (strided fused) --- */
    axis->stage_info[0].type = FFT_STAGE_FUSED;
    axis->stage_info[0].dispatch_y = 1;

    int wg_limit_0 = aligned_wg_limit(n1, n2);
    if (wg_limit_0 <= 0) return CUFFT_INTERNAL_ERROR;
    int bpw_0 = fft_fused_batch_per_wg(n1, 0, wg_limit_0);
    if (bpw_0 <= 0) return CUFFT_INTERNAL_ERROR;
    axis->dispatch_x[0] = (uint32_t)(n2 / bpw_0);

    for (int d = 0; d < 2; d++) {
        /* Column DFTs: in_bs=1, in_es=N2, out_bs=N1, out_es=1, no twiddle */
        char *wgsl = gen_fft_fused_strided(n1, gen_dirs[d], 0, wg_limit_0,
                                            n2, 1, n2, n1, 1, 0, NULL);
        if (!wgsl) return CUFFT_INTERNAL_ERROR;
        uint32_t *spirv = NULL; size_t sc = 0;
        cr = compile_wgsl(wgsl, &spirv, &sc);
        free(wgsl);
        if (cr != CUFFT_SUCCESS) return cr;
        cr = create_stage_pipeline(ctx, spirv, sc, pipe_layout,
                                    &axis->shaders[d][0], &axis->pipelines[d][0]);
        wgsl_lower_free(spirv);
        if (cr != CUFFT_SUCCESS) return cr;
    }

    /* LUT for N1-point fused sub-FFT */
    for (int d = 0; d < 2; d++) {
        int lut_count = fft_fused_lut_size(n1, gen_dirs[d], 0);
        if (lut_count > 0) {
            float *lut_data = fft_fused_compute_lut(n1, gen_dirs[d], 0);
            if (!lut_data) return CUFFT_INTERNAL_ERROR;
            cr = create_lut_buffer(ctx, lut_data, lut_count * 2,
                                    &axis->stage_info[0].lut_buf[d],
                                    &axis->stage_info[0].lut_mem[d]);
            free(lut_data);
            if (cr != CUFFT_SUCCESS) return cr;
            { VkBufferDeviceAddressInfo ai={0}; ai.sType=VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO; ai.buffer=axis->stage_info[0].lut_buf[d]; axis->stage_info[0].lut_bda[d]=ctx->pfn_get_bda(ctx->device,&ai); }
        }
    }

    /* --- Stage 1: Twiddle + Row DFTs (strided fused) --- */
    axis->stage_info[1].type = FFT_STAGE_FUSED;
    axis->stage_info[1].dispatch_y = 1;

    int wg_limit_1 = aligned_wg_limit(n2, n1);
    if (wg_limit_1 <= 0) return CUFFT_INTERNAL_ERROR;
    int bpw_1 = fft_fused_batch_per_wg(n2, 0, wg_limit_1);
    if (bpw_1 <= 0) return CUFFT_INTERNAL_ERROR;
    axis->dispatch_x[1] = (uint32_t)(n1 / bpw_1);

    for (int d = 0; d < 2; d++) {
        /* Row DFTs: in_bs=1, in_es=N1, out_bs=1, out_es=N1, twiddle=N */
        char *wgsl = gen_fft_fused_strided(n2, gen_dirs[d], 0, wg_limit_1,
                                            n1, 1, n1, 1, n1, n, NULL);
        if (!wgsl) return CUFFT_INTERNAL_ERROR;
        uint32_t *spirv = NULL; size_t sc = 0;
        cr = compile_wgsl(wgsl, &spirv, &sc);
        free(wgsl);
        if (cr != CUFFT_SUCCESS) return cr;
        cr = create_stage_pipeline(ctx, spirv, sc, pipe_layout,
                                    &axis->shaders[d][1], &axis->pipelines[d][1]);
        wgsl_lower_free(spirv);
        if (cr != CUFFT_SUCCESS) return cr;
    }

    /* LUT for N2-point fused sub-FFT */
    for (int d = 0; d < 2; d++) {
        int lut_count = fft_fused_lut_size(n2, gen_dirs[d], 0);
        if (lut_count > 0) {
            float *lut_data = fft_fused_compute_lut(n2, gen_dirs[d], 0);
            if (!lut_data) return CUFFT_INTERNAL_ERROR;
            cr = create_lut_buffer(ctx, lut_data, lut_count * 2,
                                    &axis->stage_info[1].lut_buf[d],
                                    &axis->stage_info[1].lut_mem[d]);
            free(lut_data);
            if (cr != CUFFT_SUCCESS) return cr;
            { VkBufferDeviceAddressInfo ai={0}; ai.sType=VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO; ai.buffer=axis->stage_info[1].lut_buf[d]; axis->stage_info[1].lut_bda[d]=ctx->pfn_get_bda(ctx->device,&ai); }
        }
    }

    return CUFFT_SUCCESS;
}

/*
 * Five-stage four-step FFT for batch>1 (multi-dimensional inner axes):
 *   Stage 0: Transpose N1×N2 → N2×N1
 *   Stage 1: N2 fused FFTs of size N1 (column DFTs)
 *   Stage 2: Twiddle + Transpose N2×N1 → N1×N2
 *   Stage 3: N1 fused FFTs of size N2 (row DFTs)
 *   Stage 4: Transpose N1×N2 → N2×N1 (natural output order)
 */
static cufftResult build_fourstep_5pass(struct CUctx_st *ctx, FftAxis *axis,
                                         int n, int n1, int n2, int batch,
                                         VkPipelineLayout pipe_layout)
{
    memset(axis, 0, sizeof(*axis));
    axis->n = n;
    axis->element_stride = 1;
    axis->batch_stride = n;
    axis->batch_count = batch;
    axis->n_stages = 5;

    int gen_dirs[2] = {1, -1};
    cufftResult cr;

    /* --- Stage 0: Transpose N1×N2 → N2×N1 --- */
    axis->stage_info[0].type = FFT_STAGE_TRANSPOSE;
    axis->stage_info[0].dispatch_y = (uint32_t)batch;
    axis->dispatch_x[0] = (uint32_t)((n + FFT_WORKGROUP_SIZE - 1) / FFT_WORKGROUP_SIZE);

    /* Transpose is direction-independent: compile once, reuse for both dirs */
    {
        char *wgsl = gen_fft_transpose(n1, n2, FFT_WORKGROUP_SIZE);
        if (!wgsl) return CUFFT_INTERNAL_ERROR;
        uint32_t *spirv = NULL; size_t sc = 0;
        cr = compile_wgsl(wgsl, &spirv, &sc);
        free(wgsl);
        if (cr != CUFFT_SUCCESS) return cr;
        cr = create_stage_pipeline(ctx, spirv, sc, pipe_layout,
                                    &axis->shaders[0][0], &axis->pipelines[0][0]);
        wgsl_lower_free(spirv);
        if (cr != CUFFT_SUCCESS) return cr;
        axis->shaders[1][0]  = axis->shaders[0][0];
        axis->pipelines[1][0] = axis->pipelines[0][0];
    }

    /* --- Stage 1: N2 batches of N1-point fused FFTs --- */
    axis->stage_info[1].type = FFT_STAGE_FUSED;
    axis->stage_info[1].dispatch_y = 1;

    int fused_batch_1 = n2 * batch;
    int wg_limit_1 = aligned_wg_limit(n1, fused_batch_1);
    if (wg_limit_1 <= 0) return CUFFT_INTERNAL_ERROR;
    int bpw_1 = fft_fused_batch_per_wg(n1, 0, wg_limit_1);
    if (bpw_1 <= 0) return CUFFT_INTERNAL_ERROR;
    axis->dispatch_x[1] = (uint32_t)(fused_batch_1 / bpw_1);

    for (int d = 0; d < 2; d++) {
        char *wgsl = gen_fft_fused(n1, gen_dirs[d], 0, wg_limit_1);
        if (!wgsl) return CUFFT_INTERNAL_ERROR;
        uint32_t *spirv = NULL; size_t sc = 0;
        cr = compile_wgsl(wgsl, &spirv, &sc);
        free(wgsl);
        if (cr != CUFFT_SUCCESS) return cr;
        cr = create_stage_pipeline(ctx, spirv, sc, pipe_layout,
                                    &axis->shaders[d][1], &axis->pipelines[d][1]);
        wgsl_lower_free(spirv);
        if (cr != CUFFT_SUCCESS) return cr;
    }

    for (int d = 0; d < 2; d++) {
        int lut1_count = fft_fused_lut_size(n1, gen_dirs[d], 0);
        if (lut1_count > 0) {
            float *lut1_data = fft_fused_compute_lut(n1, gen_dirs[d], 0);
            if (!lut1_data) return CUFFT_INTERNAL_ERROR;
            cr = create_lut_buffer(ctx, lut1_data, lut1_count * 2,
                                    &axis->stage_info[1].lut_buf[d],
                                    &axis->stage_info[1].lut_mem[d]);
            free(lut1_data);
            if (cr != CUFFT_SUCCESS) return cr;
            { VkBufferDeviceAddressInfo ai={0}; ai.sType=VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO; ai.buffer=axis->stage_info[1].lut_buf[d]; axis->stage_info[1].lut_bda[d]=ctx->pfn_get_bda(ctx->device,&ai); }
        }
    }

    /* --- Stage 2: Twiddle + Transpose --- */
    axis->stage_info[2].type = FFT_STAGE_TWIDDLE_TRANSPOSE;
    axis->stage_info[2].dispatch_y = (uint32_t)batch;
    axis->dispatch_x[2] = (uint32_t)((n + FFT_WORKGROUP_SIZE - 1) / FFT_WORKGROUP_SIZE);

    for (int d = 0; d < 2; d++) {
        char *wgsl = gen_fft_twiddle_transpose(n2, n1, gen_dirs[d], FFT_WORKGROUP_SIZE);
        if (!wgsl) return CUFFT_INTERNAL_ERROR;
        uint32_t *spirv = NULL; size_t sc = 0;
        cr = compile_wgsl(wgsl, &spirv, &sc);
        free(wgsl);
        if (cr != CUFFT_SUCCESS) return cr;
        cr = create_stage_pipeline(ctx, spirv, sc, pipe_layout,
                                    &axis->shaders[d][2], &axis->pipelines[d][2]);
        wgsl_lower_free(spirv);
        if (cr != CUFFT_SUCCESS) return cr;
    }

    /* --- Stage 3: N1 batches of N2-point fused FFTs --- */
    axis->stage_info[3].type = FFT_STAGE_FUSED;
    axis->stage_info[3].dispatch_y = 1;

    int fused_batch_3 = n1 * batch;
    int wg_limit_3 = aligned_wg_limit(n2, fused_batch_3);
    if (wg_limit_3 <= 0) return CUFFT_INTERNAL_ERROR;
    int bpw_3 = fft_fused_batch_per_wg(n2, 0, wg_limit_3);
    if (bpw_3 <= 0) return CUFFT_INTERNAL_ERROR;
    axis->dispatch_x[3] = (uint32_t)(fused_batch_3 / bpw_3);

    for (int d = 0; d < 2; d++) {
        char *wgsl = gen_fft_fused(n2, gen_dirs[d], 0, wg_limit_3);
        if (!wgsl) return CUFFT_INTERNAL_ERROR;
        uint32_t *spirv = NULL; size_t sc = 0;
        cr = compile_wgsl(wgsl, &spirv, &sc);
        free(wgsl);
        if (cr != CUFFT_SUCCESS) return cr;
        cr = create_stage_pipeline(ctx, spirv, sc, pipe_layout,
                                    &axis->shaders[d][3], &axis->pipelines[d][3]);
        wgsl_lower_free(spirv);
        if (cr != CUFFT_SUCCESS) return cr;
    }

    for (int d = 0; d < 2; d++) {
        int lut3_count = fft_fused_lut_size(n2, gen_dirs[d], 0);
        if (lut3_count > 0) {
            float *lut3_data = fft_fused_compute_lut(n2, gen_dirs[d], 0);
            if (!lut3_data) return CUFFT_INTERNAL_ERROR;
            cr = create_lut_buffer(ctx, lut3_data, lut3_count * 2,
                                    &axis->stage_info[3].lut_buf[d],
                                    &axis->stage_info[3].lut_mem[d]);
            free(lut3_data);
            if (cr != CUFFT_SUCCESS) return cr;
            { VkBufferDeviceAddressInfo ai={0}; ai.sType=VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO; ai.buffer=axis->stage_info[3].lut_buf[d]; axis->stage_info[3].lut_bda[d]=ctx->pfn_get_bda(ctx->device,&ai); }
        }
    }

    /* --- Stage 4: Transpose N1×N2 → N2×N1 (reuse Stage 0's pipeline) --- */
    axis->stage_info[4].type = FFT_STAGE_TRANSPOSE;
    axis->stage_info[4].dispatch_y = (uint32_t)batch;
    axis->dispatch_x[4] = axis->dispatch_x[0];

    for (int d = 0; d < 2; d++) {
        axis->shaders[d][4]  = axis->shaders[0][0];
        axis->pipelines[d][4] = axis->pipelines[0][0];
    }

    return CUFFT_SUCCESS;
}

static cufftResult build_fourstep_axis(struct CUctx_st *ctx, FftAxis *axis,
                                        int n, int n1, int n2, int batch,
                                        VkPipelineLayout pipe_layout)
{
    /* For batch=1 (1D FFT), use 2-pass strided approach (no transposes).
     * For batch>1 (multi-dim inner axes), use 5-pass with explicit transposes. */
    if (batch == 1)
        return build_fourstep_2pass(ctx, axis, n, n1, n2, pipe_layout);
    else
        return build_fourstep_5pass(ctx, axis, n, n1, n2, batch, pipe_layout);
}

/* ============================================================================
 * Helper: allocate plan resources (scratch, CBs, fence)
 * ============================================================================ */

static cufftResult alloc_plan_resources(CufftPlan *p)
{
    struct CUctx_st *ctx = p->ctx;

    /* Scratch buffer */
    VkDeviceSize scratch_size = (VkDeviceSize)p->total_elements * 2 *
                                sizeof(float) * (VkDeviceSize)p->batch;
    cufftResult cr = create_device_buffer(ctx, scratch_size,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              &p->scratch_buf, &p->scratch_mem);
    if (cr != CUFFT_SUCCESS) return cr;

    /* Get BDA for scratch buffer */
    {
        VkBufferDeviceAddressInfo ai = {0};
        ai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        ai.buffer = p->scratch_buf;
        p->scratch_bda = ctx->pfn_get_bda(ctx->device, &ai);
    }

    /* Command buffers */
    VkCommandBufferAllocateInfo cb_ai = {0};
    cb_ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cb_ai.commandPool = ctx->cmd_pool;
    cb_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cb_ai.commandBufferCount = 1;

    VkResult vr = g_cuvk.vk.vkAllocateCommandBuffers(ctx->device, &cb_ai, &p->cb_fwd);
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

    if (!ctx->has_bda) {
        fprintf(stderr, "[cufft] BDA (VK_KHR_buffer_device_address) required for FFT\n");
        return CUFFT_INTERNAL_ERROR;
    }

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

    cufftResult cr = create_bda_layouts(ctx, &p->pipe_layout);
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

        int fs_n1, fs_n2;
        if (nx > FOURSTEP_THRESHOLD &&
            fft_fourstep_decompose(nx, FOURSTEP_THRESHOLD, &fs_n1, &fs_n2) &&
            fs_n1 > 1) {
            /* Four-step hybrid: fused sub-FFTs + twiddle/transpose */
            CUVK_LOG("[cufft] using four-step: %d = %d x %d\n", nx, fs_n1, fs_n2);
            cr = build_fourstep_axis(ctx, &p->axes[0], nx, fs_n1, fs_n2,
                                      p->batch, p->pipe_layout);
        } else if (nx >= 4 && (nx & (nx - 1)) == 0 &&
                   fft_fused_workgroup_size(nx, 0, 0) > 0) {
            /* Fused single-dispatch path for power-of-2 sizes:
             * all radix stages in one shader, enables mega-CB replay.
             * N≤128: GPU-autotuned max_radix sweep.
             * N>128: default max_radix (skip tuning overhead). */
            CUVK_LOG("[cufft] using fused 1D: nx=%d batch=%d\n", nx, p->batch);
            if (nx <= 128) {
                cr = build_fused_axis(ctx, &p->axes[0], nx, p->batch,
                                       p->pipe_layout);
            } else {
                cr = build_fused_axis_mr(ctx, &p->axes[0], nx, p->batch,
                                          0, p->pipe_layout);
            }
        } else {
            /* Fallback: multi-stage Stockham */
            cr = build_axis(ctx, &p->axes[0], nx, 1, nx,
                             p->batch, p->pipe_layout);
        }
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
    VkDeviceAddress src_bda = alloc_in->device_addr;
    VkDeviceAddress dst_bda = alloc_out->device_addr;

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
        /* Flush any deferred submissions that reference the old CBs
         * before we reset and re-record them. */
        cuvk_fft_flush(ctx);
        /* Reset both CBs, record only the requested direction.
         * The other direction will be recorded on demand if needed. */
        g_cuvk.vk.vkDeviceWaitIdle(ctx->device);
        g_cuvk.vk.vkResetCommandBuffer(p->cb_fwd, 0);
        g_cuvk.vk.vkResetCommandBuffer(p->cb_inv, 0);
        p->cb_fwd_valid = 0;
        p->cb_inv_valid = 0;

        VkBuffer effective_src = inplace ? dst_buf : src_buf;
        VkDeviceAddress effective_src_bda = inplace ? dst_bda : src_bda;

        cufftResult cr = record_fft_cb(p, dir_idx, cb, effective_src, dst_buf,
                                        effective_src_bda, dst_bda);
        if (cr != CUFFT_SUCCESS) return cr;

        if (dir_idx == 0)
            p->cb_fwd_valid = 1;
        else
            p->cb_inv_valid = 1;

        p->bound_src = src_buf;
        p->bound_dst = dst_buf;
        p->bound_inplace = inplace;
        p->bound_src_bda = src_bda;
        p->bound_dst_bda = dst_bda;
    }

    /* Register the flush callback if not yet set */
    if (!ctx->fft_flush_fn)
        ctx->fft_flush_fn = cuvk_fft_flush_impl;

    /* Deferred replay: if same plan + direction, just increment counter.
     * At flush time, we record ONE command buffer with N dispatches.
     * This avoids per-exec vkQueueSubmit overhead (~25 µs on NVIDIA). */
    if (p->replay_count > 0 && p->replay_dir_idx == dir_idx) {
        p->replay_count++;
        return CUFFT_SUCCESS;
    }

    /* Different direction or first call: flush any existing replay first */
    if (p->replay_count > 0)
        flush_plan_replay(p, ctx);

    p->replay_count   = 1;
    p->replay_dir_idx = dir_idx;

    return CUFFT_SUCCESS;
}

/* ============================================================================
 * R2C/C2R exec helper: record CB with C2C stages + post/pre-process stage
 * ============================================================================ */

static cufftResult record_r2c_cb(CufftPlan *p, VkCommandBuffer cb,
                                  VkBuffer src_buf, VkBuffer dst_buf,
                                  VkDeviceAddress src_bda, VkDeviceAddress dst_bda)
{
    (void)src_buf; (void)dst_buf;

    FftAxis *axis = &p->axes[0];
    int ns = axis->n_stages;

    /* Ping-pong for C2C stages: forward from src through scratch.
     * Post-process reads from wherever the last C2C stage wrote. */
    VkDeviceAddress read_bdas[MAX_FFT_STAGES + 1];
    VkDeviceAddress write_bdas[MAX_FFT_STAGES + 1];

    if (ns == 0) {
        /* No C2C stages (half=1): post-process reads src directly */
        read_bdas[0] = src_bda;
        write_bdas[0] = dst_bda;
    } else {
        read_bdas[0] = src_bda;
        write_bdas[0] = p->scratch_bda;
        for (int i = 1; i < ns; i++) {
            read_bdas[i] = write_bdas[i - 1];
            write_bdas[i] = (read_bdas[i] == p->scratch_bda)
                           ? src_bda : p->scratch_bda;
        }
        /* Post-process: read from last C2C output, write to dst */
        read_bdas[ns] = write_bdas[ns - 1];
        write_bdas[ns] = dst_bda;
    }

    /* Begin CB */
    VkCommandBufferBeginInfo begin = {0};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VkResult vr = g_cuvk.vk.vkBeginCommandBuffer(cb, &begin);
    if (vr != VK_SUCCESS) return CUFFT_INTERNAL_ERROR;

    /* C2C stages */
    for (int s = 0; s < ns; s++) {
        if (s > 0) emit_barrier(cb);
        g_cuvk.vk.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                          axis->pipelines[0][s]);
        uint64_t pc[3];
        pc[0] = read_bdas[s];
        pc[1] = write_bdas[s];
        pc[2] = 0;
        VkDeviceAddress slut = axis->stage_info[s].lut_bda[0];
        pc[2] = slut;
        int npc = slut ? 3 : 2;
        g_cuvk.vk.vkCmdPushConstants(cb, p->pipe_layout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)(npc * 8), pc);
        uint32_t dy = axis->stage_info[s].dispatch_y;
        if (dy == 0) dy = (uint32_t)axis->batch_count;
        uint32_t dz = axis->batch_count2 > 0 ?
                          (uint32_t)axis->batch_count2 : 1;
        g_cuvk.vk.vkCmdDispatch(cb, axis->dispatch_x[s], dy, dz);
    }

    /* Post-process stage */
    emit_barrier(cb);
    g_cuvk.vk.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                      p->r2c_post_pipeline);
    {
        uint64_t pc[3];
        pc[0] = read_bdas[ns];
        pc[1] = write_bdas[ns];
        pc[2] = 0;
        g_cuvk.vk.vkCmdPushConstants(cb, p->pipe_layout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
    }
    {
        FftAxis *a0 = &p->axes[0];
        uint32_t total_batches_r2c = (uint32_t)a0->batch_count *
                                 (uint32_t)(a0->batch_count2 > 0 ?
                                            a0->batch_count2 : 1);
        g_cuvk.vk.vkCmdDispatch(cb, p->r2c_dispatch_x, total_batches_r2c, 1);
    }

    vr = g_cuvk.vk.vkEndCommandBuffer(cb);
    return (vr == VK_SUCCESS) ? CUFFT_SUCCESS : CUFFT_INTERNAL_ERROR;
}

static cufftResult record_c2r_cb(CufftPlan *p, VkCommandBuffer cb,
                                  VkBuffer src_buf, VkBuffer dst_buf,
                                  VkDeviceAddress src_bda, VkDeviceAddress dst_bda)
{
    (void)src_buf;
    /* C2R pipeline: pre-process src -> scratch, then N/2-point inverse C2C.
     * Input: N/2+1 complex bins, Output: N reals (= N/2 complex pairs). */

    int half = p->r2c_n / 2;
    FftAxis *axis = &p->axes[0];
    int ns = axis->n_stages;

    int total_batches = axis->batch_count *
                        (axis->batch_count2 > 0 ? axis->batch_count2 : 1);
    VkDeviceSize c2c_buf_size = (VkDeviceSize)half * 2 * sizeof(float) *
                                (VkDeviceSize)total_batches;

    /* Stage 0: pre-process (src -> scratch or dst)
     * Stages 1..ns: C2C inverse, ping-pong scratch/dst */
    int total = 1 + ns;
    VkDeviceAddress read_bdas[MAX_FFT_STAGES + 1];
    VkDeviceAddress write_bdas[MAX_FFT_STAGES + 1];
    VkBuffer write_bufs_last = VK_NULL_HANDLE;
    int need_final_copy = 0;

    if (ns == 0) {
        /* No C2C stages (half=1): pre-process writes directly to dst */
        read_bdas[0] = src_bda;
        write_bdas[0] = dst_bda;
        write_bufs_last = dst_buf;
    } else {
        /* Pre-process: read src, write scratch */
        read_bdas[0] = src_bda;
        write_bdas[0] = p->scratch_bda;

        /* C2C stages: ping-pong between scratch and dst_buf */
        read_bdas[1] = p->scratch_bda;
        write_bdas[1] = dst_bda;
        for (int i = 2; i < total; i++) {
            read_bdas[i] = write_bdas[i - 1];
            write_bdas[i] = (read_bdas[i] == dst_bda)
                           ? p->scratch_bda : dst_bda;
        }
        write_bufs_last = (write_bdas[total - 1] == dst_bda) ? dst_buf : p->scratch_buf;
        need_final_copy = (write_bdas[total - 1] != dst_bda);
    }

    /* Begin CB */
    VkCommandBufferBeginInfo begin = {0};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VkResult vr = g_cuvk.vk.vkBeginCommandBuffer(cb, &begin);
    if (vr != VK_SUCCESS) return CUFFT_INTERNAL_ERROR;

    /* Pre-process stage */
    g_cuvk.vk.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                      p->c2r_pre_pipeline);
    {
        uint64_t pc[3];
        pc[0] = read_bdas[0];
        pc[1] = write_bdas[0];
        pc[2] = 0;
        g_cuvk.vk.vkCmdPushConstants(cb, p->pipe_layout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
    }
    {
        FftAxis *a0 = &p->axes[0];
        uint32_t total_batches_c2r = (uint32_t)a0->batch_count *
                                 (uint32_t)(a0->batch_count2 > 0 ?
                                            a0->batch_count2 : 1);
        g_cuvk.vk.vkCmdDispatch(cb, ((uint32_t)half + FFT_WORKGROUP_SIZE - 1) /
                      FFT_WORKGROUP_SIZE, total_batches_c2r, 1);
    }

    /* C2C inverse stages */
    for (int s = 0; s < ns; s++) {
        emit_barrier(cb);
        g_cuvk.vk.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                          axis->pipelines[1][s]);  /* dir_idx=1 = inverse */
        uint64_t pc[3];
        pc[0] = read_bdas[1 + s];
        pc[1] = write_bdas[1 + s];
        VkDeviceAddress slut = axis->stage_info[s].lut_bda[1];
        pc[2] = slut;
        int npc = slut ? 3 : 2;
        g_cuvk.vk.vkCmdPushConstants(cb, p->pipe_layout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)(npc * 8), pc);
        uint32_t dy = axis->stage_info[s].dispatch_y;
        if (dy == 0) dy = (uint32_t)axis->batch_count;
        uint32_t dz = axis->batch_count2 > 0 ?
                          (uint32_t)axis->batch_count2 : 1;
        g_cuvk.vk.vkCmdDispatch(cb, axis->dispatch_x[s], dy, dz);
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

    (void)write_bufs_last;
    vr = g_cuvk.vk.vkEndCommandBuffer(cb);
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
    if (!p->valid || (!p->r2c_post_pipeline && !p->r2c_fused))
        return CUFFT_INVALID_PLAN;

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
    VkDeviceAddress src_bda = alloc_in->device_addr;
    VkDeviceAddress dst_bda = alloc_out->device_addr;

    /* Always re-record for R2C (different CB structure from C2C) */
    cuvk_fft_flush(ctx);
    g_cuvk.vk.vkDeviceWaitIdle(ctx->device);
    g_cuvk.vk.vkResetCommandBuffer(p->cb_fwd, 0);
    p->cb_fwd_valid = 0;

    /* Fused 2-dispatch R2C path */
    if (p->r2c_fused) {
        FftAxis *faxis = &p->r2c_fused_axis;
        int ns = faxis->n_stages;

        VkDeviceAddress rd[MAX_FFT_STAGES], wr[MAX_FFT_STAGES];
        wr[ns - 1] = dst_bda;
        for (int i = ns - 2; i >= 0; i--)
            wr[i] = (wr[i + 1] == dst_bda) ? p->scratch_bda : dst_bda;
        rd[0] = src_bda;
        for (int i = 1; i < ns; i++) rd[i] = wr[i - 1];

        VkCommandBufferBeginInfo begin = {0};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VkResult vr = g_cuvk.vk.vkBeginCommandBuffer(p->cb_fwd, &begin);
        if (vr != VK_SUCCESS) return CUFFT_INTERNAL_ERROR;

        for (int s = 0; s < ns; s++) {
            if (s > 0) emit_barrier(p->cb_fwd);
            g_cuvk.vk.vkCmdBindPipeline(p->cb_fwd,
                VK_PIPELINE_BIND_POINT_COMPUTE, faxis->pipelines[0][s]);
            uint64_t pc[3];
            pc[0] = rd[s]; pc[1] = wr[s];
            VkDeviceAddress slut = faxis->stage_info[s].lut_bda[0];
            pc[2] = slut;
            int npc = slut ? 3 : 2;
            g_cuvk.vk.vkCmdPushConstants(p->cb_fwd, p->pipe_layout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)(npc * 8), pc);
            uint32_t dy = faxis->stage_info[s].dispatch_y;
            if (dy == 0) dy = 1;
            g_cuvk.vk.vkCmdDispatch(p->cb_fwd, faxis->dispatch_x[s], dy, 1);
        }

        vr = g_cuvk.vk.vkEndCommandBuffer(p->cb_fwd);
        if (vr != VK_SUCCESS) {
            CUVK_LOG("[cufft] fused R2C: vkEndCommandBuffer failed: %d\n", vr);
            return CUFFT_INTERNAL_ERROR;
        }

        g_cuvk.vk.vkResetFences(ctx->device, 1, &p->fence);
        VkSubmitInfo submit = {0};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &p->cb_fwd;
        vr = g_cuvk.vk.vkQueueSubmit(ctx->compute_queue, 1, &submit, p->fence);
        if (vr != VK_SUCCESS) {
            CUVK_LOG("[cufft] fused R2C: vkQueueSubmit failed: %d\n", vr);
            return CUFFT_EXEC_FAILED;
        }
        vr = g_cuvk.vk.vkWaitForFences(ctx->device, 1, &p->fence, VK_TRUE, UINT64_MAX);
        if (vr != VK_SUCCESS) {
            CUVK_LOG("[cufft] fused R2C: vkWaitForFences failed: %d\n", vr);
            return CUFFT_EXEC_FAILED;
        }
        return CUFFT_SUCCESS;
    }

    /* Phase 1: R2C on axis 0 (C2C stages + post-process) -> dst_buf */
    cufftResult cr = record_r2c_cb(p, p->cb_fwd, src_buf, dst_buf,
                                    src_bda, dst_bda);
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
        g_cuvk.vk.vkResetCommandBuffer(p->cb_fwd, 0);

        cr = record_fft_cb_range(p, 0, p->cb_fwd, dst_buf, dst_buf,
                                 dst_bda, dst_bda, 1, p->n_axes);
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

    /* Destroy per-axis per-stage pipelines, shader modules, and LUT buffers.
     * Some stages may share handles (e.g. direction-independent transposes),
     * so null out after destroying to avoid double-free. */
    for (int a = 0; a < p->n_axes; a++) {
        FftAxis *axis = &p->axes[a];
        for (int d = 0; d < 2; d++) {
            for (int s = 0; s < axis->n_stages; s++) {
                if (axis->pipelines[d][s]) {
                    VkPipeline pl = axis->pipelines[d][s];
                    g_cuvk.vk.vkDestroyPipeline(ctx->device, pl, NULL);
                    /* Null out all matching handles to prevent double-free */
                    for (int dd = d; dd < 2; dd++)
                        for (int ss = (dd == d ? s : 0); ss < axis->n_stages; ss++)
                            if (axis->pipelines[dd][ss] == pl)
                                axis->pipelines[dd][ss] = VK_NULL_HANDLE;
                }
                if (axis->shaders[d][s]) {
                    VkShaderModule sm = axis->shaders[d][s];
                    g_cuvk.vk.vkDestroyShaderModule(ctx->device, sm, NULL);
                    for (int dd = d; dd < 2; dd++)
                        for (int ss = (dd == d ? s : 0); ss < axis->n_stages; ss++)
                            if (axis->shaders[dd][ss] == sm)
                                axis->shaders[dd][ss] = VK_NULL_HANDLE;
                }
            }
        }
        /* LUT buffers (per-direction) */
        for (int s = 0; s < axis->n_stages; s++) {
            for (int d = 0; d < 2; d++) {
                if (axis->stage_info[s].lut_buf[d]) {
                    g_cuvk.vk.vkDestroyBuffer(ctx->device, axis->stage_info[s].lut_buf[d], NULL);
                    g_cuvk.vk.vkFreeMemory(ctx->device, axis->stage_info[s].lut_mem[d], NULL);
                }
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

    /* Destroy fused R2C axis resources */
    if (p->r2c_fused)
        destroy_axis_resources(ctx, &p->r2c_fused_axis);

    /* Destroy looped pipeline resources */
    if (p->has_loop_pipeline) {
        for (int d = 0; d < 2; d++) {
            if (p->loop_pipelines[d])
                g_cuvk.vk.vkDestroyPipeline(ctx->device, p->loop_pipelines[d], NULL);
            if (p->loop_shaders[d])
                g_cuvk.vk.vkDestroyShaderModule(ctx->device, p->loop_shaders[d], NULL);
        }
        if (p->ctl_buf)
            g_cuvk.vk.vkDestroyBuffer(ctx->device, p->ctl_buf, NULL);
        if (p->ctl_mem)
            g_cuvk.vk.vkFreeMemory(ctx->device, p->ctl_mem, NULL);
    }

    /* Destroy shared layouts */
    if (p->pipe_layout)
        g_cuvk.vk.vkDestroyPipelineLayout(ctx->device, p->pipe_layout, NULL);

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
 * 2D plan builders
 * ============================================================================ */

/*
 * Build a single-axis plan with 1 stage: complete 2D FFT in one dispatch.
 * Uses gen_fft_2d_fused() — entire NxM DFT in workgroup shared memory.
 */
static cufftResult build_2d_fused(struct CUctx_st *ctx, FftAxis *axis,
                                   int nx, int ny, int max_radix,
                                   VkPipelineLayout pipe_layout)
{
    memset(axis, 0, sizeof(*axis));
    axis->n = nx * ny;
    axis->element_stride = 1;
    axis->batch_stride = nx * ny;
    axis->batch_count = 1;
    axis->n_stages = 1;

    int gen_dirs[2] = {1, -1};
    cufftResult cr;

    axis->stage_info[0].type = FFT_STAGE_2D_FUSED;
    axis->stage_info[0].dispatch_y = 1;
    axis->dispatch_x[0] = 1; /* one workgroup per 2D FFT */

    for (int d = 0; d < 2; d++) {
        char *wgsl = gen_fft_2d_fused(nx, ny, gen_dirs[d], max_radix);
        if (!wgsl) return CUFFT_INTERNAL_ERROR;
        uint32_t *spirv = NULL; size_t sc = 0;
        cr = compile_wgsl(wgsl, &spirv, &sc);
        free(wgsl);
        if (cr != CUFFT_SUCCESS) return cr;
        cr = create_stage_pipeline(ctx, spirv, sc, pipe_layout,
                                    &axis->shaders[d][0], &axis->pipelines[d][0]);
        wgsl_lower_free(spirv);
        if (cr != CUFFT_SUCCESS) return cr;
    }

    /* LUT */
    for (int d = 0; d < 2; d++) {
        int lut_count = fft_2d_fused_lut_size(nx, ny, gen_dirs[d], max_radix);
        if (lut_count > 0) {
            float *lut_data = fft_2d_fused_compute_lut(nx, ny, gen_dirs[d], max_radix);
            if (!lut_data) return CUFFT_INTERNAL_ERROR;
            cr = create_lut_buffer(ctx, lut_data, lut_count * 2,
                                    &axis->stage_info[0].lut_buf[d],
                                    &axis->stage_info[0].lut_mem[d]);
            free(lut_data);
            if (cr != CUFFT_SUCCESS) return cr;
            { VkBufferDeviceAddressInfo ai={0}; ai.sType=VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO; ai.buffer=axis->stage_info[0].lut_buf[d]; axis->stage_info[0].lut_bda[d]=ctx->pfn_get_bda(ctx->device,&ai); }
        }
    }

    return CUFFT_SUCCESS;
}

/*
 * Build a multi-axis plan using transpose-based approach:
 *   Axis 0: fused row FFTs (ny-point, batch=nx)
 *   Axis 1: tiled transpose nx×ny → ny×nx
 *   Axis 2: fused row FFTs (nx-point, batch=ny) on transposed data
 *   Axis 3: tiled transpose ny×nx → nx×ny (back to original layout)
 *
 * This uses 4 dispatches but all with coalesced memory access.
 * We build this as a single axis with 4 stages for ping-pong to work.
 */
static cufftResult build_2d_transpose_based(struct CUctx_st *ctx, FftAxis *axis,
                                             int nx, int ny, int tile_dim,
                                             VkPipelineLayout pipe_layout)
{
    memset(axis, 0, sizeof(*axis));
    axis->n = nx * ny;
    axis->element_stride = 1;
    axis->batch_stride = nx * ny;
    axis->batch_count = 1;
    axis->n_stages = 4;

    int gen_dirs[2] = {1, -1};
    cufftResult cr;

    /* Stage 0: row FFTs — ny-point on nx rows (fused, batch=nx) */
    {
        axis->stage_info[0].type = FFT_STAGE_FUSED;
        axis->stage_info[0].dispatch_y = 1;

        int wg_limit = aligned_wg_limit(ny, nx);
        if (wg_limit <= 0) return CUFFT_INTERNAL_ERROR;
        int bpw = fft_fused_batch_per_wg(ny, 0, wg_limit);
        if (bpw <= 0) return CUFFT_INTERNAL_ERROR;
        axis->dispatch_x[0] = (uint32_t)(nx / bpw);

        for (int d = 0; d < 2; d++) {
            char *wgsl = gen_fft_fused(ny, gen_dirs[d], 0, wg_limit);
            if (!wgsl) return CUFFT_INTERNAL_ERROR;
            uint32_t *spirv = NULL; size_t sc = 0;
            cr = compile_wgsl(wgsl, &spirv, &sc);
            free(wgsl);
            if (cr != CUFFT_SUCCESS) return cr;
            cr = create_stage_pipeline(ctx, spirv, sc, pipe_layout,
                                        &axis->shaders[d][0], &axis->pipelines[d][0]);
            wgsl_lower_free(spirv);
            if (cr != CUFFT_SUCCESS) return cr;
        }

        for (int d = 0; d < 2; d++) {
            int lut_count = fft_fused_lut_size(ny, gen_dirs[d], 0);
            if (lut_count > 0) {
                float *lut_data = fft_fused_compute_lut(ny, gen_dirs[d], 0);
                if (!lut_data) return CUFFT_INTERNAL_ERROR;
                cr = create_lut_buffer(ctx, lut_data, lut_count * 2,
                                        &axis->stage_info[0].lut_buf[d],
                                        &axis->stage_info[0].lut_mem[d]);
                free(lut_data);
                if (cr != CUFFT_SUCCESS) return cr;
                { VkBufferDeviceAddressInfo ai={0}; ai.sType=VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO; ai.buffer=axis->stage_info[0].lut_buf[d]; axis->stage_info[0].lut_bda[d]=ctx->pfn_get_bda(ctx->device,&ai); }
            }
        }
    }

    /* Stage 1: tiled transpose nx×ny → ny×nx
     * Dispatch: (ceil(ny/tile), ceil(nx/tile), 1)
     * wid.x = column tile, wid.y = row tile, wid.z = batch */
    {
        axis->stage_info[1].type = FFT_STAGE_TILED_TRANSPOSE;
        axis->stage_info[1].dispatch_y = (uint32_t)((nx + tile_dim - 1) / tile_dim);
        axis->dispatch_x[1] = (uint32_t)((ny + tile_dim - 1) / tile_dim);

        /* Transpose doesn't depend on direction — same kernel for fwd/inv */
        char *wgsl = gen_transpose_tiled(nx, ny, tile_dim);
        if (!wgsl) return CUFFT_INTERNAL_ERROR;
        uint32_t *spirv = NULL; size_t sc = 0;
        cr = compile_wgsl(wgsl, &spirv, &sc);
        free(wgsl);
        if (cr != CUFFT_SUCCESS) return cr;
        for (int d = 0; d < 2; d++) {
            cr = create_stage_pipeline(ctx, spirv, sc, pipe_layout,
                                        &axis->shaders[d][1], &axis->pipelines[d][1]);
            if (cr != CUFFT_SUCCESS) { wgsl_lower_free(spirv); return cr; }
        }
        wgsl_lower_free(spirv);
    }

    /* Stage 2: row FFTs — nx-point on ny rows (fused, batch=ny)
     * Now data is ny×nx after transpose. */
    {
        axis->stage_info[2].type = FFT_STAGE_FUSED;
        axis->stage_info[2].dispatch_y = 1;

        int wg_limit = aligned_wg_limit(nx, ny);
        if (wg_limit <= 0) return CUFFT_INTERNAL_ERROR;
        int bpw = fft_fused_batch_per_wg(nx, 0, wg_limit);
        if (bpw <= 0) return CUFFT_INTERNAL_ERROR;
        axis->dispatch_x[2] = (uint32_t)(ny / bpw);

        for (int d = 0; d < 2; d++) {
            char *wgsl = gen_fft_fused(nx, gen_dirs[d], 0, wg_limit);
            if (!wgsl) return CUFFT_INTERNAL_ERROR;
            uint32_t *spirv = NULL; size_t sc = 0;
            cr = compile_wgsl(wgsl, &spirv, &sc);
            free(wgsl);
            if (cr != CUFFT_SUCCESS) return cr;
            cr = create_stage_pipeline(ctx, spirv, sc, pipe_layout,
                                        &axis->shaders[d][2], &axis->pipelines[d][2]);
            wgsl_lower_free(spirv);
            if (cr != CUFFT_SUCCESS) return cr;
        }

        for (int d = 0; d < 2; d++) {
            int lut_count = fft_fused_lut_size(nx, gen_dirs[d], 0);
            if (lut_count > 0) {
                float *lut_data = fft_fused_compute_lut(nx, gen_dirs[d], 0);
                if (!lut_data) return CUFFT_INTERNAL_ERROR;
                cr = create_lut_buffer(ctx, lut_data, lut_count * 2,
                                        &axis->stage_info[2].lut_buf[d],
                                        &axis->stage_info[2].lut_mem[d]);
                free(lut_data);
                if (cr != CUFFT_SUCCESS) return cr;
                { VkBufferDeviceAddressInfo ai={0}; ai.sType=VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO; ai.buffer=axis->stage_info[2].lut_buf[d]; axis->stage_info[2].lut_bda[d]=ctx->pfn_get_bda(ctx->device,&ai); }
            }
        }
    }

    /* Stage 3: tiled transpose ny×nx → nx×ny (back to original layout)
     * Now input is ny×nx, so dispatch: (ceil(nx/tile), ceil(ny/tile), 1) */
    {
        axis->stage_info[3].type = FFT_STAGE_TILED_TRANSPOSE;
        axis->stage_info[3].dispatch_y = (uint32_t)((ny + tile_dim - 1) / tile_dim);
        axis->dispatch_x[3] = (uint32_t)((nx + tile_dim - 1) / tile_dim);

        char *wgsl = gen_transpose_tiled(ny, nx, tile_dim);
        if (!wgsl) return CUFFT_INTERNAL_ERROR;
        uint32_t *spirv = NULL; size_t sc = 0;
        cr = compile_wgsl(wgsl, &spirv, &sc);
        free(wgsl);
        if (cr != CUFFT_SUCCESS) return cr;
        for (int d = 0; d < 2; d++) {
            cr = create_stage_pipeline(ctx, spirv, sc, pipe_layout,
                                        &axis->shaders[d][3], &axis->pipelines[d][3]);
            if (cr != CUFFT_SUCCESS) { wgsl_lower_free(spirv); return cr; }
        }
        wgsl_lower_free(spirv);
    }

    return CUFFT_SUCCESS;
}

/*
 * Build a 2-stage strided plan: FFT+transpose in each dispatch.
 *   Stage 0: ny-point row FFTs with transposed write (nx×ny → ny×nx)
 *   Stage 1: nx-point row FFTs with transposed write (ny×nx → nx×ny)
 * Only 2 dispatches, but writes are strided (non-coalesced).
 */
static cufftResult build_2d_strided(struct CUctx_st *ctx, FftAxis *axis,
                                     int nx, int ny,
                                     VkPipelineLayout pipe_layout)
{
    memset(axis, 0, sizeof(*axis));
    axis->n = nx * ny;
    axis->element_stride = 1;
    axis->batch_stride = nx * ny;
    axis->batch_count = 1;
    axis->n_stages = 2;

    int gen_dirs[2] = {1, -1};
    cufftResult cr;

    /* Stage 0: row FFTs (ny-point, batch=nx) with transposed write
     * Read row-major: in_bs=ny, in_es=1
     * Write col-major: out_bs=1, out_es=nx (transpose) */
    {
        int wg_limit = aligned_wg_limit(ny, nx);
        if (wg_limit <= 0) return CUFFT_INTERNAL_ERROR;
        int bpw = fft_fused_batch_per_wg(ny, 0, wg_limit);
        if (bpw <= 0) return CUFFT_INTERNAL_ERROR;

        axis->stage_info[0].type = FFT_STAGE_FUSED;
        axis->stage_info[0].dispatch_y = 1;
        axis->dispatch_x[0] = (uint32_t)(nx / bpw);

        for (int d = 0; d < 2; d++) {
            char *wgsl = gen_fft_fused_strided(ny, gen_dirs[d], 0, wg_limit,
                                                nx, ny, 1, 1, nx, 0, NULL);
            if (!wgsl) return CUFFT_INTERNAL_ERROR;
            uint32_t *spirv = NULL; size_t sc = 0;
            cr = compile_wgsl(wgsl, &spirv, &sc);
            free(wgsl);
            if (cr != CUFFT_SUCCESS) return cr;
            cr = create_stage_pipeline(ctx, spirv, sc, pipe_layout,
                                        &axis->shaders[d][0], &axis->pipelines[d][0]);
            wgsl_lower_free(spirv);
            if (cr != CUFFT_SUCCESS) return cr;
        }

        for (int d = 0; d < 2; d++) {
            int lut_count = fft_fused_lut_size(ny, gen_dirs[d], 0);
            if (lut_count > 0) {
                float *lut_data = fft_fused_compute_lut(ny, gen_dirs[d], 0);
                if (!lut_data) return CUFFT_INTERNAL_ERROR;
                cr = create_lut_buffer(ctx, lut_data, lut_count * 2,
                                        &axis->stage_info[0].lut_buf[d],
                                        &axis->stage_info[0].lut_mem[d]);
                free(lut_data);
                if (cr != CUFFT_SUCCESS) return cr;
                { VkBufferDeviceAddressInfo ai={0}; ai.sType=VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO; ai.buffer=axis->stage_info[0].lut_buf[d]; axis->stage_info[0].lut_bda[d]=ctx->pfn_get_bda(ctx->device,&ai); }
            }
        }
    }

    /* Stage 1: row FFTs (nx-point, batch=ny) on transposed data,
     * with transposed write back to original layout.
     * Read row-major from ny×nx: in_bs=nx, in_es=1
     * Write col-major: out_bs=1, out_es=ny (transpose back to nx×ny) */
    {
        int wg_limit = aligned_wg_limit(nx, ny);
        if (wg_limit <= 0) return CUFFT_INTERNAL_ERROR;
        int bpw = fft_fused_batch_per_wg(nx, 0, wg_limit);
        if (bpw <= 0) return CUFFT_INTERNAL_ERROR;

        axis->stage_info[1].type = FFT_STAGE_FUSED;
        axis->stage_info[1].dispatch_y = 1;
        axis->dispatch_x[1] = (uint32_t)(ny / bpw);

        for (int d = 0; d < 2; d++) {
            char *wgsl = gen_fft_fused_strided(nx, gen_dirs[d], 0, wg_limit,
                                                ny, nx, 1, 1, ny, 0, NULL);
            if (!wgsl) return CUFFT_INTERNAL_ERROR;
            uint32_t *spirv = NULL; size_t sc = 0;
            cr = compile_wgsl(wgsl, &spirv, &sc);
            free(wgsl);
            if (cr != CUFFT_SUCCESS) return cr;
            cr = create_stage_pipeline(ctx, spirv, sc, pipe_layout,
                                        &axis->shaders[d][1], &axis->pipelines[d][1]);
            wgsl_lower_free(spirv);
            if (cr != CUFFT_SUCCESS) return cr;
        }

        for (int d = 0; d < 2; d++) {
            int lut_count = fft_fused_lut_size(nx, gen_dirs[d], 0);
            if (lut_count > 0) {
                float *lut_data = fft_fused_compute_lut(nx, gen_dirs[d], 0);
                if (!lut_data) return CUFFT_INTERNAL_ERROR;
                cr = create_lut_buffer(ctx, lut_data, lut_count * 2,
                                        &axis->stage_info[1].lut_buf[d],
                                        &axis->stage_info[1].lut_mem[d]);
                free(lut_data);
                if (cr != CUFFT_SUCCESS) return cr;
                { VkBufferDeviceAddressInfo ai={0}; ai.sType=VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO; ai.buffer=axis->stage_info[1].lut_buf[d]; axis->stage_info[1].lut_bda[d]=ctx->pfn_get_bda(ctx->device,&ai); }
            }
        }
    }

    return CUFFT_SUCCESS;
}

/*
 * Build a 2-stage fused R2C strided axis for 2D R2C FFT (forward only).
 * Stage 0: half_y C2C + inline R2C post-process, transposed write to [padded_y][nx]
 * Stage 1: nx-point C2C on contiguous rows, transposed write to [nx][padded_y]
 */
static cufftResult build_2d_r2c_strided(struct CUctx_st *ctx, FftAxis *axis,
                                         int nx, int ny,
                                         VkPipelineLayout pipe_layout)
{
    if (ny < 2 || ny % 2 != 0) return CUFFT_INTERNAL_ERROR;
    int half_y = ny / 2;
    int padded_y = half_y + 1;

    memset(axis, 0, sizeof(*axis));
    axis->n = nx * padded_y;
    axis->element_stride = 1;
    axis->batch_stride = nx * padded_y;
    axis->batch_count = 1;
    axis->n_stages = 2;

    cufftResult cr;

    /* Stage 0: Row R2C — half_y-point fused C2C + R2C post-process
     * Read row-major: in_bs=half_y, in_es=1
     * Write col-major (transposed): out_bs=1, out_es=nx */
    {
        /* Compute batch-per-wg using R2C sizing, ensuring it divides nx */
        int max_bpw = fft_fused_r2c_batch_per_wg(half_y, 0, 0);
        if (max_bpw <= 0) return CUFFT_INTERNAL_ERROR;
        int bpw = max_bpw;
        while (bpw > 1 && nx % bpw != 0) bpw--;
        int wg_size = fft_fused_r2c_workgroup_size(half_y, 0, 0);
        if (wg_size <= 0) return CUFFT_INTERNAL_ERROR;
        int wpf = wg_size / max_bpw;
        int wg_limit = bpw * wpf;

        axis->stage_info[0].type = FFT_STAGE_FUSED;
        axis->stage_info[0].dispatch_y = 1;
        axis->dispatch_x[0] = (uint32_t)(nx / bpw);

        /* total_batch=0: B divides nx exactly, no bounds guard needed
         * (avoids barrier divergence from early-return threads) */
        char *wgsl = gen_fft_fused_r2c_strided(half_y, 0, wg_limit,
                                                0, half_y, 1, 1, nx, NULL);
        if (!wgsl) return CUFFT_INTERNAL_ERROR;
        uint32_t *spirv = NULL; size_t sc = 0;
        cr = compile_wgsl(wgsl, &spirv, &sc);
        free(wgsl);
        if (cr != CUFFT_SUCCESS) return cr;
        cr = create_stage_pipeline(ctx, spirv, sc, pipe_layout,
                                    &axis->shaders[0][0], &axis->pipelines[0][0]);
        wgsl_lower_free(spirv);
        if (cr != CUFFT_SUCCESS) return cr;

        int lut_count = fft_fused_r2c_lut_size(half_y, 0);
        if (lut_count > 0) {
            float *lut_data = fft_fused_r2c_compute_lut(half_y, 0);
            if (!lut_data) return CUFFT_INTERNAL_ERROR;
            cr = create_lut_buffer(ctx, lut_data, lut_count * 2,
                                    &axis->stage_info[0].lut_buf[0],
                                    &axis->stage_info[0].lut_mem[0]);
            free(lut_data);
            if (cr != CUFFT_SUCCESS) return cr;
            { VkBufferDeviceAddressInfo ai={0}; ai.sType=VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO; ai.buffer=axis->stage_info[0].lut_buf[0]; axis->stage_info[0].lut_bda[0]=ctx->pfn_get_bda(ctx->device,&ai); }
        }
    }

    /* Stage 1: Column C2C — nx-point forward FFT on transposed data.
     * Read from [padded_y][nx]: in_bs=nx, in_es=1
     * Write to [nx][padded_y]: out_bs=1, out_es=padded_y
     * Cap max_radix to force shared-memory path for small nx. */
    {
        int mr1 = 0;
        if (nx <= DIRECT_MAX_N) {
            mr1 = nx / 2;
            if (mr1 > 16) mr1 = 16;
            if (mr1 < 2) mr1 = 2;
        }

        int wpf1 = fft_fused_workgroup_size(nx, mr1, 1);
        if (wpf1 <= 0) return CUFFT_INTERNAL_ERROR;
        int max_bpw1 = fft_fused_batch_per_wg(nx, mr1, 256);
        int best_b1 = 1;
        for (int b = max_bpw1; b >= 1; b--)
            if (padded_y % b == 0) { best_b1 = b; break; }
        int wg_limit = best_b1 * wpf1;
        if (wg_limit <= 0) return CUFFT_INTERNAL_ERROR;
        int bpw = fft_fused_batch_per_wg(nx, mr1, wg_limit);
        if (bpw <= 0) return CUFFT_INTERNAL_ERROR;

        axis->stage_info[1].type = FFT_STAGE_FUSED;
        axis->stage_info[1].dispatch_y = 1;
        axis->dispatch_x[1] = (uint32_t)(padded_y / bpw);

        char *wgsl = gen_fft_fused_strided(nx, 1, mr1, wg_limit,
                                            padded_y, nx, 1, 1, padded_y, 0, NULL);
        if (!wgsl) return CUFFT_INTERNAL_ERROR;
        uint32_t *spirv = NULL; size_t sc = 0;
        cr = compile_wgsl(wgsl, &spirv, &sc);
        free(wgsl);
        if (cr != CUFFT_SUCCESS) return cr;
        cr = create_stage_pipeline(ctx, spirv, sc, pipe_layout,
                                    &axis->shaders[0][1], &axis->pipelines[0][1]);
        wgsl_lower_free(spirv);
        if (cr != CUFFT_SUCCESS) return cr;

        int lut_count = fft_fused_lut_size(nx, 1, mr1);
        if (lut_count > 0) {
            float *lut_data = fft_fused_compute_lut(nx, 1, mr1);
            if (!lut_data) return CUFFT_INTERNAL_ERROR;
            cr = create_lut_buffer(ctx, lut_data, lut_count * 2,
                                    &axis->stage_info[1].lut_buf[0],
                                    &axis->stage_info[1].lut_mem[0]);
            free(lut_data);
            if (cr != CUFFT_SUCCESS) return cr;
            { VkBufferDeviceAddressInfo ai={0}; ai.sType=VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO; ai.buffer=axis->stage_info[1].lut_buf[0]; axis->stage_info[1].lut_bda[0]=ctx->pfn_get_bda(ctx->device,&ai); }
        }
    }

    return CUFFT_SUCCESS;
}

/* ============================================================================
 * Axis resource cleanup (for discarding losing benchmark candidate)
 * ============================================================================ */

static void destroy_axis_resources(struct CUctx_st *ctx, FftAxis *axis)
{
    for (int d = 0; d < 2; d++) {
        for (int s = 0; s < axis->n_stages; s++) {
            if (axis->pipelines[d][s]) {
                VkPipeline pl = axis->pipelines[d][s];
                g_cuvk.vk.vkDestroyPipeline(ctx->device, pl, NULL);
                for (int dd = d; dd < 2; dd++)
                    for (int ss = (dd == d ? s : 0); ss < axis->n_stages; ss++)
                        if (axis->pipelines[dd][ss] == pl)
                            axis->pipelines[dd][ss] = VK_NULL_HANDLE;
            }
            if (axis->shaders[d][s]) {
                VkShaderModule sm = axis->shaders[d][s];
                g_cuvk.vk.vkDestroyShaderModule(ctx->device, sm, NULL);
                for (int dd = d; dd < 2; dd++)
                    for (int ss = (dd == d ? s : 0); ss < axis->n_stages; ss++)
                        if (axis->shaders[dd][ss] == sm)
                            axis->shaders[dd][ss] = VK_NULL_HANDLE;
            }
        }
    }
    for (int s = 0; s < axis->n_stages; s++) {
        for (int d = 0; d < 2; d++) {
            if (axis->stage_info[s].lut_buf[d]) {
                g_cuvk.vk.vkDestroyBuffer(ctx->device, axis->stage_info[s].lut_buf[d], NULL);
                g_cuvk.vk.vkFreeMemory(ctx->device, axis->stage_info[s].lut_mem[d], NULL);
            }
        }
    }
}

/* ============================================================================
 * Benchmark an axis with GPU timestamps (forward direction only)
 * Returns median GPU time in nanoseconds, or -1.0 on failure.
 * ============================================================================ */

#define PLAN_BENCH_WARMUP 3
#define PLAN_BENCH_ITERS  5

static double bench_axis_gpu(struct CUctx_st *ctx, FftAxis *axis,
                              VkPipelineLayout pipe_layout,
                              int total_elements)
{
    int ns = axis->n_stages;
    VkDeviceSize buf_size = (VkDeviceSize)total_elements * 2 * sizeof(float);
    VkResult vr;
    double result = -1.0;

    /* Declare all Vulkan handles upfront to avoid goto-past-declaration */
    VkBuffer bufs[3] = {0};
    VkDeviceMemory mems[3] = {0};
    VkQueryPool ts_pool = VK_NULL_HANDLE;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    /* Temp buffers: src, dst, scratch */
    for (int i = 0; i < 3; i++) {
        cufftResult cr = create_device_buffer(ctx, buf_size,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &bufs[i], &mems[i]);
        if (cr != CUFFT_SUCCESS) goto cleanup;
    }

    /* Get BDA addresses for temp buffers */
    VkDeviceAddress buf_bdas[3];
    for (int i = 0; i < 3; i++) {
        VkBufferDeviceAddressInfo ai = {0};
        ai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        ai.buffer = bufs[i];
        buf_bdas[i] = ctx->pfn_get_bda(ctx->device, &ai);
    }

    /* Timestamp query pool */
    {
        VkQueryPoolCreateInfo qpci = {0};
        qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = 2;
        vr = g_cuvk.vk.vkCreateQueryPool(ctx->device, &qpci, NULL, &ts_pool);
        if (vr != VK_SUCCESS) goto cleanup;
    }

    /* Ping-pong BDA arrays: last stage writes to dst (buf_bdas[1]) */
    VkDeviceAddress read_bdas_arr[MAX_FFT_STAGES], write_bdas_arr[MAX_FFT_STAGES];
    {
        VkDeviceAddress src_bda_ = buf_bdas[0], dst_bda_ = buf_bdas[1], scratch_bda_ = buf_bdas[2];
        write_bdas_arr[ns - 1] = dst_bda_;
        for (int i = ns - 2; i >= 0; i--)
            write_bdas_arr[i] = (write_bdas_arr[i + 1] == dst_bda_) ? scratch_bda_ : dst_bda_;
        read_bdas_arr[0] = src_bda_;
        for (int i = 1; i < ns; i++)
            read_bdas_arr[i] = write_bdas_arr[i - 1];
    }

    /* Command buffer + fence */
    {
        VkCommandBufferAllocateInfo cbai = {0};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = ctx->cmd_pool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        vr = g_cuvk.vk.vkAllocateCommandBuffers(ctx->device, &cbai, &cb);
        if (vr != VK_SUCCESS) goto cleanup;

        VkFenceCreateInfo fci = {0};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vr = g_cuvk.vk.vkCreateFence(ctx->device, &fci, NULL, &fence);
        if (vr != VK_SUCCESS) goto cleanup;
    }

    VkCommandBufferBeginInfo cbbi = {0};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    /* Helper: record and submit the dispatch sequence */
    #define RECORD_DISPATCH(use_timestamps) do { \
        g_cuvk.vk.vkResetCommandBuffer(cb, 0); \
        g_cuvk.vk.vkBeginCommandBuffer(cb, &cbbi); \
        if (use_timestamps) { \
            g_cuvk.vk.vkCmdResetQueryPool(cb, ts_pool, 0, 2); \
            g_cuvk.vk.vkCmdWriteTimestamp(cb, \
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, ts_pool, 0); \
        } \
        for (int s_ = 0; s_ < ns; s_++) { \
            if (s_ > 0) emit_barrier(cb); \
            g_cuvk.vk.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, \
                              axis->pipelines[0][s_]); \
            uint64_t pc_[3]; \
            pc_[0] = read_bdas_arr[s_]; \
            pc_[1] = write_bdas_arr[s_]; \
            VkDeviceAddress slut_ = axis->stage_info[s_].lut_bda[0]; \
            pc_[2] = slut_; \
            int npc_ = slut_ ? 3 : 2; \
            g_cuvk.vk.vkCmdPushConstants(cb, pipe_layout, \
                VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)(npc_ * 8), pc_); \
            uint32_t dy_ = axis->stage_info[s_].dispatch_y; \
            if (dy_ == 0) dy_ = 1; \
            g_cuvk.vk.vkCmdDispatch(cb, axis->dispatch_x[s_], dy_, 1); \
        } \
        if (use_timestamps) \
            g_cuvk.vk.vkCmdWriteTimestamp(cb, \
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, ts_pool, 1); \
        g_cuvk.vk.vkEndCommandBuffer(cb); \
    } while (0)

    #define SUBMIT_AND_WAIT() do { \
        g_cuvk.vk.vkResetFences(ctx->device, 1, &fence); \
        VkSubmitInfo si_ = {0}; \
        si_.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; \
        si_.commandBufferCount = 1; si_.pCommandBuffers = &cb; \
        g_cuvk.vk.vkQueueSubmit(ctx->compute_queue, 1, &si_, fence); \
        g_cuvk.vk.vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX); \
    } while (0)

    /* Warmup */
    RECORD_DISPATCH(0);
    for (int w = 0; w < PLAN_BENCH_WARMUP; w++)
        SUBMIT_AND_WAIT();

    /* Timed runs */
    double times[PLAN_BENCH_ITERS];
    for (int it = 0; it < PLAN_BENCH_ITERS; it++) {
        RECORD_DISPATCH(1);
        SUBMIT_AND_WAIT();
        uint64_t ts[2];
        g_cuvk.vk.vkGetQueryPoolResults(ctx->device, ts_pool, 0, 2,
                              sizeof(ts), ts, sizeof(uint64_t),
                              VK_QUERY_RESULT_64_BIT);
        times[it] = (double)(ts[1] - ts[0]) *
                    (double)ctx->dev_props.limits.timestampPeriod;
    }

    #undef RECORD_DISPATCH
    #undef SUBMIT_AND_WAIT

    /* Median */
    for (int i = 0; i < PLAN_BENCH_ITERS - 1; i++)
        for (int j = i + 1; j < PLAN_BENCH_ITERS; j++)
            if (times[j] < times[i]) { double t = times[i]; times[i] = times[j]; times[j] = t; }
    result = times[PLAN_BENCH_ITERS / 2];

cleanup:
    g_cuvk.vk.vkDeviceWaitIdle(ctx->device);
    if (fence) g_cuvk.vk.vkDestroyFence(ctx->device, fence, NULL);
    if (cb) g_cuvk.vk.vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cb);
    if (ts_pool) g_cuvk.vk.vkDestroyQueryPool(ctx->device, ts_pool, NULL);
    for (int i = 0; i < 3; i++) {
        if (bufs[i]) g_cuvk.vk.vkDestroyBuffer(ctx->device, bufs[i], NULL);
        if (mems[i]) g_cuvk.vk.vkFreeMemory(ctx->device, mems[i], NULL);
    }
    return result;
}

/* ============================================================================
 * cufftPlan2d
 * ============================================================================ */

cufftResult cufftPlan2d(cufftHandle *plan, int nx, int ny, cufftType type)
{
    CUVK_LOG("[cufft] cufftPlan2d: nx=%d ny=%d type=0x%x\n", nx, ny, type);

    if (!plan || nx <= 0 || ny <= 0)
        return CUFFT_INVALID_VALUE;
    if (type != CUFFT_C2C && type != CUFFT_R2C && type != CUFFT_C2R)
        return CUFFT_INVALID_TYPE;

    CufftPlan *p;
    int handle;
    cufftResult cr = plan_init(&p, &handle, type, 1);
    if (cr != CUFFT_SUCCESS) return cr;

    struct CUctx_st *ctx = p->ctx;
    p->rank = 2;
    p->dims[0] = nx; p->dims[1] = ny;

    if (type == CUFFT_R2C || type == CUFFT_C2R) {
        /* R2C/C2R along innermost dimension (ny), C2C on remaining (nx).
         * Output layout: [nx][ny/2+1] complex. */
        if (ny < 2 || ny % 2 != 0) return CUFFT_INVALID_SIZE;
        int half_y = ny / 2;
        int padded_y = half_y + 1;

        p->r2c_n = ny;
        p->total_elements = nx * padded_y;
        p->n_axes = 2;

        /* Axis 0: half_y-point C2C on contiguous rows (used by C2R inverse). */
        cr = build_axis(ctx, &p->axes[0], half_y, 1, half_y,
                         nx, p->pipe_layout);
        if (cr != CUFFT_SUCCESS) return cr;

        /* R2C post-processing pipeline */
        {
            char *wgsl = gen_fft_r2c_postprocess(ny, FFT_WORKGROUP_SIZE);
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
            p->r2c_dispatch_x = ((uint32_t)(half_y + 1) +
                                  FFT_WORKGROUP_SIZE - 1) / FFT_WORKGROUP_SIZE;
        }

        /* C2R pre-processing pipeline */
        {
            char *wgsl = gen_fft_c2r_preprocess(ny, FFT_WORKGROUP_SIZE);
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

        /* Axis 1: x-direction C2C on padded layout (used by C2R inverse
         * and as fallback for forward R2C if fused is unavailable). */
        cr = build_axis(ctx, &p->axes[1], nx, padded_y, 1, padded_y,
                         p->pipe_layout);
        if (cr != CUFFT_SUCCESS) return cr;

        /* Try fused 2-dispatch R2C strided (always preferred when available —
         * 2 dispatches vs 3+, no benchmarking needed to know it's faster). */
        if (half_y <= FOURSTEP_THRESHOLD && nx <= FOURSTEP_THRESHOLD) {
            FftAxis fused_axis;
            cufftResult cr_fused = build_2d_r2c_strided(ctx, &fused_axis,
                                                         nx, ny, p->pipe_layout);
            if (cr_fused == CUFFT_SUCCESS) {
                CUVK_LOG("[cufft] 2D R2C: using fused strided (2 dispatches)\n");
                p->r2c_fused = 1;
                p->r2c_fused_axis = fused_axis;
            }
        }
    } else {

    p->total_elements = nx * ny;

    /* Strategy selection:
     * 1. 2D fused: both dims fit in shared memory (nx*ny ≤ 2048)
     * 2. Transpose-based: both dims ≤ FOURSTEP_THRESHOLD (fused handles each)
     * 3. Fallback: old per-axis approach (row Stockham/four-step + col Stockham) */
    if (nx * ny * 16 <= 32768 && fft_2d_fused_workgroup_size(nx, ny, 0) > 0) {
        CUVK_LOG("[cufft] 2D: using fused 2D kernel %dx%d\n", nx, ny);
        p->n_axes = 1;
        cr = build_2d_fused(ctx, &p->axes[0], nx, ny, 0, p->pipe_layout);

        /* Build looped variant for deferred replay (control-buffer repeat) */
        if (cr == CUFFT_SUCCESS) {
            int gen_dirs[2] = {1, -1};
            p->has_loop_pipeline = 1;
            for (int d = 0; d < 2; d++) {
                char *wgsl = gen_fft_2d_fused_looped(nx, ny, gen_dirs[d], 0);
                if (!wgsl) { p->has_loop_pipeline = 0; break; }
                uint32_t *spirv = NULL; size_t sc = 0;
                cufftResult lr = compile_wgsl(wgsl, &spirv, &sc);
                free(wgsl);
                if (lr != CUFFT_SUCCESS) { p->has_loop_pipeline = 0; break; }
                lr = create_stage_pipeline(ctx, spirv, sc, p->pipe_layout,
                                           &p->loop_shaders[d],
                                           &p->loop_pipelines[d]);
                wgsl_lower_free(spirv);
                if (lr != CUFFT_SUCCESS) {
                    p->has_loop_pipeline = 0; break;
                }
            }
            /* Create 4-byte host-coherent control buffer for repeat count */
            if (p->has_loop_pipeline) {
                VkBufferCreateInfo bci = {0};
                bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                bci.size = sizeof(uint32_t);
                bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
                bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                VkResult cvr = g_cuvk.vk.vkCreateBuffer(ctx->device, &bci,
                                                          NULL, &p->ctl_buf);
                if (cvr == VK_SUCCESS) {
                    VkMemoryRequirements reqs;
                    g_cuvk.vk.vkGetBufferMemoryRequirements(ctx->device,
                                                             p->ctl_buf, &reqs);
                    int32_t mt = cuvk_find_memory_type(&ctx->mem_props,
                        reqs.memoryTypeBits,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                    VkMemoryAllocateInfo mai = {0};
                    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                    mai.allocationSize = reqs.size;
                    mai.memoryTypeIndex = (uint32_t)mt;
                    VkMemoryAllocateFlagsInfo ctl_flags = {0};
                    ctl_flags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
                    ctl_flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
                    mai.pNext = &ctl_flags;
                    cvr = (mt >= 0) ? g_cuvk.vk.vkAllocateMemory(
                        ctx->device, &mai, NULL, &p->ctl_mem) : VK_ERROR_UNKNOWN;
                }
                if (cvr == VK_SUCCESS)
                    cvr = g_cuvk.vk.vkBindBufferMemory(ctx->device, p->ctl_buf,
                                                        p->ctl_mem, 0);
                if (cvr == VK_SUCCESS)
                    cvr = g_cuvk.vk.vkMapMemory(ctx->device, p->ctl_mem, 0,
                                                 sizeof(uint32_t), 0,
                                                 &p->ctl_mapped);
                if (cvr == VK_SUCCESS) {
                    VkBufferDeviceAddressInfo ai = {0};
                    ai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
                    ai.buffer = p->ctl_buf;
                    p->ctl_bda = ctx->pfn_get_bda(ctx->device, &ai);
                }
                if (cvr != VK_SUCCESS)
                    p->has_loop_pipeline = 0;
            }
        }
    } else if (nx <= FOURSTEP_THRESHOLD && ny <= FOURSTEP_THRESHOLD) {
        /* Benchmark two strategies and pick the faster one:
         *   A: 4-dispatch transpose-based (coalesced reads+writes)
         *   B: 2-dispatch strided (FFT+transpose fused, non-coalesced writes) */
        int tile_dim = 32;
        while (tile_dim > nx || tile_dim > ny) tile_dim /= 2;
        if (tile_dim < 2) tile_dim = 2;

        FftAxis axis_transpose, axis_strided;
        cufftResult cr_t = build_2d_transpose_based(ctx, &axis_transpose,
                                                      nx, ny, tile_dim,
                                                      p->pipe_layout);
        cufftResult cr_s = build_2d_strided(ctx, &axis_strided, nx, ny,
                                             p->pipe_layout);

        int use_strided = 0;
        if (cr_t == CUFFT_SUCCESS && cr_s == CUFFT_SUCCESS) {
            double t_transpose = bench_axis_gpu(ctx, &axis_transpose,
                                                 p->pipe_layout,
                                                 nx * ny);
            double t_strided   = bench_axis_gpu(ctx, &axis_strided,
                                                 p->pipe_layout,
                                                 nx * ny);
            CUVK_LOG("[cufft] 2D bench %dx%d: transpose=%.0fns strided=%.0fns\n",
                     nx, ny, t_transpose, t_strided);
            if (t_strided >= 0 && (t_transpose < 0 || t_strided < t_transpose))
                use_strided = 1;
        } else if (cr_s == CUFFT_SUCCESS) {
            use_strided = 1;
        }

        if (use_strided) {
            CUVK_LOG("[cufft] 2D: using strided %dx%d (2 dispatches)\n", nx, ny);
            if (cr_t == CUFFT_SUCCESS)
                destroy_axis_resources(ctx, &axis_transpose);
            p->n_axes = 1;
            p->axes[0] = axis_strided;
            cr = CUFFT_SUCCESS;
        } else {
            CUVK_LOG("[cufft] 2D: using transpose-based %dx%d tile=%d (4 dispatches)\n",
                     nx, ny, tile_dim);
            if (cr_s == CUFFT_SUCCESS)
                destroy_axis_resources(ctx, &axis_strided);
            p->n_axes = 1;
            p->axes[0] = axis_transpose;
            cr = cr_t;
        }
    } else {
        /* Fallback: separate row + column axes (handles four-step for large dims) */
        CUVK_LOG("[cufft] 2D: using per-axis approach %dx%d\n", nx, ny);
        p->n_axes = 2;

        /* Axis 0: row FFTs — ny-point on nx rows */
        {
            int fs_n1, fs_n2;
            if (ny > FOURSTEP_THRESHOLD &&
                fft_fourstep_decompose(ny, FOURSTEP_THRESHOLD, &fs_n1, &fs_n2) &&
                fs_n1 > 1) {
                cr = build_fourstep_axis(ctx, &p->axes[0], ny, fs_n1, fs_n2,
                                          nx, p->pipe_layout);
            } else {
                cr = build_axis(ctx, &p->axes[0], ny, 1, ny, nx, p->pipe_layout);
            }
        }
        if (cr != CUFFT_SUCCESS) return cr;

        /* Axis 1: column FFTs — nx-point on ny columns */
        cr = build_axis(ctx, &p->axes[1], nx, ny, 1, ny, p->pipe_layout);
    }

    } /* end C2C */
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
         * batch_offset = iy * nz + ix * ny*nz
         * Data is flat: batch b at offset b*nz, so four-step works
         * with total_batch = ny*nx. */
        {
            int fs_n1, fs_n2;
            if (nz > FOURSTEP_THRESHOLD &&
                fft_fourstep_decompose(nz, FOURSTEP_THRESHOLD,
                                        &fs_n1, &fs_n2) &&
                fs_n1 > 1) {
                CUVK_LOG("[cufft] 3D axis0: four-step %d = %d x %d\n",
                         nz, fs_n1, fs_n2);
                cr = build_fourstep_axis(ctx, &p->axes[0], nz, fs_n1, fs_n2,
                                          ny * nx, p->pipe_layout);
            } else {
                cr = build_axis2(ctx, &p->axes[0], nz, 1, nz, ny, slice, nx,
                                  p->pipe_layout);
            }
        }
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
    if (!plan || !n || rank < 1 || batch < 1)
        return CUFFT_INVALID_VALUE;

    /* Rank 1: delegate to cufftPlan1d */
    if (rank == 1) {
        (void)inembed; (void)istride; (void)idist;
        (void)onembed; (void)ostride; (void)odist;
        return cufftPlan1d(plan, n[0], type, batch);
    }

    /* Rank 2: batch=1 delegates to optimized cufftPlan2d */
    if (rank == 2 && batch == 1) {
        (void)inembed; (void)istride; (void)idist;
        (void)onembed; (void)ostride; (void)odist;
        return cufftPlan2d(plan, n[0], n[1], type);
    }

    /* Rank 2, batch > 1: per-axis approach with batch_count2 */
    if (rank == 2) {
        int nx = n[0], ny = n[1];
        if (nx <= 0 || ny <= 0) return CUFFT_INVALID_VALUE;
        if (type != CUFFT_C2C && type != CUFFT_R2C && type != CUFFT_C2R)
            return CUFFT_INVALID_TYPE;

        (void)inembed; (void)istride; (void)idist;
        (void)onembed; (void)ostride; (void)odist;

        CufftPlan *p;
        int handle;
        cufftResult cr = plan_init(&p, &handle, type, batch);
        if (cr != CUFFT_SUCCESS) return cr;

        struct CUctx_st *ctx = p->ctx;
        p->rank = 2;
        p->dims[0] = nx; p->dims[1] = ny;

        if (type == CUFFT_C2C) {
            p->total_elements = nx * ny;
            p->n_axes = 2;

            /* Axis 0: row FFTs — ny-point on nx rows, batch batches */
            cr = build_axis2(ctx, &p->axes[0], ny, 1, ny, nx,
                              nx * ny, batch, p->pipe_layout);
            if (cr != CUFFT_SUCCESS) return cr;

            /* Axis 1: column FFTs — nx-point on ny columns, batch batches */
            cr = build_axis2(ctx, &p->axes[1], nx, ny, 1, ny,
                              nx * ny, batch, p->pipe_layout);
            if (cr != CUFFT_SUCCESS) return cr;
        } else {
            /* R2C / C2R */
            if (ny < 2 || ny % 2 != 0) return CUFFT_INVALID_SIZE;
            int half_y = ny / 2;
            int padded_y = half_y + 1;

            p->r2c_n = ny;
            p->total_elements = nx * padded_y;
            p->n_axes = 2;

            /* Axis 0: half_y-point C2C on contiguous rows, batch batches */
            cr = build_axis2(ctx, &p->axes[0], half_y, 1, half_y, nx,
                              nx * half_y, batch, p->pipe_layout);
            if (cr != CUFFT_SUCCESS) return cr;

            /* R2C post-processing pipeline */
            {
                char *wgsl = gen_fft_r2c_postprocess(ny, FFT_WORKGROUP_SIZE);
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
                p->r2c_dispatch_x = ((uint32_t)(half_y + 1) +
                                      FFT_WORKGROUP_SIZE - 1) / FFT_WORKGROUP_SIZE;
            }

            /* C2R pre-processing pipeline */
            {
                char *wgsl = gen_fft_c2r_preprocess(ny, FFT_WORKGROUP_SIZE);
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

            /* Axis 1: nx-point C2C on padded complex layout, batch batches */
            cr = build_axis2(ctx, &p->axes[1], nx, padded_y, 1, padded_y,
                              nx * padded_y, batch, p->pipe_layout);
            if (cr != CUFFT_SUCCESS) return cr;
        }

        cr = alloc_plan_resources(p);
        if (cr != CUFFT_SUCCESS) return cr;

        p->valid = 1;
        *plan = handle;
        CUVK_LOG("[cufft] cufftPlanMany: rank=2 %dx%d batch=%d type=0x%x SUCCESS handle=%d\n",
                 nx, ny, batch, type, handle);
        return CUFFT_SUCCESS;
    }

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
    VkDeviceAddress src_bda = alloc_in->device_addr;
    VkDeviceAddress dst_bda = alloc_out->device_addr;

    cuvk_fft_flush(ctx);
    g_cuvk.vk.vkDeviceWaitIdle(ctx->device);
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
                                 src_bda, src_bda, 1, p->n_axes);
        if (cr != CUFFT_SUCCESS) return cr;

        g_cuvk.vk.vkResetFences(ctx->device, 1, &p->fence);
        vr = g_cuvk.vk.vkQueueSubmit(ctx->compute_queue, 1, &submit, p->fence);
        if (vr != VK_SUCCESS) return CUFFT_EXEC_FAILED;
        vr = g_cuvk.vk.vkWaitForFences(ctx->device, 1, &p->fence, VK_TRUE, UINT64_MAX);
        if (vr != VK_SUCCESS) return CUFFT_EXEC_FAILED;

        g_cuvk.vk.vkResetCommandBuffer(p->cb_inv, 0);
    }

    /* Phase 2: C2R on axis 0 (pre-process + C2C inverse stages) -> dst_buf */
    cr = record_c2r_cb(p, p->cb_inv, src_buf, dst_buf, src_bda, dst_bda);
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

/* ============================================================================
 * WorkPackage API (cufft_vk.h)
 * ============================================================================ */

/* cufft.h types are already defined above — prevent cufft_vk.h from
   re-including them. */
#define CUVK_CUFFT_H
#include "cufft_vk.h"

static void wp_grow_cmds(CuvkWorkPackage *wp) {
    if (wp->n_cmds < wp->cmd_cap) return;
    int new_cap = wp->cmd_cap ? wp->cmd_cap * 2 : 64;
    wp->cmds = (CuvkWorkCmd *)realloc(wp->cmds, (size_t)new_cap * sizeof(CuvkWorkCmd));
    wp->cmd_cap = new_cap;
}

static void wp_grow_bufs(CuvkWorkPackage *wp) {
    if (wp->n_bufs < wp->buf_cap) return;
    int new_cap = wp->buf_cap ? wp->buf_cap * 2 : 16;
    wp->bufs = (CuvkWorkBufRef *)realloc(wp->bufs, (size_t)new_cap * sizeof(CuvkWorkBufRef));
    wp->buf_cap = new_cap;
}

cufftResult cuvk_wp_init(CuvkWorkPackage *wp, struct CUctx_st *ctx) {
    memset(wp, 0, sizeof(*wp));
    wp->ctx = ctx;
    return CUFFT_SUCCESS;
}

void cuvk_wp_destroy(CuvkWorkPackage *wp) {
    free(wp->cmds);
    free(wp->bufs);
    memset(wp, 0, sizeof(*wp));
}

void cuvk_wp_clear(CuvkWorkPackage *wp) {
    wp->n_cmds = 0;
    wp->n_bufs = 0;
    wp->sealed = 0;
}

void cuvk_wp_barrier(CuvkWorkPackage *wp,
                     VkPipelineStageFlags src_stage,
                     VkPipelineStageFlags dst_stage,
                     VkAccessFlags src_access,
                     VkAccessFlags dst_access) {
    wp_grow_cmds(wp);
    CuvkWorkCmd *c = &wp->cmds[wp->n_cmds++];
    c->type = CUVK_WORK_BARRIER;
    c->barrier.src_stage = src_stage;
    c->barrier.dst_stage = dst_stage;
    c->barrier.src_access = src_access;
    c->barrier.dst_access = dst_access;
}

void cuvk_wp_copy(CuvkWorkPackage *wp,
                  VkBuffer src, VkBuffer dst,
                  VkDeviceSize src_off, VkDeviceSize dst_off,
                  VkDeviceSize size) {
    wp_grow_cmds(wp);
    CuvkWorkCmd *c = &wp->cmds[wp->n_cmds++];
    c->type = CUVK_WORK_COPY;
    c->copy.src = src;
    c->copy.dst = dst;
    c->copy.src_offset = src_off;
    c->copy.dst_offset = dst_off;
    c->copy.size = size;
}

void cuvk_wp_dispatch(CuvkWorkPackage *wp,
                      VkPipeline pipeline, VkPipelineLayout layout,
                      const void *push_data, uint32_t push_size,
                      uint32_t gx, uint32_t gy, uint32_t gz) {
    wp_grow_cmds(wp);
    CuvkWorkCmd *c = &wp->cmds[wp->n_cmds++];
    c->type = CUVK_WORK_DISPATCH;
    c->dispatch.pipeline = pipeline;
    c->dispatch.layout = layout;
    if (push_size > 128) push_size = 128;
    memcpy(c->dispatch.push_data, push_data, push_size);
    c->dispatch.push_size = push_size;
    c->dispatch.group_count_x = gx;
    c->dispatch.group_count_y = gy;
    c->dispatch.group_count_z = gz;
}

void cuvk_wp_ref_buf(CuvkWorkPackage *wp,
                     VkBuffer buf, VkDeviceAddress bda, uint32_t access) {
    /* Deduplicate */
    for (int i = 0; i < wp->n_bufs; i++) {
        if (wp->bufs[i].buffer == buf) {
            wp->bufs[i].access |= access;
            return;
        }
    }
    wp_grow_bufs(wp);
    CuvkWorkBufRef *b = &wp->bufs[wp->n_bufs++];
    b->buffer = buf;
    b->bda = bda;
    b->access = access;
}

void cuvk_wp_encode(const CuvkWorkPackage *wp, VkCommandBuffer cb) {
    for (int i = 0; i < wp->n_cmds; i++) {
        const CuvkWorkCmd *c = &wp->cmds[i];
        switch (c->type) {
        case CUVK_WORK_BARRIER: {
            VkMemoryBarrier bar = {0};
            bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            bar.srcAccessMask = c->barrier.src_access;
            bar.dstAccessMask = c->barrier.dst_access;
            g_cuvk.vk.vkCmdPipelineBarrier(cb,
                c->barrier.src_stage, c->barrier.dst_stage,
                0, 1, &bar, 0, NULL, 0, NULL);
            break;
        }
        case CUVK_WORK_COPY: {
            VkBufferCopy region = {0};
            region.srcOffset = c->copy.src_offset;
            region.dstOffset = c->copy.dst_offset;
            region.size = c->copy.size;
            g_cuvk.vk.vkCmdCopyBuffer(cb, c->copy.src, c->copy.dst, 1, &region);
            break;
        }
        case CUVK_WORK_DISPATCH:
            g_cuvk.vk.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                         c->dispatch.pipeline);
            g_cuvk.vk.vkCmdPushConstants(cb, c->dispatch.layout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0,
                c->dispatch.push_size, c->dispatch.push_data);
            g_cuvk.vk.vkCmdDispatch(cb,
                c->dispatch.group_count_x,
                c->dispatch.group_count_y,
                c->dispatch.group_count_z);
            break;
        }
    }
}

cufftResult cuvk_wp_seal(CuvkWorkPackage *wp) {
    (void)wp;
    return CUFFT_SUCCESS;
}

cufftResult cuvk_wp_submit(CuvkWorkPackage *wp) {
    (void)wp;
    return CUFFT_NOT_IMPLEMENTED;
}

cufftResult vkCufftExecC2C(CuvkWorkPackage *wp, cufftHandle plan_handle,
                            VkBuffer idata_buf, VkDeviceAddress idata_bda,
                            VkBuffer odata_buf, VkDeviceAddress odata_bda,
                            int direction)
{
    if (plan_handle < 0 || plan_handle >= MAX_CUFFT_PLANS)
        return CUFFT_INVALID_PLAN;
    CufftPlan *p = &g_cufft_plans[plan_handle];
    if (!p->valid) return CUFFT_INVALID_PLAN;

    int dir_idx = (direction == CUFFT_FORWARD) ? 0 : 1;
    int inplace = (idata_buf == odata_buf);

    VkBuffer src_buf = inplace ? odata_buf : idata_buf;
    VkBuffer dst_buf = odata_buf;
    VkDeviceAddress src_bda = inplace ? odata_bda : idata_bda;
    VkDeviceAddress dst_bda = odata_bda;

    VkDeviceSize buf_size = (VkDeviceSize)p->total_elements * 2 * sizeof(float)
                          * (VkDeviceSize)p->batch;

    /* Count total stages */
    int total_stages = 0;
    for (int a = 0; a < p->n_axes; a++)
        total_stages += p->axes[a].n_stages;

    /* Compute ping-pong buffer assignments (same logic as record_fft_cb_range) */
    VkBuffer read_bufs[MAX_FFT_STAGES * MAX_AXES];
    VkBuffer write_bufs[MAX_FFT_STAGES * MAX_AXES];
    VkDeviceAddress read_bdas[MAX_FFT_STAGES * MAX_AXES];
    VkDeviceAddress write_bdas[MAX_FFT_STAGES * MAX_AXES];
    int need_final_copy = 0;

    if (total_stages == 1 && inplace) {
        read_bufs[0] = dst_buf;        read_bdas[0] = dst_bda;
        write_bufs[0] = p->scratch_buf; write_bdas[0] = p->scratch_bda;
        need_final_copy = 1;
    } else if (inplace) {
        read_bufs[0] = dst_buf;        read_bdas[0] = dst_bda;
        write_bufs[0] = p->scratch_buf; write_bdas[0] = p->scratch_bda;
        for (int i = 1; i < total_stages; i++) {
            read_bufs[i] = write_bufs[i - 1];
            read_bdas[i] = write_bdas[i - 1];
            if (read_bufs[i] == p->scratch_buf) {
                write_bufs[i] = dst_buf; write_bdas[i] = dst_bda;
            } else {
                write_bufs[i] = p->scratch_buf; write_bdas[i] = p->scratch_bda;
            }
        }
        if (write_bufs[total_stages - 1] != dst_buf)
            need_final_copy = 1;
    } else {
        write_bufs[total_stages - 1] = dst_buf;
        write_bdas[total_stages - 1] = dst_bda;
        for (int i = total_stages - 2; i >= 0; i--) {
            if (write_bufs[i + 1] == dst_buf) {
                write_bufs[i] = p->scratch_buf; write_bdas[i] = p->scratch_bda;
            } else {
                write_bufs[i] = dst_buf; write_bdas[i] = dst_bda;
            }
        }
        read_bufs[0] = src_buf;  read_bdas[0] = src_bda;
        for (int i = 1; i < total_stages; i++) {
            read_bufs[i] = write_bufs[i - 1];
            read_bdas[i] = write_bdas[i - 1];
        }
    }

    /* Initial barrier */
    cuvk_wp_barrier(wp,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    /* Track buffer refs */
    cuvk_wp_ref_buf(wp, dst_buf, dst_bda, CUVK_BUF_READ | CUVK_BUF_WRITE);
    if (!inplace)
        cuvk_wp_ref_buf(wp, src_buf, src_bda, CUVK_BUF_READ);
    if (p->scratch_buf)
        cuvk_wp_ref_buf(wp, p->scratch_buf, p->scratch_bda, CUVK_BUF_READ | CUVK_BUF_WRITE);

    /* Emit all stages */
    int global_stage = 0;
    for (int a = 0; a < p->n_axes; a++) {
        FftAxis *axis = &p->axes[a];
        for (int s = 0; s < axis->n_stages; s++) {
            if (global_stage > 0)
                cuvk_wp_barrier(wp,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_SHADER_WRITE_BIT,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

            uint64_t pc[3];
            pc[0] = read_bdas[global_stage];
            pc[1] = write_bdas[global_stage];
            VkDeviceAddress stage_lut = axis->stage_info[s].lut_bda[dir_idx];
            pc[2] = stage_lut;
            int npc = stage_lut ? 3 : 2;

            uint32_t dy = axis->stage_info[s].dispatch_y;
            if (dy == 0) dy = (uint32_t)axis->batch_count;
            uint32_t dz = axis->batch_count2 > 0 ? (uint32_t)axis->batch_count2 : 1;

            cuvk_wp_dispatch(wp,
                axis->pipelines[dir_idx][s], p->pipe_layout,
                pc, (uint32_t)(npc * 8),
                axis->dispatch_x[s], dy, dz);

            if (axis->stage_info[s].lut_buf[dir_idx])
                cuvk_wp_ref_buf(wp, axis->stage_info[s].lut_buf[dir_idx],
                                stage_lut, CUVK_BUF_READ);

            global_stage++;
        }
    }

    /* Final copy if result in scratch */
    if (need_final_copy) {
        cuvk_wp_barrier(wp,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT);
        cuvk_wp_copy(wp, p->scratch_buf, dst_buf, 0, 0, buf_size);
    }

    return CUFFT_SUCCESS;
}

cufftResult vkCufftExecR2C(CuvkWorkPackage *wp, cufftHandle plan_handle,
                            VkBuffer idata_buf, VkDeviceAddress idata_bda,
                            VkBuffer odata_buf, VkDeviceAddress odata_bda) {
    if (plan_handle < 0 || plan_handle >= MAX_CUFFT_PLANS)
        return CUFFT_INVALID_PLAN;
    CufftPlan *p = &g_cufft_plans[plan_handle];
    if (!p->valid || (!p->r2c_post_pipeline && !p->r2c_fused))
        return CUFFT_INVALID_PLAN;

    /* Fused 2-dispatch R2C path */
    if (p->r2c_fused) {
        FftAxis *faxis = &p->r2c_fused_axis;
        int ns = faxis->n_stages;

        VkDeviceAddress rd[MAX_FFT_STAGES], wr[MAX_FFT_STAGES];
        wr[ns - 1] = odata_bda;
        for (int i = ns - 2; i >= 0; i--)
            wr[i] = (wr[i + 1] == odata_bda) ? p->scratch_bda : odata_bda;
        rd[0] = idata_bda;
        for (int i = 1; i < ns; i++) rd[i] = wr[i - 1];

        cuvk_wp_ref_buf(wp, odata_buf, odata_bda, CUVK_BUF_READ | CUVK_BUF_WRITE);
        if (idata_buf != odata_buf)
            cuvk_wp_ref_buf(wp, idata_buf, idata_bda, CUVK_BUF_READ);
        if (p->scratch_buf)
            cuvk_wp_ref_buf(wp, p->scratch_buf, p->scratch_bda,
                             CUVK_BUF_READ | CUVK_BUF_WRITE);

        cuvk_wp_barrier(wp,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

        for (int s = 0; s < ns; s++) {
            if (s > 0)
                cuvk_wp_barrier(wp,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_SHADER_WRITE_BIT,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

            uint64_t pc[3];
            pc[0] = rd[s]; pc[1] = wr[s];
            VkDeviceAddress slut = faxis->stage_info[s].lut_bda[0];
            pc[2] = slut;
            int npc = slut ? 3 : 2;
            uint32_t dy = faxis->stage_info[s].dispatch_y;
            if (dy == 0) dy = 1;

            cuvk_wp_dispatch(wp, faxis->pipelines[0][s], p->pipe_layout,
                             pc, (uint32_t)(npc * 8),
                             faxis->dispatch_x[s], dy, 1);

            if (faxis->stage_info[s].lut_buf[0])
                cuvk_wp_ref_buf(wp, faxis->stage_info[s].lut_buf[0],
                                slut, CUVK_BUF_READ);
        }
        return CUFFT_SUCCESS;
    }

    /* Non-fused R2C path */
    FftAxis *axis0 = &p->axes[0];
    int ns = axis0->n_stages;

    /* Phase 1: C2C stages on axis 0, ping-pong between src and scratch */
    VkDeviceAddress r2c_read[MAX_FFT_STAGES + 1];
    VkDeviceAddress r2c_write[MAX_FFT_STAGES + 1];

    if (ns == 0) {
        r2c_read[0] = idata_bda;
        r2c_write[0] = odata_bda;
    } else {
        r2c_read[0] = idata_bda;
        r2c_write[0] = p->scratch_bda;
        for (int i = 1; i < ns; i++) {
            r2c_read[i] = r2c_write[i - 1];
            r2c_write[i] = (r2c_read[i] == p->scratch_bda)
                           ? idata_bda : p->scratch_bda;
        }
        r2c_read[ns] = r2c_write[ns - 1];
        r2c_write[ns] = odata_bda;
    }

    /* Track buffer refs */
    cuvk_wp_ref_buf(wp, odata_buf, odata_bda, CUVK_BUF_READ | CUVK_BUF_WRITE);
    if (idata_buf != odata_buf)
        cuvk_wp_ref_buf(wp, idata_buf, idata_bda, CUVK_BUF_READ);
    if (p->scratch_buf)
        cuvk_wp_ref_buf(wp, p->scratch_buf, p->scratch_bda,
                         CUVK_BUF_READ | CUVK_BUF_WRITE);

    /* Initial barrier */
    cuvk_wp_barrier(wp,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    /* C2C stages */
    for (int s = 0; s < ns; s++) {
        if (s > 0)
            cuvk_wp_barrier(wp,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

        uint64_t pc[3];
        pc[0] = r2c_read[s];
        pc[1] = r2c_write[s];
        VkDeviceAddress slut = axis0->stage_info[s].lut_bda[0];
        pc[2] = slut;
        int npc = slut ? 3 : 2;

        uint32_t dy = axis0->stage_info[s].dispatch_y;
        if (dy == 0) dy = (uint32_t)axis0->batch_count;
        uint32_t dz = axis0->batch_count2 > 0 ? (uint32_t)axis0->batch_count2 : 1;
        cuvk_wp_dispatch(wp, axis0->pipelines[0][s], p->pipe_layout,
                         pc, (uint32_t)(npc * 8), axis0->dispatch_x[s], dy, dz);
    }

    /* R2C post-process */
    cuvk_wp_barrier(wp,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    {
        uint64_t pc[3];
        pc[0] = r2c_read[ns];
        pc[1] = r2c_write[ns];
        pc[2] = 0;
        uint32_t total_batches = (uint32_t)axis0->batch_count *
            (uint32_t)(axis0->batch_count2 > 0 ? axis0->batch_count2 : 1);
        cuvk_wp_dispatch(wp, p->r2c_post_pipeline, p->pipe_layout,
                         pc, 16, p->r2c_dispatch_x, total_batches, 1);
    }

    /* Phase 2: C2C on remaining axes (in-place on odata_buf) */
    if (p->n_axes > 1) {
        int phase2_stages = 0;
        for (int a = 1; a < p->n_axes; a++)
            phase2_stages += p->axes[a].n_stages;

        VkDeviceAddress p2_read[MAX_FFT_STAGES * MAX_AXES];
        VkDeviceAddress p2_write[MAX_FFT_STAGES * MAX_AXES];
        int need_final_copy = 0;

        if (phase2_stages == 1) {
            p2_read[0] = odata_bda;
            p2_write[0] = p->scratch_bda;
            need_final_copy = 1;
        } else {
            p2_read[0] = odata_bda;
            p2_write[0] = p->scratch_bda;
            for (int i = 1; i < phase2_stages; i++) {
                p2_read[i] = p2_write[i - 1];
                p2_write[i] = (p2_read[i] == p->scratch_bda)
                             ? odata_bda : p->scratch_bda;
            }
            if (p2_write[phase2_stages - 1] != odata_bda)
                need_final_copy = 1;
        }

        int gs = 0;
        for (int a = 1; a < p->n_axes; a++) {
            FftAxis *axis = &p->axes[a];
            for (int s = 0; s < axis->n_stages; s++) {
                cuvk_wp_barrier(wp,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_SHADER_WRITE_BIT,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

                uint64_t pc[3];
                pc[0] = p2_read[gs];
                pc[1] = p2_write[gs];
                VkDeviceAddress lut = axis->stage_info[s].lut_bda[0];
                pc[2] = lut;
                int npc = lut ? 3 : 2;

                uint32_t dy = axis->stage_info[s].dispatch_y;
                if (dy == 0) dy = (uint32_t)axis->batch_count;
                uint32_t dz = axis->batch_count2 > 0
                            ? (uint32_t)axis->batch_count2 : 1;
                cuvk_wp_dispatch(wp, axis->pipelines[0][s], p->pipe_layout,
                                 pc, (uint32_t)(npc * 8),
                                 axis->dispatch_x[s], dy, dz);

                if (axis->stage_info[s].lut_buf[0])
                    cuvk_wp_ref_buf(wp, axis->stage_info[s].lut_buf[0],
                                    lut, CUVK_BUF_READ);
                gs++;
            }
        }

        if (need_final_copy) {
            VkDeviceSize buf_size = (VkDeviceSize)p->total_elements * 2 *
                                    sizeof(float) * (VkDeviceSize)p->batch;
            cuvk_wp_barrier(wp,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_TRANSFER_READ_BIT);
            cuvk_wp_copy(wp, p->scratch_buf, odata_buf, 0, 0, buf_size);
        }
    }

    return CUFFT_SUCCESS;
}

cufftResult vkCufftExecC2R(CuvkWorkPackage *wp, cufftHandle plan_handle,
                            VkBuffer idata_buf, VkDeviceAddress idata_bda,
                            VkBuffer odata_buf, VkDeviceAddress odata_bda) {
    if (plan_handle < 0 || plan_handle >= MAX_CUFFT_PLANS)
        return CUFFT_INVALID_PLAN;
    CufftPlan *p = &g_cufft_plans[plan_handle];
    if (!p->valid || !p->c2r_pre_pipeline) return CUFFT_INVALID_PLAN;

    /* Track buffer refs */
    cuvk_wp_ref_buf(wp, idata_buf, idata_bda, CUVK_BUF_READ | CUVK_BUF_WRITE);
    if (odata_buf != idata_buf)
        cuvk_wp_ref_buf(wp, odata_buf, odata_bda, CUVK_BUF_WRITE);
    if (p->scratch_buf)
        cuvk_wp_ref_buf(wp, p->scratch_buf, p->scratch_bda,
                         CUVK_BUF_READ | CUVK_BUF_WRITE);

    /* Initial barrier */
    cuvk_wp_barrier(wp,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    /* Phase 1: C2C inverse on remaining axes (in-place on idata_buf) */
    if (p->n_axes > 1) {
        int phase1_stages = 0;
        for (int a = 1; a < p->n_axes; a++)
            phase1_stages += p->axes[a].n_stages;

        VkDeviceAddress p1_read[MAX_FFT_STAGES * MAX_AXES];
        VkDeviceAddress p1_write[MAX_FFT_STAGES * MAX_AXES];
        int need_copy = 0;

        if (phase1_stages == 1) {
            p1_read[0] = idata_bda;
            p1_write[0] = p->scratch_bda;
            need_copy = 1;
        } else {
            p1_read[0] = idata_bda;
            p1_write[0] = p->scratch_bda;
            for (int i = 1; i < phase1_stages; i++) {
                p1_read[i] = p1_write[i - 1];
                p1_write[i] = (p1_read[i] == p->scratch_bda)
                             ? idata_bda : p->scratch_bda;
            }
            if (p1_write[phase1_stages - 1] != idata_bda)
                need_copy = 1;
        }

        int gs = 0;
        for (int a = 1; a < p->n_axes; a++) {
            FftAxis *axis = &p->axes[a];
            for (int s = 0; s < axis->n_stages; s++) {
                if (gs > 0)
                    cuvk_wp_barrier(wp,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_SHADER_WRITE_BIT,
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

                uint64_t pc[3];
                pc[0] = p1_read[gs];
                pc[1] = p1_write[gs];
                VkDeviceAddress lut = axis->stage_info[s].lut_bda[1];
                pc[2] = lut;
                int npc = lut ? 3 : 2;

                uint32_t dy = axis->stage_info[s].dispatch_y;
                if (dy == 0) dy = (uint32_t)axis->batch_count;
                uint32_t dz = axis->batch_count2 > 0
                            ? (uint32_t)axis->batch_count2 : 1;
                cuvk_wp_dispatch(wp, axis->pipelines[1][s], p->pipe_layout,
                                 pc, (uint32_t)(npc * 8),
                                 axis->dispatch_x[s], dy, dz);

                if (axis->stage_info[s].lut_buf[1])
                    cuvk_wp_ref_buf(wp, axis->stage_info[s].lut_buf[1],
                                    lut, CUVK_BUF_READ);
                gs++;
            }
        }

        if (need_copy) {
            VkDeviceSize buf_size = (VkDeviceSize)p->total_elements * 2 *
                                    sizeof(float) * (VkDeviceSize)p->batch;
            cuvk_wp_barrier(wp,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_TRANSFER_READ_BIT);
            cuvk_wp_copy(wp, p->scratch_buf, idata_buf, 0, 0, buf_size);
        }
    }

    /* Phase 2: C2R on axis 0 (pre-process + C2C inverse stages) */
    FftAxis *axis0 = &p->axes[0];
    int half = p->r2c_n / 2;
    int ns = axis0->n_stages;
    int total = 1 + ns;

    VkDeviceAddress c2r_read[MAX_FFT_STAGES + 1];
    VkDeviceAddress c2r_write[MAX_FFT_STAGES + 1];
    int need_final_copy = 0;

    if (ns == 0) {
        c2r_read[0] = idata_bda;
        c2r_write[0] = odata_bda;
    } else {
        c2r_read[0] = idata_bda;
        c2r_write[0] = p->scratch_bda;
        c2r_read[1] = p->scratch_bda;
        c2r_write[1] = odata_bda;
        for (int i = 2; i < total; i++) {
            c2r_read[i] = c2r_write[i - 1];
            c2r_write[i] = (c2r_read[i] == odata_bda)
                           ? p->scratch_bda : odata_bda;
        }
        if (c2r_write[total - 1] != odata_bda)
            need_final_copy = 1;
    }

    /* C2R pre-process */
    cuvk_wp_barrier(wp,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    {
        uint64_t pc[3];
        pc[0] = c2r_read[0];
        pc[1] = c2r_write[0];
        pc[2] = 0;
        uint32_t total_batches = (uint32_t)axis0->batch_count *
            (uint32_t)(axis0->batch_count2 > 0 ? axis0->batch_count2 : 1);
        cuvk_wp_dispatch(wp, p->c2r_pre_pipeline, p->pipe_layout,
                         pc, 16,
                         ((uint32_t)half + FFT_WORKGROUP_SIZE - 1) /
                             FFT_WORKGROUP_SIZE,
                         total_batches, 1);
    }

    /* C2C inverse stages */
    for (int s = 0; s < ns; s++) {
        cuvk_wp_barrier(wp,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

        uint64_t pc[3];
        pc[0] = c2r_read[1 + s];
        pc[1] = c2r_write[1 + s];
        VkDeviceAddress slut = axis0->stage_info[s].lut_bda[1];
        pc[2] = slut;
        int npc = slut ? 3 : 2;

        uint32_t dy = axis0->stage_info[s].dispatch_y;
        if (dy == 0) dy = (uint32_t)axis0->batch_count;
        uint32_t dz = axis0->batch_count2 > 0 ? (uint32_t)axis0->batch_count2 : 1;
        cuvk_wp_dispatch(wp, axis0->pipelines[1][s], p->pipe_layout,
                         pc, (uint32_t)(npc * 8), axis0->dispatch_x[s], dy, dz);
    }

    if (need_final_copy) {
        VkDeviceSize c2c_size = (VkDeviceSize)half * 2 * sizeof(float) *
            (VkDeviceSize)axis0->batch_count *
            (VkDeviceSize)(axis0->batch_count2 > 0 ? axis0->batch_count2 : 1);
        cuvk_wp_barrier(wp,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT);
        cuvk_wp_copy(wp, p->scratch_buf, odata_buf, 0, 0, c2c_size);
    }

    return CUFFT_SUCCESS;
}
