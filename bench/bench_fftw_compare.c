/*
 * bench_fftw_compare.c - Compare fused FFT (Vulkan/llvmpipe) vs FFTW
 *
 * Runs both engines on the same sizes and prints a comparison table.
 * Build: linked against fft_fused_gen + wgsl_compiler + Vulkan + fftw3f.
 * Usage: VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json ./bench_fftw_compare
 */

#include "fft_fused_gen.h"
#include "simple_wgsl.h"
#include "bench_vk_common.h"

#include <fftw3.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define WARMUP 3
#define ITERS  10

/* ================================================================ */
/* Helpers                                                          */
/* ================================================================ */

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static double median_d(double *arr, int n) {
    qsort(arr, (size_t)n, sizeof(double), cmp_double);
    if (n % 2 == 0) return (arr[n/2 - 1] + arr[n/2]) / 2.0;
    return arr[n/2];
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ================================================================ */
/* GPU buffer helpers                                               */
/* ================================================================ */

typedef struct {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceSize size;
    void *mapped;
} GpuBuffer;

static int create_buffer(VkCtx *ctx, VkDeviceSize size,
                         VkBufferUsageFlags usage,
                         VkMemoryPropertyFlags mem_flags, GpuBuffer *out) {
    memset(out, 0, sizeof(*out));
    out->size = size;
    VkBufferCreateInfo bci = {0};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(ctx->device, &bci, NULL, &out->buffer) != VK_SUCCESS)
        return -1;
    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(ctx->device, out->buffer, &reqs);
    int32_t mt = find_memory_type(&ctx->mem_props, reqs.memoryTypeBits, mem_flags);
    if (mt < 0) return -1;
    VkMemoryAllocateInfo ai = {0};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = reqs.size;
    ai.memoryTypeIndex = (uint32_t)mt;
    if (vkAllocateMemory(ctx->device, &ai, NULL, &out->memory) != VK_SUCCESS)
        return -1;
    vkBindBufferMemory(ctx->device, out->buffer, out->memory, 0);
    if (mem_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        vkMapMemory(ctx->device, out->memory, 0, size, 0, &out->mapped);
    return 0;
}

static void destroy_buffer(VkCtx *ctx, GpuBuffer *b) {
    if (b->mapped) vkUnmapMemory(ctx->device, b->memory);
    vkFreeMemory(ctx->device, b->memory, NULL);
    vkDestroyBuffer(ctx->device, b->buffer, NULL);
}

/* ================================================================ */
/* WGSL -> SPIR-V                                                   */
/* ================================================================ */

static int compile_wgsl(const char *src, uint32_t **out, size_t *out_count) {
    WgslAstNode *ast = wgsl_parse(src);
    if (!ast) return -1;
    WgslResolver *res = wgsl_resolver_build(ast);
    if (!res) { wgsl_free_ast(ast); return -1; }
    WgslLowerOptions opts = {0};
    opts.spirv_version = 0x00010300;
    opts.env = WGSL_LOWER_ENV_VULKAN_1_1;
    opts.packing = WGSL_LOWER_PACK_STD430;
    WgslLowerResult lr = wgsl_lower_emit_spirv(ast, res, &opts, out, out_count);
    wgsl_resolver_free(res);
    wgsl_free_ast(ast);
    return lr == WGSL_LOWER_OK ? 0 : -1;
}

/* ================================================================ */
/* Fused FFT bench (Vulkan)                                         */
/* ================================================================ */

static double bench_fused_vk(VkCtx *ctx, int n, int batch, float *out_err) {
    *out_err = -1.0f;

    char *wgsl = gen_fft_fused(n, 1, 0, 0);
    if (!wgsl) return -1;
    uint32_t *spirv = NULL; size_t sc = 0;
    if (compile_wgsl(wgsl, &spirv, &sc) != 0) { free(wgsl); return -1; }
    free(wgsl);

    int lut_count = fft_fused_lut_size(n, 1, 0);
    int num_bindings = (lut_count > 0) ? 3 : 2;

    /* Shader + pipeline */
    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = sc * sizeof(uint32_t);
    smci.pCode = spirv;
    VkShaderModule shader;
    if (vkCreateShaderModule(ctx->device, &smci, NULL, &shader) != VK_SUCCESS) {
        wgsl_lower_free(spirv); return -1;
    }
    wgsl_lower_free(spirv);

    VkDescriptorSetLayoutBinding bindings[3] = {0};
    for (int i = 0; i < num_bindings; i++) {
        bindings[i].binding = (uint32_t)i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {0};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = (uint32_t)num_bindings;
    dslci.pBindings = bindings;
    VkDescriptorSetLayout desc_layout;
    vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL, &desc_layout);

    VkPipelineLayoutCreateInfo plci = {0};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1; plci.pSetLayouts = &desc_layout;
    VkPipelineLayout pipe_layout;
    vkCreatePipelineLayout(ctx->device, &plci, NULL, &pipe_layout);

    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = shader;
    cpci.stage.pName = "main";
    cpci.layout = pipe_layout;
    VkPipeline pipeline;
    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci,
                                  NULL, &pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(ctx->device, shader, NULL);
        vkDestroyPipelineLayout(ctx->device, pipe_layout, NULL);
        vkDestroyDescriptorSetLayout(ctx->device, desc_layout, NULL);
        return -1;
    }

    /* Batch setup */
    int bpw = fft_fused_batch_per_wg(n, 0, 0);
    if (bpw < 1) bpw = 1;
    int padded_batch = ((batch + bpw - 1) / bpw) * bpw;
    int dispatch_count = padded_batch / bpw;

    /* Buffers — use host-visible on llvmpipe (no dedicated VRAM) */
    VkDeviceSize buf_size = (VkDeviceSize)n * padded_batch * 2 * sizeof(float);
    VkMemoryPropertyFlags mem_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkBufferUsageFlags buf_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    GpuBuffer src_buf, dst_buf, lut_buf;
    memset(&lut_buf, 0, sizeof(lut_buf));

    if (create_buffer(ctx, buf_size, buf_usage, mem_flags, &src_buf) != 0)
        goto fail_cleanup;
    if (create_buffer(ctx, buf_size, buf_usage, mem_flags, &dst_buf) != 0)
        goto fail_cleanup;

    /* Init src: impulse for correctness check */
    {
        float *src_data = (float *)src_buf.mapped;
        memset(src_data, 0, (size_t)buf_size);
        src_data[0] = 1.0f;
    }

    /* LUT */
    if (lut_count > 0) {
        float *lut_data = fft_fused_compute_lut(n, 1, 0);
        VkDeviceSize lut_bytes = (VkDeviceSize)lut_count * 2 * sizeof(float);
        if (create_buffer(ctx, lut_bytes, buf_usage, mem_flags, &lut_buf) != 0) {
            free(lut_data); goto fail_cleanup;
        }
        memcpy(lut_buf.mapped, lut_data, (size_t)lut_bytes);
        free(lut_data);
    }

    /* Descriptor set */
    VkDescriptorSet desc_set;
    {
        VkDescriptorSetAllocateInfo dsai = {0};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = ctx->desc_pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &desc_layout;
        vkAllocateDescriptorSets(ctx->device, &dsai, &desc_set);

        VkDescriptorBufferInfo dbi[3] = {0};
        dbi[0].buffer = src_buf.buffer; dbi[0].range = buf_size;
        dbi[1].buffer = dst_buf.buffer; dbi[1].range = buf_size;
        if (lut_count > 0) {
            dbi[2].buffer = lut_buf.buffer;
            dbi[2].range = (VkDeviceSize)lut_count * 2 * sizeof(float);
        }
        VkWriteDescriptorSet wds[3] = {0};
        for (int i = 0; i < num_bindings; i++) {
            wds[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wds[i].dstSet = desc_set;
            wds[i].dstBinding = (uint32_t)i;
            wds[i].descriptorCount = 1;
            wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wds[i].pBufferInfo = &dbi[i];
        }
        vkUpdateDescriptorSets(ctx->device, (uint32_t)num_bindings, wds, 0, NULL);
    }

    /* Command buffer + fence */
    VkCommandBuffer cb;
    {
        VkCommandBufferAllocateInfo cbai = {0};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = ctx->cmd_pool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        vkAllocateCommandBuffers(ctx->device, &cbai, &cb);
    }
    VkFence fence;
    {
        VkFenceCreateInfo fci = {0};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vkCreateFence(ctx->device, &fci, NULL, &fence);
    }

    VkCommandBufferBeginInfo cbbi = {0};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    /* Run once for correctness */
    vkBeginCommandBuffer(cb, &cbbi);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipe_layout, 0, 1, &desc_set, 0, NULL);
    vkCmdDispatch(cb, (uint32_t)dispatch_count, 1, 1);
    vkEndCommandBuffer(cb);

    {
        VkSubmitInfo si = {0};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        vkQueueSubmit(ctx->queue, 1, &si, fence);
        vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
        vkResetFences(ctx->device, 1, &fence);
    }

    /* Check correctness: impulse -> all 1s */
    {
        float *dst_data = (float *)dst_buf.mapped;
        float max_err = 0.0f;
        for (int i = 0; i < n; i++) {
            float err_re = fabsf(dst_data[i * 2] - 1.0f);
            float err_im = fabsf(dst_data[i * 2 + 1]);
            if (err_re > max_err) max_err = err_re;
            if (err_im > max_err) max_err = err_im;
        }
        *out_err = max_err;
    }

    /* Warmup */
    for (int w = 0; w < WARMUP; w++) {
        vkResetCommandBuffer(cb, 0);
        vkBeginCommandBuffer(cb, &cbbi);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipe_layout, 0, 1, &desc_set, 0, NULL);
        vkCmdDispatch(cb, (uint32_t)dispatch_count, 1, 1);
        vkEndCommandBuffer(cb);
        VkSubmitInfo si = {0};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        vkQueueSubmit(ctx->queue, 1, &si, fence);
        vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
        vkResetFences(ctx->device, 1, &fence);
    }

    /* Timed runs — use wall clock (more reliable on llvmpipe) */
    double times[ITERS];
    for (int it = 0; it < ITERS; it++) {
        /* Reset src data for each run */
        {
            float *src_data = (float *)src_buf.mapped;
            memset(src_data, 0, (size_t)buf_size);
            for (int b = 0; b < batch; b++)
                src_data[b * n * 2] = 1.0f;
        }

        vkResetCommandBuffer(cb, 0);
        vkBeginCommandBuffer(cb, &cbbi);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipe_layout, 0, 1, &desc_set, 0, NULL);
        vkCmdDispatch(cb, (uint32_t)dispatch_count, 1, 1);
        vkEndCommandBuffer(cb);

        VkSubmitInfo si = {0};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1; si.pCommandBuffers = &cb;

        double t0 = now_ms();
        vkQueueSubmit(ctx->queue, 1, &si, fence);
        vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
        double t1 = now_ms();
        vkResetFences(ctx->device, 1, &fence);

        times[it] = (t1 - t0) / batch;  /* ms per FFT */
    }

    double result = median_d(times, ITERS);

    /* Cleanup */
    vkDestroyFence(ctx->device, fence, NULL);
    vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cb);
    vkFreeDescriptorSets(ctx->device, ctx->desc_pool, 1, &desc_set);
    if (lut_buf.buffer) destroy_buffer(ctx, &lut_buf);
    destroy_buffer(ctx, &dst_buf);
    destroy_buffer(ctx, &src_buf);
    vkDestroyPipeline(ctx->device, pipeline, NULL);
    vkDestroyShaderModule(ctx->device, shader, NULL);
    vkDestroyPipelineLayout(ctx->device, pipe_layout, NULL);
    vkDestroyDescriptorSetLayout(ctx->device, desc_layout, NULL);
    return result;

fail_cleanup:
    if (lut_buf.buffer) destroy_buffer(ctx, &lut_buf);
    if (dst_buf.buffer) destroy_buffer(ctx, &dst_buf);
    if (src_buf.buffer) destroy_buffer(ctx, &src_buf);
    vkDestroyPipeline(ctx->device, pipeline, NULL);
    vkDestroyShaderModule(ctx->device, shader, NULL);
    vkDestroyPipelineLayout(ctx->device, pipe_layout, NULL);
    vkDestroyDescriptorSetLayout(ctx->device, desc_layout, NULL);
    return -1;
}

/* ================================================================ */
/* FFTW bench                                                       */
/* ================================================================ */

static double bench_fftw(int n, int batch) {
    /* Single-precision C2C forward */
    fftwf_complex *in  = fftwf_alloc_complex(n);
    fftwf_complex *out = fftwf_alloc_complex(n);

    fftwf_plan plan = fftwf_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_MEASURE);
    if (!plan) {
        fftwf_free(in); fftwf_free(out);
        return -1;
    }

    /* Init with impulse */
    memset(in, 0, sizeof(fftwf_complex) * (size_t)n);
    in[0][0] = 1.0f; in[0][1] = 0.0f;

    /* Warmup */
    for (int w = 0; w < WARMUP; w++)
        fftwf_execute(plan);

    /* Timed runs */
    double times[ITERS];
    for (int it = 0; it < ITERS; it++) {
        /* Re-init input each time */
        memset(in, 0, sizeof(fftwf_complex) * (size_t)n);
        in[0][0] = 1.0f;

        double t0 = now_ms();
        for (int b = 0; b < batch; b++)
            fftwf_execute(plan);
        double t1 = now_ms();
        times[it] = (t1 - t0) / batch;  /* ms per FFT */
    }

    double result = median_d(times, ITERS);

    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);
    return result;
}

/* ================================================================ */
/* Main                                                             */
/* ================================================================ */

int main(void) {
    VkCtx ctx;
    if (vk_init(&ctx, 16, 32, 1) != 0) {
        fprintf(stderr, "Vulkan init failed\n");
        return 1;
    }

    int sizes[] = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    int n_sizes = (int)(sizeof(sizes) / sizeof(sizes[0]));

    printf("\n%-8s  %12s  %12s  %8s  %8s\n",
           "N", "fused_us", "fftw_us", "ratio", "err");
    printf("-------  ------------  ------------  --------  --------\n");

    for (int i = 0; i < n_sizes; i++) {
        int n = sizes[i];
        /* Smaller batch for larger N (llvmpipe is slow) */
        int batch = (n <= 64) ? 1000 : (n <= 256) ? 500 : (n <= 1024) ? 100 : 10;

        float err = 0;
        double fused_ms = bench_fused_vk(&ctx, n, batch, &err);
        double fftw_ms  = bench_fftw(n, batch);

        double fused_us = fused_ms * 1000.0;
        double fftw_us  = fftw_ms * 1000.0;

        if (fused_ms > 0 && fftw_ms > 0) {
            printf("%-8d  %12.3f  %12.3f  %7.1fx  %8.1e\n",
                   n, fused_us, fftw_us, fused_us / fftw_us, err);
        } else if (fused_ms < 0) {
            printf("%-8d  %12s  %12.3f  %8s  %8s\n",
                   n, "FAIL", fftw_us, "-", "-");
        }
        fflush(stdout);
    }

    printf("\n");
    vk_destroy(&ctx);
    return 0;
}
