/*
 * bench_ours_quick.c - Quick GPU-timestamp benchmark for our cuFFT path
 *
 * Uses our fused FFT for N<=4096, four-step via cuFFT API for N>4096.
 * GPU timestamps for accurate measurement, no host overhead.
 */

#include "fft_fused_gen.h"
#include "fft_fourstep_gen.h"
#include "simple_wgsl.h"
#include "bench_vk_common.h"

#include <math.h>
#include <string.h>

#define WARMUP 5
#define ITERS  20

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static double median_d(double *arr, int n) {
    qsort(arr, (size_t)n, sizeof(double), cmp_double);
    if (n % 2 == 0) return (arr[n/2 - 1] + arr[n/2]) / 2.0;
    return arr[n/2];
}

typedef struct {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceSize size;
} GpuBuf;

static int create_buf(VkCtx *ctx, VkDeviceSize size, GpuBuf *out) {
    memset(out, 0, sizeof(*out));
    out->size = size;
    VkBufferCreateInfo bci = {0};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(ctx->device, &bci, NULL, &out->buffer) != VK_SUCCESS)
        return -1;
    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(ctx->device, out->buffer, &reqs);
    int32_t mt = find_memory_type(&ctx->mem_props, reqs.memoryTypeBits,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mt < 0) return -1;
    VkMemoryAllocateInfo ai = {0};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = reqs.size;
    ai.memoryTypeIndex = (uint32_t)mt;
    if (vkAllocateMemory(ctx->device, &ai, NULL, &out->memory) != VK_SUCCESS)
        return -1;
    vkBindBufferMemory(ctx->device, out->buffer, out->memory, 0);
    return 0;
}

static void destroy_buf(VkCtx *ctx, GpuBuf *b) {
    vkFreeMemory(ctx->device, b->memory, NULL);
    vkDestroyBuffer(ctx->device, b->buffer, NULL);
}

/* Compile WGSL to SPIR-V */
static int compile_wgsl(const char *src, uint32_t **out, size_t *out_count) {
    WgslParseResult pr = wgsl_parse(src);
    WgslAstNode *ast = pr.value;
    if (!ast) { wgsl_diagnostic_list_free(pr.diags); return -1; }
    WgslResolveResult rr = wgsl_resolver_build(ast);
    WgslResolver *res = rr.value;
    if (!res) { wgsl_free_ast(ast); wgsl_diagnostic_list_free(pr.diags); wgsl_diagnostic_list_free(rr.diags); return -1; }
    WgslLowerOptions opts = {0};
    opts.spirv_version = 0x00010300;
    opts.env = WGSL_LOWER_ENV_VULKAN_1_1;
    opts.packing = WGSL_LOWER_PACK_STD430;
    WgslLowerSpirvResult lsr = wgsl_lower_emit_spirv(ast, res, &opts);
    wgsl_resolver_free(res);
    wgsl_diagnostic_list_free(rr.diags);
    wgsl_free_ast(ast);
    wgsl_diagnostic_list_free(pr.diags);
    if (lsr.code != SW_OK) {
        wgsl_diagnostic_list_free(lsr.diags);
        if (lsr.words) wgsl_lower_free(lsr.words);
        return -1;
    }
    *out = lsr.words;
    *out_count = lsr.word_count;
    wgsl_diagnostic_list_free(lsr.diags);
    return 0;
}

/* ============================================================ */
/* Fused FFT bench (N <= 4096)                                  */
/* ============================================================ */

static double bench_fused(VkCtx *ctx, int n) {
    int wg = fft_fused_workgroup_size(n, 0, 0);
    int bpw = fft_fused_batch_per_wg(n, 0, 0);
    if (wg <= 0 || bpw <= 0) return -1;

    int batch = 1;
    int dispatch_count = (batch + bpw - 1) / bpw;

    char *wgsl = gen_fft_fused(n, 1, 0, 0);
    if (!wgsl) return -1;
    uint32_t *spirv = NULL; size_t sc = 0;
    if (compile_wgsl(wgsl, &spirv, &sc) != 0) { free(wgsl); return -1; }
    free(wgsl);

    /* Create pipeline with 3-binding layout */
    VkDescriptorSetLayoutBinding bindings[3] = {{0}};
    for (int i = 0; i < 3; i++) {
        bindings[i].binding = (uint32_t)i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci = {0};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 3; dslci.pBindings = bindings;
    VkDescriptorSetLayout desc_layout;
    vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL, &desc_layout);

    VkPipelineLayoutCreateInfo plci = {0};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1; plci.pSetLayouts = &desc_layout;
    VkPipelineLayout pipe_layout;
    vkCreatePipelineLayout(ctx->device, &plci, NULL, &pipe_layout);

    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = sc * sizeof(uint32_t);
    smci.pCode = spirv;
    VkShaderModule shader;
    vkCreateShaderModule(ctx->device, &smci, NULL, &shader);
    wgsl_lower_free(spirv);

    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = shader;
    cpci.stage.pName = "main";
    cpci.layout = pipe_layout;
    VkPipeline pipeline;
    vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL, &pipeline);

    /* Buffers */
    VkDeviceSize buf_bytes = (VkDeviceSize)n * 2 * sizeof(float);
    GpuBuf src_buf, dst_buf, lut_buf;
    create_buf(ctx, buf_bytes, &src_buf);
    create_buf(ctx, buf_bytes, &dst_buf);

    /* LUT */
    int lut_count = fft_fused_lut_size(n, 1, 0);
    int has_lut = 0;
    if (lut_count > 0) {
        float *lut_data = fft_fused_compute_lut(n, 1, 0);
        VkDeviceSize lut_bytes = (VkDeviceSize)lut_count * 2 * sizeof(float);
        create_buf(ctx, lut_bytes, &lut_buf);

        /* Upload LUT via staging */
        GpuBuf staging;
        VkBufferCreateInfo sbci = {0};
        sbci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        sbci.size = lut_bytes;
        sbci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        sbci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(ctx->device, &sbci, NULL, &staging.buffer);
        VkMemoryRequirements reqs;
        vkGetBufferMemoryRequirements(ctx->device, staging.buffer, &reqs);
        int32_t mt = find_memory_type(&ctx->mem_props, reqs.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkMemoryAllocateInfo mai = {0};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = reqs.size;
        mai.memoryTypeIndex = (uint32_t)mt;
        vkAllocateMemory(ctx->device, &mai, NULL, &staging.memory);
        vkBindBufferMemory(ctx->device, staging.buffer, staging.memory, 0);
        void *mapped;
        vkMapMemory(ctx->device, staging.memory, 0, lut_bytes, 0, &mapped);
        memcpy(mapped, lut_data, (size_t)lut_bytes);
        vkUnmapMemory(ctx->device, staging.memory);
        free(lut_data);

        VkCommandBuffer cmd;
        VkCommandBufferAllocateInfo cbai = {0};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = ctx->cmd_pool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        vkAllocateCommandBuffers(ctx->device, &cbai, &cmd);
        VkCommandBufferBeginInfo bi = {0};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bi);
        VkBufferCopy region = {0, 0, lut_bytes};
        vkCmdCopyBuffer(cmd, staging.buffer, lut_buf.buffer, 1, &region);
        vkEndCommandBuffer(cmd);
        VkFence fence;
        VkFenceCreateInfo fci = {0};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vkCreateFence(ctx->device, &fci, NULL, &fence);
        VkSubmitInfo si = {0};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        vkQueueSubmit(ctx->queue, 1, &si, fence);
        vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
        vkDestroyFence(ctx->device, fence, NULL);
        vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cmd);
        vkFreeMemory(ctx->device, staging.memory, NULL);
        vkDestroyBuffer(ctx->device, staging.buffer, NULL);

        has_lut = 1;
    } else {
        create_buf(ctx, 64, &lut_buf);  /* dummy */
    }

    /* Descriptor set */
    VkDescriptorSet desc_set;
    VkDescriptorSetAllocateInfo dsai = {0};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = ctx->desc_pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &desc_layout;
    vkAllocateDescriptorSets(ctx->device, &dsai, &desc_set);

    VkDescriptorBufferInfo dbi[3] = {{0}};
    dbi[0].buffer = src_buf.buffer; dbi[0].range = buf_bytes;
    dbi[1].buffer = dst_buf.buffer; dbi[1].range = buf_bytes;
    dbi[2].buffer = lut_buf.buffer; dbi[2].range = VK_WHOLE_SIZE;
    VkWriteDescriptorSet wds[3] = {{0}};
    for (int i = 0; i < 3; i++) {
        wds[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wds[i].dstSet = desc_set;
        wds[i].dstBinding = (uint32_t)i;
        wds[i].descriptorCount = 1;
        wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wds[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(ctx->device, 3, wds, 0, NULL);

    /* Command buffer + fence */
    VkCommandBuffer cb;
    VkCommandBufferAllocateInfo cbai2 = {0};
    cbai2.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai2.commandPool = ctx->cmd_pool;
    cbai2.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai2.commandBufferCount = 1;
    vkAllocateCommandBuffers(ctx->device, &cbai2, &cb);

    VkFence fence;
    VkFenceCreateInfo fci2 = {0};
    fci2.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(ctx->device, &fci2, NULL, &fence);

    VkCommandBufferBeginInfo cbbi = {0};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    /* Pilot: 1 exec to estimate time */
    vkBeginCommandBuffer(cb, &cbbi);
    vkCmdResetQueryPool(cb, ctx->timestamp_pool, 0, 2);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        ctx->timestamp_pool, 0);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipe_layout, 0, 1, &desc_set, 0, NULL);
    vkCmdDispatch(cb, (uint32_t)dispatch_count, 1, 1);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        ctx->timestamp_pool, 1);
    vkEndCommandBuffer(cb);

    VkSubmitInfo si = {0};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    vkQueueSubmit(ctx->queue, 1, &si, fence);
    vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkResetFences(ctx->device, 1, &fence);

    uint64_t ts[2];
    vkGetQueryPoolResults(ctx->device, ctx->timestamp_pool, 0, 2,
                          sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
    double pilot_ms = (double)(ts[1] - ts[0]) * ctx->timestamp_period / 1e6;

    int reps = (pilot_ms > 0.001) ? (int)(50.0 / pilot_ms) : 50000;
    if (reps < 10) reps = 10;
    if (reps > 100000) reps = 100000;

    /* Warmup */
    for (int w = 0; w < WARMUP; w++) {
        vkResetCommandBuffer(cb, 0);
        vkBeginCommandBuffer(cb, &cbbi);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipe_layout, 0, 1, &desc_set, 0, NULL);
        vkCmdDispatch(cb, (uint32_t)dispatch_count, 1, 1);
        vkEndCommandBuffer(cb);
        vkQueueSubmit(ctx->queue, 1, &si, fence);
        vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
        vkResetFences(ctx->device, 1, &fence);
    }

    /* Timed runs */
    double times[ITERS];
    for (int it = 0; it < ITERS; it++) {
        vkResetCommandBuffer(cb, 0);
        vkBeginCommandBuffer(cb, &cbbi);
        vkCmdResetQueryPool(cb, ctx->timestamp_pool, 0, 2);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            ctx->timestamp_pool, 0);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipe_layout, 0, 1, &desc_set, 0, NULL);
        for (int r = 0; r < reps; r++)
            vkCmdDispatch(cb, (uint32_t)dispatch_count, 1, 1);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            ctx->timestamp_pool, 1);
        vkEndCommandBuffer(cb);
        vkQueueSubmit(ctx->queue, 1, &si, fence);
        vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
        vkResetFences(ctx->device, 1, &fence);

        vkGetQueryPoolResults(ctx->device, ctx->timestamp_pool, 0, 2,
                              sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
        times[it] = (double)(ts[1] - ts[0]) * ctx->timestamp_period / 1e6 / reps;
    }

    double result = median_d(times, ITERS);

    /* Cleanup */
    vkDestroyFence(ctx->device, fence, NULL);
    vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cb);
    vkFreeDescriptorSets(ctx->device, ctx->desc_pool, 1, &desc_set);
    destroy_buf(ctx, &lut_buf);
    destroy_buf(ctx, &dst_buf);
    destroy_buf(ctx, &src_buf);
    vkDestroyPipeline(ctx->device, pipeline, NULL);
    vkDestroyShaderModule(ctx->device, shader, NULL);
    vkDestroyPipelineLayout(ctx->device, pipe_layout, NULL);
    vkDestroyDescriptorSetLayout(ctx->device, desc_layout, NULL);

    return result;
}

/* ============================================================ */
/* Stockham multi-stage bench (for four-step sizes)             */
/* ============================================================ */

#include "fft_stockham_gen.h"

static double bench_stockham(VkCtx *ctx, int n) {
    int radices[FFT_STOCKHAM_MAX_STAGES];
    int nstg = fft_stockham_factorize(n, 0, radices);
    if (nstg == 0) return -1;

    int wg_size = 64;

    /* Create descriptor layout + pipeline layout */
    VkDescriptorSetLayoutBinding bindings[2] = {{0}};
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dslci = {0};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 2; dslci.pBindings = bindings;
    VkDescriptorSetLayout desc_layout;
    vkCreateDescriptorSetLayout(ctx->device, &dslci, NULL, &desc_layout);

    VkPipelineLayoutCreateInfo plci = {0};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1; plci.pSetLayouts = &desc_layout;
    VkPipelineLayout pipe_layout;
    vkCreatePipelineLayout(ctx->device, &plci, NULL, &pipe_layout);

    /* Create per-stage pipelines */
    VkShaderModule shaders[FFT_STOCKHAM_MAX_STAGES] = {0};
    VkPipeline pipelines[FFT_STOCKHAM_MAX_STAGES] = {0};
    uint32_t dispatch_x[FFT_STOCKHAM_MAX_STAGES];

    int stride = 1;
    for (int s = 0; s < nstg; s++) {
        int radix = radices[s];
        dispatch_x[s] = (uint32_t)fft_stockham_dispatch_x(n, radix, wg_size);

        char *wgsl = gen_fft_stockham(radix, stride, n, 1, wg_size);
        if (!wgsl) return -1;
        uint32_t *spirv = NULL; size_t sc = 0;
        if (compile_wgsl(wgsl, &spirv, &sc) != 0) { free(wgsl); return -1; }
        free(wgsl);

        VkShaderModuleCreateInfo smci = {0};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sc * sizeof(uint32_t); smci.pCode = spirv;
        vkCreateShaderModule(ctx->device, &smci, NULL, &shaders[s]);
        wgsl_lower_free(spirv);

        VkComputePipelineCreateInfo cpci = {0};
        cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = shaders[s];
        cpci.stage.pName = "main";
        cpci.layout = pipe_layout;
        vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &cpci, NULL, &pipelines[s]);

        stride *= radix;
    }

    /* Buffers */
    VkDeviceSize buf_bytes = (VkDeviceSize)n * 2 * sizeof(float);
    GpuBuf src_buf, dst_buf, scratch_buf;
    create_buf(ctx, buf_bytes, &src_buf);
    create_buf(ctx, buf_bytes, &dst_buf);
    create_buf(ctx, buf_bytes, &scratch_buf);

    /* Descriptor sets with ping-pong */
    VkDescriptorSet ds[FFT_STOCKHAM_MAX_STAGES];
    VkDescriptorSetLayout layouts[FFT_STOCKHAM_MAX_STAGES];
    for (int i = 0; i < nstg; i++) layouts[i] = desc_layout;
    VkDescriptorSetAllocateInfo dsai = {0};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = ctx->desc_pool;
    dsai.descriptorSetCount = (uint32_t)nstg;
    dsai.pSetLayouts = layouts;
    vkAllocateDescriptorSets(ctx->device, &dsai, ds);

    VkBuffer read_bufs[FFT_STOCKHAM_MAX_STAGES];
    VkBuffer write_bufs[FFT_STOCKHAM_MAX_STAGES];
    write_bufs[nstg - 1] = dst_buf.buffer;
    for (int i = nstg - 2; i >= 0; i--)
        write_bufs[i] = (write_bufs[i + 1] == dst_buf.buffer)
                       ? scratch_buf.buffer : dst_buf.buffer;
    read_bufs[0] = src_buf.buffer;
    for (int i = 1; i < nstg; i++)
        read_bufs[i] = write_bufs[i - 1];

    for (int i = 0; i < nstg; i++) {
        VkDescriptorBufferInfo dbi[2] = {{0}};
        dbi[0].buffer = read_bufs[i]; dbi[0].range = buf_bytes;
        dbi[1].buffer = write_bufs[i]; dbi[1].range = buf_bytes;
        VkWriteDescriptorSet wds[2] = {{0}};
        for (int b = 0; b < 2; b++) {
            wds[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wds[b].dstSet = ds[i]; wds[b].dstBinding = (uint32_t)b;
            wds[b].descriptorCount = 1;
            wds[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wds[b].pBufferInfo = &dbi[b];
        }
        vkUpdateDescriptorSets(ctx->device, 2, wds, 0, NULL);
    }

    VkCommandBuffer cb;
    VkCommandBufferAllocateInfo cbai = {0};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = ctx->cmd_pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    vkAllocateCommandBuffers(ctx->device, &cbai, &cb);

    VkFence fence;
    VkFenceCreateInfo fci = {0};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(ctx->device, &fci, NULL, &fence);

    VkCommandBufferBeginInfo cbbi = {0};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    VkMemoryBarrier bar = {0};
    bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    /* Pilot */
    vkBeginCommandBuffer(cb, &cbbi);
    vkCmdResetQueryPool(cb, ctx->timestamp_pool, 0, 2);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        ctx->timestamp_pool, 0);
    for (int i = 0; i < nstg; i++) {
        if (i > 0)
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &bar, 0, NULL, 0, NULL);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[i]);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipe_layout, 0, 1, &ds[i], 0, NULL);
        vkCmdDispatch(cb, dispatch_x[i], 1, 1);
    }
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        ctx->timestamp_pool, 1);
    vkEndCommandBuffer(cb);

    VkSubmitInfo si = {0};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    vkQueueSubmit(ctx->queue, 1, &si, fence);
    vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkResetFences(ctx->device, 1, &fence);

    uint64_t ts[2];
    vkGetQueryPoolResults(ctx->device, ctx->timestamp_pool, 0, 2,
                          sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
    double pilot_ms = (double)(ts[1] - ts[0]) * ctx->timestamp_period / 1e6;

    int reps = (pilot_ms > 0.001) ? (int)(50.0 / pilot_ms) : 50000;
    if (reps < 10) reps = 10;
    if (reps > 50000) reps = 50000;

    /* Warmup */
    for (int w = 0; w < WARMUP; w++) {
        vkResetCommandBuffer(cb, 0);
        vkBeginCommandBuffer(cb, &cbbi);
        for (int i = 0; i < nstg; i++) {
            if (i > 0)
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     0, 1, &bar, 0, NULL, 0, NULL);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[i]);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipe_layout, 0, 1, &ds[i], 0, NULL);
            vkCmdDispatch(cb, dispatch_x[i], 1, 1);
        }
        vkEndCommandBuffer(cb);
        vkQueueSubmit(ctx->queue, 1, &si, fence);
        vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
        vkResetFences(ctx->device, 1, &fence);
    }

    /* Timed runs */
    double times[ITERS];
    for (int it = 0; it < ITERS; it++) {
        vkResetCommandBuffer(cb, 0);
        vkBeginCommandBuffer(cb, &cbbi);
        vkCmdResetQueryPool(cb, ctx->timestamp_pool, 0, 2);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            ctx->timestamp_pool, 0);
        for (int r = 0; r < reps; r++) {
            for (int i = 0; i < nstg; i++) {
                if (i > 0 || r > 0)
                    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         0, 1, &bar, 0, NULL, 0, NULL);
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[i]);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        pipe_layout, 0, 1, &ds[i], 0, NULL);
                vkCmdDispatch(cb, dispatch_x[i], 1, 1);
            }
        }
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            ctx->timestamp_pool, 1);
        vkEndCommandBuffer(cb);
        vkQueueSubmit(ctx->queue, 1, &si, fence);
        vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
        vkResetFences(ctx->device, 1, &fence);

        vkGetQueryPoolResults(ctx->device, ctx->timestamp_pool, 0, 2,
                              sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
        times[it] = (double)(ts[1] - ts[0]) * ctx->timestamp_period / 1e6 / reps;
    }

    double result = median_d(times, ITERS);

    /* Cleanup */
    vkDestroyFence(ctx->device, fence, NULL);
    vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cb);
    vkFreeDescriptorSets(ctx->device, ctx->desc_pool, (uint32_t)nstg, ds);
    destroy_buf(ctx, &scratch_buf);
    destroy_buf(ctx, &dst_buf);
    destroy_buf(ctx, &src_buf);
    for (int s = 0; s < nstg; s++) {
        vkDestroyPipeline(ctx->device, pipelines[s], NULL);
        vkDestroyShaderModule(ctx->device, shaders[s], NULL);
    }
    vkDestroyPipelineLayout(ctx->device, pipe_layout, NULL);
    vkDestroyDescriptorSetLayout(ctx->device, desc_layout, NULL);

    return result;
}

/* ============================================================ */

int main(void) {
    VkCtx ctx;
    if (vk_init(&ctx, 128, 256, 1) != 0) {
        fprintf(stderr, "Vulkan init failed\n");
        return 1;
    }

    printf("GPU: %s (timestamp period: %.1f ns)\n\n",
           ctx.dev_props.deviceName, ctx.timestamp_period);
    printf("%-10s  %10s  %10s  %8s\n",
           "N", "fused_ms", "stockham_ms", "path");
    printf("---------  ----------  -----------  --------\n");

    for (int exp = 10; exp <= 20; exp++) {
        int n = 1 << exp;

        double fused_ms = -1, stockham_ms = -1;

        /* Fused (single dispatch) — up to 4096 */
        if (n <= 4096) {
            fused_ms = bench_fused(&ctx, n);
        }

        /* Stockham (multi-dispatch) — all sizes */
        stockham_ms = bench_stockham(&ctx, n);

        printf("%-10d  ", n);
        if (fused_ms >= 0)
            printf("%10.4f  ", fused_ms);
        else
            printf("%10s  ", "-");
        if (stockham_ms >= 0)
            printf("%10.4f  ", stockham_ms);
        else
            printf("%10s  ", "-");
        printf("%8s\n", (n <= 4096) ? "fused" : "stockham");
        fflush(stdout);
    }

    printf("\n");
    vk_destroy(&ctx);
    return 0;
}
