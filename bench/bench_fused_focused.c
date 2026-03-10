/*
 * bench_fused_focused.c - Focused fused FFT benchmark with multiple batch sizes
 *
 * Tests po2 sizes 2-4096 at various batch counts to match cuFFT comparison.
 */

#include "fft_fused_gen.h"
#include "simple_wgsl.h"
#include "bench_vk_common.h"

#include <math.h>
#include <time.h>

#define WARMUP 5
#define ITERS  50

static int64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
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
    if (vkCreateBuffer(ctx->device, &bci, NULL, &out->buffer) != VK_SUCCESS) return -1;
    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(ctx->device, out->buffer, &reqs);
    int32_t mt = find_memory_type(&ctx->mem_props, reqs.memoryTypeBits,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mt < 0) return -1;
    VkMemoryAllocateInfo ai = {0};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = reqs.size;
    ai.memoryTypeIndex = (uint32_t)mt;
    if (vkAllocateMemory(ctx->device, &ai, NULL, &out->memory) != VK_SUCCESS) return -1;
    vkBindBufferMemory(ctx->device, out->buffer, out->memory, 0);
    return 0;
}

static void destroy_buf(VkCtx *ctx, GpuBuf *b) {
    vkFreeMemory(ctx->device, b->memory, NULL);
    vkDestroyBuffer(ctx->device, b->buffer, NULL);
}

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

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static double median_d(double *arr, int n) {
    qsort(arr, (size_t)n, sizeof(double), cmp_double);
    if (n % 2 == 0) return (arr[n/2 - 1] + arr[n/2]) / 2.0;
    return arr[n/2];
}

static double bench_fused(VkCtx *ctx, int n, int mr, int wgl, int batch) {
    int wg = fft_fused_workgroup_size_ex(n, mr, wgl);
    int bpw = fft_fused_batch_per_wg_ex(n, mr, wgl);
    if (wg <= 0 || bpw <= 0) return -1;

    int padded_batch = ((batch + bpw - 1) / bpw) * bpw;
    int dispatch_count = padded_batch / bpw;

    char *wgsl = gen_fft_fused_ex(n, 1, mr, wgl);
    if (!wgsl) return -1;
    uint32_t *spirv = NULL; size_t sc = 0;
    if (compile_wgsl(wgsl, &spirv, &sc) != 0) { free(wgsl); return -1; }
    free(wgsl);

    int lut_count = fft_fused_lut_size_ex(n, 1, mr);
    int num_bindings = (lut_count > 0) ? 3 : 2;

    /* Pipeline */
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
    VkDeviceSize buf_bytes = (VkDeviceSize)n * padded_batch * 2 * sizeof(float);
    GpuBuf src_buf, dst_buf, lut_buf;
    memset(&lut_buf, 0, sizeof(lut_buf));
    create_buf(ctx, buf_bytes, &src_buf);
    create_buf(ctx, buf_bytes, &dst_buf);

    /* LUT */
    int has_lut = 0;
    if (lut_count > 0) {
        float *lut_data = fft_fused_compute_lut_ex(n, 1, mr);
        VkDeviceSize lut_bytes = (VkDeviceSize)lut_count * 2 * sizeof(float);
        create_buf(ctx, lut_bytes, &lut_buf);
        has_lut = 1;

        /* Upload via staging */
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
    } else {
        create_buf(ctx, 64, &lut_buf);
    }

    /* Descriptor set */
    VkDescriptorSetAllocateInfo dsai = {0};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = ctx->desc_pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &desc_layout;
    VkDescriptorSet desc_set;
    vkAllocateDescriptorSets(ctx->device, &dsai, &desc_set);

    VkDescriptorBufferInfo dbi[3] = {0};
    dbi[0].buffer = src_buf.buffer; dbi[0].range = buf_bytes;
    dbi[1].buffer = dst_buf.buffer; dbi[1].range = buf_bytes;
    dbi[2].buffer = lut_buf.buffer; dbi[2].range = VK_WHOLE_SIZE;
    VkWriteDescriptorSet wds[3] = {0};
    for (int i = 0; i < (has_lut ? 3 : 2); i++) {
        wds[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wds[i].dstSet = desc_set;
        wds[i].dstBinding = (uint32_t)i;
        wds[i].descriptorCount = 1;
        wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wds[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(ctx->device, (uint32_t)(has_lut ? 3 : 2), wds, 0, NULL);

    /* Command buffer + fence */
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

    VkSubmitInfo si = {0};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;

    /* Helper: record CB with N dispatches */
    #define RECORD_CB(n_disp) do { \
        vkResetCommandBuffer(cb, 0); \
        vkBeginCommandBuffer(cb, &cbbi); \
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline); \
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, \
                                pipe_layout, 0, 1, &desc_set, 0, NULL); \
        for (int _i = 0; _i < (n_disp); _i++) \
            vkCmdDispatch(cb, (uint32_t)dispatch_count, 1, 1); \
        vkEndCommandBuffer(cb); \
    } while(0)

    #define SUBMIT_WAIT() do { \
        vkQueueSubmit(ctx->queue, 1, &si, fence); \
        vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX); \
        vkResetFences(ctx->device, 1, &fence); \
    } while(0)

    /* Warmup */
    RECORD_CB(1);
    for (int w = 0; w < WARMUP; w++) SUBMIT_WAIT();

    /* Pilot: 100 dispatches in one CB */
    RECORD_CB(100);
    int64_t t0 = now_ns();
    SUBMIT_WAIT();
    double pilot_ns = (double)(now_ns() - t0) / 100.0;

    /* Target ~100ms of GPU work per timed iteration */
    int reps = (pilot_ns > 1000.0) ? (int)(100e6 / pilot_ns) : 100000;
    if (reps < 100) reps = 100;
    if (reps > 200000) reps = 200000;

    /* Timed runs: reps dispatches in one CB, measure wall-clock (ns precision) */
    RECORD_CB(reps);
    double times[ITERS];
    for (int it = 0; it < ITERS; it++) {
        int64_t start = now_ns();
        SUBMIT_WAIT();
        int64_t elapsed = now_ns() - start;
        times[it] = (double)elapsed / reps;  /* ns per dispatch */
        if (it == 0)
            fprintf(stderr, "  [dbg] n=%d reps=%d elapsed=%ldns per_disp=%.1fns\n",
                    n, reps, (long)elapsed, times[0]);
    }
    #undef RECORD_CB
    #undef SUBMIT_WAIT

    double result_ns = median_d(times, ITERS);  /* ns per dispatch */

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

    return result_ns;
}

int main(void) {
    VkCtx ctx;
    if (vk_init(&ctx, 16, 32, 0) != 0) {
        fprintf(stderr, "Vulkan init failed\n");
        return 1;
    }

    int sizes[] = {2, 4, 8, 16, 32, 64, 128};
    int batches[] = {4194304, 4194304, 4194304, 4194304, 4194304, 1048576, 262144};
    int n_batches = 0; /* unused, per-size batch below */
    /* Best mr/wgl from previous sweep (auto is fine for most) */
    int best_mr[] = {0, 0, 0, 0, 0, 0, 0};

    printf("%-6s  %8s  %4s  %4s  %6s  %12s  %12s\n",
           "N", "batch", "B", "wg", "disp", "us/dispatch", "ns/fft");
    printf("------  --------  ----  ----  ------  ------------  ------------\n");

    for (int si = 0; si < 7; si++) {
        int n = sizes[si];
        int mr = best_mr[si];
        int wgl = 0;

        int batch = batches[si];
        while (batch > 256 && (size_t)n * batch * 2 * sizeof(float) * 2 > 512ULL*1024*1024)
            batch /= 2;

        int bpw = fft_fused_batch_per_wg_ex(n, mr, wgl);
        int wg = fft_fused_workgroup_size_ex(n, mr, wgl);
        int disp = ((batch + bpw - 1) / bpw);

        double ns_per_disp = bench_fused(&ctx, n, mr, wgl, batch);
        if (ns_per_disp < 0) continue;

        double us_per_disp = ns_per_disp / 1000.0;
        double ns_per_fft = ns_per_disp / batch;
        printf("%-6d  %8d  %4d  %4d  %6d  %12.3f  %12.4f\n",
               n, batch, bpw, wg, disp, us_per_disp, ns_per_fft);
        fflush(stdout);
    }

    vk_destroy(&ctx);
    return 0;
}
