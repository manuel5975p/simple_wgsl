/*
 * bench_wp_fft.c - 1D FFT benchmark using WorkPackages + vkCmdTimestamp
 *
 * Tests all power-of-2 sizes 4..1024 and key mixed-radix sizes.
 * Uses the Vulkan-native WorkPackage API for precise GPU timing.
 *
 * Build: part of cmake (bench_wp_fft target)
 * Run:   ./build/bench_wp_fft
 */

#include "cuvk_internal.h"
#include "cufft_vk.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* cuda.h is included via cuvk_internal.h. Use versioned API names
 * directly to bypass the macro remapping (cuCtxCreate → cuCtxCreate_v4 etc.) */

#define CHECK_CU(x) do { CUresult r = (x); if (r) { \
    fprintf(stderr, "CUDA error %d at %s:%d\n", r, __FILE__, __LINE__); exit(1); \
} } while (0)

#define CHECK_CUFFT(x) do { cufftResult r = (x); if (r) { \
    fprintf(stderr, "cuFFT error %d at %s:%d\n", r, __FILE__, __LINE__); exit(1); \
} } while (0)

#define CHECK_VK(x) do { VkResult r = (x); if (r) { \
    fprintf(stderr, "Vulkan error %d at %s:%d\n", r, __FILE__, __LINE__); exit(1); \
} } while (0)

#define WARMUP 10
#define ITERS  25

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static double median(double *a, int n) {
    qsort(a, (size_t)n, sizeof(double), cmp_double);
    return (n % 2) ? a[n / 2] : (a[n / 2 - 1] + a[n / 2]) / 2.0;
}

static void bench_size(int n, int batch) {
    struct CUctx_st *ctx = g_cuvk.current_ctx;
    size_t buf_bytes = (size_t)n * batch * 2 * sizeof(float);

    /* Allocate device buffer */
    CUdeviceptr d_data;
    CHECK_CU(cuMemAlloc_v2(&d_data, buf_bytes));
    cuMemsetD8_v2(d_data, 0, buf_bytes);

    /* Create cuFFT plan (triggers auto-tuning sweep for N≤1024) */
    cufftHandle plan;
    cufftResult cr = cufftPlan1d(&plan, n, CUFFT_C2C, batch);
    if (cr != CUFFT_SUCCESS) {
        printf("%-6d  %8d  plan failed (%d)\n", n, batch, cr);
        cuMemFree_v2(d_data);
        return;
    }

    /* Look up Vulkan buffer from CUdeviceptr */
    CuvkAlloc *alloc = cuvk_alloc_lookup(ctx, d_data);
    if (!alloc) {
        printf("%-6d  %8d  alloc lookup failed\n", n, batch);
        cufftDestroy(plan);
        cuMemFree_v2(d_data);
        return;
    }
    VkBuffer buf = alloc->buffer;
    VkDeviceAddress bda = alloc->device_addr;

    /* Build WorkPackage with FFT commands */
    CuvkWorkPackage wp;
    CHECK_CUFFT(cuvk_wp_init(&wp, ctx));
    CHECK_CUFFT(vkCufftExecC2C(&wp, plan, buf, bda, buf, bda, CUFFT_FORWARD));

    /* Create timestamp query pool (2 queries: start + end) */
    VkQueryPool ts_pool = VK_NULL_HANDLE;
    {
        VkQueryPoolCreateInfo qpci = {0};
        qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = 2;
        CHECK_VK(g_cuvk.vk.vkCreateQueryPool(ctx->device, &qpci, NULL, &ts_pool));
    }

    /* Allocate command buffer + fence */
    VkCommandBuffer cb;
    {
        VkCommandBufferAllocateInfo cbai = {0};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = ctx->cmd_pool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        CHECK_VK(g_cuvk.vk.vkAllocateCommandBuffers(ctx->device, &cbai, &cb));
    }
    VkFence fence;
    {
        VkFenceCreateInfo fci = {0};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        CHECK_VK(g_cuvk.vk.vkCreateFence(ctx->device, &fci, NULL, &fence));
    }

    VkCommandBufferBeginInfo cbbi = {0};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    /* Pilot: measure one exec to calibrate reps */
    g_cuvk.vk.vkResetCommandBuffer(cb, 0);
    g_cuvk.vk.vkBeginCommandBuffer(cb, &cbbi);
    g_cuvk.vk.vkCmdResetQueryPool(cb, ts_pool, 0, 2);
    g_cuvk.vk.vkCmdWriteTimestamp(cb,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, ts_pool, 0);
    cuvk_wp_encode(&wp, cb);
    g_cuvk.vk.vkCmdWriteTimestamp(cb,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, ts_pool, 1);
    g_cuvk.vk.vkEndCommandBuffer(cb);

    /* Warmup */
    for (int w = 0; w < WARMUP; w++) {
        g_cuvk.vk.vkResetFences(ctx->device, 1, &fence);
        VkSubmitInfo si = {0};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        g_cuvk.vk.vkQueueSubmit(ctx->compute_queue, 1, &si, fence);
        g_cuvk.vk.vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
    }

    /* Read pilot timestamp to determine reps */
    {
        uint64_t ts[2];
        g_cuvk.vk.vkGetQueryPoolResults(ctx->device, ts_pool, 0, 2,
            sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
        double pilot_ns = (double)(ts[1] - ts[0]) *
                           (double)ctx->dev_props.limits.timestampPeriod;
        (void)pilot_ns;  /* single-exec timing for info */
    }

    /* Determine reps: batch enough execs so each timed window is ~1ms */
    int reps = 1;
    {
        /* Re-record with timestamps for pilot measurement */
        g_cuvk.vk.vkResetCommandBuffer(cb, 0);
        g_cuvk.vk.vkBeginCommandBuffer(cb, &cbbi);
        g_cuvk.vk.vkCmdResetQueryPool(cb, ts_pool, 0, 2);
        g_cuvk.vk.vkCmdWriteTimestamp(cb,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, ts_pool, 0);
        cuvk_wp_encode(&wp, cb);
        g_cuvk.vk.vkCmdWriteTimestamp(cb,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, ts_pool, 1);
        g_cuvk.vk.vkEndCommandBuffer(cb);

        g_cuvk.vk.vkResetFences(ctx->device, 1, &fence);
        VkSubmitInfo si = {0};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        g_cuvk.vk.vkQueueSubmit(ctx->compute_queue, 1, &si, fence);
        g_cuvk.vk.vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);

        uint64_t ts[2];
        g_cuvk.vk.vkGetQueryPoolResults(ctx->device, ts_pool, 0, 2,
            sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
        double one_ns = (double)(ts[1] - ts[0]) *
                         (double)ctx->dev_props.limits.timestampPeriod;
        if (one_ns > 0) {
            reps = (int)(1000000.0 / one_ns);  /* target ~1ms */
            if (reps < 1) reps = 1;
            if (reps > 10000) reps = 10000;
        }
    }

    /* Record command buffer with reps executions + timestamps */
    VkMemoryBarrier bar = {0};
    bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    g_cuvk.vk.vkResetCommandBuffer(cb, 0);
    g_cuvk.vk.vkBeginCommandBuffer(cb, &cbbi);
    g_cuvk.vk.vkCmdResetQueryPool(cb, ts_pool, 0, 2);
    g_cuvk.vk.vkCmdWriteTimestamp(cb,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, ts_pool, 0);
    for (int r = 0; r < reps; r++) {
        if (r > 0)
            g_cuvk.vk.vkCmdPipelineBarrier(cb,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &bar, 0, NULL, 0, NULL);
        cuvk_wp_encode(&wp, cb);
    }
    g_cuvk.vk.vkCmdWriteTimestamp(cb,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, ts_pool, 1);
    g_cuvk.vk.vkEndCommandBuffer(cb);

    /* Timed runs */
    double times[ITERS];
    for (int it = 0; it < ITERS; it++) {
        g_cuvk.vk.vkResetFences(ctx->device, 1, &fence);
        VkSubmitInfo si = {0};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        g_cuvk.vk.vkQueueSubmit(ctx->compute_queue, 1, &si, fence);
        g_cuvk.vk.vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);

        uint64_t ts[2];
        g_cuvk.vk.vkGetQueryPoolResults(ctx->device, ts_pool, 0, 2,
            sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
        double total_ns = (double)(ts[1] - ts[0]) *
                           (double)ctx->dev_props.limits.timestampPeriod;
        times[it] = total_ns / reps;
    }

    double median_ns = median(times, ITERS);
    double us_per_fft = median_ns / 1000.0 / batch;
    double ms_per_exec = median_ns / 1e6;
    int log2n = 0;
    { int t = n; while (t > 1) { t >>= 1; log2n++; } }
    double flops = 5.0 * n * log2n * batch;
    double gflops = flops / (median_ns);

    printf("%-6d  %8d  %6d  %10.4f  %12.6f  %10.2f\n",
           n, batch, reps, ms_per_exec, us_per_fft, gflops);
    fflush(stdout);

    /* Cleanup */
    g_cuvk.vk.vkDestroyFence(ctx->device, fence, NULL);
    g_cuvk.vk.vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cb);
    g_cuvk.vk.vkDestroyQueryPool(ctx->device, ts_pool, NULL);
    cuvk_wp_destroy(&wp);
    cufftDestroy(plan);
    cuMemFree_v2(d_data);
}

int main(void) {
    CHECK_CU(cuInit(0));

    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));

    char name[256];
    cuDeviceGetName(name, sizeof(name), dev);

    CUcontext ctx;
    CHECK_CU(cuCtxCreate_v4(&ctx, NULL, 0, dev));

    printf("================================================================\n");
    printf("  1D FFT Benchmark (WorkPackage + vkCmdTimestamp)\n");
    printf("  Device: %s\n", name);
    printf("  Warmup: %d  Iters: %d (median)  Target: ~1ms/window\n",
           WARMUP, ITERS);
    printf("================================================================\n\n");

    printf("%-6s  %8s  %6s  %10s  %12s  %10s\n",
           "N", "batch", "reps", "ms/exec", "us/fft", "GFLOP/s");
    printf("------  --------  ------  ----------  ------------  ----------\n");

    /* Power-of-2 sizes */
    struct { int n; int batch; } configs[] = {
        {4,    262144},
        {8,    262144},
        {16,   262144},
        {32,   262144},
        {64,   65536},
        {128,  65536},
        {256,  16384},
        {512,  16384},
        {1024, 16384},
        /* Mixed-radix sizes */
        {12,   262144},
        {24,   131072},
        {48,   65536},
        {96,   65536},
        {100,  65536},
        {120,  65536},
        {240,  16384},
        {480,  16384},
        {500,  16384},
        {720,  16384},
        {960,  16384},
    };
    int nconfigs = (int)(sizeof(configs) / sizeof(configs[0]));

    for (int i = 0; i < nconfigs; i++)
        bench_size(configs[i].n, configs[i].batch);

    printf("\n");

    cuCtxDestroy_v2(ctx);
    return 0;
}
