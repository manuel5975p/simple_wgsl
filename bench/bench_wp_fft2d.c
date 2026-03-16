/*
 * bench_wp_fft2d.c - 2D FFT benchmark using WorkPackages + vkCmdTimestamp
 *
 * Tests square and rectangular 2D sizes up to 512x128.
 * Uses the Vulkan-native WorkPackage API for precise GPU timing.
 *
 * Build: part of cmake (bench_wp_fft2d target)
 * Run:   ./build/bench_wp_fft2d
 */

#include "cuvk_internal.h"
#include "cufft_vk.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

static void bench_size(int nx, int ny) {
    struct CUctx_st *ctx = g_cuvk.current_ctx;
    int total = nx * ny;
    size_t buf_bytes = (size_t)total * 2 * sizeof(float);

    CUdeviceptr d_data;
    CHECK_CU(cuMemAlloc_v2(&d_data, buf_bytes));
    cuMemsetD8_v2(d_data, 0x3F, buf_bytes); /* non-zero init */

    cufftHandle plan;
    cufftResult cr = cufftPlan2d(&plan, nx, ny, CUFFT_C2C);
    if (cr != CUFFT_SUCCESS) {
        printf("%-4d  %-4d  %8d  plan failed (%d)\n", nx, ny, total, cr);
        cuMemFree_v2(d_data);
        return;
    }

    CuvkAlloc *alloc = cuvk_alloc_lookup(ctx, d_data);
    if (!alloc) {
        printf("%-4d  %-4d  %8d  alloc lookup failed\n", nx, ny, total);
        cufftDestroy(plan);
        cuMemFree_v2(d_data);
        return;
    }
    VkBuffer buf = alloc->buffer;
    VkDeviceAddress bda = alloc->device_addr;

    CuvkWorkPackage wp;
    CHECK_CUFFT(cuvk_wp_init(&wp, ctx));
    CHECK_CUFFT(vkCufftExecC2C(&wp, plan, buf, bda, buf, bda, CUFFT_FORWARD));

    /* Timestamp query pool */
    VkQueryPool ts_pool = VK_NULL_HANDLE;
    {
        VkQueryPoolCreateInfo qpci = {0};
        qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = 2;
        CHECK_VK(g_cuvk.vk.vkCreateQueryPool(ctx->device, &qpci, NULL, &ts_pool));
    }

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

    /* Warmup */
    g_cuvk.vk.vkResetCommandBuffer(cb, 0);
    g_cuvk.vk.vkBeginCommandBuffer(cb, &cbbi);
    g_cuvk.vk.vkCmdResetQueryPool(cb, ts_pool, 0, 2);
    g_cuvk.vk.vkCmdWriteTimestamp(cb,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, ts_pool, 0);
    cuvk_wp_encode(&wp, cb);
    g_cuvk.vk.vkCmdWriteTimestamp(cb,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, ts_pool, 1);
    g_cuvk.vk.vkEndCommandBuffer(cb);

    for (int w = 0; w < WARMUP; w++) {
        g_cuvk.vk.vkResetFences(ctx->device, 1, &fence);
        VkSubmitInfo si = {0};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        g_cuvk.vk.vkQueueSubmit(ctx->compute_queue, 1, &si, fence);
        g_cuvk.vk.vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
    }

    /* Pilot to determine reps */
    int reps = 1;
    {
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

    /* Record with reps */
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
    double us_per_fft = median_ns / 1000.0;
    int log2nx = 0; { int t = nx; while (t > 1) { t >>= 1; log2nx++; } }
    int log2ny = 0; { int t = ny; while (t > 1) { t >>= 1; log2ny++; } }
    double flops = 5.0 * ny * log2ny * nx + 5.0 * nx * log2nx * ny;
    double gflops = flops / median_ns;

    printf("%-4d  %-4d  %8d  %6d  %10.4f  %12.4f  %10.2f\n",
           nx, ny, total, reps, median_ns / 1e6, us_per_fft, gflops);
    fflush(stdout);

    /* Cleanup */
    g_cuvk.vk.vkDestroyFence(ctx->device, fence, NULL);
    g_cuvk.vk.vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cb);
    g_cuvk.vk.vkDestroyQueryPool(ctx->device, ts_pool, NULL);
    cuvk_wp_destroy(&wp);
    cufftDestroy(plan);
    cuMemFree_v2(d_data);
}

static void bench_size_r2c(int nx, int ny) {
    struct CUctx_st *ctx = g_cuvk.current_ctx;
    int padded_y = ny / 2 + 1;
    int total_c = nx * padded_y;
    size_t in_bytes = (size_t)nx * ny * sizeof(float);
    size_t out_bytes = (size_t)total_c * 2 * sizeof(float);

    CUdeviceptr d_in, d_out;
    CHECK_CU(cuMemAlloc_v2(&d_in, in_bytes));
    CHECK_CU(cuMemAlloc_v2(&d_out, out_bytes));
    cuMemsetD8_v2(d_in, 0x3F, in_bytes);

    cufftHandle plan;
    cufftResult cr = cufftPlan2d(&plan, nx, ny, CUFFT_R2C);
    if (cr != CUFFT_SUCCESS) {
        printf("%-4d  %-4d  %8d  plan failed (%d)\n", nx, ny, nx * ny, cr);
        cuMemFree_v2(d_in); cuMemFree_v2(d_out);
        return;
    }

    CuvkAlloc *alloc_in = cuvk_alloc_lookup(ctx, d_in);
    CuvkAlloc *alloc_out = cuvk_alloc_lookup(ctx, d_out);
    if (!alloc_in || !alloc_out) {
        printf("%-4d  %-4d  %8d  alloc lookup failed\n", nx, ny, nx * ny);
        cufftDestroy(plan); cuMemFree_v2(d_in); cuMemFree_v2(d_out);
        return;
    }

    CuvkWorkPackage wp;
    CHECK_CUFFT(cuvk_wp_init(&wp, ctx));
    CHECK_CUFFT(vkCufftExecR2C(&wp, plan,
                                alloc_in->buffer, alloc_in->device_addr,
                                alloc_out->buffer, alloc_out->device_addr));

    VkQueryPool ts_pool = VK_NULL_HANDLE;
    {
        VkQueryPoolCreateInfo qpci = {0};
        qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = 2;
        CHECK_VK(g_cuvk.vk.vkCreateQueryPool(ctx->device, &qpci, NULL, &ts_pool));
    }

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

    /* Warmup */
    g_cuvk.vk.vkResetCommandBuffer(cb, 0);
    g_cuvk.vk.vkBeginCommandBuffer(cb, &cbbi);
    g_cuvk.vk.vkCmdResetQueryPool(cb, ts_pool, 0, 2);
    g_cuvk.vk.vkCmdWriteTimestamp(cb,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, ts_pool, 0);
    cuvk_wp_encode(&wp, cb);
    g_cuvk.vk.vkCmdWriteTimestamp(cb,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, ts_pool, 1);
    g_cuvk.vk.vkEndCommandBuffer(cb);

    for (int w = 0; w < WARMUP; w++) {
        g_cuvk.vk.vkResetFences(ctx->device, 1, &fence);
        VkSubmitInfo si = {0};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        g_cuvk.vk.vkQueueSubmit(ctx->compute_queue, 1, &si, fence);
        g_cuvk.vk.vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
    }

    /* Pilot */
    int reps = 1;
    {
        uint64_t ts[2];
        g_cuvk.vk.vkGetQueryPoolResults(ctx->device, ts_pool, 0, 2,
            sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
        double one_ns = (double)(ts[1] - ts[0]) *
                         (double)ctx->dev_props.limits.timestampPeriod;
        if (one_ns > 0) {
            reps = (int)(1000000.0 / one_ns);
            if (reps < 1) reps = 1;
            if (reps > 10000) reps = 10000;
        }
    }

    /* Record with reps */
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
    double us_per_fft = median_ns / 1000.0;
    /* R2C FLOP count: 5*N*log2(N) for the real dimension */
    int log2nx = 0; { int t = nx; while (t > 1) { t >>= 1; log2nx++; } }
    int log2ny = 0; { int t = ny; while (t > 1) { t >>= 1; log2ny++; } }
    double flops = 2.5 * ny * log2ny * nx + 5.0 * nx * log2nx * padded_y;
    double gflops = flops / median_ns;

    printf("%-4d  %-4d  %8d  %6d  %10.4f  %12.4f  %10.2f\n",
           nx, ny, nx * ny, reps, median_ns / 1e6, us_per_fft, gflops);
    fflush(stdout);

    g_cuvk.vk.vkDestroyFence(ctx->device, fence, NULL);
    g_cuvk.vk.vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cb);
    g_cuvk.vk.vkDestroyQueryPool(ctx->device, ts_pool, NULL);
    cuvk_wp_destroy(&wp);
    cufftDestroy(plan);
    cuMemFree_v2(d_in);
    cuMemFree_v2(d_out);
}

int main(int argc, char **argv) {
    CHECK_CU(cuInit(0));

    int dev_id = (argc > 1) ? atoi(argv[1]) : 0;
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, dev_id));

    char name[256];
    cuDeviceGetName(name, sizeof(name), dev);

    CUcontext ctx;
    CHECK_CU(cuCtxCreate_v4(&ctx, NULL, 0, dev));

    printf("================================================================\n");
    printf("  2D FFT Benchmark (WorkPackage + vkCmdTimestamp)\n");
    printf("  Device: %s\n", name);
    printf("  Warmup: %d  Iters: %d (median)\n", WARMUP, ITERS);
    printf("================================================================\n\n");

    printf("%-4s  %-4s  %8s  %6s  %10s  %12s  %10s\n",
           "NX", "NY", "total", "reps", "ms/exec", "us/fft", "GFLOP/s");
    printf("----  ----  --------  ------  ----------  ------------  ----------\n");

    struct { int nx, ny; } configs[] = {
        /* Square power-of-2 */
        {4,    4},
        {8,    8},
        {16,   16},
        {32,   32},
        {64,   64},
        {128,  128},
        {256,  256},
        {512,  512},
        /* Rectangular */
        {32,   128},
        {64,   128},
        {128,  64},
        {256,  128},
        {128,  256},
        {512,  128},
        {128,  512},
        {256,  64},
        {64,   256},
        {512,  64},
        {64,   512},
        {256,  32},
        {32,   256},
        {512,  32},
        {32,   512},
    };
    int nconfigs = (int)(sizeof(configs) / sizeof(configs[0]));

    if (!getenv("R2C_ONLY")) {
        for (int i = 0; i < nconfigs; i++)
            bench_size(configs[i].nx, configs[i].ny);
    }

    printf("\n");

    /* R2C benchmarks */
    printf("%-4s  %-4s  %8s  %6s  %10s  %12s  %10s\n",
           "NX", "NY", "total", "reps", "ms/exec", "us/fft", "GFLOP/s");
    printf("----  ----  --------  ------  ----------  ------------  ----------\n");
    printf("  [R2C]\n");

    struct { int nx, ny; } r2c_configs[] = {
        {64,   64},
        {128,  128},
        {256,  256},
        {512,  128},
        {128,  512},
        {256,  128},
        {128,  256},
        {512,  256},
        {256,  512},
    };
    int nr2c = (int)(sizeof(r2c_configs) / sizeof(r2c_configs[0]));

    for (int i = 0; i < nr2c; i++)
        bench_size_r2c(r2c_configs[i].nx, r2c_configs[i].ny);

    printf("\n");

    cuCtxDestroy_v2(ctx);
    return 0;
}
