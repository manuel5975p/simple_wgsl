/*
 * bench_ours_batched.c - Benchmark our Vulkan-based cuFFT, batched, small sizes
 *
 * Uses ns-precision wall-clock timing with deferred replay.
 * Matches bench_cufft_batched.cu batch sizes for direct comparison.
 */

#include "cufft.h"
#include "cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#define WARMUP 10
#define ITERS  50

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static double median_d(double *arr, int n) {
    qsort(arr, (size_t)n, sizeof(double), cmp_double);
    if (n % 2 == 0) return (arr[n/2 - 1] + arr[n/2]) / 2.0;
    return arr[n/2];
}

static int64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

static void bench_size(int n, int batch) {
    size_t buf_bytes = (size_t)n * batch * 2 * sizeof(float);
    if (buf_bytes > 512ULL * 1024 * 1024) return;

    CUdeviceptr d_data;
    cuMemAlloc(&d_data, buf_bytes);

    float *h_data = (float *)calloc((size_t)n * batch * 2, sizeof(float));
    h_data[0] = 1.0f;
    cuMemcpyHtoD(d_data, h_data, buf_bytes);

    cufftHandle plan;
    cufftResult cr = cufftPlan1d(&plan, n, CUFFT_C2C, batch);
    if (cr != CUFFT_SUCCESS) {
        printf("%-6d  %8d  plan failed (%d)\n", n, batch, cr);
        fflush(stdout);
        free(h_data);
        cuMemFree(d_data);
        return;
    }

    /* Warmup */
    for (int w = 0; w < WARMUP; w++) {
        cufftExecC2C(plan, (cufftComplex *)d_data,
                     (cufftComplex *)d_data, CUFFT_FORWARD);
        cuCtxSynchronize();
    }

    /* Pilot: 10 deferred execs + 1 sync */
    int64_t t0 = now_ns();
    for (int i = 0; i < 10; i++)
        cufftExecC2C(plan, (cufftComplex *)d_data,
                     (cufftComplex *)d_data, CUFFT_FORWARD);
    cuCtxSynchronize();
    double pilot_ns = (double)(now_ns() - t0) / 10.0;

    /* Target ~200ms per timed iteration for stable measurements */
    int reps = (pilot_ns > 100.0) ? (int)(200e6 / pilot_ns) : 100000;
    if (reps < 10) reps = 10;
    if (reps > 100000) reps = 100000;

    /* Timed runs — deferred replay batches reps into fewer GPU submissions */
    double times[ITERS];
    for (int it = 0; it < ITERS; it++) {
        int64_t start = now_ns();
        for (int r = 0; r < reps; r++)
            cufftExecC2C(plan, (cufftComplex *)d_data,
                         (cufftComplex *)d_data, CUFFT_FORWARD);
        cuCtxSynchronize();
        times[it] = (double)(now_ns() - start) / reps;  /* ns per exec */
    }

    double ns_per_exec = median_d(times, ITERS);
    double us_per_fft = ns_per_exec / 1000.0 / batch;
    double ms_per_exec = ns_per_exec / 1e6;
    double flops = 5.0 * n * log2(n) * batch;
    double gflops = (ms_per_exec > 0) ? flops / (ms_per_exec * 1e6) : 0;

    printf("%-6d  %8d  %6d  %10.4f  %12.6f  %10.2f\n",
           n, batch, reps, ms_per_exec, us_per_fft, gflops);
    fflush(stdout);

    cufftDestroy(plan);
    cuMemFree(d_data);
    free(h_data);
}

int main(void) {
    cuInit(0);
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    CUcontext ctx;
    cuCtxCreate(&ctx, NULL, 0, dev);

    printf("Our cuFFT (Vulkan-backed), batched small sizes\n\n");
    printf("%-6s  %8s  %6s  %10s  %12s  %10s\n",
           "N", "batch", "reps", "ms/exec", "us/fft", "GFLOP/s");
    printf("------  --------  ------  ----------  ------------  ----------\n");

    /* Match cuFFT batched benchmark sizes */
    int sizes[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    int batches[] = {262144, 262144, 262144, 65536, 65536, 16384, 16384, 16384, 4096, 4096, 256};

    for (int i = 0; i < 11; i++)
        bench_size(sizes[i], batches[i]);

    printf("\n");
    cuCtxDestroy(ctx);
    return 0;
}
