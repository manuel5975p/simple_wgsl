/*
 * bench_2d_matched.c - Our Vulkan cuFFT 2D, batch=10 via repeated exec
 *
 * Uses deferred replay to batch 10 cufftExecC2C into one GPU submission.
 * Measures wall-clock time (amortized over reps * 10 execs).
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
#define BATCH  10

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

static void bench(int nx, int ny) {
    int total = nx * ny;
    size_t buf_bytes = (size_t)total * 2 * sizeof(float);

    CUdeviceptr d_data;
    cuMemAlloc(&d_data, buf_bytes);
    float *h = (float *)calloc((size_t)total * 2, sizeof(float));
    h[0] = 1.0f;
    cuMemcpyHtoD(d_data, h, buf_bytes);

    cufftHandle plan;
    cufftResult cr = cufftPlan2d(&plan, nx, ny, CUFFT_C2C);
    if (cr != CUFFT_SUCCESS) {
        printf("%-4d  %-4d  plan failed (%d)\n", nx, ny, cr);
        free(h); cuMemFree(d_data);
        return;
    }

    /* Warmup */
    for (int w = 0; w < WARMUP; w++) {
        for (int b = 0; b < BATCH; b++)
            cufftExecC2C(plan, (cufftComplex *)d_data,
                         (cufftComplex *)d_data, CUFFT_FORWARD);
        cuCtxSynchronize();
    }

    /* Pilot */
    int64_t t0 = now_ns();
    for (int b = 0; b < BATCH; b++)
        cufftExecC2C(plan, (cufftComplex *)d_data,
                     (cufftComplex *)d_data, CUFFT_FORWARD);
    cuCtxSynchronize();
    double pilot_ns = (double)(now_ns() - t0);

    int reps = (pilot_ns > 1000.0) ? (int)(200e6 / pilot_ns) : 10000;
    if (reps < 10) reps = 10;
    if (reps > 50000) reps = 50000;

    double times[ITERS];
    for (int it = 0; it < ITERS; it++) {
        int64_t start = now_ns();
        for (int r = 0; r < reps; r++)
            for (int b = 0; b < BATCH; b++)
                cufftExecC2C(plan, (cufftComplex *)d_data,
                             (cufftComplex *)d_data, CUFFT_FORWARD);
        cuCtxSynchronize();
        times[it] = (double)(now_ns() - start) / reps;
    }

    double ns_per_batch = median_d(times, ITERS);
    double us_per_fft = ns_per_batch / 1000.0 / BATCH;
    double ms_per_exec = ns_per_batch / 1e6;
    int log2nx = 0; { int t = nx; while (t > 1) { t >>= 1; log2nx++; } }
    int log2ny = 0; { int t = ny; while (t > 1) { t >>= 1; log2ny++; } }
    double flops = (5.0 * ny * log2ny * nx + 5.0 * nx * log2nx * ny) * BATCH;
    double gflops = (ms_per_exec > 0) ? flops / (ms_per_exec * 1e6) : 0;

    printf("%-4d  %-4d  %5d  %6d  %10.4f  %12.4f  %10.2f\n",
           nx, ny, BATCH, reps, ms_per_exec, us_per_fft, gflops);
    fflush(stdout);

    cufftDestroy(plan);
    cuMemFree(d_data);
    free(h);
}

int main(void) {
    cuInit(0);
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    CUcontext ctx;
    cuCtxCreate(&ctx, NULL, 0, dev);

    printf("Our cuFFT 2D (Vulkan), batch=%d via deferred replay\n\n", BATCH);
    printf("%-4s  %-4s  %5s  %6s  %10s  %12s  %10s\n",
           "NX", "NY", "batch", "reps", "ms/exec", "us/fft", "GFLOP/s");
    printf("----  ----  -----  ------  ----------  ------------  ----------\n");

    struct { int nx, ny; } cases[] = {
        {64, 64}, {128, 128}, {256, 256},
        {256, 128}, {128, 256},
        {512, 128}, {128, 512},
        {512, 256}, {256, 512},
    };
    for (int i = 0; i < 9; i++)
        bench(cases[i].nx, cases[i].ny);

    printf("\n");
    cuCtxDestroy(ctx);
    return 0;
}
