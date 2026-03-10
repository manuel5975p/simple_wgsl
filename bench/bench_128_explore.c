/*
 * bench_128_explore.c - Isolate N=128 performance: vary batch, measure single-exec
 */
#include "cufft.h"
#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <math.h>

static int64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static void bench(int n, int batch) {
    size_t buf_bytes = (size_t)n * batch * 2 * sizeof(float);
    CUdeviceptr d_data;
    cuMemAlloc(&d_data, buf_bytes);
    float *h = (float *)calloc((size_t)n * batch * 2, sizeof(float));
    h[0] = 1.0f;
    cuMemcpyHtoD(d_data, h, buf_bytes);

    cufftHandle plan;
    cufftResult cr = cufftPlan1d(&plan, n, CUFFT_C2C, batch);
    if (cr != CUFFT_SUCCESS) {
        printf("N=%-6d batch=%-8d plan failed (%d)\n", n, batch, cr);
        free(h); cuMemFree(d_data); return;
    }

    /* Warmup */
    for (int w = 0; w < 20; w++) {
        cufftExecC2C(plan, (cufftComplex *)d_data,
                     (cufftComplex *)d_data, CUFFT_FORWARD);
        cuCtxSynchronize();
    }

    /* Single-exec timing (no deferred replay batching) */
    double times[50];
    for (int it = 0; it < 50; it++) {
        int64_t t0 = now_ns();
        cufftExecC2C(plan, (cufftComplex *)d_data,
                     (cufftComplex *)d_data, CUFFT_FORWARD);
        cuCtxSynchronize();
        times[it] = (double)(now_ns() - t0);
    }
    qsort(times, 50, sizeof(double), cmp_double);
    double single_ns = times[25]; /* median */

    /* Deferred replay: 100 execs + 1 sync */
    double times2[50];
    for (int it = 0; it < 50; it++) {
        int64_t t0 = now_ns();
        for (int r = 0; r < 100; r++)
            cufftExecC2C(plan, (cufftComplex *)d_data,
                         (cufftComplex *)d_data, CUFFT_FORWARD);
        cuCtxSynchronize();
        times2[it] = (double)(now_ns() - t0) / 100.0;
    }
    qsort(times2, 50, sizeof(double), cmp_double);
    double deferred_ns = times2[25];

    double data_mb = (double)n * batch * 2 * 4 / 1e6;
    double single_bw = data_mb * 2.0 / (single_ns / 1e9) / 1e3; /* GB/s, r+w */
    double defer_bw = data_mb * 2.0 / (deferred_ns / 1e9) / 1e3;
    double flops = 5.0 * n * log2(n) * batch;

    printf("N=%-4d batch=%-8d  single: %8.1f us (%5.1f GB/s %7.1f GF/s)  "
           "defer100: %8.1f us (%5.1f GB/s %7.1f GF/s)\n",
           n, batch,
           single_ns / 1e3, single_bw, flops / (single_ns / 1e9) / 1e9,
           deferred_ns / 1e3, defer_bw, flops / (deferred_ns / 1e9) / 1e9);

    cufftDestroy(plan);
    cuMemFree(d_data);
    free(h);
}

int main(void) {
    cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, NULL, 0, dev);

    printf("N=128 exploration: single exec vs deferred replay, varying batch\n\n");

    int batches[] = {1024, 4096, 16384, 65536, 262144};
    for (int i = 0; i < 5; i++)
        bench(128, batches[i]);

    printf("\nN=64 for comparison:\n");
    for (int i = 0; i < 5; i++)
        bench(64, batches[i]);

    cuCtxDestroy(ctx);
    return 0;
}
