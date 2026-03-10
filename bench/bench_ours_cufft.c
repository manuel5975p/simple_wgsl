/*
 * bench_ours_cufft.c - Benchmark our Vulkan-based cuFFT, batch=1, 2^10..2^20
 *
 * Links against our libcufft.so (Vulkan-backed cuFFT shim).
 * Uses wall-clock timing around exec calls (GPU timestamps are internal).
 */

#include "cufft.h"
#include "cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

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

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(void) {
    /* Init CUDA (our shim) */
    cuInit(0);
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    CUcontext ctx;
    cuCtxCreate(&ctx, NULL, 0, dev);

    printf("Our cuFFT (Vulkan-backed), batch=1\n\n");
    printf("%-10s  %10s  %6s  %8s\n", "N", "ms/exec", "reps", "path");
    printf("---------  ----------  ------  --------\n");

    for (int exp = 10; exp <= 20; exp++) {
        int n = 1 << exp;

        /* Allocate GPU buffer */
        size_t buf_bytes = (size_t)n * 2 * sizeof(float);
        CUdeviceptr d_data;
        cuMemAlloc(&d_data, buf_bytes);

        /* Zero-fill */
        float *h_data = (float *)calloc((size_t)n * 2, sizeof(float));
        h_data[0] = 1.0f;
        cuMemcpyHtoD(d_data, h_data, buf_bytes);

        /* Create plan */
        cufftHandle plan;
        cufftResult cr = cufftPlan1d(&plan, n, CUFFT_C2C, 1);
        if (cr != CUFFT_SUCCESS) {
            printf("%-10d  cufftPlan1d failed (%d)\n", n, cr);
            free(h_data);
            cuMemFree(d_data);
            continue;
        }

        const char *path = (n > 4096) ? "4step" : "stockham";

        /* Warmup */
        for (int w = 0; w < WARMUP; w++)
            cufftExecC2C(plan, (cufftComplex *)d_data,
                         (cufftComplex *)d_data, CUFFT_FORWARD);

        /* Pilot to determine reps */
        double t0 = now_ms();
        cufftExecC2C(plan, (cufftComplex *)d_data,
                     (cufftComplex *)d_data, CUFFT_FORWARD);
        double pilot = now_ms() - t0;

        int reps = (pilot > 0.01) ? (int)(100.0 / pilot) : 10000;
        if (reps < 5) reps = 5;
        if (reps > 50000) reps = 50000;

        /* Timed runs */
        double times[ITERS];
        for (int it = 0; it < ITERS; it++) {
            double start = now_ms();
            for (int r = 0; r < reps; r++)
                cufftExecC2C(plan, (cufftComplex *)d_data,
                             (cufftComplex *)d_data, CUFFT_FORWARD);
            times[it] = (now_ms() - start) / reps;
        }

        printf("%-10d  %10.4f  %6d  %8s\n",
               n, median_d(times, ITERS), reps, path);
        fflush(stdout);

        cufftDestroy(plan);
        cuMemFree(d_data);
        free(h_data);
    }

    printf("\n");
    cuCtxDestroy(ctx);
    return 0;
}
