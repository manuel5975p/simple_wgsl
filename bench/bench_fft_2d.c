/*
 * bench_fft_2d.c - Benchmark 2D FFT via cuFFT API (Vulkan-backed)
 *
 * Tests NxN for N = 4, 8, 16, 32, 64, 128, 256, batch=1.
 * Uses wall-clock timing around cufftExecC2C calls.
 * Verifies correctness using impulse response (all 1s).
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

    printf("2D FFT benchmark (Vulkan-backed cuFFT), batch=1\n\n");
    printf("%-10s  %12s  %6s  %8s\n", "NxN", "us/exec", "reps", "err");
    printf("---------  ------------  ------  --------\n");

    /* Square sizes */
    int sq_sizes[] = {4, 8, 16, 32, 64, 128, 256};
    int n_sq = sizeof(sq_sizes) / sizeof(sq_sizes[0]);
    /* Rectangular sizes: {nx, ny} */
    int rect_sizes[][2] = {{4, 8}, {8, 16}, {16, 32}, {32, 64},
                           {64, 128}, {128, 256}, {8, 256}};
    int n_rect = sizeof(rect_sizes) / sizeof(rect_sizes[0]);
    int total_tests = n_sq + n_rect;

    for (int si = 0; si < total_tests; si++) {
        int nx, ny;
        if (si < n_sq) {
            nx = ny = sq_sizes[si];
        } else {
            nx = rect_sizes[si - n_sq][0];
            ny = rect_sizes[si - n_sq][1];
        }
        int total = nx * ny;

        /* Allocate GPU buffer: NxN complex floats */
        size_t buf_bytes = (size_t)total * 2 * sizeof(float);
        CUdeviceptr d_data;
        cuMemAlloc(&d_data, buf_bytes);

        /* Upload impulse: (1,0, 0,0, ...) */
        float *h_data = (float *)calloc((size_t)total * 2, sizeof(float));
        h_data[0] = 1.0f;
        cuMemcpyHtoD(d_data, h_data, buf_bytes);

        /* Create 2D plan */
        cufftHandle plan;
        cufftResult cr = cufftPlan2d(&plan, nx, ny, CUFFT_C2C);
        if (cr != CUFFT_SUCCESS) {
            printf("%-10s  cufftPlan2d failed (%d)\n", "", cr);
            free(h_data);
            cuMemFree(d_data);
            continue;
        }

        /* Warmup */
        for (int w = 0; w < WARMUP; w++) {
            cuMemcpyHtoD(d_data, h_data, buf_bytes);
            cufftExecC2C(plan, (cufftComplex *)d_data,
                         (cufftComplex *)d_data, CUFFT_FORWARD);
        }

        /* Verify correctness: re-upload impulse, do one FFT, check all 1s */
        memset(h_data, 0, buf_bytes);
        h_data[0] = 1.0f;
        cuMemcpyHtoD(d_data, h_data, buf_bytes);
        cufftExecC2C(plan, (cufftComplex *)d_data,
                     (cufftComplex *)d_data, CUFFT_FORWARD);
        cuMemcpyDtoH(h_data, d_data, buf_bytes);
        float max_err = 0.0f;
        for (int i = 0; i < total; i++) {
            float err_re = fabsf(h_data[i * 2] - 1.0f);
            float err_im = fabsf(h_data[i * 2 + 1]);
            if (err_re > max_err) max_err = err_re;
            if (err_im > max_err) max_err = err_im;
        }

        /* Re-upload impulse for timing runs */
        memset(h_data, 0, buf_bytes);
        h_data[0] = 1.0f;
        cuMemcpyHtoD(d_data, h_data, buf_bytes);

        /* Pilot to determine reps */
        cuCtxSynchronize();
        double t0 = now_ms();
        cufftExecC2C(plan, (cufftComplex *)d_data,
                     (cufftComplex *)d_data, CUFFT_FORWARD);
        cuCtxSynchronize();
        double pilot = now_ms() - t0;

        int reps = (pilot > 0.01) ? (int)(100.0 / pilot) : 10000;
        if (reps < 5) reps = 5;
        if (reps > 50000) reps = 50000;

        /* Timed runs */
        double times[ITERS];
        for (int it = 0; it < ITERS; it++) {
            cuCtxSynchronize();
            double start = now_ms();
            for (int r = 0; r < reps; r++)
                cufftExecC2C(plan, (cufftComplex *)d_data,
                             (cufftComplex *)d_data, CUFFT_FORWARD);
            cuCtxSynchronize();
            times[it] = (now_ms() - start) / reps;
        }

        char label[32];
        snprintf(label, sizeof(label), "%dx%d", nx, ny);
        printf("%-10s  %12.4f  %6d  %8.1e\n",
               label, median_d(times, ITERS) * 1000.0, reps, max_err);
        fflush(stdout);

        cufftDestroy(plan);
        cuMemFree(d_data);
        free(h_data);
    }

    printf("\n");
    cuCtxDestroy(ctx);
    return 0;
}
