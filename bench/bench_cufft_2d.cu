/*
 * bench_cufft_2d.cu - Benchmark 2D cuFFT C2C, batch=1
 *
 * Compiled with nvcc, run two ways:
 *   Real CUDA:    ./bench_cufft_2d
 *   Vulkan shim:  LD_LIBRARY_PATH=build/cuvk_runtime ./bench_cufft_2d
 *
 * Build: nvcc -O2 -o bench_cufft_2d bench/bench_cufft_2d.cu -lcufft
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>
#include <cufft.h>
#include <time.h>

#define CHECK_CUDA(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %d at %s:%d\n", e, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUFFT(call) do { \
    cufftResult r = (call); \
    if (r != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error %d at %s:%d\n", r, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define WARMUP 10
#define ITERS  50

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static double median(double *arr, int n) {
    std::sort(arr, arr + n);
    if (n % 2 == 0) return (arr[n/2 - 1] + arr[n/2]) / 2.0;
    return arr[n/2];
}

static void bench_c2c(int nx, int ny) {
    int total = nx * ny;
    size_t bytes = (size_t)total * sizeof(cufftComplex);

    /* Impulse input */
    cufftComplex *h_data = (cufftComplex *)calloc(total, sizeof(cufftComplex));
    h_data[0].x = 1.0f;

    cufftComplex *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));

    cufftHandle plan;
    CHECK_CUFFT(cufftPlan2d(&plan, nx, ny, CUFFT_C2C));

    /* Warmup */
    for (int w = 0; w < WARMUP; w++) {
        CHECK_CUDA(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
        CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    /* Verify: impulse FFT should give all (1,0) */
    CHECK_CUDA(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    CHECK_CUDA(cudaDeviceSynchronize());
    cufftComplex *h_out = (cufftComplex *)malloc(bytes);
    CHECK_CUDA(cudaMemcpy(h_out, d_data, bytes, cudaMemcpyDeviceToHost));
    float max_err = 0.0f;
    for (int i = 0; i < total; i++) {
        float e = fabsf(h_out[i].x - 1.0f);
        if (e > max_err) max_err = e;
        e = fabsf(h_out[i].y);
        if (e > max_err) max_err = e;
    }
    free(h_out);

    /* Pilot to determine reps */
    CHECK_CUDA(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());
    double t0 = now_ms();
    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    CHECK_CUDA(cudaDeviceSynchronize());
    double pilot_ms = now_ms() - t0;

    int reps = (pilot_ms > 0.01) ? (int)(100.0 / pilot_ms) : 10000;
    if (reps < 10) reps = 10;
    if (reps > 100000) reps = 100000;

    /* Timed runs */
    double times[ITERS];
    for (int it = 0; it < ITERS; it++) {
        CHECK_CUDA(cudaDeviceSynchronize());
        double start = now_ms();
        for (int r = 0; r < reps; r++)
            CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
        CHECK_CUDA(cudaDeviceSynchronize());
        times[it] = (now_ms() - start) / reps;
    }

    char label[32];
    snprintf(label, sizeof(label), "%dx%d", nx, ny);
    printf("  %-10s  %10.4f  %6d  %8.1e\n",
           label, median(times, ITERS), reps, max_err);
    fflush(stdout);

    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_data));
    free(h_data);
}

int main() {
    const char *ld_path = getenv("LD_LIBRARY_PATH");
    int is_shim = (ld_path && strstr(ld_path, "cuvk_runtime"));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    printf("================================================================\n");
    printf("  2D cuFFT C2C Benchmark (%s)\n",
           is_shim ? "CUVK Vulkan shim" : "Real NVIDIA cuFFT");
    printf("  Device: %s\n", prop.name);
    printf("  Warmup: %d  Iters: %d (median)\n", WARMUP, ITERS);
    printf("================================================================\n\n");

    printf("  %-10s  %10s  %6s  %8s\n", "NxN", "ms/exec", "reps", "err");
    printf("  ----------  ----------  ------  --------\n");

    /* Square sizes */
    int sq[] = {4, 8, 16, 32, 64, 128, 256};
    for (int i = 0; i < 7; i++)
        bench_c2c(sq[i], sq[i]);

    printf("\n");
    printf("  %-10s  %10s  %6s  %8s\n", "NxM", "ms/exec", "reps", "err");
    printf("  ----------  ----------  ------  --------\n");

    /* Rectangular sizes */
    int rect[][2] = {{4, 8}, {8, 16}, {16, 32}, {32, 64},
                     {64, 128}, {128, 256}, {8, 256}};
    for (int i = 0; i < 7; i++)
        bench_c2c(rect[i][0], rect[i][1]);

    printf("\n");
    return 0;
}
