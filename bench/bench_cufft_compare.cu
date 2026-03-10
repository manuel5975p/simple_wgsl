/*
 * bench_cufft_compare.cu - Benchmark real NVIDIA cuFFT, batch=1
 *
 * Build: nvcc -O2 -o bench_cufft_compare bench/bench_cufft_compare.cu -lcufft
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cufft.h>

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

static double median(double *arr, int n) {
    std::sort(arr, arr + n);
    if (n % 2 == 0) return (arr[n/2 - 1] + arr[n/2]) / 2.0;
    return arr[n/2];
}

int main() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n\n", prop.name);

    printf("%-10s  %10s  %6s\n", "N", "ms/exec", "reps");
    printf("---------  ----------  ------\n");

    for (int exp = 1; exp <= 12; exp++) {
        int n = 1 << exp;

        cufftComplex *d_data;
        CHECK_CUDA(cudaMalloc(&d_data, sizeof(cufftComplex) * n));
        CHECK_CUDA(cudaMemset(d_data, 0, sizeof(cufftComplex) * n));

        cufftHandle plan;
        CHECK_CUFFT(cufftPlan1d(&plan, n, CUFFT_C2C, 1));

        /* Warmup */
        for (int w = 0; w < WARMUP; w++)
            CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
        CHECK_CUDA(cudaDeviceSynchronize());

        /* Pilot to determine reps */
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float pilot_ms;
        CHECK_CUDA(cudaEventElapsedTime(&pilot_ms, start, stop));

        int reps = (pilot_ms > 0.001f) ? (int)(50.0f / pilot_ms) : 50000;
        if (reps < 10) reps = 10;
        if (reps > 100000) reps = 100000;

        /* Timed runs */
        double times[ITERS];
        for (int it = 0; it < ITERS; it++) {
            CHECK_CUDA(cudaEventRecord(start));
            for (int r = 0; r < reps; r++)
                CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));
            float ms;
            CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
            times[it] = (double)ms / reps;
        }

        printf("%-10d  %10.4f  %6d\n", n, median(times, ITERS), reps);
        fflush(stdout);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUFFT(cufftDestroy(plan));
        CHECK_CUDA(cudaFree(d_data));
    }

    printf("\n");
    return 0;
}
