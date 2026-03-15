/*
 * bench_cufft_2d_batched.cu - 2D cuFFT with batch=10, CUDA events
 *
 * Build: nvcc -O2 -o bench_cufft_2d_batched bench/bench_cufft_2d_batched.cu -lcufft
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cufft.h>

#define CHECK_CUDA(x) do { cudaError_t e = (x); if (e) { fprintf(stderr, "CUDA err %d at %s:%d\n", e, __FILE__, __LINE__); exit(1); } } while(0)
#define CHECK_CUFFT(x) do { cufftResult r = (x); if (r) { fprintf(stderr, "cuFFT err %d at %s:%d\n", r, __FILE__, __LINE__); exit(1); } } while(0)

#define WARMUP 10
#define ITERS 50
#define BATCH 10

static double median(double *a, int n) {
    std::sort(a, a+n);
    return (n%2) ? a[n/2] : (a[n/2-1]+a[n/2])/2;
}

static void bench(int nx, int ny) {
    int total = nx * ny;
    size_t bytes = (size_t)total * BATCH * sizeof(cufftComplex);

    cufftComplex *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));
    CHECK_CUDA(cudaMemset(d_data, 0, bytes));

    /* cufftPlanMany for batched 2D */
    cufftHandle plan;
    int dims[2] = {nx, ny};
    CHECK_CUFFT(cufftPlanMany(&plan, 2, dims,
                               NULL, 1, total,
                               NULL, 1, total,
                               CUFFT_C2C, BATCH));

    for (int w = 0; w < WARMUP; w++) {
        CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    /* Pilot */
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float pilot_ms;
    CHECK_CUDA(cudaEventElapsedTime(&pilot_ms, start, stop));

    int reps = (pilot_ms > 0.01f) ? (int)(100.0f / pilot_ms) : 10000;
    if (reps < 10) reps = 10;
    if (reps > 100000) reps = 100000;

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

    double ms_per_exec = median(times, ITERS);
    double us_per_fft = ms_per_exec * 1000.0 / BATCH;
    int log2nx = 0; { int t = nx; while (t > 1) { t >>= 1; log2nx++; } }
    int log2ny = 0; { int t = ny; while (t > 1) { t >>= 1; log2ny++; } }
    double flops = (5.0 * ny * log2ny * nx + 5.0 * nx * log2nx * ny) * BATCH;
    double gflops = flops / (ms_per_exec * 1e6);

    printf("%-4d  %-4d  %5d  %6d  %10.4f  %12.4f  %10.2f\n",
           nx, ny, BATCH, reps, ms_per_exec, us_per_fft, gflops);
    fflush(stdout);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_data));
}

int main() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Real cuFFT 2D, batch=%d (CUDA events) -- %s\n\n", BATCH, prop.name);
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
    return 0;
}
