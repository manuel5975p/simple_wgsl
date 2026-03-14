/*
 * bench_cufft_matched.cu - cuFFT batched, matched sizes for comparison
 *
 * Build: nvcc -O2 -o bench_cufft_matched bench/bench_cufft_matched.cu -lcufft
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

static double median(double *a, int n) {
    std::sort(a, a+n);
    return (n%2) ? a[n/2] : (a[n/2-1]+a[n/2])/2;
}

static void bench(int n, int batch) {
    size_t buf_bytes = (size_t)n * batch * sizeof(cufftComplex);

    cufftComplex *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, buf_bytes));
    CHECK_CUDA(cudaMemset(d_data, 0, buf_bytes));

    cufftHandle plan;
    int dims[] = {n};
    CHECK_CUFFT(cufftPlanMany(&plan, 1, dims,
                               NULL, 1, n, NULL, 1, n,
                               CUFFT_C2C, batch));

    for (int w = 0; w < WARMUP; w++)
        CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
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
    double us_per_fft = ms_per_exec * 1000.0 / batch;
    int log2n = 0; { int tmp = n; while (tmp > 1) { tmp >>= 1; log2n++; } }
    double flops = 5.0 * n * log2n * batch;
    double gflops = flops / (ms_per_exec * 1e6);

    printf("%-8d  %8d  %6d  %10.4f  %12.6f  %10.2f\n",
           n, batch, reps, ms_per_exec, us_per_fft, gflops);
    fflush(stdout);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_data));
}

int main() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Real cuFFT (cufftPlanMany) — %s\n\n", prop.name);
    printf("%-8s  %8s  %6s  %10s  %12s  %10s\n",
           "N", "batch", "reps", "ms/exec", "us/fft", "GFLOP/s");
    printf("-------  --------  ------  ----------  ------------  ----------\n");

    struct { int n, batch; } cases[] = {
        {    2, 32768 },
        {    4, 32768 },
        {    8, 32768 },
        {   16, 32768 },
        {   32, 16384 },
        {   64,  8192 },
        {  128,  4096 },
        {  256,  2048 },
        {  512,  1024 },
        { 1024,   512 },
        { 2048,   256 },
        { 4096,   256 },
    };

    for (int i = 0; i < 12; i++)
        bench(cases[i].n, cases[i].batch);

    printf("\n");
    return 0;
}
