/*
 * bench_cufft_batched.cu - cuFFT batched throughput for po2 sizes 2-4096
 *
 * Uses cufftPlanMany with large batch to measure per-FFT throughput,
 * comparable to our fused FFT batch benchmark.
 */
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>
#include <cufft.h>

#define CHECK_CUDA(x) do { cudaError_t e = (x); if (e) { fprintf(stderr, "CUDA err %d\n", e); exit(1); } } while(0)
#define CHECK_CUFFT(x) do { cufftResult r = (x); if (r) { fprintf(stderr, "cuFFT err %d\n", r); exit(1); } } while(0)

#define WARMUP 10
#define ITERS 50

static double median(double *a, int n) {
    std::sort(a, a+n);
    return (n%2) ? a[n/2] : (a[n/2-1]+a[n/2])/2;
}

int main() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n\n", prop.name);
    printf("%-8s  %8s  %12s  %12s\n", "N", "batch", "us/fft", "GFLOP/s");
    printf("-------  --------  ------------  ------------\n");

    int batches_small[] = {256, 4096, 65536, 262144, 1048576, 4194304};
    int n_batches_small = 6;

    for (int exp = 1; exp <= 12; exp++) {
        int n = 1 << exp;
        if (n <= 16) {
            /* Sweep multiple batch sizes for small N */
            for (int bi = 0; bi < n_batches_small; bi++) {
                int batch = batches_small[bi];
                size_t buf_bytes = (size_t)n * batch * sizeof(cufftComplex);
                if (buf_bytes > 512ULL*1024*1024) continue;

                cufftComplex *d_data;
                CHECK_CUDA(cudaMalloc(&d_data, buf_bytes));
                CHECK_CUDA(cudaMemset(d_data, 0, buf_bytes));

                cufftHandle plan;
                int dims[] = {n};
                cufftResult cr = cufftPlanMany(&plan, 1, dims,
                                               NULL, 1, n, NULL, 1, n,
                                               CUFFT_C2C, batch);
                if (cr != CUFFT_SUCCESS) { CHECK_CUDA(cudaFree(d_data)); continue; }

                for (int w = 0; w < WARMUP; w++)
                    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
                CHECK_CUDA(cudaDeviceSynchronize());

                cudaEvent_t start, stop;
                CHECK_CUDA(cudaEventCreate(&start));
                CHECK_CUDA(cudaEventCreate(&stop));
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
                double flops = 5.0 * n * __builtin_ctz(n) * batch;
                double gflops = flops / (ms_per_exec * 1e6);
                printf("%-8d  %8d  %12.6f  %12.2f\n", n, batch, us_per_fft, gflops);
                fflush(stdout);

                CHECK_CUDA(cudaEventDestroy(start));
                CHECK_CUDA(cudaEventDestroy(stop));
                CHECK_CUFFT(cufftDestroy(plan));
                CHECK_CUDA(cudaFree(d_data));
            }
            printf("\n");
            continue;
        }
        int batch = (n <= 64) ? 65536 : (n <= 512) ? 16384 : (n <= 2048) ? 4096 : 256;

        cufftComplex *d_data;
        size_t buf_bytes = (size_t)n * batch * sizeof(cufftComplex);
        CHECK_CUDA(cudaMalloc(&d_data, buf_bytes));
        CHECK_CUDA(cudaMemset(d_data, 0, buf_bytes));

        cufftHandle plan;
        int dims[] = {n};
        CHECK_CUFFT(cufftPlanMany(&plan, 1, dims,
                                   NULL, 1, n,   /* inembed, istride, idist */
                                   NULL, 1, n,   /* onembed, ostride, odist */
                                   CUFFT_C2C, batch));

        /* Warmup */
        for (int w = 0; w < WARMUP; w++)
            CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
        CHECK_CUDA(cudaDeviceSynchronize());

        /* Pilot */
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float pilot_ms;
        CHECK_CUDA(cudaEventElapsedTime(&pilot_ms, start, stop));

        int reps = (pilot_ms > 0.01f) ? (int)(100.0f / pilot_ms) : 10000;
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
            times[it] = (double)ms / reps;  /* ms per batch exec */
        }

        double ms_per_exec = median(times, ITERS);
        double us_per_fft = ms_per_exec * 1000.0 / batch;
        double flops = 5.0 * n * __builtin_ctz(n) * batch;  /* 5N log2(N) per batch */
        double gflops = flops / (ms_per_exec * 1e6);

        printf("%-8d  %8d  %12.4f  %12.2f\n", n, batch, us_per_fft, gflops);
        fflush(stdout);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUFFT(cufftDestroy(plan));
        CHECK_CUDA(cudaFree(d_data));
    }
    printf("\n");
    return 0;
}
