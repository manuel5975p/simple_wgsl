/*
 * bench_cufft_batched2.cu - cuFFT batched throughput, multiple batch sizes
 */
#include <cstdio>
#include <cstdlib>
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

int main() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n\n", prop.name);
    printf("%-8s  %8s  %12s  %12s  %12s\n", "N", "batch", "us/fft", "us/batch", "GFLOP/s");
    printf("-------  --------  ------------  ------------  ------------\n");

    int sizes[] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    int batches[] = {1, 256, 4096, 16384, 65536};

    for (int si = 0; si < 12; si++) {
        int n = sizes[si];
        for (int bi = 0; bi < 5; bi++) {
            int batch = batches[bi];
            size_t buf_bytes = (size_t)n * batch * sizeof(cufftComplex);
            if (buf_bytes > 512ULL*1024*1024) continue;  /* skip huge allocations */

            cufftComplex *d_data;
            CHECK_CUDA(cudaMalloc(&d_data, buf_bytes));
            CHECK_CUDA(cudaMemset(d_data, 0, buf_bytes));

            cufftHandle plan;
            int dims[] = {n};
            cufftResult cr = cufftPlanMany(&plan, 1, dims,
                                           NULL, 1, n, NULL, 1, n,
                                           CUFFT_C2C, batch);
            if (cr != CUFFT_SUCCESS) {
                CHECK_CUDA(cudaFree(d_data));
                continue;
            }

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
            double us_per_batch = ms_per_exec * 1000.0;
            double flops = 5.0 * n * __builtin_ctz(n) * batch;
            double gflops = flops / (ms_per_exec * 1e6);

            printf("%-8d  %8d  %12.4f  %12.2f  %12.2f\n",
                   n, batch, us_per_fft, us_per_batch, gflops);
            fflush(stdout);

            CHECK_CUDA(cudaEventDestroy(start));
            CHECK_CUDA(cudaEventDestroy(stop));
            CHECK_CUFFT(cufftDestroy(plan));
            CHECK_CUDA(cudaFree(d_data));
        }
        printf("\n");
    }
    return 0;
}
