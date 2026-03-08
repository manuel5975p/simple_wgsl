/*
 * bench_cufft_compare.cu - Benchmark real cuFFT for comparison with fused FFT
 *
 * Compiled with: nvcc -o bench_cufft_compare bench_cufft_compare.cu -lcufft
 * Uses GPU timestamp events for accurate measurement.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
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

#define WARMUP_ITERS 10
#define BENCH_ITERS  50

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static double median_d(double *arr, int n) {
    qsort(arr, n, sizeof(double), cmp_double);
    if (n % 2 == 0) return (arr[n/2 - 1] + arr[n/2]) / 2.0;
    return arr[n/2];
}

static double bench_cufft(int n, int batch, int repeats) {
    cufftComplex *h_in = (cufftComplex *)calloc((size_t)n * batch, sizeof(cufftComplex));
    h_in[0].x = 1.0f;

    cufftComplex *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, (size_t)n * batch * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMalloc(&d_out, (size_t)n * batch * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, (size_t)n * batch * sizeof(cufftComplex),
                           cudaMemcpyHostToDevice));

    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, n, CUFFT_C2C, batch));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERS; i++) {
        CHECK_CUFFT(cufftExecC2C(plan, d_in, d_out, CUFFT_FORWARD));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Timed runs — repeat `repeats` execs between events */
    double times[BENCH_ITERS];
    for (int i = 0; i < BENCH_ITERS; i++) {
        CHECK_CUDA(cudaEventRecord(start));
        for (int r = 0; r < repeats; r++)
            CHECK_CUFFT(cufftExecC2C(plan, d_in, d_out, CUFFT_FORWARD));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        times[i] = (double)ms / repeats;
    }

    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    free(h_in);

    return median_d(times, BENCH_ITERS);
}

int main(void) {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n\n", prop.name);

    /* Batched throughput test: small sizes */
    int small_sizes[] = {2, 4, 8, 16, 32, 48, 64, 128, 256, 512, 1024};
    int n_small = sizeof(small_sizes) / sizeof(small_sizes[0]);

    printf("%-8s  %6s  %6s  %10s  %10s\n", "N", "batch", "reps", "total_ms", "us/fft");
    printf("-------  ------  ------  ----------  ----------\n");

    for (int i = 0; i < n_small; i++) {
        int n = small_sizes[i];
        int batch = (n <= 64) ? 65536 : (n <= 512) ? 16384 : 4096;
        /* First estimate with reps=1, then scale to ~100ms */
        double est = bench_cufft(n, batch, 1);
        int reps = (est > 0.001) ? (int)(100.0 / est) : 100000;
        if (reps < 1) reps = 1;
        if (reps > 200000) reps = 200000;
        double ms = bench_cufft(n, batch, reps);
        double us_per = ms * 1000.0 / batch;
        printf("%-8d  %6d  %6d  %10.4f  %10.4f\n", n, batch, reps, ms, us_per);
        fflush(stdout);
    }

    /* Single-dispatch test */
    int sizes[] = {
        64, 128, 256, 512, 1024, 2048, 4096, 8192,
        48, 80, 112, 240, 336, 360, 720, 1080,
        1280, 1920, 2160, 2592, 2880, 3360,
    };
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("\n=== Single (batch=1) ===\n");
    printf("%-8s  %10s\n", "N", "cuFFT_ms");
    printf("-------  ----------\n");

    for (int i = 0; i < n_sizes; i++) {
        double ms = bench_cufft(sizes[i], 1, 1);
        printf("%-8d  %10.4f\n", sizes[i], ms);
        fflush(stdout);
    }

    return 0;
}
