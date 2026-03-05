/*
 * bench_cufft.cu - Benchmark cuFFT C2C forward transform
 *
 * Compiled once with nvcc, run two ways:
 *   1. Real CUDA:    ./bench_cufft
 *   2. Vulkan shim:  LD_LIBRARY_PATH=build/cuvk_runtime ./bench_cufft
 *
 * Uses batch=1 plans (the shim's supported mode) for a fair comparison.
 * Measures per-exec latency including sync overhead.
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

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static double median_d(double *arr, int n) {
    qsort(arr, n, sizeof(double), cmp_double);
    if (n % 2 == 0) return (arr[n/2 - 1] + arr[n/2]) / 2.0;
    return arr[n/2];
}

/* Verify FFT of impulse: input (1,0,0,...) should give all-ones spectrum */
static float verify_impulse(int n) {
    cufftComplex *h_in = (cufftComplex *)calloc(n, sizeof(cufftComplex));
    h_in[0].x = 1.0f;

    cufftComplex *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMemcpy(d_data, h_in, n * sizeof(cufftComplex),
                           cudaMemcpyHostToDevice));

    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, n, CUFFT_C2C, 1));
    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    CHECK_CUDA(cudaDeviceSynchronize());

    cufftComplex *h_out = (cufftComplex *)malloc(n * sizeof(cufftComplex));
    CHECK_CUDA(cudaMemcpy(h_out, d_data, n * sizeof(cufftComplex),
                           cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err_re = fabsf(h_out[i].x - 1.0f);
        float err_im = fabsf(h_out[i].y);
        if (err_re > max_err) max_err = err_re;
        if (err_im > max_err) max_err = err_im;
    }

    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_data));
    free(h_in);
    free(h_out);
    return max_err;
}

/* Verify roundtrip: fwd then inv should give N * original */
static float verify_roundtrip(int n) {
    cufftComplex *h_in = (cufftComplex *)malloc(n * sizeof(cufftComplex));
    srand(123);
    for (int i = 0; i < n; i++) {
        h_in[i].x = (float)(rand() % 1000) / 500.0f - 1.0f;
        h_in[i].y = (float)(rand() % 1000) / 500.0f - 1.0f;
    }

    cufftComplex *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMemcpy(d_data, h_in, n * sizeof(cufftComplex),
                           cudaMemcpyHostToDevice));

    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, n, CUFFT_C2C, 1));
    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));
    CHECK_CUDA(cudaDeviceSynchronize());

    cufftComplex *h_out = (cufftComplex *)malloc(n * sizeof(cufftComplex));
    CHECK_CUDA(cudaMemcpy(h_out, d_data, n * sizeof(cufftComplex),
                           cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err_re = fabsf(h_out[i].x - h_in[i].x * n);
        float err_im = fabsf(h_out[i].y - h_in[i].y * n);
        if (err_re > max_err) max_err = err_re;
        if (err_im > max_err) max_err = err_im;
    }

    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_data));
    free(h_in);
    free(h_out);
    return max_err;
}

#define WARMUP 5
#define ITERS 20

static void bench_size(int n) {
    size_t bytes = (size_t)n * sizeof(cufftComplex);

    cufftComplex *h_data = (cufftComplex *)malloc(bytes);
    srand(42);
    for (int i = 0; i < n; i++) {
        h_data[i].x = (float)(rand() % 1000) / 500.0f - 1.0f;
        h_data[i].y = (float)(rand() % 1000) / 500.0f - 1.0f;
    }

    cufftComplex *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, n, CUFFT_C2C, 1));

    /* Warmup */
    for (int i = 0; i < WARMUP; i++) {
        CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    /* Benchmark: time individual exec+sync calls */
    double times[ITERS];
    for (int i = 0; i < ITERS; i++) {
        CHECK_CUDA(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaDeviceSynchronize());

        double t0 = now_ms();
        CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
        CHECK_CUDA(cudaDeviceSynchronize());
        times[i] = now_ms() - t0;
    }

    double med = median_d(times, ITERS);
    int log2n = 0;
    { int tmp = n; while (tmp > 1) { tmp >>= 1; log2n++; } }
    double flops = 5.0 * n * log2n;
    double gflops = flops / (med * 1e6);

    /* Correctness */
    float imp_err = verify_impulse(n);
    float rt_err = verify_roundtrip(n);
    const char *status = (imp_err < 1e-3f && rt_err < 1.0f) ? "OK" : "FAIL";

    printf("  N=%-6d %s  exec=%8.3f ms  %7.2f GFLOP/s  imp_err=%.2e  rt_err=%.2e\n",
           n, status, med, gflops, imp_err, rt_err);

    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_data));
    free(h_data);
}

int main() {
    const char *ld_path = getenv("LD_LIBRARY_PATH");
    int is_shim = (ld_path && strstr(ld_path, "cuvk_runtime"));

    int dev;
    CHECK_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    printf("================================================================\n");
    printf("  cuFFT C2C Benchmark (%s)\n",
           is_shim ? "CUVK Vulkan shim" : "Real NVIDIA cuFFT");
    printf("  Device: %s\n", prop.name);
    printf("================================================================\n\n");

    int sizes[] = { 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 };
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < nsizes; i++)
        bench_size(sizes[i]);

    printf("\n");
    return 0;
}
