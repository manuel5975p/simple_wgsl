/*
 * bench_cufft_3d.cu - Benchmark 3D cuFFT C2C, R2C, C2R
 *
 * Compiled with nvcc, run two ways:
 *   Real CUDA:    ./bench_cufft_3d
 *   Vulkan shim:  LD_LIBRARY_PATH=build/cuvk_runtime ./bench_cufft_3d
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
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

#define WARMUP 1
#define ITERS 3

static void bench_c2c(int nx, int ny, int nz) {
    int total = nx * ny * nz;
    size_t bytes = (size_t)total * sizeof(cufftComplex);

    cufftComplex *h_data = (cufftComplex *)malloc(bytes);
    srand(42);
    for (int i = 0; i < total; i++) {
        h_data[i].x = (float)(rand() % 1000) / 500.0f - 1.0f;
        h_data[i].y = (float)(rand() % 1000) / 500.0f - 1.0f;
    }

    cufftComplex *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    cufftHandle plan;
    CHECK_CUFFT(cufftPlan3d(&plan, nx, ny, nz, CUFFT_C2C));

    for (int i = 0; i < WARMUP; i++) {
        CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    double times[ITERS];
    for (int i = 0; i < ITERS; i++) {
        CHECK_CUDA(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaDeviceSynchronize());
        double t0 = now_ms();
        CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
        CHECK_CUDA(cudaDeviceSynchronize());
        times[i] = now_ms() - t0;
    }

    /* Verify roundtrip */
    CHECK_CUDA(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));
    CHECK_CUDA(cudaDeviceSynchronize());
    cufftComplex *h_out = (cufftComplex *)malloc(bytes);
    CHECK_CUDA(cudaMemcpy(h_out, d_data, bytes, cudaMemcpyDeviceToHost));
    float max_err = 0.0f;
    for (int i = 0; i < total; i++) {
        float e = fabsf(h_out[i].x - h_data[i].x * total);
        if (e > max_err) max_err = e;
        e = fabsf(h_out[i].y - h_data[i].y * total);
        if (e > max_err) max_err = e;
    }
    const char *status = (max_err < 1.0f * total) ? "OK" : "FAIL";

    double med = median_d(times, ITERS);
    printf("  C2C  %3dx%3dx%3d  %4s  %10.3f ms  err=%.2e\n",
           nx, ny, nz, status, med, max_err);

    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_data));
    free(h_data);
    free(h_out);
}

static void bench_r2c(int nx, int ny, int nz) {
    int total_r = nx * ny * nz;
    int total_c = nx * ny * (nz / 2 + 1);
    size_t bytes_r = (size_t)total_r * sizeof(cufftReal);
    size_t bytes_c = (size_t)total_c * sizeof(cufftComplex);

    cufftReal *h_data = (cufftReal *)malloc(bytes_r);
    srand(42);
    for (int i = 0; i < total_r; i++)
        h_data[i] = (float)(rand() % 1000) / 500.0f - 1.0f;

    cufftReal *d_in;
    cufftComplex *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, bytes_r));
    CHECK_CUDA(cudaMalloc(&d_out, bytes_c));

    cufftHandle plan;
    CHECK_CUFFT(cufftPlan3d(&plan, nx, ny, nz, CUFFT_R2C));

    for (int i = 0; i < WARMUP; i++) {
        CHECK_CUDA(cudaMemcpy(d_in, h_data, bytes_r, cudaMemcpyHostToDevice));
        CHECK_CUFFT(cufftExecR2C(plan, d_in, d_out));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    double times[ITERS];
    for (int i = 0; i < ITERS; i++) {
        CHECK_CUDA(cudaMemcpy(d_in, h_data, bytes_r, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaDeviceSynchronize());
        double t0 = now_ms();
        CHECK_CUFFT(cufftExecR2C(plan, d_in, d_out));
        CHECK_CUDA(cudaDeviceSynchronize());
        times[i] = now_ms() - t0;
    }

    /* Verify: DC bin should equal sum of all input */
    CHECK_CUDA(cudaMemcpy(d_in, h_data, bytes_r, cudaMemcpyHostToDevice));
    CHECK_CUFFT(cufftExecR2C(plan, d_in, d_out));
    CHECK_CUDA(cudaDeviceSynchronize());
    cufftComplex *h_out = (cufftComplex *)malloc(bytes_c);
    CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes_c, cudaMemcpyDeviceToHost));
    double sum = 0.0;
    for (int i = 0; i < total_r; i++) sum += h_data[i];
    float dc_err = fabsf(h_out[0].x - (float)sum);
    const char *status = (dc_err < 0.1f * total_r) ? "OK" : "FAIL";

    double med = median_d(times, ITERS);
    printf("  R2C  %3dx%3dx%3d  %4s  %10.3f ms  dc_err=%.2e\n",
           nx, ny, nz, status, med, dc_err);

    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    free(h_data);
    free(h_out);
}

static void bench_c2r(int nx, int ny, int nz) {
    int total_r = nx * ny * nz;
    int total_c = nx * ny * (nz / 2 + 1);
    size_t bytes_r = (size_t)total_r * sizeof(cufftReal);
    size_t bytes_c = (size_t)total_c * sizeof(cufftComplex);

    /* Generate real data, forward R2C to get valid freq-domain input */
    cufftReal *h_real = (cufftReal *)malloc(bytes_r);
    srand(42);
    for (int i = 0; i < total_r; i++)
        h_real[i] = (float)(rand() % 1000) / 500.0f - 1.0f;

    cufftReal *d_real;
    cufftComplex *d_freq;
    CHECK_CUDA(cudaMalloc(&d_real, bytes_r));
    CHECK_CUDA(cudaMalloc(&d_freq, bytes_c));

    /* Create R2C plan to generate valid frequency input */
    cufftHandle plan_r2c;
    CHECK_CUFFT(cufftPlan3d(&plan_r2c, nx, ny, nz, CUFFT_R2C));
    CHECK_CUDA(cudaMemcpy(d_real, h_real, bytes_r, cudaMemcpyHostToDevice));
    CHECK_CUFFT(cufftExecR2C(plan_r2c, d_real, d_freq));
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Copy freq data to host for re-upload each iteration */
    cufftComplex *h_freq = (cufftComplex *)malloc(bytes_c);
    CHECK_CUDA(cudaMemcpy(h_freq, d_freq, bytes_c, cudaMemcpyDeviceToHost));
    CHECK_CUFFT(cufftDestroy(plan_r2c));

    /* Now benchmark C2R */
    cufftHandle plan_c2r;
    CHECK_CUFFT(cufftPlan3d(&plan_c2r, nx, ny, nz, CUFFT_C2R));

    for (int i = 0; i < WARMUP; i++) {
        CHECK_CUDA(cudaMemcpy(d_freq, h_freq, bytes_c, cudaMemcpyHostToDevice));
        CHECK_CUFFT(cufftExecC2R(plan_c2r, d_freq, d_real));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    double times[ITERS];
    for (int i = 0; i < ITERS; i++) {
        CHECK_CUDA(cudaMemcpy(d_freq, h_freq, bytes_c, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaDeviceSynchronize());
        double t0 = now_ms();
        CHECK_CUFFT(cufftExecC2R(plan_c2r, d_freq, d_real));
        CHECK_CUDA(cudaDeviceSynchronize());
        times[i] = now_ms() - t0;
    }

    /* Verify roundtrip: R2C then C2R should give N*original */
    CHECK_CUDA(cudaMemcpy(d_freq, h_freq, bytes_c, cudaMemcpyHostToDevice));
    CHECK_CUFFT(cufftExecC2R(plan_c2r, d_freq, d_real));
    CHECK_CUDA(cudaDeviceSynchronize());
    cufftReal *h_out = (cufftReal *)malloc(bytes_r);
    CHECK_CUDA(cudaMemcpy(h_out, d_real, bytes_r, cudaMemcpyDeviceToHost));
    float max_err = 0.0f;
    for (int i = 0; i < total_r; i++) {
        float e = fabsf(h_out[i] - h_real[i] * total_r);
        if (e > max_err) max_err = e;
    }
    const char *status = (max_err < 1.0f * total_r) ? "OK" : "FAIL";

    double med = median_d(times, ITERS);
    printf("  C2R  %3dx%3dx%3d  %4s  %10.3f ms  rt_err=%.2e\n",
           nx, ny, nz, status, med, max_err);

    CHECK_CUFFT(cufftDestroy(plan_c2r));
    CHECK_CUDA(cudaFree(d_real));
    CHECK_CUDA(cudaFree(d_freq));
    free(h_real);
    free(h_freq);
    free(h_out);
}

int main() {
    const char *ld_path = getenv("LD_LIBRARY_PATH");
    int is_shim = (ld_path && strstr(ld_path, "cuvk_runtime"));

    int dev;
    CHECK_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    printf("================================================================\n");
    printf("  3D cuFFT Benchmark (%s)\n",
           is_shim ? "CUVK Vulkan shim" : "Real NVIDIA cuFFT");
    printf("  Device: %s\n", prop.name);
    printf("  Warmup: %d  Iters: %d (median)\n", WARMUP, ITERS);
    printf("================================================================\n\n");

    struct { int nx, ny, nz; } sizes[] = {
        {32, 32, 32},
        {64, 64, 64},
        {128, 128, 128},
    };
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < nsizes; i++) {
        bench_c2c(sizes[i].nx, sizes[i].ny, sizes[i].nz);
        bench_r2c(sizes[i].nx, sizes[i].ny, sizes[i].nz);
        bench_c2r(sizes[i].nx, sizes[i].ny, sizes[i].nz);
        if (i < nsizes - 1) printf("\n");
    }

    printf("\n");
    return 0;
}
