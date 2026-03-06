/*
 * test_cufft_features.cu - Test 2D FFT, R2C, and C2R through cuFFT API
 *
 * Compile: nvcc -O2 -o test_cufft_features bench/test_cufft_features.cu -lcufft
 * Run with real cuFFT:  ./test_cufft_features
 * Run with Vulkan shim: LD_LIBRARY_PATH=build/cuvk_runtime ./test_cufft_features
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
        return r; \
    } \
} while(0)

/* ========================================================================== */
/* Test 1: 1D R2C — impulse response                                         */
/* ========================================================================== */

static int test_r2c_impulse(int n) {
    /* Input: delta at 0 → all frequency bins should be (1, 0) */
    float *h_in = (float *)calloc(n, sizeof(float));
    h_in[0] = 1.0f;

    float *d_in;
    CHECK_CUDA(cudaMalloc(&d_in, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));

    int out_n = n / 2 + 1;
    cufftComplex *d_out;
    CHECK_CUDA(cudaMalloc(&d_out, out_n * sizeof(cufftComplex)));

    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, n, CUFFT_R2C, 1));
    CHECK_CUFFT(cufftExecR2C(plan, d_in, d_out));
    CHECK_CUDA(cudaDeviceSynchronize());

    cufftComplex *h_out = (cufftComplex *)malloc(out_n * sizeof(cufftComplex));
    CHECK_CUDA(cudaMemcpy(h_out, d_out, out_n * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < out_n; i++) {
        float err_re = fabsf(h_out[i].x - 1.0f);
        float err_im = fabsf(h_out[i].y);
        if (err_re > max_err) max_err = err_re;
        if (err_im > max_err) max_err = err_im;
    }

    const char *status = (max_err < 1e-3f) ? "OK" : "FAIL";
    printf("  R2C impulse N=%-6d %s  max_err=%.2e\n", n, status, max_err);

    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    free(h_in); free(h_out);
    return (max_err < 1e-3f) ? 0 : 1;
}

/* ========================================================================== */
/* Test 2: 1D R2C→C2R roundtrip                                              */
/* ========================================================================== */

static int test_r2c_c2r_roundtrip(int n) {
    /* Forward R2C then inverse C2R should give N * original */
    float *h_in = (float *)malloc(n * sizeof(float));
    srand(42);
    for (int i = 0; i < n; i++)
        h_in[i] = (float)(rand() % 1000) / 500.0f - 1.0f;

    float *d_real;
    CHECK_CUDA(cudaMalloc(&d_real, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_real, h_in, n * sizeof(float), cudaMemcpyHostToDevice));

    int out_n = n / 2 + 1;
    cufftComplex *d_complex;
    CHECK_CUDA(cudaMalloc(&d_complex, out_n * sizeof(cufftComplex)));

    float *d_real_out;
    CHECK_CUDA(cudaMalloc(&d_real_out, n * sizeof(float)));

    cufftHandle plan_r2c, plan_c2r;
    CHECK_CUFFT(cufftPlan1d(&plan_r2c, n, CUFFT_R2C, 1));
    CHECK_CUFFT(cufftPlan1d(&plan_c2r, n, CUFFT_C2R, 1));

    CHECK_CUFFT(cufftExecR2C(plan_r2c, d_real, d_complex));
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUFFT(cufftExecC2R(plan_c2r, d_complex, d_real_out));
    CHECK_CUDA(cudaDeviceSynchronize());

    float *h_out = (float *)malloc(n * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_out, d_real_out, n * sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float expected = h_in[i] * n;  /* cuFFT doesn't normalize */
        float err = fabsf(h_out[i] - expected);
        if (err > max_err) max_err = err;
    }

    const char *status = (max_err < 1.0f) ? "OK" : "FAIL";
    printf("  R2C→C2R roundtrip N=%-6d %s  max_err=%.2e\n", n, status, max_err);

    CHECK_CUFFT(cufftDestroy(plan_r2c));
    CHECK_CUFFT(cufftDestroy(plan_c2r));
    CHECK_CUDA(cudaFree(d_real));
    CHECK_CUDA(cudaFree(d_complex));
    CHECK_CUDA(cudaFree(d_real_out));
    free(h_in); free(h_out);
    return (max_err < 1.0f) ? 0 : 1;
}

/* ========================================================================== */
/* Test 3: 2D C2C — impulse response                                         */
/* ========================================================================== */

static int test_2d_impulse(int nx, int ny) {
    int total = nx * ny;
    cufftComplex *h_in = (cufftComplex *)calloc(total, sizeof(cufftComplex));
    h_in[0].x = 1.0f;

    cufftComplex *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, total * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMemcpy(d_data, h_in, total * sizeof(cufftComplex), cudaMemcpyHostToDevice));

    cufftHandle plan;
    CHECK_CUFFT(cufftPlan2d(&plan, nx, ny, CUFFT_C2C));
    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    CHECK_CUDA(cudaDeviceSynchronize());

    cufftComplex *h_out = (cufftComplex *)malloc(total * sizeof(cufftComplex));
    CHECK_CUDA(cudaMemcpy(h_out, d_data, total * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

    /* Impulse at (0,0) → all bins should be (1,0) */
    float max_err = 0.0f;
    for (int i = 0; i < total; i++) {
        float err_re = fabsf(h_out[i].x - 1.0f);
        float err_im = fabsf(h_out[i].y);
        if (err_re > max_err) max_err = err_re;
        if (err_im > max_err) max_err = err_im;
    }

    const char *status = (max_err < 1e-3f) ? "OK" : "FAIL";
    printf("  2D impulse %dx%-6d %s  max_err=%.2e\n", nx, ny, status, max_err);

    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_data));
    free(h_in); free(h_out);
    return (max_err < 1e-3f) ? 0 : 1;
}

/* ========================================================================== */
/* Test 4: 2D C2C — roundtrip                                                */
/* ========================================================================== */

static int test_2d_roundtrip(int nx, int ny) {
    int total = nx * ny;
    cufftComplex *h_in = (cufftComplex *)malloc(total * sizeof(cufftComplex));
    srand(123);
    for (int i = 0; i < total; i++) {
        h_in[i].x = (float)(rand() % 1000) / 500.0f - 1.0f;
        h_in[i].y = (float)(rand() % 1000) / 500.0f - 1.0f;
    }

    cufftComplex *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, total * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMemcpy(d_data, h_in, total * sizeof(cufftComplex), cudaMemcpyHostToDevice));

    cufftHandle plan;
    CHECK_CUFFT(cufftPlan2d(&plan, nx, ny, CUFFT_C2C));
    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));
    CHECK_CUDA(cudaDeviceSynchronize());

    cufftComplex *h_out = (cufftComplex *)malloc(total * sizeof(cufftComplex));
    CHECK_CUDA(cudaMemcpy(h_out, d_data, total * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < total; i++) {
        float err_re = fabsf(h_out[i].x - h_in[i].x * total);
        float err_im = fabsf(h_out[i].y - h_in[i].y * total);
        if (err_re > max_err) max_err = err_re;
        if (err_im > max_err) max_err = err_im;
    }

    const char *status = (max_err < 1.0f) ? "OK" : "FAIL";
    printf("  2D roundtrip %dx%-6d %s  max_err=%.2e\n", nx, ny, status, max_err);

    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_data));
    free(h_in); free(h_out);
    return (max_err < 1.0f) ? 0 : 1;
}

/* ========================================================================== */
/* Main                                                                       */
/* ========================================================================== */

int main() {
    const char *ld_path = getenv("LD_LIBRARY_PATH");
    int is_shim = (ld_path && strstr(ld_path, "cuvk_runtime"));

    int dev;
    CHECK_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    printf("================================================================\n");
    printf("  cuFFT Feature Tests (%s)\n",
           is_shim ? "CUVK Vulkan shim" : "Real NVIDIA cuFFT");
    printf("  Device: %s\n", prop.name);
    printf("================================================================\n\n");

    int failures = 0;

    printf("--- 1D R2C impulse ---\n");
    failures += test_r2c_impulse(8);
    failures += test_r2c_impulse(32);
    failures += test_r2c_impulse(64);
    failures += test_r2c_impulse(256);
    failures += test_r2c_impulse(1024);

    printf("\n--- 1D R2C→C2R roundtrip ---\n");
    failures += test_r2c_c2r_roundtrip(8);
    failures += test_r2c_c2r_roundtrip(32);
    failures += test_r2c_c2r_roundtrip(64);
    failures += test_r2c_c2r_roundtrip(256);
    failures += test_r2c_c2r_roundtrip(1024);

    printf("\n--- 2D C2C impulse ---\n");
    failures += test_2d_impulse(4, 4);
    failures += test_2d_impulse(8, 8);
    failures += test_2d_impulse(16, 16);
    failures += test_2d_impulse(32, 32);
    failures += test_2d_impulse(8, 16);
    failures += test_2d_impulse(16, 8);

    printf("\n--- 2D C2C roundtrip ---\n");
    failures += test_2d_roundtrip(4, 4);
    failures += test_2d_roundtrip(8, 8);
    failures += test_2d_roundtrip(16, 16);
    failures += test_2d_roundtrip(32, 32);
    failures += test_2d_roundtrip(8, 16);
    failures += test_2d_roundtrip(16, 8);

    printf("\n  Results: %d failures\n\n", failures);
    return failures;
}
