// 41. cuFFT basic operations: C2C, R2C, roundtrip, mixed with kernels
//     Tests forward/inverse transforms and interop with kernel launches
//     and memory operations. Compiled by nvcc, linked against system cuFFT,
//     runs with our Vulkan libcuda.so.1 replacement.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CHECK_CUDA(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %d (%s) at %s:%d\n", \
                e, cudaGetErrorString(e), __FILE__, __LINE__); \
        return 1; \
    } \
} while(0)

#define CHECK_CUFFT(call) do { \
    cufftResult r = (call); \
    if (r != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error %d at %s:%d\n", r, __FILE__, __LINE__); \
        return 1; \
    } \
} while(0)

static int check_close_f(const char *label, float got, float expected,
                          float tol) {
    float diff = fabsf(got - expected);
    if (diff > tol) {
        fprintf(stderr, "FAIL %s: got %f expected %f (diff %e)\n",
                label, got, expected, diff);
        return 1;
    }
    return 0;
}

// Kernel: zero out frequency bins outside [0..cutoff) and [n-cutoff..n)
__global__ void lowpass_filter(cufftComplex *data, int n, int cutoff) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cutoff && idx < n - cutoff) {
        data[idx].x = 0.0f;
        data[idx].y = 0.0f;
    }
}

// Kernel: scale complex array by a real factor
__global__ void scale_complex(cufftComplex *data, float s, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx].x *= s;
        data[idx].y *= s;
    }
}

// Test 1: Forward C2C of impulse -> flat spectrum
// Input: (1,0), (0,0), ..., (0,0)
// Expected output: all bins = (1, 0)
static int test_c2c_impulse() {
    const int N = 64;
    cufftComplex h_data[N];
    memset(h_data, 0, sizeof(h_data));
    h_data[0].x = 1.0f;

    cufftComplex *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(cufftComplex),
                          cudaMemcpyHostToDevice));

    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, 1));
    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));

    cufftComplex h_result[N];
    CHECK_CUDA(cudaMemcpy(h_result, d_data, N * sizeof(cufftComplex),
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) {
        if (check_close_f("impulse_re", h_result[i].x, 1.0f, 1e-5f))
            return 1;
        if (check_close_f("impulse_im", h_result[i].y, 0.0f, 1e-5f))
            return 1;
    }

    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_data));
    return 0;
}

// Test 2: Forward + inverse C2C roundtrip
// cuFFT does NOT normalize: after fwd+inv, result = N * original
static int test_c2c_roundtrip() {
    const int N = 128;
    cufftComplex h_data[N];
    for (int i = 0; i < N; i++) {
        h_data[i].x = sinf(2.0f * (float)M_PI * 3.0f * i / N);
        h_data[i].y = cosf(2.0f * (float)M_PI * 7.0f * i / N);
    }

    cufftComplex *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(cufftComplex),
                          cudaMemcpyHostToDevice));

    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, 1));
    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));

    cufftComplex h_result[N];
    CHECK_CUDA(cudaMemcpy(h_result, d_data, N * sizeof(cufftComplex),
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) {
        float exp_re = h_data[i].x * N;
        float exp_im = h_data[i].y * N;
        if (check_close_f("roundtrip_re", h_result[i].x, exp_re, 0.1f))
            return 1;
        if (check_close_f("roundtrip_im", h_result[i].y, exp_im, 0.1f))
            return 1;
    }

    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_data));
    return 0;
}

// Test 3: R2C of cosine wave -> peak at known frequency bin
// cos(2*pi*k0*n/N) -> peak magnitude N/2 at bin k0
static int test_r2c_cosine() {
    const int N = 256;
    const int k0 = 10;
    float h_real[N];
    for (int i = 0; i < N; i++)
        h_real[i] = cosf(2.0f * (float)M_PI * k0 * i / N);

    float *d_real;
    cufftComplex *d_complex;
    CHECK_CUDA(cudaMalloc(&d_real, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_complex, (N / 2 + 1) * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMemcpy(d_real, h_real, N * sizeof(float),
                          cudaMemcpyHostToDevice));

    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, N, CUFFT_R2C, 1));
    CHECK_CUFFT(cufftExecR2C(plan, d_real, d_complex));

    cufftComplex h_result[N / 2 + 1];
    CHECK_CUDA(cudaMemcpy(h_result, d_complex,
                          (N / 2 + 1) * sizeof(cufftComplex),
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i <= N / 2; i++) {
        float mag = sqrtf(h_result[i].x * h_result[i].x +
                          h_result[i].y * h_result[i].y);
        if (i == k0) {
            if (check_close_f("r2c_peak", mag, (float)(N / 2), 0.5f))
                return 1;
        } else {
            if (check_close_f("r2c_noise", mag, 0.0f, 0.5f))
                return 1;
        }
    }

    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_real));
    CHECK_CUDA(cudaFree(d_complex));
    return 0;
}

// Test 4: FFT -> kernel low-pass filter -> inverse FFT
// Signal = cos(3*2pi*n/N) + cos(50*2pi*n/N)
// After zeroing bins [cutoff..N-cutoff), only the low-freq component remains
static int test_fft_then_kernel() {
    const int N = 256;
    const int cutoff = 8;
    cufftComplex h_data[N];
    for (int i = 0; i < N; i++) {
        h_data[i].x = cosf(2.0f * (float)M_PI * 3.0f * i / N)
                     + cosf(2.0f * (float)M_PI * 50.0f * i / N);
        h_data[i].y = 0.0f;
    }

    cufftComplex *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(cufftComplex),
                          cudaMemcpyHostToDevice));

    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, 1));
    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    lowpass_filter<<<blocks, threads>>>(d_data, N, cutoff);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));

    cufftComplex h_result[N];
    CHECK_CUDA(cudaMemcpy(h_result, d_data, N * sizeof(cufftComplex),
                          cudaMemcpyDeviceToHost));

    // Only low-freq component survives: N * cos(2pi*3*n/N)
    for (int i = 0; i < N; i++) {
        float expected = (float)N * cosf(2.0f * (float)M_PI * 3.0f * i / N);
        if (check_close_f("lowpass", h_result[i].x, expected, 1.0f))
            return 1;
    }

    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_data));
    return 0;
}

// Test 5: Full pipeline
// cudaMalloc -> cudaMemcpy H2D -> FFT fwd -> scale kernel -> FFT inv
// -> cudaMemcpy D2H -> verify
// Scaling in frequency domain by k multiplies time-domain by k
static int test_full_pipeline() {
    const int N = 512;
    const float scale = 2.0f;

    cufftComplex *h_input =
        (cufftComplex *)malloc(N * sizeof(cufftComplex));
    if (!h_input) { fprintf(stderr, "malloc failed\n"); return 1; }
    for (int i = 0; i < N; i++) {
        h_input[i].x = sinf(2.0f * (float)M_PI * 5.0f * i / N);
        h_input[i].y = 0.0f;
    }

    cufftComplex *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMemcpy(d_data, h_input, N * sizeof(cufftComplex),
                          cudaMemcpyHostToDevice));

    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, 1));
    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    scale_complex<<<blocks, threads>>>(d_data, scale, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));

    cufftComplex *h_output =
        (cufftComplex *)malloc(N * sizeof(cufftComplex));
    if (!h_output) {
        free(h_input);
        fprintf(stderr, "malloc failed\n");
        return 1;
    }
    CHECK_CUDA(cudaMemcpy(h_output, d_data, N * sizeof(cufftComplex),
                          cudaMemcpyDeviceToHost));

    // result = N * scale * original
    for (int i = 0; i < N; i++) {
        float expected = (float)N * scale * h_input[i].x;
        if (check_close_f("pipeline_re", h_output[i].x, expected, 1.0f)) {
            free(h_input); free(h_output);
            return 1;
        }
    }

    free(h_input);
    free(h_output);
    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_data));
    return 0;
}

int main() {
    struct { const char *name; int (*fn)(); } tests[] = {
        {"c2c_impulse",     test_c2c_impulse},
        {"c2c_roundtrip",   test_c2c_roundtrip},
        {"r2c_cosine",      test_r2c_cosine},
        {"fft_then_kernel", test_fft_then_kernel},
        {"full_pipeline",   test_full_pipeline},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int passed = 0, failed = 0;

    for (int i = 0; i < n; i++) {
        printf("  [%d/%d] %-25s ... ", i + 1, n, tests[i].name);
        fflush(stdout);
        int r = tests[i].fn();
        if (r == 0) {
            printf("PASS\n");
            passed++;
        } else {
            printf("FAIL\n");
            failed++;
        }
    }

    printf("\nResults: %d passed, %d failed out of %d\n",
           passed, failed, n);
    if (failed) {
        fprintf(stderr, "FAIL: 41_cufft_basic\n");
        return 1;
    }
    printf("PASS: 41_cufft_basic (all %d tests)\n", n);
    return 0;
}
