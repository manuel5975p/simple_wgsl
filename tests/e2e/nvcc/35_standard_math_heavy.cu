// 35. Heavy standard math library: sinf, cosf, tanf, sqrtf, expf, logf,
//     powf, atan2f, fmodf, ceilf, floorf, fabsf, roundf, fmaxf, fminf
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__device__ float test_sin_cos(float x) {
    return sinf(x) * sinf(x) + cosf(x) * cosf(x); // should be ~1.0
}

__device__ float test_tan_identity(float x) {
    return tanf(x) - sinf(x) / cosf(x); // should be ~0.0
}

__device__ float test_exp_log(float x) {
    return logf(expf(x)); // should be ~x
}

__device__ float test_sqrt_square(float x) {
    float s = sqrtf(fabsf(x));
    return s * s; // should be ~|x|
}

__device__ float test_pow_chain(float x) {
    float a = powf(x, 2.0f);
    float b = sqrtf(a);
    return b; // should be ~|x|
}

__device__ float test_floor_ceil(float x) {
    return ceilf(x) - floorf(x); // 0.0 or 1.0
}

__device__ float test_fmod(float x) {
    return fmodf(fabsf(x) + 3.14f, 1.0f);
}

__device__ float test_atan2(float x) {
    float a = atan2f(sinf(x), cosf(x)); // should be ~x (mod 2pi)
    return sinf(a) - sinf(x); // should be ~0
}

__device__ float test_minmax(float a, float b) {
    return fmaxf(a, b) - fminf(a, b); // |a-b|
}

__global__ void standard_math_kernel(float *results, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = in[i];

    // Store 9 results per element
    int base = i * 9;
    results[base + 0] = test_sin_cos(x);
    results[base + 1] = test_tan_identity(x * 0.3f); // keep x small to avoid tan poles
    results[base + 2] = test_exp_log(x * 0.5f + 1.0f); // positive args
    results[base + 3] = test_sqrt_square(x);
    results[base + 4] = test_pow_chain(fabsf(x) + 0.01f);
    results[base + 5] = test_floor_ceil(x);
    results[base + 6] = test_fmod(x);
    results[base + 7] = test_atan2(x * 0.9f); // keep away from boundaries
    results[base + 8] = test_minmax(x, x * 0.5f + 1.0f);
}

int main() {
    const int N = 4096;
    const int RESULTS_PER = 9;
    size_t in_bytes = N * sizeof(float);
    size_t out_bytes = N * RESULTS_PER * sizeof(float);

    float *h_in = (float *)malloc(in_bytes);
    float *h_out = (float *)malloc(out_bytes);
    for (int i = 0; i < N; i++)
        h_in[i] = (float)i / (float)N * 4.0f - 2.0f; // [-2, 2)

    float *d_in, *d_out;
    cudaMalloc(&d_in, in_bytes);
    cudaMalloc(&d_out, out_bytes);
    cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice);

    standard_math_kernel<<<(N + 255) / 256, 256>>>(d_out, d_in, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float x = h_in[i];
        int base = i * RESULTS_PER;

        // sin^2+cos^2 == 1
        if (fabsf(h_out[base + 0] - 1.0f) > 1e-3f) {
            if (errors < 3)
                fprintf(stderr, "FAIL: sin^2+cos^2 at i=%d x=%f got %f\n", i, x, h_out[base+0]);
            errors++;
        }

        // tan(x)-sin(x)/cos(x) == 0
        float xtan = x * 0.3f;
        if (fabsf(h_out[base + 1]) > 1e-3f) {
            if (errors < 3)
                fprintf(stderr, "FAIL: tan identity at i=%d x=%f got %f\n", i, xtan, h_out[base+1]);
            errors++;
        }

        // log(exp(x)) == x
        float xel = x * 0.5f + 1.0f;
        if (fabsf(h_out[base + 2] - xel) > 1e-3f) {
            if (errors < 3)
                fprintf(stderr, "FAIL: exp/log at i=%d x=%f got %f expected %f\n", i, xel, h_out[base+2], xel);
            errors++;
        }

        // sqrt(|x|)^2 == |x|
        if (fabsf(h_out[base + 3] - fabsf(x)) > 1e-3f) {
            if (errors < 3)
                fprintf(stderr, "FAIL: sqrt^2 at i=%d x=%f got %f expected %f\n", i, x, h_out[base+3], fabsf(x));
            errors++;
        }

        // pow(|x|+0.01, 2)^0.5 == |x|+0.01
        float xpow = fabsf(x) + 0.01f;
        if (fabsf(h_out[base + 4] - xpow) > 1e-2f) {
            if (errors < 3)
                fprintf(stderr, "FAIL: pow chain at i=%d x=%f got %f expected %f\n", i, x, h_out[base+4], xpow);
            errors++;
        }

        // ceil-floor: 0 or 1
        float cf = h_out[base + 5];
        if (cf < -0.01f || cf > 1.01f) {
            if (errors < 3)
                fprintf(stderr, "FAIL: floor/ceil at i=%d x=%f got %f\n", i, x, cf);
            errors++;
        }

        // fmod: in [0, 1)
        float fm = h_out[base + 6];
        float exp_fm = fmodf(fabsf(x) + 3.14f, 1.0f);
        if (fabsf(fm - exp_fm) > 1e-3f) {
            if (errors < 3)
                fprintf(stderr, "FAIL: fmod at i=%d x=%f got %f expected %f\n", i, x, fm, exp_fm);
            errors++;
        }

        // atan2 identity
        float xa = x * 0.9f;
        if (fabsf(h_out[base + 7]) > 1e-3f) {
            if (errors < 3)
                fprintf(stderr, "FAIL: atan2 at i=%d x=%f got %f\n", i, xa, h_out[base+7]);
            errors++;
        }

        // max-min = |a-b|
        float a = x, b = x * 0.5f + 1.0f;
        float exp_mm = fabsf(a - b);
        if (fabsf(h_out[base + 8] - exp_mm) > 1e-4f) {
            if (errors < 3)
                fprintf(stderr, "FAIL: minmax at i=%d got %f expected %f\n", i, h_out[base+8], exp_mm);
            errors++;
        }
    }

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    if (errors) { fprintf(stderr, "FAIL: 35_standard_math_heavy %d errors\n", errors); return 1; }
    printf("PASS: 35_standard_math_heavy (N=%d, 9 checks each)\n", N);
    return 0;
}
