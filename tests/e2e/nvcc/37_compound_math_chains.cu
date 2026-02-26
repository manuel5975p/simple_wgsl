// 37. Compound math chains: sinf*cosf products, nested exp/log/pow,
//     Horner polynomials with trig, fmaf with sinf, copysignf, etc.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// sin(2x) = 2*sin(x)*cos(x)
__device__ float double_angle(float x) {
    float lhs = sinf(2.0f * x);
    float rhs = 2.0f * sinf(x) * cosf(x);
    return lhs - rhs; // should be ~0
}

// cos(2x) = cos^2(x) - sin^2(x)
__device__ float double_angle_cos(float x) {
    float lhs = cosf(2.0f * x);
    float c = cosf(x);
    float s = sinf(x);
    float rhs = c * c - s * s;
    return lhs - rhs; // should be ~0
}

// exp(a+b) = exp(a)*exp(b)
__device__ float exp_sum_product(float a, float b) {
    return expf(a + b) - expf(a) * expf(b); // ~0
}

// log(a*b) = log(a)+log(b) for a,b>0
__device__ float log_product_sum(float a, float b) {
    return logf(a * b) - (logf(a) + logf(b)); // ~0
}

// Horner polynomial with trig: p(x) = sin(x)^3 + 2*sin(x)^2 - sin(x) + 0.5
__device__ float trig_horner(float x) {
    float s = sinf(x);
    return ((s + 2.0f) * s - 1.0f) * s + 0.5f;
}

// fmaf with sinf: fma(sinf(x), cosf(x), sinf(x))
__device__ float fma_trig(float x) {
    return fmaf(sinf(x), cosf(x), sinf(x));
}

// Newton-Raphson step for sqrt using rsqrtf:
// x_{n+1} = x_n * (1.5 - 0.5*a*x_n^2) -- but just test rsqrtf directly
__device__ float test_rsqrt(float x) {
    float r = rsqrtf(x);
    return r * r * x; // should be ~1.0
}

// copysignf
__device__ float test_copysign(float x) {
    return copysignf(1.0f, x);
}

// ldexpf / scalbnf
__device__ float test_ldexp(float x) {
    return ldexpf(x, 3) - x * 8.0f; // should be ~0
}

__global__ void compound_kernel(float *results, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = in[i];

    int base = i * 9;
    results[base + 0] = double_angle(x);
    results[base + 1] = double_angle_cos(x);
    results[base + 2] = exp_sum_product(x * 0.3f, -x * 0.2f); // small args
    results[base + 3] = log_product_sum(fabsf(x) + 0.5f, fabsf(x) + 1.0f);
    results[base + 4] = trig_horner(x);
    results[base + 5] = fma_trig(x);
    results[base + 6] = test_rsqrt(fabsf(x) + 0.01f);
    results[base + 7] = test_copysign(x);
    results[base + 8] = test_ldexp(x * 0.1f);
}

int main() {
    const int N = 4096;
    const int RPE = 9;
    size_t in_bytes = N * sizeof(float);
    size_t out_bytes = N * RPE * sizeof(float);

    float *h_in = (float *)malloc(in_bytes);
    float *h_out = (float *)malloc(out_bytes);
    for (int i = 0; i < N; i++)
        h_in[i] = (float)(i - N/2) / (float)(N/4); // [-2, 2)

    float *d_in, *d_out;
    cudaMalloc(&d_in, in_bytes);
    cudaMalloc(&d_out, out_bytes);
    cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice);

    compound_kernel<<<(N + 255) / 256, 256>>>(d_out, d_in, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float x = h_in[i];
        int base = i * RPE;

        // sin(2x) identity
        if (fabsf(h_out[base + 0]) > 1e-3f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: sin(2x) i=%d x=%f got %e\n", i, x, h_out[base+0]);
            errors++;
        }
        // cos(2x) identity
        if (fabsf(h_out[base + 1]) > 1e-3f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: cos(2x) i=%d x=%f got %e\n", i, x, h_out[base+1]);
            errors++;
        }
        // exp(a+b) identity
        float ea = x * 0.3f, eb = -x * 0.2f;
        float exp_ref = expf(ea + eb) - expf(ea) * expf(eb);
        if (fabsf(h_out[base + 2] - exp_ref) > 1e-3f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: exp sum i=%d x=%f got %e ref %e\n", i, x, h_out[base+2], exp_ref);
            errors++;
        }
        // log(a*b) identity
        if (fabsf(h_out[base + 3]) > 1e-3f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: log prod i=%d x=%f got %e\n", i, x, h_out[base+3]);
            errors++;
        }
        // trig horner
        float s = sinf(x);
        float exp4 = ((s + 2.0f) * s - 1.0f) * s + 0.5f;
        if (fabsf(h_out[base + 4] - exp4) > 1e-3f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: horner i=%d x=%f got %f exp %f\n", i, x, h_out[base+4], exp4);
            errors++;
        }
        // fma_trig
        float exp5 = sinf(x) * cosf(x) + sinf(x);
        if (fabsf(h_out[base + 5] - exp5) > 1e-3f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: fma_trig i=%d x=%f got %f exp %f\n", i, x, h_out[base+5], exp5);
            errors++;
        }
        // rsqrt: r^2*x == 1
        if (fabsf(h_out[base + 6] - 1.0f) > 1e-2f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: rsqrt i=%d x=%f got %f\n", i, x, h_out[base+6]);
            errors++;
        }
        // copysign
        float exp7 = copysignf(1.0f, x);
        if (h_out[base + 7] != exp7) {
            if (errors < 5)
                fprintf(stderr, "FAIL: copysign i=%d x=%f got %f exp %f\n", i, x, h_out[base+7], exp7);
            errors++;
        }
        // ldexp
        if (fabsf(h_out[base + 8]) > 1e-4f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: ldexp i=%d x=%f got %e\n", i, x, h_out[base+8]);
            errors++;
        }
    }

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    if (errors) { fprintf(stderr, "FAIL: 37_compound_math_chains %d errors\n", errors); return 1; }
    printf("PASS: 37_compound_math_chains (N=%d, 9 checks each)\n", N);
    return 0;
}
