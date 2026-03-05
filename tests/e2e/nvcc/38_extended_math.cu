// 38. Extended math: acosf, asinf, atanf, sinhf, coshf, tanhf, cbrtf,
//     hypotf, log2f, log10f, exp2f, exp10f, truncf, nearbyintf
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// asin(sin(x)) == x for x in [-pi/2, pi/2]
__device__ float test_asin_roundtrip(float x) {
    return asinf(sinf(x));
}

// acos(cos(x)) == x for x in [0, pi]
__device__ float test_acos_roundtrip(float x) {
    return acosf(cosf(x));
}

// atan(tan(x)) == x for x in (-pi/2, pi/2)
__device__ float test_atan_roundtrip(float x) {
    return atanf(tanf(x));
}

// sinh(x) = (exp(x)-exp(-x))/2
__device__ float test_sinh_identity(float x) {
    return sinhf(x) - (expf(x) - expf(-x)) * 0.5f; // ~0
}

// cosh^2(x) - sinh^2(x) = 1
__device__ float test_hyp_pythag(float x) {
    float ch = coshf(x);
    float sh = sinhf(x);
    return ch * ch - sh * sh; // ~1
}

// tanh(x) = sinh(x)/cosh(x)
__device__ float test_tanh_ratio(float x) {
    return tanhf(x) - sinhf(x) / coshf(x); // ~0
}

// cbrt(x^3) == x
__device__ float test_cbrt(float x) {
    return cbrtf(x * x * x);
}

// hypot(a,b) == sqrt(a^2+b^2)
__device__ float test_hypot(float a, float b) {
    return hypotf(a, b) - sqrtf(a * a + b * b); // ~0
}

// log2(2^x) == x
__device__ float test_log2(float x) {
    return log2f(exp2f(x)); // ~x
}

// log10(10^x) -- use log10f(powf(10,x))
__device__ float test_log10(float x) {
    return log10f(powf(10.0f, x)); // ~x
}

// truncf and nearbyintf
__device__ float test_trunc(float x) {
    return truncf(x);
}

__device__ float test_nearbyint(float x) {
    return nearbyintf(x);
}

__global__ void extended_math_kernel(float *results, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = in[i];

    int base = i * 12;
    results[base + 0] = test_asin_roundtrip(x * 0.45f); // keep in [-pi/2+eps, pi/2-eps]
    results[base + 1] = test_acos_roundtrip(fabsf(x) * 0.45f + 0.1f); // keep in [0.1, ~1.0]
    results[base + 2] = test_atan_roundtrip(x * 0.45f);
    results[base + 3] = test_sinh_identity(x * 0.5f);
    results[base + 4] = test_hyp_pythag(x * 0.5f);
    results[base + 5] = test_tanh_ratio(x * 0.5f);
    results[base + 6] = test_cbrt(x);
    results[base + 7] = test_hypot(x, x * 0.5f + 1.0f);
    results[base + 8] = test_log2(x * 0.5f + 2.0f); // positive
    results[base + 9] = test_log10(fabsf(x) * 0.3f + 0.1f);
    results[base + 10] = test_trunc(x * 3.7f);
    results[base + 11] = test_nearbyint(x * 3.7f);
}

int main() {
    const int N = 4096;
    const int RPE = 12;
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

    extended_math_kernel<<<(N + 255) / 256, 256>>>(d_out, d_in, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float x = h_in[i];
        int base = i * RPE;

        // asin roundtrip
        float xa = x * 0.45f;
        if (fabsf(h_out[base + 0] - xa) > 1e-3f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: asin rt i=%d x=%f got %f exp %f\n", i, xa, h_out[base+0], xa);
            errors++;
        }
        // acos roundtrip
        float xac = fabsf(x) * 0.45f + 0.1f;
        if (fabsf(h_out[base + 1] - xac) > 1e-2f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: acos rt i=%d x=%f got %f exp %f\n", i, xac, h_out[base+1], xac);
            errors++;
        }
        // atan roundtrip
        float xat = x * 0.45f;
        if (fabsf(h_out[base + 2] - xat) > 1e-3f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: atan rt i=%d x=%f got %f exp %f\n", i, xat, h_out[base+2], xat);
            errors++;
        }
        // sinh identity ~0
        if (fabsf(h_out[base + 3]) > 1e-3f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: sinh i=%d x=%f got %e\n", i, x, h_out[base+3]);
            errors++;
        }
        // cosh^2-sinh^2 == 1
        if (fabsf(h_out[base + 4] - 1.0f) > 1e-2f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: hyp pyth i=%d x=%f got %f\n", i, x, h_out[base+4]);
            errors++;
        }
        // tanh ratio ~0
        if (fabsf(h_out[base + 5]) > 1e-3f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: tanh i=%d x=%f got %e\n", i, x, h_out[base+5]);
            errors++;
        }
        // cbrt(x^3) == x
        if (fabsf(h_out[base + 6] - x) > 1e-2f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: cbrt i=%d x=%f got %f\n", i, x, h_out[base+6]);
            errors++;
        }
        // hypot ~0 diff
        if (fabsf(h_out[base + 7]) > 1e-3f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: hypot i=%d x=%f got %e\n", i, x, h_out[base+7]);
            errors++;
        }
        // log2 roundtrip
        float xl = x * 0.5f + 2.0f;
        if (fabsf(h_out[base + 8] - xl) > 1e-3f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: log2 i=%d x=%f got %f exp %f\n", i, x, h_out[base+8], xl);
            errors++;
        }
        // log10 roundtrip
        float xl10 = fabsf(x) * 0.3f + 0.1f;
        if (fabsf(h_out[base + 9] - xl10) > 1e-2f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: log10 i=%d x=%f got %f exp %f\n", i, x, h_out[base+9], xl10);
            errors++;
        }
        // trunc
        float xtr = x * 3.7f;
        float exp_tr = truncf(xtr);
        if (h_out[base + 10] != exp_tr) {
            if (errors < 5)
                fprintf(stderr, "FAIL: trunc i=%d x=%f got %f exp %f\n", i, xtr, h_out[base+10], exp_tr);
            errors++;
        }
        // nearbyint
        float exp_ni = nearbyintf(xtr);
        if (h_out[base + 11] != exp_ni) {
            if (errors < 5)
                fprintf(stderr, "FAIL: nearbyint i=%d x=%f got %f exp %f\n", i, xtr, h_out[base+11], exp_ni);
            errors++;
        }
    }

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    if (errors) { fprintf(stderr, "FAIL: 38_extended_math %d errors\n", errors); return 1; }
    printf("PASS: 38_extended_math (N=%d, 12 checks each)\n", N);
    return 0;
}
