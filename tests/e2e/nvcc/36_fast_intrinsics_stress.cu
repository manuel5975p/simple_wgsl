// 36. Stress test for __sinf, __cosf, __tanf, __expf, __logf, __powf
//     with heavy chaining and composition
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// Chain: __sinf(__cosf(__sinf(x)))
__device__ float triple_chain(float x) {
    return __sinf(__cosf(__sinf(x)));
}

// __expf(__logf(x)) == x for x>0
__device__ float exp_log_roundtrip(float x) {
    return __expf(__logf(x));
}

// Fourier-like sum: sum_k sin(k*x)/k for k=1..8
__device__ float fourier_sum(float x) {
    float s = 0.0f;
    s += __sinf(1.0f * x) / 1.0f;
    s += __sinf(2.0f * x) / 2.0f;
    s += __sinf(3.0f * x) / 3.0f;
    s += __sinf(4.0f * x) / 4.0f;
    s += __sinf(5.0f * x) / 5.0f;
    s += __sinf(6.0f * x) / 6.0f;
    s += __sinf(7.0f * x) / 7.0f;
    s += __sinf(8.0f * x) / 8.0f;
    return s;
}

// Damped oscillator: __expf(-a*t) * __sinf(w*t)
__device__ float damped_osc(float t, float a, float w) {
    return __expf(-a * t) * __sinf(w * t);
}

// __sinf^2 + __cosf^2 identity
__device__ float pythagorean(float x) {
    float s = __sinf(x);
    float c = __cosf(x);
    return s * s + c * c;
}

// __logf(__powf(a, b)) ~= b * __logf(a) for a>0
__device__ float log_pow_identity(float a, float b) {
    return __logf(__powf(a, b)) - b * __logf(a);
}

// Sigmoid: 1/(1+exp(-x)) using __expf
__device__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// Softmax-like: __expf(x) / (__expf(x) + __expf(y))
__device__ float softmax2(float x, float y) {
    float ex = __expf(x);
    float ey = __expf(y);
    return ex / (ex + ey);
}

__global__ void fast_stress_kernel(float *results, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = in[i];

    int base = i * 8;
    results[base + 0] = triple_chain(x);
    results[base + 1] = exp_log_roundtrip(fabsf(x) + 0.1f);
    results[base + 2] = fourier_sum(x);
    results[base + 3] = damped_osc(fabsf(x), 0.5f, 3.0f);
    results[base + 4] = pythagorean(x);
    results[base + 5] = log_pow_identity(fabsf(x) + 1.0f, 2.5f);
    results[base + 6] = fast_sigmoid(x);
    results[base + 7] = softmax2(x, -x);
}

int main() {
    const int N = 8192;
    const int RPE = 8;
    size_t in_bytes = N * sizeof(float);
    size_t out_bytes = N * RPE * sizeof(float);

    float *h_in = (float *)malloc(in_bytes);
    float *h_out = (float *)malloc(out_bytes);
    for (int i = 0; i < N; i++)
        h_in[i] = (float)i / (float)N * 6.28f - 3.14f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, in_bytes);
    cudaMalloc(&d_out, out_bytes);
    cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice);

    fast_stress_kernel<<<(N + 255) / 256, 256>>>(d_out, d_in, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float x = h_in[i];
        int base = i * RPE;

        // triple_chain: compare against host sinf(cosf(sinf(x)))
        float exp0 = sinf(cosf(sinf(x)));
        if (fabsf(h_out[base + 0] - exp0) > 0.1f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: triple_chain i=%d x=%f got %f exp %f\n", i, x, h_out[base+0], exp0);
            errors++;
        }

        // exp_log_roundtrip: should be ~(|x|+0.1)
        float xel = fabsf(x) + 0.1f;
        if (fabsf(h_out[base + 1] - xel) > 0.05f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: exp_log i=%d x=%f got %f exp %f\n", i, x, h_out[base+1], xel);
            errors++;
        }

        // fourier sum: compare against host
        float exp2 = 0.0f;
        for (int k = 1; k <= 8; k++)
            exp2 += sinf((float)k * x) / (float)k;
        if (fabsf(h_out[base + 2] - exp2) > 0.2f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: fourier i=%d x=%f got %f exp %f\n", i, x, h_out[base+2], exp2);
            errors++;
        }

        // damped oscillator
        float t = fabsf(x);
        float exp3 = expf(-0.5f * t) * sinf(3.0f * t);
        if (fabsf(h_out[base + 3] - exp3) > 0.1f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: damped_osc i=%d x=%f got %f exp %f\n", i, x, h_out[base+3], exp3);
            errors++;
        }

        // pythagorean identity: should be 1.0
        if (fabsf(h_out[base + 4] - 1.0f) > 1e-2f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: sin^2+cos^2 i=%d x=%f got %f\n", i, x, h_out[base+4]);
            errors++;
        }

        // log_pow identity: should be ~0
        if (fabsf(h_out[base + 5]) > 0.15f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: log_pow i=%d x=%f got %f\n", i, x, h_out[base+5]);
            errors++;
        }

        // sigmoid: compare against host
        float exp6 = 1.0f / (1.0f + expf(-x));
        if (fabsf(h_out[base + 6] - exp6) > 0.05f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: sigmoid i=%d x=%f got %f exp %f\n", i, x, h_out[base+6], exp6);
            errors++;
        }

        // softmax2: compare against host
        float exp7 = expf(x) / (expf(x) + expf(-x));
        if (fabsf(h_out[base + 7] - exp7) > 0.05f) {
            if (errors < 5)
                fprintf(stderr, "FAIL: softmax2 i=%d x=%f got %f exp %f\n", i, x, h_out[base+7], exp7);
            errors++;
        }
    }

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    if (errors) { fprintf(stderr, "FAIL: 36_fast_intrinsics_stress %d errors\n", errors); return 1; }
    printf("PASS: 36_fast_intrinsics_stress (N=%d, 8 checks each)\n", N);
    return 0;
}
