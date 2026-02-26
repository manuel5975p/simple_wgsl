// 17. Packed float2 operations: paired FMA, rcp, parallel processing
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__device__ void packed_add(float *r0, float *r1,
                           float a0, float a1, float b0, float b1) {
    *r0 = a0 + b0;
    *r1 = a1 + b1;
}

__device__ void packed_mul(float *r0, float *r1,
                           float a0, float a1, float b0, float b1) {
    *r0 = a0 * b0;
    *r1 = a1 * b1;
}

__device__ void packed_fma(float *r0, float *r1,
                           float a0, float a1, float b0, float b1,
                           float c0, float c1) {
    *r0 = __fmaf_rn(a0, b0, c0);
    *r1 = __fmaf_rn(a1, b1, c1);
}

__global__ void packed_ops(float *out, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * 2;
    if (idx + 1 < n) {
        float s0, s1, p0, p1, r0, r1;
        packed_add(&s0, &s1, a[idx], a[idx+1], b[idx], b[idx+1]);
        packed_mul(&p0, &p1, a[idx], a[idx+1], b[idx], b[idx+1]);
        packed_fma(&r0, &r1, p0, p1, 0.5f, 0.5f, s0, s1);
        out[idx] = r0;
        out[idx+1] = r1;
    }
}

int main() {
    const int N = 4096;
    size_t bytes = N * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)(i % 20) * 0.5f;
        h_b[i] = (float)(i % 15) * 0.3f;
    }

    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    packed_ops<<<(N/2 + 255) / 256, 256>>>(d_out, d_a, d_b, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float sum = h_a[i] + h_b[i];
        float prod = h_a[i] * h_b[i];
        float expected = prod * 0.5f + sum;
        if (fabsf(h_out[i] - expected) > 1e-3f) {
            if (errors == 0)
                fprintf(stderr, "FAIL: 17_half2_packed at i=%d got %f expected %f\n",
                        i, h_out[i], expected);
            errors++;
        }
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    free(h_a); free(h_b); free(h_out);

    if (errors) { fprintf(stderr, "FAIL: %d errors\n", errors); return 1; }
    printf("PASS: 17_half2_packed (N=%d)\n", N);
    return 0;
}
