// 11. Double-underscore math intrinsics: __fadd_rn, __fmul_rn, __fdiv_rn, __fsub_rn
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__device__ float intrinsic_chain(float a, float b) {
    float sum = __fadd_rn(a, b);
    float prod = __fmul_rn(sum, a);
    float diff = __fsub_rn(prod, b);
    return __fdiv_rn(diff, __fadd_rn(a, 1.0f));
}

__global__ void dunder_math(float *out, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = intrinsic_chain(a[i], b[i]);
}

int main() {
    const int N = 4096;
    size_t bytes = N * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)(i % 50) * 0.1f + 0.5f;
        h_b[i] = (float)(i % 30) * 0.2f - 1.0f;
    }

    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    dunder_math<<<(N + 255) / 256, 256>>>(d_out, d_a, d_b, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float a = h_a[i], b = h_b[i];
        float expected = ((a + b) * a - b) / (a + 1.0f);
        if (fabsf(h_out[i] - expected) > 1e-4f) {
            if (errors == 0)
                fprintf(stderr, "FAIL: 11_double_underscore_math at i=%d got %f expected %f\n",
                        i, h_out[i], expected);
            errors++;
        }
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    free(h_a); free(h_b); free(h_out);

    if (errors) { fprintf(stderr, "FAIL: %d errors\n", errors); return 1; }
    printf("PASS: 11_double_underscore_math (N=%d)\n", N);
    return 0;
}
