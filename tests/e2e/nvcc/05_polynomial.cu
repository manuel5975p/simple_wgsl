// 5. __device__ polynomial evaluation using Horner's method
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// Evaluate a0 + a1*x + a2*x^2 + a3*x^3 via Horner's method
__device__ float horner4(float x, float a0, float a1, float a2, float a3) {
    return a0 + x * (a1 + x * (a2 + x * a3));
}

// Approximate sin(x) for small x: x - x^3/6 + x^5/120
__device__ float fast_sin(float x) {
    float x2 = x * x;
    return x * horner4(x2, 1.0f, -1.0f / 6.0f, 1.0f / 120.0f, 0.0f);
}

// Approximate cos(x) for small x: 1 - x^2/2 + x^4/24
__device__ float fast_cos(float x) {
    float x2 = x * x;
    return horner4(x2, 1.0f, -0.5f, 1.0f / 24.0f, 0.0f);
}

__global__ void sincos_kernel(float *sin_out, float *cos_out,
                              const float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        sin_out[i] = fast_sin(x[i]);
        cos_out[i] = fast_cos(x[i]);
    }
}

int main() {
    const int N = 4096;
    size_t bytes = N * sizeof(float);

    float *h_x = (float *)malloc(bytes);
    float *h_sin = (float *)malloc(bytes);
    float *h_cos = (float *)malloc(bytes);
    for (int i = 0; i < N; i++)
        h_x[i] = (float)i * 0.001f - 2.0f;  // [-2, ~2]

    float *d_x, *d_sin, *d_cos;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_sin, bytes);
    cudaMalloc(&d_cos, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);

    sincos_kernel<<<(N + 255) / 256, 256>>>(d_sin, d_cos, d_x, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_sin, d_sin, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cos, d_cos, bytes, cudaMemcpyDeviceToHost);

    // Verify sin^2 + cos^2 ~ 1 (not exact due to Taylor approx)
    // Just verify the kernel produced the same as the host computation
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float x = h_x[i];
        float x2 = x * x;
        float ref_sin = x * (1.0f + x2 * (-1.0f / 6.0f + x2 * (1.0f / 120.0f)));
        float ref_cos = 1.0f + x2 * (-0.5f + x2 * (1.0f / 24.0f));
        if (fabsf(h_sin[i] - ref_sin) > 1e-5f ||
            fabsf(h_cos[i] - ref_cos) > 1e-5f) {
            fprintf(stderr, "FAIL: 05_polynomial at i=%d\n", i);
            errors++;
            if (errors > 5) break;
        }
    }

    cudaFree(d_x); cudaFree(d_sin); cudaFree(d_cos);
    free(h_x); free(h_sin); free(h_cos);

    if (errors) return 1;
    printf("PASS: 05_polynomial (N=%d)\n", N);
    return 0;
}
