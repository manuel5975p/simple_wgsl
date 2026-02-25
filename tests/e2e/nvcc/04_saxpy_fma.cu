// 4. SAXPY with __device__ fused multiply-add, larger workload
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__device__ float fma_op(float a, float x, float y) {
    return __fmaf_rn(a, x, y);
}

__global__ void saxpy(float *y, const float *x, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = fma_op(a, x[i], y[i]);
    }
}

int main() {
    const int N = 8192;
    const float A = 3.14159f;
    size_t bytes = N * sizeof(float);

    float *h_x = (float *)malloc(bytes);
    float *h_y = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_x[i] = (float)(i % 100) * 0.01f;
        h_y[i] = (float)(i % 50) * 0.02f;
        h_ref[i] = A * h_x[i] + h_y[i];
    }

    float *d_x, *d_y;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);

    saxpy<<<(N + 255) / 256, 256>>>(d_y, d_x, A, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_y[i] - h_ref[i]) > 1e-4f) {
            fprintf(stderr, "FAIL: 04_saxpy_fma at i=%d: got %f expected %f\n",
                    i, h_y[i], h_ref[i]);
            errors++;
            if (errors > 5) break;
        }
    }

    cudaFree(d_x); cudaFree(d_y);
    free(h_x); free(h_y); free(h_ref);

    if (errors) return 1;
    printf("PASS: 04_saxpy_fma (N=%d)\n", N);
    return 0;
}
