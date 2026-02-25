// 10. Matrix multiply with 2D grid, __device__ dot product helper
//     C = A * B where A is MxK, B is KxN, C is MxN
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__device__ float row_dot_col(const float *A, const float *B,
                             int row, int col, int K, int N) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    return sum;
}

__global__ void matmul(float *C, const float *A, const float *B,
                       int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        C[row * N + col] = row_dot_col(A, B, row, col, K, N);
    }
}

int main() {
    const int M = 32, K = 64, N = 48;
    size_t a_bytes = M * K * sizeof(float);
    size_t b_bytes = K * N * sizeof(float);
    size_t c_bytes = M * N * sizeof(float);

    float *h_a = (float *)malloc(a_bytes);
    float *h_b = (float *)malloc(b_bytes);
    float *h_c = (float *)malloc(c_bytes);
    float *h_ref = (float *)malloc(c_bytes);

    // Initialize with small values to avoid float precision issues
    for (int i = 0; i < M * K; i++)
        h_a[i] = (float)((i % 10) - 5) * 0.1f;
    for (int i = 0; i < K * N; i++)
        h_b[i] = (float)((i % 7) - 3) * 0.1f;

    // Reference matmul on host
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += h_a[i * K + k] * h_b[k * N + j];
            h_ref[i * N + j] = sum;
        }
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, a_bytes);
    cudaMalloc(&d_b, b_bytes);
    cudaMalloc(&d_c, c_bytes);
    cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul<<<grid, block>>>(d_c, d_a, d_b, M, K, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    float max_err = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(h_c[i] - h_ref[i]);
        if (err > max_err) max_err = err;
        if (err > 1e-3f) {
            if (errors == 0)
                fprintf(stderr, "FAIL: 10_matmul at [%d,%d] got %f expected %f\n",
                        i / N, i % N, h_c[i], h_ref[i]);
            errors++;
        }
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c); free(h_ref);

    if (errors) {
        fprintf(stderr, "FAIL: 10_matmul %d errors, max_err=%e\n",
                errors, max_err);
        return 1;
    }
    printf("PASS: 10_matmul (%dx%d * %dx%d, max_err=%e)\n",
           M, K, K, N, max_err);
    return 0;
}
