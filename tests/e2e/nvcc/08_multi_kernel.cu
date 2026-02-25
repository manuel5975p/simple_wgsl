// 8. Multiple __global__ kernels in a pipeline
//    Step 1: generate data, Step 2: transform, Step 3: reduce partial sums
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// Kernel 1: linear transform  out = weight * in + bias
__global__ void linear(float *out, const float *in,
                       float weight, float bias, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = weight * in[i] + bias;
}

// Kernel 2: activation  out[i] = sigmoid(in[i])
__global__ void activate_sigmoid(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = sigmoid(data[i]);
}

// Kernel 3: element-wise product  out[i] = a[i] * b[i]
__global__ void elementwise_mul(float *out, const float *a,
                                const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

// Kernel 4: relu activation
__global__ void activate_relu(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = relu(data[i]);
}

int main() {
    const int N = 2048;
    const float W1 = 0.5f, B1 = -0.3f;
    const float W2 = -1.2f, B2 = 0.7f;
    size_t bytes = N * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int i = 0; i < N; i++)
        h_in[i] = (float)(i % 100) * 0.04f - 2.0f;  // [-2, 1.96]

    float *d_in, *d_branch1, *d_branch2, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_branch1, bytes);
    cudaMalloc(&d_branch2, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (N + block - 1) / block;

    // Pipeline: two branches then combine
    // Branch 1: linear(W1, B1) -> sigmoid
    linear<<<grid, block>>>(d_branch1, d_in, W1, B1, N);
    activate_sigmoid<<<grid, block>>>(d_branch1, N);

    // Branch 2: linear(W2, B2) -> relu
    linear<<<grid, block>>>(d_branch2, d_in, W2, B2, N);
    activate_relu<<<grid, block>>>(d_branch2, N);

    // Combine: element-wise multiply
    elementwise_mul<<<grid, block>>>(d_out, d_branch1, d_branch2, N);

    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // Verify on host
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float x = h_in[i];
        float b1 = 1.0f / (1.0f + expf(-(W1 * x + B1)));
        float b2_pre = W2 * x + B2;
        float b2 = b2_pre > 0.0f ? b2_pre : 0.0f;
        float expected = b1 * b2;
        if (fabsf(h_out[i] - expected) > 1e-4f) {
            fprintf(stderr, "FAIL: 08_multi_kernel i=%d got %f expected %f\n",
                    i, h_out[i], expected);
            errors++;
            if (errors > 5) break;
        }
    }

    cudaFree(d_in); cudaFree(d_branch1);
    cudaFree(d_branch2); cudaFree(d_out);
    free(h_in); free(h_out);

    if (errors) return 1;
    printf("PASS: 08_multi_kernel (N=%d, 5 kernel launches)\n", N);
    return 0;
}
