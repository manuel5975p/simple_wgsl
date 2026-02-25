// 3. Chain of __device__ functions: normalize -> scale -> bias
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__device__ float my_sqrt(float x) {
    return sqrtf(x);
}

__device__ float normalize(float x, float mean, float inv_std) {
    return (x - mean) * inv_std;
}

__device__ float scale_and_bias(float x, float gamma, float beta) {
    return x * gamma + beta;
}

__global__ void layerNorm(float *out, const float *in,
                          float mean, float variance,
                          float gamma, float beta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float inv_std = 1.0f / my_sqrt(variance + 1e-5f);
        float normed = normalize(in[i], mean, inv_std);
        out[i] = scale_and_bias(normed, gamma, beta);
    }
}

int main() {
    const int N = 512;
    size_t bytes = N * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int i = 0; i < N; i++)
        h_in[i] = (float)i;

    // Compute mean and variance on host
    double sum = 0, sum2 = 0;
    for (int i = 0; i < N; i++) {
        sum += h_in[i];
        sum2 += (double)h_in[i] * h_in[i];
    }
    float mean = (float)(sum / N);
    float var = (float)(sum2 / N - (double)mean * mean);
    float gamma = 2.0f, beta = 0.5f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    layerNorm<<<(N + 127) / 128, 128>>>(d_out, d_in, mean, var, gamma, beta, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // Verify
    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = ((h_in[i] - mean) * inv_std) * gamma + beta;
        if (fabsf(h_out[i] - expected) > 1e-3f) {
            fprintf(stderr, "FAIL: 03_device_chain at i=%d: got %f expected %f\n",
                    i, h_out[i], expected);
            errors++;
            if (errors > 5) break;
        }
    }

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    if (errors) return 1;
    printf("PASS: 03_device_chain (N=%d)\n", N);
    return 0;
}
