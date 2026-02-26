// 29. Grid-stride loop pattern, gridDim.x, blockDim.x, large workloads
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__device__ float activation_swish(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void grid_stride_kernel(float *out, const float *in, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        float x = in[i];
        out[i] = activation_swish(x);
    }
}

int main() {
    const int N = 100000;
    size_t bytes = N * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int i = 0; i < N; i++)
        h_in[i] = (float)(i % 200) * 0.05f - 5.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    grid_stride_kernel<<<128, 256>>>(d_out, d_in, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float x = h_in[i];
        float expected = x / (1.0f + expf(-x));
        if (fabsf(h_out[i] - expected) > 1e-3f) {
            if (errors == 0)
                fprintf(stderr, "FAIL: 29_grid_stride_loop at i=%d got %f expected %f\n",
                        i, h_out[i], expected);
            errors++;
        }
    }

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    if (errors) { fprintf(stderr, "FAIL: %d errors\n", errors); return 1; }
    printf("PASS: 29_grid_stride_loop (N=%d)\n", N);
    return 0;
}
