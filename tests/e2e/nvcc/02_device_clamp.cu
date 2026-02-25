// 2. __device__ helper function called from __global__
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__device__ float clamp_val(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

__global__ void clampKernel(float *data, float lo, float hi, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = clamp_val(data[i], lo, hi);
    }
}

int main() {
    const int N = 1024;
    size_t bytes = N * sizeof(float);
    float *h = (float *)malloc(bytes);
    for (int i = 0; i < N; i++)
        h[i] = (float)i - 512.0f;  // range [-512, 511]

    float *d;
    cudaMalloc(&d, bytes);
    cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);

    clampKernel<<<(N + 255) / 256, 256>>>(d, -100.0f, 100.0f, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        float orig = (float)i - 512.0f;
        float expected = orig < -100.0f ? -100.0f : (orig > 100.0f ? 100.0f : orig);
        if (h[i] != expected) {
            fprintf(stderr, "FAIL: 02_device_clamp at i=%d: got %f expected %f\n",
                    i, h[i], expected);
            return 1;
        }
    }

    cudaFree(d);
    free(h);
    printf("PASS: 02_device_clamp (N=%d)\n", N);
    return 0;
}
