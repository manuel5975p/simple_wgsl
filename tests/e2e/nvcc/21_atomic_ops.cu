// 21. Parallel accumulation via warp shuffle (replaces atomics)
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__device__ float warp_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void weighted_sum_kernel(float *out, const float *vals,
                                     const float *weights, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float product = (i < n) ? vals[i] * weights[i] : 0.0f;

    float sum = warp_sum(product);

    int lane = threadIdx.x & 31;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (lane == 0 && warp_id < (n + 31) / 32)
        out[warp_id] = sum;
}

int main() {
    const int N = 2048;
    const int WARPS = (N + 31) / 32;
    size_t bytes = N * sizeof(float);
    size_t out_bytes = WARPS * sizeof(float);

    float *h_vals = (float *)malloc(bytes);
    float *h_weights = (float *)malloc(bytes);
    float *h_out = (float *)malloc(out_bytes);
    for (int i = 0; i < N; i++) {
        h_vals[i] = (float)(i % 10) * 0.1f;
        h_weights[i] = 1.0f;
    }

    float *d_vals, *d_weights, *d_out;
    cudaMalloc(&d_vals, bytes);
    cudaMalloc(&d_weights, bytes);
    cudaMalloc(&d_out, out_bytes);
    cudaMemcpy(d_vals, h_vals, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, out_bytes);

    weighted_sum_kernel<<<(N + 255) / 256, 256>>>(d_out, d_vals, d_weights, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int w = 0; w < WARPS; w++) {
        float expected = 0.0f;
        for (int j = 0; j < 32; j++) {
            int idx = w * 32 + j;
            if (idx < N) expected += h_vals[idx] * h_weights[idx];
        }
        if (fabsf(h_out[w] - expected) > 1e-2f) {
            if (errors == 0)
                fprintf(stderr, "FAIL: 21_atomic_ops warp %d got %f expected %f\n",
                        w, h_out[w], expected);
            errors++;
        }
    }

    cudaFree(d_vals); cudaFree(d_weights); cudaFree(d_out);
    free(h_vals); free(h_weights); free(h_out);

    if (errors) { fprintf(stderr, "FAIL: %d errors\n", errors); return 1; }
    printf("PASS: 21_atomic_ops (N=%d, %d warps)\n", N, WARPS);
    return 0;
}
