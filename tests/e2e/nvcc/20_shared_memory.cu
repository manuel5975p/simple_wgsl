// 20. Warp-level butterfly reduction via __shfl_xor_sync
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__global__ void warp_butterfly_reduce(float *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (i < n) ? in[i] : 0.0f;

    val += __shfl_xor_sync(0xFFFFFFFF, val, 16);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 8);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 4);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 2);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 1);

    int lane = threadIdx.x & 31;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (lane == 0 && warp_id < (n + 31) / 32)
        out[warp_id] = val;
}

int main() {
    const int N = 2048;
    const int WARPS = (N + 31) / 32;
    size_t in_bytes = N * sizeof(float);
    size_t out_bytes = WARPS * sizeof(float);

    float *h_in = (float *)malloc(in_bytes);
    float *h_out = (float *)malloc(out_bytes);
    for (int i = 0; i < N; i++)
        h_in[i] = 1.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, in_bytes);
    cudaMalloc(&d_out, out_bytes);
    cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, out_bytes);

    warp_butterfly_reduce<<<(N + 255) / 256, 256>>>(d_out, d_in, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int w = 0; w < WARPS; w++) {
        float expected = 32.0f;
        if (fabsf(h_out[w] - expected) > 1e-3f) {
            if (errors == 0)
                fprintf(stderr, "FAIL: 20_shared_memory warp %d got %f expected %f\n",
                        w, h_out[w], expected);
            errors++;
        }
    }

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    if (errors) { fprintf(stderr, "FAIL: %d errors\n", errors); return 1; }
    printf("PASS: 20_shared_memory (N=%d, %d warps)\n", N, WARPS);
    return 0;
}
