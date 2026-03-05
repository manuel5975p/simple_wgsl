// 19. Warp-level primitives: __shfl_down_sync, __shfl_xor_sync, __ballot_sync
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ float warp_broadcast(float val, int src_lane) {
    return __shfl_sync(0xFFFFFFFF, val, src_lane);
}

__global__ void warp_reduce_kernel(float *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (i < n) ? in[i] : 0.0f;

    float sum = warp_reduce_sum(val);

    int lane = threadIdx.x & 31;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    if (lane == 0 && warp_id < (n + 31) / 32)
        out[warp_id] = sum;
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

    warp_reduce_kernel<<<(N + 255) / 256, 256>>>(d_out, d_in, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int w = 0; w < WARPS; w++) {
        float expected = 32.0f;
        if (fabsf(h_out[w] - expected) > 1e-3f) {
            if (errors == 0)
                fprintf(stderr, "FAIL: 19_warp_shuffle warp %d got %f expected %f\n",
                        w, h_out[w], expected);
            errors++;
        }
    }

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    if (errors) { fprintf(stderr, "FAIL: %d errors\n", errors); return 1; }
    printf("PASS: 19_warp_shuffle (N=%d, %d warps)\n", N, WARPS);
    return 0;
}
