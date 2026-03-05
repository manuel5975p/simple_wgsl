// 27. Warp-level inclusive prefix scan via __shfl_up_sync
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__global__ void warp_inclusive_scan(float *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (i < n) ? in[i] : 0.0f;
    int lane = threadIdx.x & 31;

    float t;
    t = __shfl_up_sync(0xFFFFFFFF, val, 1);
    if (lane >= 1) val += t;
    t = __shfl_up_sync(0xFFFFFFFF, val, 2);
    if (lane >= 2) val += t;
    t = __shfl_up_sync(0xFFFFFFFF, val, 4);
    if (lane >= 4) val += t;
    t = __shfl_up_sync(0xFFFFFFFF, val, 8);
    if (lane >= 8) val += t;
    t = __shfl_up_sync(0xFFFFFFFF, val, 16);
    if (lane >= 16) val += t;

    if (i < n)
        out[i] = val;
}

int main() {
    const int N = 2048;
    size_t bytes = N * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int i = 0; i < N; i++)
        h_in[i] = 1.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    int blocks = (N + 255) / 256;
    warp_inclusive_scan<<<blocks, 256>>>(d_out, d_in, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int b = 0; b < blocks; b++) {
        for (int w = 0; w < 8; w++) {
            float running = 0.0f;
            for (int lane = 0; lane < 32; lane++) {
                int idx = b * 256 + w * 32 + lane;
                if (idx >= N) break;
                running += h_in[idx];
                if (fabsf(h_out[idx] - running) > 1e-1f) {
                    if (errors == 0)
                        fprintf(stderr, "FAIL: 27_prefix_scan at i=%d got %f expected %f\n",
                                idx, h_out[idx], running);
                    errors++;
                }
            }
        }
    }

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    if (errors) { fprintf(stderr, "FAIL: %d errors\n", errors); return 1; }
    printf("PASS: 27_prefix_scan (N=%d)\n", N);
    return 0;
}
