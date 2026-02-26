// 12. __frcp_rn and rsqrtf intrinsics
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__device__ float fast_inv_length(float x, float y, float z) {
    float len_sq = __fmaf_rn(x, x, __fmaf_rn(y, y, z * z));
    return rsqrtf(len_sq);
}

__device__ float rcp_chain(float x) {
    float r = __frcp_rn(x);
    return __frcp_rn(r + 1.0f);
}

__global__ void rsqrt_rcp_kernel(float *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i * 3 + 0];
        float y = in[i * 3 + 1];
        float z = in[i * 3 + 2];
        float inv_len = fast_inv_length(x, y, z);
        out[i] = rcp_chain(inv_len);
    }
}

int main() {
    const int N = 1024;
    size_t in_bytes = N * 3 * sizeof(float);
    size_t out_bytes = N * sizeof(float);

    float *h_in = (float *)malloc(in_bytes);
    float *h_out = (float *)malloc(out_bytes);
    for (int i = 0; i < N * 3; i++)
        h_in[i] = (float)((i % 20) + 1) * 0.5f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, in_bytes);
    cudaMalloc(&d_out, out_bytes);
    cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice);

    rsqrt_rcp_kernel<<<(N + 255) / 256, 256>>>(d_out, d_in, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float x = h_in[i*3], y = h_in[i*3+1], z = h_in[i*3+2];
        float len_sq = x*x + y*y + z*z;
        float inv_len = 1.0f / sqrtf(len_sq);
        float rcp_inv = 1.0f / inv_len;
        float expected = 1.0f / (rcp_inv + 1.0f);
        if (fabsf(h_out[i] - expected) > 1e-3f) {
            if (errors == 0)
                fprintf(stderr, "FAIL: 12_rsqrt_rcp at i=%d got %f expected %f\n",
                        i, h_out[i], expected);
            errors++;
        }
    }

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    if (errors) { fprintf(stderr, "FAIL: %d errors\n", errors); return 1; }
    printf("PASS: 12_rsqrt_rcp (N=%d)\n", N);
    return 0;
}
