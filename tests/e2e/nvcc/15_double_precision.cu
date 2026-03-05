// 15. Extended float precision: long FMA chains, Horner polynomials, Kahan sum
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__device__ float horner5(float x, float a, float b, float c, float d, float e, float f) {
    float r = a;
    r = __fmaf_rn(r, x, b);
    r = __fmaf_rn(r, x, c);
    r = __fmaf_rn(r, x, d);
    r = __fmaf_rn(r, x, e);
    r = __fmaf_rn(r, x, f);
    return r;
}

__device__ float kahan_sum4(float a, float b, float c, float d) {
    float sum = a;
    float comp = 0.0f;
    float y, t;
    y = b - comp; t = sum + y; comp = (t - sum) - y; sum = t;
    y = c - comp; t = sum + y; comp = (t - sum) - y; sum = t;
    y = d - comp; t = sum + y; comp = (t - sum) - y; sum = t;
    return sum;
}

__global__ void extended_prec_kernel(float *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        float poly = horner5(x, 0.01f, -0.05f, 0.13f, -0.27f, 1.5f, -0.3f);
        float ax = x < 0.0f ? -x : x;
        float inv = rsqrtf(ax + 1.0f);
        float sq = __frcp_rn(inv);
        float k = kahan_sum4(poly, sq, x * 0.001f, -0.5f);
        out[i] = k;
    }
}

int main() {
    const int N = 4096;
    size_t bytes = N * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int i = 0; i < N; i++)
        h_in[i] = (float)(i - 2048) * 0.005f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    extended_prec_kernel<<<(N + 255) / 256, 256>>>(d_out, d_in, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float x = h_in[i];
        float r = 0.01f;
        r = fmaf(r, x, -0.05f);
        r = fmaf(r, x, 0.13f);
        r = fmaf(r, x, -0.27f);
        r = fmaf(r, x, 1.5f);
        r = fmaf(r, x, -0.3f);
        float ax = fabsf(x);
        float inv = 1.0f / sqrtf(ax + 1.0f);
        float sq = 1.0f / inv;
        float sum = r;
        float comp = 0.0f;
        float y, t;
        y = sq - comp; t = sum + y; comp = (t - sum) - y; sum = t;
        y = x * 0.001f - comp; t = sum + y; comp = (t - sum) - y; sum = t;
        y = -0.5f - comp; t = sum + y; comp = (t - sum) - y; sum = t;
        float expected = sum;
        if (fabsf(h_out[i] - expected) > 0.01f) {
            if (errors == 0)
                fprintf(stderr, "FAIL: 15_double_precision at i=%d got %f expected %f (diff=%e)\n",
                        i, h_out[i], expected, h_out[i] - expected);
            errors++;
        }
    }

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    if (errors) { fprintf(stderr, "FAIL: %d errors\n", errors); return 1; }
    printf("PASS: 15_double_precision (N=%d)\n", N);
    return 0;
}
