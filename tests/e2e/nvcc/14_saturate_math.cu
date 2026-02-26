// 14. Saturating arithmetic: __saturatef, __fmaf_rn chains, fminf/fmaxf
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__device__ float saturate(float x) {
    return __saturatef(x);
}

__device__ float smooth_step(float edge0, float edge1, float x) {
    float t = saturate((x - edge0) / (edge1 - edge0));
    return t * t * __fmaf_rn(-2.0f, t, 3.0f);
}

__device__ float clamp_fminmax(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

__global__ void saturate_kernel(float *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        float s = smooth_step(-1.0f, 1.0f, x);
        float c = clamp_fminmax(x, -0.5f, 0.5f);
        out[i] = __fmaf_rn(s, 0.7f, c * 0.3f);
    }
}

int main() {
    const int N = 4096;
    size_t bytes = N * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int i = 0; i < N; i++)
        h_in[i] = (float)i / (float)N * 4.0f - 2.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    saturate_kernel<<<(N + 255) / 256, 256>>>(d_out, d_in, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float x = h_in[i];
        float t_raw = (x - (-1.0f)) / (1.0f - (-1.0f));
        float t = fminf(fmaxf(t_raw, 0.0f), 1.0f);
        float s = t * t * (-2.0f * t + 3.0f);
        float c = fminf(fmaxf(x, -0.5f), 0.5f);
        float expected = s * 0.7f + c * 0.3f;
        if (fabsf(h_out[i] - expected) > 1e-4f) {
            if (errors == 0)
                fprintf(stderr, "FAIL: 14_saturate_math at i=%d got %f expected %f\n",
                        i, h_out[i], expected);
            errors++;
        }
    }

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    if (errors) { fprintf(stderr, "FAIL: %d errors\n", errors); return 1; }
    printf("PASS: 14_saturate_math (N=%d)\n", N);
    return 0;
}
