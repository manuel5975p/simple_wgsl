// 16. Float precision: piecewise functions, FMA chains, smoothstep variants
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__device__ float manual_fract(float x) {
    float t = (float)(int)x;
    float f = x - t;
    if (f < 0.0f) f += 1.0f;
    return f;
}

__device__ float triangle_wave(float x) {
    float t = x * 0.5f;
    float f = manual_fract(t);
    return (f < 0.5f) ? (f * 4.0f - 1.0f) : (3.0f - f * 4.0f);
}

__device__ float cubic_interp(float t) {
    return t * t * (3.0f - 2.0f * t);
}

__global__ void precision_kernel(float *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        float tri = triangle_wave(x);
        float t = fminf(fmaxf(x * 0.1f + 0.5f, 0.0f), 1.0f);
        float interp = cubic_interp(t);
        out[i] = __fmaf_rn(tri, 0.5f, interp);
    }
}

int main() {
    const int N = 4096;
    size_t bytes = N * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int i = 0; i < N; i++)
        h_in[i] = (float)(i - 2048) * 0.01f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    precision_kernel<<<(N + 255) / 256, 256>>>(d_out, d_in, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float x = h_in[i];
        float tv = x * 0.5f;
        float t_trunc = (float)(int)tv;
        float f = tv - t_trunc;
        if (f < 0.0f) f += 1.0f;
        float tri = (f < 0.5f) ? (f * 4.0f - 1.0f) : (3.0f - f * 4.0f);
        float t = fminf(fmaxf(x * 0.1f + 0.5f, 0.0f), 1.0f);
        float interp = t * t * (3.0f - 2.0f * t);
        float expected = fmaf(tri, 0.5f, interp);
        if (fabsf(h_out[i] - expected) > 1e-4f) {
            if (errors == 0)
                fprintf(stderr, "FAIL: 16_half_float_basic at i=%d x=%f got %f expected %f\n",
                        i, x, h_out[i], expected);
            errors++;
        }
    }

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    if (errors) { fprintf(stderr, "FAIL: %d errors\n", errors); return 1; }
    printf("PASS: 16_half_float_basic (N=%d)\n", N);
    return 0;
}
