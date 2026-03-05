// 30. Complex number arithmetic with FMA intrinsics
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__device__ void complex_mul(float ar, float ai, float br, float bi,
                            float *cr, float *ci) {
    *cr = __fmaf_rn(ar, br, -ai * bi);
    *ci = __fmaf_rn(ar, bi, ai * br);
}

__device__ float complex_abs_sq(float r, float i) {
    return __fmaf_rn(r, r, i * i);
}

__device__ void complex_add(float ar, float ai, float br, float bi,
                            float *cr, float *ci) {
    *cr = ar + br;
    *ci = ai + bi;
}

// For each pair: out = a*b + a + b
__global__ void complex_kernel(float *out_r, float *out_i,
                               const float *ar, const float *ai,
                               const float *br, const float *bi, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float pr, pi;
        complex_mul(ar[i], ai[i], br[i], bi[i], &pr, &pi);

        float sr, si;
        complex_add(ar[i], ai[i], br[i], bi[i], &sr, &si);

        out_r[i] = pr + sr;
        out_i[i] = pi + si;
    }
}

int main() {
    const int N = 2048;
    size_t bytes = N * sizeof(float);

    float *h_ar = (float *)malloc(bytes);
    float *h_ai = (float *)malloc(bytes);
    float *h_br = (float *)malloc(bytes);
    float *h_bi = (float *)malloc(bytes);
    float *h_or = (float *)malloc(bytes);
    float *h_oi = (float *)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_ar[i] = (float)(i % 20) * 0.05f;
        h_ai[i] = (float)(i % 15) * 0.04f;
        h_br[i] = (float)(i % 10) * 0.03f + 0.1f;
        h_bi[i] = (float)(i % 12) * 0.02f;
    }

    float *d_ar, *d_ai, *d_br, *d_bi, *d_or, *d_oi;
    cudaMalloc(&d_ar, bytes); cudaMalloc(&d_ai, bytes);
    cudaMalloc(&d_br, bytes); cudaMalloc(&d_bi, bytes);
    cudaMalloc(&d_or, bytes); cudaMalloc(&d_oi, bytes);

    cudaMemcpy(d_ar, h_ar, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ai, h_ai, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_br, h_br, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bi, h_bi, bytes, cudaMemcpyHostToDevice);

    complex_kernel<<<(N + 255) / 256, 256>>>(d_or, d_oi, d_ar, d_ai, d_br, d_bi, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_or, d_or, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_oi, d_oi, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float a_r = h_ar[i], a_i = h_ai[i];
        float b_r = h_br[i], b_i = h_bi[i];

        float p_r = a_r * b_r - a_i * b_i;
        float p_i = a_r * b_i + a_i * b_r;

        float s_r = a_r + b_r;
        float s_i = a_i + b_i;

        float exp_r = p_r + s_r;
        float exp_i = p_i + s_i;

        if (fabsf(h_or[i] - exp_r) > 1e-4f || fabsf(h_oi[i] - exp_i) > 1e-4f) {
            if (errors == 0)
                fprintf(stderr, "FAIL: 30_complex_math at i=%d got (%f,%f) expected (%f,%f)\n",
                        i, h_or[i], h_oi[i], exp_r, exp_i);
            errors++;
        }
    }

    cudaFree(d_ar); cudaFree(d_ai); cudaFree(d_br); cudaFree(d_bi);
    cudaFree(d_or); cudaFree(d_oi);
    free(h_ar); free(h_ai); free(h_br); free(h_bi); free(h_or); free(h_oi);

    if (errors) { fprintf(stderr, "FAIL: %d errors\n", errors); return 1; }
    printf("PASS: 30_complex_math (N=%d)\n", N);
    return 0;
}
