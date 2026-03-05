// 24. Predicated execution: ternary, nested conditionals, branchless select
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__device__ float branchless_abs(float x) {
    return x < 0.0f ? -x : x;
}

__device__ float step_function(float x, float edge) {
    return x >= edge ? 1.0f : 0.0f;
}

__device__ float classify(float x) {
    return (x < -1.0f) ? -1.0f :
           (x > 1.0f)  ?  1.0f :
                           x;
}

__device__ float nested_select(float a, float b, float c) {
    float m = (a > b) ? a : b;
    m = (m > c) ? m : c;
    float n = (a < b) ? a : b;
    n = (n < c) ? n : c;
    return m - n;
}

__global__ void select_kernel(float *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        float a = branchless_abs(x);
        float s = step_function(x, 0.0f);
        float c = classify(x);
        float ns = nested_select(x, x * 0.5f, x * 2.0f);
        out[i] = a * s + c * 0.3f + ns * 0.1f;
    }
}

int main() {
    const int N = 4096;
    size_t bytes = N * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int i = 0; i < N; i++)
        h_in[i] = (float)i / (float)N * 6.0f - 3.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    select_kernel<<<(N + 255) / 256, 256>>>(d_out, d_in, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float x = h_in[i];
        float a = fabsf(x);
        float s = x >= 0.0f ? 1.0f : 0.0f;
        float c = x < -1.0f ? -1.0f : (x > 1.0f ? 1.0f : x);
        float vals[3] = { x, x * 0.5f, x * 2.0f };
        float mx = vals[0], mn = vals[0];
        for (int j = 1; j < 3; j++) {
            if (vals[j] > mx) mx = vals[j];
            if (vals[j] < mn) mn = vals[j];
        }
        float expected = a * s + c * 0.3f + (mx - mn) * 0.1f;
        if (fabsf(h_out[i] - expected) > 1e-4f) {
            if (errors == 0)
                fprintf(stderr, "FAIL: 24_ternary_select at i=%d got %f expected %f\n",
                        i, h_out[i], expected);
            errors++;
        }
    }

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    if (errors) { fprintf(stderr, "FAIL: %d errors\n", errors); return 1; }
    printf("PASS: 24_ternary_select (N=%d)\n", N);
    return 0;
}
