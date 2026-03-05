// 18. Fast transcendentals: __sinf, __cosf, __expf, __logf (approx PTX ops)
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__device__ float fast_softplus(float x) {
    return __logf(1.0f + __expf(x));
}

__device__ float oscillator(float t, float freq, float phase) {
    return __sinf(freq * t + phase) * __cosf(0.5f * freq * t);
}

__global__ void fast_trig_kernel(float *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        float sp = fast_softplus(x);
        float osc = oscillator(x, 3.14159f, 0.5f);
        out[i] = sp + osc * 0.1f;
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

    fast_trig_kernel<<<(N + 255) / 256, 256>>>(d_out, d_in, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float x = h_in[i];
        float sp = logf(1.0f + expf(x));
        float osc = sinf(3.14159f * x + 0.5f) * cosf(0.5f * 3.14159f * x);
        float expected = sp + osc * 0.1f;
        if (fabsf(h_out[i] - expected) > 0.1f) {
            if (errors == 0)
                fprintf(stderr, "FAIL: 18_fast_trig at i=%d got %f expected %f (diff=%e)\n",
                        i, h_out[i], expected, h_out[i] - expected);
            errors++;
        }
    }

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    if (errors) { fprintf(stderr, "FAIL: %d errors\n", errors); return 1; }
    printf("PASS: 18_fast_trig (N=%d)\n", N);
    return 0;
}
