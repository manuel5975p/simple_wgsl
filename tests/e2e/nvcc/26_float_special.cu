// 26. Float special: __float_as_int, __int_as_float, copysignf, isinf, isnan
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__device__ float fast_abs_via_bits(float x) {
    int bits = __float_as_int(x);
    bits &= 0x7FFFFFFF;
    return __int_as_float(bits);
}

__device__ float negate_via_bits(float x) {
    int bits = __float_as_int(x);
    bits ^= 0x80000000;
    return __int_as_float(bits);
}

__device__ float safe_div(float a, float b) {
    float result = a / b;
    if (isinf(result) || isnan(result))
        return copysignf(1e30f, a);
    return result;
}

__device__ float extract_exponent(float x) {
    int bits = __float_as_int(x);
    int exp_bits = (bits >> 23) & 0xFF;
    return (float)(exp_bits - 127);
}

__global__ void float_special_kernel(float *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        float a = fast_abs_via_bits(x);
        float neg = negate_via_bits(x);
        float sd = safe_div(x, x + 0.001f);
        float exp = extract_exponent(a + 1.0f);
        out[i] = a + neg * 0.01f + sd + exp * 0.001f;
    }
}

int main() {
    const int N = 4096;
    size_t bytes = N * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int i = 0; i < N; i++)
        h_in[i] = (float)(i - 2048) * 0.1f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    float_special_kernel<<<(N + 255) / 256, 256>>>(d_out, d_in, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float x = h_in[i];
        float a = fabsf(x);
        float neg = -x;
        float denom = x + 0.001f;
        float sd;
        if (denom == 0.0f)
            sd = copysignf(1e30f, x);
        else {
            sd = x / denom;
            if (isinf(sd) || isnan(sd))
                sd = copysignf(1e30f, x);
        }

        float val = a + 1.0f;
        int bits;
        memcpy(&bits, &val, 4);
        int exp_bits = (bits >> 23) & 0xFF;
        float exp = (float)(exp_bits - 127);

        float expected = a + neg * 0.01f + sd + exp * 0.001f;
        if (fabsf(h_out[i] - expected) > 1e-3f) {
            if (errors == 0)
                fprintf(stderr, "FAIL: 26_float_special at i=%d got %f expected %f\n",
                        i, h_out[i], expected);
            errors++;
        }
    }

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    if (errors) { fprintf(stderr, "FAIL: %d errors\n", errors); return 1; }
    printf("PASS: 26_float_special (N=%d)\n", N);
    return 0;
}
