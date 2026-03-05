// 22. Type conversions: int<->float, widening, narrowing, casting
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__device__ float int_float_round(int x) {
    float f = (float)x;
    int back = (int)(f * 0.5f);
    return (float)back + 0.5f;
}

__device__ unsigned int widen_narrow(unsigned short lo, unsigned short hi) {
    unsigned int wide = ((unsigned int)hi << 16) | (unsigned int)lo;
    return wide;
}

__global__ void type_conv_kernel(float *float_out, unsigned int *uint_out,
                                 const int *int_in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float_out[i] = int_float_round(int_in[i]);

        unsigned short lo = (unsigned short)(int_in[i] & 0xFFFF);
        unsigned short hi = (unsigned short)((int_in[i] >> 16) & 0xFFFF);
        uint_out[i] = widen_narrow(lo, hi) ^ (unsigned int)int_in[i];
    }
}

int main() {
    const int N = 2048;

    int *h_in = (int *)malloc(N * sizeof(int));
    float *h_float = (float *)malloc(N * sizeof(float));
    unsigned int *h_uint = (unsigned int *)malloc(N * sizeof(unsigned int));

    for (int i = 0; i < N; i++)
        h_in[i] = i * 3 - 1000;

    int *d_in;
    float *d_float;
    unsigned int *d_uint;
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_float, N * sizeof(float));
    cudaMalloc(&d_uint, N * sizeof(unsigned int));
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    type_conv_kernel<<<(N + 255) / 256, 256>>>(d_float, d_uint, d_in, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_float, d_float, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_uint, d_uint, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        int x = h_in[i];
        float ef = (float)((int)((float)x * 0.5f)) + 0.5f;
        if (fabsf(h_float[i] - ef) > 1.0f) {
            if (errors == 0)
                fprintf(stderr, "FAIL: 22 float at i=%d got %f expected %f\n",
                        i, h_float[i], ef);
            errors++;
        }

        unsigned short lo = (unsigned short)(x & 0xFFFF);
        unsigned short hi = (unsigned short)((x >> 16) & 0xFFFF);
        unsigned int wide = ((unsigned int)hi << 16) | (unsigned int)lo;
        unsigned int eu = wide ^ (unsigned int)x;
        if (h_uint[i] != eu) {
            if (errors == 0)
                fprintf(stderr, "FAIL: 22 uint at i=%d got %u expected %u\n",
                        i, h_uint[i], eu);
            errors++;
        }
    }

    cudaFree(d_in); cudaFree(d_float); cudaFree(d_uint);
    free(h_in); free(h_float); free(h_uint);

    if (errors) { fprintf(stderr, "FAIL: %d errors\n", errors); return 1; }
    printf("PASS: 22_type_conversions (N=%d)\n", N);
    return 0;
}
