// 23. Bitwise: shift, xor, and, or with simple patterns
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__device__ unsigned int xor_fold(unsigned int x) {
    x ^= x >> 16;
    x ^= x >> 8;
    x ^= x >> 4;
    return x;
}

__device__ unsigned int mask_spread(unsigned int x) {
    unsigned int lo = x & 0x0000FFFFu;
    unsigned int hi = x >> 16;
    return (lo ^ hi) | ((lo & hi) << 8);
}

__global__ void bitwise_kernel(unsigned int *out, const unsigned int *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned int x = in[i];
        unsigned int f = xor_fold(x);
        unsigned int m = mask_spread(x);
        unsigned int combined = (f & 0xFFFF0000u) | (m & 0x0000FFFFu);
        out[i] = combined ^ (x >> 3);
    }
}

int main() {
    const int N = 4096;
    size_t bytes = N * sizeof(unsigned int);

    unsigned int *h_in = (unsigned int *)malloc(bytes);
    unsigned int *h_out = (unsigned int *)malloc(bytes);
    for (int i = 0; i < N; i++)
        h_in[i] = (unsigned int)i * 7u + 0x12345678u;

    unsigned int *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    bitwise_kernel<<<(N + 255) / 256, 256>>>(d_out, d_in, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        unsigned int x = h_in[i];
        unsigned int f = x;
        f ^= f >> 16;
        f ^= f >> 8;
        f ^= f >> 4;
        unsigned int lo = x & 0x0000FFFFu;
        unsigned int hi = x >> 16;
        unsigned int m = (lo ^ hi) | ((lo & hi) << 8);
        unsigned int combined = (f & 0xFFFF0000u) | (m & 0x0000FFFFu);
        unsigned int expected = combined ^ (x >> 3);
        if (h_out[i] != expected) {
            if (errors == 0)
                fprintf(stderr, "FAIL: 23_bitwise_ops at i=%d got 0x%08x expected 0x%08x\n",
                        i, h_out[i], expected);
            errors++;
        }
    }

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    if (errors) { fprintf(stderr, "FAIL: %d errors\n", errors); return 1; }
    printf("PASS: 23_bitwise_ops (N=%d)\n", N);
    return 0;
}
