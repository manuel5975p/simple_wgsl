// 13. Integer operations: shifts, bitwise, wide multiply, modulo, abs
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__device__ unsigned int hash_combine(unsigned int a, unsigned int b) {
    unsigned int h = a;
    h ^= b + 0x9e3779b9u + (h << 6) + (h >> 2);
    return h;
}

__device__ int abs_diff(int a, int b) {
    return (a > b) ? (a - b) : (b - a);
}

__device__ unsigned int extract_byte(unsigned int x, int byte_idx) {
    return (x >> (byte_idx * 8)) & 0xFFu;
}

__device__ unsigned int pack_bytes(unsigned int b0, unsigned int b1,
                                    unsigned int b2, unsigned int b3) {
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

__global__ void int_ops(unsigned int *out, const unsigned int *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned int x = in[i];
        unsigned int h = hash_combine(x, x >> 16);
        int ad = abs_diff((int)x, (int)(x >> 1));
        unsigned int b0 = extract_byte(h, 0);
        unsigned int b3 = extract_byte(h, 3);
        unsigned int packed = pack_bytes(b3, b0, b0 ^ b3, (unsigned int)ad & 0xFFu);
        out[i] = packed ^ h;
    }
}

int main() {
    const int N = 4096;
    size_t bytes = N * sizeof(unsigned int);

    unsigned int *h_in = (unsigned int *)malloc(bytes);
    unsigned int *h_out = (unsigned int *)malloc(bytes);
    for (int i = 0; i < N; i++)
        h_in[i] = (unsigned int)(i * 2654435761u);

    unsigned int *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    int_ops<<<(N + 255) / 256, 256>>>(d_out, d_in, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        unsigned int x = h_in[i];
        unsigned int h = x;
        h ^= (x >> 16) + 0x9e3779b9u + (h << 6) + (h >> 2);
        int a = (int)x, b = (int)(x >> 1);
        int ad = (a > b) ? (a - b) : (b - a);
        unsigned int b0 = h & 0xFFu;
        unsigned int b3 = (h >> 24) & 0xFFu;
        unsigned int packed = b3 | (b0 << 8) | ((b0 ^ b3) << 16) |
                              (((unsigned int)ad & 0xFFu) << 24);
        unsigned int expected = packed ^ h;
        if (h_out[i] != expected) {
            if (errors == 0)
                fprintf(stderr, "FAIL: 13_integer_intrinsics at i=%d got 0x%08x expected 0x%08x\n",
                        i, h_out[i], expected);
            errors++;
        }
    }

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    if (errors) { fprintf(stderr, "FAIL: %d errors\n", errors); return 1; }
    printf("PASS: 13_integer_intrinsics (N=%d)\n", N);
    return 0;
}
