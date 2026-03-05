// 28. 64-bit integer: long long arithmetic, wide multiply, shifts
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__device__ long long wide_mul_add(int a, int b, long long c) {
    long long prod = (long long)a * (long long)b;
    return prod + c;
}

__device__ long long shift_chain(long long x) {
    long long hi = x >> 32;
    long long lo = x & 0xFFFFFFFFLL;
    return (hi * 3LL) + (lo * 5LL);
}

__global__ void int64_kernel(long long *out, const int *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int a = in[i];
        int b = in[(i + 1) % n];
        long long r1 = wide_mul_add(a, b, (long long)i * 1000LL);
        long long r2 = shift_chain(r1);
        out[i] = r1 + r2;
    }
}

int main() {
    const int N = 2048;
    size_t in_bytes = N * sizeof(int);
    size_t out_bytes = N * sizeof(long long);

    int *h_in = (int *)malloc(in_bytes);
    long long *h_out = (long long *)malloc(out_bytes);
    for (int i = 0; i < N; i++)
        h_in[i] = (i * 127 - 500);

    int *d_in;
    long long *d_out;
    cudaMalloc(&d_in, in_bytes);
    cudaMalloc(&d_out, out_bytes);
    cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice);

    int64_kernel<<<(N + 255) / 256, 256>>>(d_out, d_in, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        int a = h_in[i];
        int b = h_in[(i + 1) % N];
        long long r1 = (long long)a * (long long)b + (long long)i * 1000LL;
        long long hi = r1 >> 32;
        long long lo = r1 & 0xFFFFFFFFLL;
        long long r2 = hi * 3LL + lo * 5LL;
        long long expected = r1 + r2;
        if (h_out[i] != expected) {
            if (errors == 0)
                fprintf(stderr, "FAIL: 28_long_long_ops at i=%d got %lld expected %lld\n",
                        i, h_out[i], expected);
            errors++;
        }
    }

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    if (errors) { fprintf(stderr, "FAIL: %d errors\n", errors); return 1; }
    printf("PASS: 28_long_long_ops (N=%d)\n", N);
    return 0;
}
