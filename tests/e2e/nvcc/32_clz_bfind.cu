// 32. Count leading zeros (__clz) and bit find (__ffs)
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void clz_bfind_kernel(unsigned int *out_clz,
                                  unsigned int *out_ffs,
                                  const unsigned int *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out_clz[i] = (unsigned int)__clz((int)in[i]);
        out_ffs[i] = (unsigned int)__ffs((int)in[i]);
    }
}

static unsigned int ref_clz(unsigned int x) {
    if (x == 0) return 32;
    unsigned int n = 0;
    if ((x & 0xFFFF0000u) == 0) { n += 16; x <<= 16; }
    if ((x & 0xFF000000u) == 0) { n += 8;  x <<= 8;  }
    if ((x & 0xF0000000u) == 0) { n += 4;  x <<= 4;  }
    if ((x & 0xC0000000u) == 0) { n += 2;  x <<= 2;  }
    if ((x & 0x80000000u) == 0) { n += 1; }
    return n;
}

static unsigned int ref_ffs(unsigned int x) {
    if (x == 0) return 0;
    unsigned int pos = 1;
    while ((x & 1) == 0) { x >>= 1; pos++; }
    return pos;
}

int main() {
    const int N = 4096;
    size_t bytes = N * sizeof(unsigned int);

    unsigned int *h_in = (unsigned int *)malloc(bytes);
    unsigned int *h_clz = (unsigned int *)malloc(bytes);
    unsigned int *h_ffs = (unsigned int *)malloc(bytes);

    h_in[0] = 0x00000000u;
    h_in[1] = 0xFFFFFFFFu;
    h_in[2] = 0x00000001u;
    h_in[3] = 0x80000000u;
    h_in[4] = 0x40000000u;
    h_in[5] = 0x00010000u;
    h_in[6] = 0x00000100u;
    h_in[7] = 0x7FFFFFFFu;
    h_in[8] = 0x00000002u;
    h_in[9] = 0x00008000u;
    for (int i = 10; i < N; i++)
        h_in[i] = (unsigned int)i * 2654435761u;

    unsigned int *d_in, *d_clz, *d_ffs;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_clz, bytes);
    cudaMalloc(&d_ffs, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    clz_bfind_kernel<<<(N + 255) / 256, 256>>>(d_clz, d_ffs, d_in, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_clz, d_clz, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ffs, d_ffs, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        unsigned int x = h_in[i];
        unsigned int expected_clz = ref_clz(x);
        unsigned int expected_ffs = ref_ffs(x);
        if (h_clz[i] != expected_clz) {
            if (errors < 5)
                fprintf(stderr, "FAIL clz: i=%d in=0x%08x got=%u expected=%u\n",
                        i, x, h_clz[i], expected_clz);
            errors++;
        }
        if (h_ffs[i] != expected_ffs) {
            if (errors < 5)
                fprintf(stderr, "FAIL ffs: i=%d in=0x%08x got=%u expected=%u\n",
                        i, x, h_ffs[i], expected_ffs);
            errors++;
        }
    }

    cudaFree(d_in); cudaFree(d_clz); cudaFree(d_ffs);
    free(h_in); free(h_clz); free(h_ffs);

    if (errors) { fprintf(stderr, "FAIL: %d errors\n", errors); return 1; }
    printf("PASS: 32_clz_bfind (N=%d)\n", N);
    return 0;
}
