// 31. Population count (__popc) and bit reverse (__brev)
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void popc_brev_kernel(unsigned int *out_popc,
                                  unsigned int *out_brev,
                                  const unsigned int *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out_popc[i] = __popc(in[i]);
        out_brev[i] = __brev(in[i]);
    }
}

static unsigned int ref_popc(unsigned int x) {
    unsigned int count = 0;
    while (x) { count += x & 1; x >>= 1; }
    return count;
}

static unsigned int ref_brev(unsigned int x) {
    unsigned int r = 0;
    for (int i = 0; i < 32; i++) {
        r |= ((x >> i) & 1) << (31 - i);
    }
    return r;
}

int main() {
    const int N = 4096;
    size_t bytes = N * sizeof(unsigned int);

    unsigned int *h_in = (unsigned int *)malloc(bytes);
    unsigned int *h_popc = (unsigned int *)malloc(bytes);
    unsigned int *h_brev = (unsigned int *)malloc(bytes);

    h_in[0] = 0x00000000u;
    h_in[1] = 0xFFFFFFFFu;
    h_in[2] = 0x00000001u;
    h_in[3] = 0x80000000u;
    h_in[4] = 0xAAAAAAAAu;
    h_in[5] = 0x55555555u;
    h_in[6] = 0x0000FFFFu;
    h_in[7] = 0xFFFF0000u;
    h_in[8] = 0x12345678u;
    h_in[9] = 0xDEADBEEFu;
    for (int i = 10; i < N; i++)
        h_in[i] = (unsigned int)i * 2654435761u;

    unsigned int *d_in, *d_popc, *d_brev;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_popc, bytes);
    cudaMalloc(&d_brev, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    popc_brev_kernel<<<(N + 255) / 256, 256>>>(d_popc, d_brev, d_in, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_popc, d_popc, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_brev, d_brev, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        unsigned int x = h_in[i];
        unsigned int expected_popc = ref_popc(x);
        unsigned int expected_brev = ref_brev(x);
        if (h_popc[i] != expected_popc) {
            if (errors < 5)
                fprintf(stderr, "FAIL popc: i=%d in=0x%08x got=%u expected=%u\n",
                        i, x, h_popc[i], expected_popc);
            errors++;
        }
        if (h_brev[i] != expected_brev) {
            if (errors < 5)
                fprintf(stderr, "FAIL brev: i=%d in=0x%08x got=0x%08x expected=0x%08x\n",
                        i, x, h_brev[i], expected_brev);
            errors++;
        }
    }

    cudaFree(d_in); cudaFree(d_popc); cudaFree(d_brev);
    free(h_in); free(h_popc); free(h_brev);

    if (errors) { fprintf(stderr, "FAIL: %d errors\n", errors); return 1; }
    printf("PASS: 31_popc_brev (N=%d)\n", N);
    return 0;
}
