// 34. Funnel shift via __funnelshift_l and __funnelshift_r
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void funnel_shift_kernel(unsigned int *out_l,
                                     unsigned int *out_r,
                                     const unsigned int *lo,
                                     const unsigned int *hi,
                                     const unsigned int *shift,
                                     int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out_l[i] = __funnelshift_l(lo[i], hi[i], shift[i]);
        out_r[i] = __funnelshift_r(lo[i], hi[i], shift[i]);
    }
}

static unsigned int ref_funnelshift_l(unsigned int lo, unsigned int hi,
                                       unsigned int n) {
    n &= 31;
    if (n == 0) return hi;
    return (hi << n) | (lo >> (32 - n));
}

static unsigned int ref_funnelshift_r(unsigned int lo, unsigned int hi,
                                       unsigned int n) {
    n &= 31;
    if (n == 0) return lo;
    return (lo >> n) | (hi << (32 - n));
}

int main() {
    const int N = 4096;
    size_t bytes = N * sizeof(unsigned int);

    unsigned int *h_lo    = (unsigned int *)malloc(bytes);
    unsigned int *h_hi    = (unsigned int *)malloc(bytes);
    unsigned int *h_shift = (unsigned int *)malloc(bytes);
    unsigned int *h_out_l = (unsigned int *)malloc(bytes);
    unsigned int *h_out_r = (unsigned int *)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_lo[i]    = (unsigned int)i * 2654435761u;
        h_hi[i]    = (unsigned int)i * 0x01000193u + 0xDEADBEEFu;
        h_shift[i] = (unsigned int)(i % 33);
    }
    h_lo[0] = 0xAAAAAAAAu; h_hi[0] = 0x55555555u; h_shift[0] = 0;
    h_lo[1] = 0xAAAAAAAAu; h_hi[1] = 0x55555555u; h_shift[1] = 1;
    h_lo[2] = 0xAAAAAAAAu; h_hi[2] = 0x55555555u; h_shift[2] = 16;
    h_lo[3] = 0xAAAAAAAAu; h_hi[3] = 0x55555555u; h_shift[3] = 31;
    h_lo[4] = 0xFFFFFFFFu; h_hi[4] = 0x00000000u; h_shift[4] = 8;
    h_lo[5] = 0x00000000u; h_hi[5] = 0xFFFFFFFFu; h_shift[5] = 8;
    h_lo[6] = 0x12345678u; h_hi[6] = 0x9ABCDEF0u; h_shift[6] = 4;
    h_lo[7] = 0x12345678u; h_hi[7] = 0x9ABCDEF0u; h_shift[7] = 32;

    unsigned int *d_lo, *d_hi, *d_shift, *d_out_l, *d_out_r;
    cudaMalloc(&d_lo, bytes);
    cudaMalloc(&d_hi, bytes);
    cudaMalloc(&d_shift, bytes);
    cudaMalloc(&d_out_l, bytes);
    cudaMalloc(&d_out_r, bytes);
    cudaMemcpy(d_lo, h_lo, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hi, h_hi, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_shift, h_shift, bytes, cudaMemcpyHostToDevice);

    funnel_shift_kernel<<<(N + 255) / 256, 256>>>(d_out_l, d_out_r, d_lo,
                                                    d_hi, d_shift, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out_l, d_out_l, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_r, d_out_r, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        unsigned int el = ref_funnelshift_l(h_lo[i], h_hi[i], h_shift[i]);
        unsigned int er = ref_funnelshift_r(h_lo[i], h_hi[i], h_shift[i]);
        if (h_out_l[i] != el) {
            if (errors < 5)
                fprintf(stderr, "FAIL fsl: i=%d lo=0x%08x hi=0x%08x n=%u got=0x%08x expected=0x%08x\n",
                        i, h_lo[i], h_hi[i], h_shift[i], h_out_l[i], el);
            errors++;
        }
        if (h_out_r[i] != er) {
            if (errors < 5)
                fprintf(stderr, "FAIL fsr: i=%d lo=0x%08x hi=0x%08x n=%u got=0x%08x expected=0x%08x\n",
                        i, h_lo[i], h_hi[i], h_shift[i], h_out_r[i], er);
            errors++;
        }
    }

    cudaFree(d_lo); cudaFree(d_hi); cudaFree(d_shift);
    cudaFree(d_out_l); cudaFree(d_out_r);
    free(h_lo); free(h_hi); free(h_shift);
    free(h_out_l); free(h_out_r);

    if (errors) { fprintf(stderr, "FAIL: %d errors\n", errors); return 1; }
    printf("PASS: 34_funnel_shift (N=%d)\n", N);
    return 0;
}
