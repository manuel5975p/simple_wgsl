// 33. CVT rounding modes: floor, ceil, trunc, round
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__global__ void cvt_rounding_kernel(float *out_floor, float *out_ceil,
                                     float *out_trunc, float *out_round,
                                     const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out_floor[i] = floorf(in[i]);
        out_ceil[i]  = ceilf(in[i]);
        out_trunc[i] = truncf(in[i]);
        out_round[i] = rintf(in[i]);
    }
}

int main() {
    const int N = 1024;
    size_t bytes = N * sizeof(float);

    float *h_in    = (float *)malloc(bytes);
    float *h_floor = (float *)malloc(bytes);
    float *h_ceil  = (float *)malloc(bytes);
    float *h_trunc = (float *)malloc(bytes);
    float *h_round = (float *)malloc(bytes);

    h_in[0] = 0.0f;
    h_in[1] = 1.0f;
    h_in[2] = -1.0f;
    h_in[3] = 0.5f;
    h_in[4] = -0.5f;
    h_in[5] = 1.5f;
    h_in[6] = -1.5f;
    h_in[7] = 2.5f;
    h_in[8] = -2.5f;
    h_in[9] = 0.1f;
    h_in[10] = -0.1f;
    h_in[11] = 0.9f;
    h_in[12] = -0.9f;
    h_in[13] = 3.7f;
    h_in[14] = -3.7f;
    h_in[15] = 100.001f;
    h_in[16] = -100.001f;
    h_in[17] = 0.49999f;
    h_in[18] = -0.49999f;
    h_in[19] = 1e10f;
    for (int i = 20; i < N; i++)
        h_in[i] = (float)(i - 512) * 0.37f;

    float *d_in, *d_floor, *d_ceil, *d_trunc, *d_round;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_floor, bytes);
    cudaMalloc(&d_ceil, bytes);
    cudaMalloc(&d_trunc, bytes);
    cudaMalloc(&d_round, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    cvt_rounding_kernel<<<(N + 255) / 256, 256>>>(d_floor, d_ceil, d_trunc,
                                                    d_round, d_in, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_floor, d_floor, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ceil,  d_ceil,  bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_trunc, d_trunc, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_round, d_round, bytes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float x = h_in[i];
        float ef = floorf(x), ec = ceilf(x), et = truncf(x), er = rintf(x);
        if (h_floor[i] != ef) {
            if (errors < 5)
                fprintf(stderr, "FAIL floor: i=%d in=%f got=%f expected=%f\n",
                        i, x, h_floor[i], ef);
            errors++;
        }
        if (h_ceil[i] != ec) {
            if (errors < 5)
                fprintf(stderr, "FAIL ceil: i=%d in=%f got=%f expected=%f\n",
                        i, x, h_ceil[i], ec);
            errors++;
        }
        if (h_trunc[i] != et) {
            if (errors < 5)
                fprintf(stderr, "FAIL trunc: i=%d in=%f got=%f expected=%f\n",
                        i, x, h_trunc[i], et);
            errors++;
        }
        if (h_round[i] != er) {
            if (errors < 5)
                fprintf(stderr, "FAIL round: i=%d in=%f got=%f expected=%f\n",
                        i, x, h_round[i], er);
            errors++;
        }
    }

    cudaFree(d_in); cudaFree(d_floor); cudaFree(d_ceil);
    cudaFree(d_trunc); cudaFree(d_round);
    free(h_in); free(h_floor); free(h_ceil);
    free(h_trunc); free(h_round);

    if (errors) { fprintf(stderr, "FAIL: %d errors\n", errors); return 1; }
    printf("PASS: 33_cvt_rounding (N=%d)\n", N);
    return 0;
}
