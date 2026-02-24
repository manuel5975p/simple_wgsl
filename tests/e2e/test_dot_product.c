/*
 * test_dot_product.c - E2E test: nvcc-compiled dot product
 *
 * Computes element-wise products on GPU, sums on host.
 */
#include "e2e_common.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <dot_product.ptx>\n", argv[0]);
        return 1;
    }

    const int N = 4096;
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CU(cuCtxCreate(&ctx, NULL, 0, dev));

    char *ptx = read_ptx_file(argv[1]);
    CUmodule mod;
    CHECK_CU(cuModuleLoadData(&mod, ptx));
    CUfunction func;
    CHECK_CU(cuModuleGetFunction(&func, mod, "dotPartial"));
    free(ptx);

    size_t bytes = N * sizeof(float);
    CUdeviceptr d_a, d_b, d_p;
    CHECK_CU(cuMemAlloc(&d_a, bytes));
    CHECK_CU(cuMemAlloc(&d_b, bytes));
    CHECK_CU(cuMemAlloc(&d_p, bytes));

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_p = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f / (float)(i + 1);
        h_b[i] = (float)(i + 1);
    }

    CHECK_CU(cuMemcpyHtoD(d_a, h_a, bytes));
    CHECK_CU(cuMemcpyHtoD(d_b, h_b, bytes));

    /* dotPartial(float *A, float *B, float *partials, int N) */
    int n = N;
    void *params[] = { &d_a, &d_b, &d_p, &n };
    CHECK_CU(cuLaunchKernel(func,
        gridSize, 1, 1,
        blockSize, 1, 1,
        0, NULL, params, NULL));
    CHECK_CU(cuCtxSynchronize());

    CHECK_CU(cuMemcpyDtoH(h_p, d_p, bytes));

    /* Sum partials on host */
    double gpu_dot = 0.0;
    double cpu_dot = 0.0;
    for (int i = 0; i < N; i++) {
        gpu_dot += (double)h_p[i];
        cpu_dot += (double)(h_a[i] * h_b[i]);
    }

    cuMemFree(d_a);
    cuMemFree(d_b);
    cuMemFree(d_p);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    free(h_a);
    free(h_b);
    free(h_p);

    /* Each element is 1/(i+1) * (i+1) = 1.0, so dot product = N */
    double expected = (double)N;
    double relerr = fabs(gpu_dot - expected) / expected;
    if (relerr < 1e-5) {
        printf("PASS: dot_product (N=%d, result=%.2f, expected=%.2f)\n",
               N, gpu_dot, expected);
        return 0;
    } else {
        fprintf(stderr, "FAIL: dot_product got %.6f, expected %.6f (relerr=%.2e)\n",
                gpu_dot, expected, relerr);
        return 1;
    }
}
