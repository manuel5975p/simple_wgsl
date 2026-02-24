/*
 * test_saxpy.c - E2E test: nvcc-compiled SAXPY kernel (y = a*x + y)
 *
 * Tests scalar float parameter passing + pointer params.
 */
#include "e2e_common.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <saxpy.ptx>\n", argv[0]);
        return 1;
    }

    const int N = 2048;
    const float alpha = 2.5f;
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
    CHECK_CU(cuModuleGetFunction(&func, mod, "saxpy"));
    free(ptx);

    size_t bytes = N * sizeof(float);
    CUdeviceptr d_x, d_y, d_out;
    CHECK_CU(cuMemAlloc(&d_x, bytes));
    CHECK_CU(cuMemAlloc(&d_y, bytes));
    CHECK_CU(cuMemAlloc(&d_out, bytes));

    float *h_x = (float *)malloc(bytes);
    float *h_y = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_x[i] = (float)i * 0.5f;
        h_y[i] = (float)(N - i) * 0.3f;
    }

    CHECK_CU(cuMemcpyHtoD(d_x, h_x, bytes));
    CHECK_CU(cuMemcpyHtoD(d_y, h_y, bytes));

    /* saxpy(float a, float *x, float *y, float *out, int n) */
    float a = alpha;
    int n = N;
    void *params[] = { &a, &d_x, &d_y, &d_out, &n };
    CHECK_CU(cuLaunchKernel(func,
        gridSize, 1, 1,
        blockSize, 1, 1,
        0, NULL, params, NULL));
    CHECK_CU(cuCtxSynchronize());

    CHECK_CU(cuMemcpyDtoH(h_out, d_out, bytes));

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = alpha * h_x[i] + h_y[i];
        if (!check_float_eq(h_out[i], expected, 1e-3f)) {
            if (errors < 10)
                fprintf(stderr, "MISMATCH at %d: got %f, expected %f\n",
                        i, h_out[i], expected);
            errors++;
        }
    }

    cuMemFree(d_x);
    cuMemFree(d_y);
    cuMemFree(d_out);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    free(h_x);
    free(h_y);
    free(h_out);

    if (errors == 0) {
        printf("PASS: saxpy (%d elements, alpha=%f)\n", N, alpha);
        return 0;
    } else {
        fprintf(stderr, "FAIL: %d/%d mismatches\n", errors, N);
        return 1;
    }
}
