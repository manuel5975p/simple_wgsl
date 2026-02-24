/*
 * test_scale.c - E2E test: nvcc-compiled in-place scale kernel
 *
 * Tests in-place read-modify-write with scalar float parameter.
 */
#include "e2e_common.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <scale.ptx>\n", argv[0]);
        return 1;
    }

    const int N = 512;
    const float scalar = 3.14f;
    const int blockSize = 128;
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
    CHECK_CU(cuModuleGetFunction(&func, mod, "scaleVec"));
    free(ptx);

    size_t bytes = N * sizeof(float);
    CUdeviceptr d_data;
    CHECK_CU(cuMemAlloc(&d_data, bytes));

    float *h_data = (float *)malloc(bytes);
    float *h_orig = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)(i + 1);
        h_orig[i] = h_data[i];
    }

    CHECK_CU(cuMemcpyHtoD(d_data, h_data, bytes));

    /* scaleVec(float *data, float scalar, int n) */
    float s = scalar;
    int n = N;
    void *params[] = { &d_data, &s, &n };
    CHECK_CU(cuLaunchKernel(func,
        gridSize, 1, 1,
        blockSize, 1, 1,
        0, NULL, params, NULL));
    CHECK_CU(cuCtxSynchronize());

    CHECK_CU(cuMemcpyDtoH(h_data, d_data, bytes));

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = h_orig[i] * scalar;
        if (!check_float_eq(h_data[i], expected, 1e-3f)) {
            if (errors < 10)
                fprintf(stderr, "MISMATCH at %d: got %f, expected %f\n",
                        i, h_data[i], expected);
            errors++;
        }
    }

    cuMemFree(d_data);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    free(h_data);
    free(h_orig);

    if (errors == 0) {
        printf("PASS: scale (%d elements, scalar=%f)\n", N, scalar);
        return 0;
    } else {
        fprintf(stderr, "FAIL: %d/%d mismatches\n", errors, N);
        return 1;
    }
}
