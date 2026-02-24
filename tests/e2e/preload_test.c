/*
 * preload_test.c - Standalone CUDA driver API program for LD_PRELOAD testing.
 *
 * Compiled with gcc and linked against the CUDA stub library (-lcuda).
 * At runtime, LD_PRELOAD=libcuvk_runtime.so intercepts all driver API calls
 * and routes them through our Vulkan backend.
 *
 * Build:
 *   gcc -o preload_test preload_test.c \
 *       -I/opt/cuda/include -L/opt/cuda/lib64/stubs -lcuda -lm
 *
 * Run:
 *   LD_PRELOAD=/path/to/libcuvk_runtime.so ./preload_test vecAdd.ptx
 */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CHECK(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        const char *s = NULL; \
        cuGetErrorString(err, &s); \
        fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n", \
                __FILE__, __LINE__, s ? s : "unknown", err); \
        exit(1); \
    } \
} while(0)

static int check_float(float a, float b, float eps) {
    return fabsf(a - b) <= eps * fmaxf(1.0f, fmaxf(fabsf(a), fabsf(b)));
}

static char *read_file(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(sz + 1);
    fread(buf, 1, sz, f);
    buf[sz] = '\0';
    fclose(f);
    return buf;
}

/* ====================================================================== */
static int test_vector_add(CUmodule mod) {
    const int N = 1024;
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    CUfunction func;
    CHECK(cuModuleGetFunction(&func, mod, "vecAdd"));

    CUdeviceptr d_a, d_b, d_c;
    CHECK(cuMemAlloc(&d_a, N * sizeof(float)));
    CHECK(cuMemAlloc(&d_b, N * sizeof(float)));
    CHECK(cuMemAlloc(&d_c, N * sizeof(float)));

    float *h_a = (float *)malloc(N * sizeof(float));
    float *h_b = (float *)malloc(N * sizeof(float));
    float *h_c = (float *)calloc(N, sizeof(float));

    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    CHECK(cuMemcpyHtoD(d_a, h_a, N * sizeof(float)));
    CHECK(cuMemcpyHtoD(d_b, h_b, N * sizeof(float)));

    int n = N;
    void *params[] = { &d_a, &d_b, &d_c, &n };
    CHECK(cuLaunchKernel(func,
        gridSize, 1, 1, blockSize, 1, 1,
        0, NULL, params, NULL));
    CHECK(cuCtxSynchronize());
    CHECK(cuMemcpyDtoH(h_c, d_c, N * sizeof(float)));

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];
        if (!check_float(h_c[i], expected, 1e-5f)) {
            if (errors < 5)
                fprintf(stderr, "    MISMATCH[%d]: got %f expected %f\n",
                        i, h_c[i], expected);
            errors++;
        }
    }

    cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return errors;
}

/* ====================================================================== */
static int test_saxpy(CUmodule mod) {
    const int N = 2048;
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    const float alpha = 2.5f;

    CUfunction func;
    CHECK(cuModuleGetFunction(&func, mod, "saxpy"));

    CUdeviceptr d_x, d_y, d_out;
    CHECK(cuMemAlloc(&d_x, N * sizeof(float)));
    CHECK(cuMemAlloc(&d_y, N * sizeof(float)));
    CHECK(cuMemAlloc(&d_out, N * sizeof(float)));

    float *h_x = (float *)malloc(N * sizeof(float));
    float *h_y = (float *)malloc(N * sizeof(float));
    float *h_out = (float *)calloc(N, sizeof(float));

    for (int i = 0; i < N; i++) {
        h_x[i] = (float)i * 0.5f;
        h_y[i] = (float)(N - i) * 0.3f;
    }

    CHECK(cuMemcpyHtoD(d_x, h_x, N * sizeof(float)));
    CHECK(cuMemcpyHtoD(d_y, h_y, N * sizeof(float)));

    float a = alpha;
    int n = N;
    void *params[] = { &a, &d_x, &d_y, &d_out, &n };
    CHECK(cuLaunchKernel(func,
        gridSize, 1, 1, blockSize, 1, 1,
        0, NULL, params, NULL));
    CHECK(cuCtxSynchronize());
    CHECK(cuMemcpyDtoH(h_out, d_out, N * sizeof(float)));

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = alpha * h_x[i] + h_y[i];
        if (!check_float(h_out[i], expected, 1e-5f)) {
            if (errors < 5)
                fprintf(stderr, "    MISMATCH[%d]: got %f expected %f\n",
                        i, h_out[i], expected);
            errors++;
        }
    }

    cuMemFree(d_x); cuMemFree(d_y); cuMemFree(d_out);
    free(h_x); free(h_y); free(h_out);
    return errors;
}

/* ====================================================================== */
static int test_scale(CUmodule mod) {
    const int N = 512;
    const int blockSize = 128;
    const int gridSize = (N + blockSize - 1) / blockSize;
    const float scalar = 3.14f;

    CUfunction func;
    CHECK(cuModuleGetFunction(&func, mod, "scaleVec"));

    CUdeviceptr d_data;
    CHECK(cuMemAlloc(&d_data, N * sizeof(float)));

    float *h_data = (float *)malloc(N * sizeof(float));
    float *h_orig = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)(i + 1);
        h_orig[i] = h_data[i];
    }

    CHECK(cuMemcpyHtoD(d_data, h_data, N * sizeof(float)));

    float s = scalar;
    int n = N;
    void *params[] = { &d_data, &s, &n };
    CHECK(cuLaunchKernel(func,
        gridSize, 1, 1, blockSize, 1, 1,
        0, NULL, params, NULL));
    CHECK(cuCtxSynchronize());
    CHECK(cuMemcpyDtoH(h_data, d_data, N * sizeof(float)));

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = h_orig[i] * scalar;
        if (!check_float(h_data[i], expected, 1e-5f)) {
            if (errors < 5)
                fprintf(stderr, "    MISMATCH[%d]: got %f expected %f\n",
                        i, h_data[i], expected);
            errors++;
        }
    }

    cuMemFree(d_data);
    free(h_data); free(h_orig);
    return errors;
}

/* ====================================================================== */
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <kernel1.ptx> [kernel2.ptx ...]\n\n"
            "Loads each PTX file as a CUDA module and runs the kernels it contains.\n"
            "Recognized kernels: vecAdd, saxpy, scaleVec\n\n"
            "Example with LD_PRELOAD:\n"
            "  LD_PRELOAD=./libcuvk_runtime.so %s vecAdd.ptx saxpy.ptx scale.ptx\n",
            argv[0], argv[0]);
        return 1;
    }

    printf("=== CUDA-on-Vulkan LD_PRELOAD E2E Test ===\n\n");

    CHECK(cuInit(0));

    CUdevice dev;
    CHECK(cuDeviceGet(&dev, 0));

    char name[256];
    cuDeviceGetName(name, sizeof(name), dev);
    printf("Device: %s\n", name);

    CUcontext ctx;
    CHECK(cuCtxCreate(&ctx, NULL, 0, dev));

    int total_errors = 0;
    int tests_run = 0;

    for (int a = 1; a < argc; a++) {
        printf("\nLoading: %s\n", argv[a]);
        char *ptx = read_file(argv[a]);

        CUmodule mod;
        CHECK(cuModuleLoadData(&mod, ptx));
        free(ptx);

        /* Try each known kernel */
        CUfunction func;
        if (cuModuleGetFunction(&func, mod, "vecAdd") == CUDA_SUCCESS) {
            int e = test_vector_add(mod);
            printf("  vecAdd:   %s\n", e == 0 ? "PASS" : "FAIL");
            total_errors += e;
            tests_run++;
        }
        if (cuModuleGetFunction(&func, mod, "saxpy") == CUDA_SUCCESS) {
            int e = test_saxpy(mod);
            printf("  saxpy:    %s\n", e == 0 ? "PASS" : "FAIL");
            total_errors += e;
            tests_run++;
        }
        if (cuModuleGetFunction(&func, mod, "scaleVec") == CUDA_SUCCESS) {
            int e = test_scale(mod);
            printf("  scaleVec: %s\n", e == 0 ? "PASS" : "FAIL");
            total_errors += e;
            tests_run++;
        }

        cuModuleUnload(mod);
    }

    cuCtxDestroy(ctx);

    printf("\n=== Results: %d/%d tests passed ===\n",
           tests_run - (total_errors > 0 ? 1 : 0), tests_run);
    return total_errors ? 1 : 0;
}
