/*
 * win_test.c - Minimal CUDA-on-Vulkan test for Windows cross-compilation
 *
 * Tests: cuInit, cuDeviceGet, cuCtxCreate, cuMemAlloc, cuMemcpyHtoD,
 *        cuModuleLoadData (PTX), cuLaunchKernel, cuMemcpyDtoH, cleanup.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cuda.h"

#define CHECK(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        const char *name = "?"; \
        cuGetErrorName(err, &name); \
        fprintf(stderr, "CUDA error %d (%s) at %s:%d\n", err, name, __FILE__, __LINE__); \
        return 1; \
    } \
} while(0)

/* Simple PTX: scale each element by 2.0 */
static const char *ptx_scale =
    ".version 7.0\n"
    ".target sm_75\n"
    ".address_size 64\n"
    ".visible .entry scale(\n"
    "    .param .u64 data,\n"
    "    .param .u32 n\n"
    ") {\n"
    "    .reg .u64 %rd<4>;\n"
    "    .reg .u32 %r<4>;\n"
    "    .reg .f32 %f<3>;\n"
    "    .reg .pred %p<2>;\n"
    "    mov.u32 %r0, %ctaid.x;\n"
    "    mov.u32 %r1, %ntid.x;\n"
    "    mov.u32 %r2, %tid.x;\n"
    "    mad.lo.u32 %r0, %r0, %r1, %r2;\n"
    "    ld.param.u32 %r3, [n];\n"
    "    setp.ge.u32 %p0, %r0, %r3;\n"
    "    @%p0 bra END;\n"
    "    ld.param.u64 %rd0, [data];\n"
    "    cvt.u64.u32 %rd1, %r0;\n"
    "    shl.b64 %rd1, %rd1, 2;\n"
    "    add.u64 %rd2, %rd0, %rd1;\n"
    "    ld.global.f32 %f0, [%rd2];\n"
    "    mul.f32 %f1, %f0, 0f40000000;\n"
    "    st.global.f32 [%rd2], %f1;\n"
    "END:\n"
    "    ret;\n"
    "}\n";

int main(void) {
    printf("=== CUDA-on-Vulkan Windows Test ===\n");

    /* Init */
    CHECK(cuInit(0));
    printf("[OK] cuInit\n");

    int count = 0;
    CHECK(cuDeviceGetCount(&count));
    printf("[OK] cuDeviceGetCount: %d device(s)\n", count);
    if (count == 0) {
        printf("SKIP: no Vulkan device\n");
        return 0;
    }

    CUdevice dev;
    CHECK(cuDeviceGet(&dev, 0));

    char name[256];
    CHECK(cuDeviceGetName(name, sizeof(name), dev));
    printf("[OK] Device 0: %s\n", name);

    CUcontext ctx;
    CHECK(cuCtxCreate(&ctx, NULL, 0, dev));
    printf("[OK] cuCtxCreate\n");

    /* Memory */
    const int N = 1024;
    const size_t bytes = N * sizeof(float);
    CUdeviceptr d_data;
    CHECK(cuMemAlloc(&d_data, bytes));
    printf("[OK] cuMemAlloc %zu bytes\n", bytes);

    float *h_data = (float *)malloc(bytes);
    for (int i = 0; i < N; i++)
        h_data[i] = (float)i;

    CHECK(cuMemcpyHtoD(d_data, h_data, bytes));
    printf("[OK] cuMemcpyHtoD\n");

    /* Module + kernel */
    CUmodule mod;
    CHECK(cuModuleLoadData(&mod, ptx_scale));
    printf("[OK] cuModuleLoadData (PTX)\n");

    CUfunction func;
    CHECK(cuModuleGetFunction(&func, mod, "scale"));
    printf("[OK] cuModuleGetFunction 'scale'\n");

    /* Launch */
    unsigned int n = N;
    void *args[] = { &d_data, &n };
    CHECK(cuLaunchKernel(func,
        (N + 255) / 256, 1, 1,   /* grid */
        256, 1, 1,                /* block */
        0, NULL, args, NULL));
    printf("[OK] cuLaunchKernel\n");

    CHECK(cuCtxSynchronize());
    printf("[OK] cuCtxSynchronize\n");

    /* Read back */
    float *h_out = (float *)malloc(bytes);
    CHECK(cuMemcpyDtoH(h_out, d_data, bytes));
    printf("[OK] cuMemcpyDtoH\n");

    /* Verify */
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = (float)i * 2.0f;
        if (fabsf(h_out[i] - expected) > 0.001f) {
            if (errors < 5)
                printf("  FAIL: h_out[%d] = %f, expected %f\n", i, h_out[i], expected);
            errors++;
        }
    }

    if (errors == 0) {
        printf("[OK] All %d values correct!\n", N);
    } else {
        printf("FAIL: %d / %d values wrong\n", errors, N);
    }

    /* Cleanup */
    CHECK(cuMemFree(d_data));
    CHECK(cuModuleUnload(mod));
    CHECK(cuCtxDestroy(ctx));
    printf("[OK] Cleanup done\n");

    free(h_data);
    free(h_out);

    printf("=== %s ===\n", errors == 0 ? "PASS" : "FAIL");
    return errors != 0;
}
