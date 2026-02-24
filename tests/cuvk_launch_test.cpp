#include <gtest/gtest.h>
#include <vector>
#include <cmath>
extern "C" {
#include "cuda.h"
}

// Vector add with ONLY pointer params (no scalar N).
// Uses %ctaid.x for indexing, reqntid 1,1,1 (one thread per block).
// Grid size = number of elements.
static const char *VECTOR_ADD_PTX =
    ".version 7.0\n"
    ".target sm_70\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry vecAdd(\n"
    "    .param .u64 A,\n"
    "    .param .u64 B,\n"
    "    .param .u64 C\n"
    ")\n"
    ".reqntid 1, 1, 1\n"
    "{\n"
    "    .reg .u32 %r0;\n"
    "    .reg .u64 %rd<7>;\n"
    "    .reg .f32 %f<3>;\n"
    "\n"
    "    ld.param.u64 %rd0, [A];\n"
    "    ld.param.u64 %rd1, [B];\n"
    "    ld.param.u64 %rd2, [C];\n"
    "\n"
    "    mov.u32 %r0, %ctaid.x;\n"
    "    cvt.u64.u32 %rd3, %r0;\n"
    "    mul.lo.u64 %rd3, %rd3, 4;\n"
    "\n"
    "    add.u64 %rd4, %rd0, %rd3;\n"
    "    ld.global.f32 %f0, [%rd4];\n"
    "\n"
    "    add.u64 %rd5, %rd1, %rd3;\n"
    "    ld.global.f32 %f1, [%rd5];\n"
    "\n"
    "    add.f32 %f2, %f0, %f1;\n"
    "    add.u64 %rd6, %rd2, %rd3;\n"
    "    st.global.f32 [%rd6], %f2;\n"
    "\n"
    "    ret;\n"
    "}\n";

class CuvkLaunchTest : public ::testing::Test {
protected:
    CUcontext ctx = NULL;
    void SetUp() override {
        cuInit(0);
        CUdevice dev;
        cuDeviceGet(&dev, 0);
        cuCtxCreate(&ctx, NULL, 0, dev);
    }
    void TearDown() override {
        if (ctx) cuCtxDestroy(ctx);
    }
};

TEST_F(CuvkLaunchTest, LoadModule) {
    CUmodule mod = NULL;
    EXPECT_EQ(CUDA_SUCCESS, cuModuleLoadData(&mod, VECTOR_ADD_PTX));
    EXPECT_NE(nullptr, mod);
    cuModuleUnload(mod);
}

TEST_F(CuvkLaunchTest, GetFunction) {
    CUmodule mod = NULL;
    cuModuleLoadData(&mod, VECTOR_ADD_PTX);
    CUfunction func = NULL;
    EXPECT_EQ(CUDA_SUCCESS, cuModuleGetFunction(&func, mod, "vecAdd"));
    EXPECT_NE(nullptr, func);
    cuModuleUnload(mod);
}

TEST_F(CuvkLaunchTest, GetFunctionNotFound) {
    CUmodule mod = NULL;
    cuModuleLoadData(&mod, VECTOR_ADD_PTX);
    CUfunction func = NULL;
    EXPECT_NE(CUDA_SUCCESS, cuModuleGetFunction(&func, mod, "nonexistent"));
    cuModuleUnload(mod);
}

TEST_F(CuvkLaunchTest, InvalidPtx) {
    CUmodule mod = NULL;
    EXPECT_NE(CUDA_SUCCESS, cuModuleLoadData(&mod, "not valid ptx at all"));
}

TEST_F(CuvkLaunchTest, VectorAdd) {
    const int N = 64;
    std::vector<float> h_a(N), h_b(N), h_c(N);
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 10);
    }

    CUdeviceptr d_a, d_b, d_c;
    cuMemAlloc(&d_a, N * sizeof(float));
    cuMemAlloc(&d_b, N * sizeof(float));
    cuMemAlloc(&d_c, N * sizeof(float));

    cuMemcpyHtoD(d_a, h_a.data(), N * sizeof(float));
    cuMemcpyHtoD(d_b, h_b.data(), N * sizeof(float));

    CUmodule mod;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleLoadData(&mod, VECTOR_ADD_PTX));
    CUfunction func;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleGetFunction(&func, mod, "vecAdd"));

    // kernelParams: array of pointers to each parameter's value
    void *params[] = { &d_a, &d_b, &d_c };
    ASSERT_EQ(CUDA_SUCCESS, cuLaunchKernel(func,
        N, 1, 1,   // grid: N blocks
        1, 1, 1,   // block: 1 thread each
        0, NULL, params, NULL));

    cuCtxSynchronize();

    cuMemcpyDtoH(h_c.data(), d_c, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(h_a[i] + h_b[i], h_c[i]) << "mismatch at index " << i;
    }

    cuMemFree(d_a);
    cuMemFree(d_b);
    cuMemFree(d_c);
    cuModuleUnload(mod);
}
