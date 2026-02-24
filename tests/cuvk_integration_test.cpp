/*
 * cuvk_integration_test.cpp - Comprehensive end-to-end integration tests
 * for the CUDA-on-Vulkan runtime.
 *
 * Tests the full pipeline: init -> context -> memory -> module -> launch -> verify.
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>
extern "C" {
#include "cuda.h"
}

class CuvkIntegrationTest : public ::testing::Test {
protected:
    CUcontext ctx = NULL;
    void SetUp() override {
        cuInit(0);
        CUdevice dev;
        if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) {
            GTEST_SKIP() << "No Vulkan compute device available";
            return;
        }
        if (cuCtxCreate(&ctx, NULL, 0, dev) != CUDA_SUCCESS) {
            GTEST_SKIP() << "Could not create context";
            return;
        }
    }
    void TearDown() override {
        if (ctx) cuCtxDestroy(ctx);
    }
};

/* ============================================================================
 * PTX Kernels
 * ============================================================================ */

// Vector add: C[i] = A[i] + B[i], uses %ctaid.x as element index
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

// Scalar multiply: Out[i] = In[i] * Scale  (BDA mode: pointer + scalar params)
static const char *SCALAR_MUL_PTX =
    ".version 7.0\n"
    ".target sm_70\n"
    ".address_size 64\n"
    ".visible .entry scalarMul(\n"
    "    .param .u64 In,\n"
    "    .param .u64 Out,\n"
    "    .param .f32 Scale\n"
    ")\n"
    ".reqntid 1, 1, 1\n"
    "{\n"
    "    .reg .u32 %tid;\n"
    "    .reg .u64 %rd<4>;\n"
    "    .reg .f32 %f<3>;\n"
    "\n"
    "    mov.u32 %tid, %ctaid.x;\n"
    "    ld.param.u64 %rd0, [In];\n"
    "    ld.param.u64 %rd1, [Out];\n"
    "    ld.param.f32 %f0, [Scale];\n"
    "\n"
    "    cvt.u64.u32 %rd2, %tid;\n"
    "    mul.lo.u64 %rd2, %rd2, 4;\n"
    "\n"
    "    add.u64 %rd3, %rd0, %rd2;\n"
    "    ld.global.f32 %f1, [%rd3];\n"
    "\n"
    "    mul.f32 %f2, %f1, %f0;\n"
    "\n"
    "    add.u64 %rd3, %rd1, %rd2;\n"
    "    st.global.f32 [%rd3], %f2;\n"
    "\n"
    "    ret;\n"
    "}\n";

// Two-kernel module: vecDouble and vecNegate in one PTX string
static const char *TWO_KERNEL_PTX =
    ".version 7.0\n"
    ".target sm_70\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry vecDouble(\n"
    "    .param .u64 A,\n"
    "    .param .u64 B\n"
    ")\n"
    ".reqntid 1, 1, 1\n"
    "{\n"
    "    .reg .u32 %r0;\n"
    "    .reg .u64 %rd<4>;\n"
    "    .reg .f32 %f<2>;\n"
    "\n"
    "    ld.param.u64 %rd0, [A];\n"
    "    ld.param.u64 %rd1, [B];\n"
    "    mov.u32 %r0, %ctaid.x;\n"
    "    cvt.u64.u32 %rd2, %r0;\n"
    "    mul.lo.u64 %rd2, %rd2, 4;\n"
    "    add.u64 %rd3, %rd0, %rd2;\n"
    "    ld.global.f32 %f0, [%rd3];\n"
    "    add.f32 %f1, %f0, %f0;\n"
    "    add.u64 %rd3, %rd1, %rd2;\n"
    "    st.global.f32 [%rd3], %f1;\n"
    "    ret;\n"
    "}\n"
    "\n"
    ".visible .entry vecNegate(\n"
    "    .param .u64 A,\n"
    "    .param .u64 B\n"
    ")\n"
    ".reqntid 1, 1, 1\n"
    "{\n"
    "    .reg .u32 %r0;\n"
    "    .reg .u64 %rd<4>;\n"
    "    .reg .f32 %f<2>;\n"
    "\n"
    "    ld.param.u64 %rd0, [A];\n"
    "    ld.param.u64 %rd1, [B];\n"
    "    mov.u32 %r0, %ctaid.x;\n"
    "    cvt.u64.u32 %rd2, %r0;\n"
    "    mul.lo.u64 %rd2, %rd2, 4;\n"
    "    add.u64 %rd3, %rd0, %rd2;\n"
    "    ld.global.f32 %f0, [%rd3];\n"
    "    neg.f32 %f1, %f0;\n"
    "    add.u64 %rd3, %rd1, %rd2;\n"
    "    st.global.f32 [%rd3], %f1;\n"
    "    ret;\n"
    "}\n";

// Simple passthrough kernel: Out[i] = In[i] (used for memset-then-compute test)
static const char *PASSTHROUGH_PTX =
    ".version 7.0\n"
    ".target sm_70\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry passthrough(\n"
    "    .param .u64 In,\n"
    "    .param .u64 Out\n"
    ")\n"
    ".reqntid 1, 1, 1\n"
    "{\n"
    "    .reg .u32 %r0;\n"
    "    .reg .u64 %rd<4>;\n"
    "    .reg .f32 %f0;\n"
    "\n"
    "    ld.param.u64 %rd0, [In];\n"
    "    ld.param.u64 %rd1, [Out];\n"
    "    mov.u32 %r0, %ctaid.x;\n"
    "    cvt.u64.u32 %rd2, %r0;\n"
    "    mul.lo.u64 %rd2, %rd2, 4;\n"
    "    add.u64 %rd3, %rd0, %rd2;\n"
    "    ld.global.f32 %f0, [%rd3];\n"
    "    add.u64 %rd3, %rd1, %rd2;\n"
    "    st.global.f32 [%rd3], %f0;\n"
    "    ret;\n"
    "}\n";

/* ============================================================================
 * Test 1: LargeVectorAdd — 4096 elements, verifies correctness at scale
 * ============================================================================ */

TEST_F(CuvkIntegrationTest, LargeVectorAdd) {
    const int N = 4096;
    std::vector<float> h_a(N), h_b(N), h_c(N, 0.0f);

    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i) * 0.5f;
        h_b[i] = static_cast<float>(N - i) * 0.25f;
    }

    CUdeviceptr d_a, d_b, d_c;
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_a, N * sizeof(float)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_b, N * sizeof(float)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_c, N * sizeof(float)));

    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyHtoD(d_a, h_a.data(), N * sizeof(float)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyHtoD(d_b, h_b.data(), N * sizeof(float)));

    CUmodule mod;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleLoadData(&mod, VECTOR_ADD_PTX));
    CUfunction func;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleGetFunction(&func, mod, "vecAdd"));

    void *params[] = { &d_a, &d_b, &d_c };
    ASSERT_EQ(CUDA_SUCCESS, cuLaunchKernel(func,
        N, 1, 1,   // grid: N blocks
        1, 1, 1,   // block: 1 thread
        0, NULL, params, NULL));

    cuCtxSynchronize();

    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyDtoH(h_c.data(), d_c, N * sizeof(float)));

    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];
        EXPECT_FLOAT_EQ(expected, h_c[i]) << "mismatch at index " << i;
    }

    cuMemFree(d_a);
    cuMemFree(d_b);
    cuMemFree(d_c);
    cuModuleUnload(mod);
}

/* ============================================================================
 * Test 2: ScalarMultiply — PTX kernel with scalar + pointer params via BDA
 * ============================================================================ */

TEST_F(CuvkIntegrationTest, ScalarMultiply) {
    const int N = 256;
    std::vector<float> h_in(N), h_out(N, 0.0f);

    for (int i = 0; i < N; i++) {
        h_in[i] = static_cast<float>(i + 1);
    }

    CUdeviceptr d_in, d_out;
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_in, N * sizeof(float)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_out, N * sizeof(float)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyHtoD(d_in, h_in.data(), N * sizeof(float)));

    CUmodule mod;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleLoadData(&mod, SCALAR_MUL_PTX));
    CUfunction func;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleGetFunction(&func, mod, "scalarMul"));

    float scale = 2.5f;
    void *params[] = { &d_in, &d_out, &scale };
    ASSERT_EQ(CUDA_SUCCESS, cuLaunchKernel(func,
        N, 1, 1,
        1, 1, 1,
        0, NULL, params, NULL));

    cuCtxSynchronize();

    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyDtoH(h_out.data(), d_out, N * sizeof(float)));

    for (int i = 0; i < N; i++) {
        float expected = h_in[i] * scale;
        EXPECT_FLOAT_EQ(expected, h_out[i]) << "mismatch at index " << i;
    }

    cuMemFree(d_in);
    cuMemFree(d_out);
    cuModuleUnload(mod);
}

/* ============================================================================
 * Test 3: MemsetThenCompute — memset a buffer, then use it in a kernel
 * ============================================================================ */

TEST_F(CuvkIntegrationTest, MemsetThenCompute) {
    const int N = 128;

    CUdeviceptr d_in, d_out;
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_in, N * sizeof(float)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_out, N * sizeof(float)));

    // Memset the input buffer to a known 32-bit pattern
    // 0x3F800000 is the IEEE 754 representation of 1.0f
    ASSERT_EQ(CUDA_SUCCESS, cuMemsetD32(d_in, 0x3F800000, N));

    // Verify memset worked by reading back
    std::vector<float> h_verify(N, 0.0f);
    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyDtoH(h_verify.data(), d_in, N * sizeof(float)));
    for (int i = 0; i < N; i++) {
        ASSERT_FLOAT_EQ(1.0f, h_verify[i]) << "memset verification failed at " << i;
    }

    // Run the passthrough kernel: Out[i] = In[i]
    CUmodule mod;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleLoadData(&mod, PASSTHROUGH_PTX));
    CUfunction func;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleGetFunction(&func, mod, "passthrough"));

    void *params[] = { &d_in, &d_out };
    ASSERT_EQ(CUDA_SUCCESS, cuLaunchKernel(func,
        N, 1, 1,
        1, 1, 1,
        0, NULL, params, NULL));

    cuCtxSynchronize();

    // Verify kernel output matches the memset data
    std::vector<float> h_out(N, 0.0f);
    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyDtoH(h_out.data(), d_out, N * sizeof(float)));
    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(1.0f, h_out[i]) << "kernel output mismatch at " << i;
    }

    cuMemFree(d_in);
    cuMemFree(d_out);
    cuModuleUnload(mod);
}

/* ============================================================================
 * Test 4: MultipleKernelsFromOneModule — two entry points, run both
 * ============================================================================ */

TEST_F(CuvkIntegrationTest, MultipleKernelsFromOneModule) {
    const int N = 64;
    std::vector<float> h_src(N);
    for (int i = 0; i < N; i++) {
        h_src[i] = static_cast<float>(i + 1);
    }

    CUdeviceptr d_src, d_doubled, d_negated;
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_src, N * sizeof(float)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_doubled, N * sizeof(float)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_negated, N * sizeof(float)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyHtoD(d_src, h_src.data(), N * sizeof(float)));

    // Load module with two entry points
    CUmodule mod;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleLoadData(&mod, TWO_KERNEL_PTX));

    // Get both functions
    CUfunction funcDouble, funcNegate;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleGetFunction(&funcDouble, mod, "vecDouble"));
    ASSERT_EQ(CUDA_SUCCESS, cuModuleGetFunction(&funcNegate, mod, "vecNegate"));

    // Run vecDouble: d_doubled[i] = d_src[i] * 2
    void *paramsDouble[] = { &d_src, &d_doubled };
    ASSERT_EQ(CUDA_SUCCESS, cuLaunchKernel(funcDouble,
        N, 1, 1, 1, 1, 1, 0, NULL, paramsDouble, NULL));

    cuCtxSynchronize();

    // Run vecNegate: d_negated[i] = -d_src[i]
    void *paramsNegate[] = { &d_src, &d_negated };
    ASSERT_EQ(CUDA_SUCCESS, cuLaunchKernel(funcNegate,
        N, 1, 1, 1, 1, 1, 0, NULL, paramsNegate, NULL));

    cuCtxSynchronize();

    // Verify vecDouble results
    std::vector<float> h_doubled(N, 0.0f);
    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyDtoH(h_doubled.data(), d_doubled, N * sizeof(float)));
    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(h_src[i] * 2.0f, h_doubled[i])
            << "vecDouble mismatch at index " << i;
    }

    // Verify vecNegate results
    std::vector<float> h_negated(N, 0.0f);
    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyDtoH(h_negated.data(), d_negated, N * sizeof(float)));
    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(-h_src[i], h_negated[i])
            << "vecNegate mismatch at index " << i;
    }

    cuMemFree(d_src);
    cuMemFree(d_doubled);
    cuMemFree(d_negated);
    cuModuleUnload(mod);
}

/* ============================================================================
 * Test 5: EventTiming — Time a kernel launch using cuEventRecord/ElapsedTime
 * ============================================================================ */

TEST_F(CuvkIntegrationTest, EventTiming) {
    const int N = 512;
    std::vector<float> h_a(N), h_b(N);
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    CUdeviceptr d_a, d_b, d_c;
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_a, N * sizeof(float)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_b, N * sizeof(float)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_c, N * sizeof(float)));
    cuMemcpyHtoD(d_a, h_a.data(), N * sizeof(float));
    cuMemcpyHtoD(d_b, h_b.data(), N * sizeof(float));

    CUmodule mod;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleLoadData(&mod, VECTOR_ADD_PTX));
    CUfunction func;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleGetFunction(&func, mod, "vecAdd"));

    // Create events for timing
    CUevent start = NULL, end = NULL;
    ASSERT_EQ(CUDA_SUCCESS, cuEventCreate(&start, 0));
    ASSERT_EQ(CUDA_SUCCESS, cuEventCreate(&end, 0));

    // Record start, launch, record end
    ASSERT_EQ(CUDA_SUCCESS, cuEventRecord(start, NULL));

    void *params[] = { &d_a, &d_b, &d_c };
    ASSERT_EQ(CUDA_SUCCESS, cuLaunchKernel(func,
        N, 1, 1, 1, 1, 1, 0, NULL, params, NULL));

    ASSERT_EQ(CUDA_SUCCESS, cuEventRecord(end, NULL));
    ASSERT_EQ(CUDA_SUCCESS, cuEventSynchronize(end));

    // Measure elapsed time
    float ms = -1.0f;
    ASSERT_EQ(CUDA_SUCCESS, cuEventElapsedTime(&ms, start, end));
    EXPECT_GE(ms, 0.0f) << "Elapsed time should be non-negative";

    // Also verify the kernel produced correct results
    std::vector<float> h_c(N);
    cuMemcpyDtoH(h_c.data(), d_c, N * sizeof(float));
    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(h_a[i] + h_b[i], h_c[i]) << "mismatch at " << i;
    }

    cuEventDestroy(start);
    cuEventDestroy(end);
    cuMemFree(d_a);
    cuMemFree(d_b);
    cuMemFree(d_c);
    cuModuleUnload(mod);
}

/* ============================================================================
 * Test 6: ErrorHandling — Test error paths
 * ============================================================================ */

TEST_F(CuvkIntegrationTest, ErrorHandling_NullPointers) {
    // cuModuleLoadData with NULL module pointer
    EXPECT_EQ(CUDA_ERROR_INVALID_VALUE, cuModuleLoadData(NULL, VECTOR_ADD_PTX));

    // cuModuleLoadData with NULL image
    CUmodule mod = NULL;
    EXPECT_EQ(CUDA_ERROR_INVALID_VALUE, cuModuleLoadData(&mod, NULL));

    // cuModuleGetFunction with NULL args
    EXPECT_EQ(CUDA_ERROR_INVALID_VALUE, cuModuleGetFunction(NULL, NULL, NULL));

    // cuMemAlloc with NULL pointer
    EXPECT_EQ(CUDA_ERROR_INVALID_VALUE, cuMemAlloc(NULL, 1024));

    // cuMemcpyHtoD with NULL source
    CUdeviceptr dptr = 0;
    cuMemAlloc(&dptr, 64);
    EXPECT_NE(CUDA_SUCCESS, cuMemcpyHtoD(dptr, NULL, 64));
    cuMemFree(dptr);
}

TEST_F(CuvkIntegrationTest, ErrorHandling_BadPtx) {
    CUmodule mod = NULL;
    // Completely invalid PTX should fail
    EXPECT_NE(CUDA_SUCCESS, cuModuleLoadData(&mod, "this is not valid PTX"));
}

TEST_F(CuvkIntegrationTest, ErrorHandling_FunctionNotFound) {
    CUmodule mod = NULL;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleLoadData(&mod, VECTOR_ADD_PTX));

    CUfunction func = NULL;
    EXPECT_NE(CUDA_SUCCESS, cuModuleGetFunction(&func, mod, "nonExistentKernel"));

    cuModuleUnload(mod);
}

TEST_F(CuvkIntegrationTest, ErrorHandling_DoubleFree) {
    CUdeviceptr ptr = 0;
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&ptr, 1024));
    ASSERT_EQ(CUDA_SUCCESS, cuMemFree(ptr));
    // Second free should either succeed gracefully or return an error,
    // but must not crash
    cuMemFree(ptr);
}

TEST_F(CuvkIntegrationTest, ErrorHandling_ZeroSizeAlloc) {
    CUdeviceptr ptr = 0;
    // Zero-size allocation: implementation-defined, but should not crash
    CUresult res = cuMemAlloc(&ptr, 0);
    if (res == CUDA_SUCCESS && ptr != 0) {
        cuMemFree(ptr);
    }
}
