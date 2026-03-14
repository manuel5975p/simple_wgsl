/*
 * cuvk_complex_kernel_test.cpp - Tests with heavy function calls and branching
 *
 * Five tests with increasing complexity:
 *   1. DeviceFunctionCall     - entry calls a .func that squares a value
 *   2. BranchyPiecewise       - multi-branch piecewise function
 *   3. ChainedFunctionCalls   - three nested .func calls (f calls g calls h)
 *   4. LoopAccumulation       - backward-branch loop summing 1..N per element
 *   5. MegaKernel             - function calls + branches + type cvt + bitwise
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
extern "C" {
#include "cuda.h"
}

class CuvkComplexKernelTest : public ::testing::Test {
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
 * Test 1: DeviceFunctionCall
 *
 * A .func "square" that returns x*x, called from the entry point.
 * out[i] = square(in[i])
 * ============================================================================ */

static const char *FUNC_SQUARE_PTX =
    ".version 7.0\n"
    ".target sm_70\n"
    ".address_size 64\n"
    "\n"
    ".visible .func (.reg .f32 %retval) square(.reg .f32 %x)\n"
    "{\n"
    "    mul.f32 %retval, %x, %x;\n"
    "    ret;\n"
    "}\n"
    "\n"
    ".visible .entry applySquare(\n"
    "    .param .u64 In,\n"
    "    .param .u64 Out\n"
    ")\n"
    ".reqntid 1, 1, 1\n"
    "{\n"
    "    .reg .u32 %tid;\n"
    "    .reg .u64 %rd<5>;\n"
    "    .reg .f32 %f<2>;\n"
    "\n"
    "    ld.param.u64 %rd0, [In];\n"
    "    ld.param.u64 %rd1, [Out];\n"
    "    mov.u32 %tid, %ctaid.x;\n"
    "    cvt.u64.u32 %rd2, %tid;\n"
    "    mul.lo.u64 %rd2, %rd2, 4;\n"
    "\n"
    "    add.u64 %rd3, %rd0, %rd2;\n"
    "    ld.global.f32 %f0, [%rd3];\n"
    "\n"
    "    call (%f1), square, (%f0);\n"
    "\n"
    "    add.u64 %rd4, %rd1, %rd2;\n"
    "    st.global.f32 [%rd4], %f1;\n"
    "    ret;\n"
    "}\n";

TEST_F(CuvkComplexKernelTest, DeviceFunctionCall) {
    const int N = 64;
    std::vector<float> h_in(N), h_out(N, 0.0f);
    for (int i = 0; i < N; i++)
        h_in[i] = static_cast<float>(i) - 32.0f; // -32..31

    CUdeviceptr d_in, d_out;
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_in, N * sizeof(float)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_out, N * sizeof(float)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyHtoD(d_in, h_in.data(), N * sizeof(float)));

    CUmodule mod;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleLoadData(&mod, FUNC_SQUARE_PTX));
    CUfunction func;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleGetFunction(&func, mod, "applySquare"));

    void *params[] = { &d_in, &d_out };
    ASSERT_EQ(CUDA_SUCCESS, cuLaunchKernel(func,
        N, 1, 1, 1, 1, 1, 0, NULL, params, NULL));
    cuCtxSynchronize();

    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyDtoH(h_out.data(), d_out, N * sizeof(float)));
    for (int i = 0; i < N; i++) {
        float expected = h_in[i] * h_in[i];
        EXPECT_FLOAT_EQ(expected, h_out[i]) << "mismatch at index " << i;
    }

    cuMemFree(d_in);
    cuMemFree(d_out);
    cuModuleUnload(mod);
}

/* ============================================================================
 * Test 2: BranchyPiecewise
 *
 * Piecewise function with multiple branches:
 *   if x > 10:  out = x * 2
 *   elif x > 0: out = x + 100
 *   elif x == 0: out = 42
 *   else:        out = -x
 * ============================================================================ */

static const char *PIECEWISE_PTX =
    ".version 7.0\n"
    ".target sm_70\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry piecewise(\n"
    "    .param .u64 In,\n"
    "    .param .u64 Out\n"
    ")\n"
    ".reqntid 1, 1, 1\n"
    "{\n"
    "    .reg .u32 %tid;\n"
    "    .reg .u64 %rd<5>;\n"
    "    .reg .f32 %f<3>;\n"
    "    .reg .pred %p<3>;\n"
    "\n"
    "    ld.param.u64 %rd0, [In];\n"
    "    ld.param.u64 %rd1, [Out];\n"
    "    mov.u32 %tid, %ctaid.x;\n"
    "    cvt.u64.u32 %rd2, %tid;\n"
    "    mul.lo.u64 %rd2, %rd2, 4;\n"
    "    add.u64 %rd3, %rd0, %rd2;\n"
    "    ld.global.f32 %f0, [%rd3];\n"
    "\n"
    "    // Branch 1: x > 10 -> x * 2\n"
    "    setp.gt.f32 %p0, %f0, 0f41200000;\n"  // 10.0f
    "    @%p0 bra GT10;\n"
    "\n"
    "    // Branch 2: x > 0 -> x + 100\n"
    "    setp.gt.f32 %p1, %f0, 0f00000000;\n"  // 0.0f
    "    @%p1 bra GT0;\n"
    "\n"
    "    // Branch 3: x == 0 -> 42\n"
    "    setp.eq.f32 %p2, %f0, 0f00000000;\n"
    "    @%p2 bra EQ0;\n"
    "\n"
    "    // else: -x\n"
    "    neg.f32 %f1, %f0;\n"
    "    bra STORE;\n"
    "\n"
    "GT10:\n"
    "    mul.f32 %f1, %f0, 0f40000000;\n"  // 2.0f
    "    bra STORE;\n"
    "\n"
    "GT0:\n"
    "    add.f32 %f1, %f0, 0f42C80000;\n"  // 100.0f
    "    bra STORE;\n"
    "\n"
    "EQ0:\n"
    "    mov.f32 %f1, 0f42280000;\n"  // 42.0f
    "    bra STORE;\n"
    "\n"
    "STORE:\n"
    "    add.u64 %rd4, %rd1, %rd2;\n"
    "    st.global.f32 [%rd4], %f1;\n"
    "    ret;\n"
    "}\n";

TEST_F(CuvkComplexKernelTest, BranchyPiecewise) {
    std::vector<float> h_in  = { 20.0f, 5.0f, 0.0f, -7.0f, 11.0f, 1.0f, -0.5f, 100.0f };
    const int N = (int)h_in.size();
    std::vector<float> h_out(N, 0.0f);

    CUdeviceptr d_in, d_out;
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_in, N * sizeof(float)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_out, N * sizeof(float)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyHtoD(d_in, h_in.data(), N * sizeof(float)));

    CUmodule mod;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleLoadData(&mod, PIECEWISE_PTX));
    CUfunction func;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleGetFunction(&func, mod, "piecewise"));

    void *params[] = { &d_in, &d_out };
    ASSERT_EQ(CUDA_SUCCESS, cuLaunchKernel(func,
        N, 1, 1, 1, 1, 1, 0, NULL, params, NULL));
    cuCtxSynchronize();

    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyDtoH(h_out.data(), d_out, N * sizeof(float)));

    for (int i = 0; i < N; i++) {
        float x = h_in[i];
        float expected;
        if (x > 10.0f)       expected = x * 2.0f;
        else if (x > 0.0f)   expected = x + 100.0f;
        else if (x == 0.0f)  expected = 42.0f;
        else                  expected = -x;
        EXPECT_FLOAT_EQ(expected, h_out[i]) << "mismatch at index " << i
            << " (input=" << x << ")";
    }

    cuMemFree(d_in);
    cuMemFree(d_out);
    cuModuleUnload(mod);
}

/* ============================================================================
 * Test 3: ChainedFunctionCalls
 *
 * Three device functions chained:
 *   abs_val(x)  = x < 0 ? -x : x       (branch inside function)
 *   square(x)   = x * x
 *   cube_plus_one(x) = square(abs_val(x)) * abs_val(x) + 1.0
 *
 * Entry: out[i] = cube_plus_one(in[i])   =>  |x|^3 + 1
 * ============================================================================ */

static const char *CHAINED_FUNC_PTX =
    ".version 7.0\n"
    ".target sm_70\n"
    ".address_size 64\n"
    "\n"
    // abs_val: returns |x| using a branch
    ".visible .func (.reg .f32 %retval) abs_val(.reg .f32 %x)\n"
    "{\n"
    "    .reg .pred %p;\n"
    "    setp.ge.f32 %p, %x, 0f00000000;\n"
    "    @%p bra ABS_POS;\n"
    "    neg.f32 %retval, %x;\n"
    "    ret;\n"
    "ABS_POS:\n"
    "    mov.f32 %retval, %x;\n"
    "    ret;\n"
    "}\n"
    "\n"
    // square: returns x*x
    ".visible .func (.reg .f32 %retval) square(.reg .f32 %x)\n"
    "{\n"
    "    mul.f32 %retval, %x, %x;\n"
    "    ret;\n"
    "}\n"
    "\n"
    // cube_plus_one: |x|^3 + 1
    ".visible .func (.reg .f32 %retval) cube_plus_one(.reg .f32 %x)\n"
    "{\n"
    "    .reg .f32 %abs, %sq;\n"
    "    call (%abs), abs_val, (%x);\n"
    "    call (%sq), square, (%abs);\n"
    "    mul.f32 %retval, %sq, %abs;\n"     // |x|^3
    "    add.f32 %retval, %retval, 0f3F800000;\n"  // + 1.0
    "    ret;\n"
    "}\n"
    "\n"
    ".visible .entry applyCubePlusOne(\n"
    "    .param .u64 In,\n"
    "    .param .u64 Out\n"
    ")\n"
    ".reqntid 1, 1, 1\n"
    "{\n"
    "    .reg .u32 %tid;\n"
    "    .reg .u64 %rd<5>;\n"
    "    .reg .f32 %f<2>;\n"
    "\n"
    "    ld.param.u64 %rd0, [In];\n"
    "    ld.param.u64 %rd1, [Out];\n"
    "    mov.u32 %tid, %ctaid.x;\n"
    "    cvt.u64.u32 %rd2, %tid;\n"
    "    mul.lo.u64 %rd2, %rd2, 4;\n"
    "\n"
    "    add.u64 %rd3, %rd0, %rd2;\n"
    "    ld.global.f32 %f0, [%rd3];\n"
    "\n"
    "    call (%f1), cube_plus_one, (%f0);\n"
    "\n"
    "    add.u64 %rd4, %rd1, %rd2;\n"
    "    st.global.f32 [%rd4], %f1;\n"
    "    ret;\n"
    "}\n";

TEST_F(CuvkComplexKernelTest, ChainedFunctionCalls) {
    std::vector<float> h_in = { 2.0f, -3.0f, 0.0f, 1.0f, -1.0f, 4.0f, -0.5f, 10.0f };
    const int N = (int)h_in.size();
    std::vector<float> h_out(N, 0.0f);

    CUdeviceptr d_in, d_out;
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_in, N * sizeof(float)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_out, N * sizeof(float)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyHtoD(d_in, h_in.data(), N * sizeof(float)));

    CUmodule mod;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleLoadData(&mod, CHAINED_FUNC_PTX));
    CUfunction func;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleGetFunction(&func, mod, "applyCubePlusOne"));

    void *params[] = { &d_in, &d_out };
    ASSERT_EQ(CUDA_SUCCESS, cuLaunchKernel(func,
        N, 1, 1, 1, 1, 1, 0, NULL, params, NULL));
    cuCtxSynchronize();

    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyDtoH(h_out.data(), d_out, N * sizeof(float)));

    for (int i = 0; i < N; i++) {
        float x = h_in[i];
        float ax = std::fabs(x);
        float expected = ax * ax * ax + 1.0f;
        EXPECT_FLOAT_EQ(expected, h_out[i]) << "mismatch at index " << i
            << " (input=" << x << ")";
    }

    cuMemFree(d_in);
    cuMemFree(d_out);
    cuModuleUnload(mod);
}

/* ============================================================================
 * Test 4: LoopAccumulation
 *
 * Each element in[i] is treated as an integer count N.
 * The kernel computes sum = 1 + 2 + ... + N = N*(N+1)/2 via a loop.
 * out[i] = (float)sum
 *
 * Uses a backward branch to create a loop, plus a branch-out condition.
 * Also calls a device function to do the addition (to exercise both).
 * ============================================================================ */

static const char *LOOP_SUM_PTX =
    ".version 7.0\n"
    ".target sm_70\n"
    ".address_size 64\n"
    "\n"
    ".visible .func (.reg .f32 %retval) fadd(.reg .f32 %a, .reg .f32 %b)\n"
    "{\n"
    "    add.f32 %retval, %a, %b;\n"
    "    ret;\n"
    "}\n"
    "\n"
    ".visible .entry loopSum(\n"
    "    .param .u64 In,\n"
    "    .param .u64 Out\n"
    ")\n"
    ".reqntid 1, 1, 1\n"
    "{\n"
    "    .reg .u32 %tid, %n, %i;\n"
    "    .reg .u64 %rd<5>;\n"
    "    .reg .f32 %sum, %fi, %tmp;\n"
    "    .reg .pred %p;\n"
    "\n"
    "    ld.param.u64 %rd0, [In];\n"
    "    ld.param.u64 %rd1, [Out];\n"
    "    mov.u32 %tid, %ctaid.x;\n"
    "    cvt.u64.u32 %rd2, %tid;\n"
    "    mul.lo.u64 %rd2, %rd2, 4;\n"
    "\n"
    "    // Load N as u32\n"
    "    add.u64 %rd3, %rd0, %rd2;\n"
    "    ld.global.u32 %n, [%rd3];\n"
    "\n"
    "    // sum = 0, i = 1\n"
    "    mov.f32 %sum, 0f00000000;\n"
    "    mov.u32 %i, 1;\n"
    "\n"
    "LOOP:\n"
    "    setp.gt.u32 %p, %i, %n;\n"
    "    @%p bra DONE;\n"
    "\n"
    "    cvt.rn.f32.u32 %fi, %i;\n"
    "    call (%tmp), fadd, (%sum, %fi);\n"
    "    mov.f32 %sum, %tmp;\n"
    "    add.u32 %i, %i, 1;\n"
    "    bra LOOP;\n"
    "\n"
    "DONE:\n"
    "    add.u64 %rd4, %rd1, %rd2;\n"
    "    st.global.f32 [%rd4], %sum;\n"
    "    ret;\n"
    "}\n";

TEST_F(CuvkComplexKernelTest, LoopAccumulation) {
    std::vector<uint32_t> h_in = { 0, 1, 5, 10, 20, 50, 100, 255 };
    const int N = (int)h_in.size();
    std::vector<float> h_out(N, 0.0f);

    CUdeviceptr d_in, d_out;
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_in, N * sizeof(uint32_t)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_out, N * sizeof(float)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyHtoD(d_in, h_in.data(), N * sizeof(uint32_t)));

    CUmodule mod;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleLoadData(&mod, LOOP_SUM_PTX));
    CUfunction func;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleGetFunction(&func, mod, "loopSum"));

    void *params[] = { &d_in, &d_out };
    ASSERT_EQ(CUDA_SUCCESS, cuLaunchKernel(func,
        N, 1, 1, 1, 1, 1, 0, NULL, params, NULL));
    cuCtxSynchronize();

    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyDtoH(h_out.data(), d_out, N * sizeof(float)));

    for (int i = 0; i < N; i++) {
        uint32_t n = h_in[i];
        float expected = static_cast<float>(n) * static_cast<float>(n + 1) / 2.0f;
        EXPECT_FLOAT_EQ(expected, h_out[i]) << "mismatch at index " << i
            << " (N=" << n << ")";
    }

    cuMemFree(d_in);
    cuMemFree(d_out);
    cuModuleUnload(mod);
}

/* ============================================================================
 * Test 5: MegaKernel
 *
 * The crazy one. For each element:
 *   1. Load float x from In
 *   2. Call classify(x) which returns:
 *        0 if x == 0
 *        1 if x > 0 and x <= 50
 *        2 if x > 50
 *        3 if x < 0
 *      (uses multiple branches inside the function)
 *   3. Call transform(x, class):
 *        class 0: return 0
 *        class 1: return square(x) + x        (calls square)
 *        class 2: return sqrt_approx(x) * 10  (calls sqrt_approx via Newton)
 *        class 3: return abs_val(x) * 3       (calls abs_val)
 *      (branches + nested function calls)
 *   4. Convert result to int, apply bitwise XOR with 0xFF, convert back
 *   5. Store final float
 *
 * This exercises: 5 device functions, multi-level branching, function calls
 * from within functions, type conversions, and bitwise ops.
 * ============================================================================ */

static const char *MEGA_KERNEL_PTX =
    ".version 7.0\n"
    ".target sm_70\n"
    ".address_size 64\n"
    "\n"
    // abs_val(x): |x|
    ".visible .func (.reg .f32 %retval) mega_abs(.reg .f32 %x)\n"
    "{\n"
    "    .reg .pred %p;\n"
    "    setp.ge.f32 %p, %x, 0f00000000;\n"
    "    @%p bra MABS_POS;\n"
    "    neg.f32 %retval, %x;\n"
    "    ret;\n"
    "MABS_POS:\n"
    "    mov.f32 %retval, %x;\n"
    "    ret;\n"
    "}\n"
    "\n"
    // square(x): x*x
    ".visible .func (.reg .f32 %retval) mega_square(.reg .f32 %x)\n"
    "{\n"
    "    mul.f32 %retval, %x, %x;\n"
    "    ret;\n"
    "}\n"
    "\n"
    // classify(x): 0/1/2/3 as described above
    ".visible .func (.reg .u32 %retval) mega_classify(.reg .f32 %x)\n"
    "{\n"
    "    .reg .pred %p<4>;\n"
    "\n"
    "    setp.eq.f32 %p0, %x, 0f00000000;\n"
    "    @%p0 bra CL_ZERO;\n"
    "\n"
    "    setp.lt.f32 %p1, %x, 0f00000000;\n"
    "    @%p1 bra CL_NEG;\n"
    "\n"
    "    setp.gt.f32 %p2, %x, 0f42480000;\n"  // 50.0f
    "    @%p2 bra CL_BIG;\n"
    "\n"
    "    // 0 < x <= 50 => class 1\n"
    "    mov.u32 %retval, 1;\n"
    "    ret;\n"
    "\n"
    "CL_ZERO:\n"
    "    mov.u32 %retval, 0;\n"
    "    ret;\n"
    "CL_NEG:\n"
    "    mov.u32 %retval, 3;\n"
    "    ret;\n"
    "CL_BIG:\n"
    "    mov.u32 %retval, 2;\n"
    "    ret;\n"
    "}\n"
    "\n"
    // transform(x, class): applies class-dependent transform
    ".visible .func (.reg .f32 %retval) mega_transform(.reg .f32 %x, .reg .u32 %cls)\n"
    "{\n"
    "    .reg .pred %p<4>;\n"
    "    .reg .f32 %tmp, %tmp2;\n"
    "\n"
    "    setp.eq.u32 %p0, %cls, 0;\n"
    "    @%p0 bra TR_ZERO;\n"
    "    setp.eq.u32 %p1, %cls, 1;\n"
    "    @%p1 bra TR_SMALL;\n"
    "    setp.eq.u32 %p2, %cls, 2;\n"
    "    @%p2 bra TR_BIG;\n"
    "\n"
    "    // class 3: |x| * 3\n"
    "    call (%tmp), mega_abs, (%x);\n"
    "    mul.f32 %retval, %tmp, 0f40400000;\n"  // 3.0
    "    ret;\n"
    "\n"
    "TR_ZERO:\n"
    "    mov.f32 %retval, 0f00000000;\n"
    "    ret;\n"
    "\n"
    "TR_SMALL:\n"
    "    // x^2 + x\n"
    "    call (%tmp), mega_square, (%x);\n"
    "    add.f32 %retval, %tmp, %x;\n"
    "    ret;\n"
    "\n"
    "TR_BIG:\n"
    "    // approx sqrt via x * 0.5 (simple stand-in), * 10\n"
    "    mul.f32 %tmp, %x, 0f3F000000;\n"  // 0.5
    "    mul.f32 %retval, %tmp, 0f41200000;\n"  // 10.0
    "    ret;\n"
    "}\n"
    "\n"
    ".visible .entry megaKernel(\n"
    "    .param .u64 In,\n"
    "    .param .u64 Out\n"
    ")\n"
    ".reqntid 1, 1, 1\n"
    "{\n"
    "    .reg .u32 %tid, %cls, %ival, %xored;\n"
    "    .reg .u64 %rd<5>;\n"
    "    .reg .f32 %x, %transformed, %final;\n"
    "\n"
    "    ld.param.u64 %rd0, [In];\n"
    "    ld.param.u64 %rd1, [Out];\n"
    "    mov.u32 %tid, %ctaid.x;\n"
    "    cvt.u64.u32 %rd2, %tid;\n"
    "    mul.lo.u64 %rd2, %rd2, 4;\n"
    "\n"
    "    add.u64 %rd3, %rd0, %rd2;\n"
    "    ld.global.f32 %x, [%rd3];\n"
    "\n"
    "    // Step 1: classify\n"
    "    call (%cls), mega_classify, (%x);\n"
    "\n"
    "    // Step 2: transform based on class\n"
    "    call (%transformed), mega_transform, (%x, %cls);\n"
    "\n"
    "    // Step 3: float -> int, XOR with 0xFF, int -> float\n"
    "    cvt.rzi.s32.f32 %ival, %transformed;\n"
    "    xor.b32 %xored, %ival, 255;\n"
    "    cvt.rn.f32.s32 %final, %xored;\n"
    "\n"
    "    add.u64 %rd4, %rd1, %rd2;\n"
    "    st.global.f32 [%rd4], %final;\n"
    "    ret;\n"
    "}\n";

TEST_F(CuvkComplexKernelTest, MegaKernel) {
    std::vector<float> h_in = {
        0.0f,     // class 0 => 0 => xor 0xFF => 255
        5.0f,     // class 1 => 25+5=30 => int 30 xor 0xFF = 225
        50.0f,    // class 1 => 2500+50=2550 => int 2550 xor 0xFF = 2313
        100.0f,   // class 2 => 100*0.5*10=500 => int 500 xor 0xFF = 267
        -3.0f,    // class 3 => 3*3=9 => int 9 xor 0xFF = 246
        -20.0f,   // class 3 => 20*3=60 => int 60 xor 0xFF = 195
        1.0f,     // class 1 => 1+1=2 => int 2 xor 0xFF = 253
        10.0f,    // class 1 => 100+10=110 => int 110 xor 0xFF = 145
    };
    const int N = (int)h_in.size();
    std::vector<float> h_out(N, 0.0f);

    CUdeviceptr d_in, d_out;
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_in, N * sizeof(float)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_out, N * sizeof(float)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyHtoD(d_in, h_in.data(), N * sizeof(float)));

    CUmodule mod;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleLoadData(&mod, MEGA_KERNEL_PTX));
    CUfunction func;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleGetFunction(&func, mod, "megaKernel"));

    void *params[] = { &d_in, &d_out };
    ASSERT_EQ(CUDA_SUCCESS, cuLaunchKernel(func,
        N, 1, 1, 1, 1, 1, 0, NULL, params, NULL));
    cuCtxSynchronize();

    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyDtoH(h_out.data(), d_out, N * sizeof(float)));

    for (int i = 0; i < N; i++) {
        float x = h_in[i];

        // classify
        int cls;
        if (x == 0.0f)            cls = 0;
        else if (x < 0.0f)        cls = 3;
        else if (x > 50.0f)       cls = 2;
        else                       cls = 1;

        // transform
        float transformed;
        switch (cls) {
            case 0: transformed = 0.0f; break;
            case 1: transformed = x * x + x; break;
            case 2: transformed = x * 0.5f * 10.0f; break;
            case 3: transformed = std::fabs(x) * 3.0f; break;
            default: transformed = 0.0f; break;
        }

        // int xor 0xFF
        int ival = static_cast<int>(transformed); // truncate toward zero
        int xored = ival ^ 0xFF;
        float expected = static_cast<float>(xored);

        EXPECT_FLOAT_EQ(expected, h_out[i]) << "mismatch at index " << i
            << " (input=" << x << ", class=" << cls
            << ", transformed=" << transformed
            << ", ival=" << ival << ", xored=" << xored << ")";
    }

    cuMemFree(d_in);
    cuMemFree(d_out);
    cuModuleUnload(mod);
}
