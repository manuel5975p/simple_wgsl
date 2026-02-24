#include <gtest/gtest.h>
#include <vector>
extern "C" {
#include "cuda.h"
#include "simple_wgsl.h"
}

// Vector add WITH scalar N parameter -- only works in BDA mode
static const char *VECTOR_ADD_N_PTX =
    ".version 7.0\n"
    ".target sm_70\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry vecAddN(\n"
    "    .param .u64 A,\n"
    "    .param .u64 B,\n"
    "    .param .u64 C,\n"
    "    .param .u32 N\n"
    ")\n"
    ".reqntid 1, 1, 1\n"
    "{\n"
    "    .reg .u32 %tid, %n;\n"
    "    .reg .u64 %rd<7>;\n"
    "    .reg .f32 %f<3>;\n"
    "    .reg .pred %p;\n"
    "\n"
    "    mov.u32 %tid, %ctaid.x;\n"
    "    ld.param.u32 %n, [N];\n"
    "    setp.ge.u32 %p, %tid, %n;\n"
    "    @%p bra DONE;\n"
    "\n"
    "    ld.param.u64 %rd0, [A];\n"
    "    ld.param.u64 %rd1, [B];\n"
    "    ld.param.u64 %rd2, [C];\n"
    "\n"
    "    cvt.u64.u32 %rd3, %tid;\n"
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
    "DONE:\n"
    "    ret;\n"
    "}\n";

// Test BDA mode at the SSIR level
TEST(CuvkBda, PtxParsesWithBda) {
    SsirModule *mod = NULL;
    char *err = NULL;
    PtxToSsirOptions opts = {};
    opts.use_bda = 1;
    opts.preserve_names = 1;
    ASSERT_EQ(PTX_TO_SSIR_OK, ptx_to_ssir(VECTOR_ADD_N_PTX, &opts, &mod, &err));
    EXPECT_NE(nullptr, mod);
    if (err) {
        printf("Parse error: %s\n", err);
    }
    ptx_to_ssir_free(err);
    ssir_module_destroy(mod);
}

TEST(CuvkBda, BdaGeneratesValidSpirv) {
    SsirModule *mod = NULL;
    char *err = NULL;
    PtxToSsirOptions opts = {};
    opts.use_bda = 1;
    opts.preserve_names = 1;
    ASSERT_EQ(PTX_TO_SSIR_OK, ptx_to_ssir(VECTOR_ADD_N_PTX, &opts, &mod, &err));
    ptx_to_ssir_free(err);

    uint32_t *words = NULL;
    size_t count = 0;
    SsirToSpirvOptions spirv_opts = {};
    spirv_opts.spirv_version = 0x00010500;
    spirv_opts.enable_debug_names = 1;
    EXPECT_EQ(SSIR_TO_SPIRV_OK, ssir_to_spirv(mod, &spirv_opts, &words, &count));
    EXPECT_GT(count, 0u);
    if (words) ssir_to_spirv_free(words);
    ssir_module_destroy(mod);
}

// Verify that the push constant struct has the right member count
TEST(CuvkBda, PushConstantStructHasFourMembers) {
    SsirModule *mod = NULL;
    char *err = NULL;
    PtxToSsirOptions opts = {};
    opts.use_bda = 1;
    opts.preserve_names = 1;
    ASSERT_EQ(PTX_TO_SSIR_OK, ptx_to_ssir(VECTOR_ADD_N_PTX, &opts, &mod, &err));
    ptx_to_ssir_free(err);

    // Find the push constant global
    bool found_pc = false;
    for (uint32_t i = 0; i < mod->global_count; i++) {
        SsirGlobalVar *g = &mod->globals[i];
        SsirType *t = ssir_get_type(mod, g->type);
        if (!t || t->kind != SSIR_TYPE_PTR) continue;
        if (t->ptr.space != SSIR_ADDR_PUSH_CONSTANT) continue;

        // Found push constant -- check struct
        SsirType *st = ssir_get_type(mod, t->ptr.pointee);
        ASSERT_NE(nullptr, st);
        ASSERT_EQ(SSIR_TYPE_STRUCT, st->kind);
        // 4 kernel params + 3 hidden ntid members (__ntid_x, __ntid_y, __ntid_z)
        EXPECT_EQ(7u, st->struc.member_count);

        // Check member types: u64, u64, u64, u32, u32, u32, u32
        if (st->struc.member_count >= 7) {
            SsirType *m0 = ssir_get_type(mod, st->struc.members[0]);
            SsirType *m1 = ssir_get_type(mod, st->struc.members[1]);
            SsirType *m2 = ssir_get_type(mod, st->struc.members[2]);
            SsirType *m3 = ssir_get_type(mod, st->struc.members[3]);
            EXPECT_EQ(SSIR_TYPE_U64, m0->kind);
            EXPECT_EQ(SSIR_TYPE_U64, m1->kind);
            EXPECT_EQ(SSIR_TYPE_U64, m2->kind);
            EXPECT_EQ(SSIR_TYPE_U32, m3->kind);
            // Hidden ntid members
            SsirType *m4 = ssir_get_type(mod, st->struc.members[4]);
            SsirType *m5 = ssir_get_type(mod, st->struc.members[5]);
            SsirType *m6 = ssir_get_type(mod, st->struc.members[6]);
            EXPECT_EQ(SSIR_TYPE_U32, m4->kind);
            EXPECT_EQ(SSIR_TYPE_U32, m5->kind);
            EXPECT_EQ(SSIR_TYPE_U32, m6->kind);
        }

        found_pc = true;
        break;
    }
    EXPECT_TRUE(found_pc) << "No push constant global found in BDA mode";

    // Should NOT have any storage buffer globals
    for (uint32_t i = 0; i < mod->global_count; i++) {
        SsirGlobalVar *g = &mod->globals[i];
        SsirType *t = ssir_get_type(mod, g->type);
        if (!t || t->kind != SSIR_TYPE_PTR) continue;
        EXPECT_NE(SSIR_ADDR_STORAGE, t->ptr.space)
            << "BDA mode should not create storage buffer globals";
    }

    ssir_module_destroy(mod);
}

// Verify that non-BDA mode still works (regression test)
TEST(CuvkBda, NonBdaStillWorks) {
    // Simple PTX with only pointer params (no scalar)
    static const char *SIMPLE_PTX =
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
        "    .reg .u32 %tid;\n"
        "    .reg .u64 %rd<7>;\n"
        "    .reg .f32 %f<3>;\n"
        "\n"
        "    mov.u32 %tid, %ctaid.x;\n"
        "    ld.param.u64 %rd0, [A];\n"
        "    ld.param.u64 %rd1, [B];\n"
        "    ld.param.u64 %rd2, [C];\n"
        "    cvt.u64.u32 %rd3, %tid;\n"
        "    mul.lo.u64 %rd3, %rd3, 4;\n"
        "    add.u64 %rd4, %rd0, %rd3;\n"
        "    ld.global.f32 %f0, [%rd4];\n"
        "    add.u64 %rd5, %rd1, %rd3;\n"
        "    ld.global.f32 %f1, [%rd5];\n"
        "    add.f32 %f2, %f0, %f1;\n"
        "    add.u64 %rd6, %rd2, %rd3;\n"
        "    st.global.f32 [%rd6], %f2;\n"
        "    ret;\n"
        "}\n";

    SsirModule *mod = NULL;
    char *err = NULL;
    PtxToSsirOptions opts = {};
    opts.use_bda = 0;
    opts.preserve_names = 1;
    ASSERT_EQ(PTX_TO_SSIR_OK, ptx_to_ssir(SIMPLE_PTX, &opts, &mod, &err));
    ptx_to_ssir_free(err);

    uint32_t *words = NULL;
    size_t count = 0;
    SsirToSpirvOptions spirv_opts = {};
    spirv_opts.spirv_version = 0x00010300;
    spirv_opts.enable_debug_names = 1;
    EXPECT_EQ(SSIR_TO_SPIRV_OK, ssir_to_spirv(mod, &spirv_opts, &words, &count));
    EXPECT_GT(count, 0u);

    // Should have storage buffer globals, not push constants
    bool has_storage = false;
    for (uint32_t i = 0; i < mod->global_count; i++) {
        SsirGlobalVar *g = &mod->globals[i];
        SsirType *t = ssir_get_type(mod, g->type);
        if (t && t->kind == SSIR_TYPE_PTR && t->ptr.space == SSIR_ADDR_STORAGE) {
            has_storage = true;
            break;
        }
    }
    EXPECT_TRUE(has_storage) << "Non-BDA mode should create storage buffer globals";

    if (words) ssir_to_spirv_free(words);
    ssir_module_destroy(mod);
}

// Full end-to-end BDA test with scalar parameter
class CuvkBdaTest : public ::testing::Test {
protected:
    CUcontext ctx = NULL;
    bool has_bda = false;
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

TEST_F(CuvkBdaTest, VectorAddWithN) {
    // This test only works if BDA is available
    // The runtime auto-selects BDA mode when available
    const int N = 128;
    std::vector<float> h_a(N), h_b(N), h_c(N);
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 3);
    }

    CUdeviceptr d_a, d_b, d_c;
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_a, N * sizeof(float)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_b, N * sizeof(float)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&d_c, N * sizeof(float)));
    cuMemcpyHtoD(d_a, h_a.data(), N * sizeof(float));
    cuMemcpyHtoD(d_b, h_b.data(), N * sizeof(float));

    CUmodule mod;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleLoadData(&mod, VECTOR_ADD_N_PTX));
    CUfunction func;
    ASSERT_EQ(CUDA_SUCCESS, cuModuleGetFunction(&func, mod, "vecAddN"));

    unsigned int n = N;
    void *params[] = { &d_a, &d_b, &d_c, &n };
    // Launch with more blocks than N to test bounds check
    ASSERT_EQ(CUDA_SUCCESS, cuLaunchKernel(func,
        N + 32, 1, 1,  // grid: more blocks than elements
        1, 1, 1,        // block: 1 thread
        0, NULL, params, NULL));

    cuCtxSynchronize();
    cuMemcpyDtoH(h_c.data(), d_c, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(h_a[i] + h_b[i], h_c[i]) << "mismatch at i=" << i;
    }

    cuMemFree(d_a);
    cuMemFree(d_b);
    cuMemFree(d_c);
    cuModuleUnload(mod);
}
