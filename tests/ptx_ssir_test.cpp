/*
 * ptx_ssir_test.cpp — Tests that PTX parsing produces correct SSIR
 *
 * These tests verify the actual SSIR instructions, types, globals,
 * entry points, and SPIR-V output — not just "did it parse without error."
 */

#include <gtest/gtest.h>
#include "test_utils.h"
#include <string>
#include <vector>
extern "C" {
#include "simple_wgsl.h"
}

/* ============================================================================
 * Helpers
 * ============================================================================ */

class SsirGuard2 {
  public:
    explicit SsirGuard2(SsirModule *m) : m_(m) {}
    ~SsirGuard2() { if (m_) ssir_module_destroy(m_); }
    SsirModule *get() { return m_; }
    SsirModule *operator->() { return m_; }
  private:
    SsirModule *m_;
    SsirGuard2(const SsirGuard2 &) = delete;
    SsirGuard2 &operator=(const SsirGuard2 &) = delete;
};

static const char *PTX_HEADER =
    ".version 7.8\n.target sm_80\n.address_size 64\n";

struct PtxParseResult {
    bool success;
    std::string error;
    SsirModule *mod;
};

static PtxParseResult Parse(const std::string &ptx) {
    PtxParseResult r = {false, "", nullptr};
    char *err = nullptr;
    PtxToSsirOptions opts = {};
    opts.preserve_names = 1;
    PtxToSsirResult pr = ptx_to_ssir(ptx.c_str(), &opts, &r.mod, &err);
    if (pr != PTX_TO_SSIR_OK) {
        r.error = err ? err : "unknown";
        ptx_to_ssir_free(err);
        return r;
    }
    ptx_to_ssir_free(err);
    r.success = true;
    return r;
}

struct SpirvOutput {
    bool success;
    std::string error;
    std::vector<uint32_t> words;
};

static SpirvOutput EmitSpirv(SsirModule *mod) {
    SpirvOutput out;
    out.success = false;
    uint32_t *words = nullptr;
    size_t count = 0;
    SsirToSpirvOptions opts = {};
    opts.spirv_version = 0x00010300;
    opts.enable_debug_names = 1;
    SsirToSpirvResult sr = ssir_to_spirv(mod, &opts, &words, &count);
    if (sr != SSIR_TO_SPIRV_OK) {
        out.error = ssir_to_spirv_result_string(sr);
        return out;
    }
    out.words.assign(words, words + count);
    ssir_to_spirv_free(words);
    out.success = true;
    return out;
}

// Count instructions with a given opcode in a block
static uint32_t CountOps(SsirBlock *b, SsirOpcode op) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < b->inst_count; i++)
        if (b->insts[i].op == op) n++;
    return n;
}

// Count total instructions across all blocks in a function (excluding terminators)
static uint32_t TotalInsts(SsirFunction *f) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < f->block_count; i++)
        n += f->blocks[i].inst_count;
    return n;
}


/* ============================================================================
 * Lexer: verify token boundaries
 * ============================================================================ */

// Make sure instructions like "ld.global.f32" don't get merged into one token
// causing silent instruction skipping
TEST(PtxSsir, LexerInstructionTokenBoundary) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry lex_test()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<1>;
            .reg .u32 %r<1>;

            mov.u32 %r0, %tid.x;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    // mov emits: load from special reg global + store to %r0's local var
    // ret emits: RETURN_VOID
    // We expect more than just a single RETURN_VOID
    ASSERT_GE(f->blocks[0].inst_count, 2u)
        << "mov instruction was not parsed — likely lexer merged tokens";
}

// Verify that "add.f32" is recognized as opcode "add" + type ".f32"
TEST(PtxSsir, LexerArithInstParsing) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry arith()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<3>;
            add.f32 %f2, %f0, %f1;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    SsirBlock *b = &f->blocks[0];
    // add.f32 %f2, %f0, %f1 should produce:
    //   load %f0, load %f1, add, store %f2 = 4 SSIR instructions
    //   + RETURN_VOID = 5
    EXPECT_GE(b->inst_count, 4u)
        << "add.f32 instruction not parsed into SSIR — inst_count="
        << b->inst_count;
    EXPECT_GT(CountOps(b, SSIR_OP_ADD), 0u)
        << "No SSIR_OP_ADD found";
}

/* ============================================================================
 * Entry point structure
 * ============================================================================ */

TEST(PtxSsir, EntryPointNoParams) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry ep_test(
            .param .u64 a_ptr,
            .param .u64 b_ptr
        )
        .reqntid 64, 1, 1
        {
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    ASSERT_EQ(r.mod->entry_point_count, 1u);
    SsirEntryPoint *ep = &r.mod->entry_points[0];
    EXPECT_EQ(ep->stage, SSIR_STAGE_COMPUTE);
    EXPECT_STREQ(ep->name, "ep_test");
    EXPECT_EQ(ep->workgroup_size[0], 64u);
    EXPECT_EQ(ep->workgroup_size[1], 1u);
    EXPECT_EQ(ep->workgroup_size[2], 1u);

    // Entry point function should have NO parameters for valid SPIR-V
    SsirFunction *f = &r.mod->functions[0];
    EXPECT_EQ(f->param_count, 0u)
        << "SPIR-V entry points must not have function parameters; "
           "found " << f->param_count;
}

TEST(PtxSsir, EntryPointBindings) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry bind_test(
            .param .u64 input,
            .param .u64 output
        )
        {
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    // Two .u64 params → two storage buffer globals
    ASSERT_GE(r.mod->global_count, 2u);

    // Check group=0, binding=0 and binding=1
    bool found_b0 = false, found_b1 = false;
    for (uint32_t i = 0; i < r.mod->global_count; i++) {
        SsirGlobalVar *gv = &r.mod->globals[i];
        if (gv->has_binding && gv->binding == 0 && gv->has_group && gv->group == 0)
            found_b0 = true;
        if (gv->has_binding && gv->binding == 1 && gv->has_group && gv->group == 0)
            found_b1 = true;
    }
    EXPECT_TRUE(found_b0) << "No global with @group(0) @binding(0)";
    EXPECT_TRUE(found_b1) << "No global with @group(0) @binding(1)";
}

TEST(PtxSsir, EntryPointInterface) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry iface_test(
            .param .u64 a,
            .param .u64 b
        )
        {
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    ASSERT_EQ(r.mod->entry_point_count, 1u);
    SsirEntryPoint *ep = &r.mod->entry_points[0];
    // Storage buffer globals must be in the interface list
    EXPECT_GE(ep->interface_count, 2u)
        << "Entry point interface should include storage buffer globals";
}

/* ============================================================================
 * Instruction emission — arithmetic
 * ============================================================================ */

TEST(PtxSsir, AddF32EmitsSSIR) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry add_test()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<3>;
            add.f32 %f2, %f0, %f1;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    SsirBlock *b = &r.mod->functions[0].blocks[0];
    EXPECT_GT(CountOps(b, SSIR_OP_ADD), 0u) << "add.f32 should emit SSIR_OP_ADD";
    EXPECT_GT(CountOps(b, SSIR_OP_LOAD), 0u) << "operand reads should emit SSIR_OP_LOAD";
    EXPECT_GT(CountOps(b, SSIR_OP_STORE), 0u) << "register write should emit SSIR_OP_STORE";
}

TEST(PtxSsir, SubU32EmitsSSIR) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry sub_test()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<3>;
            sub.u32 %r2, %r0, %r1;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    EXPECT_GT(CountOps(&r.mod->functions[0].blocks[0], SSIR_OP_SUB), 0u);
}

TEST(PtxSsir, MulLoU32EmitsSSIR) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry mul_test()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<3>;
            mul.lo.u32 %r2, %r0, %r1;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    EXPECT_GT(CountOps(&r.mod->functions[0].blocks[0], SSIR_OP_MUL), 0u);
}

TEST(PtxSsir, DivF32EmitsSSIR) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry div_test()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<3>;
            div.f32 %f2, %f0, %f1;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    EXPECT_GT(CountOps(&r.mod->functions[0].blocks[0], SSIR_OP_DIV), 0u);
}

TEST(PtxSsir, NegF32EmitsSSIR) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry neg_test()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<2>;
            neg.f32 %f1, %f0;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    EXPECT_GT(CountOps(&r.mod->functions[0].blocks[0], SSIR_OP_NEG), 0u);
}

/* ============================================================================
 * Instruction emission — bitwise
 * ============================================================================ */

TEST(PtxSsir, AndB32EmitsSSIR) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry and_test()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<3>;
            and.b32 %r2, %r0, %r1;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    EXPECT_GT(CountOps(&r.mod->functions[0].blocks[0], SSIR_OP_BIT_AND), 0u);
}

TEST(PtxSsir, OrB32EmitsSSIR) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry or_test()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<3>;
            or.b32 %r2, %r0, %r1;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    EXPECT_GT(CountOps(&r.mod->functions[0].blocks[0], SSIR_OP_BIT_OR), 0u);
}

TEST(PtxSsir, ShlB32EmitsSSIR) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry shl_test()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<3>;
            shl.b32 %r2, %r0, %r1;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    EXPECT_GT(CountOps(&r.mod->functions[0].blocks[0], SSIR_OP_SHL), 0u);
}

/* ============================================================================
 * Instruction emission — comparison
 * ============================================================================ */

TEST(PtxSsir, SetpEmitsComparison) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry setp_test()
        .reqntid 1, 1, 1
        {
            .reg .pred %p<1>;
            .reg .f32 %f<2>;
            setp.gt.f32 %p0, %f0, %f1;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    EXPECT_GT(CountOps(&r.mod->functions[0].blocks[0], SSIR_OP_GT), 0u);
}

TEST(PtxSsir, SelpEmitsBuiltin) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry selp_test()
        .reqntid 1, 1, 1
        {
            .reg .pred %p<1>;
            .reg .f32 %f<3>;
            selp.f32 %f2, %f0, %f1, %p0;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    EXPECT_GT(CountOps(&r.mod->functions[0].blocks[0], SSIR_OP_BUILTIN), 0u)
        << "selp should emit SSIR_OP_BUILTIN (SELECT)";
}

/* ============================================================================
 * Instruction emission — memory
 * ============================================================================ */

TEST(PtxSsir, LdParamCreatesLoad) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry ldparam_test(
            .param .u64 ptr
        )
        {
            .reg .u64 %rd<1>;
            ld.param.u64 %rd0, [ptr];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    SsirBlock *b = &r.mod->functions[0].blocks[0];
    // ld.param should produce a LOAD + STORE (load param local, store to %rd0)
    EXPECT_GT(CountOps(b, SSIR_OP_LOAD), 0u) << "ld.param should emit LOAD";
    EXPECT_GT(CountOps(b, SSIR_OP_STORE), 0u) << "ld.param should store to dest reg";
}

TEST(PtxSsir, LdGlobalCreatesLoad) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry ldglobal_test(
            .param .u64 ptr
        )
        {
            .reg .u64 %rd<2>;
            .reg .f32 %f<1>;
            ld.param.u64 %rd0, [ptr];
            ld.global.f32 %f0, [%rd0];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    SsirBlock *b = &r.mod->functions[0].blocks[0];
    // Should have at least 2 LOADs: one for ld.param, one for ld.global
    EXPECT_GE(CountOps(b, SSIR_OP_LOAD), 2u)
        << "Expected LOADs for both ld.param and ld.global";
}

TEST(PtxSsir, StGlobalCreatesStore) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry stglobal_test(
            .param .u64 ptr
        )
        {
            .reg .u64 %rd<1>;
            .reg .f32 %f<1>;
            ld.param.u64 %rd0, [ptr];
            st.global.f32 [%rd0], %f0;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    SsirBlock *b = &r.mod->functions[0].blocks[0];
    // At least 2 STOREs: one from ld.param storing to reg, one for st.global
    EXPECT_GE(CountOps(b, SSIR_OP_STORE), 2u)
        << "Expected STOREs for ld.param dest and st.global";
}

/* ============================================================================
 * Instruction emission — mov + special registers
 * ============================================================================ */

TEST(PtxSsir, MovU32EmitsLoadStore) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry mov_test()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<2>;
            mov.u32 %r1, %r0;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    SsirBlock *b = &r.mod->functions[0].blocks[0];
    // mov %r1, %r0 → load %r0 + store %r1
    EXPECT_GT(CountOps(b, SSIR_OP_LOAD), 0u) << "mov should load src reg";
    EXPECT_GT(CountOps(b, SSIR_OP_STORE), 0u) << "mov should store to dst reg";
}

TEST(PtxSsir, MovTidXCreatesBuiltinGlobal) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry tid_test()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<1>;
            mov.u32 %r0, %tid.x;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    // Should create a builtin global for local_invocation_id
    bool found_builtin = false;
    for (uint32_t i = 0; i < r.mod->global_count; i++) {
        if (r.mod->globals[i].builtin == SSIR_BUILTIN_LOCAL_INVOCATION_ID)
            found_builtin = true;
    }
    EXPECT_TRUE(found_builtin) << "mov %tid.x should create LOCAL_INVOCATION_ID builtin global";

    SsirBlock *b = &r.mod->functions[0].blocks[0];
    // Should have LOAD (from builtin) + EXTRACT (component .x) + STORE (to %r0)
    EXPECT_GT(CountOps(b, SSIR_OP_LOAD), 0u);
    EXPECT_GT(CountOps(b, SSIR_OP_STORE), 0u);
}

TEST(PtxSsir, MovCtaidXCreatesBuiltinGlobal) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry ctaid_test()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<1>;
            mov.u32 %r0, %ctaid.x;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    bool found_builtin = false;
    for (uint32_t i = 0; i < r.mod->global_count; i++) {
        if (r.mod->globals[i].builtin == SSIR_BUILTIN_WORKGROUP_ID)
            found_builtin = true;
    }
    EXPECT_TRUE(found_builtin) << "mov %ctaid.x should create WORKGROUP_ID builtin global";
}

/* ============================================================================
 * Instruction emission — type conversion
 * ============================================================================ */

TEST(PtxSsir, CvtEmitsConvert) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry cvt_test()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<1>;
            .reg .u64 %rd<1>;
            cvt.u64.u32 %rd0, %r0;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    EXPECT_GT(CountOps(&r.mod->functions[0].blocks[0], SSIR_OP_CONVERT), 0u)
        << "cvt should emit SSIR_OP_CONVERT";
}

TEST(PtxSsir, CvtF32S32EmitsConvert) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry cvt_f32_test()
        .reqntid 1, 1, 1
        {
            .reg .s32 %r<1>;
            .reg .f32 %f<1>;
            cvt.rn.f32.s32 %f0, %r0;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    EXPECT_GT(CountOps(&r.mod->functions[0].blocks[0], SSIR_OP_CONVERT), 0u);
}

/* ============================================================================
 * Instruction emission — FMA/MAD
 * ============================================================================ */

TEST(PtxSsir, FmaF32EmitsBuiltin) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry fma_test()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<4>;
            fma.rn.f32 %f3, %f0, %f1, %f2;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    EXPECT_GT(CountOps(&r.mod->functions[0].blocks[0], SSIR_OP_BUILTIN), 0u)
        << "fma should emit SSIR_OP_BUILTIN (FMA)";
}

TEST(PtxSsir, MadLoU32EmitsMulAdd) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry mad_test()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<4>;
            mad.lo.u32 %r3, %r0, %r1, %r2;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    SsirBlock *b = &r.mod->functions[0].blocks[0];
    // Integer mad emits MUL + ADD
    EXPECT_GT(CountOps(b, SSIR_OP_MUL), 0u) << "mad.lo.u32 should emit MUL";
    EXPECT_GT(CountOps(b, SSIR_OP_ADD), 0u) << "mad.lo.u32 should emit ADD";
}

/* ============================================================================
 * Instruction emission — math builtins
 * ============================================================================ */

TEST(PtxSsir, SqrtEmitsBuiltin) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry sqrt_test()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<2>;
            sqrt.f32 %f1, %f0;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    EXPECT_GT(CountOps(&r.mod->functions[0].blocks[0], SSIR_OP_BUILTIN), 0u)
        << "sqrt should emit SSIR_OP_BUILTIN";
}

TEST(PtxSsir, MinMaxEmitsBuiltin) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry minmax_test()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<4>;
            min.f32 %f2, %f0, %f1;
            max.f32 %f3, %f0, %f1;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    // min and max each emit SSIR_OP_BUILTIN
    EXPECT_GE(CountOps(&r.mod->functions[0].blocks[0], SSIR_OP_BUILTIN), 2u)
        << "min + max should each emit SSIR_OP_BUILTIN";
}

/* ============================================================================
 * Control flow
 * ============================================================================ */

TEST(PtxSsir, BranchCreatesMultipleBlocks) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry bra_test()
        .reqntid 1, 1, 1
        {
            .reg .pred %p<1>;
            .reg .f32 %f<1>;

            setp.gt.f32 %p0, %f0, %f0;
            @%p0 bra LABEL;
            ret;
        LABEL:
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GE(f->block_count, 2u) << "Branch should create at least 2 basic blocks";
}

TEST(PtxSsir, ConditionalBranchEmitsBranchCond) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry cond_bra_test()
        .reqntid 1, 1, 1
        {
            .reg .pred %p<1>;
            .reg .f32 %f<2>;
            setp.gt.f32 %p0, %f0, %f1;
            @%p0 bra DONE;
            ret;
        DONE:
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    // First block should end with BRANCH_COND
    SsirBlock *b = &r.mod->functions[0].blocks[0];
    bool has_branch_cond = false;
    for (uint32_t i = 0; i < b->inst_count; i++) {
        if (b->insts[i].op == SSIR_OP_BRANCH_COND) has_branch_cond = true;
    }
    EXPECT_TRUE(has_branch_cond) << "Predicated bra should emit BRANCH_COND";
}

/* ============================================================================
 * Multi-instruction sequences
 * ============================================================================ */

// A sequence of: ld.param, mov, cvt, mul, add, ld.global — should all produce SSIR
TEST(PtxSsir, MultiInstructionSequence) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry multi_test(
            .param .u64 a_ptr
        )
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<1>;
            .reg .u32 %r<1>;
            .reg .u64 %rd<4>;

            ld.param.u64 %rd0, [a_ptr];
            mov.u32 %r0, %tid.x;
            cvt.u64.u32 %rd1, %r0;
            mul.lo.u64 %rd2, %rd1, 4;
            add.u64 %rd3, %rd0, %rd2;
            ld.global.f32 %f0, [%rd3];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    uint32_t total = TotalInsts(f);

    // Minimum expected:
    //   ld.param: 1 load + 1 store = 2
    //   mov %tid.x: 1 load(builtin) + 1 extract + 1 store = 3
    //   cvt: 1 load + 1 convert + 1 store = 3
    //   mul: 2 load + 1 mul + 1 store = 4
    //   add: 2 load + 1 add + 1 store = 4
    //   ld.global: 1 load(reg) + 1 load(global) + 1 store = 3
    //   ret: 1
    //   Total: ~20
    EXPECT_GE(total, 15u)
        << "Multi-instruction sequence should produce at least 15 SSIR instructions, got "
        << total;

    // Verify specific opcodes exist
    bool has_add = false, has_mul = false, has_cvt = false, has_load = false;
    for (uint32_t bi = 0; bi < f->block_count; bi++) {
        SsirBlock *b = &f->blocks[bi];
        for (uint32_t i = 0; i < b->inst_count; i++) {
            if (b->insts[i].op == SSIR_OP_ADD) has_add = true;
            if (b->insts[i].op == SSIR_OP_MUL) has_mul = true;
            if (b->insts[i].op == SSIR_OP_CONVERT) has_cvt = true;
            if (b->insts[i].op == SSIR_OP_LOAD) has_load = true;
        }
    }
    EXPECT_TRUE(has_add) << "Missing ADD instruction";
    EXPECT_TRUE(has_mul) << "Missing MUL instruction";
    EXPECT_TRUE(has_cvt) << "Missing CONVERT instruction";
    EXPECT_TRUE(has_load) << "Missing LOAD instruction";
}

/* ============================================================================
 * Full kernel instruction count
 * ============================================================================ */

TEST(PtxSsir, VectorAddKernelInstructionCount) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry vec_add(
            .param .u64 a_ptr,
            .param .u64 b_ptr,
            .param .u64 c_ptr
        )
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<3>;
            .reg .u32 %r<1>;
            .reg .u64 %rd<7>;

            ld.param.u64 %rd0, [a_ptr];
            ld.param.u64 %rd1, [b_ptr];
            ld.param.u64 %rd2, [c_ptr];
            mov.u32 %r0, %tid.x;
            cvt.u64.u32 %rd3, %r0;
            mul.lo.u64 %rd3, %rd3, 4;
            add.u64 %rd4, %rd0, %rd3;
            ld.global.f32 %f0, [%rd4];
            add.u64 %rd5, %rd1, %rd3;
            ld.global.f32 %f1, [%rd5];
            add.f32 %f2, %f0, %f1;
            add.u64 %rd6, %rd2, %rd3;
            st.global.f32 [%rd6], %f2;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    uint32_t total = TotalInsts(f);
    // This kernel has 14 PTX instructions. Each produces multiple SSIR ops.
    // If we get < 10, something is badly wrong (instructions being skipped).
    EXPECT_GE(total, 20u)
        << "Vector add kernel should produce lots of SSIR instructions, got " << total
        << " — likely instructions are being silently skipped by the parser";

    // Check for the key operations
    bool has_add_f32 = false;
    for (uint32_t bi = 0; bi < f->block_count; bi++) {
        SsirBlock *b = &f->blocks[bi];
        for (uint32_t ii = 0; ii < b->inst_count; ii++) {
            SsirInst *inst = &b->insts[ii];
            if (inst->op == SSIR_OP_ADD && inst->type != 0) {
                SsirType *t = ssir_get_type(r.mod, inst->type);
                if (t && t->kind == SSIR_TYPE_F32)
                    has_add_f32 = true;
            }
        }
    }
    EXPECT_TRUE(has_add_f32) << "Should have at least one f32 ADD for c[i] = a[i] + b[i]";
}

/* ============================================================================
 * SPIR-V output validity
 * ============================================================================ */

TEST(PtxSsir, EmptyKernelValidSpirv) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry empty()
        .reqntid 1, 1, 1
        {
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    auto spv = EmitSpirv(r.mod);
    ASSERT_TRUE(spv.success) << spv.error;

    std::string val_err;
    EXPECT_TRUE(wgsl_test::ValidateSpirv(spv.words.data(), spv.words.size(), &val_err))
        << "spirv-val: " << val_err;
}

TEST(PtxSsir, KernelWithParamsValidSpirv) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry with_params(
            .param .u64 a,
            .param .u64 b
        )
        .reqntid 1, 1, 1
        {
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    auto spv = EmitSpirv(r.mod);
    ASSERT_TRUE(spv.success) << spv.error;

    std::string val_err;
    EXPECT_TRUE(wgsl_test::ValidateSpirv(spv.words.data(), spv.words.size(), &val_err))
        << "spirv-val: " << val_err;
}

TEST(PtxSsir, ArithKernelValidSpirv) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry arith()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<3>;
            add.f32 %f2, %f0, %f1;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    auto spv = EmitSpirv(r.mod);
    ASSERT_TRUE(spv.success) << spv.error;

    std::string val_err;
    EXPECT_TRUE(wgsl_test::ValidateSpirv(spv.words.data(), spv.words.size(), &val_err))
        << "spirv-val: " << val_err;
}

TEST(PtxSsir, FullVecAddValidSpirv) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry vec_add(
            .param .u64 a_ptr,
            .param .u64 b_ptr,
            .param .u64 c_ptr
        )
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<3>;
            .reg .u32 %r<1>;
            .reg .u64 %rd<7>;

            ld.param.u64 %rd0, [a_ptr];
            ld.param.u64 %rd1, [b_ptr];
            ld.param.u64 %rd2, [c_ptr];
            mov.u32 %r0, %tid.x;
            cvt.u64.u32 %rd3, %r0;
            mul.lo.u64 %rd3, %rd3, 4;
            add.u64 %rd4, %rd0, %rd3;
            ld.global.f32 %f0, [%rd4];
            add.u64 %rd5, %rd1, %rd3;
            ld.global.f32 %f1, [%rd5];
            add.f32 %f2, %f0, %f1;
            add.u64 %rd6, %rd2, %rd3;
            st.global.f32 [%rd6], %f2;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    auto spv = EmitSpirv(r.mod);
    ASSERT_TRUE(spv.success) << spv.error;

    std::string val_err;
    EXPECT_TRUE(wgsl_test::ValidateSpirv(spv.words.data(), spv.words.size(), &val_err))
        << "spirv-val: " << val_err;
}

/* ============================================================================
 * Atomics / barriers / math
 * ============================================================================ */

TEST(PtxSsir, BarrierSyncEmitsBarrier) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry bar_test()
        .reqntid 64, 1, 1
        {
            bar.sync 0;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    EXPECT_GT(CountOps(&r.mod->functions[0].blocks[0], SSIR_OP_BARRIER), 0u)
        << "bar.sync should emit SSIR_OP_BARRIER";
}

TEST(PtxSsir, RcpEmitsDivision) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry rcp_test()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<2>;
            rcp.f32 %f1, %f0;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    // rcp is 1.0/x, should emit DIV
    EXPECT_GT(CountOps(&r.mod->functions[0].blocks[0], SSIR_OP_DIV), 0u)
        << "rcp should emit SSIR_OP_DIV (1.0/x)";
}

/* ============================================================================
 * Device functions (not entry points)
 * ============================================================================ */

TEST(PtxSsir, DeviceFunctionHasParams) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .func (.reg .f32 rv) my_func(
            .reg .f32 a,
            .reg .f32 b
        )
        {
            .reg .f32 %f<1>;
            add.f32 %f0, a, b;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    ASSERT_GE(r.mod->function_count, 1u);
    SsirFunction *f = &r.mod->functions[0];
    // Device functions CAN have parameters (unlike entry points)
    EXPECT_GE(f->param_count, 2u)
        << "Device function should have function parameters";
}

/* ============================================================================
 * Instruction count for various instructions (catch silent skipping)
 * ============================================================================ */

TEST(PtxSsir, AllArithOpsEmitInstructions) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry all_arith()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<10>;
            .reg .u32 %r<5>;
            .reg .s32 %s<3>;

            add.f32 %f2, %f0, %f1;
            sub.f32 %f3, %f0, %f1;
            mul.f32 %f4, %f0, %f1;
            div.f32 %f5, %f0, %f1;
            neg.f32 %f6, %f0;
            add.u32 %r2, %r0, %r1;
            sub.u32 %r3, %r0, %r1;
            mul.lo.u32 %r4, %r0, %r1;
            rem.s32 %s2, %s0, %s1;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    SsirBlock *b = &r.mod->functions[0].blocks[0];
    // 9 instructions → at least 9 result-producing ops
    EXPECT_GT(CountOps(b, SSIR_OP_ADD), 0u) << "add not emitted";
    EXPECT_GT(CountOps(b, SSIR_OP_SUB), 0u) << "sub not emitted";
    EXPECT_GT(CountOps(b, SSIR_OP_MUL), 0u) << "mul not emitted";
    EXPECT_GT(CountOps(b, SSIR_OP_DIV), 0u) << "div not emitted";
    EXPECT_GT(CountOps(b, SSIR_OP_NEG), 0u) << "neg not emitted";
    EXPECT_GT(CountOps(b, SSIR_OP_REM), 0u) << "rem not emitted";
}

TEST(PtxSsir, AllBitwiseOpsEmitInstructions) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry all_bitwise()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<8>;
            and.b32 %r2, %r0, %r1;
            or.b32 %r3, %r0, %r1;
            xor.b32 %r4, %r0, %r1;
            not.b32 %r5, %r0;
            shl.b32 %r6, %r0, %r1;
            shr.u32 %r7, %r0, %r1;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    SsirBlock *b = &r.mod->functions[0].blocks[0];
    EXPECT_GT(CountOps(b, SSIR_OP_BIT_AND), 0u) << "and not emitted";
    EXPECT_GT(CountOps(b, SSIR_OP_BIT_OR), 0u)  << "or not emitted";
    EXPECT_GT(CountOps(b, SSIR_OP_BIT_XOR), 0u) << "xor not emitted";
    EXPECT_GT(CountOps(b, SSIR_OP_BIT_NOT), 0u) << "not not emitted";
    EXPECT_GT(CountOps(b, SSIR_OP_SHL), 0u)     << "shl not emitted";
    // shr.u32 emits SHR_LOGICAL
    EXPECT_GT(CountOps(b, SSIR_OP_SHR_LOGICAL), 0u) << "shr.u32 not emitted";
}

TEST(PtxSsir, AllComparisonOpsEmitInstructions) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry all_cmp()
        .reqntid 1, 1, 1
        {
            .reg .pred %p<6>;
            .reg .f32 %f<2>;
            setp.eq.f32 %p0, %f0, %f1;
            setp.ne.f32 %p1, %f0, %f1;
            setp.lt.f32 %p2, %f0, %f1;
            setp.le.f32 %p3, %f0, %f1;
            setp.gt.f32 %p4, %f0, %f1;
            setp.ge.f32 %p5, %f0, %f1;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    SsirBlock *b = &r.mod->functions[0].blocks[0];
    EXPECT_GT(CountOps(b, SSIR_OP_EQ), 0u) << "setp.eq not emitted";
    EXPECT_GT(CountOps(b, SSIR_OP_NE), 0u) << "setp.ne not emitted";
    EXPECT_GT(CountOps(b, SSIR_OP_LT), 0u) << "setp.lt not emitted";
    EXPECT_GT(CountOps(b, SSIR_OP_LE), 0u) << "setp.le not emitted";
    EXPECT_GT(CountOps(b, SSIR_OP_GT), 0u) << "setp.gt not emitted";
    EXPECT_GT(CountOps(b, SSIR_OP_GE), 0u) << "setp.ge not emitted";
}

/* ============================================================================
 * nvcc-generated PTX patterns
 * ============================================================================ */

// nvcc generates "cvta.to.global.u64" which should be a no-op
TEST(PtxSsir, CvtaToGlobalIsNoop) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry cvta_test(
            .param .u64 ptr
        )
        {
            .reg .u64 %rd<3>;
            ld.param.u64 %rd0, [ptr];
            cvta.to.global.u64 %rd1, %rd0;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);
    // Just needs to not crash and parse successfully
}

// nvcc uses ".b32" and ".b64" register types
TEST(PtxSsir, NvccBitRegTypes) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry breg_test()
        .reqntid 1, 1, 1
        {
            .reg .b32 %r<3>;
            .reg .b64 %rd<2>;
            add.s32 %r2, %r0, %r1;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    EXPECT_GT(CountOps(&r.mod->functions[0].blocks[0], SSIR_OP_ADD), 0u);
}

// nvcc uses "add.s64" instead of "add.u64"
TEST(PtxSsir, NvccSignedArith) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry s64_test(
            .param .u64 ptr
        )
        {
            .reg .b64 %rd<3>;
            .reg .b32 %r<2>;
            ld.param.u64 %rd0, [ptr];
            mov.u32 %r1, %tid.x;
            mul.wide.s32 %rd1, %r1, 4;
            add.s64 %rd2, %rd0, %rd1;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    uint32_t total = TotalInsts(f);
    // Should have instructions for ld.param, mov, mul.wide, add.s64
    EXPECT_GE(total, 8u)
        << "nvcc-style instructions should produce SSIR ops, got " << total;
}

// nvcc generates "mul.wide.s32" which widens to 64-bit result
TEST(PtxSsir, MulWideS32) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry mulwide_test()
        .reqntid 1, 1, 1
        {
            .reg .b32 %r<2>;
            .reg .b64 %rd<1>;
            mul.wide.s32 %rd0, %r0, %r1;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    SsirBlock *b = &r.mod->functions[0].blocks[0];
    // mul.wide should produce something — either MUL + CONVERT or just MUL
    uint32_t total = b->inst_count;
    EXPECT_GE(total, 3u)
        << "mul.wide.s32 should produce SSIR instructions, got " << total;
}

// Full nvcc-style vector_add kernel
TEST(PtxSsir, NvccStyleVectorAdd) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry vector_add(
            .param .u64 vector_add_param_0,
            .param .u64 vector_add_param_1,
            .param .u64 vector_add_param_2
        )
        {
            .reg .f32 %f<3>;
            .reg .b32 %r<2>;
            .reg .b64 %rd<8>;

            ld.param.u64 %rd1, [vector_add_param_0];
            ld.param.u64 %rd2, [vector_add_param_1];
            ld.param.u64 %rd3, [vector_add_param_2];
            cvta.to.global.u64 %rd4, %rd1;
            cvta.to.global.u64 %rd5, %rd2;
            cvta.to.global.u64 %rd6, %rd3;
            mov.u32 %r1, %tid.x;
            mul.wide.s32 %rd7, %r1, 4;
            add.s64 %rd1, %rd4, %rd7;
            ld.global.f32 %f1, [%rd1];
            add.s64 %rd2, %rd5, %rd7;
            ld.global.f32 %f2, [%rd2];
            add.f32 %f1, %f1, %f2;
            add.s64 %rd3, %rd6, %rd7;
            st.global.f32 [%rd3], %f1;
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard2 g(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    uint32_t total = TotalInsts(f);
    EXPECT_GE(total, 20u)
        << "nvcc-style vector_add should produce many SSIR instructions, got " << total;

    // Check for the key float add
    bool has_f32_add = false;
    for (uint32_t bi = 0; bi < f->block_count; bi++) {
        SsirBlock *b = &f->blocks[bi];
        for (uint32_t i = 0; i < b->inst_count; i++) {
            if (b->insts[i].op == SSIR_OP_ADD) {
                SsirType *t = ssir_get_type(r.mod, b->insts[i].type);
                if (t && t->kind == SSIR_TYPE_F32) has_f32_add = true;
            }
        }
    }
    EXPECT_TRUE(has_f32_add) << "Missing f32 ADD for the actual vector addition";
}
