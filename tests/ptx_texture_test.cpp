/*
 * ptx_texture_test.cpp -- Tests for PTX texture and surface operations
 *
 * Covers: .texref/.samplerref/.surfref declarations, tex, tld4, suld, sust,
 * txq, suq instructions, and integration tests (SSIR verification, SPIR-V
 * validation, WGSL roundtrip).
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

class SsirGuard {
  public:
    explicit SsirGuard(SsirModule *m) : m_(m) {}
    ~SsirGuard() { if (m_) ssir_module_destroy(m_); }
    SsirModule *get() { return m_; }
    SsirModule *operator->() { return m_; }
  private:
    SsirModule *m_;
    SsirGuard(const SsirGuard &) = delete;
    SsirGuard &operator=(const SsirGuard &) = delete;
};

static const char *PTX_HEADER =
    ".version 7.8\n.target sm_80\n.address_size 64\n";

struct PtxResult {
    bool success;
    std::string error;
    SsirModule *mod;
};

static PtxResult Parse(const std::string &ptx) {
    PtxResult r = {false, "", nullptr};
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

static uint32_t CountOps(SsirBlock *b, SsirOpcode op) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < b->inst_count; i++)
        if (b->insts[i].op == op) n++;
    return n;
}

static uint32_t TotalInsts(SsirFunction *f) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < f->block_count; i++)
        n += f->blocks[i].inst_count;
    return n;
}

static uint32_t CountOpsAll(SsirFunction *f, SsirOpcode op) {
    uint32_t n = 0;
    for (uint32_t bi = 0; bi < f->block_count; bi++)
        n += CountOps(&f->blocks[bi], op);
    return n;
}

static std::string PtxToWgsl(const std::string &ptx) {
    auto res = Parse(ptx);
    if (!res.success) return "PARSE_ERROR: " + res.error;
    SsirGuard guard(res.mod);

    char *wgsl_out = nullptr;
    char *wgsl_err = nullptr;
    SsirToWgslOptions wgsl_opts = {};
    wgsl_opts.preserve_names = 1;

    SsirToWgslResult wr = ssir_to_wgsl(res.mod, &wgsl_opts, &wgsl_out, &wgsl_err);
    if (wr != SSIR_TO_WGSL_OK) {
        std::string e = wgsl_err ? wgsl_err : "unknown";
        ssir_to_wgsl_free(wgsl_out);
        ssir_to_wgsl_free(wgsl_err);
        return "WGSL_ERROR: " + e;
    }
    std::string result = wgsl_out ? wgsl_out : "";
    ssir_to_wgsl_free(wgsl_out);
    ssir_to_wgsl_free(wgsl_err);
    return result;
}

/* Helper: check if any global has a specific type kind */
static bool HasGlobalWithTypeKind(SsirModule *mod, SsirTypeKind kind) {
    for (uint32_t i = 0; i < mod->global_count; i++) {
        SsirType *t = ssir_get_type(mod, mod->globals[i].type);
        if (!t) continue;
        /* For pointer types, check the pointee */
        if (t->kind == SSIR_TYPE_PTR) {
            SsirType *pt = ssir_get_type(mod, t->ptr.pointee);
            if (pt && pt->kind == kind) return true;
        }
        if (t->kind == kind) return true;
    }
    return false;
}

/* ============================================================================
 * A. Declaration Parsing Tests
 * ============================================================================ */

TEST(PtxTexture, ParseTexrefDecl) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);
}

TEST(PtxTexture, ParseSamplerrefDecl) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .samplerref my_sampler;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);
}

TEST(PtxTexture, ParseSurfrefDecl) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .surfref my_surface;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);
}

TEST(PtxTexture, ParseMultipleTexrefs) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref tex_a;
        .global .texref tex_b;
        .global .texref tex_c;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);
}

TEST(PtxTexture, ParseTexrefInKernel) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<8>;
            .reg .s32 %r<4>;

            mov.s32 %r0, 0;
            tex.1d.v4.f32.s32 {%f0, %f1, %f2, %f3}, [my_texture, {%r0}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    /* Kernel should have generated SSIR instructions for the tex op */
    EXPECT_GT(TotalInsts(&r.mod->functions[0]), 1u)
        << "tex instruction should produce SSIR instructions";
}

TEST(PtxTexture, ParseTexrefWithSampler) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;
        .global .samplerref my_sampler;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);
}

TEST(PtxTexture, ParseAllThreeRefTypes) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;
        .global .samplerref my_sampler;
        .global .surfref my_surface;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);
}

/* ============================================================================
 * B. tex Instruction Parsing Tests
 * ============================================================================ */

TEST(PtxTexture, TexSample1D) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<8>;
            .reg .s32 %r<4>;

            mov.s32 %r0, 0;
            tex.1d.v4.f32.s32 {%f0, %f1, %f2, %f3}, [my_texture, {%r0}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GT(TotalInsts(f), 1u) << "tex.1d should produce SSIR instructions";
}

TEST(PtxTexture, TexSample2D) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<8>;

            mov.f32 %f4, 0f3F000000;
            mov.f32 %f5, 0f3F000000;
            tex.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [my_texture, {%f4, %f5}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GT(TotalInsts(f), 1u) << "tex.2d should produce SSIR instructions";
}

TEST(PtxTexture, TexSample3D) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<8>;

            mov.f32 %f4, 0f3F000000;
            mov.f32 %f5, 0f3F000000;
            mov.f32 %f6, 0f3F000000;
            tex.3d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [my_texture, {%f4, %f5, %f6}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GT(TotalInsts(f), 1u) << "tex.3d should produce SSIR instructions";
}

TEST(PtxTexture, TexSampleCube) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<8>;

            mov.f32 %f4, 0f3F800000;
            mov.f32 %f5, 0f00000000;
            mov.f32 %f6, 0f00000000;
            tex.cube.v4.f32.f32 {%f0, %f1, %f2, %f3}, [my_texture, {%f4, %f5, %f6}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);
}

TEST(PtxTexture, TexSampleArray1D) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<8>;

            mov.f32 %f4, 0f00000000;
            mov.f32 %f5, 0f3F000000;
            tex.a1d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [my_texture, {%f4, %f5}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);
}

TEST(PtxTexture, TexSampleArray2D) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<8>;

            mov.f32 %f4, 0f00000000;
            mov.f32 %f5, 0f3F000000;
            mov.f32 %f6, 0f3F000000;
            tex.a2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [my_texture, {%f4, %f5, %f6}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);
}

TEST(PtxTexture, TexSampleLevel2D) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<8>;

            mov.f32 %f4, 0f3F000000;
            mov.f32 %f5, 0f3F000000;
            mov.f32 %f6, 0f00000000;
            tex.level.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [my_texture, {%f4, %f5, %f6}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GT(TotalInsts(f), 1u)
        << "tex.level.2d should produce SSIR instructions";
}

TEST(PtxTexture, TexSampleGrad2D) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<12>;

            mov.f32 %f4, 0f3F000000;
            mov.f32 %f5, 0f3F000000;
            mov.f32 %f6, 0f3F800000;
            mov.f32 %f7, 0f00000000;
            mov.f32 %f8, 0f00000000;
            mov.f32 %f9, 0f3F800000;
            tex.grad.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [my_texture, {%f4, %f5, %f6, %f7, %f8, %f9}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GT(TotalInsts(f), 1u)
        << "tex.grad.2d should produce SSIR instructions";
}

TEST(PtxTexture, TexSampleIntCoords) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<4>;
            .reg .s32 %r<4>;

            mov.s32 %r0, 10;
            mov.s32 %r1, 20;
            tex.2d.v4.f32.s32 {%f0, %f1, %f2, %f3}, [my_texture, {%r0, %r1}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);
}

TEST(PtxTexture, TexSample2DEmitsSSIR) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<8>;

            mov.f32 %f4, 0f3F000000;
            mov.f32 %f5, 0f3F000000;
            tex.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [my_texture, {%f4, %f5}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    /* tex.2d in compute shader uses explicit LOD 0 (ImplicitLod not allowed) */
    EXPECT_GT(CountOpsAll(f, SSIR_OP_TEX_SAMPLE_LEVEL), 0u)
        << "tex.2d.v4.f32.f32 in compute should emit SSIR_OP_TEX_SAMPLE_LEVEL";
}

TEST(PtxTexture, TexSampleLevel2DEmitsSSIR) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<8>;

            mov.f32 %f4, 0f3F000000;
            mov.f32 %f5, 0f3F000000;
            mov.f32 %f6, 0f00000000;
            tex.level.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [my_texture, {%f4, %f5, %f6}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    /* tex.level should emit SSIR_OP_TEX_SAMPLE_LEVEL */
    EXPECT_GT(CountOpsAll(f, SSIR_OP_TEX_SAMPLE_LEVEL), 0u)
        << "tex.level.2d should emit SSIR_OP_TEX_SAMPLE_LEVEL";
}

TEST(PtxTexture, TexSample2DCreatesGlobals) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<8>;

            mov.f32 %f4, 0f3F000000;
            mov.f32 %f5, 0f3F000000;
            tex.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [my_texture, {%f4, %f5}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    /* The parser should create global variables for texture + sampler with
       bindings. At minimum we expect a texture global and a sampler global. */
    bool found_texture = HasGlobalWithTypeKind(r.mod, SSIR_TYPE_TEXTURE);
    bool found_sampler = HasGlobalWithTypeKind(r.mod, SSIR_TYPE_SAMPLER);

    EXPECT_TRUE(found_texture)
        << "tex.2d should create a TEXTURE global variable";
    EXPECT_TRUE(found_sampler)
        << "tex.2d should create a SAMPLER global variable";

    /* Both should have binding decorations */
    bool has_binding = false;
    for (uint32_t i = 0; i < r.mod->global_count; i++) {
        if (r.mod->globals[i].has_binding) {
            has_binding = true;
            break;
        }
    }
    EXPECT_TRUE(has_binding)
        << "Texture/sampler globals should have @binding decorations";
}

TEST(PtxTexture, TexSampleGrad2DEmitsSSIR) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<12>;

            mov.f32 %f4, 0f3F000000;
            mov.f32 %f5, 0f3F000000;
            mov.f32 %f6, 0f3F800000;
            mov.f32 %f7, 0f00000000;
            mov.f32 %f8, 0f00000000;
            mov.f32 %f9, 0f3F800000;
            tex.grad.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [my_texture, {%f4, %f5, %f6, %f7, %f8, %f9}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GT(CountOpsAll(f, SSIR_OP_TEX_SAMPLE_GRAD), 0u)
        << "tex.grad.2d should emit SSIR_OP_TEX_SAMPLE_GRAD";
}

/* ============================================================================
 * C. tld4 Instruction Tests (Texture Gather)
 * ============================================================================ */

TEST(PtxTexture, Tld4GatherRed) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<8>;

            mov.f32 %f4, 0f3F000000;
            mov.f32 %f5, 0f3F000000;
            tld4.r.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [my_texture, {%f4, %f5}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GT(CountOpsAll(f, SSIR_OP_TEX_GATHER), 0u)
        << "tld4.r should emit SSIR_OP_TEX_GATHER";
}

TEST(PtxTexture, Tld4GatherGreen) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<8>;

            mov.f32 %f4, 0f3F000000;
            mov.f32 %f5, 0f3F000000;
            tld4.g.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [my_texture, {%f4, %f5}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GT(CountOpsAll(f, SSIR_OP_TEX_GATHER), 0u)
        << "tld4.g should emit SSIR_OP_TEX_GATHER";
}

TEST(PtxTexture, Tld4GatherBlue) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<8>;

            mov.f32 %f4, 0f3F000000;
            mov.f32 %f5, 0f3F000000;
            tld4.b.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [my_texture, {%f4, %f5}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GT(CountOpsAll(f, SSIR_OP_TEX_GATHER), 0u)
        << "tld4.b should emit SSIR_OP_TEX_GATHER";
}

TEST(PtxTexture, Tld4GatherAlpha) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<8>;

            mov.f32 %f4, 0f3F000000;
            mov.f32 %f5, 0f3F000000;
            tld4.a.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [my_texture, {%f4, %f5}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GT(CountOpsAll(f, SSIR_OP_TEX_GATHER), 0u)
        << "tld4.a should emit SSIR_OP_TEX_GATHER";
}

/* ============================================================================
 * D. suld Instruction Tests (Surface Load)
 * ============================================================================ */

TEST(PtxTexture, SuldLoad1D) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .surfref my_surface;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<8>;

            mov.u32 %r4, 0;
            suld.b.1d.v4.b32 {%r0, %r1, %r2, %r3}, [my_surface, {%r4}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GT(TotalInsts(f), 1u) << "suld.b.1d should produce SSIR instructions";
}

TEST(PtxTexture, SuldLoad2D) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .surfref my_surface;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<8>;

            mov.u32 %r4, 0;
            mov.u32 %r5, 0;
            suld.b.2d.v4.b32 {%r0, %r1, %r2, %r3}, [my_surface, {%r4, %r5}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GT(TotalInsts(f), 1u) << "suld.b.2d should produce SSIR instructions";
}

TEST(PtxTexture, SuldLoad3D) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .surfref my_surface;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<8>;

            mov.u32 %r4, 0;
            mov.u32 %r5, 0;
            mov.u32 %r6, 0;
            suld.b.3d.v4.b32 {%r0, %r1, %r2, %r3}, [my_surface, {%r4, %r5, %r6}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GT(TotalInsts(f), 1u) << "suld.b.3d should produce SSIR instructions";
}

TEST(PtxTexture, SuldLoad2DEmitsSSIR) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .surfref my_surface;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<8>;

            mov.u32 %r4, 0;
            mov.u32 %r5, 0;
            suld.b.2d.v4.b32 {%r0, %r1, %r2, %r3}, [my_surface, {%r4, %r5}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GT(CountOpsAll(f, SSIR_OP_TEX_LOAD), 0u)
        << "suld.b.2d should emit SSIR_OP_TEX_LOAD";
}

/* ============================================================================
 * E. sust Instruction Tests (Surface Store)
 * ============================================================================ */

TEST(PtxTexture, SustStore1D) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .surfref my_surface;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<8>;

            mov.u32 %r4, 0;
            mov.u32 %r0, 255;
            mov.u32 %r1, 128;
            mov.u32 %r2, 64;
            mov.u32 %r3, 255;
            sust.b.1d.v4.b32 [my_surface, {%r4}], {%r0, %r1, %r2, %r3};
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GT(TotalInsts(f), 1u) << "sust.b.1d should produce SSIR instructions";
}

TEST(PtxTexture, SustStore2D) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .surfref my_surface;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<8>;

            mov.u32 %r4, 0;
            mov.u32 %r5, 0;
            mov.u32 %r0, 255;
            mov.u32 %r1, 128;
            mov.u32 %r2, 64;
            mov.u32 %r3, 255;
            sust.b.2d.v4.b32 [my_surface, {%r4, %r5}], {%r0, %r1, %r2, %r3};
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GT(TotalInsts(f), 1u) << "sust.b.2d should produce SSIR instructions";
}

TEST(PtxTexture, SustStore3D) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .surfref my_surface;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<8>;

            mov.u32 %r4, 0;
            mov.u32 %r5, 0;
            mov.u32 %r6, 0;
            mov.u32 %r0, 255;
            mov.u32 %r1, 128;
            mov.u32 %r2, 64;
            mov.u32 %r3, 255;
            sust.b.3d.v4.b32 [my_surface, {%r4, %r5, %r6}], {%r0, %r1, %r2, %r3};
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GT(TotalInsts(f), 1u) << "sust.b.3d should produce SSIR instructions";
}

TEST(PtxTexture, SustStore2DEmitsSSIR) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .surfref my_surface;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<8>;

            mov.u32 %r4, 0;
            mov.u32 %r5, 0;
            mov.u32 %r0, 255;
            mov.u32 %r1, 128;
            mov.u32 %r2, 64;
            mov.u32 %r3, 255;
            sust.b.2d.v4.b32 [my_surface, {%r4, %r5}], {%r0, %r1, %r2, %r3};
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GT(CountOpsAll(f, SSIR_OP_TEX_STORE), 0u)
        << "sust.b.2d should emit SSIR_OP_TEX_STORE";
}

/* ============================================================================
 * F. txq / suq Query Tests
 * ============================================================================ */

TEST(PtxTexture, TxqWidth) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<4>;

            txq.width.b32 %r0, [my_texture];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    /* txq.width should emit SSIR_OP_TEX_SIZE and then extract component 0 */
    EXPECT_GT(TotalInsts(f), 1u)
        << "txq.width should produce SSIR instructions";
}

TEST(PtxTexture, TxqHeight) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<4>;

            txq.height.b32 %r0, [my_texture];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GT(TotalInsts(f), 1u)
        << "txq.height should produce SSIR instructions";
}

TEST(PtxTexture, TxqDepth) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<4>;

            txq.depth.b32 %r0, [my_texture];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GT(TotalInsts(f), 1u)
        << "txq.depth should produce SSIR instructions";
}

TEST(PtxTexture, TxqNumMipmapLevels) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<4>;

            txq.num_mipmap_levels.b32 %r0, [my_texture];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    /* num_mipmap_levels should emit TEX_QUERY_LEVELS */
    EXPECT_GT(CountOpsAll(f, SSIR_OP_TEX_QUERY_LEVELS), 0u)
        << "txq.num_mipmap_levels should emit SSIR_OP_TEX_QUERY_LEVELS";
}

TEST(PtxTexture, SuqWidth) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .surfref my_surface;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<4>;

            suq.width.b32 %r0, [my_surface];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GT(TotalInsts(f), 1u)
        << "suq.width should produce SSIR instructions";
}

TEST(PtxTexture, SuqHeight) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .surfref my_surface;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<4>;

            suq.height.b32 %r0, [my_surface];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GT(TotalInsts(f), 1u)
        << "suq.height should produce SSIR instructions";
}

TEST(PtxTexture, TxqWidthEmitsTexSize) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<4>;

            txq.width.b32 %r0, [my_texture];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    /* txq.width/height/depth should typically emit TEX_SIZE + EXTRACT */
    uint32_t tex_size = CountOpsAll(f, SSIR_OP_TEX_SIZE);
    uint32_t tex_query_levels = CountOpsAll(f, SSIR_OP_TEX_QUERY_LEVELS);
    EXPECT_TRUE(tex_size > 0 || tex_query_levels > 0)
        << "txq.width should emit TEX_SIZE or similar query op";
}

/* ============================================================================
 * G. Combined / Integration Tests
 * ============================================================================ */

TEST(PtxTexture, TexSampleAndStore) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern(
            .param .u64 output_ptr
        )
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<8>;
            .reg .u64 %rd<4>;

            ld.param.u64 %rd0, [output_ptr];

            mov.f32 %f4, 0f3F000000;
            mov.f32 %f5, 0f3F000000;
            tex.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [my_texture, {%f4, %f5}];

            st.global.f32 [%rd0], %f0;

            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    uint32_t total = TotalInsts(f);
    EXPECT_GE(total, 5u)
        << "Texture sample + buffer store should produce at least 5 SSIR instructions, got "
        << total;
}

TEST(PtxTexture, SurfaceReadModifyWrite) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .surfref my_surface;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<12>;

            mov.u32 %r8, 0;
            mov.u32 %r9, 0;

            suld.b.2d.v4.b32 {%r0, %r1, %r2, %r3}, [my_surface, {%r8, %r9}];

            add.u32 %r0, %r0, 10;
            add.u32 %r1, %r1, 10;
            add.u32 %r2, %r2, 10;
            add.u32 %r3, %r3, 10;

            sust.b.2d.v4.b32 [my_surface, {%r8, %r9}], {%r0, %r1, %r2, %r3};

            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    /* Should have both TEX_LOAD and TEX_STORE plus ADDs */
    EXPECT_GT(CountOpsAll(f, SSIR_OP_TEX_LOAD), 0u)
        << "Read-modify-write should emit TEX_LOAD";
    EXPECT_GT(CountOpsAll(f, SSIR_OP_TEX_STORE), 0u)
        << "Read-modify-write should emit TEX_STORE";
    EXPECT_GT(CountOpsAll(f, SSIR_OP_ADD), 0u)
        << "Read-modify-write should emit ADD instructions";
}

TEST(PtxTexture, MultipleTexturesInKernel) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref tex_a;
        .global .texref tex_b;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<12>;

            mov.f32 %f4, 0f3F000000;
            mov.f32 %f5, 0f3F000000;

            tex.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [tex_a, {%f4, %f5}];
            tex.2d.v4.f32.f32 {%f6, %f7, %f8, %f9}, [tex_b, {%f4, %f5}];

            add.f32 %f10, %f0, %f6;

            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    /* Two texture samples + arithmetic (compute uses explicit LOD 0) */
    EXPECT_GE(CountOpsAll(f, SSIR_OP_TEX_SAMPLE_LEVEL), 2u)
        << "Two tex.2d instructions should produce at least 2 TEX_SAMPLE_LEVEL ops";
}

TEST(PtxTexture, TextureWithCompute) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern(
            .param .u64 output_ptr
        )
        .reqntid 64, 1, 1
        {
            .reg .f32 %f<8>;
            .reg .u32 %r<4>;
            .reg .u64 %rd<4>;

            ld.param.u64 %rd0, [output_ptr];
            mov.u32 %r0, %tid.x;

            cvt.rn.f32.u32 %f4, %r0;
            mov.f32 %f5, 0f3F000000;
            tex.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [my_texture, {%f4, %f5}];

            add.f32 %f0, %f0, %f1;
            mul.f32 %f0, %f0, %f2;

            cvt.u64.u32 %rd1, %r0;
            mul.lo.u64 %rd1, %rd1, 4;
            add.u64 %rd2, %rd0, %rd1;
            st.global.f32 [%rd2], %f0;

            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    uint32_t total = TotalInsts(f);
    EXPECT_GE(total, 10u)
        << "Texture + compute kernel should produce many SSIR instructions, got "
        << total;

    /* Should have both texture ops and arithmetic */
    EXPECT_GT(CountOpsAll(f, SSIR_OP_ADD), 0u) << "Missing ADD";
    EXPECT_GT(CountOpsAll(f, SSIR_OP_MUL), 0u) << "Missing MUL";
}

TEST(PtxTexture, TexSampleToWgsl) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<8>;

            mov.f32 %f4, 0f3F000000;
            mov.f32 %f5, 0f3F000000;
            tex.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [my_texture, {%f4, %f5}];
            ret;
        }
    )";
    std::string wgsl = PtxToWgsl(ptx);
    EXPECT_EQ(wgsl.find("PARSE_ERROR"), std::string::npos)
        << "PTX parse should succeed: " << wgsl;
    /* WGSL emission may or may not be implemented yet, but parse must work */
}

TEST(PtxTexture, Tex2DValidSpirv) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<8>;

            mov.f32 %f4, 0f3F000000;
            mov.f32 %f5, 0f3F000000;
            tex.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [my_texture, {%f4, %f5}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    auto spv = EmitSpirv(r.mod);
    ASSERT_TRUE(spv.success) << spv.error;

    std::string val_err;
    EXPECT_TRUE(wgsl_test::ValidateSpirv(spv.words.data(), spv.words.size(), &val_err))
        << "spirv-val: " << val_err;
}

TEST(PtxTexture, TexSampleLevelToWgsl) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<8>;

            mov.f32 %f4, 0f3F000000;
            mov.f32 %f5, 0f3F000000;
            mov.f32 %f6, 0f00000000;
            tex.level.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [my_texture, {%f4, %f5, %f6}];
            ret;
        }
    )";
    std::string wgsl = PtxToWgsl(ptx);
    EXPECT_EQ(wgsl.find("PARSE_ERROR"), std::string::npos)
        << "PTX parse should succeed: " << wgsl;
}

/* ============================================================================
 * H. Error / Edge Case Tests
 * ============================================================================ */

TEST(PtxTexture, TexUndefinedTexture) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<8>;

            mov.f32 %f4, 0f3F000000;
            mov.f32 %f5, 0f3F000000;
            tex.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [undefined_texture, {%f4, %f5}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    /* Referencing an undefined texture should either fail parse or
       handle gracefully. We check that it does not crash. */
    if (!r.success) {
        /* Expected: parse error for undefined texture */
        EXPECT_FALSE(r.error.empty());
    } else {
        SsirGuard guard(r.mod);
        /* If it parses, that is also acceptable (auto-declaration) */
    }
}

TEST(PtxTexture, SuldUndefinedSurface) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<8>;

            mov.u32 %r4, 0;
            mov.u32 %r5, 0;
            suld.b.2d.v4.b32 {%r0, %r1, %r2, %r3}, [undefined_surface, {%r4, %r5}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    /* Should either fail or handle gracefully, no crash */
    if (!r.success) {
        EXPECT_FALSE(r.error.empty());
    } else {
        SsirGuard guard(r.mod);
    }
}

TEST(PtxTexture, TexEmptyKernelWithDecl) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;
        .global .samplerref my_sampler;
        .global .surfref my_surface;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    /* Declared but unused texture/sampler/surface should not crash.
       They may or may not appear as globals depending on implementation. */
    EXPECT_GE(r.mod->entry_point_count, 1u);
}

TEST(PtxTexture, TexMultipleKernelsSharedTexture) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref shared_texture;

        .visible .entry kern_a()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<8>;

            mov.f32 %f4, 0f3F000000;
            mov.f32 %f5, 0f3F000000;
            tex.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [shared_texture, {%f4, %f5}];
            ret;
        }

        .visible .entry kern_b()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<8>;

            mov.f32 %f4, 0f3F800000;
            mov.f32 %f5, 0f3F800000;
            tex.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [shared_texture, {%f4, %f5}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    /* Two entry points, both referencing the same texture */
    EXPECT_GE(r.mod->entry_point_count, 2u)
        << "Should have two entry points";
}

/* ============================================================================
 * I. Realistic Kernel Tests
 * ============================================================================ */

TEST(PtxTexture, NvccStyleTextureSample) {
    /* Mimics nvcc-generated code for a simple texture lookup kernel */
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref tex_input;

        .visible .entry texture_lookup(
            .param .u64 output_ptr,
            .param .u32 width
        )
        .reqntid 256, 1, 1
        {
            .reg .f32 %f<8>;
            .reg .b32 %r<8>;
            .reg .b64 %rd<8>;
            .reg .pred %p<2>;

            ld.param.u64 %rd0, [output_ptr];
            ld.param.u32 %r0, [width];

            mov.u32 %r1, %tid.x;
            mov.u32 %r2, %ctaid.x;
            mov.u32 %r3, %ntid.x;
            mad.lo.u32 %r1, %r2, %r3, %r1;

            setp.ge.u32 %p0, %r1, %r0;
            @%p0 bra DONE;

            cvt.rn.f32.u32 %f4, %r1;
            mov.f32 %f5, 0f3F000000;
            tex.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [tex_input, {%f4, %f5}];

            cvt.u64.u32 %rd1, %r1;
            mul.lo.u64 %rd1, %rd1, 4;
            add.u64 %rd2, %rd0, %rd1;
            st.global.f32 [%rd2], %f0;

        DONE:
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    EXPECT_GE(r.mod->entry_point_count, 1u);
    SsirEntryPoint *ep = &r.mod->entry_points[0];
    EXPECT_EQ(ep->workgroup_size[0], 256u);

    SsirFunction *f = &r.mod->functions[0];
    uint32_t total = TotalInsts(f);
    EXPECT_GE(total, 15u)
        << "nvcc-style texture kernel should produce many SSIR instructions, got "
        << total;
}

TEST(PtxTexture, ImageProcessingKernel) {
    /* Read texture, write surface -- image processing pattern */
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref input_tex;
        .global .surfref output_surf;

        .visible .entry image_process()
        .reqntid 16, 16, 1
        {
            .reg .f32 %f<8>;
            .reg .u32 %r<12>;

            mov.u32 %r0, %tid.x;
            mov.u32 %r1, %tid.y;
            mov.u32 %r2, %ctaid.x;
            mov.u32 %r3, %ctaid.y;
            mov.u32 %r4, %ntid.x;
            mov.u32 %r5, %ntid.y;

            mad.lo.u32 %r6, %r2, %r4, %r0;
            mad.lo.u32 %r7, %r3, %r5, %r1;

            cvt.rn.f32.u32 %f4, %r6;
            cvt.rn.f32.u32 %f5, %r7;
            tex.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [input_tex, {%f4, %f5}];

            cvt.rzi.u32.f32 %r8, %f0;
            cvt.rzi.u32.f32 %r9, %f1;
            cvt.rzi.u32.f32 %r10, %f2;
            cvt.rzi.u32.f32 %r11, %f3;

            sust.b.2d.v4.b32 [output_surf, {%r6, %r7}], {%r8, %r9, %r10, %r11};

            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirEntryPoint *ep = &r.mod->entry_points[0];
    EXPECT_EQ(ep->workgroup_size[0], 16u);
    EXPECT_EQ(ep->workgroup_size[1], 16u);

    SsirFunction *f = &r.mod->functions[0];
    uint32_t total = TotalInsts(f);
    EXPECT_GE(total, 15u)
        << "Image processing kernel should produce many SSIR instructions, got "
        << total;
}

TEST(PtxTexture, TextureFetchAndCompute) {
    /* Fetch texels, do computation, write to buffer */
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry tex_compute(
            .param .u64 output_ptr
        )
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<12>;
            .reg .u32 %r<4>;
            .reg .u64 %rd<4>;

            ld.param.u64 %rd0, [output_ptr];

            mov.f32 %f4, 0f3F000000;
            mov.f32 %f5, 0f3F000000;
            tex.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [my_texture, {%f4, %f5}];

            mul.f32 %f6, %f0, %f0;
            mul.f32 %f7, %f1, %f1;
            mul.f32 %f8, %f2, %f2;
            add.f32 %f9, %f6, %f7;
            add.f32 %f9, %f9, %f8;

            sqrt.f32 %f10, %f9;

            st.global.f32 [%rd0], %f10;

            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    uint32_t total = TotalInsts(f);
    EXPECT_GE(total, 10u)
        << "Texture fetch + compute kernel should produce at least 10 SSIR instructions, got "
        << total;

    /* Should have texture sample, arithmetic, and math builtin */
    EXPECT_GT(CountOpsAll(f, SSIR_OP_MUL), 0u) << "Missing MUL";
    EXPECT_GT(CountOpsAll(f, SSIR_OP_ADD), 0u) << "Missing ADD";
    EXPECT_GT(CountOpsAll(f, SSIR_OP_BUILTIN), 0u) << "Missing BUILTIN (sqrt)";
}

TEST(PtxTexture, BilinearInterpolation) {
    /* Manual bilinear interpolation using tex.level */
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry bilinear(
            .param .u64 output_ptr
        )
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<20>;
            .reg .u64 %rd<4>;

            ld.param.u64 %rd0, [output_ptr];

            mov.f32 %f8, 0f3F000000;
            mov.f32 %f9, 0f3F000000;
            mov.f32 %f10, 0f00000000;

            tex.level.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [my_texture, {%f8, %f9, %f10}];

            mov.f32 %f11, 0f3F800000;
            add.f32 %f12, %f8, %f11;

            tex.level.2d.v4.f32.f32 {%f4, %f5, %f6, %f7}, [my_texture, {%f12, %f9, %f10}];

            mov.f32 %f13, 0f3F000000;
            mul.f32 %f14, %f0, %f13;
            mul.f32 %f15, %f4, %f13;
            add.f32 %f16, %f14, %f15;

            st.global.f32 [%rd0], %f16;

            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    /* Two tex.level instructions should produce two TEX_SAMPLE_LEVEL */
    EXPECT_GE(CountOpsAll(f, SSIR_OP_TEX_SAMPLE_LEVEL), 2u)
        << "Two tex.level calls should emit at least 2 TEX_SAMPLE_LEVEL, got "
        << CountOpsAll(f, SSIR_OP_TEX_SAMPLE_LEVEL);
}

/* ============================================================================
 * J. Additional Tests for Coverage
 * ============================================================================ */

TEST(PtxTexture, TexSample1DIntCoords) {
    /* 1D texture with integer coordinates (texture fetch pattern) */
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<4>;
            .reg .s32 %r<2>;

            mov.s32 %r0, 42;
            tex.1d.v4.f32.s32 {%f0, %f1, %f2, %f3}, [my_texture, {%r0}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);
}

TEST(PtxTexture, SustUndefinedSurface) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<8>;

            mov.u32 %r4, 0;
            mov.u32 %r0, 255;
            mov.u32 %r1, 128;
            mov.u32 %r2, 64;
            mov.u32 %r3, 255;
            sust.b.1d.v4.b32 [undefined_surface, {%r4}], {%r0, %r1, %r2, %r3};
            ret;
        }
    )";
    auto r = Parse(ptx);
    /* Should either fail or handle gracefully, no crash */
    if (!r.success) {
        EXPECT_FALSE(r.error.empty());
    } else {
        SsirGuard guard(r.mod);
    }
}

TEST(PtxTexture, TexSample2DEntryPointInterface) {
    /* Verify that texture globals are in the entry point interface */
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<8>;

            mov.f32 %f4, 0f3F000000;
            mov.f32 %f5, 0f3F000000;
            tex.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [my_texture, {%f4, %f5}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    ASSERT_GE(r.mod->entry_point_count, 1u);
    SsirEntryPoint *ep = &r.mod->entry_points[0];
    /* Entry point interface should include texture/sampler globals */
    EXPECT_GE(ep->interface_count, 1u)
        << "Entry point interface should include texture/sampler globals";
}

TEST(PtxTexture, SuldLoad2DCreatesStorageTexture) {
    /* Verify that suld creates a storage texture / texture_storage global */
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .surfref my_surface;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<8>;

            mov.u32 %r4, 0;
            mov.u32 %r5, 0;
            suld.b.2d.v4.b32 {%r0, %r1, %r2, %r3}, [my_surface, {%r4, %r5}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    /* Surface references should become storage texture globals */
    bool found_storage_tex = HasGlobalWithTypeKind(r.mod, SSIR_TYPE_TEXTURE_STORAGE);
    /* Alternatively, it might be modeled as a regular texture for read-only */
    bool found_tex = HasGlobalWithTypeKind(r.mod, SSIR_TYPE_TEXTURE);
    EXPECT_TRUE(found_storage_tex || found_tex)
        << "Surface load should create a texture or storage texture global";
}

TEST(PtxTexture, SustStore2DCreatesStorageTexture) {
    /* Verify that sust creates a storage texture global with write access */
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .surfref my_surface;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<8>;

            mov.u32 %r4, 0;
            mov.u32 %r5, 0;
            mov.u32 %r0, 255;
            mov.u32 %r1, 128;
            mov.u32 %r2, 64;
            mov.u32 %r3, 255;
            sust.b.2d.v4.b32 [my_surface, {%r4, %r5}], {%r0, %r1, %r2, %r3};
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    bool found_storage_tex = HasGlobalWithTypeKind(r.mod, SSIR_TYPE_TEXTURE_STORAGE);
    EXPECT_TRUE(found_storage_tex)
        << "Surface store should create a TEXTURE_STORAGE global";
}

TEST(PtxTexture, SuldLoad2DValidSpirv) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .surfref my_surface;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<8>;

            mov.u32 %r4, 0;
            mov.u32 %r5, 0;
            suld.b.2d.v4.b32 {%r0, %r1, %r2, %r3}, [my_surface, {%r4, %r5}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    auto spv = EmitSpirv(r.mod);
    ASSERT_TRUE(spv.success) << spv.error;

    std::string val_err;
    EXPECT_TRUE(wgsl_test::ValidateSpirv(spv.words.data(), spv.words.size(), &val_err))
        << "spirv-val: " << val_err;
}

TEST(PtxTexture, SustStore2DValidSpirv) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .surfref my_surface;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<8>;

            mov.u32 %r4, 0;
            mov.u32 %r5, 0;
            mov.u32 %r0, 255;
            mov.u32 %r1, 128;
            mov.u32 %r2, 64;
            mov.u32 %r3, 255;
            sust.b.2d.v4.b32 [my_surface, {%r4, %r5}], {%r0, %r1, %r2, %r3};
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    auto spv = EmitSpirv(r.mod);
    ASSERT_TRUE(spv.success) << spv.error;

    std::string val_err;
    EXPECT_TRUE(wgsl_test::ValidateSpirv(spv.words.data(), spv.words.size(), &val_err))
        << "spirv-val: " << val_err;
}

TEST(PtxTexture, TexSampleLevel1D) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<8>;

            mov.f32 %f4, 0f3F000000;
            mov.f32 %f5, 0f00000000;
            tex.level.1d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [my_texture, {%f4, %f5}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);
}

TEST(PtxTexture, MultipleQueryOps) {
    /* Query width and height from same texture */
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern(
            .param .u64 output_ptr
        )
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<4>;
            .reg .u64 %rd<4>;

            ld.param.u64 %rd0, [output_ptr];

            txq.width.b32 %r0, [my_texture];
            txq.height.b32 %r1, [my_texture];

            mul.lo.u32 %r2, %r0, %r1;

            cvt.u64.u32 %rd1, %r2;
            mul.lo.u64 %rd1, %rd1, 4;
            add.u64 %rd2, %rd0, %rd1;

            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GT(TotalInsts(f), 5u)
        << "Multiple txq ops + arithmetic should produce SSIR instructions";
}

TEST(PtxTexture, Tld4GatherValidSpirv) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<8>;

            mov.f32 %f4, 0f3F000000;
            mov.f32 %f5, 0f3F000000;
            tld4.r.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [my_texture, {%f4, %f5}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    auto spv = EmitSpirv(r.mod);
    ASSERT_TRUE(spv.success) << spv.error;

    std::string val_err;
    EXPECT_TRUE(wgsl_test::ValidateSpirv(spv.words.data(), spv.words.size(), &val_err))
        << "spirv-val: " << val_err;
}

TEST(PtxTexture, SurfaceReadModifyWriteValidSpirv) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .surfref my_surface;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<12>;

            mov.u32 %r8, 0;
            mov.u32 %r9, 0;

            suld.b.2d.v4.b32 {%r0, %r1, %r2, %r3}, [my_surface, {%r8, %r9}];

            add.u32 %r0, %r0, 10;

            sust.b.2d.v4.b32 [my_surface, {%r8, %r9}], {%r0, %r1, %r2, %r3};

            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    auto spv = EmitSpirv(r.mod);
    ASSERT_TRUE(spv.success) << spv.error;

    std::string val_err;
    EXPECT_TRUE(wgsl_test::ValidateSpirv(spv.words.data(), spv.words.size(), &val_err))
        << "spirv-val: " << val_err;
}

TEST(PtxTexture, TexSample2DNoReqntid) {
    /* Test that tex works without .reqntid (default workgroup size) */
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern()
        {
            .reg .f32 %f<8>;

            mov.f32 %f4, 0f3F000000;
            mov.f32 %f5, 0f3F000000;
            tex.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [my_texture, {%f4, %f5}];
            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);
}

TEST(PtxTexture, TexSampleWithBufferParam) {
    /* Texture sample combined with traditional buffer parameter */
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref my_texture;

        .visible .entry kern(
            .param .u64 buf_ptr,
            .param .u32 idx
        )
        .reqntid 64, 1, 1
        {
            .reg .f32 %f<8>;
            .reg .u32 %r<4>;
            .reg .u64 %rd<4>;

            ld.param.u64 %rd0, [buf_ptr];
            ld.param.u32 %r0, [idx];

            cvt.rn.f32.u32 %f4, %r0;
            mov.f32 %f5, 0f3F000000;
            tex.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [my_texture, {%f4, %f5}];

            cvt.u64.u32 %rd1, %r0;
            mul.lo.u64 %rd1, %rd1, 4;
            add.u64 %rd2, %rd0, %rd1;
            st.global.f32 [%rd2], %f0;

            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    /* Should have globals for both buffer params and texture/sampler */
    EXPECT_GE(r.mod->global_count, 3u)
        << "Should have buffer globals + texture globals, got "
        << r.mod->global_count;
}

TEST(PtxTexture, SurfaceLoadToWgsl) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .surfref my_surface;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<8>;

            mov.u32 %r4, 0;
            mov.u32 %r5, 0;
            suld.b.2d.v4.b32 {%r0, %r1, %r2, %r3}, [my_surface, {%r4, %r5}];
            ret;
        }
    )";
    std::string wgsl = PtxToWgsl(ptx);
    EXPECT_EQ(wgsl.find("PARSE_ERROR"), std::string::npos)
        << "PTX parse should succeed: " << wgsl;
}

TEST(PtxTexture, SurfaceStoreToWgsl) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .surfref my_surface;

        .visible .entry kern()
        .reqntid 1, 1, 1
        {
            .reg .u32 %r<8>;

            mov.u32 %r4, 0;
            mov.u32 %r5, 0;
            mov.u32 %r0, 255;
            mov.u32 %r1, 128;
            mov.u32 %r2, 64;
            mov.u32 %r3, 255;
            sust.b.2d.v4.b32 [my_surface, {%r4, %r5}], {%r0, %r1, %r2, %r3};
            ret;
        }
    )";
    std::string wgsl = PtxToWgsl(ptx);
    EXPECT_EQ(wgsl.find("PARSE_ERROR"), std::string::npos)
        << "PTX parse should succeed: " << wgsl;
}

/* ============================================================================
 * L. Complex End-to-End Vulkan Texture Tests
 *
 * Each test compiles PTX through the full pipeline (PTX → SSIR → SPIR-V)
 * and validates the SPIR-V against spirv-val --target-env vulkan1.3.
 * These test complex, realistic texture scenarios with multiple formats,
 * mip levels, dimensions, and filter modes.
 * ============================================================================ */

/* E2E-1: Multiple texture dimensions in one kernel — 2D (two different UVs)
 * and 3D sampling with explicit LOD, writing results to a storage buffer. */
TEST(PtxTexture, E2E_MultiDimensionTextureSampling) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref tex_2d;
        .global .texref tex_2d_b;
        .global .texref tex_3d;

        .visible .entry multi_dim_sample(
            .param .u64 output_ptr
        )
        .reqntid 64, 1, 1
        {
            .reg .f32 %f<24>;
            .reg .u32 %r<4>;
            .reg .u64 %rd<4>;

            ld.param.u64 %rd0, [output_ptr];
            mov.u32 %r0, %tid.x;
            cvt.rn.f32.u32 %f20, %r0;

            // Sample 2D texture at UV (tid, 0.25)
            mov.f32 %f21, 0f3E800000;
            tex.level.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [tex_2d, {%f20, %f21, 0f00000000}];

            // Sample second 2D texture at UV (tid, 0.5)
            mov.f32 %f22, 0f3F000000;
            tex.level.2d.v4.f32.f32 {%f4, %f5, %f6, %f7}, [tex_2d_b, {%f20, %f22, 0f00000000}];

            // Sample 3D texture
            mov.f32 %f23, 0f3F800000;
            tex.level.3d.v4.f32.f32 {%f8, %f9, %f10, %f11}, [tex_3d, {%f20, %f22, %f23, 0f00000000}];

            // Combine: out = tex2d_a.r + tex2d_b.g + tex3d.b
            add.f32 %f12, %f0, %f5;
            add.f32 %f12, %f12, %f10;

            cvt.u64.u32 %rd1, %r0;
            mul.lo.u64 %rd1, %rd1, 4;
            add.u64 %rd2, %rd0, %rd1;
            st.global.f32 [%rd2], %f12;

            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    auto spv = EmitSpirv(r.mod);
    ASSERT_TRUE(spv.success) << spv.error;

    std::string val_err;
    EXPECT_TRUE(wgsl_test::ValidateSpirv(spv.words.data(), spv.words.size(), &val_err))
        << "spirv-val: " << val_err;

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GE(CountOpsAll(f, SSIR_OP_TEX_SAMPLE_LEVEL), 3u);
}

/* E2E-2: tex.level at multiple mip levels — samples LOD 0, 1, 2 from the
 * same texture and blends them. Validates explicit LOD parameter forwarding. */
TEST(PtxTexture, E2E_MultiMipLevelSampling) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref mipped_tex;

        .visible .entry mip_blend(
            .param .u64 output_ptr
        )
        .reqntid 32, 1, 1
        {
            .reg .f32 %f<20>;
            .reg .u32 %r<4>;
            .reg .u64 %rd<4>;

            ld.param.u64 %rd0, [output_ptr];
            mov.u32 %r0, %tid.x;
            cvt.rn.f32.u32 %f16, %r0;
            mov.f32 %f17, 0f3F000000;

            // LOD 0 (full resolution)
            tex.level.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [mipped_tex, {%f16, %f17, 0f00000000}];

            // LOD 1 (half resolution)
            tex.level.2d.v4.f32.f32 {%f4, %f5, %f6, %f7}, [mipped_tex, {%f16, %f17, 0f3F800000}];

            // LOD 2 (quarter resolution)
            tex.level.2d.v4.f32.f32 {%f8, %f9, %f10, %f11}, [mipped_tex, {%f16, %f17, 0f40000000}];

            // Weighted average: 0.5*lod0 + 0.3*lod1 + 0.2*lod2
            mul.f32 %f12, %f0, 0f3F000000;
            mul.f32 %f13, %f4, 0f3E99999A;
            mul.f32 %f14, %f8, 0f3E4CCCCD;
            add.f32 %f15, %f12, %f13;
            add.f32 %f15, %f15, %f14;

            cvt.u64.u32 %rd1, %r0;
            mul.lo.u64 %rd1, %rd1, 4;
            add.u64 %rd2, %rd0, %rd1;
            st.global.f32 [%rd2], %f15;

            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    auto spv = EmitSpirv(r.mod);
    ASSERT_TRUE(spv.success) << spv.error;

    std::string val_err;
    EXPECT_TRUE(wgsl_test::ValidateSpirv(spv.words.data(), spv.words.size(), &val_err))
        << "spirv-val: " << val_err;

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GE(CountOpsAll(f, SSIR_OP_TEX_SAMPLE_LEVEL), 3u);
}

/* E2E-3: tex.grad with explicit gradients for anisotropic filtering.
 * Uses different dx/dy gradients to simulate stretched sampling. */
TEST(PtxTexture, E2E_GradientSamplingAnisotropic) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref aniso_tex;

        .visible .entry aniso_sample(
            .param .u64 output_ptr
        )
        .reqntid 64, 1, 1
        {
            .reg .f32 %f<20>;
            .reg .u32 %r<4>;
            .reg .u64 %rd<4>;

            ld.param.u64 %rd0, [output_ptr];
            mov.u32 %r0, %tid.x;
            cvt.rn.f32.u32 %f10, %r0;
            mov.f32 %f11, 0f3F000000;

            // dx = (0.1, 0.0), dy = (0.0, 0.5) — stretched vertically
            tex.grad.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [aniso_tex, {%f10, %f11, 0f3DCCCCCD, 0f00000000, 0f00000000, 0f3F000000}];

            // Store all 4 channels to output buffer
            cvt.u64.u32 %rd1, %r0;
            mul.lo.u64 %rd1, %rd1, 16;
            add.u64 %rd2, %rd0, %rd1;
            st.global.f32 [%rd2], %f0;
            add.u64 %rd3, %rd2, 4;
            st.global.f32 [%rd3], %f1;
            add.u64 %rd3, %rd3, 4;
            st.global.f32 [%rd3], %f2;
            add.u64 %rd3, %rd3, 4;
            st.global.f32 [%rd3], %f3;

            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    auto spv = EmitSpirv(r.mod);
    ASSERT_TRUE(spv.success) << spv.error;

    std::string val_err;
    EXPECT_TRUE(wgsl_test::ValidateSpirv(spv.words.data(), spv.words.size(), &val_err))
        << "spirv-val: " << val_err;

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GE(CountOpsAll(f, SSIR_OP_TEX_SAMPLE_GRAD), 1u);
}

/* E2E-4: Surface load/store roundtrip — read a surface, apply per-pixel
 * gamma correction, write back. Exercises storage texture read + write. */
TEST(PtxTexture, E2E_SurfaceGammaCorrection) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .surfref input_surface;
        .global .surfref output_surface;

        .visible .entry gamma_correct(
            .param .u32 width
        )
        .reqntid 64, 1, 1
        {
            .reg .u32 %r<16>;

            mov.u32 %r0, %tid.x;
            mov.u32 %r1, %ctaid.x;
            mov.u32 %r2, %ntid.x;
            mad.lo.u32 %r0, %r1, %r2, %r0;

            ld.param.u32 %r3, [width];

            // Compute 2D coords: x = tid % width, y = tid / width
            div.u32 %r4, %r0, %r3;
            rem.u32 %r5, %r0, %r3;

            // Load from input surface
            suld.b.2d.v4.b32 {%r6, %r7, %r8, %r9}, [input_surface, {%r5, %r4}];

            // Simple "gamma": multiply each channel by 2
            mul.lo.u32 %r6, %r6, 2;
            mul.lo.u32 %r7, %r7, 2;
            mul.lo.u32 %r8, %r8, 2;

            // Store to output surface
            sust.b.2d.v4.b32 [output_surface, {%r5, %r4}], {%r6, %r7, %r8, %r9};

            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    auto spv = EmitSpirv(r.mod);
    ASSERT_TRUE(spv.success) << spv.error;

    std::string val_err;
    EXPECT_TRUE(wgsl_test::ValidateSpirv(spv.words.data(), spv.words.size(), &val_err))
        << "spirv-val: " << val_err;

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GE(CountOpsAll(f, SSIR_OP_TEX_LOAD), 1u);
    EXPECT_GE(CountOpsAll(f, SSIR_OP_TEX_STORE), 1u);
}

/* E2E-5: Cube map sampling — sample all 6 faces of a cube texture using
 * direction vectors pointing at each face. Uses registers for spatial
 * coordinates and provides LOD as 5th element (parser uses ncoords=4
 * for cube, so LOD must be at index 4). */
TEST(PtxTexture, E2E_CubeMapAllFaces) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref env_cubemap;

        .visible .entry cubemap_sample(
            .param .u64 output_ptr
        )
        .reqntid 1, 1, 1
        {
            .reg .f32 %f<32>;
            .reg .f32 %cx, %cy, %cz, %cw;
            .reg .u64 %rd<4>;

            ld.param.u64 %rd0, [output_ptr];
            mov.f32 %cw, 0f00000000;

            // +X face: direction (1, 0, 0)
            mov.f32 %cx, 0f3F800000;
            mov.f32 %cy, 0f00000000;
            mov.f32 %cz, 0f00000000;
            tex.level.cube.v4.f32.f32 {%f0, %f1, %f2, %f3}, [env_cubemap, {%cx, %cy, %cz, %cw, 0f00000000}];

            // -X face: direction (-1, 0, 0)
            mov.f32 %cx, 0fBF800000;
            tex.level.cube.v4.f32.f32 {%f4, %f5, %f6, %f7}, [env_cubemap, {%cx, %cy, %cz, %cw, 0f00000000}];

            // +Y face: direction (0, 1, 0)
            mov.f32 %cx, 0f00000000;
            mov.f32 %cy, 0f3F800000;
            tex.level.cube.v4.f32.f32 {%f8, %f9, %f10, %f11}, [env_cubemap, {%cx, %cy, %cz, %cw, 0f00000000}];

            // -Y face: direction (0, -1, 0)
            mov.f32 %cy, 0fBF800000;
            tex.level.cube.v4.f32.f32 {%f12, %f13, %f14, %f15}, [env_cubemap, {%cx, %cy, %cz, %cw, 0f00000000}];

            // +Z face: direction (0, 0, 1)
            mov.f32 %cy, 0f00000000;
            mov.f32 %cz, 0f3F800000;
            tex.level.cube.v4.f32.f32 {%f16, %f17, %f18, %f19}, [env_cubemap, {%cx, %cy, %cz, %cw, 0f00000000}];

            // -Z face: direction (0, 0, -1)
            mov.f32 %cz, 0fBF800000;
            tex.level.cube.v4.f32.f32 {%f20, %f21, %f22, %f23}, [env_cubemap, {%cx, %cy, %cz, %cw, 0f00000000}];

            // Average all 6 red channels
            add.f32 %f24, %f0, %f4;
            add.f32 %f24, %f24, %f8;
            add.f32 %f24, %f24, %f12;
            add.f32 %f24, %f24, %f16;
            add.f32 %f24, %f24, %f20;
            mul.f32 %f24, %f24, 0f3E2AAAAB;

            st.global.f32 [%rd0], %f24;

            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    auto spv = EmitSpirv(r.mod);
    ASSERT_TRUE(spv.success) << spv.error;

    std::string val_err;
    EXPECT_TRUE(wgsl_test::ValidateSpirv(spv.words.data(), spv.words.size(), &val_err))
        << "spirv-val: " << val_err;

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GE(CountOpsAll(f, SSIR_OP_TEX_SAMPLE_LEVEL), 6u);
}

/* E2E-6: Texture gather (tld4) for shadow map PCF filtering.
 * Gathers 4 texels and computes a soft shadow factor. */
TEST(PtxTexture, E2E_GatherPCFShadowFilter) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref shadow_map;

        .visible .entry pcf_shadow(
            .param .u64 output_ptr,
            .param .u64 coord_ptr
        )
        .reqntid 64, 1, 1
        {
            .reg .f32 %f<20>;
            .reg .u32 %r<4>;
            .reg .u64 %rd<8>;

            ld.param.u64 %rd0, [output_ptr];
            ld.param.u64 %rd1, [coord_ptr];

            mov.u32 %r0, %tid.x;
            cvt.u64.u32 %rd2, %r0;

            // Load shadow UV from coord buffer
            mul.lo.u64 %rd3, %rd2, 8;
            add.u64 %rd4, %rd1, %rd3;
            ld.global.f32 %f10, [%rd4];
            add.u64 %rd5, %rd4, 4;
            ld.global.f32 %f11, [%rd5];

            // Gather red channel (depth) from 4 neighboring texels
            tld4.r.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [shadow_map, {%f10, %f11}];

            // Average of all 4 gathered values
            add.f32 %f4, %f0, %f1;
            add.f32 %f4, %f4, %f2;
            add.f32 %f4, %f4, %f3;
            mul.f32 %f4, %f4, 0f3E800000;

            mul.lo.u64 %rd6, %rd2, 4;
            add.u64 %rd7, %rd0, %rd6;
            st.global.f32 [%rd7], %f4;

            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    auto spv = EmitSpirv(r.mod);
    ASSERT_TRUE(spv.success) << spv.error;

    std::string val_err;
    EXPECT_TRUE(wgsl_test::ValidateSpirv(spv.words.data(), spv.words.size(), &val_err))
        << "spirv-val: " << val_err;

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GE(CountOpsAll(f, SSIR_OP_TEX_GATHER), 1u);
}

/* E2E-7: Array texture sampling — sample from a 2D texture array with
 * different layer indices, used for terrain material blending. Uses registers
 * for layer index and provides LOD as 5th element (parser uses ncoords=4
 * for a2d, so LOD must be at index 4). */
TEST(PtxTexture, E2E_TextureArrayMaterialBlend) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref material_array;

        .visible .entry blend_materials(
            .param .u64 output_ptr
        )
        .reqntid 64, 1, 1
        {
            .reg .f32 %f<24>;
            .reg .f32 %layer;
            .reg .u32 %r<4>;
            .reg .u64 %rd<4>;

            ld.param.u64 %rd0, [output_ptr];
            mov.u32 %r0, %tid.x;
            cvt.rn.f32.u32 %f20, %r0;
            mov.f32 %f21, 0f3F000000;

            // Layer 0 (grass) — LOD 0.0 as 5th element
            mov.f32 %layer, 0f00000000;
            tex.level.a2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [material_array, {%layer, %f20, %f21, 0f00000000, 0f00000000}];

            // Layer 1 (rock)
            mov.f32 %layer, 0f3F800000;
            tex.level.a2d.v4.f32.f32 {%f4, %f5, %f6, %f7}, [material_array, {%layer, %f20, %f21, 0f00000000, 0f00000000}];

            // Layer 2 (sand)
            mov.f32 %layer, 0f40000000;
            tex.level.a2d.v4.f32.f32 {%f8, %f9, %f10, %f11}, [material_array, {%layer, %f20, %f21, 0f00000000, 0f00000000}];

            // Blend: 50% grass + 30% rock + 20% sand
            mul.f32 %f12, %f0, 0f3F000000;
            mul.f32 %f13, %f4, 0f3E99999A;
            mul.f32 %f14, %f8, 0f3E4CCCCD;
            add.f32 %f15, %f12, %f13;
            add.f32 %f15, %f15, %f14;

            cvt.u64.u32 %rd1, %r0;
            mul.lo.u64 %rd1, %rd1, 4;
            add.u64 %rd2, %rd0, %rd1;
            st.global.f32 [%rd2], %f15;

            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    auto spv = EmitSpirv(r.mod);
    ASSERT_TRUE(spv.success) << spv.error;

    std::string val_err;
    EXPECT_TRUE(wgsl_test::ValidateSpirv(spv.words.data(), spv.words.size(), &val_err))
        << "spirv-val: " << val_err;

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GE(CountOpsAll(f, SSIR_OP_TEX_SAMPLE_LEVEL), 3u);
}

/* E2E-8: Mixed texture and surface operations — sample a texture, process
 * with compute, then write to a storage surface. Full image pipeline. */
TEST(PtxTexture, E2E_TextureSampleToSurfaceStore) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref color_input;
        .global .surfref processed_output;

        .visible .entry tex_to_surf(
            .param .u32 width
        )
        .reqntid 64, 1, 1
        {
            .reg .f32 %f<12>;
            .reg .u32 %r<12>;

            mov.u32 %r0, %tid.x;
            mov.u32 %r1, %ctaid.x;
            mov.u32 %r2, %ntid.x;
            mad.lo.u32 %r0, %r1, %r2, %r0;

            ld.param.u32 %r3, [width];

            // 2D coords
            div.u32 %r4, %r0, %r3;
            rem.u32 %r5, %r0, %r3;

            // Normalized UV
            cvt.rn.f32.u32 %f0, %r5;
            cvt.rn.f32.u32 %f1, %r4;

            // Sample color texture at LOD 0
            tex.level.2d.v4.f32.f32 {%f2, %f3, %f4, %f5}, [color_input, {%f0, %f1, 0f00000000}];

            // Invert colors
            mov.f32 %f6, 0f3F800000;
            sub.f32 %f2, %f6, %f2;
            sub.f32 %f3, %f6, %f3;
            sub.f32 %f4, %f6, %f4;

            // Convert to u32 for surface store (scale to 0..255)
            mul.f32 %f2, %f2, 0f437F0000;
            mul.f32 %f3, %f3, 0f437F0000;
            mul.f32 %f4, %f4, 0f437F0000;
            mul.f32 %f5, %f5, 0f437F0000;

            cvt.rzi.u32.f32 %r6, %f2;
            cvt.rzi.u32.f32 %r7, %f3;
            cvt.rzi.u32.f32 %r8, %f4;
            cvt.rzi.u32.f32 %r9, %f5;

            // Write to output surface
            sust.b.2d.v4.b32 [processed_output, {%r5, %r4}], {%r6, %r7, %r8, %r9};

            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    auto spv = EmitSpirv(r.mod);
    ASSERT_TRUE(spv.success) << spv.error;

    std::string val_err;
    EXPECT_TRUE(wgsl_test::ValidateSpirv(spv.words.data(), spv.words.size(), &val_err))
        << "spirv-val: " << val_err;

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GE(CountOpsAll(f, SSIR_OP_TEX_SAMPLE_LEVEL), 1u);
    EXPECT_GE(CountOpsAll(f, SSIR_OP_TEX_STORE), 1u);
}

/* E2E-9: Texture size query + level sampling — query mip dimensions
 * then sample at a computed LOD. */
TEST(PtxTexture, E2E_QuerySizeThenSample) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref detail_tex;

        .visible .entry query_and_sample(
            .param .u64 output_ptr
        )
        .reqntid 32, 1, 1
        {
            .reg .f32 %f<12>;
            .reg .u32 %r<8>;
            .reg .u64 %rd<4>;

            ld.param.u64 %rd0, [output_ptr];
            mov.u32 %r0, %tid.x;

            // Query texture width
            txq.width.b32 %r1, [detail_tex];

            // Query number of mip levels
            txq.num_mipmap_levels.b32 %r2, [detail_tex];

            // Sample at LOD 0
            cvt.rn.f32.u32 %f0, %r0;
            mov.f32 %f1, 0f3F000000;
            tex.level.2d.v4.f32.f32 {%f2, %f3, %f4, %f5}, [detail_tex, {%f0, %f1, 0f00000000}];

            // Sample at last mip level (convert query result to LOD)
            sub.u32 %r3, %r2, 1;
            cvt.rn.f32.u32 %f6, %r3;
            tex.level.2d.v4.f32.f32 {%f7, %f8, %f9, %f10}, [detail_tex, {%f0, %f1, %f6}];

            // Store difference between LOD 0 and last LOD
            sub.f32 %f11, %f2, %f7;

            cvt.u64.u32 %rd1, %r0;
            mul.lo.u64 %rd1, %rd1, 4;
            add.u64 %rd2, %rd0, %rd1;
            st.global.f32 [%rd2], %f11;

            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    auto spv = EmitSpirv(r.mod);
    ASSERT_TRUE(spv.success) << spv.error;

    std::string val_err;
    EXPECT_TRUE(wgsl_test::ValidateSpirv(spv.words.data(), spv.words.size(), &val_err))
        << "spirv-val: " << val_err;

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GE(CountOpsAll(f, SSIR_OP_TEX_SIZE), 1u);
    EXPECT_GE(CountOpsAll(f, SSIR_OP_TEX_QUERY_LEVELS), 1u);
    EXPECT_GE(CountOpsAll(f, SSIR_OP_TEX_SAMPLE_LEVEL), 2u);
}

/* E2E-10: Multi-pass texture gather — gather all 4 RGBA channels from
 * different texture locations for bilateral filter approximation. */
TEST(PtxTexture, E2E_MultiChannelGatherBilateral) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref filter_input;

        .visible .entry bilateral_gather(
            .param .u64 output_ptr
        )
        .reqntid 64, 1, 1
        {
            .reg .f32 %f<28>;
            .reg .u32 %r<4>;
            .reg .u64 %rd<4>;

            ld.param.u64 %rd0, [output_ptr];
            mov.u32 %r0, %tid.x;
            cvt.rn.f32.u32 %f20, %r0;
            mov.f32 %f21, 0f3F000000;

            // Gather red channel
            tld4.r.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [filter_input, {%f20, %f21}];

            // Gather green channel
            tld4.g.2d.v4.f32.f32 {%f4, %f5, %f6, %f7}, [filter_input, {%f20, %f21}];

            // Gather blue channel
            tld4.b.2d.v4.f32.f32 {%f8, %f9, %f10, %f11}, [filter_input, {%f20, %f21}];

            // Gather alpha channel
            tld4.a.2d.v4.f32.f32 {%f12, %f13, %f14, %f15}, [filter_input, {%f20, %f21}];

            // Compute per-channel averages
            add.f32 %f16, %f0, %f1;
            add.f32 %f16, %f16, %f2;
            add.f32 %f16, %f16, %f3;
            mul.f32 %f16, %f16, 0f3E800000;

            add.f32 %f17, %f4, %f5;
            add.f32 %f17, %f17, %f6;
            add.f32 %f17, %f17, %f7;
            mul.f32 %f17, %f17, 0f3E800000;

            // Luminance: 0.299R + 0.587G
            mul.f32 %f18, %f16, 0f3E991687;
            mul.f32 %f19, %f17, 0f3F1645A2;
            add.f32 %f18, %f18, %f19;

            cvt.u64.u32 %rd1, %r0;
            mul.lo.u64 %rd1, %rd1, 4;
            add.u64 %rd2, %rd0, %rd1;
            st.global.f32 [%rd2], %f18;

            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    auto spv = EmitSpirv(r.mod);
    ASSERT_TRUE(spv.success) << spv.error;

    std::string val_err;
    EXPECT_TRUE(wgsl_test::ValidateSpirv(spv.words.data(), spv.words.size(), &val_err))
        << "spirv-val: " << val_err;

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GE(CountOpsAll(f, SSIR_OP_TEX_GATHER), 4u);
}

/* E2E-11: 3D volume texture — sample along a ray through a 3D volume
 * texture for volume rendering (front-to-back compositing). */
TEST(PtxTexture, E2E_VolumeRayMarching) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref volume_data;

        .visible .entry volume_march(
            .param .u64 output_ptr
        )
        .reqntid 64, 1, 1
        {
            .reg .f32 %f<24>;
            .reg .u32 %r<4>;
            .reg .u64 %rd<4>;

            ld.param.u64 %rd0, [output_ptr];
            mov.u32 %r0, %tid.x;

            // Ray origin based on thread ID
            cvt.rn.f32.u32 %f20, %r0;
            mul.f32 %f20, %f20, 0f3C23D70A;
            mov.f32 %f21, 0f3F000000;

            // March along Z axis: sample at z=0.1, 0.3, 0.5, 0.7, 0.9
            tex.level.3d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [volume_data, {%f20, %f21, 0f3DCCCCCD, 0f00000000}];
            tex.level.3d.v4.f32.f32 {%f4, %f5, %f6, %f7}, [volume_data, {%f20, %f21, 0f3E99999A, 0f00000000}];
            tex.level.3d.v4.f32.f32 {%f8, %f9, %f10, %f11}, [volume_data, {%f20, %f21, 0f3F000000, 0f00000000}];
            tex.level.3d.v4.f32.f32 {%f12, %f13, %f14, %f15}, [volume_data, {%f20, %f21, 0f3F333333, 0f00000000}];
            tex.level.3d.v4.f32.f32 {%f16, %f17, %f18, %f19}, [volume_data, {%f20, %f21, 0f3F666666, 0f00000000}];

            // Front-to-back composite: weighted sum of red channels
            mul.f32 %f0, %f0, 0f3E4CCCCD;
            mul.f32 %f4, %f4, 0f3E4CCCCD;
            mul.f32 %f8, %f8, 0f3E4CCCCD;
            mul.f32 %f12, %f12, 0f3E4CCCCD;
            mul.f32 %f16, %f16, 0f3E4CCCCD;

            add.f32 %f0, %f0, %f4;
            add.f32 %f0, %f0, %f8;
            add.f32 %f0, %f0, %f12;
            add.f32 %f0, %f0, %f16;

            cvt.u64.u32 %rd1, %r0;
            mul.lo.u64 %rd1, %rd1, 4;
            add.u64 %rd2, %rd0, %rd1;
            st.global.f32 [%rd2], %f0;

            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    auto spv = EmitSpirv(r.mod);
    ASSERT_TRUE(spv.success) << spv.error;

    std::string val_err;
    EXPECT_TRUE(wgsl_test::ValidateSpirv(spv.words.data(), spv.words.size(), &val_err))
        << "spirv-val: " << val_err;

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GE(CountOpsAll(f, SSIR_OP_TEX_SAMPLE_LEVEL), 5u);
}

/* E2E-12: Mixed texture operations — sequential level and grad sampling
 * combined with texture query, plus surface load. No branching. */
TEST(PtxTexture, E2E_ConditionalSamplingWithQuery) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref diffuse_map;
        .global .surfref normal_surf;

        .visible .entry conditional_sample(
            .param .u64 output_ptr
        )
        .reqntid 64, 1, 1
        {
            .reg .f32 %f<16>;
            .reg .u32 %r<8>;
            .reg .u64 %rd<4>;

            ld.param.u64 %rd0, [output_ptr];
            mov.u32 %r0, %tid.x;

            // Query texture dimensions and mip levels (txq on sampled texture)
            txq.width.b32 %r1, [diffuse_map];
            txq.height.b32 %r2, [diffuse_map];
            txq.num_mipmap_levels.b32 %r3, [diffuse_map];

            cvt.rn.f32.u32 %f10, %r0;
            mov.f32 %f11, 0f3F000000;

            // Level sampling path
            tex.level.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [diffuse_map, {%f10, %f11, 0f00000000}];

            // Grad sampling path (sequential, no branching)
            tex.grad.2d.v4.f32.f32 {%f4, %f5, %f6, %f7}, [diffuse_map, {%f10, %f11, 0f3DCCCCCD, 0f00000000, 0f00000000, 0f3DCCCCCD}];

            // Load from normal surface
            rem.u32 %r5, %r0, %r1;
            rem.u32 %r6, %r0, %r2;
            suld.b.2d.v4.b32 {%r4, %r5, %r6, %r7}, [normal_surf, {%r5, %r6}];

            // Combine: average level and grad red, multiply by normal.x
            add.f32 %f8, %f0, %f4;
            mul.f32 %f8, %f8, 0f3F000000;
            cvt.rn.f32.u32 %f9, %r4;
            mul.f32 %f9, %f9, 0f3A800000;
            mul.f32 %f8, %f8, %f9;

            cvt.u64.u32 %rd1, %r0;
            mul.lo.u64 %rd1, %rd1, 4;
            add.u64 %rd2, %rd0, %rd1;
            st.global.f32 [%rd2], %f8;

            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    auto spv = EmitSpirv(r.mod);
    ASSERT_TRUE(spv.success) << spv.error;

    std::string val_err;
    EXPECT_TRUE(wgsl_test::ValidateSpirv(spv.words.data(), spv.words.size(), &val_err))
        << "spirv-val: " << val_err;

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GE(CountOpsAll(f, SSIR_OP_TEX_SAMPLE_LEVEL), 1u);
    EXPECT_GE(CountOpsAll(f, SSIR_OP_TEX_SAMPLE_GRAD), 1u);
    EXPECT_GE(CountOpsAll(f, SSIR_OP_TEX_LOAD), 1u);
    EXPECT_GE(CountOpsAll(f, SSIR_OP_TEX_SIZE), 2u);
}

/* E2E-13: 1D array texture — sample from multiple layers of a 1D array
 * texture for LUT (lookup table) based color grading. */
TEST(PtxTexture, E2E_1DArrayLUTColorGrading) {
    std::string ptx = std::string(PTX_HEADER) + R"(
        .global .texref lut_array;

        .visible .entry color_grade(
            .param .u64 input_ptr,
            .param .u64 output_ptr
        )
        .reqntid 64, 1, 1
        {
            .reg .f32 %f<16>;
            .reg .u32 %r<4>;
            .reg .u64 %rd<8>;

            ld.param.u64 %rd0, [input_ptr];
            ld.param.u64 %rd1, [output_ptr];
            mov.u32 %r0, %tid.x;

            // Load input color
            cvt.u64.u32 %rd2, %r0;
            mul.lo.u64 %rd3, %rd2, 12;
            add.u64 %rd4, %rd0, %rd3;
            ld.global.f32 %f0, [%rd4];
            add.u64 %rd5, %rd4, 4;
            ld.global.f32 %f1, [%rd5];
            add.u64 %rd5, %rd5, 4;
            ld.global.f32 %f2, [%rd5];

            // Use red channel as LUT coordinate, layer 0
            tex.level.a1d.v4.f32.f32 {%f3, %f4, %f5, %f6}, [lut_array, {0f00000000, %f0, 0f00000000}];

            // Use green channel as LUT coordinate, layer 1
            tex.level.a1d.v4.f32.f32 {%f7, %f8, %f9, %f10}, [lut_array, {0f3F800000, %f1, 0f00000000}];

            // Use blue channel as LUT coordinate, layer 2
            tex.level.a1d.v4.f32.f32 {%f11, %f12, %f13, %f14}, [lut_array, {0f40000000, %f2, 0f00000000}];

            // Store graded color
            mul.lo.u64 %rd6, %rd2, 12;
            add.u64 %rd7, %rd1, %rd6;
            st.global.f32 [%rd7], %f3;
            add.u64 %rd7, %rd7, 4;
            st.global.f32 [%rd7], %f7;
            add.u64 %rd7, %rd7, 4;
            st.global.f32 [%rd7], %f11;

            ret;
        }
    )";
    auto r = Parse(ptx);
    ASSERT_TRUE(r.success) << r.error;
    SsirGuard guard(r.mod);

    auto spv = EmitSpirv(r.mod);
    ASSERT_TRUE(spv.success) << spv.error;

    std::string val_err;
    EXPECT_TRUE(wgsl_test::ValidateSpirv(spv.words.data(), spv.words.size(), &val_err))
        << "spirv-val: " << val_err;

    SsirFunction *f = &r.mod->functions[0];
    EXPECT_GE(CountOpsAll(f, SSIR_OP_TEX_SAMPLE_LEVEL), 3u);
}
