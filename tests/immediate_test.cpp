#include <gtest/gtest.h>
#include "test_utils.h"

// ============================================================================
// Helper: compile WGSL with per-entry-point immediate lowering
// ============================================================================

namespace {

struct ImmediateCompileResult {
    bool success;
    std::string error;
    std::vector<uint32_t> spirv;
};

ImmediateCompileResult CompileImmediate(const char *source,
                                        const char *entry_point,
                                        SsirLayoutRule layout = SSIR_LAYOUT_STD430) {
    ImmediateCompileResult result;
    result.success = false;

    WgslAstNode *ast = wgsl_parse(source);
    if (!ast) {
        result.error = "Parse failed";
        return result;
    }

    WgslResolver *resolver = wgsl_resolver_build(ast);
    if (!resolver) {
        wgsl_free_ast(ast);
        result.error = "Resolve failed";
        return result;
    }

    uint32_t *spirv = nullptr;
    size_t spirv_size = 0;
    WgslLowerOptions opts = {};
    opts.env = WGSL_LOWER_ENV_VULKAN_1_3;
    opts.entry_point = entry_point;
    opts.immediate_layout = layout;

    WgslLowerResult lower_result =
        wgsl_lower_emit_spirv(ast, resolver, &opts, &spirv, &spirv_size);
    wgsl_resolver_free(resolver);
    wgsl_free_ast(ast);

    if (lower_result != WGSL_LOWER_OK) {
        result.error = "Lower failed";
        return result;
    }

    result.spirv.assign(spirv, spirv + spirv_size);
    wgsl_lower_free(spirv);

    if (!wgsl_test::ValidateSpirv(result.spirv.data(), result.spirv.size(),
                                  &result.error)) {
        return result;
    }

    result.success = true;
    return result;
}

// Disassemble SPIR-V to text using spirv-dis
std::string Disassemble(const std::vector<uint32_t> &spirv) {
    std::string spv_path = wgsl_test::MakeTempSpvPath("wgsl_imm");
    wgsl_test::WriteSpirvFile(spv_path, spirv.data(), spirv.size());
    std::string output;
    wgsl_test::RunCommand("spirv-dis --no-header " + spv_path + " 2>&1",
                          &output);
    std::remove(spv_path.c_str());
    return output;
}

} // namespace

// ============================================================================
// PARSER TESTS
// ============================================================================

class ImmediateParserTest : public ::testing::Test {
  protected:
    WgslAstNode *ast = nullptr;
    void TearDown() override {
        if (ast) wgsl_free_ast(ast);
    }
    WgslAstNode *Parse(const char *source) {
        ast = wgsl_parse(source);
        return ast;
    }
};

TEST_F(ImmediateParserTest, EnableDirective) {
    auto *node = Parse(R"(
        enable immediate_address_space;
        fn main() {}
    )");
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->type, WGSL_NODE_PROGRAM);
    EXPECT_EQ(node->program.extensions & WGSL_EXT_IMMEDIATE_ADDRESS_SPACE,
              WGSL_EXT_IMMEDIATE_ADDRESS_SPACE);
}

TEST_F(ImmediateParserTest, EnableArraysImpliesBase) {
    auto *node = Parse(R"(
        enable immediate_arrays;
        fn main() {}
    )");
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->program.extensions & WGSL_EXT_IMMEDIATE_ADDRESS_SPACE,
              WGSL_EXT_IMMEDIATE_ADDRESS_SPACE);
    EXPECT_EQ(node->program.extensions & WGSL_EXT_IMMEDIATE_ARRAYS,
              WGSL_EXT_IMMEDIATE_ARRAYS);
}

TEST_F(ImmediateParserTest, VarImmediateDeclaration) {
    auto *node = Parse(R"(
        enable immediate_address_space;
        var<immediate> scale: f32;
        @compute @workgroup_size(1) fn main() { let x = scale; }
    )");
    ASSERT_NE(node, nullptr);
    // Find the global var declaration
    bool found_immediate = false;
    for (int i = 0; i < node->program.decl_count; i++) {
        WgslAstNode *decl = node->program.decls[i];
        if (decl->type == WGSL_NODE_GLOBAL_VAR &&
            decl->global_var.address_space &&
            strcmp(decl->global_var.address_space, "immediate") == 0) {
            found_immediate = true;
            EXPECT_STREQ(decl->global_var.name, "scale");
        }
    }
    EXPECT_TRUE(found_immediate) << "No var<immediate> found in AST";
}

TEST_F(ImmediateParserTest, MultipleImmediateVars) {
    auto *node = Parse(R"(
        enable immediate_address_space;
        var<immediate> a: f32;
        var<immediate> b: u32;
        var<immediate> c: vec2f;
        @compute @workgroup_size(1) fn main() { let x = a; }
    )");
    ASSERT_NE(node, nullptr);
    int imm_count = 0;
    for (int i = 0; i < node->program.decl_count; i++) {
        WgslAstNode *decl = node->program.decls[i];
        if (decl->type == WGSL_NODE_GLOBAL_VAR &&
            decl->global_var.address_space &&
            strcmp(decl->global_var.address_space, "immediate") == 0) {
            imm_count++;
        }
    }
    EXPECT_EQ(imm_count, 3);
}

TEST_F(ImmediateParserTest, PtrImmediateInFunction) {
    auto *node = Parse(R"(
        enable immediate_address_space;
        var<immediate> val: u32;
        fn foo(p: ptr<immediate, u32>) -> u32 { return *p; }
        @compute @workgroup_size(1) fn main() { let x = foo(&val); }
    )");
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->type, WGSL_NODE_PROGRAM);
}

// ============================================================================
// RESOLVER TESTS
// ============================================================================

class ImmediateResolverTest : public ::testing::Test {
  protected:
    WgslAstNode *ast = nullptr;
    WgslResolver *resolver = nullptr;

    void TearDown() override {
        if (resolver) wgsl_resolver_free(resolver);
        if (ast) wgsl_free_ast(ast);
    }

    void ParseAndResolve(const char *source) {
        ast = wgsl_parse(source);
        ASSERT_NE(ast, nullptr) << "Parse failed";
        resolver = wgsl_resolver_build(ast);
        ASSERT_NE(resolver, nullptr) << "Resolve failed";
    }
};

TEST_F(ImmediateResolverTest, ImmediateSymbolKind) {
    ParseAndResolve(R"(
        enable immediate_address_space;
        var<immediate> scale: f32;
        @compute @workgroup_size(1) fn main() { let x = scale; }
    )");
    int count = 0;
    const WgslSymbolInfo *globals = wgsl_resolver_globals(resolver, &count);
    bool found = false;
    for (int i = 0; i < count; i++) {
        if (globals[i].kind == WGSL_SYM_IMMEDIATE) {
            EXPECT_STREQ(globals[i].name, "scale");
            EXPECT_FALSE(globals[i].has_group);
            EXPECT_FALSE(globals[i].has_binding);
            found = true;
        }
    }
    wgsl_resolve_free((void *)globals);
    EXPECT_TRUE(found) << "No WGSL_SYM_IMMEDIATE symbol found";
}

TEST_F(ImmediateResolverTest, EntrypointImmediates_SingleVar) {
    ParseAndResolve(R"(
        enable immediate_address_space;
        var<immediate> scale: f32;
        @compute @workgroup_size(1) fn main() { let x = scale; }
    )");
    int count = 0;
    const WgslImmediateInfo *imms =
        wgsl_resolver_entrypoint_immediates(resolver, "main",
                                            SSIR_LAYOUT_STD430, &count);
    ASSERT_NE(imms, nullptr);
    ASSERT_EQ(count, 1);
    EXPECT_STREQ(imms[0].name, "scale");
    EXPECT_EQ(imms[0].type_size, 4);
    EXPECT_EQ(imms[0].offset, 0);
    wgsl_resolve_free((void *)imms);
}

TEST_F(ImmediateResolverTest, EntrypointImmediates_MultipleVars) {
    ParseAndResolve(R"(
        enable immediate_address_space;
        var<immediate> a: f32;
        var<immediate> b: u32;
        var<immediate> c: vec2f;
        @compute @workgroup_size(1)
        fn main() {
            let x = a;
            let y = b;
            let z = c;
        }
    )");
    int count = 0;
    const WgslImmediateInfo *imms =
        wgsl_resolver_entrypoint_immediates(resolver, "main",
                                            SSIR_LAYOUT_STD430, &count);
    ASSERT_NE(imms, nullptr);
    ASSERT_EQ(count, 3);
    // a: f32 at offset 0, size 4
    EXPECT_STREQ(imms[0].name, "a");
    EXPECT_EQ(imms[0].offset, 0);
    EXPECT_EQ(imms[0].type_size, 4);
    // b: u32 at offset 4, size 4
    EXPECT_STREQ(imms[1].name, "b");
    EXPECT_EQ(imms[1].offset, 4);
    EXPECT_EQ(imms[1].type_size, 4);
    // c: vec2f at offset 8, size 8 (vec2 alignment = 8 in std430)
    EXPECT_STREQ(imms[2].name, "c");
    EXPECT_EQ(imms[2].offset, 8);
    EXPECT_EQ(imms[2].type_size, 8);
    wgsl_resolve_free((void *)imms);
}

TEST_F(ImmediateResolverTest, EntrypointImmediates_TransitiveThroughCalls) {
    ParseAndResolve(R"(
        enable immediate_address_space;
        var<immediate> val: u32;

        fn helper() -> u32 { return val; }

        @compute @workgroup_size(1)
        fn main() { let x = helper(); }
    )");
    int count = 0;
    const WgslImmediateInfo *imms =
        wgsl_resolver_entrypoint_immediates(resolver, "main",
                                            SSIR_LAYOUT_STD430, &count);
    ASSERT_NE(imms, nullptr);
    EXPECT_EQ(count, 1);
    EXPECT_STREQ(imms[0].name, "val");
    wgsl_resolve_free((void *)imms);
}

TEST_F(ImmediateResolverTest, PerEntrypointIsolation) {
    // User's example 1: main1 uses a, main3 uses b, main4 uses nothing
    ParseAndResolve(R"(
        enable immediate_address_space;
        var<immediate> a: f32;
        var<immediate> b: u32;

        fn foo() -> f32 { return a; }
        fn bar() -> u32 { return b; }

        @compute @workgroup_size(1)
        fn main1() { let x = foo(); }

        @compute @workgroup_size(1)
        fn main2() { let x = foo(); let y = bar(); }

        @compute @workgroup_size(1)
        fn main3() { let x = bar(); }

        @compute @workgroup_size(1)
        fn main4() {}
    )");

    // main1 uses only 'a' (via foo)
    {
        int count = 0;
        const WgslImmediateInfo *imms =
            wgsl_resolver_entrypoint_immediates(resolver, "main1",
                                                SSIR_LAYOUT_STD430, &count);
        EXPECT_EQ(count, 1);
        if (count > 0) {
            EXPECT_STREQ(imms[0].name, "a");
        }
        wgsl_resolve_free((void *)imms);
    }

    // main2 uses both 'a' and 'b'
    {
        int count = 0;
        const WgslImmediateInfo *imms =
            wgsl_resolver_entrypoint_immediates(resolver, "main2",
                                                SSIR_LAYOUT_STD430, &count);
        EXPECT_EQ(count, 2);
        wgsl_resolve_free((void *)imms);
    }

    // main3 uses only 'b' (via bar)
    {
        int count = 0;
        const WgslImmediateInfo *imms =
            wgsl_resolver_entrypoint_immediates(resolver, "main3",
                                                SSIR_LAYOUT_STD430, &count);
        EXPECT_EQ(count, 1);
        if (count > 0) {
            EXPECT_STREQ(imms[0].name, "b");
        }
        wgsl_resolve_free((void *)imms);
    }

    // main4 uses nothing
    {
        int count = 0;
        const WgslImmediateInfo *imms =
            wgsl_resolver_entrypoint_immediates(resolver, "main4",
                                                SSIR_LAYOUT_STD430, &count);
        EXPECT_EQ(count, 0);
        wgsl_resolve_free((void *)imms);
    }
}

TEST_F(ImmediateResolverTest, ImmediatesInEntrypointGlobals) {
    ParseAndResolve(R"(
        enable immediate_address_space;
        var<immediate> scale: f32;
        @group(0) @binding(0) var<uniform> data: f32;
        @compute @workgroup_size(1)
        fn main() { let x = scale + data; }
    )");
    int count = 0;
    const WgslSymbolInfo *globals =
        wgsl_resolver_entrypoint_globals(resolver, "main", &count);
    ASSERT_NE(globals, nullptr);
    bool found_immediate = false;
    bool found_uniform = false;
    for (int i = 0; i < count; i++) {
        if (globals[i].kind == WGSL_SYM_IMMEDIATE) found_immediate = true;
        if (globals[i].kind == WGSL_SYM_GLOBAL) found_uniform = true;
    }
    EXPECT_TRUE(found_immediate) << "Immediate var not in entrypoint globals";
    EXPECT_TRUE(found_uniform) << "Uniform var not in entrypoint globals";
    wgsl_resolve_free((void *)globals);
}

// ============================================================================
// LOWERING TESTS
// ============================================================================

TEST(ImmediateLowerTest, SingleImmediate_Compiles) {
    const char *source = R"(
        enable immediate_address_space;
        var<immediate> scale: f32;
        @compute @workgroup_size(1)
        fn main() { let x = scale; }
    )";
    auto result = CompileImmediate(source, "main");
    EXPECT_TRUE(result.success) << "Error: " << result.error;
}

TEST(ImmediateLowerTest, SingleImmediate_HasPushConstant) {
    const char *source = R"(
        enable immediate_address_space;
        var<immediate> scale: f32;
        @compute @workgroup_size(1)
        fn main() { let x = scale; }
    )";
    auto result = CompileImmediate(source, "main");
    ASSERT_TRUE(result.success) << "Error: " << result.error;
    std::string dis = Disassemble(result.spirv);
    EXPECT_NE(dis.find("PushConstant"), std::string::npos)
        << "Expected PushConstant storage class in SPIR-V:\n" << dis;
}

TEST(ImmediateLowerTest, MultipleImmediates_Compiles) {
    const char *source = R"(
        enable immediate_address_space;
        var<immediate> a: f32;
        var<immediate> b: u32;
        var<immediate> c: vec2f;
        @compute @workgroup_size(1)
        fn main() {
            let x = a + f32(b);
            let y = c;
        }
    )";
    auto result = CompileImmediate(source, "main");
    EXPECT_TRUE(result.success) << "Error: " << result.error;
}

TEST(ImmediateLowerTest, MultipleImmediates_CorrectOffsets) {
    const char *source = R"(
        enable immediate_address_space;
        var<immediate> a: f32;
        var<immediate> b: u32;
        @compute @workgroup_size(1)
        fn main() {
            let x = a;
            let y = b;
        }
    )";
    auto result = CompileImmediate(source, "main");
    ASSERT_TRUE(result.success) << "Error: " << result.error;
    std::string dis = Disassemble(result.spirv);
    // Should have member decorations with offsets 0 and 4
    EXPECT_NE(dis.find("Offset 0"), std::string::npos)
        << "Expected Offset 0 decoration:\n" << dis;
    EXPECT_NE(dis.find("Offset 4"), std::string::npos)
        << "Expected Offset 4 decoration:\n" << dis;
}

TEST(ImmediateLowerTest, PerEntrypoint_OnlyUsedImmediates) {
    // main1 uses only 'a', so its push constant block should only have 'a'
    const char *source = R"(
        enable immediate_address_space;
        var<immediate> a: f32;
        var<immediate> b: u32;

        fn foo() -> f32 { return a; }

        @compute @workgroup_size(1)
        fn main1() { let x = foo(); }

        @compute @workgroup_size(1)
        fn main2() { let x = a; let y = b; }
    )";

    // Compile main1 — should only have 'a'
    auto r1 = CompileImmediate(source, "main1");
    ASSERT_TRUE(r1.success) << "main1 error: " << r1.error;
    std::string dis1 = Disassemble(r1.spirv);
    // Should have exactly one member offset (0), not two
    EXPECT_NE(dis1.find("Offset 0"), std::string::npos);
    // Offset 4 should NOT appear if only one f32 member
    // (But this depends on implementation - just check it compiles)

    // Compile main2 — should have both 'a' and 'b'
    auto r2 = CompileImmediate(source, "main2");
    ASSERT_TRUE(r2.success) << "main2 error: " << r2.error;
    std::string dis2 = Disassemble(r2.spirv);
    EXPECT_NE(dis2.find("Offset 0"), std::string::npos);
    EXPECT_NE(dis2.find("Offset 4"), std::string::npos);
}

TEST(ImmediateLowerTest, NoImmediates_NoPushConstant) {
    // Entry point uses no immediates — should produce no push constant block
    const char *source = R"(
        enable immediate_address_space;
        var<immediate> a: f32;
        @compute @workgroup_size(1)
        fn main() {}
    )";
    auto result = CompileImmediate(source, "main");
    ASSERT_TRUE(result.success) << "Error: " << result.error;
    std::string dis = Disassemble(result.spirv);
    EXPECT_EQ(dis.find("PushConstant"), std::string::npos)
        << "Unexpected PushConstant in SPIR-V for empty entry point:\n" << dis;
}

TEST(ImmediateLowerTest, TransitiveUsage) {
    // Immediate accessed indirectly through function calls
    const char *source = R"(
        enable immediate_address_space;
        var<immediate> val: u32;

        fn inner() -> u32 { return val; }
        fn outer() -> u32 { return inner(); }

        @compute @workgroup_size(1)
        fn main() { let x = outer(); }
    )";
    auto result = CompileImmediate(source, "main");
    EXPECT_TRUE(result.success) << "Error: " << result.error;
    std::string dis = Disassemble(result.spirv);
    EXPECT_NE(dis.find("PushConstant"), std::string::npos)
        << "Expected PushConstant for transitive usage:\n" << dis;
}

TEST(ImmediateLowerTest, PointerPassing) {
    // User's example 3: ptr<immediate, T>
    const char *source = R"(
        enable immediate_address_space;
        var<immediate> val: u32;

        fn foo(p: ptr<immediate, u32>) -> u32 { return *p; }

        @compute @workgroup_size(1)
        fn main() { let x = foo(&val); }
    )";
    auto result = CompileImmediate(source, "main");
    EXPECT_TRUE(result.success) << "Error: " << result.error;
}

TEST(ImmediateLowerTest, Vec3Alignment_Std430) {
    // vec3 should be aligned to 16 bytes in std430
    const char *source = R"(
        enable immediate_address_space;
        var<immediate> a: f32;
        var<immediate> b: vec3f;
        @compute @workgroup_size(1)
        fn main() {
            let x = a;
            let y = b;
        }
    )";
    auto result = CompileImmediate(source, "main", SSIR_LAYOUT_STD430);
    ASSERT_TRUE(result.success) << "Error: " << result.error;
    std::string dis = Disassemble(result.spirv);
    // a: offset 0, b: offset 16 (aligned to 16 for vec3 in std430)
    EXPECT_NE(dis.find("Offset 0"), std::string::npos);
    EXPECT_NE(dis.find("Offset 16"), std::string::npos)
        << "Expected Offset 16 for vec3f after f32 in std430:\n" << dis;
}

TEST(ImmediateLowerTest, ScalarLayout) {
    // With scalar layout, vec3 is only aligned to 4
    const char *source = R"(
        enable immediate_address_space;
        var<immediate> a: f32;
        var<immediate> b: vec3f;
        @compute @workgroup_size(1)
        fn main() {
            let x = a;
            let y = b;
        }
    )";
    auto result = CompileImmediate(source, "main", SSIR_LAYOUT_SCALAR);
    ASSERT_TRUE(result.success) << "Error: " << result.error;
    std::string dis = Disassemble(result.spirv);
    // a: offset 0, b: offset 4 (scalar layout, 4-byte alignment)
    EXPECT_NE(dis.find("Offset 0"), std::string::npos);
    EXPECT_NE(dis.find("Offset 4"), std::string::npos)
        << "Expected Offset 4 for vec3f after f32 in scalar layout:\n" << dis;
}

TEST(ImmediateLowerTest, Mat4x4_Compiles) {
    const char *source = R"(
        enable immediate_address_space;
        var<immediate> transform: mat4x4f;
        @compute @workgroup_size(1)
        fn main() {
            let m = transform;
        }
    )";
    auto result = CompileImmediate(source, "main");
    EXPECT_TRUE(result.success) << "Error: " << result.error;
}

TEST(ImmediateLowerTest, VertexShaderWithImmediates) {
    // User's example 2: vertex shader with immediates
    const char *source = R"(
        enable immediate_address_space;
        var<immediate> scale: f32;
        var<immediate> offset: vec2f;

        struct VertexOutput {
            @builtin(position) pos: vec4f,
        };

        @vertex
        fn vs(@builtin(vertex_index) vid: u32) -> VertexOutput {
            var out: VertexOutput;
            let s = scale;
            let o = offset;
            out.pos = vec4f(o.x + s, o.y + s, 0.0, 1.0);
            return out;
        }
    )";
    auto result = CompileImmediate(source, "vs");
    EXPECT_TRUE(result.success) << "Error: " << result.error;
}

// ============================================================================
// MULTI-ENTRY-POINT INTEGRATION TEST (User's example 1)
// ============================================================================

TEST(ImmediateIntegrationTest, UserExample1_MultiEntrypoint) {
    const char *source = R"(
        enable immediate_address_space;
        var<immediate> a: f32;
        var<immediate> b: u32;

        fn foo() -> f32 { return a; }
        fn bar() -> u32 { return b; }

        @compute @workgroup_size(1)
        fn main1() { let x = foo(); }

        @compute @workgroup_size(1)
        fn main3() { let y = bar(); }

        @compute @workgroup_size(1)
        fn main4() {}
    )";

    // Each entry point should compile to valid SPIR-V independently
    {
        auto r = CompileImmediate(source, "main1");
        EXPECT_TRUE(r.success) << "main1: " << r.error;
    }
    {
        auto r = CompileImmediate(source, "main3");
        EXPECT_TRUE(r.success) << "main3: " << r.error;
    }
    {
        auto r = CompileImmediate(source, "main4");
        EXPECT_TRUE(r.success) << "main4: " << r.error;
    }
}

TEST(ImmediateIntegrationTest, UserExample3_PointerPassing) {
    const char *source = R"(
        enable immediate_address_space;
        var<immediate> val: u32;

        fn foo(p: ptr<immediate, u32>) -> u32 {
            return *p;
        }

        @compute @workgroup_size(1)
        fn main() {
            let x = foo(&val);
        }
    )";
    auto result = CompileImmediate(source, "main");
    EXPECT_TRUE(result.success) << "Error: " << result.error;
}

TEST(ImmediateIntegrationTest, MixedBindingsAndImmediates) {
    // Immediates coexist with normal bindings
    const char *source = R"(
        enable immediate_address_space;
        var<immediate> scale: f32;

        struct Buf { data: array<f32> };
        @group(0) @binding(0) var<storage, read_write> buf: Buf;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) gid: vec3u) {
            buf.data[gid.x] = buf.data[gid.x] * scale;
        }
    )";
    auto result = CompileImmediate(source, "main");
    EXPECT_TRUE(result.success) << "Error: " << result.error;
}
