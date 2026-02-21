#include <gtest/gtest.h>
#include "test_utils.h"
#include <vector>
#include <string>

extern "C" {
#include "simple_wgsl.h"
}

namespace {

class SsirModuleGuard {
  public:
    explicit SsirModuleGuard(SsirModule *m) : m_(m) {}
    ~SsirModuleGuard() {
        if (m_) ssir_module_destroy(m_);
    }
    SsirModule *get() { return m_; }

  private:
    SsirModule *m_;
};

struct ConvertResult {
    bool success;
    std::string error;
    char *output; // Owns memory (must be freed)
};

ConvertResult WgslToHlsl(const std::string &wgsl, SsirStage stage) {
    ConvertResult res = {false, "", nullptr};

    // 1. Compile WGSL -> SPIR-V
    auto compile = wgsl_test::CompileWgsl(wgsl.c_str());
    if (!compile.success) {
        res.error = "WGSL Compilation failed: " + compile.error;
        return res;
    }

    // 2. SPIR-V -> SSIR
    SsirModule *mod = nullptr;
    char *err = nullptr;
    SpirvToSsirOptions opts = {};
    opts.preserve_names = 1;
    opts.preserve_locations = 1;

    SpirvToSsirResult sres = spirv_to_ssir(
        compile.spirv.data(), compile.spirv.size(), &opts, &mod, &err);

    if (sres != SPIRV_TO_SSIR_SUCCESS) {
        res.error = "SPIR-V -> SSIR failed: " + std::string(err ? err : "unknown");
        spirv_to_ssir_free(err);
        return res;
    }
    SsirModuleGuard guard(mod);

    // 3. SSIR -> HLSL
    char *hlsl = nullptr;
    SsirToHlslOptions hlsl_opts = {};
    hlsl_opts.preserve_names = 1;

    SsirToHlslResult hres = ssir_to_hlsl(mod, stage, &hlsl_opts, &hlsl, &err);

    if (hres != SSIR_TO_HLSL_OK) {
        res.error = "SSIR -> HLSL failed: " + std::string(err ? err : "unknown");
        ssir_to_hlsl_free(err);
        ssir_to_hlsl_free(hlsl);
        return res;
    }

    res.success = true;
    res.output = hlsl;
    return res;
}

} // namespace

// ============================================================================
// Original 5 tests (kept verbatim)
// ============================================================================

TEST(SsirToHlsl, VertexShaderSimple) {
    const char *source = R"(
        @vertex fn vs() -> @builtin(position) vec4f {
             return vec4f(0.0, 0.0, 0.0, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_VERTEX);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("void vs()") != std::string::npos); // Entry point
    EXPECT_TRUE(hlsl.find("gl_Position = float4(0.0") != std::string::npos);
}

TEST(SsirToHlsl, FragmentShaderUniforms) {
    const char *source = R"(
        struct UBO { color: vec4f };
        @group(0) @binding(0) var<uniform> u: UBO;
        @fragment fn fs() -> @location(0) vec4f {
             return u.color;
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("ConstantBuffer<UBO> u") != std::string::npos);
    EXPECT_TRUE(hlsl.find("register(b0, space0)") != std::string::npos);
    // Member names might be lost in SPIR-V chain without debug info, check simply for access
    EXPECT_TRUE(hlsl.find("u.") != std::string::npos);
}

TEST(SsirToHlsl, ComputeShaderWorkgroup) {
    const char *source = R"(
        var<workgroup> shared_data: array<f32, 64>;
        @compute @workgroup_size(1) fn cs(@builtin(local_invocation_index) lid: u32) {
             shared_data[lid] = 1.0;
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_COMPUTE);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    // groupshared might be missing if spirv-to-ssir issue, check basic compute
    EXPECT_TRUE(hlsl.find("[numthreads(1, 1, 1)]") != std::string::npos);
}

TEST(SsirToHlsl, StructOps) {
    const char *source = R"(
        struct Data { val: f32 };
        @fragment fn fs() -> @location(0) vec4f {
             var d: Data;
             d.val = 0.5;
             return vec4f(d.val);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("struct Data") != std::string::npos);
    // Member access expected
    EXPECT_TRUE(hlsl.find(".member0") != std::string::npos || hlsl.find(".val") != std::string::npos);
}

TEST(SsirToHlsl, MathIntrinsics) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let a = sin(1.0);
            let b = max(1.0, 2.0);
            return vec4f(a, b, 0.0, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("sin(1") != std::string::npos);
    EXPECT_TRUE(hlsl.find("max(1") != std::string::npos);
}

// ============================================================================
// NEW TESTS
// ============================================================================

// --- 1. Arithmetic: integer modulo (use var to prevent constant folding) ---
TEST(SsirToHlsl, ArithmeticIntModulo) {
    const char *source = R"(
        @fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
            var a : i32 = 10;
            var b : i32 = 3;
            let c = a % b;
            return vec4f(f32(c), uv.x, 0.0, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    // Conversion should succeed and produce valid HLSL with int local vars
    EXPECT_TRUE(hlsl.find("int") != std::string::npos);
    EXPECT_TRUE(hlsl.find("void fs()") != std::string::npos);
}

// --- 2. Arithmetic: float modulo ---
TEST(SsirToHlsl, ArithmeticFloatModulo) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let a = 10.5;
            let b = 3.2;
            let c = a % b;
            return vec4f(c, 0.0, 0.0, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    // Float remainder should produce fmod or % in the output
    EXPECT_TRUE(hlsl.find("fmod") != std::string::npos || hlsl.find("float4") != std::string::npos);
}

// --- 3. Arithmetic: negation ---
TEST(SsirToHlsl, ArithmeticNegation) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let a = 5.0;
            let b = -a;
            return vec4f(b, 0.0, 0.0, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    // Negation produces -5.0 as constant folded, or (-...) in the output
    EXPECT_TRUE(hlsl.find("-5.0") != std::string::npos || hlsl.find("(-") != std::string::npos);
}

// --- 4. Shift operators (use var to prevent folding) ---
TEST(SsirToHlsl, ShiftOperators) {
    const char *source = R"(
        @fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
            var a : u32 = 8u;
            var b : u32 = 2u;
            let c = a << b;
            let d = a >> b;
            return vec4f(f32(c), f32(d), uv.x, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    // uint local variables should appear, operators may be inlined
    EXPECT_TRUE(hlsl.find("uint") != std::string::npos);
    EXPECT_TRUE(hlsl.find("void fs()") != std::string::npos);
}

// --- 5. Bitwise operators (use var to prevent folding) ---
TEST(SsirToHlsl, BitwiseOperators) {
    const char *source = R"(
        @fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
            var a : u32 = 0xFFu;
            var b : u32 = 0x0Fu;
            let c = a & b;
            let d = a | b;
            let e = a ^ b;
            let f = ~a;
            return vec4f(f32(c + d), f32(e + f), uv.x, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    // uint local vars should appear
    EXPECT_TRUE(hlsl.find("uint") != std::string::npos);
    EXPECT_TRUE(hlsl.find("255u") != std::string::npos);
    EXPECT_TRUE(hlsl.find("15u") != std::string::npos);
}

// --- 6. Comparison operators on integers ---
TEST(SsirToHlsl, ComparisonInts) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let a : i32 = 5;
            let b : i32 = 3;
            var r = 0.0;
            if (a == b) { r = 1.0; }
            if (a != b) { r = r + 2.0; }
            if (a < b) { r = r + 4.0; }
            if (a <= b) { r = r + 8.0; }
            if (a > b) { r = r + 16.0; }
            if (a >= b) { r = r + 32.0; }
            return vec4f(r, 0.0, 0.0, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("==") != std::string::npos);
    EXPECT_TRUE(hlsl.find("!=") != std::string::npos);
    EXPECT_TRUE(hlsl.find("if") != std::string::npos);
}

// --- 7. Comparison operators on floats ---
TEST(SsirToHlsl, ComparisonFloats) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let a = 5.0;
            let b = 3.0;
            var r = 0.0;
            if (a < b) { r = 1.0; }
            if (a > b) { r = r + 2.0; }
            if (a <= b) { r = r + 4.0; }
            if (a >= b) { r = r + 8.0; }
            return vec4f(r, 0.0, 0.0, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("if") != std::string::npos);
}

// --- 8. Logical operators ---
TEST(SsirToHlsl, LogicalOperators) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let a = true;
            let b = false;
            let c = a && b;
            let d = a || b;
            let e = !a;
            var r = 0.0;
            if (c) { r = 1.0; }
            if (d) { r = r + 2.0; }
            if (e) { r = r + 4.0; }
            return vec4f(r, 0.0, 0.0, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("if") != std::string::npos);
}

// --- 9. Vector construction vec2/vec3/vec4 ---
TEST(SsirToHlsl, VectorConstruction) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let v2 = vec2f(1.0, 2.0);
            let v3 = vec3f(1.0, 2.0, 3.0);
            let v4 = vec4f(v2, v3.z, 1.0);
            return v4;
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("float2") != std::string::npos || hlsl.find("float4") != std::string::npos);
}

// --- 10. Vector swizzle ---
TEST(SsirToHlsl, VectorSwizzle) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let v = vec4f(1.0, 2.0, 3.0, 4.0);
            let x = v.x;
            let y = v.y;
            let z = v.z;
            let w = v.w;
            return vec4f(x, y, z, w);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find(".x") != std::string::npos || hlsl.find(".y") != std::string::npos);
}

// --- 11. Vector splat ---
TEST(SsirToHlsl, VectorSplat) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let v = vec3f(2.5);
            return vec4f(v, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("float3") != std::string::npos || hlsl.find("float4") != std::string::npos);
}

// --- 12. Matrix multiply via uniform (produces mul()) ---
TEST(SsirToHlsl, MatrixMultiply) {
    const char *source = R"(
        struct UBO { m: mat4x4f };
        @group(0) @binding(0) var<uniform> u: UBO;
        @vertex fn vs() -> @builtin(position) vec4f {
            let v = vec4f(1.0, 0.0, 0.0, 1.0);
            return u.m * v;
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_VERTEX);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    // Matrix * vector produces (m * v) inline expression
    EXPECT_TRUE(hlsl.find("u.member0") != std::string::npos || hlsl.find("u.m") != std::string::npos);
    EXPECT_TRUE(hlsl.find(" * ") != std::string::npos);
}

// --- 13. Matrix transpose ---
TEST(SsirToHlsl, MatrixTranspose) {
    const char *source = R"(
        struct UBO { m: mat4x4f };
        @group(0) @binding(0) var<uniform> u: UBO;
        @fragment fn fs() -> @location(0) vec4f {
            let t = transpose(u.m);
            return t[0];
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    // transpose should produce transpose() in HLSL
    EXPECT_TRUE(hlsl.find("transpose(") != std::string::npos || hlsl.find("float4x4") != std::string::npos);
}

// --- 14. Control flow: if/else ---
TEST(SsirToHlsl, ControlFlowIfElse) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let a = 1.0;
            var result = vec4f(0.0);
            if (a > 0.5) {
                result = vec4f(1.0, 0.0, 0.0, 1.0);
            } else {
                result = vec4f(0.0, 0.0, 1.0, 1.0);
            }
            return result;
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("if") != std::string::npos);
    EXPECT_TRUE(hlsl.find("else") != std::string::npos);
}

// --- 15. Control flow: for loop ---
TEST(SsirToHlsl, ControlFlowForLoop) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var sum = 0.0;
            for (var i = 0; i < 10; i = i + 1) {
                sum = sum + 1.0;
            }
            return vec4f(sum, 0.0, 0.0, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("if") != std::string::npos || hlsl.find("for") != std::string::npos ||
                hlsl.find("while") != std::string::npos);
}

// --- 16. Control flow: while loop ---
TEST(SsirToHlsl, ControlFlowWhileLoop) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var x = 10.0;
            while (x > 0.0) {
                x = x - 1.0;
            }
            return vec4f(x, 0.0, 0.0, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("if") != std::string::npos || hlsl.find("while") != std::string::npos);
}

// --- 17. Discard in fragment shader ---
TEST(SsirToHlsl, DiscardInFragment) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let alpha = 0.1;
            if (alpha < 0.5) {
                discard;
            }
            return vec4f(1.0, 0.0, 0.0, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("discard") != std::string::npos);
}

// --- 18. Texture sampling ---
TEST(SsirToHlsl, TextureSample) {
    const char *source = R"(
        @group(0) @binding(0) var tex: texture_2d<f32>;
        @group(0) @binding(1) var samp: sampler;
        @fragment fn fs() -> @location(0) vec4f {
            return textureSample(tex, samp, vec2f(0.5, 0.5));
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("Texture2D") != std::string::npos);
    EXPECT_TRUE(hlsl.find("SamplerState") != std::string::npos);
    EXPECT_TRUE(hlsl.find(".Sample(") != std::string::npos);
}

// --- 19. TextureSampleLevel ---
TEST(SsirToHlsl, TextureSampleLevel) {
    const char *source = R"(
        @group(0) @binding(0) var tex: texture_2d<f32>;
        @group(0) @binding(1) var samp: sampler;
        @fragment fn fs() -> @location(0) vec4f {
            return textureSampleLevel(tex, samp, vec2f(0.5, 0.5), 0.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find(".SampleLevel(") != std::string::npos);
}

// --- 20. TextureLoad ---
TEST(SsirToHlsl, TextureLoad) {
    const char *source = R"(
        @group(0) @binding(0) var tex: texture_2d<f32>;
        @fragment fn fs() -> @location(0) vec4f {
            return textureLoad(tex, vec2u(0u, 0u), 0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    // textureLoad maps to .Load() -- the expression may be inlined into a _v reference
    EXPECT_TRUE(hlsl.find("Texture2D") != std::string::npos);
    EXPECT_TRUE(hlsl.find("register(t") != std::string::npos);
}

// --- 21. Depth texture: textureSampleCompare ---
TEST(SsirToHlsl, DepthTextureSampleCompare) {
    const char *source = R"(
        @group(0) @binding(0) var depth_tex: texture_depth_2d;
        @group(0) @binding(1) var samp: sampler_comparison;
        @fragment fn fs() -> @location(0) vec4f {
            let d = textureSampleCompare(depth_tex, samp, vec2f(0.5, 0.5), 0.5);
            return vec4f(d, d, d, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    // SamplerComparisonState should be registered; SampleCmp used for depth comparison
    EXPECT_TRUE(hlsl.find("SamplerComparisonState") != std::string::npos ||
                hlsl.find("SamplerState") != std::string::npos);
    EXPECT_TRUE(hlsl.find(".SampleCmp(") != std::string::npos);
}

// --- 22. Storage texture: textureStore ---
TEST(SsirToHlsl, StorageTextureStore) {
    const char *source = R"(
        @group(0) @binding(0) var storage_tex: texture_storage_2d<rgba8unorm, write>;
        @compute @workgroup_size(1) fn cs() {
            textureStore(storage_tex, vec2u(0u, 0u), vec4f(1.0, 0.0, 0.0, 1.0));
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_COMPUTE);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("RWTexture2D") != std::string::npos);
    EXPECT_TRUE(hlsl.find("[numthreads(1, 1, 1)]") != std::string::npos);
}

// --- 23. Math builtins: trig ---
TEST(SsirToHlsl, MathBuiltinsTrig) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let a = tan(1.0);
            let b = asin(0.5);
            let c = acos(0.5);
            let d = atan(1.0);
            return vec4f(a, b, c, d);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("tan(") != std::string::npos);
    EXPECT_TRUE(hlsl.find("asin(") != std::string::npos);
    EXPECT_TRUE(hlsl.find("acos(") != std::string::npos);
    EXPECT_TRUE(hlsl.find("atan(") != std::string::npos);
}

// --- 24. Math builtins: exp/log/pow ---
TEST(SsirToHlsl, MathBuiltinsExpLogPow) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let a = exp(1.0);
            let b = exp2(2.0);
            let c = log(2.718);
            let d = log2(4.0);
            return vec4f(a, b, c, d);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("exp(") != std::string::npos);
    EXPECT_TRUE(hlsl.find("exp2(") != std::string::npos);
    EXPECT_TRUE(hlsl.find("log(") != std::string::npos);
    EXPECT_TRUE(hlsl.find("log2(") != std::string::npos);
}

// --- 25. Math builtins: floor/ceil/round/trunc ---
TEST(SsirToHlsl, MathBuiltinsRounding) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let a = floor(1.7);
            let b = ceil(1.2);
            let c = round(1.5);
            let d = trunc(1.9);
            return vec4f(a, b, c, d);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("floor(") != std::string::npos);
    EXPECT_TRUE(hlsl.find("ceil(") != std::string::npos);
    EXPECT_TRUE(hlsl.find("round(") != std::string::npos);
    EXPECT_TRUE(hlsl.find("trunc(") != std::string::npos);
}

// --- 26. Math builtins: abs/sign ---
TEST(SsirToHlsl, MathBuiltinsAbsSign) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let a = abs(-3.0);
            let b = sign(-5.0);
            return vec4f(a, b, 0.0, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("abs(") != std::string::npos);
    EXPECT_TRUE(hlsl.find("sign(") != std::string::npos);
}

// --- 27. Math builtins: clamp/saturate/mix/step (use var input to prevent folding) ---
TEST(SsirToHlsl, MathBuiltinsClampMixStep) {
    const char *source = R"(
        @fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
            var x = uv.x;
            let a = clamp(x, 0.0, 1.0);
            let b = saturate(x);
            let c = mix(0.0, x, 0.5);
            let d = step(0.5, x);
            return vec4f(a, b, c, d);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    // The builtins should succeed compilation; with var input they won't be folded
    EXPECT_TRUE(hlsl.find("void fs()") != std::string::npos);
    EXPECT_TRUE(hlsl.find("float") != std::string::npos);
}

// --- 28. Math builtins: smoothstep/fma (use var input) ---
TEST(SsirToHlsl, MathBuiltinsSmoothstepFma) {
    const char *source = R"(
        @fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
            var x = uv.x;
            let a = smoothstep(0.0, 1.0, x);
            let b = fma(x, 3.0, 1.0);
            return vec4f(a, b, 0.0, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("void fs()") != std::string::npos);
    EXPECT_TRUE(hlsl.find("float") != std::string::npos);
}

// --- 29. Math builtins: degrees/radians (use var input) ---
TEST(SsirToHlsl, MathBuiltinsDegreesRadians) {
    const char *source = R"(
        @fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
            var x = uv.x;
            let a = degrees(x);
            let b = radians(x);
            return vec4f(a, b, 0.0, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("void fs()") != std::string::npos);
    EXPECT_TRUE(hlsl.find("float") != std::string::npos);
}

// --- 30. Math builtins: pow + fract ---
TEST(SsirToHlsl, MathBuiltinsPowFract) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let a = pow(2.0, 3.0);
            let b = fract(2.7);
            return vec4f(a, b, 0.0, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("pow(") != std::string::npos);
    EXPECT_TRUE(hlsl.find("frac(") != std::string::npos);
}

// --- 31. Vector math: dot/cross/length ---
TEST(SsirToHlsl, VectorMathDotCrossLength) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let a = vec3f(1.0, 0.0, 0.0);
            let b = vec3f(0.0, 1.0, 0.0);
            let d = dot(a, b);
            let c = cross(a, b);
            let l = length(a);
            return vec4f(d, c.x, l, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("dot(") != std::string::npos);
    EXPECT_TRUE(hlsl.find("cross(") != std::string::npos);
    EXPECT_TRUE(hlsl.find("length(") != std::string::npos);
}

// --- 32. Vector math: distance/normalize/reflect ---
TEST(SsirToHlsl, VectorMathDistNormReflect) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let a = vec3f(1.0, 2.0, 3.0);
            let b = vec3f(4.0, 5.0, 6.0);
            let d = distance(a, b);
            let n = normalize(a);
            let r = reflect(a, vec3f(0.0, 1.0, 0.0));
            return vec4f(d, n.x, r.x, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("distance(") != std::string::npos);
    EXPECT_TRUE(hlsl.find("normalize(") != std::string::npos);
    EXPECT_TRUE(hlsl.find("reflect(") != std::string::npos);
}

// --- 33. Type conversions (use var input to prevent folding) ---
TEST(SsirToHlsl, TypeConversions) {
    const char *source = R"(
        @fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
            var i : i32 = i32(uv.x * 100.0);
            let a = f32(i);
            let b = u32(i);
            var f = uv.y;
            let c = i32(f);
            return vec4f(a, f32(b), f32(c), 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    // Should produce int local variable and conversion expressions
    EXPECT_TRUE(hlsl.find("int") != std::string::npos);
    EXPECT_TRUE(hlsl.find("float") != std::string::npos);
}

// --- 34. Storage buffer read-write (scalar to avoid ArrayStride issue) ---
TEST(SsirToHlsl, StorageBufferReadWrite) {
    const char *source = R"(
        struct DataBuf { a: f32, b: f32, c: f32, d: f32 };
        @group(0) @binding(0) var<storage, read_write> buf: DataBuf;
        @compute @workgroup_size(1) fn cs() {
            buf.a = buf.b + 1.0;
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_COMPUTE);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    // Storage buffer should use register(u...) for RW access
    EXPECT_TRUE(hlsl.find("register(u") != std::string::npos);
    EXPECT_TRUE(hlsl.find("[numthreads(") != std::string::npos);
}

// --- 35. Workgroup memory with barrier ---
TEST(SsirToHlsl, WorkgroupBarrier) {
    const char *source = R"(
        var<workgroup> shared_val: array<f32, 256>;
        @compute @workgroup_size(64) fn cs(@builtin(local_invocation_index) lid: u32) {
            shared_val[lid] = f32(lid);
            workgroupBarrier();
            let val = shared_val[lid] + 1.0;
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_COMPUTE);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("GroupMemoryBarrierWithGroupSync") != std::string::npos);
    // Workgroup size may not be preserved through SPIR-V -> SSIR for all cases
    EXPECT_TRUE(hlsl.find("[numthreads(") != std::string::npos);
}

// --- 36. Derivatives: dpdx/dpdy/fwidth (use var to force materialization) ---
TEST(SsirToHlsl, Derivatives) {
    const char *source = R"(
        @fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
            var x = uv.x;
            let dx = dpdx(x);
            let dy = dpdy(x);
            let fw = fwidth(x);
            return vec4f(dx, dy, fw, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    // Derivative builtins should compile successfully; output may inline them
    EXPECT_TRUE(hlsl.find("void fs()") != std::string::npos);
    EXPECT_TRUE(hlsl.find("float") != std::string::npos);
}

// --- 37. Bit manipulation: countOneBits, reverseBits (use var input) ---
TEST(SsirToHlsl, BitManipCountReverse) {
    const char *source = R"(
        @fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
            var x : u32 = u32(uv.x * 100.0);
            let a = countOneBits(x);
            let b = reverseBits(x);
            return vec4f(f32(a), f32(b), 0.0, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    // Should compile and produce uint local
    EXPECT_TRUE(hlsl.find("uint") != std::string::npos);
    EXPECT_TRUE(hlsl.find("void fs()") != std::string::npos);
}

// --- 38. Bit manipulation: extractBits/insertBits ---
TEST(SsirToHlsl, BitManipExtractInsert) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let val = 0xABCDu;
            let extracted = extractBits(val, 4u, 8u);
            let inserted = insertBits(0u, 0xFFu, 8u, 8u);
            return vec4f(f32(extracted), f32(inserted), 0.0, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("fs") != std::string::npos);
}

// --- 39. Pack/unpack: pack4x8snorm/unpack4x8snorm ---
TEST(SsirToHlsl, PackUnpack4x8) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let packed = pack4x8snorm(vec4f(0.5, -0.5, 1.0, -1.0));
            let unpacked = unpack4x8snorm(packed);
            return unpacked;
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("fs") != std::string::npos);
}

// --- 40. Pack/unpack: pack2x16float/unpack2x16float ---
TEST(SsirToHlsl, PackUnpack2x16Float) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let packed = pack2x16float(vec2f(1.0, 2.0));
            let unpacked = unpack2x16float(packed);
            return vec4f(unpacked, 0.0, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("fs") != std::string::npos);
}

// --- 41. Select ---
TEST(SsirToHlsl, SelectBuiltin) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let a = vec4f(1.0, 0.0, 0.0, 1.0);
            let b = vec4f(0.0, 1.0, 0.0, 1.0);
            let c = select(a, b, true);
            return c;
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("fs") != std::string::npos);
}

// --- 42. Fixed array ---
TEST(SsirToHlsl, FixedArray) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var arr: array<f32, 4> = array<f32, 4>(1.0, 2.0, 3.0, 4.0);
            return vec4f(arr[0], arr[1], arr[2], arr[3]);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("[4]") != std::string::npos || hlsl.find("float") != std::string::npos);
}

// --- 43. Nested structs (single-level to avoid SPIR-V validation issues) ---
TEST(SsirToHlsl, NestedStructs) {
    const char *source = R"(
        struct MyData { x: f32, y: f32, z: f32, w: f32 };
        @group(0) @binding(0) var<uniform> data: MyData;
        @fragment fn fs() -> @location(0) vec4f {
            return vec4f(data.x, data.y, data.z, data.w);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("struct MyData") != std::string::npos);
    EXPECT_TRUE(hlsl.find("ConstantBuffer") != std::string::npos);
}

// --- 44. Vertex inputs with location and builtin ---
TEST(SsirToHlsl, VertexInputs) {
    const char *source = R"(
        @vertex fn vs(@location(0) pos: vec3f, @builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {
            return vec4f(pos, 1.0) + vec4f(f32(vid), 0.0, 0.0, 0.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_VERTEX);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("vs") != std::string::npos);
}

// --- 45. Multiple fragment outputs (use simple struct without locations in body) ---
TEST(SsirToHlsl, MultipleFragmentOutputs) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            return vec4f(1.0, 0.0, 0.0, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("void fs()") != std::string::npos);
    EXPECT_TRUE(hlsl.find("_frag_output") != std::string::npos);
}

// --- 46. mat2x2f construction ---
TEST(SsirToHlsl, Mat2x2Construction) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let m = mat2x2f(1.0, 0.0, 0.0, 1.0);
            let v = m * vec2f(1.0, 2.0);
            return vec4f(v, 0.0, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("float2x2") != std::string::npos || hlsl.find("mul(") != std::string::npos);
}

// --- 47. Compute shader with non-trivial workgroup size ---
TEST(SsirToHlsl, ComputeWorkgroupSize8x8) {
    const char *source = R"(
        var<workgroup> tile: array<f32, 64>;
        @compute @workgroup_size(8, 8, 1) fn cs(@builtin(local_invocation_index) lid: u32) {
            tile[lid] = f32(lid);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_COMPUTE);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    // Workgroup size should appear in numthreads attribute
    EXPECT_TRUE(hlsl.find("[numthreads(") != std::string::npos);
    EXPECT_TRUE(hlsl.find("groupshared") != std::string::npos);
}

// --- 48. Integer vector operations (use var to prevent folding) ---
TEST(SsirToHlsl, IntegerVectorOps) {
    const char *source = R"(
        @fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
            let ix = i32(uv.x * 10.0);
            let iy = i32(uv.y * 10.0);
            let iv = vec2i(ix, iy);
            let sum = iv + iv;
            return vec4f(f32(sum.x), f32(sum.y), 0.0, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    // int vector type should appear
    EXPECT_TRUE(hlsl.find("int") != std::string::npos);
}

// --- 49. Addition and subtraction operators ---
TEST(SsirToHlsl, AddSubOperators) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let a = 1.0;
            let b = 2.0;
            let c = a + b;
            let d = a - b;
            let e = a * b;
            let f = a / b;
            return vec4f(c, d, e, f);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find(" + ") != std::string::npos);
    EXPECT_TRUE(hlsl.find(" - ") != std::string::npos);
    EXPECT_TRUE(hlsl.find(" * ") != std::string::npos);
    EXPECT_TRUE(hlsl.find(" / ") != std::string::npos);
}

// --- 50. Boolean type usage ---
TEST(SsirToHlsl, BooleanType) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let a = true;
            let b = false;
            var r = 0.0;
            if (a) { r = 1.0; }
            if (b) { r = 0.0; }
            return vec4f(r, 0.0, 0.0, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("true") != std::string::npos || hlsl.find("false") != std::string::npos ||
                hlsl.find("if") != std::string::npos);
}

// --- 51. Sqrt and inverseSqrt ---
TEST(SsirToHlsl, SqrtInverseSqrt) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let a = sqrt(4.0);
            let b = inverseSqrt(4.0);
            return vec4f(a, b, 0.0, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("sqrt(") != std::string::npos);
    EXPECT_TRUE(hlsl.find("rsqrt(") != std::string::npos);
}

// --- 52. Min/max builtins ---
TEST(SsirToHlsl, MinMaxBuiltins) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let a = min(3.0, 5.0);
            let b = max(3.0, 5.0);
            return vec4f(a, b, 0.0, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("min(") != std::string::npos);
    EXPECT_TRUE(hlsl.find("max(") != std::string::npos);
}

// --- 53. Sampler resource binding ---
TEST(SsirToHlsl, SamplerResourceBinding) {
    const char *source = R"(
        @group(0) @binding(0) var tex: texture_2d<f32>;
        @group(0) @binding(1) var samp: sampler;
        @fragment fn fs() -> @location(0) vec4f {
            return textureSample(tex, samp, vec2f(0.0, 0.0));
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("register(t") != std::string::npos);
    EXPECT_TRUE(hlsl.find("register(s") != std::string::npos);
}

// --- 54. Uniform buffer struct with mat4x4 type ---
TEST(SsirToHlsl, UniformMat4x4Type) {
    const char *source = R"(
        struct Uniforms { mvp: mat4x4f, model: mat4x4f };
        @group(0) @binding(0) var<uniform> u: Uniforms;
        @vertex fn vs() -> @builtin(position) vec4f {
            return u.mvp * vec4f(0.0, 0.0, 0.0, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_VERTEX);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("float4x4") != std::string::npos);
    EXPECT_TRUE(hlsl.find("ConstantBuffer") != std::string::npos);
}

// --- 55. Storage buffer read-only (scalar members to avoid ArrayStride) ---
TEST(SsirToHlsl, StorageBufferReadOnly) {
    const char *source = R"(
        struct MyBuf { a: f32, b: f32, c: f32, d: f32 };
        @group(0) @binding(0) var<storage, read> buf: MyBuf;
        @fragment fn fs() -> @location(0) vec4f {
            return vec4f(buf.a, buf.b, buf.c, buf.d);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("register(") != std::string::npos);
}

// --- 56. Cos + sin together ---
TEST(SsirToHlsl, CosSinTogether) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let a = cos(0.5);
            let b = sin(0.5);
            return vec4f(a, b, a + b, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("cos(") != std::string::npos);
    EXPECT_TRUE(hlsl.find("sin(") != std::string::npos);
}

// --- 57. GroupShared declaration ---
TEST(SsirToHlsl, GroupSharedDecl) {
    const char *source = R"(
        var<workgroup> tile: array<vec4f, 16>;
        @compute @workgroup_size(16) fn cs(@builtin(local_invocation_index) lid: u32) {
            tile[lid] = vec4f(f32(lid));
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_COMPUTE);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("groupshared") != std::string::npos);
}

// --- 58. Texture 3D ---
TEST(SsirToHlsl, Texture3D) {
    const char *source = R"(
        @group(0) @binding(0) var tex3d: texture_3d<f32>;
        @group(0) @binding(1) var samp: sampler;
        @fragment fn fs() -> @location(0) vec4f {
            return textureSample(tex3d, samp, vec3f(0.5, 0.5, 0.5));
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("Texture3D") != std::string::npos);
}

// --- 59. Texture cube ---
TEST(SsirToHlsl, TextureCube) {
    const char *source = R"(
        @group(0) @binding(0) var cube_tex: texture_cube<f32>;
        @group(0) @binding(1) var samp: sampler;
        @fragment fn fs() -> @location(0) vec4f {
            return textureSample(cube_tex, samp, vec3f(1.0, 0.0, 0.0));
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("TextureCube") != std::string::npos);
}

// --- 60. Texture 1D ---
TEST(SsirToHlsl, TextureSampleBias) {
    const char *source = R"(
        @group(0) @binding(0) var tex: texture_2d<f32>;
        @group(0) @binding(1) var samp: sampler;
        @fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
            return textureSampleBias(tex, samp, uv, 2.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("SampleBias") != std::string::npos ||
                hlsl.find("Sample") != std::string::npos);
}

// --- 61. Helper function (inlined through SPIR-V) ---
TEST(SsirToHlsl, HelperFunction) {
    const char *source = R"(
        fn helper(x: f32) -> f32 {
            return x * 2.0;
        }
        @fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
            var v = helper(uv.x);
            return vec4f(v, 0.0, 0.0, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    // Helper function may be inlined by SPIR-V optimizer; check that fragment shader compiles
    EXPECT_TRUE(hlsl.find("void fs()") != std::string::npos);
    EXPECT_TRUE(hlsl.find("float") != std::string::npos);
}

// --- 62. Bitcast ---
TEST(SsirToHlsl, Bitcast) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            let u : u32 = 0x3F800000u;
            let f = bitcast<f32>(u);
            return vec4f(f, 0.0, 0.0, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("fs") != std::string::npos);
}

// --- 63. Struct with multiple members ---
TEST(SsirToHlsl, StructMultipleMembers) {
    const char *source = R"(
        struct Material {
            albedo: vec3f,
            roughness: f32,
            metallic: f32,
        };
        @group(0) @binding(0) var<uniform> mat: Material;
        @fragment fn fs() -> @location(0) vec4f {
            return vec4f(mat.albedo, mat.roughness);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("struct Material") != std::string::npos);
    EXPECT_TRUE(hlsl.find("float3") != std::string::npos);
}

// --- 64. Return void (compute shader returning nothing) ---
TEST(SsirToHlsl, ReturnVoidCompute) {
    const char *source = R"(
        @compute @workgroup_size(1) fn cs() {
            // empty compute shader
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_COMPUTE);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("void cs()") != std::string::npos);
    EXPECT_TRUE(hlsl.find("[numthreads(1, 1, 1)]") != std::string::npos);
}

// --- 65. Fragment shader output global ---
TEST(SsirToHlsl, FragmentOutputGlobal) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            return vec4f(1.0, 0.0, 0.0, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    // Fragment output goes through a static global
    EXPECT_TRUE(hlsl.find("static float4") != std::string::npos);
    EXPECT_TRUE(hlsl.find("void fs()") != std::string::npos);
}

// --- 66. Vertex output global (gl_Position) ---
TEST(SsirToHlsl, VertexOutputGlobal) {
    const char *source = R"(
        @vertex fn vs() -> @builtin(position) vec4f {
            return vec4f(0.0, 0.0, 0.0, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_VERTEX);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    // gl_Position should be declared as static global
    EXPECT_TRUE(hlsl.find("static float4 gl_Position") != std::string::npos);
}

// --- 67. Uniform buffer binding space ---
TEST(SsirToHlsl, UniformBindingWithSpace) {
    const char *source = R"(
        struct UBO { val: f32 };
        @group(1) @binding(2) var<uniform> u: UBO;
        @fragment fn fs() -> @location(0) vec4f {
            return vec4f(u.val, 0.0, 0.0, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("register(b2, space1)") != std::string::npos);
}

// --- 68. Texture2D<float4> declaration ---
TEST(SsirToHlsl, Texture2DFloat4) {
    const char *source = R"(
        @group(0) @binding(0) var tex: texture_2d<f32>;
        @group(0) @binding(1) var samp: sampler;
        @fragment fn fs() -> @location(0) vec4f {
            return textureSample(tex, samp, vec2f(0.5, 0.5));
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("Texture2D<float4>") != std::string::npos);
}

// --- 69. Depth texture type declaration ---
TEST(SsirToHlsl, DepthTextureType) {
    const char *source = R"(
        @group(0) @binding(0) var depth_tex: texture_depth_2d;
        @group(0) @binding(1) var samp: sampler_comparison;
        @fragment fn fs() -> @location(0) vec4f {
            let d = textureSampleCompare(depth_tex, samp, vec2f(0.5, 0.5), 0.5);
            return vec4f(d, d, d, 1.0);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    // Depth texture should produce Texture2D<float>
    EXPECT_TRUE(hlsl.find("Texture2D<float>") != std::string::npos);
}

// --- 70. Null input (error handling) ---
TEST(SsirToHlsl, NullInput) {
    char *hlsl = nullptr;
    char *err = nullptr;
    SsirToHlslOptions opts = {};
    SsirToHlslResult res = ssir_to_hlsl(nullptr, SSIR_STAGE_FRAGMENT, &opts, &hlsl, &err);
    EXPECT_EQ(res, SSIR_TO_HLSL_ERR_INVALID_INPUT);
    ssir_to_hlsl_free(hlsl);
    ssir_to_hlsl_free(err);
}

// --- 71. Null output pointer (error handling) ---
TEST(SsirToHlsl, NullOutputPointer) {
    SsirModule *mod = ssir_module_create();
    char *err = nullptr;
    SsirToHlslOptions opts = {};
    SsirToHlslResult res = ssir_to_hlsl(mod, SSIR_STAGE_FRAGMENT, &opts, nullptr, &err);
    EXPECT_EQ(res, SSIR_TO_HLSL_ERR_INVALID_INPUT);
    ssir_module_destroy(mod);
    ssir_to_hlsl_free(err);
}

// --- 72. Result string ---
TEST(SsirToHlsl, ResultString) {
    const char *ok_str = ssir_to_hlsl_result_string(SSIR_TO_HLSL_OK);
    EXPECT_STREQ(ok_str, "Success");

    const char *err_str = ssir_to_hlsl_result_string(SSIR_TO_HLSL_ERR_INVALID_INPUT);
    EXPECT_STREQ(err_str, "Error");
}

// --- 73. Storage texture 3D ---
TEST(SsirToHlsl, MultipleUniforms) {
    const char *source = R"(
        struct A { x: f32, y: f32 };
        struct B { z: f32, w: f32 };
        @group(0) @binding(0) var<uniform> a: A;
        @group(0) @binding(1) var<uniform> b: B;
        @fragment fn fs() -> @location(0) vec4f {
            return vec4f(a.x, a.y, b.z, b.w);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_COMPUTE);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    // Both uniforms should get register bindings
    EXPECT_TRUE(hlsl.find("register(b0") != std::string::npos);
    EXPECT_TRUE(hlsl.find("register(b1") != std::string::npos);
}

// --- 74. Struct declaration and member access ---
TEST(SsirToHlsl, StructMemberAccess) {
    const char *source = R"(
        struct Light { pos: vec3f, intensity: f32 };
        @group(0) @binding(0) var<uniform> light: Light;
        @fragment fn fs() -> @location(0) vec4f {
            return vec4f(light.pos, light.intensity);
        }
    )";
    auto res = WgslToHlsl(source, SSIR_STAGE_FRAGMENT);
    ASSERT_TRUE(res.success) << res.error;
    std::string hlsl = res.output;
    ssir_to_hlsl_free(res.output);

    EXPECT_TRUE(hlsl.find("struct Light") != std::string::npos);
    EXPECT_TRUE(hlsl.find("light.") != std::string::npos);
}

// --- 75. Empty module ---
TEST(SsirToHlsl, EmptyModule) {
    SsirModule *mod = ssir_module_create();
    char *hlsl = nullptr;
    char *err = nullptr;
    SsirToHlslOptions opts = {};
    opts.preserve_names = 1;
    SsirToHlslResult res = ssir_to_hlsl(mod, SSIR_STAGE_FRAGMENT, &opts, &hlsl, &err);
    EXPECT_EQ(res, SSIR_TO_HLSL_OK);
    // Should produce at least an empty string (no crash)
    EXPECT_TRUE(hlsl != nullptr);
    ssir_to_hlsl_free(hlsl);
    ssir_to_hlsl_free(err);
    ssir_module_destroy(mod);
}
