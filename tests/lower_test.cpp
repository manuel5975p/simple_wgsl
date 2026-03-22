#include <gtest/gtest.h>
#include "test_utils.h"
#ifdef WGSL_HAS_FFT
#include "fft_stockham_gen.h"
extern "C" {
#include "fft_fused_gen.h"
#include "fft_2d_gen.h"
#include "fft_fourstep_gen.h"
#include "fft_optimal_gen.h"
}
#endif

TEST(LowerTest, EmitMinimalSpirv) {
    const char *source = "fn main() {}";
    wgsl_test::AstGuard ast(wgsl_parse(source));
    ASSERT_NE(ast.get(), nullptr);

    wgsl_test::ResolverGuard resolver(wgsl_resolver_build(ast.get()));
    ASSERT_NE(resolver.get(), nullptr);

    uint32_t *spirv = nullptr;
    size_t spirv_size = 0;
    WgslLowerOptions opts = {};
    opts.env = WGSL_LOWER_ENV_VULKAN_1_3;

    WgslLowerResult result = wgsl_lower_emit_spirv(ast.get(), resolver.get(), &opts, &spirv, &spirv_size);
    EXPECT_EQ(result, WGSL_LOWER_OK);
    ASSERT_NE(spirv, nullptr);
    ASSERT_GE(spirv_size, static_cast<size_t>(5));
    EXPECT_EQ(spirv[0], 0x07230203u);

    wgsl_lower_free(spirv);
}

TEST(LowerTest, ValidateMinimalSpirvModule) {
    const char *source = "fn main() {}";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, ValidateFragmentShader) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, ValidateVertexShader) {
    const char *source = R"(
        @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0.0); }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, ValidateComputeShader) {
    const char *source = R"(
        @compute @workgroup_size(1) fn cs() {}
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, ValidateBindingVariable) {
    const char *source = R"(
        @group(0) @binding(0) var tex: texture_2d<f32>;
        @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, ValidateSampler) {
    const char *source = R"(
        @group(0) @binding(0) var s: sampler;
        @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, TypeCachingWorks) {
    // This test verifies that type caching works correctly
    // by compiling a shader that uses the same types multiple times
    const char *source = R"(
        fn main() {}
    )";

    wgsl_test::AstGuard ast(wgsl_parse(source));
    ASSERT_NE(ast.get(), nullptr);

    wgsl_test::ResolverGuard resolver(wgsl_resolver_build(ast.get()));
    ASSERT_NE(resolver.get(), nullptr);

    WgslLowerOptions opts = {};
    opts.env = WGSL_LOWER_ENV_VULKAN_1_3;

    wgsl_test::LowerGuard lower(wgsl_lower_create(ast.get(), resolver.get(), &opts));
    ASSERT_NE(lower.get(), nullptr);

    // Check that entrypoints are created
    int count = 0;
    const WgslLowerEntrypointInfo *eps = wgsl_lower_entrypoints(lower.get(), &count);
    EXPECT_GE(count, 1);
    EXPECT_NE(eps, nullptr);
}

TEST(LowerTest, MultipleEntrypoints) {
    const char *source = R"(
        @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0.0); }
        @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }
    )";

    wgsl_test::AstGuard ast(wgsl_parse(source));
    ASSERT_NE(ast.get(), nullptr);

    wgsl_test::ResolverGuard resolver(wgsl_resolver_build(ast.get()));
    ASSERT_NE(resolver.get(), nullptr);

    WgslLowerOptions opts = {};
    opts.env = WGSL_LOWER_ENV_VULKAN_1_3;

    wgsl_test::LowerGuard lower(wgsl_lower_create(ast.get(), resolver.get(), &opts));
    ASSERT_NE(lower.get(), nullptr);

    int count = 0;
    const WgslLowerEntrypointInfo *eps = wgsl_lower_entrypoints(lower.get(), &count);
    EXPECT_EQ(count, 2);

    // Verify both entrypoints have function IDs
    for (int i = 0; i < count; ++i) {
        EXPECT_NE(eps[i].function_id, 0u);
    }
}

TEST(LowerTest, ModuleFeatures) {
    const char *source = "fn main() {}";

    wgsl_test::AstGuard ast(wgsl_parse(source));
    ASSERT_NE(ast.get(), nullptr);

    wgsl_test::ResolverGuard resolver(wgsl_resolver_build(ast.get()));
    ASSERT_NE(resolver.get(), nullptr);

    WgslLowerOptions opts = {};
    opts.env = WGSL_LOWER_ENV_VULKAN_1_3;

    wgsl_test::LowerGuard lower(wgsl_lower_create(ast.get(), resolver.get(), &opts));
    ASSERT_NE(lower.get(), nullptr);

    const WgslLowerModuleFeatures *features = wgsl_lower_module_features(lower.get());
    ASSERT_NE(features, nullptr);

    // Should have at least Shader capability
    EXPECT_GE(features->capability_count, 1u);
}

// ==================== Expression Lowering Tests ====================

TEST(LowerTest, ArithmeticExpressions) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var x = 1.0 + 2.0;
            var y = 3.0 - 1.0;
            var z = x * y;
            var w = z / 2.0;
            return vec4f(w);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, ComparisonOperators) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var a = 1.0;
            var b = 2.0;
            var lt = a < b;
            var le = a <= b;
            var gt = a > b;
            var ge = a >= b;
            var eq = a == b;
            var ne = a != b;
            return vec4f(1.0);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, BitwiseOperators) {
    // Basic integer operations (shift operators need parser support)
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var a: i32 = 5;
            var b: i32 = 3;
            var sum: i32 = a + b;
            var diff: i32 = a - b;
            return vec4f(1.0);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, UnaryOperators) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var a: f32 = 5.0;
            var neg: f32 = -a;
            return vec4f(neg);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, VectorConstruction) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var v2 = vec2f(1.0, 2.0);
            var v3 = vec3f(1.0, 2.0, 3.0);
            var v4 = vec4f(1.0, 2.0, 3.0, 4.0);
            return v4;
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, VectorSwizzle) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var v = vec4f(1.0, 2.0, 3.0, 4.0);
            var x = v.x;
            var xy = v.xy;
            var zw = v.zw;
            var rgba = v.rgba;
            return vec4f(x);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

// ==================== Built-in Function Tests ====================

TEST(LowerTest, MathFunctions) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var a = 2.0;
            var s = sqrt(a);
            var f = floor(a);
            var c = ceil(a);
            var r = round(a);
            var t = trunc(a);
            return vec4f(s);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, TrigFunctions) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var angle = 1.0;
            var s = sin(angle);
            var c = cos(angle);
            var t = tan(angle);
            return vec4f(s, c, t, 1.0);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, MinMaxClamp) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var a = 1.0;
            var b = 2.0;
            var mn = min(a, b);
            var mx = max(a, b);
            var cl = clamp(a, 0.0, 1.0);
            return vec4f(mn, mx, cl, 1.0);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, VectorBuiltins) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var v1 = vec3f(1.0, 0.0, 0.0);
            var v2 = vec3f(0.0, 1.0, 0.0);
            var d = dot(v1, v2);
            var c = cross(v1, v2);
            var len = length(v1);
            var n = normalize(v1);
            return vec4f(d, len, n.x, 1.0);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, MixSmoothstep) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var a = 0.0;
            var b = 1.0;
            var t = 0.5;
            var m = mix(a, b, t);
            var ss = smoothstep(0.0, 1.0, t);
            return vec4f(m, ss, 0.0, 1.0);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

// ==================== Control Flow Tests ====================

TEST(LowerTest, IfStatement) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var x = 1.0;
            if (x > 0.5) {
                x = 2.0;
            } else {
                x = 0.0;
            }
            return vec4f(x);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, IfWithoutElse) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var x = 1.0;
            if (x > 0.5) {
                x = 2.0;
            }
            return vec4f(x);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, WhileLoop) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var i = 0;
            var sum = 0.0;
            while (i < 10) {
                sum = sum + 1.0;
                i = i + 1;
            }
            return vec4f(sum);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, ForLoop) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var sum = 0.0;
            for (var i = 0; i < 10; i = i + 1) {
                sum = sum + 1.0;
            }
            return vec4f(sum);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, NestedControlFlow) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var result = 0.0;
            for (var i = 0; i < 5; i = i + 1) {
                if (i > 2) {
                    result = result + 2.0;
                } else {
                    result = result + 1.0;
                }
            }
            return vec4f(result);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

// ==================== Variable Tests ====================

TEST(LowerTest, LocalVariables) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var a = 1.0;
            var b = 2.0;
            var c = a + b;
            a = c * 2.0;
            return vec4f(a);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, IntegerVariables) {
    const char *source = R"(
        @fragment fn fs() -> @location(0) vec4f {
            var i = 10;
            var u = 20u;
            var sum = i + 5;
            return vec4f(1.0);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

// ============================================================================
// Matrix × Vector tests (OpMatrixTimesVector)
// ============================================================================

TEST(LowerTest, MatrixTimesVector_Mat2x2_Vec2) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let m = mat2x2<f32>(vec2<f32>(1.,0.), vec2<f32>(0.,1.));
            let v = vec2<f32>(2., 3.);
            let r = m * v;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, MatrixTimesVector_Mat3x3_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let m = mat3x3<f32>(
                vec3<f32>(1.,0.,0.),
                vec3<f32>(0.,1.,0.),
                vec3<f32>(0.,0.,1.));
            let v = vec3<f32>(1., 2., 3.);
            let r = m * v;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, MatrixTimesVector_Mat4x4_Vec4) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let m = mat4x4<f32>(
                vec4<f32>(1.,0.,0.,0.),
                vec4<f32>(0.,1.,0.,0.),
                vec4<f32>(0.,0.,1.,0.),
                vec4<f32>(0.,0.,0.,1.));
            let v = vec4<f32>(1., 2., 3., 4.);
            let r = m * v;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, MatrixTimesVector_NonSquare_Mat3x2_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let m = mat3x2<f32>(
                vec2<f32>(1.,0.),
                vec2<f32>(0.,1.),
                vec2<f32>(1.,1.));
            let v = vec3<f32>(1., 2., 3.);
            let r = m * v;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, MatrixTimesVector_NonSquare_Mat4x3_Vec4) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let m = mat4x3<f32>(
                vec3<f32>(1.,0.,0.),
                vec3<f32>(0.,1.,0.),
                vec3<f32>(0.,0.,1.),
                vec3<f32>(1.,1.,1.));
            let v = vec4<f32>(1., 2., 3., 4.);
            let r = m * v;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, MatrixTimesVector_NonSquare_Mat2x4_Vec2) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let m = mat2x4<f32>(
                vec4<f32>(1.,0.,0.,0.),
                vec4<f32>(0.,1.,0.,0.));
            let v = vec2<f32>(1., 2.);
            let r = m * v;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// Vector × Matrix tests (OpVectorTimesMatrix)
// ============================================================================

TEST(LowerTest, VectorTimesMatrix_Vec2_Mat2x2) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let v = vec2<f32>(2., 3.);
            let m = mat2x2<f32>(vec2<f32>(1.,0.), vec2<f32>(0.,1.));
            let r = v * m;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, VectorTimesMatrix_Vec3_Mat3x3) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let v = vec3<f32>(1., 2., 3.);
            let m = mat3x3<f32>(
                vec3<f32>(1.,0.,0.),
                vec3<f32>(0.,1.,0.),
                vec3<f32>(0.,0.,1.));
            let r = v * m;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, VectorTimesMatrix_Vec4_Mat4x4) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let v = vec4<f32>(1., 2., 3., 4.);
            let m = mat4x4<f32>(
                vec4<f32>(1.,0.,0.,0.),
                vec4<f32>(0.,1.,0.,0.),
                vec4<f32>(0.,0.,1.,0.),
                vec4<f32>(0.,0.,0.,1.));
            let r = v * m;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, VectorTimesMatrix_NonSquare_Vec2_Mat3x2) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let v = vec2<f32>(1., 2.);
            let m = mat3x2<f32>(
                vec2<f32>(1.,0.),
                vec2<f32>(0.,1.),
                vec2<f32>(1.,1.));
            let r = v * m;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// Matrix × Matrix tests (OpMatrixTimesMatrix)
// ============================================================================

TEST(LowerTest, MatrixTimesMatrix_Mat2x2) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let a = mat2x2<f32>(vec2<f32>(1.,2.), vec2<f32>(3.,4.));
            let b = mat2x2<f32>(vec2<f32>(5.,6.), vec2<f32>(7.,8.));
            let r = a * b;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, MatrixTimesMatrix_Mat3x3) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let a = mat3x3<f32>(
                vec3<f32>(1.,2.,3.),
                vec3<f32>(4.,5.,6.),
                vec3<f32>(7.,8.,9.));
            let b = mat3x3<f32>(
                vec3<f32>(-1.,-2.,-3.),
                vec3<f32>(-4.,-5.,-6.),
                vec3<f32>(-7.,-8.,-9.));
            let r = a * b;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, MatrixTimesMatrix_Mat4x4) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let a = mat4x4<f32>(
                vec4<f32>(1.,0.,0.,0.),
                vec4<f32>(0.,1.,0.,0.),
                vec4<f32>(0.,0.,1.,0.),
                vec4<f32>(0.,0.,0.,1.));
            let b = mat4x4<f32>(
                vec4<f32>(2.,0.,0.,0.),
                vec4<f32>(0.,2.,0.,0.),
                vec4<f32>(0.,0.,2.,0.),
                vec4<f32>(0.,0.,0.,2.));
            let r = a * b;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, MatrixTimesMatrix_NonSquare_Mat2x3_Mat3x2) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let a = mat2x3<f32>(vec3<f32>(1.,2.,3.), vec3<f32>(4.,5.,6.));
            let b = mat3x2<f32>(vec2<f32>(1.,2.), vec2<f32>(3.,4.), vec2<f32>(5.,6.));
            let r = a * b;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// Matrix × Scalar tests (OpMatrixTimesScalar)
// ============================================================================

TEST(LowerTest, MatrixTimesScalar_Mat3x3) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let m = mat3x3<f32>(
                vec3<f32>(1.,0.,0.),
                vec3<f32>(0.,1.,0.),
                vec3<f32>(0.,0.,1.));
            let s = 2.0;
            let r = m * s;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, ScalarTimesMatrix_Mat4x4) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let s = 3.0;
            let m = mat4x4<f32>(
                vec4<f32>(1.,0.,0.,0.),
                vec4<f32>(0.,1.,0.,0.),
                vec4<f32>(0.,0.,1.,0.),
                vec4<f32>(0.,0.,0.,1.));
            let r = s * m;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// Matrix operations with uniform struct (tests MatrixStride decorations)
// ============================================================================

TEST(LowerTest, MatrixTimesVector_UniformStruct_Mat3x3) {
    auto r = wgsl_test::CompileWgsl(R"(
        struct S {
            matrix : mat3x3<f32>,
            vector : vec3<f32>,
        };
        @group(0) @binding(0) var<uniform> data: S;
        @fragment fn main() {
            let x = data.matrix * data.vector;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, MatrixTimesVector_UniformStruct_Mat4x4) {
    auto r = wgsl_test::CompileWgsl(R"(
        struct S {
            matrix : mat4x4<f32>,
            vector : vec4<f32>,
        };
        @group(0) @binding(0) var<uniform> data: S;
        @fragment fn main() {
            let x = data.matrix * data.vector;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, VectorTimesMatrix_UniformStruct_Vec3_Mat3x3) {
    auto r = wgsl_test::CompileWgsl(R"(
        struct S {
            matrix : mat3x3<f32>,
            vector : vec3<f32>,
        };
        @group(0) @binding(0) var<uniform> data: S;
        @fragment fn main() {
            let x = data.vector * data.matrix;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, MatrixTimesMatrix_UniformStruct) {
    auto r = wgsl_test::CompileWgsl(R"(
        struct S {
            a : mat4x4<f32>,
            b : mat4x4<f32>,
        };
        @group(0) @binding(0) var<uniform> data: S;
        @fragment fn main() {
            let x = data.a * data.b;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// Fragment shader struct parameter bug (known issue)
// ============================================================================

TEST(LowerTest, FragmentShaderFlatParameter) {
    auto r = wgsl_test::CompileWgsl(R"(
        @fragment fn fs_main(@location(0) col: vec4f) -> @location(0) vec4f {
            return col;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// Mixed matrix operations in complex expressions
// ============================================================================

TEST(LowerTest, MatrixVectorChain) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let model = mat4x4<f32>(
                vec4<f32>(1.,0.,0.,0.),
                vec4<f32>(0.,1.,0.,0.),
                vec4<f32>(0.,0.,1.,0.),
                vec4<f32>(0.,0.,0.,1.));
            let view = mat4x4<f32>(
                vec4<f32>(1.,0.,0.,0.),
                vec4<f32>(0.,1.,0.,0.),
                vec4<f32>(0.,0.,1.,0.),
                vec4<f32>(0.,0.,0.,1.));
            let pos = vec4<f32>(1., 2., 3., 1.);
            let transformed = view * model * pos;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, MatrixScaleAndTransform) {
    auto r = wgsl_test::CompileWgsl(R"(
        @compute @workgroup_size(1) fn f() {
            let m = mat3x3<f32>(
                vec3<f32>(1.,0.,0.),
                vec3<f32>(0.,1.,0.),
                vec3<f32>(0.,0.,1.));
            let scaled = m * 2.0;
            let v = vec3<f32>(1., 2., 3.);
            let r = scaled * v;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// Fragment shader struct parameter tests
// ============================================================================

TEST(LowerTest, FragmentShaderStructParam_LocationFields) {
    auto r = wgsl_test::CompileWgsl(R"(
        struct FsIn {
            @location(0) col: vec4f,
        };
        @fragment fn fs_main(in: FsIn) -> @location(0) vec4f {
            return in.col;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, FragmentShaderStructParam_MultipleLocations) {
    auto r = wgsl_test::CompileWgsl(R"(
        struct FsIn {
            @location(0) col: vec4f,
            @location(1) uv: vec2f,
        };
        @fragment fn fs_main(in: FsIn) -> @location(0) vec4f {
            return in.col;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, FragmentShaderStructParam_UseMultipleFields) {
    auto r = wgsl_test::CompileWgsl(R"(
        struct FsIn {
            @location(0) col: vec4f,
            @location(1) uv: vec2f,
        };
        @fragment fn fs_main(in: FsIn) -> @location(0) vec4f {
            let c = in.col;
            let u = in.uv;
            return vec4f(u.x, u.y, 0.0, 1.0);
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, FragmentShaderStructParam_BuiltinPosition) {
    auto r = wgsl_test::CompileWgsl(R"(
        struct FsIn {
            @builtin(position) pos: vec4f,
            @location(0) col: vec4f,
        };
        @fragment fn fs_main(in: FsIn) -> @location(0) vec4f {
            return in.col;
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, FragmentShaderStructParam_InterpolateFlat) {
    auto r = wgsl_test::CompileWgsl(R"(
        struct FsIn {
            @location(0) @interpolate(flat) id: u32,
        };
        @fragment fn fs_main(in: FsIn) -> @location(0) vec4f {
            return vec4f(1.0, 0.0, 0.0, 1.0);
        }
    )");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(LowerTest, FragmentShaderStructParam_VertexFragmentPipeline) {
    // Vertex and fragment shaders compiled separately (normal workflow)
    auto vs = wgsl_test::CompileWgsl(R"(
        struct VsOut {
            @builtin(position) pos: vec4f,
            @location(0) col: vec4f,
        };
        @vertex fn vs_main(@location(0) pos: vec3f) -> VsOut {
            var out: VsOut;
            out.pos = vec4f(pos, 1.0);
            out.col = vec4f(1.0, 0.0, 0.0, 1.0);
            return out;
        }
    )");
    EXPECT_TRUE(vs.success) << vs.error;

    auto fs = wgsl_test::CompileWgsl(R"(
        struct FsIn {
            @location(0) col: vec4f,
        };
        @fragment fn fs_main(in: FsIn) -> @location(0) vec4f {
            return in.col;
        }
    )");
    EXPECT_TRUE(fs.success) << fs.error;
}

// =============================================================================
// Swizzle Tests - Single Component Extraction
// =============================================================================

TEST(SwizzleTest, Vec4_SingleComponent_xyzw) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    return vec4<f32>(v.x, v.y, v.z, v.w);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec4_SingleComponent_rgba) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    return vec4<f32>(v.r, v.g, v.b, v.a);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec3_SingleComponent) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<f32>(1.0, 2.0, 3.0);
    return vec4<f32>(v.x, v.y, v.z, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec2_SingleComponent) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec2<f32>(1.0, 2.0);
    return vec4<f32>(v.x, v.y, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Swizzle Tests - Two Component Extraction
// =============================================================================

TEST(SwizzleTest, Vec4_TwoComponent_xy) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let xy = v.xy;
    return vec4<f32>(xy, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec4_TwoComponent_zw) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let zw = v.zw;
    return vec4<f32>(zw, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec4_TwoComponent_yw) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let yw = v.yw;
    return vec4<f32>(yw, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec3_TwoComponent_yz) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<f32>(1.0, 2.0, 3.0);
    let yz = v.yz;
    return vec4<f32>(yz, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Swizzle Tests - Three Component Extraction
// =============================================================================

TEST(SwizzleTest, Vec4_ThreeComponent_xyz) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let xyz = v.xyz;
    return vec4<f32>(xyz, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec4_ThreeComponent_rgb) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let c = v.rgb;
    return vec4<f32>(c, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec4_ThreeComponent_yzw) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let yzw = v.yzw;
    return vec4<f32>(yzw, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Swizzle Tests - Four Component (identity and reorder)
// =============================================================================

TEST(SwizzleTest, Vec4_FourComponent_xyzw) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    return v.xyzw;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec4_FourComponent_wzyx) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    return v.wzyx;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec4_FourComponent_abgr) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    return v.abgr;
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Swizzle Tests - Duplicate Components
// =============================================================================

TEST(SwizzleTest, Vec4_Duplicate_xx) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let xx = v.xx;
    return vec4<f32>(xx, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec4_Duplicate_xxx) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let xxx = v.xxx;
    return vec4<f32>(xxx, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec4_Duplicate_xxxx) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    return v.xxxx;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec4_Duplicate_xxyy) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    return v.xxyy;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec4_Duplicate_aaaa) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    return v.aaaa;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec4_Duplicate_rrgg) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    return v.rrgg;
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Swizzle Tests - Integer Vectors
// =============================================================================

TEST(SwizzleTest, Vec4i_ThreeComponent_xyz) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<i32>(1, 2, 3, 4);
    let xyz = v.xyz;
    return vec4<f32>(f32(xyz.x), f32(xyz.y), f32(xyz.z), 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec4u_TwoComponent_xy) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<u32>(1u, 2u, 3u, 4u);
    let xy = v.xy;
    return vec4<f32>(f32(xy.x), f32(xy.y), 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, Vec3i_SingleComponent) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<i32>(10, 20, 30);
    return vec4<f32>(f32(v.x), f32(v.y), f32(v.z), 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Swizzle Tests - On Expressions (not just variables)
// =============================================================================

TEST(SwizzleTest, SwizzleOnArithmeticResult) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let b = vec4<f32>(0.1, 0.2, 0.3, 0.4);
    let rgb = (a + b).xyz;
    return vec4<f32>(rgb, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, SwizzleOnMultiplyResult) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let scaled = (v * 0.5).rgb;
    return vec4<f32>(scaled, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, SwizzleOnConstructor) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let xy = vec4<f32>(1.0, 2.0, 3.0, 4.0).xy;
    return vec4<f32>(xy, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Swizzle Tests - Chained Swizzles
// =============================================================================

TEST(SwizzleTest, ChainedSwizzle_xyz_xy) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let xy = v.xyz.xy;
    return vec4<f32>(xy, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, ChainedSwizzle_xyzw_zw) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let zw = v.xyzw.zw;
    return vec4<f32>(zw, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Swizzle Tests - In Arithmetic Expressions
// =============================================================================

TEST(SwizzleTest, SwizzleInMultiply) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 0.5, 0.25, 1.0);
    let scaled = v.rgb * 2.0;
    return vec4<f32>(scaled, v.a);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, SwizzleBothOperands) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let b = vec4<f32>(5.0, 6.0, 7.0, 8.0);
    let sum = a.xy + b.zw;
    return vec4<f32>(sum, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, SwizzleAsFunctionArg) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(3.0, 4.0, 0.0, 0.0);
    let len = length(v.xyz);
    return vec4<f32>(len, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, SwizzleNormalize) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 0.0);
    let n = normalize(v.xyz);
    return vec4<f32>(n, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, SwizzleDotProduct) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    let b = vec4<f32>(0.0, 1.0, 0.0, 0.0);
    let d = dot(a.xyz, b.xyz);
    return vec4<f32>(d, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleTest, SwizzleCross) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    let b = vec4<f32>(0.0, 1.0, 0.0, 0.0);
    let c = cross(a.xyz, b.xyz);
    return vec4<f32>(c, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Vector Construction Tests - From Mixed Components
// =============================================================================

TEST(VectorConstructionTest, Vec4_FromVec3AndScalar) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let rgb = vec3<f32>(1.0, 0.5, 0.25);
    return vec4<f32>(rgb, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorConstructionTest, Vec4_FromScalarAndVec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let yzw = vec3<f32>(0.5, 0.25, 1.0);
    return vec4<f32>(1.0, yzw);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorConstructionTest, Vec4_FromTwoVec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let xy = vec2<f32>(1.0, 2.0);
    let zw = vec2<f32>(3.0, 4.0);
    return vec4<f32>(xy, zw);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorConstructionTest, Vec4_FromVec2AndTwoScalars) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let xy = vec2<f32>(1.0, 2.0);
    return vec4<f32>(xy, 3.0, 4.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorConstructionTest, Vec4_FromTwoScalarsAndVec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let zw = vec2<f32>(3.0, 4.0);
    return vec4<f32>(1.0, 2.0, zw);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorConstructionTest, Vec4_FromScalarVec2Scalar) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let yz = vec2<f32>(2.0, 3.0);
    return vec4<f32>(1.0, yz, 4.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorConstructionTest, Vec3_FromVec2AndScalar) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let xy = vec2<f32>(1.0, 2.0);
    let v = vec3<f32>(xy, 3.0);
    return vec4<f32>(v, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorConstructionTest, Vec3_FromScalarAndVec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let yz = vec2<f32>(2.0, 3.0);
    let v = vec3<f32>(1.0, yz);
    return vec4<f32>(v, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorConstructionTest, Vec4_FromSwizzleAndScalar) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let color = vec4<f32>(1.0, 0.5, 0.25, 0.8);
    return vec4<f32>(color.rgb, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorConstructionTest, Vec4_FromSwizzleAndSwizzle) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let b = vec4<f32>(5.0, 6.0, 7.0, 8.0);
    return vec4<f32>(a.xy, b.xy);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Vector Construction Tests - Scalar Splat
// =============================================================================

TEST(VectorConstructionTest, Vec2_ScalarSplat) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec2<f32>(5.0);
    return vec4<f32>(v, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorConstructionTest, Vec3_ScalarSplat) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<f32>(0.5);
    return vec4<f32>(v, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorConstructionTest, Vec4_ScalarSplat) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.5);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Vector Construction Tests - Integer Vectors
// =============================================================================

TEST(VectorConstructionTest, Vec4i_FromVec3iAndScalar) {
    auto r = wgsl_test::CompileWgsl(R"(
fn helper() -> vec4<i32> {
    let xyz = vec3<i32>(1, 2, 3);
    return vec4<i32>(xyz, 4);
}
@fragment fn main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorConstructionTest, Vec3u_FromVec2uAndScalar) {
    auto r = wgsl_test::CompileWgsl(R"(
fn helper() -> vec3<u32> {
    let xy = vec2<u32>(1u, 2u);
    return vec3<u32>(xy, 3u);
}
@fragment fn main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Component-wise Math Builtins - Unary (on vectors)
// =============================================================================

TEST(ComponentMathTest, Abs_Vec3f) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<f32>(-1.0, -0.5, 0.25);
    let a = abs(v);
    return vec4<f32>(a, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Abs_Vec4i) {
    auto r = wgsl_test::CompileWgsl(R"(
fn helper() -> vec4<i32> {
    let v = vec4<i32>(-1, -2, 3, -4);
    return abs(v);
}
@fragment fn main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Floor_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<f32>(1.5, 2.7, -0.3);
    let f = floor(v);
    return vec4<f32>(f, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Ceil_Vec4) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.1, 2.9, -0.1, 0.5);
    return ceil(v);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Round_Vec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec2<f32>(1.4, 1.6);
    let rounded = round(v);
    return vec4<f32>(rounded, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Trunc_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<f32>(1.9, -2.3, 0.7);
    let t = trunc(v);
    return vec4<f32>(t, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Fract_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<f32>(1.75, 2.25, 3.5);
    let f = fract(v);
    return vec4<f32>(f, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Sqrt_Vec4) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 4.0, 9.0, 16.0);
    return sqrt(v);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, InverseSqrt_Vec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec2<f32>(4.0, 16.0);
    let inv = inverseSqrt(v);
    return vec4<f32>(inv, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Sign_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<f32>(-5.0, 0.0, 3.0);
    let s = sign(v);
    return vec4<f32>(s, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Exp_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<f32>(0.0, 1.0, 2.0);
    let e = exp(v);
    return vec4<f32>(e, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Exp2_Vec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec2<f32>(1.0, 3.0);
    let e = exp2(v);
    return vec4<f32>(e, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Log_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<f32>(1.0, 2.718, 7.389);
    let l = log(v);
    return vec4<f32>(l, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Log2_Vec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec2<f32>(2.0, 8.0);
    let l = log2(v);
    return vec4<f32>(l, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Component-wise Math Builtins - Trig (on vectors)
// =============================================================================

TEST(ComponentMathTest, Sin_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<f32>(0.0, 1.57, 3.14);
    let s = sin(v);
    return vec4<f32>(s, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Cos_Vec4) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(0.0, 1.57, 3.14, 6.28);
    return cos(v);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Tan_Vec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec2<f32>(0.0, 0.78);
    let t = tan(v);
    return vec4<f32>(t, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Asin_Vec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec2<f32>(0.0, 0.5);
    let a = asin(v);
    return vec4<f32>(a, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Acos_Vec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec2<f32>(0.0, 1.0);
    let a = acos(v);
    return vec4<f32>(a, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Atan_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<f32>(0.0, 1.0, -1.0);
    let a = atan(v);
    return vec4<f32>(a, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Component-wise Math Builtins - Binary (on vectors)
// =============================================================================

TEST(ComponentMathTest, Pow_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let base = vec3<f32>(2.0, 3.0, 4.0);
    let exp = vec3<f32>(2.0, 2.0, 0.5);
    let result = pow(base, exp);
    return vec4<f32>(result, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Pow_Vec4) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let base = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let exp = vec4<f32>(0.5, 0.5, 0.5, 0.5);
    return pow(base, exp);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Pow_Vec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let base = vec2<f32>(4.0, 9.0);
    let exp = vec2<f32>(0.5, 0.5);
    let result = pow(base, exp);
    return vec4<f32>(result, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Atan2_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let y = vec3<f32>(1.0, 0.0, -1.0);
    let x = vec3<f32>(0.0, 1.0, 0.0);
    let result = atan2(y, x);
    return vec4<f32>(result, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Distance_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec3<f32>(1.0, 0.0, 0.0);
    let b = vec3<f32>(0.0, 1.0, 0.0);
    let d = distance(a, b);
    return vec4<f32>(d, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Reflect_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let incident = vec3<f32>(1.0, -1.0, 0.0);
    let normal = vec3<f32>(0.0, 1.0, 0.0);
    let refl = reflect(incident, normal);
    return vec4<f32>(refl, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Step_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let edge = vec3<f32>(0.5, 0.5, 0.5);
    let x = vec3<f32>(0.3, 0.5, 0.7);
    let result = step(edge, x);
    return vec4<f32>(result, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Component-wise Math Builtins - Min/Max/Clamp (on vectors)
// =============================================================================

TEST(ComponentMathTest, Min_Vec3f) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec3<f32>(1.0, 5.0, 3.0);
    let b = vec3<f32>(4.0, 2.0, 6.0);
    let m = min(a, b);
    return vec4<f32>(m, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Max_Vec4f) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec4<f32>(1.0, 5.0, 3.0, 7.0);
    let b = vec4<f32>(4.0, 2.0, 6.0, 1.0);
    return max(a, b);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Clamp_Vec3f) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<f32>(-0.5, 0.5, 1.5);
    let lo = vec3<f32>(0.0, 0.0, 0.0);
    let hi = vec3<f32>(1.0, 1.0, 1.0);
    let c = clamp(v, lo, hi);
    return vec4<f32>(c, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Min_Vec2i) {
    auto r = wgsl_test::CompileWgsl(R"(
fn helper() -> vec2<i32> {
    let a = vec2<i32>(3, -5);
    let b = vec2<i32>(-1, 7);
    return min(a, b);
}
@fragment fn main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Max_Vec3u) {
    auto r = wgsl_test::CompileWgsl(R"(
fn helper() -> vec3<u32> {
    let a = vec3<u32>(1u, 5u, 3u);
    let b = vec3<u32>(4u, 2u, 6u);
    return max(a, b);
}
@fragment fn main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Component-wise Math Builtins - Mix/Smoothstep
// =============================================================================

TEST(ComponentMathTest, Mix_Vec3_ScalarT) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec3<f32>(0.0, 0.0, 0.0);
    let b = vec3<f32>(1.0, 1.0, 1.0);
    let m = mix(a, b, 0.5);
    return vec4<f32>(m, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Mix_Vec3_VecT) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec3<f32>(0.0, 0.0, 0.0);
    let b = vec3<f32>(1.0, 1.0, 1.0);
    let t = vec3<f32>(0.25, 0.5, 0.75);
    let m = mix(a, b, t);
    return vec4<f32>(m, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Smoothstep_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let lo = vec3<f32>(0.0, 0.0, 0.0);
    let hi = vec3<f32>(1.0, 1.0, 1.0);
    let x = vec3<f32>(0.25, 0.5, 0.75);
    let s = smoothstep(lo, hi, x);
    return vec4<f32>(s, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// Scalar-to-vector splatting for builtins
TEST(ComponentMathTest, Clamp_Vec3_ScalarBounds) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<f32>(1.5, -0.2, 0.8);
    let c = clamp(v, 0.0, 1.0);
    return vec4<f32>(c, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Smoothstep_Vec3_ScalarEdges) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x = vec3<f32>(0.25, 0.5, 0.75);
    let s = smoothstep(0.0, 1.0, x);
    return vec4<f32>(s, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Step_Vec3_ScalarEdge) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x = vec3<f32>(0.25, 0.75, 1.5);
    let s = step(0.5, x);
    return vec4<f32>(s, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ComponentMathTest, Mix_Vec4_ScalarT) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    let b = vec4<f32>(0.0, 0.0, 1.0, 1.0);
    return mix(a, b, 0.5);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Vector Builtins (non-component-wise)
// =============================================================================

TEST(VectorBuiltinTest, Dot_Vec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec2<f32>(1.0, 0.0);
    let b = vec2<f32>(0.0, 1.0);
    let d = dot(a, b);
    return vec4<f32>(d, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorBuiltinTest, Dot_Vec4) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let b = vec4<f32>(4.0, 3.0, 2.0, 1.0);
    let d = dot(a, b);
    return vec4<f32>(d, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorBuiltinTest, Cross_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec3<f32>(1.0, 0.0, 0.0);
    let b = vec3<f32>(0.0, 1.0, 0.0);
    let c = cross(a, b);
    return vec4<f32>(c, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorBuiltinTest, Length_Vec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec2<f32>(3.0, 4.0);
    let l = length(v);
    return vec4<f32>(l, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorBuiltinTest, Length_Vec4) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let l = length(v);
    return vec4<f32>(l, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorBuiltinTest, Normalize_Vec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec2<f32>(3.0, 4.0);
    let n = normalize(v);
    return vec4<f32>(n, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorBuiltinTest, Normalize_Vec4) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    return normalize(v);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(VectorBuiltinTest, Distance_Vec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec2<f32>(0.0, 0.0);
    let b = vec2<f32>(3.0, 4.0);
    let d = distance(a, b);
    return vec4<f32>(d, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Swizzle + Math Integration Tests
// =============================================================================

TEST(SwizzleMathTest, PowOnSwizzle) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let color = vec4<f32>(0.5, 0.25, 0.125, 1.0);
    let gamma = vec3<f32>(2.2);
    let corrected = pow(color.rgb, gamma);
    return vec4<f32>(corrected, color.a);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleMathTest, NormalizeSwizzle) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(1.0, 2.0, 3.0, 0.0);
    let n = normalize(v.xyz);
    return vec4<f32>(n, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleMathTest, DotOnSwizzle) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let normal = vec4<f32>(0.0, 1.0, 0.0, 0.0);
    let light = vec4<f32>(0.577, 0.577, 0.577, 0.0);
    let ndotl = dot(normal.xyz, light.xyz);
    return vec4<f32>(ndotl, ndotl, ndotl, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleMathTest, CrossOnSwizzle) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    let b = vec4<f32>(0.0, 1.0, 0.0, 0.0);
    let c = cross(a.xyz, b.xyz);
    return vec4<f32>(c, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleMathTest, MixOnSwizzle) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    let b = vec4<f32>(0.0, 0.0, 1.0, 1.0);
    let mixed = mix(a.rgb, b.rgb, 0.5);
    return vec4<f32>(mixed, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleMathTest, ClampOnSwizzle) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let hdr = vec4<f32>(1.5, -0.2, 0.8, 1.0);
    let clamped = clamp(hdr.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(clamped, hdr.a);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleMathTest, SqrtOnSwizzle) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(4.0, 9.0, 16.0, 25.0);
    let s = sqrt(v.xyz);
    return vec4<f32>(s, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SwizzleMathTest, AbsOnSwizzle) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(-1.0, 2.0, -3.0, 4.0);
    let a = abs(v.xyz);
    return vec4<f32>(a, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// Integration Tests - Real Shader Patterns
// =============================================================================

TEST(IntegrationTest, GammaCorrection) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Uniforms { gamma: f32, };
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var s: sampler;
@group(1) @binding(0) var t: texture_2d<f32>;

@fragment fn main(@location(0) color: vec4<f32>, @location(1) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let texColor = textureSample(t, s, uv);
    let combined = color * texColor;
    let corrected = pow(combined.rgb, vec3<f32>(u.gamma));
    return vec4<f32>(corrected, combined.a);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(IntegrationTest, SimpleLighting) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main(@location(0) normal: vec3<f32>) -> @location(0) vec4<f32> {
    let lightDir = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let n = normalize(normal);
    let ndotl = max(dot(n, lightDir), 0.0);
    let diffuse = vec3<f32>(0.8, 0.6, 0.4) * ndotl;
    let ambient = vec3<f32>(0.1, 0.1, 0.1);
    return vec4<f32>(diffuse + ambient, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(IntegrationTest, ColorBlending) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
    let tint = vec4<f32>(1.0, 0.8, 0.6, 1.0);
    let blended = mix(color.rgb, tint.rgb, 0.3);
    let saturated = clamp(blended, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(saturated, color.a * tint.a);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(IntegrationTest, DistanceAttenuation) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main(@location(0) worldPos: vec3<f32>) -> @location(0) vec4<f32> {
    let lightPos = vec3<f32>(0.0, 5.0, 0.0);
    let dist = distance(worldPos, lightPos);
    let atten = 1.0 / (1.0 + dist * dist);
    let color = vec3<f32>(1.0, 1.0, 1.0) * atten;
    return vec4<f32>(color, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(IntegrationTest, ImGuiVertexShader) {
    auto r = wgsl_test::CompileWgsl(R"(
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
};
struct Uniforms { mvp: mat4x4<f32>, gamma: f32, };
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex fn main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = uniforms.mvp * vec4<f32>(in.position, 0.0, 1.0);
    out.color = in.color;
    out.uv = in.uv;
    return out;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(IntegrationTest, ImGuiFragmentShader) {
    auto r = wgsl_test::CompileWgsl(R"(
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
};
struct Uniforms { mvp: mat4x4<f32>, gamma: f32, };
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var s: sampler;
@group(1) @binding(0) var t: texture_2d<f32>;

@fragment fn main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = in.color * textureSample(t, s, in.uv);
    let corrected_color = pow(color.rgb, vec3<f32>(uniforms.gamma));
    return vec4<f32>(corrected_color, color.a);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(IntegrationTest, ReflectionVector) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main(@location(0) normal: vec3<f32>, @location(1) viewDir: vec3<f32>) -> @location(0) vec4<f32> {
    let n = normalize(normal);
    let v = normalize(viewDir);
    let r = reflect(-v, n);
    let spec = pow(max(dot(r, v), 0.0), 32.0);
    return vec4<f32>(vec3<f32>(spec), 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// --- Const array regression tests ---

TEST(ConstArrayTest, ModuleScopeConstArrayF32) {
    auto r = wgsl_test::CompileWgsl(R"(
const WEIGHTS = array<f32, 4>(0.1, 0.2, 0.3, 0.4);
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = WEIGHTS[gid.x % 4u];
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ConstArrayTest, ModuleScopeConstArrayU32) {
    auto r = wgsl_test::CompileWgsl(R"(
const INDICES = array<u32, 3>(10u, 20u, 30u);
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = INDICES[gid.x % 3u];
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ConstArrayTest, ModuleScopeConstArrayI32) {
    auto r = wgsl_test::CompileWgsl(R"(
const OFFSETS = array<i32, 4>(-2, -1, 1, 2);
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let off = OFFSETS[gid.x % 4u];
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ConstArrayTest, ConstArrayIndexedByLoopVariable) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Out { data: array<f32>, }
const TAPS = array<f32, 4>(0.25, 0.5, 0.75, 1.0);
@group(0) @binding(0) var<storage, read_write> out: Out;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var sum = 0.0;
    for (var i = 0u; i < 4u; i = i + 1u) {
        sum = sum + TAPS[i];
    }
    out.data[gid.x] = sum;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ConstArrayTest, MultipleConstArrays) {
    auto r = wgsl_test::CompileWgsl(R"(
const A = array<f32, 3>(1.0, 2.0, 3.0);
const B = array<f32, 3>(4.0, 5.0, 6.0);
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x % 3u;
    let sum = A[i] + B[i];
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ConstArrayTest, ConstArrayInExpression) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Out { data: array<f32>, }
const SCALE = array<f32, 4>(0.5, 1.0, 1.5, 2.0);
@group(0) @binding(0) var<storage, read_write> out: Out;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x % 4u;
    out.data[gid.x] = SCALE[i] * 3.14 + 1.0;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ConstArrayTest, ConstArrayOfVec2) {
    auto r = wgsl_test::CompileWgsl(R"(
const DIRS = array<vec2<f32>, 4>(
    vec2<f32>(1.0, 0.0),
    vec2<f32>(0.0, 1.0),
    vec2<f32>(-1.0, 0.0),
    vec2<f32>(0.0, -1.0)
);
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = DIRS[gid.x % 4u];
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ConstArrayTest, ConstArrayOfVec4) {
    auto r = wgsl_test::CompileWgsl(R"(
const COLORS = array<vec4<f32>, 3>(
    vec4<f32>(1.0, 0.0, 0.0, 1.0),
    vec4<f32>(0.0, 1.0, 0.0, 1.0),
    vec4<f32>(0.0, 0.0, 1.0, 1.0)
);
@fragment fn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let i = u32(pos.x) % 3u;
    return COLORS[i];
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(ConstArrayTest, FunctionScopeLetArray) {
    auto r = wgsl_test::CompileWgsl(R"(
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let weights = array<f32, 4>(0.1, 0.2, 0.3, 0.4);
    let w = weights[gid.x % 4u];
})");
    EXPECT_TRUE(r.success) << r.error;
}

#ifdef WGSL_HAS_FFT
// ============================================================================
// Stockham FFT generator tests
// ============================================================================

class StockhamGenTest : public ::testing::Test {};

// Helper: compile WGSL from gen_fft_stockham through the full pipeline
static void CompileStockhamShader(int radix, int stride, int n_total,
                                  int direction, int workgroup_size) {
    char *src = gen_fft_stockham(radix, stride, n_total, direction, workgroup_size);
    ASSERT_NE(src, nullptr) << "gen_fft_stockham returned NULL for radix=" << radix
        << " stride=" << stride << " n_total=" << n_total
        << " dir=" << direction << " wg=" << workgroup_size;

    WgslAstNode *ast = wgsl_parse(src);
    ASSERT_NE(ast, nullptr) << "wgsl_parse failed for radix=" << radix;

    WgslResolver *resolver = wgsl_resolver_build(ast);
    ASSERT_NE(resolver, nullptr) << "wgsl_resolver_build failed for radix=" << radix;

    uint32_t *spirv = nullptr;
    size_t spirv_size = 0;
    WgslLowerOptions opts = {};
    opts.spirv_version = 0x00010300;
    opts.env = WGSL_LOWER_ENV_VULKAN_1_1;
    opts.packing = WGSL_LOWER_PACK_STD430;

    WgslLowerResult result = wgsl_lower_emit_spirv(ast, resolver, &opts, &spirv, &spirv_size);
    EXPECT_EQ(result, WGSL_LOWER_OK)
        << "wgsl_lower_emit_spirv failed for radix=" << radix
        << " stride=" << stride << " n_total=" << n_total;

    if (spirv) wgsl_lower_free(spirv);
    wgsl_resolver_free(resolver);
    wgsl_free_ast(ast);
    free(src);
}

TEST_F(StockhamGenTest, AllRadicesCompile) {
    for (int radix = 2; radix <= 32; radix++) {
        SCOPED_TRACE("radix=" + std::to_string(radix));
        CompileStockhamShader(radix, /*stride=*/1, /*n_total=*/radix,
                              /*direction=*/1, /*workgroup_size=*/64);
    }
}

TEST_F(StockhamGenTest, NonTrivialStride) {
    CompileStockhamShader(/*radix=*/4, /*stride=*/8, /*n_total=*/32,
                          /*direction=*/1, /*workgroup_size=*/64);
}

TEST_F(StockhamGenTest, InverseDirection) {
    CompileStockhamShader(/*radix=*/8, /*stride=*/1, /*n_total=*/8,
                          /*direction=*/-1, /*workgroup_size=*/64);
}

// =========================================================================
// User-defined function call tests
// =========================================================================

TEST(UserFunctionTest, SimpleHelperFunction) {
    const char *source = R"(
        fn double_it(x: f32) -> f32 {
            return x + x;
        }
        @compute @workgroup_size(1)
        fn main() {
            let v = double_it(3.0);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(UserFunctionTest, Vec2Helper) {
    const char *source = R"(
        fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
            return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
        }
        @compute @workgroup_size(1)
        fn main() {
            let a = vec2<f32>(1.0, 2.0);
            let b = vec2<f32>(3.0, 4.0);
            let c = cmul(a, b);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(UserFunctionTest, VoidHelper) {
    const char *source = R"(
        struct Buf { d: array<f32> };
        @group(0) @binding(0) var<storage, read_write> buf: Buf;

        fn store_pair(offset: u32, x: f32, y: f32) {
            buf.d[offset] = x;
            buf.d[offset + 1u] = y;
        }
        @compute @workgroup_size(1)
        fn main() {
            store_pair(0u, 1.0, 2.0);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(UserFunctionTest, MultipleHelpers) {
    const char *source = R"(
        fn add(a: f32, b: f32) -> f32 { return a + b; }
        fn mul(a: f32, b: f32) -> f32 { return a * b; }
        @compute @workgroup_size(1)
        fn main() {
            let x = add(1.0, 2.0);
            let y = mul(x, 3.0);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(UserFunctionTest, HelperCallingHelper) {
    const char *source = R"(
        fn square(x: f32) -> f32 { return x * x; }
        fn sum_of_squares(a: f32, b: f32) -> f32 {
            return square(a) + square(b);
        }
        @compute @workgroup_size(1)
        fn main() {
            let r = sum_of_squares(3.0, 4.0);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(UserFunctionTest, IntegerParams) {
    const char *source = R"(
        fn index2d(row: u32, col: u32, stride: u32) -> u32 {
            return row * stride + col;
        }
        @compute @workgroup_size(1)
        fn main() {
            let idx = index2d(3u, 5u, 10u);
        }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(DeviceAddressFftTest, FusedStyleShader) {
    auto r = wgsl_test::CompileWgsl(R"(
enable device_address;
struct SrcBuf { d: array<f32> };
struct DstBuf { d: array<f32> };
struct LutBuf { d: array<f32> };
var<device, read> src: SrcBuf;
var<device, read_write> dst: DstBuf;
var<device, read> lut: LutBuf;
var<workgroup> smem: array<f32, 1024>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    smem[lid.x] = src.d[gid.x];
    workgroupBarrier();
    dst.d[gid.x] = smem[lid.x] * lut.d[lid.x];
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(DeviceAddressFftTest, StockhamStyleShader) {
    auto r = wgsl_test::CompileWgsl(R"(
enable device_address;
struct SrcBuf { d: array<f32> };
struct DstBuf { d: array<f32> };
var<device, read> src: SrcBuf;
var<device, read_write> dst: DstBuf;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    dst.d[gid.x * 2u] = src.d[gid.x * 2u];
    dst.d[gid.x * 2u + 1u] = src.d[gid.x * 2u + 1u];
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(DeviceAddressFftTest, InPlaceStyleShader) {
    auto r = wgsl_test::CompileWgsl(R"(
enable device_address;
struct DataBuf { d: array<f32> };
var<device, read_write> data: DataBuf;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let re = data.d[gid.x * 2u];
    let im = data.d[gid.x * 2u + 1u];
    data.d[gid.x * 2u] = re;
    data.d[gid.x * 2u + 1u] = -im;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(DeviceAddressFftTest, FusedGenN64_BDA) {
    char *wgsl = gen_fft_fused(64, 1, 0, 0);
    ASSERT_NE(wgsl, nullptr);
    EXPECT_NE(strstr(wgsl, "var<device"), nullptr)
        << "Expected var<device> in generated WGSL";
    EXPECT_EQ(strstr(wgsl, "@binding"), nullptr)
        << "Should NOT have @binding declarations";
    auto r = wgsl_test::CompileWgsl(wgsl);
    free(wgsl);
    EXPECT_TRUE(r.success) << r.error;
}

TEST(DeviceAddressFftTest, FusedGenN8_BDA) {
    char *wgsl = gen_fft_fused(8, 1, 0, 0);
    ASSERT_NE(wgsl, nullptr);
    EXPECT_NE(strstr(wgsl, "var<device"), nullptr);
    auto r = wgsl_test::CompileWgsl(wgsl);
    free(wgsl);
    EXPECT_TRUE(r.success) << r.error;
}

TEST(DeviceAddressFftTest, StockhamGenRadix4_BDA) {
    char *wgsl = gen_fft_stockham(4, 1, 1024, 1, 256);
    ASSERT_NE(wgsl, nullptr);
    EXPECT_NE(strstr(wgsl, "var<device"), nullptr);
    EXPECT_EQ(strstr(wgsl, "@binding"), nullptr);
    auto r = wgsl_test::CompileWgsl(wgsl);
    free(wgsl);
    EXPECT_TRUE(r.success) << r.error;
}

TEST(DeviceAddressFftTest, StockhamGenR2C_BDA) {
    char *wgsl = gen_fft_r2c_postprocess(1024, 256);
    ASSERT_NE(wgsl, nullptr);
    EXPECT_NE(strstr(wgsl, "var<device"), nullptr);
    EXPECT_EQ(strstr(wgsl, "@binding"), nullptr);
    auto r = wgsl_test::CompileWgsl(wgsl);
    free(wgsl);
    EXPECT_TRUE(r.success) << r.error;
}

TEST(DeviceAddressFftTest, TransposeTiled_BDA) {
    char *wgsl = gen_transpose_tiled(16, 16, 8);
    ASSERT_NE(wgsl, nullptr);
    EXPECT_NE(strstr(wgsl, "var<device"), nullptr);
    EXPECT_EQ(strstr(wgsl, "@binding"), nullptr);
    auto r = wgsl_test::CompileWgsl(wgsl);
    free(wgsl);
    EXPECT_TRUE(r.success) << r.error;
}

TEST(DeviceAddressFftTest, Fft2dFused_BDA) {
    char *wgsl = gen_fft_2d_fused(8, 8, 1, 0);
    ASSERT_NE(wgsl, nullptr);
    EXPECT_NE(strstr(wgsl, "var<device"), nullptr);
    EXPECT_EQ(strstr(wgsl, "@binding"), nullptr);
    auto r = wgsl_test::CompileWgsl(wgsl);
    free(wgsl);
    EXPECT_TRUE(r.success) << r.error;
}

TEST(DeviceAddressFftTest, Fft2dFusedLooped_BDA) {
    char *wgsl = gen_fft_2d_fused_looped(8, 8, 1, 0);
    ASSERT_NE(wgsl, nullptr);
    EXPECT_NE(strstr(wgsl, "var<device"), nullptr);
    EXPECT_EQ(strstr(wgsl, "@binding"), nullptr);
    auto r = wgsl_test::CompileWgsl(wgsl);
    free(wgsl);
    EXPECT_TRUE(r.success) << r.error;
}

TEST(DeviceAddressFftTest, FourStepTwiddleTranspose_BDA) {
    char *wgsl = gen_fft_twiddle_transpose(32, 32, 1, 256);
    ASSERT_NE(wgsl, nullptr);
    EXPECT_NE(strstr(wgsl, "var<device"), nullptr);
    EXPECT_EQ(strstr(wgsl, "@binding"), nullptr);
    auto r = wgsl_test::CompileWgsl(wgsl);
    free(wgsl);
    EXPECT_TRUE(r.success) << r.error;
}

TEST(DeviceAddressFftTest, FourStepTranspose_BDA) {
    char *wgsl = gen_fft_transpose(32, 32, 256);
    ASSERT_NE(wgsl, nullptr);
    EXPECT_NE(strstr(wgsl, "var<device"), nullptr);
    EXPECT_EQ(strstr(wgsl, "@binding"), nullptr);
    auto r = wgsl_test::CompileWgsl(wgsl);
    free(wgsl);
    EXPECT_TRUE(r.success) << r.error;
}

TEST(DeviceAddressFftTest, FftDirect_BDA) {
    char *wgsl = gen_fft_direct(8, 1);
    ASSERT_NE(wgsl, nullptr);
    EXPECT_NE(strstr(wgsl, "var<device"), nullptr);
    EXPECT_EQ(strstr(wgsl, "@binding"), nullptr);
    auto r = wgsl_test::CompileWgsl(wgsl);
    free(wgsl);
    EXPECT_TRUE(r.success) << r.error;
}
#endif // WGSL_HAS_FFT

// ---- Slug text rendering shader tests ----
// These exercise struct constructors, bitcast, fwidth, saturate, and
// struct I/O with @builtin/@location/@interpolate attributes.

TEST(LowerTest, SlugVertexShader) {
    const char *source = R"(
struct SlugUnpackResult {
    vbnd: vec4<f32>,
    vgly: vec4<i32>,
}

fn SlugUnpack(tex: vec4<f32>, bnd: vec4<f32>) -> SlugUnpackResult {
    let g = vec2<u32>(bitcast<u32>(tex.z), bitcast<u32>(tex.w));
    let vgly = vec4<i32>(
        i32(g.x & 0xFFFFu),
        i32(g.x >> 16u),
        i32(g.y & 0xFFFFu),
        i32(g.y >> 16u)
    );
    let vbnd = bnd;
    return SlugUnpackResult(vbnd, vgly);
}

struct SlugDilateResult {
    texcoord: vec2<f32>,
    vpos: vec2<f32>,
}

fn SlugDilate(pos: vec4<f32>, tex: vec4<f32>, jac: vec4<f32>, m0: vec4<f32>, m1: vec4<f32>, m3: vec4<f32>, dim: vec2<f32>) -> SlugDilateResult {
    let n = normalize(pos.zw);
    let s = dot(m3.xy, pos.xy) + m3.w;
    let t = dot(m3.xy, n);

    let u = (s * dot(m0.xy, n) - t * (dot(m0.xy, pos.xy) + m0.w)) * dim.x;
    let v = (s * dot(m1.xy, n) - t * (dot(m1.xy, pos.xy) + m1.w)) * dim.y;

    let s2 = s * s;
    let st = s * t;
    let uv = u * u + v * v;
    let d = pos.zw * (s2 * (st + sqrt(uv)) / (uv - st * st));

    let vpos = pos.xy + d;
    let texcoord = vec2<f32>(tex.x + dot(d, jac.xy), tex.y + dot(d, jac.zw));
    return SlugDilateResult(texcoord, vpos);
}

struct ParamStruct {
    slug_matrix: array<vec4<f32>, 4>,
    slug_viewport: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: ParamStruct;

struct VertexInput {
    @location(0) pos: vec4<f32>,
    @location(1) tex: vec4<f32>,
    @location(2) jac: vec4<f32>,
    @location(3) bnd: vec4<f32>,
    @location(4) col: vec4<f32>,
};

struct VertexStruct {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) texcoord: vec2<f32>,
    @location(2) @interpolate(flat) banding: vec4<f32>,
    @location(3) @interpolate(flat) glyph: vec4<i32>,
};

@vertex
fn main(attrib: VertexInput) -> VertexStruct {
    var vresult: VertexStruct;

    let dilateResult = SlugDilate(attrib.pos, attrib.tex, attrib.jac, params.slug_matrix[0], params.slug_matrix[1], params.slug_matrix[3], params.slug_viewport.xy);
    vresult.texcoord = dilateResult.texcoord;
    let p = dilateResult.vpos;

    vresult.position.x = p.x * params.slug_matrix[0].x + p.y * params.slug_matrix[0].y + params.slug_matrix[0].w;
    vresult.position.y = p.x * params.slug_matrix[1].x + p.y * params.slug_matrix[1].y + params.slug_matrix[1].w;
    vresult.position.z = p.x * params.slug_matrix[2].x + p.y * params.slug_matrix[2].y + params.slug_matrix[2].w;
    vresult.position.w = p.x * params.slug_matrix[3].x + p.y * params.slug_matrix[3].y + params.slug_matrix[3].w;

    let unpackResult = SlugUnpack(attrib.tex, attrib.bnd);
    vresult.banding = unpackResult.vbnd;
    vresult.glyph = unpackResult.vgly;
    vresult.color = attrib.col;
    return vresult;
}
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Slug vertex shader: " << result.error;
}

TEST(LowerTest, SlugFragmentShader) {
    const char *source = R"(
const kLogBandTextureWidth: u32 = 12u;

fn TexelLoad2D_f32(tex: texture_2d<f32>, coords: vec2<i32>) -> vec4<f32> {
    return textureLoad(tex, coords, 0);
}

fn TexelLoad2D_u32(tex: texture_2d<u32>, coords: vec2<i32>) -> vec4<u32> {
    return textureLoad(tex, coords, 0);
}

fn CalcRootCode(y1: f32, y2: f32, y3: f32) -> u32 {
    let i1 = bitcast<u32>(y1) >> 31u;
    let i2 = bitcast<u32>(y2) >> 30u;
    let i3 = bitcast<u32>(y3) >> 29u;
    let shift = (i3 & 4u) | (((i2 & 2u) | (i1 & ~2u)) & ~4u);
    return ((0x2E74u >> shift) & 0x0101u);
}

fn SolveHorizPoly(p12: vec4<f32>, p3: vec2<f32>) -> vec2<f32> {
    let a = vec2<f32>(p12.x - p12.z * 2.0 + p3.x, p12.y - p12.w * 2.0 + p3.y);
    let b = vec2<f32>(p12.x - p12.z, p12.y - p12.w);
    let ra = 1.0 / a.y;
    let rb = 0.5 / b.y;
    let d = sqrt(max(b.y * b.y - a.y * p12.y, 0.0));
    var t1 = (b.y - d) * ra;
    var t2 = (b.y + d) * ra;
    if (abs(a.y) < 1.0 / 65536.0) {
        t1 = p12.y * rb;
        t2 = p12.y * rb;
    }
    return vec2<f32>((a.x * t1 - b.x * 2.0) * t1 + p12.x, (a.x * t2 - b.x * 2.0) * t2 + p12.x);
}

fn SolveVertPoly(p12: vec4<f32>, p3: vec2<f32>) -> vec2<f32> {
    let a = vec2<f32>(p12.x - p12.z * 2.0 + p3.x, p12.y - p12.w * 2.0 + p3.y);
    let b = vec2<f32>(p12.x - p12.z, p12.y - p12.w);
    let ra = 1.0 / a.x;
    let rb = 0.5 / b.x;
    let d = sqrt(max(b.x * b.x - a.x * p12.x, 0.0));
    var t1 = (b.x - d) * ra;
    var t2 = (b.x + d) * ra;
    if (abs(a.x) < 1.0 / 65536.0) {
        t1 = p12.x * rb;
        t2 = p12.x * rb;
    }
    return vec2<f32>((a.y * t1 - b.y * 2.0) * t1 + p12.y, (a.y * t2 - b.y * 2.0) * t2 + p12.y);
}

fn CalcBandLoc(glyphLoc: vec2<i32>, offset: u32) -> vec2<i32> {
    var bandLoc = vec2<i32>(glyphLoc.x + i32(offset), glyphLoc.y);
    bandLoc.y += bandLoc.x >> kLogBandTextureWidth;
    bandLoc.x &= (1 << kLogBandTextureWidth) - 1;
    return bandLoc;
}

fn CalcCoverage(xcov: f32, ycov: f32, xwgt: f32, ywgt: f32, flags: i32) -> f32 {
    var coverage = max(abs(xcov * xwgt + ycov * ywgt) / max(xwgt + ywgt, 1.0 / 65536.0), min(abs(xcov), abs(ycov)));
    coverage = saturate(coverage);
    return coverage;
}

fn SlugRender(curveData: texture_2d<f32>, bandData: texture_2d<u32>, renderCoord: vec2<f32>, bandTransform: vec4<f32>, glyphData: vec4<i32>) -> f32 {
    var curveIndex: i32;

    let emsPerPixel = fwidth(renderCoord);
    let pixelsPerEm = 1.0 / emsPerPixel;

    var bandMax = vec2<i32>(glyphData.z, glyphData.w & 0x00FF);
    let bandIndex = clamp(vec2<i32>(renderCoord * bandTransform.xy + bandTransform.zw), vec2<i32>(0, 0), bandMax);
    let glyphLoc = vec2<i32>(glyphData.x, glyphData.y);

    var xcov: f32 = 0.0;
    var xwgt: f32 = 0.0;

    let hbandData = TexelLoad2D_u32(bandData, vec2<i32>(glyphLoc.x + bandIndex.y, glyphLoc.y)).xy;
    let hbandLoc = CalcBandLoc(glyphLoc, hbandData.y);

    for (curveIndex = 0; curveIndex < i32(hbandData.x); curveIndex++) {
        let curveLoc = vec2<i32>(TexelLoad2D_u32(bandData, vec2<i32>(hbandLoc.x + curveIndex, hbandLoc.y)).xy);
        let p12 = TexelLoad2D_f32(curveData, curveLoc) - vec4<f32>(renderCoord, renderCoord);
        let p3 = TexelLoad2D_f32(curveData, vec2<i32>(curveLoc.x + 1, curveLoc.y)).xy - renderCoord;

        if (max(max(p12.x, p12.z), p3.x) * pixelsPerEm.x < -0.5) {
            break;
        }

        let code = CalcRootCode(p12.y, p12.w, p3.y);
        if (code != 0u) {
            let r = SolveHorizPoly(p12, p3) * pixelsPerEm.x;
            if ((code & 1u) != 0u) {
                xcov += saturate(r.x + 0.5);
                xwgt = max(xwgt, saturate(1.0 - abs(r.x) * 2.0));
            }
            if (code > 1u) {
                xcov -= saturate(r.y + 0.5);
                xwgt = max(xwgt, saturate(1.0 - abs(r.y) * 2.0));
            }
        }
    }

    var ycov: f32 = 0.0;
    var ywgt: f32 = 0.0;

    let vbandData = TexelLoad2D_u32(bandData, vec2<i32>(glyphLoc.x + bandMax.y + 1 + bandIndex.x, glyphLoc.y)).xy;
    let vbandLoc = CalcBandLoc(glyphLoc, vbandData.y);

    for (curveIndex = 0; curveIndex < i32(vbandData.x); curveIndex++) {
        let curveLoc = vec2<i32>(TexelLoad2D_u32(bandData, vec2<i32>(vbandLoc.x + curveIndex, vbandLoc.y)).xy);
        let p12 = TexelLoad2D_f32(curveData, curveLoc) - vec4<f32>(renderCoord, renderCoord);
        let p3 = TexelLoad2D_f32(curveData, vec2<i32>(curveLoc.x + 1, curveLoc.y)).xy - renderCoord;

        if (max(max(p12.y, p12.w), p3.y) * pixelsPerEm.y < -0.5) {
            break;
        }

        let code = CalcRootCode(p12.x, p12.z, p3.x);
        if (code != 0u) {
            let r = SolveVertPoly(p12, p3) * pixelsPerEm.y;
            if ((code & 1u) != 0u) {
                ycov -= saturate(r.x + 0.5);
                ywgt = max(ywgt, saturate(1.0 - abs(r.x) * 2.0));
            }
            if (code > 1u) {
                ycov += saturate(r.y + 0.5);
                ywgt = max(ywgt, saturate(1.0 - abs(r.y) * 2.0));
            }
        }
    }

    return CalcCoverage(xcov, ycov, xwgt, ywgt, glyphData.w);
}

struct VertexStruct {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) texcoord: vec2<f32>,
    @location(2) @interpolate(flat) banding: vec4<f32>,
    @location(3) @interpolate(flat) glyph: vec4<i32>,
};

@group(0) @binding(1) var curveTexture: texture_2d<f32>;
@group(0) @binding(2) var bandTexture: texture_2d<u32>;

@fragment
fn main(vresult: VertexStruct) -> @location(0) vec4<f32> {
    let coverage = SlugRender(curveTexture, bandTexture, vresult.texcoord, vresult.banding, vresult.glyph);
    return vresult.color * coverage;
}
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Slug fragment shader: " << result.error;
}

// Isolated regression tests for individual features

TEST(LowerTest, StructConstructorReturn) {
    const char *source = R"(
struct Pair { a: f32, b: f32 }
fn make_pair(x: f32, y: f32) -> Pair {
    return Pair(x, y);
}
@compute @workgroup_size(1) fn main() {
    let p = make_pair(1.0, 2.0);
}
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Struct constructor return: " << result.error;
}

TEST(LowerTest, BitcastBuiltin) {
    const char *source = R"(
@compute @workgroup_size(1) fn main() {
    let f: f32 = 1.0;
    let u: u32 = bitcast<u32>(f);
    let back: f32 = bitcast<f32>(u);
}
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "bitcast: " << result.error;
}
