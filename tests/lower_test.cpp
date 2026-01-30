#include <gtest/gtest.h>
#include "test_utils.h"

TEST(LowerTest, EmitMinimalSpirv) {
    const char* source = "fn main() {}";
    wgsl_test::AstGuard ast(wgsl_parse(source));
    ASSERT_NE(ast.get(), nullptr);

    wgsl_test::ResolverGuard resolver(wgsl_resolver_build(ast.get()));
    ASSERT_NE(resolver.get(), nullptr);

    uint32_t* spirv = nullptr;
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
    const char* source = "fn main() {}";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, ValidateFragmentShader) {
    const char* source = R"(
        @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, ValidateVertexShader) {
    const char* source = R"(
        @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0.0); }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, ValidateComputeShader) {
    const char* source = R"(
        @compute @workgroup_size(1) fn cs() {}
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, ValidateBindingVariable) {
    const char* source = R"(
        @group(0) @binding(0) var tex: texture_2d<f32>;
        @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, ValidateSampler) {
    const char* source = R"(
        @group(0) @binding(0) var s: sampler;
        @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }
    )";
    auto result = wgsl_test::CompileWgsl(source);
    EXPECT_TRUE(result.success) << "Validation error: " << result.error;
}

TEST(LowerTest, TypeCachingWorks) {
    // This test verifies that type caching works correctly
    // by compiling a shader that uses the same types multiple times
    const char* source = R"(
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
    const WgslLowerEntrypointInfo* eps = wgsl_lower_entrypoints(lower.get(), &count);
    EXPECT_GE(count, 1);
    EXPECT_NE(eps, nullptr);
}

TEST(LowerTest, MultipleEntrypoints) {
    const char* source = R"(
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
    const WgslLowerEntrypointInfo* eps = wgsl_lower_entrypoints(lower.get(), &count);
    EXPECT_EQ(count, 2);

    // Verify both entrypoints have function IDs
    for (int i = 0; i < count; ++i) {
        EXPECT_NE(eps[i].function_id, 0u);
    }
}

TEST(LowerTest, ModuleFeatures) {
    const char* source = "fn main() {}";

    wgsl_test::AstGuard ast(wgsl_parse(source));
    ASSERT_NE(ast.get(), nullptr);

    wgsl_test::ResolverGuard resolver(wgsl_resolver_build(ast.get()));
    ASSERT_NE(resolver.get(), nullptr);

    WgslLowerOptions opts = {};
    opts.env = WGSL_LOWER_ENV_VULKAN_1_3;

    wgsl_test::LowerGuard lower(wgsl_lower_create(ast.get(), resolver.get(), &opts));
    ASSERT_NE(lower.get(), nullptr);

    const WgslLowerModuleFeatures* features = wgsl_lower_module_features(lower.get());
    ASSERT_NE(features, nullptr);

    // Should have at least Shader capability
    EXPECT_GE(features->capability_count, 1u);
}

// ==================== Expression Lowering Tests ====================

TEST(LowerTest, ArithmeticExpressions) {
    const char* source = R"(
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
    const char* source = R"(
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
    const char* source = R"(
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
    const char* source = R"(
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
    const char* source = R"(
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
    const char* source = R"(
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
    const char* source = R"(
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
    const char* source = R"(
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
    const char* source = R"(
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
    const char* source = R"(
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
    const char* source = R"(
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
    const char* source = R"(
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
    const char* source = R"(
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
    const char* source = R"(
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
    const char* source = R"(
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
    const char* source = R"(
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
    const char* source = R"(
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
    const char* source = R"(
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
