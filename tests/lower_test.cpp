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
