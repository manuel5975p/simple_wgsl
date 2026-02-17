#include <gtest/gtest.h>
#include "test_utils.h"

// =============================================================================
// transpose() and determinant() builtin functions
// SPIR-V: transpose -> SpvOpTranspose, determinant -> GLSLstd450Determinant
// transpose(matNxM<f32>) returns matMxN<f32>; determinant(matNxN<f32>) returns f32.
// =============================================================================

TEST(TransposeDeterminantTest, Transpose_Mat2x2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let m = mat2x2<f32>(vec2<f32>(1.0, 2.0), vec2<f32>(3.0, 4.0));
    let t = transpose(m);
    let val = t[1][0];
    return vec4<f32>(val, val, val, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TransposeDeterminantTest, Transpose_Mat3x3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let m = mat3x3<f32>(
        vec3<f32>(1.0, 2.0, 3.0),
        vec3<f32>(4.0, 5.0, 6.0),
        vec3<f32>(7.0, 8.0, 9.0)
    );
    let t = transpose(m);
    let col = t[0];
    return vec4<f32>(col.x, col.y, col.z, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TransposeDeterminantTest, Transpose_Mat4x4) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let m = mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 2.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 3.0, 0.0),
        vec4<f32>(5.0, 6.0, 7.0, 1.0)
    );
    let t = transpose(m);
    let val = t[3][0];
    return vec4<f32>(val, t[3][1], t[3][2], 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TransposeDeterminantTest, Determinant_Mat2x2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let m = mat2x2<f32>(vec2<f32>(3.0, 1.0), vec2<f32>(2.0, 4.0));
    let d = determinant(m);
    return vec4<f32>(d, d, d, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TransposeDeterminantTest, Determinant_Mat3x3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let m = mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 2.0, 0.0),
        vec3<f32>(0.0, 0.0, 3.0)
    );
    let d = determinant(m);
    return vec4<f32>(d, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TransposeDeterminantTest, Determinant_Mat4x4) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let m = mat4x4<f32>(
        vec4<f32>(2.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 3.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 4.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 5.0)
    );
    let d = determinant(m);
    return vec4<f32>(d, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}
