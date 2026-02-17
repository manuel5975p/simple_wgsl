#include <gtest/gtest.h>
#include "test_utils.h"

// =============================================================================
// faceForward() and refract() builtin functions
// SPIR-V: faceForward -> GLSLstd450FaceForward, refract -> GLSLstd450Refract
// faceForward(N, I, Nref) returns N if dot(Nref, I) < 0, else -N.
// refract(I, N, eta) computes the refraction vector given incident, normal, and
// index of refraction ratio.
// =============================================================================

TEST(FaceForwardRefractTest, FaceForward_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let N = vec3<f32>(0.0, 1.0, 0.0);
    let I = vec3<f32>(0.0, -1.0, 0.0);
    let Nref = vec3<f32>(0.0, 1.0, 0.0);
    let result = faceForward(N, I, Nref);
    return vec4<f32>(result.x, result.y, result.z, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(FaceForwardRefractTest, FaceForward_Vec4) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let N = vec4<f32>(0.0, 0.0, 1.0, 0.0);
    let I = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    let Nref = vec4<f32>(0.0, 0.0, -1.0, 0.0);
    let result = faceForward(N, I, Nref);
    return result;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(FaceForwardRefractTest, FaceForward_Vec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let N = vec2<f32>(1.0, 0.0);
    let I = vec2<f32>(-1.0, 0.0);
    let Nref = vec2<f32>(1.0, 0.0);
    let result = faceForward(N, I, Nref);
    return vec4<f32>(result.x, result.y, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(FaceForwardRefractTest, Refract_Vec3) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let I = vec3<f32>(0.707, -0.707, 0.0);
    let N = vec3<f32>(0.0, 1.0, 0.0);
    let eta = 0.5;
    let result = refract(I, N, eta);
    return vec4<f32>(result.x, result.y, result.z, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(FaceForwardRefractTest, Refract_Vec4) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let I = vec4<f32>(0.0, -1.0, 0.0, 0.0);
    let N = vec4<f32>(0.0, 1.0, 0.0, 0.0);
    let eta = 1.5;
    let result = refract(I, N, eta);
    return result;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(FaceForwardRefractTest, Refract_Vec2) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let I = vec2<f32>(0.707, -0.707);
    let N = vec2<f32>(0.0, 1.0);
    let eta = 0.9;
    let result = refract(I, N, eta);
    return vec4<f32>(result.x, result.y, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}
