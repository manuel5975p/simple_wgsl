#include <gtest/gtest.h>
#include "test_utils.h"

TEST(FmaTest, Fma_ScalarF32) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a : f32 = 2.0;
    let b : f32 = 3.0;
    let c : f32 = 0.5;
    let result : f32 = fma(a, b, c);
    return vec4<f32>(result, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(FmaTest, Fma_Vec3F32) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec3<f32>(1.0, 2.0, 3.0);
    let b = vec3<f32>(4.0, 5.0, 6.0);
    let c = vec3<f32>(0.1, 0.2, 0.3);
    let result = fma(a, b, c);
    return vec4<f32>(result.x, result.y, result.z, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(FmaTest, Fma_Vec4F32) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    let b = vec4<f32>(0.5, 0.5, 0.5, 0.5);
    let c = vec4<f32>(0.1, 0.2, 0.3, 0.4);
    return fma(a, b, c);
})");
    EXPECT_TRUE(r.success) << r.error;
}
