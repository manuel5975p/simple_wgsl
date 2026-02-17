#include <gtest/gtest.h>
#include "test_utils.h"

TEST(SaturateTest, Saturate_ScalarF32) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let above = saturate(1.5);
    let below = saturate(-0.5);
    return vec4<f32>(above, below, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SaturateTest, Saturate_Vec3F32) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<f32>(-1.0, 0.5, 2.0);
    let result = saturate(v);
    return vec4<f32>(result.x, result.y, result.z, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SaturateTest, Saturate_Vec4F32) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<f32>(-0.3, 0.7, 1.5, 0.0);
    return saturate(v);
})");
    EXPECT_TRUE(r.success) << r.error;
}
