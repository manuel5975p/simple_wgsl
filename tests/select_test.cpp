#include <gtest/gtest.h>
#include "test_utils.h"

TEST(SelectTest, Select_ScalarF32) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = 0.0;
    let b = 1.0;
    let cond = true;
    let result = select(a, b, cond);
    return vec4<f32>(result, result, result, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SelectTest, Select_Vec3F32) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec3<f32>(0.0, 0.0, 0.0);
    let b = vec3<f32>(1.0, 1.0, 1.0);
    let cond = false;
    let result = select(a, b, cond);
    return vec4<f32>(result.x, result.y, result.z, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SelectTest, Select_Vec3F32_VecBool) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let a = vec3<f32>(0.0, 0.0, 0.0);
    let b = vec3<f32>(1.0, 1.0, 1.0);
    let cond = vec3<bool>(true, false, true);
    let result = select(a, b, cond);
    return vec4<f32>(result.x, result.y, result.z, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}
