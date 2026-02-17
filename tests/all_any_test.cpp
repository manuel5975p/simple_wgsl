#include <gtest/gtest.h>
#include "test_utils.h"

// =============================================================================
// all() and any() builtin functions
// SPIR-V: all -> SpvOpAll, any -> SpvOpAny
// Both take vec<bool> and return bool.
// =============================================================================

TEST(AllAnyTest, All_Vec3Bool) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<bool>(true, true, true);
    if all(v) {
        return vec4<f32>(0.0, 1.0, 0.0, 1.0);
    }
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(AllAnyTest, Any_Vec3Bool) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec3<bool>(false, true, false);
    if any(v) {
        return vec4<f32>(0.0, 1.0, 0.0, 1.0);
    }
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(AllAnyTest, All_Vec4Bool) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec4<bool>(true, false, true, true);
    let result = all(v);
    if result {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}
