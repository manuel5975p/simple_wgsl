#include <gtest/gtest.h>
#include "test_utils.h"

// ============================================================================
// dpdx
// ============================================================================

TEST(DerivativeTest, Dpdx_ScalarF32) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x : f32 = 1.0;
    let result : f32 = dpdx(x);
    return vec4<f32>(result, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(DerivativeTest, Dpdx_Vec2F32) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x = vec2<f32>(1.0, 2.0);
    let result = dpdx(x);
    return vec4<f32>(result.x, result.y, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(DerivativeTest, Dpdx_InComputation) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x : f32 = 3.0;
    let result : f32 = dpdx(x) * 2.0;
    return vec4<f32>(result, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// dpdxCoarse
// ============================================================================

TEST(DerivativeTest, DpdxCoarse_ScalarF32) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x : f32 = 1.0;
    let result : f32 = dpdxCoarse(x);
    return vec4<f32>(result, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(DerivativeTest, DpdxCoarse_Vec3F32) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x = vec3<f32>(1.0, 2.0, 3.0);
    let result = dpdxCoarse(x);
    return vec4<f32>(result.x, result.y, result.z, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(DerivativeTest, DpdxCoarse_InComputation) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x : f32 = 5.0;
    let result : f32 = dpdxCoarse(x) + 1.0;
    return vec4<f32>(result, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// dpdxFine
// ============================================================================

TEST(DerivativeTest, DpdxFine_ScalarF32) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x : f32 = 1.0;
    let result : f32 = dpdxFine(x);
    return vec4<f32>(result, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(DerivativeTest, DpdxFine_Vec2F32) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x = vec2<f32>(4.0, 5.0);
    let result = dpdxFine(x);
    return vec4<f32>(result.x, result.y, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(DerivativeTest, DpdxFine_InComputation) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x : f32 = 2.0;
    let result : f32 = dpdxFine(x) * 0.5;
    return vec4<f32>(result, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// dpdy
// ============================================================================

TEST(DerivativeTest, Dpdy_ScalarF32) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let y : f32 = 1.0;
    let result : f32 = dpdy(y);
    return vec4<f32>(result, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(DerivativeTest, Dpdy_Vec3F32) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let y = vec3<f32>(1.0, 2.0, 3.0);
    let result = dpdy(y);
    return vec4<f32>(result.x, result.y, result.z, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(DerivativeTest, Dpdy_InComputation) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let y : f32 = 7.0;
    let result : f32 = dpdy(y) * 3.0;
    return vec4<f32>(result, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// dpdyCoarse
// ============================================================================

TEST(DerivativeTest, DpdyCoarse_ScalarF32) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let y : f32 = 1.0;
    let result : f32 = dpdyCoarse(y);
    return vec4<f32>(result, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(DerivativeTest, DpdyCoarse_Vec2F32) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let y = vec2<f32>(2.0, 4.0);
    let result = dpdyCoarse(y);
    return vec4<f32>(result.x, result.y, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(DerivativeTest, DpdyCoarse_InComputation) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let y : f32 = 6.0;
    let result : f32 = dpdyCoarse(y) - 0.5;
    return vec4<f32>(result, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// dpdyFine
// ============================================================================

TEST(DerivativeTest, DpdyFine_ScalarF32) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let y : f32 = 1.0;
    let result : f32 = dpdyFine(y);
    return vec4<f32>(result, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(DerivativeTest, DpdyFine_Vec3F32) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let y = vec3<f32>(1.0, 3.0, 5.0);
    let result = dpdyFine(y);
    return vec4<f32>(result.x, result.y, result.z, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(DerivativeTest, DpdyFine_InComputation) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let y : f32 = 4.0;
    let result : f32 = dpdyFine(y) + 2.0;
    return vec4<f32>(result, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// fwidth
// ============================================================================

TEST(DerivativeTest, Fwidth_ScalarF32) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x : f32 = 1.0;
    let result : f32 = fwidth(x);
    return vec4<f32>(result, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(DerivativeTest, Fwidth_Vec2F32) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x = vec2<f32>(1.0, 2.0);
    let result = fwidth(x);
    return vec4<f32>(result.x, result.y, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(DerivativeTest, Fwidth_InComputation) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x : f32 = 8.0;
    let result : f32 = fwidth(x) * 0.25;
    return vec4<f32>(result, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// fwidthCoarse
// ============================================================================

TEST(DerivativeTest, FwidthCoarse_ScalarF32) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x : f32 = 1.0;
    let result : f32 = fwidthCoarse(x);
    return vec4<f32>(result, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(DerivativeTest, FwidthCoarse_Vec3F32) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x = vec3<f32>(2.0, 4.0, 6.0);
    let result = fwidthCoarse(x);
    return vec4<f32>(result.x, result.y, result.z, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(DerivativeTest, FwidthCoarse_InComputation) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x : f32 = 3.0;
    let result : f32 = fwidthCoarse(x) + 0.1;
    return vec4<f32>(result, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// fwidthFine
// ============================================================================

TEST(DerivativeTest, FwidthFine_ScalarF32) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x : f32 = 1.0;
    let result : f32 = fwidthFine(x);
    return vec4<f32>(result, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(DerivativeTest, FwidthFine_Vec2F32) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x = vec2<f32>(5.0, 10.0);
    let result = fwidthFine(x);
    return vec4<f32>(result.x, result.y, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(DerivativeTest, FwidthFine_InComputation) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let x : f32 = 9.0;
    let result : f32 = fwidthFine(x) * 4.0;
    return vec4<f32>(result, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}
