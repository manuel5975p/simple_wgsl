#include <gtest/gtest.h>
#include "test_utils.h"

// =============================================================================
// Pack/Unpack builtin functions
// pack4x8snorm, pack4x8unorm, pack2x16snorm, pack2x16unorm, pack2x16float
// unpack4x8snorm, unpack4x8unorm, unpack2x16snorm, unpack2x16unorm, unpack2x16float
// =============================================================================

// -----------------------------------------------------------------------------
// pack4x8snorm(v: vec4<f32>) -> u32
// Packs four f32 values in [-1,1] into four signed normalized 8-bit integers.
// SPIR-V: GLSLstd450PackSnorm4x8
// -----------------------------------------------------------------------------

TEST(PackUnpackTest, Pack4x8snorm_BasicValues) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let packed = pack4x8snorm(vec4<f32>(0.0, 1.0, -1.0, 0.5));
    let f = f32(packed);
    return vec4<f32>(f, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(PackUnpackTest, Pack4x8snorm_AllZeros) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    output.data[0] = pack4x8snorm(vec4<f32>(0.0, 0.0, 0.0, 0.0));
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(PackUnpackTest, Pack4x8snorm_NegativeValues) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    output.data[0] = pack4x8snorm(vec4<f32>(-1.0, -0.5, -0.25, -0.75));
})");
    EXPECT_TRUE(r.success) << r.error;
}

// -----------------------------------------------------------------------------
// pack4x8unorm(v: vec4<f32>) -> u32
// Packs four f32 values in [0,1] into four unsigned normalized 8-bit integers.
// SPIR-V: GLSLstd450PackUnorm4x8
// -----------------------------------------------------------------------------

TEST(PackUnpackTest, Pack4x8unorm_BasicValues) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let packed = pack4x8unorm(vec4<f32>(0.5, 0.25, 0.75, 1.0));
    let f = f32(packed);
    return vec4<f32>(f, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(PackUnpackTest, Pack4x8unorm_AllZeros) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    output.data[0] = pack4x8unorm(vec4<f32>(0.0, 0.0, 0.0, 0.0));
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(PackUnpackTest, Pack4x8unorm_AllOnes) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    output.data[0] = pack4x8unorm(vec4<f32>(1.0, 1.0, 1.0, 1.0));
})");
    EXPECT_TRUE(r.success) << r.error;
}

// -----------------------------------------------------------------------------
// pack2x16snorm(v: vec2<f32>) -> u32
// Packs two f32 values in [-1,1] into two signed normalized 16-bit integers.
// SPIR-V: GLSLstd450PackSnorm2x16
// -----------------------------------------------------------------------------

TEST(PackUnpackTest, Pack2x16snorm_BasicValues) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let packed = pack2x16snorm(vec2<f32>(0.5, -0.5));
    let f = f32(packed);
    return vec4<f32>(f, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(PackUnpackTest, Pack2x16snorm_Extremes) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    output.data[0] = pack2x16snorm(vec2<f32>(1.0, -1.0));
    output.data[1] = pack2x16snorm(vec2<f32>(-1.0, 1.0));
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(PackUnpackTest, Pack2x16snorm_Zeros) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    output.data[0] = pack2x16snorm(vec2<f32>(0.0, 0.0));
})");
    EXPECT_TRUE(r.success) << r.error;
}

// -----------------------------------------------------------------------------
// pack2x16unorm(v: vec2<f32>) -> u32
// Packs two f32 values in [0,1] into two unsigned normalized 16-bit integers.
// SPIR-V: GLSLstd450PackUnorm2x16
// -----------------------------------------------------------------------------

TEST(PackUnpackTest, Pack2x16unorm_BasicValues) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let packed = pack2x16unorm(vec2<f32>(0.5, 0.75));
    let f = f32(packed);
    return vec4<f32>(f, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(PackUnpackTest, Pack2x16unorm_Extremes) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    output.data[0] = pack2x16unorm(vec2<f32>(0.0, 1.0));
    output.data[1] = pack2x16unorm(vec2<f32>(1.0, 0.0));
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(PackUnpackTest, Pack2x16unorm_AllOnes) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    output.data[0] = pack2x16unorm(vec2<f32>(1.0, 1.0));
})");
    EXPECT_TRUE(r.success) << r.error;
}

// -----------------------------------------------------------------------------
// pack2x16float(v: vec2<f32>) -> u32
// Packs two f32 values into two f16 values stored as a u32.
// SPIR-V: GLSLstd450PackHalf2x16
// -----------------------------------------------------------------------------

TEST(PackUnpackTest, Pack2x16float_BasicValues) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let packed = pack2x16float(vec2<f32>(1.0, 2.0));
    let f = f32(packed);
    return vec4<f32>(f, 0.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(PackUnpackTest, Pack2x16float_Zeros) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    output.data[0] = pack2x16float(vec2<f32>(0.0, 0.0));
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(PackUnpackTest, Pack2x16float_NegativeValues) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    output.data[0] = pack2x16float(vec2<f32>(-1.0, -0.5));
    output.data[1] = pack2x16float(vec2<f32>(3.14, -2.71));
})");
    EXPECT_TRUE(r.success) << r.error;
}

// -----------------------------------------------------------------------------
// unpack4x8snorm(u: u32) -> vec4<f32>
// Unpacks a u32 into four signed normalized f32 values in [-1,1].
// SPIR-V: GLSLstd450UnpackSnorm4x8
// -----------------------------------------------------------------------------

TEST(PackUnpackTest, Unpack4x8snorm_BasicValue) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = unpack4x8snorm(0x817F007Fu);
    return v;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(PackUnpackTest, Unpack4x8snorm_AllZeros) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    return unpack4x8snorm(0u);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(PackUnpackTest, Unpack4x8snorm_ComputeShader) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<f32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let v = unpack4x8snorm(0xFF80407Fu);
    output.data[0] = v.x;
    output.data[1] = v.y;
    output.data[2] = v.z;
    output.data[3] = v.w;
})");
    EXPECT_TRUE(r.success) << r.error;
}

// -----------------------------------------------------------------------------
// unpack4x8unorm(u: u32) -> vec4<f32>
// Unpacks a u32 into four unsigned normalized f32 values in [0,1].
// SPIR-V: GLSLstd450UnpackUnorm4x8
// -----------------------------------------------------------------------------

TEST(PackUnpackTest, Unpack4x8unorm_BasicValue) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = unpack4x8unorm(0xFF804000u);
    return v;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(PackUnpackTest, Unpack4x8unorm_AllOnes) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    return unpack4x8unorm(0xFFFFFFFFu);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(PackUnpackTest, Unpack4x8unorm_ComputeShader) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<f32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let v = unpack4x8unorm(0x00FF8040u);
    output.data[0] = v.x;
    output.data[1] = v.y;
    output.data[2] = v.z;
    output.data[3] = v.w;
})");
    EXPECT_TRUE(r.success) << r.error;
}

// -----------------------------------------------------------------------------
// unpack2x16snorm(u: u32) -> vec2<f32>
// Unpacks a u32 into two signed normalized f32 values in [-1,1].
// SPIR-V: GLSLstd450UnpackSnorm2x16
// -----------------------------------------------------------------------------

TEST(PackUnpackTest, Unpack2x16snorm_BasicValue) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = unpack2x16snorm(0x7FFF8001u);
    return vec4<f32>(v.x, v.y, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(PackUnpackTest, Unpack2x16snorm_Zeros) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = unpack2x16snorm(0u);
    return vec4<f32>(v.x, v.y, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(PackUnpackTest, Unpack2x16snorm_ComputeShader) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<f32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let v = unpack2x16snorm(0x40008000u);
    output.data[0] = v.x;
    output.data[1] = v.y;
})");
    EXPECT_TRUE(r.success) << r.error;
}

// -----------------------------------------------------------------------------
// unpack2x16unorm(u: u32) -> vec2<f32>
// Unpacks a u32 into two unsigned normalized f32 values in [0,1].
// SPIR-V: GLSLstd450UnpackUnorm2x16
// -----------------------------------------------------------------------------

TEST(PackUnpackTest, Unpack2x16unorm_BasicValue) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = unpack2x16unorm(0xFFFF0000u);
    return vec4<f32>(v.x, v.y, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(PackUnpackTest, Unpack2x16unorm_AllOnes) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = unpack2x16unorm(0xFFFFFFFFu);
    return vec4<f32>(v.x, v.y, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(PackUnpackTest, Unpack2x16unorm_ComputeShader) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<f32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let v = unpack2x16unorm(0x80004000u);
    output.data[0] = v.x;
    output.data[1] = v.y;
})");
    EXPECT_TRUE(r.success) << r.error;
}

// -----------------------------------------------------------------------------
// unpack2x16float(u: u32) -> vec2<f32>
// Unpacks a u32 into two f16 values converted to f32.
// SPIR-V: GLSLstd450UnpackHalf2x16
// -----------------------------------------------------------------------------

TEST(PackUnpackTest, Unpack2x16float_BasicValue) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    // 0x3C00 = f16 1.0, 0x4000 = f16 2.0
    let v = unpack2x16float(0x40003C00u);
    return vec4<f32>(v.x, v.y, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(PackUnpackTest, Unpack2x16float_Zeros) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = unpack2x16float(0u);
    return vec4<f32>(v.x, v.y, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(PackUnpackTest, Unpack2x16float_ComputeShader) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<f32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // 0xBC00 = f16 -1.0, 0x3800 = f16 0.5
    let v = unpack2x16float(0x3800BC00u);
    output.data[0] = v.x;
    output.data[1] = v.y;
})");
    EXPECT_TRUE(r.success) << r.error;
}
