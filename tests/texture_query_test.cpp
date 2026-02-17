#include <gtest/gtest.h>
#include "test_utils.h"

// =============================================================================
// Texture query and gather built-in functions
// textureDimensions, textureNumLayers, textureNumLevels, textureNumSamples,
// textureGather, textureGatherCompare
// =============================================================================

// -----------------------------------------------------------------------------
// textureDimensions(t) / textureDimensions(t, level)
// Returns the dimensions of the texture.
// texture_2d<f32> -> vec2<u32>, texture_3d<f32> -> vec3<u32>
// SPIR-V: OpImageQuerySize / OpImageQuerySizeLod
// -----------------------------------------------------------------------------

TEST(TextureQueryTest, TextureDimensions_2D_NoLevel) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@group(0) @binding(1) var t: texture_2d<f32>;
@compute @workgroup_size(1) fn main() {
    let dims = textureDimensions(t);
    output.data[0] = dims.x;
    output.data[1] = dims.y;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TextureQueryTest, TextureDimensions_2D_WithLevel) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@group(0) @binding(1) var t: texture_2d<f32>;
@compute @workgroup_size(1) fn main() {
    let dims = textureDimensions(t, 0);
    output.data[0] = dims.x;
    output.data[1] = dims.y;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TextureQueryTest, TextureDimensions_3D) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@group(0) @binding(1) var t: texture_3d<f32>;
@compute @workgroup_size(1) fn main() {
    let dims = textureDimensions(t);
    output.data[0] = dims.x;
    output.data[1] = dims.y;
    output.data[2] = dims.z;
})");
    EXPECT_TRUE(r.success) << r.error;
}

// -----------------------------------------------------------------------------
// textureNumLayers(t)
// Returns the number of layers in an array texture.
// texture_2d_array<f32> -> u32
// SPIR-V: OpImageQuerySize (array dimension component)
// -----------------------------------------------------------------------------

TEST(TextureQueryTest, TextureNumLayers_2DArray) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@group(0) @binding(1) var t: texture_2d_array<f32>;
@compute @workgroup_size(1) fn main() {
    output.data[0] = textureNumLayers(t);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TextureQueryTest, TextureNumLayers_2DArray_StoreToMultipleSlots) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@group(0) @binding(1) var t: texture_2d_array<f32>;
@compute @workgroup_size(1) fn main() {
    let layers = textureNumLayers(t);
    output.data[0] = layers;
    output.data[1] = layers + 1u;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TextureQueryTest, TextureNumLayers_2DArray_WithDimensions) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@group(0) @binding(1) var t: texture_2d_array<f32>;
@compute @workgroup_size(1) fn main() {
    let layers = textureNumLayers(t);
    let dims = textureDimensions(t);
    output.data[0] = dims.x;
    output.data[1] = dims.y;
    output.data[2] = layers;
})");
    EXPECT_TRUE(r.success) << r.error;
}

// -----------------------------------------------------------------------------
// textureNumLevels(t)
// Returns the number of mip levels in the texture.
// texture_2d<f32> -> u32
// SPIR-V: OpImageQueryLevels
// -----------------------------------------------------------------------------

TEST(TextureQueryTest, TextureNumLevels_2D) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@group(0) @binding(1) var t: texture_2d<f32>;
@compute @workgroup_size(1) fn main() {
    output.data[0] = textureNumLevels(t);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TextureQueryTest, TextureNumLevels_2D_StoreToMultipleSlots) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@group(0) @binding(1) var t: texture_2d<f32>;
@compute @workgroup_size(1) fn main() {
    let levels = textureNumLevels(t);
    output.data[0] = levels;
    output.data[1] = levels + 1u;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TextureQueryTest, TextureNumLevels_2D_WithDimensions) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@group(0) @binding(1) var t: texture_2d<f32>;
@compute @workgroup_size(1) fn main() {
    let levels = textureNumLevels(t);
    let dims = textureDimensions(t);
    output.data[0] = dims.x;
    output.data[1] = dims.y;
    output.data[2] = levels;
})");
    EXPECT_TRUE(r.success) << r.error;
}

// -----------------------------------------------------------------------------
// textureNumSamples(t)
// Returns the number of samples per texel in a multisampled texture.
// texture_multisampled_2d<f32> -> u32
// SPIR-V: OpImageQuerySamples
// -----------------------------------------------------------------------------

TEST(TextureQueryTest, TextureNumSamples_Multisampled2D) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@group(0) @binding(1) var t: texture_multisampled_2d<f32>;
@compute @workgroup_size(1) fn main() {
    output.data[0] = textureNumSamples(t);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TextureQueryTest, TextureNumSamples_Multisampled2D_StoreToMultipleSlots) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@group(0) @binding(1) var t: texture_multisampled_2d<f32>;
@compute @workgroup_size(1) fn main() {
    let samples = textureNumSamples(t);
    output.data[0] = samples;
    output.data[1] = samples + 1u;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TextureQueryTest, TextureNumSamples_Multisampled2D_WithDimensions) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@group(0) @binding(1) var t: texture_multisampled_2d<f32>;
@compute @workgroup_size(1) fn main() {
    let samples = textureNumSamples(t);
    let dims = textureDimensions(t);
    output.data[0] = dims.x;
    output.data[1] = dims.y;
    output.data[2] = samples;
})");
    EXPECT_TRUE(r.success) << r.error;
}

// -----------------------------------------------------------------------------
// textureGather(component, t, s, coord)
// Gathers a single component from 4 texels at the given coordinates.
// texture_2d<f32> + sampler -> vec4<f32>
// SPIR-V: OpImageGather
// -----------------------------------------------------------------------------

TEST(TextureQueryTest, TextureGather_2D_Component0) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@group(0) @binding(1) var t: texture_2d<f32>;
@group(0) @binding(2) var s: sampler;
@compute @workgroup_size(1) fn main() {
    let result = textureGather(0, t, s, vec2<f32>(0.5, 0.5));
    output.data[0] = bitcast<u32>(result.x);
    output.data[1] = bitcast<u32>(result.y);
    output.data[2] = bitcast<u32>(result.z);
    output.data[3] = bitcast<u32>(result.w);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TextureQueryTest, TextureGather_2D_Component1) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@group(0) @binding(1) var t: texture_2d<f32>;
@group(0) @binding(2) var s: sampler;
@compute @workgroup_size(1) fn main() {
    let result = textureGather(1, t, s, vec2<f32>(0.25, 0.75));
    output.data[0] = bitcast<u32>(result.x);
    output.data[1] = bitcast<u32>(result.y);
    output.data[2] = bitcast<u32>(result.z);
    output.data[3] = bitcast<u32>(result.w);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TextureQueryTest, TextureGather_2D_Component2) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@group(0) @binding(1) var t: texture_2d<f32>;
@group(0) @binding(2) var s: sampler;
@compute @workgroup_size(1) fn main() {
    let result = textureGather(2, t, s, vec2<f32>(0.0, 0.0));
    output.data[0] = bitcast<u32>(result.x);
    output.data[1] = bitcast<u32>(result.y);
    output.data[2] = bitcast<u32>(result.z);
    output.data[3] = bitcast<u32>(result.w);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// -----------------------------------------------------------------------------
// textureGatherCompare(t, s, coord, depthRef)
// Gathers depth comparison results from 4 texels.
// texture_depth_2d + sampler_comparison -> vec4<f32>
// SPIR-V: OpImageDrefGather
// -----------------------------------------------------------------------------

TEST(TextureQueryTest, TextureGatherCompare_Depth2D_ZeroRef) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@group(0) @binding(1) var t: texture_depth_2d;
@group(0) @binding(2) var s: sampler_comparison;
@compute @workgroup_size(1) fn main() {
    let result = textureGatherCompare(t, s, vec2<f32>(0.5, 0.5), 0.0);
    output.data[0] = bitcast<u32>(result.x);
    output.data[1] = bitcast<u32>(result.y);
    output.data[2] = bitcast<u32>(result.z);
    output.data[3] = bitcast<u32>(result.w);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TextureQueryTest, TextureGatherCompare_Depth2D_HalfRef) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@group(0) @binding(1) var t: texture_depth_2d;
@group(0) @binding(2) var s: sampler_comparison;
@compute @workgroup_size(1) fn main() {
    let result = textureGatherCompare(t, s, vec2<f32>(0.25, 0.75), 0.5);
    output.data[0] = bitcast<u32>(result.x);
    output.data[1] = bitcast<u32>(result.y);
    output.data[2] = bitcast<u32>(result.z);
    output.data[3] = bitcast<u32>(result.w);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TextureQueryTest, TextureGatherCompare_Depth2D_OneRef) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@group(0) @binding(1) var t: texture_depth_2d;
@group(0) @binding(2) var s: sampler_comparison;
@compute @workgroup_size(1) fn main() {
    let result = textureGatherCompare(t, s, vec2<f32>(0.0, 0.0), 1.0);
    output.data[0] = bitcast<u32>(result.x);
    output.data[1] = bitcast<u32>(result.y);
    output.data[2] = bitcast<u32>(result.z);
    output.data[3] = bitcast<u32>(result.w);
})");
    EXPECT_TRUE(r.success) << r.error;
}
