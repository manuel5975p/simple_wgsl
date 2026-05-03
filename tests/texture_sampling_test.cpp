#include <gtest/gtest.h>
#include "test_utils.h"

namespace {

std::string DisassembleSpirvForTextureTest(const std::vector<uint32_t> &spirv) {
    std::string spv_path = wgsl_test::MakeTempSpvPath("wgsl_tex");
    if (!wgsl_test::WriteSpirvFile(spv_path, spirv.data(), spirv.size())) return "";

    std::string output;
    int ret = wgsl_test::RunCommand("spirv-dis --no-header " + spv_path + " 2>&1", &output);
    std::remove(spv_path.c_str());
    if (ret != 0) return "";
    return output;
}

int CountSubstr(const std::string &s, const char *needle) {
    int count = 0;
    size_t pos = 0;
    while ((pos = s.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += std::strlen(needle);
    }
    return count;
}

} // namespace

// =============================================================================
// textureSampleLevel
// =============================================================================

TEST(TextureSamplingTest, TextureSampleLevel_Basic) {
    auto r = wgsl_test::CompileWgsl(R"(
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSampleLevel(t, s, uv, 0.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TextureSamplingTest, TextureSampleLevel_ExplicitMipLevel) {
    auto r = wgsl_test::CompileWgsl(R"(
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let level0 = textureSampleLevel(t, s, uv, 0.0);
    let level3 = textureSampleLevel(t, s, uv, 3.0);
    return level0 + level3;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TextureSamplingTest, TextureSampleLevel_InVertexShader) {
    auto r = wgsl_test::CompileWgsl(R"(
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

@vertex fn main(@location(0) pos: vec2<f32>) -> @builtin(position) vec4<f32> {
    let color = textureSampleLevel(t, s, pos, 0.0);
    return vec4<f32>(pos, color.r, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// textureSampleBias
// =============================================================================

TEST(TextureSamplingTest, TextureSampleBias_Basic) {
    auto r = wgsl_test::CompileWgsl(R"(
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSampleBias(t, s, uv, 2.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TextureSamplingTest, TextureSampleBias_NegativeBias) {
    auto r = wgsl_test::CompileWgsl(R"(
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSampleBias(t, s, uv, -1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TextureSamplingTest, TextureSampleBias_WithComputation) {
    auto r = wgsl_test::CompileWgsl(R"(
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let bias = clamp(uv.x * 4.0, -1.0, 3.0);
    let color = textureSampleBias(t, s, uv, bias);
    return color;
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// textureSampleGrad
// =============================================================================

TEST(TextureSamplingTest, TextureSampleGrad_Basic) {
    auto r = wgsl_test::CompileWgsl(R"(
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let ddx = vec2<f32>(0.1, 0.0);
    let ddy = vec2<f32>(0.0, 0.1);
    return textureSampleGrad(t, s, uv, ddx, ddy);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TextureSamplingTest, TextureSampleGrad_AnisotropicDerivatives) {
    auto r = wgsl_test::CompileWgsl(R"(
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let ddx = vec2<f32>(0.5, 0.1);
    let ddy = vec2<f32>(0.1, 0.5);
    return textureSampleGrad(t, s, uv, ddx, ddy);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TextureSamplingTest, TextureSampleGrad_ComputedDerivatives) {
    auto r = wgsl_test::CompileWgsl(R"(
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let scale = 2.0;
    let ddx = vec2<f32>(scale * 0.01, 0.0);
    let ddy = vec2<f32>(0.0, scale * 0.01);
    let color = textureSampleGrad(t, s, uv * scale, ddx, ddy);
    return color;
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// textureSampleCompare
// =============================================================================

TEST(TextureSamplingTest, TextureSampleCompare_Basic) {
    auto r = wgsl_test::CompileWgsl(R"(
@group(0) @binding(0) var t: texture_depth_2d;
@group(0) @binding(1) var s: sampler_comparison;

@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let depth = textureSampleCompare(t, s, uv, 0.5);
    return vec4<f32>(depth, depth, depth, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TextureSamplingTest, TextureSampleCompare_ShadowMapping) {
    auto r = wgsl_test::CompileWgsl(R"(
@group(0) @binding(0) var shadowMap: texture_depth_2d;
@group(0) @binding(1) var shadowSampler: sampler_comparison;

@fragment fn main(@location(0) shadowCoord: vec3<f32>) -> @location(0) vec4<f32> {
    let visibility = textureSampleCompare(shadowMap, shadowSampler, shadowCoord.xy, shadowCoord.z);
    return vec4<f32>(visibility, visibility, visibility, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TextureSamplingTest, TextureSampleCompare_WithAttenuation) {
    auto r = wgsl_test::CompileWgsl(R"(
@group(0) @binding(0) var depthTex: texture_depth_2d;
@group(0) @binding(1) var cmpSampler: sampler_comparison;

@fragment fn main(@location(0) uv: vec2<f32>, @location(1) refDepth: f32) -> @location(0) vec4<f32> {
    let shadow = textureSampleCompare(depthTex, cmpSampler, uv, refDepth);
    let lit = mix(0.3, 1.0, shadow);
    return vec4<f32>(lit, lit, lit, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// textureSampleCompareLevel
// =============================================================================

TEST(TextureSamplingTest, TextureSampleCompareLevel_Basic) {
    auto r = wgsl_test::CompileWgsl(R"(
@group(0) @binding(0) var t: texture_depth_2d;
@group(0) @binding(1) var s: sampler_comparison;

@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let depth = textureSampleCompareLevel(t, s, uv, 0.5);
    return vec4<f32>(depth, depth, depth, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TextureSamplingTest, TextureSampleCompareLevel_InVertexShader) {
    auto r = wgsl_test::CompileWgsl(R"(
@group(0) @binding(0) var t: texture_depth_2d;
@group(0) @binding(1) var s: sampler_comparison;

@vertex fn main(@location(0) pos: vec2<f32>) -> @builtin(position) vec4<f32> {
    let shadow = textureSampleCompareLevel(t, s, pos, 0.5);
    return vec4<f32>(pos, shadow, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TextureSamplingTest, TextureSampleCompareLevel_WithBranching) {
    auto r = wgsl_test::CompileWgsl(R"(
@group(0) @binding(0) var t: texture_depth_2d;
@group(0) @binding(1) var s: sampler_comparison;

@fragment fn main(@location(0) uv: vec2<f32>, @location(1) refDepth: f32) -> @location(0) vec4<f32> {
    let shadow = textureSampleCompareLevel(t, s, uv, refDepth);
    if (shadow > 0.5) {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }
    return vec4<f32>(0.2, 0.2, 0.2, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// textureLoad
// =============================================================================

TEST(TextureSamplingTest, TextureLoad_Basic) {
    auto r = wgsl_test::CompileWgsl(R"(
@group(0) @binding(0) var t: texture_2d<f32>;

@fragment fn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(i32(pos.x), i32(pos.y));
    return textureLoad(t, coord, 0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TextureSamplingTest, TextureLoad_MipLevel) {
    auto r = wgsl_test::CompileWgsl(R"(
@group(0) @binding(0) var t: texture_2d<f32>;

@fragment fn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(i32(pos.x), i32(pos.y));
    let level0 = textureLoad(t, coord, 0);
    let level1 = textureLoad(t, coord, 1);
    return level0 + level1;
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TextureSamplingTest, TextureLoad_ComputeShader) {
    auto r = wgsl_test::CompileWgsl(R"(
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let color = textureLoad(inputTex, coord, 0);
    textureStore(outputTex, vec2<u32>(gid.x, gid.y), color);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// =============================================================================
// textureStore
// =============================================================================

TEST(TextureSamplingTest, TextureStore_Basic) {
    auto r = wgsl_test::CompileWgsl(R"(
@group(0) @binding(0) var t: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    textureStore(t, vec2<u32>(gid.x, gid.y), vec4<f32>(1.0, 0.0, 0.0, 1.0));
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TextureSamplingTest, TextureStorage_FormatLowered) {
    auto r = wgsl_test::CompileWgsl(R"(
@group(0) @binding(0) var t: texture_storage_2d<r32float, write>;

@compute @workgroup_size(1) fn main() {
}
)");
    ASSERT_TRUE(r.success) << r.error;

    std::string spvasm = DisassembleSpirvForTextureTest(r.spirv);
    ASSERT_FALSE(spvasm.empty());
    EXPECT_NE(spvasm.find("R32f"), std::string::npos) << spvasm;
    EXPECT_EQ(spvasm.find("Rgba8"), std::string::npos) << spvasm;
}

TEST(TextureSamplingTest, TextureStorage_AllDimensionsFormatLowered) {
    auto r = wgsl_test::CompileWgsl(R"(
@group(0) @binding(0) var tex1d: texture_storage_1d<r32float, write>;
@group(0) @binding(1) var tex2d: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var tex2dArray: texture_storage_2d_array<r32float, write>;
@group(0) @binding(3) var tex3d: texture_storage_3d<r32float, write>;

@compute @workgroup_size(1) fn main() {
}
)");
    ASSERT_TRUE(r.success) << r.error;

    std::string spvasm = DisassembleSpirvForTextureTest(r.spirv);
    ASSERT_FALSE(spvasm.empty());
    EXPECT_NE(spvasm.find("1D 0 0 0 2 R32f"), std::string::npos) << spvasm;
    EXPECT_NE(spvasm.find("2D 0 0 0 2 R32f"), std::string::npos) << spvasm;
    EXPECT_NE(spvasm.find("2D 0 1 0 2 R32f"), std::string::npos) << spvasm;
    EXPECT_NE(spvasm.find("3D 0 0 0 2 R32f"), std::string::npos) << spvasm;
    EXPECT_EQ(spvasm.find("Rgba8"), std::string::npos) << spvasm;
}

TEST(TextureSamplingTest, TextureStorage_IntegerFormatsUseIntegerImageTypes) {
    auto r = wgsl_test::CompileWgsl(R"(
@group(0) @binding(0) var uintTex: texture_storage_3d<r32uint, write>;
@group(0) @binding(1) var sintTex: texture_storage_3d<r32sint, write>;
@group(0) @binding(2) var rgbaUintTex: texture_storage_2d_array<rgba8uint, write>;
@group(0) @binding(3) var rgbaSintTex: texture_storage_2d_array<rgba8sint, write>;

@compute @workgroup_size(1) fn main() {
}
)");
    ASSERT_TRUE(r.success) << r.error;

    std::string spvasm = DisassembleSpirvForTextureTest(r.spirv);
    ASSERT_FALSE(spvasm.empty());
    EXPECT_NE(spvasm.find("R32ui"), std::string::npos) << spvasm;
    EXPECT_NE(spvasm.find("R32i"), std::string::npos) << spvasm;
    EXPECT_NE(spvasm.find("Rgba8ui"), std::string::npos) << spvasm;
    EXPECT_NE(spvasm.find("Rgba8i"), std::string::npos) << spvasm;
    EXPECT_NE(spvasm.find("OpTypeImage %uint 3D 0 0 0 2 R32ui"), std::string::npos) << spvasm;
    EXPECT_NE(spvasm.find("OpTypeImage %int 3D 0 0 0 2 R32i"), std::string::npos) << spvasm;
}

TEST(TextureSamplingTest, TextureStorage_AdditionalFormatsLowered) {
    auto r = wgsl_test::CompileWgsl(R"(
@group(0) @binding(0) var a: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var b: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var c: texture_storage_2d<rg32float, write>;
@group(0) @binding(3) var d: texture_storage_2d<rgba8snorm, write>;
@group(0) @binding(4) var e: texture_storage_2d<rg16uint, write>;
@group(0) @binding(5) var f: texture_storage_2d<r8sint, write>;
@group(0) @binding(6) var g: texture_storage_2d<rgb10a2uint, write>;
@group(0) @binding(7) var h: texture_storage_2d<rgba16unorm, write>;

@compute @workgroup_size(1) fn main() {
}
)");
    ASSERT_TRUE(r.success) << r.error;

    std::string spvasm = DisassembleSpirvForTextureTest(r.spirv);
    ASSERT_FALSE(spvasm.empty());
    EXPECT_NE(spvasm.find("Rgba16f"), std::string::npos) << spvasm;
    EXPECT_NE(spvasm.find("Rgba32f"), std::string::npos) << spvasm;
    EXPECT_NE(spvasm.find("Rg32f"), std::string::npos) << spvasm;
    EXPECT_NE(spvasm.find("Rgba8Snorm"), std::string::npos) << spvasm;
    EXPECT_NE(spvasm.find("Rg16ui"), std::string::npos) << spvasm;
    EXPECT_NE(spvasm.find("R8i"), std::string::npos) << spvasm;
    EXPECT_NE(spvasm.find("Rgb10a2ui"), std::string::npos) << spvasm;
    EXPECT_NE(spvasm.find("Rgba16"), std::string::npos) << spvasm;
}

TEST(TextureSamplingTest, TextureStorage_AccessDecorationsLowered) {
    auto r = wgsl_test::CompileWgsl(R"(
@group(0) @binding(0) var readTex: texture_storage_2d<rgba8unorm, read>;
@group(0) @binding(1) var writeTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var rwTex: texture_storage_2d<rgba8unorm, read_write>;

@compute @workgroup_size(1) fn main() {
}
)");
    ASSERT_TRUE(r.success) << r.error;

    std::string spvasm = DisassembleSpirvForTextureTest(r.spirv);
    ASSERT_FALSE(spvasm.empty());
    EXPECT_EQ(CountSubstr(spvasm, "NonWritable"), 1) << spvasm;
    EXPECT_EQ(CountSubstr(spvasm, "NonReadable"), 1) << spvasm;
}

TEST(TextureSamplingTest, TextureStorage_AccessDecorationsAllDimensions) {
    auto r = wgsl_test::CompileWgsl(R"(
@group(0) @binding(0) var read1d: texture_storage_1d<rgba8unorm, read>;
@group(0) @binding(1) var write2dArray: texture_storage_2d_array<rgba8unorm, write>;
@group(0) @binding(2) var rw3d: texture_storage_3d<rgba8unorm, read_write>;

@compute @workgroup_size(1) fn main() {
}
)");
    ASSERT_TRUE(r.success) << r.error;

    std::string spvasm = DisassembleSpirvForTextureTest(r.spirv);
    ASSERT_FALSE(spvasm.empty());
    EXPECT_EQ(CountSubstr(spvasm, "NonWritable"), 1) << spvasm;
    EXPECT_EQ(CountSubstr(spvasm, "NonReadable"), 1) << spvasm;
}

TEST(TextureSamplingTest, TextureStorage_BindingArrayAccessDecorated) {
    auto r = wgsl_test::CompileWgsl(R"(
@group(0) @binding(0) var readTextures: binding_array<texture_storage_2d<rgba8unorm, read>>;
@group(0) @binding(1) var writeTextures: binding_array<texture_storage_2d<rgba8unorm, write>>;

@compute @workgroup_size(1) fn main() {
}
)");
    ASSERT_TRUE(r.success) << r.error;

    std::string spvasm = DisassembleSpirvForTextureTest(r.spirv);
    ASSERT_FALSE(spvasm.empty());
    EXPECT_EQ(CountSubstr(spvasm, "NonWritable"), 1) << spvasm;
    EXPECT_EQ(CountSubstr(spvasm, "NonReadable"), 1) << spvasm;
}

TEST(TextureSamplingTest, TextureStore_1D) {
    auto r = wgsl_test::CompileWgsl(R"(
@group(0) @binding(0) var t: texture_storage_1d<rgba8unorm, write>;

@compute @workgroup_size(1) fn main() {
    textureStore(t, 0i, vec4<f32>(1.0, 0.0, 0.0, 1.0));
}
)");
    ASSERT_TRUE(r.success) << r.error;

    std::string spvasm = DisassembleSpirvForTextureTest(r.spirv);
    ASSERT_FALSE(spvasm.empty());
    EXPECT_NE(spvasm.find("ImageWrite"), std::string::npos) << spvasm;
    EXPECT_NE(spvasm.find("1D 0 0 0 2 Rgba8"), std::string::npos) << spvasm;
}

TEST(TextureSamplingTest, TextureStore_3D) {
    auto r = wgsl_test::CompileWgsl(R"(
@group(0) @binding(0) var t: texture_storage_3d<rgba8unorm, write>;

@compute @workgroup_size(1) fn main() {
    textureStore(t, vec3<i32>(0, 0, 0), vec4<f32>(1.0, 0.0, 0.0, 1.0));
}
)");
    ASSERT_TRUE(r.success) << r.error;

    std::string spvasm = DisassembleSpirvForTextureTest(r.spirv);
    ASSERT_FALSE(spvasm.empty());
    EXPECT_NE(spvasm.find("ImageWrite"), std::string::npos) << spvasm;
    EXPECT_NE(spvasm.find("3D 0 0 0 2 Rgba8"), std::string::npos) << spvasm;
}

TEST(TextureSamplingTest, TextureLoad_Storage1DRead) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@group(0) @binding(1) var t: texture_storage_1d<r32uint, read>;

@compute @workgroup_size(1) fn main() {
    let value = textureLoad(t, 0i);
    output.data[0] = value.x;
}
)");
    ASSERT_TRUE(r.success) << r.error;

    std::string spvasm = DisassembleSpirvForTextureTest(r.spirv);
    ASSERT_FALSE(spvasm.empty());
    EXPECT_NE(spvasm.find("ImageRead"), std::string::npos) << spvasm;
    EXPECT_NE(spvasm.find("1D 0 0 0 2 R32ui"), std::string::npos) << spvasm;
}

TEST(TextureSamplingTest, TextureLoad_Storage3DRead) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Buf { data: array<u32> };
@group(0) @binding(0) var<storage, read_write> output: Buf;
@group(0) @binding(1) var t: texture_storage_3d<rgba8uint, read>;

@compute @workgroup_size(1) fn main() {
    let value = textureLoad(t, vec3<i32>(0, 0, 0));
    output.data[0] = value.x;
}
)");
    ASSERT_TRUE(r.success) << r.error;

    std::string spvasm = DisassembleSpirvForTextureTest(r.spirv);
    ASSERT_FALSE(spvasm.empty());
    EXPECT_NE(spvasm.find("ImageRead"), std::string::npos) << spvasm;
    EXPECT_NE(spvasm.find("3D 0 0 0 2 Rgba8ui"), std::string::npos) << spvasm;
}

TEST(TextureSamplingTest, TextureLoadAndStore_Storage3DReadWrite) {
    auto r = wgsl_test::CompileWgsl(R"(
@group(0) @binding(0) var t: texture_storage_3d<rgba8unorm, read_write>;

@compute @workgroup_size(1) fn main() {
    let coord = vec3<i32>(0, 0, 0);
    let value = textureLoad(t, coord);
    textureStore(t, coord, value);
}
)");
    ASSERT_TRUE(r.success) << r.error;

    std::string spvasm = DisassembleSpirvForTextureTest(r.spirv);
    ASSERT_FALSE(spvasm.empty());
    EXPECT_NE(spvasm.find("ImageRead"), std::string::npos) << spvasm;
    EXPECT_NE(spvasm.find("ImageWrite"), std::string::npos) << spvasm;
    EXPECT_EQ(spvasm.find("NonWritable"), std::string::npos) << spvasm;
    EXPECT_EQ(spvasm.find("NonReadable"), std::string::npos) << spvasm;
}

TEST(TextureSamplingTest, TextureStore_ComputedValue) {
    auto r = wgsl_test::CompileWgsl(R"(
@group(0) @binding(0) var t: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let uv = vec2<f32>(f32(gid.x) / 256.0, f32(gid.y) / 256.0);
    let color = vec4<f32>(uv.x, uv.y, 0.0, 1.0);
    textureStore(t, vec2<u32>(gid.x, gid.y), color);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(TextureSamplingTest, TextureStore_MultipleWrites) {
    auto r = wgsl_test::CompileWgsl(R"(
@group(0) @binding(0) var outA: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var outB: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coord = vec2<u32>(gid.x, gid.y);
    textureStore(outA, coord, vec4<f32>(1.0, 0.0, 0.0, 1.0));
    textureStore(outB, coord, vec4<f32>(0.0, 0.0, 1.0, 1.0));
})");
    EXPECT_TRUE(r.success) << r.error;
}
