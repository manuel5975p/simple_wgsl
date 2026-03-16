#include <gtest/gtest.h>
#include "test_utils.h"

// Regression tests for bugs discovered while compiling the Slug text
// rendering shaders (SlugVertexShader.wgsl, SlugPixelShader.wgsl).
// See slug_bugfix_insights.md for detailed descriptions of each bug.

// ============================================================================
// Bug 1: @interpolate(flat) on vertex shader outputs
// The vertex shader must emit OpDecorate Flat for outputs marked
// @interpolate(flat), especially for integer types (required by Vulkan).
// ============================================================================

TEST(SlugRegression, FlatInterpolation_VertexOutput) {
    auto r = wgsl_test::CompileWgsl(R"(
struct VertOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(flat) data: vec4<f32>,
    @location(2) @interpolate(flat) ids: vec4<i32>,
};
@vertex fn main(@location(0) pos: vec4<f32>) -> VertOut {
    var out: VertOut;
    out.position = pos;
    out.color = vec4<f32>(1.0);
    out.data = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    out.ids = vec4<i32>(0, 0, 7, 7);
    return out;
})");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// Bug 2: Struct value member extraction (OpCompositeExtract)
// Accessing members of a struct returned by a function call must emit
// OpCompositeExtract, not try to treat it as a vector swizzle.
// ============================================================================

TEST(SlugRegression, StructReturnMemberAccess) {
    auto r = wgsl_test::CompileWgsl(R"(
struct Result {
    texcoord: vec2<f32>,
    pos: vec2<f32>,
};
fn compute() -> Result {
    return Result(vec2<f32>(1.0, 2.0), vec2<f32>(3.0, 4.0));
}
@fragment fn main() -> @location(0) vec4<f32> {
    let res = compute();
    let tc = res.texcoord;
    let p = res.pos;
    return vec4<f32>(tc.x, tc.y, p.x, p.y);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SlugRegression, StructReturnMemberAccess_MultipleFields) {
    auto r = wgsl_test::CompileWgsl(R"(
struct UnpackResult {
    vbnd: vec4<f32>,
    vgly: vec4<i32>,
};
fn unpack(a: vec4<f32>, b: vec4<f32>) -> UnpackResult {
    return UnpackResult(b, vec4<i32>(0, 0, 7, 7));
}
@fragment fn main() -> @location(0) vec4<f32> {
    let r = unpack(vec4<f32>(1.0), vec4<f32>(2.0));
    return r.vbnd;
})");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// Bug 3: Output struct member access must not intercept wrong base objects
// When a vertex shader has struct output, member access on OTHER struct
// values (e.g., function return values) must not be confused with output writes.
// ============================================================================

TEST(SlugRegression, OutputStructDisambiguation) {
    auto r = wgsl_test::CompileWgsl(R"(
struct MyResult {
    texcoord: vec2<f32>,
    value: vec2<f32>,
};
struct VertOut {
    @builtin(position) position: vec4<f32>,
    @location(0) texcoord: vec2<f32>,
};
fn helper() -> MyResult {
    return MyResult(vec2<f32>(1.0, 2.0), vec2<f32>(3.0, 4.0));
}
@vertex fn main(@location(0) pos: vec4<f32>) -> VertOut {
    var out: VertOut;
    let res = helper();
    out.texcoord = res.texcoord;
    out.position = vec4<f32>(pos.xy, 0.0, 1.0);
    return out;
})");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// Bug 4: Vector component pointer access on output variables
// Writing to individual components like position.x must create an
// OpAccessChain on the output variable, not fail silently.
// ============================================================================

TEST(SlugRegression, OutputVectorComponentWrite) {
    auto r = wgsl_test::CompileWgsl(R"(
struct VertOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};
@vertex fn main(@location(0) pos: vec4<f32>) -> VertOut {
    var out: VertOut;
    out.position.x = pos.x * 2.0;
    out.position.y = pos.y * 2.0;
    out.position.z = 0.0;
    out.position.w = 1.0;
    out.color = vec4<f32>(1.0);
    return out;
})");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// Bug 5: fwidth / dpdx / dpdy must be emitted through the SSIR path
// The existing derivative_test.cpp tests simple cases. These tests ensure
// the result of fwidth is actually USED in subsequent computation (which
// was the failure mode: fwidth emitted to dead word buffer, not SSIR).
// ============================================================================

TEST(SlugRegression, FwidthResultUsedInComputation) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main(@location(0) tc: vec2<f32>) -> @location(0) vec4<f32> {
    let epp = fwidth(tc);
    let ppe = 1.0 / epp;
    return vec4<f32>(ppe.x, ppe.y, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SlugRegression, FwidthInLetBinding) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main(@location(0) tc: vec2<f32>) -> @location(0) vec4<f32> {
    let fw = fwidth(tc);
    let scaled = fw * 0.5;
    return vec4<f32>(scaled, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// Bug 6: Scalar / vector division broadcast
// Dividing a scalar by a vector (or vice versa) must splat the scalar
// to a matching vector type before emitting OpFDiv.
// ============================================================================

TEST(SlugRegression, ScalarDivideByVector) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec2<f32>(2.0, 4.0);
    let result = 1.0 / v;
    return vec4<f32>(result, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SlugRegression, VectorDivideByScalar) {
    auto r = wgsl_test::CompileWgsl(R"(
@fragment fn main() -> @location(0) vec4<f32> {
    let v = vec2<f32>(2.0, 4.0);
    let result = v / 2.0;
    return vec4<f32>(result, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// Bug 7: Module-scope const must be visible in helper functions
// A top-level `const` declaration must be accessible from ALL functions,
// not just the entry point.
// ============================================================================

TEST(SlugRegression, ModuleConstInHelperFunction) {
    auto r = wgsl_test::CompileWgsl(R"(
const kWidth: u32 = 12u;
fn helper(x: i32) -> vec2<i32> {
    var loc = vec2<i32>(x, 0);
    loc.y += loc.x >> kWidth;
    loc.x &= (1 << kWidth) - 1;
    return loc;
}
@fragment fn main() -> @location(0) vec4<f32> {
    let r = helper(5000);
    return vec4<f32>(f32(r.x), f32(r.y), 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SlugRegression, ModuleConstInMultipleHelpers) {
    auto r = wgsl_test::CompileWgsl(R"(
const PI: f32 = 3.14159;
const TWO: f32 = 2.0;
fn circle_area(radius: f32) -> f32 {
    return PI * radius * radius;
}
fn circumference(radius: f32) -> f32 {
    return TWO * PI * radius;
}
@fragment fn main() -> @location(0) vec4<f32> {
    let a = circle_area(1.0);
    let c = circumference(1.0);
    return vec4<f32>(a, c, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// Bug 8: Boolean override constants
// `override` declarations with bool type (false/true as identifiers)
// must emit OpSpecConstantFalse/True, not default to f32.
// ============================================================================

TEST(SlugRegression, BooleanOverrideConstant) {
    auto r = wgsl_test::CompileWgsl(R"(
override ENABLE_FEATURE: bool = false;
@fragment fn main() -> @location(0) vec4<f32> {
    if (ENABLE_FEATURE) {
        return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    }
    return vec4<f32>(0.0, 1.0, 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SlugRegression, BooleanOverrideInHelperFunction) {
    auto r = wgsl_test::CompileWgsl(R"(
override USE_SQRT: bool = false;
fn adjust(coverage: f32) -> f32 {
    if (USE_SQRT) {
        return sqrt(coverage);
    }
    return coverage;
}
@fragment fn main() -> @location(0) vec4<f32> {
    let c = adjust(0.5);
    return vec4<f32>(c, c, c, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// Combined: Full Slug-like vertex shader pattern
// This tests the complete pattern used by the Slug vertex shader:
// struct return with builtins + locations + flat interpolation,
// function calls returning structs, component-wise position writes.
// ============================================================================

TEST(SlugRegression, FullSlugVertexPattern) {
    auto r = wgsl_test::CompileWgsl(R"(
struct DilateResult {
    texcoord: vec2<f32>,
    vpos: vec2<f32>,
};
struct UnpackResult {
    vbnd: vec4<f32>,
    vgly: vec4<i32>,
};
struct ParamStruct {
    matrix: array<vec4<f32>, 4>,
    viewport: vec4<f32>,
};
@group(0) @binding(0) var<uniform> params: ParamStruct;

fn Dilate(pos: vec4<f32>, tex: vec4<f32>) -> DilateResult {
    return DilateResult(tex.xy, pos.xy + pos.zw * 0.5);
}
fn Unpack(tex: vec4<f32>, bnd: vec4<f32>) -> UnpackResult {
    return UnpackResult(bnd, vec4<i32>(0, 0, 7, 7));
}

struct VertOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) texcoord: vec2<f32>,
    @location(2) @interpolate(flat) banding: vec4<f32>,
    @location(3) @interpolate(flat) glyph: vec4<i32>,
};

@vertex fn main(
    @location(0) pos: vec4<f32>,
    @location(1) tex: vec4<f32>,
    @location(2) jac: vec4<f32>,
    @location(3) bnd: vec4<f32>,
    @location(4) col: vec4<f32>,
) -> VertOut {
    var vresult: VertOut;
    let dr = Dilate(pos, tex);
    vresult.texcoord = dr.texcoord;
    let p = dr.vpos;
    vresult.position.x = p.x * params.matrix[0].x + p.y * params.matrix[0].y + params.matrix[0].w;
    vresult.position.y = p.x * params.matrix[1].x + p.y * params.matrix[1].y + params.matrix[1].w;
    vresult.position.z = 0.0;
    vresult.position.w = 1.0;
    let ur = Unpack(tex, bnd);
    vresult.banding = ur.vbnd;
    vresult.glyph = ur.vgly;
    vresult.color = col;
    return vresult;
})");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// Combined: Full Slug-like fragment shader pattern
// Module-level const, override bools, helper functions with const access,
// fwidth, texture loads, loops, and the coverage computation pattern.
// ============================================================================

TEST(SlugRegression, FullSlugFragmentPattern) {
    auto r = wgsl_test::CompileWgsl(R"(
const kLogWidth: u32 = 12u;
override EVEN_ODD: bool = false;
override WEIGHT: bool = false;

fn CalcBandLoc(glyphLoc: vec2<i32>, offset: u32) -> vec2<i32> {
    var bandLoc = vec2<i32>(glyphLoc.x + i32(offset), glyphLoc.y);
    bandLoc.y += bandLoc.x >> kLogWidth;
    bandLoc.x &= (1 << kLogWidth) - 1;
    return bandLoc;
}

fn CalcCoverage(xcov: f32, ycov: f32, xwgt: f32, ywgt: f32) -> f32 {
    var coverage = max(abs(xcov * xwgt + ycov * ywgt) / max(xwgt + ywgt, 1.0 / 65536.0), min(abs(xcov), abs(ycov)));
    if (EVEN_ODD) {
        coverage = 1.0 - abs(1.0 - fract(coverage * 0.5) * 2.0);
    } else {
        coverage = saturate(coverage);
    }
    if (WEIGHT) {
        coverage = sqrt(coverage);
    }
    return coverage;
}

fn Render(renderCoord: vec2<f32>, bandTransform: vec4<f32>,
          glyphData: vec4<i32>) -> f32 {
    let emsPerPixel = fwidth(renderCoord);
    let pixelsPerEm = 1.0 / emsPerPixel;
    let bandMax = vec2<i32>(glyphData.z, glyphData.w & 0x00FF);
    let bandIndex = clamp(vec2<i32>(renderCoord * bandTransform.xy + bandTransform.zw),
                          vec2<i32>(0, 0), bandMax);
    let glyphLoc = vec2<i32>(glyphData.x, glyphData.y);
    let hbandLoc = CalcBandLoc(glyphLoc, u32(bandIndex.y));

    var xcov: f32 = 0.0;
    var xwgt: f32 = 0.0;
    for (var i: i32 = 0; i < 8; i++) {
        let px = f32(hbandLoc.x + i) * pixelsPerEm.x;
        if (px < -0.5) {
            break;
        }
        xcov += saturate(px + 0.5);
    }
    return CalcCoverage(xcov, 0.0, xwgt, 0.0);
}

struct VertexStruct {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) texcoord: vec2<f32>,
    @location(2) @interpolate(flat) banding: vec4<f32>,
    @location(3) @interpolate(flat) glyph: vec4<i32>,
};

@fragment fn main(vresult: VertexStruct) -> @location(0) vec4<f32> {
    let coverage = Render(vresult.texcoord, vresult.banding, vresult.glyph);
    return vresult.color * coverage;
})");
    EXPECT_TRUE(r.success) << r.error;
}

// ============================================================================
// Bug 9: textureLoad on texture_2d<u32> passed as function parameter
// When textureLoad is called on a texture passed as a function parameter
// (not a global variable), the SPIR-V emitter must correctly infer vec4<u32>
// as the result type. Previously, if vec4<u32> hadn't been emitted as a
// SPIR-V type yet, the result type defaulted to 0, which was misinterpreted
// as f16 by spv_type_to_ssir.
// ============================================================================

TEST(SlugRegression, TextureLoadOnParameterU32) {
    auto r = wgsl_test::CompileWgsl(R"(
fn TexelLoad2D_u32(tex: texture_2d<u32>, coords: vec2<i32>) -> vec4<u32> {
    return textureLoad(tex, coords, 0);
}

fn helper(bandData: texture_2d<u32>, glyphLoc: vec2<i32>) -> vec2<u32> {
    let data = textureLoad(bandData, glyphLoc, 0).xy;
    return data;
}

@group(0) @binding(0) var bandTexture: texture_2d<u32>;

@fragment fn main(@location(0) tc: vec2<f32>) -> @location(0) vec4<f32> {
    let r = helper(bandTexture, vec2<i32>(0, 0));
    return vec4<f32>(f32(r.x), f32(r.y), 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SlugRegression, TextureLoadOnParameterU32_HelperFirst) {
    // Regression: helper declared before any function that creates vec4<u32>
    auto r = wgsl_test::CompileWgsl(R"(
fn helper(bandData: texture_2d<u32>, glyphLoc: vec2<i32>) -> vec2<u32> {
    let data = textureLoad(bandData, glyphLoc, 0).xy;
    return data;
}

@group(0) @binding(0) var bandTexture: texture_2d<u32>;

@fragment fn main(@location(0) tc: vec2<f32>) -> @location(0) vec4<f32> {
    let r = helper(bandTexture, vec2<i32>(0, 0));
    return vec4<f32>(f32(r.x), f32(r.y), 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}

TEST(SlugRegression, TextureLoadOnParameterI32) {
    auto r = wgsl_test::CompileWgsl(R"(
fn helper(tex: texture_2d<i32>, c: vec2<i32>) -> vec2<i32> {
    return textureLoad(tex, c, 0).xy;
}

@group(0) @binding(0) var myTex: texture_2d<i32>;

@fragment fn main(@location(0) tc: vec2<f32>) -> @location(0) vec4<f32> {
    let r = helper(myTex, vec2<i32>(0, 0));
    return vec4<f32>(f32(r.x), f32(r.y), 0.0, 1.0);
})");
    EXPECT_TRUE(r.success) << r.error;
}
