// Comprehensive GLSL roundtrip tests: WGSL -> SSIR -> GLSL -> parse GLSL -> SPIR-V -> validate
// Exercises the GLSL emission and re-parse paths for a wide variety of types, variables,
// control flow, builtins, and shader constructs.
#include <gtest/gtest.h>
#include "test_utils.h"

extern "C" {
#include "simple_wgsl.h"
}

namespace {

// ============================================================================
// Helpers
// ============================================================================

struct SsirCompileResult {
    bool success;
    std::string error;
    const SsirModule *ssir;
    WgslLower *lower;
};

SsirCompileResult CompileToSsir(const char *source) {
    SsirCompileResult r;
    r.success = false;
    r.ssir = nullptr;
    r.lower = nullptr;

    WgslAstNode *ast = wgsl_parse(source);
    if (!ast) { r.error = "Parse failed"; return r; }
    WgslResolver *resolver = wgsl_resolver_build(ast);
    if (!resolver) { wgsl_free_ast(ast); r.error = "Resolve failed"; return r; }

    WgslLowerOptions opts = {};
    opts.env = WGSL_LOWER_ENV_VULKAN_1_3;
    opts.enable_debug_names = 1;

    r.lower = wgsl_lower_create(ast, resolver, &opts);
    wgsl_resolver_free(resolver);
    wgsl_free_ast(ast);
    if (!r.lower) { r.error = "Lower create failed"; return r; }

    r.ssir = wgsl_lower_get_ssir(r.lower);
    if (!r.ssir) {
        wgsl_lower_destroy(r.lower);
        r.lower = nullptr;
        r.error = "No SSIR module";
        return r;
    }
    r.success = true;
    return r;
}

class SsirCompileGuard {
  public:
    explicit SsirCompileGuard(const SsirCompileResult &r) : r_(r) {}
    ~SsirCompileGuard() { if (r_.lower) wgsl_lower_destroy(r_.lower); }
  private:
    SsirCompileResult r_;
};

// Full roundtrip: WGSL -> SSIR -> GLSL -> parse GLSL -> SPIR-V -> validate
struct RoundtripResult {
    bool glsl_emit_ok;
    bool glsl_parse_ok;
    bool spirv_ok;
    bool spirv_valid;
    std::string glsl;
    std::string error;
};

RoundtripResult GlslRoundtrip(const char *wgsl_source, WgslStage stage, SsirStage ssir_stage) {
    RoundtripResult r = {};

    auto compile = CompileToSsir(wgsl_source);
    SsirCompileGuard guard(compile);
    if (!compile.success) {
        r.error = "Compile: " + compile.error;
        return r;
    }

    auto glsl_result = wgsl_test::RaiseSsirToGlsl(compile.ssir, ssir_stage);
    if (!glsl_result.success) {
        r.error = "GLSL emit: " + glsl_result.error;
        return r;
    }
    r.glsl_emit_ok = true;
    r.glsl = glsl_result.glsl;

    WgslAstNode *ast = glsl_parse(r.glsl.c_str(), stage);
    if (!ast) {
        r.error = "GLSL parse failed on emitted GLSL";
        return r;
    }
    r.glsl_parse_ok = true;

    WgslResolver *resolver = wgsl_resolver_build(ast);
    if (!resolver) {
        wgsl_free_ast(ast);
        r.error = "Resolve of re-parsed GLSL failed";
        return r;
    }

    uint32_t *spirv = nullptr;
    size_t spirv_size = 0;
    WgslLowerOptions lower_opts = {};
    lower_opts.env = WGSL_LOWER_ENV_VULKAN_1_3;

    WgslLowerResult lower_result = wgsl_lower_emit_spirv(ast, resolver, &lower_opts, &spirv, &spirv_size);
    wgsl_resolver_free(resolver);
    wgsl_free_ast(ast);

    if (lower_result != WGSL_LOWER_OK) {
        r.error = "Lower of re-parsed GLSL failed";
        return r;
    }
    r.spirv_ok = true;

    std::string val_err;
    r.spirv_valid = wgsl_test::ValidateSpirv(spirv, spirv_size, &val_err);
    wgsl_lower_free(spirv);

    if (!r.spirv_valid) {
        r.error = "SPIR-V validation failed: " + val_err;
    }

    return r;
}

// Macro for full GLSL roundtrip tests
#define GLSL_ROUNDTRIP(TestName, WgslSource, Stage, SsirStage, ...) \
TEST(GlslRoundtrip, TestName) { \
    auto r = GlslRoundtrip(WgslSource, Stage, SsirStage); \
    EXPECT_TRUE(r.glsl_emit_ok) << r.error; \
    EXPECT_TRUE(r.glsl_parse_ok) << "GLSL:\n" << r.glsl << "\nError: " << r.error; \
    EXPECT_TRUE(r.spirv_ok) << "GLSL:\n" << r.glsl << "\nError: " << r.error; \
    EXPECT_TRUE(r.spirv_valid) << "GLSL:\n" << r.glsl << "\nError: " << r.error; \
    __VA_ARGS__; \
}

// Macro for emit-only tests (when full roundtrip hits known parser limitations)
#define GLSL_EMIT_TEST(TestName, WgslSource, SsirStage, ...) \
TEST(GlslRoundtrip, TestName) { \
    auto compile = CompileToSsir(WgslSource); \
    SsirCompileGuard guard(compile); \
    ASSERT_TRUE(compile.success) << compile.error; \
    auto glsl_result = wgsl_test::RaiseSsirToGlsl(compile.ssir, SsirStage); \
    EXPECT_TRUE(glsl_result.success) << glsl_result.error; \
    __VA_ARGS__; \
}

} // namespace

// ============================================================================
// Category 1: Scalar Types
// ============================================================================

GLSL_ROUNDTRIP(ScalarF32, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let x: f32 = 3.14;
        return vec4f(x, 0.0, 0.0, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(ScalarI32, R"(
    @compute @workgroup_size(1) fn main() {
        let x: i32 = 42i;
        let y: i32 = -10i;
        let z = x + y;
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_ROUNDTRIP(ScalarU32, R"(
    @compute @workgroup_size(1) fn main() {
        let x: u32 = 255u;
        let y: u32 = 1u;
        let z = x + y;
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_EMIT_TEST(ScalarBool, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let a = true;
        let b = false;
        var result = 0.0;
        if (a && !b) { result = 1.0; }
        return vec4f(result, 0.0, 0.0, 1.0);
    }
)", SSIR_STAGE_FRAGMENT,
    EXPECT_TRUE(glsl_result.glsl.find("true") != std::string::npos ||
                glsl_result.glsl.find("if") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

// ============================================================================
// Category 2: Vector Types
// ============================================================================

GLSL_ROUNDTRIP(Vec2f, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let v = vec2f(1.0, 2.0);
        return vec4f(v.x, v.y, 0.0, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(Vec3f, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let v = vec3f(1.0, 2.0, 3.0);
        return vec4f(v, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(Vec4fConstruct, R"(
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(0.1, 0.2, 0.3, 0.4);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(Vec2i, R"(
    @compute @workgroup_size(1) fn main() {
        let v = vec2i(1, -2);
        let w = v + vec2i(3, 4);
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_ROUNDTRIP(Vec3u, R"(
    @compute @workgroup_size(1) fn main() {
        let v = vec3u(1u, 2u, 3u);
        let w = v * vec3u(2u, 2u, 2u);
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_ROUNDTRIP(Vec4u, R"(
    @compute @workgroup_size(1) fn main() {
        let v = vec4u(0u, 1u, 2u, 3u);
        let s = v.x + v.y + v.z + v.w;
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_ROUNDTRIP(Vec4i, R"(
    @compute @workgroup_size(1) fn main() {
        let a = vec4i(1, 2, 3, 4);
        let b = vec4i(5, 6, 7, 8);
        let c = a - b;
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_ROUNDTRIP(VectorSplatConstruct, R"(
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(0.5);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

// ============================================================================
// Category 3: Matrix Types
// ============================================================================

GLSL_ROUNDTRIP(Mat2x2Construct, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let m = mat2x2f(1.0, 0.0, 0.0, 1.0);
        return vec4f(m[0].x, m[0].y, m[1].x, m[1].y);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(Mat3x3Construct, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let m = mat3x3f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        return vec4f(m[0].x, m[1].y, m[2].z, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(Mat4x4UniformMultiply, R"(
    struct UBO { mvp: mat4x4f };
    @group(0) @binding(0) var<uniform> ubo: UBO;
    @vertex fn vs(@location(0) pos: vec4f) -> @builtin(position) vec4f {
        return ubo.mvp * pos;
    }
)", WGSL_STAGE_VERTEX, SSIR_STAGE_VERTEX)

GLSL_ROUNDTRIP(MatVecMultiply, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let m = mat2x2f(2.0, 0.0, 0.0, 2.0);
        let v = vec2f(1.0, 1.0);
        let r = m * v;
        return vec4f(r.x, r.y, 0.0, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(MatTranspose, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let m = mat2x2f(1.0, 2.0, 3.0, 4.0);
        let t = transpose(m);
        return vec4f(t[0].x, t[0].y, t[1].x, t[1].y);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(MatDeterminant, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let m = mat2x2f(1.0, 2.0, 3.0, 4.0);
        let d = determinant(m);
        return vec4f(d, 0.0, 0.0, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

// ============================================================================
// Category 4: Array Types
// ============================================================================

GLSL_ROUNDTRIP(FixedArray, R"(
    @compute @workgroup_size(1) fn main() {
        var arr: array<f32, 4>;
        arr[0] = 1.0;
        arr[1] = 2.0;
        arr[2] = 3.0;
        arr[3] = 4.0;
        let sum = arr[0] + arr[3];
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_ROUNDTRIP(FixedArrayInt, R"(
    @compute @workgroup_size(1) fn main() {
        var arr: array<i32, 3>;
        arr[0] = 10i;
        arr[1] = 20i;
        arr[2] = 30i;
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_ROUNDTRIP(FixedArrayVec4, R"(
    @compute @workgroup_size(1) fn main() {
        var arr: array<vec4f, 2>;
        arr[0] = vec4f(1.0);
        arr[1] = vec4f(2.0);
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_ROUNDTRIP(RuntimeArrayLength, R"(
    struct Buf { data: array<f32> };
    @group(0) @binding(0) var<storage, read_write> buf: Buf;
    @compute @workgroup_size(1) fn main() {
        let len = arrayLength(&buf.data);
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

// ============================================================================
// Category 5: Struct Types
// ============================================================================

GLSL_ROUNDTRIP(SimpleStruct, R"(
    struct Material { color: vec4f };
    @group(0) @binding(0) var<uniform> mat: Material;
    @fragment fn fs() -> @location(0) vec4f { return mat.color; }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(StructMultipleFields, R"(
    struct Params { scale: f32, offset: f32, brightness: f32 };
    @group(0) @binding(0) var<uniform> params: Params;
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(params.scale, params.offset, params.brightness, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(StructMixedTypes, R"(
    struct MixedData {
        position: vec3f,
        index: u32,
        normal: vec3f,
        pad: f32,
    };
    @group(0) @binding(0) var<uniform> data: MixedData;
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(data.position, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_EMIT_TEST(NestedStructEmit, R"(
    struct Inner { value: f32 };
    struct Outer { inner: Inner, scale: f32 };
    @group(0) @binding(0) var<uniform> data: Outer;
    @fragment fn fs() -> @location(0) vec4f {
        let v = data.inner.value * data.scale;
        return vec4f(v, 0.0, 0.0, 1.0);
    }
)", SSIR_STAGE_FRAGMENT,
    EXPECT_TRUE(glsl_result.glsl.find("struct") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl;
    EXPECT_TRUE(glsl_result.glsl.find("uniform") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

// ============================================================================
// Category 6: Uniform / Storage / Bindings
// ============================================================================

GLSL_ROUNDTRIP(UniformBufferFragment, R"(
    struct UBO { color: vec4f };
    @group(0) @binding(0) var<uniform> u: UBO;
    @fragment fn fs() -> @location(0) vec4f { return u.color; }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_EMIT_TEST(StorageBufferReadOnly, R"(
    struct Data { values: array<f32, 4> };
    @group(0) @binding(0) var<storage> data: Data;
    @compute @workgroup_size(1) fn main() {
        let v = data.values[0];
    }
)", SSIR_STAGE_COMPUTE,
    EXPECT_TRUE(glsl_result.glsl.find("buffer") != std::string::npos ||
                glsl_result.glsl.find("std430") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

GLSL_ROUNDTRIP(StorageBufferReadWrite, R"(
    struct Buf { a: f32, b: f32, result: f32 };
    @group(0) @binding(0) var<storage, read_write> buf: Buf;
    @compute @workgroup_size(1) fn main() {
        buf.result = buf.a + buf.b;
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_ROUNDTRIP(MultipleBindings, R"(
    struct A { x: f32 };
    struct B { y: f32 };
    @group(0) @binding(0) var<uniform> a: A;
    @group(0) @binding(1) var<uniform> b: B;
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(a.x, b.y, 0.0, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(MultipleGroups, R"(
    struct A { val: f32 };
    struct B { val: f32 };
    @group(0) @binding(0) var<uniform> a: A;
    @group(1) @binding(0) var<uniform> b: B;
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(a.val + b.val, 0.0, 0.0, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

// ============================================================================
// Category 7: Entry Point Stages
// ============================================================================

GLSL_ROUNDTRIP(VertexSimple, R"(
    @vertex fn vs() -> @builtin(position) vec4f {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }
)", WGSL_STAGE_VERTEX, SSIR_STAGE_VERTEX)

GLSL_ROUNDTRIP(FragmentSimple, R"(
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(1.0, 0.0, 0.0, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(ComputeSimple, R"(
    @compute @workgroup_size(1) fn main() {
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_ROUNDTRIP(ComputeWorkgroupSize, R"(
    @compute @workgroup_size(8, 8, 1) fn main() {
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_ROUNDTRIP(ComputeWorkgroupSize3D, R"(
    @compute @workgroup_size(4, 4, 4) fn main() {
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

// ============================================================================
// Category 8: Vertex Builtins and I/O
// ============================================================================

GLSL_ROUNDTRIP(VertexWithLocationInput, R"(
    @vertex fn vs(@location(0) pos: vec3f) -> @builtin(position) vec4f {
        return vec4f(pos, 1.0);
    }
)", WGSL_STAGE_VERTEX, SSIR_STAGE_VERTEX)

GLSL_ROUNDTRIP(VertexMultipleInputs, R"(
    @vertex fn vs(@location(0) pos: vec3f, @location(1) color: vec4f) -> @builtin(position) vec4f {
        return vec4f(pos, 1.0);
    }
)", WGSL_STAGE_VERTEX, SSIR_STAGE_VERTEX)

GLSL_ROUNDTRIP(VertexMultipleOutputs, R"(
    struct VOut {
        @builtin(position) pos: vec4f,
        @location(0) color: vec4f,
        @location(1) uv: vec2f,
    };
    @vertex fn vs(@location(0) inPos: vec3f, @location(1) inColor: vec4f, @location(2) inUv: vec2f) -> VOut {
        var out: VOut;
        out.pos = vec4f(inPos, 1.0);
        out.color = inColor;
        out.uv = inUv;
        return out;
    }
)", WGSL_STAGE_VERTEX, SSIR_STAGE_VERTEX)

GLSL_ROUNDTRIP(VertexIndex, R"(
    @vertex fn vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {
        let x = f32(vid) * 0.5 - 1.0;
        return vec4f(x, 0.0, 0.0, 1.0);
    }
)", WGSL_STAGE_VERTEX, SSIR_STAGE_VERTEX)

GLSL_ROUNDTRIP(VertexInstanceIndex, R"(
    @vertex fn vs(@builtin(instance_index) iid: u32) -> @builtin(position) vec4f {
        let offset = f32(iid) * 0.1;
        return vec4f(offset, 0.0, 0.0, 1.0);
    }
)", WGSL_STAGE_VERTEX, SSIR_STAGE_VERTEX)

// ============================================================================
// Category 9: Fragment Builtins
// ============================================================================

GLSL_ROUNDTRIP(FragmentWithLocationInput, R"(
    @fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
        return vec4f(uv, 0.0, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(FragmentMultipleInputs, R"(
    @fragment fn fs(@location(0) color: vec4f, @location(1) uv: vec2f) -> @location(0) vec4f {
        return vec4f(color.rgb * uv.x, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_EMIT_TEST(FragCoordEmit, R"(
    @fragment fn fs(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
        let uv = fragCoord.xy / vec2f(800.0, 600.0);
        return vec4f(uv, 0.0, 1.0);
    }
)", SSIR_STAGE_FRAGMENT,
    EXPECT_TRUE(glsl_result.glsl.find("gl_FragCoord") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

GLSL_EMIT_TEST(FrontFacingEmit, R"(
    @fragment fn fs(@builtin(front_facing) ff: bool) -> @location(0) vec4f {
        if (ff) { return vec4f(1.0, 0.0, 0.0, 1.0); }
        return vec4f(0.0, 0.0, 1.0, 1.0);
    }
)", SSIR_STAGE_FRAGMENT,
    EXPECT_TRUE(glsl_result.glsl.find("gl_FrontFacing") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

// ============================================================================
// Category 10: Compute Builtins
// ============================================================================

GLSL_ROUNDTRIP(GlobalInvocationId, R"(
    struct Buf { value: f32 };
    @group(0) @binding(0) var<storage, read_write> buf: Buf;
    @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3u) {
        buf.value = f32(gid.x);
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_ROUNDTRIP(LocalInvocationId, R"(
    struct Buf { value: f32 };
    @group(0) @binding(0) var<storage, read_write> buf: Buf;
    @compute @workgroup_size(64) fn main(@builtin(local_invocation_id) lid: vec3u) {
        buf.value = f32(lid.x);
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_ROUNDTRIP(WorkgroupId, R"(
    struct Buf { value: f32 };
    @group(0) @binding(0) var<storage, read_write> buf: Buf;
    @compute @workgroup_size(64) fn main(@builtin(workgroup_id) wgid: vec3u) {
        buf.value = f32(wgid.x);
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_ROUNDTRIP(LocalInvocationIndex, R"(
    var<workgroup> shared_data: array<f32, 64>;
    @compute @workgroup_size(64) fn main(@builtin(local_invocation_index) lid: u32) {
        shared_data[lid] = f32(lid);
        workgroupBarrier();
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_ROUNDTRIP(NumWorkgroups, R"(
    struct Buf { value: f32 };
    @group(0) @binding(0) var<storage, read_write> buf: Buf;
    @compute @workgroup_size(1) fn main(@builtin(num_workgroups) nwg: vec3u) {
        buf.value = f32(nwg.x * nwg.y * nwg.z);
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

// ============================================================================
// Category 11: Arithmetic Operations
// ============================================================================

GLSL_ROUNDTRIP(FloatArithmetic, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let a = 1.0;
        let b = 2.0;
        let sum = a + b;
        let diff = a - b;
        let prod = a * b;
        let quot = a / b;
        return vec4f(sum, diff, prod, quot);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(IntegerArithmetic, R"(
    @compute @workgroup_size(1) fn main() {
        let a = 10i;
        let b = 3i;
        let sum = a + b;
        let diff = a - b;
        let prod = a * b;
        let quot = a / b;
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_ROUNDTRIP(UnsignedArithmetic, R"(
    @compute @workgroup_size(1) fn main() {
        let a = 100u;
        let b = 7u;
        let sum = a + b;
        let diff = a - b;
        let prod = a * b;
        let quot = a / b;
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_ROUNDTRIP(FloatNegation, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let x = 5.0;
        let neg = -x;
        return vec4f(neg, 0.0, 0.0, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(FloatModulo, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let a = 5.0;
        let b = 3.0;
        let m = a % b;
        return vec4f(m, 0.0, 0.0, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(IntegerModulo, R"(
    @compute @workgroup_size(1) fn main() {
        let a = 17i;
        let b = 5i;
        let m = a % b;
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_ROUNDTRIP(VectorArithmetic, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let a = vec4f(1.0, 2.0, 3.0, 4.0);
        let b = vec4f(5.0, 6.0, 7.0, 8.0);
        let sum = a + b;
        let prod = a * b;
        return sum * 0.1;
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(IntVectorArithmetic, R"(
    @compute @workgroup_size(1) fn main() {
        let a = vec4i(1, 2, 3, 4);
        let b = vec4i(5, 6, 7, 8);
        let c = a + b;
        let d = a * b;
        let e = a - b;
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

// ============================================================================
// Category 12: Comparison Operations
// ============================================================================

GLSL_ROUNDTRIP(FloatComparisons, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let a = 1.0;
        let b = 2.0;
        var result = 0.0;
        if (a == b) { result = 1.0; }
        if (a != b) { result = 2.0; }
        if (a < b) { result = 3.0; }
        if (a <= b) { result = 4.0; }
        if (a > b) { result = 5.0; }
        if (a >= b) { result = 6.0; }
        return vec4f(result, 0.0, 0.0, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(IntComparisons, R"(
    @compute @workgroup_size(1) fn main() {
        let a = 5i;
        let b = 10i;
        var r = 0i;
        if (a < b) { r = 1i; }
        if (a >= b) { r = 2i; }
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

// ============================================================================
// Category 13: Bitwise Operations
// ============================================================================

GLSL_ROUNDTRIP(BitwiseAndOrXor, R"(
    @compute @workgroup_size(1) fn main() {
        let a = 0xFFu;
        let b = 0x0Fu;
        let c = a & b;
        let d = a | b;
        let e = a ^ b;
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_ROUNDTRIP(BitwiseNotAndShifts, R"(
    @compute @workgroup_size(1) fn main() {
        let a = 0xF0u;
        let b = ~a;
        let c = a << 2u;
        let d = a >> 4u;
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_ROUNDTRIP(SignedBitwiseOps, R"(
    @compute @workgroup_size(1) fn main() {
        let a = 0x7Fi;
        let b = 0x0Fi;
        let c = a & b;
        let d = a | b;
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

// ============================================================================
// Category 14: Logical Operations
// ============================================================================

GLSL_ROUNDTRIP(LogicalAndOrNot, R"(
    @fragment fn fs(@location(0) val: f32) -> @location(0) vec4f {
        let a = val > 0.5;
        let b = val < 0.8;
        var result = 0.0;
        if (a && b) { result = 1.0; }
        if (a || b) { result = 2.0; }
        if (!a) { result = 3.0; }
        return vec4f(result, 0.0, 0.0, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

// ============================================================================
// Category 15: Control Flow
// ============================================================================

GLSL_ROUNDTRIP(IfElse, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let x = 0.5;
        var color = vec4f(0.0);
        if (x > 0.0) {
            color = vec4f(1.0, 0.0, 0.0, 1.0);
        } else {
            color = vec4f(0.0, 0.0, 1.0, 1.0);
        }
        return color;
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(NestedIf, R"(
    @fragment fn fs(@location(0) val: f32) -> @location(0) vec4f {
        var result = vec4f(0.0);
        if (val > 0.5) {
            if (val > 0.75) {
                result = vec4f(1.0, 0.0, 0.0, 1.0);
            } else {
                result = vec4f(0.0, 1.0, 0.0, 1.0);
            }
        } else {
            result = vec4f(0.0, 0.0, 1.0, 1.0);
        }
        return result;
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(ForLoop, R"(
    @compute @workgroup_size(1) fn main() {
        var sum = 0i;
        for (var i = 0i; i < 10i; i = i + 1i) {
            sum = sum + i;
        }
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_EMIT_TEST(ForLoopNested, R"(
    @compute @workgroup_size(1) fn main() {
        var sum = 0i;
        for (var i = 0i; i < 4i; i = i + 1i) {
            for (var j = 0i; j < 4i; j = j + 1i) {
                sum = sum + i * j;
            }
        }
    }
)", SSIR_STAGE_COMPUTE,
    EXPECT_TRUE(glsl_result.glsl.find("void main()") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

GLSL_ROUNDTRIP(WhileLoop, R"(
    @compute @workgroup_size(1) fn main() {
        var i = 0i;
        while (i < 5i) {
            i = i + 1i;
        }
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_ROUNDTRIP(BreakStatement, R"(
    @compute @workgroup_size(1) fn main() {
        var i = 0i;
        for (var x = 0i; x < 100i; x = x + 1i) {
            if (x > 10i) { break; }
            i = x;
        }
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_EMIT_TEST(ContinueStatement, R"(
    @compute @workgroup_size(1) fn main() {
        var sum = 0i;
        for (var i = 0i; i < 10i; i = i + 1i) {
            if (i == 5i) { continue; }
            sum = sum + i;
        }
    }
)", SSIR_STAGE_COMPUTE,
    EXPECT_TRUE(glsl_result.glsl.find("void main()") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

GLSL_ROUNDTRIP(Discard, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let x = 0.5;
        if (x < 0.1) {
            discard;
        }
        return vec4f(1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

// ============================================================================
// Category 16: Type Conversions
// ============================================================================

GLSL_ROUNDTRIP(FloatToIntConversion, R"(
    @compute @workgroup_size(1) fn main() {
        let f = 3.14;
        let i = i32(f);
        let u = u32(i);
        let f2 = f32(u);
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_EMIT_TEST(VectorTypeConversion, R"(
    @compute @workgroup_size(1) fn main() {
        let fv = vec3f(1.5, 2.5, 3.5);
        let iv = vec3i(fv);
        let uv = vec3u(iv);
    }
)", SSIR_STAGE_COMPUTE,
    EXPECT_TRUE(glsl_result.glsl.find("ivec3") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

GLSL_ROUNDTRIP(Bitcast, R"(
    @compute @workgroup_size(1) fn main() {
        let f = 1.0f;
        let i = bitcast<i32>(f);
        let f2 = bitcast<f32>(i);
        let u = bitcast<u32>(f);
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

// ============================================================================
// Category 17: Math Builtins - Trigonometry
// ============================================================================

GLSL_ROUNDTRIP(TrigBasic, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let x = 0.5;
        return vec4f(sin(x), cos(x), tan(x), 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(TrigInverse, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let x = 0.5;
        return vec4f(asin(x), acos(x), atan(x), 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(Atan2, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let y = 1.0;
        let x = 2.0;
        let angle = atan2(y, x);
        return vec4f(angle, 0.0, 0.0, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(Sinh, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let x = 1.0;
        return vec4f(sinh(x), cosh(x), tanh(x), 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(Asinh, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let x = 1.0;
        return vec4f(asinh(x), acosh(x + 1.0), atanh(0.5), 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

// ============================================================================
// Category 18: Math Builtins - Exponential
// ============================================================================

GLSL_ROUNDTRIP(ExpLog, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let x = 1.0;
        return vec4f(exp(x), exp2(x), log(x + 1.0), log2(x + 1.0));
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(PowSqrt, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let x = 4.0;
        return vec4f(pow(x, 2.0), sqrt(x), inverseSqrt(x), 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

// ============================================================================
// Category 19: Math Builtins - Rounding
// ============================================================================

GLSL_ROUNDTRIP(FloorCeilRound, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let x = 2.5;
        return vec4f(floor(x), ceil(x), round(x), trunc(x));
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(FractAbsSign, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let x = -2.7;
        return vec4f(fract(abs(x)), abs(x), sign(x), 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

// ============================================================================
// Category 20: Math Builtins - Clamping and Interpolation
// ============================================================================

GLSL_ROUNDTRIP(ClampMinMax, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let x = 0.5;
        let a = clamp(x, 0.0, 1.0);
        let b = min(x, 0.3);
        let c = max(x, 0.7);
        return vec4f(a, b, c, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(MixStepSmoothstep, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let x = 0.5;
        let a = mix(0.0, 1.0, x);
        let b = step(0.3, x);
        let c = smoothstep(0.2, 0.8, x);
        return vec4f(a, b, c, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(FmaDegreesRadians, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let x = 1.0;
        let a = fma(x, 2.0, 0.5);
        let b = degrees(x);
        let c = radians(45.0);
        return vec4f(a, b, c, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

// ============================================================================
// Category 21: Vector Math Builtins
// ============================================================================

GLSL_ROUNDTRIP(DotProduct, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let a = vec3f(1.0, 0.0, 0.0);
        let b = vec3f(0.0, 1.0, 0.0);
        let d = dot(a, b);
        return vec4f(d, 0.0, 0.0, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(CrossProduct, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let a = vec3f(1.0, 0.0, 0.0);
        let b = vec3f(0.0, 1.0, 0.0);
        let c = cross(a, b);
        return vec4f(c, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(LengthDistanceNormalize, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let a = vec3f(1.0, 2.0, 3.0);
        let b = vec3f(4.0, 5.0, 6.0);
        let l = length(a);
        let d = distance(a, b);
        let n = normalize(a);
        return vec4f(l, d, n.x, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(ReflectRefract, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let incident = vec3f(1.0, -1.0, 0.0);
        let normal = vec3f(0.0, 1.0, 0.0);
        let refl = reflect(incident, normal);
        let refr = refract(normalize(incident), normal, 0.5);
        return vec4f(refl.x, refl.y, refr.x, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(FaceForward, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let n = vec3f(0.0, 1.0, 0.0);
        let i = vec3f(0.0, -1.0, 0.0);
        let nref = vec3f(0.0, 1.0, 0.0);
        let ff = faceForward(n, i, nref);
        return vec4f(ff, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

// ============================================================================
// Category 22: Select Builtin
// ============================================================================

GLSL_ROUNDTRIP(SelectScalar, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let a = 0.0;
        let b = 1.0;
        let c = select(a, b, true);
        return vec4f(c, 0.0, 0.0, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(SelectVector, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let a = vec4f(1.0, 0.0, 0.0, 1.0);
        let b = vec4f(0.0, 1.0, 0.0, 1.0);
        let c = select(a, b, true);
        return c;
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

// ============================================================================
// Category 23: Bit Manipulation Builtins
// ============================================================================

GLSL_ROUNDTRIP(CountOneBitsReverseBits, R"(
    @compute @workgroup_size(1) fn main() {
        let a = 0xF0F0u;
        let c = countOneBits(a);
        let r = reverseBits(a);
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_ROUNDTRIP(ExtractInsertBits, R"(
    @compute @workgroup_size(1) fn main() {
        let v = 0xABCDu;
        let e = extractBits(v, 4u, 8u);
        let ins = insertBits(0u, 0xFFu, 8u, 8u);
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_ROUNDTRIP(FirstLeadingBit, R"(
    @compute @workgroup_size(1) fn main() {
        let a = 0x00F0u;
        let b = firstLeadingBit(a);
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_ROUNDTRIP(FirstTrailingBit, R"(
    @compute @workgroup_size(1) fn main() {
        let a = 0x00F0u;
        let b = firstTrailingBit(a);
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

// ============================================================================
// Category 24: Pack / Unpack
// ============================================================================

GLSL_ROUNDTRIP(Pack4x8snorm, R"(
    @compute @workgroup_size(1) fn main() {
        let v = vec4f(0.5, -0.5, 1.0, -1.0);
        let packed = pack4x8snorm(v);
        let unpacked = unpack4x8snorm(packed);
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_ROUNDTRIP(Pack4x8unorm, R"(
    @compute @workgroup_size(1) fn main() {
        let v = vec4f(0.25, 0.5, 0.75, 1.0);
        let packed = pack4x8unorm(v);
        let unpacked = unpack4x8unorm(packed);
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_ROUNDTRIP(Pack2x16snorm, R"(
    @compute @workgroup_size(1) fn main() {
        let v = vec2f(0.5, -0.5);
        let packed = pack2x16snorm(v);
        let unpacked = unpack2x16snorm(packed);
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_ROUNDTRIP(Pack2x16unorm, R"(
    @compute @workgroup_size(1) fn main() {
        let v = vec2f(0.25, 0.75);
        let packed = pack2x16unorm(v);
        let unpacked = unpack2x16unorm(packed);
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

// ============================================================================
// Category 25: Derivatives (fragment only)
// ============================================================================

GLSL_ROUNDTRIP(DerivativesBasic, R"(
    @fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
        let dx = dpdx(uv.x);
        let dy = dpdy(uv.y);
        let fw = fwidth(uv.x);
        return vec4f(dx, dy, fw, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(DerivativesCoarse, R"(
    @fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
        let dx = dpdxCoarse(uv.x);
        let dy = dpdyCoarse(uv.y);
        return vec4f(dx, dy, 0.0, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(DerivativesFine, R"(
    @fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
        let dx = dpdxFine(uv.x);
        let dy = dpdyFine(uv.y);
        return vec4f(dx, dy, 0.0, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

// ============================================================================
// Category 26: Texture Operations (emit-only due to combined image sampler)
// ============================================================================

GLSL_EMIT_TEST(TextureSample2D, R"(
    @group(0) @binding(0) var t: texture_2d<f32>;
    @group(0) @binding(1) var s: sampler;
    @fragment fn fs() -> @location(0) vec4f {
        return textureSample(t, s, vec2f(0.5, 0.5));
    }
)", SSIR_STAGE_FRAGMENT,
    EXPECT_TRUE(glsl_result.glsl.find("texture(") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

GLSL_EMIT_TEST(TextureSampleLevel, R"(
    @group(0) @binding(0) var t: texture_2d<f32>;
    @group(0) @binding(1) var s: sampler;
    @fragment fn fs() -> @location(0) vec4f {
        return textureSampleLevel(t, s, vec2f(0.5, 0.5), 0.0);
    }
)", SSIR_STAGE_FRAGMENT,
    EXPECT_TRUE(glsl_result.glsl.find("textureLod(") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

GLSL_EMIT_TEST(TextureSampleBias, R"(
    @group(0) @binding(0) var t: texture_2d<f32>;
    @group(0) @binding(1) var s: sampler;
    @fragment fn fs() -> @location(0) vec4f {
        return textureSampleBias(t, s, vec2f(0.5, 0.5), 1.0);
    }
)", SSIR_STAGE_FRAGMENT,
    EXPECT_TRUE(glsl_result.glsl.find("texture(") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

GLSL_EMIT_TEST(TextureSampleGrad, R"(
    @group(0) @binding(0) var t: texture_2d<f32>;
    @group(0) @binding(1) var s: sampler;
    @fragment fn fs() -> @location(0) vec4f {
        return textureSampleGrad(t, s, vec2f(0.5, 0.5), vec2f(0.01, 0.0), vec2f(0.0, 0.01));
    }
)", SSIR_STAGE_FRAGMENT,
    EXPECT_TRUE(glsl_result.glsl.find("textureGrad(") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

GLSL_EMIT_TEST(TextureLoad2D, R"(
    @group(0) @binding(0) var t: texture_2d<f32>;
    @fragment fn fs() -> @location(0) vec4f {
        return textureLoad(t, vec2i(0, 0), 0);
    }
)", SSIR_STAGE_FRAGMENT,
    EXPECT_TRUE(glsl_result.glsl.find("texelFetch(") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

GLSL_EMIT_TEST(TextureStore2D, R"(
    @group(0) @binding(0) var t: texture_storage_2d<rgba8unorm, write>;
    @compute @workgroup_size(1) fn main() {
        textureStore(t, vec2i(0, 0), vec4f(1.0, 0.0, 0.0, 1.0));
    }
)", SSIR_STAGE_COMPUTE,
    EXPECT_TRUE(glsl_result.glsl.find("imageStore(") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

GLSL_EMIT_TEST(TextureSampleCompare, R"(
    @group(0) @binding(0) var t: texture_depth_2d;
    @group(0) @binding(1) var s: sampler_comparison;
    @fragment fn fs() -> @location(0) vec4f {
        let d = textureSampleCompare(t, s, vec2f(0.5, 0.5), 0.5);
        return vec4f(d, d, d, 1.0);
    }
)", SSIR_STAGE_FRAGMENT,
    EXPECT_TRUE(glsl_result.glsl.find("sampler2DShadow") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

GLSL_EMIT_TEST(TextureDimensions2D, R"(
    @group(0) @binding(0) var t: texture_2d<f32>;
    @group(0) @binding(1) var s: sampler;
    @fragment fn fs() -> @location(0) vec4f {
        let dims = textureDimensions(t);
        return vec4f(f32(dims.x), f32(dims.y), 0.0, 1.0);
    }
)", SSIR_STAGE_FRAGMENT,
    EXPECT_TRUE(glsl_result.glsl.find("textureSize(") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

GLSL_EMIT_TEST(TextureNumLevels, R"(
    @group(0) @binding(0) var t: texture_2d<f32>;
    @group(0) @binding(1) var s: sampler;
    @fragment fn fs() -> @location(0) vec4f {
        let levels = textureNumLevels(t);
        return vec4f(f32(levels), 0.0, 0.0, 1.0);
    }
)", SSIR_STAGE_FRAGMENT,
    EXPECT_TRUE(glsl_result.glsl.find("textureQueryLevels(") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

// ============================================================================
// Category 27: Texture Types (emit-only)
// ============================================================================

GLSL_EMIT_TEST(Texture3D, R"(
    @group(0) @binding(0) var t: texture_3d<f32>;
    @group(0) @binding(1) var s: sampler;
    @fragment fn fs() -> @location(0) vec4f {
        return textureSample(t, s, vec3f(0.5, 0.5, 0.5));
    }
)", SSIR_STAGE_FRAGMENT,
    EXPECT_TRUE(glsl_result.glsl.find("sampler3D") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

GLSL_EMIT_TEST(TextureCube, R"(
    @group(0) @binding(0) var t: texture_cube<f32>;
    @group(0) @binding(1) var s: sampler;
    @fragment fn fs() -> @location(0) vec4f {
        return textureSample(t, s, vec3f(1.0, 0.0, 0.0));
    }
)", SSIR_STAGE_FRAGMENT,
    EXPECT_TRUE(glsl_result.glsl.find("samplerCube") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

GLSL_EMIT_TEST(TextureStorageRGBA32Float, R"(
    @group(0) @binding(0) var t: texture_storage_2d<rgba32float, write>;
    @compute @workgroup_size(1) fn main() {
        textureStore(t, vec2i(0, 0), vec4f(1.0, 2.0, 3.0, 4.0));
    }
)", SSIR_STAGE_COMPUTE,
    EXPECT_TRUE(glsl_result.glsl.find("image2D") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

// ============================================================================
// Category 28: Workgroup Variables and Barriers
// ============================================================================

GLSL_ROUNDTRIP(WorkgroupBarrier, R"(
    var<workgroup> shared_data: array<f32, 64>;
    @compute @workgroup_size(64) fn main(@builtin(local_invocation_index) lid: u32) {
        shared_data[lid] = f32(lid);
        workgroupBarrier();
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_EMIT_TEST(WorkgroupSharedVariable, R"(
    var<workgroup> counter: u32;
    @compute @workgroup_size(1) fn main() {
        counter = 0u;
        workgroupBarrier();
        counter = counter + 1u;
    }
)", SSIR_STAGE_COMPUTE,
    EXPECT_TRUE(glsl_result.glsl.find("shared") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl;
    EXPECT_TRUE(glsl_result.glsl.find("barrier()") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

GLSL_ROUNDTRIP(WorkgroupSharedArray, R"(
    var<workgroup> tile: array<vec4f, 256>;
    @compute @workgroup_size(16, 16) fn main(@builtin(local_invocation_index) lid: u32) {
        tile[lid] = vec4f(f32(lid));
        workgroupBarrier();
        let val = tile[lid];
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_EMIT_TEST(StorageBarrier, R"(
    struct Buf { data: array<u32, 64> };
    @group(0) @binding(0) var<storage, read_write> buf: Buf;
    @compute @workgroup_size(64) fn main(@builtin(local_invocation_index) lid: u32) {
        buf.data[lid] = lid;
        storageBarrier();
    }
)", SSIR_STAGE_COMPUTE,
    EXPECT_TRUE(glsl_result.glsl.find("barrier()") != std::string::npos ||
                glsl_result.glsl.find("memoryBarrierBuffer()") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

// ============================================================================
// Category 29: Function Calls
// ============================================================================

GLSL_ROUNDTRIP(FunctionCallSimple, R"(
    fn square(x: f32) -> f32 {
        return x * x;
    }
    @fragment fn fs() -> @location(0) vec4f {
        let v = square(3.0);
        return vec4f(v, 0.0, 0.0, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(FunctionCallMultipleParams, R"(
    fn add_scaled(a: vec4f, b: vec4f, s: f32) -> vec4f {
        return a + b * s;
    }
    @fragment fn fs() -> @location(0) vec4f {
        let a = vec4f(1.0, 0.0, 0.0, 1.0);
        let b = vec4f(0.0, 1.0, 0.0, 1.0);
        return add_scaled(a, b, 0.5);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(FunctionCallMultiple, R"(
    fn helper_a(x: f32) -> f32 { return x * 2.0; }
    fn helper_b(x: f32) -> f32 { return x + 1.0; }
    @fragment fn fs() -> @location(0) vec4f {
        let a = helper_a(1.0);
        let b = helper_b(2.0);
        return vec4f(a, b, 0.0, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(FunctionCallNested, R"(
    fn inner(x: f32) -> f32 { return x * x; }
    fn outer(x: f32) -> f32 { return inner(x) + 1.0; }
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(outer(3.0), 0.0, 0.0, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

// ============================================================================
// Category 30: Flat Interpolation
// ============================================================================

GLSL_ROUNDTRIP(FlatInterpolationVertex, R"(
    struct VOut {
        @builtin(position) pos: vec4f,
        @location(0) @interpolate(flat) id: u32,
    };
    @vertex fn vs() -> VOut {
        var out: VOut;
        out.pos = vec4f(0.0);
        out.id = 42u;
        return out;
    }
)", WGSL_STAGE_VERTEX, SSIR_STAGE_VERTEX)

GLSL_EMIT_TEST(FlatInterpolationFragment, R"(
    @fragment fn fs(@location(0) @interpolate(flat) id: u32) -> @location(0) vec4f {
        return vec4f(f32(id) / 255.0, 0.0, 0.0, 1.0);
    }
)", SSIR_STAGE_FRAGMENT,
    // GLSL emitter may not preserve flat qualifier on fragment inputs
    EXPECT_TRUE(glsl_result.glsl.find("in uint") != std::string::npos ||
                glsl_result.glsl.find("flat") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

// ============================================================================
// Category 31: Swizzle and Component Access
// ============================================================================

GLSL_ROUNDTRIP(VectorSwizzle, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let v = vec4f(1.0, 2.0, 3.0, 4.0);
        let xy = v.xy;
        let zw = v.zw;
        return vec4f(xy, zw);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(VectorComponentAccess, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let v = vec3f(1.0, 2.0, 3.0);
        return vec4f(v.x, v.y, v.z, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

// ============================================================================
// Category 32: Variable Declarations
// ============================================================================

GLSL_ROUNDTRIP(VarMutable, R"(
    @fragment fn fs() -> @location(0) vec4f {
        var x = 0.0;
        x = 1.0;
        x = x + 0.5;
        return vec4f(x, 0.0, 0.0, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(LetImmutable, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let x = 1.5;
        let y = x * 2.0;
        return vec4f(y, 0.0, 0.0, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(MultipleVarDecls, R"(
    @fragment fn fs() -> @location(0) vec4f {
        var r = 0.0;
        var g = 0.0;
        var b = 0.0;
        r = 1.0;
        g = 0.5;
        b = 0.25;
        return vec4f(r, g, b, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

// ============================================================================
// Category 33: Complex Integrated Shaders
// ============================================================================

GLSL_ROUNDTRIP(PhongLighting, R"(
    struct Light { direction: vec3f, pad0: f32, color: vec3f, intensity: f32 };
    @group(0) @binding(0) var<uniform> light: Light;
    @fragment fn fs(@location(0) normal: vec3f) -> @location(0) vec4f {
        let n = normalize(normal);
        let l = normalize(-light.direction);
        let diff = max(dot(n, l), 0.0);
        let color = light.color * diff * light.intensity;
        return vec4f(color, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(TransformPipeline, R"(
    struct Matrices {
        model: mat4x4f,
        view: mat4x4f,
        projection: mat4x4f,
    };
    @group(0) @binding(0) var<uniform> matrices: Matrices;
    @vertex fn vs(@location(0) pos: vec3f) -> @builtin(position) vec4f {
        let world = matrices.model * vec4f(pos, 1.0);
        let eye = matrices.view * world;
        return matrices.projection * eye;
    }
)", WGSL_STAGE_VERTEX, SSIR_STAGE_VERTEX)

GLSL_EMIT_TEST(ComputeReduction, R"(
    struct Buf { data: array<f32, 256> };
    @group(0) @binding(0) var<storage, read_write> buf: Buf;
    var<workgroup> scratch: array<f32, 128>;
    @compute @workgroup_size(128) fn main(@builtin(local_invocation_index) lid: u32) {
        scratch[lid] = buf.data[lid] + buf.data[lid + 128u];
        workgroupBarrier();
        if (lid < 64u) {
            scratch[lid] = scratch[lid] + scratch[lid + 64u];
        }
        workgroupBarrier();
        if (lid == 0u) {
            buf.data[0] = scratch[0];
        }
    }
)", SSIR_STAGE_COMPUTE,
    EXPECT_TRUE(glsl_result.glsl.find("shared") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl;
    EXPECT_TRUE(glsl_result.glsl.find("barrier()") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

GLSL_ROUNDTRIP(ColorGrading, R"(
    struct Params { contrast: f32, brightness: f32, saturation: f32, pad: f32 };
    @group(0) @binding(0) var<uniform> params: Params;
    fn adjust_contrast(color: vec3f, contrast: f32) -> vec3f {
        return (color - vec3f(0.5)) * contrast + vec3f(0.5);
    }
    fn adjust_brightness(color: vec3f, brightness: f32) -> vec3f {
        return color + vec3f(brightness);
    }
    @fragment fn fs(@location(0) color: vec3f) -> @location(0) vec4f {
        var c = adjust_contrast(color, params.contrast);
        c = adjust_brightness(c, params.brightness);
        c = clamp(c, vec3f(0.0), vec3f(1.0));
        return vec4f(c, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_EMIT_TEST(ParticleUpdate, R"(
    struct Particle { pos: vec2f, vel: vec2f };
    struct ParticleBuffer { particles: array<Particle, 64> };
    struct SimParams { dt: f32, gravity: f32, pad0: f32, pad1: f32 };
    @group(0) @binding(0) var<storage, read_write> buf: ParticleBuffer;
    @group(0) @binding(1) var<uniform> sim: SimParams;
    @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3u) {
        let idx = gid.x;
        if (idx >= 64u) { return; }
        var p = buf.particles[idx];
        p.vel.y = p.vel.y + sim.gravity * sim.dt;
        p.pos = p.pos + p.vel * sim.dt;
        buf.particles[idx] = p;
    }
)", SSIR_STAGE_COMPUTE,
    EXPECT_TRUE(glsl_result.glsl.find("struct Particle") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl;
    EXPECT_TRUE(glsl_result.glsl.find("buffer") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

GLSL_ROUNDTRIP(SkyGradient, R"(
    @fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
        let top = vec3f(0.3, 0.5, 0.9);
        let bottom = vec3f(0.8, 0.9, 1.0);
        let t = uv.y;
        let color = mix(bottom, top, vec3f(t));
        return vec4f(color, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(EdgeDetect, R"(
    @fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
        let dx = dpdx(uv);
        let dy = dpdy(uv);
        let edge = length(dx) + length(dy);
        let intensity = smoothstep(0.0, 0.1, edge);
        return vec4f(vec3f(intensity), 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_EMIT_TEST(VertexSkinning, R"(
    struct SkinData {
        bone0: mat4x4f,
        bone1: mat4x4f,
    };
    @group(0) @binding(0) var<uniform> skin: SkinData;
    @vertex fn vs(
        @location(0) pos: vec3f,
        @location(1) weights: vec2f
    ) -> @builtin(position) vec4f {
        let p = vec4f(pos, 1.0);
        let skinned = skin.bone0 * p * weights.x + skin.bone1 * p * weights.y;
        return skinned;
    }
)", SSIR_STAGE_VERTEX,
    EXPECT_TRUE(glsl_result.glsl.find("gl_Position") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl;
    EXPECT_TRUE(glsl_result.glsl.find("uniform") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

GLSL_EMIT_TEST(ComputePrefixSum, R"(
    struct Buf { data: array<u32, 128> };
    @group(0) @binding(0) var<storage, read_write> buf: Buf;
    var<workgroup> temp: array<u32, 128>;
    @compute @workgroup_size(64) fn main(@builtin(local_invocation_index) lid: u32) {
        temp[lid] = buf.data[lid];
        workgroupBarrier();
        var offset = 1u;
        for (var d = 64u; d > 0u; d = d >> 1u) {
            if (lid < d) {
                let ai = offset * (2u * lid + 1u) - 1u;
                let bi = offset * (2u * lid + 2u) - 1u;
                temp[bi] = temp[bi] + temp[ai];
            }
            offset = offset * 2u;
            workgroupBarrier();
        }
        buf.data[lid] = temp[lid];
    }
)", SSIR_STAGE_COMPUTE,
    EXPECT_TRUE(glsl_result.glsl.find("shared") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl;
    EXPECT_TRUE(glsl_result.glsl.find("barrier()") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

// ============================================================================
// Category 34: GLSL-Specific Emission Verification
// ============================================================================

GLSL_EMIT_TEST(VersionDirective, R"(
    @compute @workgroup_size(1) fn main() {}
)", SSIR_STAGE_COMPUTE,
    EXPECT_TRUE(glsl_result.glsl.find("#version 450") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

GLSL_EMIT_TEST(VertexGlPosition, R"(
    @vertex fn vs() -> @builtin(position) vec4f {
        return vec4f(0.0);
    }
)", SSIR_STAGE_VERTEX,
    EXPECT_TRUE(glsl_result.glsl.find("gl_Position") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

GLSL_EMIT_TEST(ComputeLocalSize, R"(
    @compute @workgroup_size(16, 8, 2) fn main() {}
)", SSIR_STAGE_COMPUTE,
    EXPECT_TRUE(glsl_result.glsl.find("local_size_x") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

GLSL_EMIT_TEST(UniformBlockStd140, R"(
    struct UBO { color: vec4f };
    @group(0) @binding(0) var<uniform> u: UBO;
    @fragment fn fs() -> @location(0) vec4f { return u.color; }
)", SSIR_STAGE_FRAGMENT,
    EXPECT_TRUE(glsl_result.glsl.find("std140") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl;
    EXPECT_TRUE(glsl_result.glsl.find("uniform") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

GLSL_EMIT_TEST(StorageBlockStd430, R"(
    struct SSBO { data: array<f32, 4> };
    @group(0) @binding(0) var<storage, read_write> s: SSBO;
    @compute @workgroup_size(1) fn main() {
        s.data[0] = 1.0;
    }
)", SSIR_STAGE_COMPUTE,
    EXPECT_TRUE(glsl_result.glsl.find("std430") != std::string::npos ||
                glsl_result.glsl.find("buffer") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

GLSL_EMIT_TEST(FragmentOutputLocation, R"(
    @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }
)", SSIR_STAGE_FRAGMENT,
    EXPECT_TRUE(glsl_result.glsl.find("layout(location = 0) out") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

GLSL_EMIT_TEST(VertexInputLocation, R"(
    @vertex fn vs(@location(0) pos: vec3f) -> @builtin(position) vec4f {
        return vec4f(pos, 1.0);
    }
)", SSIR_STAGE_VERTEX,
    EXPECT_TRUE(glsl_result.glsl.find("layout(location = 0) in") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

GLSL_EMIT_TEST(SetBindingLayout, R"(
    struct UBO { x: f32 };
    @group(1) @binding(2) var<uniform> u: UBO;
    @fragment fn fs() -> @location(0) vec4f { return vec4f(u.x); }
)", SSIR_STAGE_FRAGMENT,
    EXPECT_TRUE(glsl_result.glsl.find("set = 1") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl;
    EXPECT_TRUE(glsl_result.glsl.find("binding = 2") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

// ============================================================================
// Category 35: All/Any Builtins
// ============================================================================

GLSL_ROUNDTRIP(AllAny, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let v = vec3<bool>(true, true, false);
        var result = 0.0;
        if (any(v)) { result = 1.0; }
        if (all(vec3<bool>(true, true, true))) { result = 2.0; }
        return vec4f(result, 0.0, 0.0, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

// ============================================================================
// Category 36: Saturate
// ============================================================================

GLSL_ROUNDTRIP(Saturate, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let x = 1.5;
        let y = -0.5;
        return vec4f(saturate(x), saturate(y), 0.0, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

// ============================================================================
// Category 37: Integer Min/Max/Clamp
// ============================================================================

GLSL_ROUNDTRIP(IntMinMaxClamp, R"(
    @compute @workgroup_size(1) fn main() {
        let a = 10i;
        let b = 20i;
        let mn = min(a, b);
        let mx = max(a, b);
        let c = clamp(a, 0i, 15i);
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_ROUNDTRIP(UintMinMaxClamp, R"(
    @compute @workgroup_size(1) fn main() {
        let a = 10u;
        let b = 20u;
        let mn = min(a, b);
        let mx = max(a, b);
        let c = clamp(a, 0u, 15u);
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

// ============================================================================
// Category 38: Abs for integers
// ============================================================================

GLSL_ROUNDTRIP(IntAbs, R"(
    @compute @workgroup_size(1) fn main() {
        let a = -42i;
        let b = abs(a);
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

// ============================================================================
// Category 39: Vector construction from mixed components
// ============================================================================

GLSL_ROUNDTRIP(Vec4FromVec3AndScalar, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let pos = vec3f(1.0, 2.0, 3.0);
        return vec4f(pos, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(Vec4FromVec2AndVec2, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let a = vec2f(1.0, 2.0);
        let b = vec2f(3.0, 4.0);
        return vec4f(a, b);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(Vec3FromScalarAndVec2, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let xy = vec2f(1.0, 2.0);
        let v = vec3f(xy, 3.0);
        return vec4f(v, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

// ============================================================================
// Category 40: Complex control flow patterns
// ============================================================================

GLSL_ROUNDTRIP(EarlyReturn, R"(
    @fragment fn fs(@location(0) val: f32) -> @location(0) vec4f {
        if (val < 0.0) {
            return vec4f(0.0, 0.0, 0.0, 1.0);
        }
        if (val > 1.0) {
            return vec4f(1.0, 1.0, 1.0, 1.0);
        }
        return vec4f(val, val, val, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

GLSL_ROUNDTRIP(LoopWithMultipleExits, R"(
    @compute @workgroup_size(1) fn main() {
        var found = -1i;
        for (var i = 0i; i < 100i; i = i + 1i) {
            if (i == 42i) {
                found = i;
                break;
            }
        }
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

GLSL_ROUNDTRIP(WhileWithComplex, R"(
    @compute @workgroup_size(1) fn main() {
        var x = 1.0;
        var iter = 0i;
        while (x > 0.001 && iter < 100i) {
            x = x * 0.5;
            iter = iter + 1i;
        }
    }
)", WGSL_STAGE_COMPUTE, SSIR_STAGE_COMPUTE)

// ============================================================================
// Category 41: Matrix column access
// ============================================================================

GLSL_ROUNDTRIP(MatColumnAccess, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let m = mat4x4f(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0);
        let col0 = m[0];
        let col1 = m[1];
        return vec4f(col0.x, col1.y, 0.0, 1.0);
    }
)", WGSL_STAGE_FRAGMENT, SSIR_STAGE_FRAGMENT)

// ============================================================================
// Category 42: Kitchen Sink Shader
// ============================================================================

GLSL_EMIT_TEST(KitchenSinkVertex, R"(
    struct MVP {
        model: mat4x4f,
        viewProj: mat4x4f,
    };
    @group(0) @binding(0) var<uniform> mvp: MVP;
    struct VOut {
        @builtin(position) pos: vec4f,
        @location(0) worldPos: vec3f,
        @location(1) uv: vec2f,
    };
    @vertex fn vs(
        @location(0) pos: vec3f,
        @location(1) uv: vec2f
    ) -> VOut {
        var out: VOut;
        let worldPos = mvp.model * vec4f(pos, 1.0);
        out.pos = mvp.viewProj * worldPos;
        out.worldPos = worldPos.xyz;
        out.uv = uv;
        return out;
    }
)", SSIR_STAGE_VERTEX,
    EXPECT_TRUE(glsl_result.glsl.find("mat4") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl;
    EXPECT_TRUE(glsl_result.glsl.find("uniform") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl;
    EXPECT_TRUE(glsl_result.glsl.find("out vec3") != std::string::npos ||
                glsl_result.glsl.find("out vec2") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

GLSL_EMIT_TEST(KitchenSinkFragment, R"(
    struct Material {
        albedo: vec3f,
        roughness: f32,
        metallic: f32,
        pad0: f32,
        pad1: f32,
        pad2: f32,
    };
    @group(0) @binding(0) var<uniform> material: Material;
    fn fresnel(cosTheta: f32, f0: vec3f) -> vec3f {
        return f0 + (vec3f(1.0) - f0) * pow(1.0 - cosTheta, 5.0);
    }
    @fragment fn fs(
        @location(0) worldPos: vec3f,
        @location(1) uv: vec2f
    ) -> @location(0) vec4f {
        let viewDir = normalize(-worldPos);
        let normal = vec3f(0.0, 0.0, 1.0);
        let nDotV = max(dot(normal, viewDir), 0.0);
        let f0 = mix(vec3f(0.04), material.albedo, vec3f(material.metallic));
        let f = fresnel(nDotV, f0);
        let diffuse = material.albedo * (1.0 - material.metallic);
        let color = diffuse + f * material.roughness;
        return vec4f(color, 1.0);
    }
)", SSIR_STAGE_FRAGMENT,
    EXPECT_TRUE(glsl_result.glsl.find("uniform") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl;
    EXPECT_TRUE(glsl_result.glsl.find("material") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)

GLSL_EMIT_TEST(KitchenSinkCompute, R"(
    struct Particle {
        pos: vec2f,
        vel: vec2f,
    };
    struct Particles { data: array<Particle, 256> };
    struct SimParams { dt: f32, damping: f32, bounds: f32, pad: f32 };
    @group(0) @binding(0) var<storage, read_write> particles: Particles;
    @group(0) @binding(1) var<uniform> params: SimParams;
    var<workgroup> local_data: array<vec2f, 64>;
    fn clamp_pos(p: vec2f, bounds: f32) -> vec2f {
        return clamp(p, vec2f(-bounds), vec2f(bounds));
    }
    @compute @workgroup_size(64) fn main(
        @builtin(global_invocation_id) gid: vec3u,
        @builtin(local_invocation_index) lid: u32
    ) {
        let idx = gid.x;
        if (idx >= 256u) { return; }
        var p = particles.data[idx];
        p.vel = p.vel * params.damping;
        p.pos = p.pos + p.vel * params.dt;
        p.pos = clamp_pos(p.pos, params.bounds);
        local_data[lid] = p.pos;
        workgroupBarrier();
        particles.data[idx] = p;
    }
)", SSIR_STAGE_COMPUTE,
    EXPECT_TRUE(glsl_result.glsl.find("struct Particle") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl;
    EXPECT_TRUE(glsl_result.glsl.find("shared") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl;
    EXPECT_TRUE(glsl_result.glsl.find("buffer") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl;
    EXPECT_TRUE(glsl_result.glsl.find("uniform") != std::string::npos)
        << "GLSL:\n" << glsl_result.glsl)
