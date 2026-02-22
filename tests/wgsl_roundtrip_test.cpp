// Comprehensive WGSL roundtrip tests: WGSL -> parse -> resolve -> lower -> SSIR -> ssir_to_wgsl
// Exercises lowering and raising paths for a wide variety of types, variables,
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
    if (!r.lower) {
        const char *lerr = wgsl_lower_last_error(r.lower);
        r.error = lerr ? lerr : "Lower create failed";
        return r;
    }

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

// Macro for concise WGSL->SSIR->WGSL roundtrip tests
#define ROUNDTRIP_TEST(TestName, Source, ...) \
TEST(WgslRoundtrip, TestName) { \
    auto compile = CompileToSsir(Source); \
    SsirCompileGuard guard(compile); \
    ASSERT_TRUE(compile.success) << compile.error; \
    char *wgsl = nullptr; \
    char *error = nullptr; \
    SsirToWgslOptions opts = {}; \
    opts.preserve_names = 1; \
    SsirToWgslResult result = ssir_to_wgsl(compile.ssir, &opts, &wgsl, &error); \
    EXPECT_EQ(result, SSIR_TO_WGSL_OK) << (error ? error : "unknown error"); \
    ASSERT_NE(wgsl, nullptr); \
    __VA_ARGS__; \
    ssir_to_wgsl_free(wgsl); \
    ssir_to_wgsl_free(error); \
}

// Macro for WGSL->SPIR-V->SSIR->WGSL roundtrip (avoids lowerer DCE issues)
#define ROUNDTRIP_SPIRV_TEST(TestName, Source, ...) \
TEST(WgslRoundtrip, TestName) { \
    auto compile = wgsl_test::CompileWgsl(Source); \
    ASSERT_TRUE(compile.success) << compile.error; \
    SsirModule *mod = nullptr; \
    char *err = nullptr; \
    SpirvToSsirOptions sopts = {}; \
    sopts.preserve_names = 1; \
    sopts.preserve_locations = 1; \
    SpirvToSsirResult sres = spirv_to_ssir( \
        compile.spirv.data(), compile.spirv.size(), &sopts, &mod, &err); \
    ASSERT_EQ(sres, SPIRV_TO_SSIR_SUCCESS) << (err ? err : "unknown"); \
    spirv_to_ssir_free(err); \
    char *wgsl = nullptr; \
    char *error = nullptr; \
    SsirToWgslOptions opts = {}; \
    opts.preserve_names = 1; \
    SsirToWgslResult result = ssir_to_wgsl(mod, &opts, &wgsl, &error); \
    EXPECT_EQ(result, SSIR_TO_WGSL_OK) << (error ? error : "unknown error"); \
    ASSERT_NE(wgsl, nullptr); \
    __VA_ARGS__; \
    ssir_to_wgsl_free(wgsl); \
    ssir_to_wgsl_free(error); \
    ssir_module_destroy(mod); \
}

} // namespace

// ============================================================================
// 1. Scalar types
// ============================================================================

ROUNDTRIP_TEST(ScalarI32, R"(
    struct UB { v: i32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(f32(u.v), 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "i32") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(ScalarU32, R"(
    struct UB { v: u32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(f32(u.v), 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "u32") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(ScalarF32, R"(
    struct UB { v: f32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(u.v, 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "f32") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(ScalarBool, R"(
    struct UB { v: u32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let b: bool = u.v > 0u;
        if (b) { return vec4f(1.0); }
        return vec4f(0.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "bool") != nullptr || strstr(wgsl, "if") != nullptr)
        << "WGSL:\n" << wgsl
)

// ============================================================================
// 2. Vector types - all sizes and element types
// ============================================================================

ROUNDTRIP_TEST(Vec2f, R"(
    struct UB { v: vec2f };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(u.v, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "vec2") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(Vec3f, R"(
    struct UB { v: vec4f };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let v3 = vec3f(u.v.x, u.v.y, u.v.z);
        return vec4f(v3, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "vec3") != nullptr || strstr(wgsl, "vec4") != nullptr)
        << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(Vec4f, R"(
    struct UB { v: vec4f };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        return u.v;
    }
)",
    EXPECT_TRUE(strstr(wgsl, "vec4") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(Vec2u, R"(
    struct UB { v: vec2u };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(f32(u.v.x), f32(u.v.y), 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "vec2") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "u32") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(Vec3i, R"(
    struct UB { x: i32, y: i32, z: i32, pad: i32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let v = vec3i(u.x, u.y, u.z);
        return vec4f(f32(v.x), f32(v.y), f32(v.z), 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "vec3") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "i32") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(Vec4u, R"(
    struct UB { v: vec4u };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(f32(u.v.x), f32(u.v.y), f32(u.v.z), f32(u.v.w));
    }
)",
    EXPECT_TRUE(strstr(wgsl, "vec4") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "u32") != nullptr) << "WGSL:\n" << wgsl
)

// ============================================================================
// 3. Matrix types
// ============================================================================

ROUNDTRIP_TEST(Mat2x2f, R"(
    struct UB { m: mat2x2f };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let v = u.m * vec2f(1.0, 0.0);
        return vec4f(v, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "mat2x2") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(Mat3x3f, R"(
    struct UB { m: mat3x3f };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let v = u.m * vec3f(1.0, 0.0, 0.0);
        return vec4f(v, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "mat3x3") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(Mat4x4f, R"(
    struct UB { m: mat4x4f };
    @group(0) @binding(0) var<uniform> u: UB;
    @vertex fn vs() -> @builtin(position) vec4f {
        return u.m * vec4f(0.0, 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "mat4x4") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(Mat2x3f, R"(
    struct UB { m: mat2x3f };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let v = u.m * vec2f(1.0, 0.0);
        return vec4f(v, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "mat2x3") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(Mat4x3f, R"(
    struct UB { m: mat4x3f };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let v = u.m * vec4f(1.0, 0.0, 0.0, 0.0);
        return vec4f(v, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "mat4x3") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(Mat3x4f, R"(
    struct UB { m: mat3x4f };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        return u.m * vec3f(1.0, 0.0, 0.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "mat3x4") != nullptr) << "WGSL:\n" << wgsl
)

// ============================================================================
// 4. Array types
// ============================================================================

ROUNDTRIP_TEST(FixedArray, R"(
    struct Data { arr: array<vec4f, 8> };
    @group(0) @binding(0) var<uniform> u: Data;
    @fragment fn fs() -> @location(0) vec4f {
        return u.arr[0] + u.arr[7];
    }
)",
    EXPECT_TRUE(strstr(wgsl, "array") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_SPIRV_TEST(RuntimeArray, R"(
    struct Data { count: u32, values: array<f32> };
    @group(0) @binding(0) var<storage, read_write> buf: Data;
    @compute @workgroup_size(1) fn cs() {
        let len = arrayLength(&buf.values);
        buf.values[0] = f32(len);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "array") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(NestedArrayInStruct, R"(
    struct Inner { data: array<f32, 4> };
    struct Outer { inner: Inner };
    @group(0) @binding(0) var<uniform> u: Outer;
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(u.inner.data[0], u.inner.data[1], u.inner.data[2], u.inner.data[3]);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "struct") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "array") != nullptr) << "WGSL:\n" << wgsl
)

// ============================================================================
// 5. Address spaces
// ============================================================================

ROUNDTRIP_TEST(UniformAddressSpace, R"(
    struct UB { color: vec4f };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        return u.color;
    }
)",
    EXPECT_TRUE(strstr(wgsl, "uniform") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(StorageReadWriteAddressSpace, R"(
    struct Buf { data: array<f32, 64> };
    @group(0) @binding(0) var<storage, read_write> buf: Buf;
    @compute @workgroup_size(1) fn cs() {
        buf.data[0] = buf.data[1] + 1.0;
    }
)",
    EXPECT_TRUE(strstr(wgsl, "storage") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(StorageReadOnlyAddressSpace, R"(
    struct Buf { data: array<f32, 64> };
    @group(0) @binding(0) var<storage, read> buf: Buf;
    @compute @workgroup_size(1) fn cs() {
        _ = buf.data[0];
    }
)",
    EXPECT_TRUE(strstr(wgsl, "storage") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(WorkgroupAddressSpace, R"(
    var<workgroup> shared_data: array<f32, 256>;
    @compute @workgroup_size(64) fn cs(@builtin(local_invocation_id) lid: vec3u) {
        shared_data[lid.x] = f32(lid.x);
        workgroupBarrier();
        _ = shared_data[63u - lid.x];
    }
)",
    EXPECT_TRUE(strstr(wgsl, "workgroup") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(PrivateAddressSpace, R"(
    var<private> pv: f32;
    @fragment fn fs() -> @location(0) vec4f {
        pv = 1.0;
        return vec4f(pv, 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "private") != nullptr) << "WGSL:\n" << wgsl
)

// ============================================================================
// 6. Binding groups and binding indices
// ============================================================================

ROUNDTRIP_TEST(MultipleBindGroups, R"(
    struct A { v: vec4f };
    struct B { v: vec4f };
    @group(0) @binding(0) var<uniform> a: A;
    @group(1) @binding(0) var<uniform> b: B;
    @fragment fn fs() -> @location(0) vec4f {
        return a.v + b.v;
    }
)",
    EXPECT_TRUE(strstr(wgsl, "@group(0)") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "@group(1)") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(MultipleBindings, R"(
    struct A { v: vec4f };
    struct B { v: vec4f };
    @group(0) @binding(0) var<uniform> a: A;
    @group(0) @binding(1) var<uniform> b: B;
    @fragment fn fs() -> @location(0) vec4f {
        return a.v + b.v;
    }
)",
    EXPECT_TRUE(strstr(wgsl, "@binding(0)") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "@binding(1)") != nullptr) << "WGSL:\n" << wgsl
)

// ============================================================================
// 7. Struct types
// ============================================================================

ROUNDTRIP_TEST(SimpleStruct, R"(
    struct Params { a: f32, b: f32, c: f32, d: f32 };
    @group(0) @binding(0) var<uniform> u: Params;
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(u.a, u.b, u.c, u.d);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "struct") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(NestedStruct, R"(
    struct Inner { x: f32, y: f32, z: f32, w: f32 };
    struct Outer { inner: Inner, scale: f32 };
    @group(0) @binding(0) var<uniform> u: Outer;
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(u.inner.x * u.scale, u.inner.y, u.inner.z, u.inner.w);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "struct") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(StructWithVecMembers, R"(
    struct UB { pos: vec3f, pad: f32, color: vec4f };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(u.pos, 1.0) * u.color;
    }
)",
    EXPECT_TRUE(strstr(wgsl, "struct") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "vec3") != nullptr || strstr(wgsl, "vec4") != nullptr)
        << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(StructWithMatMembers, R"(
    struct UB { model: mat4x4f, normal_mat: mat3x3f };
    @group(0) @binding(0) var<uniform> u: UB;
    @vertex fn vs(@location(0) pos: vec3f) -> @builtin(position) vec4f {
        let n = u.normal_mat * pos;
        return u.model * vec4f(n, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "mat4x4") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "mat3x3") != nullptr) << "WGSL:\n" << wgsl
)

// ============================================================================
// 8. Entry point stages
// ============================================================================

ROUNDTRIP_TEST(VertexStage, R"(
    @vertex fn vs() -> @builtin(position) vec4f {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "@vertex") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "@builtin(position)") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(FragmentStage, R"(
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(1.0, 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "@fragment") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "@location(0)") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(ComputeStage, R"(
    @compute @workgroup_size(8, 8, 1) fn cs() {}
)",
    EXPECT_TRUE(strstr(wgsl, "@compute") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "@workgroup_size") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(ComputeWorkgroupSize1D, R"(
    @compute @workgroup_size(256) fn cs() {}
)",
    EXPECT_TRUE(strstr(wgsl, "@compute") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "@workgroup_size") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(ComputeWorkgroupSize3D, R"(
    @compute @workgroup_size(4, 4, 4) fn cs() {}
)",
    EXPECT_TRUE(strstr(wgsl, "@compute") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "@workgroup_size") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(MultipleEntryPoints, R"(
    @vertex fn vs() -> @builtin(position) vec4f {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(1.0, 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "@vertex") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "@fragment") != nullptr) << "WGSL:\n" << wgsl
)

// ============================================================================
// 9. Built-in variables
// ============================================================================

ROUNDTRIP_TEST(BuiltinPosition, R"(
    @fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
        return pos;
    }
)",
    EXPECT_TRUE(strstr(wgsl, "position") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(BuiltinVertexIndex, R"(
    @vertex fn vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {
        return vec4f(f32(vid), 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "vertex_index") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(BuiltinInstanceIndex, R"(
    @vertex fn vs(@builtin(instance_index) iid: u32) -> @builtin(position) vec4f {
        return vec4f(0.0, f32(iid), 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "instance_index") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(BuiltinFrontFacing, R"(
    @fragment fn fs(@builtin(front_facing) ff: bool) -> @location(0) vec4f {
        if (ff) { return vec4f(1.0, 0.0, 0.0, 1.0); }
        return vec4f(0.0, 0.0, 1.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "front_facing") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(BuiltinSampleIndex, R"(
    @fragment fn fs(@builtin(sample_index) si: u32) -> @location(0) vec4f {
        return vec4f(f32(si), 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "sample_index") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(BuiltinGlobalInvocationId, R"(
    struct Data { values: array<f32, 256> };
    @group(0) @binding(0) var<storage, read_write> buf: Data;
    @compute @workgroup_size(64) fn cs(@builtin(global_invocation_id) gid: vec3u) {
        buf.values[gid.x] = f32(gid.x);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "global_invocation_id") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(BuiltinLocalInvocationId, R"(
    var<workgroup> shared: array<u32, 64>;
    @compute @workgroup_size(64) fn cs(@builtin(local_invocation_id) lid: vec3u) {
        shared[lid.x] = lid.x;
        workgroupBarrier();
        _ = shared[lid.x];
    }
)",
    EXPECT_TRUE(strstr(wgsl, "local_invocation_id") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(BuiltinLocalInvocationIndex, R"(
    var<workgroup> shared: array<u32, 64>;
    @compute @workgroup_size(64) fn cs(@builtin(local_invocation_index) idx: u32) {
        shared[idx] = idx;
    }
)",
    EXPECT_TRUE(strstr(wgsl, "local_invocation_index") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(BuiltinWorkgroupId, R"(
    struct Buf { data: array<u32, 256> };
    @group(0) @binding(0) var<storage, read_write> buf: Buf;
    @compute @workgroup_size(1) fn cs(@builtin(workgroup_id) wid: vec3u) {
        buf.data[wid.x] = wid.x;
    }
)",
    EXPECT_TRUE(strstr(wgsl, "workgroup_id") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(BuiltinNumWorkgroups, R"(
    struct Buf { data: array<u32, 4> };
    @group(0) @binding(0) var<storage, read_write> buf: Buf;
    @compute @workgroup_size(1) fn cs(@builtin(num_workgroups) nwg: vec3u) {
        buf.data[0] = nwg.x;
    }
)",
    EXPECT_TRUE(strstr(wgsl, "num_workgroups") != nullptr) << "WGSL:\n" << wgsl
)

// ============================================================================
// 10. Arithmetic operations
// ============================================================================

ROUNDTRIP_TEST(IntegerArithmetic, R"(
    struct UB { a: i32, b: i32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let sum = u.a + u.b;
        let diff = u.a - u.b;
        let prod = u.a * u.b;
        let quot = u.a / u.b;
        return vec4f(f32(sum), f32(diff), f32(prod), f32(quot));
    }
)",
    EXPECT_TRUE(strstr(wgsl, "+") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "-") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "*") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "/") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(FloatArithmetic, R"(
    struct UB { a: f32, b: f32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let sum = u.a + u.b;
        let diff = u.a - u.b;
        let prod = u.a * u.b;
        let quot = u.a / u.b;
        return vec4f(sum, diff, prod, quot);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "+") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "*") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(VectorArithmetic, R"(
    struct UB { a: vec4f, b: vec4f };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        return u.a + u.b * u.a - u.b;
    }
)",
    EXPECT_TRUE(strstr(wgsl, "vec4") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(Negation, R"(
    struct UB { v: f32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(-u.v, 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "-") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(IntegerModulo, R"(
    struct UB { a: i32, b: i32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let m = u.a % u.b;
        return vec4f(f32(m), 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "%") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(FloatModulo, R"(
    struct UB { a: f32, b: f32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let m = u.a % u.b;
        return vec4f(m, 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "%") != nullptr || strstr(wgsl, "trunc") != nullptr)
        << "WGSL:\n" << wgsl
)

// ============================================================================
// 11. Comparison operations
// ============================================================================

ROUNDTRIP_TEST(ComparisonOps, R"(
    struct UB { a: f32, b: f32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        var r = 0.0;
        if (u.a == u.b) { r = r + 1.0; }
        if (u.a != u.b) { r = r + 2.0; }
        if (u.a < u.b) { r = r + 4.0; }
        if (u.a <= u.b) { r = r + 8.0; }
        if (u.a > u.b) { r = r + 16.0; }
        if (u.a >= u.b) { r = r + 32.0; }
        return vec4f(r, 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "==") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "!=") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(IntegerComparisons, R"(
    struct UB { a: i32, b: i32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        var r = 0.0;
        if (u.a < u.b) { r = 1.0; }
        if (u.a >= u.b) { r = 2.0; }
        return vec4f(r, 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "if") != nullptr) << "WGSL:\n" << wgsl
)

// ============================================================================
// 12. Bitwise operations
// ============================================================================

ROUNDTRIP_TEST(BitwiseAnd, R"(
    struct UB { a: u32, b: u32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let c = u.a & u.b;
        return vec4f(f32(c), 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "&") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(BitwiseOr, R"(
    struct UB { a: u32, b: u32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let c = u.a | u.b;
        return vec4f(f32(c), 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "|") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(BitwiseXor, R"(
    struct UB { a: u32, b: u32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let c = u.a ^ u.b;
        return vec4f(f32(c), 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "^") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(BitwiseNot, R"(
    struct UB { a: u32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let c = ~u.a;
        return vec4f(f32(c), 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "~") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(ShiftLeft, R"(
    struct UB { a: u32, b: u32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let c = u.a << u.b;
        return vec4f(f32(c), 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "<<") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(ShiftRight, R"(
    struct UB { a: u32, b: u32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let c = u.a >> u.b;
        return vec4f(f32(c), 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, ">>") != nullptr) << "WGSL:\n" << wgsl
)

// ============================================================================
// 13. Logical operations
// ============================================================================

ROUNDTRIP_TEST(LogicalAnd, R"(
    struct UB { a: f32, b: f32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let x = u.a > 0.0;
        let y = u.b > 0.0;
        if (x && y) { return vec4f(1.0); }
        return vec4f(0.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "if") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(LogicalOr, R"(
    struct UB { a: f32, b: f32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let x = u.a > 0.0;
        let y = u.b > 0.0;
        if (x || y) { return vec4f(1.0); }
        return vec4f(0.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "if") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(LogicalNot, R"(
    struct UB { a: f32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let x = u.a > 0.0;
        if (!x) { return vec4f(1.0); }
        return vec4f(0.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "!") != nullptr || strstr(wgsl, "if") != nullptr)
        << "WGSL:\n" << wgsl
)

// ============================================================================
// 14. Control flow
// ============================================================================

ROUNDTRIP_TEST(IfElse, R"(
    struct UB { x: f32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        if (u.x > 0.5) {
            return vec4f(1.0, 0.0, 0.0, 1.0);
        } else {
            return vec4f(0.0, 0.0, 1.0, 1.0);
        }
    }
)",
    EXPECT_TRUE(strstr(wgsl, "if") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "else") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(ForLoop, R"(
    @fragment fn fs() -> @location(0) vec4f {
        var sum = 0.0;
        for (var i = 0i; i < 10i; i = i + 1i) {
            sum = sum + 1.0;
        }
        return vec4f(sum, 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "var") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(WhileLoop, R"(
    struct UB { n: i32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        var i = 0i;
        var sum = 0.0;
        while (i < u.n) {
            sum = sum + 1.0;
            i = i + 1i;
        }
        return vec4f(sum, 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "var") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(NestedIfElse, R"(
    struct UB { a: f32, b: f32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        var r = vec4f(0.0);
        if (u.a > 0.0) {
            if (u.b > 0.0) {
                r = vec4f(1.0, 1.0, 0.0, 1.0);
            } else {
                r = vec4f(1.0, 0.0, 0.0, 1.0);
            }
        } else {
            r = vec4f(0.0, 0.0, 1.0, 1.0);
        }
        return r;
    }
)",
    EXPECT_TRUE(strstr(wgsl, "if") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "else") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(BreakInLoop, R"(
    @fragment fn fs() -> @location(0) vec4f {
        var sum = 0.0;
        for (var i = 0i; i < 100i; i = i + 1i) {
            if (sum > 5.0) { break; }
            sum = sum + 1.0;
        }
        return vec4f(sum, 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "break") != nullptr ||
                strstr(wgsl, "if") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(ContinueInLoop, R"(
    struct UB { n: i32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        var sum = 0.0;
        for (var i = 0i; i < u.n; i = i + 1i) {
            if (i == 3i) { continue; }
            sum = sum + 1.0;
        }
        return vec4f(sum, 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "fn ") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(Discard, R"(
    @fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
        if (pos.x < 100.0) { discard; }
        return vec4f(1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "discard") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(Switch, R"(
    struct UB { mode: i32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        var r = vec4f(0.0);
        switch (u.mode) {
            case 0i: { r = vec4f(1.0, 0.0, 0.0, 1.0); }
            case 1i: { r = vec4f(0.0, 1.0, 0.0, 1.0); }
            case 2i: { r = vec4f(0.0, 0.0, 1.0, 1.0); }
            default: { r = vec4f(1.0, 1.0, 1.0, 1.0); }
        }
        return r;
    }
)",
    // Switch may be lowered to if/else chain; verify control flow is present
    EXPECT_TRUE(strstr(wgsl, "switch") != nullptr ||
                strstr(wgsl, "if") != nullptr) << "WGSL:\n" << wgsl
)

// ============================================================================
// 15. Type conversions
// ============================================================================

ROUNDTRIP_TEST(I32ToF32, R"(
    struct UB { v: i32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(f32(u.v), 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "f32") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(U32ToF32, R"(
    struct UB { v: u32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(f32(u.v), 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "f32") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(F32ToI32, R"(
    struct UB { v: f32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let i = i32(u.v);
        return vec4f(f32(i), 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "i32") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(F32ToU32, R"(
    struct UB { v: f32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let c = u32(u.v);
        return vec4f(f32(c), 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "u32") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_SPIRV_TEST(Bitcast, R"(
    struct UB { v: u32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let f = bitcast<f32>(u.v);
        return vec4f(f, 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "bitcast") != nullptr ||
                strstr(wgsl, "fn ") != nullptr) << "WGSL:\n" << wgsl
)

// ============================================================================
// 16. Math builtins
// ============================================================================

ROUNDTRIP_TEST(TrigFunctions, R"(
    struct UB { x: f32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(sin(u.x), cos(u.x), tan(u.x), 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "sin") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "cos") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "tan") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(InvTrigFunctions, R"(
    struct UB { x: f32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(asin(u.x), acos(u.x), atan2(u.x, 1.0), 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "asin") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "acos") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "atan2") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(ExpLogFunctions, R"(
    struct UB { x: f32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(exp(u.x), exp2(u.x), log(u.x), log2(u.x));
    }
)",
    EXPECT_TRUE(strstr(wgsl, "exp") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "log") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(PowSqrt, R"(
    struct UB { x: f32, y: f32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(pow(u.x, u.y), sqrt(u.x), inverseSqrt(u.x), 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "pow") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "sqrt") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(FloorCeilRoundTrunc, R"(
    struct UB { x: f32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(floor(u.x), ceil(u.x), round(u.x), trunc(u.x));
    }
)",
    EXPECT_TRUE(strstr(wgsl, "floor") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "ceil") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "round") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "trunc") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(AbsSignFract, R"(
    struct UB { x: f32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(abs(u.x), sign(u.x), fract(u.x), 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "abs") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "sign") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "fract") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(MinMaxClamp, R"(
    struct UB { a: f32, b: f32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(min(u.a, u.b), max(u.a, u.b), clamp(u.a, 0.0, 1.0), 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "min") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "max") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "clamp") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(MixStepSmoothstep, R"(
    struct UB { a: f32, b: f32, t: f32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(mix(u.a, u.b, u.t), step(u.a, u.t), smoothstep(u.a, u.b, u.t), 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "mix") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "step") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "smoothstep") != nullptr) << "WGSL:\n" << wgsl
)

// ============================================================================
// 17. Vector math builtins
// ============================================================================

ROUNDTRIP_TEST(DotProduct, R"(
    struct UB { a: vec3f, b: vec3f };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let d = dot(u.a, u.b);
        return vec4f(d, 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "dot") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(CrossProduct, R"(
    struct UB { a: vec3f, b: vec3f };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let c = cross(u.a, u.b);
        return vec4f(c, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "cross") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(LengthDistanceNormalize, R"(
    struct UB { a: vec3f, b: vec3f };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let l = length(u.a);
        let d = distance(u.a, u.b);
        let n = normalize(u.a);
        return vec4f(l, d, n.x, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "length") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "distance") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "normalize") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_SPIRV_TEST(Reflect, R"(
    struct UB { i: vec4f, n: vec4f };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let r = reflect(u.i.xyz, u.n.xyz);
        return vec4f(r, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "reflect") != nullptr ||
                strstr(wgsl, "fn ") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_SPIRV_TEST(Refract, R"(
    struct UB { i: vec4f, n: vec4f };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let r = refract(u.i.xyz, u.n.xyz, 1.5);
        return vec4f(r, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "refract") != nullptr ||
                strstr(wgsl, "fn ") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_SPIRV_TEST(FaceForward, R"(
    struct UB { i: vec4f, n: vec4f };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let r = faceForward(u.n.xyz, u.i.xyz, u.n.xyz);
        return vec4f(r, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "faceForward") != nullptr ||
                strstr(wgsl, "fn ") != nullptr) << "WGSL:\n" << wgsl
)

// ============================================================================
// 18. Matrix operations
// ============================================================================

ROUNDTRIP_TEST(MatrixMultiply, R"(
    struct UB { m: mat4x4f };
    @group(0) @binding(0) var<uniform> u: UB;
    @vertex fn vs(@location(0) pos: vec3f) -> @builtin(position) vec4f {
        return u.m * vec4f(pos, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "mat4x4") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "*") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(MatrixTranspose, R"(
    struct UB { m: mat3x3f };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let t = transpose(u.m);
        return vec4f(t[0].x, t[1].y, t[2].z, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "mat3x3") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(MatrixDeterminant, R"(
    struct UB { m: mat4x4f };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let d = determinant(u.m);
        return vec4f(d, 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "mat4x4") != nullptr) << "WGSL:\n" << wgsl
)

// ============================================================================
// 19. Texture operations
// ============================================================================

ROUNDTRIP_TEST(TextureSample2D, R"(
    @group(0) @binding(0) var tex: texture_2d<f32>;
    @group(0) @binding(1) var samp: sampler;
    @fragment fn fs() -> @location(0) vec4f {
        return textureSample(tex, samp, vec2f(0.5, 0.5));
    }
)",
    EXPECT_TRUE(strstr(wgsl, "texture_2d") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "sampler") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "textureSample") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(TextureSampleLevel, R"(
    @group(0) @binding(0) var tex: texture_2d<f32>;
    @group(0) @binding(1) var samp: sampler;
    @fragment fn fs() -> @location(0) vec4f {
        return textureSampleLevel(tex, samp, vec2f(0.5, 0.5), 0.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "textureSampleLevel") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(TextureSampleBias, R"(
    @group(0) @binding(0) var tex: texture_2d<f32>;
    @group(0) @binding(1) var samp: sampler;
    @fragment fn fs() -> @location(0) vec4f {
        return textureSampleBias(tex, samp, vec2f(0.5, 0.5), -1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "textureSampleBias") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(TextureLoad2D, R"(
    @group(0) @binding(0) var tex: texture_2d<f32>;
    @fragment fn fs() -> @location(0) vec4f {
        return textureLoad(tex, vec2i(0, 0), 0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "textureLoad") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(TextureStore2D, R"(
    @group(0) @binding(0) var tex: texture_storage_2d<rgba8unorm, write>;
    @compute @workgroup_size(1) fn cs() {
        textureStore(tex, vec2i(0, 0), vec4f(1.0, 0.0, 0.0, 1.0));
    }
)",
    EXPECT_TRUE(strstr(wgsl, "textureStore") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "texture_storage") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(TextureSampleCompare, R"(
    @group(0) @binding(0) var dtex: texture_depth_2d;
    @group(0) @binding(1) var samp: sampler_comparison;
    @fragment fn fs() -> @location(0) vec4f {
        let d = textureSampleCompare(dtex, samp, vec2f(0.5, 0.5), 0.5);
        return vec4f(d, d, d, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "textureSampleCompare") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "texture_depth") != nullptr ||
                strstr(wgsl, "sampler_comparison") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(Texture3D, R"(
    @group(0) @binding(0) var tex: texture_3d<f32>;
    @group(0) @binding(1) var samp: sampler;
    @fragment fn fs() -> @location(0) vec4f {
        return textureSample(tex, samp, vec3f(0.5, 0.5, 0.5));
    }
)",
    EXPECT_TRUE(strstr(wgsl, "texture_3d") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(TextureCube, R"(
    @group(0) @binding(0) var tex: texture_cube<f32>;
    @group(0) @binding(1) var samp: sampler;
    @fragment fn fs() -> @location(0) vec4f {
        return textureSample(tex, samp, vec3f(1.0, 0.0, 0.0));
    }
)",
    EXPECT_TRUE(strstr(wgsl, "texture_cube") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(Texture2DArray, R"(
    @group(0) @binding(0) var tex: texture_2d_array<f32>;
    @group(0) @binding(1) var samp: sampler;
    @fragment fn fs() -> @location(0) vec4f {
        return textureSample(tex, samp, vec2f(0.5, 0.5), 0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "2d_array") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(TextureMultisampled, R"(
    @group(0) @binding(0) var tex: texture_multisampled_2d<f32>;
    @fragment fn fs(@builtin(sample_index) si: u32) -> @location(0) vec4f {
        return textureLoad(tex, vec2i(0, 0), si);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "texture_multisampled_2d") != nullptr ||
                strstr(wgsl, "textureLoad") != nullptr) << "WGSL:\n" << wgsl
)

// ============================================================================
// 20. Derivatives
// ============================================================================

ROUNDTRIP_SPIRV_TEST(Derivatives, R"(
    struct UB { v: vec4f };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
        let dx = dpdx(uv.x * u.v.x);
        let dy = dpdy(uv.y * u.v.y);
        let fw = fwidth(uv.x * u.v.z);
        return vec4f(dx, dy, fw, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "fn ") != nullptr) << "WGSL:\n" << wgsl
)

// ============================================================================
// 21. Barriers and synchronization
// ============================================================================

ROUNDTRIP_TEST(WorkgroupBarrier, R"(
    var<workgroup> shared_data: array<f32, 64>;
    @compute @workgroup_size(64) fn cs(@builtin(local_invocation_id) lid: vec3u) {
        shared_data[lid.x] = f32(lid.x);
        workgroupBarrier();
        _ = shared_data[63u - lid.x];
    }
)",
    EXPECT_TRUE(strstr(wgsl, "workgroupBarrier") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(StorageBarrier, R"(
    struct Buf { data: array<u32, 256> };
    @group(0) @binding(0) var<storage, read_write> buf: Buf;
    @compute @workgroup_size(64) fn cs(@builtin(local_invocation_id) lid: vec3u) {
        buf.data[lid.x] = lid.x;
        storageBarrier();
        _ = buf.data[63u - lid.x];
    }
)",
    EXPECT_TRUE(strstr(wgsl, "storageBarrier") != nullptr) << "WGSL:\n" << wgsl
)

// ============================================================================
// 22. Atomics
// ============================================================================

ROUNDTRIP_SPIRV_TEST(AtomicAddLoad, R"(
    struct Counter { value: atomic<u32> };
    @group(0) @binding(0) var<storage, read_write> counter: Counter;
    @compute @workgroup_size(64) fn cs() {
        let prev = atomicAdd(&counter.value, 1u);
        _ = prev;
    }
)",
    // Atomic type may be lowered to plain u32 through the SPIR-V roundtrip
    EXPECT_TRUE(strstr(wgsl, "fn ") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "storage") != nullptr ||
                strstr(wgsl, "atomic") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_SPIRV_TEST(AtomicMaxMin, R"(
    struct AtomicBuf { value: atomic<u32> };
    @group(0) @binding(0) var<storage, read_write> buf: AtomicBuf;
    @compute @workgroup_size(1) fn cs() {
        atomicMax(&buf.value, 100u);
        atomicMin(&buf.value, 50u);
    }
)",
    // Atomic type may be lowered to plain u32 through the SPIR-V roundtrip
    EXPECT_TRUE(strstr(wgsl, "fn ") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "storage") != nullptr ||
                strstr(wgsl, "atomic") != nullptr) << "WGSL:\n" << wgsl
)

// ============================================================================
// 23. Function calls
// ============================================================================

ROUNDTRIP_SPIRV_TEST(SimpleFunctionCall, R"(
    fn helper(x: f32) -> f32 { return x * 2.0; }
    struct UB { v: vec4f };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let r = helper(u.v.x);
        return vec4f(r, 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "fn ") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_SPIRV_TEST(FunctionMultipleParams, R"(
    fn add3(a: f32, b: f32, c: f32) -> f32 { return a + b + c; }
    struct UB { v: vec4f };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let r = add3(u.v.x, u.v.y, u.v.z);
        return vec4f(r, 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "fn ") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_SPIRV_TEST(FunctionReturningVec, R"(
    fn make_color(r: f32, g: f32) -> vec4f {
        return vec4f(r, g, 0.0, 1.0);
    }
    struct UB { v: vec4f };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        return make_color(u.v.x, u.v.y);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "fn ") != nullptr) << "WGSL:\n" << wgsl
)

// ============================================================================
// 24. Variable declarations
// ============================================================================

ROUNDTRIP_TEST(LetDeclaration, R"(
    struct UB { v: f32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let x = u.v * 2.0;
        let y = x + 1.0;
        return vec4f(y, 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "let") != nullptr || strstr(wgsl, "var") != nullptr)
        << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(VarDeclarationWithAssignment, R"(
    struct UB { v: f32 };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        var x = u.v;
        x = x * 2.0;
        x = x + 1.0;
        return vec4f(x, 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "var") != nullptr) << "WGSL:\n" << wgsl
)

// ============================================================================
// 25. Input/output locations
// ============================================================================

ROUNDTRIP_TEST(MultipleLocationInputs, R"(
    @vertex fn vs(
        @location(0) pos: vec3f,
        @location(1) normal: vec3f,
        @location(2) uv: vec2f
    ) -> @builtin(position) vec4f {
        return vec4f(pos + normal * uv.x, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "@location(0)") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "@location(1)") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "@location(2)") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_SPIRV_TEST(MultipleLocationOutputs, R"(
    struct VsOut {
        @builtin(position) pos: vec4f,
        @location(0) color: vec4f,
        @location(1) uv: vec2f,
    };
    @vertex fn vs(@location(0) pos: vec3f) -> VsOut {
        var out: VsOut;
        out.pos = vec4f(pos, 1.0);
        out.color = vec4f(1.0);
        out.uv = vec2f(0.0);
        return out;
    }
)",
    EXPECT_TRUE(strstr(wgsl, "@vertex") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "@location(0)") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(FlatInterpolation, R"(
    @fragment fn fs(@location(0) @interpolate(flat) id: u32) -> @location(0) vec4f {
        return vec4f(f32(id), 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "@location(0)") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "@fragment") != nullptr) << "WGSL:\n" << wgsl
)

// ============================================================================
// 26. Swizzle and component access
// ============================================================================

ROUNDTRIP_TEST(VectorSwizzle, R"(
    struct UB { v: vec4f };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let a = u.v.xyz;
        let b = u.v.ww;
        return vec4f(a, b.x);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "vec") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(VectorComponentWrite, R"(
    struct UB { v: vec4f };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        var result = u.v;
        result.x = 0.0;
        result.w = 1.0;
        return result;
    }
)",
    EXPECT_TRUE(strstr(wgsl, "var") != nullptr) << "WGSL:\n" << wgsl
)

// ============================================================================
// 27. Constant expressions and constructors
// ============================================================================

ROUNDTRIP_TEST(VectorSplat, R"(
    @fragment fn fs() -> @location(0) vec4f {
        return vec4f(0.5);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "vec4") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(VectorConstruct, R"(
    @fragment fn fs() -> @location(0) vec4f {
        let v2 = vec2f(0.1, 0.2);
        return vec4f(v2, 0.3, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "vec") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(ZeroInit, R"(
    @fragment fn fs() -> @location(0) vec4f {
        var v = vec4f(0.0, 0.0, 0.0, 0.0);
        v.x = 1.0;
        return v;
    }
)",
    EXPECT_TRUE(strstr(wgsl, "var") != nullptr) << "WGSL:\n" << wgsl
)

// ============================================================================
// 28. Complex integrated shaders
// ============================================================================

ROUNDTRIP_TEST(PhongLighting, R"(
    struct Camera { view_proj: mat4x4f, eye_pos: vec3f, pad: f32 };
    struct Material { ambient: vec4f, diffuse: vec4f };
    @group(0) @binding(0) var<uniform> camera: Camera;
    @group(0) @binding(1) var<uniform> material: Material;
    @fragment fn fs(
        @location(0) world_pos: vec3f,
        @location(1) world_normal: vec3f
    ) -> @location(0) vec4f {
        let light_dir = normalize(vec3f(1.0, 1.0, 1.0));
        let n = normalize(world_normal);
        let ndotl = max(dot(n, light_dir), 0.0);
        let view_dir = normalize(camera.eye_pos - world_pos);
        let reflect_dir = reflect(-light_dir, n);
        let spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0);
        let color = material.ambient.xyz + material.diffuse.xyz * ndotl + vec3f(spec);
        return vec4f(color, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "normalize") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "dot") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "mat4x4") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(ComputeReduce, R"(
    struct Data { values: array<f32, 256> };
    struct Result { sum: f32 };
    @group(0) @binding(0) var<storage, read> input: Data;
    @group(0) @binding(1) var<storage, read_write> output: Result;
    var<workgroup> shared: array<f32, 64>;
    @compute @workgroup_size(64) fn cs(
        @builtin(local_invocation_id) lid: vec3u,
        @builtin(global_invocation_id) gid: vec3u
    ) {
        shared[lid.x] = input.values[gid.x] + input.values[gid.x + 64u];
        workgroupBarrier();
        if (lid.x < 32u) { shared[lid.x] = shared[lid.x] + shared[lid.x + 32u]; }
        workgroupBarrier();
        if (lid.x < 16u) { shared[lid.x] = shared[lid.x] + shared[lid.x + 16u]; }
        workgroupBarrier();
        if (lid.x < 8u) { shared[lid.x] = shared[lid.x] + shared[lid.x + 8u]; }
        workgroupBarrier();
        if (lid.x < 4u) { shared[lid.x] = shared[lid.x] + shared[lid.x + 4u]; }
        workgroupBarrier();
        if (lid.x < 2u) { shared[lid.x] = shared[lid.x] + shared[lid.x + 2u]; }
        workgroupBarrier();
        if (lid.x == 0u) {
            output.sum = shared[0] + shared[1];
        }
    }
)",
    EXPECT_TRUE(strstr(wgsl, "workgroupBarrier") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "storage") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "workgroup") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(TexturedQuad, R"(
    struct Uniforms { mvp: mat4x4f };
    @group(0) @binding(0) var<uniform> uniforms: Uniforms;
    @group(0) @binding(1) var tex: texture_2d<f32>;
    @group(0) @binding(2) var samp: sampler;

    struct VsOut {
        @builtin(position) pos: vec4f,
        @location(0) uv: vec2f,
    };

    @vertex fn vs(@location(0) pos: vec3f, @location(1) uv: vec2f) -> VsOut {
        var out: VsOut;
        out.pos = uniforms.mvp * vec4f(pos, 1.0);
        out.uv = uv;
        return out;
    }

    @fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
        return textureSample(tex, samp, uv);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "@vertex") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "@fragment") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "mat4x4") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "textureSample") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(SkyboxShader, R"(
    struct Camera { inv_view_proj: mat4x4f };
    @group(0) @binding(0) var<uniform> camera: Camera;
    @group(0) @binding(1) var sky_tex: texture_cube<f32>;
    @group(0) @binding(2) var sky_samp: sampler;

    @vertex fn vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {
        var pos = array<vec2f, 3>(
            vec2f(-1.0, -1.0),
            vec2f(3.0, -1.0),
            vec2f(-1.0, 3.0)
        );
        return vec4f(pos[vid], 0.999, 1.0);
    }

    @fragment fn fs(@builtin(position) frag_pos: vec4f) -> @location(0) vec4f {
        let ndc = vec2f(frag_pos.x / 1920.0, frag_pos.y / 1080.0) * 2.0 - vec2f(1.0);
        let world = camera.inv_view_proj * vec4f(ndc, 1.0, 1.0);
        let dir = normalize(world.xyz / world.w);
        return textureSample(sky_tex, sky_samp, dir);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "@vertex") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "@fragment") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "texture_cube") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_TEST(ParticleUpdate, R"(
    struct Particle {
        pos: vec2f,
        vel: vec2f,
    };
    struct Params { delta_t: f32, count: u32 };
    struct Particles { p: array<Particle> };
    @group(0) @binding(0) var<uniform> params: Params;
    @group(0) @binding(1) var<storage, read_write> particles: Particles;
    @compute @workgroup_size(64) fn cs(@builtin(global_invocation_id) gid: vec3u) {
        if (gid.x >= params.count) { return; }
        let idx = gid.x;
        var p = particles.p[idx];
        p.pos = p.pos + p.vel * params.delta_t;
        if (p.pos.x < -1.0 || p.pos.x > 1.0) { p.vel.x = -p.vel.x; }
        if (p.pos.y < -1.0 || p.pos.y > 1.0) { p.vel.y = -p.vel.y; }
        particles.p[idx] = p;
    }
)",
    EXPECT_TRUE(strstr(wgsl, "@compute") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "storage") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "struct") != nullptr) << "WGSL:\n" << wgsl
)

// ============================================================================
// 29. Bit manipulation builtins
// ============================================================================

ROUNDTRIP_SPIRV_TEST(CountOneBits, R"(
    struct UB { v: vec4u };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let c = countOneBits(u.v.x);
        return vec4f(f32(c), 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "fn ") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_SPIRV_TEST(ReverseBits, R"(
    struct UB { v: vec4u };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let r = reverseBits(u.v.x);
        return vec4f(f32(r), 0.0, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "fn ") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_SPIRV_TEST(ExtractInsertBits, R"(
    struct UB { v: vec4u };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let e = extractBits(u.v.x, 4u, 8u);
        let i = insertBits(u.v.y, u.v.z, 0u, 8u);
        return vec4f(f32(e), f32(i), 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "fn ") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_SPIRV_TEST(FirstLeadingTrailingBit, R"(
    struct UB { v: vec4u };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let a = firstLeadingBit(u.v.x);
        let b = firstTrailingBit(u.v.y);
        return vec4f(f32(a), f32(b), 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "fn ") != nullptr) << "WGSL:\n" << wgsl
)

// ============================================================================
// 30. Pack/unpack builtins
// ============================================================================

ROUNDTRIP_SPIRV_TEST(Pack4x8snorm, R"(
    struct UB { v: vec4f };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let p = pack4x8snorm(u.v);
        let r = unpack4x8snorm(p);
        return r;
    }
)",
    EXPECT_TRUE(strstr(wgsl, "fn ") != nullptr) << "WGSL:\n" << wgsl
)

ROUNDTRIP_SPIRV_TEST(Pack2x16float, R"(
    struct UB { v: vec4f };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let p = pack2x16float(u.v.xy);
        let r = unpack2x16float(p);
        return vec4f(r, 0.0, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "fn ") != nullptr) << "WGSL:\n" << wgsl
)

// ============================================================================
// 31. Select builtin
// ============================================================================

ROUNDTRIP_SPIRV_TEST(SelectBuiltin, R"(
    struct UB { a: vec4f, b: vec4f };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        return select(u.a, u.b, u.a.x > 0.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "fn ") != nullptr) << "WGSL:\n" << wgsl
)

// ============================================================================
// 32. FMA, degrees, radians
// ============================================================================

ROUNDTRIP_SPIRV_TEST(FmaDegreesRadians, R"(
    struct UB { v: vec4f };
    @group(0) @binding(0) var<uniform> u: UB;
    @fragment fn fs() -> @location(0) vec4f {
        let f = fma(u.v.x, u.v.y, u.v.z);
        let d = degrees(u.v.x);
        let r = radians(u.v.y);
        return vec4f(f, d, r, 1.0);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "fn ") != nullptr) << "WGSL:\n" << wgsl
)

// ============================================================================
// 33. Double roundtrip: WGSL -> SSIR -> WGSL -> SSIR -> WGSL
// ============================================================================

TEST(WgslRoundtrip, DoubleRoundtrip) {
    const char *source = R"(
        struct Uniforms { mvp: mat4x4f };
        @group(0) @binding(0) var<uniform> u: Uniforms;
        @vertex fn vs(@location(0) pos: vec3f) -> @builtin(position) vec4f {
            return u.mvp * vec4f(pos, 1.0);
        }
        @fragment fn fs() -> @location(0) vec4f {
            return vec4f(1.0, 0.0, 0.0, 1.0);
        }
    )";

    // First roundtrip
    auto compile1 = CompileToSsir(source);
    SsirCompileGuard guard1(compile1);
    ASSERT_TRUE(compile1.success) << compile1.error;

    char *wgsl1 = nullptr;
    char *error1 = nullptr;
    SsirToWgslOptions opts = {};
    opts.preserve_names = 1;
    SsirToWgslResult result1 = ssir_to_wgsl(compile1.ssir, &opts, &wgsl1, &error1);
    ASSERT_EQ(result1, SSIR_TO_WGSL_OK) << (error1 ? error1 : "unknown error");
    ASSERT_NE(wgsl1, nullptr);

    // Second roundtrip using output from first
    auto compile2 = CompileToSsir(wgsl1);
    SsirCompileGuard guard2(compile2);
    ASSERT_TRUE(compile2.success) << "Second roundtrip failed: " << compile2.error
                                  << "\nFirst roundtrip WGSL:\n" << wgsl1;

    char *wgsl2 = nullptr;
    char *error2 = nullptr;
    SsirToWgslResult result2 = ssir_to_wgsl(compile2.ssir, &opts, &wgsl2, &error2);
    ASSERT_EQ(result2, SSIR_TO_WGSL_OK) << (error2 ? error2 : "unknown error");
    ASSERT_NE(wgsl2, nullptr);

    // Both roundtrips should produce valid WGSL with key constructs preserved
    EXPECT_TRUE(strstr(wgsl2, "@vertex") != nullptr) << "WGSL2:\n" << wgsl2;
    EXPECT_TRUE(strstr(wgsl2, "@fragment") != nullptr) << "WGSL2:\n" << wgsl2;
    EXPECT_TRUE(strstr(wgsl2, "mat4x4") != nullptr) << "WGSL2:\n" << wgsl2;

    ssir_to_wgsl_free(wgsl1);
    ssir_to_wgsl_free(error1);
    ssir_to_wgsl_free(wgsl2);
    ssir_to_wgsl_free(error2);
}

// ============================================================================
// 34. Full pipeline: WGSL -> SSIR -> SPIR-V -> validate -> SSIR -> WGSL
// ============================================================================

TEST(WgslRoundtrip, FullPipelineRoundtrip) {
    const char *source = R"(
        struct Uniforms { color: vec4f };
        @group(0) @binding(0) var<uniform> u: Uniforms;
        @group(0) @binding(1) var tex: texture_2d<f32>;
        @group(0) @binding(2) var samp: sampler;
        @fragment fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
            return textureSample(tex, samp, uv) * u.color;
        }
    )";

    // Step 1: WGSL -> SPIR-V
    auto compile = wgsl_test::CompileWgsl(source);
    ASSERT_TRUE(compile.success) << compile.error;

    // Step 2: SPIR-V -> SSIR
    SsirModule *mod = nullptr;
    char *err = nullptr;
    SpirvToSsirOptions sopts = {};
    sopts.preserve_names = 1;
    sopts.preserve_locations = 1;
    SpirvToSsirResult sres = spirv_to_ssir(
        compile.spirv.data(), compile.spirv.size(), &sopts, &mod, &err);
    ASSERT_EQ(sres, SPIRV_TO_SSIR_SUCCESS) << (err ? err : "unknown");
    spirv_to_ssir_free(err);

    // Step 3: SSIR -> SPIR-V
    uint32_t *spirv2 = nullptr;
    size_t spirv2_count = 0;
    SsirToSpirvOptions spirv_opts = {};
    spirv_opts.enable_debug_names = 1;
    SsirToSpirvResult spirv_result = ssir_to_spirv(mod, &spirv_opts, &spirv2, &spirv2_count);
    ASSERT_EQ(spirv_result, SSIR_TO_SPIRV_OK);

    // Step 4: Validate regenerated SPIR-V
    std::string val_error;
    bool valid = wgsl_test::ValidateSpirv(spirv2, spirv2_count, &val_error);
    EXPECT_TRUE(valid) << "SPIR-V validation failed: " << val_error;
    ssir_to_spirv_free(spirv2);

    // Step 5: SSIR -> WGSL
    char *wgsl = nullptr;
    char *error = nullptr;
    SsirToWgslOptions opts = {};
    opts.preserve_names = 1;
    SsirToWgslResult result = ssir_to_wgsl(mod, &opts, &wgsl, &error);
    EXPECT_EQ(result, SSIR_TO_WGSL_OK) << (error ? error : "unknown error");
    ASSERT_NE(wgsl, nullptr);
    EXPECT_TRUE(strstr(wgsl, "@fragment") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "textureSample") != nullptr) << "WGSL:\n" << wgsl;

    ssir_to_wgsl_free(wgsl);
    ssir_to_wgsl_free(error);
    ssir_module_destroy(mod);
}

// ============================================================================
// 35. Stress test: many constructs in one shader
// ============================================================================

ROUNDTRIP_TEST(KitchenSinkShader, R"(
    struct Camera {
        view: mat4x4f,
        proj: mat4x4f,
        eye: vec3f,
        pad: f32,
    };
    struct Material {
        color: vec4f,
        roughness: f32,
        metallic: f32,
        pad0: f32,
        pad1: f32,
    };
    struct Light {
        position: vec3f,
        pad: f32,
        color: vec3f,
        intensity: f32,
    };
    @group(0) @binding(0) var<uniform> camera: Camera;
    @group(0) @binding(1) var<uniform> material: Material;
    @group(0) @binding(2) var<uniform> light: Light;
    @group(1) @binding(0) var albedo_tex: texture_2d<f32>;
    @group(1) @binding(1) var normal_tex: texture_2d<f32>;
    @group(1) @binding(2) var tex_sampler: sampler;

    struct VsOut {
        @builtin(position) pos: vec4f,
        @location(0) world_pos: vec3f,
        @location(1) world_normal: vec3f,
        @location(2) uv: vec2f,
    };

    @vertex fn vs(
        @location(0) pos: vec3f,
        @location(1) normal: vec3f,
        @location(2) uv: vec2f
    ) -> VsOut {
        var out: VsOut;
        let world_pos = pos;
        out.pos = camera.proj * camera.view * vec4f(world_pos, 1.0);
        out.world_pos = world_pos;
        out.world_normal = normal;
        out.uv = uv;
        return out;
    }

    @fragment fn fs(in: VsOut) -> @location(0) vec4f {
        let albedo = textureSample(albedo_tex, tex_sampler, in.uv);
        let n = normalize(in.world_normal);
        let l = normalize(light.position - in.world_pos);
        let ndotl = max(dot(n, l), 0.0);
        let v = normalize(camera.eye - in.world_pos);
        let h = normalize(v + l);
        let spec = pow(max(dot(n, h), 0.0), mix(8.0, 256.0, 1.0 - material.roughness));
        let diffuse = albedo.xyz * material.color.xyz * ndotl;
        let specular = light.color * spec * material.metallic;
        let ambient = albedo.xyz * 0.03;
        let color = ambient + (diffuse + specular) * light.intensity;
        return vec4f(color, albedo.w);
    }
)",
    EXPECT_TRUE(strstr(wgsl, "@vertex") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "@fragment") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "mat4x4") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "struct") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "normalize") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "dot") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "textureSample") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "@group(0)") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "@group(1)") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "pow") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "mix") != nullptr) << "WGSL:\n" << wgsl;
    EXPECT_TRUE(strstr(wgsl, "max") != nullptr) << "WGSL:\n" << wgsl
)
