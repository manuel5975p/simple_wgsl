// FILE: wgsl_resolve_test.c
#include "wgsl_parser.h"
#include "wgsl_resolve.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        tests_failed++; \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    } else { \
        tests_passed++; \
    } \
} while (0)

static int name_eq(const char *a, const char *b) { return a && b && strcmp(a,b) == 0; }

static int find_entrypoint_index(const WgslResolverEntrypoint *eps, int n, const char *name) {
    for (int i = 0; i < n; i++) if (name_eq(eps[i].name, name)) return i;
    return -1;
}

static int has_binding_named(const WgslSymbolInfo *syms, int n, const char *name, int group, int binding) {
    for (int i = 0; i < n; i++) {
        if (name_eq(syms[i].name, name) && syms[i].has_group && syms[i].has_binding &&
            syms[i].group_index == group && syms[i].binding_index == binding)
            return 1;
    }
    return 0;
}

static int has_global_named(const WgslSymbolInfo *syms, int n, const char *name) {
    for (int i = 0; i < n; i++) if (name_eq(syms[i].name, name)) return 1;
    return 0;
}

static int count_with_location(const WgslVertexSlot *slots, int n, int loc) {
    int c = 0;
    for (int i = 0; i < n; i++) if (slots[i].location == loc) c++;
    return c;
}

static void test_vertex_fragment_suite(void) {
    const char *src = R"(
const konschtante = 0.5f;
struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv: vec2f,
    @location(2) normal: vec3f,
    @location(3) color: vec4f,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
    @location(1) color: vec4f,
};

struct LightBuffer {
    count: u32,
    positions: array<vec3f>
};

@group(0) @binding(0) var<uniform> Perspective_View: mat4x4f;
@group(0) @binding(1) var texture0: texture_2d<f32>;
@group(0) @binding(2) var texSampler: sampler;
@group(0) @binding(3) var<storage> modelMatrix: array<mat4x4f>;
@group(0) @binding(4) var<storage> lights: LightBuffer;
@group(0) @binding(5) var<storage> lights2: LightBuffer;

@vertex
fn vs_main(@builtin(instance_index) instanceIdx : u32, in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = Perspective_View * modelMatrix[instanceIdx] * vec4f(in.position.xyz, 1.0f);
    out.color = in.color;
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    return textureSample(texture0, texSampler, in.uv).rgba * in.color;
}
)";
    WgslAstNode *ast = wgsl_parse(src);
    CHECK(ast != NULL);
    WgslResolver *r = wgsl_resolver_build(ast);
    CHECK(r != NULL);

    int epn = 0;
    const WgslResolverEntrypoint *eps = wgsl_resolver_entrypoints(r, &epn);
    CHECK(eps != NULL && epn == 2);
    int vi = find_entrypoint_index(eps, epn, "vs_main");
    int fi = find_entrypoint_index(eps, epn, "fs_main");
    CHECK(vi >= 0 && fi >= 0);
    CHECK(eps[vi].stage == WGSL_STAGE_VERTEX);
    CHECK(eps[fi].stage == WGSL_STAGE_FRAGMENT);

    int rc = 0;
    const WgslSymbolInfo *vrefs = wgsl_resolver_entrypoint_binding_vars(r, "vs_main", &rc);
    CHECK(vrefs != NULL);
    CHECK(has_binding_named(vrefs, rc, "Perspective_View", 0, 0));
    CHECK(has_binding_named(vrefs, rc, "modelMatrix", 0, 3));
    CHECK(!has_binding_named(vrefs, rc, "texSampler", 0, 2));
    CHECK(!has_binding_named(vrefs, rc, "texture0", 0, 1));
    wgsl_resolve_free((void*)vrefs);

    rc = 0;
    const WgslSymbolInfo *frefs = wgsl_resolver_entrypoint_binding_vars(r, "fs_main", &rc);
    CHECK(frefs != NULL);
    CHECK(has_binding_named(frefs, rc, "texture0", 0, 1));
    CHECK(has_binding_named(frefs, rc, "texSampler", 0, 2));
    CHECK(!has_binding_named(frefs, rc, "Perspective_View", 0, 0));
    CHECK(!has_binding_named(frefs, rc, "modelMatrix", 0, 3));
    wgsl_resolve_free((void*)frefs);

    WgslVertexSlot *slots = NULL;
    int slotn = wgsl_resolver_vertex_inputs(r, "vs_main", &slots);
    CHECK(slotn == 4);
    CHECK(count_with_location(slots, slotn, 0) == 1);
    CHECK(count_with_location(slots, slotn, 1) == 1);
    CHECK(count_with_location(slots, slotn, 2) == 1);
    CHECK(count_with_location(slots, slotn, 3) == 1);
    int c0=-1,c1=-1,c2=-1,c3=-1;
    for (int i = 0; i < slotn; i++) {
        if (slots[i].location == 0) c0 = slots[i].component_count;
        if (slots[i].location == 1) c1 = slots[i].component_count;
        if (slots[i].location == 2) c2 = slots[i].component_count;
        if (slots[i].location == 3) c3 = slots[i].component_count;
    }
    CHECK(c0 == 3 && c1 == 2 && c2 == 3 && c3 == 4);
    wgsl_resolve_free(slots);

    int gn = 0;
    const WgslSymbolInfo *globals = wgsl_resolver_globals(r, &gn);
    CHECK(globals != NULL && gn >= 7);
    CHECK(has_global_named(globals, gn, "konschtante"));
    CHECK(has_global_named(globals, gn, "lights2"));
    wgsl_resolve_free((void*)globals);

    wgsl_resolve_free((void*)eps);
    wgsl_resolver_free(r);
    wgsl_free_ast(ast);
}

static void test_compute_suite(void) {
    const char *src = R"(
@group(0) @binding(0) var tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> scale: f32;
@group(0) @binding(2) var<uniform> center: vec2<f32>;

fn mandelIter(p: vec2f, c: vec2f) -> vec2f{
    let newr: f32 = p.x * p.x - p.y * p.y;
    let newi: f32 = 2.0f * p.x * p.y;
    return vec2f(newr + c.x, newi + c.y);
}

const maxiters: i32 = 150;

fn mandelBailout(z: vec2f) -> i32{
    var zn = z;
    var i: i32;
    for(i = 0;i < maxiters;i = i + 1){
        zn = mandelIter(zn, z);
        if(sqrt(zn.x * zn.x + zn.y * zn.y) > 2.0f){
            break;
        }
    }
    return i;
}

@compute @workgroup_size(16, 16, 1)
fn compute_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let p: vec2f = (vec2f(id.xy) - 640.0f) * scale + center;
    let iters = mandelBailout(p);
    if(iters == maxiters){
        textureStore(tex, id.xy, vec4<f32>(0, 0, 0, 1.0f));
    }
}
)";
    WgslAstNode *ast = wgsl_parse(src);
    CHECK(ast != NULL);
    WgslResolver *r = wgsl_resolver_build(ast);
    CHECK(r != NULL);

    int epn = 0;
    const WgslResolverEntrypoint *eps = wgsl_resolver_entrypoints(r, &epn);
    CHECK(eps != NULL && epn == 1 && name_eq(eps[0].name, "compute_main"));
    CHECK(eps[0].stage == WGSL_STAGE_COMPUTE);

    int rc = 0;
    const WgslSymbolInfo *brefs = wgsl_resolver_entrypoint_binding_vars(r, "compute_main", &rc);
    CHECK(brefs != NULL);
    CHECK(has_binding_named(brefs, rc, "tex", 0, 0));
    CHECK(has_binding_named(brefs, rc, "scale", 0, 1));
    CHECK(has_binding_named(brefs, rc, "center", 0, 2));
    wgsl_resolve_free((void*)brefs);

    rc = 0;
    const WgslSymbolInfo *grefs = wgsl_resolver_entrypoint_globals(r, "compute_main", &rc);
    CHECK(grefs != NULL);
    CHECK(has_global_named(grefs, rc, "maxiters")); /* const global should be included here */
    wgsl_resolve_free((void*)grefs);

    wgsl_resolve_free((void*)eps);
    wgsl_resolver_free(r);
    wgsl_free_ast(ast);
}

static void test_minimal_compute_no_bindings(void) {
    const char *src = R"(
@compute @workgroup_size(8, 8, 1)
fn main_cs(@builtin(global_invocation_id) id: vec3<u32>) {
    let v = id.x + id.y + id.z;
    if (v == 123u) { }
}
)";
    WgslAstNode *ast = wgsl_parse(src);
    CHECK(ast != NULL);
    WgslResolver *r = wgsl_resolver_build(ast);
    CHECK(r != NULL);

    int epn = 0;
    const WgslResolverEntrypoint *eps = wgsl_resolver_entrypoints(r, &epn);
    CHECK(eps != NULL && epn == 1 && name_eq(eps[0].name, "main_cs"));
    CHECK(eps[0].stage == WGSL_STAGE_COMPUTE);

    int rc = 0;
    const WgslSymbolInfo *brefs = wgsl_resolver_entrypoint_binding_vars(r, "main_cs", &rc);
    CHECK((brefs == NULL && rc == 0) || (brefs != NULL && rc == 0)); /* allow NULL or empty */
    wgsl_resolve_free((void*)brefs);

    WgslVertexSlot *slots = NULL;
    int slotn = wgsl_resolver_vertex_inputs(r, "main_cs", &slots);
    CHECK(slotn == 0 && slots == NULL);

    wgsl_resolve_free((void*)eps);
    wgsl_resolver_free(r);
    wgsl_free_ast(ast);
}

static void test_transitive_calls_vertex(void) {
    const char *src = R"(
@group(1) @binding(0) var<uniform> U: mat4x4f;

fn useU(v: vec4f) -> vec4f {
    return U * v;
}

fn middle(v: vec4f) -> vec4f {
    return useU(v);
}

@vertex
fn main_vs(@location(0) p: vec3f) -> @builtin(position) vec4f {
    let v = vec4f(p, 1.0f);
    return middle(v);
}
)";
    WgslAstNode *ast = wgsl_parse(src);
    CHECK(ast != NULL);
    WgslResolver *r = wgsl_resolver_build(ast);
    CHECK(r != NULL);

    int epn = 0;
    const WgslResolverEntrypoint *eps = wgsl_resolver_entrypoints(r, &epn);
    CHECK(eps != NULL && epn == 1 && name_eq(eps[0].name, "main_vs"));
    CHECK(eps[0].stage == WGSL_STAGE_VERTEX);

    int rc = 0;
    const WgslSymbolInfo *brefs = wgsl_resolver_entrypoint_binding_vars(r, "main_vs", &rc);
    CHECK(brefs != NULL);
    CHECK(has_binding_named(brefs, rc, "U", 1, 0)); /* must be discovered transitively */
    wgsl_resolve_free((void*)brefs);

    wgsl_resolve_free((void*)eps);
    wgsl_resolver_free(r);
    wgsl_free_ast(ast);
}

int main(void) {
    test_vertex_fragment_suite();
    test_compute_suite();
    test_minimal_compute_no_bindings();
    test_transitive_calls_vertex();

    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_failed);
    return tests_failed ? 1 : 0;
}
