#include "wgsl_parser.h"

const char vertexFragmentShaderSource[] = R"(
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
    out.position = Perspective_View * 
                   modelMatrix[instanceIdx] *
    vec4f(in.position.xyz, 1.0f);
    out.color = in.color;
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    return textureSample(texture0, texSampler, in.uv).rgba * in.color;
}
)";

const char computeShaderCode[] = R"(
@group(0) @binding(0) var tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> scale: f32;
@group(0) @binding(2) var<uniform> center: vec2<f32>;
fn mandelIter(p: vec2f, c: vec2f) -> vec2f{
    let newr: f32 = p.x * p.x - p.y * p.y;
    let newi: f32 = 2.0f * p.x * p.y;

    return vec2f(newr + c.x, newi + c.y);
}

const maxiters: i32 = 150;

struct termination {
    iterations: i32,
    overshoot: f32,
}

fn mandelBailout(z: vec2f) -> termination{
    var zn = z;
    var i: i32;
    for(i = 0;i < maxiters;i = i + 1){
        zn = mandelIter(zn, z);
        if(sqrt(zn.x * zn.x + zn.y * zn.y) > 2.0f){
            break;
        }
    }
    var ret: termination;
    ret.iterations = i;
    ret.overshoot = ((sqrt(zn.x * zn.x + zn.y * zn.y)) - 2.0f) / 2.0f;
    return ret;
}
@compute
@workgroup_size(16, 16, 1)
fn compute_main(@builtin(global_invocation_id) id: vec3<u32>) {
    //let ld: vec4<u32> = textureLoad(tex, id.xy);
    let mandelSample: vec2f = (vec2f(id.xy) - 640.0f) * scale + center;
    let iters = mandelBailout(mandelSample);
    if(iters.iterations == maxiters){
        textureStore(tex, id.xy, vec4<f32>(0, 0, 0, 1.0f));
    }
    else{
        let inorm: f32 = f32(iters.iterations) - iters.overshoot;
        let colorSpace = log(inorm + 1.0f);//3.0f * log(inorm + 1) / log(f32(maxiters));
        let colorr: f32 = 0.5f * sin(10.0f * colorSpace) + 0.5f;
        let colorg: f32 = 0.5f * sin(4.0f * colorSpace) + 0.5f;
        let colorb: f32 = 0.05f * sin(7.0f * colorSpace) + 0.1f;
        textureStore(tex, id.xy, vec4<f32>(colorr, colorg, colorb, 1.0f));
    }
}
)";

const char csSimple[] = R"(
@compute
@workgroup_size(16, 16, 1)
fn compute_main(@builtin(global_invocation_id) id: vec3<u32>) {
    var i: i32 = -id.x;
    //for(i = 0;i < -ix.x;i++){}
}
)";

int main(){
    WgslAstNode* vertexFragmentRootNode = wgsl_parse(vertexFragmentShaderSource);
    //WgslAstNode* computeRootNode = wgsl_parse(csSimple);
    wgsl_debug_print(vertexFragmentRootNode, 0);
    //wgsl_debug_print(computeRootNode, 0);
}