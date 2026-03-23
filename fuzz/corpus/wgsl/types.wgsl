struct S { x: f32, y: vec3f, z: mat4x4f };
@group(0) @binding(0) var<uniform> u: S;
@compute @workgroup_size(1)
fn main() { let v = u.x + u.y.x; }
