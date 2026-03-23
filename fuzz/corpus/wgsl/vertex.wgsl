struct VertexOut { @builtin(position) pos: vec4f, @location(0) color: vec4f };
@vertex fn main(@builtin(vertex_index) vi: u32) -> VertexOut {
    var out: VertexOut;
    out.pos = vec4f(0.0, 0.0, 0.0, 1.0);
    out.color = vec4f(1.0);
    return out;
}
