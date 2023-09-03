struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(1) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;

    // let x = f32(1 - i32(in_vertex_index)) * 0.5;
    // let y = f32(i32(in_vertex_index & 1u) * 2 - 1) * 0.5;
    // out.clip_position = vec4<f32>(x, y, 0.0, 1.0);

    // i ->  x  y
    // 0 -> -1  1
    // 1 -> -1 -3
    // 2 ->  3  1
    let x = f32(i32(in_vertex_index) / 2);
    let y = f32(i32(in_vertex_index & 1u));
    out.clip_position = vec4<f32>(x * 4.0 - 1.0, y * -4.0 + 1.0, 0.0, 1.0);
    out.tex_coords = vec2<f32>(x*2.0, y*2.0);
    return out;
}

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, in.tex_coords);
}
