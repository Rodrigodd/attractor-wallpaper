struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(1) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;

    // i ->  x  y
    // 0 -> -1  1
    // 1 -> -1 -3
    // 2 ->  3  1
    let x = f32(i32(in_vertex_index) / 2); // 0, 0, 1
    let y = f32(i32(in_vertex_index & 1u)); // 0, 1, 0
    out.clip_position = vec4<f32>(x * 4.0 - 1.0, y * -4.0 + 1.0, 0.0, 1.0);
    out.tex_coords = vec2<f32>(x*2.0, y*2.0);
    return out;
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> aggregate_buffer : array<u32>;

struct Uniforms {
  screenWidth: u32,
  screenHeight: u32,
  color_scale: f32,
};

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let i = u32(in.tex_coords.y * f32(uniforms.screenHeight)) * uniforms.screenWidth + u32(in.tex_coords.x * f32(uniforms.screenWidth)) ;
    let v = f32(aggregate_buffer[i]) / (f32(aggregate_buffer[0]) * 0.5);
    return colormap(v);
}

fn colormap(x: f32) -> vec4<f32> {
    var r: f32 = clamp(8.0 / 3.0 * x, 0.0, 1.0);
    var g: f32 = clamp(8.0 / 3.0 * x - 1.0, 0.0, 1.0);
    var b: f32 = clamp(4.0 * x - 3.0, 0.0, 1.0);
    return vec4<f32>(r, g, b, 1.0);
}
