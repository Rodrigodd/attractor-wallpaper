
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
@group(0) @binding(1) var<storage, read> aggregate_buffer : array<i32>;

struct Uniforms {
  screenWidth: u32,
  screenHeight: u32,
  color_scale: f32,
};

// MULTISAMPLING here is replaced by a custom preprocessor. This in the future
// should be replaced by WGSL override constructs, which is not implemented in
// naga. The code may also be relying in constant propagation, which is also
// not implemented yet.
const multisampling: u32 = MULTISAMPLING;
const lanczos_width: u32 = LANCZOS_WIDTH;
const pi: f32 = 3.1415926535897932384626433832795;

fn lanczos_kernel(x: f32) -> f32 {
    let a = f32(lanczos_width) * 0.5;
    if x == 0.0 {
        return 1.0;
    } else {
        let pi_x = x * pi;
        return a * sin(pi_x) * sin(pi_x / a) / (pi_x * pi_x);
    }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {

     let i0 = 
           u32(in.tex_coords.y * f32(uniforms.screenHeight)) * uniforms.screenWidth * multisampling * (multisampling)
         + u32(in.tex_coords.x * f32(uniforms.screenWidth)) * multisampling;

    // let v = f32(aggregate_buffer[i0]) / f32(aggregate_buffer[0]);
    // return colormap(v);

    var c = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    //CONVOLUTION
    
    c *= 1.0/c.a;

    return c;
}

fn colormap(x: f32) -> vec4<f32> {
    var r: f32 = clamp(8.0 / 3.0 * x, 0.0, 1.0);
    var g: f32 = clamp(8.0 / 3.0 * x - 1.0, 0.0, 1.0);
    var b: f32 = clamp(4.0 * x - 3.0, 0.0, 1.0);
    return vec4<f32>(r, g, b, 1.0);
}
