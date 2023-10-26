
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

    let iy = u32(in.tex_coords.y * f32(uniforms.screenHeight));
    let ix = u32(in.tex_coords.x * f32(uniforms.screenWidth));
    let i0 = 
        iy * uniforms.screenWidth * multisampling * (multisampling)
        + ix * multisampling;

    // let v = f32(aggregate_buffer[i0]) / f32(aggregate_buffer[0]);
    // return color(v);

    var c = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    //CONVOLUTION
    
    c *= 1.0/c.a;

    var d = dither(linear_to_srgb(c.rgb), vec2<f32>(f32(ix), f32(iy)));
    d = srgb_to_linear(d);

    return vec4(d,1.0);
}

fn color(c: f32, x: f32, y: f32) -> vec4<f32> {
    let background =  gradient(x, y);
    let foreground =  colormap(c);

    let alpha = clamp(c * 10.0, 0.0, 1.0);
    return mix(background, foreground, alpha);
}

// Based on bevy's dithering implementation: https://github.com/bevyengine/bevy/pull/5264
fn dither(color: vec3<f32>, pos: vec2<f32>) -> vec3<f32> {
    return color + screen_space_dither(pos.xy);
}

// Source: Advanced VR Rendering, GDC 2015, Alex Vlachos, Valve, Slide 49
// https://media.steampowered.com/apps/valve/2015/Alex_Vlachos_Advanced_VR_Rendering_GDC2015.pdf
fn screen_space_dither(frag_coord: vec2<f32>) -> vec3<f32> {
    var dither = vec3<f32>(dot(vec2<f32>(171.0, 231.0), frag_coord)).xxx;
    dither = fract(dither.rgb / vec3<f32>(103.0, 71.0, 97.0));
    return (dither - 0.5) / 255.0;
}

fn gradient(x: f32, y: f32) -> vec4<f32> { 
    let start = vec4<f32>(0.012, 0.000, 0.0, 1.0);
    let end = vec4<f32>(0.004, 0.000, 0.0, 1.0);

    let center = vec2<f32>(0.9, 0.3);
    let radius = 1.2;

    let dist = distance(vec2<f32>(x, y), center);

    let z = dist / radius;

    return mix(start, end, pow(z,0.33));
}

fn colormap(x: f32) -> vec4<f32> {
    var r: f32 = clamp(8.0 / 3.0 * x, 0.0, 1.0);
    var g: f32 = clamp(8.0 / 3.0 * x - 1.0, 0.0, 1.0);
    var b: f32 = clamp(4.0 * x - 3.0, 0.0, 1.0);

    return vec4<f32>(r, g, b, 1.0);
}

// Linear to sRGB conversion function
fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
    var srgbColor: vec3<f32>;
    
    for (var i: u32 = 0u; i < 3u; i = i + 1u) {
        if (color[i] <= 0.0031308) {
            srgbColor[i] = 12.92 * color[i];
        } else {
            srgbColor[i] = 1.055 * pow(color[i], 1.0 / 2.4) - 0.055;
        }
    }
    
    return srgbColor;
}

// sRGB to Linear conversion function
fn srgb_to_linear(color: vec3<f32>) -> vec3<f32> {
    var linearColor: vec3<f32>;
    
    for (var i: u32 = 0u; i < 3u; i = i + 1u) {
        if (color[i] <= 0.04045) {
            linearColor[i] = color[i] / 12.92;
        } else {
            linearColor[i] = pow((color[i] + 0.055) / 1.055, 2.4);
        }
    }
    
    return linearColor;
}
