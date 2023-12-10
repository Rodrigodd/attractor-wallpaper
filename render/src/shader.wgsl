
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

struct ColorPoint {
    time: f32,
    l_coef: vec4<f32>,
    a_coef: vec4<f32>,
    b_coef: vec4<f32>,
}

struct Uniforms {
  screenWidth: u32,
  screenHeight: u32,
  /// background gradient color 1
  bg_color_1: vec4<f32>,
  /// background gradient color 2
  bg_color_2: vec4<f32>,
  /// background gradient point 1, in clip space
  bg_point_1: vec2<f32>,
  /// background gradient point 2, in clip space
  bg_point_2: vec2<f32>,
  /// color map
  colormap: array<ColorPoint, 4>,
};

// MULTISAMPLING here is replaced by a custom preprocessor. This in the future
// should be replaced by WGSL override constructs, which is not implemented in
// naga. The code may also be relying in constant propagation, which is also
// not implemented yet.
// const MULTISAMPLING: u32 = 0u;
// const LANCZOS_WIDTH: u32 = 0u;
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

    let p = vec2(in.tex_coords.x, in.tex_coords.y);

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

fn color(c: f32, p: vec2<f32>) -> vec4<f32> {
    let background = gradient(p);
    // var background: vec4<f32>;
    // if p.y < 0.5 {
    //     background = sample_colormap(p.x);
    // } else {
    //     background = colormap(p.x);
    // }

    let foreground = sample_colormap(c);

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

fn gradient(p: vec2<f32>) -> vec4<f32> { 
    let c1 = uniforms.bg_color_1;
    let c2 = uniforms.bg_color_2;

    let p1 = uniforms.bg_point_1;
    let p2 = uniforms.bg_point_2;
    let radius = distance(p1, p2);

    let dist = distance(p, p1);

    let t = dot(p - p1, p2 - p1) / (radius * radius);

    return okmix(c1, c2, t);
}

fn colormap(x: f32) -> vec4<f32> {
    var r: f32 = clamp(8.0 / 3.0 * x, 0.0, 1.0);
    var g: f32 = clamp(8.0 / 3.0 * x - 1.0, 0.0, 1.0);
    var b: f32 = clamp(4.0 * x - 3.0, 0.0, 1.0);

    return vec4<f32>(r, g, b, 1.0);
}

fn sample_colormap(t: f32) -> vec4<f32> {
    // let length: u32 = arrayLength(uniforms.colormap.time);
    let length: i32 = 4;
    var right_index = length;
    for (var i: i32 = 0; i < length; i = i + 1) {
        if (t < uniforms.colormap[i].time) {
            right_index = i;
            break;
        }
    }
    let left_index = right_index - 1;

    let t0 = uniforms.colormap[left_index].time;

    var t1: f32;
    if (right_index == length) {
        t1 = 1.0;
    } else {
        t1 = uniforms.colormap[right_index].time;
    }

    var x: f32 = (t - t0) / (t1 - t0);
    x = clamp(x, 0.0, 1.0);

    var p = vec3(
        (((uniforms.colormap[left_index].l_coef[0] * x) + uniforms.colormap[left_index].l_coef[1]) * x + uniforms.colormap[left_index].l_coef[2]) * x + uniforms.colormap[left_index].l_coef[3],
        (((uniforms.colormap[left_index].a_coef[0] * x) + uniforms.colormap[left_index].a_coef[1]) * x + uniforms.colormap[left_index].a_coef[2]) * x + uniforms.colormap[left_index].a_coef[3],
        (((uniforms.colormap[left_index].b_coef[0] * x) + uniforms.colormap[left_index].b_coef[1]) * x + uniforms.colormap[left_index].b_coef[2]) * x + uniforms.colormap[left_index].b_coef[3],
    );

    return vec4(oklab_to_linear(p), 1.0);
}

fn remap(x: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    return c + (x - a) * (d - c) / (b - a);
}

fn relative_eq(a: f32, b: f32) -> bool {
    return abs(a - b) <= 1.0e-6 * max(abs(a), abs(b));
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

fn okmix(c1: vec4<f32>, c2: vec4<f32>, t: f32) -> vec4<f32> {
    let ok1 = linear_to_oklab(c1.rgb);
    let ok2 = linear_to_oklab(c2.rgb);
    let okt = mix(ok1, ok2, t);

    return vec4(oklab_to_linear(okt), 1.0);
}

fn linear_to_oklab(c: vec3<f32>) -> vec3<f32> {
    let l = 0.41222146 * c.r + 0.53633255 * c.g + 0.051445995 * c.b;
    let m = 0.2119035 * c.r + 0.6806995 * c.g + 0.10739696 * c.b;
    let s = 0.08830246 * c.r + 0.28171885 * c.g + 0.6299787 * c.b;

    let l_ = pow(l, 1.0/3.0);
    let m_ = pow(m, 1.0/3.0);
    let s_ = pow(s, 1.0/3.0);

    return vec3<f32>(
        0.21045426 * l_ + 0.7936178 * m_ - 0.004072047 * s_,
        1.9779985 * l_ - 2.4285922 * m_ + 0.4505937 * s_,
        0.025904037 * l_ + 0.78277177 * m_ - 0.80867577 * s_
    );
}

fn oklab_to_linear(c: vec3<f32>) -> vec3<f32> {
    let l_ = c[0] + 0.39633778  * c[1] + 0.21580376 * c[2];
    let m_ = c[0] - 0.105561346 * c[1] - 0.06385417 * c[2];
    let s_ = c[0] - 0.08948418  * c[1] - 1.2914855  * c[2];

    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    return vec3<f32>(
        4.0767417 * l - 3.3077116 * m + 0.23096994 * s,
        -1.268438 * l + 2.6097574 * m - 0.34131938 * s,
        -0.0041960863 * l - 0.7034186 * m + 1.7076147 * s,
    );
}
