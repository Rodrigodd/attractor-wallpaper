@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read_write> aggregate_buffer : array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> particles : array<vec2<f32>>;
@group(0) @binding(3) var<uniform> attractor : Attractor;

struct Uniforms {
  screenWidth: u32,
  screenHeight: u32,
  color_scale: f32,
};

struct Attractor {
    // need to use matrix, because arrays need to have a stride of 16 bytes.
    coeffs: mat4x4<f32>,
}

fn attractor_step(p: vec2<f32>) -> vec2<f32> {
    let a = array<f32, 6>(
        attractor.coeffs[0][0],
        attractor.coeffs[0][1],
        attractor.coeffs[0][2],
        attractor.coeffs[0][3],

        attractor.coeffs[1][0],
        attractor.coeffs[1][1],
    );
    let b = array<f32, 6>(
        attractor.coeffs[1][2],
        attractor.coeffs[1][3],

        attractor.coeffs[2][0],
        attractor.coeffs[2][1],
        attractor.coeffs[2][2],
        attractor.coeffs[2][3],
    );
    let x = a[0] + a[1] * p.x + a[2] * p.x * p.x + a[3] * p.x * p.y + a[4] * p.y + a[5] * p.y * p.y;
    let y = b[0] + b[1] * p.x + b[2] * p.x * p.x + b[3] * p.x * p.y + b[4] * p.y + b[5] * p.y * p.y;

    return vec2<f32>(x, y);
}

@compute @workgroup_size(256, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let index = global_id.x;

    var particle = particles[index];
    for (var k = 0; k < 1; k++){
        particle = attractor_step(particle);

        let x = u32(particle.x);
        let y = u32(particle.y);
        let i = y * uniforms.screenWidth + x;

        if (x < uniforms.screenWidth && y < uniforms.screenHeight) {
            let n = atomicAdd(&aggregate_buffer[i], 1);
            atomicMax(&aggregate_buffer[0], n + 1);
        }
    }
    particles[index] = particle;
}
