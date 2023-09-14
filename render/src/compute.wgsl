@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read_write> aggregate_buffer : array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> particles : array<vec2<f32>>;

const a = array<f32, 6>(408.77520299183766, 1.083786938178373, -0.0020245073471088213, -0.0019693494247534278, 0.6419168560638423, -0.0005447818575888833);
const b = array<f32, 6>(226.38406851378386, -0.8232323878194252, 0.004218763946825989, -0.006422495770489061, 1.2471152957813798, 0.0004873801656204262);
const start = vec2<f32>(576.6726382593068, 292.59368391867366);

struct Uniforms {
  screenWidth: u32,
  screenHeight: u32,
};

fn attractor_step(p: vec2<f32>) -> vec2<f32> {
    let x = a[0] + a[1] * p.x + a[2] * p.x * p.x + a[3] * p.x * p.y + a[4] * p.y + a[5] * p.y * p.y;
    let y = b[0] + b[1] * p.x + b[2] * p.x * p.x + b[3] * p.x * p.y + b[4] * p.y + b[5] * p.y * p.y;

    return vec2<f32>(x, y);
}

@compute @workgroup_size(256, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let index = global_id.x;

    var particle = particles[index];
    for (var k = 0; k < 100; k++){
        particle = attractor_step(particle);

        let x = u32(particle.x);
        let y = u32(particle.y);
        let i = y * uniforms.screenWidth + x;

        if (x < uniforms.screenWidth && y < uniforms.screenHeight) {
            atomicAdd(&aggregate_buffer[i], 1u);
        }
    }
    particles[index] = particle;
}
