mod executor;
mod renderer;

pub use crate::{
    executor::{TaskId, WinitExecutor},
    renderer::{AttractorRenderer, SurfaceState, WgpuState},
};

pub fn get_intensity(
    base_intensity: f32,
    tranform: [f64; 4],
    total_samples: u64,
    antialiasing: attractors::AntiAliasing,
) -> i32 {
    let p = match antialiasing {
        attractors::AntiAliasing::None => 1,
        attractors::AntiAliasing::Bilinear => 64,
        attractors::AntiAliasing::Lanczos => 64,
    };

    let det = tranform[0] * tranform[3] - tranform[1] * tranform[2];

    (base_intensity as f64 * total_samples as f64 * p as f64 * det * 1000.0 / 4.0).round() as i32
}
