extern crate oklab;

use oklab::{ok_color, LinSrgb, OkHsl, OkHsv, Oklab, Srgb};

fn frand() -> f32 {
    static mut SEED: u32 = 42;
    // SAFETY: we are running in a single thread
    unsafe {
        SEED = SEED.wrapping_mul(1664525).wrapping_add(1013904223);
        (SEED as f32 / u32::MAX as f32) * 2.0 - 0.5
    }
}

trait Color {
    fn print(&self);
}

impl Color for LinSrgb {
    fn print(&self) {
        println!("{:.8} {:.8} {:.8}", self.r, self.g, self.b);
    }
}
impl Color for Srgb {
    fn print(&self) {
        println!("{:.8} {:.8} {:.8}", self.r, self.g, self.b);
    }
}
impl Color for Oklab {
    fn print(&self) {
        println!("{:.8} {:.8} {:.8}", self.l, self.a, self.b);
    }
}
impl Color for OkHsl {
    fn print(&self) {
        println!("{:.8} {:.8} {:.8}", self.h, self.s, self.l);
    }
}
impl Color for OkHsv {
    fn print(&self) {
        println!("{:.8} {:.8} {:.8}", self.h, self.s, self.v);
    }
}

fn print_color(c: impl Color) {
    c.print();
}

#[rustfmt::skip]
fn main() -> std::process::ExitCode {
    for _ in 0..1000 {
        print_color(ok_color::linear_srgb_to_oklab(LinSrgb { r: frand(), g: frand(), b: frand() }));
        print_color(ok_color::oklab_to_linear_srgb(Oklab { l: frand(), a: frand(), b: frand() }));
        print_color(ok_color::gamut_clip_preserve_chroma(LinSrgb { r: frand(), g: frand(), b: frand() }));
        print_color(ok_color::gamut_clip_project_to_0_5(LinSrgb { r: frand(), g: frand(), b: frand() }));
        print_color(ok_color::gamut_clip_project_to_l_cusp(LinSrgb { r: frand(), g: frand(), b: frand() }));
        print_color(ok_color::gamut_clip_adaptive_l0_0_5(LinSrgb { r: frand(), g: frand(), b: frand() }, 0.05));
        print_color(ok_color::gamut_clip_adaptive_l0_l_cusp(LinSrgb { r: frand(), g: frand(), b: frand() }, 0.05));
        print_color(ok_color::okhsl_to_srgb(OkHsl { h: frand(), s: frand(), l: frand() }));
        print_color(ok_color::srgb_to_okhsl(Srgb { r: frand(), g: frand(), b: frand() }));
        print_color(ok_color::okhsv_to_srgb(OkHsv { h: frand(), s: frand(), v: frand() }));
        print_color(ok_color::srgb_to_okhsv(Srgb { r: frand(), g: frand(), b: frand() }));
    }
    std::process::ExitCode::from(0)
}
