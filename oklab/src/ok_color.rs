#![allow(missing_docs)]

//! Direct port of the C++ header provided by Björn Ottosson at
//! [http://bottosson.github.io/misc/ok_color.h](http://bottosson.github.io/misc/ok_color.h).

// Copyright(c) 2021 Björn Ottosson
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this softwareand associated documentation files(the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and /or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions :
// The above copyright noticeand this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
use std::f32::consts::PI;

use super::*;

#[derive(Clone, Copy)]
struct Lc {
    l: f32,
    c: f32,
}

#[derive(Clone, Copy)]
struct Cs {
    c_0: f32,
    c_mid: f32,
    c_max: f32,
}

// Alternative representation of (L_cusp, C_cusp)
// Encoded so S = C_cusp/L_cusp and T = C_cusp/(1-L_cusp)
// The maximum value for C in the triangle is then found fn fmin(S*L, T*(1-L) -> as), for a given L
struct ST {
    s: f32,
    t: f32,
}

fn fmin(a: f32, b: f32) -> f32 {
    if a < b {
        a
    } else {
        b
    }
}
fn fmax(a: f32, b: f32) -> f32 {
    if a > b {
        a
    } else {
        b
    }
}

fn clamp(x: f32, min: f32, max: f32) -> f32 {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

fn sgn(x: f32) -> f32 {
    (0. < x) as u8 as f32 - (x < 0.) as u8 as f32
}

fn srgb_transfer_function(a: f32) -> f32 {
    if 0.0031308 >= a {
        12.92 * a
    } else {
        1.055 * a.powf(0.41666666) - 0.055
    }
}

fn srgb_transfer_function_inv(a: f32) -> f32 {
    if 0.04045 < a {
        ((a + 0.055) / 1.055).powf(2.4)
    } else {
        a / 12.92
    }
}

pub fn linear_srgb_to_srgb(c: LinSrgb) -> Srgb {
    Srgb {
        r: srgb_transfer_function(c.r),
        g: srgb_transfer_function(c.g),
        b: srgb_transfer_function(c.b),
    }
}
pub fn srgb_to_linear_srgb(c: Srgb) -> LinSrgb {
    LinSrgb {
        r: srgb_transfer_function_inv(c.r),
        g: srgb_transfer_function_inv(c.g),
        b: srgb_transfer_function_inv(c.b),
    }
}

pub fn linear_srgb_to_oklab(c: LinSrgb) -> Oklab {
    let l = 0.41222146 * c.r + 0.53633255 * c.g + 0.051445995 * c.b;
    let m = 0.2119035 * c.r + 0.6806995 * c.g + 0.10739696 * c.b;
    let s = 0.08830246 * c.r + 0.28171885 * c.g + 0.6299787 * c.b;

    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();

    Oklab {
        l: 0.21045426 * l_ + 0.7936178 * m_ - 0.004072047 * s_,
        a: 1.9779985 * l_ - 2.4285922 * m_ + 0.4505937 * s_,
        b: 0.025904037 * l_ + 0.78277177 * m_ - 0.80867577 * s_,
    }
}

pub fn oklab_to_linear_srgb(c: Oklab) -> LinSrgb {
    let l_ = c.l + 0.39633778 * c.a + 0.21580376 * c.b;
    let m_ = c.l - 0.105561346 * c.a - 0.06385417 * c.b;
    let s_ = c.l - 0.08948418 * c.a - 1.2914855 * c.b;

    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    LinSrgb {
        r: 4.0767417 * l - 3.3077116 * m + 0.23096994 * s,
        g: -1.268438 * l + 2.6097574 * m - 0.34131938 * s,
        b: -0.0041960863 * l - 0.7034186 * m + 1.7076147 * s,
    }
}

// Finds the maximum saturation possible for a given hue that fits in sRGB
// Saturation here is defined as S = C/L
// a and b must be normalized so a^2 + b^2 == 1
fn compute_max_saturation(a: f32, b: f32) -> f32 {
    // Max saturation will be when one of r, g or b goes below zero.

    // Select different coefficients depending on which component goes below zero first
    let k0;
    let k1;
    let k2;
    let k3;
    let k4;
    let wl;
    let wm;
    let ws;

    if -1.8817033 * a - 0.8093649 * b > 1.0 {
        // Red component
        k0 = 1.1908628;
        k1 = 1.7657673;
        k2 = 0.5966264;
        k3 = 0.755152;
        k4 = 0.5677124;
        wl = 4.0767417;
        wm = -3.3077116;
        ws = 0.23096994;
    } else if 1.8144411 * a - 1.1944528 * b > 1.0 {
        // Green component
        k0 = 0.73956515;
        k1 = -0.45954404;
        k2 = 0.08285427;
        k3 = 0.1254107;
        k4 = 0.14503204;
        wl = -1.268438;
        wm = 2.6097574;
        ws = -0.34131938;
    } else {
        // Blue component
        k0 = 1.3573365;
        k1 = -0.00915799;
        k2 = -1.1513021;
        k3 = -0.50559606;
        k4 = 0.00692167;
        wl = -0.0041960863;
        wm = -0.7034186;
        ws = 1.7076147;
    }

    // Approximate max saturation using a polynomial:
    let mut saturation: f32 = k0 + k1 * a + k2 * b + k3 * a * a + k4 * a * b;

    // Do one step Halley's method to get closer
    // this gives an error less than 10e6, except for some blue hues where the dS/dh is close to infinite
    // this should be sufficient for most applications, otherwise do two/three steps

    let k_l = 0.39633778 * a + 0.21580376 * b;
    let k_m = -0.105561346 * a - 0.06385417 * b;
    let k_s = -0.08948418 * a - 1.2914855 * b;

    {
        let l_ = 1. + saturation * k_l;
        let m_ = 1. + saturation * k_m;
        let s_ = 1. + saturation * k_s;

        let l = l_ * l_ * l_;
        let m = m_ * m_ * m_;
        let s = s_ * s_ * s_;

        let l_ds = 3. * k_l * l_ * l_;
        let m_ds = 3. * k_m * m_ * m_;
        let s_ds = 3. * k_s * s_ * s_;

        let l_ds2 = 6. * k_l * k_l * l_;
        let m_ds2 = 6. * k_m * k_m * m_;
        let s_ds2 = 6. * k_s * k_s * s_;

        let f = wl * l + wm * m + ws * s;
        let f1 = wl * l_ds + wm * m_ds + ws * s_ds;
        let f2 = wl * l_ds2 + wm * m_ds2 + ws * s_ds2;

        saturation -= f * f1 / (f1 * f1 - 0.5 * f * f2);
    }

    saturation
}

// finds L_cusp and C_cusp for a given hue
// a and b must be normalized so a^2 + b^2 == 1
fn find_cusp(a: f32, b: f32) -> Lc {
    // First, find the maximum saturation (saturation S = C/L)
    let s_cusp = compute_max_saturation(a, b);

    // Convert to linear to: sRGB find the first point where at least one of r,g or b >= 1:
    let rgb_at_max = oklab_to_linear_srgb(Oklab {
        l: 1.0,
        a: s_cusp * a,
        b: s_cusp * b,
    });
    let l_cusp = (1. / fmax(fmax(rgb_at_max.r, rgb_at_max.g), rgb_at_max.b)).cbrt();
    let c_cusp = l_cusp * s_cusp;

    Lc {
        l: l_cusp,
        c: c_cusp,
    }
}

// Finds intersection of the line defined by
// L = L0 * (1 - t) + t * L1;
// C = t * C1;
// a and b must be normalized so a^2 + b^2 == 1
fn find_gamut_intersection_(a: f32, b: f32, l1: f32, c1: f32, l0: f32, cusp: Lc) -> f32 {
    // Find the intersection for upper and lower half seprately
    let mut t: f32;
    if ((l1 - l0) * cusp.c - (cusp.l - l0) * c1) <= 0. {
        // Lower half

        t = cusp.c * l0 / (c1 * cusp.l + cusp.c * (l0 - l1));
    } else {
        // Upper half

        // First intersect with triangle
        t = cusp.c * (l0 - 1.) / (c1 * (cusp.l - 1.) + cusp.c * (l0 - l1));

        // Then one step Halley's method
        {
            let d_ligth = l1 - l0;
            let d_chrom = c1;

            let k_l = 0.39633778 * a + 0.21580376 * b;
            let k_m = -0.105561346 * a - 0.06385417 * b;
            let k_s = -0.08948418 * a - 1.2914855 * b;

            let l_dt = d_ligth + d_chrom * k_l;
            let m_dt = d_ligth + d_chrom * k_m;
            let s_dt = d_ligth + d_chrom * k_s;

            // If higher accuracy is required, 2.0 or 3.0 iterations of the following block can be used:
            {
                let ligth = l0 * (1. - t) + t * l1;
                let chorm = t * c1;

                let l_ = ligth + chorm * k_l;
                let m_ = ligth + chorm * k_m;
                let s_ = ligth + chorm * k_s;

                let l = l_ * l_ * l_;
                let m = m_ * m_ * m_;
                let s = s_ * s_ * s_;

                let ldt = 3.0 * l_dt * l_ * l_;
                let mdt = 3.0 * m_dt * m_ * m_;
                let sdt = 3.0 * s_dt * s_ * s_;

                let ldt2 = 6.0 * l_dt * l_dt * l_;
                let mdt2 = 6.0 * m_dt * m_dt * m_;
                let sdt2 = 6.0 * s_dt * s_dt * s_;

                let r = 4.0767417 * l - 3.3077116 * m + 0.23096994 * s - 1.0;
                let r1 = 4.0767417 * ldt - 3.3077116 * mdt + 0.23096994 * sdt;
                let r2 = 4.0767417 * ldt2 - 3.3077116 * mdt2 + 0.23096994 * sdt2;

                let u_r = r1 / (r1 * r1 - 0.5 * r * r2);
                let t_r = -r * u_r;

                let g = -1.268438 * l + 2.6097574 * m - 0.34131938 * s - 1.0;
                let g1 = -1.268438 * ldt + 2.6097574 * mdt - 0.34131938 * sdt;
                let g2 = -1.268438 * ldt2 + 2.6097574 * mdt2 - 0.34131938 * sdt2;

                let u_g = g1 / (g1 * g1 - 0.5 * g * g2);
                let t_g = -g * u_g;

                let b = -0.0041960863 * l - 0.7034186 * m + 1.7076147 * s - 1.0;
                let b1 = -0.0041960863 * ldt - 0.7034186 * mdt + 1.7076147 * sdt;
                let b2 = -0.0041960863 * ldt2 - 0.7034186 * mdt2 + 1.7076147 * sdt2;

                let u_b = b1 / (b1 * b1 - 0.5 * b * b2);
                let t_b = -b * u_b;

                let t_r = if u_r >= 0. { t_r } else { f32::MAX };
                let t_g = if u_g >= 0. { t_g } else { f32::MAX };
                let t_b = if u_b >= 0. { t_b } else { f32::MAX };

                t += fmin(t_r, fmin(t_g, t_b));
            }
        }
    }

    t
}

fn find_gamut_intersection(a: f32, b: f32, l1: f32, c1: f32, l0: f32) -> f32 {
    // Find the cusp of the gamut triangle
    let cusp = find_cusp(a, b);

    find_gamut_intersection_(a, b, l1, c1, l0, cusp)
}

pub fn gamut_clip_preserve_chroma(rgb: LinSrgb) -> LinSrgb {
    if rgb.r < 1.0 && rgb.g < 1.0 && rgb.b < 1.0 && rgb.r >= 0.0 && rgb.g >= 0.0 && rgb.b >= 0.0 {
        return rgb;
    }

    let lab = linear_srgb_to_oklab(rgb);

    let l = lab.l;
    let eps = 0.00001;
    let c = fmax(eps, (lab.a * lab.a + lab.b * lab.b).sqrt());
    let a_ = lab.a / c;
    let b_ = lab.b / c;

    let l0 = clamp(l, 0.0, 1.0);

    let t = find_gamut_intersection(a_, b_, l, c, l0);
    let l_clipped = l0 * (1.0 - t) + t * l;
    let c_clipped = t * c;

    oklab_to_linear_srgb(Oklab {
        l: l_clipped,
        a: c_clipped * a_,
        b: c_clipped * b_,
    })
}

pub fn gamut_clip_project_to_0_5(rgb: LinSrgb) -> LinSrgb {
    if rgb.r < 1.0 && rgb.g < 1.0 && rgb.b < 1.0 && rgb.r > 0.0 && rgb.g > 0.0 && rgb.b > 0.0 {
        return rgb;
    }

    let lab = linear_srgb_to_oklab(rgb);

    let l = lab.l;
    let eps = 0.00001;
    let c = fmax(eps, (lab.a * lab.a + lab.b * lab.b).sqrt());
    let a_ = lab.a / c;
    let b_ = lab.b / c;

    let l0 = 0.5;

    let t = find_gamut_intersection(a_, b_, l, c, l0);
    let l_clipped = l0 * (1.0 - t) + t * l;
    let c_clipped = t * c;

    oklab_to_linear_srgb(Oklab {
        l: l_clipped,
        a: c_clipped * a_,
        b: c_clipped * b_,
    })
}

pub fn gamut_clip_project_to_l_cusp(rgb: LinSrgb) -> LinSrgb {
    if rgb.r < 1.0 && rgb.g < 1.0 && rgb.b < 1.0 && rgb.r > 0.0 && rgb.g > 0.0 && rgb.b > 0.0 {
        return rgb;
    }

    let lab = linear_srgb_to_oklab(rgb);

    let l = lab.l;
    let eps = 0.00001;
    let c = fmax(eps, (lab.a * lab.a + lab.b * lab.b).sqrt());
    let a_ = lab.a / c;
    let b_ = lab.b / c;

    // The cusp is computed here and in find_gamut_intersection, an optimized solution would only compute it once.
    let cusp = find_cusp(a_, b_);

    let l0 = cusp.l;

    let t = find_gamut_intersection(a_, b_, l, c, l0);

    let l_clipped = l0 * (1.0 - t) + t * l;
    let c_clipped = t * c;

    oklab_to_linear_srgb(Oklab {
        l: l_clipped,
        a: c_clipped * a_,
        b: c_clipped * b_,
    })
}

pub fn gamut_clip_adaptive_l0_0_5(rgb: LinSrgb, alpha: f32) -> LinSrgb {
    if rgb.r < 1.0 && rgb.g < 1.0 && rgb.b < 1.0 && rgb.r > 0.0 && rgb.g > 0.0 && rgb.b > 0.0 {
        return rgb;
    }

    let lab = linear_srgb_to_oklab(rgb);

    let l = lab.l;
    let eps = 0.00001;
    let c = fmax(eps, (lab.a * lab.a + lab.b * lab.b).sqrt());
    let a_ = lab.a / c;
    let b_ = lab.b / c;

    let ld = l - 0.5;
    let e1 = 0.5 + ld.abs() + alpha * c;
    let l0 = 0.5 * (1. + sgn(ld) * (e1 - (e1 * e1 - 2. * ld.abs()).sqrt()));

    let t = find_gamut_intersection(a_, b_, l, c, l0);
    let l_clipped = l0 * (1. - t) + t * l;
    let c_clipped = t * c;

    oklab_to_linear_srgb(Oklab {
        l: l_clipped,
        a: c_clipped * a_,
        b: c_clipped * b_,
    })
}

pub fn gamut_clip_adaptive_l0_l_cusp(rgb: LinSrgb, alpha: f32) -> LinSrgb {
    if rgb.r < 1.0 && rgb.g < 1.0 && rgb.b < 1.0 && rgb.r > 0.0 && rgb.g > 0.0 && rgb.b > 0.0 {
        return rgb;
    }

    let lab = linear_srgb_to_oklab(rgb);

    let l = lab.l;
    let eps = 0.00001;
    let c = fmax(eps, (lab.a * lab.a + lab.b * lab.b).sqrt());
    let a_ = lab.a / c;
    let b_ = lab.b / c;

    // The cusp is computed here and in find_gamut_intersection, an optimized solution would only compute it once.
    let cusp = find_cusp(a_, b_);

    let ld = l - cusp.l;
    let k = 2. * (if ld > 0.0 { 1. - cusp.l } else { cusp.l });

    let e1 = 0.5 * k + ld.abs() + alpha * c / k;
    let l0 = cusp.l + 0.5 * (sgn(ld) * (e1 - (e1 * e1 - 2. * k * ld.abs()).sqrt()));

    let t = find_gamut_intersection(a_, b_, l, c, l0);
    let l_clipped = l0 * (1. - t) + t * l;
    let c_clipped = t * c;

    oklab_to_linear_srgb(Oklab {
        l: l_clipped,
        a: c_clipped * a_,
        b: c_clipped * b_,
    })
}

fn toe(x: f32) -> f32 {
    const K_1: f32 = 0.206;
    const K_2: f32 = 0.03;
    const K_3: f32 = (1. + K_1) / (1. + K_2);
    0.5 * (K_3 * x - K_1 + ((K_3 * x - K_1) * (K_3 * x - K_1) + 4.0 * K_2 * K_3 * x).sqrt())
}

fn toe_inv(x: f32) -> f32 {
    const K_1: f32 = 0.206;
    const K_2: f32 = 0.03;
    const K_3: f32 = (1. + K_1) / (1. + K_2);
    (x * x + K_1 * x) / (K_3 * (x + K_2))
}

fn to_st(cusp: Lc) -> ST {
    let l = cusp.l;
    let c = cusp.c;
    ST {
        s: c / l,
        t: c / (1.0 - l),
    }
}

// Returns a smooth approximation of the location of the cusp
// This polynomial was created by an optimization process
// It has been designed so that S_mid < S_max and T_mid < T_max
fn get_st_mid(a_: f32, b_: f32) -> ST {
    let s = 0.11516993
        + 1. / (7.4477897
            + 4.1590123 * b_
            + a_ * (-2.1955736
                + 1.751984 * b_
                + a_ * (-2.1370494 - 10.02301 * b_
                    + a_ * (-4.2489457 + 5.387708 * b_ + 4.69891 * a_))));

    let t = 0.11239642
        + 1. / (1.6132032 - 0.6812438 * b_
            + a_ * (0.40370612
                + 0.9014812 * b_
                + a_ * (-0.27087943
                    + 0.6122399 * b_
                    + a_ * (0.00299215 - 0.45399568 * b_ - 0.14661872 * a_))));

    ST { s, t }
}

fn get_cs(l: f32, a_: f32, b_: f32) -> Cs {
    let cusp = find_cusp(a_, b_);

    let c_max = find_gamut_intersection_(a_, b_, l, 1.0, l, cusp);
    let st_max = to_st(cusp);

    // Scale factor to compensate for the curved part of gamut shape:
    let k = c_max / fmin(l * st_max.s, (1.0 - l) * st_max.t);

    let c_mid;
    {
        let st_mid = get_st_mid(a_, b_);

        // Use a soft minimum function, instead of a sharp triangle shape to get a smooth value for chroma.
        let c_a = l * st_mid.s;
        let c_b = (1. - l) * st_mid.t;
        c_mid = 0.9
            * k
            * ((1. / (1. / (c_a * c_a * c_a * c_a) + 1. / (c_b * c_b * c_b * c_b))).sqrt()).sqrt();
    }

    let c_0;
    {
        // for C_0, the shape is independent of hue, so are: ST constant. Values picked to roughly be the average values of ST.
        let c_a = l * 0.4;
        let c_b = (1. - l) * 0.8;

        // Use a soft minimum function, instead of a sharp triangle shape to get a smooth value for chroma.
        c_0 = (1. / (1. / (c_a * c_a) + 1. / (c_b * c_b))).sqrt();
    }

    Cs { c_0, c_mid, c_max }
}

pub fn okhsl_to_srgb(hsl: OkHsl) -> Srgb {
    if hsl.l == 1.0 {
        return Srgb {
            r: 1.,
            g: 1.,
            b: 1.,
        };
    } else if hsl.l == 0. {
        return Srgb {
            r: 0.,
            g: 0.,
            b: 0.,
        };
    }

    let oklab = okhsl_to_oklab(hsl);
    let rgb = oklab_to_linear_srgb(oklab);
    linear_srgb_to_srgb(rgb)
}

pub fn okhsl_to_oklab(hsl: OkHsl) -> Oklab {
    let h = hsl.h;
    let s = hsl.s;
    let l = hsl.l;

    let a_ = (2. * PI * h).cos();
    let b_ = (2. * PI * h).sin();
    let l = toe_inv(l);

    let cs = get_cs(l, a_, b_);
    let c_0 = cs.c_0;
    let c_mid = cs.c_mid;
    let c_max = cs.c_max;

    let mid = 0.8;
    let mid_inv = 1.25;

    let c;
    let t;
    let k_0;
    let k_1;
    let k_2;

    if s < mid {
        t = mid_inv * s;

        k_1 = mid * c_0;
        k_2 = 1. - k_1 / c_mid;

        c = t * k_1 / (1. - k_2 * t);
    } else {
        t = (s - mid) / (1.0 - mid);

        k_0 = c_mid;
        k_1 = (1. - mid) * c_mid * c_mid * mid_inv * mid_inv / c_0;
        k_2 = 1. - (k_1) / (c_max - c_mid);

        c = k_0 + t * k_1 / (1. - k_2 * t);
    }

    Oklab {
        l,
        a: c * a_,
        b: c * b_,
    }
}

pub fn srgb_to_okhsl(rgb: Srgb) -> OkHsl {
    let rgb = srgb_to_linear_srgb(rgb);
    let lab = linear_srgb_to_oklab(rgb);

    oklab_to_okhsl(lab)
}

pub fn oklab_to_okhsl(lab: Oklab) -> OkHsl {
    let c = (lab.a * lab.a + lab.b * lab.b).sqrt();
    let a_ = lab.a / c;
    let b_ = lab.b / c;

    let l = lab.l;
    let h = 0.5 + 0.5 * (-lab.b).atan2(-lab.a) / PI;

    let cs = get_cs(l, a_, b_);
    let c_0 = cs.c_0;
    let c_mid = cs.c_mid;
    let c_max = cs.c_max;

    // Inverse of the interpolation in okhsl_to_srgb:

    let mid = 0.8;
    let mid_inv = 1.25;

    let s = if c < c_mid {
        let k_1 = mid * c_0;
        let k_2 = 1. - k_1 / c_mid;

        let t = c / (k_1 + k_2 * c);
        t * mid
    } else {
        let k_0 = c_mid;
        let k_1 = (1. - mid) * c_mid * c_mid * mid_inv * mid_inv / c_0;
        let k_2 = 1. - (k_1) / (c_max - c_mid);

        let t = (c - k_0) / (k_1 + k_2 * (c - k_0));
        mid + (1. - mid) * t
    };

    let l = toe(l);
    OkHsl { h, s, l }
}

pub fn okhsv_to_srgb(hsv: OkHsv) -> Srgb {
    let oklab = okhsv_to_oklab(hsv);
    let rgb = oklab_to_linear_srgb(oklab);
    linear_srgb_to_srgb(rgb)
}

pub fn oklch_to_oklab(lch: OkLch) -> Oklab {
    let l = lch.l;
    let c = lch.c;
    let h = lch.h;

    // (h - 0.5) * 2 * PI = (-lab.b).atan2(-lab.a);

    let a = c * (2. * PI * h).cos();
    let b = c * (2. * PI * h).sin();

    Oklab { l, a, b }
}

pub fn okhsv_to_oklab(hsv: OkHsv) -> Oklab {
    let h = hsv.h;
    let s = hsv.s;
    let v = hsv.v;

    let a_ = (2. * PI * h).cos();
    let b_ = (2. * PI * h).sin();

    let cusp = find_cusp(a_, b_);
    let st_max = to_st(cusp);
    let s_max = st_max.s;
    let t_max = st_max.t;
    let s_0 = 0.5;
    let k = 1.0 - s_0 / s_max;

    // first we compute L and V as if the gamut is a perfect triangle:

    // L, C when v==1:
    let l_v = 1.0 - s * s_0 / (s_0 + t_max - t_max * k * s);
    let c_v = s * t_max * s_0 / (s_0 + t_max - t_max * k * s);

    let l = v * l_v;
    let c = v * c_v;

    // then we compensate for both toe and the curved top part of the triangle:
    let l_vt = toe_inv(l_v);
    let c_vt = c_v * l_vt / l_v;

    let (l, c) = if l != 0.0 {
        let l_new = toe_inv(l);
        (l_new, c * l_new / l)
    } else {
        let l_new = toe_inv(l);
        (l_new, c)
    };

    let rgb_scale = oklab_to_linear_srgb(Oklab {
        l: l_vt,
        a: a_ * c_vt,
        b: b_ * c_vt,
    });
    let scale_l = (1. / fmax(fmax(rgb_scale.r, rgb_scale.g), fmax(rgb_scale.b, 0.))).cbrt();

    let l = l * scale_l;
    let c = c * scale_l;

    Oklab {
        l,
        a: c * a_,
        b: c * b_,
    }
}

pub fn srgb_to_okhsv(rgb: Srgb) -> OkHsv {
    let rgb = srgb_to_linear_srgb(rgb);
    let lab = linear_srgb_to_oklab(rgb);

    oklab_to_okhsv(lab)
}

pub fn oklab_to_oklch(lab: Oklab) -> OkLch {
    let c = (lab.a * lab.a + lab.b * lab.b).sqrt();
    let h = 0.5 + 0.5 * (-lab.b).atan2(-lab.a) / PI;
    OkLch { l: lab.l, c, h }
}

pub fn oklab_to_okhsv(lab: Oklab) -> OkHsv {
    let OkLch { l, c, h } = oklab_to_oklch(lab);

    if l == 0.0 && c == 0.0 {
        return OkHsv { h, s: 0.0, v: 0.0 };
    }

    let (a_, b_) = if c == 0.0 {
        (1.0, 0.0)
    } else {
        (lab.a / c, lab.b / c)
    };

    let cusp = find_cusp(a_, b_);
    let st_max = to_st(cusp);
    let s_max = st_max.s;
    let t_max = st_max.t;
    let s_0 = 0.5;
    let k = 1.0 - s_0 / s_max;

    // first we find L_v, C_v, L_vt and C_vt

    let t = t_max / (c + l * t_max);
    let l_v = t * l;
    let c_v = t * c;

    let l_vt = toe_inv(l_v);
    let c_vt = c_v * l_vt / l_v;

    // we can then use these to invert the step that compensates for the toe and the curved top part of the triangle:
    let rgb_scale = oklab_to_linear_srgb(Oklab {
        l: l_vt,
        a: a_ * c_vt,
        b: b_ * c_vt,
    });
    let scale_l = (1. / fmax(fmax(rgb_scale.r, rgb_scale.g), fmax(rgb_scale.b, 0.))).cbrt();

    let l = l / scale_l;
    let c = c / scale_l;

    let _c = c * toe(l) / l;
    let l = toe(l);

    // we can now compute v and s:

    let v = l / l_v;
    let s = (s_0 + t_max) * c_v / ((t_max * s_0) + t_max * k * c_v);

    OkHsv { h, s, v }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::{Rng, SeedableRng};

    fn equal_lin_srgb(a: LinSrgb, b: LinSrgb) -> bool {
        let eps = 5e-6;
        (a.r - b.r).abs() < eps && (a.g - b.g).abs() < eps && (a.b - b.b).abs() < eps
    }

    fn seeded_rng() -> impl Rng {
        rand::rngs::StdRng::seed_from_u64(8947) // just a random seed
    }

    #[test]
    fn srgb_conversions() {
        let mut rng = seeded_rng();
        for _ in 0..100 {
            let x = LinSrgb {
                r: rng.gen(),
                g: rng.gen(),
                b: rng.gen(),
            };
            let y = ok_color::linear_srgb_to_srgb(x);
            let z = ok_color::srgb_to_linear_srgb(y);
            assert!(equal_lin_srgb(x, z));
        }
    }

    #[test]
    fn okhsl_conversions() {
        let mut rng = seeded_rng();
        for _ in 0..100 {
            let x = Srgb {
                r: rng.gen(),
                g: rng.gen(),
                b: rng.gen(),
            };
            let y = ok_color::srgb_to_okhsl(x);
            let z = ok_color::okhsl_to_srgb(y);
            assert!(
                equal_lin_srgb(x.to_linear(), z.to_linear()),
                "{:?} {:?}",
                x,
                z
            );
        }
    }

    #[test]
    fn okhsv_conversions() {
        let mut rng = seeded_rng();
        for _ in 0..100 {
            let x = Srgb {
                r: rng.gen(),
                g: rng.gen(),
                b: rng.gen(),
            };
            let y = ok_color::srgb_to_okhsv(x);
            let z = ok_color::okhsv_to_srgb(y);
            assert!(
                equal_lin_srgb(x.to_linear(), z.to_linear()),
                "{:?} {:?}",
                x,
                z
            );
        }
    }

    #[test]
    fn oklab_conversions() {
        let mut rng = seeded_rng();
        for _ in 0..100 {
            let x = LinSrgb {
                r: rng.gen(),
                g: rng.gen(),
                b: rng.gen(),
            };
            let y = ok_color::linear_srgb_to_oklab(x);
            let z = ok_color::oklab_to_linear_srgb(y);
            assert!(equal_lin_srgb(x, z), "{:?} {:?}", x, z);
        }
    }

    #[test]
    fn oklch_conversions() {
        let mut rng = seeded_rng();
        for _ in 0..100 {
            let x = Oklab {
                l: rng.gen(),
                a: rng.gen(),
                b: rng.gen(),
            };
            let y = ok_color::oklab_to_oklch(x);
            let z = ok_color::oklch_to_oklab(y);

            assert!(
                equal_lin_srgb(LinSrgb::from(y), LinSrgb::from(z)),
                "{:?} {:?}",
                x,
                z
            );
        }
    }

    #[test]
    fn hsv_to_lch() {
        let mut rng = seeded_rng();
        for _ in 0..100 {
            let x = OkHsv {
                h: rng.gen(),
                s: rng.gen(),
                v: rng.gen(),
            };

            let y = ok_color::okhsv_to_oklab(x);
            let z = ok_color::oklab_to_okhsv(y);

            assert!(
                equal_lin_srgb(LinSrgb::from(x), LinSrgb::from(z)),
                "{:?} {:?}",
                x,
                z
            );
        }
    }

    #[test]
    fn hsv_to_lch_only_value() {
        let mut rng = seeded_rng();
        for _ in 0..100 {
            let x = OkHsv {
                h: 0.0,
                s: 0.0,
                v: rng.gen(),
            };

            let y = ok_color::okhsv_to_oklab(x);
            let z = ok_color::oklab_to_okhsv(y);

            assert!(
                equal_lin_srgb(LinSrgb::from(x), LinSrgb::from(z)),
                "{:?} {:?}",
                x,
                z
            );
        }
    }

    #[test]
    fn hsv_value_zero() {
        let okhsv = OkHsv {
            h: 0.0,
            s: 1.0,
            v: 0.0,
        };

        let oklab = ok_color::okhsv_to_oklab(okhsv);

        println!("{:?}", okhsv);
        println!("{:?}", oklab);

        assert!(!oklab.l.is_nan());
        assert!(!oklab.a.is_nan());
        assert!(!oklab.b.is_nan());

        assert_eq!(oklab.l, 0.0);
        assert_eq!(oklab.a, 0.0);
        assert_eq!(oklab.b, 0.0);
    }

    #[test]
    fn hsv_saturation_zero() {
        let oklch = OkLch {
            l: 0.77861947,
            c: 0.0,
            h: 0.0,
        };

        let okhsv = OkHsv::from(oklch);

        println!("{:?}", oklch);
        println!("{:?}", okhsv);

        assert!(!okhsv.h.is_nan());
        assert!(!okhsv.s.is_nan());
        assert!(!okhsv.v.is_nan());
    }

    #[test]
    fn zero_oklab_to_okhsv() {
        let oklab = Oklab {
            l: 0.0,
            a: 0.0,
            b: 0.0,
        };
        let okhsv = ok_color::oklab_to_okhsv(oklab);

        println!("{:?}", okhsv);

        assert!(!okhsv.h.is_nan());
        assert!(!okhsv.s.is_nan());
        assert!(!okhsv.v.is_nan());
    }
}
