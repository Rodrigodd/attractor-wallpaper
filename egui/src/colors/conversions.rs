// SPDX-FileCopyrightText: 2022 Björn Ottosson
// SPDX-FileCopyrightText: 2022 Gerry Agbobada <git@gagbo.net>
//
// SPDX-License-Identifier: MIT

//! Colorspace conversions
//!
//! Yes, it could be shaders. It could.

use super::{LinSrgb, OkHsl, OkHsv, OkLCh, OkLab, Srgb};
use crate::colors::ACCEPTABLE_ERROR;

/// The matrices were updated 2021-01-25
static M1_LIN_SRGB_TO_OKLAB: [[f64; 3]; 3] = [
    [0.412_221_470_8, 0.536_332_536_3, 0.051_445_992_9],
    [0.211_903_498_2, 0.680_699_545_1, 0.107_396_956_6],
    [0.088_302_461_9, 0.281_718_837_6, 0.629_978_700_5],
];

/// The matrices were updated 2021-01-25
static M2_LIN_SRGB_TO_OKLAB: [[f64; 3]; 3] = [
    [0.210_454_255_3, 0.793_617_785_0, -0.004_072_046_8],
    [1.977_998_495_1, -2.428_592_205_0, 0.450_593_709_9],
    [0.025_904_037_1, 0.782_771_766_2, -0.808_675_766_0],
];

/// The matrices were updated 2021-01-25
static M1_OKLAB_TO_LIN_SRGB: [[f64; 3]; 3] = [
    [1.0, 0.396_337_777_4, 0.215_803_757_3],
    [1.0, -0.105_561_345_8, -0.063_854_172_8],
    [1.0, -0.089_484_177_5, -1.291_485_548_0],
];

/// The matrices were updated 2021-01-25
static M2_OKLAB_TO_LIN_SRGB: [[f64; 3]; 3] = [
    [4.076_741_662_1, -3.307_711_591_3, 0.230_969_929_2],
    [-1.268_438_004_6, 2.609_757_401_1, -0.341_319_396_5],
    [-0.004_196_086_3, -0.703_418_614_7, 1.707_614_701_0],
];

trait Matrix {
    fn dot(&self, other: &[f64; 3]) -> [f64; 3];
}
impl Matrix for [[f64; 3]; 3] {
    fn dot(&self, other: &[f64; 3]) -> [f64; 3] {
        [
            self[0][0] * other[0] + self[0][1] * other[1] + self[0][2] * other[2],
            self[1][0] * other[0] + self[1][1] * other[1] + self[1][2] * other[2],
            self[2][0] * other[0] + self[2][1] * other[1] + self[2][2] * other[2],
        ]
    }
}

impl From<LinSrgb> for Srgb {
    fn from(linear: LinSrgb) -> Self {
        fn transform(val: f64) -> f64 {
            if val >= 0.003_130_8 {
                val.powf(1.0 / 2.4).mul_add(1.055, -0.055)
            } else {
                12.92 * val
            }
        }

        Self {
            red: transform(linear.red),
            green: transform(linear.green),
            blue: transform(linear.blue),
        }
    }
}

impl From<Srgb> for LinSrgb {
    fn from(gammad: Srgb) -> Self {
        fn inverse_transform(val: f64) -> f64 {
            if val >= 0.040_45 {
                ((val + 0.055) / 1.055).powf(2.4)
            } else {
                val / 12.92
            }
        }

        Self {
            red: inverse_transform(gammad.red),
            green: inverse_transform(gammad.green),
            blue: inverse_transform(gammad.blue),
        }
    }
}

impl From<OkLab> for OkLCh {
    fn from(lab: OkLab) -> Self {
        Self {
            lightness: lab.lightness,
            chroma: (lab.a.powi(2) + lab.b.powi(2)).sqrt(),
            hue: lab.b.atan2(lab.a),
        }
    }
}

impl From<OkLCh> for OkLab {
    fn from(lch: OkLCh) -> Self {
        Self {
            lightness: lch.lightness,
            a: lch.chroma * lch.hue.cos(),
            b: lch.chroma * lch.hue.sin(),
        }
    }
}

impl From<LinSrgb> for OkLab {
    fn from(lin: LinSrgb) -> Self {
        debug_assert!(lin.is_normal(), "lin is not normal {lin:?}");

        let lin_vec: [f64; 3] = lin.into();
        let lms = M1_LIN_SRGB_TO_OKLAB.dot(&lin_vec).to_vec();
        let lms_ = [lms[0].cbrt(), lms[1].cbrt(), lms[2].cbrt()];

        let col: Self = M2_LIN_SRGB_TO_OKLAB.dot(&lms_).into();
        debug_assert!(col.is_normal(), "col is not normal: {col:?}");
        col
    }
}

impl From<OkLab> for LinSrgb {
    fn from(lab: OkLab) -> Self {
        debug_assert!(lab.is_normal(), "lab is not normal {lab:?}");

        let lab_vec = lab.into();
        let lms_ = M1_OKLAB_TO_LIN_SRGB.dot(&lab_vec).to_vec();
        let lms = [lms_[0].powi(3), lms_[1].powi(3), lms_[2].powi(3)];

        let col: Self = M2_OKLAB_TO_LIN_SRGB.dot(&lms).into();
        debug_assert!(col.is_normal(), "col is not normal: {col:?}");
        col
    }
}

impl From<Srgb> for OkHsv {
    fn from(gammad_rgb: Srgb) -> Self {
        debug_assert!(
            gammad_rgb.is_normal(),
            "gammad_rgb isn't normal {gammad_rgb:?}"
        );
        let lab = OkLab::from(LinSrgb::from(gammad_rgb));
        debug_assert!(lab.is_normal(), "lab isn't normal {lab:?}");

        let chroma = (lab.a.powi(2) + lab.b.powi(2)).sqrt();
        let saturated_a = if chroma.is_normal() {
            lab.a / chroma
        } else if lab.a.is_sign_positive() {
            // Artificially move to a forced chroma if we received 0.0
            1.0
        } else {
            -1.0
        };
        let saturated_b = if chroma.is_normal() {
            lab.b / chroma
        } else {
            0.0
        };

        let hue = saturated_b.atan2(saturated_a);
        let cusp = find_cusp(saturated_a, saturated_b);
        let st_max = ST::from_cusp(cusp);

        const S0: f64 = 0.5;
        let k = 1.0 - S0 / st_max.s;

        // first we find L_v, C_v, L_vt and C_vt
        let t = if st_max.t.is_finite() && chroma.is_normal() {
            st_max.t / (chroma + lab.lightness * st_max.t)
        } else {
            0.0
        };
        debug_assert!(t.is_finite(), "t is not normal {t:?}");
        let l_v = t * lab.lightness;
        let c_v = t * chroma;

        let l_vt = inverse_toe(l_v);
        let c_vt = if l_v.is_normal() {
            c_v * l_vt / l_v
        } else {
            c_v
        };

        let rgb_scale = LinSrgb::from(OkLab {
            lightness: l_vt,
            a: saturated_a * c_vt,
            b: saturated_b * c_vt,
        });
        let scale_l = (1.0
            / f64::max(
                f64::max(rgb_scale.red, rgb_scale.green),
                f64::max(rgb_scale.blue, 0.0),
            ))
        .cbrt();
        let scaled_lightness = lab.lightness / scale_l;
        /* Code from the original source, that is unused here
         *
         * let scaled_chroma = chroma / scale_l;
         *
         * let scaled_chroma = scaled_chroma * toe(scaled_lightness) / scaled_lightness;
         */
        let l = toe(scaled_lightness);

        Self {
            hue,
            saturation: (S0 + st_max.t) * c_v / ((st_max.t * S0) + st_max.t * k * c_v),
            value: if l_v == 0.0 { 0.0 } else { l / l_v },
        }
    }
}

impl From<OkHsv> for Srgb {
    fn from(hsv: OkHsv) -> Self {
        debug_assert!(hsv.is_normal(), "hsv isn't normal {hsv:?}");

        let a_ = (hsv.hue).cos();
        let b_ = (hsv.hue).sin();
        let cusp = find_cusp(a_, b_);
        let st_max = ST::from_cusp(cusp);

        const S0: f64 = 0.5;
        let k = 1.0 - S0 / st_max.s;
        debug_assert!(k.is_finite(), "k is not normal {k:?}");

        // first we compute L and V as if the gamut is a perfect triangle:

        // L, C when v==1:
        let l_v = 1.0 - hsv.saturation * S0 / (S0 + st_max.t - st_max.t * k * hsv.saturation);
        debug_assert!(l_v.is_finite(), "l_v is not normal {l_v:?}");
        let c_v = hsv.saturation * st_max.t * S0 / (S0 + st_max.t - st_max.t * k * hsv.saturation);

        let l = hsv.value * l_v;
        let c = hsv.value * c_v;

        // then we compensate for both toe and the curved top part of the triangle:
        let l_vt = inverse_toe(l_v);
        debug_assert!(l_vt.is_finite(), "l_vt is not normal {l_vt:?}");
        let c_vt = if l_v.is_normal() {
            c_v * l_vt / l_v
        } else {
            c_v
        };

        let l_new = inverse_toe(l);
        debug_assert!(l_new.is_finite(), "l_new is not normal {l_new:?}");
        let c = if l.is_normal() { c * l_new / l } else { c };
        let l = l_new;

        let rgb_scale = LinSrgb::from(OkLab {
            lightness: l_vt,
            a: a_ * c_vt,
            b: b_ * c_vt,
        });
        debug_assert!(
            rgb_scale.is_normal(),
            "rgb_scale is not normal {rgb_scale:?}"
        );
        let scale_l = (1.0
            / f64::max(
                f64::max(rgb_scale.red, rgb_scale.green),
                f64::max(rgb_scale.blue, std::f64::MIN_POSITIVE),
            ))
        .cbrt();
        debug_assert!(scale_l.is_finite(), "scale_l is not normal {scale_l:?}");
        let l = l * scale_l;
        let c = c * scale_l;

        let resulting_lab = OkLab {
            lightness: l,
            a: c * a_,
            b: c * b_,
        };
        debug_assert!(
            resulting_lab.is_normal(),
            "resulting_lab is not normal {resulting_lab:?}"
        );

        LinSrgb::from(resulting_lab).into()
    }
}

impl From<OkHsl> for Srgb {
    fn from(hsl: OkHsl) -> Self {
        if hsl.lightness == 1.0 {
            return Self {
                red: 1.0,
                green: 1.0,
                blue: 1.0,
            };
        }

        if hsl.lightness == 0.0 {
            return Self {
                red: 0.0,
                green: 0.0,
                blue: 0.0,
            };
        }

        let a = hsl.hue.cos();
        let b = hsl.hue.sin();
        let l = inverse_toe(hsl.lightness);

        let Cs { c_0, c_mid, c_max } = Cs::from(OkLab { lightness: l, a, b });

        let mid = 0.8;
        let mid_inv = 1.25_f64;

        let c = if hsl.saturation < mid {
            let t = mid_inv * hsl.saturation;
            let k_1 = mid * c_0;
            let k_2 = 1.0 - k_1 / c_mid;

            t * k_1 / (1.0 - k_2 * t)
        } else {
            let t = (hsl.saturation - mid) / (1.0 - mid);
            let k_0 = c_mid;
            let k_1 = (1.0 - mid) * c_mid.powi(2) * mid_inv.powi(2) / c_0;
            let k_2 = 1.0 - k_1 / (c_max - c_mid);

            k_0 + t * k_1 / (1.0 - k_2 * t)
        };

        Self::from(LinSrgb::from(OkLab {
            lightness: l,
            a: c * a,
            b: c * b,
        }))
    }
}

impl From<Srgb> for OkHsl {
    fn from(rgb: Srgb) -> Self {
        let lab = OkLab::from(LinSrgb::from(rgb));

        let chroma = (lab.a.powi(2) + lab.b.powi(2)).sqrt();
        let a_ = if chroma.is_normal() {
            lab.a / chroma
        } else {
            1.0
        };
        let b_ = if chroma.is_normal() {
            lab.b / chroma
        } else {
            0.0
        };

        let lightness = lab.lightness;
        let hue = b_.atan2(a_);

        let Cs { c_0, c_mid, c_max } = Cs::from(OkLab {
            lightness,
            a: a_,
            b: b_,
        });
        // Inverse of the interpolation in Srgb::from::<OkHsl>()
        let mid = 0.8;
        let mid_inv = 1.25_f64;

        let saturation = if c_0 == 0.0 && c_max == 0.0 {
            0.0
        } else if chroma < c_mid {
            let k_1 = mid * c_0;
            let k_2 = 1.0 - k_1 / c_mid;
            debug_assert!(k_1.is_finite(), "k1 is not normal {k_1:?}");
            debug_assert!(k_2.is_finite(), "k2 is not normal {k_2:?}");
            debug_assert!(c_mid.is_normal(), "c_mid is not normal {c_mid:?}");
            debug_assert!(c_0.is_normal(), "c_0 is not normal {c_0:?}");

            mid * chroma / (k_1 + k_2 * chroma)
        } else {
            let k_0 = c_mid;
            let k_1 = (1.0 - mid) * c_mid.powi(2) * mid_inv.powi(2) / c_0;
            let k_2 = 1.0 - k_1 / (c_max - c_mid);
            debug_assert!(k_0.is_finite(), "k0 is not normal {k_0:?}");
            debug_assert!(k_1.is_finite(), "k1 is not normal {k_1:?}");
            debug_assert!(k_2.is_finite(), "k2 is not normal {k_2:?}");
            debug_assert!(c_mid.is_normal(), "c_mid is not normal {c_mid:?}");
            debug_assert!(c_max.is_normal(), "c_max is not normal {c_max:?}");
            debug_assert!(c_0.is_normal(), "c_0 is not normal {c_0:?}");

            mid + (1.0 - mid) * (chroma - k_0) / (k_1 + k_2 * (chroma - k_0))
        };

        Self {
            hue,
            saturation,
            lightness: toe(lightness),
        }
    }
}

impl From<[f64; 3]> for LinSrgb {
    fn from(col: [f64; 3]) -> Self {
        Self {
            red: col[0],
            green: col[1],
            blue: col[2],
        }
    }
}

impl From<LinSrgb> for [f64; 3] {
    fn from(col: LinSrgb) -> Self {
        [col.red, col.green, col.blue]
    }
}

impl From<[f64; 3]> for OkLab {
    fn from(col: [f64; 3]) -> Self {
        Self {
            lightness: col[0],
            a: col[1],
            b: col[2],
        }
    }
}

impl From<OkLab> for [f64; 3] {
    fn from(col: OkLab) -> Self {
        [col.lightness, col.a, col.b]
    }
}

#[derive(Clone, Copy, Debug)]
struct LC {
    pub lightness: f64,
    pub chroma: f64,
}

/// Alternative representation of (L_cusp, C_cusp)
///
/// Encoded so S = C_cusp/L_cusp and T = C_cusp/(1-L_cusp)
/// The maximum value for C in the triangle is then found as
/// fmin(S*L, T*(1-L)), for a given L
#[derive(Clone, Copy, Debug)]
struct ST {
    pub s: f64,
    pub t: f64,
}

impl ST {
    fn from_cusp(cusp: LC) -> Self {
        let l = cusp.lightness;
        debug_assert!(l.is_normal(), "l is {l:?}");
        debug_assert!((1.0 - l).is_normal(), "l is {l:?}");
        Self {
            s: cusp.chroma / l,
            t: cusp.chroma / (1.0 - l),
        }
    }

    /// Returns a smooth approximation of the location of the cusp
    /// This polynomial was created by an optimization process
    /// It has been designed so that S_mid < S_max and T_mid < T_max
    fn mid(a: f64, b: f64) -> Self {
        Self {
            s: 0.115_169_93
                + 1.0
                    / (7.447_789_70
                        + 4.159_012_40 * b
                        + a * (-2.195_573_47
                            + 1.751_984_01 * b
                            + a * (-2.137_049_48 - 10.023_010_43 * b
                                + a * (-4.248_945_61 + 5.387_708_19 * b + 4.698_910_13 * a)))),

            t: 0.112_396_42
                + 1.0
                    / (1.613_203_20 - 0.681_243_79 * b
                        + a * (0.403_706_12
                            + 0.901_481_23 * b
                            + a * (-0.270_879_43
                                + 0.612_239_90 * b
                                + a * (0.002_992_15 - 0.453_995_68 * b - 0.146_618_72 * a)))),
        }
    }
}

/// toe function for L_r
fn toe(val: f64) -> f64 {
    const K1: f64 = 0.206;
    const K2: f64 = 0.03;
    const K3: f64 = (K1 + 1.0) / (K2 + 1.0);
    0.5 * (K3 * val - K1 + ((K3 * val - K1) * (K3 * val - K1) + 4.0 * K2 * K3 * val).sqrt())
}

/// inverse toe function for L_r
fn inverse_toe(val: f64) -> f64 {
    const K1: f64 = 0.206;
    const K2: f64 = 0.03;
    const K3: f64 = (K1 + 1.0) / (K2 + 1.0);
    (val * val + K1 * val) / (K3 * (val + K2))
}

fn find_cusp(a: f64, b: f64) -> LC {
    debug_assert!(
        (1.0 - a.powi(2) - b.powi(2)).abs() < ACCEPTABLE_ERROR,
        "Precondition failed: ({a:?}, {b:?}) isn't on unit circle (norm is {})",
        a.powi(2) + b.powi(2)
    );
    let s_cusp = compute_max_saturation(a, b);

    let max_rgb = LinSrgb::from(OkLab {
        lightness: 1.0,
        a: s_cusp * a,
        b: s_cusp * b,
    });
    debug_assert!(max_rgb.is_normal(), "max_rgb is not normal: {max_rgb:?}");
    let lightness = (1.0 / f64::max(max_rgb.red, f64::max(max_rgb.green, max_rgb.blue))).cbrt();
    LC {
        lightness,
        chroma: lightness * s_cusp,
    }
}
/// Finds the maximum saturation possible for a given hue that fits in sRGB
///
/// Saturation here is defined as S = C/L
/// a and b must be normalized so a^2 + b^2 == 1
fn compute_max_saturation(a: f64, b: f64) -> f64 {
    debug_assert!(
        (1.0 - a.powi(2) - b.powi(2)).abs() < ACCEPTABLE_ERROR,
        "Precondition failed: ({a:?}, {b:?}) isn't on unit circle (norm is {})",
        a.powi(2) + b.powi(2)
    );
    // Max saturation will be when one of r, g or b goes below zero.

    // Select different coefficients depending on which component goes below zero first
    let (k0, k1, k2, k3, k4, wl, wm, ws) = if 1.0 < a.mul_add(-1.881_703_28, -0.809_364_93 * b) {
        // red component
        (
            1.190_862_77,
            1.765_767_28,
            0.596_626_41,
            0.755_151_97,
            0.567_712_45,
            M2_OKLAB_TO_LIN_SRGB[0][0],
            M2_OKLAB_TO_LIN_SRGB[0][1],
            M2_OKLAB_TO_LIN_SRGB[0][2],
        )
    } else if 1.0 < a.mul_add(1.814_441_04, -1.194_452_76 * b) {
        // green component
        (
            0.739_565_15,
            -0.459_544_04,
            0.082_854_27,
            0.125_410_70,
            0.145_032_04,
            M2_OKLAB_TO_LIN_SRGB[1][0],
            M2_OKLAB_TO_LIN_SRGB[1][1],
            M2_OKLAB_TO_LIN_SRGB[1][2],
        )
    } else {
        // blue component
        (
            1.357_336_52,
            -0.009_157_99,
            -1.151_302_10,
            -0.505_596_06,
            0.006_921_67,
            M2_OKLAB_TO_LIN_SRGB[2][0],
            M2_OKLAB_TO_LIN_SRGB[2][1],
            M2_OKLAB_TO_LIN_SRGB[2][2],
        )
    };

    // Approximate max saturation using a polynomial:
    let mut sat = k0 + k1 * a + k2 * b + k3 * a * a + k4 * a * b;

    // Do one step Halley's method to get closer
    // this gives an error less than 10e6, except for some blue hues where the dS/dh is close to infinite
    // this should be sufficient for most applications, otherwise do two/three steps
    for _ in 0..4 {
        let (k_l, k_m, k_s) = (
            M1_OKLAB_TO_LIN_SRGB[0][1] * a + M1_OKLAB_TO_LIN_SRGB[0][2] * b,
            M1_OKLAB_TO_LIN_SRGB[1][1] * a + M1_OKLAB_TO_LIN_SRGB[1][2] * b,
            M1_OKLAB_TO_LIN_SRGB[2][1] * a + M1_OKLAB_TO_LIN_SRGB[2][2] * b,
        );

        let (l_, m_, s_) = (1.0 + sat * k_l, 1.0 + sat * k_m, 1.0 + sat * k_s);
        let (l, m, s) = (l_.powi(3), m_.powi(3), s_.powi(3));
        let (l_ds, m_ds, s_ds) = (
            3.0 * k_l * l_ * l_,
            3.0 * k_m * m_ * m_,
            3.0 * k_s * s_ * s_,
        );
        let (l_ds2, m_ds2, s_ds2) = (
            6.0 * k_l * k_l * l_,
            6.0 * k_m * k_m * m_,
            6.0 * k_s * k_s * s_,
        );
        let f = wl * l + wm * m + ws * s;
        let f1 = wl * l_ds + wm * m_ds + ws * s_ds;
        let f2 = wl * l_ds2 + wm * m_ds2 + ws * s_ds2;

        sat -= f * f1 / (f1 * f1 - 0.5 * f * f2);
    }

    debug_assert!(sat.is_finite(), "Maximum saturation is not correct {sat:?}");
    sat
}

#[derive(Clone, Copy, Debug)]
struct Cs {
    c_0: f64,
    c_mid: f64,
    c_max: f64,
}

impl From<OkLab> for Cs {
    fn from(lab: OkLab) -> Self {
        debug_assert!(lab.is_normal(), "lab isn't normal: {lab:?}");
        let cusp = find_cusp(lab.a, lab.b);
        let c_max = find_gamut_intersection(lab.a, lab.b, lab.lightness, 1.0, lab.lightness, cusp);
        let st_max = ST::from_cusp(cusp);
        let k = c_max
            / (lab.lightness * st_max.s)
                .min((1.0 - lab.lightness) * st_max.t)
                .max(std::f64::MIN_POSITIVE);
        debug_assert!(k.is_finite(), "k is not normal: {k:?}");
        let st_mid = ST::mid(lab.a, lab.b);

        debug_assert!(
            st_mid.s.is_normal() && st_mid.t.is_normal(),
            "st_mid is not normal: {st_mid:?}"
        );
        debug_assert!(
            st_max.s.is_normal() && st_max.t.is_normal(),
            "st_max is not normal: {st_max:?}"
        );

        // Use a soft minimum function, instead of a sharp triangle shape to get a smooth value for chroma.
        let c_a = lab.lightness * st_mid.s;
        let c_b = (1.0 - lab.lightness) * st_mid.t;
        let c_mid = if !(c_a.is_normal() && c_b.is_normal()) {
            0.0
        } else {
            0.9 * k * (c_a.powi(-4) + c_b.powi(-4)).powi(-1).sqrt().sqrt()
        };

        // for C_0, the shape is independent of hue, so ST are constant. Values picked to roughly be the average values of ST.
        let c_a = lab.lightness * 0.4;
        let c_b = (1.0 - lab.lightness) * 0.8;
        let c_0 = if !(c_a.is_normal() && c_b.is_normal()) {
            0.0
        } else {
            (c_a.powi(-2) + c_b.powi(-2)).powi(-1).sqrt()
        };

        debug_assert!(c_0.is_finite(), "c_0 is not normal: {c_0:?}");
        debug_assert!(c_mid.is_finite(), "c_mid is not normal: {c_mid:?}");
        debug_assert!(c_max.is_finite(), "c_max is not normal: {c_max:?}");
        Self { c_0, c_mid, c_max }
    }
}

/// Finds intersection of the line defined by
/// L = L0 * (1 - t) + t * L1;
/// C = t * C1;
/// a and b must be normalized so a^2 + b^2 == 1
fn find_gamut_intersection(a: f64, b: f64, l1: f64, c1: f64, l0: f64, cusp: LC) -> f64 {
    // Find the intersection for upper and lower half seprately
    if (l1 - l0) * cusp.chroma <= (cusp.lightness - l0) * c1 {
        // Lower half
        cusp.chroma * l0 / (c1 * cusp.lightness + cusp.chroma * (l0 - l1))
    } else {
        // Upper half

        // First intersect with triangle
        let mut target =
            cusp.chroma * (l0 - 1.0) / (c1 * (cusp.lightness - 1.0) + cusp.chroma * (l0 - l1));
        // Then one step Halley's method

        let d_l = l1 - l0;
        let d_c = c1;
        let k_l = M1_OKLAB_TO_LIN_SRGB[0][1] * a + M1_OKLAB_TO_LIN_SRGB[0][2] * b;
        let k_m = M1_OKLAB_TO_LIN_SRGB[1][1] * a + M1_OKLAB_TO_LIN_SRGB[1][2] * b;
        let k_s = M1_OKLAB_TO_LIN_SRGB[2][1] * a + M1_OKLAB_TO_LIN_SRGB[2][2] * b;

        let l_dt = d_l + d_c * k_l;
        let m_dt = d_l + d_c * k_m;
        let s_dt = d_l + d_c * k_s;

        // If higher accuracy is required, 2 or 3 iterations of the following block can be used:
        for _ in 0..4 {
            let l = l0 * (1.0 - target) + target * l1;
            let c = target * c1;

            let l_ = l + c * k_l;
            let m_ = l + c * k_m;
            let s_ = l + c * k_s;

            let l_c = l_.powi(3);
            let m_c = m_.powi(3);
            let s_c = s_.powi(3);

            let ldt = 3.0 * l_dt * l_ * l_;
            let mdt = 3.0 * m_dt * m_ * m_;
            let sdt = 3.0 * s_dt * s_ * s_;

            let ldt2 = 6.0 * l_dt * l_dt * l_;
            let mdt2 = 6.0 * m_dt * m_dt * m_;
            let sdt2 = 6.0 * s_dt * s_dt * s_;

            let r = M2_OKLAB_TO_LIN_SRGB[0][0] * l_c
                + M2_OKLAB_TO_LIN_SRGB[0][1] * m_c
                + M2_OKLAB_TO_LIN_SRGB[0][2] * s_c
                - 1.0;
            let r1 = M2_OKLAB_TO_LIN_SRGB[0][0] * ldt
                + M2_OKLAB_TO_LIN_SRGB[0][1] * mdt
                + M2_OKLAB_TO_LIN_SRGB[0][2] * sdt;
            let r2 = M2_OKLAB_TO_LIN_SRGB[0][0] * ldt2
                + M2_OKLAB_TO_LIN_SRGB[0][1] * mdt2
                + M2_OKLAB_TO_LIN_SRGB[0][2] * sdt2;

            let u_r = r1 / (r1 * r1 - 0.5 * r * r2);
            let t_r = if u_r > 0.0 { Some(-r * u_r) } else { None };

            let g = M2_OKLAB_TO_LIN_SRGB[1][0] * l_c
                + M2_OKLAB_TO_LIN_SRGB[1][1] * m_c
                + M2_OKLAB_TO_LIN_SRGB[1][2] * s_c
                - 1.0;
            let g1 = M2_OKLAB_TO_LIN_SRGB[1][0] * ldt
                + M2_OKLAB_TO_LIN_SRGB[1][1] * mdt
                + M2_OKLAB_TO_LIN_SRGB[1][2] * sdt;
            let g2 = M2_OKLAB_TO_LIN_SRGB[1][0] * ldt2
                + M2_OKLAB_TO_LIN_SRGB[1][1] * mdt2
                + M2_OKLAB_TO_LIN_SRGB[1][2] * sdt2;

            let u_g = g1 / (g1 * g1 - 0.5 * g * g2);
            let t_g = if u_g > 0.0 { Some(-g * u_g) } else { None };

            let b = M2_OKLAB_TO_LIN_SRGB[2][0] * l_c
                + M2_OKLAB_TO_LIN_SRGB[2][1] * m_c
                + M2_OKLAB_TO_LIN_SRGB[2][2] * s_c
                - 1.0;
            let b1 = M2_OKLAB_TO_LIN_SRGB[2][0] * ldt
                + M2_OKLAB_TO_LIN_SRGB[2][1] * mdt
                + M2_OKLAB_TO_LIN_SRGB[2][2] * sdt;
            let b2 = M2_OKLAB_TO_LIN_SRGB[2][0] * ldt2
                + M2_OKLAB_TO_LIN_SRGB[2][1] * mdt2
                + M2_OKLAB_TO_LIN_SRGB[2][2] * sdt2;

            let u_b = b1 / (b1 * b1 - 0.5 * b * b2);
            let t_b = if u_b > 0.0 { Some(-b * u_b) } else { None };

            target += [t_r, t_g, t_b]
                .into_iter()
                .flatten()
                .min_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0)
        }

        target
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    const STEPS: i32 = 32;

    #[test]
    fn okhsl_srgb() {
        let red_steps = STEPS;
        let green_steps = STEPS;
        let blue_steps = STEPS;
        for red_step in 0..red_steps {
            for green_step in 0..green_steps {
                for blue_step in 0..blue_steps {
                    let init_col = Srgb {
                        red: red_step as f64 / red_steps as f64,
                        green: green_step as f64 / green_steps as f64,
                        blue: blue_step as f64 / blue_steps as f64,
                    };
                    let return_col = Srgb::from(OkHsl::from(init_col));
                    assert!(
                        return_col.is_normal(),
                        "return_col is not normal\n{init_col:?} became\n{return_col:?}"
                    );
                    // Comparing with f32 epsilon to allow some leeway
                    if init_col.red.abs() > std::f32::EPSILON as f64 {
                        let error = (init_col.red - return_col.red).abs() / init_col.red.abs();
                        assert!(
                            error < ACCEPTABLE_ERROR,
                            "The red is too different: \n\tInit {init_col:?}\n\tBack {return_col:?}\n\tError: {}%",
                            error * 100.0
                        );
                    } else {
                        let ratio = (return_col.red
                            / init_col
                                .red
                                .max(init_col.green)
                                .max(init_col.blue)
                                .max(std::f32::EPSILON as f64))
                        .abs();
                        assert!(
                            ratio <= ACCEPTABLE_ERROR,
                            "The red should be negligible from\n\t{init_col:?}, got\n\t{return_col:?} instead",
                        );
                    }

                    if init_col.green.abs() > std::f32::EPSILON as f64 {
                        let error =
                            (init_col.green - return_col.green).abs() / init_col.green.abs();
                        assert!(
                            error < ACCEPTABLE_ERROR,
                            "The green is too different: \n\tInit {init_col:?}\n\tBack {return_col:?}\n\tError: {}%",
                            error * 100.0
                        );
                    } else {
                        let ratio = (return_col.green
                            / init_col
                                .red
                                .max(init_col.green)
                                .max(init_col.blue)
                                .max(std::f32::EPSILON as f64))
                        .abs();
                        assert!(
                            ratio <= ACCEPTABLE_ERROR,
                            "The green should be negligible from\n\t{init_col:?}, got\n\t{return_col:?} instead",
                        );
                    }

                    if init_col.blue.abs() > std::f32::EPSILON as f64 {
                        let error = (init_col.blue - return_col.blue).abs() / init_col.blue.abs();
                        assert!(
                            error < ACCEPTABLE_ERROR,
                            "The blue is too different: \n\tInit {init_col:?}\n\tBack {return_col:?}\n\tError: {}%",
                            error * 100.0
                        );
                    } else {
                        let ratio = (return_col.blue
                            / init_col
                                .red
                                .max(init_col.green)
                                .max(init_col.blue)
                                .max(std::f32::EPSILON as f64))
                        .abs();
                        assert!(
                            ratio <= ACCEPTABLE_ERROR,
                            "The blue should be negligible from\n\t{init_col:?}, got\n\t{return_col:?} instead",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn okhsv_srgb() {
        let red_steps = STEPS;
        let green_steps = STEPS;
        let blue_steps = STEPS;
        for red_step in 0..red_steps {
            for green_step in 0..green_steps {
                for blue_step in 0..blue_steps {
                    let init_col = Srgb {
                        red: red_step as f64 / red_steps as f64,
                        green: green_step as f64 / green_steps as f64,
                        blue: blue_step as f64 / blue_steps as f64,
                    };
                    let return_col = Srgb::from(OkHsv::from(init_col));
                    assert!(
                        return_col.is_normal(),
                        "return_col is not normal\n{init_col:?} became\n{return_col:?}"
                    );
                    // Comparing with f32 epsilon to allow some leeway
                    if init_col.red.abs() > std::f32::EPSILON as f64 {
                        let error = (init_col.red - return_col.red).abs() / init_col.red.abs();
                        assert!(
                            error < ACCEPTABLE_ERROR,
                            "The red is too different: \n\tInit {init_col:?}\n\tBack {return_col:?}\n\tError: {}%",
                            error * 100.0
                        );
                    } else {
                        let ratio = (return_col.red
                            / init_col
                                .red
                                .max(init_col.green)
                                .max(init_col.blue)
                                .max(std::f32::EPSILON as f64))
                        .abs();
                        assert!(
                            ratio <= ACCEPTABLE_ERROR,
                            "The red should be negligible from\n\t{init_col:?}, got\n\t{return_col:?} instead",
                        );
                    }

                    if init_col.green.abs() > std::f32::EPSILON as f64 {
                        let error =
                            (init_col.green - return_col.green).abs() / init_col.green.abs();
                        assert!(
                            error < ACCEPTABLE_ERROR,
                            "The green is too different: \n\tInit {init_col:?}\n\tBack {return_col:?}\n\tError: {}%",
                            error * 100.0
                        );
                    } else {
                        let ratio = (return_col.green
                            / init_col
                                .red
                                .max(init_col.green)
                                .max(init_col.blue)
                                .max(std::f32::EPSILON as f64))
                        .abs();
                        assert!(
                            ratio <= ACCEPTABLE_ERROR,
                            "The green should be negligible from\n\t{init_col:?}, got\n\t{return_col:?} instead",
                        );
                    }

                    if init_col.blue.abs() > std::f32::EPSILON as f64 {
                        let error = (init_col.blue - return_col.blue).abs() / init_col.blue.abs();
                        assert!(
                            error < ACCEPTABLE_ERROR,
                            "The blue is too different: \n\tInit {init_col:?}\n\tBack {return_col:?}\n\tError: {}%",
                            error * 100.0
                        );
                    } else {
                        let ratio = (return_col.blue
                            / init_col
                                .red
                                .max(init_col.green)
                                .max(init_col.blue)
                                .max(std::f32::EPSILON as f64))
                        .abs();
                        assert!(
                            ratio <= ACCEPTABLE_ERROR,
                            "The blue should be negligible from\n\t{init_col:?}, got\n\t{return_col:?} instead",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn oklab_srgb() {
        let red_steps = STEPS;
        let green_steps = STEPS;
        let blue_steps = STEPS;
        for red_step in 0..red_steps {
            for green_step in 0..green_steps {
                for blue_step in 0..blue_steps {
                    let init_col = LinSrgb {
                        red: red_step as f64 / red_steps as f64,
                        green: green_step as f64 / green_steps as f64,
                        blue: blue_step as f64 / blue_steps as f64,
                    };
                    let return_col = LinSrgb::from(OkLab::from(init_col));
                    assert!(
                        return_col.is_normal(),
                        "return_col is not normal\n{init_col:?} became\n{return_col:?}"
                    );
                    // Comparing with f32 epsilon to allow some leeway
                    if init_col.red.abs() > std::f32::EPSILON as f64 {
                        let error = (init_col.red - return_col.red).abs() / init_col.red.abs();
                        assert!(
                            error < ACCEPTABLE_ERROR,
                            "The red is too different: \n\tInit {init_col:?}\n\tBack {return_col:?}\n\tError: {}%",
                            error * 100.0
                        );
                    } else {
                        let ratio = (return_col.red
                            / init_col
                                .red
                                .max(init_col.green)
                                .max(init_col.blue)
                                .max(std::f32::EPSILON as f64))
                        .abs();
                        assert!(
                            ratio <= ACCEPTABLE_ERROR,
                            "The red should be negligible from\n\t{init_col:?}, got\n\t{return_col:?} instead",
                        );
                    }

                    if init_col.green.abs() > std::f32::EPSILON as f64 {
                        let error =
                            (init_col.green - return_col.green).abs() / init_col.green.abs();
                        assert!(
                            error < ACCEPTABLE_ERROR,
                            "The green is too different: \n\tInit {init_col:?}\n\tBack {return_col:?}\n\tError: {}%",
                            error * 100.0
                        );
                    } else {
                        let ratio = (return_col.green
                            / init_col
                                .red
                                .max(init_col.green)
                                .max(init_col.blue)
                                .max(std::f32::EPSILON as f64))
                        .abs();
                        assert!(
                            ratio <= ACCEPTABLE_ERROR,
                            "The green should be negligible from\n\t{init_col:?}, got\n\t{return_col:?} instead",
                        );
                    }

                    if init_col.blue.abs() > std::f32::EPSILON as f64 {
                        let error = (init_col.blue - return_col.blue).abs() / init_col.blue.abs();
                        assert!(
                            error < ACCEPTABLE_ERROR,
                            "The blue is too different: \n\tInit {init_col:?}\n\tBack {return_col:?}\n\tError: {}%",
                            error * 100.0
                        );
                    } else {
                        let ratio = (return_col.blue
                            / init_col
                                .red
                                .max(init_col.green)
                                .max(init_col.blue)
                                .max(std::f32::EPSILON as f64))
                        .abs();
                        assert!(
                            ratio <= ACCEPTABLE_ERROR,
                            "The blue should be negligible from\n\t{init_col:?}, got\n\t{return_col:?} instead",
                        );
                    }
                }
            }
        }
    }
}
