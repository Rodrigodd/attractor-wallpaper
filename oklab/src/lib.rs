#![warn(missing_docs)]

//! Rust implemention of the Oklab color space, as described by [Björn Ottosson in his blog
//! post](https://bottosson.github.io/posts/oklab/). This is a color space designed to be
//! perceptually uniform, meaning that the same amount of distance between two colors is perceived
//! as the same ammount of difference in color. Useful for picking colors, building palettes and
//! rendering gradients.

use std::ops::{Add, Mul};

pub mod ok_color;

/// Represents a color in the sRGB color space.
///
/// This color space is gamma-corrected from a linear RGB color space, so it is not suitable to be
/// interpolated.
///
/// The represented color may not be in the sRGB gamut, being outside of the range [0.0, 1.0].
/// This can happen when converting from other color spaces, or when performing operations on
/// colors. This can be fixed by calling the clip() function.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Srgb {
    /// Red component, with gamut in the range [0.0, 1.0].
    pub r: f32,
    /// Green component, with gamut in the range [0.0, 1.0].
    pub g: f32,
    /// Blue component, with gamut in the range [0.0, 1.0].
    pub b: f32,
}
impl Srgb {
    /// Create a new color with the given red, green, blue components.
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    /// Convert to linear sRGB color space.
    ///
    /// Shortcut for LinSrgb::from(self).
    pub fn to_linear(self) -> LinSrgb {
        LinSrgb::from(self)
    }

    /// Convert to sRGB color space with 8 bits per channel.
    ///
    /// Shortcut for Srgb8::from(self). The color will be clipped to the sRGB gamut, as described
    /// in [Self::clip].
    pub fn to_srgb8(self) -> Srgb8 {
        Srgb8::from(self)
    }

    /// Clip the color to the sRGB gamut, preserving lightness and hue.
    ///
    /// Color conversions may produce colors that are outside of the sRGB gamut. This function
    /// clips the color to the sRGB gamut, while preserving lightness and hue, and projecting
    /// chroma into the sRGB gamut. Lightness outside of the sRGB gamut will be clipped to 0.0 or
    /// 1.0.
    pub fn clip(self) -> Srgb {
        if self.r > 0.0
            && self.r < 1.0
            && self.g > 0.0
            && self.g < 1.0
            && self.b > 0.0
            && self.b < 1.0
        {
            return self;
        }
        self.to_linear().clip().to_srgb()
    }
}

/// Represents a color in the linear sRGB color space.
///
/// The same color model as the sRGB color space, but the values are linear in respect to light
/// intensity. This makes colors interpolation correspond to the physical mixing of light.
///
/// The represented color may not be in the sRGB gamut, being outside of the range [0.0, 1.0].
/// This can happen when converting from other color spaces, or when performing operations on
/// colors. This can be fixed by calling the clip() function.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LinSrgb {
    /// Red component, normally in the range [0.0, 1.0].
    pub r: f32,
    /// Green component, normally in the range [0.0, 1.0].
    pub g: f32,
    /// Blue component, normally in the range [0.0, 1.0].
    pub b: f32,
}
impl LinSrgb {
    /// Create a new color with the given red, green, blue components.
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    /// Convert to sRGB color space.
    ///
    /// Shortcut for Srgb::from(self).
    pub fn to_srgb(self) -> Srgb {
        Srgb::from(self)
    }

    /// Clip the color to the sRGB gamut, preserving lightness and hue.
    ///
    /// Color conversions may produce colors that are outside of the sRGB gamut. This function
    /// clips the color to the sRGB gamut, while preserving lightness and hue, and projecting
    /// chroma into the sRGB gamut. Lightness outside of the sRGB gamut will be clipped to 0.0 or
    /// 1.0.
    pub fn clip(self) -> LinSrgb {
        ok_color::gamut_clip_preserve_chroma(self)
    }
}

/// Represents a color in the Oklab color space.
///
/// Colors in this color space are perceptually uniform, meaning that the same amount of distance
/// between two colors is perceived as the same amount of difference in lightness. This makes
/// interpolations and gradients in this color space perceptually uniform.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Oklab {
    /// Perceived lightness.
    pub l: f32,
    /// Green-red component, normally around the range [-1.0, 1.0].
    pub a: f32,
    /// Blue-yellow component, normally around the range [-1.0, 1.0].
    pub b: f32,
}
impl Oklab {
    /// Create a new color with the given lightness, green-red and blue-yellow components.
    pub fn new(l: f32, a: f32, b: f32) -> Self {
        Self { l, a, b }
    }
}

/// Represents a color in the Oklch color space.
///
/// This is a transformation of the Oklab color space, where the chromaticityies `a` and `b` are
/// represented in polar coordinates, as chroma and hue.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OkLch {
    /// Perceived lightness.
    pub l: f32,
    /// Chroma, normally in the range [0.0, 1.0].
    pub c: f32,
    /// Hue, normally in the range [0.0, 1.0].
    pub h: f32,
}
impl OkLch {
    /// Create a new color with the given lightness, chroma and hue.
    pub fn new(l: f32, c: f32, h: f32) -> Self {
        Self { l, c, h }
    }
}

/// Represents a color in the okHsv color space.
///
/// This color space is similar to the HSV color space, but the with a perceptually uniform
/// saturation and value, based on the Oklab/OkLCh color space.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OkHsv {
    /// Hue, as in the OkLCh color space, in the range [0.0, 1.0].
    pub h: f32,
    /// Saturation, normally in the range [0.0, 1.0].
    pub s: f32,
    /// Value, normally in the range [0.0, 1.0].
    pub v: f32,
}

/// Represents a color in the sRGB color space, with 8 bits per channel.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Srgb8 {
    /// Red component, in the range [0, 255]
    pub r: u8,
    /// Green component, with gamut in the range [0, 255]
    pub g: u8,
    /// Blue component, with gamut in the range [0, 255]
    pub b: u8,
}
impl Srgb8 {
    /// Convert to equivalent `Srgb` color.
    pub fn to_f32(self) -> Srgb {
        Srgb::new(
            self.r as f32 / 255.0,
            self.g as f32 / 255.0,
            self.b as f32 / 255.0,
        )
    }
}
impl From<[u8; 3]> for Srgb8 {
    fn from(rgb: [u8; 3]) -> Self {
        Srgb8 {
            r: rgb[0],
            g: rgb[1],
            b: rgb[2],
        }
    }
}
impl From<Srgb> for Srgb8 {
    fn from(srgb: Srgb) -> Self {
        let srgb = srgb.clip();
        Srgb8 {
            r: (srgb.r * 255.0 + 0.5) as u8,
            g: (srgb.g * 255.0 + 0.5) as u8,
            b: (srgb.b * 255.0 + 0.5) as u8,
        }
    }
}

/// Represents a color in the okHsl color space.
///
/// This color space is similar to the HSL color space, but the with a perceptually uniform
/// saturation and lightness, based on the Oklab/OkLCh color space.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OkHsl {
    /// Hue, as in the OkLCh color space, in the range [0.0, 1.0]
    pub h: f32,
    /// Saturation, normally in the range [0.0, 1.0].
    pub s: f32,
    /// Lightness, normally in the range [0.0, 1.0], where 1.0 is white.
    pub l: f32,
}

impl From<LinSrgb> for Srgb {
    fn from(lin_srgb: LinSrgb) -> Self {
        ok_color::linear_srgb_to_srgb(lin_srgb)
    }
}
impl From<Oklab> for Srgb {
    fn from(oklab: Oklab) -> Self {
        let lin_srgb = ok_color::oklab_to_linear_srgb(oklab);
        ok_color::linear_srgb_to_srgb(lin_srgb)
    }
}
impl From<OkLch> for Srgb {
    fn from(okclh: OkLch) -> Self {
        let oklab = ok_color::oklch_to_oklab(okclh);
        let lin_srgb = ok_color::oklab_to_linear_srgb(oklab);
        ok_color::linear_srgb_to_srgb(lin_srgb)
    }
}
impl From<OkHsl> for Srgb {
    fn from(okhsl: OkHsl) -> Self {
        ok_color::okhsl_to_srgb(okhsl)
    }
}
impl From<OkHsv> for Srgb {
    fn from(okhsv: OkHsv) -> Self {
        ok_color::okhsv_to_srgb(okhsv)
    }
}

impl From<Srgb> for LinSrgb {
    fn from(srgb: Srgb) -> Self {
        ok_color::srgb_to_linear_srgb(srgb)
    }
}
impl From<Oklab> for LinSrgb {
    fn from(oklab: Oklab) -> Self {
        ok_color::oklab_to_linear_srgb(oklab)
    }
}
impl From<OkLch> for LinSrgb {
    fn from(okclh: OkLch) -> Self {
        let oklab = ok_color::oklch_to_oklab(okclh);
        ok_color::oklab_to_linear_srgb(oklab)
    }
}
impl From<OkHsl> for LinSrgb {
    fn from(okhsl: OkHsl) -> Self {
        let srgb = ok_color::okhsl_to_srgb(okhsl);
        ok_color::srgb_to_linear_srgb(srgb)
    }
}
impl From<OkHsv> for LinSrgb {
    fn from(okhsv: OkHsv) -> Self {
        let srgb = ok_color::okhsv_to_srgb(okhsv);
        ok_color::srgb_to_linear_srgb(srgb)
    }
}

impl From<Srgb> for Oklab {
    fn from(srgb: Srgb) -> Self {
        let srgb = ok_color::srgb_to_linear_srgb(srgb);
        ok_color::linear_srgb_to_oklab(srgb)
    }
}
impl From<LinSrgb> for Oklab {
    fn from(lin_srgb: LinSrgb) -> Self {
        ok_color::linear_srgb_to_oklab(lin_srgb)
    }
}
impl From<OkLch> for Oklab {
    fn from(okclh: OkLch) -> Self {
        ok_color::oklch_to_oklab(okclh)
    }
}
impl From<OkHsl> for Oklab {
    fn from(okhsl: OkHsl) -> Self {
        let srgb = ok_color::okhsl_to_srgb(okhsl);
        let lin_srgb = ok_color::srgb_to_linear_srgb(srgb);
        ok_color::linear_srgb_to_oklab(lin_srgb)
    }
}
impl From<OkHsv> for Oklab {
    fn from(okhsv: OkHsv) -> Self {
        let srgb = ok_color::okhsv_to_srgb(okhsv);
        let lin_srgb = ok_color::srgb_to_linear_srgb(srgb);
        ok_color::linear_srgb_to_oklab(lin_srgb)
    }
}

impl From<Srgb> for OkLch {
    fn from(srgb: Srgb) -> Self {
        let lin_srgb = ok_color::srgb_to_linear_srgb(srgb);
        let oklab = ok_color::linear_srgb_to_oklab(lin_srgb);
        ok_color::oklab_to_oklch(oklab)
    }
}
impl From<LinSrgb> for OkLch {
    fn from(lin_srgb: LinSrgb) -> Self {
        let oklab = ok_color::linear_srgb_to_oklab(lin_srgb);
        ok_color::oklab_to_oklch(oklab)
    }
}
impl From<Oklab> for OkLch {
    fn from(oklab: Oklab) -> Self {
        ok_color::oklab_to_oklch(oklab)
    }
}
impl From<OkHsl> for OkLch {
    fn from(okhsl: OkHsl) -> Self {
        let oklab = ok_color::okhsl_to_oklab(okhsl);
        ok_color::oklab_to_oklch(oklab)
    }
}
impl From<OkHsv> for OkLch {
    fn from(okhsv: OkHsv) -> Self {
        let oklab = ok_color::okhsv_to_oklab(okhsv);
        ok_color::oklab_to_oklch(oklab)
    }
}

impl From<Srgb> for OkHsl {
    fn from(srgb: Srgb) -> Self {
        ok_color::srgb_to_okhsl(srgb)
    }
}
impl From<LinSrgb> for OkHsl {
    fn from(lin_srgb: LinSrgb) -> Self {
        let srgb = ok_color::linear_srgb_to_srgb(lin_srgb);
        ok_color::srgb_to_okhsl(srgb)
    }
}
impl From<Oklab> for OkHsl {
    fn from(oklab: Oklab) -> Self {
        let lin_srgb = ok_color::oklab_to_linear_srgb(oklab);
        let srgb = ok_color::linear_srgb_to_srgb(lin_srgb);
        ok_color::srgb_to_okhsl(srgb)
    }
}
impl From<OkLch> for OkHsl {
    fn from(okclh: OkLch) -> Self {
        let oklab = ok_color::oklch_to_oklab(okclh);
        ok_color::oklab_to_okhsl(oklab)
    }
}
impl From<OkHsv> for OkHsl {
    fn from(okhsv: OkHsv) -> Self {
        let srgb = ok_color::okhsv_to_srgb(okhsv);
        ok_color::srgb_to_okhsl(srgb)
    }
}

impl From<Srgb> for OkHsv {
    fn from(srgb: Srgb) -> Self {
        ok_color::srgb_to_okhsv(srgb)
    }
}
impl From<LinSrgb> for OkHsv {
    fn from(lin_srgb: LinSrgb) -> Self {
        let srgb = ok_color::linear_srgb_to_srgb(lin_srgb);
        ok_color::srgb_to_okhsv(srgb)
    }
}
impl From<Oklab> for OkHsv {
    fn from(oklab: Oklab) -> Self {
        let lin_srgb = ok_color::oklab_to_linear_srgb(oklab);
        let srgb = ok_color::linear_srgb_to_srgb(lin_srgb);
        ok_color::srgb_to_okhsv(srgb)
    }
}
impl From<OkLch> for OkHsv {
    fn from(okclh: OkLch) -> Self {
        let oklab = ok_color::oklch_to_oklab(okclh);
        ok_color::oklab_to_okhsv(oklab)
    }
}
impl From<OkHsl> for OkHsv {
    fn from(okhsl: OkHsl) -> Self {
        let srgb = ok_color::okhsl_to_srgb(okhsl);
        ok_color::srgb_to_okhsv(srgb)
    }
}

// Linear operations

impl Add for Srgb {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Srgb {
            r: self.r + rhs.r,
            g: self.g + rhs.g,
            b: self.b + rhs.b,
        }
    }
}
impl Add for LinSrgb {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        LinSrgb {
            r: self.r + rhs.r,
            g: self.g + rhs.g,
            b: self.b + rhs.b,
        }
    }
}
impl Add for Oklab {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Oklab {
            l: self.l + rhs.l,
            a: self.a + rhs.a,
            b: self.b + rhs.b,
        }
    }
}
impl Add for OkLch {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        OkLch {
            l: self.l + rhs.l,
            c: self.c + rhs.c,
            h: self.h + rhs.h,
        }
    }
}
impl Add for OkHsl {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        OkHsl {
            h: self.h + rhs.h,
            s: self.s + rhs.s,
            l: self.l + rhs.l,
        }
    }
}
impl Add for OkHsv {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        OkHsv {
            h: self.h + rhs.h,
            s: self.s + rhs.s,
            v: self.v + rhs.v,
        }
    }
}

impl Mul<f32> for Srgb {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output {
        Srgb {
            r: self.r * rhs,
            g: self.g * rhs,
            b: self.b * rhs,
        }
    }
}
impl Mul<f32> for LinSrgb {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output {
        LinSrgb {
            r: self.r * rhs,
            g: self.g * rhs,
            b: self.b * rhs,
        }
    }
}
impl Mul<f32> for Oklab {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output {
        Oklab {
            l: self.l * rhs,
            a: self.a * rhs,
            b: self.b * rhs,
        }
    }
}
impl Mul<f32> for OkLch {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output {
        OkLch {
            l: self.l * rhs,
            c: self.c * rhs,
            h: self.h,
        }
    }
}
impl Mul<f32> for OkHsl {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output {
        OkHsl {
            h: self.h,
            s: self.s * rhs,
            l: self.l * rhs,
        }
    }
}
impl Mul<f32> for OkHsv {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output {
        OkHsv {
            h: self.h,
            s: self.s * rhs,
            v: self.v * rhs,
        }
    }
}

// Array conversions

impl From<[f32; 3]> for Srgb {
    fn from([r, g, b]: [f32; 3]) -> Self {
        Srgb { r, g, b }
    }
}
impl From<[f32; 3]> for LinSrgb {
    fn from([r, g, b]: [f32; 3]) -> Self {
        LinSrgb { r, g, b }
    }
}
impl From<[f32; 3]> for Oklab {
    fn from([l, a, b]: [f32; 3]) -> Self {
        Oklab { l, a, b }
    }
}
impl From<[f32; 3]> for OkLch {
    fn from([l, c, h]: [f32; 3]) -> Self {
        OkLch { l, c, h }
    }
}
impl From<[f32; 3]> for OkHsl {
    fn from([h, s, l]: [f32; 3]) -> Self {
        OkHsl { h, s, l }
    }
}
impl From<[f32; 3]> for OkHsv {
    fn from([h, s, v]: [f32; 3]) -> Self {
        OkHsv { h, s, v }
    }
}

impl From<Srgb> for [f32; 3] {
    fn from(Srgb { r, g, b }: Srgb) -> Self {
        [r, g, b]
    }
}
impl From<LinSrgb> for [f32; 3] {
    fn from(LinSrgb { r, g, b }: LinSrgb) -> Self {
        [r, g, b]
    }
}
impl From<Oklab> for [f32; 3] {
    fn from(Oklab { l, a, b }: Oklab) -> Self {
        [l, a, b]
    }
}
impl From<OkLch> for [f32; 3] {
    fn from(OkLch { l, c, h }: OkLch) -> Self {
        [l, c, h]
    }
}
impl From<OkHsl> for [f32; 3] {
    fn from(OkHsl { h, s, l }: OkHsl) -> Self {
        [h, s, l]
    }
}
impl From<OkHsv> for [f32; 3] {
    fn from(OkHsv { h, s, v }: OkHsv) -> Self {
        [h, s, v]
    }
}

#[cfg(test)]
mod test {
    use rand::Rng;

    #[test]
    fn clip() {
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let x = super::Srgb::new(rng.gen(), rng.gen(), rng.gen());
            assert_eq!(x, x.clip());
        }
    }
}
