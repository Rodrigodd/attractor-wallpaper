// SPDX-FileCopyrightText: 2022 Gerry Agbobada <git@gagbo.net>
//
// SPDX-License-Identifier: GPL-3.0-only

use egui::{Color32, Rgba};

pub mod conversions;

/// The controlling factor for accepting numerical errors in debug builds
/// preconditions/assertions, and in tests.
///
/// This is the expected precision of the computations, mostly that we bound
/// relative errors to `(100.0 * ACCEPTABLE_ERROR)%`
const ACCEPTABLE_ERROR: f64 = 0.0001;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Srgb {
    pub red: f64,
    pub green: f64,
    pub blue: f64,
}

impl Srgb {
    fn is_normal(&self) -> bool {
        self.red.is_finite() && self.green.is_finite() && self.blue.is_finite()
    }
}

impl Default for Srgb {
    fn default() -> Self {
        Self {
            red: 0.4,
            green: 0.4,
            blue: 0.7,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct LinSrgb {
    pub red: f64,
    pub green: f64,
    pub blue: f64,
}

impl LinSrgb {
    fn is_normal(&self) -> bool {
        self.red.is_finite() && self.green.is_finite() && self.blue.is_finite()
    }

    /// Gamma clipping through dumb clamping.
    ///
    /// This method should only be used for colors _really_ close to be in
    /// gamut, e.g. to fix numerical noise after conversion cycles.
    pub fn clamp(&mut self) {
        self.red = self.red.clamp(0.0, 1.0);
        self.green = self.green.clamp(0.0, 1.0);
        self.blue = self.blue.clamp(0.0, 1.0);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct OkLab {
    pub lightness: f64,
    pub a: f64,
    pub b: f64,
}

impl OkLab {
    fn is_normal(&self) -> bool {
        self.lightness.is_finite() && self.a.is_finite() && self.b.is_finite()
    }

    /// Gamma clipping through dumb clamping.
    ///
    /// This method should only be used for colors _really_ close to be in
    /// gamut, e.g. to fix numerical noise after conversion cycles.
    pub fn clamp(&mut self) {
        self.a = self.a.clamp(-1.0, 1.0);
        self.b = self.b.clamp(-1.0, 1.0);
        self.lightness = self.lightness.clamp(0.0, 1.0);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct OkLCh {
    pub lightness: f64,
    pub chroma: f64,
    pub hue: f64,
}

impl OkLCh {
    fn is_normal(&self) -> bool {
        self.lightness.is_finite() && self.chroma.is_finite() && self.hue.is_finite()
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OkHsl {
    pub hue: f64,
    pub saturation: f64,
    pub lightness: f64,
}

impl OkHsl {
    fn is_normal(&self) -> bool {
        self.lightness.is_finite() && self.saturation.is_finite() && self.hue.is_finite()
    }
}
impl Default for OkHsl {
    fn default() -> Self {
        Self {
            hue: Default::default(),
            saturation: 0.5,
            lightness: 0.5,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OkHsv {
    pub hue: f64,
    pub saturation: f64,
    pub value: f64,
}

impl OkHsv {
    fn is_normal(&self) -> bool {
        self.value.is_finite() && self.saturation.is_finite() && self.hue.is_finite()
    }
}

impl Default for OkHsv {
    fn default() -> Self {
        Self {
            hue: Default::default(),
            saturation: 0.5,
            value: 0.5,
        }
    }
}

impl From<OkHsv> for Color32 {
    fn from(hsv: OkHsv) -> Self {
        let rgb = Srgb::from(hsv);
        Self::from_rgb(
            (rgb.red * 256.0).floor() as u8,
            (rgb.green * 256.0).floor() as u8,
            (rgb.blue * 256.0).floor() as u8,
        )
    }
}

impl From<OkHsv> for Rgba {
    fn from(hsv: OkHsv) -> Self {
        let rgb = Srgb::from(hsv);
        Self::from_rgb(rgb.red as f32, rgb.green as f32, rgb.blue as f32)
    }
}

impl From<OkHsl> for Color32 {
    fn from(hsl: OkHsl) -> Self {
        let rgb = Srgb::from(hsl);
        Self::from_rgb(
            (rgb.red * 256.0).floor() as u8,
            (rgb.green * 256.0).floor() as u8,
            (rgb.blue * 256.0).floor() as u8,
        )
    }
}

impl From<OkHsl> for Rgba {
    fn from(hsl: OkHsl) -> Self {
        let rgb = Srgb::from(hsl);
        Self::from_rgb(rgb.red as f32, rgb.green as f32, rgb.blue as f32)
    }
}
