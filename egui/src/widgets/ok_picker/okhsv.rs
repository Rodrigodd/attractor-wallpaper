// SPDX-FileCopyrightText: 2022 Gerry Agbobada <git@gagbo.net>
//
// SPDX-License-Identifier: GPL-3.0-only

use std::f32::consts::PI;

use egui::{color_picker::show_color, vec2, Ui};

use super::{
    color_slider_1d, color_slider_2d, color_slider_circle, color_text_okhsv_ui,
    color_text_rgb_hex_ui, ToColor32,
};
use oklab::{OkHsv, OkLch, Srgb};
//// Shows a color picker where the user can change the given [`OkHsv`] color.
///
/// Returns `true` on change.
pub fn color_picker_2d(ui: &mut Ui, oklch: &mut OkLch) -> bool {
    let old_color = Srgb::from(*oklch);
    let mut new_oklch = *oklch;

    color_picker_2d_impl(ui, &mut new_oklch);

    let new_color = Srgb::from(new_oklch);
    let sq_distance = (old_color.r - new_color.r).powi(2)
        + (old_color.g - new_color.g).powi(2)
        + (old_color.b - new_color.b).powi(2);
    let sq_norm = old_color.r.powi(2) + old_color.g.powi(2) + old_color.b.powi(2);

    if sq_norm.is_normal() && sq_distance / sq_norm < 0.001 {
        false
    } else {
        *oklch = new_oklch;
        true
    }
}

//// Shows a color picker where the user can change the given [`OkHsv`] color.
///
/// Returns `true` on change.
pub fn color_picker_circle(ui: &mut Ui, current_color: &mut Srgb) -> bool {
    let mut new_okhsv = OkHsv::from(*current_color);

    color_picker_circle_impl(ui, &mut new_okhsv);

    let new_color = Srgb::from(new_okhsv);
    let sq_distance = (current_color.r - new_color.r).powi(2)
        + (current_color.g - new_color.g).powi(2)
        + (current_color.b - new_color.b).powi(2);
    let sq_norm = current_color.r.powi(2) + current_color.g.powi(2) + current_color.b.powi(2);

    if sq_norm.is_normal() && sq_distance / sq_norm < 0.001 {
        false
    } else {
        *current_color = new_color;
        true
    }
}

fn color_picker_2d_impl(ui: &mut Ui, oklch: &mut OkLch) {
    let current_color_size = vec2(
        2.0 * ui.spacing().slider_width,
        2.0 * ui.spacing().interact_size.y,
    );
    show_color(ui, oklch.to_color32(), current_color_size).on_hover_text("Selected color");

    // color_text_okhsv_ui(ui, *okhsv);
    color_text_rgb_hex_ui(ui, *oklch);

    let mut hue = oklch.h;
    let old_hue = hue;

    color_slider_1d(ui, &mut hue, 0.0, 1.0, |hue| {
        OkHsv {
            h: hue,
            s: 1.0,
            v: 1.0,
        }
        .to_color32()
    })
    .on_hover_text("Hue fully saturated");
    // color_slider_1d(ui, hue, -PI, PI, |hue| OkHsv { hue, ..current }).on_hover_text("Hue");

    let OkHsv {
        s: mut saturation,
        v: mut value,
        h,
    } = OkHsv::from(OkLch { h: hue, ..*oklch });

    if saturation > 1.0 {
        saturation = 1.0;
    }
    if value > 1.0 {
        value = 1.0;
    }

    let old_saturation = saturation;
    let old_value = value;

    if true {
        color_slider_1d(ui, &mut saturation, 0.0, 1.0, move |saturation| {
            OkHsv {
                s: saturation,
                v: value,
                h,
            }
            .to_color32()
        })
        .on_hover_text("Saturation");
    }

    if true {
        color_slider_1d(ui, &mut value, 0.0, 1.0, |value| {
            OkHsv {
                s: saturation,
                v: value,
                h,
            }
            .to_color32()
        })
        .on_hover_text("Value");
    }

    color_slider_2d(ui, &mut saturation, &mut value, |saturation, value| {
        OkHsv {
            s: saturation,
            v: value,
            h,
        }
        .to_color32()
    });

    // only update saturation and value when changed directly, otherwise keep it, even if outside
    // of range. This is used for keeping lighness and chroma constant while dragging hue.
    if saturation != old_saturation || value != old_value {
        *oklch = OkLch::from(OkHsv {
            h: hue,
            s: saturation,
            v: value,
        });
    }

    if hue != old_hue {
        oklch.h = hue;
    }
}

fn color_picker_circle_impl(ui: &mut Ui, okhsv: &mut OkHsv) {
    let current_color_size = vec2(
        2.0 * ui.spacing().slider_width,
        2.0 * ui.spacing().interact_size.y,
    );
    show_color(ui, okhsv.to_color32(), current_color_size).on_hover_text("Selected color");

    color_text_okhsv_ui(ui, *okhsv);
    color_text_rgb_hex_ui(ui, *okhsv);

    let current = *okhsv;

    let OkHsv {
        h: hue,
        s: saturation,
        v: value,
    } = okhsv;

    color_slider_1d(ui, hue, -PI, PI, |hue| {
        OkHsv { h: hue, ..current }.to_color32()
    })
    .on_hover_text("Hue");

    color_slider_circle(ui, saturation, hue, |saturation, hue| {
        OkHsv {
            h: hue,
            s: saturation,
            ..current
        }
        .to_color32()
    });

    if true {
        color_slider_1d(ui, value, 0.0, 1.0, |value| {
            OkHsv {
                v: value,
                ..current
            }
            .to_color32()
        })
        .on_hover_text("Value");
    }

    if true {
        color_slider_1d(ui, saturation, 0.0, 1.0, |saturation| {
            OkHsv {
                s: saturation,
                ..current
            }
            .to_color32()
        })
        .on_hover_text("Saturation");
    }
}
