// SPDX-FileCopyrightText: 2022 Gerry Agbobada <git@gagbo.net>
//
// SPDX-License-Identifier: GPL-3.0-only

use std::f64::consts::PI;

use egui::{color_picker::show_color, vec2, Ui};

use super::{
    color_slider_1d, color_slider_2d, color_slider_circle, color_text_okhsv_ui,
    color_text_rgb_hex_ui,
};
use crate::colors::{OkHsv, Srgb};
//// Shows a color picker where the user can change the given [`OkHsv`] color.
///
/// Returns `true` on change.
pub fn color_picker_2d(ui: &mut Ui, current_color: &mut Srgb) -> bool {
    let mut new_okhsv = OkHsv::from(*current_color);

    color_picker_2d_impl(ui, &mut new_okhsv);

    let new_color = Srgb::from(new_okhsv);
    let sq_distance = (current_color.red - new_color.red).powi(2)
        + (current_color.green - new_color.green).powi(2)
        + (current_color.blue - new_color.blue).powi(2);
    let sq_norm =
        current_color.red.powi(2) + current_color.green.powi(2) + current_color.blue.powi(2);

    if sq_norm.is_normal() && sq_distance / sq_norm < 0.001 {
        false
    } else {
        *current_color = new_color;
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
    let sq_distance = (current_color.red - new_color.red).powi(2)
        + (current_color.green - new_color.green).powi(2)
        + (current_color.blue - new_color.blue).powi(2);
    let sq_norm =
        current_color.red.powi(2) + current_color.green.powi(2) + current_color.blue.powi(2);

    if sq_norm.is_normal() && sq_distance / sq_norm < 0.001 {
        false
    } else {
        *current_color = new_color;
        true
    }
}

fn color_picker_2d_impl(ui: &mut Ui, okhsv: &mut OkHsv) {
    let current_color_size = vec2(
        2.0 * ui.spacing().slider_width,
        2.0 * ui.spacing().interact_size.y,
    );
    show_color(ui, *okhsv, current_color_size).on_hover_text("Selected color");

    // color_text_okhsv_ui(ui, *okhsv);
    // color_text_rgb_hex_ui(ui, *okhsv);

    let current = *okhsv;

    let OkHsv {
        hue,
        saturation,
        value,
    } = okhsv;

    color_slider_1d(ui, hue, -PI, PI, |hue| OkHsv {
        hue,
        saturation: 1.0,
        value: 1.0,
    })
    .on_hover_text("Hue fully saturated");
    // color_slider_1d(ui, hue, -PI, PI, |hue| OkHsv { hue, ..current }).on_hover_text("Hue");

    if true {
        color_slider_1d(ui, saturation, 0.0, 1.0, |saturation| OkHsv {
            saturation,
            ..current
        })
        .on_hover_text("Saturation");
    }

    if true {
        color_slider_1d(ui, value, 0.0, 1.0, |value| OkHsv { value, ..current })
            .on_hover_text("Value");
    }

    color_slider_2d(ui, saturation, value, |saturation, value| OkHsv {
        saturation,
        value,
        ..current
    });
}

fn color_picker_circle_impl(ui: &mut Ui, okhsv: &mut OkHsv) {
    let current_color_size = vec2(
        2.0 * ui.spacing().slider_width,
        2.0 * ui.spacing().interact_size.y,
    );
    show_color(ui, *okhsv, current_color_size).on_hover_text("Selected color");

    color_text_okhsv_ui(ui, *okhsv);
    color_text_rgb_hex_ui(ui, *okhsv);

    let current = *okhsv;

    let OkHsv {
        hue,
        saturation,
        value,
    } = okhsv;

    color_slider_1d(ui, hue, -PI, PI, |hue| OkHsv { hue, ..current }).on_hover_text("Hue");

    color_slider_circle(ui, saturation, hue, |saturation, hue| OkHsv {
        hue: hue as f64,
        saturation: saturation as f64,
        ..current
    });

    if true {
        color_slider_1d(ui, value, 0.0, 1.0, |value| OkHsv { value, ..current })
            .on_hover_text("Value");
    }

    if true {
        color_slider_1d(ui, saturation, 0.0, 1.0, |saturation| OkHsv {
            saturation,
            ..current
        })
        .on_hover_text("Saturation");
    }
}
