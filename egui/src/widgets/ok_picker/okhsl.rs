// SPDX-FileCopyrightText: 2022 Gerry Agbobada <git@gagbo.net>
//
// SPDX-License-Identifier: GPL-3.0-only

use std::f64::consts::PI;

use egui::Align;
use egui::{color_picker::show_color, vec2, Layout, Ui};

use super::{
    color_slider_1d, color_slider_circle, color_slider_vertical_1d, color_text_okhsl_ui,
    color_text_rgb_hex_ui,
};
use crate::colors::{OkHsl, OkHsv, Srgb};

//// Shows a color picker where the user can change the given [`OkHsl`] color.
///
/// Returns `true` on change.
pub fn color_picker_circle(ui: &mut Ui, current_color: &mut Srgb) -> bool {
    let mut new_okhsl = OkHsl::from(*current_color);

    color_picker_circle_impl(ui, &mut new_okhsl);

    let new_color = Srgb::from(new_okhsl);
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

fn color_picker_circle_impl(ui: &mut Ui, okhsl: &mut OkHsl) {
    let current_color_size = vec2(
        2.0 * ui.spacing().slider_width,
        2.0 * ui.spacing().interact_size.y,
    );
    show_color(ui, *okhsl, current_color_size).on_hover_text("Selected color");

    color_text_okhsl_ui(ui, *okhsl);
    color_text_rgb_hex_ui(ui, *okhsl);

    let current = *okhsl;

    let OkHsl {
        hue,
        saturation,
        lightness,
    } = okhsl;

    ui.allocate_ui_with_layout(
        vec2(
            2.0 * ui.spacing().slider_width + ui.spacing().interact_size.y,
            2.0 * ui.spacing().slider_width,
        ),
        Layout::left_to_right(Align::Center),
        |ui| {
            color_slider_circle(ui, saturation, hue, |saturation, hue| OkHsl {
                saturation: saturation as f64,
                hue: hue as f64,
                ..current
            });

            color_slider_vertical_1d(ui, hue, -PI, PI, |hue| {
                OkHsv::from(Srgb::from(OkHsl { hue, ..current }))
            })
            .on_hover_text("Hue");
        },
    );

    if true {
        color_slider_1d(ui, saturation, 0.0, 1.0, |saturation| {
            OkHsv::from(Srgb::from(OkHsl {
                saturation,
                ..current
            }))
        })
        .on_hover_text("Saturation");
    }

    if true {
        color_slider_1d(ui, lightness, 0.0, 1.0, |lightness| {
            OkHsv::from(Srgb::from(OkHsl {
                lightness,
                ..current
            }))
        })
        .on_hover_text("Lightness");
    }
}
