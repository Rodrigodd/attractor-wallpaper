// SPDX-FileCopyrightText: 2022 Gerry Agbobada <git@gagbo.net>
//
// SPDX-License-Identifier: GPL-3.0-only

use std::f32::consts::PI;

use egui::Align;
use egui::{color_picker::show_color, vec2, Layout, Ui};

use super::{
    color_slider_1d, color_slider_circle, color_slider_vertical_1d, color_text_okhsl_ui,
    color_text_rgb_hex_ui, ToColor32,
};
use oklab::{OkHsl, OkHsv, Srgb};

//// Shows a color picker where the user can change the given [`OkHsl`] color.
///
/// Returns `true` on change.
pub fn color_picker_circle(ui: &mut Ui, current_color: &mut Srgb) -> bool {
    let mut new_okhsl = OkHsl::from(*current_color);

    color_picker_circle_impl(ui, &mut new_okhsl);

    let new_color = Srgb::from(new_okhsl);
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

fn color_picker_circle_impl(ui: &mut Ui, okhsl: &mut OkHsl) {
    let current_color_size = vec2(
        2.0 * ui.spacing().slider_width,
        2.0 * ui.spacing().interact_size.y,
    );
    show_color(ui, okhsl.to_color32(), current_color_size).on_hover_text("Selected color");

    color_text_okhsl_ui(ui, *okhsl);
    color_text_rgb_hex_ui(ui, *okhsl);

    let current = *okhsl;

    let OkHsl { h, s, l: lightness } = okhsl;

    ui.allocate_ui_with_layout(
        vec2(
            2.0 * ui.spacing().slider_width + ui.spacing().interact_size.y,
            2.0 * ui.spacing().slider_width,
        ),
        Layout::left_to_right(Align::Center),
        |ui| {
            color_slider_circle(ui, s, h, |saturation, hue| {
                OkHsl {
                    s: saturation,
                    h: hue,
                    ..current
                }
                .to_color32()
            });

            color_slider_vertical_1d(ui, h, -PI, PI, |hue| {
                OkHsv::from(Srgb::from(OkHsl { h: hue, ..current })).to_color32()
            })
            .on_hover_text("Hue");
        },
    );

    if true {
        color_slider_1d(ui, s, 0.0, 1.0, |saturation| {
            OkHsl {
                s: saturation,
                ..current
            }
            .to_color32()
        })
        .on_hover_text("Saturation");
    }

    if true {
        color_slider_1d(ui, lightness, 0.0, 1.0, |lightness| {
            OkHsl {
                l: lightness,
                ..current
            }
            .to_color32()
        })
        .on_hover_text("Lightness");
    }
}
