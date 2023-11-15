// SPDX-FileCopyrightText: 2022 Gerry Agbobada <git@gagbo.net>
//
// SPDX-License-Identifier: GPL-3.0-only

use egui::{
    epaint, lerp, pos2, remap_clamp, vec2, Align, Color32, Layout, Mesh, Painter, Rect, Response,
    Rgba, Sense, Shape, Stroke, Ui, Vec2,
};

use oklab::{OkHsl, OkHsv, Srgb};

pub mod okhsl;
pub mod okhsv;

pub trait ToColor32 {
    fn to_color32(self) -> Color32;
}
impl<T: Into<Srgb> + Clone> ToColor32 for T {
    fn to_color32(self) -> Color32 {
        let srgb: Srgb = self.into();
        let srgb8 = srgb.to_srgb8();
        Color32::from_rgb(srgb8.r, srgb8.g, srgb8.b)
    }
}

/// Number of vertices per dimension in the color sliders.
/// We need at least 6 for hues, and more for smooth 2D areas.
/// Should always be a multiple of 6 to hit the peak hues in HSV/HSL (every 60°).
const N: u32 = 6 * 6;

fn color_text_rgb_dec_ui(ui: &mut Ui, color: impl Into<Srgb>) {
    let color = color.into();
    ui.allocate_ui_with_layout(
        vec2(
            2.0 * ui.spacing().slider_width,
            ui.spacing().interact_size.y,
        ),
        Layout::left_to_right(Align::Center),
        |ui| {
            let Srgb { r, g, b } = color;

            let r = (256.0 * r).floor() as u8;
            let g = (256.0 * g).floor() as u8;
            let b = (256.0 * b).floor() as u8;

            // if ui.button("📋").on_hover_text("Click to copy").clicked() {
            //     ui.output().copied_text = format!("{}, {}, {}", r, g, b);
            // }

            ui.label(format!("rgb({}, {}, {})", r, g, b))
                .on_hover_text("Red Green Blue");
        },
    );
}

fn color_text_rgb_hex_ui(ui: &mut Ui, color: impl Into<Srgb>) {
    let color = color.into();
    ui.allocate_ui_with_layout(
        vec2(
            2.0 * ui.spacing().slider_width,
            ui.spacing().interact_size.y,
        ),
        Layout::left_to_right(Align::Center),
        |ui| {
            let Srgb { r, g, b } = color;

            let r = (256.0 * r).floor() as u8;
            let g = (256.0 * g).floor() as u8;
            let b = (256.0 * b).floor() as u8;

            if ui.button("📋").on_hover_text("Click to copy").clicked() {
                ui.output_mut(|x| x.copied_text = format!("#{:02X}{:02X}{:02X}", r, g, b));
            }

            ui.label(format!("#{:02X}{:02X}{:02X}", r, g, b))
                .on_hover_text("Red Green Blue, Hex");
        },
    );
}
fn color_text_okhsv_ui(ui: &mut Ui, color: impl Into<OkHsv>) {
    let hsv = color.into();
    ui.allocate_ui_with_layout(
        vec2(
            2.0 * ui.spacing().slider_width,
            ui.spacing().interact_size.y,
        ),
        Layout::left_to_right(Align::Center),
        |ui| {
            // if ui.button("📋").on_hover_text("Click to copy").clicked() {
            //     ui.output().copied_text = format!("{}, {}, {}", hsv.hue, hsv.saturation, hsv.value);
            // }

            // Approx 512 even steps for the rounding
            let trunc = 1.0 / 2.0_f32.powi(8);

            ui.label(format!(
                "okhsv({}, {}, {})",
                trunc * (hsv.h / trunc).trunc(),
                trunc * (hsv.s / trunc).trunc(),
                trunc * (hsv.v / trunc).trunc()
            ))
            .on_hover_text("Hue Saturation Value, OkHSV");
        },
    );
}

fn color_text_okhsl_ui(ui: &mut Ui, color: impl Into<OkHsl>) {
    let hsl = color.into();
    ui.allocate_ui_with_layout(
        vec2(
            2.0 * ui.spacing().slider_width,
            ui.spacing().interact_size.y,
        ),
        Layout::left_to_right(Align::Center),
        |ui| {
            // if ui.button("📋").on_hover_text("Click to copy").clicked() {
            //     ui.output().copied_text =
            //         format!("{}, {}, {}", hsl.hue, hsl.saturation, hsl.lightness);
            // }

            // Approx 512 even steps for the rounding
            let trunc = 1.0 / 2.0_f32.powi(8);

            ui.label(format!(
                "okhsl({}, {}, {})",
                trunc * (hsl.h / trunc).trunc(),
                trunc * (hsl.s / trunc).trunc(),
                trunc * (hsl.l / trunc).trunc()
            ))
            .on_hover_text("Hue Saturation Lightness, OkHSL");
        },
    );
}

pub fn color_text_ui(ui: &mut Ui, color: impl Into<Srgb>) {
    let color = color.into();
    color_text_okhsl_ui(ui, color);
    color_text_okhsv_ui(ui, color);
    color_text_rgb_dec_ui(ui, color);
    color_text_rgb_hex_ui(ui, color);
}

fn color_slider_1d(
    ui: &mut Ui,
    value: &mut f32,
    min: f32,
    max: f32,
    color_at: impl Fn(f32) -> Color32,
) -> Response {
    #![allow(clippy::identity_op)]

    let span = max - min;

    let desired_size = vec2(
        2.0 * ui.spacing().slider_width,
        ui.spacing().interact_size.y,
    );
    let (rect, response) = ui.allocate_at_least(desired_size, Sense::click_and_drag());

    if let Some(mpos) = response.interact_pointer_pos() {
        *value = min + span * remap_clamp(mpos.x, rect.left()..=rect.right(), 0.0..=1.0);
    }

    if ui.is_rect_visible(rect) {
        let visuals = ui.style().interact(&response);

        background_checkers(ui.painter(), rect); // for alpha:

        {
            // fill color:
            let mut mesh = Mesh::default();
            for i in 0..=N {
                let t = min + (i as f32 * span / (N as f32));

                let color = color_at(t);

                let x = lerp(rect.left()..=rect.right(), (t - min) / span);

                mesh.colored_vertex(pos2(x, rect.top()), color);
                mesh.colored_vertex(pos2(x, rect.bottom()), color);
                if i < N {
                    mesh.add_triangle(2 * i + 0, 2 * i + 1, 2 * i + 2);
                    mesh.add_triangle(2 * i + 1, 2 * i + 2, 2 * i + 3);
                }
            }
            ui.painter().add(Shape::mesh(mesh));
        }

        ui.painter().rect_stroke(rect, 0.0, visuals.bg_stroke); // outline

        {
            // Show where the slider is at:
            let x = lerp(rect.left()..=rect.right(), (*value - min) / span);
            let r = rect.height() / 4.0;
            let picked_color = color_at(*value);
            ui.painter().add(Shape::convex_polygon(
                vec![
                    pos2(x, rect.center().y),   // tip
                    pos2(x + r, rect.bottom()), // right bottom
                    pos2(x - r, rect.bottom()), // left bottom
                ],
                picked_color,
                Stroke::new(visuals.fg_stroke.width, contrast_color(picked_color)),
            ));
        }
    }

    response
}

fn color_slider_vertical_1d(
    ui: &mut Ui,
    value: &mut f32,
    min: f32,
    max: f32,
    color_at: impl Fn(f32) -> Color32,
) -> Response {
    #![allow(clippy::identity_op)]

    let span = max - min;

    let desired_size = vec2(
        ui.spacing().interact_size.y,
        2.0 * ui.spacing().slider_width,
    );
    let (rect, response) = ui.allocate_at_least(desired_size, Sense::click_and_drag());

    if let Some(mpos) = response.interact_pointer_pos() {
        *value = min + span * remap_clamp(mpos.y, rect.top()..=rect.bottom(), 0.0..=1.0);
    }

    if ui.is_rect_visible(rect) {
        let visuals = ui.style().interact(&response);

        background_checkers(ui.painter(), rect); // for alpha:

        {
            // fill color:
            let mut mesh = Mesh::default();
            for i in 0..=N {
                let t = min + (i as f32 * span / (N as f32));
                let color = color_at(t);
                let y = lerp(rect.top()..=rect.bottom(), (t - min) / span);
                mesh.colored_vertex(pos2(rect.left(), y), color);
                mesh.colored_vertex(pos2(rect.right(), y), color);
                if i < N {
                    mesh.add_triangle(2 * i + 0, 2 * i + 1, 2 * i + 2);
                    mesh.add_triangle(2 * i + 1, 2 * i + 2, 2 * i + 3);
                }
            }
            ui.painter().add(Shape::mesh(mesh));
        }

        ui.painter().rect_stroke(rect, 0.0, visuals.bg_stroke); // outline

        {
            // Show where the slider is at:
            let y = lerp(rect.top()..=rect.bottom(), (*value - min) / span);
            let r = rect.width() / 4.0;
            let picked_color = color_at(*value);
            ui.painter().add(Shape::convex_polygon(
                vec![
                    pos2(rect.center().x, y), // tip
                    pos2(rect.left(), y + r), // left bottom
                    pos2(rect.left(), y - r), // left top
                ],
                picked_color,
                Stroke::new(visuals.fg_stroke.width, contrast_color(picked_color)),
            ));
        }
    }

    response
}

fn color_slider_2d<T>(
    ui: &mut Ui,
    x_value: &mut f32,
    y_value: &mut f32,
    color_at: impl Fn(f32, f32) -> T,
) -> Response
where
    T: Into<Color32> + Copy,
    egui::Rgba: std::convert::From<T>,
{
    let desired_size = Vec2::splat(2.0 * ui.spacing().slider_width);
    let (rect, response) = ui.allocate_at_least(desired_size, Sense::click_and_drag());

    if let Some(mpos) = response.interact_pointer_pos() {
        *x_value = remap_clamp(mpos.x, rect.left()..=rect.right(), 0.0..=1.0);
        *y_value = remap_clamp(mpos.y, rect.bottom()..=rect.top(), 0.0..=1.0);
    }

    if ui.is_rect_visible(rect) {
        let visuals = ui.style().interact(&response);
        let mut mesh = Mesh::default();

        for xi in 0..=N {
            for yi in 0..=N {
                let xt = xi as f32 / (N as f32);
                let yt = yi as f32 / (N as f32);
                let color = color_at(xt, yt);
                let x = lerp(rect.left()..=rect.right(), xt);
                let y = lerp(rect.bottom()..=rect.top(), yt);
                mesh.colored_vertex(pos2(x, y), color.into());

                if xi < N && yi < N {
                    let x_offset = 1;
                    let y_offset = N + 1;
                    let tl = yi * y_offset + xi;
                    mesh.add_triangle(tl, tl + x_offset, tl + y_offset);
                    mesh.add_triangle(tl + x_offset, tl + y_offset, tl + y_offset + x_offset);
                }
            }
        }
        ui.painter().add(Shape::mesh(mesh)); // fill

        ui.painter().rect_stroke(rect, 0.0, visuals.bg_stroke); // outline

        // Show where the slider is at:
        let x = lerp(rect.left()..=rect.right(), *x_value);
        let y = lerp(rect.bottom()..=rect.top(), *y_value);
        let picked_color = color_at(*x_value, *y_value);
        ui.painter().add(epaint::CircleShape {
            center: pos2(x, y),
            radius: rect.width() / 12.0,
            fill: picked_color.into(),
            stroke: Stroke::new(visuals.fg_stroke.width, contrast_color(picked_color)),
        });
    }

    response
}

pub fn color_slider_circle<T>(
    ui: &mut Ui,
    r: &mut f32,
    angle: &mut f32,
    color_at: impl Fn(f32, f32) -> T,
) -> Response
where
    T: Into<Color32> + Copy,
    egui::Rgba: std::convert::From<T>,
{
    let desired_size = Vec2::splat(2.0 * ui.spacing().slider_width);
    let (rect, response) = ui.allocate_at_least(desired_size, Sense::click_and_drag());
    let r_max = rect.width().min(rect.height()) / 2.0;

    if let Some(mpos) = response.interact_pointer_pos() {
        let current_pos = mpos - rect.center();
        let current_r = current_pos.length();
        *r = remap_clamp(current_r, 0.0..=r_max, 0.0..=1.0);
        // y goes down, so we flip the angle to get the
        // trigonometry normal direction
        *angle = -1.0 * current_pos.angle();
    }

    if ui.is_rect_visible(rect) {
        let visuals = ui.style().interact(&response);
        let mut mesh = Mesh::default();

        for ri in 0..=N {
            for anglei in 0..=N {
                let rt = ri as f32 / (N as f32);
                let anglet = 2.0 * std::f32::consts::PI * anglei as f32 / (N as f32);
                let color = color_at(rt, anglet);
                let (x_norm, y_norm) = (
                    (rt * anglet.cos() + 1.0) / 2.0,
                    (rt * anglet.sin() + 1.0) / 2.0,
                );
                let x = lerp(rect.left()..=rect.right(), x_norm);
                let y = lerp(rect.bottom()..=rect.top(), y_norm);
                mesh.colored_vertex(pos2(x, y), color.into());

                if ri < N && anglei < N {
                    let r_offset = 1;
                    let angle_offset = N + 1;
                    let tl = anglei * angle_offset + ri;
                    mesh.add_triangle(tl, tl + r_offset, tl + angle_offset);
                    mesh.add_triangle(
                        tl + r_offset,
                        tl + angle_offset,
                        tl + angle_offset + r_offset,
                    );
                }
            }
        }
        ui.painter().add(Shape::mesh(mesh)); // fill

        ui.painter().rect_stroke(rect, 0.0, visuals.bg_stroke); // outline

        // Show where the slider is at:
        // let actual_r = lerp(0.0..=r_max.into(), *r);
        let (x_norm, y_norm) = (
            (*r * angle.cos() + 1.0) / 2.0,
            (*r * angle.sin() + 1.0) / 2.0,
        );
        let x = lerp(rect.left()..=rect.right(), x_norm);
        let y = lerp(rect.bottom()..=rect.top(), y_norm);
        let picked_color = color_at(*r, *angle);
        ui.painter().add(epaint::CircleShape {
            center: pos2(x, y),
            radius: rect.width() / 12.0,
            fill: picked_color.into(),
            stroke: Stroke::new(visuals.fg_stroke.width, contrast_color(picked_color)),
        });
    }

    response
}

fn background_checkers(painter: &Painter, rect: Rect) {
    let rect = rect.shrink(0.5); // Small hack to avoid the checkers from peeking through the sides
    if !rect.is_positive() {
        return;
    }

    let dark_color = Color32::from_gray(32);
    let bright_color = Color32::from_gray(128);

    let checker_size = Vec2::splat(rect.height() / 2.0);
    let n = (rect.width() / checker_size.x).round() as u32;

    let mut mesh = Mesh::default();
    mesh.add_colored_rect(rect, dark_color);

    let mut top = true;
    for i in 0..n {
        let x = lerp(rect.left()..=rect.right(), i as f32 / (n as f32));
        let small_rect = if top {
            Rect::from_min_size(pos2(x, rect.top()), checker_size)
        } else {
            Rect::from_min_size(pos2(x, rect.center().y), checker_size)
        };
        mesh.add_colored_rect(small_rect, bright_color);
        top = !top;
    }
    painter.add(Shape::mesh(mesh));
}

fn contrast_color(color: impl Into<Rgba>) -> Color32 {
    if color.into().intensity() < 0.5 {
        Color32::WHITE
    } else {
        Color32::BLACK
    }
}
