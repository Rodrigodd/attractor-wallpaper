use egui::{
    lerp, pos2, remap, remap_clamp, vec2, Color32, Id, Mesh, Response, Rgba, Sense, Shape, Stroke,
    Ui,
};
use oklab::{Oklab, Srgb};

use super::ok_picker::ToColor32;

const N: u32 = 256;

pub struct Gradient<T> {
    colors: Vec<(f32, T)>,
}
impl<T: std::ops::Mul<f32, Output = T> + std::ops::Add<Output = T>> Gradient<T> {
    pub fn new(colors: Vec<(f32, T)>) -> Self {
        Self { colors }
    }

    /// Sample the gradient at `t` using linear interpolation.
    pub fn linear_sample(&self, t: f32) -> T
    where
        T: Copy,
    {
        let right_index = self
            .colors
            .iter()
            .position(|x| x.0 > t)
            .unwrap_or(self.colors.len());

        if right_index == 0 {
            return self.colors[0].1;
        } else if right_index == self.colors.len() {
            return self.colors[self.colors.len() - 1].1;
        }

        let left = self.colors[right_index - 1];
        let right = self.colors[right_index];

        let span = right.0 - left.0;
        let dt = (t - left.0) / span;

        left.1 * (1.0 - dt) + right.1 * dt
    }

    /// Sample the gradient at `t` using a monotone interpolation.
    pub fn monotone_sample(&self, t: f32) -> T
    where
        T: Copy + Into<[f32; 3]> + From<[f32; 3]>,
    {
        if self.colors.len() == 2 {
            return self.linear_sample(t);
        }

        let right_index = self
            .colors
            .iter()
            .position(|x| x.0 > t)
            .unwrap_or(self.colors.len());

        if right_index == 0 {
            return self.colors[0].1;
        } else if right_index == self.colors.len() {
            return self.colors[self.colors.len() - 1].1;
        }

        if right_index == 1 || right_index == self.colors.len() - 1 {
            let (i0, i1, i2);
            if right_index == 1 {
                i0 = right_index - 1;
                i1 = right_index;
                i2 = right_index + 1;
            } else {
                i0 = right_index - 2;
                i1 = right_index - 1;
                i2 = right_index;
            }

            let t0 = self.colors[i0].0;
            let t1 = self.colors[i1].0;
            let t2 = self.colors[i2].0;

            let p0 = self.colors[i0].1;
            let p1 = self.colors[i1].1;
            let p2 = self.colors[i2].1;

            let a1 = p0 * ((t1 - t) / (t1 - t0)) + p1 * ((t - t0) / (t1 - t0));
            let a2 = p1 * ((t2 - t) / (t2 - t1)) + p2 * ((t - t1) / (t2 - t1));

            let b1 = a1 * ((t2 - t) / (t2 - t0)) + a2 * ((t - t0) / (t2 - t0));

            return b1;
        }

        let i0 = right_index - 2;
        let i1 = right_index - 1;
        let i2 = right_index;
        let i3 = right_index + 1;

        let t0 = self.colors[i0].0;
        let t1 = self.colors[i1].0;
        let t2 = self.colors[i2].0;
        let t3 = self.colors[i3].0;

        let p0: [f32; 3] = self.colors[i0].1.into();
        let p1: [f32; 3] = self.colors[i1].1.into();
        let p2: [f32; 3] = self.colors[i2].1.into();
        let p3: [f32; 3] = self.colors[i3].1.into();

        let mut p = [0.0; 3];

        for i in 0..3 {
            p[i] = interpolate_cubic_monotonic_heckbert(
                t,
                (t0, p0[i]),
                (t1, p1[i]),
                (t2, p2[i]),
                (t3, p3[i]),
            );
        }

        p.into()
    }

    /// Sample the gradient at `t` using Catmull-Rom interpolation.
    pub fn catmull_rom_sample(&self, t: f32) -> T
    where
        T: Copy,
    {
        if self.colors.len() == 2 {
            return self.linear_sample(t);
        }

        let right_index = self
            .colors
            .iter()
            .position(|x| x.0 > t)
            .unwrap_or(self.colors.len());

        if right_index == 0 {
            return self.colors[0].1;
        } else if right_index == self.colors.len() {
            return self.colors[self.colors.len() - 1].1;
        }

        if right_index == 1 || right_index == self.colors.len() - 1 {
            let (i0, i1, i2);
            if right_index == 1 {
                i0 = right_index - 1;
                i1 = right_index;
                i2 = right_index + 1;
            } else {
                i0 = right_index - 2;
                i1 = right_index - 1;
                i2 = right_index;
            }

            let t0 = self.colors[i0].0;
            let t1 = self.colors[i1].0;
            let t2 = self.colors[i2].0;

            let p0 = self.colors[i0].1;
            let p1 = self.colors[i1].1;
            let p2 = self.colors[i2].1;

            let a1 = p0 * ((t1 - t) / (t1 - t0)) + p1 * ((t - t0) / (t1 - t0));
            let a2 = p1 * ((t2 - t) / (t2 - t1)) + p2 * ((t - t1) / (t2 - t1));

            let b1 = a1 * ((t2 - t) / (t2 - t0)) + a2 * ((t - t0) / (t2 - t0));

            return b1;
        }

        let i0 = right_index - 2;
        let i1 = right_index - 1;
        let i2 = right_index;
        let i3 = right_index + 1;

        let t0 = self.colors[i0].0;
        let t1 = self.colors[i1].0;
        let t2 = self.colors[i2].0;
        let t3 = self.colors[i3].0;

        let p0 = self.colors[i0].1;
        let p1 = self.colors[i1].1;
        let p2 = self.colors[i2].1;
        let p3 = self.colors[i3].1;

        // Barry and Goldman's pyramidal formulation
        let a1 = p0 * ((t1 - t) / (t1 - t0)) + p1 * ((t - t0) / (t1 - t0));
        let a2 = p1 * ((t2 - t) / (t2 - t1)) + p2 * ((t - t1) / (t2 - t1));
        let a3 = p2 * ((t3 - t) / (t3 - t2)) + p3 * ((t - t2) / (t3 - t2));

        let b1 = a1 * ((t2 - t) / (t2 - t0)) + a2 * ((t - t0) / (t2 - t0));
        let b2 = a2 * ((t3 - t) / (t3 - t1)) + a3 * ((t - t1) / (t3 - t1));

        let c1 = b1 * ((t2 - t) / (t2 - t1)) + b2 * ((t - t1) / (t2 - t1));

        c1
    }

    pub fn min(&self) -> f32 {
        self.colors[0].0
    }

    pub fn max(&self) -> f32 {
        self.colors[self.colors.len() - 1].0
    }
}

macro_rules! relative_eq {
    ($a:expr, $b:expr) => {
        ($a - $b).abs() <= 1e-6
    };
}

/// Monotone cubic interpolation of points (t1, y1) and (t1, y2) using x as the interpolation
/// parameter (assumed to be [0..1]). In order to maintain C1 continuity, two neighbouring
/// samples are required.
///
/// Reference: http://jbrd.github.io/2020/12/27/monotone-cubic-interpolation.html
pub fn interpolate_cubic_monotonic_heckbert(
    t: f32,
    (t0, y0): (f32, f32),
    (t1, y1): (f32, f32),
    (t2, y2): (f32, f32),
    (t3, y3): (f32, f32),
) -> f32 {
    // remap everything [t1, t2] to [0, 1]
    let x = remap(t, t1..=t2, 0.0..=1.0);
    let t0 = remap(t0, t1..=t2, 0.0..=1.0);
    let t3 = remap(t3, t1..=t2, 0.0..=1.0);
    let t1 = 0.0;
    let t2 = 1.0;

    // Calculate secant line gradients for each successive pair of data points
    let s_0 = (y1 - y0) / (t1 - t0);
    let s_1 = (y2 - y1) / (t2 - t1);
    let s_2 = (y3 - y2) / (t3 - t2);

    // Use central differences to calculate initial gradients at the end-points
    let mut m_1 = (s_0 + s_1) * 0.5;
    let mut m_2 = (s_1 + s_2) * 0.5;

    // If the central curve (joining y1 and y2) is neither increasing or decreasing, we
    // should have a horizontal line, so immediately set gradients to zero here.
    if relative_eq!(y1, y2) {
        m_1 = 0.0;
        m_2 = 0.0;
    } else {
        // If the curve to the left is horizontal, or the sign of the secants on either side
        // of the end-point are different, set the gradient to zero...
        if relative_eq!(y0, y1) || s_0 < 0.0 && s_1 >= 0.0 || s_0 > 0.0 && s_1 <= 0.0 {
            m_1 = 0.0;
        }
        // ... otherwise, ensure the magnitude of the gradient is constrained to 3 times the
        // left secant, and 3 times the right secant (whatever is smaller)
        else {
            m_1 *= (3.0 * s_0 / m_1).min(3.0 * s_1 / m_1).min(1.0);
        }

        // If the curve to the right is horizontal, or the sign of the secants on either side
        // of the end-point are different, set the gradient to zero...
        if relative_eq!(y2, y3) || s_1 < 0.0 && s_2 >= 0.0 || s_1 > 0.0 && s_2 <= 0.0 {
            m_2 = 0.0;
        }
        // ... otherwise, ensure the magnitude of the gradient is constrained to 3 times the
        // left secant, and 3 times the right secant (whatever is smaller)
        else {
            m_2 *= (3.0 * s_1 / m_2).min(3.0 * s_2 / m_2).min(1.0);
        }
    }

    // Evaluate the cubic hermite spline
    let result =
        (((((m_1 + m_2 - 2.0 * s_1) * x) + (3.0 * s_1 - 2.0 * m_1 - m_2)) * x) + m_1) * x + y1;

    // The values at the end points (y0 and y1) define an interval that the curve passes
    // through. Since the curve between the end-points is now monotonic, all interpolated
    // values between these end points should be inside this interval. However, floating
    // point rounding error can still lead to values slightly outside this range.
    // Guard against this by clamping the interpolated result to this interval...
    let min = y1.min(y2);
    let max = y1.max(y2);
    result.min(max).max(min)
}

#[derive(Clone, Copy, Debug, Default)]
struct GradientEditor {
    /// The index of the selected handle.
    selected: usize,
}

pub fn gradient_editor(ui: &mut Ui, gradient: &mut Gradient<Oklab>) -> bool {
    let id = Id::new("gradient_editor");
    let mut editor = ui
        .memory_mut(|x| x.data.get_persisted::<GradientEditor>(id))
        .unwrap_or_default();

    let mut changed = false;

    changed |= gradient_handles(ui, gradient, &mut editor).changed();

    let mut handle = gradient.colors[editor.selected];
    let mut color = handle.1.to_color32();
    let mut t = handle.0;

    ui.horizontal(|ui| {
        ui.label("Color:");
        changed |= ui.color_edit_button_srgba(&mut color).changed();
    });

    let fixed = is_fixed(editor.selected, gradient);

    ui.horizontal(|ui| {
        ui.label("Position:");
        changed |= ui
            .add_enabled(
                !fixed,
                egui::DragValue::new(&mut t)
                    .speed(0.001)
                    .clamp_range(gradient.min()..=gradient.max()),
            )
            .changed();
    });

    if changed {
        handle.0 = t;
        handle.1 = Srgb::new(
            color.r() as f32 / 255.0,
            color.g() as f32 / 255.0,
            color.b() as f32 / 255.0,
        )
        .into();

        gradient.colors[editor.selected] = handle;

        if !fixed {
            // sort the new handle position, including the selected handle:
            let sort_up =
                gradient.colors[editor.selected - 1].0 < gradient.colors[editor.selected].0;
            if sort_up {
                for i in editor.selected + 1..gradient.colors.len() {
                    if gradient.colors[i - 1].0 > gradient.colors[i].0 {
                        gradient.colors.swap(i - 1, i);

                        if i - 1 == editor.selected {
                            editor.selected = i;
                        }
                    } else {
                        break;
                    }
                }
            } else {
                for i in (1..=editor.selected).rev() {
                    if gradient.colors[i - 1].0 > gradient.colors[i].0 {
                        gradient.colors.swap(i - 1, i);

                        if i == editor.selected {
                            editor.selected = i - 1;
                        }
                    } else {
                        break;
                    }
                }
            }
        }
    }

    ui.memory_mut(|x| x.data.insert_persisted(id, editor));

    changed
}

fn gradient_handles(
    ui: &mut Ui,
    gradient: &mut Gradient<Oklab>,
    editor: &mut GradientEditor,
) -> Response {
    #![allow(clippy::identity_op)]

    let min = gradient.min();
    let max = gradient.max();
    let span = max - min;

    let desired_size = vec2(
        2.0 * ui.spacing().slider_width,
        ui.spacing().interact_size.y,
    );
    let (rect, mut response) = ui.allocate_at_least(desired_size, Sense::click_and_drag());

    const HANDLE_RADIUS: f32 = 32.0;
    let hovered = if let Some(mpos) = response.hover_pos() {
        gradient
            .colors
            .iter()
            .enumerate()
            .filter_map(|(i, (t, _))| {
                let x = lerp(rect.left()..=rect.right(), (*t - min) / span);
                let d = (mpos.x - x).abs();
                (d < HANDLE_RADIUS).then_some((i, d))
            })
            .min_by(|(_, d1), (_, d2)| d1.total_cmp(d2))
            .map(|(i, _)| i)
    } else {
        None
    };

    if let Some(mpos) = response.interact_pointer_pos() {
        if response.double_clicked() {
            let t = remap_clamp(mpos.x, rect.left()..=rect.right(), min..=max);
            let color = gradient.monotone_sample(t);

            gradient.colors.push((t, color));
            gradient.colors.sort_by(|a, b| a.0.total_cmp(&b.0));

            response.mark_changed();
        } else if response.clicked() || response.drag_started() {
            if let Some(x) = hovered {
                editor.selected = x;
                response.mark_changed();
            }
        } else if response.secondary_clicked() {
            if let Some(x) = hovered {
                if !is_fixed(x, gradient) {
                    gradient.colors.remove(x);
                    response.mark_changed();
                }
            }
        }
    }

    if response.dragged() && !is_fixed(editor.selected, gradient) {
        let delta = response.drag_delta();
        let delta_t = delta.x / rect.width() * span;
        let t = gradient.colors[editor.selected].0 + delta_t;
        gradient.colors[editor.selected].0 = t.clamp(min, max);
        response.mark_changed();
    }

    if ui.is_rect_visible(rect) {
        let visuals = ui.style().interact(&response);

        // background_checkers(ui.painter(), rect); // for alpha:

        {
            // fill color:
            let mut mesh = Mesh::default();
            for i in 0..=N {
                let t = min + (i as f32 * span / (N as f32));

                let color = gradient.monotone_sample(t).to_color32();

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

        for (i, &(t, _)) in gradient.colors.iter().enumerate() {
            let x = lerp(rect.left()..=rect.right(), (t - min) / span);
            let r = rect.height() / 4.0;
            let picked_color = gradient.monotone_sample(t).to_color32();

            let stroke_color = if i == editor.selected {
                if Rgba::from(picked_color).intensity() < 0.5 {
                    Color32::WHITE
                } else {
                    Color32::BLACK
                }
            } else if Rgba::from(picked_color).intensity() < 0.5 {
                Color32::LIGHT_GRAY
            } else {
                Color32::DARK_GRAY
            };

            let stroke_width = if Some(i) == hovered {
                visuals.fg_stroke.width * 2.0
            } else {
                visuals.fg_stroke.width
            };

            ui.painter().add(Shape::convex_polygon(
                vec![
                    pos2(x, rect.center().y),   // tip
                    pos2(x + r, rect.bottom()), // right bottom
                    pos2(x - r, rect.bottom()), // left bottom
                ],
                picked_color,
                Stroke::new(stroke_width, stroke_color),
            ));
        }
    }

    response
}

/// The position of the first and last color of the gradient are fixed.
fn is_fixed(selected: usize, gradient: &mut Gradient<Oklab>) -> bool {
    selected == 0 || selected == gradient.colors.len() - 1
}

#[cfg(test)]
mod test {
    use oklab::{OkLch, Oklab};
    use rand::{Rng, SeedableRng};

    use crate::widgets::gradient::Gradient;

    use super::interpolate_cubic_monotonic_heckbert;

    #[test]
    #[ignore]
    fn is_monotone() {
        let t0 = 1.0;
        let t1 = 2.0;
        let t2 = 3.0;
        let t3 = 4.0;

        let y0 = t0;
        let y1 = t1;
        let y2 = t2;
        let y3 = t3;

        let t = 2.5;
        let t_ = 2.7;

        let y = interpolate_cubic_monotonic_heckbert(t, (t0, y0), (t1, y1), (t2, y2), (t3, y3));
        let y_ = interpolate_cubic_monotonic_heckbert(t_, (t0, y0), (t1, y1), (t2, y2), (t3, y3));

        println!("t0: {:.2}, y0: {:.2}", t0, y0);
        println!("t1: {:.2}, y1: {:.2}", t1, y1);
        println!("t2: {:.2}, y2: {:.2}", t2, y2);
        println!("t3: {:.2}, y3: {:.2}", t3, y3);
        println!();
        println!("t : {:.2}, y : {:.2}", t, y);
        println!("t_: {:.2}, y_: {:.2}", t_, y_);

        assert!(y < y_);
    }

    fn sort2<T: PartialOrd>(a: T, b: T) -> (T, T) {
        if a < b {
            (a, b)
        } else {
            (b, a)
        }
    }

    #[test]
    fn is_monotone_fuzz() {
        for _ in 0..1000 {
            let t0 = 0.0;
            let t1 = 1.0;
            let t2 = 2.0;
            let t3 = 3.0;

            let mut rng = rand::rngs::SmallRng::seed_from_u64(0);

            let t = rng.gen_range(t1..=t2);
            let t_ = rng.gen_range(t1..=t2);

            let (t, t_) = sort2(t, t_);

            let y0 = rng.gen_range(-1.0..=1.0);
            let y1 = y0 + rng.gen_range(-1.0..=1.0);
            let y2 = y1 + rng.gen_range(-1.0..=1.0);
            let y3 = y2 + rng.gen_range(-1.0..=1.0);

            let y = interpolate_cubic_monotonic_heckbert(t, (t0, y0), (t1, y1), (t2, y2), (t3, y3));

            assert!(y >= y1.min(y2));
            assert!(y <= y1.max(y2));

            let y1_ =
                interpolate_cubic_monotonic_heckbert(t1, (t0, y0), (t1, y1), (t2, y2), (t3, y3));
            assert!((y1_ - y1).abs() < 0.0001);

            let y2_ =
                interpolate_cubic_monotonic_heckbert(t2, (t0, y0), (t1, y1), (t2, y2), (t3, y3));
            assert!((y2_ - y2).abs() < 0.0001);

            let y = interpolate_cubic_monotonic_heckbert(t, (t0, y0), (t1, y1), (t2, y2), (t3, y3));
            let y_ =
                interpolate_cubic_monotonic_heckbert(t_, (t0, y0), (t1, y1), (t2, y2), (t3, y3));

            println!("t0: {}, y0: {}", t0, y0);
            println!("t1: {}, y1: {}", t1, y1);
            println!("t2: {}, y2: {}", t2, y2);
            println!("t3: {}, y3: {}", t3, y3);
            println!();
            println!("t : {}, y : {}", t, y);
            println!("t_: {}, y_: {}", t_, y_);
            assert!(t <= t_);
            assert!((y - y_).abs() < 0.0001 || y1 < y2 && y < y_ || y1 > y2 && y > y_);

            let r1 = (y2 - y1) / (t2 - t1);

            let r = (y_ - y) / (t_ - t);
            let rb = (y_ - y1) / (t_ - t1);
            let ra = (y2 - y_) / (t2 - t_);

            println!("r1: {}", r1);
            println!("r : {}", r);
            println!("rb: {}", rb);
            println!("ra: {}", ra);

            assert!(r.abs() <= 2.0 * r1.abs());
            assert!(rb.abs() <= 2.0 * r1.abs());
            assert!(ra.abs() <= 2.0 * r1.abs());
        }
    }

    #[test]
    fn monotone_color_gradient() {
        let gradient: Gradient<Oklab> = Gradient::new(vec![
            (0.00, OkLch::new(0.00, 0.3, 29.0 / 360.0).into()),
            // (0.01, OkLch::new(0.00, 0.3, 29.0 / 360.0).into()),
            (0.50, OkLch::new(0.50, 0.3, 29.0 / 360.0).into()),
            (0.75, OkLch::new(0.75, 0.3, 69.0 / 360.0).into()),
            // (0.99, OkLch::new(1.00, 0.3, 110.0 / 360.0).into()),
            (1.00, OkLch::new(1.00, 0.3, 110.0 / 360.0).into()),
        ]);

        const N: usize = 10;
        for x in 0..=N {
            let x = x as f32 / N as f32;
            let color = gradient.monotone_sample(x);
            println!("{:4}: {:.2?}", x, color);
        }

        // panic!();
    }
}
