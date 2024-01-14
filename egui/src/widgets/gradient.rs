use egui::{
    lerp, pos2, remap_clamp, vec2, Color32, Id, Mesh, Response, Rgba, Sense, Shape, Stroke, Ui,
};
use oklab::{OkLch, Oklab, Srgb};
use render::gradient::Gradient;

use crate::{widgets::ok_picker::ToColor32, MyUiExt};

const N: u32 = 256;

#[derive(Clone, Debug)]
struct LruCache<K, V> {
    values: Vec<(K, V)>,
    capacity: usize,
}
impl<K, V> LruCache<K, V> {
    fn new(capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
            capacity,
        }
    }

    fn get_or_insert_with(&mut self, key: K, f: impl FnOnce() -> V) -> (&mut K, &mut V)
    where
        K: Clone + PartialEq + std::fmt::Debug,
        V: std::fmt::Debug,
    {
        if let Some((key, value)) = self.values.iter_mut().rev().find(|(k, _)| k == &key) {
            // HACK: avoid limitation of borrow checker
            unsafe { std::mem::transmute((key, value)) }
        } else {
            let value = f();
            self.values.push((key, value));
            if self.values.len() > self.capacity {
                self.values.remove(0);
            }
            let (key, value) = self.values.last_mut().unwrap();
            (key, value)
        }
    }
}

#[derive(Clone, Debug)]
struct GradientEditor {
    /// The index of the selected handle.
    selected: usize,
    /// Cache color conversions to avoid losing continuity in non-injective conversions.
    cache: LruCache<Oklab, OkLch>,
}

impl Default for GradientEditor {
    fn default() -> Self {
        Self {
            selected: Default::default(),
            cache: LruCache::new(2),
        }
    }
}

pub fn gradient_editor(ui: &mut Ui, gradient: &mut Gradient<Oklab>, insertible: bool) -> bool {
    let id = Id::new("gradient_editor");
    let mut editor = ui
        .memory_mut(|x| x.data.get_persisted::<GradientEditor>(id))
        .unwrap_or_default();

    let mut changed = false;

    changed |= gradient_handles(ui, gradient, &mut editor, insertible).changed();

    let mut handle = gradient.colors[editor.selected];
    let mut t = handle.0;

    ui.horizontal(|ui| {
        ui.label("Color:");
        let (key, color) = editor
            .cache
            .get_or_insert_with(handle.1, || handle.1.into());
        if ui.my_color_picker(color).changed() {
            changed |= true;
            *key = Srgb::from(*color).clip().into();
            handle.1 = *key;
        }
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
    insertible: bool,
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
        if insertible && response.double_clicked() {
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
        } else if insertible && response.secondary_clicked() {
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
