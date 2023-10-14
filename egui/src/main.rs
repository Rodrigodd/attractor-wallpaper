use std::time::{Duration, Instant};

use attractors::{Affine, AntiAliasing, Attractor};
use egui::{Checkbox, ComboBox, Grid, Response, Slider, TextEdit, Ui};
use egui_wgpu::wgpu;
use egui_winit::winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ControlFlow, EventLoopBuilder, EventLoopProxy},
    platform::wayland::WindowBuilderExtWayland,
    window::{Window, WindowBuilder},
};
use rand::prelude::*;

use render::{AttractorRenderer, SurfaceState, TaskId, WgpuState, WinitExecutor};

const BORDER: f64 = 0.1;

async fn build_renderer(window: Window, proxy: EventLoopProxy<UserEvent>) {
    let (wgpu_state, surface) = render::WgpuState::new_windowed(window).await.unwrap();
    let attractor_renderer = AttractorRenderer::new(
        &wgpu_state.device,
        surface.size(),
        surface.texture_format(),
        1,
    )
    .unwrap();

    let egui_renderer =
        egui_wgpu::Renderer::new(&wgpu_state.device, surface.texture_format(), None, 1);

    let _ = proxy.send_event(UserEvent::BuildRenderer((
        wgpu_state,
        surface,
        attractor_renderer,
        egui_renderer,
    )));
}

enum UserEvent {
    BuildRenderer(
        (
            WgpuState,
            SurfaceState,
            AttractorRenderer,
            egui_wgpu::Renderer,
        ),
    ),
    PollTask(TaskId),
}

struct RenderState {
    wgpu_state: WgpuState,
    surface: SurfaceState,
    attractor_renderer: AttractorRenderer,
    egui_renderer: egui_wgpu::Renderer,
}

struct AttractorCtx {
    attractor: Attractor,
    seed: u64,
    bitmap: Vec<i32>,
    total_samples: u64,
    base_intensity: i16,
    multisampling: u8,
    size: PhysicalSize<u32>,
    intensity: f32,
    anti_aliasing: AntiAliasing,
    random_start: bool,
    last_change: Instant,
}
impl AttractorCtx {
    fn new(gui_state: &mut GuiState, size: PhysicalSize<u32>) -> Self {
        let width = size.width as usize;
        let height = size.height as usize;
        let seed = gui_state.seed().unwrap_or(0);
        let multisampling = gui_state.multisampling().unwrap_or(1);
        let rng = rand::rngs::SmallRng::seed_from_u64(seed);
        let mut attractor = Attractor::find_strange_attractor(rng, 1_000_000).unwrap();

        let multisampling = multisampling as usize;

        {
            let points = attractor.get_points::<512>();

            // 4 KiB
            let affine = attractors::affine_from_pca(&points);
            attractor = attractor.transform_input(affine);

            let bounds = attractor.get_bounds(512);

            let dst = square_bounds(
                (width * multisampling) as f64,
                (height * multisampling) as f64,
                BORDER,
            );
            let affine = attractors::map_bounds_affine(dst, bounds);

            attractor = attractor.transform_input(affine);
        };
        let bitmap = vec![0i32; width * multisampling * height * multisampling];

        let base_intensity = attractors::get_base_intensity(&attractor);

        Self {
            attractor,
            seed,
            bitmap,
            total_samples: 0,
            base_intensity,
            multisampling: multisampling as u8,
            size,
            intensity: gui_state.intensity,
            anti_aliasing: gui_state.anti_aliasing,
            random_start: gui_state.random_start,
            last_change: Instant::now(),
        }
    }

    fn clear(&mut self) {
        self.bitmap.fill(0);
        self.total_samples = 0;
        self.last_change = Instant::now();
    }

    fn transform(&mut self, affine: Affine) {
        self.attractor = self.attractor.transform_input(affine);
        self.clear();
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>, new_multisampling: u8) {
        let old_size = self.size;
        let old_multisampling = self.multisampling;
        self.size = new_size;
        self.multisampling = new_multisampling;
        (self.bitmap, _) = render::resize_attractor(
            &mut self.attractor,
            (
                old_size.width as usize * old_multisampling as usize,
                old_size.height as usize * old_multisampling as usize,
            ),
            (
                new_size.width as usize * new_multisampling as usize,
                new_size.height as usize * new_multisampling as usize,
            ),
        );
        self.clear();
    }

    fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
        let rng = rand::rngs::SmallRng::seed_from_u64(self.seed);
        let mut attractor = Attractor::find_strange_attractor(rng, 1_000_000).unwrap();
        {
            let points = attractor.get_points::<512>();

            // 4 KiB
            let affine = attractors::affine_from_pca(&points);
            attractor = attractor.transform_input(affine);

            let bounds = attractor.get_bounds(512);

            let dst = square_bounds(
                (self.size.width * self.multisampling as u32) as f64,
                (self.size.height * self.multisampling as u32) as f64,
                BORDER,
            );
            let affine = attractors::map_bounds_affine(dst, bounds);

            attractor = attractor.transform_input(affine);
        };
        self.attractor = attractor;
        self.clear();
    }

    fn set_multisampling(&mut self, multisampling: u8) {
        self.multisampling = multisampling;
        self.resize(self.size, multisampling);
        self.clear();
    }

    fn set_antialiasing(&mut self, anti_aliasing: AntiAliasing) {
        self.anti_aliasing = anti_aliasing;
        self.clear();
    }

    fn set_random_start(&mut self, random_start: bool) {
        self.random_start = random_start;
        self.clear();
    }
}

fn square_bounds(width: f64, height: f64, border: f64) -> [f64; 4] {
    let size = width.min(height) * (1.0 - 2.0 * border);
    let start_x = (width - size) / 2.0;
    let start_y = (height - size) / 2.0;
    [start_x, start_x + size, start_y, start_y + size]
}

struct GuiState {
    seed_text: String,
    multisampling_text: String,
    anti_aliasing: AntiAliasing,
    intensity: f32,
    dragging: bool,
    rotating: bool,
    last_cursor_position: PhysicalPosition<f64>,
    random_start: bool,
}
impl GuiState {
    fn set_seed(&mut self, seed: u64) {
        self.seed_text = seed.to_string();
    }

    fn seed(&mut self) -> Option<u64> {
        self.seed_text.parse::<u64>().ok()
    }

    fn multisampling(&mut self) -> Option<u8> {
        self.multisampling_text.parse::<u8>().ok()
    }
}

fn main() {
    let event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();

    let size = egui_winit::winit::dpi::PhysicalSize::<u32>::new(800, 600);

    let wb = WindowBuilder::new()
        .with_title("A fantastic window!")
        .with_inner_size(size);

    let wb = { wb.with_name("dev", "") };

    let window = wb.build(&event_loop).unwrap();

    let egui_ctx = egui::Context::default();

    let mut executor = WinitExecutor::new(event_loop.create_proxy(), UserEvent::PollTask);

    let mut egui_state = egui_winit::State::new(&window);

    let task = build_renderer(window, event_loop.create_proxy());
    executor.spawn(task);

    let mut render_state = None;

    let mut gui_state = GuiState {
        seed_text: 0.to_string(),
        multisampling_text: 1.to_string(),
        anti_aliasing: attractors::AntiAliasing::None,
        intensity: 1.0,
        dragging: false,
        rotating: false,
        last_cursor_position: PhysicalPosition::default(),
        random_start: false,
    };

    let mut attractor = AttractorCtx::new(&mut gui_state, size);

    event_loop.run(move |event, _, control_flow| {
        // control_flow is a reference to an enum which tells us how to run the event loop.
        // See the docs for details: https://docs.rs/winit/0.22.2/winit/enum.ControlFlow.html
        *control_flow = ControlFlow::Wait;

        match event {
            Event::UserEvent(UserEvent::PollTask(id)) => return executor.poll(id),
            Event::UserEvent(UserEvent::BuildRenderer((
                wgpu_state,
                surface,
                attractor_renderer,
                egui_renderer,
            ))) => {
                println!("render state builded");
                dbg!(&surface.size());
                render_state = Some(RenderState {
                    wgpu_state,
                    surface,
                    attractor_renderer,
                    egui_renderer,
                });
                return;
            }
            _ => {}
        }

        let Some(render_state) = render_state.as_mut() else {
            return;
        };
        let window = render_state.surface.window();

        match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => {
                let res = egui_state.on_event(&egui_ctx, &event);

                if res.repaint {
                    window.request_redraw();
                }
                if res.consumed {
                    return;
                }

                match event {
                    WindowEvent::MouseInput {
                        state,
                        button: MouseButton::Left,
                        ..
                    } => match state {
                        ElementState::Pressed => gui_state.dragging = true,
                        ElementState::Released => gui_state.dragging = false,
                    },
                    WindowEvent::MouseInput {
                        state,
                        button: MouseButton::Right,
                        ..
                    } => match state {
                        ElementState::Pressed => gui_state.rotating = true,
                        ElementState::Released => gui_state.rotating = false,
                    },
                    WindowEvent::CursorLeft { .. } => {
                        gui_state.dragging = false;
                        gui_state.rotating = false;
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        if gui_state.dragging {
                            let delta_x = position.x - gui_state.last_cursor_position.x;
                            let delta_y = position.y - gui_state.last_cursor_position.y;

                            let m = attractor.multisampling as f64;
                            let mat = [1.0, 0.0, 0.0, 1.0];
                            let trans = [-delta_x * m, -delta_y * m];

                            attractor.transform((mat, trans));
                        } else if gui_state.rotating {
                            let size = render_state.surface.size();
                            let cx = size.width as f64 / 2.0;
                            let cy = size.height as f64 / 2.0;

                            let ldx = gui_state.last_cursor_position.y - cy;
                            let ldy = gui_state.last_cursor_position.x - cx;
                            let last_a = f64::atan2(ldx, ldy);

                            let dx = position.y - cy;
                            let dy = position.x - cx;
                            let a = f64::atan2(dx, dy);

                            let delta_a: f64 = a - last_a;

                            // Rot(x - c) + c => Rot(x) + (c - Rot(c))
                            let mat = [delta_a.cos(), delta_a.sin(), -delta_a.sin(), delta_a.cos()];
                            let trans = [
                                cx - mat[0] * cx - mat[1] * cy,
                                cy - mat[2] * cx - mat[3] * cy,
                            ];

                            attractor.transform((mat, trans));
                        }
                        gui_state.last_cursor_position = position;
                    }
                    WindowEvent::MouseWheel { delta, .. } => {
                        let delta = match delta {
                            MouseScrollDelta::LineDelta(_, y) => y as f64 * 12.0,
                            MouseScrollDelta::PixelDelta(PhysicalPosition { y, .. }) => y,
                        };

                        let delta_s = (-delta * 0.02).exp2();

                        // S(x - c) + c => S(x) + (c - S(c))
                        let mat = [delta_s, 0.0, 0.0, delta_s];
                        let trans = [
                            gui_state.last_cursor_position.x * (1.0 - delta_s),
                            gui_state.last_cursor_position.y * (1.0 - delta_s),
                        ];
                        attractor.transform((mat, trans));
                    }
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(new_size) => {
                        attractor.resize(new_size, attractor.multisampling);

                        render_state
                            .attractor_renderer
                            .resize(&render_state.wgpu_state.device, new_size);
                        render_state
                            .surface
                            .resize(new_size, &render_state.wgpu_state.device);
                        render_state
                            .attractor_renderer
                            .load_attractor(&render_state.wgpu_state.queue, &attractor.attractor);
                    }
                    _ => (),
                }
            }
            Event::MainEventsCleared => {
                let samples = 40_000;
                let size = render_state.surface.size();

                if attractor.random_start {
                    attractor.attractor.start = attractor
                        .attractor
                        .get_random_start_point(&mut rand::thread_rng());
                }

                attractors::aggregate_to_bitmap(
                    &mut attractor.attractor,
                    size.width as usize * attractor.multisampling as usize,
                    size.height as usize * attractor.multisampling as usize,
                    samples,
                    attractor.anti_aliasing,
                    &mut attractor.bitmap,
                );
                attractor.total_samples += samples;
                attractor.bitmap[0] = render::get_intensity(
                    (attractor.base_intensity as f32 / attractor.intensity) as i16,
                    attractor.total_samples,
                    size,
                    attractor.multisampling,
                    attractor.anti_aliasing,
                );
                render_state
                    .attractor_renderer
                    .load_aggragate_buffer(&render_state.wgpu_state.queue, &attractor.bitmap);

                let new_input = egui_state.take_egui_input(window);

                let mut full_output = egui_ctx.run(new_input, |ui| {
                    egui::Window::new("Hello world!")
                        .resizable(false) // could not figure out how make this work
                        .show(ui, |ui| {
                            build_ui(ui, &mut gui_state, &mut attractor, render_state);
                            // ui.allocate_space(ui.available_size());
                        });
                });

                let window = render_state.surface.window();
                if full_output.repaint_after == Duration::ZERO {
                    window.request_redraw();
                }

                egui_state.handle_platform_output(
                    window,
                    &egui_ctx,
                    full_output.platform_output.take(),
                );

                render_frame(&egui_ctx, full_output, render_state);
            }
            Event::RedrawRequested(id) if window.id() == id => {}
            _ => (),
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn build_ui(
    ui: &mut Ui,
    gui_state: &mut GuiState,
    attractor: &mut AttractorCtx,
    render_state: &mut RenderState,
) -> egui::InnerResponse<()> {
    Grid::new("options_grid").show(ui, |ui| {
        ui.label(format!(
            "sample per second: {:.2e}",
            attractor.total_samples as f64 / attractor.last_change.elapsed().as_secs_f64()
        ));
        ui.end_row();

        ui.label("seed: ");
        ui.horizontal(|ui| {
            if ui.my_text_field(&mut gui_state.seed_text).lost_focus() {
                if let Some(seed) = gui_state.seed() {
                    attractor.set_seed(seed);
                }
            }

            if ui.button("rand").clicked() {
                gui_state.set_seed(rand::thread_rng().gen());
                if let Some(seed) = gui_state.seed() {
                    attractor.set_seed(seed);
                }
            }
        });

        ui.end_row();

        ui.label("multisampling: ");
        if ui
            .my_text_field(&mut gui_state.multisampling_text)
            .lost_focus()
        {
            if let Some(multisampling) = gui_state.multisampling() {
                attractor.set_multisampling(multisampling);
                render_state.attractor_renderer.recreate_aggregate_buffer(
                    &render_state.wgpu_state.device,
                    attractor.size,
                    multisampling,
                );
            }
        }

        ui.end_row();

        ui.label("anti-aliasing: ");
        let prev_anti_aliasing = gui_state.anti_aliasing;

        ComboBox::new("anti-aliasing", "")
            .selected_text(format!("{:?}", gui_state.anti_aliasing))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut gui_state.anti_aliasing, AntiAliasing::None, "None");
                ui.selectable_value(
                    &mut gui_state.anti_aliasing,
                    AntiAliasing::Bilinear,
                    "Bilinear",
                );
                ui.selectable_value(
                    &mut gui_state.anti_aliasing,
                    AntiAliasing::Lanczos,
                    "Lanczos",
                );
            });

        if prev_anti_aliasing != gui_state.anti_aliasing {
            attractor.set_antialiasing(gui_state.anti_aliasing);
        }

        ui.end_row();

        ui.label("intensity: ");
        ui.add(Slider::new(&mut gui_state.intensity, 0.01..=2.0));
        attractor.intensity = gui_state.intensity;

        ui.end_row();

        ui.label("random start: ");
        if ui
            .add(Checkbox::new(&mut gui_state.random_start, ""))
            .changed()
        {
            attractor.set_random_start(gui_state.random_start);
        }
    })
}

fn render_frame(
    egui_ctx: &egui::Context,
    full_output: egui::FullOutput,
    render_state: &mut RenderState,
) {
    let paint_jobs = egui_ctx.tessellate(full_output.shapes);
    let queue = &render_state.wgpu_state.queue;
    let device = &render_state.wgpu_state.device;
    for (id, image_delta) in &full_output.textures_delta.set {
        render_state
            .egui_renderer
            .update_texture(device, queue, *id, image_delta);
    }
    let texture = match render_state.surface.current_texture() {
        Ok(texture) => texture,
        Err(e) => {
            eprintln!("Failed to acquire next surface texture: {:?}", e);
            return;
        }
    };
    let view = texture.texture.create_view(&Default::default());
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Render Encoder"),
    });
    let screen_descriptor = egui_wgpu::renderer::ScreenDescriptor {
        size_in_pixels: [
            render_state.surface.size().width,
            render_state.surface.size().height,
        ],
        pixels_per_point: 1.0,
    };
    render_state.attractor_renderer.set_uniforms(queue);
    render_state.egui_renderer.update_buffers(
        device,
        queue,
        &mut encoder,
        &paint_jobs,
        &screen_descriptor,
    );
    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Render Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: true,
            },
        })],
        depth_stencil_attachment: None,
    });
    render_state
        .attractor_renderer
        .render_aggregate_buffer(&mut render_pass);
    render_state
        .egui_renderer
        .render(&mut render_pass, &paint_jobs, &screen_descriptor);
    drop(render_pass);
    queue.submit(std::iter::once(encoder.finish()));
    texture.present();
}

trait MyUiExt {
    fn my_text_field(&mut self, text: &mut String) -> Response;
}

impl MyUiExt for Ui {
    fn my_text_field(&mut self, text: &mut String) -> Response {
        self.add_sized([180.0, self.available_height()], TextEdit::singleline(text))
    }
}
