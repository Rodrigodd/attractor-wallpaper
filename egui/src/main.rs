use std::{
    sync::{atomic::AtomicI32, mpsc::Sender},
    time::{Duration, Instant},
};

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
const SAMPLES_PER_ITERATION: u64 = 1_000_000;

mod channel;

enum AttractorMess {
    SetSeed(u64),
    SetMultisampling(u8),
    SetAntialiasing(AntiAliasing),
    SetIntensity(f32),
    SetRandomStart(bool),
    SetMultithreaded(Multithreading),
    SetSamplesPerIteration(u64),
    Resize(PhysicalSize<u32>),
    Transform(Affine),
}

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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Multithreading {
    Single,
    AtomicMulti,
    MergeMulti,
}

#[derive(Clone)]
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
    multithreaded: Multithreading,
    samples_per_iteration: u64,
    starts: Vec<[f64; 2]>,
    // pub send_mess: Option<Sender<AttractorMess>>,
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
            multithreaded: Multithreading::Single,
            samples_per_iteration: SAMPLES_PER_ITERATION,
            starts: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.bitmap.fill(0);
        self.total_samples = 0;
        self.last_change = Instant::now();
    }

    fn transform(&mut self, mut affine: Affine) {
        affine.1[0] *= self.multisampling as f64;
        affine.1[1] *= self.multisampling as f64;
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

    fn set_intensity(&mut self, intensity: f32) {
        self.intensity = intensity;
    }

    fn set_random_start(&mut self, random_start: bool) {
        self.random_start = random_start;
        self.clear();
    }

    fn set_multithreaded(&mut self, multithreaded: Multithreading) {
        self.multithreaded = multithreaded;
        self.clear();
    }

    fn set_samples_per_iteration(&mut self, samples_per_iteration: u64) {
        self.samples_per_iteration = samples_per_iteration;
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
    multithreaded: Multithreading,
    samples_per_iteration_text: String,
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

    fn samples_per_iteration(&mut self) -> Option<u64> {
        self.samples_per_iteration_text.parse::<u64>().ok()
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
    let mut last_change = Instant::now();

    let mut gui_state = GuiState {
        seed_text: 8742.to_string(),
        multisampling_text: 1.to_string(),
        anti_aliasing: attractors::AntiAliasing::None,
        intensity: 1.0,
        dragging: false,
        rotating: false,
        last_cursor_position: PhysicalPosition::default(),
        random_start: false,
        multithreaded: Multithreading::Single,
        samples_per_iteration_text: SAMPLES_PER_ITERATION.to_string(),
    };

    // let mut attractor = AttractorCtx::new(&mut gui_state, size);

    let (attractor_sender, recv_conf) = std::sync::mpsc::channel::<AttractorMess>();
    let (mut sender_bitmap, mut recv_bitmap) = channel::channel::<AttractorCtx>();
    let mut attractor = AttractorCtx::new(&mut gui_state, size);
    std::thread::spawn(move || loop {
        loop {
            match recv_conf.try_recv() {
                Ok(mess) => match mess {
                    AttractorMess::SetSeed(seed) => attractor.set_seed(seed),
                    AttractorMess::SetMultisampling(multisampling) => {
                        attractor.set_multisampling(multisampling)
                    }
                    AttractorMess::SetAntialiasing(antialising) => {
                        attractor.set_antialiasing(antialising)
                    }
                    AttractorMess::SetIntensity(intensity) => attractor.set_intensity(intensity),
                    AttractorMess::SetRandomStart(random_start) => {
                        attractor.set_random_start(random_start)
                    }
                    AttractorMess::SetMultithreaded(multithreaded) => {
                        attractor.set_multithreaded(multithreaded)
                    }
                    AttractorMess::SetSamplesPerIteration(samples_per_iteration) => {
                        attractor.set_samples_per_iteration(samples_per_iteration)
                    }
                    AttractorMess::Resize(size) => attractor.resize(size, attractor.multisampling),
                    AttractorMess::Transform(affine) => attractor.transform(affine),
                },
                Err(std::sync::mpsc::TryRecvError::Disconnected) => return,
                _ => break,
            }
        }

        match attractor.multithreaded {
            Multithreading::Single => aggregate_attractor(&mut attractor),
            Multithreading::AtomicMulti => atomic_par_aggregate_attractor(&mut attractor),
            Multithreading::MergeMulti => merge_par_aggregate_attractor(&mut attractor),
        }

        if sender_bitmap.is_closed() {
            break;
        }
        sender_bitmap.send(&mut attractor);
    });

    // attractor.send_mess = Some(sender_conf);

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

                            let mat = [1.0, 0.0, 0.0, 1.0];
                            let trans = [-delta_x, -delta_y];

                            let _ = attractor_sender.send(AttractorMess::Transform((mat, trans)));
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

                            let _ = attractor_sender.send(AttractorMess::Transform((mat, trans)));
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
                        let _ = attractor_sender.send(AttractorMess::Transform((mat, trans)));
                    }
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(new_size) => {
                        let _ = attractor_sender.send(AttractorMess::Resize(new_size));

                        render_state
                            .attractor_renderer
                            .resize(&render_state.wgpu_state.device, new_size);
                        render_state
                            .surface
                            .resize(new_size, &render_state.wgpu_state.device);

                        // make sure the attractor is resized before the rendering.
                        recv_bitmap.recv(|_| {});
                    }
                    _ => (),
                }
            }
            Event::MainEventsCleared => {
                let mut total_sampĺes = 0;
                recv_bitmap.recv(|at| {
                    if at.last_change > last_change {
                        render_state
                            .attractor_renderer
                            .load_attractor(&render_state.wgpu_state.queue, &at.attractor);
                    }

                    at.bitmap[0] = render::get_intensity(
                        (at.base_intensity as f32 / at.intensity) as i16,
                        at.total_samples,
                        at.size,
                        at.multisampling,
                        at.anti_aliasing,
                    );
                    render_state
                        .attractor_renderer
                        .load_aggragate_buffer(&render_state.wgpu_state.queue, &at.bitmap);

                    total_sampĺes = at.total_samples;
                    last_change = at.last_change;
                });

                let new_input = egui_state.take_egui_input(window);

                let mut full_output = egui_ctx.run(new_input, |ui| {
                    egui::Window::new("Hello world!")
                        .resizable(false) // could not figure out how make this work
                        .show(ui, |ui| {
                            build_ui(
                                ui,
                                &mut gui_state,
                                &attractor_sender,
                                total_sampĺes,
                                last_change,
                                render_state,
                            );
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

fn aggregate_attractor(attractor: &mut AttractorCtx) {
    let samples = attractor.samples_per_iteration;

    if attractor.random_start {
        attractor.attractor.start = attractor
            .attractor
            .get_random_start_point(&mut rand::thread_rng());
    }

    attractors::aggregate_to_bitmap(
        &mut attractor.attractor,
        attractor.size.width as usize * attractor.multisampling as usize,
        attractor.size.height as usize * attractor.multisampling as usize,
        samples,
        attractor.anti_aliasing,
        &mut attractor.bitmap[..],
        &mut 0,
    );
    attractor.total_samples += samples;
}

fn atomic_par_aggregate_attractor(attractor: &mut AttractorCtx) {
    let samples = attractor.samples_per_iteration;

    // convert &mut [i32] to &mut [AtomicI32]
    let bitmap: &mut [AtomicI32] = unsafe { std::mem::transmute(&mut attractor.bitmap[..]) };

    let threads = 4;
    if attractor.random_start {
        attractor.starts.clear();
    }
    while attractor.starts.len() < threads {
        let mut rng = rand::thread_rng();
        attractor
            .starts
            .push(attractor.attractor.get_random_start_point(&mut rng));
    }

    std::thread::scope(|s| {
        for start in attractor.starts.iter_mut() {
            attractor.total_samples += samples;
            let size = attractor.size;
            let multisampling = attractor.multisampling;
            let anti_aliasing = attractor.anti_aliasing;

            let mut at = attractor.attractor;

            let bitmap: &[AtomicI32] = &*bitmap;

            at.start = *start;
            s.spawn(move || {
                attractors::aggregate_to_bitmap(
                    &mut at,
                    size.width as usize * multisampling as usize,
                    size.height as usize * multisampling as usize,
                    samples,
                    anti_aliasing,
                    &mut &*bitmap,
                    &mut AtomicI32::new(0),
                );
                *start = at.start;
            });
        }
    });
}

fn merge_par_aggregate_attractor(attractor: &mut AttractorCtx) {
    let samples = attractor.samples_per_iteration;

    let threads = 4;
    if attractor.random_start {
        attractor.starts.clear();
    }
    while attractor.starts.len() < threads {
        let mut rng = rand::thread_rng();
        attractor
            .starts
            .push(attractor.attractor.get_random_start_point(&mut rng));
    }

    // resize bitmap to hold multiple buffers
    let len = attractor.size.width as usize
        * attractor.multisampling as usize
        * attractor.size.height as usize
        * attractor.multisampling as usize;
    attractor.bitmap.resize(threads * len, 0);

    std::thread::scope(|s| {
        let mut bitmap = attractor.bitmap.as_mut_slice();
        for start in attractor.starts.iter_mut() {
            attractor.total_samples += samples;
            let size = attractor.size;
            let multisampling = attractor.multisampling;
            let anti_aliasing = attractor.anti_aliasing;

            let mut at = attractor.attractor;

            let b;
            (b, bitmap) = bitmap.split_at_mut(len);

            at.start = *start;
            s.spawn(move || {
                attractors::aggregate_to_bitmap(
                    &mut at,
                    size.width as usize * multisampling as usize,
                    size.height as usize * multisampling as usize,
                    samples,
                    anti_aliasing,
                    b,
                    &mut 0,
                );
                *start = at.start;
            });
        }
    });

    // merge the buffers into one
    for i in 0..len {
        let sum = attractor.bitmap.iter().skip(i).step_by(len).sum::<i32>();
        attractor.bitmap[i] = sum;
    }
    attractor.bitmap.truncate(len);
}

#[allow(clippy::too_many_arguments)]
fn build_ui(
    ui: &mut Ui,
    gui_state: &mut GuiState,
    attractor_sender: &Sender<AttractorMess>,
    total_samples: u64,
    last_change: Instant,
    render_state: &mut RenderState,
) -> egui::InnerResponse<()> {
    Grid::new("options_grid").show(ui, |ui| {
        ui.label(format!(
            "sample per second: {:.2e}",
            total_samples as f64 / last_change.elapsed().as_secs_f64()
        ));
        ui.end_row();

        ui.label("seed: ");
        ui.horizontal(|ui| {
            if ui.my_text_field(&mut gui_state.seed_text).lost_focus() {
                if let Some(seed) = gui_state.seed() {
                    let _ = attractor_sender.send(AttractorMess::SetSeed(seed));
                }
            }

            if ui.button("rand").clicked() {
                gui_state.set_seed(rand::thread_rng().gen());
                if let Some(seed) = gui_state.seed() {
                    let _ = attractor_sender.send(AttractorMess::SetSeed(seed));
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
                let _ = attractor_sender.send(AttractorMess::SetMultisampling(multisampling));
                render_state.attractor_renderer.recreate_aggregate_buffer(
                    &render_state.wgpu_state.device,
                    render_state.attractor_renderer.size,
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
            let _ = attractor_sender.send(AttractorMess::SetAntialiasing(gui_state.anti_aliasing));
        }

        ui.end_row();

        ui.label("intensity: ");
        ui.add(Slider::new(&mut gui_state.intensity, 0.01..=2.0));
        let _ = attractor_sender.send(AttractorMess::SetIntensity(gui_state.intensity));

        ui.end_row();

        ui.label("random start: ");
        if ui
            .add(Checkbox::new(&mut gui_state.random_start, ""))
            .changed()
        {
            let _ = attractor_sender.send(AttractorMess::SetRandomStart(gui_state.random_start));
        }

        ui.end_row();

        ui.label("multithreading: ");
        let prev_multihreaded = gui_state.multithreaded;

        ComboBox::new("multithreading", "")
            .selected_text(format!("{:?}", gui_state.multithreaded))
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut gui_state.multithreaded,
                    Multithreading::Single,
                    "Single",
                );
                ui.selectable_value(
                    &mut gui_state.multithreaded,
                    Multithreading::AtomicMulti,
                    "AtomicMulti",
                );
                ui.selectable_value(
                    &mut gui_state.multithreaded,
                    Multithreading::MergeMulti,
                    "MergeMulti",
                );
            });

        if prev_multihreaded != gui_state.multithreaded {
            let _ = attractor_sender.send(AttractorMess::SetMultithreaded(gui_state.multithreaded));
        }

        ui.end_row();

        ui.label("samples per iteration: ");
        if ui
            .text_edit_singleline(&mut gui_state.samples_per_iteration_text)
            .lost_focus()
        {
            if let Some(samples_per_iteration) = gui_state.samples_per_iteration() {
                let _ = attractor_sender
                    .send(AttractorMess::SetSamplesPerIteration(samples_per_iteration));
            }
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
