#![allow(clippy::let_and_return)]

use std::{
    sync::{atomic::AtomicI32, mpsc::Sender, Arc},
    thread::JoinHandle,
    time::{Duration, Instant},
};

use attractors::{affine_affine, Affine, AntiAliasing, Attractor};
use egui::{Checkbox, ComboBox, Response, Slider, TextEdit, Ui, Vec2, WidgetText};
use egui_wgpu::wgpu;
use egui_winit::winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ControlFlow, EventLoopBuilder, EventLoopProxy},
    platform::wayland::WindowBuilderExtWayland,
    window::{Window, WindowBuilder},
};
use oklab::{LinSrgb, OkLch, Oklab};
use parking_lot::Mutex;
use rand::prelude::*;

use render::{AttractorRenderer, SurfaceState, TaskId, WgpuState, WinitExecutor};

use crate::widgets::ok_picker::ToColor32;

use self::widgets::gradient::Gradient;

mod channel;
pub mod widgets;

const BORDER: f64 = 0.1;
const SAMPLES_PER_ITERATION: u64 = 1_000_000;

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

/// Serializable configuration of the attractor
#[derive(Debug)]
struct AttractorConfig {
    base_attractor: Attractor,
    base_intensity: i16,
    transform: Affine,
    attractor: Attractor,
    size: PhysicalSize<u32>,

    seed: u64,
    multisampling: u8,
    anti_aliasing: AntiAliasing,
    intensity: f32,
    random_start: bool,
    multithreaded: Multithreading,
    samples_per_iteration: u64,
    background_color_1: OkLch,
    background_color_2: OkLch,
    gradient: Gradient<Oklab>,
}
impl Clone for AttractorConfig {
    fn clone(&self) -> Self {
        Self {
            base_attractor: self.base_attractor,
            base_intensity: self.base_intensity,
            transform: self.transform,
            attractor: self.attractor,
            size: self.size,
            seed: self.seed,
            multisampling: self.multisampling,
            anti_aliasing: self.anti_aliasing,
            intensity: self.intensity,
            random_start: self.random_start,
            multithreaded: self.multithreaded,
            samples_per_iteration: self.samples_per_iteration,
            background_color_1: self.background_color_1,
            background_color_2: self.background_color_2,
            gradient: self.gradient.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.base_attractor.clone_from(&source.base_attractor);
        self.base_intensity.clone_from(&source.base_intensity);
        self.transform.clone_from(&source.transform);
        self.attractor.clone_from(&source.attractor);
        self.size.clone_from(&source.size);
        self.seed.clone_from(&source.seed);
        self.multisampling.clone_from(&source.multisampling);
        self.anti_aliasing.clone_from(&source.anti_aliasing);
        self.intensity.clone_from(&source.intensity);
        self.random_start.clone_from(&source.random_start);
        self.multithreaded.clone_from(&source.multithreaded);
        self.samples_per_iteration
            .clone_from(&source.samples_per_iteration);
        self.background_color_1
            .clone_from(&source.background_color_1);
        self.background_color_2
            .clone_from(&source.background_color_2);
        self.gradient.clone_from(&source.gradient);
    }
}
impl AttractorConfig {
    fn bitmap_size(&self) -> [usize; 2] {
        let width = self.size.width as usize * self.multisampling as usize;
        let height = self.size.height as usize * self.multisampling as usize;
        [width, height]
    }
}

#[derive(Clone)]
struct AttractorCtx {
    config: Arc<Mutex<AttractorConfig>>,
    bitmap: Vec<i32>,
    total_samples: u64,
    last_change: Instant,
    starts: Vec<[f64; 2]>,
}
impl AttractorCtx {
    fn new(config: Arc<Mutex<AttractorConfig>>) -> Self {
        let bitmap = {
            let config = config.lock();
            let [width, height] = config.bitmap_size();
            vec![0; width * height]
        };
        Self {
            config,
            bitmap,
            total_samples: 0,
            last_change: Instant::now(),
            starts: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.bitmap.fill(0);
        self.total_samples = 0;
        self.last_change = Instant::now();
    }

    fn transform(&mut self, affine: Affine) {
        {
            let mut config = self.config.lock();
            config.transform = affine_affine(affine, config.transform);
            config.attractor = config.base_attractor.transform_input(config.transform);
        }
        self.clear();
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>, new_multisampling: u8) {
        {
            let mut config = self.config.lock();

            let old_size = config.bitmap_size();

            config.size = new_size;
            config.multisampling = new_multisampling;
            let new_size = config.bitmap_size();

            let src = square_bounds(old_size[0] as f64, old_size[1] as f64, BORDER);
            let dst = square_bounds(new_size[0] as f64, new_size[1] as f64, BORDER);
            let affine = attractors::map_bounds_affine(dst, src);

            config.transform = affine_affine(affine, config.transform);
            config.attractor = config.base_attractor.transform_input(config.transform);

            self.bitmap = vec![0i32; new_size[0] * new_size[1]];
            self.total_samples = 0;
        }
        self.starts.clear();
        self.clear();
    }

    fn set_seed(&mut self, seed: u64) {
        {
            self.config.lock().seed = seed;
            let rng = rand::rngs::SmallRng::seed_from_u64(seed);
            let mut attractor = Attractor::find_strange_attractor(rng, 1_000_000).unwrap();
            let points = attractor.get_points::<512>();

            // 4 KiB
            let affine = attractors::affine_from_pca(&points);
            attractor = attractor.transform_input(affine);

            let bounds = attractor.get_bounds(512);

            let mut config = self.config.lock();

            let [width, height] = config.bitmap_size();
            let dst = square_bounds(width as f64, height as f64, BORDER);
            let affine = attractors::map_bounds_affine(dst, bounds);

            attractor = attractor.transform_input(affine);
            config.base_intensity = attractors::get_base_intensity(&attractor);
            config.attractor = attractor;
        };

        self.clear();
    }

    fn set_multisampling(&mut self, multisampling: u8) {
        let size = self.config.lock().size;
        self.resize(size, multisampling);
        self.clear();
    }

    fn set_antialiasing(&mut self, anti_aliasing: AntiAliasing) {
        self.config.lock().anti_aliasing = anti_aliasing;
        self.clear();
    }

    fn set_intensity(&mut self, intensity: f32) {
        self.config.lock().intensity = intensity;
    }

    fn set_random_start(&mut self, random_start: bool) {
        self.config.lock().random_start = random_start;
        self.clear();
    }

    fn set_multithreaded(&mut self, multithreaded: Multithreading) {
        self.config.lock().multithreaded = multithreaded;
        self.clear();
    }

    fn set_samples_per_iteration(&mut self, samples_per_iteration: u64) {
        self.config.lock().samples_per_iteration = samples_per_iteration;
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
    multisampling: u8,
    anti_aliasing: AntiAliasing,
    intensity: f32,
    dragging: bool,
    rotating: bool,
    last_cursor_position: PhysicalPosition<f64>,
    random_start: bool,
    multithreaded: Multithreading,
    samples_per_iteration_text: String,
    total_samples_text: String,
    wallpaper_thread: Option<JoinHandle<AttractorCtx>>,
    background_color_1: OkLch,
    background_color_2: OkLch,
    gradient: Gradient<Oklab>,
}
impl GuiState {
    fn set_seed(&mut self, seed: u64) {
        self.seed_text = seed.to_string();
    }

    fn seed(&mut self) -> Option<u64> {
        self.seed_text.parse::<u64>().ok()
    }

    fn multisampling(&mut self) -> u8 {
        self.multisampling
    }

    fn samples_per_iteration(&mut self) -> Option<u64> {
        self.samples_per_iteration_text.parse::<u64>().ok()
    }

    fn total_samples(&mut self) -> Option<u64> {
        self.total_samples_text.parse::<u64>().ok()
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

    let seed = 8742;
    let multisampling = 1;
    let attractor = {
        let rng = rand::rngs::SmallRng::seed_from_u64(seed);
        let mut attractor = Attractor::find_strange_attractor(rng, 1_000_000).unwrap();
        let points = attractor.get_points::<512>();

        // 4 KiB
        let affine = attractors::affine_from_pca(&points);
        attractor = attractor.transform_input(affine);

        let bounds = attractor.get_bounds(512);

        let [width, height] = [
            size.width as f64 * multisampling as f64,
            size.height as f64 * multisampling as f64,
        ];
        let dst = square_bounds(width, height, BORDER);
        let affine = attractors::map_bounds_affine(dst, bounds);

        attractor.transform_input(affine)
    };

    let attractor_config = AttractorConfig {
        base_intensity: attractors::get_base_intensity(&attractor),
        base_attractor: attractor,
        transform: ([1.0, 0.0, 0.0, 1.0], [0.0, 0.0]),
        attractor,
        size,
        seed,
        multisampling,
        anti_aliasing: attractors::AntiAliasing::None,
        intensity: 1.0,
        random_start: false,
        multithreaded: Multithreading::Single,
        samples_per_iteration: SAMPLES_PER_ITERATION,
        background_color_1: OkLch::new(0.1, 1.0, 0.05),
        background_color_2: OkLch::new(0.05, 1.0, 0.05),
        gradient: Gradient::new(vec![
            (
                0.00,
                OkLch::new(0.01, 0.3, 29.0 / 360.0).to_srgb().clip().into(),
            ),
            // (0.01, OkLch::new(0.00, 0.3, 29.0 / 360.0).into()),
            (
                0.25,
                OkLch::new(0.25, 0.3, 29.0 / 360.0).to_srgb().clip().into(),
            ),
            (
                0.50,
                OkLch::new(0.50, 0.3, 29.0 / 360.0).to_srgb().clip().into(),
            ),
            (
                0.75,
                OkLch::new(0.75, 0.3, 69.0 / 360.0).to_srgb().clip().into(),
            ),
            // (0.99, OkLch::new(1.00, 0.3, 110.0 / 360.0).into()),
            (
                1.00,
                OkLch::new(1.00, 0.3, 110.0 / 360.0).to_srgb().clip().into(),
            ),
        ]),
    };

    let mut gui_state = GuiState {
        seed_text: attractor_config.seed.to_string(),
        multisampling: attractor_config.multisampling,
        anti_aliasing: attractor_config.anti_aliasing,
        intensity: attractor_config.intensity,
        dragging: false,
        rotating: false,
        last_cursor_position: PhysicalPosition::default(),
        random_start: attractor_config.random_start,
        multithreaded: attractor_config.multithreaded,
        samples_per_iteration_text: attractor_config.samples_per_iteration.to_string(),
        total_samples_text: 10_000_000.to_string(),
        wallpaper_thread: None,
        background_color_1: attractor_config.background_color_1,
        background_color_2: attractor_config.background_color_2,
        gradient: attractor_config.gradient.clone(),
    };

    let attractor_config = Arc::new(Mutex::new(attractor_config));

    // let mut attractor = AttractorCtx::new(&mut gui_state, size);

    let (attractor_sender, recv_conf) = std::sync::mpsc::channel::<AttractorMess>();
    let (mut sender_bitmap, mut recv_bitmap) = channel::channel::<AttractorCtx>();
    let mut attractor = AttractorCtx::new(attractor_config.clone());

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
                    AttractorMess::Resize(size) => {
                        let multisampling = attractor.config.lock().multisampling;
                        attractor.resize(size, multisampling)
                    }
                    AttractorMess::Transform(affine) => attractor.transform(affine),
                },
                Err(std::sync::mpsc::TryRecvError::Disconnected) => return,
                _ => break,
            }
        }

        let multithreading = attractor.config.lock().multithreaded;
        match multithreading {
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
                attractor_renderer.set_colormap(
                    &wgpu_state.queue,
                    gui_state
                        .gradient
                        .monotonic_hermit_spline_coefs()
                        .into_iter()
                        .map(|x| x.into())
                        .collect(),
                );
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

                        let s = (-delta * 0.02).exp2();

                        let multisampling = attractor_config.lock().multisampling;
                        let [x, y] = [
                            gui_state.last_cursor_position.x * multisampling as f64,
                            gui_state.last_cursor_position.y * multisampling as f64,
                        ];

                        // S(x - c) + c => S(x) + (c - S(c))
                        //              => s*x + (1 - s)*c
                        let cs = 1.0 - s;

                        let mat = [s, 0.0, 0.0, s];
                        let trans = [x * cs, y * cs];

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
                    }
                    _ => (),
                }
            }
            Event::MainEventsCleared => {
                let mut total_sampĺes = 0;
                recv_bitmap.recv(|at| {
                    let rsize = render_state.attractor_renderer.size;
                    let rsize = [
                        rsize.width * render_state.attractor_renderer.multisampling as u32,
                        rsize.height * render_state.attractor_renderer.multisampling as u32,
                    ];
                    let config = at.config.lock();
                    let [width, height] = config.bitmap_size();
                    let asize = [width as u32, height as u32];
                    if asize != rsize {
                        // don't update the bitmap if the attractor has not resized
                        // yet
                        return;
                    }

                    if at.total_samples < 100_000 {
                        // don't update bitmap if there is not enough samples yet. This avoids
                        // flickering while draggin the attractor.
                        return;
                    }

                    let base_intensity = config.base_intensity as f32 / config.intensity;
                    let total_samples = at.total_samples;
                    let size = config.size;
                    let multisampling = config.multisampling;
                    let anti_aliasing = config.anti_aliasing;

                    // drop lock before doing any heavy computation
                    drop(config);

                    at.bitmap[0] = render::get_intensity(
                        base_intensity,
                        total_samples,
                        size,
                        multisampling,
                        anti_aliasing,
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
                                &mut recv_bitmap,
                                &attractor_config,
                                total_sampĺes,
                                last_change,
                                render_state,
                                &mut executor,
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
    let mut config = attractor.config.lock();

    let samples = config.samples_per_iteration;

    if config.random_start {
        config.attractor.start = config
            .attractor
            .get_random_start_point(&mut rand::thread_rng());
    }

    let [width, height] = config.bitmap_size();
    let anti_aliasing = config.anti_aliasing;
    let mut att = config.attractor;

    // avoid holding the lock while doing the heavy computation
    drop(config);

    attractors::aggregate_to_bitmap(
        &mut att,
        width,
        height,
        samples,
        anti_aliasing,
        &mut attractor.bitmap[..],
        &mut 0,
    );
    attractor.total_samples += samples;

    attractor.config.lock().attractor = att;
}

fn atomic_par_aggregate_attractor(attractor: &mut AttractorCtx) {
    let config = attractor.config.lock();

    let samples = config.samples_per_iteration;
    let threads = 4;
    if config.random_start {
        attractor.starts.clear();
    }
    while attractor.starts.len() < threads {
        let mut rng = rand::thread_rng();
        attractor
            .starts
            .push(config.attractor.get_random_start_point(&mut rng));
    }

    let [width, height] = config.bitmap_size();
    let anti_aliasing = config.anti_aliasing;
    let att = config.attractor;

    // avoid holding the lock while doing the heavy computation
    drop(config);

    // SAFETY: AtomicI32 and i32 have the same layout.
    let bitmap: &mut [AtomicI32] = unsafe { std::mem::transmute(&mut attractor.bitmap[..]) };

    std::thread::scope(|s| {
        for start in attractor.starts.iter_mut() {
            attractor.total_samples += samples;

            let mut att = att;

            let bitmap: &[AtomicI32] = &*bitmap;

            att.start = *start;
            s.spawn(move || {
                attractors::aggregate_to_bitmap(
                    &mut att,
                    width,
                    height,
                    samples,
                    anti_aliasing,
                    &mut &*bitmap,
                    &mut AtomicI32::new(0),
                );
                *start = att.start;
            });
        }
    });
}

fn merge_par_aggregate_attractor(attractor: &mut AttractorCtx) {
    let config = attractor.config.lock();

    let samples = config.samples_per_iteration;
    let threads = 4;
    if config.random_start {
        attractor.starts.clear();
    }
    while attractor.starts.len() < threads {
        let mut rng = rand::thread_rng();
        attractor
            .starts
            .push(config.attractor.get_random_start_point(&mut rng));
    }

    let [width, height] = config.bitmap_size();
    let anti_aliasing = config.anti_aliasing;
    let att = config.attractor;

    // avoid holding the lock while doing the heavy computation
    drop(config);

    // resize bitmap to hold multiple buffers
    let len = width * height;
    attractor.bitmap.resize(threads * len, 0);

    std::thread::scope(|s| {
        let mut bitmap = attractor.bitmap.as_mut_slice();
        for start in attractor.starts.iter_mut() {
            attractor.total_samples += samples;
            let mut att = att;

            let b;
            (b, bitmap) = bitmap.split_at_mut(len);

            att.start = *start;
            s.spawn(move || {
                attractors::aggregate_to_bitmap(
                    &mut att,
                    width,
                    height,
                    samples,
                    anti_aliasing,
                    b,
                    &mut 0,
                );
                *start = att.start;
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
    attractor_recv: &mut channel::Receiver<AttractorCtx>,
    config: &Mutex<AttractorConfig>,
    total_samples: u64,
    last_change: Instant,
    render_state: &mut RenderState,
    executor: &mut WinitExecutor<UserEvent>,
) -> egui::InnerResponse<()> {
    ui.vertical(|ui| {
        ui.my_field("samples per second:", |ui| {
            ui.label(format!(
                "{:.2e}",
                total_samples as f64 / last_change.elapsed().as_secs_f64()
            ));
        });

        ui.my_field("seed:", |ui| {
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

        ui.my_field("multisampling:", |ui| {
            if ui
                .add(Slider::new(&mut gui_state.multisampling, 1..=6))
                .changed()
            {
                let _ =
                    attractor_sender.send(AttractorMess::SetMultisampling(gui_state.multisampling));
                render_state.attractor_renderer.recreate_aggregate_buffer(
                    &render_state.wgpu_state.device,
                    render_state.attractor_renderer.size,
                    gui_state.multisampling,
                );
            }
        });

        ui.my_field("anti-aliasing:", |ui| {
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
                let _ =
                    attractor_sender.send(AttractorMess::SetAntialiasing(gui_state.anti_aliasing));
            }
        });

        ui.my_field("intensity:", |ui| {
            if ui
                .add(Slider::new(&mut gui_state.intensity, 0.01..=4.0))
                .changed()
            {
                let _ = attractor_sender.send(AttractorMess::SetIntensity(gui_state.intensity));
            }
        });

        ui.end_row();

        ui.my_field("random start:", |ui| {
            if ui
                .add(Checkbox::new(&mut gui_state.random_start, ""))
                .changed()
            {
                let _ =
                    attractor_sender.send(AttractorMess::SetRandomStart(gui_state.random_start));
            }
        });

        ui.my_field("multithreading:", |ui| {
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
                let _ =
                    attractor_sender.send(AttractorMess::SetMultithreaded(gui_state.multithreaded));
            }
        });

        ui.my_field("samples per iteration:", |ui| {
            if ui
                .my_text_field(&mut gui_state.samples_per_iteration_text)
                .lost_focus()
            {
                if let Some(samples_per_iteration) = gui_state.samples_per_iteration() {
                    let _ = attractor_sender
                        .send(AttractorMess::SetSamplesPerIteration(samples_per_iteration));
                }
            }
        });

        ui.separator();

        ui.my_field("background color 1:", |ui| {
            if ui
                .my_color_picker(&mut gui_state.background_color_1)
                .changed()
            {
                config.lock().background_color_1 = gui_state.background_color_1;

                let c1 = LinSrgb::from(gui_state.background_color_1).clip();
                let c2 = LinSrgb::from(gui_state.background_color_2).clip();
                render_state.attractor_renderer.set_background_color(
                    &render_state.wgpu_state.queue,
                    [c1.r, c1.g, c1.b, 1.0],
                    [c2.r, c2.g, c2.b, 1.0],
                );
            }
        });

        ui.my_field("background color 2:", |ui| {
            if ui
                .my_color_picker(&mut gui_state.background_color_2)
                .changed()
            {
                config.lock().background_color_2 = gui_state.background_color_2;

                let c1 = LinSrgb::from(gui_state.background_color_1).clip();
                let c2 = LinSrgb::from(gui_state.background_color_2).clip();
                render_state.attractor_renderer.set_background_color(
                    &render_state.wgpu_state.queue,
                    [c1.r, c1.g, c1.b, 1.0],
                    [c2.r, c2.g, c2.b, 1.0],
                );
            }
        });

        if ui.my_gradient_picker(&mut gui_state.gradient) {
            config.lock().gradient.clone_from(&gui_state.gradient);
            render_state.attractor_renderer.set_colormap(
                &render_state.wgpu_state.queue,
                gui_state
                    .gradient
                    .monotonic_hermit_spline_coefs()
                    .into_iter()
                    .map(|x| x.into())
                    .collect(),
            )
        }

        ui.separator();

        ui.my_field("total samples:", |ui| {
            ui.my_text_field(&mut gui_state.total_samples_text);
        });

        if gui_state.wallpaper_thread.is_some() {
            ui.spinner();
        } else if ui.button("Set as wallpaper").clicked() {
            let mut attractor = attractor_recv
                .recv(|x| {
                    let config = x.config.lock().clone();
                    let config = Arc::new(Mutex::new(config));
                    AttractorCtx::new(config)
                })
                .unwrap();

            let monitor_size = render_state
                .surface
                .window()
                .current_monitor()
                .unwrap()
                .size();

            let multisampling = attractor.config.lock().multisampling;

            attractor.resize(monitor_size, multisampling);
            attractor.config.lock().samples_per_iteration = gui_state.total_samples().unwrap();

            let multithreaded = gui_state.multithreaded;

            let handle = std::thread::spawn(move || {
                match multithreaded {
                    Multithreading::Single => aggregate_attractor(&mut attractor),
                    Multithreading::AtomicMulti => atomic_par_aggregate_attractor(&mut attractor),
                    Multithreading::MergeMulti => merge_par_aggregate_attractor(&mut attractor),
                }
                attractor
            });

            gui_state.wallpaper_thread = Some(handle);
        }

        if gui_state
            .wallpaper_thread
            .as_mut()
            .map_or(false, |x| x.is_finished())
        {
            let wallpaper_handle = gui_state.wallpaper_thread.take().unwrap();

            let AttractorCtx {
                mut bitmap,
                total_samples,
                config,
                ..
            } = wallpaper_handle.join().unwrap();

            let config = Arc::try_unwrap(config).unwrap().into_inner();

            let AttractorConfig {
                size,
                base_intensity,
                multisampling,
                anti_aliasing,
                ..
            } = config;

            let task = async move {
                let wgpu_state = WgpuState::new_headless().await.unwrap();
                let mut attractor_renderer = AttractorRenderer::new(
                    &wgpu_state.device,
                    size,
                    wgpu::TextureFormat::Rgba8UnormSrgb,
                    multisampling,
                )
                .unwrap();

                bitmap[0] = render::get_intensity(
                    base_intensity as f32,
                    total_samples,
                    size,
                    multisampling,
                    anti_aliasing,
                );
                attractor_renderer.load_aggragate_buffer(&wgpu_state.queue, &bitmap);

                let texture = wgpu_state.new_target_texture(size);
                let view = texture.create_view(&Default::default());
                attractor_renderer.render(&wgpu_state.device, &wgpu_state.queue, &view);

                let bitmap = wgpu_state.copy_texture_content(texture);

                let path = "wallpaper.png";

                image::save_buffer(
                    path,
                    &bitmap,
                    size.width,
                    size.height,
                    image::ColorType::Rgba8,
                )
                .unwrap();

                // kill swaybg
                println!("killing swaybg");
                let _ = std::process::Command::new("killall").arg("swaybg").output();

                println!("setting wallpaper");
                wallpaper::set_from_path(path).unwrap();
                println!("wallpaper set");
            };
            executor.spawn(task);
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
    render_state.attractor_renderer.update_uniforms(queue);
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

trait ToSrgb {
    fn to_srgb(&self) -> oklab::Srgb;
}
impl<T> ToSrgb for T
where
    oklab::Srgb: std::convert::From<T>,
    T: Copy,
{
    fn to_srgb(&self) -> oklab::Srgb {
        oklab::Srgb::from(*self)
    }
}

trait MyUiExt {
    fn my_text_field(&mut self, text: &mut String) -> Response;
    fn my_color_picker(&mut self, oklch: &mut OkLch) -> Response;
    fn my_gradient_picker(&mut self, gradient: &mut Gradient<Oklab>) -> bool;

    /// A label with a fixed width, horizontally followed by a widget.
    fn my_field<T>(
        &mut self,
        label: impl Into<WidgetText>,
        content: impl FnOnce(&mut Ui) -> T,
    ) -> T;
}

impl MyUiExt for Ui {
    fn my_text_field(&mut self, text: &mut String) -> Response {
        self.add_sized([120.0, self.available_height()], TextEdit::singleline(text))
    }

    fn my_color_picker(&mut self, oklch: &mut OkLch) -> Response {
        let ui = self;

        popup_button(
            ui,
            oklch,
            |ui, oklch, open| color_button(ui, oklch.to_color32(), open),
            |ui, oklch, response| {
                if widgets::ok_picker::okhsv::color_picker_2d(ui, oklch) {
                    response.mark_changed();
                }
            },
        )
    }

    fn my_gradient_picker(&mut self, gradient: &mut Gradient<Oklab>) -> bool {
        let ui = self;

        widgets::gradient::gradient_editor(ui, gradient)
    }

    fn my_field<T>(
        &mut self,
        label: impl Into<WidgetText>,
        content: impl FnOnce(&mut Ui) -> T,
    ) -> T {
        self.horizontal(|ui| {
            const FIELD_WIDTH: f32 = 130.0;

            let total_size = ui.available_size();

            ui.label(label);
            let remaining_size = ui.available_size();

            let label_width = total_size.x - remaining_size.x;
            let pad_width = FIELD_WIDTH - label_width;

            if pad_width > 0.0 {
                ui.allocate_exact_size(
                    Vec2 {
                        x: pad_width,
                        y: remaining_size.y,
                    },
                    egui::Sense::hover(),
                );
            } else {
                println!("label_width: {}", label_width);
            }

            ui.horizontal(content).inner
        })
        .inner
    }
}

fn color_button(ui: &mut Ui, color: egui::Color32, open: bool) -> Response {
    let size = ui.spacing().interact_size;
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
    response.widget_info(|| egui::WidgetInfo::new(egui::WidgetType::ColorButton));

    if ui.is_rect_visible(rect) {
        let visuals = if open {
            &ui.visuals().widgets.open
        } else {
            ui.style().interact(&response)
        };
        let rect = rect.expand(visuals.expansion);

        egui::color_picker::show_color_at(ui.painter(), color, rect);

        let rounding = visuals.rounding.at_most(2.0);
        ui.painter()
            .rect_stroke(rect, rounding, (2.0, visuals.bg_fill)); // fill is intentional, because default style has no border
    }

    response
}

fn popup_button<T>(
    ui: &mut Ui,
    user_state: &mut T,
    button: impl Fn(&mut Ui, &mut T, bool) -> Response,
    mut widget: impl FnMut(&mut Ui, &mut T, &mut Response),
) -> Response {
    let popup_id = ui.auto_id_with("my_popup");
    let open = ui.memory(|mem| mem.is_popup_open(popup_id));

    let mut button_response = button(ui, user_state, open);
    if ui.style().explanation_tooltips {
        button_response = button_response.on_hover_text("Click to edit color");
    }

    if button_response.clicked() {
        ui.memory_mut(|mem| mem.toggle_popup(popup_id));
    }

    const POPUP_WIDTH: f32 = 100.0;
    if ui.memory(|mem| mem.is_popup_open(popup_id)) {
        let area_response = egui::Area::new(popup_id)
            .order(egui::Order::Foreground)
            .fixed_pos(button_response.rect.max)
            .constrain(true)
            .show(ui.ctx(), |ui| {
                ui.spacing_mut().slider_width = POPUP_WIDTH;
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    widget(ui, user_state, &mut button_response);
                });
            })
            .response;

        if !button_response.clicked()
            && (ui.input(|i| i.key_pressed(egui::Key::Escape)) || area_response.clicked_elsewhere())
        {
            ui.memory_mut(|mem| mem.close_popup());
        }
    }
    button_response
}
