use std::{
    sync::{mpsc::Sender, Arc},
    thread::JoinHandle,
    time::{Duration, Instant},
};

use attractors::AntiAliasing;
use documented::DocumentedFields;
use egui::{Checkbox, ComboBox, Response, Slider, TextEdit, Ui, Vec2, WidgetText};
use egui_wgpu::wgpu;
use egui_winit::winit::{
    dpi::PhysicalPosition,
    event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ControlFlow, EventLoopBuilder, EventLoopProxy},
    platform::wayland::WindowBuilderExtWayland,
    window::{Window, WindowBuilder},
};
use oklab::{LinSrgb, OkLch, Oklab};
use parking_lot::Mutex;
use rand::prelude::*;

use render::{
    aggregate_attractor_single_thread, aggregate_buffer, atomic_par_aggregate_attractor,
    attractor_thread, gradient::Gradient, merge_par_aggregate_attractor, update_render,
    AttractorConfig, AttractorCtx, AttractorMess, AttractorRenderer, Multithreading, SavedThemes,
    SurfaceState, Theme, WgpuState,
};
use winit::event::{KeyboardInput, ModifiersState, VirtualKeyCode};
use winit_executor::{TaskId, WinitExecutor};

use crate::widgets::ok_picker::ToColor32;

pub mod widgets;

mod cli {
    use clap::Parser;

    #[derive(Parser)]
    #[command(author, version, about, long_about = None)]
    pub struct Cli {
        /// Spawn a window in fullscreen mode. In headless mode, make the output image the same
        /// size as the focused monitor.
        #[arg(short, long, default_value = "false")]
        pub fullscreen: bool,

        /// Enable anti-aliasing.
        #[arg(short, long, default_value = "none")]
        pub anti_aliasing: AntiAliasing,

        /// The seed to use for the sequence of generated attractors.
        #[arg(short, long)]
        pub seed: Option<u64>,

        /// Render the attractor into a buffer with both dimensions scaled by this factor, and them
        /// downsample it to the expected size. Used for anti-aliasing.
        #[arg(short, long, default_value = "1")]
        pub multisampling: u8,

        /// Renders in headless mode, and outputs the attractor to the given file.
        #[arg(short, long)]
        pub output: Option<String>,

        /// The dimensions of the spawned window, or the size of the output image in headless mode.
        #[arg(short, long, default_value = "512x512", value_parser = size_value_parser, conflicts_with = "fullscreen")]
        pub dimensions: winit::dpi::PhysicalSize<u32>,

        /// The number of samples to render for in headless mode.
        #[arg(long, default_value = "10_000_000", value_parser = parse_integer)]
        pub samples: u64,

        /// If set, sets the outputed image as the wallpaper.
        #[arg(long, requires("output"))]
        pub set_wallpaper: bool,

        /// The name of one of the saved themes to use.
        #[arg(short, long)]
        pub theme: Option<String>,
    }

    #[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
    pub enum AntiAliasing {
        None,
        Bilinear,
        Lanczos,
    }
    impl AntiAliasing {
        pub fn into_attractors_antialiasing(self) -> attractors::AntiAliasing {
            match self {
                Self::None => attractors::AntiAliasing::None,
                Self::Bilinear => attractors::AntiAliasing::Bilinear,
                Self::Lanczos => attractors::AntiAliasing::Lanczos,
            }
        }
    }

    fn size_value_parser(s: &str) -> Result<winit::dpi::PhysicalSize<u32>, String> {
        let mut parts = s.split('x');
        let width = parts
            .next()
            .ok_or_else(|| "Missing width".to_string())
            .and_then(|s| {
                s.parse::<u32>()
                    .map_err(|err| format!("Invalid width: {}", err))
            })?;
        let height = parts
            .next()
            .ok_or_else(|| "Missing height".to_string())
            .and_then(|s| {
                s.parse::<u32>()
                    .map_err(|err| format!("Invalid height: {}", err))
            })?;
        Ok(winit::dpi::PhysicalSize::new(width, height))
    }

    // parse integer with underscores
    fn parse_integer(s: &str) -> Result<u64, String> {
        s.chars()
            .filter(|c| *c != '_')
            .map(|c| {
                c.to_digit(10)
                    .ok_or_else(|| format!("Invalid digit: {}", c))
            })
            .try_fold(0u64, |acc, digit| {
                let digit = digit? as u64;
                acc.checked_mul(10)
                    .and_then(|acc| acc.checked_add(digit))
                    .ok_or_else(|| "Integer overflow".to_string())
            })
    }
}

async fn build_renderer(window: Window, proxy: EventLoopProxy<UserEvent>) {
    let size = window.inner_size();
    let (wgpu_state, surface) = render::WgpuState::new_windowed(window, (size.width, size.height))
        .await
        .unwrap();
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
            SurfaceState<Window>,
            AttractorRenderer,
            egui_wgpu::Renderer,
        ),
    ),
    PollTask(TaskId),
}

struct RenderState {
    wgpu_state: WgpuState,
    surface: SurfaceState<Window>,
    attractor_renderer: AttractorRenderer,
    egui_renderer: egui_wgpu::Renderer,
}

#[derive(Default, DocumentedFields)]
struct GuiState {
    /// The seed used to generate the attractor. The underlying algorithm will use this seed to
    /// generate a random sequence of attractors, until it find one that is chaotic.
    seed_text: String,
    /// The minimum area that a attractor occupies. If the first attractor generated by a given
    /// seed doesn't satifies this requirement, a diferent attractor will be generated instead.
    min_area: u16,
    /// The amount of multisampling to be used. This value multiplies the width and height of the
    /// bitmap, so a value of 2 will result in a bitmap with 4 times the area. The bitmap is then
    /// downsampled to the output size using a lanczos kernel, in order to reduce aliasing.
    multisampling: u8,
    /// A method to reduce aliasing by drawing the points of the attractor as a antialiased point.
    /// This aliasing is done before applying the color gradient, so it causes artifacts around
    /// lines. Not recommended.
    anti_aliasing: AntiAliasing,
    /// A multiplier for the intensity of each pixel before applying the color gradient.
    intensity: f32,
    dragging: bool,
    rotating: bool,
    last_cursor_position: PhysicalPosition<f64>,
    /// If each iteration of the attractor rendering should use a random start position. Was added
    /// for resolving problems with periodic attractors, but that was never a problem to begin
    /// with.
    random_start: bool,
    /// If the attractor should be rendered using multiple threads.
    /// - Single: Use a single thread.
    /// - AtomicMulti: Use a bitmap of AtomicI32's that is rendered by multiple threads.
    /// - MergeMulti: Each thread renders to a separate bitmap, that is merged at the end of each
    /// iteration. For a big enough number of samples per iteration, this should be the fastest
    /// method, although the one that consumes more memory.
    multithreading: Multithreading,
    /// The number of samples to be generated for each iteration of the attractor. The UI will
    /// block until a the end of a iteration every time the UI is updated/rendered. This value
    /// should be small enough to avoid lagging.
    samples_per_iteration_text: String,
    /// The number of samples to use when rendering the wallpaper. A bigger number of samples will
    /// result in less noise, but will take longer to render.
    total_samples_text: String,
    /// Generates a image with the current configuration, and the resolution of the current
    /// monitor, and sets it as the wallpaper.
    wallpaper_thread: Option<JoinHandle<AttractorCtx>>,
    /// A list of saved themes.
    saved_themes: SavedThemes,
    /// The name of the current theme, used to save it in the list.
    theme_name: String,
    /// The color of the background at the top right corner.
    background_color_1: OkLch,
    /// The color of the background at the bottom left corner.
    background_color_2: OkLch,
    /// The color gradient used to colormap the attractor.
    gradient: Gradient<Oklab>,
}
impl GuiState {
    fn set_seed(&mut self, seed: u64) {
        self.seed_text = seed.to_string();
    }

    fn seed(&mut self) -> Option<u64> {
        self.seed_text.parse::<u64>().ok()
    }

    fn samples_per_iteration(&mut self) -> Option<u64> {
        self.samples_per_iteration_text.parse::<u64>().ok()
    }

    fn total_samples(&mut self) -> Option<u64> {
        self.total_samples_text.parse::<u64>().ok()
    }
}

fn main() {
    env_logger::init_from_env(
        env_logger::Env::default().default_filter_or("info,render=trace,attractor=trace"),
    );

    let cli = <cli::Cli as clap::Parser>::parse();

    let json = include_str!("default.json");
    let mut attractor_config = serde_json::from_str::<AttractorConfig>(json).unwrap();

    {
        let cli::Cli {
            fullscreen: _,
            anti_aliasing,
            seed,
            multisampling,
            output: _,
            dimensions: _,
            samples: _,
            set_wallpaper: _,
            theme,
        } = &cli;

        let config = &mut attractor_config;

        config.anti_aliasing = anti_aliasing.into_attractors_antialiasing();
        if let Some(seed) = seed {
            config.set_seed(*seed);
        } else {
            config.set_seed(rand::random::<u64>() % 1_000_000);
        }

        config.resize(config.size, *multisampling);

        if let Some(name) = theme {
            if let Some(theme) = config.saved_themes.get(name).cloned() {
                config.set_theme(name, &theme);
            } else {
                eprintln!("Theme {} not found. Avaliable themes are:", name);
                for name in config.saved_themes.keys() {
                    eprintln!("  - {}", name);
                }
                std::process::exit(1);
            }
        }
    }

    if cli.output.is_some() {
        return run_headless(attractor_config, cli);
    }

    run_ui(attractor_config, cli.fullscreen);
}

fn run_headless(mut config: AttractorConfig, mut cli: cli::Cli) {
    if cli.fullscreen {
        let ev = winit::event_loop::EventLoop::new();
        let monitor = match ev.primary_monitor() {
            Some(x) => x,
            None => {
                let Some(x) = ev.available_monitors().next() else {
                    println!("ERROR: No monitors found");
                    std::process::exit(1)
                };
                println!("WARN: No primary monitor found, falling back to first avaliable monitor");
                x
            }
        };
        cli.dimensions = monitor.size();
    }

    config.samples_per_iteration = cli.samples;

    let output = cli.output.unwrap();

    let multithreaded = config.multithreading;
    let multisampling = config.multisampling;

    let mut attractor = AttractorCtx::new(Arc::new(Mutex::new(config)));

    attractor.resize((cli.dimensions.width, cli.dimensions.height), multisampling);

    aggregate_buffer(multithreaded, &mut attractor);

    let AttractorCtx {
        bitmap,
        total_samples,
        config,
        ..
    } = attractor;

    let config = Arc::try_unwrap(config).unwrap().into_inner();

    let AttractorConfig {
        size,
        base_intensity,
        multisampling,
        anti_aliasing,
        gradient,
        background_color_1,
        background_color_2,
        transform: (mat, _),
        ..
    } = config;

    pollster::block_on(async move {
        let bitmap = render_to_bitmap(
            size,
            multisampling,
            bitmap,
            base_intensity,
            mat,
            total_samples,
            anti_aliasing,
            gradient,
            background_color_1,
            background_color_2,
        )
        .await;

        image::save_buffer(&output, &bitmap, size.0, size.1, image::ColorType::Rgba8).unwrap();

        if cli.set_wallpaper {
            // kill swaybg
            println!("killing swaybg");
            let _ = std::process::Command::new("killall").arg("swaybg").output();

            println!("setting wallpaper");
            wallpaper::set_from_path(&output).unwrap();
            println!("wallpaper set");
        }
    });
}

fn run_ui(attractor_config: AttractorConfig, fullscreen: bool) {
    let event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();

    let size = egui_winit::winit::dpi::PhysicalSize::<u32>::new(800, 600);

    let wb = WindowBuilder::new()
        .with_title("A fantastic window!")
        .with_inner_size(size);

    let wb = { wb.with_name("dev", "") };

    let wb = if fullscreen {
        wb.with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)))
    } else {
        wb
    };

    let window = wb.build(&event_loop).unwrap();

    let egui_ctx = egui::Context::default();

    let mut executor = WinitExecutor::new(event_loop.create_proxy(), UserEvent::PollTask);

    let mut egui_state = egui_winit::State::new(egui::ViewportId::ROOT, &window, None, None);

    let task = build_renderer(window, event_loop.create_proxy());
    executor.spawn(task);

    let mut render_state = None;
    let mut last_change = Instant::now();

    let mut last_noise = 0.0;
    let mut exp_moving_avg = 0.0;

    let mut gui_state = GuiState {
        total_samples_text: 10_000_000.to_string(),
        ..GuiState::default()
    };

    update_gui_state_from_config(&mut gui_state, &attractor_config);

    let attractor_config = Arc::new(Mutex::new(attractor_config));

    // let mut attractor = AttractorCtx::new(&mut gui_state, size);

    let (attractor_sender, recv_conf) = std::sync::mpsc::channel::<AttractorMess>();
    let (sender_bitmap, mut recv_bitmap) = render::channel::channel::<AttractorCtx>();
    let attractor = AttractorCtx::new(attractor_config.clone());

    std::thread::spawn(move || attractor_thread(recv_conf, attractor, sender_bitmap));

    // attractor.send_mess = Some(sender_conf);

    let mut modifiers = ModifiersState::empty();

    event_loop.run(move |event, _, control_flow| {
        // control_flow is a reference to an enum which tells us how to run the event loop.
        // See the docs for details: https://docs.rs/winit/0.22.2/winit/enum.ControlFlow.html
        *control_flow = ControlFlow::Wait;

        match event {
            Event::UserEvent(UserEvent::PollTask(id)) => return executor.poll(id),
            Event::UserEvent(UserEvent::BuildRenderer((
                wgpu_state,
                surface,
                mut attractor_renderer,
                egui_renderer,
            ))) => {
                update_render(
                    &mut attractor_renderer,
                    &wgpu_state,
                    &gui_state.gradient,
                    gui_state.multisampling,
                    gui_state.background_color_1,
                    gui_state.background_color_2,
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
                let res = egui_state.on_window_event(&egui_ctx, &event);

                if res.repaint {
                    window.request_redraw();
                }
                if res.consumed {
                    return;
                }

                match event {
                    WindowEvent::ModifiersChanged(m) => modifiers = m,
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(virtual_keycode),
                                ..
                            },
                        ..
                    } => match virtual_keycode {
                        VirtualKeyCode::NumpadEnter | VirtualKeyCode::Return if modifiers.alt() => {
                            println!("toggling fullscreen");
                            let window = render_state.surface.window();
                            if window.fullscreen().is_some() {
                                // window.set_decorations(true);
                                window.set_fullscreen(None);
                            } else {
                                // window.set_decorations(false);
                                // window.set_fullscreen(Some(
                                //     winit::window::Fullscreen::Borderless(None),
                                // ));
                                if let Some(monitor) = window.current_monitor() {
                                    window.set_fullscreen(Some(
                                        winit::window::Fullscreen::Borderless(Some(monitor)),
                                    ));
                                } else {
                                    window.set_fullscreen(Some(
                                        winit::window::Fullscreen::Borderless(None),
                                    ));
                                }
                            }
                        }
                        VirtualKeyCode::Q => {
                            *control_flow = ControlFlow::Exit;
                        }
                        _ => {}
                    },
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
                            let trans = [
                                -delta_x * gui_state.multisampling as f64,
                                -delta_y * gui_state.multisampling as f64,
                            ];

                            let _ = attractor_sender.send(AttractorMess::Transform((mat, trans)));
                        } else if gui_state.rotating {
                            let size = render_state.surface.size();
                            let cx = size.0 as f64 / 2.0;
                            let cy = size.1 as f64 / 2.0;

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
                        let _ = attractor_sender
                            .send(AttractorMess::Resize((new_size.width, new_size.height)));

                        let new_size = (new_size.width, new_size.height);
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
                let mut stop_time = Instant::now();
                recv_bitmap.recv(|at| {
                    let rsize = render_state.attractor_renderer.size;
                    let rsize = [
                        rsize.0 * render_state.attractor_renderer.multisampling as u32,
                        rsize.1 * render_state.attractor_renderer.multisampling as u32,
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
                    let anti_aliasing = config.anti_aliasing;
                    let mat = config.transform.0;

                    // drop lock before doing any heavy computation
                    drop(config);

                    if at.stop_time.is_none() {
                        let noise = attractors::estimate_noise(&at.bitmap, width, height);
                        let diff = noise - last_noise;
                        last_noise = noise;

                        let exp = 0.03;
                        exp_moving_avg = exp_moving_avg * (1.0 - exp) + diff * 0.03;
                    }

                    at.bitmap[0] =
                        render::get_intensity(base_intensity, mat, total_samples, anti_aliasing);
                    render_state
                        .attractor_renderer
                        .load_aggregate_buffer(&render_state.wgpu_state.queue, &at.bitmap);

                    total_sampĺes = at.total_samples;
                    last_change = at.last_change;
                    if let Some(x) = at.stop_time {
                        stop_time = x;
                    } else {
                        stop_time = Instant::now();
                    }
                });

                let new_input = egui_state.take_egui_input(window);

                let mut full_output = egui_ctx.run(new_input, |ui| {
                    egui::Window::new("Configuration")
                        .resizable(true)
                        .scroll2([false, true])
                        .show(ui, |ui| {
                            build_ui(
                                ui,
                                &mut gui_state,
                                &attractor_sender,
                                &mut recv_bitmap,
                                &attractor_config,
                                total_sampĺes,
                                stop_time - last_change,
                                exp_moving_avg,
                                render_state,
                                &mut executor,
                            );
                            // ui.allocate_space(ui.available_size());
                        });
                });

                let window = render_state.surface.window();
                let view_output = full_output
                    .viewport_output
                    .get(&egui::ViewportId::ROOT)
                    .unwrap();
                if view_output.repaint_delay == Duration::ZERO {
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
    attractor_sender: &Sender<AttractorMess>,
    attractor_recv: &mut render::channel::Receiver<AttractorCtx>,
    config: &Mutex<AttractorConfig>,
    total_samples: u64,
    elapsed_time: Duration,
    convergence: f64,
    render_state: &mut RenderState,
    executor: &mut WinitExecutor<UserEvent>,
) -> egui::InnerResponse<()> {
    ui.vertical(|ui| {
        ui.collapsing("stats", |ui| {
            ui.my_field("samples per second:", |ui| {
                ui.label(format!(
                    "{:.2e} ({:.2e} / {:.2}s))",
                    total_samples as f64 / elapsed_time.as_secs_f64(),
                    total_samples,
                    elapsed_time.as_secs_f64()
                ));
            });
            ui.my_field("convergence:", |ui| {
                ui.label(format!("{:.4e}", convergence,));
            });
        });

        ui.collapsing("rendering", |ui| {
            ui.my_field("seed:", |ui| {
                if ui.my_text_field(&mut gui_state.seed_text).lost_focus() {
                    if let Some(seed) = gui_state.seed() {
                        let _ = attractor_sender.send(AttractorMess::SetSeed(seed));
                    }
                }

                if ui.button("rand").clicked() {
                    gui_state.set_seed(rand::thread_rng().gen_range(0..1_000_000));
                    if let Some(seed) = gui_state.seed() {
                        let _ = attractor_sender.send(AttractorMess::SetSeed(seed));
                    }
                }
            })
            .doc("seed_text");

            ui.my_field("min area:", |ui| {
                if ui
                    .add(Slider::new(&mut gui_state.min_area, 0..=2048))
                    .changed()
                {
                    let _ = attractor_sender.send(AttractorMess::SetMinArea(gui_state.min_area));
                }
            })
            .doc("min_area");

            ui.my_field("multisampling:", |ui| {
                if ui
                    .add(Slider::new(&mut gui_state.multisampling, 1..=6))
                    .changed()
                {
                    let _ = attractor_sender
                        .send(AttractorMess::SetMultisampling(gui_state.multisampling));
                    render_state.attractor_renderer.recreate_aggregate_buffer(
                        &render_state.wgpu_state.device,
                        render_state.attractor_renderer.size,
                        gui_state.multisampling,
                    );
                }
            })
            .doc("multisampling");

            ui.my_field("anti-aliasing:", |ui| {
                let prev_anti_aliasing = gui_state.anti_aliasing;

                ComboBox::new("anti-aliasing", "")
                    .selected_text(format!("{:?}", gui_state.anti_aliasing))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut gui_state.anti_aliasing,
                            AntiAliasing::None,
                            "None",
                        );
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
                    let _ = attractor_sender
                        .send(AttractorMess::SetAntialiasing(gui_state.anti_aliasing));
                }
            })
            .doc("anti_aliasing");

            ui.my_field("intensity:", |ui| {
                if ui
                    .add(Slider::new(&mut gui_state.intensity, 0.01..=4.0))
                    .changed()
                {
                    let _ = attractor_sender.send(AttractorMess::SetIntensity(gui_state.intensity));
                }
            })
            .doc("intensity");

            ui.my_field("random start:", |ui| {
                if ui
                    .add(Checkbox::new(&mut gui_state.random_start, ""))
                    .changed()
                {
                    let _ = attractor_sender
                        .send(AttractorMess::SetRandomStart(gui_state.random_start));
                }
            })
            .doc("random_start");

            ui.my_field("multithreading:", |ui| {
                let prev_multihreaded = gui_state.multithreading;

                ComboBox::new("multithreading", "")
                    .selected_text(format!("{:?}", gui_state.multithreading))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut gui_state.multithreading,
                            Multithreading::Single,
                            "Single",
                        );
                        ui.selectable_value(
                            &mut gui_state.multithreading,
                            Multithreading::AtomicMulti,
                            "AtomicMulti",
                        );
                        ui.selectable_value(
                            &mut gui_state.multithreading,
                            Multithreading::MergeMulti,
                            "MergeMulti",
                        );
                    });

                if prev_multihreaded != gui_state.multithreading {
                    let _ = attractor_sender
                        .send(AttractorMess::SetMultithreaded(gui_state.multithreading));
                }
            })
            .doc("multithreading");

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
            })
            .doc("samples_per_iteration_text");
        });

        ui.collapsing("theme", |ui| {
            ui.my_field("themes:", |ui| {
                ComboBox::new("saved_themes", "")
                    // .selected_text(gui_state.theme_name.clone())
                    .show_ui(ui, |ui| {
                        let mut changed = false;
                        for (name, theme) in gui_state.saved_themes.iter() {
                            if ui
                                .selectable_label(false, name)
                                .on_hover_text("load gradient")
                                .clicked()
                            {
                                gui_state.theme_name = name.clone();
                                gui_state.background_color_1 = theme.background_color_1;
                                gui_state.background_color_2 = theme.background_color_2;
                                gui_state.gradient = theme.gradient.clone();

                                config.lock().set_theme(name, theme);

                                changed = true;
                            }
                        }

                        if changed {
                            update_render(
                                &mut render_state.attractor_renderer,
                                &render_state.wgpu_state,
                                &gui_state.gradient,
                                gui_state.multisampling,
                                gui_state.background_color_1,
                                gui_state.background_color_2,
                            );
                        }
                    });
            })
            .doc("saved_themes");

            ui.my_field("theme name:", |ui| {
                if ui.my_text_field(&mut gui_state.theme_name).changed() {
                    config
                        .lock()
                        .saved_themes
                        .clone_from(&gui_state.saved_themes);
                }
            })
            .doc("theme_name");

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
            })
            .doc("background_color_1");

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
            })
            .doc("background_color_2");

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

            ui.my_field("save theme:", |ui| {
                if ui.button("save").clicked() {
                    gui_state.saved_themes.insert(
                        gui_state.theme_name.clone(),
                        Theme {
                            background_color_1: gui_state.background_color_1,
                            background_color_2: gui_state.background_color_2,
                            gradient: gui_state.gradient.clone(),
                        },
                    );
                    config
                        .lock()
                        .saved_themes
                        .clone_from(&gui_state.saved_themes);
                }
            });
        });

        ui.collapsing("wallpaper", |ui| {
            ui.my_field("total samples:", |ui| {
                ui.my_text_field(&mut gui_state.total_samples_text);
            })
            .doc("total_samples_text");

            if gui_state.wallpaper_thread.is_some() {
                ui.spinner();
            } else if ui
                .button("Set as wallpaper")
                .doc("wallpaper_thread")
                .clicked()
            {
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

                attractor.resize((monitor_size.width, monitor_size.height), multisampling);
                attractor.config.lock().samples_per_iteration = gui_state.total_samples().unwrap();

                let multithreaded = gui_state.multithreading;

                let handle = std::thread::spawn(move || {
                    aggregate_buffer(multithreaded, &mut attractor);
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
                    bitmap,
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
                    gradient,
                    background_color_1,
                    background_color_2,
                    transform: (mat, _),
                    ..
                } = config;

                let task = async move {
                    let bitmap = render_to_bitmap(
                        size,
                        multisampling,
                        bitmap,
                        base_intensity,
                        mat,
                        total_samples,
                        anti_aliasing,
                        gradient,
                        background_color_1,
                        background_color_2,
                    )
                    .await;

                    let path = "wallpaper.png";

                    image::save_buffer(path, &bitmap, size.0, size.1, image::ColorType::Rgba8)
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
        });

        ui.horizontal(|ui| {
            if ui.button("Save").clicked() {
                let config = config.lock();

                let j: Result<String, _> = serde_json::to_string_pretty(&*config);

                drop(config);

                match j {
                    Err(e) => {
                        println!("Error serializing: {e:?}")
                    }
                    Ok(x) => {
                        std::fs::write("config.json", x).unwrap();
                    }
                }
            }

            if ui.button("load").clicked() {
                let json = std::fs::read_to_string("config.json");
                match json {
                    Ok(json) => {
                        let mut config = config.lock();
                        let c = match serde_json::from_str::<AttractorConfig>(&json) {
                            Ok(c) => c,
                            Err(e) => {
                                println!("could not parse json: {}", e);
                                return;
                            }
                        };
                        *config = c;
                        update_render(
                            &mut render_state.attractor_renderer,
                            &render_state.wgpu_state,
                            &gui_state.gradient,
                            gui_state.multisampling,
                            gui_state.background_color_1,
                            gui_state.background_color_2,
                        );
                        update_gui_state_from_config(gui_state, &config);
                        drop(config);
                        let _ = attractor_sender.send(AttractorMess::Update);
                        let size = render_state.surface.size();
                        let _ = attractor_sender.send(AttractorMess::Resize(size));
                        attractor_recv.recv(|_| {});
                    }
                    _ => println!("could not open config.json"),
                };
            }
        });
    })
}

#[allow(clippy::too_many_arguments)]
async fn render_to_bitmap(
    size: (u32, u32),
    multisampling: u8,
    mut bitmap: Vec<i32>,
    base_intensity: i16,
    mat: [f64; 4],
    total_samples: u64,
    anti_aliasing: AntiAliasing,
    gradient: Gradient<Oklab>,
    background_color_1: OkLch,
    background_color_2: OkLch,
) -> Vec<u8> {
    let wgpu_state = WgpuState::new_headless().await.unwrap();
    let mut attractor_renderer = AttractorRenderer::new(
        &wgpu_state.device,
        size,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        multisampling,
    )
    .unwrap();

    bitmap[0] = render::get_intensity(base_intensity as f32, mat, total_samples, anti_aliasing);
    attractor_renderer.load_aggregate_buffer(&wgpu_state.queue, &bitmap);

    update_render(
        &mut attractor_renderer,
        &wgpu_state,
        &gradient,
        multisampling,
        background_color_1,
        background_color_2,
    );

    let texture = wgpu_state.new_target_texture(size);
    let view = texture.create_view(&Default::default());
    attractor_renderer.render(&wgpu_state.device, &wgpu_state.queue, &view);

    wgpu_state.copy_texture_content(texture)
}

fn update_gui_state_from_config(gui_state: &mut GuiState, config: &AttractorConfig) {
    gui_state.seed_text = config.seed.to_string();
    gui_state.min_area = config.min_area;
    gui_state.multisampling = config.multisampling;
    gui_state.anti_aliasing = config.anti_aliasing;
    gui_state.intensity = config.intensity;
    gui_state.random_start = config.random_start;
    gui_state.samples_per_iteration_text = config.samples_per_iteration.to_string();
    gui_state.multithreading = config.multithreading;
    gui_state.saved_themes = config.saved_themes.clone();
    gui_state.theme_name = config.theme_name.clone();
    gui_state.background_color_1 = config.background_color_1;
    gui_state.background_color_2 = config.background_color_2;
    gui_state.gradient.clone_from(&config.gradient);
}

fn render_frame(
    egui_ctx: &egui::Context,
    full_output: egui::FullOutput,
    render_state: &mut RenderState,
) {
    let paint_jobs = egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
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
        size_in_pixels: [render_state.surface.size().0, render_state.surface.size().1],
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
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        ..Default::default()
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

trait MyInnerResponseExt {
    fn doc(self, field: &str) -> Response;
}

impl<T> MyInnerResponseExt for egui::InnerResponse<T> {
    fn doc(self, field: &str) -> Response {
        self.response
            .on_hover_text(GuiState::get_field_comment(field).unwrap())
    }
}
impl MyInnerResponseExt for Response {
    fn doc(self, field: &str) -> Response {
        self.on_hover_text(GuiState::get_field_comment(field).unwrap())
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
    ) -> egui::InnerResponse<T>;
}

impl MyUiExt for Ui {
    fn my_text_field(&mut self, text: &mut String) -> Response {
        self.add_sized([100.0, self.available_height()], TextEdit::singleline(text))
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
    ) -> egui::InnerResponse<T> {
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
