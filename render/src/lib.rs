use std::path::PathBuf;

use attractors::map_bounds_affine;
use clap::Parser;
use rand::{Rng, SeedableRng};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoopBuilder, EventLoopProxy},
    platform::wayland::WindowBuilderExtWayland,
    window::WindowBuilder,
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum RenderBackend {
    Cpu,
    Gpu,
}

#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum AntiAliasing {
    None,
    Bilinear,
    Lanczos,
}
impl AntiAliasing {
    fn into_attractors_antialiasing(self) -> attractors::AntiAliasing {
        match self {
            Self::None => attractors::AntiAliasing::None,
            Self::Bilinear => attractors::AntiAliasing::Bilinear,
            Self::Lanczos => attractors::AntiAliasing::Lanczos,
        }
    }
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// The backend to use for rendering.
    #[arg(short, long, default_value = "cpu")]
    backend: RenderBackend,

    /// Spawn a window in fullscreen mode. In headless mode, make the output image the same size as
    /// the focused monitor.
    #[arg(short, long, default_value = "false")]
    fullscreen: bool,

    /// Enable anti-aliasing.
    #[arg(short, long, default_value = "none")]
    anti_aliasing: AntiAliasing,

    /// The seed to use for the sequence of generated attractors.
    #[arg(short, long)]
    seed: Option<u64>,

    /// Render the attractor into a buffer with both dimensions scaled by this factor, and them
    /// downsample it to the expected size. Used for anti-aliasing.
    #[arg(short, long, default_value = "1")]
    multisampling: u8,

    /// Renders in headless mode, and outputs the attractor to the given file.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// The dimensions for the rendered attractor, in pixels.
    #[arg(short, long, default_value = "512x512", value_parser = size_value_parser, conflicts_with = "fullscreen")]
    dimensions: winit::dpi::PhysicalSize<u32>,
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

mod executor;
mod renderer;

pub use crate::{
    executor::{TaskId, WinitExecutor},
    renderer::{AttractorRenderer, SurfaceState, WgpuState},
};

fn init_logger() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Info).unwrap();
        } else {
            env_logger::init();
        }
    }
}

enum UserEvent {
    PollTask(executor::TaskId),
    RebuildRenderer((WgpuState, SurfaceState, AttractorRenderer)),
}

pub async fn run(mut cli: Cli) {
    if let Some(output) = cli.output.take() {
        run_headless(cli, output).await
    } else {
        run_windowed(cli).await
    }
}

pub async fn run_headless(mut cli: Cli, output: PathBuf) {
    if cli.fullscreen {
        let ev = winit::event_loop::EventLoop::new();
        let monitor = match ev.primary_monitor() {
            Some(x) => x,
            None => {
                let Some(x) = ev.available_monitors().next()else {
                    println!("ERROR: No monitors found");
                    return;
                };
                println!("WARN: No primary monitor found, falling back to first available monitor");
                x
            }
        };
        cli.dimensions = monitor.size();
    }

    let wgpu_state = WgpuState::new_headless().await.unwrap();
    let mut renderer = AttractorRenderer::new(
        &wgpu_state.device,
        cli.dimensions,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        cli.multisampling,
    )
    .unwrap();

    let size = cli.dimensions;

    let (mut attractor, mut bitmap, _, base_intensity) = gen_attractor(
        size.width as usize,
        size.height as usize,
        cli.seed.unwrap_or_else(|| rand::rngs::OsRng.gen()),
        cli.multisampling,
    );

    let start = std::time::Instant::now();

    let total_samples = 400_000_000;
    let max = attractors::aggregate_to_bitmap(
        &mut attractor,
        size.width as usize * cli.multisampling as usize,
        size.height as usize * cli.multisampling as usize,
        total_samples,
        cli.anti_aliasing.into_attractors_antialiasing(),
        &mut bitmap,
    );

    println!(
        "Rendered {} samples in {}s",
        total_samples,
        start.elapsed().as_secs_f32()
    );

    if max == i32::MAX {
        println!("max reached");
    }

    bitmap[0] = get_intensity(
        base_intensity,
        total_samples,
        size,
        cli.multisampling,
        cli.anti_aliasing.into_attractors_antialiasing(),
    );
    renderer.load_aggragate_buffer(&wgpu_state.queue, &bitmap);

    let texture = wgpu_state.new_target_texture(size);

    let view = texture.create_view(&Default::default());
    renderer.render(
        &wgpu_state.device,
        &wgpu_state.queue,
        cli.backend == RenderBackend::Gpu,
        &view,
    );

    let bitmap = wgpu_state.copy_texture_content(texture);

    image::save_buffer(
        output,
        &bitmap,
        size.width,
        size.height,
        image::ColorType::Rgba8,
    )
    .unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run_windowed(cli: Cli) {
    init_logger();

    log::info!("Initializing window()...");

    let mut screen_size = winit::dpi::PhysicalSize::new(768, 512);

    let event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();
    let wb = WindowBuilder::new()
        .with_inner_size(screen_size)
        .with_title("My WGPU App");

    let wb = { wb.with_name("dev", "") };

    let wb = if cli.fullscreen {
        wb.with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)))
    } else {
        wb
    };

    let window = wb.build(&event_loop).unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        // Winit prevents sizing with CSS, so we have to set
        // the size manually when on web.
        use winit::dpi::PhysicalSize;
        window().set_inner_size(PhysicalSize::new(450, 400));

        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.get_element_by_id("wasm-example")?;
                let canvas = web_sys::Element::from(window().canvas());
                dst.append_child(&canvas).ok()?;
                Some(())
            })
            .expect("Couldn't append canvas to document body.");
    }

    let (mut wgpu_state, surface) = WgpuState::new_windowed(window).await.unwrap();
    let mut renderer = AttractorRenderer::new(
        &wgpu_state.device,
        surface.size(),
        surface.texture_format(),
        cli.multisampling,
    )
    .map_err(|err| log::error!("{}", err))
    .unwrap();

    let mut surface = Some(surface);

    let mut tasks = WinitExecutor::new(event_loop.create_proxy(), UserEvent::PollTask);
    let event_loop_proxy = event_loop.create_proxy();

    let mut frame_times = Vec::new();
    let mut last_frame_time: Option<std::time::Instant> = None;

    let mut seed = if let Some(seed) = cli.seed {
        seed
    } else {
        rand::rngs::OsRng.gen()
    };

    let mut modifiers = ModifiersState::empty();

    let (mut attractor, mut bitmap, mut total_samples, mut base_intensity) = gen_attractor(
        screen_size.width as usize,
        screen_size.height as usize,
        seed,
        cli.multisampling,
    );

    renderer.load_attractor(&wgpu_state.queue, &attractor);

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::UserEvent(UserEvent::RebuildRenderer((w, s, r))) => {
                renderer = r;
                surface = Some(s);
                wgpu_state = w;
                *control_flow = ControlFlow::Poll;
                return;
            }
            Event::WindowEvent {
                event:
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::R),
                                ..
                            },
                        ..
                    },
                ..
            } => {
                let window = surface.take().unwrap().destroy();
                log::info!("rebuilding renderer...");
                rebuild_renderer(event_loop_proxy.clone(), window, &mut tasks, &cli);
                return;
            }
            _ => (),
        };
        let Some(state) = surface.as_mut() else {
            return;
        };

        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window().id() => match event {
                &WindowEvent::Resized(physical_size)
                | WindowEvent::ScaleFactorChanged {
                    new_inner_size: &mut physical_size,
                    ..
                } => {
                    screen_size = physical_size;

                    (attractor, bitmap, total_samples, base_intensity) = gen_attractor(
                        screen_size.width as usize,
                        screen_size.height as usize,
                        seed,
                        cli.multisampling,
                    );

                    renderer.resize(&wgpu_state.device, physical_size);
                    state.resize(physical_size, &wgpu_state.device);
                    renderer.load_attractor(&wgpu_state.queue, &attractor);
                }
                WindowEvent::CloseRequested
                | WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::Q),
                            ..
                        },
                    ..
                } => *control_flow = ControlFlow::Exit,
                WindowEvent::ModifiersChanged(m) => modifiers = *m,
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(virtual_keycode),
                            ..
                        },
                    ..
                } => match virtual_keycode {
                    VirtualKeyCode::R => {
                        let window = surface.take().unwrap().destroy();
                        log::info!("rebuilding renderer...");
                        rebuild_renderer(event_loop_proxy.clone(), window, &mut tasks, &cli);
                    }
                    VirtualKeyCode::NumpadEnter | VirtualKeyCode::Return if modifiers.alt() => {
                        println!("toggling fullscreen");
                        if state.window().fullscreen().is_some() {
                            state.window().set_decorations(true);
                            state.window().set_fullscreen(None);
                        } else {
                            state.window().set_decorations(false);
                            state
                                .window()
                                .set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
                        }
                        state.window().set_fullscreen(None);
                    }
                    _ => {}
                },
                WindowEvent::MouseInput {
                    state: ElementState::Pressed,
                    button: MouseButton::Left,
                    ..
                } => {
                    seed += 1;
                    (attractor, bitmap, total_samples, base_intensity) = gen_attractor(
                        screen_size.width as usize,
                        screen_size.height as usize,
                        seed,
                        cli.multisampling,
                    );
                    renderer.load_attractor(&wgpu_state.queue, &attractor);
                    println!("attractor: {:?}", attractor);
                }
                _ => {}
            },
            Event::RedrawRequested(window_id) if window_id == state.window().id() => {
                if let Some(last_frame) = last_frame_time {
                    frame_times.push(last_frame.elapsed());
                    if frame_times.len() > 30 {
                        let sum: std::time::Duration = frame_times.iter().sum();
                        let avg = sum / frame_times.len() as u32;
                        println!(
                            "avg frame time: {:.2?} (after {:e} samples)",
                            avg, total_samples as f64
                        );
                        frame_times.clear();
                    }
                }
                last_frame_time = Some(std::time::Instant::now());

                if let RenderBackend::Cpu = cli.backend {
                    let samples = 400_000;
                    let size = state.window().inner_size();
                    let max = attractors::aggregate_to_bitmap(
                        &mut attractor,
                        size.width as usize * cli.multisampling as usize,
                        size.height as usize * cli.multisampling as usize,
                        samples,
                        cli.anti_aliasing.into_attractors_antialiasing(),
                        &mut bitmap,
                    );
                    if max == i32::MAX {
                        println!("max reached");
                    }
                    total_samples += samples;
                    bitmap[0] = get_intensity(
                        base_intensity,
                        total_samples,
                        size,
                        cli.multisampling,
                        cli.anti_aliasing.into_attractors_antialiasing(),
                    );
                    renderer.load_aggragate_buffer(&wgpu_state.queue, &bitmap);
                }

                match state.current_texture() {
                    Ok(texture) => {
                        let view = texture.texture.create_view(&Default::default());
                        renderer.render(
                            &wgpu_state.device,
                            &wgpu_state.queue,
                            cli.backend == RenderBackend::Gpu,
                            &view,
                        );
                        texture.present();
                    }
                    Err(e) => {
                        eprintln!("{:?}", e);
                        match e {
                            wgpu::SurfaceError::Timeout => {}
                            wgpu::SurfaceError::Outdated => {}
                            wgpu::SurfaceError::Lost => {
                                state.resize(cli.dimensions, &wgpu_state.device)
                            }
                            wgpu::SurfaceError::OutOfMemory => {
                                let window = surface.take().unwrap().destroy();
                                rebuild_renderer(event_loop_proxy.clone(), window, &mut tasks, &cli)
                            }
                        }
                    }
                }
            }
            Event::MainEventsCleared => {
                state.window().request_redraw();
            }
            Event::UserEvent(UserEvent::PollTask(task_id)) => tasks.poll(task_id),
            _ => {}
        }
    });
}

pub fn get_intensity(
    base_intensity: i16,
    total_samples: u64,
    size: winit::dpi::PhysicalSize<u32>,
    multisampling: u8,
    antialiasing: attractors::AntiAliasing,
) -> i32 {
    let p = match antialiasing {
        attractors::AntiAliasing::None => 1,
        attractors::AntiAliasing::Bilinear => 64,
        attractors::AntiAliasing::Lanczos => 64,
    };

    (base_intensity as u64 * total_samples * 32 * p
        / size.width as u64
        / size.height as u64
        / multisampling as u64
        / multisampling as u64) as i32
}

pub fn gen_attractor(
    width: usize,
    height: usize,
    seed: u64,
    multisampling: u8,
) -> (attractors::Attractor, Vec<i32>, u64, i16) {
    let rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let mut attractor = attractors::Attractor::find_strange_attractor(rng, 1_000_000).unwrap();

    let border = 15.0;
    let multisampling = multisampling as usize;

    attractor_to_within_border(
        &mut attractor,
        border * multisampling as f64,
        width * multisampling,
        height * multisampling,
    );
    let bitmap = vec![0i32; width * multisampling * height * multisampling];

    let base_intensity = attractors::get_base_intensity(&attractor);

    (attractor, bitmap, 0u64, base_intensity)
}

pub fn resize_attractor(
    attractor: &mut attractors::Attractor,
    old_size: (usize, usize),
    new_size: (usize, usize),
) -> (Vec<i32>, u64) {
    let border = 0.0;

    let affine = map_bounds_affine(
        [
            border,
            new_size.0 as f64 - border,
            border,
            new_size.1 as f64 - border,
        ],
        [
            border,
            old_size.0 as f64 - border,
            border,
            old_size.1 as f64 - border,
        ],
    );

    *attractor = attractor.transform_input(affine);

    let bitmap = vec![0i32; new_size.0 * new_size.1];

    (bitmap, 0u64)
}

fn attractor_to_within_border(
    attractor: &mut attractors::Attractor,
    border: f64,
    width: usize,
    height: usize,
) {
    let points = attractor.get_points::<512>();

    // 4 KiB
    let affine = attractors::affine_from_pca(&points);
    *attractor = attractor.transform_input(affine);

    let bounds = attractor.get_bounds(512);
    let dst = [
        border,
        width as f64 - border,
        border,
        height as f64 - border,
    ];
    let affine = attractors::map_bounds_affine(dst, bounds);

    *attractor = attractor.transform_input(affine);
}

fn rebuild_renderer(
    event_loop_proxy: EventLoopProxy<UserEvent>,
    window: winit::window::Window,
    tasks: &mut WinitExecutor<UserEvent>,
    cli: &Cli,
) {
    let multisampling = cli.multisampling;
    let task = async move {
        let Ok(( wgpu_state, surface )) = WgpuState::new_windowed(window).await.map_err(|err| log::error!("{}", err)) else {
            return;
        };
        let renderer = AttractorRenderer::new(
            &wgpu_state.device,
            surface.size(),
            wgpu::TextureFormat::Rgba8UnormSrgb,
            multisampling,
        )
        .unwrap();
        event_loop_proxy
            .send_event(UserEvent::RebuildRenderer((wgpu_state, surface, renderer)))
            .unwrap_or_else(|_| panic!("failed to send event"));
    };
    tasks.spawn(task);
}
