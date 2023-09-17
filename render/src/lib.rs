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

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// The backend to use for rendering.
    #[arg(short, long, default_value = "cpu")]
    backend: RenderBackend,

    /// Spawn a window in fullscreen mode.
    #[arg(short, long, default_value = "false")]
    fullscreen: bool,
}

mod renderer;
use renderer::Renderer;
mod executor;
use executor::WinitExecutor;

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

pub enum UserEvent {
    PollTask(executor::TaskId),
    SetRenderer(Renderer),
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run(cli: Cli) {
    init_logger();

    log::info!("Initializing window...");

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
        window.set_inner_size(PhysicalSize::new(450, 400));

        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.get_element_by_id("wasm-example")?;
                let canvas = web_sys::Element::from(window.canvas());
                dst.append_child(&canvas).ok()?;
                Some(())
            })
            .expect("Couldn't append canvas to document body.");
    }

    let mut some_state = Renderer::new(window)
        .await
        .map_err(|err| log::error!("{}", err))
        .ok();

    let mut tasks = WinitExecutor::new(event_loop.create_proxy());
    let event_loop_proxy = event_loop.create_proxy();

    let mut frame_times = Vec::new();
    let mut last_frame_time: Option<std::time::Instant> = None;

    let mut seed = rand::rngs::OsRng.gen();
    let gen_attractor = |width, height, seed: u64| {
        let rng = rand::rngs::SmallRng::seed_from_u64(seed);
        let border = 15.0;

        let attractor = attractor_within_border(rng, border, width, height);
        let bitmap = vec![0u32; width * height];
        (attractor, bitmap, 0u64)
    };

    let mut modifiers = ModifiersState::empty();

    let (mut attractor, mut bitmap, mut total_samples) = gen_attractor(
        screen_size.width as usize,
        screen_size.height as usize,
        seed,
    );

    if let Some(state) = &mut some_state {
        state.load_attractor(&attractor);
    }

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::UserEvent(UserEvent::SetRenderer(renderer)) => {
                some_state = Some(renderer);
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
                let window = some_state.take().unwrap().destroy();
                log::info!("rebuilding renderer...");
                rebuild_renderer(event_loop_proxy.clone(), window, &mut tasks);
                return;
            }
            _ => (),
        };
        let Some(state) = some_state.as_mut() else {
            return;
        };

        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window.id() => match event {
                &WindowEvent::Resized(physical_size)
                | WindowEvent::ScaleFactorChanged {
                    new_inner_size: &mut physical_size,
                    ..
                } => {
                    screen_size = physical_size;

                    (attractor, bitmap, total_samples) = gen_attractor(
                        screen_size.width as usize,
                        screen_size.height as usize,
                        seed,
                    );

                    state.resize(physical_size);
                    state.load_attractor(&attractor);
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
                        let window = some_state.take().unwrap().destroy();
                        log::info!("rebuilding renderer...");
                        rebuild_renderer(event_loop_proxy.clone(), window, &mut tasks);
                    }
                    VirtualKeyCode::NumpadEnter | VirtualKeyCode::Return if modifiers.alt() => {
                        println!("toggling fullscreen");
                        if state.window.fullscreen().is_some() {
                            state.window.set_decorations(true);
                            state.window.set_fullscreen(None);
                        } else {
                            state.window.set_decorations(false);
                            state
                                .window
                                .set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
                        }
                        state.window.set_fullscreen(None);
                    }
                    _ => {}
                },
                WindowEvent::MouseInput {
                    state: ElementState::Pressed,
                    button: MouseButton::Left,
                    ..
                } => {
                    seed += 1;
                    (attractor, bitmap, total_samples) = gen_attractor(
                        screen_size.width as usize,
                        screen_size.height as usize,
                        seed,
                    );
                    state.load_attractor(&attractor);
                    println!("attractor: {:?}", attractor);
                }
                _ => {}
            },
            Event::RedrawRequested(window_id) if window_id == state.window.id() => {
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
                    let max = attractors::aggregate_to_bitmap(
                        &mut attractor,
                        state.size.width as usize,
                        state.size.height as usize,
                        samples,
                        &mut bitmap,
                    );
                    total_samples += samples as u64;
                    bitmap[0] = max;
                    state.load_aggragate_buffer(&bitmap);
                }

                // match state.render(false, range as f32 / 5000.0) {
                match state.render(cli.backend == RenderBackend::Gpu, 0.0) {
                    Ok(_) => {}
                    Err(e) => {
                        eprintln!("{:?}", e);
                        match e {
                            wgpu::SurfaceError::Timeout => {}
                            wgpu::SurfaceError::Outdated => {}
                            wgpu::SurfaceError::Lost => state.resize(state.size),
                            wgpu::SurfaceError::OutOfMemory => {
                                let window = some_state.take().unwrap().destroy();
                                rebuild_renderer(event_loop_proxy.clone(), window, &mut tasks)
                            }
                        }
                    }
                }
            }
            Event::MainEventsCleared => {
                state.window.request_redraw();
            }
            Event::UserEvent(UserEvent::PollTask(task_id)) => tasks.poll(task_id),
            _ => {}
        }
    });
}

fn attractor_within_border(
    rng: rand::rngs::SmallRng,
    border: f64,
    width: usize,
    height: usize,
) -> attractors::Attractor {
    let attractor = attractors::Attractor::find_strange_attractor(rng, 1_000_000).unwrap();
    let points = attractor.get_points::<512>();

    // 4 KiB
    let affine = attractors::affine_from_pca(&points);
    let attractor = attractor.transform_input(affine);

    let bounds = attractor.get_bounds(512);
    let dst = [
        border,
        width as f64 - border,
        border,
        height as f64 - border,
    ];
    let affine = attractors::map_bounds_affine(dst, bounds);

    attractor.transform_input(affine)
}

fn rebuild_renderer(
    event_loop_proxy: EventLoopProxy<UserEvent>,
    window: winit::window::Window,
    tasks: &mut WinitExecutor,
) {
    let task = async move {
        let Ok(renderer) = Renderer::new(window).await.map_err(|err| log::error!("{}", err)) else {
            return;
        };
        event_loop_proxy
            .send_event(UserEvent::SetRenderer(renderer))
            .unwrap_or_else(|_| panic!("failed to send event"));
    };
    tasks.spawn(task);
}
