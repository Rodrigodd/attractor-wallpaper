use rand::SeedableRng;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoopBuilder, EventLoopProxy},
    platform::wayland::WindowBuilderExtWayland,
    window::WindowBuilder,
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

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
pub async fn run() {
    init_logger();

    log::info!("Initializing window...");

    let event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();
    let wb = WindowBuilder::new()
        .with_inner_size(winit::dpi::PhysicalSize::new(512, 512))
        .with_title("My WGPU App");

    let wb = { wb.with_name("dev", "") };

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

    let mut render_image = false;

    let mut frame_times = Vec::new();
    let mut last_frame_time: Option<std::time::Instant> = None;

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
                WindowEvent::Resized(physical_size) => {
                    state.resize(*physical_size);
                }
                WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    state.resize(**new_inner_size);
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
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::R),
                            ..
                        },
                    ..
                } => {
                    let window = some_state.take().unwrap().destroy();
                    log::info!("rebuilding renderer...");
                    rebuild_renderer(event_loop_proxy.clone(), window, &mut tasks);
                }
                WindowEvent::MouseInput {
                    state: ElementState::Pressed,
                    button: MouseButton::Left,
                    ..
                } => render_image = true,
                _ => {}
            },
            Event::RedrawRequested(window_id) if window_id == state.window.id() => {
                if let Some(last_frame) = last_frame_time {
                    frame_times.push(last_frame.elapsed());
                    if frame_times.len() > 30 {
                        let sum: std::time::Duration = frame_times.iter().sum();
                        let avg = sum / frame_times.len() as u32;
                        println!("avg frame time: {:?}", avg);
                        frame_times.clear();
                    }
                }
                last_frame_time = Some(std::time::Instant::now());

                let img = render_image.then(|| {
                    let rng = rand::rngs::SmallRng::from_entropy();

                    let attractor =
                        attractors::Attractor::find_strange_attractor(rng, 1_000_000).unwrap();
                    let points = attractor.get_points::<512>(); // 4 KiB
                    let affine = attractors::affine_from_pca(&points);
                    let attractor = attractor.transform_input(affine);
                    let bounds = attractor.get_bounds(512);
                    let border = 15.0;
                    let width = 512;
                    let height = 512;
                    let dst = [
                        border,
                        width as f64 - border,
                        border,
                        height as f64 - border,
                    ];
                    let affine = attractors::map_bounds_affine(dst, bounds);
                    let attractor = attractor.transform_input(affine);

                    println!("attractor: {:?}", attractor);

                    let img = attractors::render_to_bitmap(
                        &attractor,
                        state.size.width as usize,
                        state.size.height as usize,
                        1_000_000,
                    );
                    let img = image::GrayImage::from_raw(state.size.width, state.size.height, img)
                        .unwrap();
                    const PALETTE: [(u8, u8, u8); 256] = {
                        let mut pallete = [(0, 0, 0); 256];
                        let mut i = 0;
                        loop {
                            pallete[i] = (i as u8, i as u8, i as u8);
                            if i == 255 {
                                break;
                            }
                            i += 1;
                        }
                        pallete
                    };
                    img.expand_palette(&PALETTE, None)
                });
                render_image = false;
                match state.render() {
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
