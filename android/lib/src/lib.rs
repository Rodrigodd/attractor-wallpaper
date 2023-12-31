use android_logger::FilterBuilder;
use oklab::{OkLch, Oklab};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use render::gradient::Gradient;
use render::{
    update_render, AttractorConfig, AttractorCtx, AttractorMess, AttractorRenderer, SurfaceState,
    WgpuState,
};

use std::error::Error;
use std::ops;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;

fn init() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        set_panic_hook();
        init_logger();
    });
}

fn init_logger() {
    android_logger::init_once(
        android_logger::Config::default()
            .with_max_level(log::LevelFilter::Trace)
            .with_filter(
                FilterBuilder::new()
                    .parse("info,attractor_android=trace,attractor=trace")
                    .build(),
            )
            .with_tag("attractor-rust"),
    );
}

fn set_panic_hook() {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info: &std::panic::PanicInfo<'_>| {
        info.payload()
            .downcast_ref::<&str>()
            .map(|msg| log::error!("panic occurred: {:}", msg));
        info.payload()
            .downcast_ref::<String>()
            .map(|msg| log::error!("panic occurred: {:}", msg));
        info.location().map(|loc| {
            log::error!(
                "panic occurred in file '{}' at line {}",
                loc.file(),
                loc.line()
            );
        });
        hook(info);
    }))
}

mod bindings {
    use jni::{
        objects::{JClass, JObject},
        sys::{jint, jlong},
        JNIEnv,
    };

    // JNI binding to io.github.rodrigodd.attractorwallpaper.AttractorSurfaceView#nativeSurfaceCreated(surface: Surface): Long
    #[no_mangle]
    pub extern "system" fn Java_io_github_rodrigodd_attractorwallpaper_AttractorSurfaceView_nativeSurfaceCreated(
        env: JNIEnv,
        _: JClass,
        surface: JObject,
    ) -> jlong {
        super::init();
        log::info!("nativeSurfaceCreated: {:?}", surface);

        let window = unsafe {
            ndk::native_window::NativeWindow::from_surface(
                env.get_native_interface(),
                surface.as_raw(),
            )
        };

        let Some(window) = window else {
            return 0;
        };

        let ctx = super::on_create(super::NativeWindow(window));

        let ctx = match ctx {
            Ok(ctx) => ctx,
            Err(err) => {
                log::error!("Failed to create context: {:?}", err);
                return 0;
            }
        };

        let ptr = Box::into_raw(ctx);

        log::debug!("nativeSurfaceCreated: {:p} {:?}", ptr, surface);

        ptr as usize as i64
    }

    // JNI binding to io.github.rodrigodd.attractorwallpaper.AttractorSurfaceView#nativeSurfaceChanged(ctx: Long, surface: Surface, format: int, width: int, height: int)
    #[no_mangle]
    pub extern "system" fn Java_io_github_rodrigodd_attractorwallpaper_AttractorSurfaceView_nativeSurfaceChanged(
        _env: JNIEnv,
        _: JClass,
        ctx: jlong,
        surface: JObject,
        format: jint,
        width: jint,
        height: jint,
    ) {
        let ctx = ctx as *mut super::Context;
        log::info!(
            "nativeSurfaceChanged: {:p} {:?} {} {} {}",
            ctx,
            surface,
            format,
            width,
            height
        );
        unsafe {
            super::on_resize(&mut *ctx, width, height);
        }
    }

    // JNI binding to io.github.rodrigodd.attractorwallpaper.AttractorSurfaceView#nativeSurfaceDestroyed(ctx: Long, surface: Surface)
    #[no_mangle]
    pub extern "system" fn Java_io_github_rodrigodd_attractorwallpaper_AttractorSurfaceView_nativeSurfaceDestroyed(
        _env: JNIEnv,
        _: JClass,
        ctx: jlong,
        surface: JObject,
    ) {
        let ctx = ctx as *mut super::Context;
        log::info!("nativeSurfaceDestroyed: {:p} {:?}", ctx, surface,);
        unsafe {
            super::on_destroy(Box::from_raw(ctx));
        }
    }
}

enum Event {
    Created,
    Resized((u32, u32)),
    Destroyed,
    Redraw,
}

struct Context {
    thread: std::thread::JoinHandle<()>,
    sender: std::sync::mpsc::Sender<Event>,
}

fn on_create(window: NativeWindow) -> Result<Box<Context>, Box<dyn Error>> {
    let (sender, receiver) = std::sync::mpsc::channel();

    let render_state = pollster::block_on(build_renderer(window))?;

    let thread = std::thread::spawn(|| main_loop(render_state, receiver));

    sender.send(Event::Created).expect("channel");
    Ok(Box::new(Context { thread, sender }))
}

fn on_resize(ctx: &mut Context, width: i32, height: i32) {
    ctx.sender
        .send(Event::Resized((width as u32, height as u32)))
        .expect("channel");
    ctx.sender.send(Event::Redraw).expect("channel");
}

fn on_destroy(ctx: Box<Context>) {
    ctx.sender.send(Event::Destroyed).expect("channel");
    ctx.thread.join().expect("channel");
}

fn main_loop(mut render_state: RenderState, events: Receiver<Event>) {
    let mut attractor_config = AttractorConfig {
        size: (1, 1),
        transform: ([1.0, 0.0, 0.0, 1.0], [0.0, 0.0]),

        intensity: 1.0,
        samples_per_iteration: 100_000,
        multisampling: 1,

        background_color_1: OkLch::new(0.27, 0.11, 0.07),
        background_color_2: OkLch::new(0.10, 0.04, 0.07),
        gradient: Gradient::new(vec![
            (0.00, Oklab::new(0.09, 0.02, 0.02)),
            (0.03, Oklab::new(0.25, 0.09, 0.05)),
            (0.30, Oklab::new(0.5, 0.18, 0.1)),
            (0.75, Oklab::new(0.92, -0.05, 0.19)),
            (1.00, Oklab::new(1.0, 0.0, 0.0)),
        ]),

        ..Default::default()
    };

    attractor_config.set_seed(rand::random::<u64>() % 1_000_000);

    update_render(
        &mut render_state.attractor_renderer,
        &render_state.wgpu_state,
        &attractor_config.gradient,
        attractor_config.multisampling,
        attractor_config.background_color_1,
        attractor_config.background_color_2,
    );

    let attractor_config = Arc::new(Mutex::new(attractor_config));

    let (attractor_sender, recv_conf) = std::sync::mpsc::channel::<AttractorMess>();
    let (sender_bitmap, mut recv_bitmap) = render::channel::channel::<AttractorCtx>();
    let attractor = AttractorCtx::new(attractor_config.clone());

    std::thread::spawn(move || render::attractor_thread(recv_conf, attractor, sender_bitmap));

    let mut redraw = false;
    loop {
        let event = match events.try_recv() {
            Ok(event) => event,
            Err(TryRecvError::Empty) => {
                if redraw {
                    redraw = false;
                    Event::Redraw
                } else {
                    match events.recv() {
                        Ok(event) => event,
                        Err(_) => break,
                    }
                }
            }
            Err(TryRecvError::Disconnected) => break,
        };

        match event {
            Event::Created => {
                log::info!("Created");
            }
            Event::Resized(new_size) => {
                let _ = attractor_sender.send(AttractorMess::Resize(new_size));

                render_state
                    .attractor_renderer
                    .resize(&render_state.wgpu_state.device, new_size);
                render_state
                    .surface
                    .resize(new_size, &render_state.wgpu_state.device);
            }
            Event::Redraw => {
                log::info!("Rendering Frame");

                let mut total_sampĺes = 0;
                let mut stop_time = Instant::now();
                redraw = true;
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

                    at.bitmap[0] =
                        render::get_intensity(base_intensity, mat, total_samples, anti_aliasing);
                    render_state
                        .attractor_renderer
                        .load_aggregate_buffer(&render_state.wgpu_state.queue, &at.bitmap);

                    total_sampĺes = at.total_samples;
                    if let Some(x) = at.stop_time {
                        stop_time = x;
                        redraw = false;
                    } else {
                        stop_time = Instant::now();
                    }
                });

                render_frame(&mut render_state);
            }
            Event::Destroyed => {
                log::info!("Destroyed");
                return;
            }
        }
    }
}

struct NativeWindow(ndk::native_window::NativeWindow);
unsafe impl HasRawWindowHandle for NativeWindow {
    fn raw_window_handle(&self) -> raw_window_handle::RawWindowHandle {
        self.0.raw_window_handle()
    }
}
unsafe impl HasRawDisplayHandle for NativeWindow {
    fn raw_display_handle(&self) -> raw_window_handle::RawDisplayHandle {
        raw_window_handle::AndroidDisplayHandle::empty().into()
    }
}
impl ops::Deref for NativeWindow {
    type Target = ndk::native_window::NativeWindow;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

struct RenderState {
    wgpu_state: WgpuState,
    surface: SurfaceState<NativeWindow>,
    attractor_renderer: AttractorRenderer,
}

async fn build_renderer(window: NativeWindow) -> Result<RenderState, Box<dyn Error>> {
    let size = (window.width() as u32, window.height() as u32);
    let (wgpu_state, surface) = WgpuState::new_windowed(window, size).await?;
    let attractor_renderer = AttractorRenderer::new(
        &wgpu_state.device,
        surface.size(),
        surface.texture_format(),
        1,
    )?;

    attractor_renderer.set_background_color(
        &wgpu_state.queue,
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 1.0],
    );

    Ok(RenderState {
        wgpu_state,
        surface,
        attractor_renderer,
    })
}

fn render_frame(render_state: &mut RenderState) {
    let queue = &render_state.wgpu_state.queue;
    let device = &render_state.wgpu_state.device;

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

    render_state.attractor_renderer.update_uniforms(queue);
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
    drop(render_pass);
    queue.submit(std::iter::once(encoder.finish()));
    texture.present();
}
