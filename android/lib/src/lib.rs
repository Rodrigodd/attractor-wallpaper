use android_logger::FilterBuilder;
use oklab::{OkLch, Oklab, Srgb8};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use render::gradient::Gradient;
use render::{
    render_to_bitmap, update_render, AttractorConfig, AttractorCtx, AttractorMess,
    AttractorRenderer, SurfaceState, Theme, WgpuState,
};

use std::{
    error::Error,
    ops,
    sync::{
        mpsc::{Receiver, TryRecvError},
        Arc,
    },
    time::Instant,
};

use parking_lot::Mutex;

mod bindings;

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
                    .parse("info,attractor_android=trace,attractor=trace,render=trace")
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

enum ConfigKey {
    Seed(u64),
    Multisampling(u8),
    Intensity(f32),
    Exponent(f32),
    MinArea(f32),
    BackgroundColor1(u32),
    BackgroundColor2(u32),
}

enum Event {
    Created,
    Resized((u32, u32)),
    Destroyed,
    Redraw,
    UpdateConfig(ConfigKey),
    UpdateTheme(Theme),
}

struct Context {
    thread: std::thread::JoinHandle<()>,
    sender: std::sync::mpsc::Sender<Event>,
    config: Arc<Mutex<AttractorConfig>>,
}

fn on_create(window: NativeWindow) -> Result<Box<Context>, Box<dyn Error>> {
    let (sender, receiver) = std::sync::mpsc::channel();

    let mut render_state = pollster::block_on(build_renderer(window))?;

    let (attractor_sender, recv_bitmap, config) = init_attractor(&mut render_state);

    let config_clone = config.clone();
    let thread = std::thread::spawn(|| {
        main_loop(
            render_state,
            receiver,
            config_clone,
            attractor_sender,
            recv_bitmap,
        )
    });

    sender.send(Event::Created).expect("event channel");
    Ok(Box::new(Context {
        thread,
        sender,
        config,
    }))
}

fn on_resize(ctx: &Context, width: i32, height: i32) {
    ctx.sender
        .send(Event::Resized((width as u32, height as u32)))
        .expect("event channel");
    ctx.sender.send(Event::Redraw).expect("event channel");
}

fn on_destroy(ctx: Box<Context>) {
    ctx.sender.send(Event::Destroyed).expect("event channel");
    ctx.thread.join().expect("event channel");
}

fn on_redraw(ctx: &Context) {
    ctx.sender.send(Event::Redraw).expect("event channel");
}

fn on_update_config_int(ctx: &Context, key: &str, value: i32) {
    log::debug!("update config: {} = {}", key, value);
    let config_key = match key {
        "seed" => ConfigKey::Seed(value as u64),
        "multisampling" => ConfigKey::Multisampling(value as u8),
        "intensity" => ConfigKey::Intensity(value as f32 / 100.0),
        "exponent" => ConfigKey::Exponent(value as f32 / 100.0),
        "min_area" => ConfigKey::MinArea(value as f32 / 100.0),
        "background_color1" => ConfigKey::BackgroundColor1(value as u32),
        "background_color2" => ConfigKey::BackgroundColor2(value as u32),
        _ => {
            log::error!("unknown config key: {}", key);
            return;
        }
    };
    ctx.sender
        .send(Event::UpdateConfig(config_key))
        .expect("event channel");
}

fn u32_to_srgb8(color: u32) -> Srgb8 {
    let color = color.to_be_bytes();
    let color = [color[1], color[2], color[3]];
    Srgb8::from(color)
}

fn on_update_theme(ctx: &Context, theme: &[u8]) {
    // deserialize the theme:
    // [ background_color1:u32, background_color2:u32, [ t:f32, color:u32][5]]

    let mut offset = 0;
    let background_color_1 = u32::from_le_bytes(theme[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let background_color_2 = u32::from_le_bytes(theme[offset..offset + 4].try_into().unwrap());
    offset += 4;

    let background_color_1 = u32_to_srgb8(background_color_1);
    let background_color_2 = u32_to_srgb8(background_color_2);

    let mut gradient = [(0.0, Oklab::new(0.0, 0.0, 0.0)); 5];

    for (_, entry) in gradient.iter_mut().enumerate() {
        let t = f32::from_le_bytes(theme[offset..offset + 4].try_into().unwrap());
        offset += 4;
        let color = u32::from_le_bytes(theme[offset..offset + 4].try_into().unwrap());
        offset += 4;

        let color = u32_to_srgb8(color);

        *entry = (t, Oklab::from(color.to_f32()));
    }

    let gradient = Gradient::new(gradient.to_vec());

    let theme = Theme {
        background_color_1: background_color_1.to_f32().into(),
        background_color_2: background_color_2.to_f32().into(),
        gradient,
    };

    ctx.sender
        .send(Event::UpdateTheme(theme))
        .expect("event channel");
}

fn on_get_wallpaper(
    ctx: &Context,
    width: u32,
    height: u32,
    view_width: u32,
    _view_height: u32,
    bitmap: &mut [u8],
) {
    let mut config = ctx.config.lock().clone();
    log::debug!(
        "resizing from {}x{} to {}x{}",
        config.size.0,
        config.size.1,
        width,
        height
    );

    config.resize((width, height), config.multisampling);

    // the wallpaper image is much wider than the display size, so decrease the size of the
    // attractor to fit the display.
    // this scales the view, so the division is reversed.
    let scale = width as f64 / view_width as f64;
    config.transform.0[0] *= scale;
    config.transform.0[1] *= scale;
    config.transform.0[2] *= scale;
    config.transform.0[3] *= scale;
    config.transform.1[0] *= scale;
    config.transform.1[1] *= scale;

    config.samples_per_iteration = 100_000_000;

    let result = render_wallpaper(config);

    bitmap.clone_from_slice(&result);
}

fn init_attractor(
    render_state: &mut RenderState,
) -> (
    std::sync::mpsc::Sender<AttractorMess>,
    render::channel::Receiver<AttractorCtx>,
    Arc<Mutex<AttractorConfig>>,
) {
    let mut attractor_config = AttractorConfig {
        size: (1, 1),
        transform: ([1.0, 0.0, 0.0, 1.0], [0.0, 0.0]),

        intensity: 1.0,
        exponent: 1.0,
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
        attractor_config.intensity,
        attractor_config.exponent,
    );

    let attractor_config = Arc::new(Mutex::new(attractor_config));

    let (attractor_sender, recv_conf) = std::sync::mpsc::channel::<AttractorMess>();
    let (sender_bitmap, recv_bitmap) = render::channel::channel::<AttractorCtx>();
    let attractor = AttractorCtx::new(attractor_config.clone());

    std::thread::spawn(move || render::attractor_thread(recv_conf, attractor, sender_bitmap));
    (attractor_sender, recv_bitmap, attractor_config)
}

fn main_loop(
    mut render_state: RenderState,
    events: Receiver<Event>,
    config: Arc<Mutex<AttractorConfig>>,
    attractor_sender: std::sync::mpsc::Sender<AttractorMess>,
    mut recv_bitmap: render::channel::Receiver<AttractorCtx>,
) {
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
            Event::UpdateConfig(config_key) => {
                redraw = true;
                match config_key {
                    ConfigKey::Seed(seed) => {
                        let _ = attractor_sender.send(AttractorMess::SetSeed(seed));
                    }
                    ConfigKey::Multisampling(multisampling) => {
                        let _ =
                            attractor_sender.send(AttractorMess::SetMultisampling(multisampling));
                        render_state.attractor_renderer.recreate_aggregate_buffer(
                            &render_state.wgpu_state.device,
                            render_state.attractor_renderer.size,
                            multisampling,
                        );
                    }
                    ConfigKey::Intensity(intensity) => {
                        let _ = attractor_sender.send(AttractorMess::SetIntensity(intensity));
                        render_state
                            .attractor_renderer
                            .set_intensity(&render_state.wgpu_state.queue, intensity);
                    }
                    ConfigKey::Exponent(exponent) => {
                        let _ = attractor_sender.send(AttractorMess::SetExponent(exponent));
                        render_state
                            .attractor_renderer
                            .set_exponent(&render_state.wgpu_state.queue, exponent);
                    }
                    ConfigKey::MinArea(min_area) => {
                        let min_area = (min_area * 4096.0).round() as u16;
                        let _ = attractor_sender.send(AttractorMess::SetMinArea(min_area));
                    }
                    ConfigKey::BackgroundColor1(color) => {
                        let color = color.to_be_bytes();
                        let color = [color[1], color[2], color[3]];
                        let color = oklab::Srgb8::from(color);
                        let mut config = config.lock();
                        config.background_color_1 = color.to_f32().into();
                        update_render(
                            &mut render_state.attractor_renderer,
                            &render_state.wgpu_state,
                            &config.gradient,
                            config.multisampling,
                            config.background_color_1,
                            config.background_color_2,
                            config.intensity,
                            config.exponent,
                        );
                    }
                    ConfigKey::BackgroundColor2(color) => {
                        let color = color.to_be_bytes();
                        let color = [color[1], color[2], color[3]];
                        let color = oklab::Srgb8::from(color);
                        let mut config = config.lock();
                        config.background_color_2 = color.to_f32().into();
                        update_render(
                            &mut render_state.attractor_renderer,
                            &render_state.wgpu_state,
                            &config.gradient,
                            config.multisampling,
                            config.background_color_1,
                            config.background_color_2,
                            config.intensity,
                            config.exponent,
                        );
                    }
                }
                // wait for the attractor to update to start redrawing
                recv_bitmap.recv(|_| {});
            }
            Event::UpdateTheme(theme) => {
                redraw = true;
                log::debug!("Updating theme: {:?}", theme);
                {
                    let mut config = config.lock();
                    config.gradient = theme.gradient;
                    config.background_color_1 = theme.background_color_1;
                    config.background_color_2 = theme.background_color_2;
                    update_render(
                        &mut render_state.attractor_renderer,
                        &render_state.wgpu_state,
                        &config.gradient,
                        config.multisampling,
                        config.background_color_1,
                        config.background_color_2,
                        config.intensity,
                        config.exponent,
                    );
                }
                // wait for the attractor to update to start redrawing
                recv_bitmap.recv(|_| {});
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

                    let base_intensity = config.base_intensity as f32;
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

fn render_wallpaper(config: AttractorConfig) -> Vec<u8> {
    log::info!("rendering wallpaper");
    let multithreaded = config.multithreading;
    let multisampling = config.multisampling;
    let size = config.size;

    let mut attractor = AttractorCtx::new(Arc::new(Mutex::new(config)));

    attractor.resize((size.0, size.1), multisampling);

    render::aggregate_buffer(multithreaded, &mut attractor);

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
        intensity,
        exponent,
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
            intensity,
            exponent,
        )
        .await;

        bitmap
    })
}
