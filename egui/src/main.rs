use std::time::Duration;

use attractors::AntiAliasing;
use egui::{ComboBox, Grid, Response, Slider, TextEdit, Ui};
use egui_wgpu::wgpu;
use egui_winit::winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoopBuilder, EventLoopProxy},
    platform::wayland::WindowBuilderExtWayland,
    window::{Window, WindowBuilder},
};
use rand::prelude::*;

use render::{AttractorRenderer, SurfaceState, TaskId, WgpuState, WinitExecutor};

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

struct GuiState {
    seed: u64,
    seed_text: String,
    multisampling: u8,
    multisampling_text: String,
    anti_aliasing: AntiAliasing,
    intensity: f32,
}
impl GuiState {
    fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
        self.seed_text = seed.to_string();
    }

    fn seed(&mut self) -> u64 {
        if let Ok(seed) = self.seed_text.parse::<u64>() {
            self.seed = seed;
        }
        self.seed
    }

    fn update_seed(&mut self) {
        if let Ok(seed) = self.seed_text.parse::<u64>() {
            self.seed = seed;
        }
    }

    fn multisampling(&mut self) -> u8 {
        self.multisampling
    }

    fn update_multisampling(&mut self) {
        if let Ok(multisampling) = self.multisampling_text.parse::<u8>() {
            self.multisampling = multisampling;
        }
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
        seed: 0,
        seed_text: 0.to_string(),
        multisampling: 1,
        multisampling_text: 1.to_string(),
        anti_aliasing: attractors::AntiAliasing::None,
        intensity: 1.0,
    };

    let (mut attractor, mut bitmap, mut total_samples, mut base_intensity) = render::gen_attractor(
        size.width as usize,
        size.height as usize,
        gui_state.seed(),
        gui_state.multisampling(),
    );

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

                if let WindowEvent::CloseRequested = event {
                    *control_flow = ControlFlow::Exit
                } else if let WindowEvent::Resized(new_size) = event {
                    let old_size = render_state.surface.size();
                    (bitmap, total_samples) = render::resize_attractor(
                        &mut attractor,
                        (
                            old_size.width as usize * gui_state.multisampling as usize,
                            old_size.height as usize * gui_state.multisampling as usize,
                        ),
                        (
                            new_size.width as usize * gui_state.multisampling as usize,
                            new_size.height as usize * gui_state.multisampling as usize,
                        ),
                    );

                    render_state
                        .attractor_renderer
                        .resize(&render_state.wgpu_state.device, new_size);
                    render_state
                        .surface
                        .resize(new_size, &render_state.wgpu_state.device);
                    render_state
                        .attractor_renderer
                        .load_attractor(&render_state.wgpu_state.queue, &attractor);
                }
            }
            Event::MainEventsCleared => {
                let samples = 400_000;
                let size = render_state.surface.size();
                attractors::aggregate_to_bitmap(
                    &mut attractor,
                    size.width as usize * gui_state.multisampling() as usize,
                    size.height as usize * gui_state.multisampling() as usize,
                    samples,
                    gui_state.anti_aliasing,
                    &mut bitmap,
                );
                total_samples += samples;
                bitmap[0] = render::get_intensity(
                    (base_intensity as f32 / gui_state.intensity) as i16,
                    total_samples,
                    size,
                    gui_state.multisampling,
                    gui_state.anti_aliasing,
                );
                render_state
                    .attractor_renderer
                    .load_aggragate_buffer(&render_state.wgpu_state.queue, &bitmap);

                let new_input = egui_state.take_egui_input(window);

                let mut full_output = egui_ctx.run(new_input, |ui| {
                    egui::Window::new("Hello world!")
                        .resizable(false) // could not figure out how make this work
                        .show(ui, |ui| {
                            build_ui(
                                ui,
                                &mut gui_state,
                                &mut attractor,
                                &mut bitmap,
                                &mut total_samples,
                                &mut base_intensity,
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

fn build_ui(
    ui: &mut Ui,
    gui_state: &mut GuiState,
    attractor: &mut attractors::Attractor,
    bitmap: &mut Vec<i32>,
    total_samples: &mut u64,
    base_intensity: &mut i16,
    render_state: &mut RenderState,
) -> egui::InnerResponse<()> {
    let size = render_state.surface.size();
    Grid::new("options_grid").show(ui, |ui| {
        ui.label("seed: ");
        ui.horizontal(|ui| {
            if ui.my_text_field(&mut gui_state.seed_text).lost_focus() {
                gui_state.update_seed();
                (*attractor, *bitmap, *total_samples, *base_intensity) = render::gen_attractor(
                    size.width as usize,
                    size.height as usize,
                    gui_state.seed(),
                    gui_state.multisampling(),
                );
            }

            if ui.button("rand").clicked() {
                gui_state.update_seed();
                gui_state.set_seed(rand::thread_rng().gen());
                (*attractor, *bitmap, *total_samples, *base_intensity) = render::gen_attractor(
                    size.width as usize,
                    size.height as usize,
                    gui_state.seed(),
                    gui_state.multisampling(),
                );
            }
        });

        ui.end_row();

        ui.label("multisampling: ");
        let prev_multisampling = gui_state.multisampling();
        if ui
            .my_text_field(&mut gui_state.multisampling_text)
            .lost_focus()
        {
            gui_state.update_multisampling();
            render_state.attractor_renderer.recreate_aggregate_buffer(
                &render_state.wgpu_state.device,
                size,
                gui_state.multisampling(),
            );
            (*bitmap, *total_samples) = render::resize_attractor(
                attractor,
                (
                    size.width as usize * prev_multisampling as usize,
                    size.height as usize * prev_multisampling as usize,
                ),
                (
                    size.width as usize * gui_state.multisampling() as usize,
                    size.height as usize * gui_state.multisampling() as usize,
                ),
            );
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
            bitmap.fill(0);
            *total_samples = 0;
        }

        ui.end_row();

        ui.label("intensity: ");
        ui.add(Slider::new(&mut gui_state.intensity, 0.01..=2.0));
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
