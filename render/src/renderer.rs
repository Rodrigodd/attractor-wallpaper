use std::error::Error;

use image::RgbaImage;
use wgpu::{
    Backends, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BlendState, Color, ColorTargetState,
    ColorWrites, CommandEncoderDescriptor, Device, DeviceDescriptor, Extent3d, Face, Features,
    FilterMode, FragmentState, FrontFace, ImageCopyTexture, ImageDataLayout, Instance,
    InstanceDescriptor, Limits, LoadOp, MultisampleState, Operations, Origin3d,
    PipelineLayoutDescriptor, PolygonMode, PowerPreference, PrimitiveState, PrimitiveTopology,
    Queue, RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline,
    RenderPipelineDescriptor, RequestAdapterOptions, SamplerBindingType, SamplerDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, Surface, SurfaceConfiguration,
    SurfaceError, TextureAspect, TextureDescriptor, TextureDimension, TextureFormat,
    TextureSampleType, TextureUsages, TextureViewDescriptor, TextureViewDimension, VertexState,
};
use winit::window::Window;

pub struct Renderer {
    surface: Surface,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
    // SAFETY: window must come after surface, because surface must be dropped before window.
    pub window: Window,
    render_pipeline: RenderPipeline,
    diffuse_bind_group: BindGroup,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    diffuse_texture: wgpu::Texture,
}

impl Renderer {
    pub async fn new(window: Window) -> Result<Self, Box<dyn Error>> {
        let size = window.inner_size();

        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            dx12_shader_compiler: Default::default(),
        });

        // SAFETY: The surface need to be dropped before the window. This is ensured by owning the
        // window in the struct and by the order of the fields.
        let surface = unsafe { instance.create_surface(&window)? };

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    features: Features::empty(),
                    // limits: if cfg!(target_arch = "wasm32") {
                    //     Limits::downlevel_webgl2_defaults()
                    // } else {
                    //     Limits::default()
                    // },
                    limits: Limits::downlevel_webgl2_defaults(),
                    label: None,
                },
                None,
            )
            .await?;

        device.on_uncaptured_error(Box::new(|e| {
            log::error!("wgpu error: {:?}", e);
        }));

        let surface_caps = surface.get_capabilities(&adapter);

        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let texture_size = Extent3d {
            width: size.width,
            height: size.height,
            depth_or_array_layers: 1,
        };
        let diffuse_texture = device.create_texture(&TextureDescriptor {
            label: Some("diffuse_texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("texture_bind_group_layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let diffuse_texture_view = diffuse_texture.create_view(&TextureViewDescriptor::default());
        let diffuse_sampler = device.create_sampler(&SamplerDescriptor {
            mag_filter: FilterMode::Linear,
            ..Default::default()
        });
        let diffuse_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("diffuse_bind_group"),
            layout: &texture_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&diffuse_texture_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&diffuse_sampler),
                },
            ],
        });

        let source = std::fs::read_to_string("render/src/shader.wgsl")?;
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Shader"),
            source: ShaderSource::Wgsl(source.into()),
        });

        let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Shader"),
            layout: Some(&render_pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(ColorTargetState {
                    format: config.format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
            window,
            render_pipeline,
            texture_bind_group_layout,
            diffuse_texture,
            diffuse_bind_group,
        })
    }

    pub fn destroy(self) -> Window {
        self.window
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            let texture_size = Extent3d {
                width: self.size.width,
                height: self.size.height,
                depth_or_array_layers: 1,
            };
            let diffuse_texture = self.device.create_texture(&TextureDescriptor {
                label: Some("diffuse_texture"),
                size: texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8UnormSrgb,
                usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });

            // self.queue.write_texture(
            //     ImageCopyTexture {
            //         texture: &diffuse_texture,
            //         mip_level: 0,
            //         origin: Origin3d::ZERO,
            //         aspect: TextureAspect::All,
            //     },
            //     &texture_data,
            //     ImageDataLayout {
            //         offset: 0,
            //         bytes_per_row: Some(4 * texture_data.width()),
            //         rows_per_image: Some(texture_data.height()),
            //     },
            //     texture_size,
            // );

            let diffuse_texture_view =
                diffuse_texture.create_view(&TextureViewDescriptor::default());
            let diffuse_sampler = self.device.create_sampler(&SamplerDescriptor {
                mag_filter: FilterMode::Linear,
                ..Default::default()
            });
            self.diffuse_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("diffuse_bind_group"),
                layout: &self.texture_bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&diffuse_texture_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Sampler(&diffuse_sampler),
                    },
                ],
            });
        }
    }

    pub fn render(&mut self, texture_data: Option<&RgbaImage>) -> Result<(), SurfaceError> {
        let output = self.surface.get_current_texture()?;

        let view = output
            .texture
            .create_view(&TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);

            render_pass.draw(0..3, 0..1);
        }

        if let Some(texture_data) = texture_data {
            if texture_data.width() != self.size.width || texture_data.height() != self.size.height
            {
                log::error!("Image size is wrong!!");
            } else {
                let texture_size = Extent3d {
                    width: texture_data.width(),
                    height: texture_data.height(),
                    depth_or_array_layers: 1,
                };
                self.queue.write_texture(
                    ImageCopyTexture {
                        texture: &self.diffuse_texture,
                        mip_level: 0,
                        origin: Origin3d::ZERO,
                        aspect: TextureAspect::All,
                    },
                    texture_data,
                    ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * texture_data.width()),
                        rows_per_image: Some(texture_data.height()),
                    },
                    texture_size,
                );
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
