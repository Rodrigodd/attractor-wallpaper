use std::error::Error;

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    Backends, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BlendState, Color, ColorTargetState, ColorWrites,
    CommandEncoderDescriptor, Device, DeviceDescriptor, Face, Features, FragmentState, FrontFace,
    Instance, InstanceDescriptor, Limits, LoadOp, MultisampleState, Operations,
    PipelineLayoutDescriptor, PolygonMode, PowerPreference, PrimitiveState, PrimitiveTopology,
    Queue, RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline,
    RenderPipelineDescriptor, RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource,
    ShaderStages, Surface, SurfaceConfiguration, SurfaceError, TextureUsages,
    TextureViewDescriptor, VertexState,
};
use winit::window::Window;

const NUM_OF_PARTICLES: u32 = 512;

pub struct Renderer {
    surface: Surface,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
    // SAFETY: window must come after surface, because surface must be dropped before window.
    pub window: Window,
    render_pipeline: RenderPipeline,
    compute_pipeline: wgpu::ComputePipeline,
    bind_group: BindGroup,
    aggregate_buffer: wgpu::Buffer,
    uniforms_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    particles: wgpu::Buffer,
    attractor_buffer: wgpu::Buffer,
}

impl Renderer {
    pub async fn new(window: Window) -> Result<Self, Box<dyn Error>> {
        let size = window.inner_size();

        let instance = Instance::new(InstanceDescriptor {
            // backends: Backends::all(),
            backends: Backends::GL,
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
                    limits: if cfg!(target_arch = "wasm32") {
                        Limits::downlevel_webgl2_defaults()
                    } else {
                        Limits::default()
                    },
                    // limits: Limits::downlevel_webgl2_defaults(),
                    label: None,
                },
                None,
            )
            .await?;

        // device.on_uncaptured_error(Box::new(|e| {
        //     log::error!("wgpu error: {:?}", e);
        // }));

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
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let aggregate_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size.width as u64 * size.height as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let particles = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: NUM_OF_PARTICLES as u64 * 2 * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniforms = Uniforms {
            screen_width: size.width,
            screen_height: size.height,
            color_scale: 1.0,
        };

        let uniforms_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
        });

        let attractor_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("attractor"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // uniform
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE | ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // aggregate_buffer
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE | ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // particles
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // attractor
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let bind_group = create_bind_group(
            &device,
            &bind_group_layout,
            &uniforms_buffer,
            &aggregate_buffer,
            &particles,
            &attractor_buffer,
        );

        let compute_pipeline = create_compute_pipeline(&device, &pipeline_layout)?;
        let render_pipeline = create_render_pipeline(&device, &config, &pipeline_layout)?;

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
            window,
            render_pipeline,
            compute_pipeline,
            bind_group,
            aggregate_buffer,
            uniforms_buffer,
            bind_group_layout,
            particles,
            attractor_buffer,
        })
    }

    pub fn destroy(self) -> Window {
        self.window
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 && new_size.height == 0 {
            return;
        }

        if new_size == self.size {
            return;
        }

        self.size = new_size;

        self.config.width = new_size.width;
        self.config.height = new_size.height;

        self.surface.configure(&self.device, &self.config);

        self.aggregate_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: new_size.width as u64 * new_size.height as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.bind_group = create_bind_group(
            &self.device,
            &self.bind_group_layout,
            &self.uniforms_buffer,
            &self.aggregate_buffer,
            &self.particles,
            &self.attractor_buffer,
        );
    }

    pub fn load_attractor(&mut self, attractor: &attractors::Attractor) {
        let attractor = AttractorF32 {
            a: attractor.a.map(|x| x as f32),
            b: attractor.b.map(|x| x as f32),
            start: attractor.start.map(|x| x as f32),
        };

        self.queue.write_buffer(
            &self.attractor_buffer,
            0,
            bytemuck::cast_slice(&[attractor]),
        );
        self.load_particles([attractor.start[0], attractor.start[1]]);
    }

    pub fn load_particles(&mut self, start: [f32; 2]) {
        let content = std::array::from_fn::<[f32; 2], { NUM_OF_PARTICLES as usize }, _>(|i| {
            let sx = start[0];
            let sy = start[1];
            let radius = 0.001;
            let w = (NUM_OF_PARTICLES as f32).sqrt().ceil() as usize;
            let dx = (i % w) as f32 / w as f32 * radius;
            let dy = (i / w) as f32 / w as f32 * radius;
            [sx + dx, sy + dy]
        });
        self.queue
            .write_buffer(&self.particles, 0, bytemuck::cast_slice(&content));
    }

    pub fn load_aggragate_buffer(&mut self, buffer: &[u32]) {
        self.queue
            .write_buffer(&self.aggregate_buffer, 0, bytemuck::cast_slice(buffer));
    }

    pub fn render(&mut self, compute: bool, color_scale: f32) -> Result<(), SurfaceError> {
        let output = self.surface.get_current_texture()?;

        let view = output
            .texture
            .create_view(&TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        self.queue.write_buffer(
            &self.uniforms_buffer,
            0,
            bytemuck::cast_slice(&[Uniforms {
                screen_width: self.size.width,
                screen_height: self.size.height,
                color_scale,
            }]),
        );

        if compute {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
            });

            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);

            compute_pass.dispatch_workgroups(NUM_OF_PARTICLES, 1, 1)
        }

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
            render_pass.set_bind_group(0, &self.bind_group, &[]);

            render_pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn create_bind_group(
    device: &Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    uniforms_buffer: &wgpu::Buffer,
    aggregate_buffer: &wgpu::Buffer,
    particles: &wgpu::Buffer,
    attractor_buffer: &wgpu::Buffer,
) -> BindGroup {
    device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: bind_group_layout,
        entries: &[
            // uniform
            BindGroupEntry {
                binding: 0,
                resource: uniforms_buffer.as_entire_binding(),
            },
            // aggregate_buffer
            BindGroupEntry {
                binding: 1,
                resource: aggregate_buffer.as_entire_binding(),
            },
            // particles
            BindGroupEntry {
                binding: 2,
                resource: particles.as_entire_binding(),
            },
            // attractor
            BindGroupEntry {
                binding: 3,
                resource: attractor_buffer.as_entire_binding(),
            },
        ],
    })
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    screen_width: u32,
    screen_height: u32,
    color_scale: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AttractorF32 {
    a: [f32; 6],
    b: [f32; 6],
    start: [f32; 2],
}

fn create_compute_pipeline(
    device: &Device,
    pipeline_layout: &wgpu::PipelineLayout,
) -> Result<wgpu::ComputePipeline, Box<dyn Error>> {
    let source = std::fs::read_to_string("render/src/compute.wgsl")?;
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("Shader"),
        source: ShaderSource::Wgsl(source.into()),
    });

    let compute_pipeline_layout =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

    Ok(compute_pipeline_layout)
}

fn create_render_pipeline(
    device: &Device,
    config: &SurfaceConfiguration,
    pipeline_layout: &wgpu::PipelineLayout,
) -> Result<RenderPipeline, Box<dyn Error>> {
    let source = std::fs::read_to_string("render/src/shader.wgsl")?;
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("Shader"),
        source: ShaderSource::Wgsl(source.into()),
    });

    let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("Shader"),
        layout: Some(pipeline_layout),
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

    Ok(render_pipeline)
}
