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
    ShaderStages, Surface, SurfaceConfiguration, TextureUsages, VertexState,
};
use winit::window::Window;

const NUM_OF_PARTICLES: u32 = 512;

pub struct SurfaceState {
    surface: Surface,
    // SAFETY: window must come after surface, because surface must be dropped before window.
    window: Window,
    config: SurfaceConfiguration,
}
impl SurfaceState {
    fn new(window: Window, instance: &Instance) -> Self {
        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            width: window.inner_size().width,
            height: window.inner_size().height,
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        Self {
            // SAFETY: The surface need to be dropped before the window. This is ensured by owning the
            // window in the struct and by the order of the fields.
            surface: unsafe { instance.create_surface(&window).unwrap() },
            window,
            config,
        }
    }

    fn set_configuration(&mut self, adapter: &wgpu::Adapter, device: &Device) {
        let surface_caps = self.surface.get_capabilities(adapter);

        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        self.config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: self.window.inner_size().width,
            height: self.window.inner_size().height,
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        self.surface.configure(device, &self.config);
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>, device: &Device) {
        self.config.width = new_size.width;
        self.config.height = new_size.height;

        self.surface.configure(device, &self.config);
    }

    pub fn destroy(self) -> Window {
        self.window
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn current_texture(&self) -> Result<wgpu::SurfaceTexture, wgpu::SurfaceError> {
        self.surface.get_current_texture()
    }
}

pub struct Renderer {
    device: Device,
    queue: Queue,
    pub size: winit::dpi::PhysicalSize<u32>,
    multisampling: u8,
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
    pub async fn new_windowed(
        window: Window,
        multisampling: u8,
    ) -> Result<(Self, SurfaceState), Box<dyn Error>> {
        let size = window.inner_size();
        Self::new(Some(window), size, multisampling)
            .await
            .map(|(renderer, surface)| (renderer, surface.unwrap()))
    }

    pub async fn new_headless(
        size: winit::dpi::PhysicalSize<u32>,
        multisampling: u8,
    ) -> Result<Self, Box<dyn Error>> {
        Self::new(None, size, multisampling)
            .await
            .map(|(renderer, surface)| {
                assert!(surface.is_none());
                renderer
            })
    }

    async fn new(
        window: Option<Window>,
        size: winit::dpi::PhysicalSize<u32>,
        multisampling: u8,
    ) -> Result<(Self, Option<SurfaceState>), Box<dyn Error>> {
        let instance = Instance::new(InstanceDescriptor {
            // backends: Backends::all(),
            backends: Backends::GL,
            dx12_shader_compiler: Default::default(),
        });

        let (mut surface, adapter) = if let Some(window) = window {
            let surface = SurfaceState::new(window, &instance);

            let adapter = instance
                .request_adapter(&RequestAdapterOptions {
                    power_preference: PowerPreference::default(),
                    force_fallback_adapter: false,
                    compatible_surface: Some(&surface.surface),
                })
                .await
                .unwrap();

            (Some(surface), adapter)
        } else {
            let adapter = instance
                .request_adapter(&RequestAdapterOptions {
                    power_preference: PowerPreference::default(),
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await
                .unwrap();

            (None, adapter)
        };

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

        let output_format = if let Some(surface) = &mut surface {
            surface.set_configuration(&adapter, &device);
            surface.config.format
        } else {
            wgpu::TextureFormat::Rgba8UnormSrgb
        };

        let aggregate_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size.width as u64
                * multisampling as u64
                * size.height as u64
                * multisampling as u64
                * std::mem::size_of::<u32>() as u64,
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
        let render_pipeline =
            create_render_pipeline(&device, output_format, &pipeline_layout, multisampling)?;

        let renderer = Self {
            device,
            queue,
            size,
            multisampling,
            render_pipeline,
            compute_pipeline,
            bind_group,
            aggregate_buffer,
            uniforms_buffer,
            bind_group_layout,
            particles,
            attractor_buffer,
        };
        Ok((renderer, surface))
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 && new_size.height == 0 {
            return;
        }

        if new_size == self.size {
            return;
        }

        self.size = new_size;

        self.aggregate_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: self.size.width as u64
                * self.multisampling as u64
                * self.size.height as u64
                * self.multisampling as u64
                * std::mem::size_of::<u32>() as u64,
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

    pub fn load_aggragate_buffer(&mut self, buffer: &[i32]) {
        self.queue
            .write_buffer(&self.aggregate_buffer, 0, bytemuck::cast_slice(buffer));
    }

    pub fn render(&mut self, compute: bool, color_scale: f32, view: &wgpu::TextureView) {
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
                    view,
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
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn new_target_texture(&self, dimensions: winit::dpi::PhysicalSize<u32>) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: dimensions.width,
                height: dimensions.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::COPY_SRC
                | TextureUsages::COPY_DST
                | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    }

    pub fn copy_texture_content(&self, texture: wgpu::Texture) -> Vec<u8> {
        let dimensions = texture.size();

        let align_up = |x, y| ((x + y - 1) / y) * y;

        let bytes_per_row = align_up(
            dimensions.width * std::mem::size_of::<u32>() as u32,
            wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
        );

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes_per_row as u64 * dimensions.height as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());

        // wgpu::COPY_BYTES_PER_ROW_ALIGNMENT

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(dimensions.height),
                },
            },
            dimensions,
        );

        let id = self.queue.submit(std::iter::once(encoder.finish()));

        self.device.poll(wgpu::Maintain::WaitForSubmissionIndex(id));

        let (tx, rx) = std::sync::mpsc::channel();
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |_| {
            println!("map_async");
            tx.send(()).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);
        println!("waited for map_async");
        rx.recv().unwrap();

        let mut x = buffer.slice(..).get_mapped_range().to_vec();

        // crop padding bytes per row
        for y in 0..dimensions.height as usize {
            let src = y * bytes_per_row as usize..(y + 1) * bytes_per_row as usize;
            let dst = y * dimensions.width as usize * 4;
            x.copy_within(src, dst);
        }
        x.truncate(dimensions.width as usize * dimensions.height as usize * 4);

        x
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

fn convolution_code(kernel: &[f32], multisampling: u8, side: usize) -> String {
    // var c = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    let mut code: String = String::new();
    for j in 0..side {
        for i in 0..side {
            let k = kernel[j * side + i];
            if k == 0.0 {
                continue;
            }

            // i0 is pointing to the top left sample of the current pixel. So we shift di and j
            // here apropiately, in such a way that the kernel is centered on the center sample.
            //
            // -side /2 => center kernel on the current sample
            //
            // + multisampling / 2 => center kernel on the center sample

            let shift = -(side as i32) / 2 + multisampling as i32 / 2;

            // dj must be multiplied by the number of samples per row, which is multisampling *
            // screenWidth.
            let di = i as i32 + shift;
            let dj = (j as i32 + shift) * multisampling as i32;

            let di = if di < 0 {
                format!("- {}u", -di)
            } else {
                format!("+ {}u", di)
            };
            let dj = if dj < 0 {
                format!("- {}u", -dj)
            } else {
                format!("+ {}u", dj)
            };

            let v = format!(
                "aggregate_buffer[i0 {} * uniforms.screenWidth {}]\n",
                dj, di
            );
            let c = format!(
                "c += colormap(f32({}) / f32(aggregate_buffer[0])) * {:?};\n",
                v, k
            );
            code.push_str(&c);
        }
    }
    code
}

fn box_kernel(multisampling: u8) -> (Vec<f32>, usize) {
    let side = multisampling as usize;
    let kernel = vec![1.0; side * side];
    (kernel, side)
}

fn lanczos_kernel(multisampling: u8, a: u8) -> (Vec<f32>, usize) {
    use std::f64::consts::PI;

    let m = multisampling as usize;
    let a = a as usize;

    let side = m * a * 2 - 1;
    let mut kernel = vec![0.0; side * side];

    let l = |x: f64| {
        let a = a as f64;
        if x == 0.0 {
            1.0
        } else if x.abs() < a {
            let pi_x = x * PI;
            a * pi_x.sin() * (pi_x / a).sin() / (pi_x * pi_x)
        } else {
            0.0
        }
    };

    // -a..a, 1/m
    // -a+1/m ..= a-1/m => a*m*2 - 1
    // 0 -> -a + 1/m
    // 1 -> -a + 2/m
    // i -> -a + (i + 1)/m = (i + 1)/m - a = f64(i + 1 - a*m) / f64(m)

    let mut sum = 0.0;
    for j in 0..side {
        for i in 0..side {
            let a = a as i32;
            let m = m as i32;
            let x = (i as i32 + 1 - a * m) as f64 / m as f64;
            let y = (j as i32 + 1 - a * m) as f64 / m as f64;
            let k = l(x) * l(y);
            sum += k;
            kernel[j * side + i] = k as f32;
        }
    }
    for k in kernel.iter_mut() {
        *k /= sum as f32;
    }

    (kernel, side)
}

#[test]
fn lanczos_kernel_identity() {
    let (kernel, side) = lanczos_kernel(1, 4);
    assert_eq!(side, 1 * 4 * 2 - 1);
    println!("{:.2?}", kernel.chunks(side).collect::<Vec<_>>());
    for (i, k) in kernel.into_iter().enumerate() {
        let center = (side / 2 * side) + side / 2;
        let v = if i == center { 1.0 } else { 0.0 };
        println!("{k} ~= {v}");
        assert!((k - v).abs() < 0.0001);
    }
}

#[test]
fn lanczos_kernel_2x() {
    let (kernel, side) = lanczos_kernel(2, 3);
    assert_eq!(side, 2 * 1 * 2 - 1);

    println!("{:.2?}", kernel.chunks(side).collect::<Vec<_>>());
    let expected = [
        0.05010604, 0.12363171, 0.05010604, //
        0.12363171, 0.30504901, 0.12363171, //
        0.05010604, 0.12363171, 0.05010604, //
    ];
    for (k, v) in kernel.into_iter().zip(expected.into_iter()) {
        println!("{k} ~= {v}");
        assert!((k - v).abs() < 0.0001);
    }
}

fn create_render_pipeline(
    device: &Device,
    format: wgpu::TextureFormat,
    pipeline_layout: &wgpu::PipelineLayout,
    multisampling: u8,
) -> Result<RenderPipeline, Box<dyn Error>> {
    let mut source = std::fs::read_to_string("render/src/shader.wgsl")?;
    source = source.replace("MULTISAMPLING", &format!("{}u", multisampling));
    source = source.replace("LANCZOS_WIDTH", &format!("{}u", multisampling * 2));

    // let (kernel, side) = box_kernel(multisampling);
    let (kernel, side) = lanczos_kernel(multisampling, 1);
    source = source.replace(
        "//CONVOLUTION",
        &convolution_code(&kernel, multisampling, side),
    );

    println!("{}", source);

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
                format,
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
