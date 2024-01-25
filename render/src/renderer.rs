use std::error::Error;

use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Backends, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BlendState, Color, ColorTargetState, ColorWrites,
    CommandEncoderDescriptor, Device, DeviceDescriptor, Face, Features, FragmentState, FrontFace,
    Instance, InstanceDescriptor, Limits, LoadOp, MultisampleState, Operations,
    PipelineLayoutDescriptor, PolygonMode, PowerPreference, PrimitiveState, PrimitiveTopology,
    Queue, RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline,
    RenderPipelineDescriptor, RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource,
    ShaderStages, Surface, SurfaceConfiguration, TextureUsages, VertexState,
};

pub struct SurfaceState<W: HasRawWindowHandle + HasRawDisplayHandle> {
    surface: Surface,
    // SAFETY: window must come after surface, because surface must be dropped before window.
    window: W,
    config: SurfaceConfiguration,
}
impl<W: HasRawWindowHandle + HasRawDisplayHandle> SurfaceState<W> {
    fn new(window: W, size: (u32, u32), instance: &Instance) -> Self {
        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            width: size.0,
            height: size.1,
            present_mode: wgpu::PresentMode::AutoVsync,
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

    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    pub fn texture_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    fn set_configuration(&mut self, size: (u32, u32), adapter: &wgpu::Adapter, device: &Device) {
        let surface_caps = self.surface.get_capabilities(adapter);

        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        log::trace!("Surface capabilities: {:?}", surface_caps);
        log::debug!("Surface format: {:?}", surface_format);

        self.config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.0,
            height: size.1,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        self.surface.configure(device, &self.config);
    }

    pub fn resize(&mut self, new_size: (u32, u32), device: &Device) {
        self.config.width = new_size.0;
        self.config.height = new_size.1;

        self.surface.configure(device, &self.config);
    }

    pub fn destroy(self) -> W {
        self.window
    }

    pub fn window(&self) -> &W {
        &self.window
    }

    pub fn current_texture(&self) -> Result<wgpu::SurfaceTexture, wgpu::SurfaceError> {
        self.surface.get_current_texture()
    }
}

pub struct WgpuState {
    pub adapter: wgpu::Adapter,
    pub device: Device,
    pub queue: Queue,
}
impl WgpuState {
    pub async fn new_windowed<W: HasRawWindowHandle + HasRawDisplayHandle>(
        window: W,
        size: (u32, u32),
    ) -> Result<(Self, SurfaceState<W>), Box<dyn Error>> {
        Self::new(Some((window, size)))
            .await
            .map(|(renderer, surface)| (renderer, surface.unwrap()))
    }

    pub async fn new_headless() -> Result<Self, Box<dyn Error>> {
        // TODO: This could be replace with the never type (`!`), when it stabilizes.
        trait HeadlessTrait: HasRawWindowHandle + HasRawDisplayHandle {}
        type Headless = &'static dyn HeadlessTrait;

        Self::new(None::<(Headless, _)>)
            .await
            .map(|(renderer, surface)| {
                assert!(surface.is_none());
                renderer
            })
    }

    async fn new<W: HasRawWindowHandle + HasRawDisplayHandle>(
        window: Option<(W, (u32, u32))>,
    ) -> Result<(Self, Option<SurfaceState<W>>), Box<dyn Error>> {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            // backends: Backends::GL,
            dx12_shader_compiler: Default::default(),
            flags: wgpu::InstanceFlags::debugging(),
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });

        let (mut surface, adapter) = if let Some((window, size)) = window {
            let surface = SurfaceState::new(window, size, &instance);

            let adapter = instance
                .request_adapter(&RequestAdapterOptions {
                    power_preference: PowerPreference::default(),
                    force_fallback_adapter: false,
                    compatible_surface: Some(&surface.surface),
                })
                .await
                .unwrap();

            (Some((surface, size)), adapter)
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
                    limits: Limits {
                        // My android device only supports 128 for workgroup_size_y, smaller than
                        // the default 256. But I am not using compute shaders anyway, so keep
                        // everthing as 0 for now.
                        max_compute_workgroup_storage_size: 0,
                        max_compute_invocations_per_workgroup: 0,
                        max_compute_workgroup_size_x: 0,
                        max_compute_workgroup_size_y: 0,
                        max_compute_workgroup_size_z: 0,
                        max_compute_workgroups_per_dimension: 0,

                        ..Limits::downlevel_defaults()
                    },
                    // limits: Limits::downlevel_webgl2_defaults(),
                    label: None,
                },
                None,
            )
            .await?;

        if let Some((surface, size)) = surface.as_mut() {
            surface.set_configuration(*size, &adapter, &device);
        }

        let wgpu_state = Self {
            adapter,
            device,
            queue,
        };

        Ok((wgpu_state, surface.map(|x| x.0)))
    }

    pub fn new_target_texture(&self, dimensions: (u32, u32)) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: dimensions.0,
                height: dimensions.1,
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
            tx.send(()).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);
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

pub struct AttractorRenderer {
    pub size: (u32, u32),
    pub multisampling: u8,
    output_format: wgpu::TextureFormat,
    render_pipeline: RenderPipeline,
    bind_group: BindGroup,
    pipeline_layout: wgpu::PipelineLayout,
    aggregate_buffer: wgpu::Buffer,
    uniforms_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl AttractorRenderer {
    pub fn new(
        device: &Device,
        size: (u32, u32),
        output_format: wgpu::TextureFormat,
        multisampling: u8,
    ) -> Result<Self, Box<dyn Error>> {
        let aggregate_buffer = gen_aggreate_buffer(device, size, multisampling);

        let uniforms = Uniforms {
            screen_width: size.0,
            screen_height: size.1,
            _padding: [0; 8],
            bg_color_1: [0.012, 0.0, 0.0, 1.0],
            bg_color_2: [0.004, 0.0, 0.0, 1.0],
            bg_point_1: [1.0, 0.0],
            bg_point_2: [0.2, 1.0],
            color_map: Default::default(),
        };

        let uniforms_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("attractor_frag_bind_group_layout"),
            entries: &[
                // uniform
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
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
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
            device,
            &bind_group_layout,
            &uniforms_buffer,
            &aggregate_buffer,
        );

        let render_pipeline =
            create_render_pipeline(device, output_format, &pipeline_layout, multisampling)?;

        Ok(Self {
            size,
            multisampling,
            output_format,
            render_pipeline,
            bind_group,
            pipeline_layout,
            aggregate_buffer,
            uniforms_buffer,
            bind_group_layout,
        })
    }

    pub fn recreate_aggregate_buffer(
        &mut self,
        device: &Device,
        size: (u32, u32),
        multisampling: u8,
    ) {
        if self.size == size && self.multisampling == multisampling {
            return;
        }

        self.size = size;
        self.aggregate_buffer = gen_aggreate_buffer(device, self.size, multisampling);

        self.bind_group = create_bind_group(
            device,
            &self.bind_group_layout,
            &self.uniforms_buffer,
            &self.aggregate_buffer,
        );

        if multisampling != self.multisampling {
            self.multisampling = multisampling;

            self.render_pipeline = create_render_pipeline(
                device,
                self.output_format,
                &self.pipeline_layout,
                self.multisampling,
            )
            .unwrap();
        }
    }

    pub fn resize(&mut self, device: &Device, new_size: (u32, u32)) {
        if new_size.0 == 0 && new_size.1 == 0 {
            return;
        }

        if new_size == self.size {
            return;
        }

        self.recreate_aggregate_buffer(device, new_size, self.multisampling);
    }

    pub fn load_aggregate_buffer(&mut self, queue: &Queue, buffer: &[i32]) {
        queue.write_buffer(&self.aggregate_buffer, 0, bytemuck::cast_slice(buffer));
    }

    pub fn render(&mut self, device: &Device, queue: &Queue, view: &wgpu::TextureView) {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        self.update_uniforms(queue);

        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            self.render_aggregate_buffer(&mut render_pass);
        }

        queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn render_aggregate_buffer<'a>(&'a mut self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);

        render_pass.draw(0..3, 0..1);
    }

    #[allow(clippy::ptr_eq)]
    pub fn update_uniforms(&mut self, queue: &Queue) {
        assert!({
            let x = Uniforms {
                screen_width: self.size.0,
                screen_height: self.size.1,
                ..Default::default()
            };
            (&x as *const Uniforms as usize) == (&x.screen_width as *const u32 as usize)
                && (&x as *const Uniforms as usize + 4) == (&x.screen_height as *const u32 as usize)
        });

        queue.write_buffer(
            &self.uniforms_buffer,
            0,
            bytemuck::cast_slice(&[self.size.0, self.size.1]),
        );
    }

    #[allow(clippy::ptr_eq)]
    pub fn set_background_color(
        &self,
        queue: &Queue,
        background_color_1: [f32; 4],
        background_color_2: [f32; 4],
    ) {
        assert!({
            let x = Uniforms::default();
            (&x as *const Uniforms as usize) == (&x.screen_width as *const u32 as usize)
                && (&x as *const Uniforms as usize + 4) == (&x.screen_height as *const u32 as usize)
        });

        queue.write_buffer(
            &self.uniforms_buffer,
            16,
            bytemuck::cast_slice(&[background_color_1, background_color_2]),
        );
    }

    #[allow(clippy::ptr_eq)]
    pub fn set_colormap(&self, queue: &Queue, colormap: Vec<ColorPoint>) {
        assert!({
            let x = Uniforms::default();
            (&x as *const Uniforms as usize + 64) == (&x.color_map as *const _ as usize)
        });

        assert!(colormap.len() <= 4);

        queue.write_buffer(
            &self.uniforms_buffer,
            64,
            bytemuck::cast_slice(colormap.as_slice()),
        );
    }
}

fn gen_aggreate_buffer(device: &Device, size: (u32, u32), multisampling: u8) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size.0 as u64
            * multisampling as u64
            * size.1 as u64
            * multisampling as u64
            * std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn create_bind_group(
    device: &Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    uniforms_buffer: &wgpu::Buffer,
    aggregate_buffer: &wgpu::Buffer,
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
        ],
    })
}

#[repr(C)]
#[derive(Copy, Clone, Default, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    screen_width: u32,
    screen_height: u32,
    _padding: [u8; 8], // the next field is 16 bytes aligned
    bg_color_1: [f32; 4],
    bg_color_2: [f32; 4],
    bg_point_1: [f32; 2],
    bg_point_2: [f32; 2],
    color_map: [ColorPoint; 4],
}

#[repr(C, align(16))]
#[derive(Copy, Clone, Default, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ColorPoint {
    time: f32,
    _padding: [u8; 12], // the next field is 16 bytes aligned
    l_coef: [f32; 4],
    a_coef: [f32; 4],
    b_coef: [f32; 4],
}
impl From<(f32, [f32; 4], [f32; 4], [f32; 4])> for ColorPoint {
    fn from((time, l_coef, a_coef, b_coef): (f32, [f32; 4], [f32; 4], [f32; 4])) -> Self {
        Self {
            time,
            _padding: [0; 12],
            l_coef,
            a_coef,
            b_coef,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AttractorF32 {
    a: [f32; 6],
    b: [f32; 6],
    start: [f32; 2],
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

            let v = format!("aggregate_buffer[i0 {} * uniforms.screenWidth {}]", dj, di);
            let c = format!(
                "c += color(f32({}) / f32(aggregate_buffer[0]) * 1000.0 , p) * {:?};\n",
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
    assert_eq!(side, 4 * 2 - 1);
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
    assert_eq!(side, 2 * 2 - 1);

    println!("{:.2?}", kernel.chunks(side).collect::<Vec<_>>());
    let expected = [
        0.050106, 0.123632, 0.050106, //
        0.123632, 0.305049, 0.123632, //
        0.050106, 0.123632, 0.050106, //
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
    #[cfg(not(target_os = "android"))]
    let mut source = std::fs::read_to_string("render/src/shader.wgsl")?;
    #[cfg(target_os = "android")]
    let mut source = include_str!("shader.wgsl").to_string();

    source = source.replace("MULTISAMPLING", &format!("{}u", multisampling));
    source = source.replace("LANCZOS_WIDTH", &format!("{}u", multisampling * 2));

    let (kernel, side) = if multisampling == 1 {
        box_kernel(multisampling)
    } else {
        lanczos_kernel(multisampling, 1)
    };
    source = source.replace(
        "//CONVOLUTION",
        &convolution_code(&kernel, multisampling, side),
    );

    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("Shader"),
        source: ShaderSource::Wgsl(source.into()),
    });

    let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("attractor_shader"),
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
