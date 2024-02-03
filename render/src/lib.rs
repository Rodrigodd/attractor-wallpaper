use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::atomic::AtomicI32;
use std::sync::Arc;
use std::time::{Duration, Instant};

use attractors::{affine_affine, Affine, AntiAliasing, Attractor};
use oklab::{LinSrgb, OkLch, Oklab};

pub mod channel;
pub mod gradient;
mod renderer;

use gradient::Gradient;
use rand::SeedableRng as _;

pub use crate::renderer::{AttractorRenderer, SurfaceState, WgpuState};

const BORDER: f64 = 0.1;

pub fn get_intensity(
    base_intensity: f32,
    tranform: [f64; 4],
    total_samples: u64,
    antialiasing: attractors::AntiAliasing,
) -> i32 {
    let p = match antialiasing {
        attractors::AntiAliasing::None => 1,
        attractors::AntiAliasing::Bilinear => 64,
        attractors::AntiAliasing::Lanczos => 64,
    };

    let det = tranform[0] * tranform[3] - tranform[1] * tranform[2];

    (base_intensity as f64 * total_samples as f64 * p as f64 * det * 1000.0 / 4.0).round() as i32
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Multithreading {
    #[default]
    Single,
    AtomicMulti,
    MergeMulti,
}

#[cfg(feature = "serde")]
mod gradient_serde {
    use super::Gradient;
    use serde::{Deserialize, Serialize};

    pub fn serialize<S>(gradient: &Gradient<oklab::Oklab>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let gradient = gradient.map(|x| oklab::Srgb::from(*x).to_srgb8());
        gradient.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Gradient<oklab::Oklab>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let gradient = <Gradient<oklab::Srgb8>>::deserialize(deserializer)?;
        Ok(gradient.map(|x| x.to_f32().into()))
    }
}

#[cfg(feature = "serde")]
mod color_serde {
    use oklab::OkLch;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(color: &OkLch, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let color = oklab::Srgb::from(*color).to_srgb8();
        color.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<OkLch, D::Error>
    where
        D: Deserializer<'de>,
    {
        let color = <oklab::Srgb8>::deserialize(deserializer)?;
        let color = OkLch::from(color.to_f32());
        Ok(color)
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Theme {
    #[cfg_attr(feature = "serde", serde(with = "color_serde"))]
    pub background_color_1: OkLch,
    #[cfg_attr(feature = "serde", serde(with = "color_serde"))]
    pub background_color_2: OkLch,
    #[cfg_attr(feature = "serde", serde(with = "gradient_serde"))]
    pub gradient: Gradient<Oklab>,
}

/// List of themes, keyed by name.
pub type SavedThemes = BTreeMap<String, Theme>;

/// Serializable configuration of the attractor
#[derive(Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AttractorConfig {
    pub base_attractor: Attractor,
    pub base_intensity: i16,
    pub transform: Affine,
    pub size: (u32, u32),

    pub seed: u64,
    pub min_area: u16,
    pub multisampling: u8,
    pub anti_aliasing: AntiAliasing,
    pub intensity: f32,
    pub exponent: f32,
    pub random_start: bool,
    pub multithreading: Multithreading,
    pub samples_per_iteration: u64,

    pub saved_themes: SavedThemes,
    pub theme_name: String,
    pub background_color_1: OkLch,
    pub background_color_2: OkLch,
    #[cfg_attr(feature = "serde", serde(with = "gradient_serde"))]
    pub gradient: Gradient<Oklab>,
}
impl Clone for AttractorConfig {
    fn clone(&self) -> Self {
        Self {
            base_attractor: self.base_attractor,
            base_intensity: self.base_intensity,
            transform: self.transform,
            size: self.size,
            seed: self.seed,
            min_area: self.min_area,
            multisampling: self.multisampling,
            anti_aliasing: self.anti_aliasing,
            intensity: self.intensity,
            exponent: self.exponent,
            random_start: self.random_start,
            multithreading: self.multithreading,
            samples_per_iteration: self.samples_per_iteration,
            background_color_1: self.background_color_1,
            background_color_2: self.background_color_2,
            saved_themes: self.saved_themes.clone(),
            theme_name: self.theme_name.clone(),
            gradient: self.gradient.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.base_attractor.clone_from(&source.base_attractor);
        self.base_intensity.clone_from(&source.base_intensity);
        self.transform.clone_from(&source.transform);
        self.size.clone_from(&source.size);
        self.seed.clone_from(&source.seed);
        self.multisampling.clone_from(&source.multisampling);
        self.anti_aliasing.clone_from(&source.anti_aliasing);
        self.intensity.clone_from(&source.intensity);
        self.random_start.clone_from(&source.random_start);
        self.multithreading.clone_from(&source.multithreading);
        self.samples_per_iteration
            .clone_from(&source.samples_per_iteration);
        self.background_color_1
            .clone_from(&source.background_color_1);
        self.background_color_2
            .clone_from(&source.background_color_2);
        self.saved_themes.clone_from(&source.saved_themes);
        self.theme_name.clone_from(&source.theme_name);
        self.gradient.clone_from(&source.gradient);
    }
}
impl AttractorConfig {
    pub fn bitmap_size(&self) -> [usize; 2] {
        let width = self.size.0 as usize * self.multisampling as usize;
        let height = self.size.1 as usize * self.multisampling as usize;
        [width, height]
    }

    pub fn set_theme(&mut self, name: &String, theme: &Theme) {
        self.theme_name.clone_from(name);
        self.background_color_1 = theme.background_color_1;
        self.background_color_2 = theme.background_color_2;
        self.gradient = theme.gradient.clone();
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
        let rng = rand::rngs::SmallRng::seed_from_u64(seed);
        let mut attractor =
            Attractor::find_strange_attractor(rng, self.min_area, u16::MAX, usize::MAX).unwrap();
        let points = attractor.get_points::<512>();

        // 4 KiB
        let affine = attractors::affine_from_pca(&points);
        attractor = attractor.transform_input(affine);

        let bounds = attractor.get_bounds(512);

        let [width, height] = self.bitmap_size();
        let dst = square_bounds(width as f64, height as f64, BORDER);
        let affine = attractors::map_bounds_affine(dst, bounds);

        self.base_intensity = attractors::get_base_intensity(&attractor);
        self.base_attractor = attractor;
        self.transform = affine;
    }

    pub fn resize(&mut self, new_size: (u32, u32), new_multisampling: u8) {
        let old_size = self.bitmap_size();

        self.size = new_size;
        self.multisampling = new_multisampling;
        let new_size = self.bitmap_size();

        let src = square_bounds(old_size[0] as f64, old_size[1] as f64, BORDER);
        let dst = square_bounds(new_size[0] as f64, new_size[1] as f64, BORDER);
        let affine = attractors::map_bounds_affine(dst, src);

        self.transform = affine_affine(affine, self.transform);
    }
}

#[derive(Clone)]
pub struct AttractorCtx {
    pub config: Arc<Mutex<AttractorConfig>>,
    pub attractor: Attractor,
    pub bitmap: Vec<i32>,
    pub total_samples: u64,
    pub last_change: Instant,
    pub stop_time: Option<Instant>,
    pub starts: Vec<[f64; 2]>,
}
impl AttractorCtx {
    pub fn new(config: Arc<Mutex<AttractorConfig>>) -> Self {
        let bitmap;
        let attractor;
        {
            let config = config.lock();
            let [width, height] = config.bitmap_size();
            bitmap = vec![0; width * height];
            attractor = config.base_attractor.transform_input(config.transform);
        };
        Self {
            config,
            attractor,
            bitmap,
            total_samples: 0,
            last_change: Instant::now(),
            stop_time: None,
            starts: Vec::new(),
        }
    }

    pub fn update(&mut self) {
        self.bitmap = {
            let config = self.config.lock();
            let [width, height] = config.bitmap_size();
            vec![0; width * height]
        };
        self.total_samples = 0;
        self.last_change = Instant::now();
        self.stop_time = None;
        self.attractor = {
            let config = self.config.lock();
            config.base_attractor.transform_input(config.transform)
        };
        self.starts.clear();
    }

    pub fn clear(&mut self) {
        self.bitmap.fill(0);
        self.total_samples = 0;
        self.last_change = Instant::now();
        self.stop_time = None;
    }

    pub fn transform(&mut self, affine: Affine) {
        {
            let mut config = self.config.lock();
            config.transform = affine_affine(affine, config.transform);
            self.attractor = config.base_attractor.transform_input(config.transform);
        }
        self.clear();
        self.starts.clear();
    }

    pub fn resize(&mut self, new_size: (u32, u32), new_multisampling: u8) {
        {
            let mut config = self.config.lock();

            config.resize(new_size, new_multisampling);

            let new_size = config.bitmap_size();
            self.attractor = config.base_attractor.transform_input(config.transform);
            self.bitmap = vec![0i32; new_size[0] * new_size[1]];
            self.total_samples = 0;
        }
        self.starts.clear();
        self.clear();
    }

    pub fn set_seed(&mut self, seed: u64) {
        {
            let mut config = self.config.lock();
            config.set_seed(seed);
            self.attractor = config.base_attractor.transform_input(config.transform);
        };

        self.clear();
    }

    pub fn set_min_area(&mut self, min_area: u16) {
        {
            let mut config = self.config.lock();
            config.min_area = min_area;

            let seed = config.seed;
            config.set_seed(seed);

            self.attractor = config.base_attractor.transform_input(config.transform);
        };

        self.clear();
    }

    pub fn set_multisampling(&mut self, multisampling: u8) {
        let size = self.config.lock().size;
        self.resize(size, multisampling);
        self.clear();
    }

    pub fn set_antialiasing(&mut self, anti_aliasing: AntiAliasing) {
        self.config.lock().anti_aliasing = anti_aliasing;
        self.clear();
    }

    pub fn set_intensity(&mut self, intensity: f32) {
        self.config.lock().intensity = intensity;
    }

    pub fn set_exponent(&mut self, exponent: f32) {
        self.config.lock().exponent = exponent;
    }

    pub fn set_random_start(&mut self, random_start: bool) {
        self.config.lock().random_start = random_start;
        self.clear();
    }

    pub fn set_multithreaded(&mut self, multithreaded: Multithreading) {
        self.config.lock().multithreading = multithreaded;
        self.clear();
    }

    pub fn set_samples_per_iteration(&mut self, samples_per_iteration: u64) {
        self.config.lock().samples_per_iteration = samples_per_iteration;
        self.clear();
    }
}

fn square_bounds(width: f64, height: f64, border: f64) -> [f64; 4] {
    let size = width.min(height) * (1.0 - 2.0 * border);
    let start_x = (width - size) / 2.0;
    let start_y = (height - size) / 2.0;
    [start_x, start_x + size, start_y, start_y + size]
}

#[allow(clippy::too_many_arguments)]
pub fn update_render(
    attractor_renderer: &mut AttractorRenderer,
    wgpu_state: &WgpuState,
    gradient: &Gradient<Oklab>,
    multisampling: u8,
    background_color_1: OkLch,
    background_color_2: OkLch,
    intensity: f32,
    exponent: f32,
) {
    attractor_renderer.recreate_aggregate_buffer(
        &wgpu_state.device,
        attractor_renderer.size,
        multisampling,
    );

    let c1 = LinSrgb::from(background_color_1).clip();
    let c2 = LinSrgb::from(background_color_2).clip();
    attractor_renderer.set_background_color(
        &wgpu_state.queue,
        [c1.r, c1.g, c1.b, 1.0],
        [c2.r, c2.g, c2.b, 1.0],
    );
    attractor_renderer.set_colormap(
        &wgpu_state.queue,
        gradient
            .monotonic_hermit_spline_coefs()
            .into_iter()
            .map(|x| x.into())
            .collect(),
    );
    attractor_renderer.set_intensity_exponent(&wgpu_state.queue, intensity, exponent);
}

pub enum AttractorMess {
    SetSeed(u64),
    SetMinArea(u16),
    SetMultisampling(u8),
    SetAntialiasing(AntiAliasing),
    SetIntensity(f32),
    SetExponent(f32),
    SetRandomStart(bool),
    SetMultithreaded(Multithreading),
    SetSamplesPerIteration(u64),
    Resize((u32, u32)),
    Transform(Affine),
    Update,
}

pub fn attractor_thread(
    recv_conf: std::sync::mpsc::Receiver<AttractorMess>,
    mut attractor: AttractorCtx,
    mut sender_bitmap: channel::Sender<AttractorCtx>,
) {
    loop {
        loop {
            match recv_conf.try_recv() {
                Ok(mess) => match mess {
                    AttractorMess::SetSeed(seed) => attractor.set_seed(seed),
                    AttractorMess::SetMinArea(min_area) => attractor.set_min_area(min_area),
                    AttractorMess::SetMultisampling(multisampling) => {
                        attractor.set_multisampling(multisampling)
                    }
                    AttractorMess::SetAntialiasing(antialising) => {
                        attractor.set_antialiasing(antialising)
                    }
                    AttractorMess::SetIntensity(intensity) => attractor.set_intensity(intensity),
                    AttractorMess::SetExponent(exponent) => attractor.set_exponent(exponent),
                    AttractorMess::SetRandomStart(random_start) => {
                        attractor.set_random_start(random_start)
                    }
                    AttractorMess::SetMultithreaded(multithreaded) => {
                        attractor.set_multithreaded(multithreaded)
                    }
                    AttractorMess::SetSamplesPerIteration(samples_per_iteration) => {
                        attractor.set_samples_per_iteration(samples_per_iteration)
                    }
                    AttractorMess::Resize(size) => {
                        let multisampling = attractor.config.lock().multisampling;
                        attractor.resize(size, multisampling)
                    }
                    AttractorMess::Transform(affine) => attractor.transform(affine),
                    AttractorMess::Update => attractor.update(),
                },
                Err(std::sync::mpsc::TryRecvError::Disconnected) => return,
                _ => break,
            }
        }

        let det = |mat: [f64; 4]| mat[0] * mat[3] - mat[1] * mat[2];

        let max_total_samples = {
            let mat = attractor.config.lock().transform.0;

            // `transform` maps from [-1, 1] to the image size, so we need to divide by 2x2 here.
            let baseline_zoom = 500.0 * 500.0 / 4.0;
            let zoom = 1.0 / det(mat).abs();

            // for a 500x500 image, 10_000_000 samples is a good baseline
            10_000_000.0 * zoom / baseline_zoom
        };

        if attractor.total_samples < max_total_samples as u64 {
            let multithreading = attractor.config.lock().multithreading;
            aggregate_buffer(multithreading, &mut attractor);
        } else {
            if attractor.stop_time.is_none() {
                attractor.stop_time = Some(Instant::now());
            }
            std::thread::sleep(Duration::from_millis(5));
        }

        if sender_bitmap.is_closed() {
            break;
        }
        sender_bitmap.send(&mut attractor);
    }
}

pub fn aggregate_buffer(multithreading: Multithreading, attractor: &mut AttractorCtx) {
    match multithreading {
        Multithreading::Single => aggregate_attractor_single_thread(attractor),
        Multithreading::AtomicMulti => atomic_par_aggregate_attractor(attractor),
        Multithreading::MergeMulti => merge_par_aggregate_attractor(attractor),
    }
}

pub fn aggregate_attractor_single_thread(attractor: &mut AttractorCtx) {
    let config = attractor.config.lock();

    let [width, height] = config.bitmap_size();

    // the config could have being modified in the meantime.
    if width * height != attractor.bitmap.len() {
        return;
    }

    let samples = config.samples_per_iteration;

    if config.random_start {
        attractor.attractor.start = attractor
            .attractor
            .get_random_start_point(&mut rand::thread_rng());
    }

    let anti_aliasing = config.anti_aliasing;

    // avoid holding the lock while doing the heavy computation
    drop(config);

    attractors::aggregate_to_bitmap(
        &mut attractor.attractor,
        width,
        height,
        samples,
        anti_aliasing,
        &mut attractor.bitmap[..],
        &mut 0,
    );
    attractor.total_samples += samples;
}

pub fn atomic_par_aggregate_attractor(attractor: &mut AttractorCtx) {
    let config = attractor.config.lock();

    let samples = config.samples_per_iteration;
    let threads = 4;
    if config.random_start {
        attractor.starts.clear();
    }
    while attractor.starts.len() < threads {
        let mut rng = rand::thread_rng();
        attractor
            .starts
            .push(attractor.attractor.get_random_start_point(&mut rng));
    }

    let [width, height] = config.bitmap_size();
    let anti_aliasing = config.anti_aliasing;

    // avoid holding the lock while doing the heavy computation
    drop(config);

    // SAFETY: AtomicI32 and i32 have the same layout.
    let bitmap: &mut [AtomicI32] = unsafe { std::mem::transmute(&mut attractor.bitmap[..]) };

    std::thread::scope(|s| {
        for start in attractor.starts.iter_mut() {
            attractor.total_samples += samples;

            let mut att = attractor.attractor;

            let bitmap: &[AtomicI32] = &*bitmap;

            att.start = *start;
            s.spawn(move || {
                attractors::aggregate_to_bitmap(
                    &mut att,
                    width,
                    height,
                    samples,
                    anti_aliasing,
                    &mut &*bitmap,
                    &mut AtomicI32::new(0),
                );
                *start = att.start;
            });
        }
    });
}

pub fn merge_par_aggregate_attractor(attractor: &mut AttractorCtx) {
    let config = attractor.config.lock();

    let samples = config.samples_per_iteration;
    let threads = 4;
    if config.random_start {
        attractor.starts.clear();
    }
    while attractor.starts.len() < threads {
        let mut rng = rand::thread_rng();
        attractor
            .starts
            .push(attractor.attractor.get_random_start_point(&mut rng));
    }

    let [width, height] = config.bitmap_size();
    let anti_aliasing = config.anti_aliasing;

    // avoid holding the lock while doing the heavy computation
    drop(config);

    // resize bitmap to hold multiple buffers
    let len = width * height;
    attractor.bitmap.resize(threads * len, 0);

    std::thread::scope(|s| {
        let mut bitmap = attractor.bitmap.as_mut_slice();
        for start in attractor.starts.iter_mut() {
            attractor.total_samples += samples;
            let mut att = attractor.attractor;

            let b;
            (b, bitmap) = bitmap.split_at_mut(len);

            att.start = *start;
            s.spawn(move || {
                attractors::aggregate_to_bitmap(
                    &mut att,
                    width,
                    height,
                    samples,
                    anti_aliasing,
                    b,
                    &mut 0,
                );
                *start = att.start;
            });
        }
    });

    // merge the buffers into one
    for i in 0..len {
        let sum = attractor.bitmap.iter().skip(i).step_by(len).sum::<i32>();
        attractor.bitmap[i] = sum;
    }
    attractor.bitmap.truncate(len);
}

/// Returns a RGBA8 bitmap of the rendered attractor.
#[allow(clippy::too_many_arguments)]
pub async fn render_to_bitmap(
    size: (u32, u32),
    multisampling: u8,
    mut bitmap: Vec<i32>,
    base_intensity: i16,
    mat: [f64; 4],
    total_samples: u64,
    anti_aliasing: AntiAliasing,
    gradient: Gradient<Oklab>,
    background_color_1: OkLch,
    background_color_2: OkLch,
    intensity: f32,
    exponent: f32,
) -> Vec<u8> {
    let wgpu_state = WgpuState::new_headless().await.unwrap();
    let mut attractor_renderer = AttractorRenderer::new(
        &wgpu_state.device,
        size,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        multisampling,
    )
    .unwrap();

    bitmap[0] = get_intensity(base_intensity as f32, mat, total_samples, anti_aliasing);
    attractor_renderer.load_aggregate_buffer(&wgpu_state.queue, &bitmap);

    update_render(
        &mut attractor_renderer,
        &wgpu_state,
        &gradient,
        multisampling,
        background_color_1,
        background_color_2,
        intensity,
        exponent,
    );

    let texture = wgpu_state.new_target_texture(size);
    let view = texture.create_view(&Default::default());
    attractor_renderer.render(&wgpu_state.device, &wgpu_state.queue, &view);

    wgpu_state.copy_texture_content(texture)
}
