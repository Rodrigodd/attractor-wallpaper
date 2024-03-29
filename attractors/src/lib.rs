use std::sync::atomic::{AtomicI32, Ordering};

use rand::prelude::*;

mod sympy;

type Point = [f64; 2];
/// (min_x, max_x, min_y, max_y)
type Bounds = [f64; 4];

/// Matrix and translation vector. The transformation formula is `A*x + t`, where `A` is the 2x2
/// matrix and `t` is the translation vector.
pub type Affine = ([f64; 4], [f64; 2]);

const THUMB_WIDTH: usize = 64;
const THUMB_HEIGHT: usize = 64;

#[derive(Debug, Clone, Copy)]
pub enum Behavior {
    Convergent { after: usize, to: Point },
    Divergent { after: usize },
    Chaotic { lyapunov: f64, to: Point },
    Periodic { lyapunov: f64, to: Point },
}

#[derive(Debug, Default, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Attractor {
    pub a: [f64; 6],
    pub b: [f64; 6],
    pub start: Point,
}
impl Attractor {
    pub fn random(mut rng: impl rand::RngCore) -> Self {
        let a = std::array::from_fn(|_| rng.gen_range(-1.0..1.0));
        let b = std::array::from_fn(|_| rng.gen_range(-1.0..1.0));
        Self {
            a,
            b,
            start: [0.0, 0.0],
        }
    }

    /// Return a attractor with a new set of coefficients, such that it behaves like the original
    /// attractor, but with the given affine transformation applied to the input.
    pub fn transform_input(self, (mat, translation): Affine) -> Self {
        let (new_a, new_b, new_p) = sympy::apply_affine_transform_to_attractor(
            self.a,
            self.b,
            self.start,
            mat,
            translation,
        );

        Self {
            a: new_a,
            b: new_b,
            start: new_p,
        }
    }

    pub fn step(&self, [x, y]: Point) -> Point {
        let x_ = self.a[0]
            + self.a[1] * x
            + self.a[2] * x * x
            + self.a[3] * x * y
            + self.a[4] * y
            + self.a[5] * y * y;
        let y_ = self.b[0]
            + self.b[1] * x
            + self.b[2] * x * x
            + self.b[3] * x * y
            + self.b[4] * y
            + self.b[5] * y * y;
        debug_assert!(x.is_finite());
        debug_assert!(y.is_finite());
        debug_assert!(x_.is_finite());
        debug_assert!(y_.is_finite());
        [x_, y_]
    }

    pub fn check_behavior(&self) -> Behavior {
        /// Threshold for convergence
        const EPSILON: f64 = 1e-10;
        /// Threshold for divergence
        const THRESHOLD: f64 = 1e10;

        // helper closures
        let delta = |[x1, y1]: Point, [x2, y2]: Point| [(x2 - x1), (y2 - y1)];
        let sqdist = |[x1, y1]: Point, [x2, y2]: Point| (x1 - x2).powi(2) + (y1 - y2).powi(2);

        /// How many steps to wait before calculating the lyapunov exponent
        const WAIT: usize = 1000;
        /// The total number of steps to take when calculating the lyapunov exponent
        const STEPS: usize = 100_000 - WAIT;

        let mut lyapunov = 0.0;

        let mut p0 = self.start;
        let mut pn0 = [p0[0] + 0.0001, p0[1]];
        let d0 = sqdist(p0, pn0).sqrt();

        let mut fixed = p0;
        for i in 0..WAIT + STEPS {
            let p1 = self.step(p0);

            // check for divergence
            let diverge = |p: Point| p[0].abs() > THRESHOLD || p[1].abs() > THRESHOLD;
            if diverge(p1) {
                return Behavior::Divergent { after: i };
            };

            // check for convergence
            let too_close = |(p1, p2)| {
                let [dx, dy] = delta(p1, p2);
                dx.abs() < EPSILON && dy.abs() < EPSILON
            };
            if too_close((p0, p1)) {
                return Behavior::Convergent { after: i, to: p1 };
            }

            // wait for the point to settle into the attractor
            if i < WAIT {
                p0 = p1;
                continue;
            }
            if i == WAIT {
                fixed = p1;
            }
            // check if is periodic
            if i > WAIT && fixed == p1 {
                return Behavior::Periodic {
                    lyapunov: 0.0,
                    to: fixed,
                };
            }

            // calculate lyapunov exponent
            let pn1 = self.step(pn0);

            let d1 = sqdist(p1, pn1).sqrt();
            if d1 == 0.0 {
                return Behavior::Convergent { after: i, to: p1 };
            }

            lyapunov += (d1 / d0).log10();

            let [dx, dy] = delta(p1, pn1);

            p0 = p1;
            pn0 = [p1[0] + d0 * dx / d1, p1[1] + d0 * dy / d1];
        }

        // check for chaos
        let lyapunov = lyapunov / STEPS as f64; // arbitrary scaling factor, for aesthetics
        if lyapunov < 0.01 {
            return Behavior::Periodic { lyapunov, to: p0 };
        }

        Behavior::Chaotic { lyapunov, to: p0 }
    }

    pub fn find_strange_attractor(
        mut rng: impl rand::RngCore,
        min_area: u16,
        max_area: u16,
        tries: usize,
    ) -> Option<Self> {
        for _ in 0..tries {
            let mut attractor = Self::random(&mut rng);
            if let Behavior::Chaotic { lyapunov, to } = attractor.check_behavior() {
                log::debug!("found attractor with lyapunov exponent {}", lyapunov);

                let area = get_base_area(&attractor);
                if area < min_area || area > max_area {
                    continue;
                }

                attractor.start = to;
                return Some(attractor);
            }
        }
        None
    }

    pub fn get_bounds(&self, samples: usize) -> Bounds {
        let mut p = self.start;

        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for _ in 0..samples {
            p = self.step(p);

            if p[0] < min_x {
                min_x = p[0];
            }
            if p[0] > max_x {
                max_x = p[0];
            }
            if p[1] < min_y {
                min_y = p[1];
            }
            if p[1] > max_y {
                max_y = p[1];
            }
        }

        [min_x, max_x, min_y, max_y]
    }

    pub fn get_points<const N: usize>(&self) -> [[f64; 2]; N] {
        let mut p = self.start;
        std::array::from_fn(|_| {
            p = self.step(p);
            p
        })
    }

    pub fn get_random_start_point(&self, rng: &mut impl Rng) -> [f64; 2] {
        loop {
            const RADIUS: f64 = 0.001;
            let dx = rng.gen_range(-1.0..1.0) * RADIUS;
            let dy = rng.gen_range(-1.0..1.0) * RADIUS;

            let px = self.start[0] + dx;
            let py = self.start[1] + dy;

            let mut attractor = *self;
            attractor.start = [px, py];
            match attractor.check_behavior() {
                Behavior::Chaotic { to, .. } => return to,
                _ => continue,
            }
        }
    }
}

pub fn map_bounds(p: Point, src: Bounds, dst: Bounds) -> Point {
    let [min_x, max_x, min_y, max_y] = src;
    let [min_x_, max_x_, min_y_, max_y_] = dst;
    let x = (p[0] - min_x) / (max_x - min_x);
    let y = (p[1] - min_y) / (max_y - min_y);
    let x_ = min_x_ + x * (max_x_ - min_x_);
    let y_ = min_y_ + y * (max_y_ - min_y_);
    [x_, y_]
}

pub fn map_bounds_affine(src: Bounds, dst: Bounds) -> ([f64; 4], [f64; 2]) {
    let src_width = src[1] - src[0];
    let src_height = src[3] - src[2];
    let dst_width = dst[1] - dst[0];
    let dst_height = dst[3] - dst[2];

    let scale_x = dst_width / src_width;
    let scale_y = dst_height / src_height;

    let scaled_src = [
        src[0] * scale_x,
        src[1] * scale_x,
        src[2] * scale_y,
        src[3] * scale_y,
    ];

    debug_assert!((dst[1] - dst[0]) - (scaled_src[1] - scaled_src[0]) < 0.00001);
    debug_assert!((dst[3] - dst[2]) - (scaled_src[3] - scaled_src[2]) < 0.00001);

    let translate_x = dst[0] - scaled_src[0];
    let translate_y = dst[2] - scaled_src[2];

    ([scale_x, 0.0, 0.0, scale_y], [translate_x, translate_y])
}

/// Derives a affine transformation from Principal Component Analysis.
///
/// Find an affine transformation that maps the bounding box aligned with the principal components
/// of the given points to the `[-1, -1, 1, 1]` bounding box.
pub fn affine_from_pca(points: &[Point]) -> Affine {
    let [mean_x, mean_y] = points
        .iter()
        .fold([0.0, 0.0], |acc, p| [acc[0] + p[0], acc[1] + p[1]])
        .map(|x| x / points.len() as f64);

    let e = |v: &dyn Fn(&[f64; 2]) -> f64| points.iter().map(v).sum::<f64>() / points.len() as f64;

    let cov_xx = e(&|[x, _]| (x - mean_x).powi(2));
    let cov_xy = e(&|[x, y]| (x - mean_x) * (y - mean_y));
    let cov_yy = e(&|[_, y]| (y - mean_y).powi(2));

    let cov_matrix = [[cov_xx, cov_xy], [cov_xy, cov_yy]];

    // Eigenvalues and eigenvectors of 2x2 matrices
    // From: https://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html
    let trace = cov_matrix[0][0] + cov_matrix[1][1];
    let determinant = cov_matrix[0][0] * cov_matrix[1][1] - cov_matrix[0][1] * cov_matrix[1][0];

    let eigenvalue = trace / 2.0 + (trace.powi(2) / 4.0 - determinant).sqrt();

    let eigenvector = if cov_matrix[1][0] != 0.0 {
        [eigenvalue - cov_matrix[1][1], cov_matrix[1][0]]
    } else if cov_matrix[0][1] != 0.0 {
        [cov_matrix[0][1], eigenvalue - cov_matrix[0][0]]
    } else {
        [1.0, 0.0]
    };

    // find the bounding box of the points aligned with the eigenvector
    let norm = (eigenvector[0].powi(2) + eigenvector[1].powi(2)).sqrt();

    let axis1 = [eigenvector[0] / norm, eigenvector[1] / norm];
    let axis2 = [-axis1[1], axis1[0]];

    let max1 = points
        .iter()
        .map(|p| p[0] * axis1[0] + p[1] * axis1[1])
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let min1 = points
        .iter()
        .map(|p| p[0] * axis1[0] + p[1] * axis1[1])
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    let max2 = points
        .iter()
        .map(|p| p[0] * axis2[0] + p[1] * axis2[1])
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let min2 = points
        .iter()
        .map(|p| p[0] * axis2[0] + p[1] * axis2[1])
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    let mean1 = (max1 + min1) / 2.0;
    let mean2 = (max2 + min2) / 2.0;
    let scale1 = (max1 - min1) / 2.0;
    let scale2 = (max2 - min2) / 2.0;

    // find the affine transform that maps the unit square to the bounding box
    let t = [
        axis1[0] * mean1 + axis2[0] * mean2,
        axis1[1] * mean1 + axis2[1] * mean2,
    ];
    #[rustfmt::skip]
    let mat = [
        axis1[0] * scale1, axis2[0] * scale2,
        axis1[1] * scale1, axis2[1] * scale2,
    ];

    (mat, t)
}

/// Returns the affine transformation equivalent to applying `a` and then `b`.
pub fn affine_affine(a: Affine, b: Affine) -> Affine {
    // C*x + c = B*(A*x + a) + b
    //    =>   = B*A*x + B*a + b
    //    => C = B*A
    //       c = B*a + b

    let ([a00, a01, a10, a11], [a20, a21]) = a;
    let ([b00, b01, b10, b11], [b20, b21]) = b;

    let c00 = b00 * a00 + b01 * a10;
    let c01 = b00 * a01 + b01 * a11;
    let c10 = b10 * a00 + b11 * a10;
    let c11 = b10 * a01 + b11 * a11;

    let c20 = b00 * a20 + b01 * a21 + b20;
    let c21 = b10 * a20 + b11 * a21 + b21;

    ([c00, c01, c10, c11], [c20, c21])
}

#[allow(clippy::len_without_is_empty)]
pub trait Buffer {
    type Element;

    fn len(&self) -> usize;

    fn aggregate(&mut self, i: usize, v: i8, max: &mut Self::Element);
}

impl<P: From<i8> + std::ops::AddAssign + Ord + Clone> Buffer for [P] {
    type Element = P;

    fn len(&self) -> usize {
        <[P]>::len(self)
    }

    fn aggregate(&mut self, i: usize, v: i8, max: &mut Self::Element) {
        self[i] += v.into();
        let p = self[i].clone();
        if p > *max {
            *max = p;
        }
    }
}
impl Buffer for &[AtomicI32] {
    type Element = AtomicI32;

    fn len(&self) -> usize {
        (*self).len()
    }

    fn aggregate(&mut self, i: usize, v: i8, max: &mut Self::Element) {
        let v = self[i].fetch_add(v as i32, Ordering::Relaxed) + v as i32;
        max.fetch_max(v, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AntiAliasing {
    #[default]
    None,
    Bilinear,
    Lanczos,
}

/// Renders the attractor to a 8-bit grayscale bitmap.
pub fn aggregate_to_bitmap<B: Buffer + ?Sized>(
    attractor: &mut Attractor,
    width: usize,
    height: usize,
    samples: u64,
    anti_aliasing: AntiAliasing,
    buffer: &mut B,
    max: &mut B::Element,
) {
    assert_eq!(buffer.len(), width * height);

    let mut p = attractor.start;
    for _ in 0..samples {
        p = attractor.step(p);
        match anti_aliasing {
            AntiAliasing::None => draw_point(p, width, height, buffer, max),
            AntiAliasing::Bilinear => draw_point_bilinear(p, width, height, buffer, max),
            AntiAliasing::Lanczos => draw_point_lanczos::<3, _>(p, width, height, buffer, max),
        }
    }

    attractor.start = p;
}

pub fn generate_thumbnail(attractor: &Attractor) -> [i16; THUMB_WIDTH * THUMB_HEIGHT] {
    let samples = 1 << 16;

    let mut buffer = [0i16; THUMB_WIDTH * THUMB_HEIGHT];
    let mut attractor = *attractor;

    let bounds = attractor.get_bounds(samples);

    assert_eq!(buffer.len(), THUMB_WIDTH * THUMB_HEIGHT);

    let mut p = attractor.start;
    let mut max = 0i16;

    for _ in 0..samples {
        p = attractor.step(p);
        let pos = map_bounds(
            p,
            bounds,
            [0.0, THUMB_WIDTH as f64, 0.0, THUMB_HEIGHT as f64],
        );
        draw_point(pos, THUMB_WIDTH, THUMB_HEIGHT, &mut buffer[..], &mut max);
    }

    attractor.start = p;

    buffer
}

fn select_nth(values: &mut [i16], nth: usize) -> i16 {
    let pivot = values[0];

    let mut last = values.len();
    let mut i = 0;

    while i < last {
        if values[i] <= pivot {
            i += 1;
            continue;
        }
        last -= 1;
        values.swap(i, last);
    }

    let j = last - 1;
    values.swap(0, j);

    match j {
        _ if j == nth => pivot,
        _ if j < nth => select_nth(&mut values[j + 1..], nth - (j + 1)),
        _ => select_nth(&mut values[..j], nth),
    }
}

/// Get a reference color intensity of the attractor, to be used as a base for the color range of
/// the attractor.
pub fn get_base_intensity(attractor: &Attractor) -> i16 {
    let mut thumbnail = generate_thumbnail(attractor);

    const THRESHOLD: i16 = 2;

    // compute the median intensity of values greater than THRESHOLD
    let mut count = thumbnail.len();

    let mut i = 0;
    while i < count {
        if thumbnail[i] > THRESHOLD {
            i += 1;
            continue;
        }
        count -= 1;
        thumbnail.swap(i, count);
    }

    // ascii_histogram(
    //     "thumbnail intensity",
    //     &thumbnail.map(|x| x as f64),
    //     10.0,
    //     false,
    // );
    // ascii_histogram(
    //     "thumbnail non-zero intensity",
    //     &thumbnail.map(|x| x as f64)[0..count],
    //     10.0,
    //     false,
    // );

    select_nth(&mut thumbnail[0..count], count * 3 / 4)
}

/// Get a reference for the area covered by the attractor.
pub fn get_base_area(attractor: &Attractor) -> u16 {
    let mut thumbnail = generate_thumbnail(attractor);

    const THRESHOLD: i16 = 2;

    // compute the median intensity of values greater than THRESHOLD
    let mut count = thumbnail.len();

    let mut i = 0;
    while i < count {
        if thumbnail[i] > THRESHOLD {
            i += 1;
            continue;
        }
        count -= 1;
        thumbnail.swap(i, count);
    }

    // Make sure that the maximum possible area is within the return type.
    const _: () = assert!(THUMB_WIDTH * THUMB_HEIGHT <= u16::MAX as usize);

    count as u16
}

/// Estimate the ammount of noise in the bitmap by the sum of the sum of absolute differences along
/// each row of the image, divide by its mean.
///
/// As the attractor render converges to a clear image, the noise will tend to a constant value. So
/// we can use the rate of change of this values a estimative of how close we are to convergence.
pub fn estimate_noise(bitmap: &[i32], width: usize, height: usize) -> f64 {
    let mut i = 0;
    let mut noise = 0u64;
    let mut sum = 0u64;
    for _ in 0..width {
        sum += bitmap[i].unsigned_abs() as u64;
        i += 1;

        for _ in 1..height {
            noise += (bitmap[i] - bitmap[i - 1]).unsigned_abs() as u64;
            sum += bitmap[i].unsigned_abs() as u64;
            i += 1;
        }
    }

    noise as f64 / sum as f64
}

fn draw_point<B: Buffer + ?Sized>(
    p: [f64; 2],
    width: usize,
    height: usize,
    buffer: &mut B,
    max: &mut B::Element,
) {
    let x = p[0] as usize;
    let y = p[1] as usize;

    if x < width && y < height {
        buffer.aggregate(y * width + x, 1, max);
    } else {
        // println!("out! {:?}", p);
    }
}

fn draw_point_bilinear<B: Buffer + ?Sized>(
    p: [f64; 2],
    width: usize,
    height: usize,
    buffer: &mut B,
    max: &mut B::Element,
) {
    // anti-aliasing with bilinear interpolation
    let x0 = p[0].floor() as usize;
    let y0 = p[1].floor() as usize;
    let x1 = p[0].ceil() as usize;
    let y1 = p[1].ceil() as usize;

    let fx = p[0] - x0 as f64;
    let fy = p[1] - y0 as f64;

    let c0 = (1.0 - fx) * (1.0 - fy);
    let c1 = fx * (1.0 - fy);
    let c2 = (1.0 - fx) * fy;
    let c3 = fx * fy;

    let mut add = |buffer: &mut B, x, y, c: i8| {
        if x < width && y < height {
            buffer.aggregate(y * width + x, c, max);
        }
    };

    add(buffer, x0, y0, (c0 * 64.0) as i8);
    add(buffer, x1, y0, (c1 * 64.0) as i8);
    add(buffer, x0, y1, (c2 * 64.0) as i8);
    add(buffer, x1, y1, (c3 * 64.0) as i8);
}

/// Draw a anti-aliased point using 2x2 Lanczos kernel.
fn draw_point_lanczos<const W: usize, B: Buffer + ?Sized>(
    p: [f64; 2],
    width: usize,
    height: usize,
    buffer: &mut B,
    max: &mut B::Element,
) {
    // anti-aliasing with bilinear interpolation
    let x: [usize; W] = std::array::from_fn(|i| (p[0] - W as f64 / 2.0 + i as f64) as usize);
    let y: [usize; W] = std::array::from_fn(|i| (p[1] - W as f64 / 2.0 + i as f64) as usize);

    // The lanczos kernel
    let l = |x: f64| {
        let a = W as f64 / 2.0;
        if x == 0.0 {
            1.0
        } else if x.abs() < a {
            let pi_x = x * std::f64::consts::PI;
            a * pi_x.sin() * (pi_x / a).sin() / (pi_x * pi_x)
        } else {
            0.0
        }
    };

    for y in y {
        if y >= height {
            break;
        }
        for x in x {
            if x >= width {
                continue;
            }

            let dx = (x as f64 - p[0]).abs();
            let dy = (y as f64 - p[1]).abs();

            let c = l(dx) * l(dy);

            buffer.aggregate(y * width + x, (c * 64.0) as i8, max);
        }
    }
}

/// Plot a histogram of the given data.
/// The bins are show in the vertical axis, and the values as horizontal bars.
pub fn ascii_histogram(title: &str, data: &[f64], step: f64, log: bool) -> String {
    let min = *data.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
    let max = *data.iter().max_by(|a, b| a.total_cmp(b)).unwrap();

    if min.is_nan() || max.is_nan() {
        return "NaN".to_string();
    }

    let (min, max) = ((min / step).round() * step, (max / step).round() * step);

    let map = |x: f64| {
        if log {
            x.log2().round() as usize
        } else {
            ((x - min) / step).round() as usize
        }
    };

    let range = map(max) + 1;

    let mut hist = vec![0; range];
    for &v in data {
        if v < min || v > max {
            continue;
        }
        let i = map(v);
        hist[i] += 1;
    }

    let max_count = *hist.iter().max().unwrap();

    let mut out = format!("     / {}\n", title);

    for (i, &count) in hist.iter().enumerate() {
        let v = (count as f64 / max_count as f64) * 60.0;
        let v = v.ceil() as usize;
        let x = if log {
            2usize.pow(i as u32) as f64
        } else {
            i as f64 * step + min
        };
        let digits_after_dot = if log {
            0
        } else {
            -step.log10().ceil() as usize
        };
        out.push_str(&format!(
            "{:5.d$} |{:=>count$}\n",
            x,
            "",
            count = v,
            d = digits_after_dot
        ));
    }

    println!("{}", out);
    out
}

#[cfg(test)]
mod test;
