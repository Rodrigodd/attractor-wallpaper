use rand::prelude::*;

mod sympy;

type Point = [f64; 2];
/// (min_x, max_x, min_y, max_y)
type Bounds = [f64; 4];
/// Matrix and translation vector
type Affine = ([f64; 4], [f64; 2]);

#[derive(Debug, Clone, Copy)]
pub enum Behavior {
    Convergent { after: usize, to: Point },
    Divergent { after: usize },
    Chaotic { lyapunov: f64, to: Point },
    Periodic { lyapunov: f64, to: Point },
}

#[derive(Debug, Clone, Copy)]
pub struct Attractor {
    a: [f64; 6],
    b: [f64; 6],
    start: Point,
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
            if i > WAIT && too_close((fixed, p1)) {
                // println!("periodic after {} steps", i - WAIT);
                // return Behavior::Convergent { after: i, to: p1 };
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

    pub fn find_strange_attractor(mut rng: impl rand::RngCore, tries: usize) -> Option<Self> {
        for _ in 0..tries {
            let mut attractor = Self::random(&mut rng);
            if let Behavior::Chaotic { lyapunov, to } = attractor.check_behavior() {
                println!("found attractor with lyapunov exponent {}", lyapunov);
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

/// Renders the attractor to a 8-bit grayscale bitmap.
pub fn render_to_bitmap(
    attractor: &Attractor,
    width: usize,
    height: usize,
    samples: usize,
) -> Vec<u8> {
    let mut buffer = vec![0u8; width * height];

    let points = attractor.get_points::<512>(); // 4 KiB
    let affine = affine_from_pca(&points);
    let attractor = attractor.transform_input(affine);
    // println!("{:?}", attractor);

    let bounds = attractor.get_bounds(512);

    let border = 15.0;
    let dst = [
        border,
        width as f64 - border,
        border,
        height as f64 - border,
    ];
    let affine = map_bounds_affine(dst, bounds);
    let attractor = attractor.transform_input(affine);
    // println!("{:?}", attractor);

    let end_bounds = attractor.get_bounds(512);

    let mut p = attractor.start;
    let mut max = 0;
    for _ in 0..samples {
        p = attractor.step(p);

        let x = p[0] as usize;
        let y = p[1] as usize;

        if x < width && y < height {
            let p = &mut buffer[y * width + x];
            if *p == 0 {
                *p = 25;
            }
            *p = p.saturating_add(1);
            if *p > max {
                max = *p;
            }
        } else {
            println!("out! {:?} {:?}", p, end_bounds);
        }
    }

    println!("max: {}", max);

    buffer
}

#[cfg(test)]
mod test {
    use super::*;

    /// Test if `find_strange_attractor` don't returns a attractor that diverges. This test is used
    /// to tweak the `NUM_POINTS` constant, in `Attractor::check_behaviour`. If this fails too
    /// often, increase `NUM_POINTS`.
    #[test]
    fn test_for_nan() {
        let mut rng = rand::rngs::SmallRng::from_entropy();
        for _ in 0..500 {
            let Some(a) = Attractor::find_strange_attractor(&mut rng, 1000) else {
                println!("no attractor found");
                continue;
            };
            let mut p = a.start;
            for _ in 0..10000 {
                p = a.step(p)
            }
        }
    }

    #[test]
    #[ignore]
    fn generate_svg_scatter_plot() {
        let rng = rand::rngs::SmallRng::from_entropy();
        let attractor = Attractor::find_strange_attractor(rng, 1000).unwrap();
        let samples = 10_000;

        let src_bounds = attractor.get_bounds(100);

        let mut svg = r##"
        <svg
            width="800px"
            height="800px"
            viewBox="0 0 1 1"
            version="1"
            xmlns="http://www.w3.org/2000/svg">
        <polygon
            fill="#000000"
            points="0,0 0,1 1,1 1,0"
        />
        <g fill="#FF0000">"##
            .to_string();

        let mut p = attractor.start;

        for _ in 0..samples {
            p = attractor.step(p);

            let pos = map_bounds(p, src_bounds, [0.0, 1.0, 0.0, 1.0]);

            svg.push_str(&format!(
                r#"<circle cx="{:.3}" cy="{:.3}" r="0.002" opacity="0.2"/>"#,
                pos[0], pos[1]
            ));
        }

        svg.push_str(r##"</g></svg>"##);

        std::fs::write("scatter.svg", svg).unwrap();
        panic!("writed to scatter.svg");
    }

    #[test]
    #[ignore]
    fn render_to_png_test() {
        let mut rng = rand::rngs::SmallRng::from_entropy();
        let attractor = Attractor::find_strange_attractor(&mut rng, 1000).unwrap();
        let img = render_to_bitmap(&attractor, 512, 512, 1_000_000);

        image::GrayImage::from_raw(512, 512, img)
            .unwrap()
            .save("scatter.png")
            .unwrap();
        panic!("writed to scatter.png");
    }

    /// Plot a histogram of the given data.
    /// The bins are show in the vertical axis, and the values as horizontal bars.
    fn ascii_histogram(title: &str, data: &[f64], step: f64, log: bool) -> String {
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

    #[test]
    #[ignore]
    fn stats() {
        let mut converge = Vec::new();
        let mut diverge = Vec::new();
        let mut lyapunovs = Vec::new();
        let mut caotic = 0;
        let mut periodic = 0;

        let total = 10_000;
        for _ in 0..total {
            let mut rng = rand::rngs::SmallRng::from_entropy();
            let attractor = Attractor::random(&mut rng);
            let behavior = attractor.check_behavior();
            match behavior {
                Behavior::Convergent { after, .. } => converge.push(after),
                Behavior::Divergent { after, .. } => diverge.push(after),
                Behavior::Chaotic { lyapunov, .. } => {
                    lyapunovs.push(lyapunov);
                    caotic += 1;
                }
                Behavior::Periodic { lyapunov, .. } => {
                    lyapunovs.push(lyapunov);
                    periodic += 1;
                }
            }
        }

        ascii_histogram(
            "coverge after n steps",
            &converge.iter().map(|x| *x as f64).collect::<Vec<_>>(),
            1.0,
            true,
        );
        ascii_histogram(
            "diverge after n steps",
            &diverge.iter().map(|x| *x as f64).collect::<Vec<_>>(),
            1.0,
            true,
        );
        ascii_histogram("lyapunov exponent", &lyapunovs, 0.01, false);

        let total = total as f64;
        println!("converge: {:.2}%", converge.len() as f64 / total * 100.0);
        println!("diverge: {:.2}%", diverge.len() as f64 / total * 100.0);
        println!("caotic: {:.2}%", caotic as f64 / total * 100.0);
        println!("periodic: {:.2}%", periodic as f64 / total * 100.0);
        panic!();
    }

    #[test]
    fn test_map_bounds_affine() {
        let src = [0.0, 1.0, 0.0, 1.0];
        let dst = [0.0, 1.0, 0.0, 1.0];

        let (a, t) = map_bounds_affine(src, dst);
        assert_eq!(a, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(t, [0.0, 0.0]);

        //
        let src = [0.0, 1.0, 0.0, 1.0];
        let dst = [0.0, 2.0, 0.0, 2.0];

        let (a, t) = map_bounds_affine(src, dst);
        assert_eq!(a, [2.0, 0.0, 0.0, 2.0]);
        assert_eq!(t, [0.0, 0.0]);

        //
        let src = [1.0, 2.0, 3.0, 4.0];
        let dst = [0.0, 1.0, 0.0, 1.0];

        let (a, t) = map_bounds_affine(src, dst);
        assert_eq!(a, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(t, [-1.0, -3.0]);
    }

    #[test]
    fn test_transform_input() {
        let attractor = Attractor {
            a: [1.0; 6],
            b: [1.0; 6],
            start: [0.0, 0.0],
        };

        const N: usize = 3;

        let mut p = attractor.start;
        let points: [Point; N] = std::array::from_fn(|_| {
            p = attractor.step(p);
            p
        });

        let transform: Affine = ([1.0, 0.0, 0.0, 1.0], [1.0, 1.0]);
        let attractor_trans = attractor.transform_input(transform);
        let mut p = [
            attractor.start[0] - transform.1[0],
            attractor.start[1] - transform.1[1],
        ];
        let points_trans: [Point; N] = std::array::from_fn(|_| {
            p = attractor_trans.step(p);
            [p[0] + transform.1[0], p[1] + transform.1[1]]
        });

        assert_eq!(points, points_trans);
    }
}
