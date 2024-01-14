#![allow(clippy::let_and_return)]

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Gradient<T> {
    pub colors: Vec<(f32, T)>,
}

/// Coefficients for a cubic spline, in three dimensions, in the form
/// (t, x, y, z), where `t` is the time, and `x`, `y`, and `z` are the
/// 4 coefficients for the cubic spline in each dimension.
type SplineCoefs = Vec<(f32, [f32; 4], [f32; 4], [f32; 4])>;

impl<T> Gradient<T> {
    pub fn new(colors: Vec<(f32, T)>) -> Self {
        Self { colors }
    }

    pub fn map<U>(&self, mut f: impl FnMut(&T) -> U) -> Gradient<U> {
        Gradient {
            colors: self.colors.iter().map(|(t, c)| (*t, f(c))).collect(),
        }
    }
}

impl<T: std::ops::Mul<f32, Output = T> + std::ops::Add<Output = T>> Gradient<T> {
    /// Sample the gradient at `t` using linear interpolation.
    pub fn linear_sample(&self, t: f32) -> T
    where
        T: Copy,
    {
        let right_index = self
            .colors
            .iter()
            .position(|x| t < x.0)
            .unwrap_or(self.colors.len());

        if right_index == 0 {
            return self.colors[0].1;
        } else if right_index == self.colors.len() {
            return self.colors[self.colors.len() - 1].1;
        }

        let left = self.colors[right_index - 1];
        let right = self.colors[right_index];

        let span = right.0 - left.0;
        let dt = (t - left.0) / span;

        left.1 * (1.0 - dt) + right.1 * dt
    }

    /// Sample the gradient at `t` using a monotone interpolation.
    pub fn monotone_sample(&self, t: f32) -> T
    where
        T: Copy + Into<[f32; 3]> + From<[f32; 3]>,
    {
        if self.colors.len() == 2 {
            return self.linear_sample(t);
        }

        let right_index = self
            .colors
            .iter()
            .position(|x| t < x.0)
            .unwrap_or(self.colors.len());

        if right_index == 0 {
            return self.colors[0].1;
        } else if right_index == self.colors.len() {
            return self.colors[self.colors.len() - 1].1;
        }

        if right_index == 1 || right_index == self.colors.len() - 1 {
            let (i0, i1, i2);
            if right_index == 1 {
                i0 = right_index - 1;
                i1 = right_index;
                i2 = right_index + 1;
            } else {
                i0 = right_index - 2;
                i1 = right_index - 1;
                i2 = right_index;
            }

            let t0 = self.colors[i0].0;
            let t1 = self.colors[i1].0;
            let t2 = self.colors[i2].0;

            let p0 = self.colors[i0].1;
            let p1 = self.colors[i1].1;
            let p2 = self.colors[i2].1;

            let a1 = p0 * ((t1 - t) / (t1 - t0)) + p1 * ((t - t0) / (t1 - t0));
            let a2 = p1 * ((t2 - t) / (t2 - t1)) + p2 * ((t - t1) / (t2 - t1));

            let b1 = a1 * ((t2 - t) / (t2 - t0)) + a2 * ((t - t0) / (t2 - t0));

            return b1;
        }

        let i0 = right_index - 2;
        let i1 = right_index - 1;
        let i2 = right_index;
        let i3 = right_index + 1;

        let t0 = self.colors[i0].0;
        let t1 = self.colors[i1].0;
        let t2 = self.colors[i2].0;
        let t3 = self.colors[i3].0;

        let p0: [f32; 3] = self.colors[i0].1.into();
        let p1: [f32; 3] = self.colors[i1].1.into();
        let p2: [f32; 3] = self.colors[i2].1.into();
        let p3: [f32; 3] = self.colors[i3].1.into();

        let mut p = [0.0; 3];

        for i in 0..3 {
            p[i] = interpolate_cubic_monotonic_heckbert(
                t,
                (t0, p0[i]),
                (t1, p1[i]),
                (t2, p2[i]),
                (t3, p3[i]),
            );
        }

        p.into()
    }

    pub fn monotonic_hermit_spline_coefs(&self) -> SplineCoefs
    where
        T: Copy + Into<[f32; 3]> + From<[f32; 3]>,
    {
        let mut coefs = Vec::with_capacity(self.colors.len() - 1);
        for i in 0..self.colors.len() - 1 {
            let t0;
            let p0: [f32; 3];
            if i == 0 {
                t0 = self.colors[0].0 - 1.0;
                p0 = self.colors[0].1.into();
            } else {
                let i0 = i - 1;
                t0 = self.colors[i0].0;
                p0 = self.colors[i0].1.into();
            }

            let i1 = i;
            let t1 = self.colors[i1].0;
            let p1: [f32; 3] = self.colors[i1].1.into();

            let i2 = i + 1;
            let t2 = self.colors[i2].0;
            let p2: [f32; 3] = self.colors[i2].1.into();

            let t3;
            let p3: [f32; 3];
            if i == self.colors.len() - 2 {
                t3 = self.colors[self.colors.len() - 1].0 + 1.0;
                p3 = self.colors[self.colors.len() - 1].1.into();
            } else {
                let i3 = i + 2;
                t3 = self.colors[i3].0;
                p3 = self.colors[i3].1.into();
            }

            let mut p = [[0.0; 4]; 3];

            for i in 0..3 {
                p[i] = cubic_monotonic_heckbert_spline(
                    (t0, p0[i]),
                    (t1, p1[i]),
                    (t2, p2[i]),
                    (t3, p3[i]),
                );
            }

            coefs.push((self.colors[i].0, p[0], p[1], p[2]));
        }
        coefs
    }

    /// Sample the gradient at `t` using Catmull-Rom interpolation.
    pub fn catmull_rom_sample(&self, t: f32) -> T
    where
        T: Copy,
    {
        if self.colors.len() == 2 {
            return self.linear_sample(t);
        }

        let right_index = self
            .colors
            .iter()
            .position(|x| t < x.0)
            .unwrap_or(self.colors.len());

        if right_index == 0 {
            return self.colors[0].1;
        } else if right_index == self.colors.len() {
            return self.colors[self.colors.len() - 1].1;
        }

        if right_index == 1 || right_index == self.colors.len() - 1 {
            let (i0, i1, i2);
            if right_index == 1 {
                i0 = right_index - 1;
                i1 = right_index;
                i2 = right_index + 1;
            } else {
                i0 = right_index - 2;
                i1 = right_index - 1;
                i2 = right_index;
            }

            let t0 = self.colors[i0].0;
            let t1 = self.colors[i1].0;
            let t2 = self.colors[i2].0;

            let p0 = self.colors[i0].1;
            let p1 = self.colors[i1].1;
            let p2 = self.colors[i2].1;

            let a1 = p0 * ((t1 - t) / (t1 - t0)) + p1 * ((t - t0) / (t1 - t0));
            let a2 = p1 * ((t2 - t) / (t2 - t1)) + p2 * ((t - t1) / (t2 - t1));

            let b1 = a1 * ((t2 - t) / (t2 - t0)) + a2 * ((t - t0) / (t2 - t0));

            return b1;
        }

        let i0 = right_index - 2;
        let i1 = right_index - 1;
        let i2 = right_index;
        let i3 = right_index + 1;

        let t0 = self.colors[i0].0;
        let t1 = self.colors[i1].0;
        let t2 = self.colors[i2].0;
        let t3 = self.colors[i3].0;

        let p0 = self.colors[i0].1;
        let p1 = self.colors[i1].1;
        let p2 = self.colors[i2].1;
        let p3 = self.colors[i3].1;

        // Barry and Goldman's pyramidal formulation
        let a1 = p0 * ((t1 - t) / (t1 - t0)) + p1 * ((t - t0) / (t1 - t0));
        let a2 = p1 * ((t2 - t) / (t2 - t1)) + p2 * ((t - t1) / (t2 - t1));
        let a3 = p2 * ((t3 - t) / (t3 - t2)) + p3 * ((t - t2) / (t3 - t2));

        let b1 = a1 * ((t2 - t) / (t2 - t0)) + a2 * ((t - t0) / (t2 - t0));
        let b2 = a2 * ((t3 - t) / (t3 - t1)) + a3 * ((t - t1) / (t3 - t1));

        let c1 = b1 * ((t2 - t) / (t2 - t1)) + b2 * ((t - t1) / (t2 - t1));

        c1
    }

    pub fn min(&self) -> f32 {
        self.colors[0].0
    }

    pub fn max(&self) -> f32 {
        self.colors[self.colors.len() - 1].0
    }
}

macro_rules! relative_eq {
    ($a:expr, $b:expr) => {
        ($a - $b).abs() <= 1e-6
    };
}

pub fn interpolate_cubic_monotonic_heckbert(
    t: f32,
    (t0, y0): (f32, f32),
    (t1, y1): (f32, f32),
    (t2, y2): (f32, f32),
    (t3, y3): (f32, f32),
) -> f32 {
    let [a, b, c, d] = cubic_monotonic_heckbert_spline((t0, y0), (t1, y1), (t2, y2), (t3, y3));
    let x = (t - t1) / (t2 - t1);
    ((((a * x) + b) * x) + c) * x + d
}

fn remap(v: f32, from: std::ops::RangeInclusive<f32>, to: std::ops::RangeInclusive<f32>) -> f32 {
    let from = *from.start()..*from.end();
    let to = *to.start()..*to.end();
    (v - from.start) / (from.end - from.start) * (to.end - to.start) + to.start
}

/// Monotone cubic interpolation of points (t1, y1) and (t1, y2) using x as the interpolation
/// parameter (assumed to be [0..1]). In order to maintain C1 continuity, two neighbouring
/// samples are required.
///
/// Reference: http://jbrd.github.io/2020/12/27/monotone-cubic-interpolation.html
pub fn cubic_monotonic_heckbert_spline(
    (t0, y0): (f32, f32),
    (t1, y1): (f32, f32),
    (t2, y2): (f32, f32),
    (t3, y3): (f32, f32),
) -> [f32; 4] {
    // remap everything [t1, t2] to [0, 1]
    let t0 = remap(t0, t1..=t2, 0.0..=1.0);
    let t3 = remap(t3, t1..=t2, 0.0..=1.0);
    let t1 = 0.0;
    let t2 = 1.0;

    // Calculate secant line gradients for each successive pair of data points
    let s_0 = (y1 - y0) / (t1 - t0);
    let s_1 = (y2 - y1) / (t2 - t1);
    let s_2 = (y3 - y2) / (t3 - t2);

    // Use central differences to calculate initial gradients at the end-points
    let mut m_1 = (s_0 + s_1) * 0.5;
    let mut m_2 = (s_1 + s_2) * 0.5;

    // If the central curve (joining y1 and y2) is neither increasing or decreasing, we
    // should have a horizontal line, so immediately set gradients to zero here.
    if relative_eq!(y1, y2) {
        m_1 = 0.0;
        m_2 = 0.0;
    } else {
        // If the curve to the left is horizontal, or the sign of the secants on either side
        // of the end-point are different, set the gradient to zero...
        if relative_eq!(y0, y1) || s_0 < 0.0 && s_1 >= 0.0 || s_0 > 0.0 && s_1 <= 0.0 {
            m_1 = 0.0;
        }
        // ... otherwise, ensure the magnitude of the gradient is constrained to 3 times the
        // left secant, and 3 times the right secant (whatever is smaller)
        else {
            m_1 *= (3.0 * s_0 / m_1).min(3.0 * s_1 / m_1).min(1.0);
        }

        // If the curve to the right is horizontal, or the sign of the secants on either side
        // of the end-point are different, set the gradient to zero...
        if relative_eq!(y2, y3) || s_1 < 0.0 && s_2 >= 0.0 || s_1 > 0.0 && s_2 <= 0.0 {
            m_2 = 0.0;
        }
        // ... otherwise, ensure the magnitude of the gradient is constrained to 3 times the
        // left secant, and 3 times the right secant (whatever is smaller)
        else {
            m_2 *= (3.0 * s_1 / m_2).min(3.0 * s_2 / m_2).min(1.0);
        }
    }

    // Evaluate the cubic hermite spline
    let a = m_1 + m_2 - 2.0 * s_1;
    let b = 3.0 * s_1 - 2.0 * m_1 - m_2;
    let c = m_1;
    let d = y1;

    [a, b, c, d]
}

#[cfg(test)]
mod test {
    use oklab::{OkLch, Oklab};
    use rand::{Rng, SeedableRng};

    use super::{interpolate_cubic_monotonic_heckbert, Gradient};

    #[test]
    #[ignore]
    fn is_monotone() {
        let t0 = 1.0;
        let t1 = 2.0;
        let t2 = 3.0;
        let t3 = 4.0;

        let y0 = t0;
        let y1 = t1;
        let y2 = t2;
        let y3 = t3;

        let t = 2.5;
        let t_ = 2.7;

        let y = interpolate_cubic_monotonic_heckbert(t, (t0, y0), (t1, y1), (t2, y2), (t3, y3));
        let y_ = interpolate_cubic_monotonic_heckbert(t_, (t0, y0), (t1, y1), (t2, y2), (t3, y3));

        println!("t0: {:.2}, y0: {:.2}", t0, y0);
        println!("t1: {:.2}, y1: {:.2}", t1, y1);
        println!("t2: {:.2}, y2: {:.2}", t2, y2);
        println!("t3: {:.2}, y3: {:.2}", t3, y3);
        println!();
        println!("t : {:.2}, y : {:.2}", t, y);
        println!("t_: {:.2}, y_: {:.2}", t_, y_);

        assert!(y < y_);
    }

    fn sort2<T: PartialOrd>(a: T, b: T) -> (T, T) {
        if a < b {
            (a, b)
        } else {
            (b, a)
        }
    }

    #[test]
    fn is_monotone_fuzz() {
        for _ in 0..1000 {
            let t0 = 0.0;
            let t1 = 1.0;
            let t2 = 2.0;
            let t3 = 3.0;

            let mut rng = rand::rngs::SmallRng::seed_from_u64(0);

            let t = rng.gen_range(t1..=t2);
            let t_ = rng.gen_range(t1..=t2);

            let (t, t_) = sort2(t, t_);

            let y0 = rng.gen_range(-1.0..=1.0);
            let y1 = y0 + rng.gen_range(-1.0..=1.0);
            let y2 = y1 + rng.gen_range(-1.0..=1.0);
            let y3 = y2 + rng.gen_range(-1.0..=1.0);

            let y = interpolate_cubic_monotonic_heckbert(t, (t0, y0), (t1, y1), (t2, y2), (t3, y3));

            assert!(y >= y1.min(y2));
            assert!(y <= y1.max(y2));

            let y1_ =
                interpolate_cubic_monotonic_heckbert(t1, (t0, y0), (t1, y1), (t2, y2), (t3, y3));
            assert!((y1_ - y1).abs() < 0.0001);

            let y2_ =
                interpolate_cubic_monotonic_heckbert(t2, (t0, y0), (t1, y1), (t2, y2), (t3, y3));
            assert!((y2_ - y2).abs() < 0.0001);

            let y = interpolate_cubic_monotonic_heckbert(t, (t0, y0), (t1, y1), (t2, y2), (t3, y3));
            let y_ =
                interpolate_cubic_monotonic_heckbert(t_, (t0, y0), (t1, y1), (t2, y2), (t3, y3));

            println!("t0: {}, y0: {}", t0, y0);
            println!("t1: {}, y1: {}", t1, y1);
            println!("t2: {}, y2: {}", t2, y2);
            println!("t3: {}, y3: {}", t3, y3);
            println!();
            println!("t : {}, y : {}", t, y);
            println!("t_: {}, y_: {}", t_, y_);
            assert!(t <= t_);
            assert!((y - y_).abs() < 0.0001 || y1 < y2 && y < y_ || y1 > y2 && y > y_);

            let r1 = (y2 - y1) / (t2 - t1);

            let r = (y_ - y) / (t_ - t);
            let rb = (y_ - y1) / (t_ - t1);
            let ra = (y2 - y_) / (t2 - t_);

            println!("r1: {}", r1);
            println!("r : {}", r);
            println!("rb: {}", rb);
            println!("ra: {}", ra);

            assert!(r.abs() <= 2.0 * r1.abs());
            assert!(rb.abs() <= 2.0 * r1.abs());
            assert!(ra.abs() <= 2.0 * r1.abs());
        }
    }

    #[test]
    fn monotone_color_gradient() {
        let gradient: Gradient<Oklab> = Gradient::new(vec![
            (0.00, OkLch::new(0.00, 0.3, 29.0 / 360.0).into()),
            // (0.01, OkLch::new(0.00, 0.3, 29.0 / 360.0).into()),
            (0.50, OkLch::new(0.50, 0.3, 29.0 / 360.0).into()),
            (0.75, OkLch::new(0.75, 0.3, 69.0 / 360.0).into()),
            // (0.99, OkLch::new(1.00, 0.3, 110.0 / 360.0).into()),
            (1.00, OkLch::new(1.00, 0.3, 110.0 / 360.0).into()),
        ]);

        const N: usize = 10;
        for x in 0..=N {
            let x = x as f32 / N as f32;
            let color = gradient.monotone_sample(x);
            println!("{:4}: {:.2?}", x, color);
        }

        // panic!();
    }
}
