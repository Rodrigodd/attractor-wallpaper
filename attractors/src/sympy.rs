//! Module with code generated using SymPy. Check notebook for the code generator.

#[allow(non_snake_case)]
pub fn apply_affine_transform_to_attractor(
    a: [f64; 6],
    b: [f64; 6],
    p: [f64; 2],
    A: [f64; 4],
    t: [f64; 2],
) -> ([f64; 6], [f64; 6], [f64; 2]) {
    let [x, y] = p;
    let x0 = A[0] * A[3];
    let x1 = A[1] * A[2];
    let x2 = (x0 - x1).recip();
    let x3 = A[3] * t[0];
    let x4 = A[1] * b[1];
    let x5 = A[1] * t[1];
    let x6 = (t[0]).powi(2);
    let x7 = A[1] * b[2];
    let x8 = (t[1]).powi(2);
    let x9 = A[1] * b[5];
    let x10 = x5 * b[3];
    let x11 = x1 * b[3];
    let x12 = 2.0 * t[1];
    let x13 = x1 * b[5];
    let x14 = A[2] * a[3];
    let x15 = A[3] * a[5];
    let x16 = A[2] * t[1];
    let x17 = A[0] * t[0];
    let x18 =
        x10 * A[0] - x14 * x3 - 2.0 * x15 * x16 + 2.0 * x17 * x7 + x4 * A[0] - A[2] * A[3] * a[4];
    let x19 = (A[0]).powi(2);
    let x20 = x19 * x7;
    let x21 = (A[2]).powi(2);
    let x22 = -x0 * x14 + x11 * A[0];
    let x23 = (A[1]).powi(2);
    let x24 = x23 * b[3];
    let x25 = (A[3]).powi(2);
    let x26 = A[0] * b[2];
    let x27 = x23 * x26;
    let x28 = 2.0 * A[3];
    let x29 = A[1] * b[3];
    let x30 = x0 * x29 - x1 * A[3] * a[3];
    let x31 = 2.0 * t[0];
    let x32 = x31 * b[2];
    let x33 = A[2] * t[0];
    let x34 = A[0] * t[1];
    let x35 = A[0] * b[5];
    let x36 = A[2] * a[2];
    let x37 = A[2] * a[5];
    let x38 = b[3] * t[0];
    let x39 = a[3] * t[1];
    let x40 = A[0] * A[2];
    let x41 = x19 * b[3];
    let x42 = x21 * a[3];
    let x43 = x33 * A[0];
    let x44 = 2.0 * a[2];
    let x45 = 2.0 * A[2] * b[5];
    let new_a = [
        x2 * (-x10 * t[0] - x3 - x4 * t[0] - x5 * b[4] - x6 * x7 + x6 * A[3] * a[2] - x8 * x9
            + x8 * A[3] * a[5]
            - A[1] * b[0]
            + A[1] * t[1]
            + A[3] * a[0]
            + A[3] * a[1] * t[0]
            + A[3] * a[3] * t[0] * t[1]
            + A[3] * a[4] * t[1]),
        x2 * (-x1 * b[4] - x11 * t[0] - x12 * x13 - x18
            + A[0] * A[3] * a[1]
            + 2.0 * A[0] * A[3] * a[2] * t[0]
            + A[0] * A[3] * a[3] * t[1]),
        x2 * (x19 * A[3] * a[2] - x20 - x21 * x9 + x21 * A[3] * a[5] - x22),
        x2 * (-x13 * x28 - x24 * A[2] + x25 * A[0] * a[3] + 2.0 * x25 * A[2] * a[5]
            - 2.0 * x27
            - x30
            + 2.0 * A[0] * A[1] * A[3] * a[2]),
        x2 * (-x23 * x32 - x23 * b[1] - x24 * t[1]
            + x25 * a[3] * t[0]
            + x25 * a[4]
            + 2.0 * x25 * a[5] * t[1]
            - x28 * x5 * b[5]
            - x29 * x3
            + A[1] * A[3] * a[1]
            + 2.0 * A[1] * A[3] * a[2] * t[0]
            + A[1] * A[3] * a[3] * t[1]
            - A[1] * A[3] * b[4]),
        x2 * (x23 * A[3] * a[2] - x24 * A[3] - x25 * x9 + x25 * A[1] * a[3]
            - (A[1]).powi(3) * b[2]
            + (A[3]).powi(3) * a[5]),
    ];
    let new_b = [
        x2 * (-x16 * a[4] + x17 * b[1] + x26 * x6 - x33 * x39 - x33 * a[1]
            + x33
            + x34 * x38
            + x34 * b[4]
            - x34
            + x35 * x8
            - x36 * x6
            - x37 * x8
            + A[0] * b[0]
            - A[2] * a[0]),
        x2 * (-x12 * x21 * a[5] - x14 * x34 + x19 * x32 + x19 * b[1] - x21 * a[4] + x34 * x45
            - x40 * a[1]
            + x40 * b[4]
            + x41 * t[1]
            - x42 * t[0]
            - x43 * x44
            + x43 * b[3]),
        x2 * (-x19 * x36 + x21 * x35 + x41 * A[2] - x42 * A[0] + (A[0]).powi(3) * b[2]
            - (A[2]).powi(3) * a[5]),
        x2 * (x0 * x45 - x1 * x44 * A[0] - 2.0 * x15 * x21 + 2.0 * x20 + x22 + x41 * A[3]
            - x42 * A[1]),
        x2 * (x0 * x12 * b[5] + x0 * x38 + x0 * b[4] - x1 * x31 * a[2] - x1 * x39 - x1 * a[1]
            + x18),
        x2 * (-x23 * x36 + x25 * x35 - x25 * x37 + x27 + x30),
    ];
    let new_p = [
        (x - t[0]) * A[3] / (A[0] * A[3] - A[1] * A[2])
            - (y - t[1]) * A[1] / (A[0] * A[3] - A[1] * A[2]),
        -(x - t[0]) * A[2] / (A[0] * A[3] - A[1] * A[2])
            + (y - t[1]) * A[0] / (A[0] * A[3] - A[1] * A[2]),
    ];
    (new_a, new_b, new_p)
}
