use std::io::{BufWriter, Write};

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
#[ignore]
fn check_period_length() {
    let rng = rand::rngs::SmallRng::from_entropy();
    let attractor = Attractor::find_strange_attractor(rng, 1000).unwrap();

    let affine = super::affine_from_pca(&attractor.get_points::<512>());
    let attractor = attractor.transform_input(affine);

    // after the tranformation, the attractor should be in the range [-1, 1]. We are using a
    // f64, so we can assume that there are a maximu of 2^63 valid points, in each axis, or
    // 2^126 in total. This means that the maximum period length is 2^126.
    //
    // If we consider that the step function is a completely random function, the expected period
    // length should be around (n+1) / 2 ~= 2^125.
    // https://en.wikipedia.org/wiki/Random_permutation_statistics#Expected_cycle_size_of_a_random_element
    //
    // If we consider that only a small subset of the points are valid, like 1/1000, the mean
    // period length should be around 2^116. If we consider that some region of space is about
    // 1,000,000,000 times more dense than the average, the mean period length should be around
    // 2^96.
    //
    // From this analisis, I conclude that I maybe should not worry that the function will
    // cycle before the attractor render converges to a clear image. And that the test below
    // will likely not find a period.

    let mut p = attractor.start;
    for k in 0..32 {
        let mut i = 0u64;
        let start = p;
        let length = 1 << k;
        println!("length: {}", length);
        if p[0] * p[0] + p[1] * p[1] > 1000.0 {
            println!("diverged!, p: {:?}", p);
            panic!("diverged!");
        }
        while i < length {
            p = attractor.step(p);
            i += 1;
            if p == start {
                println!("period length: {}", i);
                panic!();
            }
        }
    }

    panic!();
}

#[test]
#[ignore]
fn noise_estimate_curve() {
    const WIDTH: usize = 512;
    const HEIGHT: usize = 512;

    let zoom = 10.0;

    // let mut rng = rand::rngs::SmallRng::seed_from_u64(145);
    let mut rng = rand::rngs::SmallRng::from_entropy();
    let mut attractor = loop {
        let attractor = Attractor::find_strange_attractor(&mut rng, 1000).unwrap();

        let affine = super::affine_from_pca(&attractor.get_points::<512>());
        let attractor = attractor.transform_input(affine);

        let area = super::get_base_area(&attractor);

        if area < 1 {
            println!("too thin");
            continue;
        }

        let affine = super::map_bounds_affine(
            [0.0, WIDTH as f64, 0.0, HEIGHT as f64],
            [-zoom, zoom, -zoom, zoom],
        );

        break attractor.transform_input(affine);
    };

    let len = 500;
    let samples = 10_000;
    let mut data = Vec::with_capacity(len);

    let mut bitmap = vec![0i32; WIDTH * HEIGHT].into_boxed_slice();

    println!("{}", str::repeat("-", len / (len / 20)));
    for i in 0..len {
        super::aggregate_to_bitmap(
            &mut attractor,
            WIDTH,
            HEIGHT,
            samples,
            AntiAliasing::None,
            &mut bitmap[..],
            &mut 0,
        );

        let noise = super::estimate_noise(&bitmap[..], WIDTH, HEIGHT);

        data.push(noise);

        if i % (len / 20) == 0 {
            print!(".");
            std::io::stdout().flush().unwrap();
        }
    }
    println!();

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("data.txt")
        .unwrap();

    let mut w = BufWriter::new(&mut file);
    for x in data {
        writeln!(w, "{}", x).unwrap();
    }
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

#[test]
fn test_median() {
    const N: usize = 10;
    let mut values: [i16; N] = std::array::from_fn(|i| i as i16);

    let mut rng = rand::rngs::SmallRng::seed_from_u64(0x123);
    for _ in 0..1000 {
        let nth = rng.gen_range(0..N);
        values.shuffle(&mut rng);
        dbg!(&values);
        let m = select_nth(&mut values, nth);
        assert_eq!(m, nth as i16);
    }
}
