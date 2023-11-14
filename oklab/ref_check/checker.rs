use std::process::Command;

fn main() {
    // Execute ref_tester and capture its output
    let ref_output = execute_program("ref_tester");

    // Execute our_tester and capture its output
    let our_output = execute_program("our_tester");

    // Split the output strings into vectors of f64 values
    let ref_values: Vec<f64> = ref_output
        .split_whitespace()
        .map(|s| s.parse::<f64>().expect("ref is not float"))
        .collect();

    let our_values: Vec<f64> = our_output
        .split_whitespace()
        .map(|s| s.parse::<f64>().expect("our is not float"))
        .collect();

    if ref_values.len() != our_values.len() {
        println!(
            "The number of values is not equal. {} vs {}",
            ref_values.len(),
            our_values.len()
        );
        return;
    }

    println!("checking {} values", ref_values.len());

    // Check if the values are equal
    for (i, (x, y)) in ref_values.iter().zip(our_values.iter()).enumerate() {
        // println!("{} {}", x, y);
        if (x - y).abs() >= 5e-6 {
            let line = i / 3 % 11;
            let column = i % 3;
            println!("{} {}:{} {} {}", i / 3, line, column, x, y);
            println!("The values are not equal.");
            return;
        }
    }
    println!("The values are equal.");
}

fn execute_program(program_name: &str) -> String {
    let current_dir = std::env::current_dir().expect("Failed to get current working directory.");
    let absolute = current_dir.join(program_name);

    let output = Command::new(absolute)
        .output()
        .expect("Failed to execute program.");

    // Convert the output bytes to a UTF-8 string
    String::from_utf8_lossy(&output.stdout).to_string()
}
