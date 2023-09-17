use clap::Parser;

fn main() {
    let cli = render::Cli::parse();

    pollster::block_on(render::run(cli));
}
