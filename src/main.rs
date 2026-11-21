//! Binary entry point for `stereo-pipeline` (CLI + Hybrid Native X11 / Web UI).

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    match stereo_pipeline::cli::run_cli(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("Error: {}", err);
            ExitCode::FAILURE
        }
    }
}
