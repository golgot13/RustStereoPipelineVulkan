//! Binary entry point for `stereo-pipeline` with standardized POSIX exit codes.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    match stereo_pipeline::cli::run_cli(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("Error [{}]: {}", err.kind(), err);
            ExitCode::from(err.exit_code())
        }
    }
}
