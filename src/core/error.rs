//! Typed zero-panic error hierarchy with standardized POSIX exit codes.

use std::fmt;
use std::io;

/// Standardized error enum for all pipeline, I/O, geometry, compute, and CLI operations.
#[derive(Debug)]
pub enum PipelineError {
    Cli(String),
    Config(String),
    Io(String),
    Format(String),
    DimensionMismatch {
        expected: (u32, u32),
        actual: (u32, u32),
    },
    Geometry(String),
    Compute(String),
    MemoryBudget {
        required_mb: u64,
        budget_mb: u64,
    },
    Cancelled,
}

pub type Result<T> = std::result::Result<T, PipelineError>;

impl PipelineError {
    /// POSIX-compliant exit code for automated scripts and CI pipelines.
    /// - `0`   : success (not an error)
    /// - `2`   : CLI / config argument error
    /// - `3`   : I/O or file format error
    /// - `4`   : compute / Vulkan / memory budget error
    /// - `5`   : photogrammetric geometry / dimension error
    /// - `130` : cancelled by user or signal
    pub fn exit_code(&self) -> u8 {
        match self {
            Self::Cli(_) | Self::Config(_) => 2,
            Self::Io(_) | Self::Format(_) => 3,
            Self::Compute(_) | Self::MemoryBudget { .. } => 4,
            Self::DimensionMismatch { .. } | Self::Geometry(_) => 5,
            Self::Cancelled => 130,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Self::Cli(_) => "cli_error",
            Self::Config(_) => "config_error",
            Self::Io(_) => "io_error",
            Self::Format(_) => "format_error",
            Self::DimensionMismatch { .. } => "dimension_mismatch",
            Self::Geometry(_) => "geometry_error",
            Self::Compute(_) => "compute_error",
            Self::MemoryBudget { .. } => "memory_budget_exceeded",
            Self::Cancelled => "cancelled",
        }
    }
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cli(m) => write!(f, "CLI error: {}", m),
            Self::Config(m) => write!(f, "Configuration error: {}", m),
            Self::Io(m) => write!(f, "I/O error: {}", m),
            Self::Format(m) => write!(f, "Format error: {}", m),
            Self::DimensionMismatch { expected, actual } => write!(
                f,
                "Dimension mismatch: left is {}x{}, right is {}x{}",
                expected.0, expected.1, actual.0, actual.1
            ),
            Self::Geometry(m) => write!(f, "Geometry error: {}", m),
            Self::Compute(m) => write!(f, "Compute error: {}", m),
            Self::MemoryBudget {
                required_mb,
                budget_mb,
            } => write!(
                f,
                "Memory budget exceeded: estimated {} MB > budget {} MB (reduce tile size or search window)",
                required_mb, budget_mb
            ),
            Self::Cancelled => write!(f, "Operation cancelled"),
        }
    }
}

impl std::error::Error for PipelineError {}

impl From<io::Error> for PipelineError {
    fn from(err: io::Error) -> Self {
        Self::Io(err.to_string())
    }
}

impl From<String> for PipelineError {
    fn from(s: String) -> Self {
        Self::Compute(s)
    }
}
