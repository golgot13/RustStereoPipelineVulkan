//! Core foundational primitives: zero-panic errors, cancellation, hardware governance,
//! pure-Rust JSON, f64 CRS/geodesy, unified JobSpec contract, and aerial chantier ingestion.

pub mod cancel;
pub mod chantier;
pub mod chantier_calib;
pub mod crs;
pub mod error;
pub mod hardware;
pub mod job;
pub mod json;
pub mod lidar_mnt;
