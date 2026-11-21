//! Hybrid Zero-Dependency GUI Launcher:
//! - Native Linux X11 desktop window (`libX11.so.6` dynamic FFI + pure-Rust framebuffer renderer)
//! - Embedded HTTP Web UI (`std::net::TcpListener` + HTML5 Canvas & 3D Terrain viewer)
//! Automatically selects X11 when `$DISPLAY` is available, or falls back to Web UI.

pub mod framebuffer;
pub mod web;
pub mod x11;

use crate::camera::AnyCamera;
use crate::config::StereoSettings;
use crate::image::ImageF32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiMode {
    Auto,
    X11,
    Web,
}

pub fn launch_gui(
    mode: UiMode,
    left: ImageF32,
    right: ImageF32,
    cam_left: AnyCamera,
    cam_right: AnyCamera,
    settings: StereoSettings,
    web_port: u16,
) -> Result<(), String> {
    match mode {
        UiMode::X11 => x11::run_x11_gui(left, right, cam_left, cam_right, settings),
        UiMode::Web => web::run_web_ui(left, right, cam_left, cam_right, settings, web_port),
        UiMode::Auto => {
            if x11::can_open_x11_display() {
                println!("[GUI] Detected live X11 DISPLAY — launching Native X11 Workstation...");
                x11::run_x11_gui(left, right, cam_left, cam_right, settings)
            } else {
                println!("[GUI] No X11 DISPLAY detected — falling back to Embedded Web UI...");
                web::run_web_ui(left, right, cam_left, cam_right, settings, web_port)
            }
        }
    }
}
