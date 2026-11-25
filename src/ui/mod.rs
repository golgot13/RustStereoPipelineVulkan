//! Zero-Dependency Hybrid GUI Launcher supporting:
//! 1. Embedded HTTP/1.1 + SSE Real-Time 2D/3D Photogrammetry Workstation (`std::net::TcpListener`, 0 external crates, auto-opens browser on Windows/macOS/Linux)
//! 2. Native Linux X11 desktop window (`libX11.so.6` dynamic FFI, 0 external crates)

pub mod framebuffer;
pub mod live_monitor;
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
        UiMode::Web | UiMode::Auto => {
            web::run_web_ui(left, right, cam_left, cam_right, settings, web_port, true)
        }
    }
}
