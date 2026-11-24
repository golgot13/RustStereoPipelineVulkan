//! Hybrid GUI Launcher supporting three backends:
//! 1. `egui` / `eframe` GPU-accelerated desktop studio (optional `egui-ui` feature, enabled by default)
//! 2. Native Linux X11 desktop window (`libX11.so.6` dynamic FFI, zero crates)
//! 3. Embedded HTTP/1.1 Web UI (`std::net::TcpListener`, zero crates, works headless/SSH)

#[cfg(feature = "egui-ui")]
pub mod egui_app;
pub mod framebuffer;
pub mod web;
pub mod x11;

use crate::camera::AnyCamera;
use crate::config::StereoSettings;
use crate::image::ImageF32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiMode {
    Auto,
    Egui,
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
        UiMode::Egui => launch_egui(left, right, cam_left, cam_right, settings),
        UiMode::X11 => x11::run_x11_gui(left, right, cam_left, cam_right, settings),
        UiMode::Web => web::run_web_ui(left, right, cam_left, cam_right, settings, web_port),
        UiMode::Auto => {
            #[cfg(feature = "egui-ui")]
            {
                let has_display = !std::env::var("DISPLAY").unwrap_or_default().is_empty()
                    || !std::env::var("WAYLAND_DISPLAY").unwrap_or_default().is_empty()
                    || cfg!(target_os = "windows")
                    || cfg!(target_os = "macos");
                if has_display {
                    println!("[GUI] Launching egui/eframe Desktop Studio...");
                    return egui_app::run_egui_gui(left, right, cam_left, cam_right, settings);
                }
            }
            if x11::can_open_x11_display() {
                println!("[GUI] Detected live X11 DISPLAY — launching Native X11 Workstation...");
                x11::run_x11_gui(left, right, cam_left, cam_right, settings)
            } else {
                println!("[GUI] No desktop display detected — falling back to Embedded Web UI...");
                web::run_web_ui(left, right, cam_left, cam_right, settings, web_port)
            }
        }
    }
}

#[cfg(feature = "egui-ui")]
fn launch_egui(
    left: ImageF32,
    right: ImageF32,
    cam_left: AnyCamera,
    cam_right: AnyCamera,
    settings: StereoSettings,
) -> Result<(), String> {
    egui_app::run_egui_gui(left, right, cam_left, cam_right, settings)
}

#[cfg(not(feature = "egui-ui"))]
fn launch_egui(
    _left: ImageF32,
    _right: ImageF32,
    _cam_left: AnyCamera,
    _cam_right: AnyCamera,
    _settings: StereoSettings,
) -> Result<(), String> {
    Err("Binary compiled without `egui-ui` feature. Rebuild with default features (`cargo build --release`) or use `--x11` / `--web`.".into())
}
