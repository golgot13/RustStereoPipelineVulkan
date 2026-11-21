//! Native Linux X11 desktop GUI via zero-dependency dynamic FFI (`dlopen("libX11.so.6")`).
//! Renders interactive stereo layers, controls, Vulkan telemetry, and 3D DEM viewport.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::{c_char, c_int, c_long, c_uint, c_ulong, c_void, CString};
use std::path::Path;
use std::ptr;

use crate::camera::AnyCamera;
use crate::config::{
    ComputeBackendPreference, CostMode, PrefilterMode, StereoAlgorithm, StereoSettings, SubpixelMode,
};
use crate::image::{Colormap, ImageF32};
use crate::pipeline::{synthetic, PipelineResult, StereoPipelineEngine};
use crate::ui::framebuffer::*;
use crate::vulkan::ffi::DynamicLib;

type Display = c_void;
type Window = c_ulong;
type Drawable = c_ulong;
type GC = *mut c_void;
type Visual = c_void;
type Atom = c_ulong;

const EXPOSURE_MASK: c_long = 1 << 15;
const KEY_PRESS_MASK: c_long = 1 << 0;
const BUTTON_PRESS_MASK: c_long = 1 << 2;
const BUTTON_RELEASE_MASK: c_long = 1 << 3;
const POINTER_MOTION_MASK: c_long = 1 << 6;
const STRUCTURE_NOTIFY_MASK: c_long = 1 << 17;

const KEY_PRESS: c_int = 2;
const BUTTON_PRESS: c_int = 4;
const BUTTON_RELEASE: c_int = 5;
const MOTION_NOTIFY: c_int = 6;
const EXPOSE: c_int = 12;
const CONFIGURE_NOTIFY: c_int = 22;
const CLIENT_MESSAGE: c_int = 33;
const Z_PIXMAP: c_int = 2;

#[repr(C)]
#[derive(Clone, Copy)]
struct XKeyEvent {
    type_: c_int,
    serial: c_ulong,
    send_event: c_int,
    display: *mut Display,
    window: Window,
    root: Window,
    subwindow: Window,
    time: c_ulong,
    x: c_int,
    y: c_int,
    x_root: c_int,
    y_root: c_int,
    state: c_uint,
    keycode: c_uint,
    same_screen: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct XButtonEvent {
    type_: c_int,
    serial: c_ulong,
    send_event: c_int,
    display: *mut Display,
    window: Window,
    root: Window,
    subwindow: Window,
    time: c_ulong,
    x: c_int,
    y: c_int,
    x_root: c_int,
    y_root: c_int,
    state: c_uint,
    button: c_uint,
    same_screen: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct XMotionEvent {
    type_: c_int,
    serial: c_ulong,
    send_event: c_int,
    display: *mut Display,
    window: Window,
    root: Window,
    subwindow: Window,
    time: c_ulong,
    x: c_int,
    y: c_int,
    x_root: c_int,
    y_root: c_int,
    state: c_uint,
    is_hint: c_char,
    same_screen: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct XConfigureEvent {
    type_: c_int,
    serial: c_ulong,
    send_event: c_int,
    display: *mut Display,
    event: Window,
    window: Window,
    x: c_int,
    y: c_int,
    width: c_int,
    height: c_int,
    border_width: c_int,
    above: Window,
    override_redirect: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct XClientMessageEvent {
    type_: c_int,
    serial: c_ulong,
    send_event: c_int,
    display: *mut Display,
    window: Window,
    message_type: Atom,
    format: c_int,
    data_l: [c_long; 5],
}

#[repr(C)]
union XEvent {
    type_: c_int,
    xkey: XKeyEvent,
    xbutton: XButtonEvent,
    xmotion: XMotionEvent,
    xconfigure: XConfigureEvent,
    xclient: XClientMessageEvent,
    pad: [c_long; 24],
}

#[repr(C)]
struct XImage {
    width: c_int,
    height: c_int,
    xoffset: c_int,
    format: c_int,
    data: *mut c_char,
    byte_order: c_int,
    bitmap_unit: c_int,
    bitmap_bit_order: c_int,
    bitmap_pad: c_int,
    depth: c_int,
    bytes_per_line: c_int,
    bits_per_pixel: c_int,
    red_mask: c_ulong,
    green_mask: c_ulong,
    blue_mask: c_ulong,
    obdata: *mut c_void,
}

struct X11Api {
    _lib: DynamicLib,
    XOpenDisplay: unsafe extern "C" fn(*const c_char) -> *mut Display,
    XCloseDisplay: unsafe extern "C" fn(*mut Display) -> c_int,
    XDefaultScreen: unsafe extern "C" fn(*mut Display) -> c_int,
    XRootWindow: unsafe extern "C" fn(*mut Display, c_int) -> Window,
    XDefaultVisual: unsafe extern "C" fn(*mut Display, c_int) -> *mut Visual,
    XDefaultDepth: unsafe extern "C" fn(*mut Display, c_int) -> c_int,
    XCreateSimpleWindow: unsafe extern "C" fn(*mut Display, Window, c_int, c_int, c_uint, c_uint, c_uint, c_ulong, c_ulong) -> Window,
    XStoreName: unsafe extern "C" fn(*mut Display, Window, *const c_char) -> c_int,
    XSelectInput: unsafe extern "C" fn(*mut Display, Window, c_long) -> c_int,
    XMapWindow: unsafe extern "C" fn(*mut Display, Window) -> c_int,
    XCreateGC: unsafe extern "C" fn(*mut Display, Drawable, c_ulong, *mut c_void) -> GC,
    XFreeGC: unsafe extern "C" fn(*mut Display, GC) -> c_int,
    XNextEvent: unsafe extern "C" fn(*mut Display, *mut XEvent) -> c_int,
    XPending: unsafe extern "C" fn(*mut Display) -> c_int,
    XCreateImage: unsafe extern "C" fn(*mut Display, *mut Visual, c_uint, c_int, c_int, *mut c_char, c_uint, c_uint, c_int, c_int) -> *mut XImage,
    XPutImage: unsafe extern "C" fn(*mut Display, Drawable, GC, *mut XImage, c_int, c_int, c_int, c_int, c_uint, c_uint) -> c_int,
    XInternAtom: unsafe extern "C" fn(*mut Display, *const c_char, c_int) -> Atom,
    XSetWMProtocols: unsafe extern "C" fn(*mut Display, Window, *mut Atom, c_int) -> c_int,
    XFlush: unsafe extern "C" fn(*mut Display) -> c_int,
}

impl X11Api {
    fn load() -> Result<Self, String> {
        let lib = DynamicLib::open(&[b"libX11.so.6\0", b"libX11.so\0"])
            .ok_or_else(|| "libX11.so.6 not found".to_string())?;
        macro_rules! sym {
            ($name:literal) => {
                unsafe { lib.sym($name) }
                    .ok_or_else(|| format!("Missing X11 symbol: {}", String::from_utf8_lossy($name)))?
            };
        }
        Ok(Self {
            XOpenDisplay: sym!(b"XOpenDisplay\0"),
            XCloseDisplay: sym!(b"XCloseDisplay\0"),
            XDefaultScreen: sym!(b"XDefaultScreen\0"),
            XRootWindow: sym!(b"XRootWindow\0"),
            XDefaultVisual: sym!(b"XDefaultVisual\0"),
            XDefaultDepth: sym!(b"XDefaultDepth\0"),
            XCreateSimpleWindow: sym!(b"XCreateSimpleWindow\0"),
            XStoreName: sym!(b"XStoreName\0"),
            XSelectInput: sym!(b"XSelectInput\0"),
            XMapWindow: sym!(b"XMapWindow\0"),
            XCreateGC: sym!(b"XCreateGC\0"),
            XFreeGC: sym!(b"XFreeGC\0"),
            XNextEvent: sym!(b"XNextEvent\0"),
            XPending: sym!(b"XPending\0"),
            XCreateImage: sym!(b"XCreateImage\0"),
            XPutImage: sym!(b"XPutImage\0"),
            XInternAtom: sym!(b"XInternAtom\0"),
            XSetWMProtocols: sym!(b"XSetWMProtocols\0"),
            XFlush: sym!(b"XFlush\0"),
            _lib: lib,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewLayer {
    Left,
    Right,
    Prefiltered,
    DisparityRaw,
    DisparityFiltered,
    Dem,
    Hillshade,
    ErrorMap,
    Terrain3D,
}

const LAYERS: [(ViewLayer, &str); 9] = [
    (ViewLayer::Left, "1:Left"),
    (ViewLayer::Right, "2:Right"),
    (ViewLayer::Prefiltered, "3:PPRC"),
    (ViewLayer::DisparityRaw, "4:DispRaw"),
    (ViewLayer::DisparityFiltered, "5:DispFltr"),
    (ViewLayer::Dem, "6:DEM"),
    (ViewLayer::Hillshade, "7:Shade"),
    (ViewLayer::ErrorMap, "8:TriErr"),
    (ViewLayer::Terrain3D, "9:3D DEM"),
];

pub fn can_open_x11_display() -> bool {
    if std::env::var("DISPLAY").unwrap_or_default().is_empty() {
        return false;
    }
    if let Ok(x11) = X11Api::load() {
        let dpy = unsafe { (x11.XOpenDisplay)(ptr::null()) };
        if !dpy.is_null() {
            unsafe { (x11.XCloseDisplay)(dpy) };
            return true;
        }
    }
    false
}

/// Launch the interactive native Linux X11 desktop workstation window.
pub fn run_x11_gui(
    mut left: ImageF32,
    mut right: ImageF32,
    mut cam_left: AnyCamera,
    mut cam_right: AnyCamera,
    mut settings: StereoSettings,
) -> Result<(), String> {
    let x11 = X11Api::load()?;
    let dpy = unsafe { (x11.XOpenDisplay)(ptr::null()) };
    if dpy.is_null() {
        return Err("Cannot open X11 DISPLAY (is X11/XWayland running?)".into());
    }

    let engine = StereoPipelineEngine::new(settings.backend);
    let mut result = engine.run_full(&left, &right, &cam_left, &cam_right, &settings, |_, _| {});
    let mut active_layer = ViewLayer::Dem;
    let mut cmap = Colormap::Turbo;
    let mut zoom = 1.0f32;
    let mut pan_x = 0.0f32;
    let mut pan_y = 0.0f32;
    let mut yaw = 0.65f32;
    let mut pitch = 0.55f32;
    let mut dragging = false;
    let mut last_mx = 0i32;
    let mut last_my = 0i32;
    let mut mouse_x = 0i32;
    let mut mouse_y = 0i32;
    let mut status_msg = String::from("Ready — Click controls or press R to re-run pipeline");

    let mut win_w = 1280u32;
    let mut win_h = 800u32;
    let mut fb = Framebuffer::new(win_w, win_h);

    let screen = unsafe { (x11.XDefaultScreen)(dpy) };
    let root = unsafe { (x11.XRootWindow)(dpy, screen) };
    let visual = unsafe { (x11.XDefaultVisual)(dpy, screen) };
    let depth = unsafe { (x11.XDefaultDepth)(dpy, screen) };

    let win = unsafe {
        (x11.XCreateSimpleWindow)(dpy, root, 40, 40, win_w, win_h, 0, 0, COL_BG as c_ulong)
    };
    let title = CString::new("RustStereoPipelineVulkan — Native X11 Stereo & DEM Workstation").unwrap();
    unsafe {
        (x11.XStoreName)(dpy, win, title.as_ptr());
        (x11.XSelectInput)(
            dpy,
            win,
            EXPOSURE_MASK
                | KEY_PRESS_MASK
                | BUTTON_PRESS_MASK
                | BUTTON_RELEASE_MASK
                | POINTER_MOTION_MASK
                | STRUCTURE_NOTIFY_MASK,
        );
        let wm_delete_str = CString::new("WM_DELETE_WINDOW").unwrap();
        let mut wm_delete = (x11.XInternAtom)(dpy, wm_delete_str.as_ptr(), 0);
        (x11.XSetWMProtocols)(dpy, win, &mut wm_delete, 1);
        (x11.XMapWindow)(dpy, win);
    }

    let gc = unsafe { (x11.XCreateGC)(dpy, win, 0, ptr::null_mut()) };
    let wm_delete_atom = unsafe {
        let s = CString::new("WM_DELETE_WINDOW").unwrap();
        (x11.XInternAtom)(dpy, s.as_ptr(), 0)
    };

    let mut running = true;
    let mut needs_redraw = true;

    while running {
        if needs_redraw {
            render_ui(
                &mut fb,
                &engine,
                &settings,
                &result,
                active_layer,
                cmap,
                zoom,
                pan_x,
                pan_y,
                yaw,
                pitch,
                mouse_x,
                mouse_y,
                &status_msg,
            );
            unsafe {
                let ximg = (x11.XCreateImage)(
                    dpy,
                    visual,
                    depth as c_uint,
                    Z_PIXMAP,
                    0,
                    fb.pixels.as_mut_ptr() as *mut c_char,
                    fb.width,
                    fb.height,
                    32,
                    (fb.width * 4) as c_int,
                );
                if !ximg.is_null() {
                    (x11.XPutImage)(dpy, win, gc, ximg, 0, 0, 0, 0, fb.width, fb.height);
                    // Prevent XDestroyImage from freeing Rust Vec memory by just dropping header
                    (*ximg).data = ptr::null_mut();
                }
                (x11.XFlush)(dpy);
            }
            needs_redraw = false;
        }

        let mut ev: XEvent = unsafe { std::mem::zeroed() };
        unsafe { (x11.XNextEvent)(dpy, &mut ev) };
        let ev_type = unsafe { ev.type_ };

        match ev_type {
            EXPOSE => needs_redraw = true,
            CONFIGURE_NOTIFY => {
                let c = unsafe { ev.xconfigure };
                let nw = c.width.max(640) as u32;
                let nh = c.height.max(480) as u32;
                if nw != win_w || nh != win_h {
                    win_w = nw;
                    win_h = nh;
                    fb.resize(win_w, win_h);
                    needs_redraw = true;
                }
            }
            CLIENT_MESSAGE => {
                let cm = unsafe { ev.xclient };
                if cm.data_l[0] as Atom == wm_delete_atom {
                    running = false;
                }
            }
            MOTION_NOTIFY => {
                let m = unsafe { ev.xmotion };
                mouse_x = m.x;
                mouse_y = m.y;
                if dragging {
                    let dx = (m.x - last_mx) as f32;
                    let dy = (m.y - last_my) as f32;
                    if active_layer == ViewLayer::Terrain3D {
                        yaw += dx * 0.01;
                        pitch = (pitch + dy * 0.01).clamp(0.15, 1.45);
                    } else {
                        pan_x += dx;
                        pan_y += dy;
                    }
                    last_mx = m.x;
                    last_my = m.y;
                }
                needs_redraw = true;
            }
            BUTTON_RELEASE => {
                dragging = false;
            }
            BUTTON_PRESS => {
                let b = unsafe { ev.xbutton };
                mouse_x = b.x;
                mouse_y = b.y;
                if b.button == 4 {
                    zoom = (zoom * 1.15).min(10.0);
                    needs_redraw = true;
                } else if b.button == 5 {
                    zoom = (zoom / 1.15).max(0.3);
                    needs_redraw = true;
                } else if b.button == 1 {
                    let layout = UiLayout::compute(win_w as i32, win_h as i32);
                    if layout.viewport.contains(b.x, b.y) {
                        dragging = true;
                        last_mx = b.x;
                        last_my = b.y;
                    } else if let Some(action) = layout.hit_test(b.x, b.y) {
                        match action {
                            UiAction::SelectLayer(l) => {
                                active_layer = l;
                                zoom = 1.0;
                                pan_x = 0.0;
                                pan_y = 0.0;
                            }
                            UiAction::CycleAlgo => {
                                settings.stereo_algorithm = match settings.stereo_algorithm {
                                    StereoAlgorithm::AspBm => StereoAlgorithm::AspSgm,
                                    StereoAlgorithm::AspSgm => StereoAlgorithm::AspMgm,
                                    StereoAlgorithm::AspMgm => StereoAlgorithm::AspBm,
                                };
                            }
                            UiAction::CycleCost => {
                                settings.cost_mode = match settings.cost_mode {
                                    CostMode::Sad => CostMode::Ssd,
                                    CostMode::Ssd => CostMode::Ncc,
                                    CostMode::Ncc => CostMode::Census,
                                    CostMode::Census => CostMode::Sad,
                                };
                            }
                            UiAction::CyclePrefilter => {
                                settings.prefilter_mode = match settings.prefilter_mode {
                                    PrefilterMode::None => PrefilterMode::SubtractedMean,
                                    PrefilterMode::SubtractedMean => PrefilterMode::LaplacianOfGaussian,
                                    PrefilterMode::LaplacianOfGaussian => PrefilterMode::None,
                                };
                            }
                            UiAction::CycleSubpixel => {
                                settings.subpixel_mode = match settings.subpixel_mode {
                                    SubpixelMode::None => SubpixelMode::Parabola,
                                    SubpixelMode::Parabola => SubpixelMode::BayesEmAffine,
                                    SubpixelMode::BayesEmAffine => SubpixelMode::Affine,
                                    SubpixelMode::Affine => SubpixelMode::None,
                                };
                            }
                            UiAction::CycleBackend => {
                                settings.backend = match settings.backend {
                                    ComputeBackendPreference::Auto => ComputeBackendPreference::Vulkan,
                                    ComputeBackendPreference::Vulkan => ComputeBackendPreference::Cpu,
                                    ComputeBackendPreference::Cpu => ComputeBackendPreference::Auto,
                                };
                            }
                            UiAction::CycleColormap => cmap = cmap.next(),
                            UiAction::RunPipeline => {
                                result = engine.run_full(&left, &right, &cam_left, &cam_right, &settings, |_, _| {});
                                status_msg = format!("Completed in {:.1} ms", result.timings.total_ms);
                            }
                            UiAction::GenSynthetic => {
                                let ds = synthetic::generate_synthetic_stereo(192, 144);
                                left = ds.left_image;
                                right = ds.right_image;
                                cam_left = ds.left_cam;
                                cam_right = ds.right_cam;
                                result = engine.run_full(&left, &right, &cam_left, &cam_right, &settings, |_, _| {});
                                status_msg = format!("Synthetic 192x144 solved in {:.1} ms", result.timings.total_ms);
                            }
                            UiAction::ExportOutputs => {
                                match StereoPipelineEngine::save_outputs(Path::new("output/gui_run"), &result) {
                                    Ok(_) => status_msg = "Saved to output/gui_run-*.tif/.ply".into(),
                                    Err(e) => status_msg = format!("Export failed: {}", e),
                                }
                            }
                        }
                        needs_redraw = true;
                    }
                }
            }
            KEY_PRESS => {
                let k = unsafe { ev.xkey };
                // X11 Linux keycodes: ESC=9, 1..9 = 10..18, R=27, C=54, Q=24
                match k.keycode {
                    9 | 24 => running = false,
                    10..=18 => {
                        active_layer = LAYERS[(k.keycode - 10) as usize].0;
                        needs_redraw = true;
                    }
                    27 => {
                        result = engine.run_full(&left, &right, &cam_left, &cam_right, &settings, |_, _| {});
                        status_msg = format!("Re-ran pipeline in {:.1} ms", result.timings.total_ms);
                        needs_redraw = true;
                    }
                    54 => {
                        cmap = cmap.next();
                        needs_redraw = true;
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        // Drain queued motion events for responsiveness
        while unsafe { (x11.XPending)(dpy) } > 16 {
            unsafe { (x11.XNextEvent)(dpy, &mut ev) };
        }
    }

    unsafe {
        (x11.XFreeGC)(dpy, gc);
        (x11.XCloseDisplay)(dpy);
    }
    Ok(())
}

enum UiAction {
    SelectLayer(ViewLayer),
    CycleAlgo,
    CycleCost,
    CyclePrefilter,
    CycleSubpixel,
    CycleBackend,
    CycleColormap,
    RunPipeline,
    GenSynthetic,
    ExportOutputs,
}

struct UiLayout {
    sidebar: Rect,
    topbar: Rect,
    viewport: Rect,
    tabs: Vec<(Rect, ViewLayer, &'static str)>,
    btn_algo: Rect,
    btn_cost: Rect,
    btn_pre: Rect,
    btn_sub: Rect,
    btn_backend: Rect,
    btn_cmap: Rect,
    btn_run: Rect,
    btn_synth: Rect,
    btn_export: Rect,
}

impl UiLayout {
    fn compute(w: i32, h: i32) -> Self {
        let side_w = 270;
        let top_h = 44;
        let sidebar = Rect::new(0, top_h, side_w, h - top_h);
        let topbar = Rect::new(0, 0, w, top_h);
        let viewport = Rect::new(side_w + 8, top_h + 44, w - side_w - 16, h - top_h - 76);

        let mut tabs = Vec::new();
        let tab_w = ((w - side_w - 20) / LAYERS.len() as i32).max(70);
        for (i, &(layer, name)) in LAYERS.iter().enumerate() {
            tabs.push((
                Rect::new(side_w + 8 + i as i32 * (tab_w + 2), top_h + 8, tab_w - 2, 28),
                layer,
                name,
            ));
        }

        let bx = 14;
        let bw = side_w - 28;
        let bh = 28;
        let y0 = top_h + 36;
        Self {
            sidebar,
            topbar,
            viewport,
            tabs,
            btn_algo: Rect::new(bx, y0, bw, bh),
            btn_cost: Rect::new(bx, y0 + 36, bw, bh),
            btn_pre: Rect::new(bx, y0 + 72, bw, bh),
            btn_sub: Rect::new(bx, y0 + 108, bw, bh),
            btn_backend: Rect::new(bx, y0 + 144, bw, bh),
            btn_cmap: Rect::new(bx, y0 + 180, bw, bh),
            btn_run: Rect::new(bx, y0 + 228, bw, 34),
            btn_synth: Rect::new(bx, y0 + 270, bw, 30),
            btn_export: Rect::new(bx, y0 + 308, bw, 30),
        }
    }

    fn hit_test(&self, mx: i32, my: i32) -> Option<UiAction> {
        for &(r, layer, _) in &self.tabs {
            if r.contains(mx, my) {
                return Some(UiAction::SelectLayer(layer));
            }
        }
        if self.btn_algo.contains(mx, my) { return Some(UiAction::CycleAlgo); }
        if self.btn_cost.contains(mx, my) { return Some(UiAction::CycleCost); }
        if self.btn_pre.contains(mx, my) { return Some(UiAction::CyclePrefilter); }
        if self.btn_sub.contains(mx, my) { return Some(UiAction::CycleSubpixel); }
        if self.btn_backend.contains(mx, my) { return Some(UiAction::CycleBackend); }
        if self.btn_cmap.contains(mx, my) { return Some(UiAction::CycleColormap); }
        if self.btn_run.contains(mx, my) { return Some(UiAction::RunPipeline); }
        if self.btn_synth.contains(mx, my) { return Some(UiAction::GenSynthetic); }
        if self.btn_export.contains(mx, my) { return Some(UiAction::ExportOutputs); }
        None
    }
}

#[allow(clippy::too_many_arguments)]
fn render_ui(
    fb: &mut Framebuffer,
    engine: &StereoPipelineEngine,
    settings: &StereoSettings,
    res: &PipelineResult,
    active_layer: ViewLayer,
    cmap: Colormap,
    zoom: f32,
    pan_x: f32,
    pan_y: f32,
    yaw: f32,
    pitch: f32,
    mx: i32,
    my: i32,
    status: &str,
) {
    fb.clear(COL_BG);
    let layout = UiLayout::compute(fb.width as i32, fb.height as i32);

    // Top bar
    fb.fill_rect(layout.topbar, COL_PANEL);
    fb.stroke_rect(layout.topbar, COL_BORDER);
    fb.draw_text(14, 14, "RUST STEREO PIPELINE VULKAN", COL_ACCENT_HI, 2);
    let backend_badge = if engine.vulkan.is_some() { "VULKAN GPU READY" } else { "CPU MULTI-THREAD" };
    let badge_col = if engine.vulkan.is_some() { COL_GREEN } else { COL_AMBER };
    fb.draw_text(fb.width as i32 - 220, 18, backend_badge, badge_col, 1);

    // Sidebar
    fb.fill_rect(layout.sidebar, COL_PANEL);
    fb.stroke_rect(layout.sidebar, COL_BORDER);
    fb.draw_text(14, layout.sidebar.y + 12, "PIPELINE PARAMETERS", COL_TEXT_DIM, 1);

    fb.draw_button(layout.btn_algo, &format!("Algo: {}", settings.stereo_algorithm.name()), false, layout.btn_algo.contains(mx, my), false);
    fb.draw_button(layout.btn_cost, &format!("Cost: {}", settings.cost_mode.name()), false, layout.btn_cost.contains(mx, my), false);
    fb.draw_button(layout.btn_pre, &format!("Prefilter: {}", settings.prefilter_mode.name()), false, layout.btn_pre.contains(mx, my), false);
    fb.draw_button(layout.btn_sub, &format!("Subpixel: {}", settings.subpixel_mode.name()), false, layout.btn_sub.contains(mx, my), false);
    fb.draw_button(layout.btn_backend, &format!("Backend: {:?}", settings.backend), false, layout.btn_backend.contains(mx, my), false);
    fb.draw_button(layout.btn_cmap, &format!("Colormap: {}", cmap.name()), false, layout.btn_cmap.contains(mx, my), false);
    fb.draw_button(layout.btn_run, "RUN STEREO PIPELINE [R]", false, layout.btn_run.contains(mx, my), true);
    fb.draw_button(layout.btn_synth, "NEW SYNTHETIC TERRAIN", false, layout.btn_synth.contains(mx, my), false);
    fb.draw_button(layout.btn_export, "EXPORT TIF/PLY/DEM", false, layout.btn_export.contains(mx, my), false);

    // Stage timings & telemetry panel
    let ty = layout.btn_export.y + 46;
    fb.draw_text(14, ty, "STAGE TELEMETRY (MS)", COL_TEXT_DIM, 1);
    let t = &res.timings;
    let lines = [
        format!("0. PPRC: {:7.1} ms", t.pprc_ms),
        format!("1. CORR: {:7.1} ms", t.corr_ms),
        format!("2. RFNE: {:7.1} ms", t.rfne_ms),
        format!("3. FLTR: {:7.1} ms", t.fltr_ms),
        format!("4. TRI : {:7.1} ms", t.tri_ms),
        format!("5. DEM : {:7.1} ms", t.dem_ms),
        format!("TOTAL  : {:7.1} ms", t.total_ms),
        format!("VALID  : {:6.1}%", res.disp_filtered.valid_fraction() * 100.0),
        format!("POINTS : {}", res.point_cloud.valid_count()),
        format!("TRI ERR: {:.4} m", res.point_cloud.mean_error()),
        format!("VK DISP: {}", t.vk_dispatches),
    ];
    for (i, line) in lines.iter().enumerate() {
        let col = if i == 6 { COL_GREEN } else { COL_TEXT };
        fb.draw_text(14, ty + 18 + i as i32 * 16, line, col, 1);
    }

    // Layer tabs
    for &(r, layer, name) in &layout.tabs {
        fb.draw_button(r, name, layer == active_layer, r.contains(mx, my), false);
    }

    // Main viewport
    match active_layer {
        ViewLayer::Left => fb.blit_image_f32(layout.viewport, &res.left_raw, Colormap::Grayscale, zoom, pan_x, pan_y),
        ViewLayer::Right => fb.blit_image_f32(layout.viewport, &res.right_raw, Colormap::Grayscale, zoom, pan_x, pan_y),
        ViewLayer::Prefiltered => fb.blit_image_f32(layout.viewport, &res.left_pprc, Colormap::Grayscale, zoom, pan_x, pan_y),
        ViewLayer::DisparityRaw => fb.blit_image_f32(layout.viewport, &res.disp_raw.dx_image(), cmap, zoom, pan_x, pan_y),
        ViewLayer::DisparityFiltered => fb.blit_image_f32(layout.viewport, &res.disp_filtered.dx_image(), cmap, zoom, pan_x, pan_y),
        ViewLayer::Dem => fb.blit_image_f32(layout.viewport, &res.dem.elevation, cmap, zoom, pan_x, pan_y),
        ViewLayer::Hillshade => fb.blit_image_f32(layout.viewport, &res.dem.hillshade, Colormap::Grayscale, zoom, pan_x, pan_y),
        ViewLayer::ErrorMap => fb.blit_image_f32(layout.viewport, &res.point_cloud.error_image(), Colormap::Inferno, zoom, pan_x, pan_y),
        ViewLayer::Terrain3D => fb.render_dem_3d(layout.viewport, &res.dem.elevation, cmap, yaw, pitch, zoom),
    }
    fb.stroke_rect(layout.viewport, COL_BORDER);

    // Status bar
    let sb = Rect::new(layout.viewport.x, layout.viewport.y + layout.viewport.h + 4, layout.viewport.w, 22);
    fb.fill_rect(sb, COL_PANEL);
    fb.stroke_rect(sb, COL_BORDER);
    fb.draw_text(sb.x + 8, sb.y + 7, status, COL_TEXT_DIM, 1);
}
