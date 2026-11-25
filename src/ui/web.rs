//! Hardened zero-dependency embedded HTTP/1.1 + SSE Real-Time 2D/3D Workstation (`std::net::TcpListener`).
//!
//! Works in two seamless modes with 0 external crates:
//! 1. **Standalone Workstation** (`stereo-pipeline gui` / `stereo-pipeline web-ui`):
//!    Configure and run Full-Scale 1:1 aerial chantiers or Stereo Workbench jobs interactively from the browser.
//! 2. **CLI Live Monitor** (`stereo-pipeline chantier <dir> ... --live-ui [--port 8080]`):
//!    Spawns on a background thread and auto-opens the browser to stream live tile-by-tile
//!    2D RGB/Elevation/Grid mosaics, incremental WebGL 3D colorized point clouds, stage timings, and logs.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use crate::camera::AnyCamera;
use crate::config::{
    ComputeBackendPreference, CostMode, PrefilterMode, StereoAlgorithm, StereoSettings, SubpixelMode,
};
use crate::core::cancel::CancellationToken;
use crate::core::chantier::Chantier;
use crate::core::chantier_runner::{run_full_model_with_monitor, ChantierRunConfig};
use crate::core::crs::CrsId;
use crate::core::hardware::HardwareProfile;
use crate::core::job::{JobInput, JobSpec};
use crate::core::json::parse_json;
use crate::image::io::encode_png_rgba;
use crate::image::{Colormap, ImageF32};
use crate::pipeline::dag::{QualityReport, WorkflowEvent, WorkflowExecutor};
use crate::pipeline::{synthetic, PipelineResult, StereoPipelineEngine};
use crate::ui::live_monitor::{esc_json, open_browser_url, LiveChantierMonitor};
use crate::vulkan::shaders::shader_catalog;

struct WorkbenchState {
    left: ImageF32,
    right: ImageF32,
    cam_left: AnyCamera,
    cam_right: AnyCamera,
    job: JobSpec,
    result: PipelineResult,
    report: Option<QualityReport>,
    engine: Arc<StereoPipelineEngine>,
    cancel: CancellationToken,
    running: bool,
    progress_pct: f32,
    progress_msg: String,
    last_error: Option<String>,
    event_seq: u64,
    last_event_json: String,
}

struct ServerContext {
    monitor: Arc<LiveChantierMonitor>,
    workbench: Option<Arc<Mutex<WorkbenchState>>>,
    chantier_running: Mutex<bool>,
    default_settings: StereoSettings,
    live_cli_attached: bool,
}

fn lock_or_recover<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn bind_first_available_port(start_port: u16) -> Result<(TcpListener, u16), String> {
    if start_port == 0 {
        let listener = TcpListener::bind("127.0.0.1:0")
            .map_err(|e| format!("Impossible d'écouter sur un port libre : {}", e))?;
        let bound = listener.local_addr().map(|a| a.port()).unwrap_or(0);
        return Ok((listener, bound));
    }
    for offset in 0..16u16 {
        let port = start_port.saturating_add(offset);
        let addr = format!("127.0.0.1:{}", port);
        if let Ok(listener) = TcpListener::bind(&addr) {
            return Ok((listener, port));
        }
    }
    Err(format!(
        "Impossible d'écouter sur les ports {}..{}",
        start_port,
        start_port.saturating_add(15)
    ))
}

/// Spawn a non-blocking Live Monitor HTTP+SSE server attached to an ongoing CLI `chantier` run.
/// Returns the bound port number and opens the user's browser automatically when `open_browser` is true.
pub fn spawn_live_monitor_server(
    monitor: Arc<LiveChantierMonitor>,
    port: u16,
    open_browser: bool,
) -> Result<u16, String> {
    let (listener, bound_port) = bind_first_available_port(port)?;
    let ctx = Arc::new(ServerContext {
        monitor,
        workbench: None,
        chantier_running: Mutex::new(false),
        default_settings: StereoSettings::default(),
        live_cli_attached: true,
    });

    let url = format!("http://127.0.0.1:{}", bound_port);
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let c = Arc::clone(&ctx);
            std::thread::spawn(move || {
                let _ = handle_client(stream, c);
            });
        }
    });

    if open_browser {
        open_browser_url(&url);
    }
    Ok(bound_port)
}

/// Start the full standalone zero-dependency Photogrammetry Workstation HTTP/SSE server.
pub fn run_web_ui(
    left: ImageF32,
    right: ImageF32,
    cam_left: AnyCamera,
    cam_right: AnyCamera,
    settings: StereoSettings,
    port: u16,
    open_browser: bool,
) -> Result<(), String> {
    let engine = Arc::new(StereoPipelineEngine::new(settings.backend));
    let hw = HardwareProfile::detect();
    let mut job = JobSpec::default();
    job.input = JobInput::Synthetic {
        width: left.width,
        height: left.height,
    };
    job.output_prefix = "output/webui_run".to_string();
    job.tile_size = hw.tile_size;
    job.write_outputs = false;
    job.write_checkpoints = false;
    job.settings = settings.clone();

    println!(
        "[Studio] Initialisation du moteur stéréo ({})...",
        engine.vulkan_status
    );
    let cancel = CancellationToken::new();
    let initial = WorkflowExecutor::execute_in_memory(
        &job,
        &left,
        &right,
        &cam_left,
        &cam_right,
        None,
        &engine,
        &cancel,
        |_| {},
    )
    .map_err(|e| e.to_string())?;

    let monitor = LiveChantierMonitor::new();
    {
        let mut st = monitor.lock();
        st.backend_info = format!(
            "{} | {} threads CPU (AVX2: {})",
            engine.vulkan_status, settings.threads, hw.avx2_supported
        );
    }

    let workbench = Arc::new(Mutex::new(WorkbenchState {
        left,
        right,
        cam_left,
        cam_right,
        job,
        result: initial.result,
        report: Some(initial.report),
        engine,
        cancel,
        running: false,
        progress_pct: 1.0,
        progress_msg: "Prêt".to_string(),
        last_error: None,
        event_seq: 1,
        last_event_json: r#"{"event":"ready","pct":1.0}"#.to_string(),
    }));

    let ctx = Arc::new(ServerContext {
        monitor,
        workbench: Some(workbench),
        chantier_running: Mutex::new(false),
        default_settings: settings,
        live_cli_attached: false,
    });

    let (listener, bound_port) = bind_first_available_port(port)?;
    let url = format!("http://127.0.0.1:{}", bound_port);
    println!("==================================================================");
    println!(" RustStereoPipelineVulkan — Station Photogrammétrique 1:1 (0 crate)");
    println!(" URL      : {}", url);
    println!(" Fonctions: Suivi Temps Réel SSE | Mosaïque 2D | Nuage 3D WebGL");
    println!(" Appuyez sur Ctrl+C pour arrêter le serveur.");
    println!("==================================================================");
    if open_browser {
        open_browser_url(&url);
    }

    for stream in listener.incoming().flatten() {
        let c = Arc::clone(&ctx);
        std::thread::spawn(move || {
            let _ = handle_client(stream, c);
        });
    }
    Ok(())
}

// ============================================================================
// HTTP/1.1 Request Parser & Router
// ============================================================================

fn read_http_request(stream: &mut TcpStream) -> std::io::Result<Option<(String, String, String, String)>> {
    let mut raw = Vec::with_capacity(8192);
    let mut chunk = [0u8; 4096];
    let mut header_end: Option<usize> = None;
    let mut content_length: usize = 0;

    loop {
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            if raw.is_empty() {
                return Ok(None);
            }
            break;
        }
        raw.extend_from_slice(&chunk[..n]);
        if raw.len() > 512 * 1024 {
            break;
        }
        if header_end.is_none() {
            if let Some(pos) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                header_end = Some(pos + 4);
                let hdr_str = String::from_utf8_lossy(&raw[..pos]);
                for line in hdr_str.lines() {
                    if let Some((k, v)) = line.split_once(':') {
                        if k.trim().eq_ignore_ascii_case("content-length") {
                            content_length = v.trim().parse::<usize>().unwrap_or(0).min(256 * 1024);
                        }
                    }
                }
            }
        }
        if let Some(he) = header_end {
            if raw.len() >= he + content_length {
                break;
            }
        }
    }

    let he = header_end.unwrap_or(raw.len());
    let hdr_str = String::from_utf8_lossy(&raw[..he]);
    let first_line = hdr_str.lines().next().unwrap_or("");
    let mut parts = first_line.split_whitespace();
    let method = parts.next().unwrap_or("GET").to_string();
    let full_path = parts.next().unwrap_or("/");
    let (path, query) = full_path.split_once('?').unwrap_or((full_path, ""));
    let body_end = (he + content_length).min(raw.len());
    let body = String::from_utf8_lossy(&raw[he..body_end]).into_owned();
    Ok(Some((method, path.to_string(), query.to_string(), body)))
}

fn handle_client(mut stream: TcpStream, ctx: Arc<ServerContext>) -> std::io::Result<()> {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(15)));

    let Some((method, path, query, body)) = read_http_request(&mut stream)? else {
        return Ok(());
    };

    match (method.as_str(), path.as_str()) {
        ("GET", "/") | ("GET", "/index.html") => {
            send_response(&mut stream, "200 OK", "text/html; charset=utf-8", INDEX_HTML.as_bytes())
        }

        // ==================== Live Chantier Monitor Endpoints ====================
        ("GET", "/api/live/status") => {
            let json = ctx.monitor.to_status_json();
            send_response(&mut stream, "200 OK", "application/json", json.as_bytes())
        }
        ("GET", "/api/live/events") => {
            let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));
            let hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n\r\n";
            stream.write_all(hdr.as_bytes())?;
            stream.flush()?;
            let mut last_seq = 0u64;
            for _ in 0..240 {
                let (seq, frame) = {
                    let st = ctx.monitor.lock();
                    let pct = if st.total_tiles > 0 {
                        st.tiles_done as f64 / st.total_tiles as f64
                    } else if st.phase == "done" {
                        1.0
                    } else {
                        0.0
                    };
                    (
                        st.event_seq,
                        format!(
                            r#"{{"seq":{},"phase":"{}","pct":{:.4},"tiles_done":{},"total_tiles":{},"mosaic_rev":{},"cloud_rev":{},"cloud_count":{},"total_points":{},"mpx_per_sec":{:.2},"eta_sec":{:.1},"status_msg":"{}"}}"#,
                            st.event_seq,
                            esc_json(&st.phase),
                            pct,
                            st.tiles_done,
                            st.total_tiles,
                            st.mosaic_rev,
                            st.cloud_rev,
                            st.cloud_pts.len(),
                            st.total_points,
                            st.mpx_per_sec,
                            st.eta_sec,
                            esc_json(&st.status_msg)
                        ),
                    )
                };
                if seq != last_seq {
                    last_seq = seq;
                    let msg = format!("data: {}\n\n", frame);
                    if stream.write_all(msg.as_bytes()).is_err() || stream.flush().is_err() {
                        break;
                    }
                }
                std::thread::sleep(Duration::from_millis(200));
            }
            Ok(())
        }
        ("GET", "/api/live/mosaic.png") => {
            let layer = query_param(&query, "layer").unwrap_or("rgb");
            let cmap = query_param(&query, "cmap").unwrap_or("turbo");
            let png = ctx.monitor.render_mosaic_png(layer, cmap);
            send_response(&mut stream, "200 OK", "image/png", &png)
        }
        ("GET", "/api/live/cloud.bin") => {
            let from = query_param(&query, "from")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0);
            let bin = ctx.monitor.encode_cloud_chunk_bin(from);
            send_response(&mut stream, "200 OK", "application/octet-stream", &bin)
        }
        ("POST", "/api/chantier/discover") => handle_chantier_discover(&mut stream, &ctx, &body),
        ("POST", "/api/chantier/start") => handle_chantier_start(&mut stream, &ctx, &body),
        ("POST", "/api/chantier/cancel") => {
            ctx.monitor.cancel.cancel();
            send_response(
                &mut stream,
                "200 OK",
                "application/json",
                br#"{"ok":true,"cancelled":true}"#,
            )
        }

        // ==================== Stereo Workbench Endpoints ====================
        ("GET", "/api/status") => {
            if let Some(ref wb) = ctx.workbench {
                let st = lock_or_recover(wb);
                let json = build_workbench_status_json(&st, ctx.live_cli_attached);
                send_response(&mut stream, "200 OK", "application/json", json.as_bytes())
            } else {
                let json = format!(
                    r#"{{"workbench_available":false,"live_cli_attached":{}}}"#,
                    ctx.live_cli_attached
                );
                send_response(&mut stream, "200 OK", "application/json", json.as_bytes())
            }
        }
        ("GET", "/api/job") => {
            if let Some(ref wb) = ctx.workbench {
                let st = lock_or_recover(wb);
                send_response(&mut stream, "200 OK", "application/json", st.job.to_json_pretty().as_bytes())
            } else {
                send_response(&mut stream, "200 OK", "application/json", b"{}")
            }
        }
        ("POST", "/api/job") => {
            let Some(ref wb) = ctx.workbench else {
                return send_response(&mut stream, "400 Bad Request", "application/json", br#"{"ok":false,"error":"Workbench unavailable in CLI live mode"}"#);
            };
            match JobSpec::from_json_str(&body) {
                Ok(mut spec) => {
                    spec.write_outputs = false;
                    spec.write_checkpoints = false;
                    {
                        let mut st = lock_or_recover(wb);
                        if let JobInput::Synthetic { width, height } = spec.input {
                            if width != st.left.width || height != st.left.height {
                                let ds = synthetic::generate_synthetic_stereo(width, height);
                                st.left = ds.left_image;
                                st.right = ds.right_image;
                                st.cam_left = ds.left_cam;
                                st.cam_right = ds.right_cam;
                            }
                        }
                        st.job = spec;
                    }
                    spawn_workbench_job(Arc::clone(wb));
                    let st = lock_or_recover(wb);
                    let json = build_workbench_status_json(&st, ctx.live_cli_attached);
                    send_response(&mut stream, "200 OK", "application/json", json.as_bytes())
                }
                Err(e) => {
                    let msg = format!(r#"{{"ok":false,"error":"{}"}}"#, esc_json(&e.to_string()));
                    send_response(&mut stream, "400 Bad Request", "application/json", msg.as_bytes())
                }
            }
        }
        ("GET", "/api/cli-command") => {
            if let Some(ref wb) = ctx.workbench {
                let st = lock_or_recover(wb);
                let json = format!(r#"{{"command":"{}"}}"#, esc_json(&st.job.to_cli_command()));
                send_response(&mut stream, "200 OK", "application/json", json.as_bytes())
            } else {
                let st = ctx.monitor.lock();
                let json = format!(r#"{{"command":"{}"}}"#, esc_json(&st.cli_command));
                send_response(&mut stream, "200 OK", "application/json", json.as_bytes())
            }
        }
        ("GET", "/api/quality-report") => {
            if let Some(ref wb) = ctx.workbench {
                let st = lock_or_recover(wb);
                let json = st
                    .report
                    .as_ref()
                    .map(|r| r.to_json_pretty())
                    .unwrap_or_else(|| "{}".to_string());
                send_response(&mut stream, "200 OK", "application/json", json.as_bytes())
            } else {
                send_response(&mut stream, "200 OK", "application/json", b"{}")
            }
        }
        ("GET", "/api/events") => {
            let Some(ref wb) = ctx.workbench else {
                return send_response(&mut stream, "204 No Content", "text/plain", b"");
            };
            let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));
            let hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n\r\n";
            stream.write_all(hdr.as_bytes())?;
            stream.flush()?;
            let mut last_seq = 0u64;
            for _ in 0..120 {
                let (seq, payload) = {
                    let st = lock_or_recover(wb);
                    (st.event_seq, st.last_event_json.clone())
                };
                if seq != last_seq {
                    last_seq = seq;
                    let frame = format!("data: {}\n\n", payload);
                    if stream.write_all(frame.as_bytes()).is_err() || stream.flush().is_err() {
                        break;
                    }
                }
                std::thread::sleep(Duration::from_millis(250));
            }
            Ok(())
        }
        ("POST", "/api/cancel") => {
            ctx.monitor.cancel.cancel();
            if let Some(ref wb) = ctx.workbench {
                lock_or_recover(wb).cancel.cancel();
            }
            send_response(&mut stream, "200 OK", "application/json", br#"{"ok":true,"cancelled":true}"#)
        }
        ("GET", "/api/image") => {
            let Some(ref wb) = ctx.workbench else {
                return send_response(&mut stream, "404 Not Found", "text/plain", b"No workbench image");
            };
            let layer = query_param(&query, "layer").unwrap_or("dem");
            let cmap = Colormap::from_name(query_param(&query, "cmap").unwrap_or("turbo"));
            let st = lock_or_recover(wb);
            let (img, eff_cmap) = match layer {
                "left" => (st.result.left_raw.clone(), Colormap::Grayscale),
                "right" => (st.result.right_raw.clone(), Colormap::Grayscale),
                "pprc" => (st.result.left_pprc.clone(), Colormap::Grayscale),
                "disp_raw" => (st.result.disp_raw.dx_image(), cmap),
                "disp_fltr" => (st.result.disp_filtered.dx_image(), cmap),
                "hillshade" => (st.result.dem.hillshade.clone(), Colormap::Grayscale),
                "error" => (st.result.point_cloud.error_image(), Colormap::Inferno),
                _ => (st.result.dem.elevation.clone(), cmap),
            };
            let rgba = img.to_rgba8(eff_cmap);
            let png = encode_png_rgba(img.width, img.height, &rgba);
            send_response(&mut stream, "200 OK", "image/png", &png)
        }
        ("GET", "/api/dem3d") => {
            let Some(ref wb) = ctx.workbench else {
                return send_response(&mut stream, "404 Not Found", "application/json", b"{}");
            };
            let st = lock_or_recover(wb);
            let json = build_dem3d_json(&st.result.dem.elevation);
            send_response(&mut stream, "200 OK", "application/json", json.as_bytes())
        }
        ("POST", "/api/run") | ("GET", "/api/run") => {
            let Some(ref wb) = ctx.workbench else {
                return send_response(&mut stream, "400 Bad Request", "application/json", br#"{"ok":false}"#);
            };
            let is_async = query_param(&query, "async").unwrap_or("0") == "1";
            {
                let mut st = lock_or_recover(wb);
                if let Some(a) = query_param(&query, "algo") {
                    st.job.settings.stereo_algorithm = StereoAlgorithm::from_str(a);
                }
                if let Some(c) = query_param(&query, "cost").and_then(|v| v.parse().ok()) {
                    st.job.settings.cost_mode = CostMode::from_int(c);
                }
                if let Some(p) = query_param(&query, "prefilter").and_then(|v| v.parse().ok()) {
                    st.job.settings.prefilter_mode = PrefilterMode::from_int(p);
                }
                if let Some(sp) = query_param(&query, "subpixel").and_then(|v| v.parse().ok()) {
                    st.job.settings.subpixel_mode = SubpixelMode::from_int(sp);
                }
                if let Some(k) = query_param(&query, "kernel").and_then(|v| v.parse::<u32>().ok()) {
                    let kc = k.clamp(3, 21);
                    st.job.settings.corr_kernel_x = kc;
                    st.job.settings.corr_kernel_y = kc;
                }
                if let Some(ts) = query_param(&query, "tile").and_then(|v| v.parse::<u32>().ok()) {
                    st.job.tile_size = ts.clamp(64, 4096);
                }
                if let Some(crs) = query_param(&query, "crs") {
                    if let Ok(parsed) = CrsId::parse(crs) {
                        st.job.crs = parsed;
                    }
                }
                if let Some(b) = query_param(&query, "backend") {
                    st.job.settings.backend = ComputeBackendPreference::from_str(b);
                }
            }
            if is_async {
                spawn_workbench_job(Arc::clone(wb));
                let st = lock_or_recover(wb);
                let json = build_workbench_status_json(&st, ctx.live_cli_attached);
                send_response(&mut stream, "202 Accepted", "application/json", json.as_bytes())
            } else {
                let (engine, job, left, right, cl, cr, cancel) = {
                    let st = lock_or_recover(wb);
                    st.cancel.reset();
                    (
                        Arc::clone(&st.engine),
                        st.job.clone(),
                        st.left.clone(),
                        st.right.clone(),
                        st.cam_left.clone(),
                        st.cam_right.clone(),
                        st.cancel.clone(),
                    )
                };
                match WorkflowExecutor::execute_in_memory(&job, &left, &right, &cl, &cr, None, &engine, &cancel, |_| {}) {
                    Ok(out) => {
                        let mut st = lock_or_recover(wb);
                        st.result = out.result;
                        st.report = Some(out.report);
                        st.progress_pct = 1.0;
                        st.progress_msg = format!("Terminé en {:.1} ms", st.result.timings.total_ms);
                        let json = build_workbench_status_json(&st, ctx.live_cli_attached);
                        send_response(&mut stream, "200 OK", "application/json", json.as_bytes())
                    }
                    Err(e) => {
                        let msg = format!(r#"{{"ok":false,"error":"{}"}}"#, esc_json(&e.to_string()));
                        send_response(&mut stream, "500 Internal Server Error", "application/json", msg.as_bytes())
                    }
                }
            }
        }
        ("POST", "/api/synthetic") | ("GET", "/api/synthetic") => {
            let Some(ref wb) = ctx.workbench else {
                return send_response(&mut stream, "400 Bad Request", "application/json", br#"{"ok":false}"#);
            };
            let w = query_param(&query, "w").and_then(|v| v.parse::<u32>().ok()).unwrap_or(160).clamp(64, 512);
            let h = query_param(&query, "h").and_then(|v| v.parse::<u32>().ok()).unwrap_or(120).clamp(48, 384);
            let ds = synthetic::generate_synthetic_stereo(w, h);
            let (engine, job, left, right, cl, cr, cancel) = {
                let mut st = lock_or_recover(wb);
                st.left = ds.left_image;
                st.right = ds.right_image;
                st.cam_left = ds.left_cam;
                st.cam_right = ds.right_cam;
                st.job.input = JobInput::Synthetic { width: w, height: h };
                st.cancel.reset();
                (
                    Arc::clone(&st.engine),
                    st.job.clone(),
                    st.left.clone(),
                    st.right.clone(),
                    st.cam_left.clone(),
                    st.cam_right.clone(),
                    st.cancel.clone(),
                )
            };
            match WorkflowExecutor::execute_in_memory(&job, &left, &right, &cl, &cr, None, &engine, &cancel, |_| {}) {
                Ok(out) => {
                    let mut st = lock_or_recover(wb);
                    st.result = out.result;
                    st.report = Some(out.report);
                    st.progress_pct = 1.0;
                    let json = build_workbench_status_json(&st, ctx.live_cli_attached);
                    send_response(&mut stream, "200 OK", "application/json", json.as_bytes())
                }
                Err(e) => {
                    let msg = format!(r#"{{"ok":false,"error":"{}"}}"#, esc_json(&e.to_string()));
                    send_response(&mut stream, "500 Internal Server Error", "application/json", msg.as_bytes())
                }
            }
        }
        ("POST", "/api/export") | ("GET", "/api/export") => {
            let Some(ref wb) = ctx.workbench else {
                return send_response(&mut stream, "400 Bad Request", "application/json", br#"{"ok":false}"#);
            };
            let st = lock_or_recover(wb);
            let out_p = Path::new(&st.job.output_prefix);
            let msg = match StereoPipelineEngine::save_outputs(out_p, &st.result) {
                Ok(_) => {
                    if let Some(ref rep) = st.report {
                        let _ = rep.save(out_p);
                    }
                    let _ = st.job.save_file(Path::new(&format!("{}-job.json", st.job.output_prefix)));
                    format!(
                        r#"{{"ok":true,"message":"Exporté : {}-{{L,R,D,RD,F,DEM,hillshade}}.tif, -PC.ply & -quality_report.json"}}"#,
                        esc_json(&st.job.output_prefix)
                    )
                }
                Err(e) => format!(r#"{{"ok":false,"message":"Erreur export: {}"}}"#, esc_json(&e.to_string())),
            };
            send_response(&mut stream, "200 OK", "application/json", msg.as_bytes())
        }
        _ => send_response(&mut stream, "404 Not Found", "text/plain", b"Not Found"),
    }
}

// ============================================================================
// Live Chantier Background Discovery & Full-Model Execution Handlers
// ============================================================================

fn parse_chantier_config(body: &str, defaults: &StereoSettings) -> Result<ChantierRunConfig, String> {
    let val = parse_json(body).map_err(|e| format!("JSON invalide: {}", e))?;
    let dir = val
        .get("dir")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "Veuillez renseigner le répertoire du chantier.".to_string())?;

    let mut settings = defaults.clone();
    if let Some(a) = val.get("algo").and_then(|v| v.as_str()) {
        settings.stereo_algorithm = StereoAlgorithm::from_str(a);
    }
    if let Some(c) = val.get("cost").and_then(|v| v.as_i64()) {
        settings.cost_mode = CostMode::from_int(c as i32);
    }
    if let Some(p) = val.get("prefilter").and_then(|v| v.as_i64()) {
        settings.prefilter_mode = PrefilterMode::from_int(p as i32);
    }
    if let Some(sp) = val.get("subpixel").and_then(|v| v.as_i64()) {
        settings.subpixel_mode = SubpixelMode::from_int(sp as i32);
    }
    if let Some(k) = val.get("kernel").and_then(|v| v.as_u64()) {
        let kc = (k as u32).clamp(3, 21);
        settings.corr_kernel_x = kc;
        settings.corr_kernel_y = kc;
    }
    if let Some(p1) = val.get("sgm_p1").and_then(|v| v.as_f64()) {
        settings.sgm_p1 = p1.clamp(0.01, 2.0) as f32;
    }
    if let Some(p2) = val.get("sgm_p2").and_then(|v| v.as_f64()) {
        settings.sgm_p2 = p2.clamp(0.05, 4.0) as f32;
    }
    if let Some(lr) = val.get("lr_threshold").and_then(|v| v.as_f64()) {
        settings.lr_threshold = lr.clamp(0.5, 10.0) as f32;
    }
    if let Some(b) = val.get("backend").and_then(|v| v.as_str()) {
        settings.backend = ComputeBackendPreference::from_str(b);
    }
    if let Some(th) = val.get("threads").and_then(|v| v.as_u64()) {
        settings.threads = (th as usize).clamp(1, 64);
    }

    let output_dir = val
        .get("output_dir")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(PathBuf::from);
    let pair_idx = val.get("pair_idx").and_then(|v| v.as_u64()).map(|v| v as usize);
    let all_pairs = val.get("all_pairs").and_then(|v| v.as_bool()).unwrap_or(false);
    let roi_size = val
        .get("roi_size")
        .and_then(|v| v.as_u64())
        .map(|v| (v as u32).clamp(256, 4096))
        .unwrap_or(1024);
    let express = val.get("express").and_then(|v| v.as_bool()).unwrap_or(true);
    let merge_voxel = val
        .get("merge_voxel")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
        .max(0.0);
    let no_ign_lidar = val.get("no_ign_lidar").and_then(|v| v.as_bool()).unwrap_or(false);
    let no_gcp_refine = val.get("no_gcp_refine").and_then(|v| v.as_bool()).unwrap_or(false);

    Ok(ChantierRunConfig {
        chantier_dir: PathBuf::from(dir),
        output_dir,
        pair_idx,
        all_pairs,
        roi_size,
        express,
        no_merge_cloud: false,
        merge_voxel,
        custom_search: false,
        gcp_picks_path: None,
        footprints_path: None,
        no_gcp_refine,
        ign_data_path: None,
        no_ign_lidar,
        ndjson: false,
        settings,
    })
}

fn handle_chantier_discover(
    stream: &mut TcpStream,
    ctx: &Arc<ServerContext>,
    body: &str,
) -> std::io::Result<()> {
    let cfg = match parse_chantier_config(body, &ctx.default_settings) {
        Ok(c) => c,
        Err(e) => {
            let msg = format!(r#"{{"ok":false,"error":"{}"}}"#, esc_json(&e));
            return send_response(stream, "400 Bad Request", "application/json", msg.as_bytes());
        }
    };
    {
        let mut running = lock_or_recover(&ctx.chantier_running);
        if *running {
            return send_response(
                stream,
                "409 Conflict",
                "application/json",
                r#"{"ok":false,"error":"Un calcul chantier est déjà en cours."}"#.as_bytes(),
            );
        }
        *running = true;
    }

    let ctx_bg = Arc::clone(ctx);
    ctx.monitor
        .set_discovering(&cfg.chantier_dir.display().to_string(), &cfg.to_cli_command());
    std::thread::spawn(move || {
        let out_dir = cfg
            .output_dir
            .clone()
            .unwrap_or_else(|| cfg.chantier_dir.join("Output_stereopipeline"));
        let res = Chantier::discover_with_opts(
            &cfg.chantier_dir,
            cfg.gcp_picks_path.as_deref(),
            cfg.footprints_path.as_deref(),
            !cfg.no_gcp_refine,
            cfg.ign_data_path.as_deref(),
            !cfg.no_ign_lidar,
        );
        match res {
            Ok(chantier) => {
                let engine = StereoPipelineEngine::new(cfg.settings.backend);
                let hw = HardwareProfile::detect();
                let backend_str = format!(
                    "{} | {} threads CPU (AVX2: {})",
                    engine.vulkan_status, cfg.settings.threads, hw.avx2_supported
                );
                ctx_bg.monitor.set_chantier_info(
                    &chantier,
                    &out_dir.display().to_string(),
                    &backend_str,
                    &cfg.to_cli_command(),
                );
                ctx_bg.monitor.push_log(format!(
                    "[OK] {} paires détectées | CRS={} | {} poses | {} GCP",
                    chantier.pairs.len(),
                    chantier.survey_crs.code(),
                    chantier.poses.len(),
                    chantier.gcps.len()
                ));
            }
            Err(e) => {
                ctx_bg.monitor.finish_error(e.to_string());
            }
        }
        *lock_or_recover(&ctx_bg.chantier_running) = false;
    });

    send_response(stream, "202 Accepted", "application/json", br#"{"ok":true}"#)
}

fn handle_chantier_start(
    stream: &mut TcpStream,
    ctx: &Arc<ServerContext>,
    body: &str,
) -> std::io::Result<()> {
    let cfg = match parse_chantier_config(body, &ctx.default_settings) {
        Ok(c) => c,
        Err(e) => {
            let msg = format!(r#"{{"ok":false,"error":"{}"}}"#, esc_json(&e));
            return send_response(stream, "400 Bad Request", "application/json", msg.as_bytes());
        }
    };
    {
        let mut running = lock_or_recover(&ctx.chantier_running);
        if *running {
            return send_response(
                stream,
                "409 Conflict",
                "application/json",
                r#"{"ok":false,"error":"Un calcul chantier est déjà en cours."}"#.as_bytes(),
            );
        }
        *running = true;
    }

    let ctx_bg = Arc::clone(ctx);
    ctx.monitor
        .set_discovering(&cfg.chantier_dir.display().to_string(), &cfg.to_cli_command());
    std::thread::spawn(move || {
        let out_dir = cfg
            .output_dir
            .clone()
            .unwrap_or_else(|| cfg.chantier_dir.join("Output_stereopipeline"));
        let _ = std::fs::create_dir_all(&out_dir);
        let run_res = (|| -> crate::core::error::Result<()> {
            let chantier = Chantier::discover_with_opts(
                &cfg.chantier_dir,
                cfg.gcp_picks_path.as_deref(),
                cfg.footprints_path.as_deref(),
                !cfg.no_gcp_refine,
                cfg.ign_data_path.as_deref(),
                !cfg.no_ign_lidar,
            )?;
            if chantier.pairs.is_empty() {
                return Err(crate::core::error::PipelineError::Geometry(
                    "Aucune paire stéréo détectée dans ce chantier.".into(),
                ));
            }
            let selected: Vec<usize> = if cfg.all_pairs {
                (0..chantier.pairs.len()).collect()
            } else {
                let idx = cfg.pair_idx.unwrap_or(0).min(chantier.pairs.len() - 1);
                vec![idx]
            };
            let engine = StereoPipelineEngine::new(cfg.settings.backend);
            run_full_model_with_monitor(
                &chantier,
                &selected,
                &out_dir,
                &cfg,
                &engine,
                Some(&ctx_bg.monitor),
                true,
            )
        })();
        if let Err(e) = run_res {
            if !matches!(e, crate::core::error::PipelineError::Cancelled) {
                ctx_bg.monitor.finish_error(e.to_string());
            }
        }
        *lock_or_recover(&ctx_bg.chantier_running) = false;
    });

    send_response(stream, "202 Accepted", "application/json", br#"{"ok":true}"#)
}

// ============================================================================
// Stereo Workbench Background Helper & Formatters
// ============================================================================

fn spawn_workbench_job(state: Arc<Mutex<WorkbenchState>>) {
    let (engine, job, left, right, cam_l, cam_r, cancel) = {
        let mut st = lock_or_recover(&state);
        if st.running {
            return;
        }
        st.cancel.reset();
        st.running = true;
        st.progress_pct = 0.02;
        st.progress_msg = "Démarrage pipeline DAG...".to_string();
        st.last_error = None;
        st.event_seq += 1;
        st.last_event_json = r#"{"event":"started","pct":0.02}"#.to_string();
        (
            Arc::clone(&st.engine),
            st.job.clone(),
            st.left.clone(),
            st.right.clone(),
            st.cam_left.clone(),
            st.cam_right.clone(),
            st.cancel.clone(),
        )
    };

    std::thread::spawn(move || {
        let state_cb = Arc::clone(&state);
        let outcome = WorkflowExecutor::execute_in_memory(
            &job,
            &left,
            &right,
            &cam_l,
            &cam_r,
            None,
            &engine,
            &cancel,
            move |ev| {
                let mut st = lock_or_recover(&state_cb);
                st.event_seq += 1;
                st.last_event_json = ev.to_ndjson();
                match ev {
                    WorkflowEvent::StageStart { label, pct, .. } => {
                        st.progress_pct = pct;
                        st.progress_msg = label;
                    }
                    WorkflowEvent::TileProgress { tile, total, pct } => {
                        st.progress_pct = pct;
                        st.progress_msg = format!("Dalle 1:1 {}/{}", tile, total);
                    }
                    WorkflowEvent::StageComplete { pct, .. } => {
                        st.progress_pct = pct;
                    }
                    _ => {}
                }
            },
        );

        let mut st = lock_or_recover(&state);
        st.running = false;
        st.event_seq += 1;
        match outcome {
            Ok(out) => {
                st.progress_pct = 1.0;
                st.progress_msg = format!("Terminé en {:.1} ms", out.result.timings.total_ms);
                st.result = out.result;
                st.report = Some(out.report);
                st.last_event_json = r#"{"event":"done","pct":1.0}"#.to_string();
            }
            Err(e) => {
                st.progress_msg = format!("{}", e);
                st.last_error = Some(e.to_string());
                st.last_event_json = format!(
                    r#"{{"event":"error","kind":"{}","message":"{}"}}"#,
                    e.kind(),
                    esc_json(&e.to_string())
                );
            }
        }
    });
}

fn send_response(stream: &mut TcpStream, status: &str, content_type: &str, body: &[u8]) -> std::io::Result<()> {
    let header = format!(
        "HTTP/1.1 {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        status,
        content_type,
        body.len()
    );
    stream.write_all(header.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

fn query_param<'a>(query: &'a str, key: &str) -> Option<&'a str> {
    for pair in query.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            if k == key {
                return Some(v);
            }
        }
    }
    None
}

fn build_workbench_status_json(st: &WorkbenchState, live_cli_attached: bool) -> String {
    let t = &st.result.timings;
    let hw = st.job.effective_hardware();
    let (dem_min, dem_max, dem_mean, _) = st.result.dem.elevation.stats();
    let shaders_count = shader_catalog().len();
    let cli_cmd = st.job.to_cli_command();
    format!(
        r#"{{"workbench_available":true,"live_cli_attached":{},"running":{},"progress_pct":{:.3},"progress_msg":"{}","width":{},"height":{},"tile_size":{},"crs":"{}","hw_tier":"{}","ram_budget_mb":{},"avx2":{},"vulkan_available":{},"vulkan_status":"{}","active_backend":"{}","vk_dispatches":{},"spirv_shaders":{},"algo":"{}","cost":{},"prefilter":{},"subpixel":{},"kernel":{},"valid_pct":{:.2},"points":{},"tri_error":{:.4},"dem_min":{:.2},"dem_max":{:.2},"dem_mean":{:.2},"cli_command":"{}","timings":{{"pprc":{:.2},"corr":{:.2},"rfne":{:.2},"fltr":{:.2},"tri":{:.2},"dem":{:.2},"total":{:.2}}}}}"#,
        live_cli_attached,
        st.running,
        st.progress_pct,
        esc_json(&st.progress_msg),
        st.left.width,
        st.left.height,
        st.job.tile_size,
        esc_json(&st.job.crs.code()),
        hw.tier.name(),
        hw.ram_budget_mb,
        hw.avx2_supported,
        st.engine.vulkan.is_some(),
        esc_json(&st.engine.vulkan_status),
        esc_json(&t.active_backend),
        t.vk_dispatches,
        shaders_count,
        st.job.settings.stereo_algorithm.name(),
        st.job.settings.cost_mode as i32,
        st.job.settings.prefilter_mode as i32,
        st.job.settings.subpixel_mode as i32,
        st.job.settings.corr_kernel_x,
        st.result.disp_filtered.valid_fraction() * 100.0,
        st.result.point_cloud.valid_count(),
        st.result.point_cloud.mean_error(),
        dem_min,
        dem_max,
        dem_mean,
        esc_json(&cli_cmd),
        t.pprc_ms,
        t.corr_ms,
        t.rfne_ms,
        t.fltr_ms,
        t.tri_ms,
        t.dem_ms,
        t.total_ms
    )
}

fn build_dem3d_json(dem: &ImageF32) -> String {
    let step_x = (dem.width / 52).max(1);
    let step_y = (dem.height / 52).max(1);
    let gw = (dem.width + step_x - 1) / step_x;
    let gh = (dem.height + step_y - 1) / step_y;
    let (min_z, max_z, _, _) = dem.stats();
    let mut vals = String::with_capacity((gw * gh * 7) as usize);
    let mut first = true;
    for j in 0..gh {
        let y = (j * step_y).min(dem.height - 1);
        for i in 0..gw {
            let x = (i * step_x).min(dem.width - 1);
            let v = dem.get(x, y);
            if !first {
                vals.push(',');
            }
            first = false;
            if v.is_finite() {
                vals.push_str(&format!("{:.2}", v));
            } else {
                vals.push_str("null");
            }
        }
    }
    format!(
        r#"{{"gw":{},"gh":{},"min":{:.3},"max":{:.3},"z":[{}]}}"#,
        gw, gh, min_z, max_z, vals
    )
}

// ============================================================================
// Self-Contained Real-Time 2D Mosaic + WebGL 3D Point Cloud Workstation HTML
// ============================================================================

const INDEX_HTML: &str = r##"<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>RustStereoPipelineVulkan — Station Photogrammétrique 1:1 Temps Réel</title>
<style>
  :root {
    --bg:#0b0f17; --panel:#121824; --panel2:#1a2232; --border:#2b3648;
    --accent:#38bdf8; --accent2:#0ea5e9; --green:#22c55e; --amber:#f59e0b; --red:#ef4444;
    --text:#e6edf3; --dim:#8b9bb4;
  }
  * { box-sizing:border-box; margin:0; padding:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; }
  code, pre, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  body { background:var(--bg); color:var(--text); height:100vh; display:flex; flex-direction:column; overflow:hidden; }

  header {
    height:48px; background:var(--panel); border-bottom:1px solid var(--border);
    display:flex; align-items:center; justify-content:space-between; padding:0 16px; gap:10px;
  }
  .brand { display:flex; align-items:center; gap:10px; }
  .brand h1 { font-size:13.5px; letter-spacing:0.6px; color:var(--accent); font-weight:700; }
  .pill { font-size:11px; padding:3px 10px; border-radius:999px; border:1px solid var(--border); background:var(--panel2); font-weight:600; }
  .pill.running { color:#38bdf8; border-color:#38bdf8; background:#0c2d48; }
  .pill.done { color:#22c55e; border-color:#22c55e; background:#0d3320; }
  .pill.error, .pill.cancelled { color:#f87171; border-color:#ef4444; background:#3b1518; }
  .pill.discovering { color:#f59e0b; border-color:#f59e0b; background:#3a2608; }
  .badges { display:flex; gap:8px; align-items:center; }

  main { flex:1; display:flex; min-height:0; }

  aside.left {
    width:330px; background:var(--panel); border-right:1px solid var(--border);
    padding:10px; overflow-y:auto; display:flex; flex-direction:column; gap:10px;
  }
  aside.right {
    width:310px; background:var(--panel); border-left:1px solid var(--border);
    padding:10px; overflow-y:auto; display:flex; flex-direction:column; gap:10px;
  }
  .card { background:var(--panel2); border:1px solid var(--border); border-radius:8px; padding:10px; }
  .card h2 { font-size:11px; color:var(--dim); text-transform:uppercase; letter-spacing:0.7px; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center; }

  label.row { display:flex; justify-content:space-between; align-items:center; font-size:11.5px; margin:5px 0; gap:8px; }
  label.chk { display:flex; align-items:center; gap:6px; font-size:11.5px; margin:4px 0; cursor:pointer; }
  input[type=text], input[type=number], select {
    background:var(--bg); color:var(--text); border:1px solid var(--border);
    border-radius:5px; padding:4px 7px; font-size:11.5px;
  }
  input[type=text].full { width:100%; margin:4px 0 6px 0; }
  select, input[type=number] { width:148px; }

  .btn-row { display:flex; gap:6px; margin-top:6px; }
  button {
    flex:1; padding:7px 9px; border-radius:6px; border:1px solid var(--border);
    background:#233044; color:var(--text); font-size:11.5px; font-weight:600; cursor:pointer;
  }
  button:hover { background:#2e405a; }
  button.primary { background:var(--accent2); border-color:var(--accent); color:#04121f; }
  button.primary:hover { background:var(--accent); }
  button.danger { background:#451a1d; border-color:var(--red); color:#fca5a5; flex:0 0 88px; }
  button.danger:hover { background:#5c2024; }

  .center { flex:1; display:flex; flex-direction:column; min-width:0; }
  .tabs {
    display:flex; flex-wrap:wrap; align-items:center; gap:5px; padding:7px 12px;
    background:var(--panel); border-bottom:1px solid var(--border);
  }
  .tab {
    padding:5px 11px; font-size:11.5px; border-radius:6px; border:1px solid var(--border);
    background:var(--panel2); color:var(--dim); cursor:pointer; font-weight:600;
  }
  .tab.active { background:#10395c; border-color:var(--accent); color:var(--text); }
  .tab-sep { width:1px; height:20px; background:var(--border); margin:0 4px; }

  .viewport {
    flex:1; position:relative; display:flex; align-items:center; justify-content:center;
    background:#06090f; overflow:hidden; min-height:0;
  }
  #mosaicImg, #wbImg {
    max-width:97%; max-height:97%; object-fit:contain;
    border:1px solid var(--border); box-shadow:0 10px 32px rgba(0,0,0,0.65);
  }
  #cloudCanvas { width:100%; height:100%; display:none; cursor:grab; }
  #cloudCanvas:active { cursor:grabbing; }

  .hud3d {
    position:absolute; top:10px; left:10px; display:none; gap:8px; align-items:center; flex-wrap:wrap;
    background:rgba(12,17,27,0.88); border:1px solid var(--border); border-radius:7px; padding:6px 10px; font-size:11px;
    backdrop-filter: blur(4px);
  }
  .hud3d select, .hud3d input[type=range] { width:auto; padding:2px 5px; font-size:11px; }
  .hud3d button { flex:none; padding:3px 8px; font-size:11px; }
  .hud-hint {
    position:absolute; bottom:10px; left:12px; font-size:11px; color:var(--dim);
    background:rgba(11,15,23,0.78); padding:4px 8px; border-radius:5px; pointer-events:none;
  }

  .kpi-grid { display:grid; grid-template-columns:1fr 1fr; gap:6px; }
  .kpi { background:var(--bg); border:1px solid var(--border); border-radius:6px; padding:7px 8px; }
  .kpi .k { font-size:10px; color:var(--dim); text-transform:uppercase; }
  .kpi .v { font-size:14px; font-weight:700; color:var(--text); margin-top:2px; }
  .kpi .v.accent { color:var(--accent); }
  .kpi .v.green { color:var(--green); }

  .kv { display:flex; justify-content:space-between; gap:8px; font-size:11px; margin:3.5px 0; }
  .kv span:first-child { color:var(--dim); flex-shrink:0; }
  .kv span:last-child { text-align:right; word-break:break-word; }

  .prog-track { width:100%; height:10px; background:var(--bg); border:1px solid var(--border); border-radius:999px; overflow:hidden; margin:6px 0; }
  .prog-fill { height:100%; width:0%; background:linear-gradient(90deg, #0ea5e9, #22c55e); transition:width 0.2s; }

  .bar-row { display:flex; align-items:center; gap:6px; font-size:10.5px; margin:3.5px 0; }
  .bar-label { width:54px; color:var(--dim); }
  .bar-track { flex:1; height:7px; background:var(--bg); border-radius:4px; overflow:hidden; }
  .bar-fill { height:100%; background:var(--accent); }
  .bar-val { width:56px; text-align:right; }

  .cli-box {
    background:var(--bg); border:1px solid var(--border); border-radius:5px; padding:6px;
    font-size:10.5px; color:#7dd3fc; word-break:break-all; max-height:66px; overflow-y:auto;
  }
  .log-box {
    background:#070a10; border:1px solid var(--border); border-radius:5px; padding:6px;
    font-size:10px; line-height:1.45; color:#cbd5e1; height:190px; overflow-y:auto; white-space:pre-wrap;
  }

  footer {
    height:28px; background:var(--panel); border-top:1px solid var(--border);
    display:flex; align-items:center; justify-content:space-between; padding:0 14px; font-size:11px; color:var(--dim);
  }
</style>
</head>
<body>
<header>
  <div class="brand">
    <h1>RUST STEREO PIPELINE VULKAN &mdash; STATION 1:1 TEMPS RÉEL</h1>
    <span id="phasePill" class="pill">PRÊT</span>
  </div>
  <div class="badges">
    <span id="modelBadge" class="pill mono">Modèle : -</span>
    <span id="backendBadge" class="pill mono">Backend : ...</span>
  </div>
</header>

<main>
  <!-- LEFT SIDEBAR: CHANTIER CONTROLS & PARAMETERS -->
  <aside class="left">
    <div class="card">
      <h2>Chantier Aérien 1:1</h2>
      <div style="font-size:11px;color:var(--dim)">Répertoire du chantier (.CAM + Traj + GeoTIFF + IGN_DATA)</div>
      <input id="chDir" class="full mono" type="text" placeholder="E:\Chantiers\MEYREUIL" />
      <div class="btn-row">
        <button onclick="discoverChantier()">&#128269; Analyser</button>
        <button class="primary" onclick="startChantier()">&#9654; Lancer 1:1</button>
        <button class="danger" onclick="cancelChantier()">&#9632; Stop</button>
      </div>
      <label class="row" style="margin-top:8px">Paire stéréo
        <select id="chPair"><option value="0">Paire #0 (auto)</option></select>
      </label>
      <label class="chk"><input id="chAllPairs" type="checkbox" onchange="updateCliPreview()" /> Traiter toutes les paires (--all-pairs)</label>
      <label class="row">Taille dalle 1:1
        <select id="chRoi" onchange="updateCliPreview()">
          <option value="512">512 px</option>
          <option value="1024" selected>1024 px</option>
          <option value="1536">1536 px</option>
          <option value="2048">2048 px</option>
        </select>
      </label>
      <label class="row">Voxel fusion (m) <input id="chVoxel" type="number" min="0" max="5" step="0.01" value="0" onchange="updateCliPreview()" /></label>
      <label class="chk"><input id="chExpress" type="checkbox" checked onchange="updateCliPreview()" /> Mode Express (0 intermédiaire, PLY direct)</label>
      <label class="chk"><input id="chUseLidar" type="checkbox" checked onchange="updateCliPreview()" /> Guidage MNT/MNS LiDAR HD (IGN_DATA)</label>
      <label class="chk"><input id="chUseBa" type="checkbox" checked onchange="updateCliPreview()" /> Raffinement poses BA sur GCP pointés</label>
    </div>

    <div class="card">
      <h2>Paramètres Stéréo 1:1</h2>
      <label class="row">Algorithme
        <select id="algo" onchange="updateCliPreview()">
          <option value="asp_sgm" selected>ASP SGM (8 chemins)</option>
          <option value="asp_mgm">ASP MGM (Quadrant)</option>
          <option value="asp_bm">ASP Block Matching</option>
        </select>
      </label>
      <label class="row">Métrique coût
        <select id="cost" onchange="updateCliPreview()">
          <option value="2" selected>ZNCC (damped bruit)</option>
          <option value="3">Census 5x5</option>
          <option value="0">SAD</option>
          <option value="1">SSD</option>
        </select>
      </label>
      <label class="row">Préfiltre
        <select id="prefilter" onchange="updateCliPreview()">
          <option value="1" selected>Subtracted Mean</option>
          <option value="2">LoG (Laplacian)</option>
          <option value="0">Aucun</option>
        </select>
      </label>
      <label class="row">Sous-pixel
        <select id="subpixel" onchange="updateCliPreview()">
          <option value="1" selected>Parabole ZNCC pur</option>
          <option value="2">Bayes-EM Affine</option>
          <option value="3">Affine LK</option>
          <option value="0">Désactivé</option>
        </select>
      </label>
      <label class="row">Fenêtre (px) <input id="kernel" type="number" min="3" max="21" step="2" value="9" onchange="updateCliPreview()" /></label>
      <label class="row">SGM P1 / P2
        <span style="display:flex;gap:4px;width:148px">
          <input id="sgmP1" type="number" step="0.02" min="0.02" max="1.5" value="0.22" style="width:72px" />
          <input id="sgmP2" type="number" step="0.05" min="0.1" max="3.0" value="0.85" style="width:72px" />
        </span>
      </label>
      <label class="row">Backend
        <select id="backend" onchange="updateCliPreview()">
          <option value="auto" selected>Auto (Vulkan GPU)</option>
          <option value="vulkan">Vulkan forcé</option>
          <option value="cpu">CPU AVX2</option>
        </select>
      </label>
      <label class="row">Palette Z
        <select id="cmap" onchange="refreshVisuals(true)">
          <option value="turbo" selected>Turbo</option>
          <option value="terrain">Terrain</option>
          <option value="viridis">Viridis</option>
          <option value="inferno">Inferno</option>
          <option value="grayscale">Niveaux de gris</option>
        </select>
      </label>
    </div>

    <div class="card">
      <h2>Parité Commande CLI</h2>
      <div id="cliBox" class="cli-box mono">-</div>
      <div class="btn-row">
        <button onclick="copyCli()">Copier la commande CLI</button>
      </div>
    </div>

    <div class="card">
      <h2>Métadonnées Chantier</h2>
      <div class="kv"><span>Caméra</span><span id="infoCam" class="mono">-</span></div>
      <div class="kv"><span>CRS</span><span id="infoCrs" class="mono">-</span></div>
      <div class="kv"><span>LiDAR HD</span><span id="infoLidar" class="mono">-</span></div>
      <div class="kv"><span>Empreintes</span><span id="infoFp" class="mono">-</span></div>
      <div class="kv"><span>Raffinement BA</span><span id="infoBa" class="mono">-</span></div>
      <div class="kv"><span>Épipolaire</span><span id="infoEpi" class="mono">-</span></div>
      <div class="kv"><span>Sortie PLY</span><span id="infoPly" class="mono">-</span></div>
    </div>
  </aside>

  <!-- CENTER: LIVE 2D PROGRESSIVE MOSAIC & INTERACTIVE 3D WEBGL CLOUD -->
  <section class="center">
    <div class="tabs" id="viewTabs">
      <div class="tab active" data-view="rgb">Mosaïque 2D RGB</div>
      <div class="tab" data-view="elev">Altitude MNS (Relief)</div>
      <div class="tab" data-view="grid">Grille de Dalles</div>
      <div class="tab" data-view="overlay">RGB + Altitude</div>
      <div class="tab" data-view="cloud3d">Nuage 3D Temps Réel (WebGL)</div>
      <div class="tab-sep"></div>
      <div class="tab" data-view="wb_dem" id="tabWb">Banc Stéréo Synthétique</div>
    </div>

    <div class="viewport">
      <img id="mosaicImg" alt="Mosaïque 2D progressive" />
      <img id="wbImg" alt="Banc stéréo" style="display:none" />
      <canvas id="cloudCanvas"></canvas>

      <div class="hud3d" id="hud3d">
        <span>Couleur:
          <select id="colorMode3d" onchange="renderCloud3D()">
            <option value="rgb" selected>Vraies Couleurs RGB</option>
            <option value="turbo">Altitude (Turbo)</option>
            <option value="terrain">Altitude (Terrain)</option>
          </select>
        </span>
        <span>Points: <input id="ptSize3d" type="range" min="1" max="7" step="0.5" value="2.2" oninput="renderCloud3D()" /></span>
        <span>Échelle Z: <input id="zScale3d" type="range" min="0.5" max="3.0" step="0.1" value="1.0" oninput="renderCloud3D()" /></span>
        <button onclick="resetCamera3D()">Recentrer</button>
        <span id="hudPts3d" class="mono" style="color:var(--accent)">0 pts</span>
      </div>

      <div class="hud-hint mono" id="hudHint">
        Vue temps réel — la mosaïque se met à jour automatiquement à chaque dalle 1:1 terminée
      </div>
    </div>
  </section>

  <!-- RIGHT SIDEBAR: LIVE PROGRESS, STAGE TIMINGS & CONSOLE -->
  <aside class="right">
    <div class="card">
      <h2>Avancement en Direct <span id="pctLabel" class="mono" style="color:var(--accent)">0.0%</span></h2>
      <div class="prog-track"><div id="progFill" class="prog-fill"></div></div>
      <div id="statusMsg" style="font-size:11px;color:var(--dim);margin-bottom:8px">En attente...</div>
      <div class="kpi-grid">
        <div class="kpi"><div class="k">Dalles 1:1</div><div id="kTiles" class="v mono">0 / 0</div></div>
        <div class="kpi"><div class="k">Points 3D</div><div id="kPts" class="v green mono">0</div></div>
        <div class="kpi"><div class="k">Débit</div><div id="kSpeed" class="v accent mono">0.00 Mpx/s</div></div>
        <div class="kpi"><div class="k">Temps restant</div><div id="kEta" class="v mono">-</div></div>
        <div class="kpi"><div class="k">Densité moy.</div><div id="kValid" class="v mono">0.0%</div></div>
        <div class="kpi"><div class="k">Altitude Z (m)</div><div id="kZ" class="v mono">-</div></div>
      </div>
      <div class="kv" style="margin-top:8px"><span>Bande source</span><span id="kBand" class="mono">-</span></div>
      <div class="kv"><span>Recouvrement 1:1</span><span id="kOverlap" class="mono">-</span></div>
      <div class="kv"><span>Dispatches Vulkan</span><span id="kVk" class="mono">0</span></div>
    </div>

    <div class="card">
      <h2>Ventilation Temps par Étape</h2>
      <div id="timingsBox"></div>
    </div>

    <div class="card" style="flex:1;display:flex;flex-direction:column;min-height:200px">
      <h2>Journal Temps Réel</h2>
      <div id="logBox" class="log-box mono">Prêt.</div>
    </div>
  </aside>
</main>

<footer>
  <span id="footerLeft">0 crate externe &bull; Pure Rust HTTP/1.1 + SSE + LAS/LAZ + Vulkan Compute SPIR-V 1.0</span>
  <span id="footerRight" class="mono">SSE: connexion...</span>
</footer>

<script>
// ============================================================================
// State & Tabs
// ============================================================================
let activeView = 'rgb';
let lastMosaicRev = -1;
let lastCloudRev = -1;
let cloudLoadedCount = 0;
let lastStatus = null;
let fetchingCloud = false;

// 3D Cloud buffers (interleaved f32 xyz + u8 rgba stored in flat typed arrays)
let maxCap3d = 420000;
let ptsPos = new Float32Array(maxCap3d * 3);
let ptsRgb = new Uint8Array(maxCap3d * 3);
let ptsCount = 0;
let cloudCenter = [0, 0, 0];
let cloudExtent = 100.0;
let cloudZMin = 0.0, cloudZMax = 1.0;

// Camera 3D orbit state
let camYaw = 0.65, camPitch = 0.68, camDist = 1.65;
let panX = 0.0, panY = 0.0;
let dragging = false, panMode = false, lastMX = 0, lastMY = 0;

// WebGL context
let gl = null, glProg = null, glPosBuf = null, glColBuf = null, glDirty = false;

document.querySelectorAll('.tab').forEach(tab => {
  tab.onclick = () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    activeView = tab.dataset.view;
    updateViewVisibility();
    refreshVisuals(true);
  };
});

function updateViewVisibility() {
  const mImg = document.getElementById('mosaicImg');
  const wImg = document.getElementById('wbImg');
  const c3d = document.getElementById('cloudCanvas');
  const hud = document.getElementById('hud3d');
  const hint = document.getElementById('hudHint');

  mImg.style.display = 'none';
  wImg.style.display = 'none';
  c3d.style.display = 'none';
  hud.style.display = 'none';

  if (activeView === 'cloud3d') {
    c3d.style.display = 'block';
    hud.style.display = 'flex';
    hint.textContent = 'Clic gauche: Orbite 3D | Shift+Clic ou Clic droit: Panoramique | Molette: Zoom — mise à jour directe dalle par dalle';
    initWebGLIfNeeded();
    fetchCloudDelta();
    renderCloud3D();
  } else if (activeView === 'wb_dem') {
    wImg.style.display = 'block';
    hint.textContent = 'Banc Stéréo synthétique — vérifiez le backend Vulkan/CPU rapidement';
  } else {
    mImg.style.display = 'block';
    hint.textContent = 'Mosaïque 2D progressive — cadre cyan = dalle 1:1 en cours de calcul';
  }
}

// ============================================================================
// SSE Real-Time Stream + Status Sync
// ============================================================================
function connectLiveSSE() {
  const es = new EventSource('/api/live/events');
  es.onopen = () => {
    document.getElementById('footerRight').textContent = 'SSE: connecté (temps réel)';
  };
  es.onmessage = async (ev) => {
    try {
      const m = JSON.parse(ev.data);
      applyQuickFrame(m);
      await fetchFullLiveStatus();
      if (m.mosaic_rev !== lastMosaicRev) {
        lastMosaicRev = m.mosaic_rev;
        if (activeView === 'rgb' || activeView === 'elev' || activeView === 'grid' || activeView === 'overlay') {
          refreshMosaicImage();
        }
      }
      if (m.cloud_rev !== lastCloudRev) {
        lastCloudRev = m.cloud_rev;
        if (m.cloud_count < cloudLoadedCount) {
          resetCloudBuffer();
        }
        if (activeView === 'cloud3d') {
          fetchCloudDelta();
        }
      }
    } catch (_) {}
  };
  es.onerror = () => {
    document.getElementById('footerRight').textContent = 'SSE: reconnexion...';
    es.close();
    setTimeout(connectLiveSSE, 1500);
  };
}

function applyQuickFrame(m) {
  const pct = (m.pct * 100).toFixed(1) + '%';
  document.getElementById('pctLabel').textContent = pct;
  document.getElementById('progFill').style.width = pct;
  document.getElementById('statusMsg').textContent = m.status_msg || '';
  document.getElementById('kTiles').textContent = `${m.tiles_done} / ${m.total_tiles}`;
  document.getElementById('kPts').textContent = fmtPoints(m.total_points);
  document.getElementById('kSpeed').textContent = `${m.mpx_per_sec.toFixed(2)} Mpx/s`;
  document.getElementById('kEta').textContent = m.eta_sec > 0.5 ? fmtTime(m.eta_sec) : (m.phase === 'done' ? 'Terminé' : '-');
  setPhasePill(m.phase);
}

function setPhasePill(phase) {
  const pill = document.getElementById('phasePill');
  const labels = {
    idle: 'PRÊT',
    discovering: 'ANALYSE LIDAR / CHANTIER...',
    running: 'CALCUL 1:1 EN COURS',
    done: 'TERMINÉ',
    cancelled: 'ARRÊTÉ',
    error: 'ERREUR'
  };
  pill.textContent = labels[phase] || phase.toUpperCase();
  pill.className = 'pill ' + (phase || '');
}

async function fetchFullLiveStatus() {
  try {
    const r = await fetch('/api/live/status');
    const d = await r.json();
    lastStatus = d;
    applyFullStatus(d);
  } catch (_) {}
}

function applyFullStatus(d) {
  setPhasePill(d.phase);
  const pct = (d.pct * 100).toFixed(1) + '%';
  document.getElementById('pctLabel').textContent = pct;
  document.getElementById('progFill').style.width = pct;
  document.getElementById('statusMsg').textContent = d.status_msg || '-';

  document.getElementById('modelBadge').textContent = d.model_name && d.model_name !== '-'
    ? `Modèle ${d.model_index}/${d.total_models} : ${d.model_name}`
    : 'Modèle : -';
  if (d.backend && d.backend !== '-') {
    document.getElementById('backendBadge').textContent = d.backend;
  }

  document.getElementById('kTiles').textContent = `${d.tiles_done} / ${d.total_tiles}`;
  document.getElementById('kPts').textContent = fmtPoints(d.total_points);
  document.getElementById('kSpeed').textContent = `${d.mpx_per_sec.toFixed(2)} Mpx/s`;
  document.getElementById('kEta').textContent = d.eta_sec > 0.5 ? fmtTime(d.eta_sec) : (d.phase === 'done' ? fmtTime(d.elapsed_sec) : '-');
  document.getElementById('kValid').textContent = `${d.avg_valid_pct.toFixed(1)}%`;
  document.getElementById('kZ').textContent = d.z_max > d.z_min ? `${d.z_min.toFixed(1)}..${d.z_max.toFixed(1)}` : '-';
  document.getElementById('kBand').textContent = d.total_bands > 0 ? `${d.band_idx} / ${d.total_bands}` : '-';
  document.getElementById('kOverlap').textContent = d.overlap_w > 0
    ? `${d.overlap_w}x${d.overlap_h} px` + (d.area_ha > 0 ? ` (${d.area_ha.toFixed(1)} ha)` : '')
    : '-';
  document.getElementById('kVk').textContent = d.vk_dispatches.toLocaleString('fr-FR');

  if (d.chantier_dir && !document.getElementById('chDir').value) {
    document.getElementById('chDir').value = d.chantier_dir;
  }
  document.getElementById('infoCam').textContent = d.camera || '-';
  document.getElementById('infoCrs').textContent = d.crs || '-';
  document.getElementById('infoLidar').textContent = d.lidar || '-';
  document.getElementById('infoFp').textContent = d.footprints || '-';
  document.getElementById('infoBa').textContent = d.ba || '-';
  document.getElementById('infoEpi').textContent = d.epipolar || '-';
  document.getElementById('infoPly').textContent = d.output_ply || '-';

  if (d.cli_command) {
    document.getElementById('cliBox').textContent = d.cli_command;
  } else {
    updateCliPreview();
  }

  // Populate stereo pair dropdown when pairs arrive
  if (d.pairs && d.pairs.length > 0) {
    const sel = document.getElementById('chPair');
    if (sel.options.length !== d.pairs.length) {
      const prev = sel.value;
      sel.innerHTML = d.pairs.map(p =>
        `<option value="${p.index}">#${p.index} ${p.name} (B=${p.baseline}m)</option>`
      ).join('');
      if (prev && Number(prev) < d.pairs.length) sel.value = prev;
    }
  }

  // Stage timings
  const t = d.timings;
  const rows = [
    ['GeoTIFF', t.band_io, '#a78bfa'],
    ['Rectif', t.rect, '#38bdf8'],
    ['PPRC', t.pprc, '#60a5fa'],
    ['Corr/SGM', t.corr, '#f59e0b'],
    ['Subpx', t.rfne, '#fb923c'],
    ['Filtre', t.fltr, '#34d399'],
    ['Tri 3D', t.tri, '#22d3ee'],
    ['PLY', t.ply, '#94a3b8']
  ];
  const maxT = Math.max(...rows.map(r => r[1]), 1.0);
  document.getElementById('timingsBox').innerHTML = rows.map(([name, ms, col]) => {
    const w = Math.min(100, (ms / maxT) * 100);
    const lbl = ms >= 1000 ? (ms / 1000).toFixed(1) + 's' : ms.toFixed(0) + 'ms';
    return `<div class="bar-row"><span class="bar-label">${name}</span><div class="bar-track"><div class="bar-fill" style="width:${w}%;background:${col}"></div></div><span class="bar-val mono">${lbl}</span></div>`;
  }).join('');

  // Live logs
  if (d.logs && d.logs.length > 0) {
    const lb = document.getElementById('logBox');
    const atBottom = lb.scrollHeight - lb.scrollTop - lb.clientHeight < 30;
    lb.textContent = d.logs.join('\n');
    if (atBottom) lb.scrollTop = lb.scrollHeight;
  }

  // Sync cloud/mosaic if needed on first load
  if (d.mosaic_rev !== lastMosaicRev) {
    lastMosaicRev = d.mosaic_rev;
    if (activeView !== 'cloud3d' && activeView !== 'wb_dem') refreshMosaicImage();
  }
  if (d.cloud_count > cloudLoadedCount && activeView === 'cloud3d') {
    fetchCloudDelta();
  }
}

function refreshMosaicImage() {
  const cmap = document.getElementById('cmap').value;
  document.getElementById('mosaicImg').src = `/api/live/mosaic.png?layer=${activeView}&cmap=${cmap}&r=${Date.now()}`;
}

function refreshVisuals(force) {
  if (activeView === 'cloud3d') {
    fetchCloudDelta();
    renderCloud3D();
  } else if (activeView === 'wb_dem') {
    const cmap = document.getElementById('cmap').value;
    document.getElementById('wbImg').src = `/api/image?layer=dem&cmap=${cmap}&t=${Date.now()}`;
  } else if (force) {
    refreshMosaicImage();
  }
}

// ============================================================================
// Chantier Actions (100% CLI Parity)
// ============================================================================
function gatherPayload() {
  return {
    dir: document.getElementById('chDir').value.trim(),
    pair_idx: parseInt(document.getElementById('chPair').value, 10) || 0,
    all_pairs: document.getElementById('chAllPairs').checked,
    roi_size: parseInt(document.getElementById('chRoi').value, 10) || 1024,
    merge_voxel: parseFloat(document.getElementById('chVoxel').value) || 0.0,
    express: document.getElementById('chExpress').checked,
    no_ign_lidar: !document.getElementById('chUseLidar').checked,
    no_gcp_refine: !document.getElementById('chUseBa').checked,
    algo: document.getElementById('algo').value,
    cost: parseInt(document.getElementById('cost').value, 10),
    prefilter: parseInt(document.getElementById('prefilter').value, 10),
    subpixel: parseInt(document.getElementById('subpixel').value, 10),
    kernel: parseInt(document.getElementById('kernel').value, 10) || 9,
    sgm_p1: parseFloat(document.getElementById('sgmP1').value) || 0.22,
    sgm_p2: parseFloat(document.getElementById('sgmP2').value) || 0.85,
    backend: document.getElementById('backend').value
  };
}

function updateCliPreview() {
  const p = gatherPayload();
  const dir = p.dir || 'E:\\Chantiers\\MEYREUIL';
  let cmd = `stereo-pipeline chantier "${dir}" --full-model`;
  cmd += p.all_pairs ? ' --all-pairs' : ` --pair ${p.pair_idx}`;
  if (p.express) cmd += ' --express';
  cmd += ` --roi-size ${p.roi_size}`;
  if (p.merge_voxel > 0) cmd += ` --merge-voxel ${p.merge_voxel.toFixed(3)}`;
  if (p.no_ign_lidar) cmd += ' --no-ign-lidar';
  if (p.no_gcp_refine) cmd += ' --no-gcp-refine';
  cmd += ` --stereo-algorithm ${p.algo} --cost-mode ${p.cost} --prefilter-mode ${p.prefilter} --subpixel-mode ${p.subpixel} --corr-kernel ${p.kernel} ${p.kernel} --backend ${p.backend} --live-ui`;
  document.getElementById('cliBox').textContent = cmd;
}

async function discoverChantier() {
  const payload = gatherPayload();
  if (!payload.dir) { alert('Indiquez le dossier du chantier (ex: E:\\Chantiers\\MEYREUIL)'); return; }
  const r = await fetch('/api/chantier/discover', {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload)
  });
  const d = await r.json();
  if (!d.ok && d.error) alert(d.error);
}

async function startChantier() {
  const payload = gatherPayload();
  if (!payload.dir) { alert('Indiquez le dossier du chantier (ex: E:\\Chantiers\\MEYREUIL)'); return; }
  resetCloudBuffer();
  const r = await fetch('/api/chantier/start', {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload)
  });
  const d = await r.json();
  if (!d.ok && d.error) alert(d.error);
}

async function cancelChantier() {
  await fetch('/api/chantier/cancel', { method: 'POST' });
}

function copyCli() {
  navigator.clipboard?.writeText(document.getElementById('cliBox').textContent);
}

// ============================================================================
// Incremental Binary 3D Point Cloud Streaming + WebGL Renderer
// ============================================================================
function resetCloudBuffer() {
  ptsCount = 0;
  cloudLoadedCount = 0;
  glDirty = true;
}

async function fetchCloudDelta() {
  if (fetchingCloud) return;
  fetchingCloud = true;
  try {
    const r = await fetch('/api/live/cloud.bin?from=' + cloudLoadedCount);
    const buf = await r.arrayBuffer();
    if (buf.byteLength < 36) return;
    const dv = new DataView(buf);
    // Magic "PC3D"
    if (dv.getUint8(0) !== 80 || dv.getUint8(1) !== 67) return;
    const total = dv.getUint32(4, true);
    const count = dv.getUint32(8, true);
    cloudZMin = dv.getFloat32(12, true);
    cloudZMax = dv.getFloat32(16, true);
    cloudExtent = Math.max(dv.getFloat32(20, true), 1.0);
    cloudCenter = [dv.getFloat32(24, true), dv.getFloat32(28, true), dv.getFloat32(32, true)];

    if (total < cloudLoadedCount) {
      resetCloudBuffer();
      fetchingCloud = false;
      return fetchCloudDelta();
    }
    if (count > 0) {
      let off = 36;
      for (let i = 0; i < count && ptsCount < maxCap3d; i++) {
        const i3 = ptsCount * 3;
        ptsPos[i3]     = dv.getFloat32(off, true);
        ptsPos[i3 + 1] = dv.getFloat32(off + 4, true);
        ptsPos[i3 + 2] = dv.getFloat32(off + 8, true);
        ptsRgb[i3]     = dv.getUint8(off + 12);
        ptsRgb[i3 + 1] = dv.getUint8(off + 13);
        ptsRgb[i3 + 2] = dv.getUint8(off + 14);
        ptsCount++;
        off += 16;
      }
      cloudLoadedCount = total;
      glDirty = true;
      document.getElementById('hudPts3d').textContent =
        `${ptsCount.toLocaleString('fr-FR')} pts aperçu 3D`;
      if (activeView === 'cloud3d') renderCloud3D();
    }
  } finally {
    fetchingCloud = false;
  }
}

function initWebGLIfNeeded() {
  if (gl) return;
  const canvas = document.getElementById('cloudCanvas');
  gl = canvas.getContext('webgl', { antialias: true, preserveDrawingBuffer: false });
  if (!gl) return;

  const vsSrc = `
    attribute vec3 aPos;
    attribute vec3 aCol;
    uniform mat4 uMvp;
    uniform float uPtSize;
    uniform float uColorMode;
    uniform float uZMin;
    uniform float uZMax;
    varying vec3 vCol;
    vec3 turbo(float t) {
      t = clamp(t, 0.0, 1.0);
      vec3 c1 = vec3(0.19, 0.07, 0.23);
      vec3 c2 = vec3(0.16, 0.56, 0.94);
      vec3 c3 = vec3(0.28, 0.93, 0.44);
      vec3 c4 = vec3(0.98, 0.72, 0.16);
      vec3 c5 = vec3(0.65, 0.07, 0.05);
      if (t < 0.25) return mix(c1, c2, t * 4.0);
      if (t < 0.50) return mix(c2, c3, (t - 0.25) * 4.0);
      if (t < 0.75) return mix(c3, c4, (t - 0.50) * 4.0);
      return mix(c4, c5, (t - 0.75) * 4.0);
    }
    vec3 terrain(float t) {
      t = clamp(t, 0.0, 1.0);
      if (t < 0.35) return mix(vec3(0.15, 0.42, 0.22), vec3(0.48, 0.65, 0.28), t / 0.35);
      if (t < 0.70) return mix(vec3(0.48, 0.65, 0.28), vec3(0.62, 0.48, 0.32), (t - 0.35) / 0.35);
      return mix(vec3(0.62, 0.48, 0.32), vec3(0.95, 0.96, 0.98), (t - 0.70) / 0.30);
    }
    void main() {
      gl_Position = uMvp * vec4(aPos, 1.0);
      gl_PointSize = uPtSize;
      float zn = (aPos.z - uZMin) / max(uZMax - uZMin, 0.01);
      if (uColorMode < 0.5) vCol = aCol;
      else if (uColorMode < 1.5) vCol = turbo(zn);
      else vCol = terrain(zn);
    }
  `;
  const fsSrc = `
    precision mediump float;
    varying vec3 vCol;
    void main() { gl_FragColor = vec4(vCol, 1.0); }
  `;
  const compile = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    return s;
  };
  glProg = gl.createProgram();
  gl.attachShader(glProg, compile(gl.VERTEX_SHADER, vsSrc));
  gl.attachShader(glProg, compile(gl.FRAGMENT_SHADER, fsSrc));
  gl.linkProgram(glProg);
  glPosBuf = gl.createBuffer();
  glColBuf = gl.createBuffer();
  gl.enable(gl.DEPTH_TEST);
}

function makeMvpMatrix(aspect, zScale) {
  const s = 1.6 / Math.max(cloudExtent, 1.0);
  const cx = cloudCenter[0], cy = cloudCenter[1], cz = cloudCenter[2];
  const cyaw = Math.cos(camYaw), syaw = Math.sin(camYaw);
  const cp = Math.cos(camPitch), sp = Math.sin(camPitch);

  // Perspective projection (fov ~ 45 deg)
  const f = 1.0 / Math.tan(Math.PI / 8.0);
  const zn = 0.02, zf = 40.0;
  const nf = 1.0 / (zn - zf);

  // Compose: P * T_cam * Rx(pitch) * Rz(yaw) * S(s, s, s*zScale) * T(-center)
  // Compute column-major 4x4 directly
  const m = new Float32Array(16);
  const rx0 = cyaw * s,        ry0 = -syaw * s,        rz0 = 0;
  const rx1 = syaw * cp * s,   ry1 = cyaw * cp * s,    rz1 = sp * s * zScale;
  const rx2 = -syaw * sp * s,  ry2 = -cyaw * sp * s,   rz2 = cp * s * zScale;

  const tx = -(rx0 * cx + ry0 * cy + rz0 * cz) + panX;
  const ty = -(rx1 * cx + ry1 * cy + rz1 * cz) + panY;
  const tz = -(rx2 * cx + ry2 * cy + rz2 * cz) - camDist;

  // Apply perspective
  const px = f / aspect, py = f;
  const A = (zf + zn) * nf, B = (2.0 * zf * zn) * nf;

  // Note: swap y/z camera axes cleanly -> view coords (vx, vy, vz) = (x_cam, y_cam, z_cam)
  // Actually use right-handed view: X right = rx0, Y up = rx1, Z forward = rx2
  m[0] = px * rx0;  m[4] = px * ry0;  m[8]  = px * rz0;  m[12] = px * tx;
  m[1] = py * rx1;  m[5] = py * ry1;  m[9]  = py * rz1;  m[13] = py * ty;
  m[2] = A  * rx2;  m[6] = A  * ry2;  m[10] = A  * rz2;  m[14] = A  * tz + B;
  m[3] = -rx2;      m[7] = -ry2;      m[11] = -rz2;      m[15] = -tz;
  return m;
}

function renderCloud3D() {
  const canvas = document.getElementById('cloudCanvas');
  if (canvas.clientWidth === 0 || canvas.clientHeight === 0) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.floor(canvas.clientWidth * dpr);
  const h = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }

  if (!gl) {
    renderCloud2DFallback(canvas);
    return;
  }

  gl.viewport(0, 0, w, h);
  gl.clearColor(0.025, 0.035, 0.06, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  if (ptsCount === 0) return;

  gl.useProgram(glProg);
  const aPos = gl.getAttribLocation(glProg, 'aPos');
  const aCol = gl.getAttribLocation(glProg, 'aCol');

  if (glDirty) {
    gl.bindBuffer(gl.ARRAY_BUFFER, glPosBuf);
    gl.bufferData(gl.ARRAY_BUFFER, ptsPos.subarray(0, ptsCount * 3), gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, glColBuf);
    gl.bufferData(gl.ARRAY_BUFFER, ptsRgb.subarray(0, ptsCount * 3), gl.DYNAMIC_DRAW);
    glDirty = false;
  }

  gl.bindBuffer(gl.ARRAY_BUFFER, glPosBuf);
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, glColBuf);
  gl.enableVertexAttribArray(aCol);
  gl.vertexAttribPointer(aCol, 3, gl.UNSIGNED_BYTE, true, 0, 0);

  const zScale = parseFloat(document.getElementById('zScale3d').value) || 1.0;
  const ptSize = (parseFloat(document.getElementById('ptSize3d').value) || 2.2) * dpr;
  const modeStr = document.getElementById('colorMode3d').value;
  const modeNum = modeStr === 'rgb' ? 0.0 : (modeStr === 'turbo' ? 1.0 : 2.0);

  const mvp = makeMvpMatrix(w / Math.max(h, 1), zScale);
  gl.uniformMatrix4fv(gl.getUniformLocation(glProg, 'uMvp'), false, mvp);
  gl.uniform1f(gl.getUniformLocation(glProg, 'uPtSize'), ptSize);
  gl.uniform1f(gl.getUniformLocation(glProg, 'uColorMode'), modeNum);
  gl.uniform1f(gl.getUniformLocation(glProg, 'uZMin'), cloudZMin);
  gl.uniform1f(gl.getUniformLocation(glProg, 'uZMax'), cloudZMax);

  gl.drawArrays(gl.POINTS, 0, ptsCount);
}

function renderCloud2DFallback(canvas) {
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#06090f';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (ptsCount === 0) return;
  const step = Math.max(1, Math.floor(ptsCount / 60000));
  const mvp = makeMvpMatrix(canvas.width / Math.max(canvas.height, 1), parseFloat(document.getElementById('zScale3d').value) || 1.0);
  const hw = canvas.width * 0.5, hh = canvas.height * 0.5;
  for (let i = 0; i < ptsCount; i += step) {
    const i3 = i * 3;
    const x = ptsPos[i3], y = ptsPos[i3 + 1], z = ptsPos[i3 + 2];
    const cw = mvp[3]*x + mvp[7]*y + mvp[11]*z + mvp[15];
    if (cw <= 0.05) continue;
    const sx = ((mvp[0]*x + mvp[4]*y + mvp[8]*z + mvp[12]) / cw) * hw + hw;
    const sy = hh - ((mvp[1]*x + mvp[5]*y + mvp[9]*z + mvp[13]) / cw) * hh;
    ctx.fillStyle = `rgb(${ptsRgb[i3]},${ptsRgb[i3+1]},${ptsRgb[i3+2]})`;
    ctx.fillRect(sx, sy, 2, 2);
  }
}

function resetCamera3D() {
  camYaw = 0.65; camPitch = 0.68; camDist = 1.65; panX = 0; panY = 0;
  renderCloud3D();
}

const c3d = document.getElementById('cloudCanvas');
c3d.oncontextmenu = e => e.preventDefault();
c3d.onmousedown = e => {
  dragging = true;
  panMode = e.button === 2 || e.shiftKey;
  lastMX = e.clientX; lastMY = e.clientY;
};
window.onmouseup = () => { dragging = false; };
window.onmousemove = e => {
  if (!dragging || activeView !== 'cloud3d') return;
  const dx = e.clientX - lastMX, dy = e.clientY - lastMY;
  lastMX = e.clientX; lastMY = e.clientY;
  if (panMode) {
    panX += dx * 0.0025 * camDist;
    panY -= dy * 0.0025 * camDist;
  } else {
    camYaw += dx * 0.008;
    camPitch = Math.max(-1.5, Math.min(1.5, camPitch + dy * 0.008));
  }
  renderCloud3D();
};
c3d.onwheel = e => {
  e.preventDefault();
  camDist = Math.max(0.18, Math.min(8.0, camDist * (e.deltaY < 0 ? 0.88 : 1.13)));
  renderCloud3D();
};
window.onresize = () => { if (activeView === 'cloud3d') renderCloud3D(); };

// ============================================================================
// Formatting Helpers & Boot
// ============================================================================
function fmtPoints(n) {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + ' M';
  if (n >= 1_000) return (n / 1_000).toFixed(1) + ' k';
  return String(n);
}
function fmtTime(sec) {
  const s = Math.max(0, Math.round(sec));
  const m = Math.floor(s / 60), r = s % 60;
  return m > 0 ? `${m}m ${r}s` : `${r}s`;
}

// Check if Workbench is available (hidden in CLI --live-ui attached mode)
fetch('/api/status').then(r => r.json()).then(d => {
  if (!d.workbench_available) {
    document.getElementById('tabWb').style.display = 'none';
  }
}).catch(() => {});

updateViewVisibility();
updateCliPreview();
fetchFullLiveStatus();
connectLiveSSE();
</script>
</body>
</html>
"##;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::pinhole::PinholeCamera;
    use crate::core::chantier::{FullModelPlan, ModelTileSpec};
    use crate::image::{ImageRgb, PointCloud};
    use crate::math::{Mat3, Vec3};

    fn http_get(port: u16, path: &str) -> Vec<u8> {
        let mut s = TcpStream::connect(format!("127.0.0.1:{}", port)).unwrap();
        let req = format!("GET {} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n", path);
        s.write_all(req.as_bytes()).unwrap();
        let mut resp = Vec::new();
        s.read_to_end(&mut resp).unwrap();
        resp
    }

    #[test]
    fn test_live_monitor_http_2d_mosaic_and_3d_binary_cloud() {
        let mon = LiveChantierMonitor::new();
        let cam = PinholeCamera::new(
            1000.0,
            1000.0,
            128.0,
            128.0,
            Vec3::new(0.0, 0.0, 1000.0),
            Mat3::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        );
        let tile = ModelTileSpec {
            tx: 0,
            ty: 0,
            px0: 0,
            py0: 0,
            tw: 64,
            th: 64,
            emit_x0: 0,
            emit_y0: 0,
            emit_x1: 64,
            emit_y1: 64,
        };
        let plan = FullModelPlan {
            full_cam_l: cam.clone(),
            full_cam_r: cam.clone(),
            rect_cam_l: cam.clone(),
            rect_cam_r: cam,
            full_dims_l: (256, 256),
            full_dims_r: (256, 256),
            overlap_w: 128,
            overlap_h: 128,
            ground_z_ref: 240.0,
            half_span: 32,
            tiles_x: 2,
            tiles_y: 2,
            bands: vec![vec![tile.clone()]],
            total_tiles: 4,
            validated_overlap_area_m2: Some(12_345.0),
        };
        mon.begin_model(1, 1, "L0_TEST_PAIR", &plan, "dy=0.05 px".into(), "out.ply".into());

        let left_img = ImageF32::new(64, 64);
        let mut left_rgb = ImageRgb::new(64, 64);
        let mut pc = PointCloud::new(64, 64);
        let mut colors = vec![[0u8; 3]; 64 * 64];
        for y in 0..64u32 {
            for x in 0..64u32 {
                let i = (y * 64 + x) as usize;
                left_rgb.set(x, y, [80, 140, 210]);
                pc.points[i] = Vec3::new(1_900_000.0 + x as f64, 3_200_000.0 + y as f64, 240.0 + 0.1 * x as f64);
                pc.valid[i] = 1;
                colors[i] = [80, 140, 210];
            }
        }
        pc.colors = Some(colors);
        mon.on_tile_start(0, 0);
        mon.on_tile_complete(
            &tile,
            &left_img,
            Some(&left_rgb),
            &pc,
            4096,
            100.0,
            4,
            5.0,
            10.0,
            80.0,
            12.0,
            25.0,
            6.0,
            4.0,
            12,
            0.14,
        );

        let port = spawn_live_monitor_server(Arc::clone(&mon), 0, false).unwrap();
        assert!(port > 0);

        // 1. HTML landing page
        let html = http_get(port, "/");
        assert!(html.starts_with(b"HTTP/1.1 200 OK"));
        assert!(String::from_utf8_lossy(&html).contains("STATION 1:1 TEMPS"));

        // 2. JSON live status
        let status = http_get(port, "/api/live/status");
        let body_start = status.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
        let parsed = parse_json(std::str::from_utf8(&status[body_start..]).unwrap()).unwrap();
        assert_eq!(parsed.get("tiles_done").and_then(|v| v.as_u64()), Some(1));
        assert_eq!(parsed.get("total_points").and_then(|v| v.as_u64()), Some(4096));

        // 3. All 4 progressive 2D PNG layers
        for layer in ["rgb", "elev", "grid", "overlay"] {
            let png_resp = http_get(port, &format!("/api/live/mosaic.png?layer={}&cmap=turbo", layer));
            let bs = png_resp.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
            assert_eq!(&png_resp[bs..bs + 4], b"\x89PNG", "layer {} invalid PNG", layer);
        }

        // 4. Incremental binary 3D cloud chunk
        let bin_resp = http_get(port, "/api/live/cloud.bin?from=0");
        let bs = bin_resp.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
        let payload = &bin_resp[bs..];
        assert_eq!(&payload[0..4], b"PC3D");
        let total = u32::from_le_bytes(payload[4..8].try_into().unwrap()) as usize;
        let count = u32::from_le_bytes(payload[8..12].try_into().unwrap()) as usize;
        assert!(total > 0 && total == count);
        assert_eq!(payload.len(), 36 + count * 16);

        // Second incremental request from `total` must return 0 new points!
        let bin_delta = http_get(port, &format!("/api/live/cloud.bin?from={}", total));
        let bs2 = bin_delta.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
        let payload2 = &bin_delta[bs2..];
        assert_eq!(u32::from_le_bytes(payload2[8..12].try_into().unwrap()), 0);
    }
}

