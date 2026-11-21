//! Zero-dependency embedded HTTP/1.1 Web UI server (`std::net::TcpListener`).
//! Serves a rich interactive HTML5 Canvas / 3D Terrain Stereo Photogrammetry GUI + PNG/JSON API.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::camera::AnyCamera;
use crate::config::{
    ComputeBackendPreference, CostMode, PrefilterMode, StereoAlgorithm, StereoSettings, SubpixelMode,
};
use crate::image::io::encode_png_rgba;
use crate::image::{Colormap, ImageF32};
use crate::pipeline::{synthetic, PipelineResult, StereoPipelineEngine};
use crate::vulkan::shaders::shader_catalog;

struct AppState {
    left: ImageF32,
    right: ImageF32,
    cam_left: AnyCamera,
    cam_right: AnyCamera,
    settings: StereoSettings,
    result: PipelineResult,
    engine: StereoPipelineEngine,
}

/// Start the embedded zero-dependency Web UI HTTP server on `127.0.0.1:<port>`.
pub fn run_web_ui(
    left: ImageF32,
    right: ImageF32,
    cam_left: AnyCamera,
    cam_right: AnyCamera,
    settings: StereoSettings,
    port: u16,
) -> Result<(), String> {
    let engine = StereoPipelineEngine::new(settings.backend);
    println!("[Web UI] Initializing stereo pipeline ({})...", engine.vulkan_status);
    let result = engine.run_full(&left, &right, &cam_left, &cam_right, &settings, |stage, pct| {
        println!("[Web UI] {:3.0}% - {}", pct * 100.0, stage);
    });

    let state = Arc::new(Mutex::new(AppState {
        left,
        right,
        cam_left,
        cam_right,
        settings,
        result,
        engine,
    }));

    let addr = format!("0.0.0.0:{}", port);
    let listener = TcpListener::bind(&addr).map_err(|e| format!("Failed to bind {}: {}", addr, e))?;
    println!("==================================================================");
    println!(" RustStereoPipelineVulkan Web UI listening on http://localhost:{}", port);
    println!(" Press Ctrl+C to stop.");
    println!("==================================================================");

    for stream in listener.incoming().flatten() {
        let st = Arc::clone(&state);
        std::thread::spawn(move || {
            let _ = handle_client(stream, st);
        });
    }
    Ok(())
}

fn handle_client(mut stream: TcpStream, state: Arc<Mutex<AppState>>) -> std::io::Result<()> {
    let mut buf = [0u8; 8192];
    let n = stream.read(&mut buf)?;
    if n == 0 {
        return Ok(());
    }
    let req = String::from_utf8_lossy(&buf[..n]);
    let first_line = req.lines().next().unwrap_or("");
    let mut parts = first_line.split_whitespace();
    let method = parts.next().unwrap_or("GET");
    let full_path = parts.next().unwrap_or("/");
    let (path, query) = full_path.split_once('?').unwrap_or((full_path, ""));

    match (method, path) {
        ("GET", "/") | ("GET", "/index.html") => {
            send_response(&mut stream, "200 OK", "text/html; charset=utf-8", INDEX_HTML.as_bytes())
        }
        ("GET", "/api/status") => {
            let st = state.lock().unwrap();
            let json = build_status_json(&st);
            send_response(&mut stream, "200 OK", "application/json", json.as_bytes())
        }
        ("GET", "/api/image") => {
            let layer = query_param(query, "layer").unwrap_or("dem");
            let cmap = Colormap::from_name(query_param(query, "cmap").unwrap_or("turbo"));
            let st = state.lock().unwrap();
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
            let st = state.lock().unwrap();
            let json = build_dem3d_json(&st.result.dem.elevation);
            send_response(&mut stream, "200 OK", "application/json", json.as_bytes())
        }
        ("POST", "/api/run") | ("GET", "/api/run") => {
            let mut st = state.lock().unwrap();
            if let Some(a) = query_param(query, "algo") {
                st.settings.stereo_algorithm = StereoAlgorithm::from_str(a);
            }
            if let Some(c) = query_param(query, "cost").and_then(|v| v.parse().ok()) {
                st.settings.cost_mode = CostMode::from_int(c);
            }
            if let Some(p) = query_param(query, "prefilter").and_then(|v| v.parse().ok()) {
                st.settings.prefilter_mode = PrefilterMode::from_int(p);
            }
            if let Some(sp) = query_param(query, "subpixel").and_then(|v| v.parse().ok()) {
                st.settings.subpixel_mode = SubpixelMode::from_int(sp);
            }
            if let Some(k) = query_param(query, "kernel").and_then(|v| v.parse::<u32>().ok()) {
                let kc = k.clamp(3, 21);
                st.settings.corr_kernel_x = kc;
                st.settings.corr_kernel_y = kc;
            }
            if let Some(b) = query_param(query, "backend") {
                st.settings.backend = ComputeBackendPreference::from_str(b);
            }
            let left = st.left.clone();
            let right = st.right.clone();
            let cl = st.cam_left.clone();
            let cr = st.cam_right.clone();
            let cfg = st.settings.clone();
            st.result = st.engine.run_full(&left, &right, &cl, &cr, &cfg, |_, _| {});
            let json = build_status_json(&st);
            send_response(&mut stream, "200 OK", "application/json", json.as_bytes())
        }
        ("POST", "/api/synthetic") | ("GET", "/api/synthetic") => {
            let w = query_param(query, "w").and_then(|v| v.parse::<u32>().ok()).unwrap_or(160).clamp(64, 384);
            let h = query_param(query, "h").and_then(|v| v.parse::<u32>().ok()).unwrap_or(120).clamp(48, 288);
            let mut st = state.lock().unwrap();
            let ds = synthetic::generate_synthetic_stereo(w, h);
            st.left = ds.left_image;
            st.right = ds.right_image;
            st.cam_left = ds.left_cam;
            st.cam_right = ds.right_cam;
            let left = st.left.clone();
            let right = st.right.clone();
            let cl = st.cam_left.clone();
            let cr = st.cam_right.clone();
            let cfg = st.settings.clone();
            st.result = st.engine.run_full(&left, &right, &cl, &cr, &cfg, |_, _| {});
            let json = build_status_json(&st);
            send_response(&mut stream, "200 OK", "application/json", json.as_bytes())
        }
        ("POST", "/api/export") | ("GET", "/api/export") => {
            let st = state.lock().unwrap();
            let msg = match StereoPipelineEngine::save_outputs(Path::new("output/webui_run"), &st.result) {
                Ok(_) => r#"{"ok":true,"message":"Exported output/webui_run-{L,R,D,RD,F,DEM,hillshade}.tif & -PC.ply"}"#.to_string(),
                Err(e) => format!(r#"{{"ok":false,"message":"Export error: {}"}}"#, escape_json(&e.to_string())),
            };
            send_response(&mut stream, "200 OK", "application/json", msg.as_bytes())
        }
        _ => send_response(&mut stream, "404 Not Found", "text/plain", b"Not Found"),
    }
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

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn build_status_json(st: &AppState) -> String {
    let t = &st.result.timings;
    let (dem_min, dem_max, dem_mean, _) = st.result.dem.elevation.stats();
    let shaders_count = shader_catalog().len();
    format!(
        r#"{{"width":{},"height":{},"vulkan_available":{},"vulkan_status":"{}","active_backend":"{}","vk_dispatches":{},"spirv_shaders":{},"algo":"{}","cost":{},"prefilter":{},"subpixel":{},"kernel":{},"valid_pct":{:.2},"points":{},"tri_error":{:.4},"dem_min":{:.2},"dem_max":{:.2},"dem_mean":{:.2},"timings":{{"pprc":{:.2},"corr":{:.2},"rfne":{:.2},"fltr":{:.2},"tri":{:.2},"dem":{:.2},"total":{:.2}}}}}"#,
        st.left.width,
        st.left.height,
        st.engine.vulkan.is_some(),
        escape_json(&st.engine.vulkan_status),
        escape_json(&t.active_backend),
        t.vk_dispatches,
        shaders_count,
        st.settings.stereo_algorithm.name(),
        st.settings.cost_mode as i32,
        st.settings.prefilter_mode as i32,
        st.settings.subpixel_mode as i32,
        st.settings.corr_kernel_x,
        st.result.disp_filtered.valid_fraction() * 100.0,
        st.result.point_cloud.valid_count(),
        st.result.point_cloud.mean_error(),
        dem_min,
        dem_max,
        dem_mean,
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

const INDEX_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>RustStereoPipelineVulkan — Stereo & DEM Workstation</title>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --panel2:#1f2633; --border:#30363d; --accent:#3b9eff; --green:#2ecc71; --amber:#f5b041; --text:#e6edf3; --dim:#8b949e; }
  * { box-sizing:border-box; margin:0; padding:0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  body { background:var(--bg); color:var(--text); height:100vh; display:flex; flex-direction:column; overflow:hidden; }
  header { height:50px; background:var(--panel); border-bottom:1px solid var(--border); display:flex; align-items:center; justify-content:space-between; padding:0 18px; }
  header h1 { font-size:15px; letter-spacing:0.8px; color:var(--accent); }
  .badge { font-size:11px; padding:4px 10px; border-radius:999px; border:1px solid var(--border); background:var(--panel2); }
  .badge.gpu { color:var(--green); border-color:var(--green); }
  .badge.cpu { color:var(--amber); border-color:var(--amber); }
  main { flex:1; display:flex; min-height:0; }
  aside { width:310px; background:var(--panel); border-right:1px solid var(--border); padding:14px; overflow-y:auto; display:flex; flex-direction:column; gap:12px; }
  .section { background:var(--panel2); border:1px solid var(--border); border-radius:8px; padding:10px; }
  .section h2 { font-size:11px; color:var(--dim); text-transform:uppercase; margin-bottom:8px; letter-spacing:0.7px; }
  label { display:flex; justify-content:space-between; align-items:center; font-size:12px; margin:6px 0; color:var(--text); }
  select, input[type=number] { background:var(--bg); color:var(--text); border:1px solid var(--border); border-radius:4px; padding:3px 6px; font-size:12px; width:135px; }
  button { width:100%; padding:8px 10px; border-radius:6px; border:1px solid var(--border); background:#253041; color:var(--text); font-size:12px; cursor:pointer; font-weight:600; margin-top:6px; }
  button:hover { background:#2e3d52; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#071320; }
  button.primary:hover { filter:brightness(1.1); }
  .content { flex:1; display:flex; flex-direction:column; min-width:0; }
  .tabs { display:flex; flex-wrap:wrap; gap:4px; padding:8px 12px; background:var(--panel); border-bottom:1px solid var(--border); }
  .tab { padding:6px 10px; font-size:11px; border-radius:5px; border:1px solid var(--border); background:var(--panel2); color:var(--dim); cursor:pointer; }
  .tab.active { background:#1d4270; border-color:var(--accent); color:var(--text); }
  .viewport { flex:1; position:relative; display:flex; align-items:center; justify-content:center; background:#070a0f; overflow:hidden; }
  #imgView { max-width:95%; max-height:95%; image-rendering:pixelated; border:1px solid var(--border); box-shadow:0 8px 30px rgba(0,0,0,0.6); }
  #canvas3d { width:100%; height:100%; display:none; cursor:grab; }
  .bar-row { display:flex; align-items:center; gap:6px; font-size:11px; margin:4px 0; }
  .bar-label { width:56px; color:var(--dim); }
  .bar-track { flex:1; height:7px; background:var(--bg); border-radius:4px; overflow:hidden; }
  .bar-fill { height:100%; background:var(--accent); }
  .bar-val { width:58px; text-align:right; }
  .kv { display:flex; justify-content:space-between; font-size:11px; margin:3px 0; }
  .kv span:first-child { color:var(--dim); }
  footer { height:28px; background:var(--panel); border-top:1px solid var(--border); display:flex; align-items:center; justify-content:space-between; padding:0 14px; font-size:11px; color:var(--dim); }
</style>
</head>
<body>
<header>
  <h1>RUST STEREO PIPELINE VULKAN &mdash; ZERO-DEP COMPUTE WORKSTATION</h1>
  <div id="backendBadge" class="badge">Connecting...</div>
</header>
<main>
  <aside>
    <div class="section">
      <h2>ASP Stereo Controls</h2>
      <label>Algorithm
        <select id="algo">
          <option value="asp_bm">ASP BM (Vulkan/CPU)</option>
          <option value="asp_sgm" selected>ASP SGM (4-Path)</option>
          <option value="asp_mgm">ASP MGM (Quadrant)</option>
        </select>
      </label>
      <label>Cost Metric
        <select id="cost">
          <option value="0">SAD</option>
          <option value="1">SSD</option>
          <option value="2" selected>NCC (Zero-Mean)</option>
          <option value="3">Census 5x5</option>
        </select>
      </label>
      <label>Prefilter
        <select id="prefilter">
          <option value="0">None</option>
          <option value="1">Subtracted Mean</option>
          <option value="2" selected>LoG (Laplacian)</option>
        </select>
      </label>
      <label>Subpixel
        <select id="subpixel">
          <option value="0">Disabled</option>
          <option value="1" selected>Parabola</option>
          <option value="2">Bayes-EM Affine</option>
          <option value="3">Affine LK</option>
        </select>
      </label>
      <label>Kernel Size <input id="kernel" type="number" min="3" max="21" step="2" value="9" /></label>
      <label>Colormap
        <select id="cmap" onchange="refreshLayer()">
          <option value="turbo" selected>Turbo</option>
          <option value="viridis">Viridis</option>
          <option value="inferno">Inferno</option>
          <option value="terrain">Terrain</option>
          <option value="grayscale">Grayscale</option>
        </select>
      </label>
      <button class="primary" onclick="runPipeline()">&#9654; Run 5-Stage Pipeline</button>
      <button onclick="genSynthetic(160,120)">&#9733; Synthetic Crater Terrain (160x120)</button>
      <button onclick="genSynthetic(256,192)">&#9733; High-Res Synthetic (256x192)</button>
      <button onclick="exportOutputs()">&#128190; Export GeoTIFF / PLY / DEM</button>
    </div>

    <div class="section">
      <h2>Stage Timings</h2>
      <div id="timingsBox"></div>
    </div>

    <div class="section">
      <h2>Photogrammetry Telemetry</h2>
      <div class="kv"><span>Resolution</span><span id="statRes">-</span></div>
      <div class="kv"><span>Valid Disparity</span><span id="statValid">-</span></div>
      <div class="kv"><span>3D Points</span><span id="statPts">-</span></div>
      <div class="kv"><span>Triangulation Err</span><span id="statErr">-</span></div>
      <div class="kv"><span>DEM Range (m)</span><span id="statDem">-</span></div>
      <div class="kv"><span>SPIR-V Shaders</span><span id="statSpirv">-</span></div>
    </div>
  </aside>

  <section class="content">
    <div class="tabs" id="tabBar">
      <div class="tab" data-layer="left">1. Left</div>
      <div class="tab" data-layer="right">2. Right</div>
      <div class="tab" data-layer="pprc">3. Prefiltered</div>
      <div class="tab" data-layer="disp_raw">4. Disp Raw</div>
      <div class="tab" data-layer="disp_fltr">5. Disp Filtered</div>
      <div class="tab active" data-layer="dem">6. DEM Elevation</div>
      <div class="tab" data-layer="hillshade">7. Hillshade</div>
      <div class="tab" data-layer="error">8. Tri Error</div>
      <div class="tab" data-layer="3d">9. 3D Terrain</div>
    </div>
    <div class="viewport">
      <img id="imgView" alt="Stereo Layer" />
      <canvas id="canvas3d"></canvas>
    </div>
  </section>
</main>
<footer>
  <span id="statusText">Ready</span>
  <span>Pure Rust Zero-Dependency Server &bull; SPIR-V 1.0 Vulkan Compute</span>
</footer>
<script>
let currentLayer = 'dem';
let dem3d = null;
let yaw = 0.65, pitch = 0.55, zoom3d = 1.0, drag = false, lx = 0, ly = 0;

document.querySelectorAll('.tab').forEach(t => {
  t.onclick = () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    currentLayer = t.dataset.layer;
    refreshLayer();
  };
});

async function fetchStatus() {
  const r = await fetch('/api/status');
  const d = await r.json();
  applyStatus(d);
}

function applyStatus(d) {
  const badge = document.getElementById('backendBadge');
  badge.textContent = d.active_backend;
  badge.className = 'badge ' + (d.vulkan_available ? 'gpu' : 'cpu');
  document.getElementById('statRes').textContent = d.width + 'x' + d.height;
  document.getElementById('statValid').textContent = d.valid_pct.toFixed(1) + '%';
  document.getElementById('statPts').textContent = d.points.toLocaleString();
  document.getElementById('statErr').textContent = d.tri_error.toFixed(4) + ' m';
  document.getElementById('statDem').textContent = d.dem_min.toFixed(1) + ' .. ' + d.dem_max.toFixed(1);
  document.getElementById('statSpirv').textContent = d.spirv_shaders + ' modules (VK dispatches: ' + d.vk_dispatches + ')';

  const t = d.timings;
  const maxT = Math.max(t.pprc, t.corr, t.rfne, t.fltr, t.tri, t.dem, 1);
  const rows = [
    ['PPRC', t.pprc], ['CORR', t.corr], ['RFNE', t.rfne],
    ['FLTR', t.fltr], ['TRI', t.tri], ['DEM', t.dem], ['TOTAL', t.total]
  ];
  document.getElementById('timingsBox').innerHTML = rows.map(([name, val]) => {
    const pct = name === 'TOTAL' ? 100 : Math.min(100, (val / maxT) * 100);
    const col = name === 'TOTAL' ? '#2ecc71' : '#3b9eff';
    return `<div class="bar-row"><span class="bar-label">${name}</span><div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${col}"></div></div><span class="bar-val">${val.toFixed(1)}ms</span></div>`;
  }).join('');
}

async function refreshLayer() {
  const img = document.getElementById('imgView');
  const c3d = document.getElementById('canvas3d');
  if (currentLayer === '3d') {
    img.style.display = 'none';
    c3d.style.display = 'block';
    const r = await fetch('/api/dem3d?t=' + Date.now());
    dem3d = await r.json();
    draw3D();
  } else {
    c3d.style.display = 'none';
    img.style.display = 'block';
    const cmap = document.getElementById('cmap').value;
    img.src = `/api/image?layer=${currentLayer}&cmap=${cmap}&t=${Date.now()}`;
  }
}

async function runPipeline() {
  document.getElementById('statusText').textContent = 'Running 5-stage stereo pipeline...';
  const q = new URLSearchParams({
    algo: document.getElementById('algo').value,
    cost: document.getElementById('cost').value,
    prefilter: document.getElementById('prefilter').value,
    subpixel: document.getElementById('subpixel').value,
    kernel: document.getElementById('kernel').value
  });
  const r = await fetch('/api/run?' + q.toString(), { method: 'POST' });
  const d = await r.json();
  applyStatus(d);
  await refreshLayer();
  document.getElementById('statusText').textContent = `Completed in ${d.timings.total.toFixed(1)} ms`;
}

async function genSynthetic(w, h) {
  document.getElementById('statusText').textContent = `Generating ${w}x${h} synthetic stereo pair...`;
  const r = await fetch(`/api/synthetic?w=${w}&h=${h}`, { method: 'POST' });
  const d = await r.json();
  applyStatus(d);
  await refreshLayer();
  document.getElementById('statusText').textContent = `Generated & solved ${w}x${h} terrain in ${d.timings.total.toFixed(1)} ms`;
}

async function exportOutputs() {
  const r = await fetch('/api/export', { method: 'POST' });
  const d = await r.json();
  document.getElementById('statusText').textContent = d.message;
}

const c3d = document.getElementById('canvas3d');
c3d.onmousedown = e => { drag = true; lx = e.clientX; ly = e.clientY; };
window.onmouseup = () => drag = false;
window.onmousemove = e => {
  if (!drag || currentLayer !== '3d') return;
  yaw += (e.clientX - lx) * 0.01;
  pitch = Math.max(0.15, Math.min(1.45, pitch + (e.clientY - ly) * 0.01));
  lx = e.clientX; ly = e.clientY;
  draw3D();
};
c3d.onwheel = e => {
  e.preventDefault();
  zoom3d = Math.max(0.4, Math.min(3.5, zoom3d * (e.deltaY < 0 ? 1.1 : 0.9)));
  draw3D();
};

function draw3D() {
  if (!dem3d) return;
  c3d.width = c3d.clientWidth;
  c3d.height = c3d.clientHeight;
  const ctx = c3d.getContext('2d');
  ctx.fillStyle = '#070a0f';
  ctx.fillRect(0, 0, c3d.width, c3d.height);

  const { gw, gh, min, max, z } = dem3d;
  const range = Math.max(max - min, 1e-4);
  const sy = Math.sin(yaw), cy = Math.cos(yaw);
  const sp = Math.sin(pitch), cp = Math.cos(pitch);
  const sc = Math.min(c3d.width, c3d.height) * 0.42 * zoom3d;
  const cx = c3d.width * 0.5, cy0 = c3d.height * 0.56;

  const proj = new Array(gw * gh).fill(null);
  for (let j = 0; j < gh; j++) {
    const ny = (j / Math.max(gh - 1, 1)) * 2 - 1;
    for (let i = 0; i < gw; i++) {
      const v = z[j * gw + i];
      if (v === null) continue;
      const nx = (i / Math.max(gw - 1, 1)) * 2 - 1;
      const t = Math.max(0, Math.min(1, (v - min) / range));
      const nz = (t - 0.5) * 0.65;
      const rx = nx * cy - ny * sy;
      const ry = nx * sy + ny * cy;
      const py3 = ry * cp - nz * sp;
      const pz3 = ry * sp + nz * cp;
      const depth = 3.2 - py3;
      if (depth < 0.3) continue;
      const p = (sc / depth) * 2.3;
      proj[j * gw + i] = [cx + rx * p, cy0 - pz3 * p, t];
    }
  }
  ctx.lineWidth = 1.1;
  for (let j = 0; j < gh; j++) {
    for (let i = 0; i < gw; i++) {
      const a = proj[j * gw + i];
      if (!a) continue;
      const hue = 220 - a[2] * 210;
      ctx.strokeStyle = `hsl(${hue}, 82%, 58%)`;
      if (i + 1 < gw && proj[j * gw + i + 1]) {
        const b = proj[j * gw + i + 1];
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      }
      if (j + 1 < gh && proj[(j + 1) * gw + i]) {
        const b = proj[(j + 1) * gw + i];
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      }
    }
  }
  ctx.fillStyle = '#8b949e';
  ctx.font = '12px monospace';
  ctx.fillText('Drag to orbit 3D DEM | Scroll to zoom', 14, c3d.height - 14);
}

fetchStatus().then(refreshLayer);
</script>
</body>
</html>
"##;
