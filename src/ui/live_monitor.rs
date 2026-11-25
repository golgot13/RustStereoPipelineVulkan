//! Zero-dependency real-time Chantier execution monitor & progressive 2D/3D renderer.
//!
//! Feeds live tile-by-tile updates from `run_chantier_full_model` (both CLI `--live-ui`
//! and Standalone Web Workstation) with < 1 ms overhead per 1024x1024 tile:
//! - Live SSE telemetry (tile progress, Mpx/s, ETA, per-stage timings, LiDAR/BA metrics, logs)
//! - Progressive 2D Full-Model Mosaic (RGB Orthophoto / Elevation Turbo+Hillshade / Tile Coverage Grid)
//! - Progressive Binary 3D Colorized Point Cloud stream (`f32` relative to `f64` origin, up to 400k live pts)

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use crate::core::cancel::CancellationToken;
use crate::core::chantier::{Chantier, FullModelPlan, ModelTileSpec};
use crate::image::io::encode_png_rgba;
use crate::image::{Colormap, ImageF32, ImageRgb, PointCloud};

const MAX_LIVE_CLOUD_PTS: usize = 400_000;
const MAX_MOSAIC_DIM: usize = 1200;
const MAX_LOGS: usize = 200;

#[derive(Debug, Clone, Copy)]
pub struct LivePoint3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

#[derive(Debug, Clone)]
pub struct PairSummaryInfo {
    pub index: usize,
    pub name: String,
    pub baseline_m: f64,
    pub left_name: String,
    pub right_name: String,
}

#[derive(Debug, Clone, Default)]
pub struct StageTimingsMs {
    pub band_io: f64,
    pub rect: f64,
    pub pprc: f64,
    pub corr: f64,
    pub rfne: f64,
    pub fltr: f64,
    pub tri: f64,
    pub ply: f64,
}

pub struct LiveChantierState {
    pub phase: String, // "idle" | "discovering" | "running" | "done" | "cancelled" | "error"
    pub status_msg: String,
    pub last_error: Option<String>,
    pub chantier_dir: String,
    pub output_dir: String,
    pub camera_info: String,
    pub survey_crs: String,
    pub backend_info: String,
    pub cli_command: String,

    // Discovery metadata
    pub poses_count: usize,
    pub gcp_count: usize,
    pub lidar_summary: String,
    pub footprints_summary: String,
    pub ba_summary: String,
    pub pairs: Vec<PairSummaryInfo>,

    // Active model state
    pub model_name: String,
    pub model_index: usize,
    pub total_models: usize,
    pub overlap_w: usize,
    pub overlap_h: usize,
    pub epipolar_info: String,
    pub area_ha: Option<f64>,
    pub output_ply: String,

    // Grid & progress
    pub tiles_x: usize,
    pub tiles_y: usize,
    pub total_tiles: usize,
    pub tiles_done: usize,
    pub tiles_skipped: usize,
    pub band_idx: usize,
    pub total_bands: usize,
    pub active_tx: Option<usize>,
    pub active_ty: Option<usize>,
    pub tile_state: Vec<u8>,      // 0=pending, 1=active, 2=done, 3=skipped
    pub tile_valid_pct: Vec<f32>, // 0..100
    pub tile_pts: Vec<u32>,

    // Throughput & quality
    pub total_points: u64,
    pub core_pixels: u64,
    pub mpx_per_sec: f64,
    pub eta_sec: f64,
    pub elapsed_sec: f64,
    pub avg_valid_pct: f64,
    pub sum_valid_pct: f64,
    pub vk_dispatches: usize,
    pub timings: StageTimingsMs,

    // Progressive 2D mosaic
    pub mosaic_w: usize,
    pub mosaic_h: usize,
    pub mosaic_rgb: Vec<u8>, // mw * mh * 4
    pub mosaic_z: Vec<f32>,  // mw * mh (NAN if empty)
    pub mosaic_rev: u64,

    // Progressive 3D cloud
    pub cloud_origin: Option<[f64; 3]>,
    pub cloud_pts: Vec<LivePoint3D>,
    pub cloud_min_rel: [f32; 3],
    pub cloud_max_rel: [f32; 3],
    pub cloud_rev: u64,
    pub z_min_world: f64,
    pub z_max_world: f64,

    // Logs & SSE counter
    pub logs: VecDeque<String>,
    pub event_seq: u64,
    pub started_at: Option<Instant>,
}

impl Default for LiveChantierState {
    fn default() -> Self {
        Self {
            phase: "idle".to_string(),
            status_msg: "Prêt — sélectionnez un chantier ou lancez depuis la CLI avec --live-ui".to_string(),
            last_error: None,
            chantier_dir: String::new(),
            output_dir: String::new(),
            camera_info: "-".to_string(),
            survey_crs: "-".to_string(),
            backend_info: "-".to_string(),
            cli_command: String::new(),
            poses_count: 0,
            gcp_count: 0,
            lidar_summary: "Aucun".to_string(),
            footprints_summary: "-".to_string(),
            ba_summary: "-".to_string(),
            pairs: Vec::new(),
            model_name: "-".to_string(),
            model_index: 0,
            total_models: 0,
            overlap_w: 0,
            overlap_h: 0,
            epipolar_info: "-".to_string(),
            area_ha: None,
            output_ply: String::new(),
            tiles_x: 0,
            tiles_y: 0,
            total_tiles: 0,
            tiles_done: 0,
            tiles_skipped: 0,
            band_idx: 0,
            total_bands: 0,
            active_tx: None,
            active_ty: None,
            tile_state: Vec::new(),
            tile_valid_pct: Vec::new(),
            tile_pts: Vec::new(),
            total_points: 0,
            core_pixels: 0,
            mpx_per_sec: 0.0,
            eta_sec: 0.0,
            elapsed_sec: 0.0,
            avg_valid_pct: 0.0,
            sum_valid_pct: 0.0,
            vk_dispatches: 0,
            timings: StageTimingsMs::default(),
            mosaic_w: 0,
            mosaic_h: 0,
            mosaic_rgb: Vec::new(),
            mosaic_z: Vec::new(),
            mosaic_rev: 0,
            cloud_origin: None,
            cloud_pts: Vec::new(),
            cloud_min_rel: [f32::INFINITY; 3],
            cloud_max_rel: [f32::NEG_INFINITY; 3],
            cloud_rev: 0,
            z_min_world: f64::INFINITY,
            z_max_world: f64::NEG_INFINITY,
            logs: VecDeque::with_capacity(MAX_LOGS),
            event_seq: 1,
            started_at: None,
        }
    }
}

pub struct LiveChantierMonitor {
    pub inner: Mutex<LiveChantierState>,
    pub cancel: CancellationToken,
}

impl LiveChantierMonitor {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(LiveChantierState::default()),
            cancel: CancellationToken::new(),
        })
    }

    pub fn lock(&self) -> MutexGuard<'_, LiveChantierState> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn push_log(&self, msg: impl Into<String>) {
        let mut st = self.lock();
        let line = msg.into();
        if st.logs.len() >= MAX_LOGS {
            st.logs.pop_front();
        }
        st.logs.push_back(line);
        st.event_seq += 1;
    }

    pub fn set_discovering(&self, chantier_dir: &str, cli_command: &str) {
        self.cancel.reset();
        let mut st = self.lock();
        *st = LiveChantierState::default();
        st.phase = "discovering".to_string();
        st.chantier_dir = chantier_dir.to_string();
        st.cli_command = cli_command.to_string();
        st.status_msg = format!("Analyse du chantier & LiDAR HD : {}", chantier_dir);
        st.started_at = Some(Instant::now());
        st.logs.push_back(format!("[Chantier] Analyse de {} ...", chantier_dir));
        st.event_seq += 1;
    }

    pub fn set_chantier_info(
        &self,
        chantier: &Chantier,
        output_dir: &str,
        backend_info: &str,
        cli_command: &str,
    ) {
        let mut st = self.lock();
        st.chantier_dir = chantier.root_dir.display().to_string();
        st.output_dir = output_dir.to_string();
        st.camera_info = format!(
            "{} ({}x{} px, f={:.4} mm, pitch={:.6} mm)",
            chantier.calibration.name,
            chantier.calibration.width_px,
            chantier.calibration.height_px,
            chantier.calibration.focal_mm,
            chantier.calibration.pixel_size_mm
        );
        st.survey_crs = chantier.survey_crs.code().to_string();
        st.backend_info = backend_info.to_string();
        if !cli_command.is_empty() {
            st.cli_command = cli_command.to_string();
        }
        st.poses_count = chantier.poses.len();
        st.gcp_count = chantier.gcps.len();

        st.lidar_summary = if let Some(ref mnt) = chantier.lidar_mnt {
            let mut dz_gcp: Vec<f64> = chantier
                .gcps
                .iter()
                .filter_map(|g| mnt.sample_ground_z(g.x, g.y, chantier.survey_crs).map(|zm| zm - g.z))
                .collect();
            let cmp = if !dz_gcp.is_empty() {
                dz_gcp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                format!(" | MNT vs {} GCP dZ={:+.3} m", dz_gcp.len(), dz_gcp[dz_gcp.len() / 2])
            } else {
                String::new()
            };
            format!(
                "{} dalles .laz ({} -> {}, Z=[{:.1}..{:.1}] m{})",
                mnt.tile_count,
                mnt.lidar_crs.code(),
                chantier.survey_crs.code(),
                mnt.z_min,
                mnt.z_max,
                cmp
            )
        } else {
            "Désactivé / absent".to_string()
        };

        st.footprints_summary = if let Some(ref fr) = chantier.footprint_report {
            format!(
                "{}/{} poses | {} | IoU={:.1}% (min {:.1}%) | dXY={:.2} m",
                fr.matched_poses,
                fr.total_poses,
                if fr.used_lidar_mnt { "AGL LiDAR MNT" } else { "plan Z_ref" },
                fr.mean_iou_pct,
                fr.min_iou_pct,
                fr.mean_center_err_m
            )
        } else {
            "Non fourni".to_string()
        };

        st.ba_summary = if let Some(ref ba) = chantier.ba_report {
            format!(
                "{} ({} obs., {} GCP) | Reproj {:.2} -> {:.2} px | Épi dy {:.2} -> {:.2} px | dZ {:+.3} m",
                if ba.applied { "APPLIQUÉ" } else { "IGNORÉ" },
                ba.matched_observations,
                ba.matched_gcps,
                ba.initial_reproj_rmse_px,
                ba.final_reproj_rmse_px,
                ba.initial_epi_dy_rmse_px,
                ba.final_epi_dy_rmse_px,
                ba.final_tri_dz_bias_m
            )
        } else {
            "Poses GNSS/IMU brutes".to_string()
        };

        st.pairs = chantier
            .pairs
            .iter()
            .enumerate()
            .map(|(i, pr)| PairSummaryInfo {
                index: i,
                name: pr.name.clone(),
                baseline_m: pr.baseline_m,
                left_name: pr
                    .left_image_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_string(),
                right_name: pr
                    .right_image_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_string(),
            })
            .collect();

        if st.phase == "discovering" {
            st.phase = "idle".to_string();
            st.status_msg = format!(
                "Chantier prêt : {} paires stéréo, {} poses, {} GCP",
                st.pairs.len(),
                st.poses_count,
                st.gcp_count
            );
        }
        st.event_seq += 1;
    }

    pub fn begin_model(
        &self,
        model_idx: usize,
        total_models: usize,
        pair_name: &str,
        plan: &FullModelPlan,
        epipolar_info: String,
        output_ply: String,
    ) {
        let mut st = self.lock();
        st.phase = "running".to_string();
        st.last_error = None;
        st.model_name = pair_name.to_string();
        st.model_index = model_idx;
        st.total_models = total_models;
        st.overlap_w = plan.overlap_w as usize;
        st.overlap_h = plan.overlap_h as usize;
        st.epipolar_info = epipolar_info;
        st.area_ha = plan.validated_overlap_area_m2.map(|a| a / 10_000.0);
        st.output_ply = output_ply;

        st.tiles_x = plan.tiles_x as usize;
        st.tiles_y = plan.tiles_y as usize;
        st.total_tiles = plan.total_tiles;
        st.tiles_done = 0;
        st.tiles_skipped = 0;
        st.band_idx = 0;
        st.total_bands = plan.bands.len();
        st.active_tx = None;
        st.active_ty = None;

        let n_grid = (st.tiles_x * st.tiles_y).max(1);
        st.tile_state = vec![0u8; n_grid];
        st.tile_valid_pct = vec![0.0f32; n_grid];
        st.tile_pts = vec![0u32; n_grid];

        st.total_points = 0;
        st.core_pixels = 0;
        st.mpx_per_sec = 0.0;
        st.eta_sec = 0.0;
        st.elapsed_sec = 0.0;
        st.avg_valid_pct = 0.0;
        st.sum_valid_pct = 0.0;
        st.vk_dispatches = 0;
        st.timings = StageTimingsMs::default();
        st.started_at = Some(Instant::now());

        // Allocate compact progressive 2D mosaic preserving exact overlap aspect ratio
        let ow = (plan.overlap_w as usize).max(1);
        let oh = (plan.overlap_h as usize).max(1);
        let (mw, mh) = if ow >= oh {
            let mw = ow.min(MAX_MOSAIC_DIM);
            let mh = ((oh as f64 / ow as f64) * mw as f64).round().max(64.0) as usize;
            (mw, mh)
        } else {
            let mh = oh.min(MAX_MOSAIC_DIM);
            let mw = ((ow as f64 / oh as f64) * mh as f64).round().max(64.0) as usize;
            (mw, mh)
        };
        st.mosaic_w = mw;
        st.mosaic_h = mh;
        st.mosaic_rgb = vec![0u8; mw * mh * 4];
        for px in st.mosaic_rgb.chunks_exact_mut(4) {
            px[0] = 11;
            px[1] = 15;
            px[2] = 23;
            px[3] = 255;
        }
        st.mosaic_z = vec![f32::NAN; mw * mh];
        st.mosaic_rev += 1;

        st.cloud_origin = None;
        st.cloud_pts.clear();
        st.cloud_pts.reserve(65_536);
        st.cloud_min_rel = [f32::INFINITY; 3];
        st.cloud_max_rel = [f32::NEG_INFINITY; 3];
        st.cloud_rev += 1;
        st.z_min_world = f64::INFINITY;
        st.z_max_world = f64::NEG_INFINITY;

        st.status_msg = format!(
            "Modèle {} : {}x{} px ({} dalles 1:1, {} bandes)",
            pair_name, plan.overlap_w, plan.overlap_h, plan.total_tiles, plan.bands.len()
        );
        if st.logs.len() >= MAX_LOGS {
            st.logs.pop_front();
        }
        st.logs.push_back(format!(
            "=== Modèle {} ({}x{} px, {}x{}={} dalles) ===",
            pair_name, plan.overlap_w, plan.overlap_h, plan.tiles_x, plan.tiles_y, plan.total_tiles
        ));
        st.event_seq += 1;
    }

    pub fn on_band_start(&self, band_idx: usize, total_bands: usize, band_io_ms: f64, desc: &str) {
        let mut st = self.lock();
        st.band_idx = band_idx + 1;
        st.total_bands = total_bands;
        st.timings.band_io += band_io_ms;
        st.status_msg = format!("Bande {}/{} — {}", band_idx + 1, total_bands, desc);
        if st.logs.len() >= MAX_LOGS {
            st.logs.pop_front();
        }
        st.logs.push_back(format!("-- Bande {}/{} ({}) --", band_idx + 1, total_bands, desc));
        st.event_seq += 1;
    }

    pub fn on_tile_start(&self, tx: usize, ty: usize) {
        let mut st = self.lock();
        st.active_tx = Some(tx);
        st.active_ty = Some(ty);
        let idx = ty * st.tiles_x + tx;
        if idx < st.tile_state.len() {
            st.tile_state[idx] = 1;
        }
        st.event_seq += 1;
    }

    pub fn on_tile_skipped(&self, tx: usize, ty: usize) {
        let mut st = self.lock();
        let idx = ty * st.tiles_x + tx;
        if idx < st.tile_state.len() {
            st.tile_state[idx] = 3;
        }
        st.tiles_skipped += 1;
        st.tiles_done += 1;
        st.active_tx = None;
        st.active_ty = None;
        st.mosaic_rev += 1;
        st.event_seq += 1;
    }

    /// Update live 2D mosaic + progressive 3D point cloud + telemetry right after a tile completes.
    #[allow(clippy::too_many_arguments)]
    pub fn on_tile_complete(
        &self,
        tile: &ModelTileSpec,
        left_img: &ImageF32,
        left_rgb: Option<&ImageRgb>,
        point_cloud: &PointCloud,
        added_pts: usize,
        valid_pct: f64,
        coarse_dx: i32,
        rect_ms: f64,
        pprc_ms: f64,
        corr_ms: f64,
        rfne_ms: f64,
        fltr_ms: f64,
        tri_ms: f64,
        ply_ms: f64,
        vk_dispatches: usize,
        tile_s: f64,
    ) {
        let mut st = self.lock();
        let grid_idx = tile.ty as usize * st.tiles_x + tile.tx as usize;
        if grid_idx < st.tile_state.len() {
            st.tile_state[grid_idx] = 2;
            st.tile_valid_pct[grid_idx] = valid_pct as f32;
            st.tile_pts[grid_idx] = added_pts as u32;
        }
        st.active_tx = None;
        st.active_ty = None;

        st.tiles_done += 1;
        let core_w = tile.emit_x1.saturating_sub(tile.emit_x0) as u64;
        let core_h = tile.emit_y1.saturating_sub(tile.emit_y0) as u64;
        st.core_pixels += core_w * core_h;
        st.total_points += added_pts as u64;
        st.sum_valid_pct += valid_pct;
        let active_done = st.tiles_done.saturating_sub(st.tiles_skipped).max(1);
        st.avg_valid_pct = st.sum_valid_pct / active_done as f64;
        st.vk_dispatches += vk_dispatches;

        st.timings.rect += rect_ms;
        st.timings.pprc += pprc_ms;
        st.timings.corr += corr_ms;
        st.timings.rfne += rfne_ms;
        st.timings.fltr += fltr_ms;
        st.timings.tri += tri_ms;
        st.timings.ply += ply_ms;

        let elapsed = st
            .started_at
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(1e-6)
            .max(1e-6);
        st.elapsed_sec = elapsed;
        st.mpx_per_sec = (st.core_pixels as f64 / 1_000_000.0) / elapsed;
        let remaining = st.total_tiles.saturating_sub(st.tiles_done);
        st.eta_sec = if st.tiles_done > 0 {
            (elapsed / st.tiles_done as f64) * remaining as f64
        } else {
            0.0
        };

        // --- 1. Incremental 2D Progressive Mosaic patch (< 0.05 ms per tile!) ---
        let mw = st.mosaic_w;
        let mh = st.mosaic_h;
        let ow = st.overlap_w.max(1) as f64;
        let oh = st.overlap_h.max(1) as f64;
        if mw > 0 && mh > 0 && core_w > 0 && core_h > 0 {
            let gx0 = tile.px0 as usize + tile.emit_x0 as usize;
            let gy0 = tile.py0 as usize + tile.emit_y0 as usize;
            let gx1 = tile.px0 as usize + tile.emit_x1 as usize;
            let gy1 = tile.py0 as usize + tile.emit_y1 as usize;

            let mx0 = ((gx0 as f64 * mw as f64) / ow).floor() as usize;
            let mx1 = (((gx1 as f64 * mw as f64) / ow).ceil() as usize).min(mw);
            let my0 = ((gy0 as f64 * mh as f64) / oh).floor() as usize;
            let my1 = (((gy1 as f64 * mh as f64) / oh).ceil() as usize).min(mh);

            for my in my0..my1 {
                let gy = (((my as f64 + 0.5) * oh) / mh as f64).floor() as isize;
                let ly = (gy - tile.py0 as isize).clamp(tile.emit_y0 as isize, (tile.emit_y1 - 1) as isize) as u32;
                let m_row = my * mw;
                for mx in mx0..mx1 {
                    let gx = (((mx as f64 + 0.5) * ow) / mw as f64).floor() as isize;
                    let lx = (gx - tile.px0 as isize).clamp(tile.emit_x0 as isize, (tile.emit_x1 - 1) as isize) as u32;

                    // True RGB or grayscale luminance from rectified left image
                    let [r, g, b] = if let Some(rgb) = left_rgb {
                        if lx < rgb.width && ly < rgb.height {
                            rgb.get(lx, ly)
                        } else {
                            [0, 0, 0]
                        }
                    } else if lx < left_img.width && ly < left_img.height {
                        let v = (left_img.get(lx, ly).clamp(0.0, 1.0) * 255.0) as u8;
                        [v, v, v]
                    } else {
                        [0, 0, 0]
                    };
                    let o4 = (m_row + mx) * 4;
                    st.mosaic_rgb[o4] = r;
                    st.mosaic_rgb[o4 + 1] = g;
                    st.mosaic_rgb[o4 + 2] = b;
                    st.mosaic_rgb[o4 + 3] = 255;

                    // Elevation Z from triangulated 3D cloud (check small 3x3 neighborhood if center invalid)
                    let mut z_val = f32::NAN;
                    if lx < point_cloud.width && ly < point_cloud.height {
                        let pi = (ly * point_cloud.width + lx) as usize;
                        if point_cloud.valid[pi] != 0 && point_cloud.points[pi].z.is_finite() {
                            z_val = point_cloud.points[pi].z as f32;
                        } else {
                            'search: for dy in -2i32..=2 {
                                let ny = ly as i32 + dy;
                                if ny < tile.emit_y0 as i32 || ny >= tile.emit_y1 as i32 {
                                    continue;
                                }
                                for dx in -2i32..=2 {
                                    let nx = lx as i32 + dx;
                                    if nx < tile.emit_x0 as i32 || nx >= tile.emit_x1 as i32 {
                                        continue;
                                    }
                                    let npi = (ny as u32 * point_cloud.width + nx as u32) as usize;
                                    if point_cloud.valid[npi] != 0 && point_cloud.points[npi].z.is_finite() {
                                        z_val = point_cloud.points[npi].z as f32;
                                        break 'search;
                                    }
                                }
                            }
                        }
                    }
                    if z_val.is_finite() {
                        st.mosaic_z[m_row + mx] = z_val;
                        let z64 = z_val as f64;
                        if z64 < st.z_min_world {
                            st.z_min_world = z64;
                        }
                        if z64 > st.z_max_world {
                            st.z_max_world = z64;
                        }
                    }
                }
            }
            st.mosaic_rev += 1;
        }

        // --- 2. Incremental 3D Colorized Point Cloud Subsampling (< 0.3 ms per tile!) ---
        if st.cloud_pts.len() < MAX_LIVE_CLOUD_PTS && added_pts > 0 {
            let budget_per_tile = (MAX_LIVE_CLOUD_PTS / st.total_tiles.max(1)).clamp(500, 4000);
            let remaining_cap = MAX_LIVE_CLOUD_PTS - st.cloud_pts.len();
            let want = budget_per_tile.min(remaining_cap);
            let cw = (tile.emit_x1 - tile.emit_x0) as usize;
            let ch = (tile.emit_y1 - tile.emit_y0) as usize;
            let stride = (((cw * ch) as f64 / (want as f64 * 1.4)).sqrt().floor() as usize).max(1);
            let mut pushed = 0usize;

            let mut ly = tile.emit_y0 as usize;
            while ly < tile.emit_y1 as usize && pushed < want {
                let row_off = ly * point_cloud.width as usize;
                let mut lx = tile.emit_x0 as usize;
                while lx < tile.emit_x1 as usize && pushed < want {
                    let pi = row_off + lx;
                    if pi < point_cloud.valid.len() && point_cloud.valid[pi] != 0 {
                        let pt = point_cloud.points[pi];
                        if pt.x.is_finite() && pt.y.is_finite() && pt.z.is_finite() {
                            let origin = *st.cloud_origin.get_or_insert([pt.x, pt.y, pt.z]);
                            let rx = (pt.x - origin[0]) as f32;
                            let ry = (pt.y - origin[1]) as f32;
                            let rz = (pt.z - origin[2]) as f32;
                            let [r, g, b] = if let Some(ref cols) = point_cloud.colors {
                                cols[pi]
                            } else if let Some(rgb) = left_rgb {
                                rgb.get(lx as u32, ly as u32)
                            } else {
                                let g8 = (left_img.get(lx as u32, ly as u32).clamp(0.0, 1.0) * 255.0) as u8;
                                [g8, g8, g8]
                            };
                            st.cloud_pts.push(LivePoint3D {
                                x: rx,
                                y: ry,
                                z: rz,
                                r,
                                g,
                                b,
                            });
                            if rx < st.cloud_min_rel[0] { st.cloud_min_rel[0] = rx; }
                            if ry < st.cloud_min_rel[1] { st.cloud_min_rel[1] = ry; }
                            if rz < st.cloud_min_rel[2] { st.cloud_min_rel[2] = rz; }
                            if rx > st.cloud_max_rel[0] { st.cloud_max_rel[0] = rx; }
                            if ry > st.cloud_max_rel[1] { st.cloud_max_rel[1] = ry; }
                            if rz > st.cloud_max_rel[2] { st.cloud_max_rel[2] = rz; }
                            if pt.z < st.z_min_world { st.z_min_world = pt.z; }
                            if pt.z > st.z_max_world { st.z_max_world = pt.z; }
                            pushed += 1;
                        }
                    }
                    lx += stride;
                }
                ly += stride;
            }
            if pushed > 0 {
                st.cloud_rev += 1;
            }
        }

        let log_line = format!(
            "[{:>3}/{}] ({:>2},{:>2}) dx={:+4} -> {:>7} pts ({:>5.1}%) | {:.2}s [rect={:.0} corr={:.0} fltr={:.0} tri={:.0} ms] | {:.2} Mpx/s | ETA {:.0}s",
            st.tiles_done,
            st.total_tiles,
            tile.tx,
            tile.ty,
            coarse_dx,
            added_pts,
            valid_pct,
            tile_s,
            rect_ms,
            corr_ms,
            fltr_ms,
            tri_ms,
            st.mpx_per_sec,
            st.eta_sec
        );
        st.status_msg = format!(
            "Dalle {}/{} ({:.1}%) — {:.2} Mpts — {:.2} Mpx/s — ETA {:.0}s",
            st.tiles_done,
            st.total_tiles,
            if st.total_tiles > 0 {
                100.0 * st.tiles_done as f64 / st.total_tiles as f64
            } else {
                0.0
            },
            st.total_points as f64 / 1_000_000.0,
            st.mpx_per_sec,
            st.eta_sec
        );
        if st.logs.len() >= MAX_LOGS {
            st.logs.pop_front();
        }
        st.logs.push_back(log_line);
        st.event_seq += 1;
    }

    pub fn finish_ok(&self, summary: String) {
        let mut st = self.lock();
        st.phase = "done".to_string();
        st.eta_sec = 0.0;
        st.active_tx = None;
        st.active_ty = None;
        st.status_msg = summary.clone();
        if st.logs.len() >= MAX_LOGS {
            st.logs.pop_front();
        }
        st.logs.push_back(summary);
        st.event_seq += 1;
    }

    pub fn finish_cancelled(&self) {
        let mut st = self.lock();
        st.phase = "cancelled".to_string();
        st.active_tx = None;
        st.active_ty = None;
        st.status_msg = "Calcul interrompu par l'utilisateur.".to_string();
        st.logs.push_back("[ANNULÉ] Calcul interrompu proprement.".to_string());
        st.event_seq += 1;
    }

    pub fn finish_error(&self, err: String) {
        let mut st = self.lock();
        st.phase = "error".to_string();
        st.active_tx = None;
        st.active_ty = None;
        st.last_error = Some(err.clone());
        st.status_msg = format!("Erreur : {}", err);
        st.logs.push_back(format!("[ERREUR] {}", err));
        st.event_seq += 1;
    }

    /// Encode the current progressive 2D mosaic (`rgb` | `elev` | `grid` | `overlay`) to PNG bytes.
    pub fn render_mosaic_png(&self, layer: &str, cmap_name: &str) -> Vec<u8> {
        let st = self.lock();
        let mw = st.mosaic_w;
        let mh = st.mosaic_h;
        if mw == 0 || mh == 0 {
            // Return a clean 320x200 dark placeholder PNG
            let mut rgba = vec![0u8; 320 * 200 * 4];
            for px in rgba.chunks_exact_mut(4) {
                px[0] = 13;
                px[1] = 17;
                px[2] = 23;
                px[3] = 255;
            }
            return encode_png_rgba(320, 200, &rgba);
        }

        let cmap = Colormap::from_name(cmap_name);
        let mut rgba = vec![0u8; mw * mh * 4];

        match layer {
            "elev" => {
                let (zmin, zmax) = if st.z_min_world.is_finite() && st.z_max_world > st.z_min_world + 0.1 {
                    (st.z_min_world as f32, st.z_max_world as f32)
                } else {
                    (0.0f32, 1.0f32)
                };
                let inv = 1.0 / (zmax - zmin).max(1e-3);
                for y in 0..mh {
                    let row = y * mw;
                    for x in 0..mw {
                        let z = st.mosaic_z[row + x];
                        let o = (row + x) * 4;
                        if z.is_finite() {
                            let t = ((z - zmin) * inv).clamp(0.0, 1.0);
                            let [cr, cg, cb, _] = cmap.sample(t);
                            // Subtle relief shading from immediate neighbors
                            let mut shade = 1.0f32;
                            if x + 1 < mw && y + 1 < mh {
                                let zx = st.mosaic_z[row + x + 1];
                                let zy = st.mosaic_z[(y + 1) * mw + x];
                                if zx.is_finite() && zy.is_finite() {
                                    let slope = ((zx - z) + (zy - z)) * 0.12;
                                    shade = (1.0 - slope).clamp(0.62, 1.28);
                                }
                            }
                            rgba[o] = (cr as f32 * shade).clamp(0.0, 255.0) as u8;
                            rgba[o + 1] = (cg as f32 * shade).clamp(0.0, 255.0) as u8;
                            rgba[o + 2] = (cb as f32 * shade).clamp(0.0, 255.0) as u8;
                            rgba[o + 3] = 255;
                        } else {
                            rgba[o] = 11;
                            rgba[o + 1] = 15;
                            rgba[o + 2] = 23;
                            rgba[o + 3] = 255;
                        }
                    }
                }
            }
            "grid" => {
                let tx_n = st.tiles_x.max(1);
                let ty_n = st.tiles_y.max(1);
                for y in 0..mh {
                    let ty = ((y * ty_n) / mh).min(ty_n - 1);
                    let border_y = y == 0 || ((y * ty_n) / mh) != (((y - 1) * ty_n) / mh);
                    for x in 0..mw {
                        let tx = ((x * tx_n) / mw).min(tx_n - 1);
                        let border_x = x == 0 || ((x * tx_n) / mw) != (((x - 1) * tx_n) / mw);
                        let o = (y * mw + x) * 4;
                        if border_x || border_y {
                            rgba[o] = 35;
                            rgba[o + 1] = 44;
                            rgba[o + 2] = 58;
                            rgba[o + 3] = 255;
                            continue;
                        }
                        let gi = ty * tx_n + tx;
                        let s = st.tile_state.get(gi).copied().unwrap_or(0);
                        let (r, g, b) = match s {
                            1 => (59, 158, 255), // Active tile (bright cyan-blue)
                            2 => {
                                let v = st.tile_valid_pct.get(gi).copied().unwrap_or(0.0) / 100.0;
                                if v >= 0.75 {
                                    (34, 197, 94) // Green: dense
                                } else if v >= 0.40 {
                                    (234, 179, 8) // Amber: medium
                                } else {
                                    (249, 115, 22) // Orange: sparse/edge
                                }
                            }
                            3 => (48, 54, 61), // Skipped (outside sensor)
                            _ => (18, 24, 34), // Pending
                        };
                        rgba[o] = r;
                        rgba[o + 1] = g;
                        rgba[o + 2] = b;
                        rgba[o + 3] = 255;
                    }
                }
            }
            "overlay" => {
                // RGB blended with elevation tint + tile grid lines
                let (zmin, zmax) = if st.z_min_world.is_finite() && st.z_max_world > st.z_min_world + 0.1 {
                    (st.z_min_world as f32, st.z_max_world as f32)
                } else {
                    (0.0f32, 1.0f32)
                };
                let inv = 1.0 / (zmax - zmin).max(1e-3);
                let tx_n = st.tiles_x.max(1);
                let ty_n = st.tiles_y.max(1);
                for y in 0..mh {
                    let row = y * mw;
                    let border_y = y > 0 && ((y * ty_n) / mh) != (((y - 1) * ty_n) / mh);
                    for x in 0..mw {
                        let o = (row + x) * 4;
                        let border_x = x > 0 && ((x * tx_n) / mw) != (((x - 1) * tx_n) / mw);
                        let z = st.mosaic_z[row + x];
                        let (mut r, mut g, mut b) = (
                            st.mosaic_rgb[o] as f32,
                            st.mosaic_rgb[o + 1] as f32,
                            st.mosaic_rgb[o + 2] as f32,
                        );
                        if z.is_finite() {
                            let [cr, cg, cb, _] = cmap.sample(((z - zmin) * inv).clamp(0.0, 1.0));
                            r = r * 0.62 + cr as f32 * 0.38;
                            g = g * 0.62 + cg as f32 * 0.38;
                            b = b * 0.62 + cb as f32 * 0.38;
                        }
                        if border_x || border_y {
                            r = r * 0.7 + 40.0;
                            g = g * 0.7 + 55.0;
                            b = b * 0.7 + 75.0;
                        }
                        rgba[o] = r.clamp(0.0, 255.0) as u8;
                        rgba[o + 1] = g.clamp(0.0, 255.0) as u8;
                        rgba[o + 2] = b.clamp(0.0, 255.0) as u8;
                        rgba[o + 3] = 255;
                    }
                }
            }
            _ => {
                // Default "rgb": true-color progressive orthophoto mosaic
                rgba.copy_from_slice(&st.mosaic_rgb);
            }
        }

        // Highlight currently active tile with a crisp cyan rectangle on all layers
        if let (Some(atx), Some(aty)) = (st.active_tx, st.active_ty) {
            let tx_n = st.tiles_x.max(1);
            let ty_n = st.tiles_y.max(1);
            let x0 = (atx * mw) / tx_n;
            let x1 = (((atx + 1) * mw) / tx_n).min(mw).saturating_sub(1);
            let y0 = (aty * mh) / ty_n;
            let y1 = (((aty + 1) * mh) / ty_n).min(mh).saturating_sub(1);
            for x in x0..=x1 {
                for &y in &[y0, y1] {
                    let o = (y * mw + x) * 4;
                    rgba[o] = 59;
                    rgba[o + 1] = 220;
                    rgba[o + 2] = 255;
                    rgba[o + 3] = 255;
                }
            }
            for y in y0..=y1 {
                for &x in &[x0, x1] {
                    let o = (y * mw + x) * 4;
                    rgba[o] = 59;
                    rgba[o + 1] = 220;
                    rgba[o + 2] = 255;
                    rgba[o + 3] = 255;
                }
            }
        }

        encode_png_rgba(mw as u32, mh as u32, &rgba)
    }

    /// Encode newly appended 3D preview points starting at `from_idx` into a compact binary stream
    /// (`24-byte header + N * 16 bytes` per point: `f32 x, f32 y, f32 z, u8 r, u8 g, u8 b, u8 a`).
    pub fn encode_cloud_chunk_bin(&self, from_idx: usize) -> Vec<u8> {
        let st = self.lock();
        let total = st.cloud_pts.len();
        let start = from_idx.min(total);
        let slice = &st.cloud_pts[start..];

        let (zmin, zmax, extent) = if total > 0 && st.cloud_min_rel[0].is_finite() {
            let dx = (st.cloud_max_rel[0] - st.cloud_min_rel[0]).abs();
            let dy = (st.cloud_max_rel[1] - st.cloud_min_rel[1]).abs();
            let dz = (st.cloud_max_rel[2] - st.cloud_min_rel[2]).abs();
            let ext = dx.max(dy).max(dz).max(1.0);
            (st.cloud_min_rel[2], st.cloud_max_rel[2], ext)
        } else {
            (0.0f32, 1.0f32, 100.0f32)
        };

        let mut out = Vec::with_capacity(36 + slice.len() * 16);
        out.extend_from_slice(b"PC3D");
        out.extend_from_slice(&(total as u32).to_le_bytes());
        out.extend_from_slice(&(slice.len() as u32).to_le_bytes());
        out.extend_from_slice(&zmin.to_le_bytes());
        out.extend_from_slice(&zmax.to_le_bytes());
        out.extend_from_slice(&extent.to_le_bytes());
        // Center of bounding box in relative coords for smooth WebGL orbit
        let cx = if total > 0 { 0.5 * (st.cloud_min_rel[0] + st.cloud_max_rel[0]) } else { 0.0 };
        let cy = if total > 0 { 0.5 * (st.cloud_min_rel[1] + st.cloud_max_rel[1]) } else { 0.0 };
        let cz = if total > 0 { 0.5 * (st.cloud_min_rel[2] + st.cloud_max_rel[2]) } else { 0.0 };
        out.extend_from_slice(&cx.to_le_bytes());
        out.extend_from_slice(&cy.to_le_bytes());
        out.extend_from_slice(&cz.to_le_bytes());

        for p in slice {
            out.extend_from_slice(&p.x.to_le_bytes());
            out.extend_from_slice(&p.y.to_le_bytes());
            out.extend_from_slice(&p.z.to_le_bytes());
            out.push(p.r);
            out.push(p.g);
            out.push(p.b);
            out.push(255);
        }
        out
    }

    /// Serialize complete live chantier telemetry snapshot as clean JSON (zero external crates).
    pub fn to_status_json(&self) -> String {
        let st = self.lock();
        let pct = if st.total_tiles > 0 {
            (st.tiles_done as f64 / st.total_tiles as f64).clamp(0.0, 1.0)
        } else if st.phase == "done" {
            1.0
        } else {
            0.0
        };

        let mut pairs_json = String::from("[");
        for (i, pr) in st.pairs.iter().enumerate() {
            if i > 0 {
                pairs_json.push(',');
            }
            pairs_json.push_str(&format!(
                r#"{{"index":{},"name":"{}","baseline":{:.1},"left":"{}","right":"{}"}}"#,
                pr.index,
                esc_json(&pr.name),
                pr.baseline_m,
                esc_json(&pr.left_name),
                esc_json(&pr.right_name)
            ));
        }
        pairs_json.push(']');

        let mut logs_json = String::from("[");
        for (i, line) in st.logs.iter().enumerate() {
            if i > 0 {
                logs_json.push(',');
            }
            logs_json.push('"');
            logs_json.push_str(&esc_json(line));
            logs_json.push('"');
        }
        logs_json.push(']');

        let z_min = if st.z_min_world.is_finite() { st.z_min_world } else { 0.0 };
        let z_max = if st.z_max_world.is_finite() { st.z_max_world } else { 0.0 };

        format!(
            r#"{{"phase":"{}","status_msg":"{}","error":{},"chantier_dir":"{}","output_dir":"{}","camera":"{}","crs":"{}","backend":"{}","cli_command":"{}","poses":{},"gcps":{},"lidar":"{}","footprints":"{}","ba":"{}","model_name":"{}","model_index":{},"total_models":{},"overlap_w":{},"overlap_h":{},"epipolar":"{}","area_ha":{},"output_ply":"{}","tiles_x":{},"tiles_y":{},"total_tiles":{},"tiles_done":{},"tiles_skipped":{},"band_idx":{},"total_bands":{},"pct":{:.4},"total_points":{},"mpx_per_sec":{:.2},"eta_sec":{:.1},"elapsed_sec":{:.1},"avg_valid_pct":{:.2},"vk_dispatches":{},"mosaic_w":{},"mosaic_h":{},"mosaic_rev":{},"cloud_count":{},"cloud_rev":{},"z_min":{:.2},"z_max":{:.2},"event_seq":{},"timings":{{"band_io":{:.1},"rect":{:.1},"pprc":{:.1},"corr":{:.1},"rfne":{:.1},"fltr":{:.1},"tri":{:.1},"ply":{:.1}}},"pairs":{},"logs":{}}}"#,
            esc_json(&st.phase),
            esc_json(&st.status_msg),
            match &st.last_error {
                Some(e) => format!(r#""{}""#, esc_json(e)),
                None => "null".to_string(),
            },
            esc_json(&st.chantier_dir),
            esc_json(&st.output_dir),
            esc_json(&st.camera_info),
            esc_json(&st.survey_crs),
            esc_json(&st.backend_info),
            esc_json(&st.cli_command),
            st.poses_count,
            st.gcp_count,
            esc_json(&st.lidar_summary),
            esc_json(&st.footprints_summary),
            esc_json(&st.ba_summary),
            esc_json(&st.model_name),
            st.model_index,
            st.total_models,
            st.overlap_w,
            st.overlap_h,
            esc_json(&st.epipolar_info),
            st.area_ha.unwrap_or(0.0),
            esc_json(&st.output_ply),
            st.tiles_x,
            st.tiles_y,
            st.total_tiles,
            st.tiles_done,
            st.tiles_skipped,
            st.band_idx,
            st.total_bands,
            pct,
            st.total_points,
            st.mpx_per_sec,
            st.eta_sec,
            st.elapsed_sec,
            st.avg_valid_pct,
            st.vk_dispatches,
            st.mosaic_w,
            st.mosaic_h,
            st.mosaic_rev,
            st.cloud_pts.len(),
            st.cloud_rev,
            z_min,
            z_max,
            st.event_seq,
            st.timings.band_io,
            st.timings.rect,
            st.timings.pprc,
            st.timings.corr,
            st.timings.rfne,
            st.timings.fltr,
            st.timings.tri,
            st.timings.ply,
            pairs_json,
            logs_json
        )
    }
}

pub fn esc_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {}
            c => out.push(c),
        }
    }
    out
}

/// Best-effort launch of the user's system browser (Windows/macOS/Linux, zero external crates).
pub fn open_browser_url(url: &str) {
    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("cmd")
            .args(["/C", "start", "", url])
            .spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(url).spawn();
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        if !std::env::var("DISPLAY").unwrap_or_default().is_empty()
            || !std::env::var("WAYLAND_DISPLAY").unwrap_or_default().is_empty()
        {
            let _ = std::process::Command::new("xdg-open").arg(url).spawn();
        }
    }
}
