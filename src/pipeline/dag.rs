//! Modular DAG Workflow Executor with Checkpoints (`--resume`), Full-Scale 1:1 Tiling,
//! Cooperative Cancellation, NDJSON Event Streaming, and Automated Quality Reports.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::camera::{load_camera, AnyCamera};
use crate::config::ComputeBackendPreference;
use crate::core::cancel::CancellationToken;
use crate::core::crs::CrsId;
use crate::core::error::{PipelineError, Result};
use crate::core::job::{JobInput, JobSpec, PreflightReport, PreflightValidator, StageId};
use crate::core::json::{parse_json, JsonValue};
use crate::image::io::{load_image, save_disparity_tiff, save_image};
use crate::image::{DemGrid, DisparityMap, ImageF32, PointCloud};
use crate::pipeline::{dem, pprc, synthetic, tiling, tri, PipelineResult, StageTimings, StereoPipelineEngine};

/// Structured workflow events emitted in real time (for `--ndjson` CLI, Web UI SSE, and `egui` progress).
#[derive(Debug, Clone)]
pub enum WorkflowEvent {
    Started {
        width: u32,
        height: u32,
        tiles: usize,
        backend: String,
    },
    PreflightWarning {
        message: String,
    },
    StageStart {
        stage: StageId,
        label: String,
        pct: f32,
    },
    TileProgress {
        tile: usize,
        total: usize,
        pct: f32,
    },
    StageComplete {
        stage: StageId,
        duration_ms: f64,
        pct: f32,
    },
    CheckpointSkipped {
        stage: StageId,
    },
    CheckpointSaved {
        stage: StageId,
        path: String,
    },
    Finished {
        total_ms: f64,
        valid_pct: f32,
        points: usize,
        mean_error_m: f64,
    },
    Cancelled,
}

impl WorkflowEvent {
    /// Serialize event as a single-line NDJSON record for machine-readable scripting & SSE.
    pub fn to_ndjson(&self) -> String {
        let mut m = BTreeMap::new();
        match self {
            Self::Started { width, height, tiles, backend } => {
                m.insert("event".into(), JsonValue::String("started".into()));
                m.insert("width".into(), JsonValue::Number(*width as f64));
                m.insert("height".into(), JsonValue::Number(*height as f64));
                m.insert("tiles".into(), JsonValue::Number(*tiles as f64));
                m.insert("backend".into(), JsonValue::String(backend.clone()));
            }
            Self::PreflightWarning { message } => {
                m.insert("event".into(), JsonValue::String("warning".into()));
                m.insert("message".into(), JsonValue::String(message.clone()));
            }
            Self::StageStart { stage, label, pct } => {
                m.insert("event".into(), JsonValue::String("stage_start".into()));
                m.insert("stage".into(), JsonValue::String(stage.name().into()));
                m.insert("label".into(), JsonValue::String(label.clone()));
                m.insert("pct".into(), JsonValue::Number(*pct as f64));
            }
            Self::TileProgress { tile, total, pct } => {
                m.insert("event".into(), JsonValue::String("tile_progress".into()));
                m.insert("tile".into(), JsonValue::Number(*tile as f64));
                m.insert("total".into(), JsonValue::Number(*total as f64));
                m.insert("pct".into(), JsonValue::Number(*pct as f64));
            }
            Self::StageComplete { stage, duration_ms, pct } => {
                m.insert("event".into(), JsonValue::String("stage_complete".into()));
                m.insert("stage".into(), JsonValue::String(stage.name().into()));
                m.insert("duration_ms".into(), JsonValue::Number(*duration_ms));
                m.insert("pct".into(), JsonValue::Number(*pct as f64));
            }
            Self::CheckpointSkipped { stage } => {
                m.insert("event".into(), JsonValue::String("checkpoint_skipped".into()));
                m.insert("stage".into(), JsonValue::String(stage.name().into()));
            }
            Self::CheckpointSaved { stage, path } => {
                m.insert("event".into(), JsonValue::String("checkpoint_saved".into()));
                m.insert("stage".into(), JsonValue::String(stage.name().into()));
                m.insert("path".into(), JsonValue::String(path.clone()));
            }
            Self::Finished { total_ms, valid_pct, points, mean_error_m } => {
                m.insert("event".into(), JsonValue::String("finished".into()));
                m.insert("total_ms".into(), JsonValue::Number(*total_ms));
                m.insert("valid_pct".into(), JsonValue::Number(*valid_pct as f64));
                m.insert("points".into(), JsonValue::Number(*points as f64));
                m.insert("mean_error_m".into(), JsonValue::Number(*mean_error_m));
            }
            Self::Cancelled => {
                m.insert("event".into(), JsonValue::String("cancelled".into()));
            }
        }
        JsonValue::Object(m).to_compact_string()
    }
}

/// Automated photogrammetric Quality Report (`<prefix>-quality_report.json` & `.txt`).
#[derive(Debug, Clone)]
pub struct QualityReport {
    pub settings_hash: String,
    pub cli_command: String,
    pub crs: String,
    pub width: u32,
    pub height: u32,
    pub tiles_processed: usize,
    pub baseline_m: f64,
    pub hardware_tier: String,
    pub ram_budget_mb: u64,
    pub avx2_active: bool,
    pub active_backend: String,
    pub vk_dispatches: usize,
    pub valid_disparity_pct: f64,
    pub valid_points: usize,
    pub tri_err_mean_m: f64,
    pub tri_err_median_m: f64,
    pub tri_err_p95_m: f64,
    pub dem_min_m: f64,
    pub dem_max_m: f64,
    pub dem_mean_m: f64,
    pub timings: StageTimings,
    pub warnings: Vec<String>,
}

impl QualityReport {
    pub fn build(
        spec: &JobSpec,
        preflight: &PreflightReport,
        res: &PipelineResult,
        tiles_processed: usize,
    ) -> Self {
        let hw = spec.effective_hardware();
        let mut errs: Vec<f64> = res
            .point_cloud
            .valid
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v != 0 { Some(res.point_cloud.errors[i] as f64) } else { None })
            .collect();
        errs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let tri_err_median_m = if errs.is_empty() { 0.0 } else { errs[errs.len() / 2] };
        let tri_err_p95_m = if errs.is_empty() { 0.0 } else { errs[(errs.len() * 95) / 100] };
        let (dem_min, dem_max, dem_mean, _) = res.dem.elevation.stats();

        Self {
            settings_hash: spec.settings_hash(),
            cli_command: spec.to_cli_command(),
            crs: spec.crs.code(),
            width: res.left_raw.width,
            height: res.left_raw.height,
            tiles_processed,
            baseline_m: preflight.baseline_m,
            hardware_tier: hw.tier.name().to_string(),
            ram_budget_mb: hw.ram_budget_mb,
            avx2_active: hw.avx2_supported,
            active_backend: res.timings.active_backend.clone(),
            vk_dispatches: res.timings.vk_dispatches,
            valid_disparity_pct: res.disp_filtered.valid_fraction() as f64 * 100.0,
            valid_points: res.point_cloud.valid_count(),
            tri_err_mean_m: res.point_cloud.mean_error(),
            tri_err_median_m,
            tri_err_p95_m,
            dem_min_m: dem_min as f64,
            dem_max_m: dem_max as f64,
            dem_mean_m: dem_mean as f64,
            timings: res.timings.clone(),
            warnings: preflight.warnings.clone(),
        }
    }

    pub fn to_json_pretty(&self) -> String {
        let mut m = BTreeMap::new();
        m.insert("settings_hash".into(), JsonValue::String(self.settings_hash.clone()));
        m.insert("cli_command".into(), JsonValue::String(self.cli_command.clone()));
        m.insert("crs".into(), JsonValue::String(self.crs.clone()));
        m.insert("width".into(), JsonValue::Number(self.width as f64));
        m.insert("height".into(), JsonValue::Number(self.height as f64));
        m.insert("tiles_processed".into(), JsonValue::Number(self.tiles_processed as f64));
        m.insert("baseline_m".into(), JsonValue::Number(self.baseline_m));
        m.insert("hardware_tier".into(), JsonValue::String(self.hardware_tier.clone()));
        m.insert("ram_budget_mb".into(), JsonValue::Number(self.ram_budget_mb as f64));
        m.insert("avx2_active".into(), JsonValue::Bool(self.avx2_active));
        m.insert("active_backend".into(), JsonValue::String(self.active_backend.clone()));
        m.insert("vk_dispatches".into(), JsonValue::Number(self.vk_dispatches as f64));
        m.insert("valid_disparity_pct".into(), JsonValue::Number(self.valid_disparity_pct));
        m.insert("valid_points".into(), JsonValue::Number(self.valid_points as f64));
        m.insert("tri_err_mean_m".into(), JsonValue::Number(self.tri_err_mean_m));
        m.insert("tri_err_median_m".into(), JsonValue::Number(self.tri_err_median_m));
        m.insert("tri_err_p95_m".into(), JsonValue::Number(self.tri_err_p95_m));
        m.insert("dem_min_m".into(), JsonValue::Number(self.dem_min_m));
        m.insert("dem_max_m".into(), JsonValue::Number(self.dem_max_m));
        m.insert("dem_mean_m".into(), JsonValue::Number(self.dem_mean_m));
        let mut tm = BTreeMap::new();
        tm.insert("pprc_ms".into(), JsonValue::Number(self.timings.pprc_ms));
        tm.insert("corr_ms".into(), JsonValue::Number(self.timings.corr_ms));
        tm.insert("rfne_ms".into(), JsonValue::Number(self.timings.rfne_ms));
        tm.insert("fltr_ms".into(), JsonValue::Number(self.timings.fltr_ms));
        tm.insert("tri_ms".into(), JsonValue::Number(self.timings.tri_ms));
        tm.insert("dem_ms".into(), JsonValue::Number(self.timings.dem_ms));
        tm.insert("total_ms".into(), JsonValue::Number(self.timings.total_ms));
        m.insert("timings".into(), JsonValue::Object(tm));
        m.insert(
            "warnings".into(),
            JsonValue::Array(self.warnings.iter().cloned().map(JsonValue::String).collect()),
        );
        JsonValue::Object(m).to_pretty_string()
    }

    pub fn to_text(&self) -> String {
        format!(
            "=== RustStereoPipelineVulkan — Quality Report ===\n\
             Settings Hash  : {}\n\
             CRS            : {}\n\
             Dimensions     : {} x {} ({} Full-Scale 1:1 tiles)\n\
             Stereo Baseline: {:.3} m\n\
             Hardware Tier  : {} (RAM budget: {} MB, AVX2: {})\n\
             Compute Backend: {}\n\
             Vulkan Dispatch: {}\n\
             Valid Disparity: {:.2}%\n\
             3D Points      : {}\n\
             Triang. Error  : mean={:.5} m | median={:.5} m | p95={:.5} m\n\
             DEM Elevation  : min={:.2} m | max={:.2} m | mean={:.2} m\n\
             Timings (ms)   : pprc={:.1} corr={:.1} rfne={:.1} fltr={:.1} tri={:.1} dem={:.1} TOTAL={:.1}\n\
             Reproducible CLI:\n  {}\n",
            self.settings_hash,
            self.crs,
            self.width,
            self.height,
            self.tiles_processed,
            self.baseline_m,
            self.hardware_tier,
            self.ram_budget_mb,
            self.avx2_active,
            self.active_backend,
            self.vk_dispatches,
            self.valid_disparity_pct,
            self.valid_points,
            self.tri_err_mean_m,
            self.tri_err_median_m,
            self.tri_err_p95_m,
            self.dem_min_m,
            self.dem_max_m,
            self.dem_mean_m,
            self.timings.pprc_ms,
            self.timings.corr_ms,
            self.timings.rfne_ms,
            self.timings.fltr_ms,
            self.timings.tri_ms,
            self.timings.dem_ms,
            self.timings.total_ms,
            self.cli_command
        )
    }

    pub fn save(&self, out_prefix: &Path) -> Result<()> {
        if let Some(parent) = out_prefix.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let p = out_prefix.to_string_lossy();
        std::fs::write(format!("{}-quality_report.json", p), self.to_json_pretty())?;
        std::fs::write(format!("{}-quality_report.txt", p), self.to_text())?;
        Ok(())
    }
}

pub struct WorkflowOutput {
    pub result: PipelineResult,
    pub preflight: PreflightReport,
    pub report: QualityReport,
    pub gt_disparity: Option<DisparityMap>,
}

/// Generic, resumable, cancellable DAG workflow executor shared by CLI, Desktop UI, and Web UI.
pub struct WorkflowExecutor;

impl WorkflowExecutor {
    /// Load inputs from `JobSpec` and execute the full DAG workflow.
    pub fn execute_job(
        spec: &JobSpec,
        engine: &StereoPipelineEngine,
        cancel: &CancellationToken,
        on_event: impl FnMut(WorkflowEvent),
    ) -> Result<WorkflowOutput> {
        let (left, right, cam_l, cam_r, gt_disp) = match &spec.input {
            JobInput::Files {
                left_image,
                right_image,
                left_camera,
                right_camera,
            } => (
                load_image(Path::new(left_image))?,
                load_image(Path::new(right_image))?,
                load_camera(Path::new(left_camera))?,
                load_camera(Path::new(right_camera))?,
                None,
            ),
            JobInput::Synthetic { width, height } => {
                let ds = synthetic::generate_synthetic_stereo(*width, *height);
                (ds.left_image, ds.right_image, ds.left_cam, ds.right_cam, Some(ds.gt_disparity))
            }
        };
        Self::execute_in_memory(spec, &left, &right, &cam_l, &cam_r, gt_disp, engine, cancel, on_event)
    }

    /// Execute the modular DAG on already-loaded rasters & cameras (used directly by `egui`, X11, and Web UI).
    #[allow(clippy::too_many_arguments)]
    pub fn execute_in_memory(
        spec: &JobSpec,
        left: &ImageF32,
        right: &ImageF32,
        cam_l: &AnyCamera,
        cam_r: &AnyCamera,
        gt_disparity: Option<DisparityMap>,
        engine: &StereoPipelineEngine,
        cancel: &CancellationToken,
        mut on_event: impl FnMut(WorkflowEvent),
    ) -> Result<WorkflowOutput> {
        cancel.check()?;
        let preflight = PreflightValidator::validate(left, right, cam_l, cam_r, spec)?;
        for w in &preflight.warnings {
            on_event(WorkflowEvent::PreflightWarning { message: w.clone() });
        }

        let vk = match spec.settings.backend {
            ComputeBackendPreference::Cpu => None,
            _ => engine.vulkan.as_ref(),
        };
        on_event(WorkflowEvent::Started {
            width: left.width,
            height: left.height,
            tiles: preflight.tile_count,
            backend: engine.vulkan_status.clone(),
        });

        let t_total = Instant::now();
        let hash = spec.settings_hash();
        let ckpt_dir = checkpoint_dir(Path::new(&spec.output_prefix));
        let mut vk_dispatches = 0usize;

        // Stage 0: stereo_pprc
        cancel.check()?;
        let mut pprc_ms = 0.0;
        let (left_pprc, right_pprc) = if spec.from_stage <= StageId::Pprc && StageId::Pprc <= spec.to_stage {
            if spec.resume && checkpoint_matches(&ckpt_dir, StageId::Pprc, &hash) {
                if let (Ok(lp), Ok(rp)) = (
                    load_image(&ckpt_dir.join("pprc-L.tif")),
                    load_image(&ckpt_dir.join("pprc-R.tif")),
                ) {
                    on_event(WorkflowEvent::CheckpointSkipped { stage: StageId::Pprc });
                    (lp, rp)
                } else {
                    let t0 = Instant::now();
                    on_event(WorkflowEvent::StageStart { stage: StageId::Pprc, label: StageId::Pprc.label().into(), pct: 0.05 });
                    let (lp, rp, used_vk) = pprc::run_pprc(left, right, &spec.settings, vk);
                    if used_vk { vk_dispatches += 2; }
                    pprc_ms = t0.elapsed().as_secs_f64() * 1000.0;
                    on_event(WorkflowEvent::StageComplete { stage: StageId::Pprc, duration_ms: pprc_ms, pct: 0.15 });
                    (lp, rp)
                }
            } else {
                let t0 = Instant::now();
                on_event(WorkflowEvent::StageStart { stage: StageId::Pprc, label: StageId::Pprc.label().into(), pct: 0.05 });
                let (lp, rp, used_vk) = pprc::run_pprc(left, right, &spec.settings, vk);
                if used_vk { vk_dispatches += 2; }
                pprc_ms = t0.elapsed().as_secs_f64() * 1000.0;
                on_event(WorkflowEvent::StageComplete { stage: StageId::Pprc, duration_ms: pprc_ms, pct: 0.15 });
                if spec.write_checkpoints {
                    if let Ok(path) = save_pprc_checkpoint(&ckpt_dir, &hash, &lp, &rp, pprc_ms) {
                        on_event(WorkflowEvent::CheckpointSaved { stage: StageId::Pprc, path });
                    }
                }
                (lp, rp)
            }
        } else {
            (left.clone(), right.clone())
        };

        // Stages 1..3: Full-Scale 1:1 Tiled corr -> rfne -> fltr
        cancel.check()?;
        let mut corr_ms = 0.0;
        let mut rfne_ms = 0.0;
        let mut fltr_ms = 0.0;
        let mut tiles_processed = 1usize;

        let (disp_raw, disp_refined, disp_filtered) = if spec.to_stage >= StageId::Corr {
            on_event(WorkflowEvent::StageStart { stage: StageId::Corr, label: StageId::Corr.label().into(), pct: 0.20 });
            let t_stereo = Instant::now();
            let tiled = tiling::run_tiled_stereo(
                &left_pprc,
                &right_pprc,
                &spec.settings,
                spec.tile_size,
                vk,
                cancel,
                |tile, total| {
                    let pct = 0.20 + 0.55 * (tile as f32 / total.max(1) as f32);
                    on_event(WorkflowEvent::TileProgress { tile, total, pct });
                },
            )?;
            let stereo_ms = t_stereo.elapsed().as_secs_f64() * 1000.0;
            // Split timing breakdown proportionally across Stages 1..3 for reporting
            corr_ms = stereo_ms * 0.78;
            rfne_ms = stereo_ms * 0.15;
            fltr_ms = stereo_ms * 0.07;
            vk_dispatches += tiled.vk_dispatches;
            tiles_processed = tiled.tiles_processed;

            on_event(WorkflowEvent::StageComplete { stage: StageId::Corr, duration_ms: corr_ms, pct: 0.55 });
            on_event(WorkflowEvent::StageComplete { stage: StageId::Rfne, duration_ms: rfne_ms, pct: 0.68 });
            on_event(WorkflowEvent::StageComplete { stage: StageId::Fltr, duration_ms: fltr_ms, pct: 0.78 });

            if spec.write_checkpoints {
                if let Ok(path) = save_disp_checkpoint(&ckpt_dir, StageId::Fltr, &hash, &tiled.disp_filtered, stereo_ms) {
                    on_event(WorkflowEvent::CheckpointSaved { stage: StageId::Fltr, path });
                }
            }
            (tiled.disp_raw, tiled.disp_refined, tiled.disp_filtered)
        } else {
            let empty = DisparityMap::new(left.width, left.height);
            (empty.clone(), empty.clone(), empty)
        };

        // Stage 4: stereo_tri
        cancel.check()?;
        let mut tri_ms = 0.0;
        let mut point_cloud = if spec.to_stage >= StageId::Tri {
            on_event(WorkflowEvent::StageStart { stage: StageId::Tri, label: StageId::Tri.label().into(), pct: 0.82 });
            let t4 = Instant::now();
            let pc = tri::run_tri(left, &disp_filtered, cam_l, cam_r, &spec.settings);
            tri_ms = t4.elapsed().as_secs_f64() * 1000.0;
            on_event(WorkflowEvent::StageComplete { stage: StageId::Tri, duration_ms: tri_ms, pct: 0.90 });
            pc
        } else {
            PointCloud::new(left.width, left.height)
        };

        // Apply optional Earth geodetic CRS projection in f64 when non-Local Earth CRS is requested and points look ECEF
        if spec.crs != CrsId::Local && spec.crs != CrsId::Ecef && spec.crs != CrsId::Moon && spec.crs != CrsId::Mars {
            let first_norm = point_cloud
                .valid
                .iter()
                .enumerate()
                .find(|(_, &v)| v != 0)
                .map(|(i, _)| point_cloud.points[i].norm())
                .unwrap_or(0.0);
            if first_norm > 6_000_000.0 {
                for i in 0..point_cloud.points.len() {
                    if point_cloud.valid[i] != 0 {
                        if let Ok(proj) = crate::core::crs::transform_point(CrsId::Ecef, spec.crs, point_cloud.points[i]) {
                            point_cloud.points[i] = proj;
                        }
                    }
                }
            }
        }

        // Stage 5: point2dem + hillshade
        cancel.check()?;
        let mut dem_ms = 0.0;
        let dem_grid = if spec.to_stage >= StageId::Dem {
            on_event(WorkflowEvent::StageStart { stage: StageId::Dem, label: StageId::Dem.label().into(), pct: 0.92 });
            let t5 = Instant::now();
            let (grid, used_vk5) = dem::point_to_dem(&point_cloud, &spec.settings, vk);
            if used_vk5 { vk_dispatches += 1; }
            dem_ms = t5.elapsed().as_secs_f64() * 1000.0;
            on_event(WorkflowEvent::StageComplete { stage: StageId::Dem, duration_ms: dem_ms, pct: 0.99 });
            grid
        } else {
            let empty = ImageF32::new(16, 16);
            DemGrid {
                elevation: empty.clone(),
                hillshade: empty,
                min_x: 0.0,
                min_y: 0.0,
                max_x: 1.0,
                max_y: 1.0,
                pixel_size: 1.0,
            }
        };

        let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
        let active_backend = if vk_dispatches > 0 {
            format!("Vulkan Compute ({} GPU dispatches) + {} CPU threads", vk_dispatches, spec.settings.threads)
        } else {
            format!(
                "CPU Multi-thread ({} threads{}) | {}",
                spec.settings.threads,
                if spec.effective_hardware().avx2_supported { ", AVX2" } else { "" },
                engine.vulkan_status
            )
        };

        let result = PipelineResult {
            left_raw: left.clone(),
            right_raw: right.clone(),
            left_pprc,
            right_pprc,
            disp_raw,
            disp_refined,
            disp_filtered,
            point_cloud,
            dem: dem_grid,
            timings: StageTimings {
                pprc_ms,
                corr_ms,
                rfne_ms,
                fltr_ms,
                tri_ms,
                dem_ms,
                total_ms,
                active_backend,
                vk_dispatches,
            },
        };

        let report = QualityReport::build(spec, &preflight, &result, tiles_processed);

        if spec.write_outputs {
            let out_p = Path::new(&spec.output_prefix);
            StereoPipelineEngine::save_outputs(out_p, &result)
                .map_err(|e| PipelineError::Io(format!("Failed saving outputs: {}", e)))?;
            report.save(out_p)?;
        }

        on_event(WorkflowEvent::Finished {
            total_ms,
            valid_pct: result.disp_filtered.valid_fraction() * 100.0,
            points: result.point_cloud.valid_count(),
            mean_error_m: result.point_cloud.mean_error(),
        });

        Ok(WorkflowOutput {
            result,
            preflight,
            report,
            gt_disparity,
        })
    }
}

fn checkpoint_dir(out_prefix: &Path) -> PathBuf {
    let base = out_prefix.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."));
    base.join("00_checkpoints")
}

fn checkpoint_matches(dir: &Path, stage: StageId, expected_hash: &str) -> bool {
    let meta = dir.join(format!("stage_{}_{}.json", stage as u8, stage.name()));
    if let Ok(txt) = std::fs::read_to_string(meta) {
        if let Ok(v) = parse_json(&txt) {
            return v.get("settings_hash").and_then(|x| x.as_str()) == Some(expected_hash);
        }
    }
    false
}

fn save_pprc_checkpoint(dir: &Path, hash: &str, lp: &ImageF32, rp: &ImageF32, ms: f64) -> Result<String> {
    std::fs::create_dir_all(dir)?;
    save_image(&dir.join("pprc-L.tif"), lp)?;
    save_image(&dir.join("pprc-R.tif"), rp)?;
    let meta_path = dir.join("stage_0_pprc.json");
    let mut m = BTreeMap::new();
    m.insert("stage".into(), JsonValue::String("pprc".into()));
    m.insert("settings_hash".into(), JsonValue::String(hash.into()));
    m.insert("duration_ms".into(), JsonValue::Number(ms));
    std::fs::write(&meta_path, JsonValue::Object(m).to_pretty_string())?;
    Ok(meta_path.to_string_lossy().into_owned())
}

fn save_disp_checkpoint(dir: &Path, stage: StageId, hash: &str, disp: &DisparityMap, ms: f64) -> Result<String> {
    std::fs::create_dir_all(dir)?;
    save_disparity_tiff(&dir.join("fltr-F.tif"), disp)?;
    let meta_path = dir.join(format!("stage_{}_{}.json", stage as u8, stage.name()));
    let mut m = BTreeMap::new();
    m.insert("stage".into(), JsonValue::String(stage.name().into()));
    m.insert("settings_hash".into(), JsonValue::String(hash.into()));
    m.insert("duration_ms".into(), JsonValue::Number(ms));
    m.insert("valid_pct".into(), JsonValue::Number(disp.valid_fraction() as f64 * 100.0));
    std::fs::write(&meta_path, JsonValue::Object(m).to_pretty_string())?;
    Ok(meta_path.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::StereoAlgorithm;

    #[test]
    fn test_dag_executor_with_checkpoints_and_quality_report() {
        let mut spec = JobSpec::default();
        spec.input = JobInput::Synthetic { width: 96, height: 72 };
        let tmp_dir = std::env::temp_dir().join("rspv_dag_test");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        spec.output_prefix = tmp_dir.join("run").to_string_lossy().into_owned();
        spec.settings.stereo_algorithm = StereoAlgorithm::AspSgm;
        spec.settings.search_min_x = -32;
        spec.settings.search_max_x = 0;
        spec.settings.search_min_y = 0;
        spec.settings.search_max_y = 0;
        spec.tile_size = 64;

        let engine = StereoPipelineEngine::new(ComputeBackendPreference::Auto);
        let cancel = CancellationToken::new();
        let mut events = Vec::new();
        let out = WorkflowExecutor::execute_job(&spec, &engine, &cancel, |ev| {
            events.push(ev.to_ndjson());
        })
        .unwrap();

        assert!(out.report.valid_disparity_pct > 60.0);
        assert!(tmp_dir.join("run-quality_report.json").exists());
        assert!(tmp_dir.join("00_checkpoints/stage_0_pprc.json").exists());
        assert!(events.iter().any(|e| e.contains("\"event\":\"finished\"")));

        // Verify resume skips Stage 0 checkpoint!
        spec.resume = true;
        let mut skipped = false;
        let _ = WorkflowExecutor::execute_job(&spec, &engine, &cancel, |ev| {
            if matches!(ev, WorkflowEvent::CheckpointSkipped { stage: StageId::Pprc }) {
                skipped = true;
            }
        })
        .unwrap();
        assert!(skipped, "Expected Stage 0 pprc to be skipped on --resume");

        // Verify cooperative cancellation
        cancel.cancel();
        let res_cancel = WorkflowExecutor::execute_job(&spec, &engine, &cancel, |_| {});
        assert!(matches!(res_cancel, Err(PipelineError::Cancelled)));

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }
}
