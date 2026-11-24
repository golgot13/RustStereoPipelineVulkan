//! Unified `JobSpec` contract and `PreflightValidator` shared identically across
//! CLI, Desktop UI (`egui` / X11), and Web UI.

use std::collections::BTreeMap;
use std::path::Path;

use crate::camera::{AnyCamera, CameraModel};
use crate::config::{
    ComputeBackendPreference, CostMode, PrefilterMode, StereoAlgorithm, StereoSettings, SubpixelMode,
};
use crate::core::crs::CrsId;
use crate::core::error::{PipelineError, Result};
use crate::core::hardware::{HardwareProfile, ProfileTier};
use crate::core::json::{parse_json, JsonValue};
use crate::image::ImageF32;
use crate::math::{triangulate_rays, Vec2};

/// Discrete pipeline stages for DAG execution, selective runs (`--from-stage`/`--to-stage`), and checkpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StageId {
    Pprc = 0,
    Corr = 1,
    Rfne = 2,
    Fltr = 3,
    Tri = 4,
    Dem = 5,
}

impl StageId {
    pub const ALL: [StageId; 6] = [
        Self::Pprc,
        Self::Corr,
        Self::Rfne,
        Self::Fltr,
        Self::Tri,
        Self::Dem,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            Self::Pprc => "pprc",
            Self::Corr => "corr",
            Self::Rfne => "rfne",
            Self::Fltr => "fltr",
            Self::Tri => "tri",
            Self::Dem => "dem",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Pprc => "Stage 0/5: stereo_pprc (Normalization & Prefiltering)",
            Self::Corr => "Stage 1/5: stereo_corr (Stereo Matching & LR Check)",
            Self::Rfne => "Stage 2/5: stereo_rfne (Subpixel Refinement)",
            Self::Fltr => "Stage 3/5: stereo_fltr (Outlier Erosion & Smoothing)",
            Self::Tri => "Stage 4/5: stereo_tri (3D Ray-Ray Triangulation)",
            Self::Dem => "Stage 5/5: point2dem (DEM Grid & Hillshade)",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "0" | "pprc" | "stereo_pprc" => Ok(Self::Pprc),
            "1" | "corr" | "stereo_corr" => Ok(Self::Corr),
            "2" | "rfne" | "stereo_rfne" => Ok(Self::Rfne),
            "3" | "fltr" | "stereo_fltr" => Ok(Self::Fltr),
            "4" | "tri" | "stereo_tri" => Ok(Self::Tri),
            "5" | "dem" | "point2dem" => Ok(Self::Dem),
            other => Err(PipelineError::Cli(format!(
                "Invalid stage '{}' (expected pprc|corr|rfne|fltr|tri|dem or 0..5)",
                other
            ))),
        }
    }
}

#[derive(Debug, Clone)]
pub enum JobInput {
    Files {
        left_image: String,
        right_image: String,
        left_camera: String,
        right_camera: String,
    },
    Synthetic {
        width: u32,
        height: u32,
    },
}

/// Complete, serializable specification of a Stereo Pipeline job (single source of truth).
#[derive(Debug, Clone)]
pub struct JobSpec {
    pub schema_version: u32,
    pub input: JobInput,
    pub output_prefix: String,
    pub crs: CrsId,
    pub from_stage: StageId,
    pub to_stage: StageId,
    pub resume: bool,
    pub tile_size: u32,
    pub profile: Option<ProfileTier>,
    pub write_outputs: bool,
    pub write_checkpoints: bool,
    pub settings: StereoSettings,
}

impl Default for JobSpec {
    fn default() -> Self {
        let hw = HardwareProfile::detect();
        let mut settings = StereoSettings::default();
        settings.threads = hw.recommended_threads;
        Self {
            schema_version: 1,
            input: JobInput::Synthetic {
                width: 192,
                height: 144,
            },
            output_prefix: "output/stereo_run".to_string(),
            crs: CrsId::Local,
            from_stage: StageId::Pprc,
            to_stage: StageId::Dem,
            resume: false,
            tile_size: hw.tile_size,
            profile: None,
            write_outputs: true,
            write_checkpoints: true,
            settings,
        }
    }
}

impl JobSpec {
    pub fn effective_hardware(&self) -> HardwareProfile {
        match self.profile {
            Some(t) => HardwareProfile::from_tier_auto(t),
            None => HardwareProfile::detect(),
        }
    }

    /// Deterministic 64-bit FNV-1a fingerprint of algorithm/geometry settings for checkpoint validation.
    pub fn settings_hash(&self) -> String {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mix = |h: &mut u64, bytes: &[u8]| {
            for &b in bytes {
                *h ^= b as u64;
                *h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        let s = &self.settings;
        mix(&mut h, s.stereo_algorithm.name().as_bytes());
        mix(&mut h, &[s.cost_mode as u8, s.prefilter_mode as u8, s.subpixel_mode as u8]);
        mix(&mut h, &s.prefilter_kernel_width.to_le_bytes());
        mix(&mut h, &s.corr_kernel_x.to_le_bytes());
        mix(&mut h, &s.corr_kernel_y.to_le_bytes());
        mix(&mut h, &s.search_min_x.to_le_bytes());
        mix(&mut h, &s.search_max_x.to_le_bytes());
        mix(&mut h, &s.search_min_y.to_le_bytes());
        mix(&mut h, &s.search_max_y.to_le_bytes());
        mix(&mut h, &s.sgm_p1.to_le_bytes());
        mix(&mut h, &s.sgm_p2.to_le_bytes());
        mix(&mut h, &s.lr_threshold.to_le_bytes());
        mix(&mut h, &s.rm_threshold.to_le_bytes());
        mix(&mut h, &s.rm_min_matches.to_le_bytes());
        mix(&mut h, self.crs.code().as_bytes());
        mix(&mut h, &self.tile_size.to_le_bytes());
        format!("{:016x}", h)
    }

    /// Generate the exact reproducible CLI invocation corresponding to this `JobSpec`.
    pub fn to_cli_command(&self) -> String {
        let s = &self.settings;
        let mut parts = vec!["stereo-pipeline".to_string()];
        match &self.input {
            JobInput::Files {
                left_image,
                right_image,
                left_camera,
                right_camera,
            } => {
                parts.push("stereo".into());
                parts.push(shell_quote(left_image));
                parts.push(shell_quote(right_image));
                parts.push(shell_quote(left_camera));
                parts.push(shell_quote(right_camera));
                parts.push(shell_quote(&self.output_prefix));
            }
            JobInput::Synthetic { width, height } => {
                parts.push("synthetic".into());
                parts.push(shell_quote(&self.output_prefix));
                parts.push(format!("--width {}", width));
                parts.push(format!("--height {}", height));
            }
        }
        parts.push(format!("--stereo-algorithm {}", s.stereo_algorithm.name()));
        parts.push(format!("--cost-mode {}", s.cost_mode as i32));
        parts.push(format!("--prefilter-mode {}", s.prefilter_mode as i32));
        parts.push(format!("--prefilter-kernel-width {:.2}", s.prefilter_kernel_width));
        parts.push(format!("--corr-kernel {} {}", s.corr_kernel_x, s.corr_kernel_y));
        parts.push(format!(
            "--corr-search {} {} {} {}",
            s.search_min_x, s.search_min_y, s.search_max_x, s.search_max_y
        ));
        parts.push(format!("--subpixel-mode {}", s.subpixel_mode as i32));
        parts.push(format!("--tile-size {}", self.tile_size));
        parts.push(format!("--crs {}", self.crs.code()));
        parts.push(format!(
            "--backend {}",
            match s.backend {
                ComputeBackendPreference::Auto => "auto",
                ComputeBackendPreference::Vulkan => "vulkan",
                ComputeBackendPreference::Cpu => "cpu",
            }
        ));
        parts.push(format!("--threads {}", s.threads));
        if self.from_stage != StageId::Pprc {
            parts.push(format!("--from-stage {}", self.from_stage.name()));
        }
        if self.to_stage != StageId::Dem {
            parts.push(format!("--to-stage {}", self.to_stage.name()));
        }
        if self.resume {
            parts.push("--resume".into());
        }
        if let Some(p) = self.profile {
            parts.push(format!("--profile {}", p.name()));
        }
        parts.join(" ")
    }

    pub fn to_json_value(&self) -> JsonValue {
        let mut root = BTreeMap::new();
        root.insert("schema_version".into(), JsonValue::Number(self.schema_version as f64));
        root.insert("output_prefix".into(), JsonValue::String(self.output_prefix.clone()));
        root.insert("crs".into(), JsonValue::String(self.crs.code()));
        root.insert("from_stage".into(), JsonValue::String(self.from_stage.name().into()));
        root.insert("to_stage".into(), JsonValue::String(self.to_stage.name().into()));
        root.insert("resume".into(), JsonValue::Bool(self.resume));
        root.insert("tile_size".into(), JsonValue::Number(self.tile_size as f64));
        root.insert("write_outputs".into(), JsonValue::Bool(self.write_outputs));
        root.insert("write_checkpoints".into(), JsonValue::Bool(self.write_checkpoints));
        if let Some(p) = self.profile {
            root.insert("profile".into(), JsonValue::String(p.name().into()));
        }

        let mut inp = BTreeMap::new();
        match &self.input {
            JobInput::Files {
                left_image,
                right_image,
                left_camera,
                right_camera,
            } => {
                inp.insert("kind".into(), JsonValue::String("files".into()));
                inp.insert("left_image".into(), JsonValue::String(left_image.clone()));
                inp.insert("right_image".into(), JsonValue::String(right_image.clone()));
                inp.insert("left_camera".into(), JsonValue::String(left_camera.clone()));
                inp.insert("right_camera".into(), JsonValue::String(right_camera.clone()));
            }
            JobInput::Synthetic { width, height } => {
                inp.insert("kind".into(), JsonValue::String("synthetic".into()));
                inp.insert("width".into(), JsonValue::Number(*width as f64));
                inp.insert("height".into(), JsonValue::Number(*height as f64));
            }
        }
        root.insert("input".into(), JsonValue::Object(inp));

        let s = &self.settings;
        let mut st = BTreeMap::new();
        st.insert("normalize_intensity".into(), JsonValue::Bool(s.normalize_intensity));
        st.insert("prefilter_mode".into(), JsonValue::Number(s.prefilter_mode as i32 as f64));
        st.insert("prefilter_kernel_width".into(), JsonValue::Number(s.prefilter_kernel_width as f64));
        st.insert("stereo_algorithm".into(), JsonValue::String(s.stereo_algorithm.name().into()));
        st.insert("cost_mode".into(), JsonValue::Number(s.cost_mode as i32 as f64));
        st.insert("corr_kernel_x".into(), JsonValue::Number(s.corr_kernel_x as f64));
        st.insert("corr_kernel_y".into(), JsonValue::Number(s.corr_kernel_y as f64));
        st.insert("search_min_x".into(), JsonValue::Number(s.search_min_x as f64));
        st.insert("search_max_x".into(), JsonValue::Number(s.search_max_x as f64));
        st.insert("search_min_y".into(), JsonValue::Number(s.search_min_y as f64));
        st.insert("search_max_y".into(), JsonValue::Number(s.search_max_y as f64));
        st.insert("sgm_p1".into(), JsonValue::Number(s.sgm_p1 as f64));
        st.insert("sgm_p2".into(), JsonValue::Number(s.sgm_p2 as f64));
        st.insert("lr_threshold".into(), JsonValue::Number(s.lr_threshold as f64));
        st.insert("subpixel_mode".into(), JsonValue::Number(s.subpixel_mode as i32 as f64));
        st.insert("subpixel_kernel_x".into(), JsonValue::Number(s.subpixel_kernel_x as f64));
        st.insert("subpixel_kernel_y".into(), JsonValue::Number(s.subpixel_kernel_y as f64));
        st.insert("filter_mode".into(), JsonValue::Number(s.filter_mode as f64));
        st.insert("rm_threshold".into(), JsonValue::Number(s.rm_threshold as f64));
        st.insert("rm_min_matches".into(), JsonValue::Number(s.rm_min_matches as f64));
        st.insert("max_mean_diff".into(), JsonValue::Number(s.max_mean_diff as f64));
        st.insert("fill_holes".into(), JsonValue::Bool(s.fill_holes));
        st.insert("dem_spacing".into(), JsonValue::Number(s.dem_spacing));
        st.insert("hillshade_azimuth_deg".into(), JsonValue::Number(s.hillshade_azimuth_deg));
        st.insert("hillshade_elevation_deg".into(), JsonValue::Number(s.hillshade_elevation_deg));
        st.insert(
            "backend".into(),
            JsonValue::String(match s.backend {
                ComputeBackendPreference::Auto => "auto",
                ComputeBackendPreference::Vulkan => "vulkan",
                ComputeBackendPreference::Cpu => "cpu",
            }.into()),
        );
        st.insert("threads".into(), JsonValue::Number(s.threads as f64));
        root.insert("settings".into(), JsonValue::Object(st));

        JsonValue::Object(root)
    }

    pub fn to_json_pretty(&self) -> String {
        self.to_json_value().to_pretty_string()
    }

    pub fn from_json_str(text: &str) -> Result<Self> {
        let v = parse_json(text)?;
        let mut spec = Self::default();
        if let Some(sv) = v.get("schema_version").and_then(|x| x.as_u64()) {
            spec.schema_version = sv as u32;
        }
        if let Some(op) = v.get("output_prefix").and_then(|x| x.as_str()) {
            spec.output_prefix = op.to_string();
        }
        if let Some(crs) = v.get("crs").and_then(|x| x.as_str()) {
            spec.crs = CrsId::parse(crs)?;
        }
        if let Some(fs) = v.get("from_stage").and_then(|x| x.as_str()) {
            spec.from_stage = StageId::parse(fs)?;
        }
        if let Some(ts) = v.get("to_stage").and_then(|x| x.as_str()) {
            spec.to_stage = StageId::parse(ts)?;
        }
        if spec.from_stage > spec.to_stage {
            return Err(PipelineError::Config(format!(
                "from_stage ({}) cannot be after to_stage ({})",
                spec.from_stage.name(),
                spec.to_stage.name()
            )));
        }
        if let Some(r) = v.get("resume").and_then(|x| x.as_bool()) {
            spec.resume = r;
        }
        if let Some(ts) = v.get("tile_size").and_then(|x| x.as_u64()) {
            spec.tile_size = (ts as u32).clamp(64, 4096);
        }
        if let Some(wo) = v.get("write_outputs").and_then(|x| x.as_bool()) {
            spec.write_outputs = wo;
        }
        if let Some(wc) = v.get("write_checkpoints").and_then(|x| x.as_bool()) {
            spec.write_checkpoints = wc;
        }
        if let Some(pr) = v.get("profile").and_then(|x| x.as_str()) {
            spec.profile = ProfileTier::from_str(pr);
        }

        if let Some(inp) = v.get("input") {
            let kind = inp.get("kind").and_then(|x| x.as_str()).unwrap_or("synthetic");
            if kind == "files" {
                let req = |k: &str| -> Result<String> {
                    inp.get(k)
                        .and_then(|x| x.as_str())
                        .map(|s| s.to_string())
                        .ok_or_else(|| PipelineError::Config(format!("Missing input.{}", k)))
                };
                spec.input = JobInput::Files {
                    left_image: req("left_image")?,
                    right_image: req("right_image")?,
                    left_camera: req("left_camera")?,
                    right_camera: req("right_camera")?,
                };
            } else {
                let w = inp.get("width").and_then(|x| x.as_u64()).unwrap_or(192) as u32;
                let h = inp.get("height").and_then(|x| x.as_u64()).unwrap_or(144) as u32;
                spec.input = JobInput::Synthetic {
                    width: w.clamp(32, 4096),
                    height: h.clamp(32, 4096),
                };
            }
        }

        if let Some(st) = v.get("settings") {
            let s = &mut spec.settings;
            if let Some(b) = st.get("normalize_intensity").and_then(|x| x.as_bool()) { s.normalize_intensity = b; }
            if let Some(n) = st.get("prefilter_mode").and_then(|x| x.as_i64()) { s.prefilter_mode = PrefilterMode::from_int(n as i32); }
            if let Some(n) = st.get("prefilter_kernel_width").and_then(|x| x.as_f64()) { s.prefilter_kernel_width = n as f32; }
            if let Some(a) = st.get("stereo_algorithm").and_then(|x| x.as_str()) { s.stereo_algorithm = StereoAlgorithm::from_str(a); }
            if let Some(n) = st.get("cost_mode").and_then(|x| x.as_i64()) { s.cost_mode = CostMode::from_int(n as i32); }
            if let Some(n) = st.get("corr_kernel_x").and_then(|x| x.as_u64()) { s.corr_kernel_x = (n as u32).clamp(3, 31); }
            if let Some(n) = st.get("corr_kernel_y").and_then(|x| x.as_u64()) { s.corr_kernel_y = (n as u32).clamp(3, 31); }
            if let Some(n) = st.get("search_min_x").and_then(|x| x.as_i64()) { s.search_min_x = n as i32; }
            if let Some(n) = st.get("search_max_x").and_then(|x| x.as_i64()) { s.search_max_x = n as i32; }
            if let Some(n) = st.get("search_min_y").and_then(|x| x.as_i64()) { s.search_min_y = n as i32; }
            if let Some(n) = st.get("search_max_y").and_then(|x| x.as_i64()) { s.search_max_y = n as i32; }
            if let Some(n) = st.get("sgm_p1").and_then(|x| x.as_f64()) { s.sgm_p1 = n as f32; }
            if let Some(n) = st.get("sgm_p2").and_then(|x| x.as_f64()) { s.sgm_p2 = n as f32; }
            if let Some(n) = st.get("lr_threshold").and_then(|x| x.as_f64()) { s.lr_threshold = n as f32; }
            if let Some(n) = st.get("subpixel_mode").and_then(|x| x.as_i64()) { s.subpixel_mode = SubpixelMode::from_int(n as i32); }
            if let Some(n) = st.get("subpixel_kernel_x").and_then(|x| x.as_u64()) { s.subpixel_kernel_x = (n as u32).clamp(3, 31); }
            if let Some(n) = st.get("subpixel_kernel_y").and_then(|x| x.as_u64()) { s.subpixel_kernel_y = (n as u32).clamp(3, 31); }
            if let Some(n) = st.get("filter_mode").and_then(|x| x.as_i64()) { s.filter_mode = n as i32; }
            if let Some(n) = st.get("rm_threshold").and_then(|x| x.as_f64()) { s.rm_threshold = n as f32; }
            if let Some(n) = st.get("rm_min_matches").and_then(|x| x.as_u64()) { s.rm_min_matches = n as u32; }
            if let Some(n) = st.get("max_mean_diff").and_then(|x| x.as_f64()) { s.max_mean_diff = n as f32; }
            if let Some(b) = st.get("fill_holes").and_then(|x| x.as_bool()) { s.fill_holes = b; }
            if let Some(n) = st.get("dem_spacing").and_then(|x| x.as_f64()) { s.dem_spacing = n; }
            if let Some(n) = st.get("hillshade_azimuth_deg").and_then(|x| x.as_f64()) { s.hillshade_azimuth_deg = n; }
            if let Some(n) = st.get("hillshade_elevation_deg").and_then(|x| x.as_f64()) { s.hillshade_elevation_deg = n; }
            if let Some(b) = st.get("backend").and_then(|x| x.as_str()) { s.backend = ComputeBackendPreference::from_str(b); }
            if let Some(n) = st.get("threads").and_then(|x| x.as_u64()) { s.threads = (n as usize).clamp(1, 128); }
        }

        Ok(spec)
    }

    pub fn load_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Self::from_json_str(&text)
    }

    pub fn save_file(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(path, self.to_json_pretty())?;
        Ok(())
    }
}

fn shell_quote(s: &str) -> String {
    if s.contains(' ') || s.contains('"') || s.contains('\\') {
        format!("\"{}\"", s.replace('"', "\\\""))
    } else {
        s.to_string()
    }
}

#[derive(Debug, Clone)]
pub struct PreflightReport {
    pub width: u32,
    pub height: u32,
    pub estimated_tile_ram_mb: u64,
    pub ram_budget_mb: u64,
    pub tile_count: usize,
    pub baseline_m: f64,
    pub warnings: Vec<String>,
}

/// Pre-execution sanity & geometry validator ("0 panic" guarantee before launching heavy compute).
pub struct PreflightValidator;

impl PreflightValidator {
    pub fn validate(
        left: &ImageF32,
        right: &ImageF32,
        cam_left: &AnyCamera,
        cam_right: &AnyCamera,
        spec: &JobSpec,
    ) -> Result<PreflightReport> {
        if left.width == 0 || left.height == 0 || right.width == 0 || right.height == 0 {
            return Err(PipelineError::Format("Input stereo images cannot be empty".into()));
        }
        if left.width != right.width || left.height != right.height {
            return Err(PipelineError::DimensionMismatch {
                expected: (left.width, left.height),
                actual: (right.width, right.height),
            });
        }
        let s = &spec.settings;
        if s.search_min_x > s.search_max_x || s.search_min_y > s.search_max_y {
            return Err(PipelineError::Config(format!(
                "Invalid disparity search range: X=[{}..{}], Y=[{}..{}]",
                s.search_min_x, s.search_max_x, s.search_min_y, s.search_max_y
            )));
        }

        let mut warnings = Vec::new();
        let (lmin, lmax, _, _) = left.stats();
        let (rmin, rmax, _, _) = right.stats();
        if (lmax - lmin).abs() < 1e-6 || (rmax - rmin).abs() < 1e-6 {
            warnings.push("Flat dynamic range detected on input imagery (correlation may fail)".into());
        }

        let center_px = Vec2::new(left.width as f64 * 0.5, left.height as f64 * 0.5);
        let o1 = cam_left.camera_center(center_px);
        let o2 = cam_right.camera_center(center_px);
        let baseline_m = (o2 - o1).norm();
        if baseline_m < 1e-6 {
            return Err(PipelineError::Geometry(
                "Zero stereo baseline between left and right cameras".into(),
            ));
        }

        // Check ray convergence at image center with mid-search disparity
        let mid_dx = ((s.search_min_x + s.search_max_x) as f64) * 0.5;
        let d1 = cam_left.pixel_to_vector(center_px);
        let d2 = cam_right.pixel_to_vector(Vec2::new(center_px.x + mid_dx, center_px.y));
        if triangulate_rays(o1, d1, o2, d2).is_none() {
            warnings.push("Center camera rays are nearly parallel at mid-search disparity".into());
        }

        let hw = spec.effective_hardware();
        let tile = spec.tile_size.max(64);
        let tiles_x = ((left.width + tile - 1) / tile) as usize;
        let tiles_y = ((left.height + tile - 1) / tile) as usize;
        let tile_count = tiles_x * tiles_y;
        let num_disp = (s.search_max_x - s.search_min_x + 1).max(1) as u32;
        let est_tw = left.width.min(tile);
        let est_th = left.height.min(tile);
        let est_mb = HardwareProfile::estimate_tile_memory_mb(est_tw, est_th, num_disp);

        if est_mb > hw.ram_budget_mb {
            return Err(PipelineError::MemoryBudget {
                required_mb: est_mb,
                budget_mb: hw.ram_budget_mb,
            });
        }

        Ok(PreflightReport {
            width: left.width,
            height: left.height,
            estimated_tile_ram_mb: est_mb,
            ram_budget_mb: hw.ram_budget_mb,
            tile_count,
            baseline_m,
            warnings,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jobspec_json_and_cli_roundtrip() {
        let mut spec = JobSpec::default();
        spec.crs = CrsId::Lambert93;
        spec.settings.stereo_algorithm = StereoAlgorithm::AspMgm;
        spec.tile_size = 512;
        spec.resume = true;
        let json = spec.to_json_pretty();
        let parsed = JobSpec::from_json_str(&json).unwrap();
        assert_eq!(parsed.crs, CrsId::Lambert93);
        assert_eq!(parsed.settings.stereo_algorithm, StereoAlgorithm::AspMgm);
        assert_eq!(parsed.tile_size, 512);
        assert!(parsed.resume);
        assert_eq!(spec.settings_hash(), parsed.settings_hash());
        let cmd = parsed.to_cli_command();
        assert!(cmd.contains("--stereo-algorithm asp_mgm"));
        assert!(cmd.contains("--crs EPSG:2154"));
        assert!(cmd.contains("--resume"));
    }
}
