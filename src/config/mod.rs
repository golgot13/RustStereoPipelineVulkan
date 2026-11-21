//! ASP-compatible stereo pipeline configuration & `stereo.default` parser.

use std::io;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefilterMode {
    None = 0,
    SubtractedMean = 1,
    LaplacianOfGaussian = 2,
}

impl PrefilterMode {
    pub fn from_int(v: i32) -> Self {
        match v {
            0 => Self::None,
            1 => Self::SubtractedMean,
            _ => Self::LaplacianOfGaussian,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::None => "None",
            Self::SubtractedMean => "SubtractedMean",
            Self::LaplacianOfGaussian => "LoG",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StereoAlgorithm {
    AspBm = 0,
    AspSgm = 1,
    AspMgm = 2,
}

impl StereoAlgorithm {
    pub fn from_str(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "0" | "asp_bm" | "bm" => Self::AspBm,
            "1" | "asp_sgm" | "sgm" => Self::AspSgm,
            "2" | "3" | "asp_mgm" | "asp_final_mgm" | "mgm" => Self::AspMgm,
            _ => Self::AspSgm,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::AspBm => "asp_bm",
            Self::AspSgm => "asp_sgm",
            Self::AspMgm => "asp_mgm",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostMode {
    Sad = 0,
    Ssd = 1,
    Ncc = 2,
    Census = 3,
}

impl CostMode {
    pub fn from_int(v: i32) -> Self {
        match v {
            0 => Self::Sad,
            1 => Self::Ssd,
            3 => Self::Census,
            _ => Self::Ncc,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Sad => "SAD",
            Self::Ssd => "SSD",
            Self::Ncc => "NCC",
            Self::Census => "Census",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubpixelMode {
    None = 0,
    Parabola = 1,
    BayesEmAffine = 2,
    Affine = 3,
}

impl SubpixelMode {
    pub fn from_int(v: i32) -> Self {
        match v {
            0 => Self::None,
            2 => Self::BayesEmAffine,
            3 => Self::Affine,
            _ => Self::Parabola,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Parabola => "Parabola",
            Self::BayesEmAffine => "BayesEM-Affine",
            Self::Affine => "Affine",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBackendPreference {
    Auto,
    Vulkan,
    Cpu,
}

impl ComputeBackendPreference {
    pub fn from_str(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "vulkan" | "gpu" => Self::Vulkan,
            "cpu" => Self::Cpu,
            _ => Self::Auto,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Auto => "Auto (Vulkan -> CPU)",
            Self::Vulkan => "Vulkan Compute",
            Self::Cpu => "CPU Multi-thread",
        }
    }
}

/// Complete ASP stereo pipeline settings.
#[derive(Debug, Clone)]
pub struct StereoSettings {
    // Stage 0: Pre-processing (stereo_pprc)
    pub alignment_method: String,
    pub normalize_intensity: bool,
    pub prefilter_mode: PrefilterMode,
    pub prefilter_kernel_width: f32,

    // Stage 1: Correlation (stereo_corr)
    pub stereo_algorithm: StereoAlgorithm,
    pub cost_mode: CostMode,
    pub corr_kernel_x: u32,
    pub corr_kernel_y: u32,
    pub search_min_x: i32,
    pub search_max_x: i32,
    pub search_min_y: i32,
    pub search_max_y: i32,
    pub sgm_p1: f32,
    pub sgm_p2: f32,
    pub lr_threshold: f32,
    pub pyramid_levels: u32,

    // Stage 2: Subpixel Refinement (stereo_rfne)
    pub subpixel_mode: SubpixelMode,
    pub subpixel_kernel_x: u32,
    pub subpixel_kernel_y: u32,

    // Stage 3: Post Filtering (stereo_fltr)
    pub filter_mode: i32,
    pub rm_half_kernel_x: i32,
    pub rm_half_kernel_y: i32,
    pub max_mean_diff: f32,
    pub rm_min_matches: u32,
    pub rm_threshold: f32,
    pub rm_cleanup_passes: u32,
    pub median_filter_size: u32,
    pub fill_holes: bool,

    // Stage 4: Triangulation (stereo_tri) & DEM (point2dem)
    pub near_universe_radius: f64,
    pub far_universe_radius: f64,
    pub max_triangulation_error: f64,
    pub dem_spacing: f64,
    pub hillshade_azimuth_deg: f64,
    pub hillshade_elevation_deg: f64,

    // Runtime Compute Execution
    pub backend: ComputeBackendPreference,
    pub threads: usize,
}

impl Default for StereoSettings {
    fn default() -> Self {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Self {
            alignment_method: "epipolar".to_string(),
            normalize_intensity: true,
            prefilter_mode: PrefilterMode::LaplacianOfGaussian,
            prefilter_kernel_width: 1.4,

            stereo_algorithm: StereoAlgorithm::AspSgm,
            cost_mode: CostMode::Ncc,
            corr_kernel_x: 9,
            corr_kernel_y: 9,
            search_min_x: -48,
            search_max_x: 48,
            search_min_y: -2,
            search_max_y: 2,
            sgm_p1: 0.12,
            sgm_p2: 0.48,
            lr_threshold: 2.0,
            pyramid_levels: 3,

            subpixel_mode: SubpixelMode::Parabola,
            subpixel_kernel_x: 9,
            subpixel_kernel_y: 9,

            filter_mode: 1,
            rm_half_kernel_x: 3,
            rm_half_kernel_y: 3,
            max_mean_diff: 3.0,
            rm_min_matches: 45,
            rm_threshold: 2.5,
            rm_cleanup_passes: 1,
            median_filter_size: 3,
            fill_holes: true,

            near_universe_radius: 0.0,
            far_universe_radius: 0.0,
            max_triangulation_error: 0.0,
            dem_spacing: 0.0,
            hillshade_azimuth_deg: 315.0,
            hillshade_elevation_deg: 45.0,

            backend: ComputeBackendPreference::Auto,
            threads,
        }
    }
}

impl StereoSettings {
    /// Load settings from an ASP `stereo.default` configuration file.
    pub fn load_file(path: &Path) -> io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut settings = Self::default();
        settings.parse_str(&content);
        Ok(settings)
    }

    pub fn parse_str(&mut self, content: &str) {
        for raw in content.lines() {
            let line = raw.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }
            let key = parts[0].trim_start_matches('-');
            let args = &parts[1..];
            match key {
                "alignment-method" if !args.is_empty() => {
                    self.alignment_method = args[0].to_ascii_lowercase();
                }
                "force-use-entire-range" => self.normalize_intensity = true,
                "prefilter-mode" if !args.is_empty() => {
                    if let Ok(v) = args[0].parse() {
                        self.prefilter_mode = PrefilterMode::from_int(v);
                    }
                }
                "prefilter-kernel-width" if !args.is_empty() => {
                    if let Ok(v) = args[0].parse() {
                        self.prefilter_kernel_width = v;
                    }
                }
                "stereo-algorithm" if !args.is_empty() => {
                    self.stereo_algorithm = StereoAlgorithm::from_str(args[0]);
                }
                "cost-mode" if !args.is_empty() => {
                    if let Ok(v) = args[0].parse() {
                        self.cost_mode = CostMode::from_int(v);
                    }
                }
                "corr-kernel" if args.len() >= 2 => {
                    if let (Ok(x), Ok(y)) = (args[0].parse(), args[1].parse()) {
                        self.corr_kernel_x = x;
                        self.corr_kernel_y = y;
                    }
                }
                "corr-search" if args.len() >= 4 => {
                    if let (Ok(x0), Ok(y0), Ok(x1), Ok(y1)) = (
                        args[0].parse(),
                        args[1].parse(),
                        args[2].parse(),
                        args[3].parse(),
                    ) {
                        self.search_min_x = x0;
                        self.search_min_y = y0;
                        self.search_max_x = x1;
                        self.search_max_y = y1;
                    }
                }
                "subpixel-mode" if !args.is_empty() => {
                    if let Ok(v) = args[0].parse() {
                        self.subpixel_mode = SubpixelMode::from_int(v);
                    }
                }
                "subpixel-kernel" if args.len() >= 2 => {
                    if let (Ok(x), Ok(y)) = (args[0].parse(), args[1].parse()) {
                        self.subpixel_kernel_x = x;
                        self.subpixel_kernel_y = y;
                    }
                }
                "filter-mode" if !args.is_empty() => {
                    if let Ok(v) = args[0].parse() {
                        self.filter_mode = v;
                    }
                }
                "rm-half-kernel" if args.len() >= 2 => {
                    if let (Ok(x), Ok(y)) = (args[0].parse(), args[1].parse()) {
                        self.rm_half_kernel_x = x;
                        self.rm_half_kernel_y = y;
                    }
                }
                "max-mean-diff" if !args.is_empty() => {
                    if let Ok(v) = args[0].parse() {
                        self.max_mean_diff = v;
                    }
                }
                "rm-min-matches" if !args.is_empty() => {
                    if let Ok(v) = args[0].parse() {
                        self.rm_min_matches = v;
                    }
                }
                "rm-threshold" if !args.is_empty() => {
                    if let Ok(v) = args[0].parse() {
                        self.rm_threshold = v;
                    }
                }
                "rm-cleanup-passes" if !args.is_empty() => {
                    if let Ok(v) = args[0].parse() {
                        self.rm_cleanup_passes = v;
                    }
                }
                "near-universe-radius" if !args.is_empty() => {
                    if let Ok(v) = args[0].parse() {
                        self.near_universe_radius = v;
                    }
                }
                "far-universe-radius" if !args.is_empty() => {
                    if let Ok(v) = args[0].parse() {
                        self.far_universe_radius = v;
                    }
                }
                "backend" if !args.is_empty() => {
                    self.backend = ComputeBackendPreference::from_str(args[0]);
                }
                "threads" if !args.is_empty() => {
                    if let Ok(v) = args[0].parse::<usize>() {
                        if v > 0 {
                            self.threads = v;
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_stereo_default() {
        let sample = std::fs::read_to_string("stereo.default.example").unwrap();
        let mut s = StereoSettings::default();
        s.parse_str(&sample);
        assert_eq!(s.prefilter_mode, PrefilterMode::LaplacianOfGaussian);
        assert_eq!(s.stereo_algorithm, StereoAlgorithm::AspBm);
        assert_eq!(s.cost_mode, CostMode::Ncc);
        assert_eq!(s.corr_kernel_x, 21);
        assert_eq!(s.subpixel_mode, SubpixelMode::Parabola);
        assert_eq!(s.rm_min_matches, 60);
    }
}
