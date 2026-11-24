//! End-to-end Ames Stereo Pipeline orchestrator (Stages 0..4 + DEM + Hillshade)
//! with Vulkan Compute execution, multi-threaded CPU fallback, timing telemetry, and artifact export.

pub mod align;
pub mod corr;
pub mod dag;
pub mod dem;
pub mod fltr;
pub mod pprc;
pub mod rfne;
pub mod simd;
pub mod synthetic;
pub mod tiling;
pub mod tri;

use std::io;
use std::path::Path;
use std::time::Instant;

use crate::camera::AnyCamera;
use crate::config::{ComputeBackendPreference, StereoSettings};
use crate::image::io::{save_disparity_tiff, save_image, save_point_cloud_ply, save_tiff_f32, GeoTransform};
use crate::image::{DemGrid, DisparityMap, ImageF32, PointCloud};
use crate::vulkan::{VulkanCompute, VulkanDeviceInfo};

#[derive(Debug, Clone, Default)]
pub struct StageTimings {
    pub pprc_ms: f64,
    pub corr_ms: f64,
    pub rfne_ms: f64,
    pub fltr_ms: f64,
    pub tri_ms: f64,
    pub dem_ms: f64,
    pub total_ms: f64,
    pub active_backend: String,
    pub vk_dispatches: usize,
}

#[derive(Debug, Clone)]
pub struct PipelineResult {
    pub left_raw: ImageF32,
    pub right_raw: ImageF32,
    pub left_pprc: ImageF32,
    pub right_pprc: ImageF32,
    pub disp_raw: DisparityMap,
    pub disp_refined: DisparityMap,
    pub disp_filtered: DisparityMap,
    pub point_cloud: PointCloud,
    pub dem: DemGrid,
    pub timings: StageTimings,
}

/// Top-level Stereo Pipeline Engine holding an optional live Vulkan Compute context.
pub struct StereoPipelineEngine {
    pub vulkan: Option<VulkanCompute>,
    pub vulkan_status: String,
}

impl StereoPipelineEngine {
    pub fn new(pref: ComputeBackendPreference) -> Self {
        match pref {
            ComputeBackendPreference::Cpu => Self {
                vulkan: None,
                vulkan_status: "CPU Multi-thread forced by settings".to_string(),
            },
            ComputeBackendPreference::Auto | ComputeBackendPreference::Vulkan => {
                match VulkanCompute::try_new() {
                    Ok(vk) => {
                        let status = format!(
                            "Vulkan {} ({} - {}, {} MB VRAM)",
                            vk.info.api_version,
                            vk.info.device_name,
                            vk.info.device_type,
                            vk.info.memory_heap_mb
                        );
                        Self {
                            vulkan: Some(vk),
                            vulkan_status: status,
                        }
                    }
                    Err(e) => Self {
                        vulkan: None,
                        vulkan_status: format!("CPU Multi-thread fallback ({})", e),
                    },
                }
            }
        }
    }

    pub fn vulkan_info(&self) -> Option<&VulkanDeviceInfo> {
        self.vulkan.as_ref().map(|v| &v.info)
    }

    /// Execute all 5 ASP stereo stages + DEM & hillshade generation.
    pub fn run_full(
        &self,
        left: &ImageF32,
        right: &ImageF32,
        cam_left: &AnyCamera,
        cam_right: &AnyCamera,
        settings: &StereoSettings,
        mut progress: impl FnMut(&str, f32),
    ) -> PipelineResult {
        let vk = match settings.backend {
            ComputeBackendPreference::Cpu => None,
            _ => self.vulkan.as_ref(),
        };
        let t_all = Instant::now();
        let mut vk_dispatches = 0usize;

        progress("Stage 0/5: stereo_pprc (Normalization & Prefiltering)", 0.05);
        let t0 = Instant::now();
        let (left_pprc, right_pprc, used_vk0) = pprc::run_pprc(left, right, settings, vk);
        let pprc_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if used_vk0 { vk_dispatches += 2; }

        progress("Stage 1/5: stereo_corr (Stereo Matching & LR Check)", 0.22);
        let t1 = Instant::now();
        let (disp_raw, used_vk1) = corr::run_corr_with_guide(&left_pprc, &right_pprc, Some(left), settings, vk);
        let corr_ms = t1.elapsed().as_secs_f64() * 1000.0;
        if used_vk1 { vk_dispatches += 2; }

        progress("Stage 2/5: stereo_rfne (Subpixel Refinement)", 0.55);
        let t2 = Instant::now();
        let disp_refined = rfne::run_rfne(&left_pprc, &right_pprc, &disp_raw, settings);
        let rfne_ms = t2.elapsed().as_secs_f64() * 1000.0;

        progress("Stage 3/5: stereo_fltr (Outlier Erosion & Smoothing)", 0.72);
        let t3 = Instant::now();
        let disp_filtered = fltr::run_fltr(&disp_refined, settings);
        let fltr_ms = t3.elapsed().as_secs_f64() * 1000.0;

        progress("Stage 4/5: stereo_tri (3D Ray-Ray Triangulation)", 0.84);
        let t4 = Instant::now();
        let point_cloud = tri::run_tri(left, &disp_filtered, cam_left, cam_right, settings);
        let tri_ms = t4.elapsed().as_secs_f64() * 1000.0;

        progress("Stage 5/5: point2dem (DEM Grid & Hillshade)", 0.93);
        let t5 = Instant::now();
        let (dem_grid, used_vk5) = dem::point_to_dem(&point_cloud, settings, vk);
        let dem_ms = t5.elapsed().as_secs_f64() * 1000.0;
        if used_vk5 { vk_dispatches += 1; }

        let total_ms = t_all.elapsed().as_secs_f64() * 1000.0;
        progress("Pipeline Complete", 1.0);

        let active_backend = if vk_dispatches > 0 {
            format!("Vulkan Compute ({} GPU dispatches) + {} CPU threads", vk_dispatches, settings.threads)
        } else {
            format!("CPU Multi-thread ({} threads) | {}", settings.threads, self.vulkan_status)
        };

        PipelineResult {
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
        }
    }

    /// Save standard ASP output products (`<prefix>-L.tif`, `-R.tif`, `-D.tif`, `-F.tif`, `-PC.ply`, `-DEM.tif`, `-hillshade.tif`).
    pub fn save_outputs(prefix: &Path, res: &PipelineResult) -> io::Result<()> {
        Self::save_outputs_opts(prefix, res, true, true)
    }

    /// Save output products with fine-grained control over PLY and intermediate raster writing (`--fast-io`).
    pub fn save_outputs_opts(
        prefix: &Path,
        res: &PipelineResult,
        write_ply: bool,
        write_intermediates: bool,
    ) -> io::Result<()> {
        if let Some(parent) = prefix.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let p = prefix.to_string_lossy();
        if write_intermediates {
            save_image(Path::new(&format!("{}-L.tif", p)), &res.left_pprc)?;
            save_image(Path::new(&format!("{}-R.tif", p)), &res.right_pprc)?;
            save_disparity_tiff(Path::new(&format!("{}-D.tif", p)), &res.disp_raw)?;
            save_disparity_tiff(Path::new(&format!("{}-RD.tif", p)), &res.disp_refined)?;
            save_disparity_tiff(Path::new(&format!("{}-F.tif", p)), &res.disp_filtered)?;
        }
        if write_ply {
            save_point_cloud_ply(Path::new(&format!("{}-PC.ply", p)), &res.point_cloud)?;
        }
        save_tiff_f32(
            Path::new(&format!("{}-DEM.tif", p)),
            &res.dem.elevation,
            Some(GeoTransform {
                origin_x: res.dem.min_x,
                origin_y: res.dem.max_y,
                pixel_size_x: res.dem.pixel_size,
                pixel_size_y: res.dem.pixel_size,
            }),
        )?;
        save_image(Path::new(&format!("{}-hillshade.tif", p)), &res.dem.hillshade)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CostMode, StereoAlgorithm, SubpixelMode};

    #[test]
    fn test_end_to_end_synthetic_stereo_pipeline() {
        let ds = synthetic::generate_synthetic_stereo(96, 72);
        let mut settings = StereoSettings::default();
        settings.stereo_algorithm = StereoAlgorithm::AspSgm;
        settings.cost_mode = CostMode::Ncc;
        settings.subpixel_mode = SubpixelMode::Parabola;
        settings.search_min_x = -32;
        settings.search_max_x = 0;
        settings.search_min_y = 0;
        settings.search_max_y = 0;
        settings.corr_kernel_x = 7;
        settings.corr_kernel_y = 7;

        let engine = StereoPipelineEngine::new(ComputeBackendPreference::Auto);
        let res = engine.run_full(
            &ds.left_image,
            &ds.right_image,
            &ds.left_cam,
            &ds.right_cam,
            &settings,
            |_, _| {},
        );

        let valid_frac = res.disp_filtered.valid_fraction();
        assert!(valid_frac > 0.60, "Expected >60% valid disparity, got {:.1}%", valid_frac * 100.0);
        assert!(res.point_cloud.valid_count() > 3000, "Too few triangulated 3D points");
        assert!(res.point_cloud.mean_error() < 0.25, "High triangulation error: {}", res.point_cloud.mean_error());
    }
}
