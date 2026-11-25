//! Unified Full-Scale 1:1 Chantier Execution Engine shared by CLI (`chantier`) and Live Web Workstation (`gui` / `--live-ui`).
//!
//! Guarantees 100% parameter & behavioral parity between CLI invocations and interactive UI runs,
//! while streaming zero-overhead live 2D/3D telemetry to `LiveChantierMonitor` when attached.

use std::io::Write;
use std::path::{Path, PathBuf};

use crate::config::StereoSettings;
use crate::core::chantier::{
    load_source_band_for_tiles, plan_full_model, rectify_full_model_tile, Chantier,
};
use crate::core::chantier_calib::pair_gcp_epipolar_dy;
use crate::core::error::{PipelineError, Result};
use crate::core::hardware::HardwareProfile;
use crate::image::geotiff::GeoTiffReader;
use crate::image::io::PointCloudMerger;
use crate::pipeline::StereoPipelineEngine;
use crate::ui::live_monitor::LiveChantierMonitor;

/// Unified configuration for a Full-Model 1:1 aerial chantier reconstruction run.
#[derive(Debug, Clone)]
pub struct ChantierRunConfig {
    pub chantier_dir: PathBuf,
    pub output_dir: Option<PathBuf>,
    pub pair_idx: Option<usize>,
    pub all_pairs: bool,
    pub roi_size: u32,
    pub express: bool,
    pub no_merge_cloud: bool,
    pub merge_voxel: f64,
    pub custom_search: bool,
    pub gcp_picks_path: Option<PathBuf>,
    pub footprints_path: Option<PathBuf>,
    pub no_gcp_refine: bool,
    pub ign_data_path: Option<PathBuf>,
    pub no_ign_lidar: bool,
    pub ndjson: bool,
    pub settings: StereoSettings,
}

impl ChantierRunConfig {
    /// Build the exact reproducible CLI command corresponding to this configuration.
    pub fn to_cli_command(&self) -> String {
        let mut cmd = format!(
            "stereo-pipeline chantier \"{}\" --full-model",
            self.chantier_dir.display()
        );
        if self.all_pairs {
            cmd.push_str(" --all-pairs");
        } else {
            cmd.push_str(&format!(" --pair {}", self.pair_idx.unwrap_or(0)));
        }
        if self.express {
            cmd.push_str(" --express");
        }
        cmd.push_str(&format!(" --roi-size {}", self.roi_size));
        if let Some(ref od) = self.output_dir {
            cmd.push_str(&format!(" --output \"{}\"", od.display()));
        }
        if self.merge_voxel > 0.0 {
            cmd.push_str(&format!(" --merge-voxel {:.3}", self.merge_voxel));
        }
        if self.no_merge_cloud {
            cmd.push_str(" --no-merge-cloud");
        }
        if self.no_ign_lidar {
            cmd.push_str(" --no-ign-lidar");
        }
        if self.no_gcp_refine {
            cmd.push_str(" --no-gcp-refine");
        }
        cmd.push_str(&format!(
            " --stereo-algorithm {} --cost-mode {} --prefilter-mode {} --subpixel-mode {} --corr-kernel {} {}",
            self.settings.stereo_algorithm.name(),
            self.settings.cost_mode as i32,
            self.settings.prefilter_mode as i32,
            self.settings.subpixel_mode as i32,
            self.settings.corr_kernel_x,
            self.settings.corr_kernel_y
        ));
        cmd.push_str(" --live-ui");
        cmd
    }
}

/// Run Full-Scale 1:1 complete stereo model reconstruction for the selected pairs of `chantier`,
/// streaming real-time 2D mosaic patches, 3D points, and telemetry to `monitor` when provided.
pub fn run_full_model_with_monitor(
    chantier: &Chantier,
    selected: &[usize],
    out_dir: &Path,
    cfg: &ChantierRunConfig,
    engine: &StereoPipelineEngine,
    monitor: Option<&LiveChantierMonitor>,
    quiet_stdout: bool,
) -> Result<()> {
    let hw = HardwareProfile::detect();
    let global_merge = selected.len() > 1 && !cfg.no_merge_cloud;
    let global_ply_path = out_dir.join("chantier_merged-PC.ply");
    let mut global_merger = if global_merge {
        Some(PointCloudMerger::new(&global_ply_path, cfg.merge_voxel)?)
    } else {
        None
    };

    let backend_str = format!(
        "{} | {} threads CPU (AVX2: {})",
        engine.vulkan_status, cfg.settings.threads, hw.avx2_supported
    );
    if let Some(mon) = monitor {
        mon.set_chantier_info(
            chantier,
            &out_dir.display().to_string(),
            &backend_str,
            &cfg.to_cli_command(),
        );
    }

    if !cfg.ndjson && !quiet_stdout {
        println!(
            "  Mode Calcul  : FULL-MODEL 1:1 {} (dalles {}x{}, 0 sortie intermédiaire)",
            if cfg.express { "EXPRESS" } else { "COMPLET" },
            cfg.roi_size,
            cfg.roi_size
        );
        println!("  Backend      : {}", backend_str);
        if global_merge {
            println!("  Nuage Global : {}", global_ply_path.display());
        }
    }

    let t_global = std::time::Instant::now();
    let mut global_tiles_done = 0usize;
    let mut global_core_pixels = 0u64;
    let mut global_points = 0usize;
    let mut was_cancelled = false;

    for (sel_pos, &idx) in selected.iter().enumerate() {
        if monitor.map_or(false, |m| m.cancel.is_cancelled()) {
            was_cancelled = true;
            break;
        }
        let pair = &chantier.pairs[idx];
        let t_model = std::time::Instant::now();
        let reader_l = GeoTiffReader::open(&pair.left_image_path)?;
        let reader_r = GeoTiffReader::open(&pair.right_image_path)?;
        let plan = plan_full_model(
            chantier,
            pair,
            reader_l.meta.width,
            reader_l.meta.height,
            reader_r.meta.width,
            reader_r.meta.height,
            cfg.roi_size,
        )?;

        let model_ply_path = out_dir.join(format!("{}-FULL-PC.ply", pair.name));
        let write_model_ply = !global_merge || !cfg.express;
        let mut model_merger = if write_model_ply {
            Some(PointCloudMerger::new(&model_ply_path, cfg.merge_voxel)?)
        } else {
            None
        };

        let epi_str = match pair_gcp_epipolar_dy(&chantier.calibration, pair, &chantier.gcp_picks) {
            Some((n_gcp, dy_rmse, dy_max)) => {
                format!("{} GCP communs | dy RMSE={:.3} px (max={:.3} px)", n_gcp, dy_rmse, dy_max)
            }
            None => "Fusiello rectifié (dy=0)".to_string(),
        };

        if let Some(mon) = monitor {
            mon.begin_model(
                sel_pos + 1,
                selected.len(),
                &pair.name,
                &plan,
                epi_str.clone(),
                model_ply_path.display().to_string(),
            );
        }

        let overlap_mpx = (plan.overlap_w as f64 * plan.overlap_h as f64) / 1_000_000.0;
        if !cfg.ndjson && !quiet_stdout {
            println!(
                "\n=== Modèle {}/{} : {} (baseline={:.1} m) ===",
                idx + 1,
                chantier.pairs.len(),
                pair.name,
                pair.baseline_m
            );
            println!(
                "  Left         : {} ({}x{} px)",
                pair.left_image_path.display(),
                plan.full_dims_l.0,
                plan.full_dims_l.1
            );
            println!(
                "  Right        : {} ({}x{} px)",
                pair.right_image_path.display(),
                plan.full_dims_r.0,
                plan.full_dims_r.1
            );
            let area_str = match plan.validated_overlap_area_m2 {
                Some(a) => format!(" | Surface validée={:.2} ha", a / 10_000.0),
                None => String::new(),
            };
            println!(
                "  Recouvrement : {}x{} px ({:.1} Mpx 1:1) | Grille {}x{} = {} dalles ({} bandes source) | Z_ref={:.2} m{}",
                plan.overlap_w,
                plan.overlap_h,
                overlap_mpx,
                plan.tiles_x,
                plan.tiles_y,
                plan.total_tiles,
                plan.bands.len(),
                plan.ground_z_ref,
                area_str
            );
            if epi_str.contains("GCP") {
                println!("  Épipolaire   : {}", epi_str);
            }
            if write_model_ply {
                println!("  Nuage Modèle : {}", model_ply_path.display());
            }
        }

        let mut band_io_ms = 0.0f64;
        let mut rect_ms = 0.0f64;
        let mut pprc_ms = 0.0f64;
        let mut corr_ms = 0.0f64;
        let mut rfne_ms = 0.0f64;
        let mut fltr_ms = 0.0f64;
        let mut tri_ms = 0.0f64;
        let mut ply_ms = 0.0f64;
        let mut total_vk_dispatches = 0usize;
        let mut tiles_done = 0usize;
        let mut tiles_skipped = 0usize;
        let mut model_core_px = 0u64;
        let mut model_pts = 0usize;
        let mut sum_valid_pct = 0.0f64;
        let mut prev_median_dx = 0i32;

        'bands: for (b_idx, band) in plan.bands.iter().enumerate() {
            if monitor.map_or(false, |m| m.cancel.is_cancelled()) {
                was_cancelled = true;
                break 'bands;
            }
            let t_band = std::time::Instant::now();
            let band_l = load_source_band_for_tiles(
                &reader_l,
                &plan.full_cam_l,
                &plan.rect_cam_l,
                band,
                true,
            )?;
            let band_r = load_source_band_for_tiles(
                &reader_r,
                &plan.full_cam_r,
                &plan.rect_cam_r,
                band,
                false,
            )?;
            let b_ms = t_band.elapsed().as_secs_f64() * 1000.0;
            band_io_ms += b_ms;

            let cache_str = match (&band_l, &band_r) {
                (Some(bl), Some(br)) => format!(
                    "bande RAM L={}x{} R={}x{} décodée en {:.0} ms",
                    bl.gray.width, bl.gray.height, br.gray.width, br.gray.height, b_ms
                ),
                _ => "streaming direct par dalle".to_string(),
            };
            if let Some(mon) = monitor {
                mon.on_band_start(b_idx, plan.bands.len(), b_ms, &cache_str);
            }
            if !cfg.ndjson && !quiet_stdout {
                println!(
                    "  -- Bande {}/{} ({} dalles, {}) --",
                    b_idx + 1,
                    plan.bands.len(),
                    band.len(),
                    cache_str
                );
                let _ = std::io::stdout().flush();
            }

            for tile in band {
                if monitor.map_or(false, |m| m.cancel.is_cancelled()) {
                    was_cancelled = true;
                    break 'bands;
                }
                if let Some(mon) = monitor {
                    mon.on_tile_start(tile.tx as usize, tile.ty as usize);
                }

                let t_tile = std::time::Instant::now();
                let t_rect = std::time::Instant::now();
                let prep_opt = rectify_full_model_tile(
                    &plan,
                    tile,
                    &reader_l,
                    &reader_r,
                    band_l.as_ref(),
                    band_r.as_ref(),
                    prev_median_dx,
                    chantier.lidar_mnt.as_ref(),
                    chantier.survey_crs,
                )?;
                let r_ms = t_rect.elapsed().as_secs_f64() * 1000.0;
                rect_ms += r_ms;

                let Some(prep) = prep_opt else {
                    tiles_skipped += 1;
                    tiles_done += 1;
                    if let Some(mon) = monitor {
                        mon.on_tile_skipped(tile.tx as usize, tile.ty as usize);
                    }
                    continue;
                };

                let mut tile_settings = cfg.settings.clone();
                if !cfg.custom_search {
                    tile_settings.search_min_x = prep.search_box.0;
                    tile_settings.search_min_y = prep.search_box.1;
                    tile_settings.search_max_x = prep.search_box.2;
                    tile_settings.search_max_y = prep.search_box.3;
                }

                let out = engine.run_express_tile(
                    &prep.left_img,
                    prep.left_rgb.as_ref(),
                    &prep.right_img,
                    &prep.left_cam,
                    &prep.right_cam,
                    &tile_settings,
                );
                pprc_ms += out.pprc_ms;
                corr_ms += out.corr_ms;
                rfne_ms += out.rfne_ms;
                fltr_ms += out.fltr_ms;
                tri_ms += out.tri_ms;
                total_vk_dispatches += out.vk_dispatches;

                if let Some(mdx) = out.median_dx {
                    if out.valid_pct >= 25.0 {
                        prev_median_dx = mdx.round() as i32;
                    }
                }

                let t_ply = std::time::Instant::now();
                let mut added_pts = 0usize;
                if let Some(ref mut m) = model_merger {
                    added_pts = m.add_point_cloud_rect(
                        &out.point_cloud,
                        tile.emit_x0,
                        tile.emit_y0,
                        tile.emit_x1,
                        tile.emit_y1,
                    )?;
                }
                if let Some(ref mut gm) = global_merger {
                    let g_pts = gm.add_point_cloud_rect(
                        &out.point_cloud,
                        tile.emit_x0,
                        tile.emit_y0,
                        tile.emit_x1,
                        tile.emit_y1,
                    )?;
                    if !write_model_ply {
                        added_pts = g_pts;
                    }
                }
                let p_ms = t_ply.elapsed().as_secs_f64() * 1000.0;
                ply_ms += p_ms;

                tiles_done += 1;
                let core_px =
                    (tile.emit_x1 - tile.emit_x0) as u64 * (tile.emit_y1 - tile.emit_y0) as u64;
                model_core_px += core_px;
                model_pts += added_pts;
                sum_valid_pct += out.valid_pct;
                let tile_s = t_tile.elapsed().as_secs_f64();

                if let Some(mon) = monitor {
                    mon.on_tile_complete(
                        tile,
                        &prep.left_img,
                        prep.left_rgb.as_ref(),
                        &out.point_cloud,
                        added_pts,
                        out.valid_pct,
                        prep.coarse_dx_px,
                        r_ms,
                        out.pprc_ms,
                        out.corr_ms,
                        out.rfne_ms,
                        out.fltr_ms,
                        out.tri_ms,
                        p_ms,
                        out.vk_dispatches,
                        tile_s,
                    );
                }

                if !cfg.ndjson && !quiet_stdout {
                    let elapsed_s = t_model.elapsed().as_secs_f64();
                    let mpx_s = (model_core_px as f64 / 1_000_000.0) / elapsed_s.max(1e-6);
                    let remaining = plan.total_tiles.saturating_sub(tiles_done);
                    let eta_s = if tiles_done > 0 {
                        (elapsed_s / tiles_done as f64) * remaining as f64
                    } else {
                        0.0
                    };
                    println!(
                        "    [{:>3}/{}] ({:>2},{:>2}) {:>4}x{:<4} dx={:+4} -> {:>7} pts ({:>5.1}%) | {:.2}s [rect={:.0} pprc={:.0} corr={:.0} fltr={:.0} tri={:.0} ply={:.0} ms] | {:.2} Mpx/s | ETA {:.0}s",
                        tiles_done,
                        plan.total_tiles,
                        tile.tx,
                        tile.ty,
                        tile.tw,
                        tile.th,
                        prep.coarse_dx_px,
                        added_pts,
                        out.valid_pct,
                        tile_s,
                        r_ms,
                        out.pprc_ms,
                        out.corr_ms,
                        out.fltr_ms,
                        out.tri_ms,
                        p_ms,
                        mpx_s,
                        eta_s
                    );
                    let _ = std::io::stdout().flush();
                }
            }
        }

        let model_stats = if let Some(m) = model_merger {
            Some(m.finish()?)
        } else {
            None
        };

        let elapsed_s = t_model.elapsed().as_secs_f64();
        let active_tiles = tiles_done.saturating_sub(tiles_skipped).max(1);
        let avg_valid = sum_valid_pct / active_tiles as f64;
        let avg_tile_ms = (elapsed_s * 1000.0) / active_tiles as f64;
        let mpx_total = model_core_px as f64 / 1_000_000.0;
        let mpx_s = mpx_total / elapsed_s.max(1e-6);
        let pts_s = model_pts as f64 / elapsed_s.max(1e-6);

        global_tiles_done += active_tiles;
        global_core_pixels += model_core_px;
        global_points += model_pts;

        if !cfg.ndjson && !quiet_stdout {
            let total_ms = (elapsed_s * 1000.0).max(1e-6);
            println!("\n=== Bilan Express Modèle {} ===", pair.name);
            println!(
                "  Dalles 1:1       : {} traitées ({} ignorées hors capteur) | {:.1} Mpx utiles",
                active_tiles, tiles_skipped, mpx_total
            );
            println!(
                "  Vitesse Pipeline : {:.2} s total | {:.0} ms/dalle | {:.2} Mpx/s | {:.2} Mpts/s",
                elapsed_s,
                avg_tile_ms,
                mpx_s,
                pts_s / 1_000_000.0
            );
            println!(
                "  Ventilation      : GeoTIFF={:.1}s ({:.1}%) | Rect={:.1}s ({:.1}%) | PPRC={:.1}s ({:.1}%) | Corr/SGM={:.1}s ({:.1}%) | Rfne={:.1}s ({:.1}%) | Fltr={:.1}s ({:.1}%) | Tri={:.1}s ({:.1}%) | PLY={:.1}s ({:.1}%)",
                band_io_ms / 1000.0, 100.0 * band_io_ms / total_ms,
                rect_ms / 1000.0, 100.0 * rect_ms / total_ms,
                pprc_ms / 1000.0, 100.0 * pprc_ms / total_ms,
                corr_ms / 1000.0, 100.0 * corr_ms / total_ms,
                rfne_ms / 1000.0, 100.0 * rfne_ms / total_ms,
                fltr_ms / 1000.0, 100.0 * fltr_ms / total_ms,
                tri_ms / 1000.0, 100.0 * tri_ms / total_ms,
                ply_ms / 1000.0, 100.0 * ply_ms / total_ms,
            );
            println!(
                "  Qualité / GPU    : densité moy. {:.2}% | {} dispatches Vulkan",
                avg_valid, total_vk_dispatches
            );
            if let Some(ref st) = model_stats {
                println!(
                    "  Nuage Dense RGB  : {} ({} pts, {:.1} MB)",
                    model_ply_path.display(),
                    st.vertex_count,
                    st.file_size_bytes as f64 / (1024.0 * 1024.0)
                );
                println!(
                    "  Emprise 3D       : X=[{:.2}..{:.2}] Y=[{:.2}..{:.2}] Z=[{:.2}..{:.2}] m",
                    st.min_x, st.max_x, st.min_y, st.max_y, st.min_z, st.max_z
                );
            }
        }
    }

    if let Some(gm) = global_merger {
        let st = gm.finish()?;
        let elapsed_s = t_global.elapsed().as_secs_f64();
        if !cfg.ndjson && !quiet_stdout {
            println!(
                "\n=== Chantier Global Full-Model Terminé ({} modèles, {} dalles, {:.1} s) ===",
                selected.len(),
                global_tiles_done,
                elapsed_s
            );
            println!(
                "  Débit Global     : {:.2} Mpx/s ({:.1} Mpx, {} pts générés)",
                (global_core_pixels as f64 / 1_000_000.0) / elapsed_s.max(1e-6),
                global_core_pixels as f64 / 1_000_000.0,
                global_points
            );
            println!(
                "  Nuage Global RGB : {} ({} pts, {:.1} MB)",
                global_ply_path.display(),
                st.vertex_count,
                st.file_size_bytes as f64 / (1024.0 * 1024.0)
            );
            println!(
                "  Emprise 3D       : X=[{:.2}..{:.2}] Y=[{:.2}..{:.2}] Z=[{:.2}..{:.2}] m",
                st.min_x, st.max_x, st.min_y, st.max_y, st.min_z, st.max_z
            );
        }
    }

    if was_cancelled {
        if let Some(mon) = monitor {
            mon.finish_cancelled();
        }
        return Err(PipelineError::Cancelled);
    }

    if let Some(mon) = monitor {
        let total_s = t_global.elapsed().as_secs_f64();
        let mpx_s = (global_core_pixels as f64 / 1_000_000.0) / total_s.max(1e-6);
        mon.finish_ok(format!(
            "Terminé en {:.1} s — {} dalles — {:.2} Mpts ({:.2} Mpx/s)",
            total_s,
            global_tiles_done,
            global_points as f64 / 1_000_000.0,
            mpx_s
        ));
    }

    Ok(())
}
