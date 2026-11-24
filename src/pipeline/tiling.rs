//! Full-Scale 1:1 Sliding-Window `TileScheduler` with overlap halo padding.
//! Inspired by AEROPHOTOGRAM's Full Scale 1:1 tiling discipline: never decimates imagery,
//! processes arbitrary large rasters within bounded RAM, and stitches tiles seamlessly.

use crate::config::StereoSettings;
use crate::core::cancel::CancellationToken;
use crate::core::error::Result;
use crate::image::{DisparityMap, ImageF32};
use crate::pipeline::{corr, fltr, rfne};
use crate::vulkan::VulkanCompute;

/// Rectangular core tile region plus padded halo window inside the full-resolution image.
#[derive(Debug, Clone, Copy)]
pub struct TileSpec {
    pub index: usize,
    pub total: usize,
    /// Core output region in full image `[core_x..core_x+core_w, core_y..core_y+core_h]`
    pub core_x: u32,
    pub core_y: u32,
    pub core_w: u32,
    pub core_h: u32,
    /// Padded ROI region extracted from full image `[pad_x..pad_x+pad_w, pad_y..pad_y+pad_h]`
    pub pad_x: u32,
    pub pad_y: u32,
    pub pad_w: u32,
    pub pad_h: u32,
}

/// Plan 1:1 tiles covering `(width, height)` with automatic overlap halo for correlation & filtering.
pub fn plan_tiles(width: u32, height: u32, tile_size: u32, halo_x: u32, halo_y: u32) -> Vec<TileSpec> {
    let ts = tile_size.max(64);
    let nx = (width + ts - 1) / ts;
    let ny = (height + ts - 1) / ts;
    let total = (nx * ny) as usize;
    let mut tiles = Vec::with_capacity(total);
    let mut idx = 0usize;

    for ty in 0..ny {
        let core_y = ty * ts;
        let core_h = (height - core_y).min(ts);
        let pad_y0 = core_y.saturating_sub(halo_y);
        let pad_y1 = (core_y + core_h + halo_y).min(height);
        for tx in 0..nx {
            let core_x = tx * ts;
            let core_w = (width - core_x).min(ts);
            let pad_x0 = core_x.saturating_sub(halo_x);
            let pad_x1 = (core_x + core_w + halo_x).min(width);
            tiles.push(TileSpec {
                index: idx,
                total,
                core_x,
                core_y,
                core_w,
                core_h,
                pad_x: pad_x0,
                pad_y: pad_y0,
                pad_w: pad_x1 - pad_x0,
                pad_h: pad_y1 - pad_y0,
            });
            idx += 1;
        }
    }
    tiles
}

fn crop_image(img: &ImageF32, x0: u32, y0: u32, w: u32, h: u32) -> ImageF32 {
    let mut out = ImageF32::new(w, h);
    out.nodata = img.nodata;
    let src_w = img.width as usize;
    let dst_w = w as usize;
    for y in 0..h as usize {
        let src_start = (y0 as usize + y) * src_w + x0 as usize;
        let dst_start = y * dst_w;
        out.data[dst_start..dst_start + dst_w]
            .copy_from_slice(&img.data[src_start..src_start + dst_w]);
    }
    out
}

fn stitch_disparity(dst: &mut DisparityMap, src: &DisparityMap, tile: &TileSpec) {
    let ox = tile.core_x - tile.pad_x;
    let oy = tile.core_y - tile.pad_y;
    for y in 0..tile.core_h {
        let sy = (oy + y) as usize * src.width as usize + ox as usize;
        let dy = (tile.core_y + y) as usize * dst.width as usize + tile.core_x as usize;
        let n = tile.core_w as usize;
        dst.dx[dy..dy + n].copy_from_slice(&src.dx[sy..sy + n]);
        dst.dy[dy..dy + n].copy_from_slice(&src.dy[sy..sy + n]);
        dst.valid[dy..dy + n].copy_from_slice(&src.valid[sy..sy + n]);
    }
}

/// Output of tiled Full-Scale 1:1 correlation + refinement + filtering.
pub struct TiledDisparityResult {
    pub disp_raw: DisparityMap,
    pub disp_refined: DisparityMap,
    pub disp_filtered: DisparityMap,
    pub vk_dispatches: usize,
    pub tiles_processed: usize,
}

/// Execute stereo stages 1..3 (corr -> rfne -> fltr) in Full-Scale 1:1 sliding tiles with halo overlap
/// and cooperative cancellation between tiles.
pub fn run_tiled_stereo(
    left_pprc: &ImageF32,
    right_pprc: &ImageF32,
    settings: &StereoSettings,
    tile_size: u32,
    vk: Option<&VulkanCompute>,
    cancel: &CancellationToken,
    mut on_tile: impl FnMut(usize, usize),
) -> Result<TiledDisparityResult> {
    let w = left_pprc.width;
    let h = left_pprc.height;
    let max_dx = settings.search_min_x.abs().max(settings.search_max_x.abs()) as u32;
    let max_dy = settings.search_min_y.abs().max(settings.search_max_y.abs()) as u32;
    let halo_x = max_dx + (settings.corr_kernel_x / 2) + settings.rm_half_kernel_x.max(1) as u32 + 4;
    let halo_y = max_dy + (settings.corr_kernel_y / 2) + settings.rm_half_kernel_y.max(1) as u32 + 4;

    let tiles = plan_tiles(w, h, tile_size, halo_x, halo_y);
    let mut disp_raw = DisparityMap::new(w, h);
    let mut disp_refined = DisparityMap::new(w, h);
    let mut disp_filtered = DisparityMap::new(w, h);
    let mut vk_dispatches = 0usize;

    for tile in &tiles {
        cancel.check()?;
        on_tile(tile.index + 1, tile.total);

        if tile.total == 1 {
            let (raw, used_vk) = corr::run_corr(left_pprc, right_pprc, settings, vk);
            if used_vk {
                vk_dispatches += 2;
            }
            cancel.check()?;
            let refined = rfne::run_rfne(left_pprc, right_pprc, &raw, settings);
            cancel.check()?;
            let filtered = fltr::run_fltr(&refined, settings);
            return Ok(TiledDisparityResult {
                disp_raw: raw,
                disp_refined: refined,
                disp_filtered: filtered,
                vk_dispatches,
                tiles_processed: 1,
            });
        }

        let sub_l = crop_image(left_pprc, tile.pad_x, tile.pad_y, tile.pad_w, tile.pad_h);
        let sub_r = crop_image(right_pprc, tile.pad_x, tile.pad_y, tile.pad_w, tile.pad_h);

        let (sub_raw, used_vk) = corr::run_corr(&sub_l, &sub_r, settings, vk);
        if used_vk {
            vk_dispatches += 2;
        }
        cancel.check()?;
        let sub_ref = rfne::run_rfne(&sub_l, &sub_r, &sub_raw, settings);
        cancel.check()?;
        let sub_flt = fltr::run_fltr(&sub_ref, settings);

        stitch_disparity(&mut disp_raw, &sub_raw, tile);
        stitch_disparity(&mut disp_refined, &sub_ref, tile);
        stitch_disparity(&mut disp_filtered, &sub_flt, tile);
    }

    Ok(TiledDisparityResult {
        disp_raw,
        disp_refined,
        disp_filtered,
        vk_dispatches,
        tiles_processed: tiles.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CostMode, StereoAlgorithm, SubpixelMode};
    use crate::pipeline::{pprc, synthetic};

    #[test]
    fn test_tiled_full_scale_matches_untiled() {
        let ds = synthetic::generate_synthetic_stereo(128, 96);
        let mut s = StereoSettings::default();
        s.stereo_algorithm = StereoAlgorithm::AspSgm;
        s.cost_mode = CostMode::Ncc;
        s.subpixel_mode = SubpixelMode::Parabola;
        s.search_min_x = -32;
        s.search_max_x = 0;
        s.search_min_y = 0;
        s.search_max_y = 0;
        s.corr_kernel_x = 7;
        s.corr_kernel_y = 7;

        let (lp, rp, _) = pprc::run_pprc(&ds.left_image, &ds.right_image, &s, None);
        let cancel = CancellationToken::new();
        let tiled = run_tiled_stereo(&lp, &rp, &s, 64, None, &cancel, |_, _| {}).unwrap();
        assert_eq!(tiled.tiles_processed, 4);

        let mut abs_err = 0.0f64;
        let mut cnt = 0usize;
        for i in 0..tiled.disp_filtered.dx.len() {
            if tiled.disp_filtered.valid[i] != 0 && ds.gt_disparity.valid[i] != 0 {
                abs_err += (tiled.disp_filtered.dx[i] - ds.gt_disparity.dx[i]).abs() as f64;
                cnt += 1;
            }
        }
        assert!(cnt > 5000);
        let mae = abs_err / cnt as f64;
        assert!(mae < 0.45, "Tiled 1:1 MAE too high: {:.3} px", mae);
    }
}
