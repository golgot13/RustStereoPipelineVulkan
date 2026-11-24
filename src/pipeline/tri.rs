//! Stage 4: `stereo_tri` — Multi-threaded 3D ray-ray stereo triangulation.

use crate::camera::{AnyCamera, CameraModel};
use crate::config::StereoSettings;
use crate::image::{DisparityMap, ImageF32, PointCloud};
use crate::math::{triangulate_rays, Vec2};

/// Triangulate all valid pixels in a disparity map into a 3D PointCloud with intersection errors.
pub fn run_tri(
    left_img: &ImageF32,
    disp: &DisparityMap,
    cam_left: &AnyCamera,
    cam_right: &AnyCamera,
    s: &StereoSettings,
) -> PointCloud {
    let w = disp.width as usize;
    let h = disp.height as usize;
    let mut pc = PointCloud::new(disp.width, disp.height);
    let n_threads = s.threads.clamp(1, h.max(1));
    let rows_per = (h + n_threads - 1) / n_threads;

    let near_r = s.near_universe_radius;
    let far_r = s.far_universe_radius;
    let max_err = s.max_triangulation_error;

    std::thread::scope(|scope| {
        for (t, (((ptc, erc), inc), vac)) in pc
            .points
            .chunks_mut(rows_per * w)
            .zip(pc.errors.chunks_mut(rows_per * w))
            .zip(pc.intensities.chunks_mut(rows_per * w))
            .zip(pc.valid.chunks_mut(rows_per * w))
            .enumerate()
        {
            let y_start = t * rows_per;
            scope.spawn(move || {
                let rows = ptc.len() / w;
                for ly in 0..rows {
                    let y = y_start + ly;
                    for x in 0..w {
                        let idx = y * w + x;
                        let local = ly * w + x;
                        if disp.valid[idx] == 0 {
                            continue;
                        }
                        let p1 = Vec2::new(x as f64, y as f64);
                        let p2 = Vec2::new(x as f64 + disp.dx[idx] as f64, y as f64 + disp.dy[idx] as f64);
                        let o1 = cam_left.camera_center(p1);
                        let d1 = cam_left.pixel_to_vector(p1);
                        let o2 = cam_right.camera_center(p2);
                        let d2 = cam_right.pixel_to_vector(p2);

                        if let Some((pt, err, t1, t2)) = triangulate_rays(o1, d1, o2, d2) {
                            if t1 <= 0.0 || t2 <= 0.0 || !pt.is_finite() {
                                continue;
                            }
                            if max_err > 0.0 && err > max_err {
                                continue;
                            }
                            if near_r > 0.0 || far_r > 0.0 {
                                let r = pt.norm();
                                if near_r > 0.0 && r < near_r {
                                    continue;
                                }
                                if far_r > 0.0 && r > far_r {
                                    continue;
                                }
                            }
                            ptc[local] = pt;
                            erc[local] = err as f32;
                            inc[local] = left_img.data[idx];
                            vac[local] = 1;
                        }
                    }
                }
            });
        }
    });

    if s.filter_mode > 0 {
        filter_point_cloud_outliers(&mut pc);
    }

    pc
}

/// Fast 2-stage 3D point cloud outlier rejection (removes floating moving vehicles & periodic
/// underground stereo ghosts before PLY export and DEM gridding):
/// 1. Global 3xIQR Z-envelope pruning
/// 2. 16x16 image-grid local Z-median consistency filter (prunes points > max(3.0 * local_mad, 4.0 m) from local surface)
fn filter_point_cloud_outliers(pc: &mut PointCloud) {
    let mut zs: Vec<f64> = pc
        .valid
        .iter()
        .enumerate()
        .filter(|&(i, &v)| v != 0 && pc.points[i].is_finite())
        .map(|(i, _)| pc.points[i].z)
        .collect();
    if zs.len() < 64 {
        return;
    }
    zs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = zs.len();
    let q25 = zs[n / 4];
    let q75 = zs[(3 * n) / 4];
    let iqr = (q75 - q25).max(2.0);
    let z_lo = q25 - 3.0 * iqr;
    let z_hi = q75 + 3.0 * iqr;

    for i in 0..pc.points.len() {
        if pc.valid[i] != 0 {
            let z = pc.points[i].z;
            if !z.is_finite() || z < z_lo || z > z_hi {
                pc.valid[i] = 0;
            }
        }
    }

    // Stage 2: Local 64x64 pixel-block robust Z-median + MAD filter (eliminates floating cars & ghosts)
    let w = pc.width as usize;
    let h = pc.height as usize;
    let blk = 64usize;
    let mut block_z = Vec::with_capacity(blk * blk);
    let mut block_mad = Vec::with_capacity(blk * blk);
    for by in (0..h).step_by(blk) {
        let y1 = (by + blk).min(h);
        for bx in (0..w).step_by(blk) {
            let x1 = (bx + blk).min(w);
            block_z.clear();
            for y in by..y1 {
                for x in bx..x1 {
                    let idx = y * w + x;
                    if pc.valid[idx] != 0 {
                        block_z.push(pc.points[idx].z);
                    }
                }
            }
            if block_z.len() < 32 {
                continue;
            }
            block_z.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let med = block_z[block_z.len() / 2];
            block_mad.clear();
            for &z in &block_z {
                block_mad.push((z - med).abs());
            }
            block_mad.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mad = block_mad[block_mad.len() / 2];
            // Preserve genuine buildings/trees (up to 6*MAD or 6 m) while removing +/-15 m stereo ghosts
            let max_dev = (6.0 * mad).max(6.0);
            for y in by..y1 {
                for x in bx..x1 {
                    let idx = y * w + x;
                    if pc.valid[idx] != 0 && (pc.points[idx].z - med).abs() > max_dev {
                        pc.valid[idx] = 0;
                    }
                }
            }
        }
    }
}
