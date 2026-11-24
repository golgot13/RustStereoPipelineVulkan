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

/// Robust 4-stage `f64` 3D point cloud outlier rejection:
/// 1. Robust global IRLS plane detrending (`Z_plane(X,Y) = aX + bY + c`) so sloped terrain does not hide ghosts
/// 2. Coarse multi-cell detrended surface median with neighbor propagation (never skips sparse `< 32` blocks!)
/// 3. Local 3D vertical-coherence filter (`9x9` neighborhood) removing vertical hanging point trails & mid-air splatter
/// 4. 3D Z-connected-component BFS removing detached floating moving vehicles & underground stereo ghosts
fn filter_point_cloud_outliers(pc: &mut PointCloud) {
    let w = pc.width as usize;
    let h = pc.height as usize;
    let n = w * h;
    if n == 0 || pc.points.len() != n {
        return;
    }

    let valid_indices: Vec<usize> = (0..n)
        .filter(|&i| pc.valid[i] != 0 && pc.points[i].is_finite())
        .collect();
    let total_valid = valid_indices.len();
    if total_valid < 64 {
        return;
    }

    // --- Stage 1: Robust global IRLS plane fit Z(X,Y) in centered f64 coordinates ---
    let step = (total_valid / 4096).max(1);
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut sz = 0.0f64;
    let mut scnt = 0usize;
    for &idx in valid_indices.iter().step_by(step) {
        let p = pc.points[idx];
        sx += p.x;
        sy += p.y;
        sz += p.z;
        scnt += 1;
    }
    let cx = sx / scnt as f64;
    let cy = sy / scnt as f64;
    let cz = sz / scnt as f64;

    let mut pa = 0.0f64;
    let mut pb = 0.0f64;
    let mut pc_off = 0.0f64;
    let mut scale = 2.0f64;

    for iter in 0..6 {
        let mut m00 = 1e-9;
        let mut m01 = 0.0;
        let mut m02 = 0.0;
        let mut m11 = 1e-9;
        let mut m12 = 0.0;
        let mut m22 = 1e-9;
        let mut b0 = 0.0;
        let mut b1 = 0.0;
        let mut b2 = 0.0;

        for &idx in valid_indices.iter().step_by(step) {
            let p = pc.points[idx];
            let dx = p.x - cx;
            let dy = p.y - cy;
            let dz = p.z - cz;
            let wt = if iter == 0 {
                1.0
            } else {
                let res = (dz - (pa * dx + pb * dy + pc_off)).abs();
                let u = res / scale.max(0.5);
                if u <= 1.0 { 1.0 } else { 1.0 / (u * u) }
            };
            m00 += wt * dx * dx;
            m01 += wt * dx * dy;
            m02 += wt * dx;
            m11 += wt * dy * dy;
            m12 += wt * dy;
            m22 += wt;
            b0 += wt * dx * dz;
            b1 += wt * dy * dz;
            b2 += wt * dz;
        }

        if let Some((na, nb, nc)) = solve_sym_3x3(m00, m01, m02, m11, m12, m22, b0, b1, b2) {
            pa = na;
            pb = nb;
            pc_off = nc;
        }

        let mut res_sample: Vec<f64> = valid_indices
            .iter()
            .step_by(step)
            .map(|&idx| {
                let p = pc.points[idx];
                ((p.z - cz) - (pa * (p.x - cx) + pb * (p.y - cy) + pc_off)).abs()
            })
            .collect();
        res_sample.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        scale = (res_sample[res_sample.len() / 2] * 1.4826).max(0.5);
    }

    // Compute detrended height residuals r_i = Z_i - Z_plane(X_i, Y_i)
    let mut resid = vec![0.0f64; n];
    let mut abs_res = Vec::with_capacity(total_valid);
    for &idx in &valid_indices {
        let p = pc.points[idx];
        let r = (p.z - cz) - (pa * (p.x - cx) + pb * (p.y - cy) + pc_off);
        resid[idx] = r;
        abs_res.push(r.abs());
    }
    abs_res.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let global_mad = (abs_res[abs_res.len() / 2] * 1.4826).max(0.4);

    // Prune extreme gross sky/underground outliers relative to the sloped trend plane
    let gross_lo = -(6.0 * global_mad).max(6.0);
    let gross_hi = (12.0 * global_mad).max(30.0);
    for &idx in &valid_indices {
        let r = resid[idx];
        if !r.is_finite() || r < gross_lo || r > gross_hi {
            pc.valid[idx] = 0;
        }
    }

    // --- Stage 2: Coarse multi-cell detrended surface median with sparse-cell propagation ---
    let blk = (w.max(h) / 4).clamp(64, 256);
    let gx_n = (w + blk - 1) / blk;
    let gy_n = (h + blk - 1) / blk;
    let mut cell_med = vec![0.0f64; gx_n * gy_n];
    let mut cell_mad = vec![global_mad; gx_n * gy_n];
    let mut cell_pop = vec![false; gx_n * gy_n];
    let mut vals = Vec::with_capacity(blk * blk);
    let mut devs = Vec::with_capacity(blk * blk);

    for cy_i in 0..gy_n {
        let y0 = cy_i * blk;
        let y1 = (y0 + blk).min(h);
        for cx_i in 0..gx_n {
            let x0 = cx_i * blk;
            let x1 = (x0 + blk).min(w);
            vals.clear();
            for y in y0..y1 {
                for x in x0..x1 {
                    let idx = y * w + x;
                    if pc.valid[idx] != 0 {
                        vals.push(resid[idx]);
                    }
                }
            }
            if vals.len() >= 64 {
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let med = vals[vals.len() / 2];
                devs.clear();
                for &v in &vals {
                    devs.push((v - med).abs());
                }
                devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let mad = (devs[devs.len() / 2] * 1.4826).max(0.35);
                let cidx = cy_i * gx_n + cx_i;
                cell_med[cidx] = med;
                cell_mad[cidx] = mad;
                cell_pop[cidx] = true;
            }
        }
    }

    // Fill sparse cells from nearest populated cells (so sparse outlier regions are never skipped!)
    for cy_i in 0..gy_n {
        for cx_i in 0..gx_n {
            let cidx = cy_i * gx_n + cx_i;
            if cell_pop[cidx] {
                continue;
            }
            let mut sum_m = 0.0;
            let mut sum_d = 0.0;
            let mut sum_w = 0.0;
            for ny in 0..gy_n {
                for nx in 0..gx_n {
                    let nidx = ny * gx_n + nx;
                    if cell_pop[nidx] {
                        let d2 = ((nx as f64 - cx_i as f64).powi(2) + (ny as f64 - cy_i as f64).powi(2)).max(1.0);
                        let wt = 1.0 / d2;
                        sum_m += cell_med[nidx] * wt;
                        sum_d += cell_mad[nidx] * wt;
                        sum_w += wt;
                    }
                }
            }
            if sum_w > 0.0 {
                cell_med[cidx] = sum_m / sum_w;
                cell_mad[cidx] = sum_d / sum_w;
            }
        }
    }

    let sample_surface = |x: usize, y: usize| -> (f64, f64) {
        let fx = ((x as f64 + 0.5) / blk as f64 - 0.5).clamp(0.0, (gx_n - 1) as f64);
        let fy = ((y as f64 + 0.5) / blk as f64 - 0.5).clamp(0.0, (gy_n - 1) as f64);
        let x0 = fx.floor() as usize;
        let y0 = fy.floor() as usize;
        let x1 = (x0 + 1).min(gx_n - 1);
        let y1 = (y0 + 1).min(gy_n - 1);
        let tx = fx - x0 as f64;
        let ty = fy - y0 as f64;
        let w00 = (1.0 - tx) * (1.0 - ty);
        let w10 = tx * (1.0 - ty);
        let w01 = (1.0 - tx) * ty;
        let w11 = tx * ty;
        let m = cell_med[y0 * gx_n + x0] * w00
            + cell_med[y0 * gx_n + x1] * w10
            + cell_med[y1 * gx_n + x0] * w01
            + cell_med[y1 * gx_n + x1] * w11;
        let d = cell_mad[y0 * gx_n + x0] * w00
            + cell_mad[y0 * gx_n + x1] * w10
            + cell_mad[y1 * gx_n + x0] * w01
            + cell_mad[y1 * gx_n + x1] * w11;
        (m, d)
    };

    // Estimate characteristic pixel-to-pixel Z step for adaptive 3D vertical coherence
    let mut adj_steps = Vec::with_capacity(2048);
    let stride_y = (h / 64).max(1);
    let stride_x = (w / 64).max(1);
    for y in (0..h.saturating_sub(1)).step_by(stride_y) {
        for x in (0..w.saturating_sub(1)).step_by(stride_x) {
            let idx = y * w + x;
            if pc.valid[idx] != 0 && pc.valid[idx + 1] != 0 {
                adj_steps.push((pc.points[idx].z - pc.points[idx + 1].z).abs());
            }
        }
    }
    let z_tol = if adj_steps.len() >= 16 {
        adj_steps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        (adj_steps[adj_steps.len() * 3 / 4] * 4.0).clamp(0.75, 2.5)
    } else {
        1.0
    };

    // --- Stage 3: Local 3D vertical-coherence filter (removes vertical trails & mid-air splatter) ---
    let mut keep_coh = vec![1u8; n];
    for y in 0..h {
        let y0 = y.saturating_sub(4);
        let y1 = (y + 4).min(h - 1);
        for x in 0..w {
            let idx = y * w + x;
            if pc.valid[idx] == 0 {
                continue;
            }
            let z0 = pc.points[idx].z;
            let x0 = x.saturating_sub(4);
            let x1 = (x + 4).min(w - 1);
            let mut coh_cnt = 0u32;
            for ny in (y0..=y1).step_by(2) {
                let row = ny * w;
                for nx in (x0..=x1).step_by(2) {
                    let nidx = row + nx;
                    if nidx != idx && pc.valid[nidx] != 0 && (pc.points[nidx].z - z0).abs() <= z_tol {
                        coh_cnt += 1;
                    }
                }
            }
            if coh_cnt < 4 {
                keep_coh[idx] = 0;
            }
        }
    }
    for i in 0..n {
        if keep_coh[i] == 0 {
            pc.valid[i] = 0;
        }
    }

    // --- Stage 4: 3D Z-connected-component BFS (removes detached floating vehicles & underground ghosts) ---
    let remaining_valid = pc.valid.iter().filter(|&&v| v != 0).count();
    if remaining_valid < 64 {
        return;
    }
    let mut visited = vec![0u8; n];
    let mut stack: Vec<usize> = Vec::with_capacity(512);
    let mut comp: Vec<usize> = Vec::with_capacity(512);

    for start in 0..n {
        if pc.valid[start] == 0 || visited[start] != 0 {
            continue;
        }
        visited[start] = 1;
        stack.clear();
        comp.clear();
        stack.push(start);

        while let Some(idx) = stack.pop() {
            comp.push(idx);
            let x = idx % w;
            let y = idx / w;
            let zc = pc.points[idx].z;

            let mut try_push = |nx: usize, ny: usize| {
                let nidx = ny * w + nx;
                if visited[nidx] == 0
                    && pc.valid[nidx] != 0
                    && (pc.points[nidx].z - zc).abs() <= z_tol
                {
                    visited[nidx] = 1;
                    stack.push(nidx);
                }
            };
            if x > 0 { try_push(x - 1, y); }
            if x + 1 < w { try_push(x + 1, y); }
            if y > 0 { try_push(x, y - 1); }
            if y + 1 < h { try_push(x, y + 1); }
        }

        let cnt = comp.len();
        if cnt * 4 >= remaining_valid {
            // Main terrain component: never prune as a detached ghost
            continue;
        }

        let mut sum_dz = 0.0f64;
        let mut sum_mad = 0.0f64;
        for &idx in &comp {
            let x = idx % w;
            let y = idx / w;
            let (sm, sd) = sample_surface(x, y);
            sum_dz += resid[idx] - sm;
            sum_mad += sd;
        }
        let mean_dz = sum_dz / cnt as f64;
        let mean_mad = sum_mad / cnt as f64;

        let is_underground_ghost =
            mean_dz < -(3.5 * mean_mad).max(2.0) && cnt * 5 < remaining_valid;
        let is_floating_ghost =
            mean_dz > (5.0 * mean_mad).max(4.0) && cnt < 20_000 && cnt * 15 < remaining_valid;
        let is_tiny_detached_outlier =
            mean_dz.abs() > (3.5 * mean_mad).max(2.5) && cnt < 512;

        if is_underground_ghost || is_floating_ghost || is_tiny_detached_outlier {
            for &idx in &comp {
                pc.valid[idx] = 0;
            }
        }
    }
}

/// Solve symmetric 3x3 linear system `M * [a, b, c]^T = rhs` in `f64` via Cramer's rule.
#[allow(clippy::too_many_arguments)]
fn solve_sym_3x3(
    m00: f64,
    m01: f64,
    m02: f64,
    m11: f64,
    m12: f64,
    m22: f64,
    b0: f64,
    b1: f64,
    b2: f64,
) -> Option<(f64, f64, f64)> {
    let det = m00 * (m11 * m22 - m12 * m12)
        - m01 * (m01 * m22 - m12 * m02)
        + m02 * (m01 * m12 - m11 * m02);
    if det.abs() < 1e-18 || !det.is_finite() {
        return None;
    }
    let inv = 1.0 / det;
    let a = (b0 * (m11 * m22 - m12 * m12)
        - m01 * (b1 * m22 - m12 * b2)
        + m02 * (b1 * m12 - m11 * b2))
        * inv;
    let b = (m00 * (b1 * m22 - m12 * b2)
        - b0 * (m01 * m22 - m12 * m02)
        + m02 * (m01 * b2 - b1 * m02))
        * inv;
    let c = (m00 * (m11 * b2 - b1 * m12)
        - m01 * (m01 * b2 - b1 * m02)
        + b0 * (m01 * m12 - m11 * m02))
        * inv;
    if a.is_finite() && b.is_finite() && c.is_finite() {
        Some((a, b, c))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3;

    #[test]
    fn test_filter_removes_floating_and_underground_ghosts_on_sloped_ground() {
        let w = 128u32;
        let h = 128u32;
        let mut pc = PointCloud::new(w, h);
        // Sloped ground plane: Z = 200.0 + 0.15 * x - 0.10 * y (spans ~19m across tile!)
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                let gx = 1_901_000.0 + x as f64 * 0.5;
                let gy = 3_143_000.0 + y as f64 * 0.5;
                let gz = 200.0 + 0.15 * x as f64 - 0.10 * y as f64;
                pc.points[idx] = Vec3::new(gx, gy, gz);
                pc.valid[idx] = 1;
            }
        }
        // Inject a 16x16 floating moving-vehicle ghost at +12m above ground
        for y in 20..36 {
            for x in 30..46 {
                let idx = (y * w + x) as usize;
                pc.points[idx].z += 12.0;
            }
        }
        // Inject a 10x10 underground ghost at -10m below ground
        for y in 80..90 {
            for x in 70..80 {
                let idx = (y * w + x) as usize;
                pc.points[idx].z -= 10.0;
            }
        }
        // Inject a 1px-wide vertical hanging point trail
        for k in 0..10u32 {
            let idx = ((36 + k) * w + 38) as usize;
            pc.points[idx].z += 1.2 * (k + 1) as f64;
        }
        // Inject a genuine parked car (+1.6m above ground, 12x20 px)
        for y in 50..62 {
            for x in 50..70 {
                let idx = (y * w + x) as usize;
                pc.points[idx].z += 1.6;
            }
        }

        filter_point_cloud_outliers(&mut pc);

        // Floating ghost must be 100% removed
        for y in 20..36 {
            for x in 30..46 {
                assert_eq!(pc.valid[(y * w + x) as usize], 0, "Floating ghost survived at ({},{})", x, y);
            }
        }
        // Underground ghost must be 100% removed
        for y in 80..90 {
            for x in 70..80 {
                assert_eq!(pc.valid[(y * w + x) as usize], 0, "Underground ghost survived at ({},{})", x, y);
            }
        }
        // Genuine parked car (+1.6m) must survive
        let mut car_kept = 0usize;
        for y in 52..60 {
            for x in 52..68 {
                if pc.valid[(y * w + x) as usize] != 0 {
                    car_kept += 1;
                }
            }
        }
        assert!(car_kept > 100, "Genuine parked car was erroneously removed (kept={})", car_kept);
        // Sloped ground plane must remain overwhelmingly intact
        assert!(pc.valid_count() > (w * h) as usize - 600);
    }
}
