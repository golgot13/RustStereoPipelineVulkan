//! Stage 3: `stereo_fltr` — Disparity outlier erosion, median smoothing, and hole filling.

use crate::config::StereoSettings;
use crate::image::{DisparityMap, ImageF32};

/// Run ASP-style disparity filtering without a radiometric guide image.
pub fn run_fltr(disp_in: &DisparityMap, s: &StereoSettings) -> DisparityMap {
    run_fltr_guided(disp_in, None, s)
}

/// Run ASP-style disparity filtering:
/// 1. Multi-pass neighbor consistency erosion (`rm-half-kernel`, `rm-threshold`, `rm-min-matches`, `max-mean-diff`)
/// 2. Median filter smoothing
/// 3. Optional small-gap coplanar hole interpolation
/// 4. Edge-preserving joint bilateral subpixel smoothing (smooths flat planes while keeping vertical cliffs crisp)
pub fn run_fltr_guided(
    disp_in: &DisparityMap,
    left_guide: Option<&ImageF32>,
    s: &StereoSettings,
) -> DisparityMap {
    let mut cur = disp_in.clone();

    if s.filter_mode > 0 {
        for _ in 0..s.rm_cleanup_passes.max(1) {
            cur = outlier_erode(&cur, s);
        }
        let max_speckle = ((cur.width as usize * cur.height as usize) / 1024).clamp(24, 512);
        remove_small_speckles(&mut cur, max_speckle, s.rm_threshold.min(2.0).max(1.0));
    }

    if s.median_filter_size >= 3 {
        cur = median_filter(&cur, (s.median_filter_size / 2) as i32, s.threads);
    }

    if s.fill_holes {
        cur = fill_coplanar_holes(&cur, 12);
        cur = fill_small_holes(&cur, 3);
    }

    if s.filter_mode > 0 {
        cur = joint_bilateral_filter(&cur, left_guide, s.threads);
    }

    cur
}

fn outlier_erode(disp: &DisparityMap, s: &StereoSettings) -> DisparityMap {
    let w = disp.width as i32;
    let h = disp.height as i32;
    let wu = disp.width as usize;
    let hu = disp.height as usize;
    let hkx = s.rm_half_kernel_x.max(1);
    let hky = s.rm_half_kernel_y.max(1);
    let min_frac = (s.rm_min_matches as f32 / 100.0).clamp(0.05, 0.95);
    let rm_thresh = s.rm_threshold;
    let max_md = s.max_mean_diff;

    let mut out = disp.clone();
    let n_threads = s.threads.clamp(1, hu.max(1));
    let rows_per = (hu + n_threads - 1) / n_threads;

    std::thread::scope(|scope| {
        for (t, vchunk) in out.valid.chunks_mut(rows_per * wu).enumerate() {
            let y_start = t * rows_per;
            scope.spawn(move || {
                let rows = vchunk.len() / wu;
                for ly in 0..rows {
                    let y = (y_start + ly) as i32;
                    let row_off = ly * wu;
                    for x in 0..w {
                        let idx = (y * w + x) as usize;
                        if disp.valid[idx] == 0 {
                            continue;
                        }
                        let cx = disp.dx[idx];
                        let cy = disp.dy[idx];
                        let mut total = 0u32;
                        let mut matching = 0u32;
                        let mut mean_x = 0.0f32;
                        let mut valid_n = 0u32;

                        for ky in -hky..=hky {
                            let ny = y + ky;
                            if ny < 0 || ny >= h {
                                continue;
                            }
                            for kx in -hkx..=hkx {
                                let nx = x + kx;
                                if nx < 0 || nx >= w || (kx == 0 && ky == 0) {
                                    continue;
                                }
                                total += 1;
                                let nidx = (ny * w + nx) as usize;
                                if disp.valid[nidx] != 0 {
                                    let ndx = disp.dx[nidx];
                                    let ndy = disp.dy[nidx];
                                    mean_x += ndx;
                                    valid_n += 1;
                                    if (ndx - cx).abs() <= rm_thresh && (ndy - cy).abs() <= rm_thresh {
                                        matching += 1;
                                    }
                                }
                            }
                        }

                        let frac = if total > 0 { matching as f32 / total as f32 } else { 0.0 };
                        let mean_diff = if valid_n > 0 {
                            (cx - mean_x / valid_n as f32).abs()
                        } else {
                            f32::INFINITY
                        };
                        if frac < min_frac || mean_diff > max_md {
                            vchunk[row_off + x as usize] = 0;
                        }
                    }
                }
            });
        }
    });
    out
}

fn median_filter(disp: &DisparityMap, rad: i32, threads: usize) -> DisparityMap {
    let w = disp.width as i32;
    let h = disp.height as i32;
    let wu = disp.width as usize;
    let hu = disp.height as usize;
    let mut out = disp.clone();
    let n_threads = threads.clamp(1, hu.max(1));
    let rows_per = (hu + n_threads - 1) / n_threads;

    std::thread::scope(|scope| {
        for (t, (dxc, dyc)) in out
            .dx
            .chunks_mut(rows_per * wu)
            .zip(out.dy.chunks_mut(rows_per * wu))
            .enumerate()
        {
            let y_start = t * rows_per;
            scope.spawn(move || {
                let rows = dxc.len() / wu;
                let cap = ((2 * rad + 1) * (2 * rad + 1)) as usize;
                let mut vals_x = Vec::with_capacity(cap);
                let mut vals_y = Vec::with_capacity(cap);
                for ly in 0..rows {
                    let y = (y_start + ly) as i32;
                    let row_off = ly * wu;
                    for x in 0..w {
                        let idx = (y * w + x) as usize;
                        if disp.valid[idx] == 0 {
                            continue;
                        }
                        vals_x.clear();
                        vals_y.clear();
                        for ky in -rad..=rad {
                            let ny = (y + ky).clamp(0, h - 1);
                            for kx in -rad..=rad {
                                let nx = (x + kx).clamp(0, w - 1);
                                let nidx = (ny * w + nx) as usize;
                                if disp.valid[nidx] != 0 {
                                    vals_x.push(disp.dx[nidx]);
                                    vals_y.push(disp.dy[nidx]);
                                }
                            }
                        }
                        if !vals_x.is_empty() {
                            vals_x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                            vals_y.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                            let mid = vals_x.len() / 2;
                            dxc[row_off + x as usize] = vals_x[mid];
                            dyc[row_off + x as usize] = vals_y[mid];
                        }
                    }
                }
            });
        }
    });
    out
}

/// Edge-preserving 4-axis coplanar hole filler: closes homogeneous asphalt/shadow gaps up to `2*max_dist` px
/// ONLY where opposing boundaries agree within `<= 2.5 px` disparity (never bridges car roofs or building cliffs).
fn fill_coplanar_holes(disp: &DisparityMap, max_dist: i32) -> DisparityMap {
    let w = disp.width as i32;
    let h = disp.height as i32;
    let mut out = disp.clone();
    let axes: [(i32, i32); 4] = [(1, 0), (0, 1), (1, 1), (1, -1)];

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            if disp.valid[idx] != 0 {
                continue;
            }
            let mut sum_x = 0.0f32;
            let mut sum_y = 0.0f32;
            let mut sum_w = 0.0f32;
            let mut coplanar_axes = 0u32;

            for &(ax, ay) in &axes {
                let mut neg_hit: Option<(i32, f32, f32)> = None;
                for step in 1..=max_dist {
                    let nx = x - ax * step;
                    let ny = y - ay * step;
                    if nx < 0 || nx >= w || ny < 0 || ny >= h {
                        break;
                    }
                    let nidx = (ny * w + nx) as usize;
                    if disp.valid[nidx] != 0 {
                        neg_hit = Some((step, disp.dx[nidx], disp.dy[nidx]));
                        break;
                    }
                }
                let mut pos_hit: Option<(i32, f32, f32)> = None;
                for step in 1..=max_dist {
                    let nx = x + ax * step;
                    let ny = y + ay * step;
                    if nx < 0 || nx >= w || ny < 0 || ny >= h {
                        break;
                    }
                    let nidx = (ny * w + nx) as usize;
                    if disp.valid[nidx] != 0 {
                        pos_hit = Some((step, disp.dx[nidx], disp.dy[nidx]));
                        break;
                    }
                }

                if let (Some((dn, dxn, dyn_)), Some((dp, dxp, dyp))) = (neg_hit, pos_hit) {
                    if (dxn - dxp).abs() <= 2.5 && (dyn_ - dyp).abs() <= 2.0 {
                        let span = (dn + dp) as f32;
                        let t = dn as f32 / span;
                        let interp_x = dxn * (1.0 - t) + dxp * t;
                        let interp_y = dyn_ * (1.0 - t) + dyp * t;
                        let wt = 1.0 / (span + 0.5);
                        sum_x += interp_x * wt;
                        sum_y += interp_y * wt;
                        sum_w += wt;
                        coplanar_axes += 1;
                    }
                }
            }

            if coplanar_axes >= 2 && sum_w > 0.0 {
                out.dx[idx] = sum_x / sum_w;
                out.dy[idx] = sum_y / sum_w;
                out.valid[idx] = 1;
            }
        }
    }
    out
}

fn fill_small_holes(disp: &DisparityMap, radius: i32) -> DisparityMap {
    let w = disp.width as i32;
    let h = disp.height as i32;
    let mut out = disp.clone();

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            if disp.valid[idx] != 0 {
                continue;
            }
            let mut sum_x = 0.0f32;
            let mut sum_y = 0.0f32;
            let mut w_sum = 0.0f32;
            let mut count = 0u32;
            let mut min_dx = f32::INFINITY;
            let mut max_dx = f32::NEG_INFINITY;
            for ky in -radius..=radius {
                let ny = y + ky;
                if ny < 0 || ny >= h {
                    continue;
                }
                for kx in -radius..=radius {
                    let nx = x + kx;
                    if nx < 0 || nx >= w {
                        continue;
                    }
                    let nidx = (ny * w + nx) as usize;
                    if disp.valid[nidx] != 0 {
                        let dist_sq = (kx * kx + ky * ky) as f32;
                        let wt = 1.0 / (dist_sq + 0.5);
                        let dxv = disp.dx[nidx];
                        if dxv < min_dx { min_dx = dxv; }
                        if dxv > max_dx { max_dx = dxv; }
                        sum_x += dxv * wt;
                        sum_y += disp.dy[nidx] * wt;
                        w_sum += wt;
                        count += 1;
                    }
                }
            }
            if count >= 4 && w_sum > 0.0 && (max_dx - min_dx) <= 2.5 {
                out.dx[idx] = sum_x / w_sum;
                out.dy[idx] = sum_y / w_sum;
                out.valid[idx] = 1;
            }
        }
    }
    out
}

/// ASP `erode-max-size` / OpenCV `filterSpeckles`: remove small isolated disparity islands
/// (e.g. false matches floating above/below real roofs and ground in 3D point clouds).
fn remove_small_speckles(disp: &mut DisparityMap, max_speckle_size: usize, max_diff: f32) {
    let w = disp.width as usize;
    let h = disp.height as usize;
    let n = w * h;
    if n == 0 || max_speckle_size == 0 {
        return;
    }
    let mut visited = vec![0u8; n];
    let mut stack: Vec<usize> = Vec::with_capacity(256);
    let mut component: Vec<usize> = Vec::with_capacity(256);

    for start in 0..n {
        if disp.valid[start] == 0 || visited[start] != 0 {
            continue;
        }
        visited[start] = 1;
        stack.clear();
        component.clear();
        stack.push(start);

        while let Some(idx) = stack.pop() {
            component.push(idx);
            let x = idx % w;
            let y = idx / w;
            let d = disp.dx[idx];

            let mut try_push = |nx: usize, ny: usize| {
                let nidx = ny * w + nx;
                if visited[nidx] == 0 && disp.valid[nidx] != 0 && (disp.dx[nidx] - d).abs() <= max_diff {
                    visited[nidx] = 1;
                    stack.push(nidx);
                }
            };
            if x > 0 { try_push(x - 1, y); }
            if x + 1 < w { try_push(x + 1, y); }
            if y > 0 { try_push(x, y - 1); }
            if y + 1 < h { try_push(x, y + 1); }
        }

        if component.len() <= max_speckle_size {
            for &idx in &component {
                disp.valid[idx] = 0;
            }
        }
    }
}

/// Solve a 3x3 symmetric positive-definite linear system `A * [a, b, c]^T = rhs` in `f64`.
#[inline]
fn solve_sym_3x3_f64(
    m00: f64, m01: f64, m02: f64,
    m11: f64, m12: f64, m22: f64,
    b0: f64, b1: f64, b2: f64,
) -> Option<(f64, f64, f64)> {
    let det = m00 * (m11 * m22 - m12 * m12)
        - m01 * (m01 * m22 - m12 * m02)
        + m02 * (m01 * m12 - m11 * m02);
    if det.abs() < 1e-12 || !det.is_finite() {
        return None;
    }
    let inv = 1.0 / det;
    let a = (b0 * (m11 * m22 - m12 * m12) - m01 * (b1 * m22 - m12 * b2) + m02 * (b1 * m12 - m11 * b2)) * inv;
    let b = (m00 * (b1 * m22 - m12 * b2) - b0 * (m01 * m22 - m12 * m02) + m02 * (m01 * b2 - b1 * m02)) * inv;
    let c = (m00 * (m11 * b2 - b1 * m12) - m01 * (m01 * b2 - b1 * m02) + b0 * (m01 * m12 - m11 * m02)) * inv;
    if a.is_finite() && b.is_finite() && c.is_finite() {
        Some((a, b, c))
    } else {
        None
    }
}

/// Multi-threaded 9x9 Edge-Preserving Local Planar Filter (`d(dx, dy) = a*dx + b*dy + c` in `f64`):
/// - Fits a weighted local plane over coplanar same-material neighbors (`9x9`, ~42 cm on Meyreuil)
/// - Eliminates granulometry and residual undulations on flat roofs, asphalt, and lawns
/// - Preserves sloped pitched roofs and roads with ZERO slope bias (unlike box/mean bilateral)
/// - Hard disparity cliff gate (`|d_q - d_p| <= 1.25 px`) + radiometric guide weight (`sigma_I = 0.085`)
///   strictly prevents rounding or bleeding across building walls, curbs, and vehicles.
fn joint_bilateral_filter(
    disp: &DisparityMap,
    left_guide: Option<&ImageF32>,
    threads: usize,
) -> DisparityMap {
    let w = disp.width as usize;
    let h = disp.height as usize;
    if w < 9 || h < 9 {
        return disp.clone();
    }

    // Normalize optional radiometric guide once to [0, 1]
    let guide_norm: Option<Vec<f32>> = left_guide
        .filter(|g| g.width == disp.width && g.height == disp.height)
        .map(|g| {
            let (mn, mx, _, _) = g.stats();
            let inv = 1.0 / (mx - mn).max(1e-6);
            g.data
                .iter()
                .map(|&v| if v.is_finite() { ((v - mn) * inv).clamp(0.0, 1.0) } else { 0.0 })
                .collect()
        });

    let rad = 4i32; // 9x9 local planar support
    let max_step = 1.25f32;
    let inv_2sig_s2 = 1.0f32 / (2.0 * 3.5 * 3.5); // sigma_s = 3.5 px
    let inv_2sig_d2 = 1.0f32 / (2.0 * 0.80 * 0.80); // sigma_d = 0.80 px
    let inv_2sig_i2 = 1.0f32 / (2.0 * 0.085 * 0.085); // sigma_I = 0.085

    // Precompute 9x9 spatial Gaussian weights
    let mut sp_w = [[0.0f32; 9]; 9];
    for ky in -rad..=rad {
        for kx in -rad..=rad {
            let r2 = (kx * kx + ky * ky) as f32;
            sp_w[(ky + rad) as usize][(kx + rad) as usize] = (-r2 * inv_2sig_s2).exp();
        }
    }

    let mut cur = disp.clone();
    let n_threads = threads.clamp(1, h.max(1));
    let rows_per = (h + n_threads - 1) / n_threads;
    let g_ref = guide_norm.as_deref();

    for _pass in 0..2 {
        let src = &cur;
        let mut next = cur.clone();
        std::thread::scope(|scope| {
            for (t, (dxc, dyc)) in next
                .dx
                .chunks_mut(rows_per * w)
                .zip(next.dy.chunks_mut(rows_per * w))
                .enumerate()
            {
                let y_start = t * rows_per;
                scope.spawn(move || {
                    let rows = dxc.len() / w;
                    for ly in 0..rows {
                        let y = (y_start + ly) as i32;
                        let row_off = ly * w;
                        for x in 0..w as i32 {
                            let idx = (y as usize) * w + (x as usize);
                            if src.valid[idx] == 0 {
                                continue;
                            }
                            let cx = src.dx[idx];
                            let gi_c = g_ref.map(|g| g[idx]);
                            let mut m00 = 0.0f64;
                            let mut m01 = 0.0f64;
                            let mut m02 = 0.0f64;
                            let mut m11 = 0.0f64;
                            let mut m12 = 0.0f64;
                            let mut m22 = 0.0f64;
                            let mut b0 = 0.0f64;
                            let mut b1 = 0.0f64;
                            let mut b2 = 0.0f64;
                            let mut sum_y = 0.0f64;
                            let mut min_d = f32::INFINITY;
                            let mut max_d = f32::NEG_INFINITY;
                            let mut cnt = 0u32;

                            for ky in -rad..=rad {
                                let ny = y + ky;
                                if ny < 0 || ny >= h as i32 {
                                    continue;
                                }
                                let nrow = (ny as usize) * w;
                                let sp_row = &sp_w[(ky + rad) as usize];
                                let ky_f = ky as f64;
                                for kx in -rad..=rad {
                                    let nx = x + kx;
                                    if nx < 0 || nx >= w as i32 {
                                        continue;
                                    }
                                    let nidx = nrow + (nx as usize);
                                    if src.valid[nidx] == 0 {
                                        continue;
                                    }
                                    let ndx = src.dx[nidx];
                                    let dd = ndx - cx;
                                    // Hard cliff gate prevents any smoothing across building roofs/walls/curbs
                                    if dd.abs() > max_step {
                                        continue;
                                    }
                                    let mut wt = sp_row[(kx + rad) as usize] * (-(dd * dd) * inv_2sig_d2).exp();
                                    if let (Some(gc), Some(g)) = (gi_c, g_ref) {
                                        let di = g[nidx] - gc;
                                        wt *= (-(di * di) * inv_2sig_i2).exp();
                                    }
                                    let wf = wt as f64;
                                    let kx_f = kx as f64;
                                    let d_f = ndx as f64;
                                    m00 += wf * kx_f * kx_f;
                                    m01 += wf * kx_f * ky_f;
                                    m02 += wf * kx_f;
                                    m11 += wf * ky_f * ky_f;
                                    m12 += wf * ky_f;
                                    m22 += wf;
                                    b0 += wf * kx_f * d_f;
                                    b1 += wf * ky_f * d_f;
                                    b2 += wf * d_f;
                                    sum_y += wf * src.dy[nidx] as f64;
                                    if ndx < min_d { min_d = ndx; }
                                    if ndx > max_d { max_d = ndx; }
                                    cnt += 1;
                                }
                            }

                            if m22 > 1e-6 {
                                let est = if cnt >= 9 {
                                    // Exact f64 local plane fit evaluated at center (0, 0) = c (zero slope bias!)
                                    solve_sym_3x3_f64(
                                        m00 + 1e-6, m01, m02,
                                        m11 + 1e-6, m12, m22,
                                        b0, b1, b2,
                                    )
                                    .map(|(_, _, c)| (c as f32).clamp(min_d, max_d))
                                    .unwrap_or((b2 / m22) as f32)
                                } else {
                                    (b2 / m22) as f32
                                };
                                dxc[row_off + x as usize] = est;
                                dyc[row_off + x as usize] = (sum_y / m22) as f32;
                            }
                        }
                    }
                });
            }
        });
        cur = next;
    }

    cur
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coplanar_hole_fill_preserves_depth_discontinuities() {
        let w = 40u32;
        let h = 40u32;
        let mut disp = DisparityMap::new(w, h);
        // Flat ground at dx = -5.0
        for y in 0..h {
            for x in 0..w {
                disp.set(x, y, -5.0, 0.0, true);
            }
        }
        // Punch a 8x8 flat asphalt hole in the middle -> MUST be filled
        for y in 16..24 {
            for x in 16..24 {
                disp.valid[(y * w + x) as usize] = 0;
            }
        }
        // Create a raised roof block (dx = -20.0) with a 4px vertical occlusion gap next to it -> MUST NOT bridge
        for y in 4..12 {
            for x in 4..12 {
                disp.set(x, y, -20.0, 0.0, true);
            }
            for x in 12..16 {
                disp.valid[(y * w + x) as usize] = 0;
            }
        }

        let filled = fill_coplanar_holes(&disp, 12);
        // Center of flat asphalt hole should be smoothly filled at -5.0
        assert!(filled.is_valid(20, 20));
        assert!((filled.dx[filled.idx(20, 20)] - (-5.0)).abs() < 0.05);
        // Occlusion gap between -20.0 roof and -5.0 ground must NOT be interpolated to -12.5!
        let occ_idx = filled.idx(13, 8);
        if filled.valid[occ_idx] != 0 {
            assert!(
                (filled.dx[occ_idx] - (-5.0)).abs() < 1.5 || (filled.dx[occ_idx] - (-20.0)).abs() < 1.5,
                "Unexpected cliff-bridging disparity: {}",
                filled.dx[occ_idx]
            );
        }
    }

    #[test]
    fn test_joint_bilateral_smooths_flat_plane_and_preserves_cliff() {
        let w = 40u32;
        let h = 40u32;
        let mut disp = DisparityMap::new(w, h);
        let mut guide = ImageF32::new(w, h);
        // Left half: tilted roof pitch dx(x) = -0.15 * x with +/-0.35 px granulometry
        // Right half: tall vertical cliff down to -30.0 (10 m step!)
        for y in 0..h {
            for x in 0..w {
                if x < 20 {
                    let noise = if (x + y) % 2 == 0 { 0.35 } else { -0.35 };
                    disp.set(x, y, -0.15 * x as f32 + noise, 0.0, true);
                    guide.set(x, y, 0.3);
                } else {
                    disp.set(x, y, -30.0, 0.0, true);
                    guide.set(x, y, 0.85);
                }
            }
        }

        let smooth = joint_bilateral_filter(&disp, Some(&guide), 2);
        // Sloped plane noise must collapse from 0.35 px down to < 0.05 px WITH ZERO SLOPE BIAS even near edge x=19!
        let mut max_slope_err = 0.0f32;
        for y in 4..36 {
            for x in 4..20 {
                let expected = -0.15 * x as f32;
                max_slope_err = max_slope_err.max((smooth.dx[smooth.idx(x, y)] - expected).abs());
            }
        }
        assert!(max_slope_err < 0.05, "Sloped plane noise/bias too high: {}", max_slope_err);

        // Building roof edge right at x = 20 MUST remain razor-sharp at -30.0 (zero bleeding!)
        for y in 4..36 {
            let roof_edge = smooth.dx[smooth.idx(20, y)];
            assert!((roof_edge - (-30.0)).abs() < 1e-4, "Roof cliff rounded: {}", roof_edge);
        }
    }
}
