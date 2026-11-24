//! Stage 1: `stereo_corr` — Integer stereo correlation (ASP BM, ASP SGM 4/8-path, ASP MGM).
//! Supports SAD, SSD, zero-mean NCC, and 5x5 Census cost metrics + Left-Right consistency check.
//! Executes on Vulkan Compute GPU when available, or multi-threaded CPU (`std::thread::scope`).

use crate::config::{CostMode, StereoAlgorithm, StereoSettings};
use crate::image::{DisparityMap, ImageF32};
use crate::vulkan::VulkanCompute;

/// Run Stage 1 stereo correlation returning an integer disparity map and a flag indicating Vulkan GPU execution.
pub fn run_corr(
    left: &ImageF32,
    right: &ImageF32,
    settings: &StereoSettings,
    vk: Option<&VulkanCompute>,
) -> (DisparityMap, bool) {
    let w = left.width;
    let h = left.height;

    // For ASP BM with horizontal epipolar search (dy == 0) and Vulkan available, dispatch directly to GPU!
    if settings.stereo_algorithm == StereoAlgorithm::AspBm
        && settings.cost_mode != CostMode::Census
        && settings.search_min_y == 0
        && settings.search_max_y == 0
    {
        if let Some(gpu) = vk {
            let hkw = (settings.corr_kernel_x / 2).max(1) as i32;
            let hkh = (settings.corr_kernel_y / 2).max(1) as i32;
            if let Ok((dx_lr, _cost_lr)) = gpu.run_stereo_bm(
                &left.data,
                &right.data,
                w,
                h,
                hkw,
                hkh,
                settings.search_min_x,
                settings.search_max_x,
                settings.cost_mode as i32,
            ) {
                // Reverse pass on GPU for Left-Right consistency check
                let mut disp = DisparityMap::new(w, h);
                if let Ok((dx_rl, _)) = gpu.run_stereo_bm(
                    &right.data,
                    &left.data,
                    w,
                    h,
                    hkw,
                    hkh,
                    -settings.search_max_x,
                    -settings.search_min_x,
                    settings.cost_mode as i32,
                ) {
                    apply_lr_check(w, h, &dx_lr, &dx_rl, settings.lr_threshold, &mut disp);
                    return (disp, true);
                }
            }
        }
    }

    // Build cost volume & solve using selected algorithm (BM / SGM / MGM)
    match settings.stereo_algorithm {
        StereoAlgorithm::AspBm => (correlate_bm_cpu(left, right, settings), false),
        StereoAlgorithm::AspSgm => correlate_sgm(left, right, settings, false, vk),
        StereoAlgorithm::AspMgm => correlate_sgm(left, right, settings, true, vk),
    }
}

/// Precompute 5x5 Census bitstrings (25 bits packed into u32) for fast Hamming matching.
fn compute_census_5x5(img: &ImageF32) -> Vec<u32> {
    let w = img.width as usize;
    let h = img.height as usize;
    let mut out = vec![0u32; w * h];
    for y in 0..h {
        for x in 0..w {
            let center = img.data[y * w + x];
            let mut bits = 0u32;
            let mut bit = 0u32;
            for ky in -2..=2 {
                let sy = (y as i32 + ky).clamp(0, h as i32 - 1) as usize;
                for kx in -2..=2 {
                    if kx == 0 && ky == 0 {
                        continue;
                    }
                    let sx = (x as i32 + kx).clamp(0, w as i32 - 1) as usize;
                    if img.data[sy * w + sx] > center {
                        bits |= 1u32 << bit;
                    }
                    bit += 1;
                }
            }
            out[y * w + x] = bits;
        }
    }
    out
}

/// Compute patch matching cost at `(x, y)` in left vs `(x + dx, y + dy)` in right.
/// Automatically uses runtime-detected AVX2 SIMD for interior rows with safe scalar fallback.
#[inline]
pub fn patch_cost(
    left: &ImageF32,
    right: &ImageF32,
    x: i32,
    y: i32,
    dx: i32,
    dy: i32,
    hkw: i32,
    hkh: i32,
    mode: CostMode,
) -> f32 {
    let w = left.width as i32;
    let h = left.height as i32;
    let lx0 = x - hkw;
    let lx1 = x + hkw;
    let rx0 = lx0 + dx;
    let rx1 = lx1 + dx;
    let interior_x = lx0 >= 0 && lx1 < w && rx0 >= 0 && rx1 < w;
    let row_len = (2 * hkw + 1) as usize;

    match mode {
        CostMode::Sad => {
            let mut sum = 0.0f32;
            let mut cnt = 0.0f32;
            for ky in -hkh..=hkh {
                let ly = (y + ky).clamp(0, h - 1) as usize;
                let ry = (y + ky + dy).clamp(0, h - 1) as usize;
                let lrow = &left.data[ly * w as usize..(ly + 1) * w as usize];
                let rrow = &right.data[ry * w as usize..(ry + 1) * w as usize];
                if interior_x {
                    sum += crate::pipeline::simd::row_sad(
                        &lrow[lx0 as usize..lx0 as usize + row_len],
                        &rrow[rx0 as usize..rx0 as usize + row_len],
                    );
                    cnt += row_len as f32;
                } else {
                    for kx in -hkw..=hkw {
                        let lx = (x + kx).clamp(0, w - 1) as usize;
                        let rx = (x + kx + dx).clamp(0, w - 1) as usize;
                        sum += (lrow[lx] - rrow[rx]).abs();
                        cnt += 1.0;
                    }
                }
            }
            sum / cnt.max(1.0)
        }
        CostMode::Ssd => {
            let mut sum = 0.0f32;
            let mut cnt = 0.0f32;
            for ky in -hkh..=hkh {
                let ly = (y + ky).clamp(0, h - 1) as usize;
                let ry = (y + ky + dy).clamp(0, h - 1) as usize;
                let lrow = &left.data[ly * w as usize..(ly + 1) * w as usize];
                let rrow = &right.data[ry * w as usize..(ry + 1) * w as usize];
                if interior_x {
                    sum += crate::pipeline::simd::row_ssd(
                        &lrow[lx0 as usize..lx0 as usize + row_len],
                        &rrow[rx0 as usize..rx0 as usize + row_len],
                    );
                    cnt += row_len as f32;
                } else {
                    for kx in -hkw..=hkw {
                        let lx = (x + kx).clamp(0, w - 1) as usize;
                        let rx = (x + kx + dx).clamp(0, w - 1) as usize;
                        let d = lrow[lx] - rrow[rx];
                        sum += d * d;
                        cnt += 1.0;
                    }
                }
            }
            sum / cnt.max(1.0)
        }
        CostMode::Ncc | CostMode::Census => {
            // Zero-mean Normalized Cross-Correlation cost = 1.0 - ZNCC in [0, 2]
            let mut sum_l = 0.0f32;
            let mut sum_r = 0.0f32;
            let mut sum_ll = 0.0f32;
            let mut sum_rr = 0.0f32;
            let mut sum_lr = 0.0f32;
            let mut cnt = 0.0f32;
            for ky in -hkh..=hkh {
                let ly = (y + ky).clamp(0, h - 1) as usize;
                let ry = (y + ky + dy).clamp(0, h - 1) as usize;
                let lrow = &left.data[ly * w as usize..(ly + 1) * w as usize];
                let rrow = &right.data[ry * w as usize..(ry + 1) * w as usize];
                if interior_x {
                    let (sl, sr, sll, srr, slr) = crate::pipeline::simd::row_ncc_moments(
                        &lrow[lx0 as usize..lx0 as usize + row_len],
                        &rrow[rx0 as usize..rx0 as usize + row_len],
                    );
                    sum_l += sl;
                    sum_r += sr;
                    sum_ll += sll;
                    sum_rr += srr;
                    sum_lr += slr;
                    cnt += row_len as f32;
                } else {
                    for kx in -hkw..=hkw {
                        let lx = (x + kx).clamp(0, w - 1) as usize;
                        let rx = (x + kx + dx).clamp(0, w - 1) as usize;
                        let lv = lrow[lx];
                        let rv = rrow[rx];
                        sum_l += lv;
                        sum_r += rv;
                        sum_ll += lv * lv;
                        sum_rr += rv * rv;
                        sum_lr += lv * rv;
                        cnt += 1.0;
                    }
                }
            }
            let inv_n = 1.0 / cnt.max(1.0);
            let mean_l = sum_l * inv_n;
            let mean_r = sum_r * inv_n;
            let var_l = (sum_ll * inv_n - mean_l * mean_l).max(0.0);
            let var_r = (sum_rr * inv_n - mean_r * mean_r).max(0.0);
            let cov = sum_lr * inv_n - mean_l * mean_r;
            let denom = (var_l * var_r).sqrt();
            if denom < 1e-6 {
                1.0
            } else {
                1.0 - (cov / denom).clamp(-1.0, 1.0)
            }
        }
    }
}

/// Multi-threaded ASP Block Matching with 2D search `(dx, dy)` and LR consistency check.
fn correlate_bm_cpu(left: &ImageF32, right: &ImageF32, s: &StereoSettings) -> DisparityMap {
    let w = left.width as usize;
    let h = left.height as usize;
    let hkw = (s.corr_kernel_x / 2).max(1) as i32;
    let hkh = (s.corr_kernel_y / 2).max(1) as i32;
    let n_threads = s.threads.clamp(1, h.max(1));
    let rows_per = (h + n_threads - 1) / n_threads;

    let mut dx_lr = vec![0.0f32; w * h];
    let mut dy_lr = vec![0.0f32; w * h];
    let mut dx_rl = vec![0.0f32; w * h];

    let census_l = if s.cost_mode == CostMode::Census { Some(compute_census_5x5(left)) } else { None };
    let census_r = if s.cost_mode == CostMode::Census { Some(compute_census_5x5(right)) } else { None };

    // Left-to-Right pass
    std::thread::scope(|scope| {
        for (t, (dxc, dyc)) in dx_lr
            .chunks_mut(rows_per * w)
            .zip(dy_lr.chunks_mut(rows_per * w))
            .enumerate()
        {
            let y_start = t * rows_per;
            let cl = census_l.as_deref();
            let cr = census_r.as_deref();
            scope.spawn(move || {
                let rows = dxc.len() / w;
                for ly in 0..rows {
                    let y = (y_start + ly) as i32;
                    for x in 0..w as i32 {
                        let mut best_cost = f32::INFINITY;
                        let mut best_dx = 0i32;
                        let mut best_dy = 0i32;
                        for dy in s.search_min_y..=s.search_max_y {
                            if y + dy < 0 || y + dy >= h as i32 {
                                continue;
                            }
                            for dx in s.search_min_x..=s.search_max_x {
                                if x + dx < 0 || x + dx >= w as i32 {
                                    continue;
                                }
                                let cost = if let (Some(c1), Some(c2)) = (cl, cr) {
                                    census_patch_cost(c1, c2, w as i32, h as i32, x, y, dx, dy, hkw.min(2), hkh.min(2))
                                } else {
                                    patch_cost(left, right, x, y, dx, dy, hkw, hkh, s.cost_mode)
                                };
                                if cost < best_cost {
                                    best_cost = cost;
                                    best_dx = dx;
                                    best_dy = dy;
                                }
                            }
                        }
                        dxc[ly * w + x as usize] = best_dx as f32;
                        dyc[ly * w + x as usize] = best_dy as f32;
                    }
                }
            });
        }
    });

    // Right-to-Left pass (for LR consistency validation)
    std::thread::scope(|scope| {
        for (t, dxc) in dx_rl.chunks_mut(rows_per * w).enumerate() {
            let y_start = t * rows_per;
            let cl = census_l.as_deref();
            let cr = census_r.as_deref();
            scope.spawn(move || {
                let rows = dxc.len() / w;
                for ly in 0..rows {
                    let y = (y_start + ly) as i32;
                    for x in 0..w as i32 {
                        let mut best_cost = f32::INFINITY;
                        let mut best_dx = 0i32;
                        for dx in -s.search_max_x..=-s.search_min_x {
                            if x + dx < 0 || x + dx >= w as i32 {
                                continue;
                            }
                            let cost = if let (Some(c1), Some(c2)) = (cl, cr) {
                                census_patch_cost(c2, c1, w as i32, h as i32, x, y, dx, 0, hkw.min(2), hkh.min(2))
                            } else {
                                patch_cost(right, left, x, y, dx, 0, hkw, hkh, s.cost_mode)
                            };
                            if cost < best_cost {
                                best_cost = cost;
                                best_dx = dx;
                            }
                        }
                        dxc[ly * w + x as usize] = best_dx as f32;
                    }
                }
            });
        }
    });

    let mut disp = DisparityMap::new(left.width, left.height);
    disp.dy = dy_lr;
    apply_lr_check(left.width, left.height, &dx_lr, &dx_rl, s.lr_threshold, &mut disp);
    disp
}

#[inline]
fn census_patch_cost(
    cl: &[u32],
    cr: &[u32],
    w: i32,
    h: i32,
    x: i32,
    y: i32,
    dx: i32,
    dy: i32,
    hkw: i32,
    hkh: i32,
) -> f32 {
    let mut sum = 0u32;
    let mut cnt = 0u32;
    for ky in -hkh..=hkh {
        let ly = (y + ky).clamp(0, h - 1) as usize;
        let ry = (y + ky + dy).clamp(0, h - 1) as usize;
        for kx in -hkw..=hkw {
            let lx = (x + kx).clamp(0, w - 1) as usize;
            let rx = (x + kx + dx).clamp(0, w - 1) as usize;
            sum += (cl[ly * w as usize + lx] ^ cr[ry * w as usize + rx]).count_ones();
            cnt += 24;
        }
    }
    sum as f32 / cnt.max(1) as f32
}

/// Semi-Global Matching (ASP SGM) & More Global Matching (ASP MGM) with GPU Vulkan Compute
/// 3D cost volume acceleration (with automatic AVX2 multi-thread CPU fallback) + 4-direction
/// dynamic programming (or 2D quadrant aggregation for MGM) + LR consistency check.
fn correlate_sgm(
    left: &ImageF32,
    right: &ImageF32,
    s: &StereoSettings,
    use_mgm: bool,
    vk: Option<&VulkanCompute>,
) -> (DisparityMap, bool) {
    let w = left.width as usize;
    let h = left.height as usize;
    let min_d = s.search_min_x;
    let max_d = s.search_max_x;
    let num_d = (max_d - min_d + 1).max(1) as usize;
    let hkw = (s.corr_kernel_x / 2).clamp(1, 4) as i32;
    let hkh = (s.corr_kernel_y / 2).clamp(1, 4) as i32;
    let n_threads = s.threads.clamp(1, h.max(1));
    let rows_per = (h + n_threads - 1) / n_threads;

    // 1. Build 3D matching cost volume C[y][x][d] on Vulkan GPU when available, else AVX2 CPU
    let mut used_vk = false;
    let cost_vol: Vec<f32> = if s.cost_mode != CostMode::Census {
        if let Some(gpu) = vk {
            match gpu.run_stereo_cost_volume(
                &left.data,
                &right.data,
                left.width,
                left.height,
                hkw,
                hkh,
                min_d,
                num_d,
                s.cost_mode as i32,
            ) {
                Ok(vol) if vol.len() == w * h * num_d => {
                    used_vk = true;
                    vol
                }
                _ => build_cost_volume_cpu(left, right, s, min_d, num_d, hkw, hkh, rows_per),
            }
        } else {
            build_cost_volume_cpu(left, right, s, min_d, num_d, hkw, hkh, rows_per)
        }
    } else {
        build_cost_volume_cpu(left, right, s, min_d, num_d, hkw, hkh, rows_per)
    };

    // 2. Multi-path SGM / MGM cost aggregation
    let p1 = s.sgm_p1;
    let p2 = s.sgm_p2;
    let mut agg = vec![0.0f32; w * h * num_d];

    // Four cardinal passes (+X, -X, +Y, -Y) and for MGM, 2D quadrant coupling
    let dirs: &[(i32, i32)] = &[(1, 0), (-1, 0), (0, 1), (0, -1)];
    for &(dx_dir, dy_dir) in dirs {
        let mut path_l = vec![0.0f32; w * h * num_d];
        let y_range: Vec<usize> = if dy_dir >= 0 { (0..h).collect() } else { (0..h).rev().collect() };
        let x_range: Vec<usize> = if dx_dir >= 0 { (0..w).collect() } else { (0..w).rev().collect() };

        for &y in &y_range {
            for &x in &x_range {
                let idx = (y * w + x) * num_d;
                let px = x as i32 - dx_dir;
                let py = y as i32 - dy_dir;
                let has_prev = px >= 0 && px < w as i32 && py >= 0 && py < h as i32;

                if !has_prev {
                    for d in 0..num_d {
                        path_l[idx + d] = cost_vol[idx + d];
                    }
                } else {
                    let pidx = (py as usize * w + px as usize) * num_d;
                    let mut min_prev = f32::INFINITY;
                    for d in 0..num_d {
                        if path_l[pidx + d] < min_prev {
                            min_prev = path_l[pidx + d];
                        }
                    }
                    // For MGM, blend perpendicular already-visited neighbor in scan order
                    let (q_valid, qidx, min_q) = if use_mgm {
                        let y_step = if dy_dir >= 0 { 1 } else { -1 };
                        let x_step = if dx_dir >= 0 { 1 } else { -1 };
                        let (qx, qy) = if dx_dir != 0 {
                            (x as i32, y as i32 - y_step)
                        } else {
                            (x as i32 - x_step, y as i32)
                        };
                        if qx >= 0 && qx < w as i32 && qy >= 0 && qy < h as i32 {
                            let qi = (qy as usize * w + qx as usize) * num_d;
                            let mut mq = f32::INFINITY;
                            for d in 0..num_d {
                                if path_l[qi + d] < mq {
                                    mq = path_l[qi + d];
                                }
                            }
                            (true, qi, mq)
                        } else {
                            (false, 0, 0.0)
                        }
                    } else {
                        (false, 0, 0.0)
                    };

                    for d in 0..num_d {
                        let c0 = path_l[pidx + d];
                        let cm = if d > 0 { path_l[pidx + d - 1] + p1 } else { f32::INFINITY };
                        let cp = if d + 1 < num_d { path_l[pidx + d + 1] + p1 } else { f32::INFINITY };
                        let best_p = c0.min(cm).min(cp).min(min_prev + p2) - min_prev;

                        let smooth_term = if q_valid {
                            let q0 = path_l[qidx + d];
                            let qm = if d > 0 { path_l[qidx + d - 1] + p1 } else { f32::INFINITY };
                            let qp = if d + 1 < num_d { path_l[qidx + d + 1] + p1 } else { f32::INFINITY };
                            let best_q = q0.min(qm).min(qp).min(min_q + p2) - min_q;
                            0.5 * (best_p + best_q)
                        } else {
                            best_p
                        };
                        path_l[idx + d] = cost_vol[idx + d] + smooth_term;
                    }
                }
                for d in 0..num_d {
                    agg[idx + d] += path_l[idx + d];
                }
            }
        }
    }

    // 3. Winner-Takes-All on Left + implicit Right-to-Left WTA from the same aggregated volume!
    let mut dx_lr = vec![0.0f32; w * h];
    let mut dx_rl = vec![0.0f32; w * h];

    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * num_d;
            let mut best_c = f32::INFINITY;
            let mut best_d = 0;
            for di in 0..num_d {
                let c = agg[base + di];
                if c < best_c {
                    best_c = c;
                    best_d = min_d + di as i32;
                }
            }
            dx_lr[y * w + x] = best_d as f32;

            // Right pixel x matches left pixel x_l = x - d  => cost is agg[y][x_l][d]
            let mut best_rc = f32::INFINITY;
            let mut best_rd = 0i32;
            for di in 0..num_d {
                let d = min_d + di as i32;
                let xl = x as i32 - d;
                if xl >= 0 && xl < w as i32 {
                    let c = agg[(y * w + xl as usize) * num_d + di];
                    if c < best_rc {
                        best_rc = c;
                        best_rd = -d;
                    }
                }
            }
            dx_rl[y * w + x] = best_rd as f32;
        }
    }

    let mut disp = DisparityMap::new(left.width, left.height);
    apply_lr_check(left.width, left.height, &dx_lr, &dx_rl, s.lr_threshold, &mut disp);
    (disp, used_vk)
}

fn build_cost_volume_cpu(
    left: &ImageF32,
    right: &ImageF32,
    s: &StereoSettings,
    min_d: i32,
    num_d: usize,
    hkw: i32,
    hkh: i32,
    rows_per: usize,
) -> Vec<f32> {
    let w = left.width as usize;
    let h = left.height as usize;
    let census_l = compute_census_5x5(left);
    let census_r = compute_census_5x5(right);

    let mut cost_vol = vec![1.0f32; w * h * num_d];
    std::thread::scope(|scope| {
        for (t, chunk) in cost_vol.chunks_mut(rows_per * w * num_d).enumerate() {
            let y_start = t * rows_per;
            let cl = &census_l;
            let cr = &census_r;
            scope.spawn(move || {
                let rows = chunk.len() / (w * num_d);
                for ly in 0..rows {
                    let y = (y_start + ly) as i32;
                    for x in 0..w as i32 {
                        let base = (ly * w + x as usize) * num_d;
                        for di in 0..num_d {
                            let d = min_d + di as i32;
                            if x + d < 0 || x + d >= w as i32 {
                                chunk[base + di] = 2.0;
                                continue;
                            }
                            let c = match s.cost_mode {
                                CostMode::Census => {
                                    census_patch_cost(cl, cr, w as i32, h as i32, x, y, d, 0, hkw.min(2), hkh.min(2))
                                }
                                _ => {
                                    let pc = patch_cost(left, right, x, y, d, 0, hkw, hkh, s.cost_mode);
                                    let cc = census_patch_cost(cl, cr, w as i32, h as i32, x, y, d, 0, 1, 1);
                                    0.7 * pc + 0.3 * cc
                                }
                            };
                            chunk[base + di] = c;
                        }
                    }
                }
            });
        }
    });
    cost_vol
}

fn apply_lr_check(
    w: u32,
    h: u32,
    dx_lr: &[f32],
    dx_rl: &[f32],
    lr_thresh: f32,
    out: &mut DisparityMap,
) {
    let wu = w as usize;
    for y in 0..h as usize {
        for x in 0..wu {
            let idx = y * wu + x;
            let d = dx_lr[idx];
            let rx = (x as f32 + d).round() as i32;
            if rx < 0 || rx >= w as i32 {
                out.dx[idx] = d;
                out.valid[idx] = 0;
                continue;
            }
            let d_back = dx_rl[y * wu + rx as usize];
            let consistent = (d + d_back).abs() <= lr_thresh;
            out.dx[idx] = d;
            out.valid[idx] = if consistent { 1 } else { 0 };
        }
    }
}
