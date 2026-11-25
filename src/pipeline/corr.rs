//! Stage 1: `stereo_corr` — Integer stereo correlation (ASP BM, ASP SGM 4/8-path, ASP MGM).
//! Supports SAD, SSD, zero-mean NCC, and 5x5 Census cost metrics + Left-Right consistency check.
//! Executes on Vulkan Compute GPU when available, or multi-threaded CPU (`std::thread::scope`).

use crate::config::{CostMode, StereoAlgorithm, StereoSettings, SubpixelMode};
use crate::image::{DisparityMap, ImageF32};
use crate::vulkan::VulkanCompute;

/// Run Stage 1 stereo correlation returning a disparity map and a flag indicating Vulkan GPU execution.
pub fn run_corr(
    left: &ImageF32,
    right: &ImageF32,
    settings: &StereoSettings,
    vk: Option<&VulkanCompute>,
) -> (DisparityMap, bool) {
    run_corr_with_guide(left, right, None, settings, vk)
}

/// Run Stage 1 stereo correlation with an optional raw radiometric left guide image for edge-aware SGM.
pub fn run_corr_with_guide(
    left: &ImageF32,
    right: &ImageF32,
    left_guide: Option<&ImageF32>,
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
        StereoAlgorithm::AspSgm => correlate_sgm(left, right, left_guide, settings, false, vk),
        StereoAlgorithm::AspMgm => correlate_sgm(left, right, left_guide, settings, true, vk),
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
            // Physical noise-floor variance damping (sigma_n = 0.025) prevents noise amplification on flat roofs/asphalt
            let var_floor = 0.000625f32;
            let denom = ((var_l + var_floor) * (var_r + var_floor)).sqrt();
            1.0 - (cov / denom).clamp(-1.0, 1.0)
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
/// 3D cost volume acceleration (with automatic AVX2 multi-thread CPU fallback), 8-path edge-aware
/// dynamic programming using raw radiometric guide gradients, texture-aware uniqueness filtering,
/// LR consistency check, and exact subpixel parabola fitting on the aggregated SGM cost volume.
fn correlate_sgm(
    left: &ImageF32,
    right: &ImageF32,
    left_guide: Option<&ImageF32>,
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

    let guide = match left_guide {
        Some(g) if g.width == left.width && g.height == left.height => g,
        _ => left,
    };

    // 1. Build 3D matching cost volume C[y][x][d] on Vulkan GPU (supports SAD, SSD, hybrid NCC+Census, Census)
    let mut used_vk = false;
    let cost_vol: Vec<f32> = if let Some(gpu) = vk {
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
            Err(e) => {
                eprintln!("[WARN] Vulkan 3D cost volume fallback to CPU: {}", e);
                build_cost_volume_cpu(left, right, s, min_d, num_d, hkw, hkh, rows_per)
            }
            Ok(_) => build_cost_volume_cpu(left, right, s, min_d, num_d, hkw, hkh, rows_per),
        }
    } else {
        build_cost_volume_cpu(left, right, s, min_d, num_d, hkw, hkh, rows_per)
    };

    // 2. 8-path Edge-Aware SGM / MGM cost aggregation using AVX2 SIMD + L2-cache-friendly 2-row DP buffers
    //    Uses raw radiometric guide gradients so P2 stays strong on flat asphalt and drops cleanly at real edges.
    let p1 = s.sgm_p1;
    let p2_base = s.sgm_p2;
    let mut agg = vec![0.0f32; w * h * num_d];
    let stride = num_d + 2;

    // 2a. Horizontal passes (+X and -X) parallelized across all CPU threads with AVX2 SIMD
    std::thread::scope(|scope| {
        for (t, agg_chunk) in agg.chunks_mut(rows_per * w * num_d).enumerate() {
            let y_start = t * rows_per;
            let cv = &cost_vol;
            let gdata = &guide.data;
            scope.spawn(move || {
                let rows = agg_chunk.len() / (w * num_d);
                let mut prev_l = vec![f32::INFINITY; stride];
                let mut cur_l = vec![f32::INFINITY; stride];
                for ly in 0..rows {
                    let y = y_start + ly;
                    let row_pix = y * w;
                    let row_cv = row_pix * num_d;
                    let row_agg = ly * w * num_d;

                    // Forward +X pass
                    let mut min_prev = f32::INFINITY;
                    for d in 0..num_d {
                        let c = cv[row_cv + d];
                        prev_l[d + 1] = c;
                        agg_chunk[row_agg + d] += c;
                        if c < min_prev { min_prev = c; }
                    }
                    for x in 1..w {
                        let grad = (gdata[row_pix + x] - gdata[row_pix + x - 1]).abs();
                        let (p1_eff, p2_eff) = adaptive_sgm_penalties(p1, p2_base, grad);
                        let c_off = row_cv + x * num_d;
                        let a_off = row_agg + x * num_d;
                        min_prev = crate::pipeline::simd::sgm_dp_step_add(
                            &cv[c_off..c_off + num_d],
                            &prev_l,
                            &mut cur_l,
                            &mut agg_chunk[a_off..a_off + num_d],
                            min_prev,
                            p1_eff,
                            p2_eff,
                        );
                        std::mem::swap(&mut prev_l, &mut cur_l);
                    }

                    // Backward -X pass
                    let last_off = (w - 1) * num_d;
                    min_prev = f32::INFINITY;
                    for d in 0..num_d {
                        let c = cv[row_cv + last_off + d];
                        prev_l[d + 1] = c;
                        agg_chunk[row_agg + last_off + d] += c;
                        if c < min_prev { min_prev = c; }
                    }
                    for x in (0..w - 1).rev() {
                        let grad = (gdata[row_pix + x] - gdata[row_pix + x + 1]).abs();
                        let (p1_eff, p2_eff) = adaptive_sgm_penalties(p1, p2_base, grad);
                        let c_off = row_cv + x * num_d;
                        let a_off = row_agg + x * num_d;
                        min_prev = crate::pipeline::simd::sgm_dp_step_add(
                            &cv[c_off..c_off + num_d],
                            &prev_l,
                            &mut cur_l,
                            &mut agg_chunk[a_off..a_off + num_d],
                            min_prev,
                            p1_eff,
                            p2_eff,
                        );
                        std::mem::swap(&mut prev_l, &mut cur_l);
                    }
                }
            });
        }
    });

    // 2b. Vertical & Diagonal passes (+Y, +1/+1, -1/+1 and -Y, +1/-1, -1/-1)
    if !use_mgm {
        // Multi-threaded AVX2 fused 3-direction sweeps (cuts DRAM traffic 3x + parallelizes across CPU cores)
        run_sgm_fused_vd_sweep(&cost_vol, &guide.data, &mut agg, w, h, num_d, p1, p2_base, true, s.threads);
        run_sgm_fused_vd_sweep(&cost_vol, &guide.data, &mut agg, w, h, num_d, p1, p2_base, false, s.threads);
    } else {
        // MGM 2D-quadrant passes with compact 2-row L2-cache buffers
        let vd_dirs: &[(i32, i32)] = &[(0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)];
        let mut prev_row = vec![0.0f32; w * num_d];
        let mut cur_row = vec![0.0f32; w * num_d];
        let mut prev_mins = vec![0.0f32; w];

        for &(dx_dir, dy_dir) in vd_dirs {
            let y_iter: Vec<usize> = if dy_dir > 0 { (0..h).collect() } else { (0..h).rev().collect() };
            let x_iter: Vec<usize> = if dx_dir >= 0 { (0..w).collect() } else { (0..w).rev().collect() };
            let x_step = if dx_dir >= 0 { 1i32 } else { -1i32 };

            for (step_y, &y) in y_iter.iter().enumerate() {
                let row_pix = y * w;
                let py = y as i32 - dy_dir;
                let has_prev_row = step_y > 0 && py >= 0 && py < h as i32;

                if has_prev_row {
                    for x in 0..w {
                        let off = x * num_d;
                        let mut mn = f32::INFINITY;
                        for d in 0..num_d {
                            if prev_row[off + d] < mn { mn = prev_row[off + d]; }
                        }
                        prev_mins[x] = mn;
                    }
                }

                for &x in &x_iter {
                    let px = x as i32 - dx_dir;
                    let idx = (row_pix + x) * num_d;
                    let dst = x * num_d;

                    if !has_prev_row || px < 0 || px >= w as i32 {
                        for d in 0..num_d {
                            let c = cost_vol[idx + d];
                            cur_row[dst + d] = c;
                            agg[idx + d] += c;
                        }
                    } else {
                        let px_u = px as usize;
                        let p_off = px_u * num_d;
                        let min_prev = prev_mins[px_u];
                        let grad = (guide.data[row_pix + x] - guide.data[py as usize * w + px_u]).abs();
                        let (p1_eff, p2_eff) = adaptive_sgm_penalties(p1, p2_base, grad);
                        let mp2 = min_prev + p2_eff;

                        let qx = x as i32 - x_step;
                        let (q_valid, q_off, min_q) = if qx >= 0 && qx < w as i32 {
                            let qo = qx as usize * num_d;
                            let mut mq = f32::INFINITY;
                            for d in 0..num_d {
                                if cur_row[qo + d] < mq { mq = cur_row[qo + d]; }
                            }
                            (true, qo, mq)
                        } else {
                            (false, 0, 0.0)
                        };
                        let mq2 = min_q + p2_eff;

                        for d in 0..num_d {
                            let c0 = prev_row[p_off + d];
                            let cm = if d > 0 { prev_row[p_off + d - 1] + p1_eff } else { f32::INFINITY };
                            let cp = if d + 1 < num_d { prev_row[p_off + d + 1] + p1_eff } else { f32::INFINITY };
                            let best_p = c0.min(cm).min(cp).min(mp2) - min_prev;

                            let smooth = if q_valid {
                                let q0 = cur_row[q_off + d];
                                let qm = if d > 0 { cur_row[q_off + d - 1] + p1_eff } else { f32::INFINITY };
                                let qp = if d + 1 < num_d { cur_row[q_off + d + 1] + p1_eff } else { f32::INFINITY };
                                let best_q = q0.min(qm).min(qp).min(mq2) - min_q;
                                0.5 * (best_p + best_q)
                            } else {
                                best_p
                            };
                            let v = cost_vol[idx + d] + smooth;
                            cur_row[dst + d] = v;
                            agg[idx + d] += v;
                        }
                    }
                }
                std::mem::swap(&mut prev_row, &mut cur_row);
            }
        }
    }

    // 3. Parallel WTA + Aggregated-Cost Subpixel Parabola + Texture-Aware Uniqueness + Implicit R->L WTA
    let do_subpixel = s.subpixel_mode != SubpixelMode::None;
    let mut dx_lr_int = vec![0.0f32; w * h];
    let mut dx_lr_sub = vec![0.0f32; w * h];
    let mut dx_rl = vec![0.0f32; w * h];
    let mut unique_ok = vec![1u8; w * h];

    std::thread::scope(|scope| {
        for (t, (((lr_i, lr_s), rl_c), uq_c)) in dx_lr_int
            .chunks_mut(rows_per * w)
            .zip(dx_lr_sub.chunks_mut(rows_per * w))
            .zip(dx_rl.chunks_mut(rows_per * w))
            .zip(unique_ok.chunks_mut(rows_per * w))
            .enumerate()
        {
            let y_start = t * rows_per;
            let ag = &agg;
            let cv = &cost_vol;
            let gdata = &guide.data;
            scope.spawn(move || {
                let rows = lr_i.len() / w;
                for ly in 0..rows {
                    let y = y_start + ly;
                    let ym = y.saturating_sub(1);
                    let yp = (y + 1).min(h - 1);
                    for x in 0..w {
                        let base = (y * w + x) * num_d;
                        let mut best_c = f32::INFINITY;
                        let mut best_di = 0usize;
                        for di in 0..num_d {
                            let c = ag[base + di];
                            if c < best_c {
                                best_c = c;
                                best_di = di;
                            }
                        }

                        // Unbiased subpixel parabola around SGM-regularized `best_di`:
                        // Prefer raw `cost_vol` when strictly convex (avoids SGM +16*P1 pixel-locking bias),
                        // falling back to strictly convex `agg` parabola on low-texture surfaces.
                        let d_int = (min_d + best_di as i32) as f32;
                        let mut d_sub = d_int;
                        if do_subpixel && best_di > 0 && best_di + 1 < num_d {
                            let rc0 = cv[base + best_di];
                            let rcm = cv[base + best_di - 1];
                            let rcp = cv[base + best_di + 1];
                            let rden = rcm - 2.0 * rc0 + rcp;
                            if rden > 1e-5 && rcm >= rc0 && rcp >= rc0 {
                                d_sub += (0.5 * (rcm - rcp) / rden).clamp(-0.5, 0.5);
                            } else {
                                let cm = ag[base + best_di - 1];
                                let cp = ag[base + best_di + 1];
                                let den = cm - 2.0 * best_c + cp;
                                if den > 1e-6 && cm >= best_c && cp >= best_c {
                                    d_sub += (0.5 * (cm - cp) / den).clamp(-0.5, 0.5);
                                }
                            }
                        }

                        // Local radiometric texture contrast (3x3 max neighbor diff on guide image)
                        let xm = x.saturating_sub(1);
                        let xp = (x + 1).min(w - 1);
                        let gc = gdata[y * w + x];
                        let local_tex = (gdata[y * w + xm] - gc)
                            .abs()
                            .max((gdata[y * w + xp] - gc).abs())
                            .max((gdata[ym * w + x] - gc).abs())
                            .max((gdata[yp * w + x] - gc).abs());

                        // Texture-aware uniqueness check: reject ambiguous periodic peaks on textured regions
                        // and search-boundary locks, while letting SGM smoothly bridge uniform asphalt/stalls
                        let mut second_c = f32::INFINITY;
                        for di in 0..num_d {
                            if di.abs_diff(best_di) >= 4 {
                                let c = ag[base + di];
                                if c < second_c {
                                    second_c = c;
                                }
                            }
                        }

                        let loc = ly * w + x;
                        lr_i[loc] = d_int;
                        lr_s[loc] = d_sub;
                        if num_d >= 16 && second_c.is_finite() && best_c > 1e-4 {
                            let at_border = best_di == 0 || best_di + 1 == num_d;
                            if at_border && second_c < best_c * 1.01 {
                                uq_c[loc] = 0;
                            } else if local_tex > 0.035 && second_c < best_c * 1.0035 {
                                uq_c[loc] = 0;
                            }
                        }

                        let mut best_rc = f32::INFINITY;
                        let mut best_rd = 0i32;
                        for di in 0..num_d {
                            let d = min_d + di as i32;
                            let xl = x as i32 - d;
                            if xl >= 0 && xl < w as i32 {
                                let c = ag[(y * w + xl as usize) * num_d + di];
                                if c < best_rc {
                                    best_rc = c;
                                    best_rd = -d;
                                }
                            }
                        }
                        rl_c[loc] = best_rd as f32;
                    }
                }
            });
        }
    });

    let mut disp = DisparityMap::new(left.width, left.height);
    apply_lr_check(left.width, left.height, &dx_lr_int, &dx_rl, s.lr_threshold, &mut disp);
    for i in 0..disp.valid.len() {
        disp.dx[i] = dx_lr_sub[i];
        if unique_ok[i] == 0 {
            disp.valid[i] = 0;
        }
    }
    disp.is_subpixel = do_subpixel;
    (disp, used_vk)
}

#[derive(Clone, Copy)]
struct SyncMutF32Ptr(*mut f32);
// SAFETY: Used exclusively inside `run_sgm_fused_vd_sweep` where each worker owns a strictly
// disjoint column range `[x0..x1)` for writes, and double-buffered parity buffers are separated
// by an acquire/release `std::sync::Barrier` between rows.
unsafe impl Send for SyncMutF32Ptr {}
unsafe impl Sync for SyncMutF32Ptr {}

/// Texture-adaptive SGM regularization: reinforces P1/P2 on radiometrically flat surfaces
/// (`grad < 0.015`: white roofs, smooth asphalt, lawns) to suppress wavy blobs, while releasing
/// P2 sharply across real image edges (`grad > 0.04`).
#[inline(always)]
fn adaptive_sgm_penalties(p1_base: f32, p2_base: f32, grad: f32) -> (f32, f32) {
    let flat_boost = if grad < 0.015 {
        1.0 + 0.85 * (1.0 - grad / 0.015)
    } else {
        1.0
    };
    let p1_eff = p1_base * flat_boost;
    let p2_eff = (p2_base * flat_boost / (1.0 + 12.0 * grad)).max(p1_eff * 1.5);
    (p1_eff, p2_eff)
}

/// Multi-threaded AVX2 fused 3-direction SGM sweep:
/// - `downward == true`:  fuses `(0, +1)`, `(+1, +1)`, `(-1, +1)` in ONE top-to-bottom scan
/// - `downward == false`: fuses `(0, -1)`, `(+1, -1)`, `(-1, -1)` in ONE bottom-to-top scan
/// Reduces `cost_vol` + `agg` DRAM bandwidth by 3x and parallelizes across CPU cores via a single `Barrier` scope.
#[allow(clippy::too_many_arguments)]
fn run_sgm_fused_vd_sweep(
    cost_vol: &[f32],
    guide: &[f32],
    agg: &mut [f32],
    w: usize,
    h: usize,
    num_d: usize,
    p1: f32,
    p2_base: f32,
    downward: bool,
    threads: usize,
) {
    if w == 0 || h == 0 || num_d == 0 {
        return;
    }
    let stride = num_d + 2;
    let dir_stride = w * stride;
    // Double-buffered ping-pong buffers holding 3 directions [v, dl, dr] per parity (0 or 1)
    let mut bufs = [vec![f32::INFINITY; 3 * dir_stride], vec![f32::INFINITY; 3 * dir_stride]];
    let mut mins = [vec![0.0f32; 3 * w], vec![0.0f32; 3 * w]];

    let n_workers = threads.clamp(1, 8).min((w / 32).max(1));
    let cols_per = (w + n_workers - 1) / n_workers;
    let active_workers = (w + cols_per - 1) / cols_per;
    let barrier = std::sync::Barrier::new(active_workers);

    let agg_ptr = SyncMutF32Ptr(agg.as_mut_ptr());
    let buf0_ptr = SyncMutF32Ptr(bufs[0].as_mut_ptr());
    let buf1_ptr = SyncMutF32Ptr(bufs[1].as_mut_ptr());
    let min0_ptr = SyncMutF32Ptr(mins[0].as_mut_ptr());
    let min1_ptr = SyncMutF32Ptr(mins[1].as_mut_ptr());

    std::thread::scope(|scope| {
        for t in 0..active_workers {
            let x0 = t * cols_per;
            let x1 = (x0 + cols_per).min(w);
            let bar = &barrier;
            scope.spawn(move || {
                let agg_p = agg_ptr;
                let buf_ptrs = [buf0_ptr, buf1_ptr];
                let min_ptrs = [min0_ptr, min1_ptr];

                for step_y in 0..h {
                    let y = if downward { step_y } else { h - 1 - step_y };
                    let py = if downward { y.wrapping_sub(1) } else { y + 1 };
                    let cur_p = step_y & 1;
                    let prev_p = 1 - cur_p;

                    let row_pix = y * w;
                    let prev_row_pix = py.wrapping_mul(w);

                    // SAFETY: All workers read only `prev_p` (immutable after previous barrier)
                    // and write exclusively to disjoint column slice `[x0..x1)` in `cur_p` and `agg[y]`.
                    let prev_buf = unsafe { std::slice::from_raw_parts(buf_ptrs[prev_p].0, 3 * dir_stride) };
                    let prev_min = unsafe { std::slice::from_raw_parts(min_ptrs[prev_p].0, 3 * w) };
                    let cur_buf_ptr = buf_ptrs[cur_p].0;
                    let cur_min_ptr = min_ptrs[cur_p].0;
                    let agg_row_ptr = unsafe { agg_p.0.add(row_pix * num_d) };

                    for x in x0..x1 {
                        let cv_off = (row_pix + x) * num_d;
                        let cv_s = &cost_vol[cv_off..cv_off + num_d];
                        let agg_s = unsafe { std::slice::from_raw_parts_mut(agg_row_ptr.add(x * num_d), num_d) };
                        let cur_v = unsafe { std::slice::from_raw_parts_mut(cur_buf_ptr.add(x * stride), stride) };
                        let cur_dl = unsafe { std::slice::from_raw_parts_mut(cur_buf_ptr.add(dir_stride + x * stride), stride) };
                        let cur_dr = unsafe { std::slice::from_raw_parts_mut(cur_buf_ptr.add(2 * dir_stride + x * stride), stride) };

                        if step_y == 0 {
                            let mut mn = f32::INFINITY;
                            for d in 0..num_d {
                                let c = cv_s[d];
                                cur_v[d + 1] = c;
                                cur_dl[d + 1] = c;
                                cur_dr[d + 1] = c;
                                agg_s[d] += 3.0 * c;
                                if c < mn { mn = c; }
                            }
                            unsafe {
                                *cur_min_ptr.add(x) = mn;
                                *cur_min_ptr.add(w + x) = mn;
                                *cur_min_ptr.add(2 * w + x) = mn;
                            }
                        } else {
                            let gx = guide[row_pix + x];

                            // Direction 0: Vertical (0, ±1) from (x, py)
                            let grad_v = (gx - guide[prev_row_pix + x]).abs();
                            let (p1_v, p2_v) = adaptive_sgm_penalties(p1, p2_base, grad_v);
                            let mv = crate::pipeline::simd::sgm_dp_step_add(
                                cv_s,
                                &prev_buf[x * stride..(x + 1) * stride],
                                cur_v,
                                agg_s,
                                prev_min[x],
                                p1_v,
                                p2_v,
                            );

                            // Direction 1: Diagonal (+1, ±1) from (x - 1, py)
                            let mdl = if x > 0 {
                                let px = x - 1;
                                let grad_dl = (gx - guide[prev_row_pix + px]).abs();
                                let (p1_dl, p2_dl) = adaptive_sgm_penalties(p1, p2_base, grad_dl);
                                crate::pipeline::simd::sgm_dp_step_add(
                                    cv_s,
                                    &prev_buf[dir_stride + px * stride..dir_stride + (px + 1) * stride],
                                    cur_dl,
                                    agg_s,
                                    prev_min[w + px],
                                    p1_dl,
                                    p2_dl,
                                )
                            } else {
                                let mut mn = f32::INFINITY;
                                for d in 0..num_d {
                                    let c = cv_s[d];
                                    cur_dl[d + 1] = c;
                                    agg_s[d] += c;
                                    if c < mn { mn = c; }
                                }
                                mn
                            };

                            // Direction 2: Diagonal (-1, ±1) from (x + 1, py)
                            let mdr = if x + 1 < w {
                                let px = x + 1;
                                let grad_dr = (gx - guide[prev_row_pix + px]).abs();
                                let (p1_dr, p2_dr) = adaptive_sgm_penalties(p1, p2_base, grad_dr);
                                crate::pipeline::simd::sgm_dp_step_add(
                                    cv_s,
                                    &prev_buf[2 * dir_stride + px * stride..2 * dir_stride + (px + 1) * stride],
                                    cur_dr,
                                    agg_s,
                                    prev_min[2 * w + px],
                                    p1_dr,
                                    p2_dr,
                                )
                            } else {
                                let mut mn = f32::INFINITY;
                                for d in 0..num_d {
                                    let c = cv_s[d];
                                    cur_dr[d + 1] = c;
                                    agg_s[d] += c;
                                    if c < mn { mn = c; }
                                }
                                mn
                            };

                            unsafe {
                                *cur_min_ptr.add(x) = mv;
                                *cur_min_ptr.add(w + x) = mdl;
                                *cur_min_ptr.add(2 * w + x) = mdr;
                            }
                        }
                    }

                    bar.wait();
                }
            });
        }
    });
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
    // Precompute local 3x3 variance on left image for texture-gated Census matching GPU shader
    let mut left_var = vec![0.0f32; w * h];
    for y in 0..h {
        let y0 = y.saturating_sub(1);
        let y1 = (y + 1).min(h - 1);
        for x in 0..w {
            let x0 = x.saturating_sub(1);
            let x1 = (x + 1).min(w - 1);
            let mut s1 = 0.0f32;
            let mut s2 = 0.0f32;
            let mut cnt = 0.0f32;
            for ny in y0..=y1 {
                for nx in x0..=x1 {
                    let v = left.data[ny * w + nx];
                    s1 += v;
                    s2 += v * v;
                    cnt += 1.0;
                }
            }
            let m = s1 / cnt;
            left_var[y * w + x] = (s2 / cnt - m * m).max(0.0);
        }
    }

    let mut cost_vol = vec![1.0f32; w * h * num_d];
    std::thread::scope(|scope| {
        for (t, chunk) in cost_vol.chunks_mut(rows_per * w * num_d).enumerate() {
            let y_start = t * rows_per;
            let cl = &census_l;
            let cr = &census_r;
            let lv_ref = &left_var;
            scope.spawn(move || {
                let rows = chunk.len() / (w * num_d);
                for ly in 0..rows {
                    let y = (y_start + ly) as i32;
                    for x in 0..w as i32 {
                        let base = (ly * w + x as usize) * num_d;
                        let lv = lv_ref[(y as usize) * w + (x as usize)];
                        let tex_conf = lv / (lv + 0.000625);
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
                                CostMode::Ncc => {
                                    let pc = patch_cost(left, right, x, y, d, 0, hkw, hkh, s.cost_mode);
                                    let cc = census_patch_cost(cl, cr, w as i32, h as i32, x, y, d, 0, 1, 1);
                                    let cc_gated = 0.5 + tex_conf * (cc - 0.5);
                                    0.65 * pc + 0.35 * cc_gated
                                }
                                _ => patch_cost(left, right, x, y, d, 0, hkw, hkh, s.cost_mode),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::pprc;

    #[test]
    fn test_sgm_aggregated_subpixel_fractional_continuity() {
        let w = 64u32;
        let h = 48u32;
        let mut left = ImageF32::new(w, h);
        let mut right = ImageF32::new(w, h);
        let true_dx = -3.4f32;
        let tex = |x: f32, y: f32| -> f32 {
            0.5 + 0.25 * (x * 0.45).sin() * (y * 0.37).cos()
                + 0.15 * (x * 1.13 + y * 0.71).sin()
                + 0.10 * ((x * 0.83).cos() - (y * 0.91).sin())
        };
        for y in 0..h {
            for x in 0..w {
                left.set(x, y, tex(x as f32, y as f32));
                right.set(x, y, tex(x as f32 - true_dx, y as f32));
            }
        }
        let mut s = StereoSettings::default();
        s.stereo_algorithm = StereoAlgorithm::AspSgm;
        s.cost_mode = CostMode::Ncc;
        s.subpixel_mode = SubpixelMode::Parabola;
        s.search_min_x = -12;
        s.search_max_x = 4;
        s.search_min_y = 0;
        s.search_max_y = 0;
        s.corr_kernel_x = 7;
        s.corr_kernel_y = 7;

        let (lp, rp, _) = pprc::run_pprc(&left, &right, &s, None);
        let (disp, _) = run_corr_with_guide(&lp, &rp, Some(&left), &s, None);
        assert!(disp.is_subpixel);

        let mut err_sum = 0.0f64;
        let mut frac_cnt = 0usize;
        let mut cnt = 0usize;
        for y in 8..(h - 8) {
            for x in 14..(w - 14) {
                let idx = disp.idx(x, y);
                if disp.valid[idx] != 0 {
                    let d = disp.dx[idx];
                    err_sum += (d - true_dx).abs() as f64;
                    if (d - d.round()).abs() > 0.15 {
                        frac_cnt += 1;
                    }
                    cnt += 1;
                }
            }
        }
        assert!(cnt > 800, "Too few valid interior pixels: {}", cnt);
        let mae = err_sum / cnt as f64;
        assert!(mae < 0.15, "Subpixel SGM MAE too high for -3.4 px shift: {:.4}", mae);
        assert!(frac_cnt * 2 > cnt, "Expected continuous fractional subpixel estimates, got {}/{}", frac_cnt, cnt);
    }

    #[test]
    fn test_fused_sgm_matches_reference_6_passes() {
        let w = 96usize;
        let h = 40usize;
        let num_d = 25usize;
        let p1 = 0.12f32;
        let p2_base = 0.48f32;

        let mut cv = vec![0.0f32; w * h * num_d];
        let mut guide = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                guide[y * w + x] = ((x * 7 + y * 13) % 100) as f32 / 100.0;
                for d in 0..num_d {
                    cv[(y * w + x) * num_d + d] =
                        0.1 + (((x * 31 + y * 17 + d * 11) % 97) as f32) / 100.0;
                }
            }
        }

        // Reference sequential 6-direction aggregation
        let mut agg_ref = vec![0.0f32; w * h * num_d];
        for &(dx_dir, dy_dir) in &[(0i32, 1i32), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)] {
            let mut prev = vec![0.0f32; w * num_d];
            let mut cur = vec![0.0f32; w * num_d];
            let y_iter: Vec<usize> = if dy_dir > 0 { (0..h).collect() } else { (0..h).rev().collect() };
            for (step_y, &y) in y_iter.iter().enumerate() {
                let py = y as i32 - dy_dir;
                let has_prev = step_y > 0 && py >= 0 && py < h as i32;
                for x in 0..w {
                    let px = x as i32 - dx_dir;
                    let idx = (y * w + x) * num_d;
                    if !has_prev || px < 0 || px >= w as i32 {
                        for d in 0..num_d {
                            let c = cv[idx + d];
                            cur[x * num_d + d] = c;
                            agg_ref[idx + d] += c;
                        }
                    } else {
                        let px_u = px as usize;
                        let po = px_u * num_d;
                        let mut mn = f32::INFINITY;
                        for d in 0..num_d {
                            if prev[po + d] < mn { mn = prev[po + d]; }
                        }
                        let grad = (guide[y * w + x] - guide[py as usize * w + px_u]).abs();
                        let (p1_eff, p2_eff) = adaptive_sgm_penalties(p1, p2_base, grad);
                        let mp2 = mn + p2_eff;
                        for d in 0..num_d {
                            let c0 = prev[po + d];
                            let cm = if d > 0 { prev[po + d - 1] + p1_eff } else { f32::INFINITY };
                            let cp = if d + 1 < num_d { prev[po + d + 1] + p1_eff } else { f32::INFINITY };
                            let v = cv[idx + d] + (c0.min(cm).min(cp).min(mp2) - mn);
                            cur[x * num_d + d] = v;
                            agg_ref[idx + d] += v;
                        }
                    }
                }
                std::mem::swap(&mut prev, &mut cur);
            }
        }

        // Parallel fused AVX2 downward + upward sweeps (4 workers)
        let mut agg_fused = vec![0.0f32; w * h * num_d];
        run_sgm_fused_vd_sweep(&cv, &guide, &mut agg_fused, w, h, num_d, p1, p2_base, true, 4);
        run_sgm_fused_vd_sweep(&cv, &guide, &mut agg_fused, w, h, num_d, p1, p2_base, false, 4);

        let mut max_diff = 0.0f32;
        for i in 0..agg_ref.len() {
            let diff = (agg_ref[i] - agg_fused[i]).abs();
            if diff > max_diff { max_diff = diff; }
        }
        assert!(max_diff < 1e-4, "Fused SGM diverged from reference: max_diff={}", max_diff);
    }
}
