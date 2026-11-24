//! Stage 3: `stereo_fltr` — Disparity outlier erosion, median smoothing, and hole filling.

use crate::config::StereoSettings;
use crate::image::DisparityMap;

/// Run ASP-style disparity filtering:
/// 1. Multi-pass neighbor consistency erosion (`rm-half-kernel`, `rm-threshold`, `rm-min-matches`, `max-mean-diff`)
/// 2. Median filter smoothing
/// 3. Optional small-gap hole interpolation
pub fn run_fltr(disp_in: &DisparityMap, s: &StereoSettings) -> DisparityMap {
    let mut cur = disp_in.clone();

    if s.filter_mode > 0 {
        for _ in 0..s.rm_cleanup_passes.max(1) {
            cur = outlier_erode(&cur, s);
        }
        let max_speckle = ((cur.width as usize * cur.height as usize) / 4096).clamp(16, 128);
        remove_small_speckles(&mut cur, max_speckle, s.rm_threshold.min(2.0).max(1.0));
    }

    if s.median_filter_size >= 3 {
        cur = median_filter(&cur, (s.median_filter_size / 2) as i32);
    }

    if s.fill_holes {
        cur = fill_small_holes(&cur, 3);
    }

    cur
}

fn outlier_erode(disp: &DisparityMap, s: &StereoSettings) -> DisparityMap {
    let w = disp.width as i32;
    let h = disp.height as i32;
    let hkx = s.rm_half_kernel_x.max(1);
    let hky = s.rm_half_kernel_y.max(1);
    let min_frac = (s.rm_min_matches as f32 / 100.0).clamp(0.05, 0.95);

    let mut out = disp.clone();
    for y in 0..h {
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
                        if (ndx - cx).abs() <= s.rm_threshold && (ndy - cy).abs() <= s.rm_threshold {
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
            if frac < min_frac || mean_diff > s.max_mean_diff {
                out.valid[idx] = 0;
            }
        }
    }
    out
}

fn median_filter(disp: &DisparityMap, rad: i32) -> DisparityMap {
    let w = disp.width as i32;
    let h = disp.height as i32;
    let mut out = disp.clone();
    let mut vals_x = Vec::with_capacity(((2 * rad + 1) * (2 * rad + 1)) as usize);
    let mut vals_y = Vec::with_capacity(((2 * rad + 1) * (2 * rad + 1)) as usize);

    for y in 0..h {
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
                out.dx[idx] = vals_x[mid];
                out.dy[idx] = vals_y[mid];
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
                        sum_x += disp.dx[nidx] * wt;
                        sum_y += disp.dy[nidx] * wt;
                        w_sum += wt;
                        count += 1;
                    }
                }
            }
            if count >= 4 && w_sum > 0.0 {
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
