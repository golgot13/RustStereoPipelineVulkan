//! Stage 2: `stereo_rfne` — Subpixel disparity refinement.
//! Implements:
//! - Mode 0: Disabled
//! - Mode 1: Parabola subpixel peak fitting
//! - Mode 2: Bayes-EM robust weighted Lucas-Kanade / Affine Gauss-Newton refinement
//! - Mode 3: Affine Gauss-Newton subpixel refinement

use crate::config::{CostMode, StereoSettings, SubpixelMode};
use crate::image::{DisparityMap, ImageF32};
use crate::pipeline::corr::patch_cost;

pub fn run_rfne(
    left: &ImageF32,
    right: &ImageF32,
    disp_in: &DisparityMap,
    settings: &StereoSettings,
) -> DisparityMap {
    run_rfne_with_raw(left, right, left, right, disp_in, settings)
}

/// Subpixel refinement using clean raw radiometric images for continuous ZNCC parabola fitting
/// (eliminates discrete Census Hamming quantization noise on flat surfaces while keeping SGM integer accuracy).
pub fn run_rfne_with_raw(
    left_pprc: &ImageF32,
    right_pprc: &ImageF32,
    left_raw: &ImageF32,
    right_raw: &ImageF32,
    disp_in: &DisparityMap,
    settings: &StereoSettings,
) -> DisparityMap {
    match settings.subpixel_mode {
        SubpixelMode::None => disp_in.clone(),
        SubpixelMode::Parabola => refine_parabola_pure_zncc(left_raw, right_raw, disp_in, settings),
        SubpixelMode::Affine => refine_lucas_kanade(left_pprc, right_pprc, disp_in, settings, false),
        SubpixelMode::BayesEmAffine => refine_lucas_kanade(left_pprc, right_pprc, disp_in, settings, true),
    }
}

/// Quadratic parabola fitting around integer disparity `(dx, dy)` using pure continuous ZNCC
/// (zero discrete Census bit-flipping noise) on raw images.
fn refine_parabola_pure_zncc(
    left: &ImageF32,
    right: &ImageF32,
    disp_in: &DisparityMap,
    s: &StereoSettings,
) -> DisparityMap {
    let w = left.width as usize;
    let h = left.height as usize;
    let hkw = (s.subpixel_kernel_x / 2).clamp(1, 5) as i32;
    let hkh = (s.subpixel_kernel_y / 2).clamp(1, 5) as i32;
    let n_threads = s.threads.clamp(1, h.max(1));
    let rows_per = (h + n_threads - 1) / n_threads;
    // Always use continuous cost (pure ZNCC for Ncc/Census, or native Sad/Ssd) for smooth subpixel peak
    let sub_cost_mode = match s.cost_mode {
        CostMode::Sad => CostMode::Sad,
        CostMode::Ssd => CostMode::Ssd,
        CostMode::Ncc | CostMode::Census => CostMode::Ncc,
    };

    let mut out = disp_in.clone();
    std::thread::scope(|scope| {
        for (t, (dxc, dyc)) in out
            .dx
            .chunks_mut(rows_per * w)
            .zip(out.dy.chunks_mut(rows_per * w))
            .enumerate()
        {
            let y_start = t * rows_per;
            scope.spawn(move || {
                let rows = dxc.len() / w;
                for ly in 0..rows {
                    let y = (y_start + ly) as i32;
                    for x in 0..w as i32 {
                        let idx = (y as usize) * w + (x as usize);
                        if disp_in.valid[idx] == 0 {
                            continue;
                        }
                        let d0x = disp_in.dx[idx].round() as i32;
                        let d0y = disp_in.dy[idx].round() as i32;

                        let c0 = patch_cost(left, right, x, y, d0x, d0y, hkw, hkh, sub_cost_mode);
                        let cmx = patch_cost(left, right, x, y, d0x - 1, d0y, hkw, hkh, sub_cost_mode);
                        let cpx = patch_cost(left, right, x, y, d0x + 1, d0y, hkw, hkh, sub_cost_mode);
                        let den_x = cmx - 2.0 * c0 + cpx;
                        // Strictly convex continuous ZNCC minimum eliminates Census quantization & pixel-locking
                        if den_x > 1e-4 && cmx >= c0 && cpx >= c0 {
                            let delta_x = (0.5 * (cmx - cpx) / den_x).clamp(-0.5, 0.5);
                            dxc[ly * w + x as usize] = d0x as f32 + delta_x;
                        } else if !disp_in.is_subpixel {
                            dxc[ly * w + x as usize] = d0x as f32;
                        }

                        if s.search_min_y != s.search_max_y {
                            let cmy = patch_cost(left, right, x, y, d0x, d0y - 1, hkw, hkh, sub_cost_mode);
                            let cpy = patch_cost(left, right, x, y, d0x, d0y + 1, hkw, hkh, sub_cost_mode);
                            let den_y = cmy - 2.0 * c0 + cpy;
                            if den_y > 1e-4 && cmy >= c0 && cpy >= c0 {
                                let delta_y = (0.5 * (cmy - cpy) / den_y).clamp(-0.5, 0.5);
                                dyc[ly * w + x as usize] = d0y as f32 + delta_y;
                            } else if !disp_in.is_subpixel {
                                dyc[ly * w + x as usize] = d0y as f32;
                            }
                        } else {
                            dyc[ly * w + x as usize] = d0y as f32;
                        }
                    }
                }
            });
        }
    });
    out.is_subpixel = true;
    out
}

/// Iterative Lucas-Kanade / Affine subpixel refinement with optional Bayes-EM robust pixel weighting.
fn refine_lucas_kanade(
    left: &ImageF32,
    right: &ImageF32,
    disp_in: &DisparityMap,
    s: &StereoSettings,
    bayes_em_weights: bool,
) -> DisparityMap {
    let w = left.width as usize;
    let h = left.height as usize;
    let hkw = (s.subpixel_kernel_x / 2).clamp(2, 6) as i32;
    let hkh = (s.subpixel_kernel_y / 2).clamp(2, 6) as i32;
    let n_threads = s.threads.clamp(1, h.max(1));
    let rows_per = (h + n_threads - 1) / n_threads;

    // Precompute spatial gradients of the right image
    let mut grad_x = vec![0.0f32; w * h];
    let mut grad_y = vec![0.0f32; w * h];
    for y in 0..h {
        let ym = y.saturating_sub(1);
        let yp = (y + 1).min(h - 1);
        for x in 0..w {
            let xm = x.saturating_sub(1);
            let xp = (x + 1).min(w - 1);
            grad_x[y * w + x] = 0.5 * (right.data[y * w + xp] - right.data[y * w + xm]);
            grad_y[y * w + x] = 0.5 * (right.data[yp * w + x] - right.data[ym * w + x]);
        }
    }
    let gx_img = ImageF32::from_vec(right.width, right.height, grad_x);
    let gy_img = ImageF32::from_vec(right.width, right.height, grad_y);

    let mut out = if disp_in.is_subpixel {
        disp_in.clone()
    } else {
        refine_parabola_pure_zncc(left, right, disp_in, s)
    };
    let initial = out.clone();

    std::thread::scope(|scope| {
        let gx_ref = &gx_img;
        let gy_ref = &gy_img;
        let init_ref = &initial;

        for (t, (dxc, dyc)) in out
            .dx
            .chunks_mut(rows_per * w)
            .zip(out.dy.chunks_mut(rows_per * w))
            .enumerate()
        {
            let y_start = t * rows_per;
            scope.spawn(move || {
                let rows = dxc.len() / w;
                for ly in 0..rows {
                    let y = (y_start + ly) as i32;
                    for x in 0..w as i32 {
                        let idx = (y as usize) * w + (x as usize);
                        if init_ref.valid[idx] == 0 {
                            continue;
                        }
                        let mut dx = init_ref.dx[idx];
                        let mut dy = init_ref.dy[idx];
                        let orig_dx = dx;
                        let orig_dy = dy;

                        for _iter in 0..6 {
                            let mut h00 = 0.0f64;
                            let mut h01 = 0.0f64;
                            let mut h11 = 0.0f64;
                            let mut b0 = 0.0f64;
                            let mut b1 = 0.0f64;

                            for ky in -hkh..=hkh {
                                for kx in -hkw..=hkw {
                                    let lx = x + kx;
                                    let ly_px = y + ky;
                                    let rx = lx as f32 + dx;
                                    let ry = ly_px as f32 + dy;
                                    if rx < 1.0 || rx >= (w as f32 - 2.0) || ry < 1.0 || ry >= (h as f32 - 2.0) {
                                        continue;
                                    }
                                    let lv = left.get_clamped(lx, ly_px) as f64;
                                    let rv = right.sample_bilinear(rx, ry) as f64;
                                    let err = lv - rv;
                                    let ix = gx_ref.sample_bilinear(rx, ry) as f64;
                                    let iy = gy_ref.sample_bilinear(rx, ry) as f64;

                                    // Gaussian spatial weight + Bayes-EM Cauchy/Huber outlier downweighting
                                    let spatial = (-((kx * kx + ky * ky) as f64) / (2.0 * (hkw as f64).powi(2))).exp();
                                    let w_px = if bayes_em_weights {
                                        spatial / (1.0 + (err * err) / 0.02)
                                    } else {
                                        spatial
                                    };

                                    h00 += w_px * ix * ix;
                                    h01 += w_px * ix * iy;
                                    h11 += w_px * iy * iy;
                                    b0 += w_px * ix * err;
                                    b1 += w_px * iy * err;
                                }
                            }

                            let det = h00 * h11 - h01 * h01;
                            if det.abs() < 1e-10 {
                                break;
                            }
                            let step_x = ((h11 * b0 - h01 * b1) / det) as f32;
                            let step_y = ((-h01 * b0 + h00 * b1) / det) as f32;
                            if !step_x.is_finite() || !step_y.is_finite() {
                                break;
                            }
                            dx += step_x.clamp(-0.35, 0.35);
                            if s.search_min_y != s.search_max_y {
                                dy += step_y.clamp(-0.35, 0.35);
                            }
                            if step_x.abs() < 1e-3 && step_y.abs() < 1e-3 {
                                break;
                            }
                        }

                        // Reject diverging updates
                        if (dx - orig_dx).abs() <= 1.2 && (dy - orig_dy).abs() <= 1.2 {
                            dxc[ly * w + x as usize] = dx;
                            dyc[ly * w + x as usize] = dy;
                        }
                    }
                }
            });
        }
    });

    out
}
