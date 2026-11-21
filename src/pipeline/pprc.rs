//! Stage 0: `stereo_pprc` — Intensity normalization and prefiltering (Subtracted Mean / LoG).
//! Dispatches to Vulkan Compute when available, or uses multi-threaded CPU via `std::thread::scope`.

use crate::config::{PrefilterMode, StereoSettings};
use crate::image::ImageF32;
use crate::vulkan::VulkanCompute;

/// Run Stage 0 preprocessing on a stereo image pair.
pub fn run_pprc(
    left: &ImageF32,
    right: &ImageF32,
    settings: &StereoSettings,
    vk: Option<&VulkanCompute>,
) -> (ImageF32, ImageF32, bool) {
    let l_norm = if settings.normalize_intensity {
        normalize_image(left)
    } else {
        left.clone()
    };
    let r_norm = if settings.normalize_intensity {
        normalize_image(right)
    } else {
        right.clone()
    };

    match settings.prefilter_mode {
        PrefilterMode::None => (l_norm, r_norm, false),
        PrefilterMode::SubtractedMean => {
            let radius = (settings.prefilter_kernel_width * 0.5).round().max(1.0) as i32;
            if let Some(gpu) = vk {
                if let (Ok(ld), Ok(rd)) = (
                    gpu.run_prefilter(&l_norm.data, l_norm.width, l_norm.height, radius, true),
                    gpu.run_prefilter(&r_norm.data, r_norm.width, r_norm.height, radius, true),
                ) {
                    return (
                        ImageF32::from_vec(l_norm.width, l_norm.height, ld),
                        ImageF32::from_vec(r_norm.width, r_norm.height, rd),
                        true,
                    );
                }
            }
            (
                subtracted_mean_cpu(&l_norm, radius),
                subtracted_mean_cpu(&r_norm, radius),
                false,
            )
        }
        PrefilterMode::LaplacianOfGaussian => {
            let sigma = settings.prefilter_kernel_width.max(0.5);
            (
                laplacian_of_gaussian_cpu(&l_norm, sigma, settings.threads),
                laplacian_of_gaussian_cpu(&r_norm, sigma, settings.threads),
                false,
            )
        }
    }
}

/// Normalize pixel values to `[0.0, 1.0]` using robust min-max scaling.
pub fn normalize_image(img: &ImageF32) -> ImageF32 {
    let (min_v, max_v, _, _) = img.stats();
    let range = (max_v - min_v).max(1e-6);
    let data = img
        .data
        .iter()
        .map(|&v| {
            if v.is_finite() {
                ((v - min_v) / range).clamp(0.0, 1.0)
            } else {
                0.0
            }
        })
        .collect();
    ImageF32::from_vec(img.width, img.height, data)
}

/// Fast O(1)-per-pixel Subtracted Mean prefilter using a 2D summed-area table (integral image).
pub fn subtracted_mean_cpu(img: &ImageF32, radius: i32) -> ImageF32 {
    let w = img.width as usize;
    let h = img.height as usize;
    let mut sat = vec![0.0f64; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row_sum = 0.0f64;
        for x in 0..w {
            row_sum += img.data[y * w + x] as f64;
            sat[(y + 1) * (w + 1) + (x + 1)] = sat[y * (w + 1) + (x + 1)] + row_sum;
        }
    }

    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        let y0 = (y as i32 - radius).max(0) as usize;
        let y1 = (y as i32 + radius + 1).min(h as i32) as usize;
        for x in 0..w {
            let x0 = (x as i32 - radius).max(0) as usize;
            let x1 = (x as i32 + radius + 1).min(w as i32) as usize;
            let sum = sat[y1 * (w + 1) + x1]
                - sat[y0 * (w + 1) + x1]
                - sat[y1 * (w + 1) + x0]
                + sat[y0 * (w + 1) + x0];
            let area = ((x1 - x0) * (y1 - y0)) as f64;
            let mean = (sum / area) as f32;
            out[y * w + x] = img.data[y * w + x] - mean;
        }
    }
    ImageF32::from_vec(img.width, img.height, out)
}

/// Separable Gaussian smoothing followed by discrete 2D Laplacian (LoG) — multi-threaded.
pub fn laplacian_of_gaussian_cpu(img: &ImageF32, sigma: f32, threads: usize) -> ImageF32 {
    let rad = (sigma * 3.0).ceil().max(1.0) as i32;
    let mut kernel = Vec::with_capacity((2 * rad + 1) as usize);
    let mut ksum = 0.0f32;
    for k in -rad..=rad {
        let v = (-(k * k) as f32 / (2.0 * sigma * sigma)).exp();
        kernel.push(v);
        ksum += v;
    }
    for v in &mut kernel {
        *v /= ksum;
    }

    let w = img.width as usize;
    let h = img.height as usize;
    let n_threads = threads.clamp(1, h.max(1));

    // Horizontal pass
    let mut tmp = vec![0.0f32; w * h];
    let rows_per = (h + n_threads - 1) / n_threads;
    std::thread::scope(|s| {
        for (t, chunk) in tmp.chunks_mut(rows_per * w).enumerate() {
            let y_start = t * rows_per;
            let kref = &kernel;
            s.spawn(move || {
                let rows = chunk.len() / w;
                for ly in 0..rows {
                    let y = y_start + ly;
                    let src_row = &img.data[y * w..(y + 1) * w];
                    let dst_row = &mut chunk[ly * w..(ly + 1) * w];
                    for x in 0..w {
                        let mut acc = 0.0f32;
                        for (ki, &kv) in kref.iter().enumerate() {
                            let sx = (x as i32 + ki as i32 - rad).clamp(0, w as i32 - 1) as usize;
                            acc += src_row[sx] * kv;
                        }
                        dst_row[x] = acc;
                    }
                }
            });
        }
    });

    // Vertical pass
    let mut smooth = vec![0.0f32; w * h];
    std::thread::scope(|s| {
        for (t, chunk) in smooth.chunks_mut(rows_per * w).enumerate() {
            let y_start = t * rows_per;
            let kref = &kernel;
            let tref = &tmp;
            s.spawn(move || {
                let rows = chunk.len() / w;
                for ly in 0..rows {
                    let y = y_start + ly;
                    let dst_row = &mut chunk[ly * w..(ly + 1) * w];
                    for x in 0..w {
                        let mut acc = 0.0f32;
                        for (ki, &kv) in kref.iter().enumerate() {
                            let sy = (y as i32 + ki as i32 - rad).clamp(0, h as i32 - 1) as usize;
                            acc += tref[sy * w + x] * kv;
                        }
                        dst_row[x] = acc;
                    }
                }
            });
        }
    });

    // 2D Laplacian operator: L(x,y) = 4*I(x,y) - I(x-1,y) - I(x+1,y) - I(x,y-1) - I(x,y+1)
    let mut log_img = vec![0.0f32; w * h];
    std::thread::scope(|s| {
        for (t, chunk) in log_img.chunks_mut(rows_per * w).enumerate() {
            let y_start = t * rows_per;
            let sref = &smooth;
            s.spawn(move || {
                let rows = chunk.len() / w;
                for ly in 0..rows {
                    let y = y_start + ly;
                    let ym = y.saturating_sub(1);
                    let yp = (y + 1).min(h - 1);
                    let dst_row = &mut chunk[ly * w..(ly + 1) * w];
                    for x in 0..w {
                        let xm = x.saturating_sub(1);
                        let xp = (x + 1).min(w - 1);
                        let c = sref[y * w + x];
                        dst_row[x] = 4.0 * c
                            - sref[y * w + xm]
                            - sref[y * w + xp]
                            - sref[ym * w + x]
                            - sref[yp * w + x];
                    }
                }
            });
        }
    });

    ImageF32::from_vec(img.width, img.height, log_img)
}
