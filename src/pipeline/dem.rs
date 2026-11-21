//! `point2dem` — Gridded Digital Elevation Model (DEM) rasterizer and Vulkan/CPU Lambertian Hillshade.

use crate::config::StereoSettings;
use crate::image::{DemGrid, ImageF32, PointCloud};
use crate::vulkan::VulkanCompute;

/// Rasterize a 3D PointCloud onto a regular `(X, Y) -> Z` Digital Elevation Model grid + hillshade.
pub fn point_to_dem(
    pc: &PointCloud,
    s: &StereoSettings,
    vk: Option<&VulkanCompute>,
) -> (DemGrid, bool) {
    // Collect valid points and apply ASP-style IQR outlier rejection to prevent near-parallel ray explosions
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut zs = Vec::new();
    for (i, &v) in pc.valid.iter().enumerate() {
        if v != 0 && pc.points[i].is_finite() {
            let p = pc.points[i];
            xs.push(p.x);
            ys.push(p.y);
            zs.push(p.z);
        }
    }

    if xs.len() < 4 {
        let empty = ImageF32::new(16, 16);
        return (
            DemGrid {
                elevation: empty.clone(),
                hillshade: empty,
                min_x: 0.0,
                min_y: 0.0,
                max_x: 1.0,
                max_y: 1.0,
                pixel_size: 1.0,
            },
            false,
        );
    }

    let iqr_bounds = |vals: &mut Vec<f64>| -> (f64, f64) {
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = vals.len();
        let q25 = vals[n / 4];
        let q75 = vals[(3 * n) / 4];
        let iqr = (q75 - q25).max(1.0);
        (q25 - 3.0 * iqr, q75 + 3.0 * iqr)
    };
    let (x_lo, x_hi) = iqr_bounds(&mut xs);
    let (y_lo, y_hi) = iqr_bounds(&mut ys);
    let (z_lo, z_hi) = iqr_bounds(&mut zs);

    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut inlier_indices = Vec::with_capacity(xs.len());
    for (i, &v) in pc.valid.iter().enumerate() {
        if v != 0 && pc.points[i].is_finite() {
            let p = pc.points[i];
            if p.x >= x_lo && p.x <= x_hi && p.y >= y_lo && p.y <= y_hi && p.z >= z_lo && p.z <= z_hi {
                if p.x < min_x { min_x = p.x; }
                if p.x > max_x { max_x = p.x; }
                if p.y < min_y { min_y = p.y; }
                if p.y > max_y { max_y = p.y; }
                inlier_indices.push(i);
            }
        }
    }

    let span_x = (max_x - min_x).max(1e-3);
    let span_y = (max_y - min_y).max(1e-3);
    let pixel_size = if s.dem_spacing > 0.0 {
        s.dem_spacing
    } else {
        (span_x / pc.width.max(1) as f64).max(span_y / pc.height.max(1) as f64)
    };

    let grid_w = ((span_x / pixel_size).ceil() as u32 + 1).clamp(16, 2048);
    let grid_h = ((span_y / pixel_size).ceil() as u32 + 1).clamp(16, 2048);
    let n = (grid_w as usize) * (grid_h as usize);

    let mut sum_z = vec![0.0f64; n];
    let mut sum_w = vec![0.0f64; n];

    // Gaussian weighted splatting into 3x3 neighborhood (inliers only)
    for i in inlier_indices {
        let p = pc.points[i];
        let fx = (p.x - min_x) / pixel_size;
        let fy = (max_y - p.y) / pixel_size; // North-up raster
        let cx = fx.round() as i32;
        let cy = fy.round() as i32;
        for dy in -1..=1 {
            let gy = cy + dy;
            if gy < 0 || gy >= grid_h as i32 {
                continue;
            }
            for dx in -1..=1 {
                let gx = cx + dx;
                if gx < 0 || gx >= grid_w as i32 {
                    continue;
                }
                let rx = gx as f64 - fx;
                let ry = gy as f64 - fy;
                let w = (-(rx * rx + ry * ry) / 0.85).exp();
                let idx = (gy as usize) * (grid_w as usize) + (gx as usize);
                sum_z[idx] += p.z * w;
                sum_w[idx] += w;
            }
        }
    }

    let mut elev = ImageF32::new(grid_w, grid_h);
    elev.nodata = Some(-9999.0);
    for i in 0..n {
        if sum_w[i] > 1e-6 {
            elev.data[i] = (sum_z[i] / sum_w[i]) as f32;
        } else {
            elev.data[i] = f32::NAN;
        }
    }

    // Fill small DEM nodata gaps (2 passes)
    for _ in 0..2 {
        let prev = elev.data.clone();
        for y in 0..grid_h as i32 {
            for x in 0..grid_w as i32 {
                let idx = (y as usize) * (grid_w as usize) + (x as usize);
                if prev[idx].is_finite() {
                    continue;
                }
                let mut acc = 0.0f32;
                let mut cnt = 0u32;
                for ky in -1..=1 {
                    let ny = y + ky;
                    if ny < 0 || ny >= grid_h as i32 {
                        continue;
                    }
                    for kx in -1..=1 {
                        let nx = x + kx;
                        if nx < 0 || nx >= grid_w as i32 {
                            continue;
                        }
                        let v = prev[(ny as usize) * (grid_w as usize) + (nx as usize)];
                        if v.is_finite() {
                            acc += v;
                            cnt += 1;
                        }
                    }
                }
                if cnt >= 3 {
                    elev.data[idx] = acc / cnt as f32;
                }
            }
        }
    }

    let (hillshade, used_vk) = compute_hillshade(
        &elev,
        pixel_size as f32,
        s.hillshade_azimuth_deg,
        s.hillshade_elevation_deg,
        vk,
    );

    (
        DemGrid {
            elevation: elev,
            hillshade,
            min_x,
            min_y,
            max_x,
            max_y,
            pixel_size,
        },
        used_vk,
    )
}

/// Compute Lambertian hillshade (dispatches to Vulkan Compute SPIR-V shader when GPU is active).
pub fn compute_hillshade(
    dem: &ImageF32,
    pixel_size: f32,
    azimuth_deg: f64,
    elevation_deg: f64,
    vk: Option<&VulkanCompute>,
) -> (ImageF32, bool) {
    let az = azimuth_deg.to_radians();
    let el = elevation_deg.to_radians();
    let sun = [
        (el.cos() * az.sin()) as f32,
        (el.cos() * az.cos()) as f32,
        el.sin() as f32,
    ];

    if let Some(gpu) = vk {
        // Replace NaNs with local mean for GPU buffer, then restore NaN mask
        let (_, _, mean_z, _) = dem.stats();
        let clean: Vec<f32> = dem
            .data
            .iter()
            .map(|&v| if v.is_finite() { v } else { mean_z })
            .collect();
        if let Ok(mut shaded) = gpu.run_hillshade(&clean, dem.width, dem.height, pixel_size, sun) {
            for (i, &v) in dem.data.iter().enumerate() {
                if !v.is_finite() {
                    shaded[i] = f32::NAN;
                }
            }
            return (ImageF32::from_vec(dem.width, dem.height, shaded), true);
        }
    }

    // CPU implementation
    let w = dem.width as i32;
    let h = dem.height as i32;
    let mut out = ImageF32::new(dem.width, dem.height);
    let two_ps = 2.0 * pixel_size.max(1e-4);

    for y in 0..h {
        for x in 0..w {
            let c = dem.get(x as u32, y as u32);
            if !c.is_finite() {
                out.set(x as u32, y as u32, f32::NAN);
                continue;
            }
            let zl = dem.get_clamped(x - 1, y);
            let zr = dem.get_clamped(x + 1, y);
            let zu = dem.get_clamped(x, y - 1);
            let zd = dem.get_clamped(x, y + 1);
            let dzdx = if zl.is_finite() && zr.is_finite() { (zr - zl) / two_ps } else { 0.0 };
            let dzdy = if zu.is_finite() && zd.is_finite() { (zu - zd) / two_ps } else { 0.0 };
            let nx = -dzdx;
            let ny = -dzdy;
            let nz = 1.0f32;
            let nlen = (nx * nx + ny * ny + nz * nz).sqrt();
            let dot = (nx * sun[0] + ny * sun[1] + nz * sun[2]) / nlen;
            out.set(x as u32, y as u32, dot.clamp(0.0, 1.0));
        }
    }
    (out, false)
}
