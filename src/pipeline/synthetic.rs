//! Realistic synthetic planetary/lunar stereo dataset generator.
//! Generates a cratered mountainous terrain DEM, high-frequency rock texture,
//! left/right pinhole stereo cameras, rendered stereo pair, and exact ground-truth disparity/DEM.

use crate::camera::{AnyCamera, CameraModel, PinholeCamera};
use crate::image::{DisparityMap, ImageF32};
use crate::math::Vec3;

#[derive(Debug, Clone)]
pub struct SyntheticDataset {
    pub left_image: ImageF32,
    pub right_image: ImageF32,
    pub left_cam: AnyCamera,
    pub right_cam: AnyCamera,
    pub gt_disparity: DisparityMap,
    pub gt_dem: ImageF32,
}

/// Deterministic hash-based value noise for procedural planetary surface texture & topography.
#[inline]
fn hash2(ix: i32, iy: i32) -> f64 {
    let mut n = (ix.wrapping_mul(374_761_393)) ^ (iy.wrapping_mul(668_265_263));
    n = (n ^ (n >> 13)).wrapping_mul(1_274_126_177);
    ((n ^ (n >> 16)) as u32) as f64 / (u32::MAX as f64)
}

fn value_noise(x: f64, y: f64) -> f64 {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let fx = x - ix as f64;
    let fy = y - iy as f64;
    let ux = fx * fx * (3.0 - 2.0 * fx);
    let uy = fy * fy * (3.0 - 2.0 * fy);
    let a = hash2(ix, iy);
    let b = hash2(ix + 1, iy);
    let c = hash2(ix, iy + 1);
    let d = hash2(ix + 1, iy + 1);
    a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy
}

fn fbm(x: f64, y: f64, octaves: usize) -> f64 {
    let mut val = 0.0;
    let mut amp = 0.5;
    let mut freq = 1.0;
    for _ in 0..octaves {
        val += amp * value_noise(x * freq, y * freq);
        freq *= 2.03;
        amp *= 0.5;
    }
    val
}

/// Procedural planetary terrain elevation `Z(x, y)` in meters (impact craters + mountain ridge).
fn terrain_height(x: f64, y: f64) -> f64 {
    // Broad mountain ridge
    let ridge = 14.0 * (-((x - 18.0) * 0.025).powi(2) - ((y + 10.0) * 0.02).powi(2)).exp()
        + 9.0 * (x * 0.035).sin() * (y * 0.03).cos();

    // Impact craters (raised rim + depressed bowl)
    let craters: [(f64, f64, f64, f64); 4] = [
        (-16.0, -12.0, 22.0, 10.5),
        (24.0, 18.0, 16.0, 7.5),
        (-28.0, 22.0, 11.0, 5.0),
        (5.0, -26.0, 13.0, 5.5),
    ];
    let mut h = ridge + (fbm(x * 0.04 + 10.0, y * 0.04 + 20.0, 4) - 0.5) * 8.0;
    for &(cx, cy, rad, depth) in &craters {
        let r = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() / rad;
        if r < 1.35 {
            let bowl = if r < 1.0 {
                -depth * (1.0 - r * r).powf(1.3)
            } else {
                0.0
            };
            let rim = depth * 0.28 * (-((r - 1.0) / 0.16).powi(2)).exp();
            h += bowl + rim;
        }
    }
    h
}

/// High-frequency textured albedo + sunlit shading for crisp stereo correlation.
fn surface_radiance(x: f64, y: f64) -> f32 {
    let eps = 0.35;
    let dzdx = (terrain_height(x + eps, y) - terrain_height(x - eps, y)) / (2.0 * eps);
    let dzdy = (terrain_height(x, y + eps) - terrain_height(x, y - eps)) / (2.0 * eps);
    let n = Vec3::new(-dzdx, -dzdy, 1.0).normalized();
    let sun = Vec3::new(-0.55, 0.45, 0.70).normalized();
    let shade = (0.28 + 0.72 * n.dot(sun).max(0.0)).clamp(0.1, 1.0);

    // Rich multi-scale regolith & boulder texture (critical for subpixel stereo matching)
    let tex = 0.42
        + 0.32 * fbm(x * 0.55 + 3.1, y * 0.55 + 7.4, 4)
        + 0.26 * (value_noise(x * 2.1, y * 2.1) - 0.5);
    (shade * tex).clamp(0.02, 0.99) as f32
}

/// Ray-march a camera ray onto the procedural terrain surface `z = terrain_height(x, y)`.
fn intersect_terrain(origin: Vec3, dir: Vec3) -> Vec3 {
    // Analytic intersection with z = 0 plane as initial guess, then refine along ray
    let mut t = if dir.z.abs() > 1e-6 { -origin.z / dir.z } else { 200.0 };
    for _ in 0..14 {
        let p = origin + dir * t;
        let h = terrain_height(p.x, p.y);
        let diff = p.z - h;
        if diff.abs() < 1e-4 {
            return Vec3::new(p.x, p.y, h);
        }
        t -= diff / dir.z.min(-0.2);
    }
    let p = origin + dir * t;
    Vec3::new(p.x, p.y, terrain_height(p.x, p.y))
}

/// Generate a complete synthetic stereo dataset of resolution `width x height`.
pub fn generate_synthetic_stereo(width: u32, height: u32) -> SyntheticDataset {
    let focal = (width as f64) * 1.45;
    let altitude = 220.0;
    let baseline = 26.0;

    let left_pin = PinholeCamera::nadir(width, height, focal, Vec3::new(-baseline * 0.5, 0.0, altitude));
    let right_pin = PinholeCamera::nadir(width, height, focal, Vec3::new(baseline * 0.5, 0.0, altitude));
    let left_cam = AnyCamera::Pinhole(left_pin);
    let right_cam = AnyCamera::Pinhole(right_pin);

    let mut left_img = ImageF32::new(width, height);
    let mut right_img = ImageF32::new(width, height);
    let mut gt_disp = DisparityMap::new(width, height);
    let mut gt_dem = ImageF32::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let px = crate::math::Vec2::new(x as f64, y as f64);

            // Render left camera
            let lo = left_cam.camera_center(px);
            let ld = left_cam.pixel_to_vector(px);
            let lp = intersect_terrain(lo, ld);
            left_img.set(x, y, surface_radiance(lp.x, lp.y));
            gt_dem.set(x, y, lp.z as f32);

            // Project left 3D hit point into right camera for ground-truth disparity
            let rp_proj = right_cam.point_to_pixel(lp);
            let dx = (rp_proj.x - x as f64) as f32;
            let dy = (rp_proj.y - y as f64) as f32;
            let in_bounds = rp_proj.x >= 0.0 && rp_proj.x < width as f64 && rp_proj.y >= 0.0 && rp_proj.y < height as f64;
            gt_disp.set(x, y, dx, dy, in_bounds);

            // Render right camera
            let ro = right_cam.camera_center(px);
            let rd = right_cam.pixel_to_vector(px);
            let rp = intersect_terrain(ro, rd);
            right_img.set(x, y, surface_radiance(rp.x, rp.y));
        }
    }

    SyntheticDataset {
        left_image: left_img,
        right_image: right_img,
        left_cam,
        right_cam,
        gt_disparity: gt_disp,
        gt_dem,
    }
}
