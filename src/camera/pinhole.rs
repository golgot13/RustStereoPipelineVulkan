//! Pinhole camera model with ASP `.tsai` file format support and radial/tangential distortion.

use std::io;
use std::path::Path;

use crate::camera::CameraModel;
use crate::math::{Mat3, Vec2, Vec3};

#[derive(Debug, Clone)]
pub struct PinholeCamera {
    /// Focal length in pixels along x and y.
    pub fu: f64,
    pub fv: f64,
    /// Principal point in pixels.
    pub cu: f64,
    pub cv: f64,
    /// Pixel pitch (meters/pixel or mm/pixel scale; default 1.0 when fu/fv are already in pixels).
    pub pitch: f64,
    /// Camera optical center in world coordinates.
    pub center: Vec3,
    /// Camera-to-world rotation matrix (`world_dir = rotation * cam_dir`).
    pub rotation: Mat3,
    /// Brown-Conrady radial/tangential distortion `(k1, k2, p1, p2)`.
    pub distortion: [f64; 4],
}

impl PinholeCamera {
    pub fn new(fu: f64, fv: f64, cu: f64, cv: f64, center: Vec3, rotation: Mat3) -> Self {
        Self {
            fu,
            fv,
            cu,
            cv,
            pitch: 1.0,
            center,
            rotation,
            distortion: [0.0; 4],
        }
    }

    /// Create a nadir-looking camera at `center` looking toward `-Z` with `+X` right and `-Y` down (image rows increase south).
    pub fn nadir(width: u32, height: u32, focal_px: f64, center: Vec3) -> Self {
        // Camera frame: +x right, +y down (image row), +z forward (looking down = -Z world)
        let rotation = Mat3::new([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]);
        Self::new(
            focal_px,
            focal_px,
            width as f64 * 0.5,
            height as f64 * 0.5,
            center,
            rotation,
        )
    }

    /// Load an ASP `.tsai` pinhole camera file.
    pub fn load_tsai(path: &Path) -> io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::parse_tsai(&content)
    }

    pub fn parse_tsai(content: &str) -> io::Result<Self> {
        let mut fu = 1000.0;
        let mut fv = 1000.0;
        let mut cu = 0.0;
        let mut cv = 0.0;
        let mut pitch = 1.0;
        let mut center = Vec3::ZERO;
        let mut rotation = Mat3::IDENTITY;
        let mut distortion = [0.0; 4];

        for raw in content.lines() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((k, v)) = line.split_once('=') {
                let key = k.trim();
                let nums: Vec<f64> = v
                    .split_whitespace()
                    .filter_map(|t| t.parse::<f64>().ok())
                    .collect();
                match key {
                    "fu" if !nums.is_empty() => fu = nums[0],
                    "fv" if !nums.is_empty() => fv = nums[0],
                    "cu" if !nums.is_empty() => cu = nums[0],
                    "cv" if !nums.is_empty() => cv = nums[0],
                    "pitch" if !nums.is_empty() => pitch = nums[0],
                    "C" if nums.len() >= 3 => center = Vec3::new(nums[0], nums[1], nums[2]),
                    "R" if nums.len() >= 9 => {
                        rotation = Mat3::new([
                            [nums[0], nums[1], nums[2]],
                            [nums[3], nums[4], nums[5]],
                            [nums[6], nums[7], nums[8]],
                        ]);
                    }
                    "k1" if !nums.is_empty() => distortion[0] = nums[0],
                    "k2" if !nums.is_empty() => distortion[1] = nums[0],
                    "p1" if !nums.is_empty() => distortion[2] = nums[0],
                    "p2" if !nums.is_empty() => distortion[3] = nums[0],
                    _ => {}
                }
            }
        }

        Ok(Self {
            fu,
            fv,
            cu,
            cv,
            pitch,
            center,
            rotation,
            distortion,
        })
    }

    /// Save camera in standard ASP `.tsai` format.
    pub fn save_tsai(&self, path: &Path) -> io::Result<()> {
        let m = &self.rotation.m;
        let s = format!(
            "VERSION_4\nPINHOLE\nfu = {:.9}\nfv = {:.9}\ncu = {:.9}\ncv = {:.9}\n\
             u_direction = 1 0 0\nv_direction = 0 1 0\nw_direction = 0 0 1\n\
             C = {:.9} {:.9} {:.9}\n\
             R = {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9}\n\
             pitch = {:.9}\nNULL\n",
            self.fu,
            self.fv,
            self.cu,
            self.cv,
            self.center.x,
            self.center.y,
            self.center.z,
            m[0][0], m[0][1], m[0][2],
            m[1][0], m[1][1], m[1][2],
            m[2][0], m[2][1], m[2][2],
            self.pitch
        );
        std::fs::write(path, s)
    }

    /// Undistort normalized sensor coordinates via iterative inversion of Brown-Conrady model.
    fn undistort_normalized(&self, xd: f64, yd: f64) -> (f64, f64) {
        let [k1, k2, p1, p2] = self.distortion;
        if k1 == 0.0 && k2 == 0.0 && p1 == 0.0 && p2 == 0.0 {
            return (xd, yd);
        }
        let mut x = xd;
        let mut y = yd;
        for _ in 0..8 {
            let r2 = x * x + y * y;
            let rad = 1.0 + k1 * r2 + k2 * r2 * r2;
            let dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
            let dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
            x = (xd - dx) / rad;
            y = (yd - dy) / rad;
        }
        (x, y)
    }
}

impl CameraModel for PinholeCamera {
    fn model_type(&self) -> &'static str {
        "PINHOLE"
    }

    fn point_to_pixel(&self, point: Vec3) -> Vec2 {
        let rel = point - self.center;
        let cam = self.rotation.transpose().mul_vec(rel);
        let z = if cam.z.abs() < 1e-12 { 1e-12 } else { cam.z };
        let xn = cam.x / z;
        let yn = cam.y / z;
        let [k1, k2, p1, p2] = self.distortion;
        let r2 = xn * xn + yn * yn;
        let rad = 1.0 + k1 * r2 + k2 * r2 * r2;
        let xd = xn * rad + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn);
        let yd = yn * rad + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn;
        let inv_pitch = 1.0 / self.pitch.max(1e-12);
        Vec2::new(
            (xd * self.fu + self.cu) * inv_pitch,
            (yd * self.fv + self.cv) * inv_pitch,
        )
    }

    fn pixel_to_vector(&self, pixel: Vec2) -> Vec3 {
        let xd = (pixel.x * self.pitch - self.cu) / self.fu;
        let yd = (pixel.y * self.pitch - self.cv) / self.fv;
        let (xn, yn) = self.undistort_normalized(xd, yd);
        let cam_ray = Vec3::new(xn, yn, 1.0).normalized();
        self.rotation.mul_vec(cam_ray).normalized()
    }

    fn camera_center(&self, _pixel: Vec2) -> Vec3 {
        self.center
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pinhole_projection_roundtrip() {
        let cam = PinholeCamera::nadir(512, 512, 800.0, Vec3::new(10.0, -5.0, 200.0));
        let pt = Vec3::new(25.0, 12.0, 40.0);
        let px = cam.point_to_pixel(pt);
        let ray = cam.pixel_to_vector(px);
        let expected = (pt - cam.center).normalized();
        assert!((ray - expected).norm() < 1e-9);
    }

    #[test]
    fn test_tsai_roundtrip() {
        let cam = PinholeCamera::nadir(640, 480, 920.0, Vec3::new(1.5, 2.5, 300.0));
        let path = std::env::temp_dir().join("rspv_test_cam.tsai");
        cam.save_tsai(&path).unwrap();
        let loaded = PinholeCamera::load_tsai(&path).unwrap();
        assert!((loaded.fu - 920.0).abs() < 1e-6);
        assert!((loaded.center - cam.center).norm() < 1e-6);
        let _ = std::fs::remove_file(&path);
    }
}
