//! Core raster image, disparity map, DEM grid, point cloud, and colormap structures.

pub mod geotiff;
pub mod io;

use crate::math::Vec3;

/// Single-channel floating-point 2D raster image (row-major).
#[derive(Debug, Clone)]
pub struct ImageF32 {
    pub width: u32,
    pub height: u32,
    pub data: Vec<f32>,
    pub nodata: Option<f32>,
}

impl ImageF32 {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; (width as usize) * (height as usize)],
            nodata: None,
        }
    }

    pub fn from_vec(width: u32, height: u32, data: Vec<f32>) -> Self {
        let expected = (width as usize).saturating_mul(height as usize);
        if data.len() == expected {
            Self {
                width,
                height,
                data,
                nodata: None,
            }
        } else {
            let mut safe = data;
            safe.resize(expected, 0.0);
            Self {
                width,
                height,
                data: safe,
                nodata: None,
            }
        }
    }

    pub fn try_from_vec(width: u32, height: u32, data: Vec<f32>) -> Option<Self> {
        let expected = (width as usize).checked_mul(height as usize)?;
        if data.len() == expected {
            Some(Self {
                width,
                height,
                data,
                nodata: None,
            })
        } else {
            None
        }
    }

    #[inline]
    pub fn get_checked(&self, x: u32, y: u32) -> Option<f32> {
        if x < self.width && y < self.height {
            self.data.get((y as usize) * (self.width as usize) + (x as usize)).copied()
        } else {
            None
        }
    }

    #[inline]
    pub fn get(&self, x: u32, y: u32) -> f32 {
        self.data[(y as usize) * (self.width as usize) + (x as usize)]
    }

    #[inline]
    pub fn set(&mut self, x: u32, y: u32, v: f32) {
        self.data[(y as usize) * (self.width as usize) + (x as usize)] = v;
    }

    #[inline]
    pub fn get_clamped(&self, x: i32, y: i32) -> f32 {
        let cx = x.clamp(0, self.width as i32 - 1) as u32;
        let cy = y.clamp(0, self.height as i32 - 1) as u32;
        self.get(cx, cy)
    }

    /// Bilinear interpolation at floating-point pixel coordinates.
    #[inline]
    pub fn sample_bilinear(&self, x: f32, y: f32) -> f32 {
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;
        let v00 = self.get_clamped(x0, y0);
        let v10 = self.get_clamped(x0 + 1, y0);
        let v01 = self.get_clamped(x0, y0 + 1);
        let v11 = self.get_clamped(x0 + 1, y0 + 1);
        let top = v00 + (v10 - v00) * fx;
        let bot = v01 + (v11 - v01) * fx;
        top + (bot - top) * fy
    }

    /// Compute min, max, mean, stddev ignoring NaN and nodata.
    pub fn stats(&self) -> (f32, f32, f32, f32) {
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut count = 0usize;
        for &v in &self.data {
            if !v.is_finite() {
                continue;
            }
            if let Some(nd) = self.nodata {
                if (v - nd).abs() < 1e-6 {
                    continue;
                }
            }
            if v < min_v {
                min_v = v;
            }
            if v > max_v {
                max_v = v;
            }
            sum += v as f64;
            sum_sq += (v as f64) * (v as f64);
            count += 1;
        }
        if count == 0 {
            return (0.0, 0.0, 0.0, 0.0);
        }
        let mean = sum / count as f64;
        let var = (sum_sq / count as f64 - mean * mean).max(0.0);
        (min_v, max_v, mean as f32, var.sqrt() as f32)
    }

    /// Downsample image by factor of 2 using a 2x2 box average.
    pub fn downsample_2x(&self) -> Self {
        let nw = (self.width / 2).max(1);
        let nh = (self.height / 2).max(1);
        let mut out = Self::new(nw, nh);
        out.nodata = self.nodata;
        for y in 0..nh {
            for x in 0..nw {
                let sx = (x * 2) as i32;
                let sy = (y * 2) as i32;
                let v = 0.25
                    * (self.get_clamped(sx, sy)
                        + self.get_clamped(sx + 1, sy)
                        + self.get_clamped(sx, sy + 1)
                        + self.get_clamped(sx + 1, sy + 1));
                out.set(x, y, v);
            }
        }
        out
    }

    /// Render grayscale image to RGBA8 buffer with auto-contrast stretching.
    pub fn to_rgba8(&self, colormap: Colormap) -> Vec<u8> {
        let (min_v, max_v, _, _) = self.stats();
        let range = (max_v - min_v).max(1e-6);
        let mut rgba = vec![0u8; self.data.len() * 4];
        for (i, &v) in self.data.iter().enumerate() {
            let is_valid = v.is_finite()
                && self.nodata.map_or(true, |nd| (v - nd).abs() > 1e-6);
            let [r, g, b, a] = if !is_valid {
                [18, 22, 28, 255]
            } else {
                let t = ((v - min_v) / range).clamp(0.0, 1.0);
                colormap.sample(t)
            };
            rgba[i * 4] = r;
            rgba[i * 4 + 1] = g;
            rgba[i * 4 + 2] = b;
            rgba[i * 4 + 3] = a;
        }
        rgba
    }
}

/// 2D Stereo Disparity Map (horizontal `dx`, vertical `dy`, and validity mask).
#[derive(Debug, Clone)]
pub struct DisparityMap {
    pub width: u32,
    pub height: u32,
    pub dx: Vec<f32>,
    pub dy: Vec<f32>,
    pub valid: Vec<u8>,
}

impl DisparityMap {
    pub fn new(width: u32, height: u32) -> Self {
        let n = (width as usize) * (height as usize);
        Self {
            width,
            height,
            dx: vec![0.0; n],
            dy: vec![0.0; n],
            valid: vec![0; n],
        }
    }

    #[inline]
    pub fn idx(&self, x: u32, y: u32) -> usize {
        (y as usize) * (self.width as usize) + (x as usize)
    }

    #[inline]
    pub fn is_valid(&self, x: u32, y: u32) -> bool {
        self.valid[self.idx(x, y)] != 0
    }

    #[inline]
    pub fn set(&mut self, x: u32, y: u32, dx: f32, dy: f32, valid: bool) {
        let i = self.idx(x, y);
        self.dx[i] = dx;
        self.dy[i] = dy;
        self.valid[i] = if valid { 1 } else { 0 };
    }

    pub fn valid_count(&self) -> usize {
        self.valid.iter().filter(|&&v| v != 0).count()
    }

    pub fn valid_fraction(&self) -> f32 {
        if self.valid.is_empty() {
            return 0.0;
        }
        self.valid_count() as f32 / self.valid.len() as f32
    }

    /// Extract horizontal disparity channel as an ImageF32 (invalid pixels = NaN).
    pub fn dx_image(&self) -> ImageF32 {
        let mut img = ImageF32::new(self.width, self.height);
        for i in 0..self.dx.len() {
            img.data[i] = if self.valid[i] != 0 {
                self.dx[i]
            } else {
                f32::NAN
            };
        }
        img
    }

    /// Extract vertical disparity channel as an ImageF32 (invalid pixels = NaN).
    pub fn dy_image(&self) -> ImageF32 {
        let mut img = ImageF32::new(self.width, self.height);
        for i in 0..self.dy.len() {
            img.data[i] = if self.valid[i] != 0 {
                self.dy[i]
            } else {
                f32::NAN
            };
        }
        img
    }
}

/// 3D Triangulated Point Cloud with intensity and intersection error.
#[derive(Debug, Clone, Default)]
pub struct PointCloud {
    pub width: u32,
    pub height: u32,
    pub points: Vec<Vec3>,
    pub errors: Vec<f32>,
    pub intensities: Vec<f32>,
    pub valid: Vec<u8>,
}

impl PointCloud {
    pub fn new(width: u32, height: u32) -> Self {
        let n = (width as usize) * (height as usize);
        Self {
            width,
            height,
            points: vec![Vec3::ZERO; n],
            errors: vec![0.0; n],
            intensities: vec![0.0; n],
            valid: vec![0; n],
        }
    }

    pub fn valid_count(&self) -> usize {
        self.valid.iter().filter(|&&v| v != 0).count()
    }

    pub fn mean_error(&self) -> f64 {
        let mut sum = 0.0;
        let mut cnt = 0usize;
        for (i, &v) in self.valid.iter().enumerate() {
            if v != 0 {
                sum += self.errors[i] as f64;
                cnt += 1;
            }
        }
        if cnt == 0 { 0.0 } else { sum / cnt as f64 }
    }

    pub fn error_image(&self) -> ImageF32 {
        let mut img = ImageF32::new(self.width, self.height);
        for i in 0..self.errors.len() {
            img.data[i] = if self.valid[i] != 0 {
                self.errors[i]
            } else {
                f32::NAN
            };
        }
        img
    }
}

/// Digital Elevation Model (DEM) raster grid with geospatial bounds.
#[derive(Debug, Clone)]
pub struct DemGrid {
    pub elevation: ImageF32,
    pub hillshade: ImageF32,
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
    pub pixel_size: f64,
}

/// Scientific colormaps for visualizing disparity, DEM, and error maps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Colormap {
    Grayscale,
    Turbo,
    Viridis,
    Inferno,
    Terrain,
}

impl Colormap {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Grayscale => "Grayscale",
            Self::Turbo => "Turbo",
            Self::Viridis => "Viridis",
            Self::Inferno => "Inferno",
            Self::Terrain => "Terrain",
        }
    }

    pub fn next(&self) -> Self {
        match self {
            Self::Grayscale => Self::Turbo,
            Self::Turbo => Self::Viridis,
            Self::Viridis => Self::Inferno,
            Self::Inferno => Self::Terrain,
            Self::Terrain => Self::Grayscale,
        }
    }

    pub fn from_name(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "gray" | "grayscale" => Self::Grayscale,
            "viridis" => Self::Viridis,
            "inferno" => Self::Inferno,
            "terrain" => Self::Terrain,
            _ => Self::Turbo,
        }
    }

    #[inline]
    pub fn sample(&self, t: f32) -> [u8; 4] {
        let t = t.clamp(0.0, 1.0);
        let (r, g, b) = match self {
            Self::Grayscale => (t, t, t),
            Self::Turbo => {
                // Anton Mikhailov's Turbo polynomial approximation
                let r = (0.13572138
                    + t * (4.61539260
                        + t * (-42.66032258
                            + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943)))))
                    .clamp(0.0, 1.0);
                let g = (0.09140261
                    + t * (2.19418839
                        + t * (4.84296658
                            + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604)))))
                    .clamp(0.0, 1.0);
                let b = (0.10667330
                    + t * (12.64194608
                        + t * (-60.58204836
                            + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973)))))
                    .clamp(0.0, 1.0);
                (r, g, b)
            }
            Self::Viridis => {
                let stops: [(f32, f32, f32); 5] = [
                    (0.267, 0.005, 0.329),
                    (0.230, 0.322, 0.546),
                    (0.128, 0.567, 0.551),
                    (0.369, 0.789, 0.383),
                    (0.993, 0.906, 0.144),
                ];
                lerp_stops(&stops, t)
            }
            Self::Inferno => {
                let stops: [(f32, f32, f32); 5] = [
                    (0.001, 0.000, 0.014),
                    (0.341, 0.062, 0.429),
                    (0.735, 0.216, 0.330),
                    (0.978, 0.557, 0.035),
                    (0.988, 0.998, 0.645),
                ];
                lerp_stops(&stops, t)
            }
            Self::Terrain => {
                let stops: [(f32, f32, f32); 5] = [
                    (0.10, 0.28, 0.52),
                    (0.20, 0.56, 0.36),
                    (0.78, 0.72, 0.42),
                    (0.58, 0.40, 0.28),
                    (0.96, 0.96, 0.98),
                ];
                lerp_stops(&stops, t)
            }
        };
        [
            (r * 255.0).round() as u8,
            (g * 255.0).round() as u8,
            (b * 255.0).round() as u8,
            255,
        ]
    }
}

fn lerp_stops(stops: &[(f32, f32, f32)], t: f32) -> (f32, f32, f32) {
    let scaled = t * (stops.len() - 1) as f32;
    let i = (scaled.floor() as usize).min(stops.len() - 2);
    let f = scaled - i as f32;
    let a = stops[i];
    let b = stops[i + 1];
    (
        a.0 + (b.0 - a.0) * f,
        a.1 + (b.1 - a.1) * f,
        a.2 + (b.2 - a.2) * f,
    )
}
