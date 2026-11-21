//! Rational Polynomial Camera (RPC) model compatible with ASP/WorldView `.RPB` files.
//! Follows Grodecki, Dial & Lutes (ASPRS 2004).

use std::io;
use std::path::Path;

use crate::camera::CameraModel;
use crate::math::{Datum, Vec2, Vec3};

#[derive(Debug, Clone)]
pub struct RpcCamera {
    pub datum: Datum,
    pub line_num: [f64; 20],
    pub line_den: [f64; 20],
    pub samp_num: [f64; 20],
    pub samp_den: [f64; 20],
    pub samp_off: f64,
    pub line_off: f64,
    pub samp_scale: f64,
    pub line_scale: f64,
    pub lon_off: f64,
    pub lat_off: f64,
    pub height_off: f64,
    pub lon_scale: f64,
    pub lat_scale: f64,
    pub height_scale: f64,
}

impl RpcCamera {
    /// Evaluate the 20 standard RPC monomial terms from normalized `(P=lon, L=lat, H=height)`.
    #[inline]
    pub fn terms(lon: f64, lat: f64, h: f64) -> [f64; 20] {
        let p = lon;
        let l = lat;
        let p2 = p * p;
        let l2 = l * l;
        let h2 = h * h;
        [
            1.0,
            p,
            l,
            h,
            p * l,
            p * h,
            l * h,
            p2,
            l2,
            h2,
            p * l * h,
            p2 * p,
            p * l2,
            p * h2,
            p2 * l,
            l2 * l,
            l * h2,
            p2 * h,
            l2 * h,
            h2 * h,
        ]
    }

    #[inline]
    fn dot20(a: &[f64; 20], b: &[f64; 20]) -> f64 {
        let mut s = 0.0;
        for i in 0..20 {
            s += a[i] * b[i];
        }
        s
    }

    /// Project geodetic `(lon_deg, lat_deg, height_m)` to pixel `(sample, line)`.
    pub fn geodetic_to_pixel(&self, llh: Vec3) -> Vec2 {
        let p = (llh.x - self.lon_off) / self.lon_scale;
        let l = (llh.y - self.lat_off) / self.lat_scale;
        let h = (llh.z - self.height_off) / self.height_scale;
        let t = Self::terms(p, l, h);
        let rn = Self::dot20(&self.line_num, &t) / Self::dot20(&self.line_den, &t).max(1e-12);
        let cn = Self::dot20(&self.samp_num, &t) / Self::dot20(&self.samp_den, &t).max(1e-12);
        Vec2::new(
            cn * self.samp_scale + self.samp_off,
            rn * self.line_scale + self.line_off,
        )
    }

    /// Solve for geodetic `(lon_deg, lat_deg)` at a fixed `height_m` matching target pixel `(col, row)` via Newton-Raphson.
    pub fn pixel_to_geodetic_at_height(&self, pixel: Vec2, height_m: f64) -> Vec3 {
        let mut lon = self.lon_off;
        let mut lat = self.lat_off;
        let eps_lon = 1e-6 * self.lon_scale.max(1e-4);
        let eps_lat = 1e-6 * self.lat_scale.max(1e-4);

        for _ in 0..12 {
            let cur = self.geodetic_to_pixel(Vec3::new(lon, lat, height_m));
            let rx = cur.x - pixel.x;
            let ry = cur.y - pixel.y;
            if rx * rx + ry * ry < 1e-12 {
                break;
            }
            let dlon = self.geodetic_to_pixel(Vec3::new(lon + eps_lon, lat, height_m));
            let dlat = self.geodetic_to_pixel(Vec3::new(lon, lat + eps_lat, height_m));
            let j00 = (dlon.x - cur.x) / eps_lon;
            let j10 = (dlon.y - cur.y) / eps_lon;
            let j01 = (dlat.x - cur.x) / eps_lat;
            let j11 = (dlat.y - cur.y) / eps_lat;
            let det = j00 * j11 - j01 * j10;
            if det.abs() < 1e-18 {
                break;
            }
            lon -= (j11 * rx - j01 * ry) / det;
            lat -= (-j10 * rx + j00 * ry) / det;
        }
        Vec3::new(lon, lat, height_m)
    }

    pub fn load_rpb(path: &Path) -> io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Self::parse_rpb(&text)
    }

    pub fn parse_rpb(text: &str) -> io::Result<Self> {
        let mut cam = Self {
            datum: Datum::WGS84,
            line_num: [0.0; 20],
            line_den: [0.0; 20],
            samp_num: [0.0; 20],
            samp_den: [0.0; 20],
            samp_off: 0.0,
            line_off: 0.0,
            samp_scale: 1.0,
            line_scale: 1.0,
            lon_off: 0.0,
            lat_off: 0.0,
            height_off: 0.0,
            lon_scale: 1.0,
            lat_scale: 1.0,
            height_scale: 1.0,
        };
        cam.line_den[0] = 1.0;
        cam.samp_den[0] = 1.0;

        let cleaned = text.replace(['(', ')', ';', ','], " ");
        let tokens: Vec<&str> = cleaned.split_whitespace().collect();
        let mut i = 0;
        while i < tokens.len() {
            let key = tokens[i].trim_end_matches('=');
            let next_val = |idx: usize| -> Option<f64> {
                let mut j = idx + 1;
                if j < tokens.len() && tokens[j] == "=" {
                    j += 1;
                }
                tokens.get(j).and_then(|s| s.parse::<f64>().ok())
            };
            let read_20 = |idx: usize| -> ([f64; 20], usize) {
                let mut arr = [0.0; 20];
                let mut j = idx + 1;
                if j < tokens.len() && tokens[j] == "=" {
                    j += 1;
                }
                let mut k = 0;
                while k < 20 && j < tokens.len() {
                    if let Ok(v) = tokens[j].parse::<f64>() {
                        arr[k] = v;
                        k += 1;
                    }
                    j += 1;
                }
                (arr, j)
            };

            match key {
                "lineOffset" | "LINE_OFF" => if let Some(v) = next_val(i) { cam.line_off = v; },
                "sampOffset" | "SAMP_OFF" => if let Some(v) = next_val(i) { cam.samp_off = v; },
                "lineScale" | "LINE_SCALE" => if let Some(v) = next_val(i) { cam.line_scale = v; },
                "sampScale" | "SAMP_SCALE" => if let Some(v) = next_val(i) { cam.samp_scale = v; },
                "latOffset" | "LAT_OFF" => if let Some(v) = next_val(i) { cam.lat_off = v; },
                "longOffset" | "LONG_OFF" => if let Some(v) = next_val(i) { cam.lon_off = v; },
                "heightOffset" | "HEIGHT_OFF" => if let Some(v) = next_val(i) { cam.height_off = v; },
                "latScale" | "LAT_SCALE" => if let Some(v) = next_val(i) { cam.lat_scale = v; },
                "longScale" | "LONG_SCALE" => if let Some(v) = next_val(i) { cam.lon_scale = v; },
                "heightScale" | "HEIGHT_SCALE" => if let Some(v) = next_val(i) { cam.height_scale = v; },
                "lineNumCoef" | "LINE_NUM_COEFF" => {
                    let (arr, ni) = read_20(i);
                    cam.line_num = arr;
                    i = ni;
                    continue;
                }
                "lineDenCoef" | "LINE_DEN_COEFF" => {
                    let (arr, ni) = read_20(i);
                    cam.line_den = arr;
                    i = ni;
                    continue;
                }
                "sampNumCoef" | "SAMP_NUM_COEFF" => {
                    let (arr, ni) = read_20(i);
                    cam.samp_num = arr;
                    i = ni;
                    continue;
                }
                "sampDenCoef" | "SAMP_DEN_COEFF" => {
                    let (arr, ni) = read_20(i);
                    cam.samp_den = arr;
                    i = ni;
                    continue;
                }
                _ => {}
            }
            i += 1;
        }
        Ok(cam)
    }
}

impl CameraModel for RpcCamera {
    fn model_type(&self) -> &'static str {
        "RPC"
    }

    fn point_to_pixel(&self, point: Vec3) -> Vec2 {
        let llh = self.datum.ecef_to_geodetic(point);
        self.geodetic_to_pixel(llh)
    }

    fn pixel_to_vector(&self, pixel: Vec2) -> Vec3 {
        let p_hi = self.datum.geodetic_to_ecef(
            self.pixel_to_geodetic_at_height(pixel, self.height_off + self.height_scale).x,
            self.pixel_to_geodetic_at_height(pixel, self.height_off + self.height_scale).y,
            self.height_off + self.height_scale,
        );
        let p_lo_llh = self.pixel_to_geodetic_at_height(pixel, self.height_off - self.height_scale);
        let p_lo = self.datum.geodetic_to_ecef(p_lo_llh.x, p_lo_llh.y, p_lo_llh.z);
        (p_lo - p_hi).normalized()
    }

    fn camera_center(&self, pixel: Vec2) -> Vec3 {
        let hi_llh = self.pixel_to_geodetic_at_height(pixel, self.height_off + self.height_scale);
        self.datum.geodetic_to_ecef(hi_llh.x, hi_llh.y, hi_llh.z)
    }
}
