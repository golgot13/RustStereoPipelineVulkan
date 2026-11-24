//! Pure-Rust double-precision (`f64`) Geodetic & Coordinate Reference System (CRS) engine.
//! Supports Local Cartesian, WGS84 (EPSG:4326), ECEF (EPSG:4978), French Lambert-93 (EPSG:2154),
//! UTM North/South zones (EPSG:326xx / 327xx), Moon, and Mars.

use crate::core::error::{PipelineError, Result};
use crate::math::{Datum, Vec3};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrsId {
    Local,
    Wgs84,
    Ecef,
    Lambert93,
    Utm { zone: u8, north: bool },
    Moon,
    Mars,
}

impl CrsId {
    pub fn code(&self) -> String {
        match self {
            Self::Local => "LOCAL".to_string(),
            Self::Wgs84 => "EPSG:4326".to_string(),
            Self::Ecef => "EPSG:4978".to_string(),
            Self::Lambert93 => "EPSG:2154".to_string(),
            Self::Utm { zone, north } => {
                if *north {
                    format!("EPSG:326{:02}", zone)
                } else {
                    format!("EPSG:327{:02}", zone)
                }
            }
            Self::Moon => "IAU:MOON".to_string(),
            Self::Mars => "IAU:MARS".to_string(),
        }
    }

    pub fn display_name(&self) -> String {
        match self {
            Self::Local => "Local Cartesian (m)".to_string(),
            Self::Wgs84 => "WGS84 Geographic (EPSG:4326)".to_string(),
            Self::Ecef => "WGS84 ECEF (EPSG:4978)".to_string(),
            Self::Lambert93 => "RGF93 / Lambert-93 (EPSG:2154)".to_string(),
            Self::Utm { zone, north } => format!(
                "WGS84 / UTM Zone {}{} ({})",
                zone,
                if *north { "N" } else { "S" },
                self.code()
            ),
            Self::Moon => "Moon Geographic (IAU:MOON)".to_string(),
            Self::Mars => "Mars Geographic (IAU:MARS)".to_string(),
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        let up = s.trim().to_ascii_uppercase();
        let norm = up.strip_prefix("EPSG:").unwrap_or(&up);
        match norm {
            "LOCAL" | "CARTESIAN" => Ok(Self::Local),
            "4326" | "WGS84" | "WGS_1984" => Ok(Self::Wgs84),
            "4978" | "ECEF" => Ok(Self::Ecef),
            "2154" | "LAMBERT93" | "LAMBERT-93" | "L93" | "RGF93" => Ok(Self::Lambert93),
            "MOON" | "IAU:MOON" => Ok(Self::Moon),
            "MARS" | "IAU:MARS" => Ok(Self::Mars),
            code if code.len() == 5 && (code.starts_with("326") || code.starts_with("327")) => {
                let north = code.starts_with("326");
                let zone: u8 = code[3..]
                    .parse()
                    .map_err(|_| PipelineError::Config(format!("Invalid UTM EPSG code: {}", s)))?;
                if (1..=60).contains(&zone) {
                    Ok(Self::Utm { zone, north })
                } else {
                    Err(PipelineError::Config(format!("UTM zone out of range 1..=60: {}", zone)))
                }
            }
            code if code.starts_with("UTM") => {
                let rest = code.trim_start_matches("UTM").trim_start_matches(['_', '-', ' ']);
                let (num_str, north) = if let Some(n) = rest.strip_suffix('N') {
                    (n, true)
                } else if let Some(n) = rest.strip_suffix('S') {
                    (n, false)
                } else {
                    (rest, true)
                };
                let zone: u8 = num_str
                    .parse()
                    .map_err(|_| PipelineError::Config(format!("Invalid UTM zone: {}", s)))?;
                if (1..=60).contains(&zone) {
                    Ok(Self::Utm { zone, north })
                } else {
                    Err(PipelineError::Config(format!("UTM zone out of range 1..=60: {}", zone)))
                }
            }
            _ => Err(PipelineError::Config(format!("Unsupported CRS '{}'", s))),
        }
    }

    pub fn catalog() -> Vec<Self> {
        vec![
            Self::Local,
            Self::Wgs84,
            Self::Ecef,
            Self::Lambert93,
            Self::Utm { zone: 31, north: true },
            Self::Utm { zone: 32, north: true },
            Self::Utm { zone: 20, north: true },
            Self::Utm { zone: 40, north: false },
            Self::Moon,
            Self::Mars,
        ]
    }
}

/// Exact GRS80/IGN Lambert-93 (`EPSG:2154`) parameters (IAG / IGN Alg0003 & Alg0004).
mod lambert93 {
    use crate::math::Vec3;

    const A: f64 = 6_378_137.0;
    const INV_F: f64 = 298.257_222_101;
    const E2: f64 = 2.0 / INV_F - 1.0 / (INV_F * INV_F);
    const PHI_1_DEG: f64 = 44.0;
    const PHI_2_DEG: f64 = 49.0;
    const PHI_0_DEG: f64 = 46.5;
    const LON_0_DEG: f64 = 3.0;
    const X_S: f64 = 700_000.0;
    const Y_0: f64 = 6_600_000.0;

    #[inline]
    fn isometric_lat(phi: f64, e: f64) -> f64 {
        let s = e * phi.sin();
        ((std::f64::consts::FRAC_PI_4 + phi * 0.5).tan()).ln()
            - 0.5 * e * ((1.0 + s) / (1.0 - s)).ln()
    }

    fn params() -> (f64, f64, f64, f64) {
        let e = E2.sqrt();
        let phi1 = PHI_1_DEG.to_radians();
        let phi2 = PHI_2_DEG.to_radians();
        let phi0 = PHI_0_DEG.to_radians();
        let l1 = isometric_lat(phi1, e);
        let l2 = isometric_lat(phi2, e);
        let l0 = isometric_lat(phi0, e);

        let w1 = (1.0 - E2 * phi1.sin().powi(2)).sqrt();
        let w2 = (1.0 - E2 * phi2.sin().powi(2)).sqrt();
        let n = ((phi2.cos() / w2) / (phi1.cos() / w1)).ln() / (l1 - l2);
        let c = (A * phi1.cos() / (n * w1)) * (n * l1).exp();
        let y_s = Y_0 + c * (-n * l0).exp();
        (e, n, c, y_s)
    }

    /// `(lon_deg, lat_deg, h_m) -> (Easting_m, Northing_m, h_m)` in EPSG:2154.
    pub fn forward(llh: Vec3) -> Vec3 {
        let (e, n, c, y_s) = params();
        let lon = llh.x.to_radians();
        let lat = llh.y.to_radians();
        let lon0 = LON_0_DEG.to_radians();
        let l = isometric_lat(lat, e);
        let r = c * (-n * l).exp();
        let gamma = n * (lon - lon0);
        Vec3::new(X_S + r * gamma.sin(), y_s - r * gamma.cos(), llh.z)
    }

    /// `(Easting_m, Northing_m, h_m) -> (lon_deg, lat_deg, h_m)` from EPSG:2154.
    pub fn inverse(enh: Vec3) -> Vec3 {
        let (e, n, c, y_s) = params();
        let dx = enh.x - X_S;
        let dy = y_s - enh.y;
        let r = dx.hypot(dy);
        let gamma = dx.atan2(dy);
        let lon = LON_0_DEG.to_radians() + gamma / n;
        let l = -(r / c).ln() / n;

        // Iterative inverse isometric latitude (converges to < 1e-13 rad in 6 iterations)
        let mut phi = 2.0 * l.exp().atan() - std::f64::consts::FRAC_PI_2;
        for _ in 0..10 {
            let s = e * phi.sin();
            let next = 2.0 * (l.exp() * ((1.0 + s) / (1.0 - s)).powf(e * 0.5)).atan()
                - std::f64::consts::FRAC_PI_2;
            if (next - phi).abs() < 1e-14 {
                phi = next;
                break;
            }
            phi = next;
        }
        Vec3::new(lon.to_degrees(), phi.to_degrees(), enh.z)
    }
}

/// WGS84 Transverse Mercator (UTM) projection in double precision (`f64`).
mod utm {
    use crate::math::Vec3;

    const A: f64 = 6_378_137.0;
    const F: f64 = 1.0 / 298.257_223_563;
    const E2: f64 = 2.0 * F - F * F;
    const EP2: f64 = E2 / (1.0 - E2);
    const K0: f64 = 0.9996;

    pub fn forward(llh: Vec3, zone: u8, north: bool) -> Vec3 {
        let lon0 = ((zone as f64 - 1.0) * 6.0 - 180.0 + 3.0).to_radians();
        let phi = llh.y.to_radians();
        let lam = llh.x.to_radians();
        let sin_p = phi.sin();
        let cos_p = phi.cos();
        let tan_p = phi.tan();

        let n = A / (1.0 - E2 * sin_p * sin_p).sqrt();
        let t = tan_p * tan_p;
        let c = EP2 * cos_p * cos_p;
        let a_term = cos_p * (lam - lon0);

        let e4 = E2 * E2;
        let e6 = e4 * E2;
        let m = A
            * ((1.0 - E2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0) * phi
                - (3.0 * E2 / 8.0 + 3.0 * e4 / 32.0 + 45.0 * e6 / 1024.0) * (2.0 * phi).sin()
                + (15.0 * e4 / 256.0 + 45.0 * e6 / 1024.0) * (4.0 * phi).sin()
                - (35.0 * e6 / 3072.0) * (6.0 * phi).sin());

        let easting = 500_000.0
            + K0 * n
                * (a_term
                    + (1.0 - t + c) * a_term.powi(3) / 6.0
                    + (5.0 - 18.0 * t + t * t + 72.0 * c - 58.0 * EP2) * a_term.powi(5) / 120.0);
        let mut northing = K0
            * (m + n
                * tan_p
                * (a_term.powi(2) / 2.0
                    + (5.0 - t + 9.0 * c + 4.0 * c * c) * a_term.powi(4) / 24.0
                    + (61.0 - 58.0 * t + t * t + 600.0 * c - 330.0 * EP2) * a_term.powi(6) / 720.0));
        if !north {
            northing += 10_000_000.0;
        }
        Vec3::new(easting, northing, llh.z)
    }

    pub fn inverse(enh: Vec3, zone: u8, north: bool) -> Vec3 {
        let lon0 = ((zone as f64 - 1.0) * 6.0 - 180.0 + 3.0).to_radians();
        let x = enh.x - 500_000.0;
        let y = if north { enh.y } else { enh.y - 10_000_000.0 };

        let e4 = E2 * E2;
        let e6 = e4 * E2;
        let e1 = (1.0 - (1.0 - E2).sqrt()) / (1.0 + (1.0 - E2).sqrt());
        let m = y / K0;
        let mu = m / (A * (1.0 - E2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0));

        let phi1 = mu
            + (3.0 * e1 / 2.0 - 27.0 * e1.powi(3) / 32.0) * (2.0 * mu).sin()
            + (21.0 * e1.powi(2) / 16.0 - 55.0 * e1.powi(4) / 32.0) * (4.0 * mu).sin()
            + (151.0 * e1.powi(3) / 96.0) * (6.0 * mu).sin()
            + (1097.0 * e1.powi(4) / 512.0) * (8.0 * mu).sin();

        let sin_p = phi1.sin();
        let cos_p = phi1.cos();
        let tan_p = phi1.tan();
        let c1 = EP2 * cos_p * cos_p;
        let t1 = tan_p * tan_p;
        let n1 = A / (1.0 - E2 * sin_p * sin_p).sqrt();
        let r1 = A * (1.0 - E2) / (1.0 - E2 * sin_p * sin_p).powf(1.5);
        let d = x / (n1 * K0);

        let lat = phi1
            - (n1 * tan_p / r1)
                * (d * d / 2.0
                    - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1 * c1 - 9.0 * EP2) * d.powi(4) / 24.0
                    + (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * t1 * t1 - 252.0 * EP2 - 3.0 * c1 * c1)
                        * d.powi(6)
                        / 720.0);
        let lon = lon0
            + (d - (1.0 + 2.0 * t1 + c1) * d.powi(3) / 6.0
                + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1 * c1 + 8.0 * EP2 + 24.0 * t1 * t1)
                    * d.powi(5)
                    / 120.0)
                / cos_p;
        Vec3::new(lon.to_degrees(), lat.to_degrees(), enh.z)
    }
}

/// Convert a 3D point `(x, y, z)` between any two supported Earth CRS definitions in `f64`.
pub fn transform_point(src: CrsId, dst: CrsId, p: Vec3) -> Result<Vec3> {
    if src == dst {
        return Ok(p);
    }
    if src == CrsId::Local || dst == CrsId::Local {
        return Err(PipelineError::Geometry(
            "Cannot geodetically reproject Local Cartesian coordinates without a ground anchor".into(),
        ));
    }
    if src == CrsId::Moon || dst == CrsId::Moon || src == CrsId::Mars || dst == CrsId::Mars {
        return Err(PipelineError::Geometry(format!(
            "Cross-planetary reprojection ({} -> {}) is invalid",
            src.code(),
            dst.code()
        )));
    }

    // Step 1: Convert source -> WGS84 geographic (lon_deg, lat_deg, h_m)
    let wgs = match src {
        CrsId::Wgs84 => p,
        CrsId::Ecef => Datum::WGS84.ecef_to_geodetic(p),
        CrsId::Lambert93 => lambert93::inverse(p),
        CrsId::Utm { zone, north } => utm::inverse(p, zone, north),
        _ => unreachable!(),
    };

    // Step 2: Convert WGS84 geographic -> destination
    let out = match dst {
        CrsId::Wgs84 => wgs,
        CrsId::Ecef => Datum::WGS84.geodetic_to_ecef(wgs.x, wgs.y, wgs.z),
        CrsId::Lambert93 => lambert93::forward(wgs),
        CrsId::Utm { zone, north } => utm::forward(wgs, zone, north),
        _ => unreachable!(),
    };
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lambert93_roundtrip_submillimeter() {
        // Paris Notre-Dame approx: lon=2.3499, lat=48.8530, h=35.0
        let wgs = Vec3::new(2.3499, 48.8530, 35.0);
        let l93 = transform_point(CrsId::Wgs84, CrsId::Lambert93, wgs).unwrap();
        assert!((l93.x - 652_300.0).abs() < 500.0, "Unexpected L93 Easting: {}", l93.x);
        assert!((l93.y - 6_861_600.0).abs() < 500.0, "Unexpected L93 Northing: {}", l93.y);

        let back = transform_point(CrsId::Lambert93, CrsId::Wgs84, l93).unwrap();
        assert!((back.x - wgs.x).abs() < 1e-8);
        assert!((back.y - wgs.y).abs() < 1e-8);
        assert!((back.z - wgs.z).abs() < 1e-6);
    }

    #[test]
    fn test_utm_roundtrip_and_l93_to_utm() {
        let wgs = Vec3::new(5.3698, 43.2965, 120.5); // Marseille (UTM 31N)
        let utm31 = CrsId::Utm { zone: 31, north: true };
        let proj = transform_point(CrsId::Wgs84, utm31, wgs).unwrap();
        let back = transform_point(utm31, CrsId::Wgs84, proj).unwrap();
        assert!((back.x - wgs.x).abs() < 1e-7);
        assert!((back.y - wgs.y).abs() < 1e-7);

        let l93 = transform_point(utm31, CrsId::Lambert93, proj).unwrap();
        let back_utm = transform_point(CrsId::Lambert93, utm31, l93).unwrap();
        assert!((back_utm.x - proj.x).abs() < 1e-3);
        assert!((back_utm.y - proj.y).abs() < 1e-3);
    }
}
