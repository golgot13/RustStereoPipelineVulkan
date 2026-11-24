//! Linear algebra, 3D geometry, geodesy (WGS84/Moon/Mars), and numerical solvers.
//! Pure Rust standard library implementation with zero external dependencies.

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    #[inline]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    #[inline]
    pub fn dot(self, o: Self) -> f64 {
        self.x * o.x + self.y * o.y
    }

    #[inline]
    pub fn norm_sq(self) -> f64 {
        self.dot(self)
    }

    #[inline]
    pub fn norm(self) -> f64 {
        self.norm_sq().sqrt()
    }
}

impl Add for Vec2 {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y)
    }
}

impl Sub for Vec2 {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y)
    }
}

impl Mul<f64> for Vec2 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f64) -> Self {
        Self::new(self.x * s, self.y * s)
    }
}

impl Div<f64> for Vec2 {
    type Output = Self;
    #[inline]
    fn div(self, s: f64) -> Self {
        Self::new(self.x / s, self.y / s)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };

    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn dot(self, o: Self) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    #[inline]
    pub fn cross(self, o: Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    #[inline]
    pub fn norm_sq(self) -> f64 {
        self.dot(self)
    }

    #[inline]
    pub fn norm(self) -> f64 {
        self.norm_sq().sqrt()
    }

    #[inline]
    pub fn normalized(self) -> Self {
        let n = self.norm();
        if n > 1e-15 {
            self / n
        } else {
            Self::ZERO
        }
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    #[inline]
    pub fn to_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
}

impl Add for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl AddAssign for Vec3 {
    #[inline]
    fn add_assign(&mut self, o: Self) {
        self.x += o.x;
        self.y += o.y;
        self.z += o.z;
    }
}

impl Sub for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

impl SubAssign for Vec3 {
    #[inline]
    fn sub_assign(&mut self, o: Self) {
        self.x -= o.x;
        self.y -= o.y;
        self.z -= o.z;
    }
}

impl Neg for Vec3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f64) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
}

impl Div<f64> for Vec3 {
    type Output = Self;
    #[inline]
    fn div(self, s: f64) -> Self {
        Self::new(self.x / s, self.y / s, self.z / s)
    }
}

/// Row-major 3x3 matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3 {
    pub m: [[f64; 3]; 3],
}

impl Default for Mat3 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Mat3 {
    pub const IDENTITY: Self = Self {
        m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };

    pub const ZERO: Self = Self {
        m: [[0.0; 3]; 3],
    };

    #[inline]
    pub const fn new(m: [[f64; 3]; 3]) -> Self {
        Self { m }
    }

    pub fn from_euler_xyz(rx: f64, ry: f64, rz: f64) -> Self {
        let (sx, cx) = rx.sin_cos();
        let (sy, cy) = ry.sin_cos();
        let (sz, cz) = rz.sin_cos();

        let mx = Self::new([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]]);
        let my = Self::new([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]]);
        let mz = Self::new([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]]);
        mz.mul_mat(&my).mul_mat(&mx)
    }

    /// Standard aerial photogrammetry OPK camera-to-world rotation (PMS3D / IGN / AEROPHOTOGRAM YXZ):
    /// `R_c2w = Ry(phi) * Rx(omega) * Rz(kappa)`.
    pub fn from_opk_yxz(omega: f64, phi: f64, kappa: f64) -> Self {
        let (sw, cw) = omega.sin_cos();
        let (sp, cp) = phi.sin_cos();
        let (sk, ck) = kappa.sin_cos();

        let rx = Self::new([[1.0, 0.0, 0.0], [0.0, cw, -sw], [0.0, sw, cw]]);
        let ry = Self::new([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]]);
        let rz = Self::new([[ck, -sk, 0.0], [sk, ck, 0.0], [0.0, 0.0, 1.0]]);
        ry.mul_mat(&rx).mul_mat(&rz)
    }

    #[inline]
    pub fn transpose(&self) -> Self {
        Self::new([
            [self.m[0][0], self.m[1][0], self.m[2][0]],
            [self.m[0][1], self.m[1][1], self.m[2][1]],
            [self.m[0][2], self.m[1][2], self.m[2][2]],
        ])
    }

    #[inline]
    pub fn mul_vec(&self, v: Vec3) -> Vec3 {
        Vec3::new(
            self.m[0][0] * v.x + self.m[0][1] * v.y + self.m[0][2] * v.z,
            self.m[1][0] * v.x + self.m[1][1] * v.y + self.m[1][2] * v.z,
            self.m[2][0] * v.x + self.m[2][1] * v.y + self.m[2][2] * v.z,
        )
    }

    pub fn mul_mat(&self, o: &Self) -> Self {
        let mut r = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = self.m[i][0] * o.m[0][j]
                    + self.m[i][1] * o.m[1][j]
                    + self.m[i][2] * o.m[2][j];
            }
        }
        Self::new(r)
    }

    pub fn det(&self) -> f64 {
        let m = &self.m;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    pub fn inverse(&self) -> Option<Self> {
        let d = self.det();
        if d.abs() < 1e-18 {
            return None;
        }
        let inv_d = 1.0 / d;
        let m = &self.m;
        Some(Self::new([
            [
                (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_d,
                (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_d,
                (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_d,
            ],
            [
                (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_d,
                (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_d,
                (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_d,
            ],
            [
                (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_d,
                (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_d,
                (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_d,
            ],
        ]))
    }

    pub fn solve(&self, b: Vec3) -> Option<Vec3> {
        self.inverse().map(|inv| inv.mul_vec(b))
    }

    /// Jacobi eigendecomposition of a symmetric 3x3 matrix.
    /// Returns `(eigenvalues, eigenvectors_columns)`.
    pub fn symmetric_eigen(&self) -> ([f64; 3], Mat3) {
        let mut a = self.m;
        let mut v = Mat3::IDENTITY.m;

        for _ in 0..40 {
            let mut p = 0;
            let mut q = 1;
            let mut max_off = a[0][1].abs();
            if a[0][2].abs() > max_off {
                p = 0;
                q = 2;
                max_off = a[0][2].abs();
            }
            if a[1][2].abs() > max_off {
                p = 1;
                q = 2;
                max_off = a[1][2].abs();
            }
            if max_off < 1e-15 {
                break;
            }
            let diff = a[q][q] - a[p][p];
            let t = if a[p][q].abs() < 1e-30 * diff.abs() {
                a[p][q] / diff
            } else {
                let phi = diff / (2.0 * a[p][q]);
                let mut t = 1.0 / (phi.abs() + (phi * phi + 1.0).sqrt());
                if phi < 0.0 {
                    t = -t;
                }
                t
            };
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;
            let tau = s / (1.0 + c);

            let apq = a[p][q];
            a[p][q] = 0.0;
            a[q][p] = 0.0;
            a[p][p] -= t * apq;
            a[q][q] += t * apq;

            for r in 0..3 {
                if r != p && r != q {
                    let arp = a[r][p];
                    let arq = a[r][q];
                    a[r][p] = arp - s * (arq + tau * arp);
                    a[p][r] = a[r][p];
                    a[r][q] = arq + s * (arp - tau * arq);
                    a[q][r] = a[r][q];
                }
            }
            for r in 0..3 {
                let vrp = v[r][p];
                let vrq = v[r][q];
                v[r][p] = vrp - s * (vrq + tau * vrp);
                v[r][q] = vrq + s * (vrp - tau * vrq);
            }
        }
        ([a[0][0], a[1][1], a[2][2]], Mat3::new(v))
    }

    /// Nearest proper rotation matrix R (det(R) = +1) maximizing trace(R^T * self)
    /// via polar decomposition (Kabsch algorithm).
    pub fn closest_rotation(&self) -> Mat3 {
        let ata = self.transpose().mul_mat(self);
        let (evals, v) = ata.symmetric_eigen();
        // Sort descending by eigenvalue
        let mut order = [0usize, 1, 2];
        order.sort_by(|&i, &j| evals[j].partial_cmp(&evals[i]).unwrap_or(std::cmp::Ordering::Equal));

        let mut v_sorted = [[0.0; 3]; 3];
        let mut s_sorted = [0.0; 3];
        for (col_new, &col_old) in order.iter().enumerate() {
            s_sorted[col_new] = evals[col_old].max(0.0).sqrt();
            for row in 0..3 {
                v_sorted[row][col_new] = v.m[row][col_old];
            }
        }
        let v_mat = Mat3::new(v_sorted);
        let av = self.mul_mat(&v_mat);

        // Build U columns from A * V / sigma
        let mut u_cols = [Vec3::ZERO; 3];
        for k in 0..3 {
            let col = Vec3::new(av.m[0][k], av.m[1][k], av.m[2][k]);
            if s_sorted[k] > 1e-12 {
                u_cols[k] = col / s_sorted[k];
            } else if k == 2 {
                u_cols[2] = u_cols[0].cross(u_cols[1]).normalized();
            } else {
                u_cols[k] = Vec3::new(if k == 0 { 1.0 } else { 0.0 }, if k == 1 { 1.0 } else { 0.0 }, 0.0);
            }
        }
        // Gram-Schmidt orthonormalize U
        u_cols[0] = u_cols[0].normalized();
        u_cols[1] = (u_cols[1] - u_cols[0] * u_cols[0].dot(u_cols[1])).normalized();
        u_cols[2] = u_cols[0].cross(u_cols[1]).normalized();

        let u_mat = Mat3::new([
            [u_cols[0].x, u_cols[1].x, u_cols[2].x],
            [u_cols[0].y, u_cols[1].y, u_cols[2].y],
            [u_cols[0].z, u_cols[1].z, u_cols[2].z],
        ]);

        let mut r = u_mat.mul_mat(&v_mat.transpose());
        if r.det() < 0.0 {
            let fix = Mat3::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]);
            r = u_mat.mul_mat(&fix).mul_mat(&v_mat.transpose());
        }
        r
    }
}

/// 2D integer bounding box `[min_x, max_x) x [min_y, max_y)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BBox2i {
    pub min_x: i32,
    pub min_y: i32,
    pub max_x: i32,
    pub max_y: i32,
}

impl BBox2i {
    pub const fn new(min_x: i32, min_y: i32, max_x: i32, max_y: i32) -> Self {
        Self { min_x, min_y, max_x, max_y }
    }

    pub const fn width(&self) -> i32 {
        let d = self.max_x - self.min_x;
        if d > 0 { d } else { 0 }
    }

    pub const fn height(&self) -> i32 {
        let d = self.max_y - self.min_y;
        if d > 0 { d } else { 0 }
    }

    pub const fn is_empty(&self) -> bool {
        self.max_x <= self.min_x || self.max_y <= self.min_y
    }
}

/// Planetary geodetic datum (WGS84, Moon, Mars, Local Cartesian).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Datum {
    pub name: &'static str,
    pub semi_major_axis: f64,
    pub semi_minor_axis: f64,
}

impl Datum {
    pub const WGS84: Self = Self {
        name: "WGS_1984",
        semi_major_axis: 6_378_137.0,
        semi_minor_axis: 6_356_752.314_245_179,
    };

    pub const MOON: Self = Self {
        name: "D_MOON",
        semi_major_axis: 1_737_400.0,
        semi_minor_axis: 1_737_400.0,
    };

    pub const MARS: Self = Self {
        name: "D_MARS",
        semi_major_axis: 3_396_190.0,
        semi_minor_axis: 3_376_200.0,
    };

    pub fn from_name(name: &str) -> Self {
        match name.to_ascii_uppercase().as_str() {
            "MOON" | "D_MOON" => Self::MOON,
            "MARS" | "D_MARS" => Self::MARS,
            _ => Self::WGS84,
        }
    }

    /// Convert geodetic `(lon_deg, lat_deg, height_m)` to ECEF XYZ meters.
    pub fn geodetic_to_ecef(&self, lon_deg: f64, lat_deg: f64, height_m: f64) -> Vec3 {
        let lon = lon_deg.to_radians();
        let lat = lat_deg.to_radians();
        let a = self.semi_major_axis;
        let b = self.semi_minor_axis;
        let e2 = 1.0 - (b * b) / (a * a);
        let sin_lat = lat.sin();
        let cos_lat = lat.cos();
        let n = a / (1.0 - e2 * sin_lat * sin_lat).sqrt();
        Vec3::new(
            (n + height_m) * cos_lat * lon.cos(),
            (n + height_m) * cos_lat * lon.sin(),
            (n * (1.0 - e2) + height_m) * sin_lat,
        )
    }

    /// Convert ECEF XYZ meters to geodetic `(lon_deg, lat_deg, height_m)` using Bowring iteration.
    pub fn ecef_to_geodetic(&self, p: Vec3) -> Vec3 {
        let a = self.semi_major_axis;
        let b = self.semi_minor_axis;
        let e2 = 1.0 - (b * b) / (a * a);
        let lon = p.y.atan2(p.x);
        let r = (p.x * p.x + p.y * p.y).sqrt();
        let mut lat = (p.z / (r * (1.0 - e2))).atan();
        for _ in 0..8 {
            let sin_lat = lat.sin();
            let n = a / (1.0 - e2 * sin_lat * sin_lat).sqrt();
            lat = ((p.z + e2 * n * sin_lat) / r).atan();
        }
        let sin_lat = lat.sin();
        let cos_lat = lat.cos();
        let n = a / (1.0 - e2 * sin_lat * sin_lat).sqrt();
        let height = if cos_lat.abs() > 1e-8 {
            r / cos_lat - n
        } else {
            p.z.abs() / sin_lat.abs() - n * (1.0 - e2)
        };
        Vec3::new(lon.to_degrees(), lat.to_degrees(), height)
    }
}

/// Closest midpoint between two 3D rays `(o1, d1)` and `(o2, d2)`.
/// Returns `(midpoint, intersection_distance_error, depth_t1, depth_t2)`.
#[inline]
pub fn triangulate_rays(o1: Vec3, d1: Vec3, o2: Vec3, d2: Vec3) -> Option<(Vec3, f64, f64, f64)> {
    let u = d1.normalized();
    let v = d2.normalized();
    let w0 = o1 - o2;
    let a = u.dot(u);
    let b = u.dot(v);
    let c = v.dot(v);
    let d = u.dot(w0);
    let e = v.dot(w0);
    let denom = a * c - b * b;
    if denom.abs() < 1e-7 {
        return None;
    }
    let t1 = (b * e - c * d) / denom;
    let t2 = (a * e - b * d) / denom;
    let p1 = o1 + u * t1;
    let p2 = o2 + v * t2;
    let mid = (p1 + p2) * 0.5;
    let err = (p1 - p2).norm();
    Some((mid, err, t1, t2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geodetic_roundtrip() {
        let d = Datum::WGS84;
        let ecef = d.geodetic_to_ecef(-122.057, 37.415, 120.5);
        let llh = d.ecef_to_geodetic(ecef);
        assert!((llh.x - (-122.057)).abs() < 1e-6);
        assert!((llh.y - 37.415).abs() < 1e-6);
        assert!((llh.z - 120.5).abs() < 1e-3);
    }

    #[test]
    fn test_triangulate_rays() {
        let o1 = Vec3::new(-10.0, 0.0, 100.0);
        let o2 = Vec3::new(10.0, 0.0, 100.0);
        let target = Vec3::new(2.0, -3.0, 15.0);
        let d1 = (target - o1).normalized();
        let d2 = (target - o2).normalized();
        let (pt, err, t1, t2) = triangulate_rays(o1, d1, o2, d2).unwrap();
        assert!((pt - target).norm() < 1e-9);
        assert!(err < 1e-9);
        assert!(t1 > 0.0 && t2 > 0.0);
    }

    #[test]
    fn test_closest_rotation() {
        let rot = Mat3::from_euler_xyz(0.3, -0.5, 0.8);
        let rec = rot.closest_rotation();
        for i in 0..3 {
            for j in 0..3 {
                assert!((rot.m[i][j] - rec.m[i][j]).abs() < 1e-9);
            }
        }
    }
}
