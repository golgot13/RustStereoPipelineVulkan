//! Aerial Chantier Geometric Validation & Subpixel GCP-Pick Pose Refinement (`f64`).
//! - Parses `aerophotogram_footprints_validation.json` (64+ validated ground footprints) and
//!   cross-validates pose positions, corner winding/orientation, and 2D polygon IoU per flight line.
//! - Parses `GCP_picks_user_validated_*.json` (subpixel `[gcp_id, image_id, col_px, row_px]` picks)
//!   and runs two-stage damped `f64` Levenberg-Marquardt refinement:
//!     1. Global 7-DOF Block Calibration `(dX, dY, dZ, d_omega, d_phi, d_kappa, d_focal_mm)`
//!        absorbing GNSS datum shift, IMU boresight, and CC44/elevation scale across ALL poses.
//!     2. Per-Image MAP 6-DOF Levenberg-Marquardt `(X, Y, Z, omega, phi, kappa)` with Huber
//!        robust weighting and GNSS/IMU prior regularization for subpixel epipolar alignment.

use std::collections::HashMap;
use std::path::Path;

use crate::camera::{CameraModel, PinholeCamera};
use crate::core::chantier::{
    build_epipolar_rectified_cameras, build_pinhole_for_pose, CameraCalibration, Chantier,
    GroundControlPoint, ImagePose, StereoPairEntry,
};
use crate::core::crs::CrsId;
use crate::core::error::{PipelineError, Result};
use crate::core::json::parse_json;
use crate::core::lidar_mnt::LidarTerrainModel;
use crate::math::{triangulate_rays, Vec2, Vec3};

/// One user-validated subpixel GCP image observation `[gcp_id, image_id, col_px, row_px]`.
#[derive(Debug, Clone)]
pub struct GcpImagePick {
    pub gcp_id: String,
    pub image_id: String,
    pub col_px: f64,
    pub row_px: f64,
}

/// One validated ground footprint polygon from `aerophotogram_footprints_validation.json`.
#[derive(Debug, Clone)]
pub struct ValidatedFootprint {
    pub pose_id: String,
    pub x: f64,
    pub y: f64,
    pub corners_sol: [[f64; 2]; 4],
}

/// Cross-validation summary comparing camera poses against validated ground footprints.
#[derive(Debug, Clone)]
pub struct FootprintValidationReport {
    pub matched_poses: usize,
    pub total_poses: usize,
    pub mean_center_err_m: f64,
    pub max_center_err_m: f64,
    pub mean_iou_pct: f64,
    pub min_iou_pct: f64,
    pub orientation_ok: usize,
    pub orientation_failed_ids: Vec<String>,
    pub used_lidar_mnt: bool,
}

/// Before/After metrics of the two-stage `f64` LM GCP-pick block refinement.
#[derive(Debug, Clone)]
pub struct GcpRefinementReport {
    pub matched_observations: usize,
    pub matched_gcps: usize,
    pub refined_images: usize,
    pub applied: bool,
    pub initial_reproj_rmse_px: f64,
    pub stage1_reproj_rmse_px: f64,
    pub final_reproj_rmse_px: f64,
    pub initial_epi_dy_rmse_px: f64,
    pub final_epi_dy_rmse_px: f64,
    pub initial_tri_dz_bias_m: f64,
    pub final_tri_dz_bias_m: f64,
    pub initial_tri_xyz_rmse_m: f64,
    pub final_tri_xyz_rmse_m: f64,
    pub global_dx_m: f64,
    pub global_dy_m: f64,
    pub global_dz_m: f64,
    pub global_domega_deg: f64,
    pub global_dphi_deg: f64,
    pub global_dkappa_deg: f64,
    pub global_dfocal_mm: f64,
}

fn normalize_id(s: &str) -> String {
    let low = s.trim().to_ascii_lowercase();
    Path::new(&low)
        .file_stem()
        .and_then(|x| x.to_str())
        .unwrap_or(&low)
        .to_string()
}

/// Parse `GCP_picks_user_validated_*.json` (array of `[gcp_id, image_id, col_px, row_px]`
/// or array of objects `{gcp_id, image_id, u/col, v/row}`).
pub fn parse_gcp_picks_file(path: &Path) -> Result<Vec<GcpImagePick>> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| PipelineError::Io(format!("Cannot read {}: {}", path.display(), e)))?;
    parse_gcp_picks_str(&content)
}

pub fn parse_gcp_picks_str(content: &str) -> Result<Vec<GcpImagePick>> {
    let val = parse_json(content)?;
    let arr = val
        .as_array()
        .or_else(|| val.get("picks").and_then(|v| v.as_array()))
        .ok_or_else(|| PipelineError::Format("GCP picks JSON must be an array".into()))?;
    let mut picks = Vec::with_capacity(arr.len());
    for item in arr {
        if let Some(tuple) = item.as_array() {
            if tuple.len() >= 4 {
                let gcp_id = tuple[0]
                    .as_str()
                    .map(|s| s.trim().to_string())
                    .or_else(|| tuple[0].as_i64().map(|n| n.to_string()));
                let image_id = tuple[1].as_str().map(|s| s.trim().to_string());
                let col = tuple[2].as_f64();
                let row = tuple[3].as_f64();
                if let (Some(g), Some(img), Some(c), Some(r)) = (gcp_id, image_id, col, row) {
                    if c.is_finite() && r.is_finite() {
                        picks.push(GcpImagePick {
                            gcp_id: g,
                            image_id: img,
                            col_px: c,
                            row_px: r,
                        });
                    }
                }
            }
        } else if let Some(obj) = item.as_object() {
            let gcp_id = obj
                .get("gcp_id")
                .or_else(|| obj.get("gcp"))
                .and_then(|v| v.as_str().map(|s| s.to_string()).or_else(|| v.as_i64().map(|n| n.to_string())));
            let image_id = obj
                .get("image_id")
                .or_else(|| obj.get("image"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let col = obj.get("col").or_else(|| obj.get("u")).or_else(|| obj.get("x")).and_then(|v| v.as_f64());
            let row = obj.get("row").or_else(|| obj.get("v")).or_else(|| obj.get("y")).and_then(|v| v.as_f64());
            if let (Some(g), Some(img), Some(c), Some(r)) = (gcp_id, image_id, col, row) {
                if c.is_finite() && r.is_finite() {
                    picks.push(GcpImagePick {
                        gcp_id: g,
                        image_id: img,
                        col_px: c,
                        row_px: r,
                    });
                }
            }
        }
    }
    Ok(picks)
}

/// Parse `aerophotogram_footprints_validation.json` into a normalized-ID map.
pub fn parse_footprints_file(path: &Path) -> Result<HashMap<String, ValidatedFootprint>> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| PipelineError::Io(format!("Cannot read {}: {}", path.display(), e)))?;
    parse_footprints_str(&content)
}

pub fn parse_footprints_str(content: &str) -> Result<HashMap<String, ValidatedFootprint>> {
    let val = parse_json(content)?;
    let arr = val
        .get("footprints")
        .and_then(|v| v.as_array())
        .or_else(|| val.as_array())
        .ok_or_else(|| PipelineError::Format("Footprints JSON missing 'footprints' array".into()))?;
    let mut map = HashMap::with_capacity(arr.len());
    for item in arr {
        let pose_id = match item.get("pose_id").and_then(|v| v.as_str()) {
            Some(s) => s.trim().to_string(),
            None => continue,
        };
        let x = item.get("x").and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
        let y = item.get("y").and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
        let corners_arr = match item.get("corners_sol").and_then(|v| v.as_array()) {
            Some(a) if a.len() >= 4 => a,
            _ => continue,
        };
        let mut corners = [[0.0f64; 2]; 4];
        let mut ok = x.is_finite() && y.is_finite();
        for ci in 0..4 {
            if let Some(pair) = corners_arr[ci].as_array() {
                if pair.len() >= 2 {
                    if let (Some(cx), Some(cy)) = (pair[0].as_f64(), pair[1].as_f64()) {
                        if cx.is_finite() && cy.is_finite() {
                            corners[ci] = [cx, cy];
                            continue;
                        }
                    }
                }
            }
            ok = false;
        }
        if ok {
            let key = normalize_id(&pose_id);
            map.insert(
                key,
                ValidatedFootprint {
                    pose_id,
                    x,
                    y,
                    corners_sol: corners,
                },
            );
        }
    }
    Ok(map)
}

/// 2D polygon signed area (`> 0` for CCW, `< 0` for CW).
fn polygon_signed_area(poly: &[[f64; 2]]) -> f64 {
    if poly.len() < 3 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..poly.len() {
        let a = poly[i];
        let b = poly[(i + 1) % poly.len()];
        sum += a[0] * b[1] - b[0] * a[1];
    }
    0.5 * sum
}

fn ensure_ccw(poly: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut v = poly.to_vec();
    if polygon_signed_area(&v) < 0.0 {
        v.reverse();
    }
    v
}

/// Exact `f64` Sutherland-Hodgman intersection of two convex 2D polygons.
pub fn intersect_convex_polygons(a: &[[f64; 2]], b: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut output = ensure_ccw(a);
    let clip = ensure_ccw(b);
    if output.len() < 3 || clip.len() < 3 {
        return Vec::new();
    }

    for i in 0..clip.len() {
        if output.is_empty() {
            break;
        }
        let cp1 = clip[i];
        let cp2 = clip[(i + 1) % clip.len()];
        let edge_x = cp2[0] - cp1[0];
        let edge_y = cp2[1] - cp1[1];
        let inside = |p: [f64; 2]| -> bool { edge_x * (p[1] - cp1[1]) - edge_y * (p[0] - cp1[0]) >= -1e-9 };

        let input = std::mem::take(&mut output);
        let mut s = *input.last().unwrap();
        for &e in &input {
            let e_in = inside(e);
            let s_in = inside(s);
            let compute_inter = |p1: [f64; 2], p2: [f64; 2]| -> [f64; 2] {
                let d1 = edge_x * (p1[1] - cp1[1]) - edge_y * (p1[0] - cp1[0]);
                let d2 = edge_x * (p2[1] - cp1[1]) - edge_y * (p2[0] - cp1[0]);
                let denom = d1 - d2;
                let t = if denom.abs() > 1e-12 { (d1 / denom).clamp(0.0, 1.0) } else { 0.5 };
                [p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])]
            };
            if e_in {
                if !s_in {
                    output.push(compute_inter(s, e));
                }
                output.push(e);
            } else if s_in {
                output.push(compute_inter(s, e));
            }
            s = e;
        }
    }
    output
}

fn polygon_iou(a: &[[f64; 2]], b: &[[f64; 2]]) -> f64 {
    let area_a = polygon_signed_area(a).abs();
    let area_b = polygon_signed_area(b).abs();
    if area_a <= 1e-6 || area_b <= 1e-6 {
        return 0.0;
    }
    let inter = intersect_convex_polygons(a, b);
    let area_i = polygon_signed_area(&inter).abs();
    let union = (area_a + area_b - area_i).max(1e-6);
    (area_i / union).clamp(0.0, 1.0)
}

fn local_ground_z_at(gcps: &[GroundControlPoint], x: f64, y: f64, fallback_z: f64) -> f64 {
    if gcps.is_empty() {
        return fallback_z;
    }
    let mut by_dist: Vec<(f64, f64)> = gcps.iter().map(|g| ((g.x - x).hypot(g.y - y), g.z)).collect();
    by_dist.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let k = by_dist.len().min(5);
    let mut sw = 0.0;
    let mut sz = 0.0;
    for &(d, z) in &by_dist[..k] {
        let w = 1.0 / (d * d + 25.0);
        sw += w;
        sz += w * z;
    }
    if sw > 0.0 { sz / sw } else { fallback_z }
}

/// Cross-validate all discovered camera poses against `aerophotogram_footprints_validation.json`.
/// When `lidar_mnt` is available, projects all 4 sensor corners via exact iterative `f64` ray-MNT
/// intersection (real local AGL terrain elevation) instead of a flat reference plane.
pub fn validate_chantier_footprints(
    cal: &CameraCalibration,
    poses: &[ImagePose],
    gcps: &[GroundControlPoint],
    footprints: &HashMap<String, ValidatedFootprint>,
    lidar_mnt: Option<&LidarTerrainModel>,
    survey_crs: CrsId,
) -> FootprintValidationReport {
    let w = cal.width_px.max(1);
    let h = cal.height_px.max(1);
    let wf = w as f64;
    let hf = h as f64;
    let corners_px = [(0.0, 0.0), (wf, 0.0), (wf, hf), (0.0, hf)];

    let mut matched = 0usize;
    let mut sum_center_err = 0.0;
    let mut max_center_err = 0.0f64;
    let mut sum_iou = 0.0;
    let mut min_iou = 1.0f64;
    let mut orient_ok = 0usize;
    let mut orient_failed = Vec::new();
    let mut used_lidar_mnt = false;

    for pose in poses {
        let key = normalize_id(&pose.id);
        let Some(fp) = footprints.get(&key) else {
            continue;
        };
        let fallback_z = if cal.flight_height_m > 10.0 {
            pose.z - cal.flight_height_m
        } else {
            pose.z - 1000.0
        };
        let gz = lidar_mnt
            .and_then(|m| m.sample_ground_z(pose.x, pose.y, survey_crs))
            .unwrap_or_else(|| local_ground_z_at(gcps, pose.x, pose.y, fallback_z));
        let cam = build_pinhole_for_pose(cal, pose, w, h, 0, 0);

        let mut proj_corners = [[0.0f64; 2]; 4];
        let mut valid = true;
        for (ci, &(u, v)) in corners_px.iter().enumerate() {
            let ray = cam.pixel_to_vector(Vec2::new(u, v));
            if ray.z.abs() < 1e-8 {
                valid = false;
                break;
            }
            let hit = if let Some(mnt) = lidar_mnt {
                if let Some(p) = mnt.intersect_ray_ground(cam.center, ray, gz, survey_crs) {
                    used_lidar_mnt = true;
                    Some(p)
                } else {
                    let t = (gz - cam.center.z) / ray.z;
                    if t > 0.0 && t.is_finite() { Some(cam.center + ray * t) } else { None }
                }
            } else {
                let t = (gz - cam.center.z) / ray.z;
                if t > 0.0 && t.is_finite() { Some(cam.center + ray * t) } else { None }
            };
            match hit {
                Some(p) => proj_corners[ci] = [p.x, p.y],
                None => {
                    valid = false;
                    break;
                }
            }
        }
        if !valid {
            orient_failed.push(pose.id.clone());
            continue;
        }

        matched += 1;
        let c_err = (pose.x - fp.x).hypot(pose.y - fp.y);
        sum_center_err += c_err;
        if c_err > max_center_err {
            max_center_err = c_err;
        }

        let iou = polygon_iou(&proj_corners, &fp.corners_sol);
        sum_iou += iou;
        if iou < min_iou {
            min_iou = iou;
        }

        // Check corner-order orientation (catches 180-deg kappa inversion or flipped axes!)
        let d_direct: f64 = (0..4)
            .map(|i| (proj_corners[i][0] - fp.corners_sol[i][0]).hypot(proj_corners[i][1] - fp.corners_sol[i][1]))
            .sum();
        let d_flipped: f64 = (0..4)
            .map(|i| {
                let j = (i + 2) % 4;
                (proj_corners[i][0] - fp.corners_sol[j][0]).hypot(proj_corners[i][1] - fp.corners_sol[j][1])
            })
            .sum();
        if iou > 0.70 && d_direct < d_flipped * 0.5 {
            orient_ok += 1;
        } else {
            orient_failed.push(pose.id.clone());
        }
    }

    FootprintValidationReport {
        matched_poses: matched,
        total_poses: poses.len(),
        mean_center_err_m: if matched > 0 { sum_center_err / matched as f64 } else { 0.0 },
        max_center_err_m: max_center_err,
        mean_iou_pct: if matched > 0 { 100.0 * sum_iou / matched as f64 } else { 0.0 },
        min_iou_pct: if matched > 0 { 100.0 * min_iou } else { 0.0 },
        orientation_ok: orient_ok,
        orientation_failed_ids: orient_failed,
        used_lidar_mnt,
    }
}

/// Compute validated ground overlap area (m^2) and projected rectified bounds `(u_min, u_max, v_min, v_max)`
/// from two validated ground footprints when available (using local LiDAR MNT elevation if loaded).
pub fn validated_pair_overlap_bounds(
    footprints: &HashMap<String, ValidatedFootprint>,
    left_id: &str,
    right_id: &str,
    ground_z: f64,
    rect_cam_l: &PinholeCamera,
    lidar_mnt: Option<&LidarTerrainModel>,
    survey_crs: CrsId,
) -> Option<(f64, f64, f64, f64, f64)> {
    let fp_l = footprints.get(&normalize_id(left_id))?;
    let fp_r = footprints.get(&normalize_id(right_id))?;
    let inter = intersect_convex_polygons(&fp_l.corners_sol, &fp_r.corners_sol);
    if inter.len() < 3 {
        return None;
    }
    let area_m2 = polygon_signed_area(&inter).abs();
    if area_m2 < 10.0 {
        return None;
    }
    let mut min_u = f64::INFINITY;
    let mut max_u = f64::NEG_INFINITY;
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    // Sample intersection polygon across local terrain/canopy vertical margin so relief is never clipped
    for &[gx, gy] in &inter {
        let zv = lidar_mnt
            .and_then(|m| m.sample_ground_z(gx, gy, survey_crs))
            .unwrap_or(ground_z);
        for &dz in &[-25.0, 0.0, 40.0] {
            let rp = rect_cam_l.point_to_pixel(Vec3::new(gx, gy, zv + dz));
            if rp.x.is_finite() && rp.y.is_finite() {
                if rp.x < min_u { min_u = rp.x; }
                if rp.x > max_u { max_u = rp.x; }
                if rp.y < min_v { min_v = rp.y; }
                if rp.y > max_v { max_v = rp.y; }
            }
        }
    }
    if min_u.is_finite() && min_v.is_finite() && max_u > min_u && max_v > min_v {
        Some((min_u, max_u, min_v, max_v, area_m2))
    } else {
        None
    }
}

/// Solve small dense symmetric/general `N x N` linear system `A * x = b` in `f64` via Gaussian elimination with partial pivoting.
fn solve_linear_f64<const N: usize>(mut a: [[f64; N]; N], mut b: [f64; N]) -> Option<[f64; N]> {
    for col in 0..N {
        let mut pivot = col;
        let mut max_v = a[col][col].abs();
        for row in (col + 1)..N {
            let v = a[row][col].abs();
            if v > max_v {
                max_v = v;
                pivot = row;
            }
        }
        if max_v < 1e-18 || !max_v.is_finite() {
            return None;
        }
        if pivot != col {
            a.swap(col, pivot);
            b.swap(col, pivot);
        }
        let inv_diag = 1.0 / a[col][col];
        for row in (col + 1)..N {
            let factor = a[row][col] * inv_diag;
            if factor.abs() < 1e-24 {
                continue;
            }
            for k in col..N {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = [0.0f64; N];
    for i in (0..N).rev() {
        let mut s = b[i];
        for j in (i + 1)..N {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
        if !x[i].is_finite() {
            return None;
        }
    }
    Some(x)
}

#[inline]
fn huber_weight(res_norm: f64, delta: f64) -> f64 {
    if res_norm <= delta {
        1.0
    } else {
        delta / res_norm
    }
}

struct MatchedObs {
    pose_idx: usize,
    gcp_idx: usize,
    u_obs: f64,
    v_obs: f64,
}

fn eval_reproj_rmse(
    cal: &CameraCalibration,
    poses: &[ImagePose],
    gcps: &[GroundControlPoint],
    obs: &[MatchedObs],
) -> f64 {
    if obs.is_empty() {
        return 0.0;
    }
    let w = cal.width_px.max(1);
    let h = cal.height_px.max(1);
    let cams: Vec<PinholeCamera> = poses.iter().map(|p| build_pinhole_for_pose(cal, p, w, h, 0, 0)).collect();
    let mut sq = 0.0;
    let mut cnt = 0usize;
    for o in obs {
        let g = &gcps[o.gcp_idx];
        let pr = cams[o.pose_idx].point_to_pixel(Vec3::new(g.x, g.y, g.z));
        if pr.x.is_finite() && pr.y.is_finite() {
            let du = pr.x - o.u_obs;
            let dv = pr.y - o.v_obs;
            sq += du * du + dv * dv;
            cnt += 1;
        }
    }
    if cnt > 0 { (sq / (2.0 * cnt as f64)).sqrt() } else { 0.0 }
}

/// Measure epipolar vertical residual `dy` (px) and 3D triangulated GCP errors `(dZ bias, XYZ RMSE)`
/// across all stereo pairs sharing user-validated GCP picks.
fn eval_epipolar_and_triangulation(
    cal: &CameraCalibration,
    poses: &[ImagePose],
    pairs: &[StereoPairEntry],
    gcps: &[GroundControlPoint],
    obs: &[MatchedObs],
) -> (f64, f64, f64) {
    let w = cal.width_px.max(1);
    let h = cal.height_px.max(1);
    let pose_idx_by_id: HashMap<String, usize> = poses
        .iter()
        .enumerate()
        .map(|(i, p)| (normalize_id(&p.id), i))
        .collect();
    let mut obs_by_pose: HashMap<usize, HashMap<usize, (f64, f64)>> = HashMap::new();
    for o in obs {
        obs_by_pose.entry(o.pose_idx).or_default().insert(o.gcp_idx, (o.u_obs, o.v_obs));
    }

    let mut dy_sq = 0.0;
    let mut dy_cnt = 0usize;
    let mut dz_sum = 0.0;
    let mut xyz_sq = 0.0;
    let mut tri_cnt = 0usize;

    for pair in pairs {
        let Some(&il) = pose_idx_by_id.get(&normalize_id(&pair.left_pose.id)) else { continue };
        let Some(&ir) = pose_idx_by_id.get(&normalize_id(&pair.right_pose.id)) else { continue };
        let (Some(ol), Some(or_)) = (obs_by_pose.get(&il), obs_by_pose.get(&ir)) else { continue };

        let cam_l = build_pinhole_for_pose(cal, &poses[il], w, h, 0, 0);
        let cam_r = build_pinhole_for_pose(cal, &poses[ir], w, h, 0, 0);
        let anchor = Vec3::new(
            0.5 * (poses[il].x + poses[ir].x),
            0.5 * (poses[il].y + poses[ir].y),
            poses[il].z - cal.flight_height_m.max(500.0),
        );
        let (rect_l, rect_r) = build_epipolar_rectified_cameras(&cam_l, &cam_r, anchor, 1024, 1024);

        for (&g_idx, &(ul, vl)) in ol {
            let Some(&(ur, vr)) = or_.get(&g_idx) else { continue };
            let dl = cam_l.pixel_to_vector(Vec2::new(ul, vl));
            let dr = cam_r.pixel_to_vector(Vec2::new(ur, vr));

            let rpl = rect_l.point_to_pixel(cam_l.center + dl * 1000.0);
            let rpr = rect_r.point_to_pixel(cam_r.center + dr * 1000.0);
            if rpl.y.is_finite() && rpr.y.is_finite() {
                let dy = rpr.y - rpl.y;
                dy_sq += dy * dy;
                dy_cnt += 1;
            }
            if let Some((pt, _, t1, t2)) = triangulate_rays(cam_l.center, dl, cam_r.center, dr) {
                if t1 > 0.0 && t2 > 0.0 && pt.is_finite() {
                    let g = &gcps[g_idx];
                    let ex = pt.x - g.x;
                    let ey = pt.y - g.y;
                    let ez = pt.z - g.z;
                    dz_sum += ez;
                    xyz_sq += ex * ex + ey * ey + ez * ez;
                    tri_cnt += 1;
                }
            }
        }
    }

    let dy_rmse = if dy_cnt > 0 { (dy_sq / dy_cnt as f64).sqrt() } else { 0.0 };
    let dz_bias = if tri_cnt > 0 { dz_sum / tri_cnt as f64 } else { 0.0 };
    let xyz_rmse = if tri_cnt > 0 { (xyz_sq / (3.0 * tri_cnt as f64)).sqrt() } else { 0.0 };
    (dy_rmse, dz_bias, xyz_rmse)
}

/// Measure exact epipolar `dy` RMSE (px) on a single stereo pair's shared GCP picks.
pub fn pair_gcp_epipolar_dy(
    cal: &CameraCalibration,
    pair: &StereoPairEntry,
    picks: &[GcpImagePick],
) -> Option<(usize, f64, f64)> {
    let lid = normalize_id(&pair.left_pose.id);
    let rid = normalize_id(&pair.right_pose.id);
    let mut map_l = HashMap::new();
    let mut map_r = HashMap::new();
    for p in picks {
        let img = normalize_id(&p.image_id);
        if img == lid {
            map_l.insert(p.gcp_id.trim().to_ascii_lowercase(), (p.col_px, p.row_px));
        } else if img == rid {
            map_r.insert(p.gcp_id.trim().to_ascii_lowercase(), (p.col_px, p.row_px));
        }
    }
    let w = cal.width_px.max(1);
    let h = cal.height_px.max(1);
    let cam_l = build_pinhole_for_pose(cal, &pair.left_pose, w, h, 0, 0);
    let cam_r = build_pinhole_for_pose(cal, &pair.right_pose, w, h, 0, 0);
    let anchor = Vec3::new(
        0.5 * (pair.left_pose.x + pair.right_pose.x),
        0.5 * (pair.left_pose.y + pair.right_pose.y),
        pair.left_pose.z - cal.flight_height_m.max(500.0),
    );
    let (rect_l, rect_r) = build_epipolar_rectified_cameras(&cam_l, &cam_r, anchor, 1024, 1024);

    let mut sq = 0.0;
    let mut max_abs = 0.0f64;
    let mut cnt = 0usize;
    for (gid, &(ul, vl)) in &map_l {
        if let Some(&(ur, vr)) = map_r.get(gid) {
            let dl = cam_l.pixel_to_vector(Vec2::new(ul, vl));
            let dr = cam_r.pixel_to_vector(Vec2::new(ur, vr));
            let rpl = rect_l.point_to_pixel(cam_l.center + dl * 1000.0);
            let rpr = rect_r.point_to_pixel(cam_r.center + dr * 1000.0);
            if rpl.y.is_finite() && rpr.y.is_finite() {
                let dy = (rpr.y - rpl.y).abs();
                sq += dy * dy;
                if dy > max_abs { max_abs = dy; }
                cnt += 1;
            }
        }
    }
    if cnt > 0 {
        Some((cnt, (sq / cnt as f64).sqrt(), max_abs))
    } else {
        None
    }
}

/// Two-stage `f64` Levenberg-Marquardt GCP-pick refinement:
/// 1. Global 7-DOF calibration `[dX, dY, dZ, d_omega, d_phi, d_kappa, d_focal_mm]` across all poses
/// 2. Per-image MAP 6-DOF refinement with Huber robust weighting & GNSS/IMU prior
/// Commits refined poses and focal length into `chantier` ONLY on strict improvement.
pub fn refine_chantier_with_gcp_picks(chantier: &mut Chantier) -> Option<GcpRefinementReport> {
    if chantier.gcp_picks.is_empty() || chantier.gcps.is_empty() || chantier.poses.is_empty() {
        return None;
    }

    let gcp_map: HashMap<String, usize> = chantier
        .gcps
        .iter()
        .enumerate()
        .map(|(i, g)| (g.id.trim().to_ascii_lowercase(), i))
        .collect();
    let pose_map: HashMap<String, usize> = chantier
        .poses
        .iter()
        .enumerate()
        .map(|(i, p)| (normalize_id(&p.id), i))
        .collect();

    let mut obs = Vec::new();
    let mut unique_gcps = std::collections::HashSet::new();
    for p in &chantier.gcp_picks {
        let gk = p.gcp_id.trim().to_ascii_lowercase();
        let pk = normalize_id(&p.image_id);
        if let (Some(&gi), Some(&pi)) = (gcp_map.get(&gk), pose_map.get(&pk)) {
            obs.push(MatchedObs {
                pose_idx: pi,
                gcp_idx: gi,
                u_obs: p.col_px,
                v_obs: p.row_px,
            });
            unique_gcps.insert(gi);
        }
    }
    if obs.len() < 4 {
        return None;
    }

    let w = chantier.calibration.width_px.max(1);
    let h = chantier.calibration.height_px.max(1);

    let initial_reproj = eval_reproj_rmse(&chantier.calibration, &chantier.poses, &chantier.gcps, &obs);
    let (init_dy, init_dz_bias, init_xyz_rmse) = eval_epipolar_and_triangulation(
        &chantier.calibration,
        &chantier.poses,
        &chantier.pairs,
        &chantier.gcps,
        &obs,
    );

    // Filter extreme gross pick outliers (> 60 px initial residual) so they cannot pull global fit
    let inlier_obs: Vec<MatchedObs> = {
        let cams: Vec<PinholeCamera> = chantier
            .poses
            .iter()
            .map(|p| build_pinhole_for_pose(&chantier.calibration, p, w, h, 0, 0))
            .collect();
        obs.iter()
            .filter(|o| {
                let g = &chantier.gcps[o.gcp_idx];
                let pr = cams[o.pose_idx].point_to_pixel(Vec3::new(g.x, g.y, g.z));
                (pr.x - o.u_obs).hypot(pr.y - o.v_obs) < 60.0
            })
            .map(|o| MatchedObs { ..*o })
            .collect()
    };
    if inlier_obs.len() < 4 {
        return None;
    }

    // --- Stage 1: Global 7-DOF Block Calibration [dX, dY, dZ, d_omega_deg, d_phi_deg, d_kappa_deg, d_focal_mm] ---
    let mut theta = [0.0f64; 7];
    let eps_step = [1e-3, 1e-3, 1e-3, 1e-5, 1e-5, 1e-5, 1e-4];
    // Weak physical prior sigmas (in parameter units) keeping normal matrix well-conditioned
    let prior_inv_var = [
        1.0 / (5.0 * 5.0),
        1.0 / (5.0 * 5.0),
        1.0 / (5.0 * 5.0),
        1.0 / (1.0 * 1.0),
        1.0 / (1.0 * 1.0),
        1.0 / (1.0 * 1.0),
        1.0 / (0.25 * 0.25),
    ];
    let sigma_px_inv_var = 1.0 / (0.5 * 0.5);
    let mut lambda = 1e-3f64;

    let apply_global = |base_cal: &CameraCalibration, base_poses: &[ImagePose], th: &[f64; 7]| -> (CameraCalibration, Vec<ImagePose>) {
        let mut c = base_cal.clone();
        c.focal_mm = (base_cal.focal_mm + th[6]).max(1.0);
        let ps = base_poses
            .iter()
            .map(|p| ImagePose {
                id: p.id.clone(),
                x: p.x + th[0],
                y: p.y + th[1],
                z: p.z + th[2],
                omega_deg: p.omega_deg + th[3],
                phi_deg: p.phi_deg + th[4],
                kappa_deg: p.kappa_deg + th[5],
            })
            .collect();
        (c, ps)
    };

    let global_cost = |th: &[f64; 7]| -> f64 {
        let (c, ps) = apply_global(&chantier.calibration, &chantier.poses, th);
        let cams: Vec<PinholeCamera> = ps.iter().map(|p| build_pinhole_for_pose(&c, p, w, h, 0, 0)).collect();
        let mut cost = 0.0;
        for o in &inlier_obs {
            let g = &chantier.gcps[o.gcp_idx];
            let pr = cams[o.pose_idx].point_to_pixel(Vec3::new(g.x, g.y, g.z));
            let ru = pr.x - o.u_obs;
            let rv = pr.y - o.v_obs;
            let r = ru.hypot(rv);
            let hw = huber_weight(r, 3.0);
            cost += hw * sigma_px_inv_var * (ru * ru + rv * rv);
        }
        for k in 0..7 {
            cost += prior_inv_var[k] * th[k] * th[k];
        }
        cost
    };

    let mut cur_cost = global_cost(&theta);
    for _iter in 0..25 {
        let (c0, ps0) = apply_global(&chantier.calibration, &chantier.poses, &theta);
        let cams0: Vec<PinholeCamera> = ps0.iter().map(|p| build_pinhole_for_pose(&c0, p, w, h, 0, 0)).collect();

        // Precompute perturbed camera sets for central-difference Jacobians (only 14 tiny arrays!)
        let mut cams_plus = Vec::with_capacity(7);
        let mut cams_minus = Vec::with_capacity(7);
        for k in 0..7 {
            let mut tp = theta;
            tp[k] += eps_step[k];
            let (cp, psp) = apply_global(&chantier.calibration, &chantier.poses, &tp);
            cams_plus.push(psp.iter().map(|p| build_pinhole_for_pose(&cp, p, w, h, 0, 0)).collect::<Vec<_>>());

            let mut tm = theta;
            tm[k] -= eps_step[k];
            let (cm, psm) = apply_global(&chantier.calibration, &chantier.poses, &tm);
            cams_minus.push(psm.iter().map(|p| build_pinhole_for_pose(&cm, p, w, h, 0, 0)).collect::<Vec<_>>());
        }

        let mut h_mat = [[0.0f64; 7]; 7];
        let mut g_vec = [0.0f64; 7];
        for o in &inlier_obs {
            let g = &chantier.gcps[o.gcp_idx];
            let gp = Vec3::new(g.x, g.y, g.z);
            let pr0 = cams0[o.pose_idx].point_to_pixel(gp);
            let ru = pr0.x - o.u_obs;
            let rv = pr0.y - o.v_obs;
            let wt = huber_weight(ru.hypot(rv), 3.0) * sigma_px_inv_var;

            let mut ju = [0.0f64; 7];
            let mut jv = [0.0f64; 7];
            for k in 0..7 {
                let pp = cams_plus[k][o.pose_idx].point_to_pixel(gp);
                let pm = cams_minus[k][o.pose_idx].point_to_pixel(gp);
                let inv_2e = 0.5 / eps_step[k];
                ju[k] = (pp.x - pm.x) * inv_2e;
                jv[k] = (pp.y - pm.y) * inv_2e;
            }
            for r in 0..7 {
                g_vec[r] += wt * (ju[r] * ru + jv[r] * rv);
                for c in r..7 {
                    h_mat[r][c] += wt * (ju[r] * ju[c] + jv[r] * jv[c]);
                }
            }
        }
        for r in 0..7 {
            for c in 0..r {
                h_mat[r][c] = h_mat[c][r];
            }
            h_mat[r][r] += prior_inv_var[r];
            g_vec[r] += prior_inv_var[r] * theta[r];
        }

        let mut accepted = false;
        for _ in 0..8 {
            let mut h_damp = h_mat;
            let mut rhs = [0.0f64; 7];
            for r in 0..7 {
                h_damp[r][r] += lambda * h_mat[r][r].max(1e-6);
                rhs[r] = -g_vec[r];
            }
            if let Some(delta) = solve_linear_f64(h_damp, rhs) {
                let mut cand = theta;
                let mut step_norm = 0.0f64;
                for k in 0..7 {
                    cand[k] += delta[k];
                    step_norm += delta[k] * delta[k];
                }
                let cand_cost = global_cost(&cand);
                if cand_cost < cur_cost {
                    theta = cand;
                    let gain = (cur_cost - cand_cost) / cur_cost.max(1e-12);
                    cur_cost = cand_cost;
                    lambda = (lambda * 0.35).max(1e-9);
                    accepted = true;
                    if step_norm.sqrt() < 1e-7 || gain < 1e-7 {
                        break;
                    }
                    break;
                }
            }
            lambda = (lambda * 4.0).min(1e9);
        }
        if !accepted {
            break;
        }
    }

    let (stage1_cal, mut refined_poses) = apply_global(&chantier.calibration, &chantier.poses, &theta);
    let stage1_reproj = eval_reproj_rmse(&stage1_cal, &refined_poses, &chantier.gcps, &obs);

    // --- Stage 2: Per-Image MAP 6-DOF Levenberg-Marquardt with GNSS/IMU Prior ---
    let mut obs_per_pose: HashMap<usize, Vec<&MatchedObs>> = HashMap::new();
    for o in &inlier_obs {
        obs_per_pose.entry(o.pose_idx).or_default().push(o);
    }

    let pose_eps = [1e-3, 1e-3, 1e-3, 1e-5, 1e-5, 1e-5];
    // Regularized GNSS/IMU prior around Stage-1 pose: sigma_pos = 0.25 m, sigma_ang = 0.02 deg
    let pose_prior_iv = [
        1.0 / (0.25 * 0.25),
        1.0 / (0.25 * 0.25),
        1.0 / (0.25 * 0.25),
        1.0 / (0.02 * 0.02),
        1.0 / (0.02 * 0.02),
        1.0 / (0.02 * 0.02),
    ];
    let mut refined_images = 0usize;

    for (&p_idx, p_obs) in &obs_per_pose {
        if p_obs.len() < 2 {
            continue;
        }
        let base_pose = refined_poses[p_idx].clone();
        let mut dp = [0.0f64; 6];
        let mut lam = 1e-3f64;

        let make_pose = |d: &[f64; 6]| -> ImagePose {
            ImagePose {
                id: base_pose.id.clone(),
                x: base_pose.x + d[0],
                y: base_pose.y + d[1],
                z: base_pose.z + d[2],
                omega_deg: base_pose.omega_deg + d[3],
                phi_deg: base_pose.phi_deg + d[4],
                kappa_deg: base_pose.kappa_deg + d[5],
            }
        };
        let pose_cost = |d: &[f64; 6]| -> f64 {
            let cam = build_pinhole_for_pose(&stage1_cal, &make_pose(d), w, h, 0, 0);
            let mut c = 0.0;
            for o in p_obs {
                let g = &chantier.gcps[o.gcp_idx];
                let pr = cam.point_to_pixel(Vec3::new(g.x, g.y, g.z));
                let ru = pr.x - o.u_obs;
                let rv = pr.y - o.v_obs;
                let hw = huber_weight(ru.hypot(rv), 2.0);
                c += hw * sigma_px_inv_var * (ru * ru + rv * rv);
            }
            for k in 0..6 {
                c += pose_prior_iv[k] * d[k] * d[k];
            }
            c
        };

        let mut cur_c = pose_cost(&dp);
        for _ in 0..20 {
            let cam0 = build_pinhole_for_pose(&stage1_cal, &make_pose(&dp), w, h, 0, 0);
            let mut h_m = [[0.0f64; 6]; 6];
            let mut g_v = [0.0f64; 6];

            let mut cams_p = Vec::with_capacity(6);
            let mut cams_m = Vec::with_capacity(6);
            for k in 0..6 {
                let mut dpp = dp;
                dpp[k] += pose_eps[k];
                cams_p.push(build_pinhole_for_pose(&stage1_cal, &make_pose(&dpp), w, h, 0, 0));
                let mut dpm = dp;
                dpm[k] -= pose_eps[k];
                cams_m.push(build_pinhole_for_pose(&stage1_cal, &make_pose(&dpm), w, h, 0, 0));
            }

            for o in p_obs {
                let g = &chantier.gcps[o.gcp_idx];
                let gp = Vec3::new(g.x, g.y, g.z);
                let pr0 = cam0.point_to_pixel(gp);
                let ru = pr0.x - o.u_obs;
                let rv = pr0.y - o.v_obs;
                let wt = huber_weight(ru.hypot(rv), 2.0) * sigma_px_inv_var;
                let mut ju = [0.0f64; 6];
                let mut jv = [0.0f64; 6];
                for k in 0..6 {
                    let pp = cams_p[k].point_to_pixel(gp);
                    let pm = cams_m[k].point_to_pixel(gp);
                    let inv_2e = 0.5 / pose_eps[k];
                    ju[k] = (pp.x - pm.x) * inv_2e;
                    jv[k] = (pp.y - pm.y) * inv_2e;
                }
                for r in 0..6 {
                    g_v[r] += wt * (ju[r] * ru + jv[r] * rv);
                    for c in r..6 {
                        h_m[r][c] += wt * (ju[r] * ju[c] + jv[r] * jv[c]);
                    }
                }
            }
            for r in 0..6 {
                for c in 0..r {
                    h_m[r][c] = h_m[c][r];
                }
                h_m[r][r] += pose_prior_iv[r];
                g_v[r] += pose_prior_iv[r] * dp[r];
            }

            let mut acc = false;
            for _ in 0..6 {
                let mut hd = h_m;
                let mut rhs = [0.0f64; 6];
                for r in 0..6 {
                    hd[r][r] += lam * h_m[r][r].max(1e-6);
                    rhs[r] = -g_v[r];
                }
                if let Some(step) = solve_linear_f64(hd, rhs) {
                    let mut cand = dp;
                    for k in 0..6 {
                        cand[k] += step[k];
                    }
                    let cc = pose_cost(&cand);
                    if cc < cur_c {
                        dp = cand;
                        cur_c = cc;
                        lam = (lam * 0.35).max(1e-9);
                        acc = true;
                        break;
                    }
                }
                lam = (lam * 4.0).min(1e9);
            }
            if !acc {
                break;
            }
        }
        refined_poses[p_idx] = make_pose(&dp);
        refined_images += 1;
    }

    let final_reproj = eval_reproj_rmse(&stage1_cal, &refined_poses, &chantier.gcps, &obs);
    let (final_dy, final_dz_bias, final_xyz_rmse) = eval_epipolar_and_triangulation(
        &stage1_cal,
        &refined_poses,
        &chantier.pairs,
        &chantier.gcps,
        &obs,
    );

    // Safety guard: commit ONLY when reprojection RMSE strictly improves
    let applied = final_reproj.is_finite() && final_reproj < initial_reproj;
    if applied {
        chantier.calibration = stage1_cal;
        chantier.poses = refined_poses;
        // Synchronize all stereo pairs with the refined poses
        let updated_map: HashMap<String, ImagePose> = chantier
            .poses
            .iter()
            .map(|p| (normalize_id(&p.id), p.clone()))
            .collect();
        for pair in &mut chantier.pairs {
            if let Some(lp) = updated_map.get(&normalize_id(&pair.left_pose.id)) {
                pair.left_pose = lp.clone();
            }
            if let Some(rp) = updated_map.get(&normalize_id(&pair.right_pose.id)) {
                pair.right_pose = rp.clone();
            }
            pair.baseline_m = (pair.right_pose.x - pair.left_pose.x)
                .hypot(pair.right_pose.y - pair.left_pose.y);
        }
    }

    Some(GcpRefinementReport {
        matched_observations: obs.len(),
        matched_gcps: unique_gcps.len(),
        refined_images,
        applied,
        initial_reproj_rmse_px: initial_reproj,
        stage1_reproj_rmse_px: stage1_reproj,
        final_reproj_rmse_px: final_reproj,
        initial_epi_dy_rmse_px: init_dy,
        final_epi_dy_rmse_px: final_dy,
        initial_tri_dz_bias_m: init_dz_bias,
        final_tri_dz_bias_m: final_dz_bias,
        initial_tri_xyz_rmse_m: init_xyz_rmse,
        final_tri_xyz_rmse_m: final_xyz_rmse,
        global_dx_m: theta[0],
        global_dy_m: theta[1],
        global_dz_m: theta[2],
        global_domega_deg: theta[3],
        global_dphi_deg: theta[4],
        global_dkappa_deg: theta[5],
        global_dfocal_mm: theta[6],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_parse_gcp_picks_and_footprints_json() {
        let picks_json = r#"[
            ["10", "026_0730", 29041.573, 4456.584],
            ["10", "026_0731", 28992.887, 10070.674]
        ]"#;
        let picks = parse_gcp_picks_str(picks_json).unwrap();
        assert_eq!(picks.len(), 2);
        assert_eq!(picks[0].gcp_id, "10");
        assert_eq!(picks[0].image_id, "026_0730");
        assert!((picks[0].col_px - 29041.573).abs() < 1e-6);

        let fp_json = r#"{
            "footprints": [
                {
                    "pose_id": "025_0709",
                    "x": 1901814.123,
                    "y": 3143265.536,
                    "corners_sol": [[0, 100], [100, 100], [100, 0], [0, 0]]
                }
            ]
        }"#;
        let fps = parse_footprints_str(fp_json).unwrap();
        assert_eq!(fps.len(), 1);
        let fp = fps.get("025_0709").unwrap();
        assert!((fp.x - 1901814.123).abs() < 1e-6);
        assert!((polygon_iou(&fp.corners_sol, &fp.corners_sol) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_two_stage_gcp_refinement_removes_scale_z_bias_and_boresight() {
        // Synthetic DMC-IV 2-line opposite-heading block (L0 + L2) with true poses,
        // then corrupt raw GNSS/IMU with +0.20 m Z offset, -0.015 mm focal scale error,
        // and 0.01 deg boresight + per-image jitter, and verify LM recovers sub-pixel & sub-cm accuracy!
        let true_cal = CameraCalibration::parse_str(
            "CAMERA;DMCIV;143.0016;0.0;0.0;1742.0;0;0.02;31520;13760;0.003760;0\n",
        )
        .unwrap();
        let true_poses = vec![
            ImagePose { id: "025_0727".into(), x: 1_904_800.0, y: 3_146_730.0, z: 2024.5, omega_deg: 0.01, phi_deg: -0.01, kappa_deg: -40.4 },
            ImagePose { id: "025_0728".into(), x: 1_904_968.0, y: 3_146_928.0, z: 2024.2, omega_deg: -0.01, phi_deg: 0.01, kappa_deg: -40.4 },
            ImagePose { id: "026_0730".into(), x: 1_904_970.0, y: 3_146_935.0, z: 1980.0, omega_deg: 0.00, phi_deg: 0.01, kappa_deg: 139.6 },
            ImagePose { id: "026_0731".into(), x: 1_904_802.0, y: 3_146_737.0, z: 1982.0, omega_deg: -0.01, phi_deg: 0.02, kappa_deg: 139.6 },
        ];
        // Generate 12 well-distributed GCPs across the overlap
        let mut gcps = Vec::new();
        for iy in 0..3 {
            for ix in 0..4 {
                let id = format!("G{}", iy * 4 + ix + 1);
                let gx = 1_904_650.0 + ix as f64 * 150.0;
                let gy = 3_146_650.0 + iy as f64 * 160.0;
                let gz = 210.0 + (ix as f64) * 12.0 - (iy as f64) * 8.0;
                gcps.push(GroundControlPoint { id, x: gx, y: gy, z: gz });
            }
        }
        // Generate exact noise-free subpixel picks from true cameras
        let mut picks = Vec::new();
        for p in &true_poses {
            let cam = build_pinhole_for_pose(&true_cal, p, 31520, 13760, 0, 0);
            for g in &gcps {
                let px = cam.point_to_pixel(Vec3::new(g.x, g.y, g.z));
                if px.x > 100.0 && px.x < 31420.0 && px.y > 100.0 && px.y < 13660.0 {
                    picks.push(GcpImagePick {
                        gcp_id: g.id.clone(),
                        image_id: p.id.clone(),
                        col_px: px.x,
                        row_px: px.y,
                    });
                }
            }
        }
        assert!(picks.len() >= 24);

        // Perturb initial GNSS/IMU poses (simulating raw trajectory with +0.20 m vertical bias, scale, boresight, jitter)
        let mut raw_cal = true_cal.clone();
        raw_cal.focal_mm -= 0.015; // ~105 ppm scale error
        let raw_poses: Vec<ImagePose> = true_poses
            .iter()
            .enumerate()
            .map(|(i, p)| ImagePose {
                id: p.id.clone(),
                x: p.x + 0.08 + (i as f64 * 0.02 - 0.03),
                y: p.y - 0.06,
                z: p.z + 0.20 + if i % 2 == 0 { 0.03 } else { -0.02 },
                omega_deg: p.omega_deg + 0.008,
                phi_deg: p.phi_deg - 0.006,
                kappa_deg: p.kappa_deg + 0.010,
            })
            .collect();

        let pairs = vec![
            StereoPairEntry {
                index: 0,
                flight_line: 0,
                name: "L0_025_0727_025_0728".into(),
                left_pose: raw_poses[0].clone(),
                right_pose: raw_poses[1].clone(),
                left_image_path: PathBuf::new(),
                right_image_path: PathBuf::new(),
                baseline_m: 259.0,
            },
            StereoPairEntry {
                index: 1,
                flight_line: 1,
                name: "L1_026_0730_026_0731".into(),
                left_pose: raw_poses[2].clone(),
                right_pose: raw_poses[3].clone(),
                left_image_path: PathBuf::new(),
                right_image_path: PathBuf::new(),
                baseline_m: 259.0,
            },
        ];

        let mut chantier = Chantier {
            root_dir: PathBuf::new(),
            input_dir: PathBuf::new(),
            camera_file: PathBuf::new(),
            trajectory_file: PathBuf::new(),
            gcp_file: None,
            gcp_picks_file: None,
            footprints_file: None,
            survey_crs: CrsId::CcZone(44),
            lidar_mnt: None,
            calibration: raw_cal,
            poses: raw_poses,
            gcps,
            gcp_picks: picks,
            footprints: HashMap::new(),
            footprint_report: None,
            ba_report: None,
            pairs,
        };

        let rep = refine_chantier_with_gcp_picks(&mut chantier).unwrap();
        assert!(rep.applied);
        assert!(rep.initial_reproj_rmse_px > 2.0, "expected corrupted initial reproj, got {}", rep.initial_reproj_rmse_px);
        assert!(rep.final_reproj_rmse_px < 0.15, "final reproj RMSE too high: {}", rep.final_reproj_rmse_px);
        assert!(rep.final_epi_dy_rmse_px < 0.15, "final epipolar dy too high: {}", rep.final_epi_dy_rmse_px);
        assert!(rep.final_tri_dz_bias_m.abs() < 0.02, "final dZ bias too high: {}", rep.final_tri_dz_bias_m);
        assert!(rep.final_tri_xyz_rmse_m < 0.02, "final 3D XYZ RMSE too high: {}", rep.final_tri_xyz_rmse_m);
    }

    #[test]
    fn test_footprint_validation_with_sloped_lidar_mnt_cross_crs() {
        use crate::core::crs::transform_point;

        // Build a sloped Lambert-93 LiDAR MNT covering Meyreuil (~150 m relief across footprint)
        let w = 250usize;
        let h = 250usize;
        let cs = 20.0;
        let min_x = 895_000.0;
        let min_y = 6_264_000.0;
        let mut zg = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let px = min_x + x as f64 * cs;
                let py = min_y + y as f64 * cs;
                zg[y * w + x] = (180.0 + 0.06 * (px - min_x) + 0.04 * (py - min_y)) as f32;
            }
        }
        let mnt = LidarTerrainModel {
            source_dir: PathBuf::from("IGN_DATA"),
            tile_count: 4,
            lidar_crs: CrsId::Lambert93,
            min_x,
            min_y,
            cell_size: cs,
            width: w,
            height: h,
            z_ground: zg.clone(),
            z_surface: zg,
            z_min: 180.0,
            z_max: 680.0,
            from_cache: false,
        };

        // Place DMC-IV pose in CC44 (EPSG:3944) directly over center of the Lambert-93 MNT
        let center_cc44 = transform_point(CrsId::Lambert93, CrsId::CcZone(44), Vec3::new(897_500.0, 6_266_500.0, 0.0)).unwrap();
        let cal = CameraCalibration::parse_str(
            "CAMERA;DMCIV;143.0016;0.0;0.0;1742.0;0;0.02;31520;13760;0.003760;0\n",
        )
        .unwrap();
        let pose = ImagePose {
            id: "025_0727".into(),
            x: center_cc44.x,
            y: center_cc44.y,
            z: 2024.0,
            omega_deg: 0.01,
            phi_deg: -0.01,
            kappa_deg: -40.4,
        };
        let cam = build_pinhole_for_pose(&cal, &pose, 31520, 13760, 0, 0);
        let corners_px = [(0.0, 0.0), (31520.0, 0.0), (31520.0, 13760.0), (0.0, 13760.0)];
        let mut exact_corners = [[0.0f64; 2]; 4];
        for (i, &(u, v)) in corners_px.iter().enumerate() {
            let ray = cam.pixel_to_vector(Vec2::new(u, v));
            let hit = mnt.intersect_ray_ground(cam.center, ray, 300.0, CrsId::CcZone(44)).unwrap();
            exact_corners[i] = [hit.x, hit.y];
        }
        let mut fps = HashMap::new();
        fps.insert(
            "025_0727".into(),
            ValidatedFootprint {
                pose_id: "025_0727".into(),
                x: pose.x,
                y: pose.y,
                corners_sol: exact_corners,
            },
        );

        let rep_mnt = validate_chantier_footprints(&cal, &[pose.clone()], &[], &fps, Some(&mnt), CrsId::CcZone(44));
        assert!(rep_mnt.used_lidar_mnt);
        assert_eq!(rep_mnt.orientation_ok, 1);
        assert!(rep_mnt.mean_iou_pct > 99.9, "Expected >99.9% IoU with LiDAR MNT, got {:.3}%", rep_mnt.mean_iou_pct);

        let rep_flat = validate_chantier_footprints(&cal, &[pose], &[], &fps, None, CrsId::CcZone(44));
        assert!(!rep_flat.used_lidar_mnt);
        assert!(rep_mnt.mean_iou_pct > rep_flat.mean_iou_pct, "LiDAR MNT IoU should beat flat-plane IoU");
    }
}
