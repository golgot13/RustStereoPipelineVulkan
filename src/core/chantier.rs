//! Aerial Photogrammetry Chantier Ingestion Engine (`.CAM`, Trajectory `.TXT`, `GCP.txt`, GeoTIFF pairs).
//! Compatible with real aerial survey blocks like Meyreuil (DMC-IV 433 Mpx, Lambert-93 / CC44)
//! using streaming Full-Scale 1:1 GeoTIFF ROI extraction and exact double-precision (`f64`) geometry.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use crate::camera::{AnyCamera, CameraModel, PinholeCamera};
use crate::core::error::{PipelineError, Result};
use crate::core::json::JsonValue;
use crate::image::geotiff::GeoTiffReader;
use crate::image::{DemGrid, ImageF32};
use crate::math::{Mat3, Vec2, Vec3};

/// Aerial metric camera calibration parsed from `.CAM` (DMC-IV / UltraCam / IGN format).
#[derive(Debug, Clone)]
pub struct CameraCalibration {
    pub name: String,
    pub focal_mm: f64,
    pub pp_x_mm: f64,
    pub pp_y_mm: f64,
    pub flight_height_m: f64,
    pub scale: f64,
    pub width_px: u32,
    pub height_px: u32,
    pub pixel_size_mm: f64,
    pub distortion: [f64; 4], // [k1, k2, p1, p2]
}

impl CameraCalibration {
    pub fn focal_px(&self) -> f64 {
        self.focal_mm / self.pixel_size_mm.max(1e-9)
    }

    pub fn parse_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| PipelineError::Io(format!("Cannot read {}: {}", path.display(), e)))?;
        Self::parse_str(&content)
    }

    pub fn parse_str(content: &str) -> Result<Self> {
        let mut cal = Self {
            name: String::new(),
            focal_mm: 0.0,
            pp_x_mm: 0.0,
            pp_y_mm: 0.0,
            flight_height_m: 0.0,
            scale: 0.0,
            width_px: 0,
            height_px: 0,
            pixel_size_mm: 0.0,
            distortion: [0.0; 4],
        };
        let mut found = false;

        for raw in content.lines() {
            let line = raw.trim();
            if line.starts_with("CAMERA;") {
                let parts: Vec<&str> = line.split(';').map(|s| s.trim()).collect();
                if parts.len() < 11 {
                    return Err(PipelineError::Format(format!(
                        "CAMERA record expects >= 11 fields, got {}",
                        parts.len()
                    )));
                }
                let p_f64 = |i: usize| -> f64 {
                    parts[i].replace(',', ".").parse::<f64>().unwrap_or(0.0)
                };
                cal.name = parts[1].to_string();
                cal.focal_mm = p_f64(2);
                cal.pp_x_mm = p_f64(3);
                cal.pp_y_mm = p_f64(4);
                cal.flight_height_m = p_f64(5);
                cal.scale = p_f64(7);
                cal.width_px = parts[8].parse::<u32>().unwrap_or(0);
                cal.height_px = parts[9].parse::<u32>().unwrap_or(0);
                let raw_px = p_f64(10);
                // AEROPHOTOGRAM heuristic: values > 0.1 are in microns -> convert to mm
                cal.pixel_size_mm = if raw_px > 0.1 { raw_px / 1000.0 } else { raw_px };
                found = true;
            } else if line.starts_with("CAMDIST;") {
                let parts: Vec<&str> = line.split(';').map(|s| s.trim()).collect();
                if parts.len() >= 6 {
                    let p_f64 = |i: usize| -> f64 {
                        parts[i].replace(',', ".").parse::<f64>().unwrap_or(0.0)
                    };
                    cal.distortion = [p_f64(1), p_f64(2), p_f64(4), p_f64(5)];
                }
            }
        }

        if !found || cal.focal_mm <= 0.0 || cal.pixel_size_mm <= 0.0 {
            return Err(PipelineError::Format(
                "Invalid .CAM file: missing valid CAMERA; record".into(),
            ));
        }
        Ok(cal)
    }
}

/// Aerial exterior orientation pose `(X, Y, Z, omega, phi, kappa)` in meters & degrees.
#[derive(Debug, Clone)]
pub struct ImagePose {
    pub id: String,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub omega_deg: f64,
    pub phi_deg: f64,
    pub kappa_deg: f64,
}

/// Ground Control Point `(X, Y, Z)` in projected ground coordinates.
#[derive(Debug, Clone)]
pub struct GroundControlPoint {
    pub id: String,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Parse aerial trajectory files (CSV header, IGN `!`, or PMS3D classic whitespace/delimited).
pub fn parse_trajectory_file(path: &Path) -> Result<Vec<ImagePose>> {
    let bytes = std::fs::read(path)
        .map_err(|e| PipelineError::Io(format!("Cannot read {}: {}", path.display(), e)))?;
    let content = String::from_utf8_lossy(&bytes);
    parse_trajectory_str(&content)
}

pub fn parse_trajectory_str(content: &str) -> Result<Vec<ImagePose>> {
    let delimiter = detect_delimiter(content);
    let mut poses = Vec::new();
    let mut header_map: Option<HashMap<String, usize>> = None;
    let mut header_declares_gons = false;

    for raw in content.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('#') || line.starts_with("//") || line.starts_with('[') {
            if line_declares_gons(line) {
                header_declares_gons = true;
            }
            continue;
        }
        if line.starts_with("CAMDIST;") {
            continue;
        }
        let low = line.to_ascii_lowercase();
        if low.contains("statistiques") || low.contains("constante") {
            break;
        }

        let parts: Vec<&str> = if delimiter == ' ' {
            line.split_whitespace().collect()
        } else {
            line.split(delimiter).map(|s| s.trim()).filter(|s| !s.is_empty()).collect()
        };
        if parts.len() < 7 {
            continue;
        }

        // Check for CSV header row on first data line
        if header_map.is_none() && poses.is_empty() {
            let has_header = parts.iter().any(|&p| {
                matches!(
                    p.to_ascii_lowercase().as_str(),
                    "filename" | "file" | "image" | "name" | "omega" | "phi" | "kappa" | "x" | "y" | "z"
                )
            });
            if has_header {
                if line_declares_gons(line) {
                    header_declares_gons = true;
                }
                let mut m = HashMap::new();
                for (i, &p) in parts.iter().enumerate() {
                    m.insert(p.to_ascii_lowercase(), i);
                }
                header_map = Some(m);
                continue;
            }
        }

        let (id, x, y, z, mut omega, mut phi, mut kappa) = if let Some(ref hm) = header_map {
            let col = |keys: &[&str]| -> Option<usize> {
                keys.iter().find_map(|k| hm.get(*k).copied())
            };
            let ci = col(&["filename", "file", "image", "image name", "name", "id"]).unwrap_or(0);
            let cx = col(&["x", "easting", "longitude", "lon"]).unwrap_or(1);
            let cy = col(&["y", "northing", "latitude", "lat"]).unwrap_or(2);
            let cz = col(&["z", "altitude", "alt", "height"]).unwrap_or(3);
            let co = col(&["omega", "roll"]).unwrap_or(4);
            let cp = col(&["phi", "pitch"]).unwrap_or(5);
            let ck = col(&["kappa", "yaw", "heading"]).unwrap_or(6);
            if [ci, cx, cy, cz, co, cp, ck].iter().any(|&idx| idx >= parts.len()) {
                continue;
            }
            let pf = |idx: usize| -> Option<f64> { parts[idx].replace(',', ".").parse::<f64>().ok() };
            match (pf(cx), pf(cy), pf(cz), pf(co), pf(cp), pf(ck)) {
                (Some(x), Some(y), Some(z), Some(o), Some(p), Some(k)) => (parts[ci].to_string(), x, y, z, o, p, k),
                _ => continue,
            }
        } else if delimiter == '!' && parts.len() >= 8 {
            // IGN Aviation format: Name ! Focal ! X ! Y ! Z ! Omega ! Phi ! Kappa
            let pf = |idx: usize| -> Option<f64> { parts[idx].replace(',', ".").parse::<f64>().ok() };
            match (pf(2), pf(3), pf(4), pf(5), pf(6), pf(7)) {
                (Some(x), Some(y), Some(z), Some(o), Some(p), Some(k)) => (parts[0].to_string(), x, y, z, o, p, k),
                _ => continue,
            }
        } else {
            // PMS3D Classic: Name X Y Z Omega Phi Kappa (or Name Extra X Y Z Omega Phi Kappa)
            let pf = |idx: usize| -> Option<f64> { parts[idx].replace(',', ".").parse::<f64>().ok() };
            if parts.len() >= 8 {
                if let (Some(p1), Some(p2), Some(p3), Some(p4), Some(p5), Some(p6), Some(p7)) =
                    (pf(1), pf(2), pf(3), pf(4), pf(5), pf(6), pf(7))
                {
                    if (p2.abs() > 1000.0 || p3.abs() > 1000.0) && (p1.abs() < 1000.0 || p1 > 100_000.0) {
                        (parts[0].to_string(), p2, p3, p4, p5, p6, p7)
                    } else {
                        (parts[0].to_string(), p1, p2, p3, p4, p5, p6)
                    }
                } else {
                    continue;
                }
            } else if let (Some(x), Some(y), Some(z), Some(o), Some(p), Some(k)) =
                (pf(1), pf(2), pf(3), pf(4), pf(5), pf(6))
            {
                (parts[0].to_string(), x, y, z, o, p, k)
            } else {
                continue;
            }
        };

        // Normalize micro-degrees -> degrees
        if omega.abs() > 10_000.0 { omega /= 1_000_000.0; }
        if phi.abs() > 10_000.0 { phi /= 1_000_000.0; }
        if kappa.abs() > 10_000.0 { kappa /= 1_000_000.0; }

        poses.push(ImagePose {
            id,
            x,
            y,
            z,
            omega_deg: omega,
            phi_deg: phi,
            kappa_deg: kappa,
        });
    }

    if poses.is_empty() {
        return Err(PipelineError::Format("No valid trajectory poses found".into()));
    }

    // File-level gons -> degrees conversion ONLY if explicitly declared in header/comment
    // or if any angle lies in (360, 400] (impossible in degrees). Never corrupt degree
    // trajectories with |kappa| in (90, 360] (such as reverse flight lines at kappa ~ +/-180 deg).
    let file_uses_gons = header_declares_gons
        || poses.iter().any(|p| {
            is_strictly_gon_range(p.omega_deg)
                || is_strictly_gon_range(p.phi_deg)
                || is_strictly_gon_range(p.kappa_deg)
        });
    if file_uses_gons {
        for p in &mut poses {
            p.omega_deg *= 0.9;
            p.phi_deg *= 0.9;
            p.kappa_deg *= 0.9;
        }
    }

    Ok(poses)
}

#[inline]
fn is_strictly_gon_range(angle: f64) -> bool {
    let a = angle.abs();
    a > 360.0 && a <= 400.0
}

fn line_declares_gons(line: &str) -> bool {
    line.split(|c: char| !c.is_ascii_alphanumeric())
        .any(|tok| matches!(tok.to_ascii_lowercase().as_str(), "gon" | "gons" | "grad" | "grads" | "grade" | "grades"))
}

fn detect_delimiter(content: &str) -> char {
    for raw in content.lines() {
        let l = raw.trim();
        if l.is_empty() || l.starts_with('#') || l.starts_with("//") {
            continue;
        }
        if l.matches('!').count() >= 6 { return '!'; }
        if l.matches(';').count() >= 6 { return ';'; }
        if l.matches(',').count() >= 6 { return ','; }
        if l.matches('\t').count() >= 6 { return '\t'; }
        return ' ';
    }
    ' '
}

/// Parse Ground Control Points (`ID X Y Z ...` or `ID Name X Y Z ...`).
pub fn parse_gcp_file(path: &Path) -> Result<Vec<GroundControlPoint>> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| PipelineError::Io(format!("Cannot read {}: {}", path.display(), e)))?;
    let mut gcps = Vec::new();
    for raw in content.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
            continue;
        }
        let parts: Vec<&str> = line
            .split([' ', ',', ';', '\t'])
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        if parts.len() >= 5 && parts[1].replace(',', ".").parse::<f64>().is_err() {
            if let (Ok(x), Ok(y), Ok(z)) = (
                parts[2].replace(',', ".").parse::<f64>(),
                parts[3].replace(',', ".").parse::<f64>(),
                parts[4].replace(',', ".").parse::<f64>(),
            ) {
                gcps.push(GroundControlPoint { id: parts[0].to_string(), x, y, z });
                continue;
            }
        }
        if parts.len() >= 4 {
            if let (Ok(x), Ok(y), Ok(z)) = (
                parts[1].replace(',', ".").parse::<f64>(),
                parts[2].replace(',', ".").parse::<f64>(),
                parts[3].replace(',', ".").parse::<f64>(),
            ) {
                gcps.push(GroundControlPoint { id: parts[0].to_string(), x, y, z });
            }
        }
    }
    Ok(gcps)
}

#[derive(Debug, Clone)]
pub struct StereoPairEntry {
    pub index: usize,
    pub flight_line: usize,
    pub name: String,
    pub left_pose: ImagePose,
    pub right_pose: ImagePose,
    pub left_image_path: PathBuf,
    pub right_image_path: PathBuf,
    pub baseline_m: f64,
}

#[derive(Debug, Clone)]
pub struct Chantier {
    pub root_dir: PathBuf,
    pub input_dir: PathBuf,
    pub camera_file: PathBuf,
    pub trajectory_file: PathBuf,
    pub gcp_file: Option<PathBuf>,
    pub calibration: CameraCalibration,
    pub poses: Vec<ImagePose>,
    pub gcps: Vec<GroundControlPoint>,
    pub pairs: Vec<StereoPairEntry>,
}

impl Chantier {
    /// Auto-discover `.CAM`, trajectory `.TXT`, optional `GCP*.txt`, and GeoTIFF stereo pairs
    /// inside `<dir>` (checking `<dir>/input` automatically if present).
    pub fn discover(dir: &Path) -> Result<Self> {
        let input_dir = if dir.join("input").is_dir() {
            dir.join("input")
        } else if dir.join("INPUT").is_dir() {
            dir.join("INPUT")
        } else {
            dir.to_path_buf()
        };

        let mut all_files = Vec::new();
        collect_files_recursive(&input_dir, 3, &mut all_files);
        if input_dir != dir {
            collect_files_recursive(dir, 1, &mut all_files);
        }

        let mut cam_file: Option<PathBuf> = None;
        let mut txt_candidates: Vec<PathBuf> = Vec::new();
        let mut gcp_file: Option<PathBuf> = None;
        let mut tif_map: HashMap<String, PathBuf> = HashMap::new();

        for path in all_files {
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            let fname_low = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            match ext.as_str() {
                "cam" => {
                    if cam_file.is_none() {
                        cam_file = Some(path);
                    }
                }
                "tif" | "tiff" => {
                    let stem = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("")
                        .to_ascii_lowercase();
                    tif_map.insert(stem, path.clone());
                    tif_map.insert(fname_low, path);
                }
                "txt" | "csv" | "eo" | "ori" => {
                    if fname_low.contains("gcp") || fname_low.contains("appui") || fname_low.contains("calage") {
                        if gcp_file.is_none() {
                            gcp_file = Some(path);
                        }
                    } else {
                        txt_candidates.push(path);
                    }
                }
                _ => {}
            }
        }

        let camera_file = cam_file.ok_or_else(|| {
            PipelineError::Config(format!("No .CAM calibration file found in {}", input_dir.display()))
        })?;
        let calibration = CameraCalibration::parse_file(&camera_file)?;

        // Pick the trajectory file that parses the most valid poses
        let mut best_traj: Option<(PathBuf, Vec<ImagePose>)> = None;
        for cand in txt_candidates {
            if let Ok(poses) = parse_trajectory_file(&cand) {
                if best_traj.as_ref().map_or(true, |(_, bp)| poses.len() > bp.len()) {
                    best_traj = Some((cand, poses));
                }
            }
        }
        let (trajectory_file, poses) = best_traj.ok_or_else(|| {
            PipelineError::Config(format!("No valid trajectory .TXT/.CSV found in {}", input_dir.display()))
        })?;

        let gcps = gcp_file
            .as_ref()
            .and_then(|p| parse_gcp_file(p).ok())
            .unwrap_or_default();

        // Match poses to GeoTIFF files on disk
        let matched: Vec<(ImagePose, PathBuf)> = poses
            .iter()
            .filter_map(|pose| {
                find_image_for_pose(&pose.id, &tif_map).map(|p| (pose.clone(), p))
            })
            .collect();

        if matched.len() < 2 {
            return Err(PipelineError::Config(format!(
                "Found {} trajectory poses and {} GeoTIFFs, but fewer than 2 matched by ID in {}",
                poses.len(),
                tif_map.len() / 2,
                input_dir.display()
            )));
        }

        // Segment into flight lines by heading continuity and build consecutive stereo pairs
        let mut pairs = Vec::new();
        let mut line_id = 0usize;
        let mut prev_heading: Option<f64> = None;

        for w in matched.windows(2) {
            let (p0, path0) = &w[0];
            let (p1, path1) = &w[1];
            let dx = p1.x - p0.x;
            let dy = p1.y - p0.y;
            let baseline = dx.hypot(dy);
            if baseline < 1.0 {
                continue;
            }
            let heading = dy.atan2(dx).to_degrees();
            let same_line = prev_heading.map_or(true, |ph| angle_diff_deg(heading, ph) < 40.0);
            if !same_line {
                line_id += 1;
                prev_heading = Some(heading);
                continue; // Skip turn-around jump between two flight lines
            }
            prev_heading = Some(heading);
            let idx = pairs.len();
            pairs.push(StereoPairEntry {
                index: idx,
                flight_line: line_id,
                name: format!("L{}_{}_{}", line_id, sanitize_id(&p0.id), sanitize_id(&p1.id)),
                left_pose: p0.clone(),
                right_pose: p1.clone(),
                left_image_path: path0.clone(),
                right_image_path: path1.clone(),
                baseline_m: baseline,
            });
        }

        Ok(Self {
            root_dir: dir.to_path_buf(),
            input_dir,
            camera_file,
            trajectory_file,
            gcp_file,
            calibration,
            poses,
            gcps,
            pairs,
        })
    }

    pub fn estimated_ground_z(&self, pair: &StereoPairEntry) -> f64 {
        if !self.gcps.is_empty() {
            let mx = 0.5 * (pair.left_pose.x + pair.right_pose.x);
            let my = 0.5 * (pair.left_pose.y + pair.right_pose.y);
            let mut by_dist: Vec<(f64, f64)> = self
                .gcps
                .iter()
                .map(|g| ((g.x - mx).hypot(g.y - my), g.z))
                .collect();
            by_dist.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let k = by_dist.len().min(5);
            let mut sum_w = 0.0;
            let mut sum_wz = 0.0;
            for &(d, z) in &by_dist[..k] {
                let w = 1.0 / (d * d + 25.0);
                sum_w += w;
                sum_wz += w * z;
            }
            if sum_w > 0.0 {
                return sum_wz / sum_w;
            }
        }
        let cam_z = 0.5 * (pair.left_pose.z + pair.right_pose.z);
        if self.calibration.flight_height_m > 10.0 {
            cam_z - self.calibration.flight_height_m
        } else {
            cam_z - 1000.0
        }
    }
}

/// Build an exact `f64` ASP-compatible `PinholeCamera` from `.CAM` + pose `(X, Y, Z, omega, phi, kappa)`
/// with optional 1:1 ROI origin offset `(roi_x, roi_y)`.
pub fn build_pinhole_for_pose(
    cal: &CameraCalibration,
    pose: &ImagePose,
    img_w: u32,
    img_h: u32,
    roi_x: u32,
    roi_y: u32,
) -> PinholeCamera {
    let f_px = cal.focal_px();
    let w = if cal.width_px > 0 { cal.width_px } else { img_w } as f64;
    let h = if cal.height_px > 0 { cal.height_px } else { img_h } as f64;
    let cu = w * 0.5 + (cal.pp_x_mm / cal.pixel_size_mm) - roi_x as f64;
    let cv = h * 0.5 - (cal.pp_y_mm / cal.pixel_size_mm) - roi_y as f64;

    // Validated PMS3D / IGN / AEROPHOTOGRAM YXZ photogrammetric rotation:
    // R_c2w_photo = Ry(phi) * Rx(omega) * Rz(kappa), converted to image sensor frame (+u right, +v down, +w nadir)
    let r_opk = Mat3::from_opk_yxz(
        pose.omega_deg.to_radians(),
        pose.phi_deg.to_radians(),
        pose.kappa_deg.to_radians(),
    );
    let img_to_photo = Mat3::new([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ]);
    let rotation = r_opk.mul_mat(&img_to_photo);

    let mut cam = PinholeCamera::new(
        f_px,
        f_px,
        cu,
        cv,
        Vec3::new(pose.x, pose.y, pose.z),
        rotation,
    );
    // Convert .CAM metric distortion (mm^-2, mm^-4, mm^-1) to PinholeCamera normalized coordinates (r_norm = r_mm / f_mm)
    let f_mm = cal.focal_mm;
    let f2 = f_mm * f_mm;
    let [k1_mm, k2_mm, p1_mm, p2_mm] = cal.distortion;
    cam.distortion = [
        k1_mm * f2,
        k2_mm * f2 * f2,
        -p2_mm * f_mm,
        p1_mm * f_mm,
    ];
    cam
}

/// Build exact Fusiello epipolar-rectified `PinholeCamera` pair in `f64` where:
/// - `+u` aligns strictly along the 3D baseline `C_R - C_L` (so `v_R == v_L` everywhere, `dy == 0`)
/// - `ground_anchor` projects to `(out_w/2, out_h/2)` in both rectified images (zero disparity at `ground_anchor.z`)
/// - Lens distortion is removed during rectification, preserving exact 3D world triangulation.
pub fn build_epipolar_rectified_cameras(
    full_cam_l: &PinholeCamera,
    full_cam_r: &PinholeCamera,
    ground_anchor: Vec3,
    out_w: u32,
    out_h: u32,
) -> (PinholeCamera, PinholeCamera) {
    let e1 = (full_cam_r.center - full_cam_l.center).normalized();
    let z_l = Vec3::new(
        full_cam_l.rotation.m[0][2],
        full_cam_l.rotation.m[1][2],
        full_cam_l.rotation.m[2][2],
    );
    let z_r = Vec3::new(
        full_cam_r.rotation.m[0][2],
        full_cam_r.rotation.m[1][2],
        full_cam_r.rotation.m[2][2],
    );
    let z_avg = (z_l + z_r).normalized();
    let e2 = z_avg.cross(e1).normalized();
    let e3 = e1.cross(e2).normalized();

    // Camera-to-world rotation matrix whose columns are [e1 (+u), e2 (+v), e3 (+w nadir)]
    let r_rect = Mat3::new([
        [e1.x, e2.x, e3.x],
        [e1.y, e2.y, e3.y],
        [e1.z, e2.z, e3.z],
    ]);
    let r_rect_t = r_rect.transpose();
    let f_px = 0.5 * (full_cam_l.fu + full_cam_r.fu);

    let pl_c = r_rect_t.mul_vec(ground_anchor - full_cam_l.center);
    let pr_c = r_rect_t.mul_vec(ground_anchor - full_cam_r.center);

    let cx = out_w as f64 * 0.5;
    let cy = out_h as f64 * 0.5;
    let cu_l = cx - f_px * (pl_c.x / pl_c.z.max(1e-6));
    let cv_l = cy - f_px * (pl_c.y / pl_c.z.max(1e-6));
    let cu_r = cx - f_px * (pr_c.x / pr_c.z.max(1e-6));
    let cv_r = cy - f_px * (pr_c.y / pr_c.z.max(1e-6));

    let rect_l = PinholeCamera::new(f_px, f_px, cu_l, cv_l, full_cam_l.center, r_rect);
    let rect_r = PinholeCamera::new(f_px, f_px, cu_r, cv_r, full_cam_r.center, r_rect);
    (rect_l, rect_r)
}

/// Stream only the minimal source bounding box needed from a 433 Mpx GeoTIFF and bilinearly warp
/// into the exact epipolar-rectified `out_w x out_h` grid (optionally preserving true RGB color).
fn stream_and_rectify_roi(
    reader: &GeoTiffReader,
    full_cam: &PinholeCamera,
    rect_cam: &PinholeCamera,
    out_w: u32,
    out_h: u32,
    keep_rgb: bool,
) -> Result<(ImageF32, Option<crate::image::ImageRgb>, (u32, u32, u32, u32))> {
    let fw = reader.meta.width as f64;
    let fh = reader.meta.height as f64;

    let mut min_sx = f64::INFINITY;
    let mut max_sx = f64::NEG_INFINITY;
    let mut min_sy = f64::INFINITY;
    let mut max_sy = f64::NEG_INFINITY;

    let samples = 9usize;
    for j in 0..samples {
        let v = (j as f64 / (samples - 1) as f64) * (out_h.saturating_sub(1)) as f64;
        for i in 0..samples {
            let u = (i as f64 / (samples - 1) as f64) * (out_w.saturating_sub(1)) as f64;
            let ray = rect_cam.pixel_to_vector(Vec2::new(u, v));
            let src = full_cam.point_to_pixel(rect_cam.center + ray * 1000.0);
            if src.x.is_finite() && src.y.is_finite() {
                if src.x < min_sx { min_sx = src.x; }
                if src.x > max_sx { max_sx = src.x; }
                if src.y < min_sy { min_sy = src.y; }
                if src.y > max_sy { max_sy = src.y; }
            }
        }
    }

    if !min_sx.is_finite() || !min_sy.is_finite() {
        return Err(PipelineError::Geometry(
            "Rectified ROI projects outside the source GeoTIFF sensor footprint".into(),
        ));
    }

    let x0 = (min_sx.floor() as i64 - 4).clamp(0, (reader.meta.width.saturating_sub(1)) as i64) as u32;
    let y0 = (min_sy.floor() as i64 - 4).clamp(0, (reader.meta.height.saturating_sub(1)) as i64) as u32;
    let x1 = (max_sx.ceil() as i64 + 5).clamp((x0 + 1) as i64, fw as i64) as u32;
    let y1 = (max_sy.ceil() as i64 + 5).clamp((y0 + 1) as i64, fh as i64) as u32;
    let sw = (x1 - x0).max(1);
    let sh = (y1 - y0).max(1);

    let (patch, patch_rgb) = if keep_rgb {
        reader.read_roi_with_rgb(x0, y0, sw, sh)?
    } else {
        (reader.read_roi(x0, y0, sw, sh)?, None)
    };
    let mut out = ImageF32::new(out_w, out_h);
    let mut out_rgb = patch_rgb.as_ref().map(|_| crate::image::ImageRgb::new(out_w, out_h));
    let w_usize = out_w as usize;
    let h_usize = out_h as usize;
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, h_usize.max(1));
    let rows_per = (h_usize + n_threads - 1) / n_threads;

    let x0_f = x0 as f64;
    let y0_f = y0 as f64;
    let pw = patch.width as i32;
    let ph = patch.height as i32;
    let pdata = &patch.data;
    let prgb_ref = patch_rgb.as_ref();

    std::thread::scope(|scope| {
        if let Some(ref mut dst_rgb) = out_rgb {
            for (t, (chunk, rgb_chunk)) in out
                .data
                .chunks_mut(rows_per * w_usize)
                .zip(dst_rgb.data.chunks_mut(rows_per * w_usize))
                .enumerate()
            {
                let y_start = t * rows_per;
                scope.spawn(move || {
                    let rows = chunk.len() / w_usize;
                    for ly in 0..rows {
                        let v = (y_start + ly) as f64;
                        let row_off = ly * w_usize;
                        for u_idx in 0..w_usize {
                            let ray = rect_cam.pixel_to_vector(Vec2::new(u_idx as f64, v));
                            let src = full_cam.point_to_pixel(rect_cam.center + ray * 1000.0);
                            let lx = src.x - x0_f;
                            let ly_s = src.y - y0_f;
                            if lx >= 0.0 && ly_s >= 0.0 && lx < (pw - 1) as f64 && ly_s < (ph - 1) as f64 {
                                let ix = lx.floor() as i32;
                                let iy = ly_s.floor() as i32;
                                let fx = (lx - ix as f64) as f32;
                                let fy = (ly_s - iy as f64) as f32;
                                let i00 = (iy * pw + ix) as usize;
                                let i10 = i00 + 1;
                                let i01 = i00 + pw as usize;
                                let i11 = i01 + 1;
                                let top = pdata[i00] * (1.0 - fx) + pdata[i10] * fx;
                                let bot = pdata[i01] * (1.0 - fx) + pdata[i11] * fx;
                                chunk[row_off + u_idx] = top * (1.0 - fy) + bot * fy;
                                if let Some(pr) = prgb_ref {
                                    rgb_chunk[row_off + u_idx] = pr.sample_bilinear_clamped(lx, ly_s);
                                }
                            } else {
                                chunk[row_off + u_idx] = 0.0;
                                rgb_chunk[row_off + u_idx] = [0, 0, 0];
                            }
                        }
                    }
                });
            }
        } else {
            for (t, chunk) in out.data.chunks_mut(rows_per * w_usize).enumerate() {
                let y_start = t * rows_per;
                scope.spawn(move || {
                    let rows = chunk.len() / w_usize;
                    for ly in 0..rows {
                        let v = (y_start + ly) as f64;
                        let row_off = ly * w_usize;
                        for u_idx in 0..w_usize {
                            let ray = rect_cam.pixel_to_vector(Vec2::new(u_idx as f64, v));
                            let src = full_cam.point_to_pixel(rect_cam.center + ray * 1000.0);
                            let lx = src.x - x0_f;
                            let ly_s = src.y - y0_f;
                            if lx >= 0.0 && ly_s >= 0.0 && lx < (pw - 1) as f64 && ly_s < (ph - 1) as f64 {
                                let ix = lx.floor() as i32;
                                let iy = ly_s.floor() as i32;
                                let fx = (lx - ix as f64) as f32;
                                let fy = (ly_s - iy as f64) as f32;
                                let i00 = (iy * pw + ix) as usize;
                                let i10 = i00 + 1;
                                let i01 = i00 + pw as usize;
                                let i11 = i01 + 1;
                                let top = pdata[i00] * (1.0 - fx) + pdata[i10] * fx;
                                let bot = pdata[i01] * (1.0 - fx) + pdata[i11] * fx;
                                chunk[row_off + u_idx] = top * (1.0 - fy) + bot * fy;
                            } else {
                                chunk[row_off + u_idx] = 0.0;
                            }
                        }
                    }
                });
            }
        }
    });

    Ok((out, out_rgb, (x0, y0, sw, sh)))
}

/// Fast (~3 ms) 4x-downsampled 1D horizontal NCC sweep to lock the rectified epipolar search box
/// directly onto the real local terrain elevation (handles +/- 80m elevation offset automatically).
fn estimate_coarse_epipolar_shift(left: &ImageF32, right: &ImageF32, max_shift_px: i32) -> i32 {
    let scale = 4u32;
    let dw = (left.width / scale) as usize;
    let dh = (left.height / scale) as usize;
    if dw < 32 || dh < 32 {
        return 0;
    }

    let downsample = |img: &ImageF32| -> Vec<f32> {
        let mut out = vec![0.0f32; dw * dh];
        for y in 0..dh {
            for x in 0..dw {
                let mut sum = 0.0f32;
                for sy in 0..scale as usize {
                    let row = (y * scale as usize + sy) * img.width as usize;
                    for sx in 0..scale as usize {
                        sum += img.data[row + x * scale as usize + sx];
                    }
                }
                out[y * dw + x] = sum / (scale * scale) as f32;
            }
        }
        out
    };

    let l4 = downsample(left);
    let r4 = downsample(right);
    let max_d4 = ((max_shift_px / scale as i32).min((dw / 3) as i32)).max(4);
    let y0 = dh / 5;
    let y1 = dh - y0;
    let x_margin = max_d4 as usize + 4;
    if dw <= 2 * x_margin + 8 {
        return 0;
    }

    let mut best_d4 = 0i32;
    let mut best_ncc = 0.15f64;
    for d4 in -max_d4..=max_d4 {
        let mut sum_l = 0.0f64;
        let mut sum_r = 0.0f64;
        let mut cnt = 0.0f64;
        for y in y0..y1 {
            let row = y * dw;
            for x in x_margin..(dw - x_margin) {
                let rx = (x as i32 + d4) as usize;
                sum_l += l4[row + x] as f64;
                sum_r += r4[row + rx] as f64;
                cnt += 1.0;
            }
        }
        let mean_l = sum_l / cnt.max(1.0);
        let mean_r = sum_r / cnt.max(1.0);
        let mut num = 0.0f64;
        let mut vl = 0.0f64;
        let mut vr = 0.0f64;
        for y in y0..y1 {
            let row = y * dw;
            for x in x_margin..(dw - x_margin) {
                let rx = (x as i32 + d4) as usize;
                let dl = l4[row + x] as f64 - mean_l;
                let dr = r4[row + rx] as f64 - mean_r;
                num += dl * dr;
                vl += dl * dl;
                vr += dr * dr;
            }
        }
        let denom = (vl * vr).sqrt();
        if denom > 1e-6 {
            let ncc = num / denom;
            if ncc > best_ncc {
                best_ncc = ncc;
                best_d4 = d4;
            }
        }
    }
    best_d4 * scale as i32
}

pub struct PreparedStereoRoi {
    pub left_img: ImageF32,
    pub left_rgb: Option<crate::image::ImageRgb>,
    pub right_img: ImageF32,
    pub left_cam: AnyCamera,
    pub right_cam: AnyCamera,
    pub left_roi: (u32, u32, u32, u32),
    pub right_roi: (u32, u32, u32, u32),
    pub search_box: (i32, i32, i32, i32), // (min_x, min_y, max_x, max_y)
    pub full_dims: (u32, u32),
    pub ground_z_ref: f64,
    pub coarse_dx_px: i32,
    pub anchored_gcp: Option<String>,
}

/// Stream and epipolar-rectify a Full-Scale 1:1 stereo ROI directly from disk GeoTIFFs
/// (handles 433 Mpx DMC-IV imagery in < 15 MB RAM, aligns epipolar lines strictly along +X with dy=0).
pub fn prepare_stereo_pair_roi(
    chantier: &Chantier,
    pair: &StereoPairEntry,
    roi_size: u32,
) -> Result<PreparedStereoRoi> {
    prepare_stereo_pair_roi_opts(chantier, pair, roi_size, false)
}

/// Same as `prepare_stereo_pair_roi`, optionally centering the 1:1 ROI on the closest GCP
/// inside the pair's overlap footprint for direct vertical GCP RMSE validation.
pub fn prepare_stereo_pair_roi_opts(
    chantier: &Chantier,
    pair: &StereoPairEntry,
    roi_size: u32,
    center_gcp: bool,
) -> Result<PreparedStereoRoi> {
    let reader_l = GeoTiffReader::open(&pair.left_image_path)?;
    let reader_r = GeoTiffReader::open(&pair.right_image_path)?;
    let lw = reader_l.meta.width;
    let lh = reader_l.meta.height;
    let rw = reader_r.meta.width;
    let rh = reader_r.meta.height;

    let full_cam_l = build_pinhole_for_pose(&chantier.calibration, &pair.left_pose, lw, lh, 0, 0);
    let full_cam_r = build_pinhole_for_pose(&chantier.calibration, &pair.right_pose, rw, rh, 0, 0);

    let eff_w = roi_size.min(lw).min(rw).max(64);
    let eff_h = roi_size.min(lh).min(rh).max(64);

    let mid_x = 0.5 * (pair.left_pose.x + pair.right_pose.x);
    let mid_y = 0.5 * (pair.left_pose.y + pair.right_pose.y);
    let mut ground_z = chantier.estimated_ground_z(pair);
    let mut ground_anchor = Vec3::new(mid_x, mid_y, ground_z);
    let mut anchored_gcp = None;

    if center_gcp && !chantier.gcps.is_empty() {
        let margin = (eff_w.max(eff_h) as f64) * 0.55 + 32.0;
        let mut best: Option<(f64, &GroundControlPoint)> = None;
        for g in &chantier.gcps {
            let gp = Vec3::new(g.x, g.y, g.z);
            let pl = full_cam_l.point_to_pixel(gp);
            let pr = full_cam_r.point_to_pixel(gp);
            let inside_l = pl.x >= margin && pl.x <= lw as f64 - margin && pl.y >= margin && pl.y <= lh as f64 - margin;
            let inside_r = pr.x >= margin && pr.x <= rw as f64 - margin && pr.y >= margin && pr.y <= rh as f64 - margin;
            if inside_l && inside_r {
                let d = (g.x - mid_x).hypot(g.y - mid_y);
                if best.map_or(true, |(bd, _)| d < bd) {
                    best = Some((d, g));
                }
            }
        }
        if let Some((_, g)) = best {
            ground_z = g.z;
            ground_anchor = Vec3::new(g.x, g.y, g.z);
            anchored_gcp = Some(g.id.clone());
        }
    }

    // Build exact Fusiello epipolar-rectified pinhole cameras (dy == 0.000 px everywhere!)
    let (rect_cam_l, rect_cam_r) =
        build_epipolar_rectified_cameras(&full_cam_l, &full_cam_r, ground_anchor, eff_w, eff_h);

    // Stream ONLY the intersecting source tiles/strips from disk & warp to epipolar grid
    let (left_img, left_rgb, left_roi) =
        stream_and_rectify_roi(&reader_l, &full_cam_l, &rect_cam_l, eff_w, eff_h, true)?;
    let (right_img, _, right_roi) =
        stream_and_rectify_roi(&reader_r, &full_cam_r, &rect_cam_r, eff_w, eff_h, false)?;

    // Lock coarse horizontal disparity shift onto actual terrain elevation in ~3 ms
    let coarse_dx = estimate_coarse_epipolar_shift(&left_img, &right_img, 240);

    // Compute exact horizontal parallax sensitivity for +/- 25 m local relief around locked terrain
    let p_lo_l = rect_cam_l.point_to_pixel(Vec3::new(ground_anchor.x, ground_anchor.y, ground_z - 25.0));
    let p_lo_r = rect_cam_r.point_to_pixel(Vec3::new(ground_anchor.x, ground_anchor.y, ground_z - 25.0));
    let p_hi_l = rect_cam_l.point_to_pixel(Vec3::new(ground_anchor.x, ground_anchor.y, ground_z + 25.0));
    let p_hi_r = rect_cam_r.point_to_pixel(Vec3::new(ground_anchor.x, ground_anchor.y, ground_z + 25.0));

    let dx_span = ((p_hi_r.x - p_hi_l.x).abs().max((p_lo_r.x - p_lo_l.x).abs())).round() as i32;
    let half_span = dx_span.clamp(32, 64);

    let min_x = (coarse_dx - half_span).clamp(-256, 256);
    let max_x = (coarse_dx + half_span).clamp(-256, 256);

    Ok(PreparedStereoRoi {
        left_img,
        left_rgb,
        right_img,
        left_cam: AnyCamera::Pinhole(rect_cam_l),
        right_cam: AnyCamera::Pinhole(rect_cam_r),
        left_roi,
        right_roi,
        search_box: (min_x, 0, max_x, 0),
        full_dims: (lw, lh),
        ground_z_ref: ground_z,
        coarse_dx_px: coarse_dx,
        anchored_gcp,
    })
}

/// Per-GCP vertical validation detail (`dz_m = z_dem - z_gcp`).
#[derive(Debug, Clone)]
pub struct GcpSampleError {
    pub id: String,
    pub x: f64,
    pub y: f64,
    pub z_gcp: f64,
    pub z_dem: f64,
    pub dz_m: f64,
}

/// Vertical accuracy report comparing triangulated DEM against known GCPs inside the DEM extent.
#[derive(Debug, Clone)]
pub struct GcpValidationReport {
    pub matched_gcps: usize,
    pub dz_bias_m: f64,
    pub dz_rmse_m: f64,
    pub dz_max_abs_m: f64,
    pub samples: Vec<GcpSampleError>,
}

/// Bilinearly sample a `DemGrid` at continuous raster coordinate `(fx, fy)`, falling back to nearest
/// finite neighbor when adjacent cells contain nodata.
fn sample_dem_bilinear(dem: &DemGrid, fx: f64, fy: f64) -> Option<f64> {
    let w = dem.elevation.width;
    let h = dem.elevation.height;
    if w == 0 || h == 0 || !fx.is_finite() || !fy.is_finite() {
        return None;
    }
    if fx < -0.5 || fy < -0.5 || fx > (w as f64 - 0.5) || fy > (h as f64 - 0.5) {
        return None;
    }
    let fx_c = fx.clamp(0.0, (w - 1) as f64);
    let fy_c = fy.clamp(0.0, (h - 1) as f64);
    let x0 = fx_c.floor() as u32;
    let y0 = fy_c.floor() as u32;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let tx = fx_c - x0 as f64;
    let ty = fy_c - y0 as f64;

    let corners = [
        (dem.elevation.get_checked(x0, y0), (1.0 - tx) * (1.0 - ty)),
        (dem.elevation.get_checked(x1, y0), tx * (1.0 - ty)),
        (dem.elevation.get_checked(x0, y1), (1.0 - tx) * ty),
        (dem.elevation.get_checked(x1, y1), tx * ty),
    ];

    let mut sum_w = 0.0f64;
    let mut sum_z = 0.0f64;
    for (val_opt, wt) in corners {
        if let Some(v) = val_opt {
            if v.is_finite() && wt > 0.0 {
                sum_w += wt;
                sum_z += (v as f64) * wt;
            }
        }
    }
    if sum_w > 1e-6 {
        Some(sum_z / sum_w)
    } else {
        let nx = fx_c.round() as u32;
        let ny = fy_c.round() as u32;
        dem.elevation
            .get_checked(nx, ny)
            .filter(|v| v.is_finite())
            .map(|v| v as f64)
    }
}

pub fn validate_dem_against_gcps(dem: &DemGrid, gcps: &[GroundControlPoint]) -> GcpValidationReport {
    let mut samples = Vec::new();
    for g in gcps {
        if g.x >= dem.min_x && g.x <= dem.max_x && g.y >= dem.min_y && g.y <= dem.max_y {
            let fx = (g.x - dem.min_x) / dem.pixel_size;
            let fy = (dem.max_y - g.y) / dem.pixel_size;
            if let Some(z_dem) = sample_dem_bilinear(dem, fx, fy) {
                samples.push(GcpSampleError {
                    id: g.id.clone(),
                    x: g.x,
                    y: g.y,
                    z_gcp: g.z,
                    z_dem,
                    dz_m: z_dem - g.z,
                });
            }
        }
    }
    if samples.is_empty() {
        return GcpValidationReport {
            matched_gcps: 0,
            dz_bias_m: 0.0,
            dz_rmse_m: 0.0,
            dz_max_abs_m: 0.0,
            samples,
        };
    }
    let n = samples.len() as f64;
    let bias = samples.iter().map(|s| s.dz_m).sum::<f64>() / n;
    let rmse = (samples.iter().map(|s| s.dz_m * s.dz_m).sum::<f64>() / n).sqrt();
    let max_abs = samples.iter().map(|s| s.dz_m.abs()).fold(0.0f64, f64::max);
    GcpValidationReport {
        matched_gcps: samples.len(),
        dz_bias_m: bias,
        dz_rmse_m: rmse,
        dz_max_abs_m: max_abs,
        samples,
    }
}

pub fn chantier_summary_json(chantier: &Chantier, pair_reports: &[(String, f64, f64, usize, GcpValidationReport)]) -> String {
    let mut root = BTreeMap::new();
    root.insert("root_dir".into(), JsonValue::String(chantier.root_dir.to_string_lossy().into_owned()));
    root.insert("camera_name".into(), JsonValue::String(chantier.calibration.name.clone()));
    root.insert("sensor_width_px".into(), JsonValue::Number(chantier.calibration.width_px as f64));
    root.insert("sensor_height_px".into(), JsonValue::Number(chantier.calibration.height_px as f64));
    root.insert("focal_mm".into(), JsonValue::Number(chantier.calibration.focal_mm));
    root.insert("pixel_size_mm".into(), JsonValue::Number(chantier.calibration.pixel_size_mm));
    root.insert("total_poses".into(), JsonValue::Number(chantier.poses.len() as f64));
    root.insert("total_gcps".into(), JsonValue::Number(chantier.gcps.len() as f64));
    root.insert("detected_pairs".into(), JsonValue::Number(chantier.pairs.len() as f64));

    let pairs_arr: Vec<JsonValue> = pair_reports
        .iter()
        .map(|(name, ms, valid_pct, pts, gcp)| {
            let mut m = BTreeMap::new();
            m.insert("pair".into(), JsonValue::String(name.clone()));
            m.insert("total_ms".into(), JsonValue::Number(*ms));
            m.insert("valid_pct".into(), JsonValue::Number(*valid_pct));
            m.insert("points_3d".into(), JsonValue::Number(*pts as f64));
            m.insert("gcp_matched".into(), JsonValue::Number(gcp.matched_gcps as f64));
            m.insert("gcp_dz_bias_m".into(), JsonValue::Number(gcp.dz_bias_m));
            m.insert("gcp_dz_rmse_m".into(), JsonValue::Number(gcp.dz_rmse_m));
            m.insert("gcp_dz_max_abs_m".into(), JsonValue::Number(gcp.dz_max_abs_m));
            if !gcp.samples.is_empty() {
                let g_arr: Vec<JsonValue> = gcp
                    .samples
                    .iter()
                    .map(|s| {
                        let mut gm = BTreeMap::new();
                        gm.insert("id".into(), JsonValue::String(s.id.clone()));
                        gm.insert("x".into(), JsonValue::Number(s.x));
                        gm.insert("y".into(), JsonValue::Number(s.y));
                        gm.insert("z_gcp".into(), JsonValue::Number(s.z_gcp));
                        gm.insert("z_dem".into(), JsonValue::Number(s.z_dem));
                        gm.insert("dz_m".into(), JsonValue::Number(s.dz_m));
                        JsonValue::Object(gm)
                    })
                    .collect();
                m.insert("gcp_samples".into(), JsonValue::Array(g_arr));
            }
            JsonValue::Object(m)
        })
        .collect();
    root.insert("processed_pairs".into(), JsonValue::Array(pairs_arr));
    JsonValue::Object(root).to_pretty_string()
}

fn collect_files_recursive(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();
                if depth > 0 && !name.starts_with("output") && !name.starts_with('.') {
                    collect_files_recursive(&p, depth - 1, out);
                }
            } else {
                out.push(p);
            }
        }
    }
}

fn find_image_for_pose(pose_id: &str, tif_map: &HashMap<String, PathBuf>) -> Option<PathBuf> {
    let low = pose_id.trim().to_ascii_lowercase();
    if let Some(p) = tif_map.get(&low) {
        return Some(p.clone());
    }
    let stem = Path::new(&low)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(&low);
    if let Some(p) = tif_map.get(stem) {
        return Some(p.clone());
    }
    // Suffix match (e.g. pose_id "0042" matches "MEYREUIL_0042.tif")
    for (k, v) in tif_map {
        if k.ends_with(stem) && !k.contains('.') {
            return Some(v.clone());
        }
    }
    None
}

fn angle_diff_deg(a: f64, b: f64) -> f64 {
    let mut d = (a - b) % 360.0;
    if d > 180.0 { d -= 360.0; }
    if d < -180.0 { d += 360.0; }
    d.abs()
}

fn sanitize_id(s: &str) -> String {
    Path::new(s)
        .file_stem()
        .and_then(|x| x.to_str())
        .unwrap_or(s)
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_dmciv_cam_and_trajectory() {
        let cam_txt = "CAMERA;DMCIV_H_44610;143.0;0.0;0.0;1800.0;0;12500;31520;13760;3.76;0\n\
                       CAMDIST;-1.2e-9;0.0;0.0;0.0;0.0\n";
        let cal = CameraCalibration::parse_str(cam_txt).unwrap();
        assert_eq!(cal.width_px, 31520);
        assert_eq!(cal.height_px, 13760);
        assert!((cal.pixel_size_mm - 0.00376).abs() < 1e-9);
        assert!((cal.focal_px() - 143.0 / 0.00376).abs() < 1e-4);

        let traj_txt = "# Meyreuil trajectory\n\
                        IMG_001 895000.0 6265000.0 1950.0 0.12 -0.05 90.0\n\
                        IMG_002 895200.0 6265000.0 1951.0 0.10 -0.04 90.1\n";
        let poses = parse_trajectory_str(traj_txt).unwrap();
        assert_eq!(poses.len(), 2);
        assert!((poses[1].x - poses[0].x - 200.0).abs() < 1e-6);
    }

    #[test]
    fn test_epipolar_rectification_y_axis_flight_zero_dy() {
        // Real DMC-IV Meyreuil-like geometry: flying along Y (dy ~ 5259 px), non-zero omega/phi/kappa
        let cal = CameraCalibration::parse_str(
            "CAMERA;DMCIV;143.0;0.0;0.0;1800.0;0;12500;31520;13760;3.76;0\n",
        )
        .unwrap();
        let p0 = ImagePose {
            id: "025_0709".into(),
            x: 1_901_500.0,
            y: 3_145_000.0,
            z: 2_030.0,
            omega_deg: 0.35,
            phi_deg: -0.28,
            kappa_deg: 0.42,
        };
        let p1 = ImagePose {
            id: "025_0710".into(),
            x: 1_901_502.0,
            y: 3_145_250.0,
            z: 2_031.5,
            omega_deg: -0.22,
            phi_deg: 0.31,
            kappa_deg: 0.15,
        };
        let cam_l = build_pinhole_for_pose(&cal, &p0, 31520, 13760, 0, 0);
        let cam_r = build_pinhole_for_pose(&cal, &p1, 31520, 13760, 0, 0);
        let anchor = Vec3::new(1_901_501.0, 3_145_125.0, 225.0);

        // Before rectification: flight parallax is ~5270 px along Y!
        let raw_l = cam_l.point_to_pixel(anchor);
        let raw_r = cam_r.point_to_pixel(anchor);
        assert!((raw_r.y - raw_l.y).abs() > 4000.0);

        let (rect_l, rect_r) = build_epipolar_rectified_cameras(&cam_l, &cam_r, anchor, 1024, 1024);

        // Anchor must project to (512, 512) in both rectified views
        let al = rect_l.point_to_pixel(anchor);
        let ar = rect_r.point_to_pixel(anchor);
        assert!((al.x - 512.0).abs() < 1e-6 && (al.y - 512.0).abs() < 1e-6);
        assert!((ar.x - 512.0).abs() < 1e-6 && (ar.y - 512.0).abs() < 1e-6);

        // Across +/- 40 m of elevation and +/- 20 m XY offsets, vertical parallax dy MUST be 0!
        let mut max_dx = 0.0f64;
        for &dz in &[-40.0, -15.0, 0.0, 15.0, 40.0] {
            for &ox in &[-20.0, 0.0, 20.0] {
                let pt = Vec3::new(anchor.x + ox, anchor.y - ox * 0.5, anchor.z + dz);
                let pl = rect_l.point_to_pixel(pt);
                let pr = rect_r.point_to_pixel(pt);
                assert!(
                    (pr.y - pl.y).abs() < 1e-6,
                    "Non-zero epipolar dy={} at dz={}",
                    pr.y - pl.y,
                    dz
                );
                max_dx = max_dx.max((pr.x - pl.x).abs());
            }
        }
        // Horizontal parallax must now carry the full height sensitivity (> 100 px for 40 m relief)
        assert!(max_dx > 100.0, "Expected strong horizontal parallax, got {}", max_dx);
    }

    #[test]
    fn test_bilinear_gcp_validation_submillimeter() {
        let mut elev = ImageF32::new(10, 10);
        // Tilted plane: z(fx, fy) = 200.0 + 2.0 * fx + 4.0 * fy
        for y in 0..10 {
            for x in 0..10 {
                elev.set(x, y, 200.0 + 2.0 * x as f32 + 4.0 * y as f32);
            }
        }
        let dem = DemGrid {
            elevation: elev.clone(),
            hillshade: elev,
            min_x: 1_901_000.0,
            min_y: 3_143_000.0,
            max_x: 1_901_009.0,
            max_y: 3_143_009.0,
            pixel_size: 1.0,
        };
        // GCP at fractional pixel fx = 3.25, fy = 5.5 -> exact z = 200 + 6.5 + 22.0 = 228.5 m
        let gcp = GroundControlPoint {
            id: "GCP_MEY_01".into(),
            x: 1_901_003.25,
            y: 3_143_003.50,
            z: 228.25, // +0.25 m bias (DEM is 0.25 m above GCP)
        };
        let rep = validate_dem_against_gcps(&dem, &[gcp]);
        assert_eq!(rep.matched_gcps, 1);
        assert_eq!(rep.samples.len(), 1);
        assert!((rep.samples[0].z_dem - 228.5).abs() < 1e-4);
        assert!((rep.dz_bias_m - 0.25).abs() < 1e-4);
        assert!((rep.dz_rmse_m - 0.25).abs() < 1e-4);
    }

    #[test]
    fn test_trajectory_reverse_flight_line_kappa_degrees_preserved() {
        // Flight line L0 (kappa ~ 0 deg) + reverse line L2 (kappa ~ +/-180 deg, 90.1 deg, 270 deg)
        // Must NEVER be multiplied by 0.9!
        let traj = "\
# Meyreuil polygon survey (degrees)
025_0709 1901500.0 3145000.0 2030.0 0.35 -0.28 0.42
026_0731 1902200.0 3146000.0 2031.0 -0.15 0.22 179.80
026_0732 1902201.0 3145750.0 2030.5 0.11 -0.19 -179.40
026_0733 1902202.0 3145500.0 2030.2 0.05 0.08 90.10
026_0734 1902203.0 3145250.0 2030.1 -0.04 0.03 270.00
";
        let poses = parse_trajectory_str(traj).unwrap();
        assert_eq!(poses.len(), 5);
        assert!((poses[0].kappa_deg - 0.42).abs() < 1e-9);
        assert!((poses[1].kappa_deg - 179.80).abs() < 1e-9);
        assert!((poses[2].kappa_deg - (-179.40)).abs() < 1e-9);
        assert!((poses[3].kappa_deg - 90.10).abs() < 1e-9);
        assert!((poses[4].kappa_deg - 270.00).abs() < 1e-9);
    }

    #[test]
    fn test_trajectory_file_level_gons_conversion() {
        let traj_header = "\
# Units: gons
IMG_1 1901500.0 3145000.0 2030.0 0.50 -0.40 200.00
IMG_2 1901500.0 3145250.0 2030.0 0.10 0.20 100.00
";
        let p_hdr = parse_trajectory_str(traj_header).unwrap();
        assert!((p_hdr[0].kappa_deg - 180.0).abs() < 1e-9);
        assert!((p_hdr[0].omega_deg - 0.45).abs() < 1e-9);
        assert!((p_hdr[1].kappa_deg - 90.0).abs() < 1e-9);

        let traj_gt360 = "\
IMG_1 1901500.0 3145000.0 2030.0 0.50 -0.40 398.00
IMG_2 1901500.0 3145250.0 2030.0 0.10 0.20 200.00
";
        let p_gt = parse_trajectory_str(traj_gt360).unwrap();
        assert!((p_gt[0].kappa_deg - 358.2).abs() < 1e-9);
        assert!((p_gt[1].kappa_deg - 180.0).abs() < 1e-9);
    }
}
