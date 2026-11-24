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
use crate::math::{Mat3, Vec3};

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

    for raw in content.lines() {
        let line = raw.trim();
        if line.is_empty()
            || line.starts_with('#')
            || line.starts_with("//")
            || line.starts_with('[')
            || line.starts_with("CAMDIST;")
        {
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

        // Normalize micro-degrees and gons -> degrees (AEROPHOTOGRAM rules)
        if omega.abs() > 10_000.0 { omega /= 1_000_000.0; }
        if phi.abs() > 10_000.0 { phi /= 1_000_000.0; }
        if kappa.abs() > 10_000.0 { kappa /= 1_000_000.0; }
        if omega.abs() > 90.0 && omega.abs() <= 400.0 { omega *= 0.9; }
        if phi.abs() > 90.0 && phi.abs() <= 400.0 { phi *= 0.9; }
        if kappa.abs() > 90.0 && kappa.abs() <= 400.0 { kappa *= 0.9; }

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
    Ok(poses)
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
            let sum: f64 = self.gcps.iter().map(|g| g.z).sum();
            return sum / self.gcps.len() as f64;
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

    // Photogrammetric rotation R = Rz(kappa) * Ry(phi) * Rx(omega) (camera: +x right, +y up, -z nadir)
    // Convert image sensor frame (+u right, +v down, +w forward/nadir) to world frame via diag(1, -1, -1)
    let r_opk = Mat3::from_euler_xyz(
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
    cam.distortion = cal.distortion;
    cam
}

pub struct PreparedStereoRoi {
    pub left_img: ImageF32,
    pub right_img: ImageF32,
    pub left_cam: AnyCamera,
    pub right_cam: AnyCamera,
    pub left_roi: (u32, u32, u32, u32),
    pub right_roi: (u32, u32, u32, u32),
    pub search_box: (i32, i32, i32, i32), // (min_x, min_y, max_x, max_y)
    pub full_dims: (u32, u32),
}

/// Stream a Full-Scale 1:1 ground-aligned stereo ROI directly from disk GeoTIFFs (handles 433 Mpx in < 10 MB RAM!).
pub fn prepare_stereo_pair_roi(
    chantier: &Chantier,
    pair: &StereoPairEntry,
    roi_size: u32,
) -> Result<PreparedStereoRoi> {
    let reader_l = GeoTiffReader::open(&pair.left_image_path)?;
    let reader_r = GeoTiffReader::open(&pair.right_image_path)?;
    let lw = reader_l.meta.width;
    let lh = reader_l.meta.height;
    let rw = reader_r.meta.width;
    let rh = reader_r.meta.height;

    let full_cam_l = build_pinhole_for_pose(&chantier.calibration, &pair.left_pose, lw, lh, 0, 0);
    let full_cam_r = build_pinhole_for_pose(&chantier.calibration, &pair.right_pose, rw, rh, 0, 0);

    // Project ground midpoint between left and right cameras onto both sensors to align 1:1 ROIs on the exact same terrain!
    let ground_z = chantier.estimated_ground_z(pair);
    let ground_mid = Vec3::new(
        0.5 * (pair.left_pose.x + pair.right_pose.x),
        0.5 * (pair.left_pose.y + pair.right_pose.y),
        ground_z,
    );
    let pl = full_cam_l.point_to_pixel(ground_mid);
    let pr = full_cam_r.point_to_pixel(ground_mid);

    let eff_w = roi_size.min(lw).min(rw).max(64);
    let eff_h = roi_size.min(lh).min(rh).max(64);

    let clamp_origin = |center: f64, span: u32, max_dim: u32| -> u32 {
        let raw = (center - span as f64 * 0.5).round() as i64;
        raw.clamp(0, (max_dim.saturating_sub(span)) as i64) as u32
    };

    let lx0 = if pl.x.is_finite() && pl.x >= 0.0 && pl.x < lw as f64 {
        clamp_origin(pl.x, eff_w, lw)
    } else {
        (lw.saturating_sub(eff_w)) / 2
    };
    let ly0 = if pl.y.is_finite() && pl.y >= 0.0 && pl.y < lh as f64 {
        clamp_origin(pl.y, eff_h, lh)
    } else {
        (lh.saturating_sub(eff_h)) / 2
    };
    let rx0 = if pr.x.is_finite() && pr.x >= 0.0 && pr.x < rw as f64 {
        clamp_origin(pr.x, eff_w, rw)
    } else {
        (rw.saturating_sub(eff_w)) / 2
    };
    let ry0 = if pr.y.is_finite() && pr.y >= 0.0 && pr.y < rh as f64 {
        clamp_origin(pr.y, eff_h, rh)
    } else {
        (rh.saturating_sub(eff_h)) / 2
    };

    // Stream ONLY the intersecting tiles/strips from disk!
    let left_img = reader_l.read_roi(lx0, ly0, eff_w, eff_h)?;
    let right_img = reader_r.read_roi(rx0, ry0, eff_w, eff_h)?;

    let roi_cam_l = build_pinhole_for_pose(&chantier.calibration, &pair.left_pose, lw, lh, lx0, ly0);
    let roi_cam_r = build_pinhole_for_pose(&chantier.calibration, &pair.right_pose, rw, rh, rx0, ry0);

    // Estimate residual disparity search window around +/- 120m terrain height variation
    let p_lo_l = roi_cam_l.point_to_pixel(Vec3::new(ground_mid.x, ground_mid.y, ground_z - 120.0));
    let p_lo_r = roi_cam_r.point_to_pixel(Vec3::new(ground_mid.x, ground_mid.y, ground_z - 120.0));
    let p_hi_l = roi_cam_l.point_to_pixel(Vec3::new(ground_mid.x, ground_mid.y, ground_z + 120.0));
    let p_hi_r = roi_cam_r.point_to_pixel(Vec3::new(ground_mid.x, ground_mid.y, ground_z + 120.0));

    let dx_lo = (p_lo_r.x - p_lo_l.x).round() as i32;
    let dx_hi = (p_hi_r.x - p_hi_l.x).round() as i32;
    let dy_lo = (p_lo_r.y - p_lo_l.y).round() as i32;
    let dy_hi = (p_hi_r.y - p_hi_l.y).round() as i32;

    let min_x = dx_lo.min(dx_hi).min(-16).clamp(-96, 0);
    let max_x = dx_lo.max(dx_hi).max(16).clamp(0, 96);
    let min_y = dy_lo.min(dy_hi).min(-4).clamp(-16, 0);
    let max_y = dy_lo.max(dy_hi).max(4).clamp(0, 16);

    Ok(PreparedStereoRoi {
        left_img,
        right_img,
        left_cam: AnyCamera::Pinhole(roi_cam_l),
        right_cam: AnyCamera::Pinhole(roi_cam_r),
        left_roi: (lx0, ly0, eff_w, eff_h),
        right_roi: (rx0, ry0, eff_w, eff_h),
        search_box: (min_x, min_y, max_x, max_y),
        full_dims: (lw, lh),
    })
}

/// Vertical accuracy report comparing triangulated DEM against known GCPs inside the DEM extent.
#[derive(Debug, Clone)]
pub struct GcpValidationReport {
    pub matched_gcps: usize,
    pub dz_bias_m: f64,
    pub dz_rmse_m: f64,
    pub dz_max_abs_m: f64,
}

pub fn validate_dem_against_gcps(dem: &DemGrid, gcps: &[GroundControlPoint]) -> GcpValidationReport {
    let mut diffs = Vec::new();
    for g in gcps {
        if g.x >= dem.min_x && g.x <= dem.max_x && g.y >= dem.min_y && g.y <= dem.max_y {
            let fx = ((g.x - dem.min_x) / dem.pixel_size) as f32;
            let fy = ((dem.max_y - g.y) / dem.pixel_size) as f32;
            let ix = fx.round() as u32;
            let iy = fy.round() as u32;
            if let Some(z_dem) = dem.elevation.get_checked(ix, iy) {
                if z_dem.is_finite() {
                    diffs.push(z_dem as f64 - g.z);
                }
            }
        }
    }
    if diffs.is_empty() {
        return GcpValidationReport {
            matched_gcps: 0,
            dz_bias_m: 0.0,
            dz_rmse_m: 0.0,
            dz_max_abs_m: 0.0,
        };
    }
    let n = diffs.len() as f64;
    let bias = diffs.iter().sum::<f64>() / n;
    let rmse = (diffs.iter().map(|d| d * d).sum::<f64>() / n).sqrt();
    let max_abs = diffs.iter().map(|d| d.abs()).fold(0.0f64, f64::max);
    GcpValidationReport {
        matched_gcps: diffs.len(),
        dz_bias_m: bias,
        dz_rmse_m: rmse,
        dz_max_abs_m: max_abs,
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
            m.insert("gcp_dz_rmse_m".into(), JsonValue::Number(gcp.dz_rmse_m));
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
}
