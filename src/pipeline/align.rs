//! `pc_align` — Pure-Rust 3D Iterative Closest Point (ICP) rigid point-cloud alignment.
//! Solves for optimal rotation `R` and translation `t` minimizing `sum || R * src_i + t - ref_j ||^2`
//! using Kabsch SVD (`Mat3::closest_rotation`).

use crate::math::{Mat3, Vec3};

#[derive(Debug, Clone)]
pub struct IcpResult {
    pub rotation: Mat3,
    pub translation: Vec3,
    pub initial_rmse: f64,
    pub final_rmse: f64,
    pub iterations: usize,
}

/// Align `source` point cloud onto `reference` point cloud using trimmed rigid ICP.
pub fn align_icp(
    reference: &[Vec3],
    source: &[Vec3],
    max_iterations: usize,
    max_displacement: f64,
) -> IcpResult {
    if reference.is_empty() || source.is_empty() {
        return IcpResult {
            rotation: Mat3::IDENTITY,
            translation: Vec3::ZERO,
            initial_rmse: 0.0,
            final_rmse: 0.0,
            iterations: 0,
        };
    }

    // Subsample large clouds deterministically for fast ICP convergence
    let ref_step = (reference.len() / 2000).max(1);
    let src_step = (source.len() / 1500).max(1);
    let ref_pts: Vec<Vec3> = reference.iter().step_by(ref_step).copied().collect();
    let src_pts: Vec<Vec3> = source.iter().step_by(src_step).copied().collect();

    let mut r_total = Mat3::IDENTITY;
    let mut t_total = Vec3::ZERO;
    let mut initial_rmse = 0.0;
    let mut final_rmse = 0.0;
    let mut iters_done = 0;

    let max_dist_sq = if max_displacement > 0.0 {
        max_displacement * max_displacement
    } else {
        f64::INFINITY
    };

    for iter in 0..max_iterations {
        iters_done = iter + 1;
        let mut pairs: Vec<(Vec3, Vec3, f64)> = Vec::with_capacity(src_pts.len());
        let mut err_sq_sum = 0.0;

        for &sp in &src_pts {
            let tp = r_total.mul_vec(sp) + t_total;
            let mut best_d2 = f64::INFINITY;
            let mut best_rp = ref_pts[0];
            for &rp in &ref_pts {
                let d2 = (tp - rp).norm_sq();
                if d2 < best_d2 {
                    best_d2 = d2;
                    best_rp = rp;
                }
            }
            if best_d2 <= max_dist_sq {
                pairs.push((tp, best_rp, best_d2));
                err_sq_sum += best_d2;
            }
        }

        if pairs.len() < 4 {
            break;
        }
        let rmse = (err_sq_sum / pairs.len() as f64).sqrt();
        if iter == 0 {
            initial_rmse = rmse;
        }
        final_rmse = rmse;

        // Trimmed ICP: keep closest 85% correspondences to reject outliers
        pairs.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        let keep = ((pairs.len() as f64 * 0.85) as usize).max(4);
        let inliers = &pairs[..keep];

        let mut mean_s = Vec3::ZERO;
        let mut mean_r = Vec3::ZERO;
        for &(s, r, _) in inliers {
            mean_s += s;
            mean_r += r;
        }
        mean_s = mean_s / inliers.len() as f64;
        mean_r = mean_r / inliers.len() as f64;

        // Cross-covariance matrix H = sum (r_i - mean_r) * (s_i - mean_s)^T
        let mut h = [[0.0; 3]; 3];
        for &(s, r, _) in inliers {
            let ds = s - mean_s;
            let dr = r - mean_r;
            let sa = ds.to_array();
            let ra = dr.to_array();
            for row in 0..3 {
                for col in 0..3 {
                    h[row][col] += ra[row] * sa[col];
                }
            }
        }
        let delta_r = Mat3::new(h).closest_rotation();
        let delta_t = mean_r - delta_r.mul_vec(mean_s);

        r_total = delta_r.mul_mat(&r_total);
        t_total = delta_r.mul_vec(t_total) + delta_t;

        if delta_t.norm() < 1e-6 {
            break;
        }
    }

    IcpResult {
        rotation: r_total,
        translation: t_total,
        initial_rmse,
        final_rmse,
        iterations: iters_done,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_icp_recovers_rigid_transform() {
        let mut ref_pts = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                let x = i as f64 * 2.0;
                let y = j as f64 * 2.0;
                let z = (x * 0.3).sin() * 3.0 + (y * 0.2).cos() * 2.0;
                ref_pts.push(Vec3::new(x, y, z));
            }
        }
        let rot = Mat3::from_euler_xyz(0.05, -0.04, 0.08);
        let trans = Vec3::new(1.2, -0.8, 0.5);
        let inv_r = rot.transpose();
        let src_pts: Vec<Vec3> = ref_pts
            .iter()
            .map(|&p| inv_r.mul_vec(p - trans))
            .collect();

        let res = align_icp(&ref_pts, &src_pts, 40, 20.0);
        assert!(res.final_rmse < 0.05, "ICP RMSE too high: {}", res.final_rmse);
    }
}
