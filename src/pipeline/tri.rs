//! Stage 4: `stereo_tri` — Multi-threaded 3D ray-ray stereo triangulation.

use crate::camera::{AnyCamera, CameraModel};
use crate::config::StereoSettings;
use crate::image::{DisparityMap, ImageF32, PointCloud};
use crate::math::{triangulate_rays, Vec2};

/// Triangulate all valid pixels in a disparity map into a 3D PointCloud with intersection errors.
pub fn run_tri(
    left_img: &ImageF32,
    disp: &DisparityMap,
    cam_left: &AnyCamera,
    cam_right: &AnyCamera,
    s: &StereoSettings,
) -> PointCloud {
    let w = disp.width as usize;
    let h = disp.height as usize;
    let mut pc = PointCloud::new(disp.width, disp.height);
    let n_threads = s.threads.clamp(1, h.max(1));
    let rows_per = (h + n_threads - 1) / n_threads;

    let near_r = s.near_universe_radius;
    let far_r = s.far_universe_radius;
    let max_err = s.max_triangulation_error;

    std::thread::scope(|scope| {
        for (t, (((ptc, erc), inc), vac)) in pc
            .points
            .chunks_mut(rows_per * w)
            .zip(pc.errors.chunks_mut(rows_per * w))
            .zip(pc.intensities.chunks_mut(rows_per * w))
            .zip(pc.valid.chunks_mut(rows_per * w))
            .enumerate()
        {
            let y_start = t * rows_per;
            scope.spawn(move || {
                let rows = ptc.len() / w;
                for ly in 0..rows {
                    let y = y_start + ly;
                    for x in 0..w {
                        let idx = y * w + x;
                        let local = ly * w + x;
                        if disp.valid[idx] == 0 {
                            continue;
                        }
                        let p1 = Vec2::new(x as f64, y as f64);
                        let p2 = Vec2::new(x as f64 + disp.dx[idx] as f64, y as f64 + disp.dy[idx] as f64);
                        let o1 = cam_left.camera_center(p1);
                        let d1 = cam_left.pixel_to_vector(p1);
                        let o2 = cam_right.camera_center(p2);
                        let d2 = cam_right.pixel_to_vector(p2);

                        if let Some((pt, err, t1, t2)) = triangulate_rays(o1, d1, o2, d2) {
                            if t1 <= 0.0 || t2 <= 0.0 || !pt.is_finite() {
                                continue;
                            }
                            if max_err > 0.0 && err > max_err {
                                continue;
                            }
                            if near_r > 0.0 || far_r > 0.0 {
                                let r = pt.norm();
                                if near_r > 0.0 && r < near_r {
                                    continue;
                                }
                                if far_r > 0.0 && r > far_r {
                                    continue;
                                }
                            }
                            ptc[local] = pt;
                            erc[local] = err as f32;
                            inc[local] = left_img.data[idx];
                            vac[local] = 1;
                        }
                    }
                }
            });
        }
    });

    pc
}
