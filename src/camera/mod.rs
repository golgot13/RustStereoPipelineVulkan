//! Camera models (Pinhole .tsai and Satellite RPC .rpb/.xml) compatible with Ames Stereo Pipeline.

pub mod pinhole;
pub mod rpc;

use std::io;
use std::path::Path;

use crate::math::{Vec2, Vec3};
pub use pinhole::PinholeCamera;
pub use rpc::RpcCamera;

/// Unified camera model interface matching ASP/VW CameraModel.
pub trait CameraModel: Send + Sync {
    fn model_type(&self) -> &'static str;

    /// Project a 3D world point (meters) to 2D image pixel coordinates `(col, row)`.
    fn point_to_pixel(&self, point: Vec3) -> Vec2;

    /// Unit viewing direction vector in world frame from pixel `(col, row)`.
    fn pixel_to_vector(&self, pixel: Vec2) -> Vec3;

    /// 3D camera optical center (or ray origin for RPC) in world frame.
    fn camera_center(&self, pixel: Vec2) -> Vec3;
}

/// Enum wrapper so cameras can be stored, cloned, and serialized cleanly without dynamic trait overhead.
#[derive(Debug, Clone)]
pub enum AnyCamera {
    Pinhole(PinholeCamera),
    Rpc(RpcCamera),
}

impl CameraModel for AnyCamera {
    fn model_type(&self) -> &'static str {
        match self {
            Self::Pinhole(c) => c.model_type(),
            Self::Rpc(c) => c.model_type(),
        }
    }

    fn point_to_pixel(&self, point: Vec3) -> Vec2 {
        match self {
            Self::Pinhole(c) => c.point_to_pixel(point),
            Self::Rpc(c) => c.point_to_pixel(point),
        }
    }

    fn pixel_to_vector(&self, pixel: Vec2) -> Vec3 {
        match self {
            Self::Pinhole(c) => c.pixel_to_vector(pixel),
            Self::Rpc(c) => c.pixel_to_vector(pixel),
        }
    }

    fn camera_center(&self, pixel: Vec2) -> Vec3 {
        match self {
            Self::Pinhole(c) => c.camera_center(pixel),
            Self::Rpc(c) => c.camera_center(pixel),
        }
    }
}

/// Load a camera model from a `.tsai` (Pinhole) or `.rpb` / `.xml` (RPC) file.
pub fn load_camera(path: &Path) -> io::Result<AnyCamera> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "tsai" => Ok(AnyCamera::Pinhole(PinholeCamera::load_tsai(path)?)),
        "rpb" | "rpc" => Ok(AnyCamera::Rpc(RpcCamera::load_rpb(path)?)),
        _ => {
            let content = std::fs::read_to_string(path)?;
            if content.contains("lineNumCoef") || content.contains("LINE_NUM_COEFF") {
                Ok(AnyCamera::Rpc(RpcCamera::parse_rpb(&content)?))
            } else {
                Ok(AnyCamera::Pinhole(PinholeCamera::parse_tsai(&content)?))
            }
        }
    }
}
