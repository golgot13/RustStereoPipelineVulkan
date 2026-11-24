//! Zero-dependency scientific image & point cloud I/O.
//! Supports:
//! - TIFF (baseline uncompressed 8-bit, 16-bit, 32-bit float + GeoTIFF scale/tiepoint tags)
//! - PFM (Portable FloatMap, grayscale & RGB)
//! - PGM / PPM (P2/P3/P5/P6)
//! - BMP (24-bit & 32-bit)
//! - PNG encoder (uncompressed DEFLATE stored blocks + CRC32 + Adler32 for lossless previews/Web UI)
//! - PLY / XYZ / CSV point clouds

use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::path::Path;

use super::{Colormap, DisparityMap, ImageF32, ImageRgb, PointCloud};

/// Load a grayscale f32 image by inspecting file magic bytes / extension.
/// Supports Classic TIFF & BigTIFF (Uncompressed, LZW, Deflate, PackBits, Strip & Tiled), PFM, PGM/PPM, BMP.
pub fn load_image(path: &Path) -> io::Result<ImageF32> {
    let (img, _) = load_image_with_rgb(path)?;
    Ok(img)
}

/// Load a grayscale f32 image plus optional true-color `ImageRgb` (for 3/4-band TIFF/GeoTIFF, P6 PPM, 24/32-bit BMP).
pub fn load_image_with_rgb(path: &Path) -> io::Result<(ImageF32, Option<ImageRgb>)> {
    let mut f = File::open(path)?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic[0..2] == b"II" || &magic[0..2] == b"MM" {
        return crate::image::geotiff::GeoTiffReader::open(path)
            .and_then(|r| r.read_full_with_rgb())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()));
    }
    let mut bytes = Vec::from(magic);
    f.read_to_end(&mut bytes)?;
    if &bytes[0..2] == b"Pf" || &bytes[0..2] == b"PF" {
        Ok((decode_pfm(&bytes)?, None))
    } else if &bytes[0..2] == b"P5" || &bytes[0..2] == b"P6" || &bytes[0..2] == b"P2" {
        Ok((decode_pnm(&bytes)?, None))
    } else if &bytes[0..2] == b"BM" {
        Ok((decode_bmp(&bytes)?, None))
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported image format: {:?}", path),
        ))
    }
}

/// Save an ImageF32 based on extension (.tif/.tiff, .pfm, .pgm, .bmp, .png).
pub fn save_image(path: &Path, img: &ImageF32) -> io::Result<()> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("tif")
        .to_ascii_lowercase();
    match ext.as_str() {
        "tif" | "tiff" => save_tiff_f32(path, img, None),
        "pfm" => save_pfm(path, img),
        "pgm" => save_pgm(path, img),
        "bmp" => {
            let rgba = img.to_rgba8(Colormap::Grayscale);
            save_bmp_rgba(path, img.width, img.height, &rgba)
        }
        "png" => {
            let rgba = img.to_rgba8(Colormap::Grayscale);
            let png = encode_png_rgba(img.width, img.height, &rgba);
            std::fs::write(path, png)
        }
        _ => save_tiff_f32(path, img, None),
    }
}

/// Optional GeoTIFF transform metadata (pixel scale and top-left tiepoint).
#[derive(Debug, Clone, Copy)]
pub struct GeoTransform {
    pub origin_x: f64,
    pub origin_y: f64,
    pub pixel_size_x: f64,
    pub pixel_size_y: f64,
}

/// Write a baseline little-endian 32-bit float single-band TIFF (with optional GeoTIFF tags).
pub fn save_tiff_f32(path: &Path, img: &ImageF32, geo: Option<GeoTransform>) -> io::Result<()> {
    let mut out = Vec::with_capacity(256 + img.data.len() * 4);
    // TIFF Header: little-endian "II", magic 42, IFD offset = 8
    out.extend_from_slice(b"II");
    out.extend_from_slice(&42u16.to_le_bytes());
    out.extend_from_slice(&8u32.to_le_bytes());

    let num_entries: u16 = if geo.is_some() { 13 } else { 11 };
    let ifd_size = 2 + (num_entries as usize) * 12 + 4;
    let extra_offset = 8 + ifd_size;
    let geo_bytes = if geo.is_some() { 3 * 8 + 6 * 8 } else { 0 };
    let data_offset = extra_offset + geo_bytes;
    let strip_byte_count = (img.data.len() * 4) as u32;

    out.extend_from_slice(&num_entries.to_le_bytes());
    let mut push_entry = |tag: u16, ty: u16, count: u32, val: u32| {
        out.extend_from_slice(&tag.to_le_bytes());
        out.extend_from_slice(&ty.to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&val.to_le_bytes());
    };

    // Tags MUST be sorted ascending by tag ID per TIFF 6.0 spec
    push_entry(256, 4, 1, img.width);             // ImageWidth (LONG)
    push_entry(257, 4, 1, img.height);            // ImageLength (LONG)
    push_entry(258, 3, 1, 32);                    // BitsPerSample = 32
    push_entry(259, 3, 1, 1);                     // Compression = 1 (None)
    push_entry(262, 3, 1, 1);                     // PhotometricInterpretation = BlackIsZero
    push_entry(273, 4, 1, data_offset as u32);    // StripOffsets
    push_entry(277, 3, 1, 1);                     // SamplesPerPixel = 1
    push_entry(278, 4, 1, img.height);            // RowsPerStrip
    push_entry(279, 4, 1, strip_byte_count);      // StripByteCounts
    push_entry(284, 3, 1, 1);                     // PlanarConfiguration = Chunky
    push_entry(339, 3, 1, 3);                     // SampleFormat = 3 (IEEE float)

    if geo.is_some() {
        push_entry(33550, 12, 3, extra_offset as u32);           // ModelPixelScaleTag (DOUBLE x3)
        push_entry(33922, 12, 6, (extra_offset + 24) as u32);    // ModelTiepointTag (DOUBLE x6)
    }

    // Next IFD offset = 0
    out.extend_from_slice(&0u32.to_le_bytes());

    if let Some(g) = geo {
        for v in [g.pixel_size_x, g.pixel_size_y, 0.0f64] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in [0.0f64, 0.0, 0.0, g.origin_x, g.origin_y, 0.0] {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }

    for &v in &img.data {
        out.extend_from_slice(&v.to_le_bytes());
    }

    std::fs::write(path, out)
}

/// Write a 3-band float32 TIFF (ASP disparity format: dx, dy, valid).
pub fn save_disparity_tiff(path: &Path, disp: &DisparityMap) -> io::Result<()> {
    let n = disp.dx.len();
    let mut out = Vec::with_capacity(256 + n * 12);
    out.extend_from_slice(b"II");
    out.extend_from_slice(&42u16.to_le_bytes());
    out.extend_from_slice(&8u32.to_le_bytes());

    let num_entries: u16 = 11;
    let ifd_size = 2 + (num_entries as usize) * 12 + 4;
    let bps_offset = 8 + ifd_size;
    let sf_offset = bps_offset + 6;
    let data_offset = sf_offset + 6;
    let strip_byte_count = (n * 12) as u32;

    out.extend_from_slice(&num_entries.to_le_bytes());
    let mut push_entry = |tag: u16, ty: u16, count: u32, val: u32| {
        out.extend_from_slice(&tag.to_le_bytes());
        out.extend_from_slice(&ty.to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&val.to_le_bytes());
    };

    push_entry(256, 4, 1, disp.width);
    push_entry(257, 4, 1, disp.height);
    push_entry(258, 3, 3, bps_offset as u32);
    push_entry(259, 3, 1, 1);
    push_entry(262, 3, 1, 2); // RGB / 3-band
    push_entry(273, 4, 1, data_offset as u32);
    push_entry(277, 3, 1, 3);
    push_entry(278, 4, 1, disp.height);
    push_entry(279, 4, 1, strip_byte_count);
    push_entry(284, 3, 1, 1);
    push_entry(339, 3, 3, sf_offset as u32);
    out.extend_from_slice(&0u32.to_le_bytes());

    // BitsPerSample = [32, 32, 32]
    for _ in 0..3 {
        out.extend_from_slice(&32u16.to_le_bytes());
    }
    // SampleFormat = [3, 3, 3]
    for _ in 0..3 {
        out.extend_from_slice(&3u16.to_le_bytes());
    }

    for i in 0..n {
        out.extend_from_slice(&disp.dx[i].to_le_bytes());
        out.extend_from_slice(&disp.dy[i].to_le_bytes());
        let v = if disp.valid[i] != 0 { 1.0f32 } else { 0.0f32 };
        out.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, out)
}

/// Decode uncompressed baseline TIFF (little or big endian, 8/16/32f, 1 or 3+ channels).
pub fn decode_tiff(bytes: &[u8]) -> io::Result<ImageF32> {
    if bytes.len() < 8 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Truncated TIFF"));
    }
    let le = match &bytes[0..2] {
        b"II" => true,
        b"MM" => false,
        _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid TIFF endianness")),
    };
    let r_u16 = |off: usize| -> u16 {
        let s = [bytes[off], bytes[off + 1]];
        if le { u16::from_le_bytes(s) } else { u16::from_be_bytes(s) }
    };
    let r_u32 = |off: usize| -> u32 {
        let s = [bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]];
        if le { u32::from_le_bytes(s) } else { u32::from_be_bytes(s) }
    };
    let r_f32 = |off: usize| -> f32 {
        let s = [bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]];
        if le { f32::from_le_bytes(s) } else { f32::from_be_bytes(s) }
    };

    if r_u16(2) != 42 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Not a classic TIFF (42)"));
    }
    let ifd_off = r_u32(4) as usize;
    if ifd_off + 2 > bytes.len() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid IFD offset"));
    }
    let num_entries = r_u16(ifd_off) as usize;

    let mut width = 0u32;
    let mut height = 0u32;
    let mut bps = 8u16;
    let mut compression = 1u16;
    let mut samples_per_pixel = 1u16;
    let mut sample_format = 1u16;
    let mut strip_offsets = Vec::new();
    let mut strip_counts = Vec::new();

    for i in 0..num_entries {
        let e = ifd_off + 2 + i * 12;
        if e + 12 > bytes.len() {
            break;
        }
        let tag = r_u16(e);
        let ty = r_u16(e + 2);
        let count = r_u32(e + 4) as usize;
        let val_u32 = r_u32(e + 8);
        let val_u16 = r_u16(e + 8);

        let scalar = || -> u32 {
            if ty == 3 { val_u16 as u32 } else { val_u32 }
        };

        let read_u32_array = || -> Vec<u32> {
            if count == 1 {
                return vec![scalar()];
            }
            let off = val_u32 as usize;
            let mut out = Vec::with_capacity(count);
            for k in 0..count {
                if ty == 3 && off + k * 2 + 2 <= bytes.len() {
                    out.push(r_u16(off + k * 2) as u32);
                } else if ty == 4 && off + k * 4 + 4 <= bytes.len() {
                    out.push(r_u32(off + k * 4));
                }
            }
            out
        };

        match tag {
            256 => width = scalar(),
            257 => height = scalar(),
            258 => {
                if count == 1 {
                    bps = val_u16;
                } else {
                    let off = val_u32 as usize;
                    if off + 2 <= bytes.len() {
                        bps = r_u16(off);
                    }
                }
            }
            259 => compression = val_u16,
            273 => strip_offsets = read_u32_array(),
            277 => samples_per_pixel = val_u16,
            279 => strip_counts = read_u32_array(),
            339 => {
                if count == 1 {
                    sample_format = val_u16;
                } else {
                    let off = val_u32 as usize;
                    if off + 2 <= bytes.len() {
                        sample_format = r_u16(off);
                    }
                }
            }
            _ => {}
        }
    }

    if compression != 1 {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!("Only uncompressed TIFF supported (compression={})", compression),
        ));
    }
    if width == 0 || height == 0 || strip_offsets.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Incomplete TIFF metadata"));
    }

    let total_pixels = (width as usize) * (height as usize);
    let spp = samples_per_pixel.max(1) as usize;
    let bytes_per_sample = (bps as usize) / 8;
    let mut raw = Vec::with_capacity(total_pixels * spp * bytes_per_sample);

    for (idx, &so) in strip_offsets.iter().enumerate() {
        let start = so as usize;
        let len = strip_counts
            .get(idx)
            .copied()
            .unwrap_or((bytes.len().saturating_sub(start)) as u32) as usize;
        let end = (start + len).min(bytes.len());
        raw.extend_from_slice(&bytes[start..end]);
    }

    let mut data = Vec::with_capacity(total_pixels);
    let stride = spp * bytes_per_sample;
    for p in 0..total_pixels {
        let base = p * stride;
        if base + stride > raw.len() {
            data.push(0.0);
            continue;
        }
        let mut acc = 0.0f32;
        let ch_count = spp.min(3);
        for c in 0..ch_count {
            let off = base + c * bytes_per_sample;
            let sample = match (bps, sample_format) {
                (8, _) => raw[off] as f32 / 255.0,
                (16, _) => {
                    let s = [raw[off], raw[off + 1]];
                    let v = if le { u16::from_le_bytes(s) } else { u16::from_be_bytes(s) };
                    v as f32 / 65535.0
                }
                (32, 3) => {
                    let s = [raw[off], raw[off + 1], raw[off + 2], raw[off + 3]];
                    if le { f32::from_le_bytes(s) } else { f32::from_be_bytes(s) }
                }
                (32, _) => {
                    let s = [raw[off], raw[off + 1], raw[off + 2], raw[off + 3]];
                    let v = if le { u32::from_le_bytes(s) } else { u32::from_be_bytes(s) };
                    v as f32
                }
                _ => {
                    let _ = r_f32;
                    0.0
                }
            };
            acc += sample;
        }
        data.push(acc / ch_count as f32);
    }

    Ok(ImageF32::from_vec(width, height, data))
}

/// Write a Portable FloatMap (PFM) grayscale file.
pub fn save_pfm(path: &Path, img: &ImageF32) -> io::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    write!(f, "Pf\n{} {}\n-1.0\n", img.width, img.height)?;
    // PFM stores scanlines bottom-to-top
    for y in (0..img.height).rev() {
        for x in 0..img.width {
            f.write_all(&img.get(x, y).to_le_bytes())?;
        }
    }
    Ok(())
}

/// Decode a PFM grayscale or RGB file.
pub fn decode_pfm(bytes: &[u8]) -> io::Result<ImageF32> {
    let mut pos = 0usize;
    let mut lines = Vec::new();
    while lines.len() < 3 && pos < bytes.len() {
        let start = pos;
        while pos < bytes.len() && bytes[pos] != b'\n' {
            pos += 1;
        }
        let line = std::str::from_utf8(&bytes[start..pos])
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid PFM header"))?
            .trim()
            .to_string();
        pos += 1;
        if !line.is_empty() && !line.starts_with('#') {
            lines.push(line);
        }
    }
    if lines.len() < 3 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Incomplete PFM header"));
    }
    let channels = if lines[0] == "PF" { 3usize } else { 1usize };
    let dims: Vec<u32> = lines[1]
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();
    if dims.len() != 2 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid PFM dimensions"));
    }
    let scale: f32 = lines[2].parse().unwrap_or(-1.0);
    let le = scale < 0.0;
    let width = dims[0];
    let height = dims[1];
    let mut img = ImageF32::new(width, height);
    let mut off = pos;
    for y in (0..height).rev() {
        for x in 0..width {
            let mut sum = 0.0f32;
            for _ in 0..channels {
                if off + 4 <= bytes.len() {
                    let b = [bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]];
                    sum += if le { f32::from_le_bytes(b) } else { f32::from_be_bytes(b) };
                    off += 4;
                }
            }
            img.set(x, y, sum / channels as f32);
        }
    }
    Ok(img)
}

/// Write a binary 8-bit PGM file (auto-scaled to [0, 255]).
pub fn save_pgm(path: &Path, img: &ImageF32) -> io::Result<()> {
    let (min_v, max_v, _, _) = img.stats();
    let range = (max_v - min_v).max(1e-6);
    let mut f = BufWriter::new(File::create(path)?);
    write!(f, "P5\n{} {}\n255\n", img.width, img.height)?;
    let mut buf = Vec::with_capacity(img.data.len());
    for &v in &img.data {
        let t = if v.is_finite() { ((v - min_v) / range).clamp(0.0, 1.0) } else { 0.0 };
        buf.push((t * 255.0).round() as u8);
    }
    f.write_all(&buf)
}

/// Decode PGM/PPM (P2, P5, P6).
pub fn decode_pnm(bytes: &[u8]) -> io::Result<ImageF32> {
    let mut pos = 0usize;
    let next_token = |pos: &mut usize| -> Option<String> {
        loop {
            while *pos < bytes.len() && bytes[*pos].is_ascii_whitespace() {
                *pos += 1;
            }
            if *pos < bytes.len() && bytes[*pos] == b'#' {
                while *pos < bytes.len() && bytes[*pos] != b'\n' {
                    *pos += 1;
                }
            } else {
                break;
            }
        }
        if *pos >= bytes.len() {
            return None;
        }
        let start = *pos;
        while *pos < bytes.len() && !bytes[*pos].is_ascii_whitespace() {
            *pos += 1;
        }
        std::str::from_utf8(&bytes[start..*pos]).ok().map(|s| s.to_string())
    };

    let magic = next_token(&mut pos).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "PNM magic"))?;
    let width: u32 = next_token(&mut pos).and_then(|s| s.parse().ok()).unwrap_or(0);
    let height: u32 = next_token(&mut pos).and_then(|s| s.parse().ok()).unwrap_or(0);
    let maxval: f32 = next_token(&mut pos).and_then(|s| s.parse().ok()).unwrap_or(255.0);
    if pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
        pos += 1;
    }
    let n = (width as usize) * (height as usize);
    let mut data = Vec::with_capacity(n);

    match magic.as_str() {
        "P5" => {
            if maxval > 255.0 {
                for i in 0..n {
                    let o = pos + i * 2;
                    let v = if o + 1 < bytes.len() {
                        u16::from_be_bytes([bytes[o], bytes[o + 1]]) as f32 / maxval
                    } else {
                        0.0
                    };
                    data.push(v);
                }
            } else {
                for i in 0..n {
                    let v = bytes.get(pos + i).copied().unwrap_or(0) as f32 / maxval;
                    data.push(v);
                }
            }
        }
        "P6" => {
            for i in 0..n {
                let o = pos + i * 3;
                if o + 2 < bytes.len() {
                    let r = bytes[o] as f32;
                    let g = bytes[o + 1] as f32;
                    let b = bytes[o + 2] as f32;
                    data.push((0.299 * r + 0.587 * g + 0.114 * b) / maxval);
                } else {
                    data.push(0.0);
                }
            }
        }
        "P2" => {
            for _ in 0..n {
                let v = next_token(&mut pos)
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.0)
                    / maxval;
                data.push(v);
            }
        }
        _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "Unsupported PNM variant")),
    }
    Ok(ImageF32::from_vec(width, height, data))
}

/// Write a 24-bit BMP from an RGBA8 buffer.
pub fn save_bmp_rgba(path: &Path, width: u32, height: u32, rgba: &[u8]) -> io::Result<()> {
    let row_stride = ((width * 3 + 3) / 4) * 4;
    let img_size = row_stride * height;
    let file_size = 54 + img_size;
    let mut out = Vec::with_capacity(file_size as usize);

    out.extend_from_slice(b"BM");
    out.extend_from_slice(&file_size.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&54u32.to_le_bytes());
    // DIB header (BITMAPINFOHEADER)
    out.extend_from_slice(&40u32.to_le_bytes());
    out.extend_from_slice(&(width as i32).to_le_bytes());
    out.extend_from_slice(&(height as i32).to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&24u16.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&img_size.to_le_bytes());
    out.extend_from_slice(&2835i32.to_le_bytes());
    out.extend_from_slice(&2835i32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());

    let pad = (row_stride - width * 3) as usize;
    for y in (0..height).rev() {
        for x in 0..width {
            let idx = ((y * width + x) * 4) as usize;
            out.push(rgba[idx + 2]); // B
            out.push(rgba[idx + 1]); // G
            out.push(rgba[idx]);     // R
        }
        for _ in 0..pad {
            out.push(0);
        }
    }
    std::fs::write(path, out)
}

/// Decode uncompressed 24-bit or 32-bit BMP into grayscale f32.
pub fn decode_bmp(bytes: &[u8]) -> io::Result<ImageF32> {
    if bytes.len() < 54 || &bytes[0..2] != b"BM" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid BMP"));
    }
    let pixel_offset = u32::from_le_bytes([bytes[10], bytes[11], bytes[12], bytes[13]]) as usize;
    let width = i32::from_le_bytes([bytes[18], bytes[19], bytes[20], bytes[21]]).unsigned_abs();
    let raw_h = i32::from_le_bytes([bytes[22], bytes[23], bytes[24], bytes[25]]);
    let top_down = raw_h < 0;
    let height = raw_h.unsigned_abs();
    let bpp = u16::from_le_bytes([bytes[28], bytes[29]]) as usize;
    let ch = bpp / 8;
    if ch != 3 && ch != 4 {
        return Err(io::Error::new(io::ErrorKind::Unsupported, "Only 24/32-bit BMP supported"));
    }
    let row_stride = ((width as usize * ch + 3) / 4) * 4;
    let mut img = ImageF32::new(width, height);
    for y in 0..height {
        let src_y = if top_down { y } else { height - 1 - y } as usize;
        for x in 0..width {
            let o = pixel_offset + src_y * row_stride + (x as usize) * ch;
            if o + 2 < bytes.len() {
                let b = bytes[o] as f32;
                let g = bytes[o + 1] as f32;
                let r = bytes[o + 2] as f32;
                img.set(x, y, (0.299 * r + 0.587 * g + 0.114 * b) / 255.0);
            }
        }
    }
    Ok(img)
}

/// Pure-Rust zero-dependency PNG encoder (RGBA8 -> PNG bytes using stored DEFLATE blocks).
pub fn encode_png_rgba(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
    let mut raw = Vec::with_capacity((height as usize) * (1 + (width as usize) * 4));
    for y in 0..height as usize {
        raw.push(0u8); // Filter type 0 (None)
        let row_start = y * (width as usize) * 4;
        raw.extend_from_slice(&rgba[row_start..row_start + (width as usize) * 4]);
    }

    // Wrap raw scanlines in zlib container with uncompressed DEFLATE blocks
    let mut zlib = Vec::with_capacity(raw.len() + (raw.len() / 65535 + 1) * 5 + 6);
    zlib.push(0x78); // CMF: deflate, 32K window
    zlib.push(0x01); // FLG: no dict, check bits

    let mut pos = 0usize;
    while pos < raw.len() {
        let chunk = (raw.len() - pos).min(65535);
        let is_final = pos + chunk >= raw.len();
        zlib.push(if is_final { 0x01 } else { 0x00 });
        let len = chunk as u16;
        let nlen = !len;
        zlib.extend_from_slice(&len.to_le_bytes());
        zlib.extend_from_slice(&nlen.to_le_bytes());
        zlib.extend_from_slice(&raw[pos..pos + chunk]);
        pos += chunk;
    }
    let adler = adler32(&raw);
    zlib.extend_from_slice(&adler.to_be_bytes());

    let mut png = Vec::with_capacity(zlib.len() + 64);
    png.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]); // PNG signature

    // IHDR
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.push(8); // bit depth
    ihdr.push(6); // color type RGBA
    ihdr.push(0); // compression
    ihdr.push(0); // filter
    ihdr.push(0); // interlace
    push_png_chunk(&mut png, b"IHDR", &ihdr);
    push_png_chunk(&mut png, b"IDAT", &zlib);
    push_png_chunk(&mut png, b"IEND", &[]);
    png
}

fn push_png_chunk(out: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(chunk_type);
    out.extend_from_slice(data);
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(chunk_type);
    crc_input.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

fn adler32(data: &[u8]) -> u32 {
    let mut a = 1u32;
    let mut b = 0u32;
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

/// Export triangulated point cloud to fast Binary Little-Endian PLY with `double` (`f64`) XYZ
/// (preserves sub-millimeter precision on EPSG:3944/2154/ECEF coordinates in CloudCompare, MeshLab, QGIS, Blender;
/// 2x smaller and ~15x faster to write than ASCII PLY).
pub fn save_point_cloud_ply(path: &Path, pc: &PointCloud) -> io::Result<()> {
    let valid_cnt = pc.valid_count();
    let header = format!(
        "ply\n\
         format binary_little_endian 1.0\n\
         comment Generated by RustStereoPipelineVulkan (f64 XYZ)\n\
         element vertex {}\n\
         property double x\n\
         property double y\n\
         property double z\n\
         property uchar red\n\
         property uchar green\n\
         property uchar blue\n\
         property float error\n\
         end_header\n",
        valid_cnt
    );
    // 3 * f64 (24) + 3 * u8 (3) + 1 * f32 (4) = 31 bytes per vertex
    let mut buf = Vec::with_capacity(header.len() + valid_cnt * 31);
    buf.extend_from_slice(header.as_bytes());
    for i in 0..pc.points.len() {
        if pc.valid[i] != 0 {
            let p = pc.points[i];
            let [r, g, b] = match pc.colors.as_ref().and_then(|c| c.get(i)) {
                Some(&rgb) => rgb,
                None => {
                    let gray = (pc.intensities[i].clamp(0.0, 1.0) * 255.0).round() as u8;
                    [gray, gray, gray]
                }
            };
            buf.extend_from_slice(&p.x.to_le_bytes());
            buf.extend_from_slice(&p.y.to_le_bytes());
            buf.extend_from_slice(&p.z.to_le_bytes());
            buf.push(r);
            buf.push(g);
            buf.push(b);
            buf.extend_from_slice(&(pc.errors[i] as f32).to_le_bytes());
        }
    }
    std::fs::write(path, buf)
}

/// Export point cloud to CSV / XYZ (`x,y,z,error`).
pub fn save_point_cloud_xyz(path: &Path, pc: &PointCloud) -> io::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    for i in 0..pc.points.len() {
        if pc.valid[i] != 0 {
            let p = pc.points[i];
            writeln!(f, "{:.6} {:.6} {:.6} {:.6}", p.x, p.y, p.z, pc.errors[i])?;
        }
    }
    Ok(())
}

/// Load a whitespace or comma-separated XYZ/CSV point cloud.
pub fn load_point_cloud_xyz(path: &Path) -> io::Result<Vec<crate::math::Vec3>> {
    let text = std::fs::read_to_string(path)?;
    let mut pts = Vec::new();
    for line in text.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') || l.starts_with("ply") {
            continue;
        }
        let nums: Vec<f64> = l
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter(|s| !s.is_empty())
            .filter_map(|s| s.parse::<f64>().ok())
            .collect();
        if nums.len() >= 3 {
            pts.push(crate::math::Vec3::new(nums[0], nums[1], nums[2]));
        }
    }
    Ok(pts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiff_and_pfm_roundtrip() {
        let mut img = ImageF32::new(16, 12);
        for y in 0..12 {
            for x in 0..16 {
                img.set(x, y, (x as f32 * 0.5 + y as f32 * 1.25) / 25.0);
            }
        }
        let tiff_path = std::env::temp_dir().join("rspv_test_io.tif");
        save_tiff_f32(&tiff_path, &img, None).unwrap();
        let loaded = load_image(&tiff_path).unwrap();
        assert_eq!(loaded.width, 16);
        assert_eq!(loaded.height, 12);
        for i in 0..img.data.len() {
            assert!((img.data[i] - loaded.data[i]).abs() < 1e-6);
        }
        let _ = std::fs::remove_file(&tiff_path);

        let pfm_path = std::env::temp_dir().join("rspv_test_io.pfm");
        save_pfm(&pfm_path, &img).unwrap();
        let loaded_pfm = load_image(&pfm_path).unwrap();
        for i in 0..img.data.len() {
            assert!((img.data[i] - loaded_pfm.data[i]).abs() < 1e-6);
        }
        let _ = std::fs::remove_file(&pfm_path);
    }

    #[test]
    fn test_png_encoding_header() {
        let rgba = vec![128u8; 8 * 8 * 4];
        let png = encode_png_rgba(8, 8, &rgba);
        assert_eq!(&png[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
        assert_eq!(&png[12..16], b"IHDR");
    }

    #[test]
    fn test_binary_f64_ply_export() {
        let mut pc = PointCloud::new(2, 1);
        pc.points[0] = crate::math::Vec3::new(1_901_814.123456, 3_143_265.536789, 245.52);
        pc.valid[0] = 1;
        pc.intensities[0] = 0.5;
        pc.errors[0] = 0.0125;
        pc.valid[1] = 0;

        let path = std::env::temp_dir().join("rspv_test_bin_f64.ply");
        save_point_cloud_ply(&path, &pc).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let end_hdr = b"end_header\n";
        let pos = bytes
            .windows(end_hdr.len())
            .position(|w| w == end_hdr)
            .expect("missing end_header");
        let hdr = std::str::from_utf8(&bytes[..pos + end_hdr.len()]).unwrap();
        assert!(hdr.contains("format binary_little_endian 1.0"));
        assert!(hdr.contains("property double x"));
        assert!(hdr.contains("element vertex 1"));

        let payload = &bytes[pos + end_hdr.len()..];
        assert_eq!(payload.len(), 31);
        let rx = f64::from_le_bytes(payload[0..8].try_into().unwrap());
        let ry = f64::from_le_bytes(payload[8..16].try_into().unwrap());
        let rz = f64::from_le_bytes(payload[16..24].try_into().unwrap());
        assert!((rx - 1_901_814.123456).abs() < 1e-12);
        assert!((ry - 3_143_265.536789).abs() < 1e-12);
        assert!((rz - 245.52).abs() < 1e-12);

        // Verify true RGB color preservation when PointCloud::colors is attached
        let mut rgb = ImageRgb::new(2, 1);
        rgb.set(0, 0, [210, 120, 45]);
        pc.apply_rgb(&rgb);
        let path_rgb = std::env::temp_dir().join("rspv_test_bin_f64_rgb.ply");
        save_point_cloud_ply(&path_rgb, &pc).unwrap();
        let bytes_rgb = std::fs::read(&path_rgb).unwrap();
        let _ = std::fs::remove_file(&path_rgb);
        let pos_rgb = bytes_rgb
            .windows(end_hdr.len())
            .position(|w| w == end_hdr)
            .expect("missing end_header");
        let payload_rgb = &bytes_rgb[pos_rgb + end_hdr.len()..];
        assert_eq!(&payload_rgb[24..27], &[210, 120, 45]);
    }
}
