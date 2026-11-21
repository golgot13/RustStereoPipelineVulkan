//! Pure-Rust 2D UI & 3D Terrain software framebuffer renderer (BGRA8 for direct X11 XPutImage).
//! Includes an embedded 8x8 ASCII bitmap font, UI widgets, image blitting, and 3D terrain projection.

use crate::image::{Colormap, ImageF32};

/// Packed 0x00RRGGBB color (native little-endian BGRA byte order for 24/32-bit X11 TrueColor).
#[inline]
pub const fn rgb(r: u8, g: u8, b: u8) -> u32 {
    ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

pub const COL_BG: u32 = rgb(14, 17, 23);
pub const COL_PANEL: u32 = rgb(22, 27, 36);
pub const COL_PANEL_ALT: u32 = rgb(30, 37, 48);
pub const COL_BORDER: u32 = rgb(48, 58, 74);
pub const COL_ACCENT: u32 = rgb(56, 152, 255);
pub const COL_ACCENT_HI: u32 = rgb(92, 178, 255);
pub const COL_GREEN: u32 = rgb(46, 204, 113);
pub const COL_AMBER: u32 = rgb(245, 176, 65);
pub const COL_TEXT: u32 = rgb(232, 238, 245);
pub const COL_TEXT_DIM: u32 = rgb(148, 162, 180);

#[derive(Debug, Clone, Copy)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

impl Rect {
    pub const fn new(x: i32, y: i32, w: i32, h: i32) -> Self {
        Self { x, y, w, h }
    }

    #[inline]
    pub fn contains(&self, px: i32, py: i32) -> bool {
        px >= self.x && px < self.x + self.w && py >= self.y && py < self.y + self.h
    }
}

pub struct Framebuffer {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u32>,
}

impl Framebuffer {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![COL_BG; (width as usize) * (height as usize)],
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if self.width != width || self.height != height {
            self.width = width.max(1);
            self.height = height.max(1);
            self.pixels.resize((self.width as usize) * (self.height as usize), COL_BG);
        }
    }

    pub fn clear(&mut self, color: u32) {
        self.pixels.fill(color);
    }

    #[inline]
    pub fn put_pixel(&mut self, x: i32, y: i32, color: u32) {
        if x >= 0 && y >= 0 && (x as u32) < self.width && (y as u32) < self.height {
            self.pixels[(y as usize) * (self.width as usize) + (x as usize)] = color;
        }
    }

    pub fn fill_rect(&mut self, r: Rect, color: u32) {
        let x0 = r.x.max(0);
        let y0 = r.y.max(0);
        let x1 = (r.x + r.w).min(self.width as i32);
        let y1 = (r.y + r.h).min(self.height as i32);
        for y in y0..y1 {
            let row = (y as usize) * (self.width as usize);
            for x in x0..x1 {
                self.pixels[row + x as usize] = color;
            }
        }
    }

    pub fn stroke_rect(&mut self, r: Rect, color: u32) {
        for x in r.x..(r.x + r.w) {
            self.put_pixel(x, r.y, color);
            self.put_pixel(x, r.y + r.h - 1, color);
        }
        for y in r.y..(r.y + r.h) {
            self.put_pixel(r.x, y, color);
            self.put_pixel(r.x + r.w - 1, y, color);
        }
    }

    pub fn draw_line(&mut self, mut x0: i32, mut y0: i32, x1: i32, y1: i32, color: u32) {
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        loop {
            self.put_pixel(x0, y0, color);
            if x0 == x1 && y0 == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x0 += sx;
            }
            if e2 <= dx {
                err += dx;
                y0 += sy;
            }
        }
    }

    /// Draw crisp bitmap text using embedded 8x8 pixel glyphs.
    pub fn draw_text(&mut self, x: i32, y: i32, text: &str, color: u32, scale: i32) {
        let s = scale.max(1);
        let mut cx = x;
        for ch in text.chars() {
            if ch == '\n' {
                cx = x;
                continue;
            }
            let glyph = glyph_8x8(ch);
            for (row, &bits) in glyph.iter().enumerate() {
                for col in 0..8 {
                    if (bits & (1 << (7 - col))) != 0 {
                        for sy in 0..s {
                            for sx in 0..s {
                                self.put_pixel(cx + (col as i32) * s + sx, y + (row as i32) * s + sy, color);
                            }
                        }
                    }
                }
            }
            cx += 8 * s;
        }
    }

    /// Draw a clickable UI button and return whether it is hovered.
    pub fn draw_button(&mut self, r: Rect, label: &str, active: bool, hovered: bool, primary: bool) {
        let bg = if primary {
            if hovered { COL_ACCENT_HI } else { COL_ACCENT }
        } else if active {
            rgb(38, 78, 128)
        } else if hovered {
            rgb(40, 49, 64)
        } else {
            COL_PANEL_ALT
        };
        let border = if active || primary { COL_ACCENT_HI } else { COL_BORDER };
        self.fill_rect(r, bg);
        self.stroke_rect(r, border);
        let text_w = (label.len() as i32) * 8;
        let tx = r.x + ((r.w - text_w) / 2).max(4);
        let ty = r.y + (r.h - 8) / 2;
        self.draw_text(tx, ty, label, COL_TEXT, 1);
    }

    /// Blit an `ImageF32` into `viewport` using a scientific colormap, zoom factor, and pan offset.
    pub fn blit_image_f32(
        &mut self,
        viewport: Rect,
        img: &ImageF32,
        cmap: Colormap,
        zoom: f32,
        pan_x: f32,
        pan_y: f32,
    ) {
        self.fill_rect(viewport, rgb(10, 12, 16));
        let (min_v, max_v, _, _) = img.stats();
        let range = (max_v - min_v).max(1e-6);

        let fit_scale = ((viewport.w as f32) / (img.width.max(1) as f32))
            .min((viewport.h as f32) / (img.height.max(1) as f32))
            * 0.94
            * zoom.clamp(0.25, 12.0);
        let draw_w = img.width as f32 * fit_scale;
        let draw_h = img.height as f32 * fit_scale;
        let ox = viewport.x as f32 + (viewport.w as f32 - draw_w) * 0.5 + pan_x;
        let oy = viewport.y as f32 + (viewport.h as f32 - draw_h) * 0.5 + pan_y;

        let x0 = (ox.floor() as i32).max(viewport.x);
        let y0 = (oy.floor() as i32).max(viewport.y);
        let x1 = ((ox + draw_w).ceil() as i32).min(viewport.x + viewport.w);
        let y1 = ((oy + draw_h).ceil() as i32).min(viewport.y + viewport.h);

        for py in y0..y1 {
            let iy = (((py as f32 - oy) / fit_scale).floor() as i32).clamp(0, img.height as i32 - 1) as u32;
            let row = (py as usize) * (self.width as usize);
            for px in x0..x1 {
                let ix = (((px as f32 - ox) / fit_scale).floor() as i32).clamp(0, img.width as i32 - 1) as u32;
                let v = img.get(ix, iy);
                let col = if !v.is_finite() {
                    rgb(22, 26, 34)
                } else {
                    let t = ((v - min_v) / range).clamp(0.0, 1.0);
                    let [r, g, b, _] = cmap.sample(t);
                    rgb(r, g, b)
                };
                self.pixels[row + px as usize] = col;
            }
        }

        // Draw colorbar legend at bottom-right of viewport
        let bar = Rect::new(viewport.x + viewport.w - 190, viewport.y + viewport.h - 32, 170, 12);
        for i in 0..bar.w {
            let t = i as f32 / (bar.w - 1).max(1) as f32;
            let [r, g, b, _] = cmap.sample(t);
            let c = rgb(r, g, b);
            for y in bar.y..(bar.y + bar.h) {
                self.put_pixel(bar.x + i, y, c);
            }
        }
        self.stroke_rect(bar, COL_BORDER);
        let legend = format!("{:.1} .. {:.1}", min_v, max_v);
        self.draw_text(bar.x, bar.y - 12, &legend, COL_TEXT, 1);
    }

    /// Render an interactive 3D perspective wireframe/elevation-colored terrain mesh of a DEM.
    pub fn render_dem_3d(
        &mut self,
        viewport: Rect,
        dem: &ImageF32,
        cmap: Colormap,
        yaw: f32,
        pitch: f32,
        zoom: f32,
    ) {
        self.fill_rect(viewport, rgb(10, 13, 19));
        let (min_z, max_z, _, _) = dem.stats();
        let z_range = (max_z - min_z).max(1e-4);

        let step_x = (dem.width / 56).max(1);
        let step_y = (dem.height / 56).max(1);
        let gx_n = ((dem.width + step_x - 1) / step_x) as usize;
        let gy_n = ((dem.height + step_y - 1) / step_y) as usize;

        let (sy, cy) = yaw.sin_cos();
        let (sp, cp) = pitch.sin_cos();
        let scale = (viewport.w.min(viewport.h) as f32) * 0.42 * zoom.clamp(0.4, 4.0);
        let cx = viewport.x as f32 + viewport.w as f32 * 0.5;
        let cy_s = viewport.y as f32 + viewport.h as f32 * 0.56;

        let mut projected: Vec<Option<(i32, i32, u32)>> = vec![None; gx_n * gy_n];
        for j in 0..gy_n {
            let y = (j as u32 * step_y).min(dem.height - 1);
            let ny = (y as f32 / dem.height.max(1) as f32) * 2.0 - 1.0;
            for i in 0..gx_n {
                let x = (i as u32 * step_x).min(dem.width - 1);
                let v = dem.get(x, y);
                if !v.is_finite() {
                    continue;
                }
                let nx = (x as f32 / dem.width.max(1) as f32) * 2.0 - 1.0;
                let t = ((v - min_z) / z_range).clamp(0.0, 1.0);
                let nz = (t - 0.5) * 0.65;

                // Rotate yaw around Z, pitch around X
                let rx = nx * cy - ny * sy;
                let ry = nx * sy + ny * cy;
                let py3 = ry * cp - nz * sp;
                let pz3 = ry * sp + nz * cp;
                let depth = 3.2 - py3;
                if depth < 0.3 {
                    continue;
                }
                let persp = scale / depth * 2.3;
                let sx_i = (cx + rx * persp) as i32;
                let sy_i = (cy_s - pz3 * persp) as i32;
                let [r, g, b, _] = cmap.sample(t);
                projected[j * gx_n + i] = Some((sx_i, sy_i, rgb(r, g, b)));
            }
        }

        // Draw grid lines clipped to viewport
        for j in 0..gy_n {
            for i in 0..gx_n {
                if let Some((x0, y0, c0)) = projected[j * gx_n + i] {
                    if !viewport.contains(x0, y0) {
                        continue;
                    }
                    if i + 1 < gx_n {
                        if let Some((x1, y1, _)) = projected[j * gx_n + (i + 1)] {
                            if viewport.contains(x1, y1) {
                                self.draw_line(x0, y0, x1, y1, c0);
                            }
                        }
                    }
                    if j + 1 < gy_n {
                        if let Some((x1, y1, _)) = projected[(j + 1) * gx_n + i] {
                            if viewport.contains(x1, y1) {
                                self.draw_line(x0, y0, x1, y1, c0);
                            }
                        }
                    }
                }
            }
        }
        self.draw_text(
            viewport.x + 12,
            viewport.y + viewport.h - 20,
            "Drag mouse to orbit 3D DEM | Scroll to zoom",
            COL_TEXT_DIM,
            1,
        );
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self.pixels.as_ptr() as *const u8, self.pixels.len() * 4)
        }
    }
}

/// Compact 8x8 ASCII bitmap font (printable chars 32..=126).
fn glyph_8x8(ch: char) -> [u8; 8] {
    match ch {
        ' ' => [0x00; 8],
        '!' => [0x18, 0x18, 0x18, 0x18, 0x18, 0x00, 0x18, 0x00],
        '#' => [0x24, 0x7E, 0x24, 0x24, 0x7E, 0x24, 0x00, 0x00],
        '%' => [0x62, 0x64, 0x08, 0x10, 0x26, 0x46, 0x00, 0x00],
        '(' => [0x0C, 0x18, 0x30, 0x30, 0x30, 0x18, 0x0C, 0x00],
        ')' => [0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x18, 0x30, 0x00],
        '+' => [0x00, 0x18, 0x18, 0x7E, 0x18, 0x18, 0x00, 0x00],
        ',' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x30],
        '-' => [0x00, 0x00, 0x00, 0x7E, 0x00, 0x00, 0x00, 0x00],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00],
        '/' => [0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x00, 0x00],
        '0' => [0x3C, 0x66, 0x6E, 0x76, 0x66, 0x66, 0x3C, 0x00],
        '1' => [0x18, 0x38, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00],
        '2' => [0x3C, 0x66, 0x06, 0x1C, 0x30, 0x60, 0x7E, 0x00],
        '3' => [0x3C, 0x66, 0x06, 0x1C, 0x06, 0x66, 0x3C, 0x00],
        '4' => [0x0C, 0x1C, 0x3C, 0x6C, 0x7E, 0x0C, 0x0C, 0x00],
        '5' => [0x7E, 0x60, 0x7C, 0x06, 0x06, 0x66, 0x3C, 0x00],
        '6' => [0x3C, 0x60, 0x7C, 0x66, 0x66, 0x66, 0x3C, 0x00],
        '7' => [0x7E, 0x06, 0x0C, 0x18, 0x30, 0x30, 0x30, 0x00],
        '8' => [0x3C, 0x66, 0x66, 0x3C, 0x66, 0x66, 0x3C, 0x00],
        '9' => [0x3C, 0x66, 0x66, 0x3E, 0x06, 0x0C, 0x38, 0x00],
        ':' => [0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x00, 0x00],
        '<' => [0x06, 0x18, 0x60, 0x60, 0x18, 0x06, 0x00, 0x00],
        '=' => [0x00, 0x00, 0x7E, 0x00, 0x7E, 0x00, 0x00, 0x00],
        '>' => [0x60, 0x18, 0x06, 0x06, 0x18, 0x60, 0x00, 0x00],
        '[' => [0x3C, 0x30, 0x30, 0x30, 0x30, 0x30, 0x3C, 0x00],
        ']' => [0x3C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x3C, 0x00],
        '_' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7E, 0x00],
        '|' => [0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00],
        'A' | 'a' => [0x18, 0x3C, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x00],
        'B' | 'b' => [0x7C, 0x66, 0x66, 0x7C, 0x66, 0x66, 0x7C, 0x00],
        'C' | 'c' => [0x3C, 0x66, 0x60, 0x60, 0x60, 0x66, 0x3C, 0x00],
        'D' | 'd' => [0x78, 0x6C, 0x66, 0x66, 0x66, 0x6C, 0x78, 0x00],
        'E' | 'e' => [0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x7E, 0x00],
        'F' | 'f' => [0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x60, 0x00],
        'G' | 'g' => [0x3C, 0x66, 0x60, 0x6E, 0x66, 0x66, 0x3E, 0x00],
        'H' | 'h' => [0x66, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x66, 0x00],
        'I' | 'i' => [0x3C, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C, 0x00],
        'J' | 'j' => [0x1E, 0x0C, 0x0C, 0x0C, 0x6C, 0x6C, 0x38, 0x00],
        'K' | 'k' => [0x66, 0x6C, 0x78, 0x70, 0x78, 0x6C, 0x66, 0x00],
        'L' | 'l' => [0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x7E, 0x00],
        'M' | 'm' => [0x63, 0x77, 0x7F, 0x6B, 0x63, 0x63, 0x63, 0x00],
        'N' | 'n' => [0x66, 0x76, 0x7E, 0x7E, 0x6E, 0x66, 0x66, 0x00],
        'O' | 'o' => [0x3C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00],
        'P' | 'p' => [0x7C, 0x66, 0x66, 0x7C, 0x60, 0x60, 0x60, 0x00],
        'Q' | 'q' => [0x3C, 0x66, 0x66, 0x66, 0x6A, 0x6C, 0x36, 0x00],
        'R' | 'r' => [0x7C, 0x66, 0x66, 0x7C, 0x6C, 0x66, 0x66, 0x00],
        'S' | 's' => [0x3C, 0x66, 0x60, 0x3C, 0x06, 0x66, 0x3C, 0x00],
        'T' | 't' => [0x7E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00],
        'U' | 'u' => [0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00],
        'V' | 'v' => [0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x18, 0x00],
        'W' | 'w' => [0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00],
        'X' | 'x' => [0x66, 0x66, 0x3C, 0x18, 0x3C, 0x66, 0x66, 0x00],
        'Y' | 'y' => [0x66, 0x66, 0x66, 0x3C, 0x18, 0x18, 0x18, 0x00],
        'Z' | 'z' => [0x7E, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x7E, 0x00],
        _ => [0x3C, 0x42, 0x04, 0x18, 0x18, 0x00, 0x18, 0x00],
    }
}
