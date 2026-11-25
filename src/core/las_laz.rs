//! Zero-dependency pure-Rust LAS 1.0–1.4 + LAZ v2/v3 + COPC 1.0 point cloud reader.
//!
//! Designed for fast, `f64`-exact IGN LiDAR HD (`.copc.laz`, `.laz`, `.las`) ingestion:
//! - Instant 227–375 byte Public Header & VLR parsing (LAS 1.0 through 1.4).
//! - Exact LASzip arithmetic range coder (`ArithmeticModel`, `ArithmeticBitModel`,
//!   `ArithmeticDecoder`, `IntegerDecompressor`) and compressed ChunkTable reader.
//! - Zero-copy LASzip v3/v4 layered chunk selective decoder (`Point14` / PDRF 6–10 + COPC 1.0):
//!   decodes ONLY `XY + Z + Classification` layers and skips unused layers (Flags, Intensity,
//!   ScanAngle, UserData, PointSourceId, GpsTime, RGB14, RGBNIR14, WavePacket14, Byte14).
//! - LASzip v2 sequential chunk decoder (`Point10` / PDRF 0–3 + GpsTime + RGB12 + ExtraBytes)
//!   for legacy LAS 1.0–1.3 `.laz` archives.

use std::io::{self, Read, Seek, SeekFrom};

// ============================================================================
// Little-Endian Binary Helpers
// ============================================================================

#[inline(always)]
fn read_u8<R: Read>(r: &mut R) -> io::Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

#[inline(always)]
fn read_u32_le<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

#[inline(always)]
fn read_i64_le<R: Read>(r: &mut R) -> io::Result<i64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(i64::from_le_bytes(b))
}

#[inline(always)]
fn le_u16(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

#[inline(always)]
fn le_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

#[inline(always)]
fn le_i32(b: &[u8], off: usize) -> i32 {
    i32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

#[inline(always)]
fn le_u64(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes([
        b[off],
        b[off + 1],
        b[off + 2],
        b[off + 3],
        b[off + 4],
        b[off + 5],
        b[off + 6],
        b[off + 7],
    ])
}

#[inline(always)]
fn le_i64(b: &[u8], off: usize) -> i64 {
    i64::from_le_bytes([
        b[off],
        b[off + 1],
        b[off + 2],
        b[off + 3],
        b[off + 4],
        b[off + 5],
        b[off + 6],
        b[off + 7],
    ])
}

#[inline(always)]
fn le_f64(b: &[u8], off: usize) -> f64 {
    f64::from_bits(le_u64(b, off))
}

// ============================================================================
// LAS 1.0–1.4 Public Header & VLR Parser
// ============================================================================

#[derive(Debug, Clone)]
pub struct LasHeader {
    pub version_major: u8,
    pub version_minor: u8,
    pub header_size: u16,
    pub offset_to_point_data: u32,
    pub number_of_vlrs: u32,
    pub point_data_record_format: u8,
    pub point_data_record_length: u16,
    pub legacy_number_of_point_records: u32,
    pub extended_number_of_point_records: Option<u64>,
    pub x_scale_factor: f64,
    pub y_scale_factor: f64,
    pub z_scale_factor: f64,
    pub x_offset: f64,
    pub y_offset: f64,
    pub z_offset: f64,
    pub max_x: f64,
    pub min_x: f64,
    pub max_y: f64,
    pub min_y: f64,
    pub max_z: f64,
    pub min_z: f64,
}

impl LasHeader {
    /// Read LAS 1.0–1.4 Public Header Block and advance reader to `header_size` (start of VLRs).
    pub fn read_from<R: Read>(src: &mut R) -> io::Result<Self> {
        let mut base = [0u8; 227];
        src.read_exact(&mut base)?;
        if &base[0..4] != b"LASF" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid LAS signature (expected 'LASF')",
            ));
        }

        let version_major = base[24];
        let version_minor = base[25];
        let header_size = le_u16(&base, 94);
        let offset_to_point_data = le_u32(&base, 96);
        let number_of_vlrs = le_u32(&base, 100);
        let point_data_record_format = base[104];
        let point_data_record_length = le_u16(&base, 105);
        let legacy_number_of_point_records = le_u32(&base, 107);

        let x_scale_factor = le_f64(&base, 131);
        let y_scale_factor = le_f64(&base, 139);
        let z_scale_factor = le_f64(&base, 147);
        let x_offset = le_f64(&base, 155);
        let y_offset = le_f64(&base, 163);
        let z_offset = le_f64(&base, 171);
        let max_x = le_f64(&base, 179);
        let min_x = le_f64(&base, 187);
        let max_y = le_f64(&base, 195);
        let min_y = le_f64(&base, 203);
        let max_z = le_f64(&base, 211);
        let min_z = le_f64(&base, 219);

        let mut extended_number_of_point_records = None;
        if header_size > 227 {
            let extra_len = (header_size - 227) as usize;
            let mut ext = vec![0u8; extra_len];
            src.read_exact(&mut ext)?;
            // LAS 1.4: offset 247..255 in file = offset 20..28 in ext
            if version_major == 1 && version_minor >= 4 && extra_len >= 28 {
                extended_number_of_point_records = Some(le_u64(&ext, 20));
            }
        }

        Ok(Self {
            version_major,
            version_minor,
            header_size,
            offset_to_point_data,
            number_of_vlrs,
            point_data_record_format,
            point_data_record_length,
            legacy_number_of_point_records,
            extended_number_of_point_records,
            x_scale_factor,
            y_scale_factor,
            z_scale_factor,
            x_offset,
            y_offset,
            z_offset,
            max_x,
            min_x,
            max_y,
            min_y,
            max_z,
            min_z,
        })
    }

    #[inline]
    pub fn is_compressed(&self) -> bool {
        (self.point_data_record_format & 0x80) != 0
    }

    #[inline]
    pub fn point_format_id(&self) -> u8 {
        self.point_data_record_format & 0x3F
    }

    #[inline]
    pub fn total_points(&self) -> u64 {
        match self.extended_number_of_point_records {
            Some(ext) if ext > 0 => ext,
            _ => self.legacy_number_of_point_records as u64,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LasVlr {
    pub user_id: [u8; 16],
    pub record_id: u16,
    pub data: Vec<u8>,
}

impl LasVlr {
    pub fn read_from<R: Read>(src: &mut R) -> io::Result<Self> {
        let mut hdr = [0u8; 54];
        src.read_exact(&mut hdr)?;
        let mut user_id = [0u8; 16];
        user_id.copy_from_slice(&hdr[2..18]);
        let record_id = le_u16(&hdr, 18);
        let record_len = le_u16(&hdr, 20) as usize;
        let mut data = vec![0u8; record_len];
        src.read_exact(&mut data)?;
        Ok(Self {
            user_id,
            record_id,
            data,
        })
    }

    pub fn user_id_str(&self) -> &str {
        let len = self
            .user_id
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(self.user_id.len());
        std::str::from_utf8(&self.user_id[..len]).unwrap_or("")
    }
}

// ============================================================================
// LASzip VLR & ChunkTable
// ============================================================================

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum CompressorType {
    None,
    PointWise,
    PointWiseChunked,
    LayeredChunked,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum LazItemType {
    Byte(u16),
    Point10,
    GpsTime,
    Rgb12,
    WavePacket13,
    Point14,
    Rgb14,
    RgbNir14,
    WavePacket14,
    Byte14(u16),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct LazItem {
    pub item_type: LazItemType,
    pub size: u16,
    pub version: u16,
}

#[derive(Debug, Clone)]
pub struct LazVlr {
    pub compressor: CompressorType,
    pub chunk_size: u32,
    pub items: Vec<LazItem>,
}

impl LazVlr {
    pub const VARIABLE_CHUNK_SIZE: u32 = u32::MAX;

    pub fn parse(data: &[u8]) -> io::Result<Self> {
        if data.len() < 34 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "LAZ VLR too short",
            ));
        }
        let compressor = match le_u16(data, 0) {
            0 => CompressorType::None,
            1 => CompressorType::PointWise,
            2 => CompressorType::PointWiseChunked,
            3 => CompressorType::LayeredChunked,
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Unknown LAZ compressor type {}", other),
                ))
            }
        };
        let raw_chunk = le_u32(data, 12);
        let chunk_size = if raw_chunk == 0 {
            Self::VARIABLE_CHUNK_SIZE
        } else {
            raw_chunk
        };
        let num_items = le_u16(data, 32) as usize;
        if data.len() < 34 + num_items * 6 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "LAZ VLR truncated items",
            ));
        }
        let mut items = Vec::with_capacity(num_items);
        for i in 0..num_items {
            let off = 34 + i * 6;
            let raw_type = le_u16(data, off);
            let size = le_u16(data, off + 2);
            let version = le_u16(data, off + 4);
            let item_type = match raw_type {
                0 => LazItemType::Byte(size),
                6 => LazItemType::Point10,
                7 => LazItemType::GpsTime,
                8 => LazItemType::Rgb12,
                9 => LazItemType::WavePacket13,
                10 => LazItemType::Point14,
                11 => LazItemType::Rgb14,
                12 => LazItemType::RgbNir14,
                13 => LazItemType::WavePacket14,
                14 => LazItemType::Byte14(size),
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Unknown LAZ item type {}", other),
                    ))
                }
            };
            items.push(LazItem {
                item_type,
                size,
                version,
            });
        }
        Ok(Self {
            compressor,
            chunk_size,
            items,
        })
    }

    #[inline]
    pub fn uses_variable_size_chunks(&self) -> bool {
        self.chunk_size == Self::VARIABLE_CHUNK_SIZE
    }

    #[inline]
    pub fn items_size(&self) -> usize {
        self.items.iter().map(|i| i.size as usize).sum()
    }
}

#[derive(Debug, Copy, Clone)]
pub struct ChunkEntry {
    pub file_offset: u64,
    pub byte_count: u64,
    pub point_count: u64,
}

/// Read the LASzip chunk table from a seekable LAZ stream positioned at `point_data_offset`.
pub fn read_chunk_table<R: Read + Seek>(
    src: &mut R,
    point_data_offset: u64,
    vlr: &LazVlr,
    total_points: u64,
) -> io::Result<Vec<ChunkEntry>> {
    if vlr.compressor == CompressorType::PointWise {
        let end = src.seek(SeekFrom::End(0))?;
        let byte_count = end.saturating_sub(point_data_offset);
        return Ok(vec![ChunkEntry {
            file_offset: point_data_offset,
            byte_count,
            point_count: total_points,
        }]);
    }

    src.seek(SeekFrom::Start(point_data_offset))?;
    let mut table_off = read_i64_le(src)?;
    if table_off <= point_data_offset as i64 {
        // Writer could not update the header offset; fallback to last 8 bytes of file
        src.seek(SeekFrom::End(-8))?;
        table_off = read_i64_le(src)?;
        if table_off <= point_data_offset as i64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Missing LAZ chunk table offset",
            ));
        }
    }

    src.seek(SeekFrom::Start(table_off as u64))?;
    let _version = read_u32_le(src)?;
    let num_chunks = read_u32_le(src)? as usize;
    if num_chunks == 0 {
        return Ok(Vec::new());
    }

    let variable = vlr.uses_variable_size_chunks();
    let mut ic = IntegerDecompressor::new(32, 2);
    let mut dec = ArithmeticDecoder::new(src);
    dec.read_init_bytes()?;

    let mut entries = Vec::with_capacity(num_chunks);
    let mut prev_pts = 0i32;
    let mut prev_bytes = 0i32;
    let mut cur_off = point_data_offset + 8;
    let mut remaining_pts = total_points;

    for _ in 0..num_chunks {
        let pt_count = if variable {
            let v = ic.decompress(&mut dec, prev_pts, 0)?;
            prev_pts = v;
            v as u32 as u64
        } else {
            let cs = (vlr.chunk_size as u64).min(remaining_pts);
            remaining_pts = remaining_pts.saturating_sub(cs);
            cs
        };
        let byte_count = {
            let v = ic.decompress(&mut dec, prev_bytes, 1)?;
            prev_bytes = v;
            v as u32 as u64
        };
        entries.push(ChunkEntry {
            file_offset: cur_off,
            byte_count,
            point_count: pt_count,
        });
        cur_off = cur_off.saturating_add(byte_count);
    }

    Ok(entries)
}

// ============================================================================
// Arithmetic Range Decoder & Integer Decompressor (Exact LASzip Algorithm)
// ============================================================================

const DM_LENGTH_SHIFT: u32 = 15;
const DM_MAX_COUNT: u32 = 1 << DM_LENGTH_SHIFT;
const BM_LENGTH_SHIFT: u32 = 13;
const BM_MAX_COUNT: u32 = 1 << BM_LENGTH_SHIFT;
const AC_MAX_LENGTH: u32 = 0xFFFF_FFFF;
const AC_MIN_LENGTH: u32 = 0x0100_0000;

#[derive(Clone)]
struct ArithmeticModel {
    symbols: u32,
    distribution: Vec<u32>,
    symbol_count: Vec<u32>,
    decoder_table: Vec<u32>,
    total_count: u32,
    update_cycle: u32,
    symbols_until_update: u32,
    last_symbol: u32,
    table_size: u32,
    table_shift: u32,
}

impl ArithmeticModel {
    fn new(symbols: u32) -> Self {
        let mut table_size = 0u32;
        let mut table_shift = 0u32;
        let mut decoder_table = Vec::new();
        if symbols > 16 {
            let mut table_bits = 3u32;
            while symbols > (1u32 << (table_bits + 2)) {
                table_bits += 1;
            }
            table_size = 1 << table_bits;
            table_shift = DM_LENGTH_SHIFT - table_bits;
            decoder_table = vec![0u32; (table_size + 2) as usize];
        }
        let mut m = Self {
            symbols,
            distribution: vec![0u32; symbols as usize],
            symbol_count: vec![1u32; symbols as usize],
            decoder_table,
            total_count: 0,
            update_cycle: symbols,
            symbols_until_update: 0,
            last_symbol: symbols - 1,
            table_size,
            table_shift,
        };
        m.update();
        let init_cycle = (symbols + 6) >> 1;
        m.symbols_until_update = init_cycle;
        m.update_cycle = init_cycle;
        m
    }

    fn update(&mut self) {
        self.total_count += self.update_cycle;
        if self.total_count > DM_MAX_COUNT {
            self.total_count = 0;
            for c in &mut self.symbol_count {
                *c = (*c + 1) >> 1;
                self.total_count += *c;
            }
        }
        let mut sum = 0u32;
        let scale = 0x8000_0000u32 / self.total_count;
        if self.table_size == 0 {
            for (d, &c) in self.distribution.iter_mut().zip(&self.symbol_count) {
                *d = (scale * sum) >> (31 - DM_LENGTH_SHIFT);
                sum += c;
            }
        } else {
            let mut s = 0usize;
            for (k, (d, &c)) in self
                .distribution
                .iter_mut()
                .zip(&self.symbol_count)
                .enumerate()
            {
                *d = (scale * sum) >> (31 - DM_LENGTH_SHIFT);
                sum += c;
                let w = (*d >> self.table_shift) as usize;
                while s < w {
                    s += 1;
                    self.decoder_table[s] = (k - 1) as u32;
                }
            }
            self.decoder_table[0] = 0;
            while s <= self.table_size as usize {
                s += 1;
                self.decoder_table[s] = self.symbols - 1;
            }
        }
        self.update_cycle = (5 * self.update_cycle) >> 2;
        let max_cycle = (self.symbols + 6) << 3;
        if self.update_cycle > max_cycle {
            self.update_cycle = max_cycle;
        }
        self.symbols_until_update = self.update_cycle;
    }
}

#[derive(Copy, Clone)]
struct ArithmeticBitModel {
    bit_0_count: u32,
    bit_count: u32,
    bit_0_prob: u32,
    bits_until_update: u32,
    update_cycle: u32,
}

impl Default for ArithmeticBitModel {
    fn default() -> Self {
        Self {
            bit_0_count: 1,
            bit_count: 2,
            bit_0_prob: 1u32 << (BM_LENGTH_SHIFT - 1),
            bits_until_update: 4,
            update_cycle: 4,
        }
    }
}

impl ArithmeticBitModel {
    fn update(&mut self) {
        self.bit_count += self.update_cycle;
        if self.bit_count > BM_MAX_COUNT {
            self.bit_count = (self.bit_count + 1) >> 1;
            self.bit_0_count = (self.bit_0_count + 1) >> 1;
            if self.bit_0_count == self.bit_count {
                self.bit_count += 1;
            }
        }
        let scale = 0x8000_0000u32 / self.bit_count;
        self.bit_0_prob = (self.bit_0_count * scale) >> (31 - BM_LENGTH_SHIFT);
        self.update_cycle = (5 * self.update_cycle) >> 2;
        if self.update_cycle > 64 {
            self.update_cycle = 64;
        }
        self.bits_until_update = self.update_cycle;
    }
}

struct ArithmeticDecoder<R: Read> {
    src: R,
    value: u32,
    length: u32,
}

impl<R: Read> ArithmeticDecoder<R> {
    fn new(src: R) -> Self {
        Self {
            src,
            value: 0,
            length: AC_MAX_LENGTH,
        }
    }

    fn read_init_bytes(&mut self) -> io::Result<()> {
        let mut v = [0u8; 4];
        self.src.read_exact(&mut v)?;
        self.value = u32::from_be_bytes(v);
        Ok(())
    }

    #[inline]
    fn renorm(&mut self) -> io::Result<()> {
        while self.length < AC_MIN_LENGTH {
            self.value = (self.value << 8) | u32::from(read_u8(&mut self.src)?);
            self.length <<= 8;
        }
        Ok(())
    }

    #[inline]
    fn decode_bit(&mut self, model: &mut ArithmeticBitModel) -> io::Result<u32> {
        let x = model.bit_0_prob * (self.length >> BM_LENGTH_SHIFT);
        let sym = self.value >= x;
        if !sym {
            self.length = x;
            model.bit_0_count += 1;
        } else {
            self.value -= x;
            self.length -= x;
        }
        if self.length < AC_MIN_LENGTH {
            self.renorm()?;
        }
        model.bits_until_update -= 1;
        if model.bits_until_update == 0 {
            model.update();
        }
        Ok(sym as u32)
    }

    #[inline]
    fn decode_symbol(&mut self, model: &mut ArithmeticModel) -> io::Result<u32> {
        let mut sym;
        let mut n;
        let mut x;
        let mut y = self.length;
        self.length >>= DM_LENGTH_SHIFT;
        if self.length == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "ArithmeticDecoder length reached zero",
            ));
        }

        if !model.decoder_table.is_empty() {
            let dv = self.value / self.length;
            let t = (dv >> model.table_shift) as usize;
            sym = model.decoder_table[t];
            n = model.decoder_table[t + 1] + 1;
            while n > sym + 1 {
                let k = (sym + n) >> 1;
                if model.distribution[k as usize] > dv {
                    n = k;
                } else {
                    sym = k;
                }
            }
            x = model.distribution[sym as usize] * self.length;
            if sym != model.last_symbol {
                y = model.distribution[sym as usize + 1] * self.length;
            }
        } else {
            x = 0;
            sym = 0;
            n = model.symbols;
            let mut k = n >> 1;
            loop {
                let z = self.length * model.distribution[k as usize];
                if z > self.value {
                    n = k;
                    y = z;
                } else {
                    sym = k;
                    x = z;
                }
                k = (sym + n) >> 1;
                if k == sym {
                    break;
                }
            }
        }

        self.value -= x;
        self.length = y - x;
        if self.length < AC_MIN_LENGTH {
            self.renorm()?;
        }
        model.symbol_count[sym as usize] += 1;
        model.symbols_until_update -= 1;
        if model.symbols_until_update == 0 {
            model.update();
        }
        Ok(sym)
    }

    #[inline]
    fn read_short(&mut self) -> io::Result<u16> {
        self.length >>= 16;
        let sym = self.value / self.length;
        self.value -= self.length * sym;
        if self.length < AC_MIN_LENGTH {
            self.renorm()?;
        }
        Ok(sym as u16)
    }

    #[inline]
    fn read_int(&mut self) -> io::Result<u32> {
        let lo = u32::from(self.read_short()?);
        let hi = u32::from(self.read_short()?);
        Ok((hi << 16) | lo)
    }

    #[inline]
    fn read_bits(&mut self, bits: u32) -> io::Result<u32> {
        if bits > 19 {
            let tmp = u32::from(self.read_short()?);
            let hi = self.read_bits(bits - 16)? << 16;
            Ok(hi | tmp)
        } else {
            self.length >>= bits;
            let sym = self.value / self.length;
            self.value -= self.length * sym;
            if self.length < AC_MIN_LENGTH {
                self.renorm()?;
            }
            Ok(sym)
        }
    }
}

#[derive(Clone)]
struct IntegerDecompressor {
    k: u32,
    bits_high: u32,
    corr_range: u32,
    corr_min: i32,
    m_bits: Vec<ArithmeticModel>,
    m_corrector0: ArithmeticBitModel,
    m_corrector: Vec<ArithmeticModel>,
}

impl IntegerDecompressor {
    fn new(bits: u32, contexts: u32) -> Self {
        let bits_high = 8u32;
        let (corr_bits, corr_range, corr_min) = if bits != 0 && bits < 32 {
            let cr = 1u32 << bits;
            (bits, cr, -(cr as i32 / 2))
        } else {
            (32u32, 0u32, i32::MIN)
        };
        let m_bits = (0..contexts)
            .map(|_| ArithmeticModel::new(corr_bits + 1))
            .collect();
        let mut m_corrector = Vec::with_capacity(corr_bits as usize);
        for i in 1..=corr_bits {
            let v = if i <= bits_high {
                1u32 << i
            } else {
                1u32 << bits_high
            };
            m_corrector.push(ArithmeticModel::new(v));
        }
        Self {
            k: 0,
            bits_high,
            corr_range,
            corr_min,
            m_bits,
            m_corrector0: ArithmeticBitModel::default(),
            m_corrector,
        }
    }

    #[inline]
    fn decompress<R: Read>(
        &mut self,
        dec: &mut ArithmeticDecoder<R>,
        pred: i32,
        context: u32,
    ) -> io::Result<i32> {
        self.k = dec.decode_symbol(&mut self.m_bits[context as usize])?;
        let corr = if self.k != 0 {
            if self.k < 32 {
                let mut c = if self.k <= self.bits_high {
                    dec.decode_symbol(&mut self.m_corrector[(self.k - 1) as usize])? as i32
                } else {
                    let k1 = self.k - self.bits_high;
                    let hi =
                        dec.decode_symbol(&mut self.m_corrector[(self.k - 1) as usize])? as i32;
                    let lo = dec.read_bits(k1)? as i32;
                    (hi << k1) | lo
                };
                if c >= (1u32 << (self.k - 1)) as i32 {
                    c += 1;
                } else {
                    c -= ((1u32 << self.k) - 1) as i32;
                }
                c
            } else {
                self.corr_min
            }
        } else {
            dec.decode_bit(&mut self.m_corrector0)? as i32
        };

        let mut real = pred.wrapping_add(corr);
        if real < 0 {
            real = real.wrapping_add(self.corr_range as i32);
        } else if real >= self.corr_range as i32 {
            real = real.wrapping_sub(self.corr_range as i32);
        }
        Ok(real)
    }
}

// ============================================================================
// Streaming Median & Return Context Lookup Tables
// ============================================================================

#[derive(Copy, Clone)]
struct StreamingMedian5 {
    values: [i32; 5],
    high: bool,
}

impl StreamingMedian5 {
    fn new() -> Self {
        Self {
            values: [0; 5],
            high: true,
        }
    }

    #[inline]
    fn add(&mut self, v: i32) {
        if self.high {
            if v < self.values[2] {
                self.values[4] = self.values[3];
                self.values[3] = self.values[2];
                if v < self.values[0] {
                    self.values[2] = self.values[1];
                    self.values[1] = self.values[0];
                    self.values[0] = v;
                } else if v < self.values[1] {
                    self.values[2] = self.values[1];
                    self.values[1] = v;
                } else {
                    self.values[2] = v;
                }
            } else {
                if v < self.values[3] {
                    self.values[4] = self.values[3];
                    self.values[3] = v;
                } else {
                    self.values[4] = v;
                }
                self.high = false;
            }
        } else if self.values[2] < v {
            self.values[0] = self.values[1];
            self.values[1] = self.values[2];
            if self.values[4] < v {
                self.values[2] = self.values[3];
                self.values[3] = self.values[4];
                self.values[4] = v;
            } else if self.values[3] < v {
                self.values[2] = self.values[3];
                self.values[3] = v;
            } else {
                self.values[2] = v;
            }
        } else {
            if self.values[1] < v {
                self.values[0] = self.values[1];
                self.values[1] = v;
            } else {
                self.values[0] = v;
            }
            self.high = true;
        }
    }

    #[inline]
    fn get(&self) -> i32 {
        self.values[2]
    }
}

const NUMBER_RETURN_MAP_8: [[u8; 8]; 8] = [
    [15, 14, 13, 12, 11, 10, 9, 8],
    [14, 0, 1, 3, 6, 10, 10, 9],
    [13, 1, 2, 4, 7, 11, 11, 10],
    [12, 3, 4, 5, 8, 12, 12, 11],
    [11, 6, 7, 8, 9, 13, 13, 12],
    [10, 10, 11, 12, 13, 14, 14, 13],
    [9, 10, 11, 12, 13, 14, 15, 14],
    [8, 9, 10, 11, 12, 13, 14, 15],
];

const NUMBER_RETURN_LEVEL_8: [[u8; 8]; 8] = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [1, 0, 1, 2, 3, 4, 5, 6],
    [2, 1, 0, 1, 2, 3, 4, 5],
    [3, 2, 1, 0, 1, 2, 3, 4],
    [4, 3, 2, 1, 0, 1, 2, 3],
    [5, 4, 3, 2, 1, 0, 1, 2],
    [6, 5, 4, 3, 2, 1, 0, 1],
    [7, 6, 5, 4, 3, 2, 1, 0],
];

const NUMBER_RETURN_MAP_6CTX: [[u8; 16]; 16] = [
    [0, 1, 2, 3, 4, 5, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5],
    [1, 0, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [2, 1, 2, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3],
    [3, 3, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [4, 3, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [5, 3, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [3, 3, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [4, 3, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4],
    [4, 3, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4],
    [5, 3, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4],
    [5, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4],
    [5, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 4, 4, 4],
    [5, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 4, 4],
    [5, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 4],
    [5, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5],
    [5, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5],
];

const NUMBER_RETURN_LEVEL_8CT: [[u8; 16]; 16] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    [1, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7],
    [2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7],
    [3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7],
    [4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7],
    [5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7],
    [6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7],
    [7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 7],
    [7, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7],
    [7, 7, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6],
    [7, 7, 7, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5],
    [7, 7, 7, 7, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4],
    [7, 7, 7, 7, 7, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3],
    [7, 7, 7, 7, 7, 7, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2],
    [7, 7, 7, 7, 7, 7, 7, 7, 6, 5, 4, 3, 2, 1, 0, 1],
    [7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 5, 4, 3, 2, 1, 0],
];

// ============================================================================
// LASzip v3/v4 Selective Layered Chunk Decoder (Point14: PDRF 6..10 & COPC)
// Decodes ONLY X, Y, Z, Classification — skips all unused layers with zero copy!
// ============================================================================

#[derive(Copy, Clone, Default)]
struct Point14State {
    x: i32,
    y: i32,
    z: i32,
    return_number: u8,
    number_of_returns: u8,
    classification: u8,
    gps_time_change: bool,
}

impl Point14State {
    #[inline]
    fn last_point_return_ctx(&self) -> usize {
        let mut lpr = if self.return_number == 1 { 1 } else { 0 };
        if self.return_number >= self.number_of_returns {
            lpr += 2;
        }
        if self.gps_time_change {
            lpr += 4;
        }
        lpr
    }
}

struct Point14ChannelCtx {
    unused: bool,
    last: Point14State,
    last_x_diff_median5: [StreamingMedian5; 12],
    last_y_diff_median5: [StreamingMedian5; 12],
    last_z: [i32; 8],
    changed_values: Vec<ArithmeticModel>,
    scanner_channel: ArithmeticModel,
    number_of_returns: Vec<Option<ArithmeticModel>>,
    return_number: Vec<Option<ArithmeticModel>>,
    return_number_gps_same: ArithmeticModel,
    classification: Vec<Option<ArithmeticModel>>,
    dx: IntegerDecompressor,
    dy: IntegerDecompressor,
    z: IntegerDecompressor,
}

impl Point14ChannelCtx {
    fn unused() -> Self {
        Self {
            unused: true,
            last: Point14State::default(),
            last_x_diff_median5: [StreamingMedian5::new(); 12],
            last_y_diff_median5: [StreamingMedian5::new(); 12],
            last_z: [0; 8],
            changed_values: Vec::new(),
            scanner_channel: ArithmeticModel::new(3),
            number_of_returns: Vec::new(),
            return_number: Vec::new(),
            return_number_gps_same: ArithmeticModel::new(13),
            classification: Vec::new(),
            dx: IntegerDecompressor::new(32, 2),
            dy: IntegerDecompressor::new(32, 22),
            z: IntegerDecompressor::new(32, 20),
        }
    }

    fn from_last(p: Point14State) -> Self {
        let mut last = p;
        last.gps_time_change = false;
        Self {
            unused: false,
            last,
            last_x_diff_median5: [StreamingMedian5::new(); 12],
            last_y_diff_median5: [StreamingMedian5::new(); 12],
            last_z: [p.z; 8],
            changed_values: (0..8).map(|_| ArithmeticModel::new(128)).collect(),
            scanner_channel: ArithmeticModel::new(3),
            number_of_returns: vec![None; 16],
            return_number: vec![None; 16],
            return_number_gps_same: ArithmeticModel::new(13),
            classification: vec![None; 64],
            dx: IntegerDecompressor::new(32, 2),
            dy: IntegerDecompressor::new(32, 22),
            z: IntegerDecompressor::new(32, 20),
        }
    }
}

/// Decode up to `max_points` from a LASzip v3/v4 layered chunk (COPC node or LAZ 1.4 chunk),
/// yielding `(raw_x, raw_y, raw_z, classification)` for every point.
pub fn decode_layered_chunk_xyz_cls<F>(
    chunk: &[u8],
    vlr: &LazVlr,
    max_points: usize,
    mut on_point: F,
) -> io::Result<usize>
where
    F: FnMut(i32, i32, i32, u8),
{
    if max_points == 0 || vlr.items.is_empty() {
        return Ok(0);
    }
    let first_item = vlr.items[0];
    if first_item.item_type != LazItemType::Point14 || (first_item.version != 3 && first_item.version != 4) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Expected Point14 v3/v4 as first item in layered chunk",
        ));
    }

    let first_point_size = vlr.items_size();
    if chunk.len() < first_point_size || first_point_size < 30 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "Layered chunk shorter than first point",
        ));
    }

    // 1. Uncompressed first point (Point14 occupies bytes 0..30)
    let x0 = le_i32(chunk, 0);
    let y0 = le_i32(chunk, 4);
    let z0 = le_i32(chunk, 8);
    let bit_fields0 = chunk[14];
    let flags0 = chunk[15];
    let cls0 = chunk[16];
    let init_channel = ((flags0 & 0b0011_0000) >> 4) as usize;

    let p0 = Point14State {
        x: x0,
        y: y0,
        z: z0,
        return_number: bit_fields0 & 0x0F,
        number_of_returns: (bit_fields0 >> 4) & 0x0F,
        classification: cls0,
        gps_time_change: false,
    };
    on_point(x0, y0, z0, cls0);

    if max_points == 1 || chunk.len() < first_point_size + 4 {
        return Ok(1);
    }

    // 2. Remaining point count in this chunk
    let mut pos = first_point_size;
    let remaining_in_chunk = le_u32(chunk, pos) as usize;
    pos += 4;
    let to_decode = remaining_in_chunk.min(max_points - 1);
    if to_decode == 0 {
        return Ok(1);
    }

    // 3. Layer sizes: 9 u32 for Point14 + trailing item layer size u32s
    let mut extra_layer_size_u32s = 0usize;
    for item in &vlr.items[1..] {
        extra_layer_size_u32s += match item.item_type {
            LazItemType::Rgb14 | LazItemType::WavePacket14 => 1,
            LazItemType::RgbNir14 => 2,
            LazItemType::Byte14(n) => n as usize,
            _ => 0,
        };
    }
    let layer_sizes_bytes = (9 + extra_layer_size_u32s) * 4;
    if chunk.len() < pos + layer_sizes_bytes {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "Layered chunk truncated in layer sizes",
        ));
    }

    let xy_len = le_u32(chunk, pos) as usize;
    let z_len = le_u32(chunk, pos + 4) as usize;
    let cls_len = le_u32(chunk, pos + 8) as usize;
    pos += layer_sizes_bytes;

    if chunk.len() < pos + xy_len + z_len + cls_len {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "Layered chunk truncated in XYZ/Classification layer payload",
        ));
    }

    let xy_slice = &chunk[pos..pos + xy_len];
    let z_slice = &chunk[pos + xy_len..pos + xy_len + z_len];
    let cls_slice = &chunk[pos + xy_len + z_len..pos + xy_len + z_len + cls_len];

    let mut dec_xy = ArithmeticDecoder::new(xy_slice);
    dec_xy.read_init_bytes()?;

    let has_z = z_len > 0;
    let mut dec_z = ArithmeticDecoder::new(z_slice);
    if has_z {
        dec_z.read_init_bytes()?;
    }

    let has_cls = cls_len > 0;
    let mut dec_cls = ArithmeticDecoder::new(cls_slice);
    if has_cls {
        dec_cls.read_init_bytes()?;
    }

    let mut cur_ctx = init_channel & 3;
    let mut contexts = [
        Point14ChannelCtx::unused(),
        Point14ChannelCtx::unused(),
        Point14ChannelCtx::unused(),
        Point14ChannelCtx::unused(),
    ];
    contexts[cur_ctx] = Point14ChannelCtx::from_last(p0);

    let mut decoded = 1usize;
    for _ in 0..to_decode {
        let changed_values = {
            let ctx = &mut contexts[cur_ctx];
            let lpr = ctx.last.last_point_return_ctx();
            dec_xy.decode_symbol(&mut ctx.changed_values[lpr])?
        };

        // Bit 6: scanner channel changed
        if (changed_values & (1 << 6)) != 0 {
            let diff = dec_xy.decode_symbol(&mut contexts[cur_ctx].scanner_channel)?;
            let next_ch = (cur_ctx + diff as usize + 1) & 3;
            if contexts[next_ch].unused {
                let prev_last = contexts[cur_ctx].last;
                contexts[next_ch] = Point14ChannelCtx::from_last(prev_last);
            }
            cur_ctx = next_ch;
        }

        let gps_time_changed = (changed_values & (1 << 4)) != 0;
        let ctx = &mut contexts[cur_ctx];

        let last_n = ctx.last.number_of_returns;
        let last_r = ctx.last.return_number;

        // Bit 2: number of returns changed
        let n = if (changed_values & (1 << 2)) != 0 {
            dec_xy.decode_symbol(
                ctx.number_of_returns[last_n as usize]
                    .get_or_insert_with(|| ArithmeticModel::new(16)),
            )?
        } else {
            u32::from(last_n)
        };
        ctx.last.number_of_returns = n as u8;

        // Bits 0..2: return number delta
        let r = match changed_values & 3 {
            0 => u32::from(last_r),
            1 => u32::from((last_r + 1) & 15),
            2 => u32::from((last_r + 15) & 15),
            _ => {
                if gps_time_changed {
                    dec_xy.decode_symbol(
                        ctx.return_number[last_r as usize]
                            .get_or_insert_with(|| ArithmeticModel::new(16)),
                    )?
                } else {
                    let sym = dec_xy.decode_symbol(&mut ctx.return_number_gps_same)?;
                    (u32::from(last_r) + sym + 2) & 15
                }
            }
        };
        ctx.last.return_number = r as u8;

        let m = usize::from(NUMBER_RETURN_MAP_6CTX[n as usize][r as usize]);
        let l = usize::from(NUMBER_RETURN_LEVEL_8CT[n as usize][r as usize]);
        let cpr = (if r == 1 { 2 } else { 0 }) + (if r >= n { 1 } else { 0 });

        // X
        let idx = (m << 1) | (gps_time_changed as usize);
        let med_x = ctx.last_x_diff_median5[idx].get();
        let diff_x = ctx.dx.decompress(&mut dec_xy, med_x, (n == 1) as u32)?;
        ctx.last.x = ctx.last.x.wrapping_add(diff_x);
        ctx.last_x_diff_median5[idx].add(diff_x);

        // Y
        let med_y = ctx.last_y_diff_median5[idx].get();
        let k_dx = ctx.dx.k;
        let ctx_y = ((n == 1) as u32) + if k_dx < 20 { k_dx & 0xFFFF_FFFE } else { 20 };
        let diff_y = ctx.dy.decompress(&mut dec_xy, med_y, ctx_y)?;
        ctx.last.y = ctx.last.y.wrapping_add(diff_y);
        ctx.last_y_diff_median5[idx].add(diff_y);

        // Z
        if has_z {
            let k_z = (ctx.dx.k + ctx.dy.k) / 2;
            let ctx_z = ((n == 1) as u32) + if k_z < 18 { k_z & 0xFFFF_FFFE } else { 18 };
            ctx.last.z = ctx.z.decompress(&mut dec_z, ctx.last_z[l], ctx_z)?;
            ctx.last_z[l] = ctx.last.z;
        }

        // Classification
        if has_cls {
            let ccc = (((ctx.last.classification & 0x1F) << 1) + (if cpr == 3 { 1 } else { 0 }))
                as usize;
            ctx.last.classification = dec_cls.decode_symbol(
                ctx.classification[ccc].get_or_insert_with(|| ArithmeticModel::new(256)),
            )? as u8;
        }

        ctx.last.gps_time_change = gps_time_changed;
        on_point(ctx.last.x, ctx.last.y, ctx.last.z, ctx.last.classification);
        decoded += 1;
    }

    Ok(decoded)
}

// ============================================================================
// LASzip v2 Sequential Chunk Decoder (Legacy LAS 1.0–1.3: Point10 + GPS + RGB + Byte)
// ============================================================================

struct GpsV2State {
    gps_time_multi: ArithmeticModel,
    gps_time_0_diff: ArithmeticModel,
    ic: IntegerDecompressor,
    last: usize,
    next: usize,
    last_times: [i64; 4],
    last_diffs: [i32; 4],
    extreme: [i32; 4],
}

impl GpsV2State {
    fn new(first_val: i64) -> Self {
        const MULTI_TOTAL: u32 = (500 - (-10) + 6) as u32;
        let mut last_times = [0i64; 4];
        last_times[0] = first_val;
        Self {
            gps_time_multi: ArithmeticModel::new(MULTI_TOTAL),
            gps_time_0_diff: ArithmeticModel::new(6),
            ic: IntegerDecompressor::new(32, 9),
            last: 0,
            next: 0,
            last_times,
            last_diffs: [0; 4],
            extreme: [0; 4],
        }
    }

    fn step<R: Read>(&mut self, dec: &mut ArithmeticDecoder<R>) -> io::Result<()> {
        const MULTI: i32 = 500;
        const MULTI_MINUS: i32 = -10;
        const MULTI_UNCHANGED: i32 = MULTI - MULTI_MINUS + 1;
        const MULTI_CODE_FULL: i32 = MULTI - MULTI_MINUS + 2;

        loop {
            if self.last_diffs[self.last] == 0 {
                let multi = dec.decode_symbol(&mut self.gps_time_0_diff)? as i32;
                if multi == 1 {
                    let d = self.ic.decompress(dec, 0, 0)?;
                    self.last_diffs[self.last] = d;
                    self.last_times[self.last] = self.last_times[self.last].wrapping_add(i64::from(d));
                    self.extreme[self.last] = 0;
                    return Ok(());
                } else if multi == 2 {
                    self.next = (self.next + 1) & 3;
                    let hi = self
                        .ic
                        .decompress(dec, (self.last_times[self.last] >> 32) as i32, 8)?;
                    let lo = dec.read_int()?;
                    self.last_times[self.next] = (i64::from(hi) << 32) | i64::from(lo);
                    self.last = self.next;
                    self.last_diffs[self.last] = 0;
                    self.extreme[self.last] = 0;
                    return Ok(());
                } else if multi > 2 {
                    self.last = (self.last + multi as usize - 2) & 3;
                    continue;
                } else {
                    return Ok(());
                }
            } else {
                let mut multi = dec.decode_symbol(&mut self.gps_time_multi)? as i32;
                if multi == 1 {
                    let d = self.ic.decompress(dec, self.last_diffs[self.last], 1)?;
                    self.last_times[self.last] = self.last_times[self.last].wrapping_add(i64::from(d));
                    self.extreme[self.last] = 0;
                    return Ok(());
                } else if multi < MULTI_UNCHANGED {
                    let d = if multi == 0 {
                        let d = self.ic.decompress(dec, 0, 7)?;
                        self.extreme[self.last] += 1;
                        if self.extreme[self.last] > 3 {
                            self.last_diffs[self.last] = d;
                            self.extreme[self.last] = 0;
                        }
                        d
                    } else if multi < MULTI {
                        let ctx = if multi < 10 { 2 } else { 3 };
                        self.ic
                            .decompress(dec, multi.wrapping_mul(self.last_diffs[self.last]), ctx)?
                    } else if multi == MULTI {
                        let d = self.ic.decompress(
                            dec,
                            multi.wrapping_mul(self.last_diffs[self.last]),
                            4,
                        )?;
                        self.extreme[self.last] += 1;
                        if self.extreme[self.last] > 3 {
                            self.last_diffs[self.last] = d;
                            self.extreme[self.last] = 0;
                        }
                        d
                    } else {
                        multi = MULTI - multi;
                        if multi > MULTI_MINUS {
                            self.ic.decompress(
                                dec,
                                multi.wrapping_mul(self.last_diffs[self.last]),
                                5,
                            )?
                        } else {
                            let d = self.ic.decompress(
                                dec,
                                MULTI_MINUS.wrapping_mul(self.last_diffs[self.last]),
                                6,
                            )?;
                            self.extreme[self.last] += 1;
                            if self.extreme[self.last] > 3 {
                                self.last_diffs[self.last] = d;
                                self.extreme[self.last] = 0;
                            }
                            d
                        }
                    };
                    self.last_times[self.last] = self.last_times[self.last].wrapping_add(i64::from(d));
                    return Ok(());
                } else if multi == MULTI_CODE_FULL {
                    self.next = (self.next + 1) & 3;
                    let hi = self
                        .ic
                        .decompress(dec, (self.last_times[self.last] >> 32) as i32, 8)?;
                    let lo = dec.read_int()?;
                    self.last_times[self.next] = (i64::from(hi) << 32) | i64::from(lo);
                    self.last = self.next;
                    self.last_diffs[self.last] = 0;
                    self.extreme[self.last] = 0;
                    return Ok(());
                } else if multi > MULTI_CODE_FULL {
                    self.last = (self.last + multi as usize - MULTI_CODE_FULL as usize) & 3;
                    continue;
                } else {
                    return Ok(());
                }
            }
        }
    }
}

struct RgbV2State {
    last: [u16; 3],
    byte_used: ArithmeticModel,
    models: [ArithmeticModel; 6],
}

impl RgbV2State {
    fn new(r: u16, g: u16, b: u16) -> Self {
        Self {
            last: [r, g, b],
            byte_used: ArithmeticModel::new(128),
            models: [
                ArithmeticModel::new(256),
                ArithmeticModel::new(256),
                ArithmeticModel::new(256),
                ArithmeticModel::new(256),
                ArithmeticModel::new(256),
                ArithmeticModel::new(256),
            ],
        }
    }

    fn step<R: Read>(&mut self, dec: &mut ArithmeticDecoder<R>) -> io::Result<()> {
        let sym = dec.decode_symbol(&mut self.byte_used)? as u8;
        let mut r = if (sym & 1) != 0 {
            let c = dec.decode_symbol(&mut self.models[0])? as u8;
            u16::from(c.wrapping_add((self.last[0] & 0xFF) as u8))
        } else {
            self.last[0] & 0xFF
        };
        if (sym & 2) != 0 {
            let c = dec.decode_symbol(&mut self.models[1])? as u8;
            r |= u16::from(c.wrapping_add((self.last[0] >> 8) as u8)) << 8;
        } else {
            r |= self.last[0] & 0xFF00;
        }
        let (g, b) = if (sym & (1 << 6)) != 0 {
            let mut diff_l = i32::from((r & 0xFF) as u8) - i32::from((self.last[0] & 0xFF) as u8);
            let mut g = if (sym & 4) != 0 {
                let c = dec.decode_symbol(&mut self.models[2])? as u8;
                let pred = (diff_l + i32::from((self.last[1] & 0xFF) as u8)).clamp(0, 255) as u8;
                u16::from(c.wrapping_add(pred))
            } else {
                self.last[1] & 0xFF
            };
            let mut b = if (sym & 16) != 0 {
                let c = dec.decode_symbol(&mut self.models[4])? as u8;
                diff_l = (diff_l + i32::from((g & 0xFF) as u8) - i32::from((self.last[1] & 0xFF) as u8)) / 2;
                let pred = (diff_l + i32::from((self.last[2] & 0xFF) as u8)).clamp(0, 255) as u8;
                u16::from(c.wrapping_add(pred))
            } else {
                self.last[2] & 0xFF
            };
            let mut diff_h = i32::from((r >> 8) as u8) - i32::from((self.last[0] >> 8) as u8);
            if (sym & 8) != 0 {
                let c = dec.decode_symbol(&mut self.models[3])? as u8;
                let pred = (diff_h + i32::from((self.last[1] >> 8) as u8)).clamp(0, 255) as u8;
                g |= u16::from(c.wrapping_add(pred)) << 8;
            } else {
                g |= self.last[1] & 0xFF00;
            }
            if (sym & 32) != 0 {
                let c = dec.decode_symbol(&mut self.models[5])? as u8;
                diff_h = (diff_h + i32::from((g >> 8) as u8) - i32::from((self.last[1] >> 8) as u8)) / 2;
                let pred = (diff_h + i32::from((self.last[2] >> 8) as u8)).clamp(0, 255) as u8;
                b |= u16::from(c.wrapping_add(pred)) << 8;
            } else {
                b |= self.last[2] & 0xFF00;
            }
            (g, b)
        } else {
            (r, r)
        };
        self.last = [r, g, b];
        Ok(())
    }
}

enum SeqV2Extra {
    Gps(GpsV2State),
    Rgb(RgbV2State),
    Bytes(Vec<u8>, Vec<ArithmeticModel>),
}

/// Decode up to `max_points` from a LASzip v2 sequential chunk (LAS 1.0–1.3 PDRF 0–3).
pub fn decode_sequential_v2_chunk_xyz_cls<F>(
    chunk: &[u8],
    vlr: &LazVlr,
    max_points: usize,
    mut on_point: F,
) -> io::Result<usize>
where
    F: FnMut(i32, i32, i32, u8),
{
    if max_points == 0 || vlr.items.is_empty() {
        return Ok(0);
    }
    let first_item = vlr.items[0];
    if first_item.item_type != LazItemType::Point10 || first_item.version != 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Expected Point10 v2 as first item in sequential chunk",
        ));
    }
    let rec_size = vlr.items_size();
    if chunk.len() < rec_size || rec_size < 20 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "Sequential v2 chunk shorter than first point",
        ));
    }

    let mut x = le_i32(chunk, 0);
    let mut y = le_i32(chunk, 4);
    let mut z = le_i32(chunk, 8);
    let mut bit_fields = chunk[14];
    let mut classification = chunk[15];
    let mut scan_angle_rank = chunk[16] as i8;
    let mut user_data = chunk[17];
    let mut point_source_id = le_u16(chunk, 18);

    on_point(x, y, z, classification & 0x1F);
    if max_points == 1 || chunk.len() < rec_size + 4 {
        return Ok(1);
    }

    let mut pos = 20usize;
    let mut extras = Vec::new();
    for item in &vlr.items[1..] {
        match item.item_type {
            LazItemType::GpsTime if item.version == 2 => {
                extras.push(SeqV2Extra::Gps(GpsV2State::new(le_i64(chunk, pos))));
                pos += 8;
            }
            LazItemType::Rgb12 if item.version == 2 => {
                extras.push(SeqV2Extra::Rgb(RgbV2State::new(
                    le_u16(chunk, pos),
                    le_u16(chunk, pos + 2),
                    le_u16(chunk, pos + 4),
                )));
                pos += 6;
            }
            LazItemType::Byte(n) if item.version == 2 => {
                let n = n as usize;
                let init = chunk[pos..pos + n].to_vec();
                let models = (0..n).map(|_| ArithmeticModel::new(256)).collect();
                extras.push(SeqV2Extra::Bytes(init, models));
                pos += n;
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!("Unsupported sequential LAZ item {:?} v{}", item.item_type, item.version),
                ));
            }
        }
    }

    let mut dec = ArithmeticDecoder::new(&chunk[rec_size..]);
    dec.read_init_bytes()?;

    let mut ic_intensity = IntegerDecompressor::new(16, 4);
    let mut ic_psid = IntegerDecompressor::new(16, 1);
    let mut ic_dx = IntegerDecompressor::new(32, 2);
    let mut ic_dy = IntegerDecompressor::new(32, 22);
    let mut ic_z = IntegerDecompressor::new(32, 20);

    let mut last_intensity = [0u16; 16];
    let mut last_x_med = [StreamingMedian5::new(); 16];
    let mut last_y_med = [StreamingMedian5::new(); 16];
    let mut last_height = [0i32; 8];
    let mut changed_model = ArithmeticModel::new(64);
    let mut scan_angle_models = [ArithmeticModel::new(256), ArithmeticModel::new(256)];
    let mut bit_byte_models: Vec<Option<ArithmeticModel>> = vec![None; 256];
    let mut cls_models: Vec<Option<ArithmeticModel>> = vec![None; 256];
    let mut user_data_models: Vec<Option<ArithmeticModel>> = vec![None; 256];

    let mut decoded = 1usize;
    for _ in 1..max_points {
        let changed = match dec.decode_symbol(&mut changed_model) {
            Ok(v) => v as i32,
            Err(_) => break,
        };
        let (r, n, m, l);
        if changed != 0 {
            if (changed & (1 << 5)) != 0 {
                let mdl = bit_byte_models[bit_fields as usize]
                    .get_or_insert_with(|| ArithmeticModel::new(256));
                bit_fields = dec.decode_symbol(mdl)? as u8;
            }
            r = (bit_fields & 0x07) as usize;
            n = ((bit_fields >> 3) & 0x07) as usize;
            m = NUMBER_RETURN_MAP_8[n][r] as usize;
            l = NUMBER_RETURN_LEVEL_8[n][r] as usize;

            if (changed & (1 << 4)) != 0 {
                let v = ic_intensity.decompress(
                    &mut dec,
                    i32::from(last_intensity[m]),
                    if m < 3 { m as u32 } else { 3 },
                )? as u16;
                last_intensity[m] = v;
            }
            if (changed & (1 << 3)) != 0 {
                let mdl = cls_models[classification as usize]
                    .get_or_insert_with(|| ArithmeticModel::new(256));
                classification = dec.decode_symbol(mdl)? as u8;
            }
            if (changed & (1 << 2)) != 0 {
                let scan_dir = ((bit_fields >> 6) & 1) as usize;
                let dv = dec.decode_symbol(&mut scan_angle_models[scan_dir])? as i8;
                scan_angle_rank = scan_angle_rank.wrapping_add(dv);
            }
            if (changed & (1 << 1)) != 0 {
                let mdl = user_data_models[user_data as usize]
                    .get_or_insert_with(|| ArithmeticModel::new(256));
                user_data = dec.decode_symbol(mdl)? as u8;
            }
            if (changed & 1) != 0 {
                point_source_id = ic_psid.decompress(&mut dec, i32::from(point_source_id), 0)? as u16;
            }
        } else {
            r = (bit_fields & 0x07) as usize;
            n = ((bit_fields >> 3) & 0x07) as usize;
            m = NUMBER_RETURN_MAP_8[n][r] as usize;
            l = NUMBER_RETURN_LEVEL_8[n][r] as usize;
        }

        // X
        let dx = ic_dx.decompress(&mut dec, last_x_med[m].get(), (n == 1) as u32)?;
        x = x.wrapping_add(dx);
        last_x_med[m].add(dx);

        // Y
        let k_dx = ic_dx.k;
        let ctx_y = ((n == 1) as u32) + if k_dx < 20 { k_dx & 0xFFFF_FFFE } else { 20 };
        let dy = ic_dy.decompress(&mut dec, last_y_med[m].get(), ctx_y)?;
        y = y.wrapping_add(dy);
        last_y_med[m].add(dy);

        // Z
        let k_z = (ic_dx.k + ic_dy.k) / 2;
        let ctx_z = ((n == 1) as u32) + if k_z < 18 { k_z & 0xFFFF_FFFE } else { 18 };
        z = ic_z.decompress(&mut dec, last_height[l], ctx_z)?;
        last_height[l] = z;

        for ex in &mut extras {
            match ex {
                SeqV2Extra::Gps(g) => g.step(&mut dec)?,
                SeqV2Extra::Rgb(rgb) => rgb.step(&mut dec)?,
                SeqV2Extra::Bytes(last, models) => {
                    for (b, mdl) in last.iter_mut().zip(models.iter_mut()) {
                        let s = dec.decode_symbol(mdl)? as u8;
                        *b = b.wrapping_add(s);
                    }
                }
            }
        }

        on_point(x, y, z, classification & 0x1F);
        decoded += 1;
    }

    Ok(decoded)
}

/// Dispatch chunk decompression automatically based on `LazVlr` compressor type (v3/v4 layered or v2 sequential).
pub fn decode_laz_chunk_xyz_cls<F>(
    chunk: &[u8],
    vlr: &LazVlr,
    max_points: usize,
    on_point: F,
) -> io::Result<usize>
where
    F: FnMut(i32, i32, i32, u8),
{
    match vlr.compressor {
        CompressorType::LayeredChunked => decode_layered_chunk_xyz_cls(chunk, vlr, max_points, on_point),
        CompressorType::PointWise | CompressorType::PointWiseChunked => {
            decode_sequential_v2_chunk_xyz_cls(chunk, vlr, max_points, on_point)
        }
        CompressorType::None => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Uncompressed LAZ compressor type",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // Real LASzip-compressed LAS 1.4 PDRF 6 (25 pts, layered chunks, exact IGN LiDAR HD format)
    const FIXTURE_LAS14_PDRF6_LAZ: &[u8] = &[
        0x4c, 0x41, 0x53, 0x46, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x01, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x77, 0x01,
        0xd5, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x86, 0x1e, 0x00, 0x19, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x08,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xfc, 0xa9, 0xf1, 0xd2, 0x4d, 0x62, 0x50, 0x3f, 0xfc, 0xa9, 0xf1, 0xd2, 0x4d,
        0x62, 0x50, 0x3f, 0xfc, 0xa9, 0xf1, 0xd2, 0x4d, 0x62, 0x50, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x70, 0x6f, 0x2b, 0x41, 0x00, 0x00, 0x00, 0x00, 0x18,
        0xe9, 0x57, 0x41, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xd7, 0xa3, 0x70, 0x3d, 0x06, 0x70, 0x2b, 0x41, 0x00, 0x00, 0x00, 0x00, 0x70,
        0x6f, 0x2b, 0x41, 0xf2, 0xd2, 0x4d, 0x0a, 0x18, 0xe9, 0x57, 0x41, 0x00, 0x00, 0x00, 0x00, 0x18, 0xe9, 0x57, 0x41, 0x33, 0x33, 0x33, 0x33, 0x33,
        0x33, 0x68, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x66, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x6c, 0x61, 0x73, 0x7a, 0x69, 0x70, 0x20,
        0x65, 0x6e, 0x63, 0x6f, 0x64, 0x65, 0x64, 0x00, 0x00, 0xbc, 0x56, 0x28, 0x00, 0x68, 0x74, 0x74, 0x70, 0x73, 0x3a, 0x2f, 0x2f, 0x6c, 0x61, 0x73,
        0x7a, 0x69, 0x70, 0x2e, 0x6f, 0x72, 0x67, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00,
        0x00, 0x02, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x50, 0xc3, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0x01, 0x00, 0x0a, 0x00, 0x1e, 0x00, 0x03, 0x00, 0x30, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0xbf, 0x02, 0x00, 0x00, 0x00, 0x31, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x6a, 0xf8, 0x40, 0x19, 0x00, 0x00, 0x00, 0x6e, 0x00, 0x00, 0x00, 0x38, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x27,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x00, 0x00, 0x00, 0x82, 0x41, 0xc4, 0x67, 0x7a,
        0x79, 0xb4, 0xb6, 0xae, 0x44, 0xcb, 0xab, 0x37, 0x7b, 0x9f, 0x47, 0xfa, 0x5f, 0x67, 0x0a, 0x60, 0x90, 0xd7, 0x8f, 0xbd, 0xa8, 0x17, 0xee, 0xfd,
        0xf4, 0x65, 0x25, 0xc1, 0x63, 0xe8, 0xd2, 0x2b, 0x3e, 0x55, 0xb3, 0xa0, 0x62, 0xc6, 0x52, 0x05, 0x29, 0xf5, 0x40, 0x14, 0x3d, 0xbe, 0x90, 0xc7,
        0x20, 0xd6, 0xaa, 0x91, 0x37, 0xac, 0x38, 0x6c, 0x2c, 0xad, 0xf2, 0xe9, 0x67, 0xd4, 0x7d, 0x7e, 0x67, 0x26, 0x32, 0xee, 0xbe, 0xa3, 0xf9, 0xf5,
        0x61, 0xd4, 0xb2, 0x30, 0xda, 0xd7, 0x66, 0x37, 0x21, 0x90, 0x82, 0x9c, 0xea, 0xe3, 0x14, 0x91, 0x8f, 0x1d, 0x08, 0xf5, 0x25, 0xc3, 0xcb, 0x86,
        0xd6, 0x01, 0x7f, 0x5a, 0xda, 0x09, 0x00, 0x00, 0x00, 0x62, 0x1a, 0x19, 0x1d, 0xb8, 0xf7, 0x9f, 0x2f, 0xc0, 0x0f, 0x84, 0x07, 0x95, 0x2f, 0xce,
        0x0a, 0x91, 0x32, 0x94, 0xfd, 0x71, 0xb7, 0x4e, 0x40, 0x26, 0xea, 0x95, 0xe2, 0xee, 0x83, 0x54, 0xce, 0x7d, 0xd5, 0xf4, 0x4d, 0x1d, 0x08, 0x60,
        0x0f, 0xc2, 0xc2, 0xc8, 0x09, 0xb6, 0x10, 0x7e, 0x99, 0xe1, 0x1f, 0xb8, 0xe2, 0xb4, 0x00, 0x00, 0x00, 0x03, 0x05, 0x04, 0x84, 0x7e, 0xff, 0x84,
        0x04, 0x84, 0x7e, 0xff, 0x84, 0x04, 0x84, 0x7e, 0xff, 0x84, 0x04, 0x84, 0x7e, 0xff, 0x84, 0x04, 0x84, 0x7d, 0x00, 0x00, 0x00, 0x62, 0xec, 0x3d,
        0xb2, 0x84, 0xf7, 0xcb, 0x26, 0x00, 0x2b, 0xa3, 0x45, 0x8b, 0xee, 0xf7, 0x0e, 0x20, 0xbb, 0x0b, 0x9d, 0x43, 0x05, 0xa5, 0xa9, 0x91, 0xe1, 0x53,
        0x91, 0x40, 0x15, 0x0f, 0xb0, 0xe9, 0x6e, 0x1e, 0xc6, 0xc8, 0x00, 0x00, 0x27, 0x93, 0x93, 0x6a, 0x46, 0x9c, 0x46, 0x7f, 0x7c, 0x4b, 0x4e, 0x91,
        0xb1, 0x1e, 0x9d, 0x5a, 0xbe, 0xc2, 0x7b, 0x02, 0x44, 0x83, 0xd3, 0x0e, 0xd7, 0x3c, 0x35, 0x5e, 0x84, 0x59, 0xeb, 0x1e, 0x20, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x4a, 0xef, 0x00, 0x00, 0x00,
    ];

    // Real LASzip-compressed LAS 1.2 PDRF 3 (25 pts, Point10 + GPS + RGB12 v2)
    const FIXTURE_LAS12_PDRF3_LAZ: &[u8] = &[
        0x4c, 0x41, 0x53, 0x46, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xe3, 0x00, 0x4d, 0x01, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x83, 0x22, 0x00, 0x19, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x08,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xfc, 0xa9, 0xf1, 0xd2, 0x4d, 0x62, 0x50, 0x3f, 0xfc,
        0xa9, 0xf1, 0xd2, 0x4d, 0x62, 0x50, 0x3f, 0xfc, 0xa9, 0xf1, 0xd2, 0x4d, 0x62, 0x50, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x70,
        0x6f, 0x2b, 0x41, 0x00, 0x00, 0x00, 0x00, 0x18, 0xe9, 0x57, 0x41, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xd7,
        0xa3, 0x70, 0x3d, 0x06, 0x70, 0x2b, 0x41, 0x00, 0x00, 0x00, 0x00, 0x70, 0x6f, 0x2b, 0x41, 0xf2, 0xd2, 0x4d, 0x0a, 0x18,
        0xe9, 0x57, 0x41, 0x00, 0x00, 0x00, 0x00, 0x18, 0xe9, 0x57, 0x41, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x68, 0x40, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x80, 0x66, 0x40, 0x00, 0x00, 0x6c, 0x61, 0x73, 0x7a, 0x69, 0x70, 0x20, 0x65, 0x6e, 0x63, 0x6f,
        0x64, 0x65, 0x64, 0x00, 0x00, 0xbc, 0x56, 0x34, 0x00, 0x68, 0x74, 0x74, 0x70, 0x73, 0x3a, 0x2f, 0x2f, 0x6c, 0x61, 0x73,
        0x7a, 0x69, 0x70, 0x2e, 0x6f, 0x72, 0x67, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x50, 0xc3, 0x00, 0x00, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x03, 0x00, 0x06, 0x00, 0x14, 0x00, 0x02,
        0x00, 0x07, 0x00, 0x08, 0x00, 0x02, 0x00, 0x08, 0x00, 0x06, 0x00, 0x02, 0x00, 0x17, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0xbf, 0x02, 0x00, 0x00, 0x00, 0x19, 0x02, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x6a, 0xf8, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xe0, 0x69, 0x0e, 0x19, 0x02,
        0x7a, 0xcb, 0xf8, 0x22, 0x16, 0xd2, 0xd7, 0x54, 0x0c, 0xb4, 0x22, 0x95, 0x15, 0xf7, 0xb8, 0x1f, 0x54, 0xf0, 0x16, 0xdb,
        0x5a, 0x16, 0x85, 0xd4, 0x03, 0x23, 0x88, 0xc0, 0x86, 0x87, 0xcb, 0x5d, 0x43, 0x4a, 0x5f, 0x45, 0x97, 0x7e, 0x5f, 0x12,
        0x87, 0xa6, 0xe9, 0xa4, 0x79, 0xfd, 0x76, 0xd4, 0xd3, 0x31, 0xdd, 0x53, 0x22, 0x46, 0x73, 0x6a, 0xee, 0xb3, 0xad, 0xc3,
        0x61, 0x84, 0x24, 0xe4, 0x6b, 0x71, 0xea, 0xa6, 0xb1, 0x63, 0xce, 0xc6, 0x66, 0x28, 0xf3, 0xd9, 0x27, 0x6c, 0xae, 0x05,
        0xc1, 0xc8, 0x17, 0x9f, 0xb4, 0x31, 0x62, 0xa7, 0x21, 0x6a, 0x29, 0x26, 0x1c, 0xfc, 0x87, 0x89, 0x50, 0x19, 0xf1, 0xb6,
        0x53, 0x5c, 0xe5, 0x2a, 0xa0, 0x88, 0xfe, 0x58, 0x29, 0xc1, 0xe5, 0xf4, 0x31, 0xbc, 0xc2, 0x37, 0x58, 0x31, 0xaf, 0x5c,
        0x22, 0x98, 0xc8, 0xaf, 0x4d, 0x70, 0x17, 0xe4, 0x28, 0x2d, 0x7f, 0x2d, 0x7f, 0x00, 0xc2, 0xc4, 0x62, 0x15, 0xd6, 0xa0,
        0x64, 0x9a, 0x1f, 0xc2, 0xbd, 0x06, 0xa2, 0xf4, 0x78, 0xdb, 0x57, 0x8e, 0x62, 0x77, 0x33, 0x10, 0x09, 0x25, 0xc6, 0x7b,
        0xb7, 0x19, 0x9d, 0xd4, 0x92, 0x30, 0xd4, 0xae, 0x20, 0x6c, 0x0d, 0xf8, 0x5f, 0xd9, 0xff, 0xe2, 0x64, 0xe2, 0xe6, 0x72,
        0xcb, 0x69, 0x87, 0xf7, 0x12, 0xd7, 0x78, 0x52, 0xf0, 0x83, 0xbe, 0x40, 0xff, 0x70, 0xed, 0xa5, 0x0d, 0x28, 0x6b, 0x59,
        0x40, 0x69, 0x78, 0xab, 0x46, 0xc9, 0x45, 0x5e, 0xe9, 0xc3, 0x4e, 0xa1, 0x1a, 0xef, 0x90, 0x8e, 0x41, 0x0f, 0x35, 0xb5,
        0xa5, 0xf9, 0xd0, 0xf0, 0x6d, 0xc5, 0x39, 0x26, 0x73, 0x93, 0x26, 0x8c, 0x56, 0x8f, 0x91, 0x23, 0x50, 0xe6, 0xae, 0xcb,
        0x8d, 0xa3, 0xb9, 0x38, 0x17, 0xf1, 0x35, 0x3e, 0x75, 0x53, 0xeb, 0x72, 0xe2, 0x92, 0x8f, 0x97, 0xe0, 0xb1, 0x12, 0x98,
        0xa0, 0x86, 0x68, 0xf9, 0x05, 0x81, 0xc3, 0x72, 0x5f, 0x88, 0xba, 0x37, 0x71, 0x7c, 0x68, 0xa9, 0x47, 0x89, 0xc8, 0x31,
        0xaa, 0x76, 0xe4, 0xd1, 0x85, 0x18, 0xc8, 0x35, 0x65, 0xe1, 0x26, 0x46, 0xf8, 0x99, 0xc7, 0x42, 0x0b, 0xeb, 0xf1, 0x1b,
        0x60, 0x76, 0x84, 0xc1, 0xe9, 0x0f, 0x67, 0x84, 0xf6, 0xf5, 0x88, 0xd1, 0x9f, 0xd3, 0x0f, 0x24, 0x24, 0x82, 0x15, 0xd3,
        0xec, 0x9e, 0x94, 0xdb, 0x07, 0x02, 0x65, 0x1b, 0x0d, 0x25, 0x89, 0x31, 0x0f, 0x22, 0x48, 0x4b, 0x74, 0x18, 0xde, 0x7e,
        0x68, 0xcb, 0x0e, 0x6e, 0x80, 0x8d, 0x8e, 0xe4, 0x4f, 0x8c, 0x81, 0x79, 0xe3, 0x26, 0x5f, 0x2a, 0x4f, 0xf2, 0xea, 0xb1,
        0x82, 0xbe, 0x07, 0x78, 0xa8, 0x4a, 0xc8, 0x9f, 0x0a, 0x2e, 0xe2, 0xde, 0x9b, 0xe3, 0x50, 0x40, 0xc2, 0x9f, 0xb2, 0xbb,
        0xed, 0x11, 0x61, 0xf7, 0x72, 0x7d, 0x4c, 0x73, 0x16, 0x28, 0x70, 0xd4, 0xf0, 0x07, 0x64, 0x8e, 0xa8, 0x2f, 0x1d, 0x56,
        0xf3, 0x2f, 0x5f, 0x1c, 0x76, 0x8c, 0xc1, 0x6e, 0xbe, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x4c,
        0x9e, 0x00, 0x00, 0x00,
    ];

    fn check_fixture(bytes: &[u8], expected_fmt: u8) {
        let mut cur = Cursor::new(bytes);
        let hdr = LasHeader::read_from(&mut cur).unwrap();
        assert!(hdr.is_compressed());
        assert_eq!(hdr.point_format_id(), expected_fmt);
        assert_eq!(hdr.total_points(), 25);

        let mut laz_vlr = None;
        for _ in 0..hdr.number_of_vlrs {
            let vlr = LasVlr::read_from(&mut cur).unwrap();
            if vlr.user_id_str().eq_ignore_ascii_case("laszip encoded") && vlr.record_id == 22204 {
                laz_vlr = Some(LazVlr::parse(&vlr.data).unwrap());
            }
        }
        let vlr = laz_vlr.expect("missing laszip vlr");
        let chunks = read_chunk_table(&mut cur, hdr.offset_to_point_data as u64, &vlr, hdr.total_points()).unwrap();
        assert_eq!(chunks.len(), 1);

        let mut got = Vec::new();
        let ch = chunks[0];
        let start = ch.file_offset as usize;
        let end = start + ch.byte_count as usize;
        decode_laz_chunk_xyz_cls(&bytes[start..end], &vlr, ch.point_count as usize, |rx, ry, rz, cls| {
            let x = rx as f64 * hdr.x_scale_factor + hdr.x_offset;
            let y = ry as f64 * hdr.y_scale_factor + hdr.y_offset;
            let z = rz as f64 * hdr.z_scale_factor + hdr.z_offset;
            got.push((x, y, z, cls));
        })
        .unwrap();

        assert_eq!(got.len(), 25);
        for (i, &(gx, gy, gz, gc)) in got.iter().enumerate() {
            let ix = (i % 25) as f64;
            let iy = (i / 25) as f64;
            let ex = 899_000.0 + ix * 3.125 + ((i * 7) % 13) as f64 * 0.01;
            let ey = 6_268_000.0 + iy * 2.750 + ((i * 11) % 17) as f64 * 0.01;
            let ez = 180.0 + 0.15 * ix + 0.08 * iy + ((i % 5) as f64) * 2.5;
            let ec = match i % 5 {
                0 => 2,
                1 => 3,
                2 => 5,
                3 => 6,
                _ => 7,
            };
            assert!((gx - ex).abs() < 1e-3, "fmt={} pt {} X: {} vs {}", expected_fmt, i, gx, ex);
            assert!((gy - ey).abs() < 1e-3, "fmt={} pt {} Y: {} vs {}", expected_fmt, i, gy, ey);
            assert!((gz - ez).abs() < 1e-3, "fmt={} pt {} Z: {} vs {}", expected_fmt, i, gz, ez);
            assert_eq!(gc, ec, "fmt={} pt {} cls: {} vs {}", expected_fmt, i, gc, ec);
        }
    }

    #[test]
    fn test_zero_dep_laz_v3_and_v2_real_fixtures() {
        check_fixture(FIXTURE_LAS14_PDRF6_LAZ, 6);
        check_fixture(FIXTURE_LAS12_PDRF3_LAZ, 3);
    }
}


