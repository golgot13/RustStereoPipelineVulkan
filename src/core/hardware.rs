//! HardwareProfile auto-detection (RAM budget, Full-Scale 1:1 tile size, CPU threads).
//! Inspired by AEROPHOTOGRAM's SurfacePro7/Standard/Workstation memory governance, implemented with 0 external crates.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileTier {
    Compact,
    Standard,
    Workstation,
}

impl ProfileTier {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Compact => "compact",
            Self::Standard => "standard",
            Self::Workstation => "workstation",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "compact" | "low" | "8gb" => Some(Self::Compact),
            "standard" | "medium" | "16gb" => Some(Self::Standard),
            "workstation" | "high" | "32gb" => Some(Self::Workstation),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub tier: ProfileTier,
    pub total_ram_mb: u64,
    pub ram_budget_mb: u64,
    pub tile_size: u32,
    pub recommended_threads: usize,
    pub avx2_supported: bool,
}

impl HardwareProfile {
    /// Detect system RAM and CPU threads using OS-native zero-dependency probes.
    pub fn detect() -> Self {
        let total_ram_mb = detect_total_ram_mb();
        let cpu_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let tier = if total_ram_mb <= 10_240 {
            ProfileTier::Compact
        } else if total_ram_mb <= 24_576 {
            ProfileTier::Standard
        } else {
            ProfileTier::Workstation
        };
        Self::from_tier(tier, total_ram_mb, cpu_cores)
    }

    pub fn from_tier_auto(tier: ProfileTier) -> Self {
        let total_ram_mb = detect_total_ram_mb();
        let cpu_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Self::from_tier(tier, total_ram_mb, cpu_cores)
    }

    fn from_tier(tier: ProfileTier, total_ram_mb: u64, cpu_cores: usize) -> Self {
        let (ram_budget_mb, tile_size, max_threads) = match tier {
            ProfileTier::Compact => (3_072, 512, 4),
            ProfileTier::Standard => (8_192, 1024, 8),
            ProfileTier::Workstation => (20_480, 2048, 32),
        };
        let avx2_supported = detect_avx2();
        Self {
            tier,
            total_ram_mb,
            ram_budget_mb: ram_budget_mb.min((total_ram_mb * 75 / 100).max(512)),
            tile_size,
            recommended_threads: cpu_cores.clamp(1, max_threads),
            avx2_supported,
        }
    }

    /// Estimate peak memory in MB for running SGM/MGM cost volume on a tile of `(tw, th)` with `num_disp` disparities.
    pub fn estimate_tile_memory_mb(tw: u32, th: u32, num_disp: u32) -> u64 {
        let pixels = (tw as u64) * (th as u64);
        // Cost volume + aggregated volume + path buffer = ~3 * W * H * D * 4 bytes + ~12 * W * H * 4 bytes for rasters
        let vol_bytes = pixels * (num_disp.max(1) as u64) * 12;
        let img_bytes = pixels * 48;
        ((vol_bytes + img_bytes) / (1024 * 1024)).max(1)
    }
}

fn detect_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_os = "linux")]
fn detect_total_ram_mb() -> u64 {
    if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
        for line in content.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                if let Some(kb_str) = rest.split_whitespace().next() {
                    if let Ok(kb) = kb_str.parse::<u64>() {
                        return (kb / 1024).max(512);
                    }
                }
            }
        }
    }
    8_192
}

#[cfg(windows)]
fn detect_total_ram_mb() -> u64 {
    #[repr(C)]
    struct MemoryStatusEx {
        dw_length: u32,
        dw_memory_load: u32,
        ull_total_phys: u64,
        ull_avail_phys: u64,
        ull_total_page_file: u64,
        ull_avail_page_file: u64,
        ull_total_virtual: u64,
        ull_avail_virtual: u64,
        ull_avail_extended_virtual: u64,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GlobalMemoryStatusEx(lp_buffer: *mut MemoryStatusEx) -> i32;
    }
    let mut status = MemoryStatusEx {
        dw_length: std::mem::size_of::<MemoryStatusEx>() as u32,
        dw_memory_load: 0,
        ull_total_phys: 0,
        ull_avail_phys: 0,
        ull_total_page_file: 0,
        ull_avail_page_file: 0,
        ull_total_virtual: 0,
        ull_avail_virtual: 0,
        ull_avail_extended_virtual: 0,
    };
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
    if ok != 0 && status.ull_total_phys > 0 {
        (status.ull_total_phys / (1024 * 1024)).max(512)
    } else {
        8_192
    }
}

#[cfg(not(any(target_os = "linux", windows)))]
fn detect_total_ram_mb() -> u64 {
    8_192
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_profile_detection() {
        let p = HardwareProfile::detect();
        assert!(p.total_ram_mb >= 512);
        assert!(p.ram_budget_mb >= 512);
        assert!(p.tile_size >= 512);
        assert!(p.recommended_threads >= 1);
    }
}
