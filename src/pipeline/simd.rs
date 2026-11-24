//! Runtime-detected CPU SIMD AVX2 kernels for stereo matching costs (SAD, SSD, ZNCC).
//! Guarded by `is_x86_feature_detected!("avx2")` with automatic safe scalar fallback (zero SIGILL risk).

/// Compute SAD (sum of absolute differences) between two equal-length `f32` slices.
#[inline]
pub fn row_sad(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && a.len() >= 8 {
            // SAFETY: AVX2 availability verified at runtime above.
            return unsafe { row_sad_avx2(a, b) };
        }
    }
    row_sad_scalar(a, b)
}

/// Compute SSD (sum of squared differences) between two equal-length `f32` slices.
#[inline]
pub fn row_ssd(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && a.len() >= 8 {
            // SAFETY: AVX2 availability verified at runtime above.
            return unsafe { row_ssd_avx2(a, b) };
        }
    }
    row_ssd_scalar(a, b)
}

/// Accumulate `(sum_a, sum_b, sum_aa, sum_bb, sum_ab)` for Zero-Mean NCC over two slices.
#[inline]
pub fn row_ncc_moments(a: &[f32], b: &[f32]) -> (f32, f32, f32, f32, f32) {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && a.len() >= 8 {
            // SAFETY: AVX2 availability verified at runtime above.
            return unsafe { row_ncc_moments_avx2(a, b) };
        }
    }
    row_ncc_moments_scalar(a, b)
}

#[inline]
pub fn row_sad_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        sum += (x - y).abs();
    }
    sum
}

#[inline]
pub fn row_ssd_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

#[inline]
pub fn row_ncc_moments_scalar(a: &[f32], b: &[f32]) -> (f32, f32, f32, f32, f32) {
    let mut sa = 0.0f32;
    let mut sb = 0.0f32;
    let mut saa = 0.0f32;
    let mut sbb = 0.0f32;
    let mut sab = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        sa += x;
        sb += y;
        saa += x * x;
        sbb += y * y;
        sab += x * y;
    }
    (sa, sb, saa, sbb, sab)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum256_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(s128);
    let sums = _mm_add_ps(s128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn row_sad_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let sign_mask = _mm256_set1_ps(-0.0);
    let mut acc = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(va, vb);
        let abs_d = _mm256_andnot_ps(sign_mask, diff);
        acc = _mm256_add_ps(acc, abs_d);
        i += 8;
    }
    let mut sum = hsum256_ps(acc);
    while i < n {
        sum += (a[i] - b[i]).abs();
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn row_ssd_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(va, vb);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(diff, diff));
        i += 8;
    }
    let mut sum = hsum256_ps(acc);
    while i < n {
        let d = a[i] - b[i];
        sum += d * d;
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn row_ncc_moments_avx2(a: &[f32], b: &[f32]) -> (f32, f32, f32, f32, f32) {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut vsa = _mm256_setzero_ps();
    let mut vsb = _mm256_setzero_ps();
    let mut vsaa = _mm256_setzero_ps();
    let mut vsbb = _mm256_setzero_ps();
    let mut vsab = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        vsa = _mm256_add_ps(vsa, va);
        vsb = _mm256_add_ps(vsb, vb);
        vsaa = _mm256_add_ps(vsaa, _mm256_mul_ps(va, va));
        vsbb = _mm256_add_ps(vsbb, _mm256_mul_ps(vb, vb));
        vsab = _mm256_add_ps(vsab, _mm256_mul_ps(va, vb));
        i += 8;
    }
    let mut sa = hsum256_ps(vsa);
    let mut sb = hsum256_ps(vsb);
    let mut saa = hsum256_ps(vsaa);
    let mut sbb = hsum256_ps(vsbb);
    let mut sab = hsum256_ps(vsab);
    while i < n {
        let x = a[i];
        let y = b[i];
        sa += x;
        sb += y;
        saa += x * x;
        sbb += y * y;
        sab += x * y;
        i += 1;
    }
    (sa, sb, saa, sbb, sab)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_matches_scalar() {
        let a: Vec<f32> = (0..37).map(|i| (i as f32 * 0.13).sin()).collect();
        let b: Vec<f32> = (0..37).map(|i| (i as f32 * 0.21 + 0.5).cos()).collect();
        assert!((row_sad(&a, &b) - row_sad_scalar(&a, &b)).abs() < 1e-4);
        assert!((row_ssd(&a, &b) - row_ssd_scalar(&a, &b)).abs() < 1e-4);
        let m1 = row_ncc_moments(&a, &b);
        let m2 = row_ncc_moments_scalar(&a, &b);
        assert!((m1.0 - m2.0).abs() < 1e-4);
        assert!((m1.1 - m2.1).abs() < 1e-4);
        assert!((m1.2 - m2.2).abs() < 1e-4);
        assert!((m1.3 - m2.3).abs() < 1e-4);
        assert!((m1.4 - m2.4).abs() < 1e-4);
    }
}
