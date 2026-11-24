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

/// Branchless SGM dynamic programming recurrence for one pixel over `num_d` disparities:
/// `v[d] = cv[d] + (min(prev_pad[d+1], prev_pad[d]+p1, prev_pad[d+2]+p1, min_prev+p2_eff) - min_prev)`
/// Writes `cur_pad[d+1] = v[d]`, accumulates `agg[d] += v[d]`, and returns `min_d(v[d])`.
/// Requires `prev_pad.len() == cv.len() + 2` and `cur_pad.len() == cv.len() + 2` with `+INF` sentinels at ends.
#[inline]
pub fn sgm_dp_step_add(
    cv: &[f32],
    prev_pad: &[f32],
    cur_pad: &mut [f32],
    agg: &mut [f32],
    min_prev: f32,
    p1: f32,
    p2_eff: f32,
) -> f32 {
    debug_assert_eq!(prev_pad.len(), cv.len() + 2);
    debug_assert_eq!(cur_pad.len(), cv.len() + 2);
    debug_assert_eq!(agg.len(), cv.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && cv.len() >= 8 {
            // SAFETY: AVX2 availability verified at runtime above, slice bounds checked by caller/debug_assert.
            return unsafe { sgm_dp_step_add_avx2(cv, prev_pad, cur_pad, agg, min_prev, p1, p2_eff) };
        }
    }
    sgm_dp_step_add_scalar(cv, prev_pad, cur_pad, agg, min_prev, p1, p2_eff)
}

#[inline]
pub fn sgm_dp_step_add_scalar(
    cv: &[f32],
    prev_pad: &[f32],
    cur_pad: &mut [f32],
    agg: &mut [f32],
    min_prev: f32,
    p1: f32,
    p2_eff: f32,
) -> f32 {
    let mp2 = min_prev + p2_eff;
    let mut min_cur = f32::INFINITY;
    let n = cv.len();
    for d in 0..n {
        let c0 = prev_pad[d + 1];
        let cm = prev_pad[d] + p1;
        let cp = prev_pad[d + 2] + p1;
        let v = cv[d] + (c0.min(cm).min(cp).min(mp2) - min_prev);
        cur_pad[d + 1] = v;
        agg[d] += v;
        if v < min_cur {
            min_cur = v;
        }
    }
    min_cur
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hmin256_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let m128 = _mm_min_ps(lo, hi);
    let shuf = _mm_movehdup_ps(m128);
    let m2 = _mm_min_ps(m128, shuf);
    let shuf2 = _mm_movehl_ps(m2, m2);
    _mm_cvtss_f32(_mm_min_ss(m2, shuf2))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sgm_dp_step_add_avx2(
    cv: &[f32],
    prev_pad: &[f32],
    cur_pad: &mut [f32],
    agg: &mut [f32],
    min_prev: f32,
    p1: f32,
    p2_eff: f32,
) -> f32 {
    use std::arch::x86_64::*;
    let n = cv.len();
    let vp1 = _mm256_set1_ps(p1);
    let vmin_prev = _mm256_set1_ps(min_prev);
    let vmp2 = _mm256_set1_ps(min_prev + p2_eff);
    let mut vmin_cur = _mm256_set1_ps(f32::INFINITY);

    let cv_ptr = cv.as_ptr();
    let prev_ptr = prev_pad.as_ptr();
    let cur_ptr = cur_pad.as_mut_ptr();
    let agg_ptr = agg.as_mut_ptr();

    let mut d = 0usize;
    while d + 8 <= n {
        let vm = _mm256_add_ps(_mm256_loadu_ps(prev_ptr.add(d)), vp1);
        let v0 = _mm256_loadu_ps(prev_ptr.add(d + 1));
        let vp = _mm256_add_ps(_mm256_loadu_ps(prev_ptr.add(d + 2)), vp1);
        let best = _mm256_min_ps(_mm256_min_ps(v0, vm), _mm256_min_ps(vp, vmp2));
        let smooth = _mm256_sub_ps(best, vmin_prev);
        let vcv = _mm256_loadu_ps(cv_ptr.add(d));
        let v = _mm256_add_ps(vcv, smooth);
        _mm256_storeu_ps(cur_ptr.add(d + 1), v);
        let va = _mm256_loadu_ps(agg_ptr.add(d));
        _mm256_storeu_ps(agg_ptr.add(d), _mm256_add_ps(va, v));
        vmin_cur = _mm256_min_ps(vmin_cur, v);
        d += 8;
    }

    let mut min_cur = hmin256_ps(vmin_cur);
    let mp2 = min_prev + p2_eff;
    while d < n {
        let c0 = prev_pad[d + 1];
        let cm = prev_pad[d] + p1;
        let cp = prev_pad[d + 2] + p1;
        let v = cv[d] + (c0.min(cm).min(cp).min(mp2) - min_prev);
        cur_pad[d + 1] = v;
        agg[d] += v;
        if v < min_cur {
            min_cur = v;
        }
        d += 1;
    }
    min_cur
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

        // Verify AVX2 sgm_dp_step_add matches scalar identically on non-multiple-of-8 num_d=129
        let num_d = 129usize;
        let cv: Vec<f32> = (0..num_d).map(|d| 0.2 + 0.5 * ((d as f32 * 0.17).sin().abs())).collect();
        let mut prev_pad = vec![f32::INFINITY; num_d + 2];
        let mut min_prev = f32::INFINITY;
        for d in 0..num_d {
            let v = 0.3 + 0.4 * ((d as f32 * 0.11).cos().abs());
            prev_pad[d + 1] = v;
            if v < min_prev { min_prev = v; }
        }
        let mut cur_simd = vec![f32::INFINITY; num_d + 2];
        let mut cur_scal = vec![f32::INFINITY; num_d + 2];
        let mut agg_simd = vec![1.0f32; num_d];
        let mut agg_scal = vec![1.0f32; num_d];
        let mn_simd = sgm_dp_step_add(&cv, &prev_pad, &mut cur_simd, &mut agg_simd, min_prev, 0.12, 0.48);
        let mn_scal = sgm_dp_step_add_scalar(&cv, &prev_pad, &mut cur_scal, &mut agg_scal, min_prev, 0.12, 0.48);
        assert!((mn_simd - mn_scal).abs() < 1e-5);
        for d in 0..num_d {
            assert!((cur_simd[d + 1] - cur_scal[d + 1]).abs() < 1e-5);
            assert!((agg_simd[d] - agg_scal[d]).abs() < 1e-5);
        }
    }
}
