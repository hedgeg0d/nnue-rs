#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn dot_u8_i8(input: &[u8], weights: &[i8]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { dot_u8_i8_avx2(input, weights) };
        }
    }
    dot_u8_i8_scalar(input, weights)
}

fn dot_u8_i8_scalar(input: &[u8], weights: &[i8]) -> i32 {
    let mut sum = 0i32;
    for (a, w) in input.iter().zip(weights) {
        sum += *a as i32 * *w as i32;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_u8_i8_avx2(input: &[u8], weights: &[i8]) -> i32 {
    let n = input.len();
    let ones = _mm256_set1_epi16(1);
    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let mut acc2 = _mm256_setzero_si256();
    let mut acc3 = _mm256_setzero_si256();
    let inp = input.as_ptr();
    let wgt = weights.as_ptr();

    let mut i = 0;
    while i + 128 <= n {
        macro_rules! block {
            ($acc:ident, $off:expr) => {{
                let a = _mm256_loadu_si256(inp.add(i + $off) as *const __m256i);
                let w = _mm256_loadu_si256(wgt.add(i + $off) as *const __m256i);
                let wide = _mm256_madd_epi16(_mm256_maddubs_epi16(a, w), ones);
                $acc = _mm256_add_epi32($acc, wide);
            }};
        }
        block!(acc0, 0);
        block!(acc1, 32);
        block!(acc2, 64);
        block!(acc3, 96);
        i += 128;
    }
    while i + 32 <= n {
        let a = _mm256_loadu_si256(inp.add(i) as *const __m256i);
        let w = _mm256_loadu_si256(wgt.add(i) as *const __m256i);
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_maddubs_epi16(a, w), ones));
        i += 32;
    }

    let acc = _mm256_add_epi32(_mm256_add_epi32(acc0, acc1), _mm256_add_epi32(acc2, acc3));
    let lo = _mm256_castsi256_si128(acc);
    let hi = _mm256_extracti128_si256(acc, 1);
    let mut s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_srli_si128(s, 8));
    s = _mm_add_epi32(s, _mm_srli_si128(s, 4));
    let mut sum = _mm_cvtsi128_si32(s);
    while i < n {
        sum += input[i] as i32 * weights[i] as i32;
        i += 1;
    }
    sum
}

/// Computes `out[j] = (clamp(a[j],0,hi) * clamp(b[j],0,hi)) >> shift` as `u8`,
/// the pairwise feature-transformer activation used by SFNNv5 networks.
pub fn pairwise_clip_mul(a: &[i16], b: &[i16], out: &mut [u8], hi: i16, shift: i32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { pairwise_clip_mul_avx2(a, b, out, hi, shift) };
            return;
        }
    }
    pairwise_clip_mul_scalar(a, b, out, hi, shift)
}

fn pairwise_clip_mul_scalar(a: &[i16], b: &[i16], out: &mut [u8], hi: i16, shift: i32) {
    let hi = hi as i32;
    for j in 0..out.len() {
        let s0 = (a[j] as i32).clamp(0, hi);
        let s1 = (b[j] as i32).clamp(0, hi);
        out[j] = ((s0 * s1) as u32 >> shift) as u8;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pairwise_clip_mul_avx2(a: &[i16], b: &[i16], out: &mut [u8], hi: i16, shift: i32) {
    let m = out.len();
    let lo = _mm256_setzero_si256();
    let hiv = _mm256_set1_epi16(hi);
    let cnt = _mm_cvtsi32_si128(shift);
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let op = out.as_mut_ptr();

    let clip_mul = |off: usize| -> __m256i {
        let mut x = _mm256_loadu_si256(ap.add(off) as *const __m256i);
        let mut y = _mm256_loadu_si256(bp.add(off) as *const __m256i);
        x = _mm256_min_epi16(_mm256_max_epi16(x, lo), hiv);
        y = _mm256_min_epi16(_mm256_max_epi16(y, lo), hiv);
        _mm256_srl_epi16(_mm256_mullo_epi16(x, y), cnt)
    };

    let mut j = 0;
    while j + 32 <= m {
        let r0 = clip_mul(j);
        let r1 = clip_mul(j + 16);
        let packed = _mm256_permute4x64_epi64(_mm256_packus_epi16(r0, r1), 0xD8);
        _mm256_storeu_si256(op.add(j) as *mut __m256i, packed);
        j += 32;
    }
    let hi = hi as i32;
    while j < m {
        let s0 = (a[j] as i32).clamp(0, hi);
        let s1 = (b[j] as i32).clamp(0, hi);
        out[j] = ((s0 * s1) as u32 >> shift) as u8;
        j += 1;
    }
}

/// Computes `out[j] = clamp(a[j], 0, 127)` as `u8`, the clipped-ReLU
/// feature-transformer activation used by HalfKP and HalfKAv2 networks.
pub fn clip_u8(a: &[i16], out: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { clip_u8_avx2(a, out) };
            return;
        }
    }
    for (o, &x) in out.iter_mut().zip(a) {
        *o = (x as i32).clamp(0, 127) as u8;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn clip_u8_avx2(a: &[i16], out: &mut [u8]) {
    let m = out.len();
    let lo = _mm256_setzero_si256();
    let hiv = _mm256_set1_epi16(127);
    let ap = a.as_ptr();
    let op = out.as_mut_ptr();

    let clip = |off: usize| -> __m256i {
        let x = _mm256_loadu_si256(ap.add(off) as *const __m256i);
        _mm256_min_epi16(_mm256_max_epi16(x, lo), hiv)
    };

    let mut j = 0;
    while j + 32 <= m {
        let packed = _mm256_permute4x64_epi64(_mm256_packus_epi16(clip(j), clip(j + 16)), 0xD8);
        _mm256_storeu_si256(op.add(j) as *mut __m256i, packed);
        j += 32;
    }
    while j < m {
        *out.get_unchecked_mut(j) = (*a.get_unchecked(j) as i32).clamp(0, 127) as u8;
        j += 1;
    }
}

pub fn add_i8_i16(acc: &mut [i16], w: &[i8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { add_i8_i16_avx2(acc, w) };
            return;
        }
    }
    for (a, &wi) in acc.iter_mut().zip(w) {
        *a = a.wrapping_add(wi as i16);
    }
}

pub fn sub_i8_i16(acc: &mut [i16], w: &[i8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { sub_i8_i16_avx2(acc, w) };
            return;
        }
    }
    for (a, &wi) in acc.iter_mut().zip(w) {
        *a = a.wrapping_sub(wi as i16);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_i8_i16_avx2(acc: &mut [i16], w: &[i8]) {
    let n = acc.len();
    let ap = acc.as_mut_ptr();
    let wp = w.as_ptr();
    let mut i = 0;
    while i + 32 <= n {
        let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i) as *const __m128i));
        let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i + 16) as *const __m128i));
        let a0 = _mm256_loadu_si256(ap.add(i) as *const __m256i);
        let a1 = _mm256_loadu_si256(ap.add(i + 16) as *const __m256i);
        _mm256_storeu_si256(ap.add(i) as *mut __m256i, _mm256_add_epi16(a0, w0));
        _mm256_storeu_si256(ap.add(i + 16) as *mut __m256i, _mm256_add_epi16(a1, w1));
        i += 32;
    }
    while i < n {
        acc[i] = acc[i].wrapping_add(w[i] as i16);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sub_i8_i16_avx2(acc: &mut [i16], w: &[i8]) {
    let n = acc.len();
    let ap = acc.as_mut_ptr();
    let wp = w.as_ptr();
    let mut i = 0;
    while i + 32 <= n {
        let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i) as *const __m128i));
        let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i + 16) as *const __m128i));
        let a0 = _mm256_loadu_si256(ap.add(i) as *const __m256i);
        let a1 = _mm256_loadu_si256(ap.add(i + 16) as *const __m256i);
        _mm256_storeu_si256(ap.add(i) as *mut __m256i, _mm256_sub_epi16(a0, w0));
        _mm256_storeu_si256(ap.add(i + 16) as *mut __m256i, _mm256_sub_epi16(a1, w1));
        i += 32;
    }
    while i < n {
        acc[i] = acc[i].wrapping_sub(w[i] as i16);
        i += 1;
    }
}

pub fn add_i16(acc: &mut [i16], w: &[i16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { add_i16_avx2(acc, w) };
            return;
        }
    }
    for (a, &wi) in acc.iter_mut().zip(w) {
        *a = a.wrapping_add(wi);
    }
}

pub fn sub_i16(acc: &mut [i16], w: &[i16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { sub_i16_avx2(acc, w) };
            return;
        }
    }
    for (a, &wi) in acc.iter_mut().zip(w) {
        *a = a.wrapping_sub(wi);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_i16_avx2(acc: &mut [i16], w: &[i16]) {
    let n = acc.len();
    let ap = acc.as_mut_ptr();
    let wp = w.as_ptr();
    let mut i = 0;
    while i + 32 <= n {
        let a0 = _mm256_loadu_si256(ap.add(i) as *const __m256i);
        let a1 = _mm256_loadu_si256(ap.add(i + 16) as *const __m256i);
        let b0 = _mm256_loadu_si256(wp.add(i) as *const __m256i);
        let b1 = _mm256_loadu_si256(wp.add(i + 16) as *const __m256i);
        _mm256_storeu_si256(ap.add(i) as *mut __m256i, _mm256_add_epi16(a0, b0));
        _mm256_storeu_si256(ap.add(i + 16) as *mut __m256i, _mm256_add_epi16(a1, b1));
        i += 32;
    }
    while i + 16 <= n {
        let a = _mm256_loadu_si256(ap.add(i) as *const __m256i);
        let b = _mm256_loadu_si256(wp.add(i) as *const __m256i);
        _mm256_storeu_si256(ap.add(i) as *mut __m256i, _mm256_add_epi16(a, b));
        i += 16;
    }
    while i < n {
        acc[i] = acc[i].wrapping_add(w[i]);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sub_i16_avx2(acc: &mut [i16], w: &[i16]) {
    let n = acc.len();
    let ap = acc.as_mut_ptr();
    let wp = w.as_ptr();
    let mut i = 0;
    while i + 32 <= n {
        let a0 = _mm256_loadu_si256(ap.add(i) as *const __m256i);
        let a1 = _mm256_loadu_si256(ap.add(i + 16) as *const __m256i);
        let b0 = _mm256_loadu_si256(wp.add(i) as *const __m256i);
        let b1 = _mm256_loadu_si256(wp.add(i + 16) as *const __m256i);
        _mm256_storeu_si256(ap.add(i) as *mut __m256i, _mm256_sub_epi16(a0, b0));
        _mm256_storeu_si256(ap.add(i + 16) as *mut __m256i, _mm256_sub_epi16(a1, b1));
        i += 32;
    }
    while i + 16 <= n {
        let a = _mm256_loadu_si256(ap.add(i) as *const __m256i);
        let b = _mm256_loadu_si256(wp.add(i) as *const __m256i);
        _mm256_storeu_si256(ap.add(i) as *mut __m256i, _mm256_sub_epi16(a, b));
        i += 16;
    }
    while i < n {
        acc[i] = acc[i].wrapping_sub(w[i]);
        i += 1;
    }
}
