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
    let mut acc = _mm256_setzero_si256();
    let mut i = 0;
    while i + 32 <= n {
        let a = _mm256_loadu_si256(input.as_ptr().add(i) as *const __m256i);
        let w = _mm256_loadu_si256(weights.as_ptr().add(i) as *const __m256i);
        let prod = _mm256_maddubs_epi16(a, w);
        let wide = _mm256_madd_epi16(prod, ones);
        acc = _mm256_add_epi32(acc, wide);
        i += 32;
    }
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
    let mut i = 0;
    while i + 16 <= n {
        let a = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
        let b = _mm256_loadu_si256(w.as_ptr().add(i) as *const __m256i);
        let r = _mm256_add_epi16(a, b);
        _mm256_storeu_si256(acc.as_mut_ptr().add(i) as *mut __m256i, r);
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
    let mut i = 0;
    while i + 16 <= n {
        let a = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
        let b = _mm256_loadu_si256(w.as_ptr().add(i) as *const __m256i);
        let r = _mm256_sub_epi16(a, b);
        _mm256_storeu_si256(acc.as_mut_ptr().add(i) as *mut __m256i, r);
        i += 16;
    }
    while i < n {
        acc[i] = acc[i].wrapping_sub(w[i]);
        i += 1;
    }
}
