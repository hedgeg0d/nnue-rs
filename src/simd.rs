#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Copy and apply all changed feature rows in a single pass. The const sizes
/// let each common move shape specialize without per-lane dynamic loops.
pub fn update_i16<const R: usize, const A: usize>(
    parent: &[i16],
    removed: [&[i16]; R],
    added: [&[i16]; A],
    child: &mut [i16],
) {
    assert_eq!(parent.len(), child.len());
    for row in removed.iter().chain(added.iter()) {
        assert_eq!(row.len(), parent.len());
    }
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // All rows have child.len() elements. Full chunks and the scalar tail
        // stay within those slices; mutable child cannot alias the inputs.
        unsafe { update_i16_avx2(parent, removed, added, child) };
        return;
    }
    update_i16_scalar(parent, removed, added, child);
}

fn update_i16_scalar<const R: usize, const A: usize>(
    parent: &[i16],
    removed: [&[i16]; R],
    added: [&[i16]; A],
    child: &mut [i16],
) {
    for (i, out) in child.iter_mut().enumerate() {
        let mut value = parent[i];
        for row in removed {
            value = value.wrapping_sub(row[i]);
        }
        for row in added {
            value = value.wrapping_add(row[i]);
        }
        *out = value;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn update_i16_avx2<const R: usize, const A: usize>(
    parent: &[i16],
    removed: [&[i16]; R],
    added: [&[i16]; A],
    child: &mut [i16],
) {
    let mut i = 0;
    while i + 16 <= child.len() {
        let mut value = _mm256_loadu_si256(parent.as_ptr().add(i).cast());
        for row in removed {
            value = _mm256_sub_epi16(value, _mm256_loadu_si256(row.as_ptr().add(i).cast()));
        }
        for row in added {
            value = _mm256_add_epi16(value, _mm256_loadu_si256(row.as_ptr().add(i).cast()));
        }
        _mm256_storeu_si256(child.as_mut_ptr().add(i).cast(), value);
        i += 16;
    }
    for j in i..child.len() {
        let mut value = parent[j];
        for row in removed {
            value = value.wrapping_sub(row[j]);
        }
        for row in added {
            value = value.wrapping_add(row[j]);
        }
        child[j] = value;
    }
}

/// Four adjacent inputs are stored together for each of the sixteen outputs.
/// This layout reuses a broadcast input block across output lanes, without
/// changing the network file format or the portable row-major fallback.
pub fn pack_affine_16(weights: &[i8], inputs: usize) -> Vec<i8> {
    assert_eq!(inputs % 4, 0);
    assert_eq!(weights.len(), inputs * 16);
    let mut packed = Vec::with_capacity(weights.len());
    for block in (0..inputs).step_by(4) {
        for output in 0..16 {
            let offset = output * inputs + block;
            packed.extend_from_slice(&weights[offset..offset + 4]);
        }
    }
    packed
}

pub fn affine_16(
    input: &[u8],
    dense: &[i8],
    packed: &[i8],
    bias: &[i32],
    output: &mut [i32; 16],
) {
    assert_eq!(input.len() % 4, 0);
    assert_eq!(dense.len(), input.len() * 16);
    assert_eq!(packed.len(), dense.len());
    assert_eq!(bias.len(), 16);
    // NNUE activations are in 0..=127; this prevents saturation in maddubs.
    debug_assert!(input.iter().all(|&x| x <= 127));
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // Each four-byte input block has 64 packed weight bytes. Lengths above
        // cover every unaligned load and all sixteen bias/output lanes.
        unsafe { affine_16_avx2(input, packed, bias, output) };
        return;
    }
    affine_16_scalar(input, dense, bias, output);
}

fn affine_16_scalar(input: &[u8], dense: &[i8], bias: &[i32], output: &mut [i32; 16]) {
    for (o, value) in output.iter_mut().enumerate() {
        let row = &dense[o * input.len()..(o + 1) * input.len()];
        *value = bias[o] + dot_u8_i8_scalar(input, row);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn affine_16_avx2(input: &[u8], packed: &[i8], bias: &[i32], output: &mut [i32; 16]) {
    let ones = _mm256_set1_epi16(1);
    let mut low = _mm256_loadu_si256(bias.as_ptr().cast());
    let mut high = _mm256_loadu_si256(bias.as_ptr().add(8).cast());
    for block in (0..input.len()).step_by(4) {
        let bytes = std::ptr::read_unaligned(input.as_ptr().add(block).cast::<i32>());
        if bytes == 0 {
            continue;
        }
        let values = _mm256_set1_epi32(bytes);
        let weights = packed.as_ptr().add(block * 16);
        let w0 = _mm256_loadu_si256(weights.cast());
        let w1 = _mm256_loadu_si256(weights.add(32).cast());
        let p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(values, w0), ones);
        let p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(values, w1), ones);
        low = _mm256_add_epi32(low, p0);
        high = _mm256_add_epi32(high, p1);
    }
    _mm256_storeu_si256(output.as_mut_ptr().cast(), low);
    _mm256_storeu_si256(output.as_mut_ptr().add(8).cast(), high);
}

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

#[cfg(test)]
mod tests {
    use super::*;

    fn fused_case<const R: usize, const A: usize>(n: usize) {
        let mut seed = 42u64;
        let mut values = || -> Vec<i16> {
            (0..n + 2).map(|i| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                match i % 7 {
                    0 => i16::MIN,
                    1 => i16::MAX,
                    2 => -1,
                    3 => 0,
                    _ => (seed >> 32) as i16,
                }
            }).collect()
        };
        let parent_storage = values();
        let removed_storage: [Vec<i16>; R] = std::array::from_fn(|_| values());
        let added_storage: [Vec<i16>; A] = std::array::from_fn(|_| values());
        // Offset slices exercise unaligned loads as well as SIMD tails.
        let parent = &parent_storage[1..n + 1];
        let removed: [&[i16]; R] = std::array::from_fn(|i| &removed_storage[i][1..n + 1]);
        let added: [&[i16]; A] = std::array::from_fn(|i| &added_storage[i][1..n + 1]);
        let mut expected = parent.to_vec();
        for row in removed {
            for (out, value) in expected.iter_mut().zip(row) {
                *out = out.wrapping_sub(*value);
            }
        }
        for row in added {
            for (out, value) in expected.iter_mut().zip(row) {
                *out = out.wrapping_add(*value);
            }
        }
        let mut scalar = vec![0; n];
        update_i16_scalar(parent, removed, added, &mut scalar);
        assert_eq!(scalar, expected);
        let mut output = vec![1234; n + 2];
        update_i16(parent, removed, added, &mut output[1..n + 1]);
        assert_eq!(&output[1..n + 1], expected);
        assert_eq!(output[0], 1234);
        assert_eq!(output[n + 1], 1234);
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            output.fill(1234);
            unsafe { update_i16_avx2(parent, removed, added, &mut output[1..n + 1]); }
            assert_eq!(&output[1..n + 1], expected);
            assert_eq!(output[0], 1234);
            assert_eq!(output[n + 1], 1234);
        }
    }

    #[test]
    fn fused_updates_match_wrapping_reference() {
        for n in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 128, 256, 1024, 3072] {
            fused_case::<0, 0>(n);
            fused_case::<1, 1>(n);
            fused_case::<2, 1>(n);
            fused_case::<2, 2>(n);
            fused_case::<4, 4>(n);
        }
    }

    #[test]
    #[should_panic]
    fn fused_updates_reject_short_rows() {
        update_i16(&[0; 32], [&[0; 16][..]], [&[0; 32][..]], &mut [0; 32]);
    }

    fn check_affine(input: &[u8], weights: &[i8], bias: &[i32]) {
        let packed = pack_affine_16(weights, input.len());
        let mut expected = [0; 16];
        for o in 0..16 {
            expected[o] = bias[o] + dot_u8_i8_scalar(input, &weights[o * input.len()..(o + 1) * input.len()]);
        }
        let mut actual = [0; 16];
        affine_16_scalar(input, weights, bias, &mut actual);
        assert_eq!(actual, expected);
        affine_16(input, weights, &packed, bias, &mut actual);
        assert_eq!(actual, expected);
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            // Explicitly offset both inputs and packed weights for loadu coverage.
            let mut inp = vec![0];
            inp.extend_from_slice(input);
            let mut rows = vec![0];
            rows.extend_from_slice(&packed);
            unsafe { affine_16_avx2(&inp[1..], &rows[1..], bias, &mut actual); }
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn packed_affine_matches_dense_at_all_sparsities() {
        let mut seed = 42u64;
        let bias: Vec<i32> = (0..16).map(|i| i * 137 - 1000).collect();
        for n in [0, 4, 16, 32, 128, 256, 1024, 3072] {
            let weights: Vec<i8> = (0..n * 16).map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                (seed >> 32) as i8
            }).collect();
            for sparsity in [0, 1, 2, 4, 8, 16, 32] {
                let input: Vec<u8> = (0..n).map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    if sparsity == 0 || seed as usize % sparsity != 0 { 0 }
                    else { (seed >> 32) as u8 & 127 }
                }).collect();
                check_affine(&input, &weights, &bias);
            }
            check_affine(&vec![127; n], &vec![i8::MIN; n * 16], &bias);
            check_affine(&vec![127; n], &vec![i8::MAX; n * 16], &bias);
            for offset in 0..n.min(128) {
                let mut one = vec![0; n];
                one[offset] = 127;
                check_affine(&one, &weights, &bias);
            }
        }
    }

    #[test]
    #[should_panic]
    fn affine_rejects_short_packed_weights() {
        affine_16(&[0; 16], &[0; 256], &[0; 128], &[0; 16], &mut [0; 16]);
    }
}
