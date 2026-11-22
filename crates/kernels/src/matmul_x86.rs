//! x86-64 matmul variants.
//!
//! `tile_f32`: weights packed in panels of 16 output columns laid out `[k][16]` bf16; an
//! `MR × 32` block of `y` is held in 2·MR AVX-512 f32 accumulators while `k` is walked, each step
//! widening 32 bf16 weights to f32 once and reusing them for all MR rows.

#![allow(clippy::missing_safety_doc)]

use std::arch::x86_64::*;

use crate::par::{num_threads, par_ranges, SendPtr};
use crate::registry::PackedWeight;

pub const NR: usize = 16;

/// `[n, k]` row-major → panels of 16 rows of `w` (output columns), each `[k][16]`, zero-padded.
pub fn pack_panels16(w: &[u16], n: usize, k: usize) -> Vec<u16> {
    let panels = n.div_ceil(NR);
    let mut out = vec![0u16; panels * k * NR];
    for p in 0..panels {
        let dst = &mut out[p * k * NR..(p + 1) * k * NR];
        for c in 0..NR {
            let j = p * NR + c;
            if j >= n {
                break;
            }
            let src = &w[j * k..(j + 1) * k];
            for kk in 0..k {
                dst[kk * NR + c] = src[kk];
            }
        }
    }
    out
}

#[inline(always)]
unsafe fn widen(p: *const u16) -> __m512 {
    _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(_mm256_loadu_si256(p as *const __m256i))))
}

/// `y[i0..i0+MR, j0..j0+32]` for two adjacent panels starting at `panel`.
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn micro2<const MR: usize>(x: *const f32, k: usize, wp: *const u16, y: *mut f32, n: usize, j0: usize) {
    let mut acc = [[_mm512_setzero_ps(); 2]; MR];
    let w1 = wp.add(k * NR);
    for kk in 0..k {
        let b0 = widen(wp.add(kk * NR));
        let b1 = widen(w1.add(kk * NR));
        for r in 0..MR {
            let a = _mm512_set1_ps(*x.add(r * k + kk));
            acc[r][0] = _mm512_fmadd_ps(a, b0, acc[r][0]);
            acc[r][1] = _mm512_fmadd_ps(a, b1, acc[r][1]);
        }
    }
    let m0 = mask_for(n, j0);
    let m1 = mask_for(n, j0 + NR);
    for r in 0..MR {
        _mm512_mask_storeu_ps(y.add(r * n + j0), m0, acc[r][0]);
        if m1 != 0 {
            _mm512_mask_storeu_ps(y.add(r * n + j0 + NR), m1, acc[r][1]);
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn micro1<const MR: usize>(x: *const f32, k: usize, wp: *const u16, y: *mut f32, n: usize, j0: usize) {
    let mut acc = [_mm512_setzero_ps(); MR];
    for kk in 0..k {
        let b0 = widen(wp.add(kk * NR));
        for r in 0..MR {
            acc[r] = _mm512_fmadd_ps(_mm512_set1_ps(*x.add(r * k + kk)), b0, acc[r]);
        }
    }
    let m0 = mask_for(n, j0);
    for r in 0..MR {
        _mm512_mask_storeu_ps(y.add(r * n + j0), m0, acc[r]);
    }
}

#[inline(always)]
fn mask_for(n: usize, j0: usize) -> u16 {
    if j0 >= n {
        0
    } else if n - j0 >= 16 {
        0xffff
    } else {
        ((1u32 << (n - j0)) - 1) as u16
    }
}

const MR: usize = 6;

unsafe fn rows_block(x: *const f32, m: usize, k: usize, wp: *const u16, y: *mut f32, n: usize, j0: usize, two: bool) {
    let mut i = 0;
    macro_rules! go {
        ($r:literal) => {
            if two {
                micro2::<$r>(x.add(i * k), k, wp, y.add(i * n), n, j0)
            } else {
                micro1::<$r>(x.add(i * k), k, wp, y.add(i * n), n, j0)
            }
        };
    }
    while i + MR <= m {
        go!(6);
        i += MR;
    }
    match m - i {
        0 => {}
        1 => go!(1),
        2 => go!(2),
        3 => go!(3),
        4 => go!(4),
        5 => go!(5),
        _ => unreachable!(),
    }
}

pub fn tile_f32(x: &[f32], m: usize, w: &PackedWeight, y: &mut [f32]) {
    let (n, k) = (w.n, w.k);
    assert!(x.len() >= m * k && y.len() >= m * n);
    let panels = n.div_ceil(NR);
    let pairs = panels.div_ceil(2);
    let yp = SendPtr(y.as_mut_ptr());
    let xp = SendPtr(x.as_ptr() as *mut f32);
    let wd = SendPtr(w.data.as_ptr() as *mut u16);
    let chunk = (pairs / (num_threads() * 4)).max(1);
    par_ranges(pairs, chunk, |p0, p1| {
        for pp in p0..p1 {
            let panel = pp * 2;
            let two = panel + 1 < panels;
            // SAFETY: each task writes only output columns of its own panels; packed data is
            // panels * k * NR long; x is m * k long.
            unsafe { rows_block(xp.ptr(), m, k, wd.ptr().add(panel * k * NR), yp.ptr(), n, panel * NR, two) };
        }
    });
}
