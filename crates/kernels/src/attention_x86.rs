//! AVX-512 paged attention.
//!
//! Same tasks as [`crate::attention::flash`]: one per (sequence, kv head, up to 16 query
//! tokens). Per KV block, each query row is held in registers while its scores against the block's
//! keys are computed; then each row's output accumulator is held in registers while the block's
//! values are folded in. K/V are bf16, widened to f32 as they are loaded. Scratch is a per-thread buffer sized to the rows in use.

use std::arch::x86_64::*;
use std::cell::RefCell;

use crate::attention::{AttnArgs, AttnSeq, QT};
use crate::ops::exp_fast;
use crate::par::par_for;

#[inline(always)]
unsafe fn ld(p: *const u16) -> __m512 {
    _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(_mm256_loadu_si256(p as *const __m256i))))
}

pub fn flash_avx512(a: &AttnArgs) {
    match a.head_dim {
        64 => run::<4>(a),
        128 => run::<8>(a),
        256 => run::<16>(a),
        _ => crate::attention::flash(a),
    }
}

thread_local! {
    static SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

fn run<const NV: usize>(a: &AttnArgs) {
    let group = a.n_heads / a.n_kv_heads;
    let mut tasks: Vec<(u32, u16, u32)> = Vec::new();
    for (si, s) in a.seqs.iter().enumerate() {
        for g in 0..a.n_kv_heads {
            for t0 in (0..s.q_len).step_by(QT) {
                tasks.push((si as u32, g as u16, t0 as u32));
            }
        }
    }
    par_for(tasks.len(), |ti| {
        let (si, g, t0) = tasks[ti];
        SCRATCH.with_borrow_mut(|buf| {
            // SAFETY: AVX-512F is enabled for this build (target-cpu=native on a machine with it).
            unsafe { task::<NV>(a, &a.seqs[si as usize], g as usize, t0 as usize, group, buf) }
        });
    });
}

#[target_feature(enable = "avx512f")]
unsafe fn task<const NV: usize>(a: &AttnArgs, s: &AttnSeq, g: usize, t0: usize, group: usize, buf: &mut Vec<f32>) {
    let d = NV * 16;
    let (kvh, bs) = (a.n_kv_heads, a.block_size);
    let nt = (s.q_len - t0).min(QT);
    let rows = nt * group;
    let first_pos = s.ctx_len - s.q_len + t0;
    let last_pos = first_pos + nt - 1;
    let need = rows * (d + bs + 2);
    if buf.len() < need {
        buf.resize(need, 0.0);
    }
    let (acc, rest) = buf.split_at_mut(rows * d);
    let (p, rest) = rest.split_at_mut(rows * bs);
    let (mx, rest) = rest.split_at_mut(rows);
    let l = &mut rest[..rows];
    acc.fill(0.0);
    mx.fill(f32::NEG_INFINITY);
    l.fill(0.0);
    let lo = a.window.map(|w| (first_pos + 1).saturating_sub(w)).unwrap_or(0);
    let row_q = |r: usize| -> *const f32 {
        let (t, j) = (r / group, r % group);
        a.q.as_ptr().add((s.q_start + t0 + t) * a.q_stride + (g * group + j) * d)
    };
    let scale = _mm512_set1_ps(a.scale);
    let mut blk = lo / bs;
    while blk * bs <= last_pos {
        let phys = s.block_table[blk] as usize;
        let kb = a.k_cache.as_ptr().add((phys * kvh + g) * bs * d);
        let vb = a.v_cache.as_ptr().add((phys * kvh + g) * bs * d);
        let p0 = blk * bs;
        let n_keys = (last_pos + 1 - p0).min(bs);
        for r in 0..rows {
            let pos = first_pos + r / group;
            let pr = &mut p[r * bs..r * bs + bs];
            let kend = if pos >= p0 { (pos - p0 + 1).min(n_keys) } else { 0 };
            let kstart = a.window.map(|w| (pos + 1).saturating_sub(w).saturating_sub(p0)).unwrap_or(0).min(kend);
            pr.fill(0.0);
            if kstart == kend {
                continue;
            }
            let qp = row_q(r);
            let mut q = [_mm512_setzero_ps(); NV];
            for i in 0..NV {
                q[i] = _mm512_mul_ps(_mm512_loadu_ps(qp.add(16 * i)), scale);
            }
            let mut bmax = f32::NEG_INFINITY;
            for kk in kstart..kend {
                let kp = kb.add(kk * d);
                let mut s0 = _mm512_mul_ps(q[0], ld(kp));
                let mut s1 = _mm512_mul_ps(q[1 % NV], ld(kp.add(16 * (1 % NV))));
                if NV == 1 {
                    s1 = _mm512_setzero_ps();
                }
                let mut i = 2;
                while i + 1 < NV {
                    s0 = _mm512_fmadd_ps(q[i], ld(kp.add(16 * i)), s0);
                    s1 = _mm512_fmadd_ps(q[i + 1], ld(kp.add(16 * (i + 1))), s1);
                    i += 2;
                }
                let sc = _mm512_reduce_add_ps(_mm512_add_ps(s0, s1));
                pr[kk] = sc;
                bmax = bmax.max(sc);
            }
            let new_m = mx[r].max(bmax);
            let corr = exp_fast(mx[r] - new_m);
            mx[r] = new_m;
            if corr != 1.0 {
                l[r] *= corr;
                let c = _mm512_set1_ps(corr);
                let ar = acc.as_mut_ptr().add(r * d);
                for i in 0..NV {
                    _mm512_storeu_ps(ar.add(16 * i), _mm512_mul_ps(_mm512_loadu_ps(ar.add(16 * i)), c));
                }
            }
            let mut sum = 0.0;
            for e in pr[kstart..kend].iter_mut() {
                *e = exp_fast(*e - new_m);
                sum += *e;
            }
            l[r] += sum;
            // Fold this block's values into the row's accumulator.
            let ar = acc.as_mut_ptr().add(r * d);
            let mut o = [_mm512_setzero_ps(); NV];
            for i in 0..NV {
                o[i] = _mm512_loadu_ps(ar.add(16 * i));
            }
            for kk in kstart..kend {
                let w = _mm512_set1_ps(pr[kk]);
                let vp = vb.add(kk * d);
                for i in 0..NV {
                    o[i] = _mm512_fmadd_ps(w, ld(vp.add(16 * i)), o[i]);
                }
            }
            for i in 0..NV {
                _mm512_storeu_ps(ar.add(16 * i), o[i]);
            }
        }
        blk += 1;
    }
    for r in 0..rows {
        let (t, j) = (r / group, r % group);
        let row = s.q_start + t0 + t;
        let inv = _mm512_set1_ps(1.0 / l[r]);
        let dst = a.out.add((row * a.n_heads + g * group + j) * d);
        let ar = acc.as_ptr().add(r * d);
        for i in 0..NV {
            _mm512_storeu_ps(dst.add(16 * i), _mm512_mul_ps(_mm512_loadu_ps(ar.add(16 * i)), inv));
        }
    }
}
