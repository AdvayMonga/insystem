//! Paged causal attention over a ragged batch.
//!
//! K/V cache per layer: `[num_blocks, n_kv_heads, block_size, head_dim]` bf16, so one kv head's
//! keys within a block are one contiguous `block_size * head_dim` run. Each sequence's
//! new tokens sit at the end of its context; new token `t` of a sequence with `ctx_len` tokens
//! and `q_len` new tokens is at position `ctx_len - q_len + t` and sees keys up to and including
//! that position (and, with a sliding window, not before `pos - window + 1`).

use crate::bf16_to_f32;
use crate::matmul::dot_bf16;
use crate::par::{par_for, SendPtr};

/// Offset of kv head `g` of flat slot `slot` (block * block_size + offset) in a K or V cache.
#[inline(always)]
pub fn kv_offset(slot: usize, g: usize, kvh: usize, d: usize, bs: usize) -> usize {
    ((slot / bs * kvh + g) * bs + slot % bs) * d
}

pub struct AttnSeq<'a> {
    pub q_start: usize,
    pub q_len: usize,
    pub ctx_len: usize,
    pub block_table: &'a [u32],
}

pub struct AttnArgs<'a> {
    /// Row `r`, head `h` is at `q[r * q_stride + h * head_dim..]`.
    pub q: &'a [f32],
    pub q_stride: usize,
    pub k_cache: &'a [u16],
    pub v_cache: &'a [u16],
    /// `[T, n_heads, head_dim]`
    pub out: SendPtr<f32>,
    pub seqs: &'a [AttnSeq<'a>],
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub block_size: usize,
    pub scale: f32,
    pub window: Option<usize>,
}

pub fn naive(a: &AttnArgs) {
    let (h, kvh, d, bs) = (a.n_heads, a.n_kv_heads, a.head_dim, a.block_size);
    let group = h / kvh;
    // Flatten (sequence, token, head) into one task index.
    let mut starts = Vec::with_capacity(a.seqs.len() + 1);
    let mut acc = 0;
    for s in a.seqs {
        starts.push(acc);
        acc += s.q_len * h;
    }
    starts.push(acc);
    par_for(acc, |task| {
        let si = starts.partition_point(|&x| x <= task) - 1;
        let s = &a.seqs[si];
        let local = task - starts[si];
        let (t, head) = (local / h, local % h);
        let kv_head = head / group;
        let pos = s.ctx_len - s.q_len + t;
        let lo = a.window.map(|w| (pos + 1).saturating_sub(w)).unwrap_or(0);
        let row = s.q_start + t;
        let q = &a.q[row * a.q_stride + head * d..row * a.q_stride + (head + 1) * d];
        let mut scores = Vec::with_capacity(pos + 1 - lo);
        let mut mx = f32::NEG_INFINITY;
        for p in lo..=pos {
            let slot = s.block_table[p / bs] as usize * bs + p % bs;
            let off = kv_offset(slot, kv_head, kvh, d, bs);
            let k = &a.k_cache[off..off + d];
            let sc = dot_bf16(q, k) * a.scale;
            mx = mx.max(sc);
            scores.push(sc);
        }
        let mut sum = 0.0;
        for sc in scores.iter_mut() {
            *sc = (*sc - mx).exp();
            sum += *sc;
        }
        let inv = 1.0 / sum;
        let mut o = vec![0f32; d];
        for (i, p) in (lo..=pos).enumerate() {
            let slot = s.block_table[p / bs] as usize * bs + p % bs;
            let off = kv_offset(slot, kv_head, kvh, d, bs);
            let v = &a.v_cache[off..off + d];
            let w = scores[i] * inv;
            for (oo, &vv) in o.iter_mut().zip(v) {
                *oo += w * bf16_to_f32(vv);
            }
        }
        // SAFETY: each (row, head) output is written by exactly one task.
        unsafe { std::ptr::copy_nonoverlapping(o.as_ptr(), a.out.add((row * h + head) * d), d) };
    });
}

/// Query tokens per task.
pub const QT: usize = 16;
const MAX_GROUP: usize = 8;
const MAX_D: usize = 256;

/// One task per (sequence, kv head, chunk of up to 16 query tokens). Each K/V row is loaded once
/// and used by every query head of the group and every query token of the chunk; softmax is
/// computed online one KV block at a time; no heap allocation.
pub fn flash(a: &AttnArgs) {
    let (h, kvh, d, bs) = (a.n_heads, a.n_kv_heads, a.head_dim, a.block_size);
    let group = h / kvh;
    assert!(group <= MAX_GROUP && d <= MAX_D && bs <= 64);
    let mut tasks: Vec<(u32, u16, u32)> = Vec::new();
    for (si, s) in a.seqs.iter().enumerate() {
        for g in 0..kvh {
            for t0 in (0..s.q_len).step_by(QT) {
                tasks.push((si as u32, g as u16, t0 as u32));
            }
        }
    }
    par_for(tasks.len(), |ti| {
        let (si, g, t0) = tasks[ti];
        flash_task(a, &a.seqs[si as usize], g as usize, t0 as usize, group, d, bs);
    });
}

#[inline]
fn flash_task(a: &AttnArgs, s: &AttnSeq, g: usize, t0: usize, group: usize, d: usize, bs: usize) {
    let kvh = a.n_kv_heads;
    let nt = (s.q_len - t0).min(QT);
    let rows = nt * group;
    let first_pos = s.ctx_len - s.q_len + t0;
    let last_pos = first_pos + nt - 1;
    let lo = a.window.map(|w| (first_pos + 1).saturating_sub(w)).unwrap_or(0);
    let mut acc = [[0f32; MAX_D]; QT * MAX_GROUP];
    let mut mx = [f32::NEG_INFINITY; QT * MAX_GROUP];
    let mut sum = [0f32; QT * MAX_GROUP];
    let mut sc = [[0f32; 64]; QT * MAX_GROUP];
    let qrow = |t: usize, j: usize| {
        let r = s.q_start + t0 + t;
        let hh = g * group + j;
        &a.q[r * a.q_stride + hh * d..r * a.q_stride + (hh + 1) * d]
    };
    let mut blk = lo / bs;
    while blk * bs <= last_pos {
        let base = s.block_table[blk] as usize * bs;
        let p0 = blk * bs;
        let n_keys = (last_pos + 1 - p0).min(bs);
        // Scores for this block.
        for t in 0..nt {
            let pos = first_pos + t;
            let wlo = a.window.map(|w| (pos + 1).saturating_sub(w)).unwrap_or(0);
            for j in 0..group {
                let q = qrow(t, j);
                let ri = t * group + j;
                let mut bmax = f32::NEG_INFINITY;
                for kk in 0..n_keys {
                    let p = p0 + kk;
                    let v = if p > pos || p < wlo {
                        f32::NEG_INFINITY
                    } else {
                        let off = kv_offset(base + kk, g, kvh, d, bs);
                        dot_bf16(q, &a.k_cache[off..off + d]) * a.scale
                    };
                    sc[ri][kk] = v;
                    bmax = bmax.max(v);
                }
                if bmax == f32::NEG_INFINITY {
                    sc[ri][..n_keys].fill(0.0);
                    continue;
                }
                let new_m = mx[ri].max(bmax);
                let corr = (mx[ri] - new_m).exp();
                if corr != 1.0 {
                    for x in acc[ri][..d].iter_mut() {
                        *x *= corr;
                    }
                    sum[ri] *= corr;
                }
                mx[ri] = new_m;
                for kk in 0..n_keys {
                    sc[ri][kk] = (sc[ri][kk] - new_m).exp();
                    sum[ri] += sc[ri][kk];
                }
            }
        }
        // Weighted V, each V row loaded once for all rows.
        for kk in 0..n_keys {
            let off = kv_offset(base + kk, g, kvh, d, bs);
            let v = &a.v_cache[off..off + d];
            for ri in 0..rows {
                let w = sc[ri][kk];
                if w == 0.0 || mx[ri] == f32::NEG_INFINITY {
                    continue;
                }
                for (o, &vv) in acc[ri][..d].iter_mut().zip(v) {
                    *o = w.mul_add(bf16_to_f32(vv), *o);
                }
            }
        }
        blk += 1;
    }
    for t in 0..nt {
        let r = s.q_start + t0 + t;
        for j in 0..group {
            let ri = t * group + j;
            let inv = 1.0 / sum[ri];
            let hh = g * group + j;
            // SAFETY: each (row, head) is written by exactly one task.
            let o = unsafe { std::slice::from_raw_parts_mut(a.out.add((r * a.n_heads + hh) * d), d) };
            for (x, &y) in o.iter_mut().zip(&acc[ri][..d]) {
                *x = y * inv;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matmul::tests::rand_vec;
    use crate::registry::ATTENTION_KERNELS;

    /// Dense reference: one sequence stored contiguously in blocks 0..n.
    fn dense(q: &[f32], k: &[f32], v: &[f32], q_len: usize, ctx: usize, h: usize, kvh: usize, d: usize, scale: f32) -> Vec<f32> {
        let mut out = vec![0f32; q_len * h * d];
        for t in 0..q_len {
            let pos = ctx - q_len + t;
            for head in 0..h {
                let kh = head / (h / kvh);
                let qv = &q[(t * h + head) * d..][..d];
                let sc: Vec<f32> = (0..=pos).map(|p| qv.iter().zip(&k[(p * kvh + kh) * d..][..d]).map(|(a, b)| a * b).sum::<f32>() * scale).collect();
                let m = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = sc.iter().map(|x| (x - m).exp()).collect();
                let s: f32 = e.iter().sum();
                for p in 0..=pos {
                    for i in 0..d {
                        out[(t * h + head) * d + i] += e[p] / s * v[(p * kvh + kh) * d + i];
                    }
                }
            }
        }
        out
    }

    #[test]
    fn kernels_match_dense() {
        let (h, kvh, d, bs) = (4, 2, 64, 4);
        for &(q_len, ctx) in &[(1usize, 1usize), (1, 9), (5, 5), (3, 11), (16, 37)] {
            let q = rand_vec(q_len * h * d, 3);
            let round = |v: Vec<f32>| -> Vec<f32> { v.into_iter().map(|x| bf16_to_f32(crate::f32_to_bf16(x))).collect() };
            let k = round(rand_vec(ctx.div_ceil(bs) * bs * kvh * d, 4));
            let v = round(rand_vec(ctx.div_ceil(bs) * bs * kvh * d, 5));
            let bt: Vec<u32> = (0..ctx.div_ceil(bs) as u32).collect();
            let want = dense(&q, &k, &v, q_len, ctx, h, kvh, d, 0.17);
            let to_paged = |x: &[f32]| {
                let mut out = vec![0u16; x.len()];
                for slot in 0..ctx.div_ceil(bs) * bs {
                    for g in 0..kvh {
                        let o = kv_offset(slot, g, kvh, d, bs);
                        for (dst, &src) in out[o..o + d].iter_mut().zip(&x[(slot * kvh + g) * d..(slot * kvh + g + 1) * d]) {
                            *dst = crate::f32_to_bf16(src);
                        }
                    }
                }
                out
            };
            let (kp, vp) = (to_paged(&k), to_paged(&v));
            for kern in ATTENTION_KERNELS {
                let mut out = vec![0f32; q_len * h * d];
                let seqs = [AttnSeq { q_start: 0, q_len, ctx_len: ctx, block_table: &bt }];
                (kern.run)(&AttnArgs {
                    q: &q,
                    q_stride: h * d,
                    k_cache: &kp,
                    v_cache: &vp,
                    out: SendPtr(out.as_mut_ptr()),
                    seqs: &seqs,
                    n_heads: h,
                    n_kv_heads: kvh,
                    head_dim: d,
                    block_size: bs,
                    scale: 0.17,
                    window: None,
                });
                for (a, b) in want.iter().zip(&out) {
                    assert!((a - b).abs() < 1e-4, "{}: q_len={q_len} ctx={ctx}: {a} vs {b}", kern.name);
                }
            }
        }
    }
}
