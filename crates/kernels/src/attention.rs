//! Paged causal attention over a ragged batch.
//!
//! K/V cache per layer: `[num_blocks, block_size, n_kv_heads, head_dim]` f32. Each sequence's
//! new tokens sit at the end of its context; new token `t` of a sequence with `ctx_len` tokens
//! and `q_len` new tokens is at position `ctx_len - q_len + t` and sees keys up to and including
//! that position (and, with a sliding window, not before `pos - window + 1`).

use crate::par::{par_for, SendPtr};

pub struct AttnSeq<'a> {
    pub q_start: usize,
    pub q_len: usize,
    pub ctx_len: usize,
    pub block_table: &'a [u32],
}

pub struct AttnArgs<'a> {
    /// `[T, n_heads, head_dim]`
    pub q: &'a [f32],
    pub k_cache: &'a [f32],
    pub v_cache: &'a [f32],
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
        let q = &a.q[(row * h + head) * d..(row * h + head + 1) * d];
        let mut scores = Vec::with_capacity(pos + 1 - lo);
        let mut mx = f32::NEG_INFINITY;
        for p in lo..=pos {
            let slot = s.block_table[p / bs] as usize * bs + p % bs;
            let k = &a.k_cache[(slot * kvh + kv_head) * d..(slot * kvh + kv_head + 1) * d];
            let sc = crate::matmul::dot(q, k) * a.scale;
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
            let v = &a.v_cache[(slot * kvh + kv_head) * d..(slot * kvh + kv_head + 1) * d];
            let w = scores[i] * inv;
            for (oo, vv) in o.iter_mut().zip(v) {
                *oo += w * vv;
            }
        }
        // SAFETY: each (row, head) output is written by exactly one task.
        unsafe { std::ptr::copy_nonoverlapping(o.as_ptr(), a.out.add((row * h + head) * d), d) };
    });
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
        let (h, kvh, d, bs) = (4, 2, 32, 4);
        for &(q_len, ctx) in &[(1usize, 1usize), (1, 9), (5, 5), (3, 11), (16, 37)] {
            let q = rand_vec(q_len * h * d, 3);
            let k = rand_vec(ctx.div_ceil(bs) * bs * kvh * d, 4);
            let v = rand_vec(ctx.div_ceil(bs) * bs * kvh * d, 5);
            let bt: Vec<u32> = (0..ctx.div_ceil(bs) as u32).collect();
            let want = dense(&q, &k, &v, q_len, ctx, h, kvh, d, 0.17);
            for kern in ATTENTION_KERNELS {
                let mut out = vec![0f32; q_len * h * d];
                let seqs = [AttnSeq { q_start: 0, q_len, ctx_len: ctx, block_table: &bt }];
                (kern.run)(&AttnArgs {
                    q: &q,
                    k_cache: &k,
                    v_cache: &v,
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
