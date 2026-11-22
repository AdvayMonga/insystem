//! Row-wise and elementwise ops.

use crate::bf16_to_f32;
use crate::par::{par_ranges, SendPtr};

/// `e^x` for f32 by range reduction to `[-ln2/2, ln2/2]` and a degree-6 polynomial; relative
/// error below 1e-6 for `x` in `[-87, 88]`; inputs below -87.3 return `e^-87.3`.
#[inline(always)]
pub fn exp_fast(x: f32) -> f32 {
    let x = x.clamp(-87.3, 88.7);
    let n = (x * std::f32::consts::LOG2_E).round();
    let r = n.mul_add(-0.693_145_75, x);
    let r = n.mul_add(-1.428_606_8e-6, r);
    let p: f32 = 1.0 / 720.0;
    let p = p.mul_add(r, 1.0 / 120.0);
    let p = p.mul_add(r, 1.0 / 24.0);
    let p = p.mul_add(r, 1.0 / 6.0);
    let p = p.mul_add(r, 0.5);
    let p = p.mul_add(r, 1.0);
    let p = p.mul_add(r, 1.0);
    f32::from_bits(((n as i32 + 127) as u32) << 23) * p
}

/// `out[t] = table[ids[t]]` for a bf16 `[vocab, h]` table.
pub fn embedding(ids: &[u32], table: &[u16], h: usize, out: &mut [f32]) {
    for (t, &id) in ids.iter().enumerate() {
        let src = &table[id as usize * h..(id as usize + 1) * h];
        for (o, &s) in out[t * h..(t + 1) * h].iter_mut().zip(src) {
            *o = bf16_to_f32(s);
        }
    }
}

#[inline]
fn rms_row(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ss + eps).sqrt();
    for ((o, &v), &g) in out.iter_mut().zip(x).zip(w) {
        *o = v * r * g;
    }
}

/// RMSNorm over rows of width `w.len()`.
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let h = w.len();
    let rows = x.len() / h;
    let op = SendPtr(out.as_mut_ptr());
    par_ranges(rows, 8, |r0, r1| {
        for r in r0..r1 {
            // SAFETY: each row is written by one task.
            let o = unsafe { std::slice::from_raw_parts_mut(op.add(r * h), h) };
            rms_row(&x[r * h..(r + 1) * h], w, eps, o);
        }
    });
}

/// `resid += x; out = rmsnorm(resid)`.
pub fn add_rmsnorm(resid: &mut [f32], x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let h = w.len();
    let rows = resid.len() / h;
    let rp = SendPtr(resid.as_mut_ptr());
    let op = SendPtr(out.as_mut_ptr());
    par_ranges(rows, 8, |r0, r1| {
        for r in r0..r1 {
            // SAFETY: each row is written by one task.
            let (rr, o) = unsafe { (std::slice::from_raw_parts_mut(rp.add(r * h), h), std::slice::from_raw_parts_mut(op.add(r * h), h)) };
            for (a, &b) in rr.iter_mut().zip(&x[r * h..(r + 1) * h]) {
                *a += b;
            }
            rms_row(rr, w, eps, o);
        }
    });
}

/// In-place RMSNorm over each `head_dim` slice of rows of width `stride`, for heads
/// `[h0, h0 + n)` of each row.
pub fn head_rmsnorm(x: &mut [f32], stride: usize, h0: usize, n: usize, w: &[f32], eps: f32) {
    let d = w.len();
    let rows = x.len() / stride;
    let mut tmp = vec![0f32; d];
    for r in 0..rows {
        for hh in h0..h0 + n {
            let s = &mut x[r * stride + hh * d..r * stride + (hh + 1) * d];
            rms_row(s, w, eps, &mut tmp);
            s.copy_from_slice(&tmp);
        }
    }
}

/// Rotary embedding tables: `cos[pos * half + i]`, `sin[...]` for `half = rotary_dim / 2`.
pub struct RopeTable {
    pub half: usize,
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
}

impl RopeTable {
    pub fn new(rotary_dim: usize, theta: f32, max_pos: usize, llama3: Option<(f32, f32, f32, f32)>) -> Self {
        let half = rotary_dim / 2;
        let mut inv: Vec<f64> = (0..half).map(|i| 1.0 / (theta as f64).powf(2.0 * i as f64 / rotary_dim as f64)).collect();
        if let Some((factor, lo, hi, orig)) = llama3 {
            let (lo_wl, hi_wl) = (orig as f64 / lo as f64, orig as f64 / hi as f64);
            for f in inv.iter_mut() {
                let wl = 2.0 * std::f64::consts::PI / *f;
                if wl > lo_wl {
                    *f /= factor as f64;
                } else if wl >= hi_wl {
                    let smooth = (orig as f64 / wl - lo as f64) / (hi as f64 - lo as f64);
                    *f = (1.0 - smooth) * *f / factor as f64 + smooth * *f;
                }
            }
        }
        let mut cos = vec![0f32; max_pos * half];
        let mut sin = vec![0f32; max_pos * half];
        for p in 0..max_pos {
            for i in 0..half {
                let a = p as f64 * inv[i];
                cos[p * half + i] = a.cos() as f32;
                sin[p * half + i] = a.sin() as f32;
            }
        }
        Self { half, cos, sin }
    }
}

/// NeoX-style (rotate-half) RoPE, in place, on heads `[h0, h0 + n)` of rows of width `stride`.
pub fn rope(x: &mut [f32], stride: usize, h0: usize, n: usize, d: usize, positions: &[u32], t: &RopeTable) {
    let half = t.half;
    for (r, &p) in positions.iter().enumerate() {
        let c = &t.cos[p as usize * half..(p as usize + 1) * half];
        let s = &t.sin[p as usize * half..(p as usize + 1) * half];
        for hh in h0..h0 + n {
            let v = &mut x[r * stride + hh * d..r * stride + hh * d + 2 * half];
            for i in 0..half {
                let (a, b) = (v[i], v[i + half]);
                v[i] = a * c[i] - b * s[i];
                v[i + half] = b * c[i] + a * s[i];
            }
        }
    }
}

/// `out[t, i] = silu(gu[t, i]) * gu[t, inter + i]` for rows of `gu` of width `2 * inter`.
pub fn silu_mul(gu: &[f32], inter: usize, out: &mut [f32]) {
    let rows = gu.len() / (2 * inter);
    let op = SendPtr(out.as_mut_ptr());
    par_ranges(rows, 4, |r0, r1| {
        for r in r0..r1 {
            let g = &gu[r * 2 * inter..r * 2 * inter + inter];
            let u = &gu[r * 2 * inter + inter..(r + 1) * 2 * inter];
            // SAFETY: each row is written by one task.
            let o = unsafe { std::slice::from_raw_parts_mut(op.add(r * inter), inter) };
            for i in 0..inter {
                o[i] = g[i] / (1.0 + (-g[i]).exp()) * u[i];
            }
        }
    });
}

pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn exp_fast_accuracy() {
        let mut worst = 0f32;
        let mut x = -80.0f32;
        while x < 80.0 {
            let (a, b) = (super::exp_fast(x), x.exp());
            worst = worst.max(((a - b) / b).abs());
            x += 0.0137;
        }
        assert!(worst < 1e-6, "{worst}");
    }
}
