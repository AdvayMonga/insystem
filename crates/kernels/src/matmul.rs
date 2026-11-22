//! Matmul variants: `y[m, n] = x[m, k] · wᵀ` with bf16 `w` and f32 `x`, `y`.

use crate::bf16_to_f32;
use crate::registry::PackedWeight;

pub fn pack_rowmajor(w: &[u16], _n: usize, _k: usize) -> Vec<u16> {
    w.to_vec()
}

pub fn naive(x: &[f32], m: usize, w: &PackedWeight, y: &mut [f32]) {
    let (n, k) = (w.n, w.k);
    assert!(x.len() >= m * k && y.len() >= m * n);
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += x[i * k + p] * bf16_to_f32(w.data[j * k + p]);
            }
            y[i * n + j] = acc;
        }
    }
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    let ca = a.chunks_exact(16);
    let cb = b.chunks_exact(16);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for l in 0..16 {
            acc[l] = x[l].mul_add(y[l], acc[l]);
        }
    }
    let mut s = 0.0;
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s + acc.iter().sum::<f32>()
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::registry::{MatmulKernel, MATMUL_KERNELS};
    use crate::f32_to_bf16;

    pub fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect()
    }

    pub fn check_kernel(kern: &MatmulKernel) {
        let reference = &MATMUL_KERNELS[0];
        for &(m, n, k) in &[(1, 64, 128), (3, 48, 96), (17, 256, 512), (64, 96, 2048), (5, 33, 70)] {
            let x = rand_vec(m * k, 1);
            let wf = rand_vec(n * k, 2);
            let wb: Vec<u16> = wf.iter().map(|&v| f32_to_bf16(v)).collect();
            let pr = PackedWeight::new(reference, &wb, n, k);
            let pk = PackedWeight::new(kern, &wb, n, k);
            let mut yr = vec![0f32; m * n];
            let mut yk = vec![0f32; m * n];
            (reference.run)(&x, m, &pr, &mut yr);
            (kern.run)(&x, m, &pk, &mut yk);
            for (a, b) in yr.iter().zip(&yk) {
                assert!((a - b).abs() <= 2e-2 * (1.0 + a.abs()), "{} m={m} n={n} k={k}: {a} vs {b}", kern.name);
            }
        }
    }

    #[test]
    fn all_matmul_kernels_match_naive() {
        for k in MATMUL_KERNELS {
            check_kernel(k);
        }
    }
}
