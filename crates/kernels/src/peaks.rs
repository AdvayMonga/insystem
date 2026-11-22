//! Measured machine peaks: memory read bandwidth with all threads, and AMX bf16 throughput with
//! operands resident in tile registers.

use std::arch::asm;
use std::time::Instant;

use rayon::prelude::*;

/// GB/s reading a `bytes`-sized buffer with every thread, best of `reps`.
pub fn read_bandwidth(bytes: usize, reps: usize) -> f64 {
    let n = bytes / 8;
    let v: Vec<u64> = (0..n as u64).into_par_iter().collect();
    let mut best = 0f64;
    for _ in 0..reps {
        let t = Instant::now();
        let s: u64 = v.par_chunks(1 << 16).map(|c| c.iter().fold(0u64, |a, &x| a.wrapping_add(x))).reduce(|| 0, u64::wrapping_add);
        let dt = t.elapsed().as_secs_f64();
        std::hint::black_box(s);
        best = best.max(bytes as f64 / dt / 1e9);
    }
    best
}

/// TFLOP/s of `tdpbf16ps` on 16×16×32 tiles held in registers, all threads, or `None` without
/// AMX.
pub fn amx_tflops(iters: usize) -> Option<f64> {
    if !crate::matmul_amx::available() {
        return None;
    }
    let threads = rayon::current_num_threads();
    let t = Instant::now();
    (0..threads).into_par_iter().for_each(|_| {
        crate::matmul_amx::configure_current_thread();
        // SAFETY: tiles are configured on this thread; only register operands are used.
        unsafe {
            asm!("tilezero tmm0", "tilezero tmm1", "tilezero tmm2", "tilezero tmm3", "tilezero tmm4", "tilezero tmm5", "tilezero tmm6", "tilezero tmm7", options(nostack, nomem));
            for _ in 0..iters {
                asm!(
                    "tdpbf16ps tmm0, tmm4, tmm6",
                    "tdpbf16ps tmm1, tmm4, tmm7",
                    "tdpbf16ps tmm2, tmm5, tmm6",
                    "tdpbf16ps tmm3, tmm5, tmm7",
                    options(nostack, nomem)
                );
            }
        }
    });
    let dt = t.elapsed().as_secs_f64();
    // Each tdpbf16ps: 16 x 16 outputs x 32 products x 2 flops.
    Some((threads * iters * 4) as f64 * 16.0 * 16.0 * 32.0 * 2.0 / dt / 1e12)
}
