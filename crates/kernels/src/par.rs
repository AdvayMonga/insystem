//! Parallel-for helpers over disjoint output ranges.

use rayon::prelude::*;

/// A raw pointer that may be shared across threads; each user writes a disjoint range.
#[derive(Clone, Copy)]
pub struct SendPtr<T>(pub *mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    /// # Safety
    /// The caller writes only indices no other thread writes.
    #[inline]
    pub unsafe fn add(self, i: usize) -> *mut T {
        self.0.add(i)
    }
}

/// Runs `f(start, end)` over `[0, n)` split into chunks of `chunk`, in parallel.
pub fn par_ranges(n: usize, chunk: usize, f: impl Fn(usize, usize) + Sync + Send) {
    let chunk = chunk.max(1);
    let tasks = n.div_ceil(chunk);
    if tasks <= 1 {
        if n > 0 {
            f(0, n);
        }
        return;
    }
    (0..tasks).into_par_iter().for_each(|t| {
        let s = t * chunk;
        f(s, (s + chunk).min(n));
    });
}

/// Runs `f(i)` for `i` in `[0, n)` in parallel.
pub fn par_for(n: usize, f: impl Fn(usize) + Sync + Send) {
    if n == 1 {
        f(0);
        return;
    }
    (0..n).into_par_iter().for_each(f);
}

pub fn num_threads() -> usize {
    rayon::current_num_threads()
}
