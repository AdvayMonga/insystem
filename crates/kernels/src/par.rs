//! Parallel-for over disjoint output ranges, on a persistent spinning thread pool.
//!
//! The calling thread and `threads - 1` workers split each job's tasks through a shared atomic
//! counter. Between jobs, workers spin on an epoch counter (a forward pass issues hundreds of
//! short jobs, and waking sleeping threads for each costs more than the job); after ~2 ms without
//! work they fall back to short sleeps, so an idle engine does not hold every core.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

/// A raw pointer that may be shared across threads; each user writes a disjoint range.
#[derive(Clone, Copy)]
pub struct SendPtr<T>(pub *mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    #[inline]
    pub fn ptr(self) -> *mut T {
        self.0
    }

    /// # Safety
    /// The caller writes only indices no other thread writes.
    #[inline]
    pub unsafe fn add(self, i: usize) -> *mut T {
        self.0.add(i)
    }
}

type Job = *const (dyn Fn(usize) + Sync);

struct Shared {
    epoch: AtomicU64,
    job: std::sync::atomic::AtomicPtr<()>,
    job_vtable: AtomicUsize,
    next: AtomicUsize,
    ntasks: AtomicUsize,
    finished: AtomicUsize,
    stop: AtomicBool,
}

struct Pool {
    threads: usize,
    shared: Arc<Shared>,
    busy: AtomicBool,
}

static POOL: OnceLock<Pool> = OnceLock::new();
static REQUESTED: AtomicUsize = AtomicUsize::new(0);

/// Sets the pool size (calling thread included). Takes effect only before the first parallel call.
pub fn set_threads(n: usize) {
    REQUESTED.store(n, Ordering::Relaxed);
}

fn pool() -> &'static Pool {
    POOL.get_or_init(|| {
        let n = match REQUESTED.load(Ordering::Relaxed) {
            0 => std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
            n => n,
        };
        let shared = Arc::new(Shared {
            epoch: AtomicU64::new(0),
            job: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()),
            job_vtable: AtomicUsize::new(0),
            next: AtomicUsize::new(0),
            ntasks: AtomicUsize::new(0),
            finished: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
        });
        for i in 1..n {
            let sh = shared.clone();
            std::thread::Builder::new().name(format!("insystem-worker-{i}")).spawn(move || worker(&sh)).expect("spawn worker");
        }
        Pool { threads: n, shared, busy: AtomicBool::new(false) }
    })
}

/// Rebuilds the job's fat pointer from its two halves.
fn load_job(sh: &Shared) -> Job {
    let data = sh.job.load(Ordering::Acquire);
    let vt = sh.job_vtable.load(Ordering::Acquire);
    // SAFETY: (data, vtable) were stored from one `*const dyn Fn(usize) + Sync` by `par_for`,
    // which keeps the closure alive until every worker has reported finishing this epoch.
    unsafe { std::mem::transmute::<(*mut (), usize), Job>((data, vt)) }
}

fn run_tasks(sh: &Shared, job: Job) {
    let n = sh.ntasks.load(Ordering::Acquire);
    loop {
        let i = sh.next.fetch_add(1, Ordering::AcqRel);
        if i >= n {
            break;
        }
        // SAFETY: see `load_job`.
        unsafe { (*job)(i) };
    }
}

fn worker(sh: &Shared) {
    let mut seen = 0u64;
    let mut idle_since = Instant::now();
    loop {
        let e = sh.epoch.load(Ordering::Acquire);
        if e == seen {
            if sh.stop.load(Ordering::Relaxed) {
                return;
            }
            if idle_since.elapsed() > Duration::from_millis(2) {
                std::thread::sleep(Duration::from_micros(100));
            } else {
                std::hint::spin_loop();
            }
            continue;
        }
        seen = e;
        run_tasks(sh, load_job(sh));
        sh.finished.fetch_add(1, Ordering::AcqRel);
        idle_since = Instant::now();
    }
}

/// Runs `f(i)` for `i` in `[0, n)` in parallel.
pub fn par_for(n: usize, f: impl Fn(usize) + Sync + Send) {
    if n == 0 {
        return;
    }
    let p = pool();
    // One job at a time; a nested or concurrent call runs inline on its own thread.
    if n == 1 || p.threads == 1 || p.busy.swap(true, Ordering::AcqRel) {
        (0..n).for_each(f);
        return;
    }
    let sh = &*p.shared;
    let local: *const (dyn Fn(usize) + Sync + '_) = &f;
    // SAFETY: erases the closure's lifetime; this function does not return until every worker
    // has finished with it.
    let job: Job = unsafe { std::mem::transmute::<*const (dyn Fn(usize) + Sync + '_), Job>(local) };
    // SAFETY: splits a fat pointer into data and vtable halves; `load_job` rejoins them.
    let (data, vt) = unsafe { std::mem::transmute::<Job, (*mut (), usize)>(job) };
    sh.job.store(data, Ordering::Release);
    sh.job_vtable.store(vt, Ordering::Release);
    sh.ntasks.store(n, Ordering::Release);
    sh.next.store(0, Ordering::Release);
    sh.finished.store(0, Ordering::Release);
    sh.epoch.fetch_add(1, Ordering::AcqRel);
    run_tasks(sh, job);
    while sh.finished.load(Ordering::Acquire) < p.threads - 1 {
        std::hint::spin_loop();
    }
    p.busy.store(false, Ordering::Release);
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
    par_for(tasks, |t| {
        let s = t * chunk;
        f(s, (s + chunk).min(n));
    });
}

pub fn num_threads() -> usize {
    pool().threads
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covers_every_index_once() {
        for n in [1usize, 2, 7, 64, 1000] {
            let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            par_for(n, |i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
        }
        // Nested calls run inline.
        let total = AtomicUsize::new(0);
        par_for(8, |_| par_for(8, |_| {
            total.fetch_add(1, Ordering::Relaxed);
        }));
        assert_eq!(total.load(Ordering::Relaxed), 64);
    }
}
