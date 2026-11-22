//! Intel AMX bf16 matmul.
//!
//! Weights are packed per 16-column panel into 1 KiB B tiles: for each 32-wide chunk of `k`, 16
//! rows of k-pairs × 16 columns × 2 bf16 (the VNNI layout `tdpbf16ps` reads). Panels are padded
//! to a multiple of 4 and `k` to a multiple of 32 with zeros. Activations are rounded to bf16
//! once per call into a row-major buffer padded to 16-row tiles. Each task owns a group of 4
//! panels (64 output columns) and walks every 16-row tile of `x`, accumulating four 16×16 f32
//! C tiles over `k`.

use std::arch::asm;
use std::sync::OnceLock;

use crate::par::{num_threads, par_ranges, SendPtr};
use crate::registry::PackedWeight;
use crate::f32_to_bf16;

const GROUP: usize = 4;

fn kpad(k: usize) -> usize {
    k.div_ceil(32) * 32
}

fn npanels(n: usize) -> usize {
    n.div_ceil(16).div_ceil(GROUP) * GROUP
}

pub fn pack(w: &[u16], n: usize, k: usize) -> Vec<u16> {
    let kp = kpad(k);
    let panels = npanels(n);
    let chunks = kp / 32;
    let mut out = vec![0u16; panels * chunks * 512];
    for p in 0..panels {
        for c in 0..16 {
            let j = p * 16 + c;
            if j >= n {
                break;
            }
            let src = &w[j * k..(j + 1) * k];
            for (kk, &v) in src.iter().enumerate() {
                let (ch, within) = (kk / 32, kk % 32);
                let (q, e) = (within / 2, within % 2);
                out[(p * chunks + ch) * 512 + q * 32 + c * 2 + e] = v;
            }
        }
    }
    out
}

/// Asks the kernel for permission to use AMX tile data in this process.
fn amx_enabled() -> bool {
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        // CPUID.(EAX=7, ECX=0):EDX bit 22 = AMX-BF16, bit 24 = AMX-TILE.
        // SAFETY: cpuid is always available on x86-64.
        let edx = std::arch::x86_64::__cpuid_count(7, 0).edx;
        if edx & (1 << 22) == 0 || edx & (1 << 24) == 0 {
            return false;
        }
        const SYS_ARCH_PRCTL: usize = 158;
        const ARCH_REQ_XCOMP_PERM: usize = 0x1023;
        const XFEATURE_XTILEDATA: usize = 18;
        let ret: isize;
        // SAFETY: arch_prctl(ARCH_REQ_XCOMP_PERM, XTILEDATA) only changes this process's
        // permission to use AMX state.
        unsafe {
            asm!("syscall", inlateout("rax") SYS_ARCH_PRCTL as isize => ret, in("rdi") ARCH_REQ_XCOMP_PERM,
                 in("rsi") XFEATURE_XTILEDATA, lateout("rcx") _, lateout("r11") _, options(nostack));
        }
        ret == 0
    })
}

pub fn available() -> bool {
    amx_enabled()
}

thread_local! {
    static CONFIGURED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[repr(C, align(64))]
struct TileCfg([u8; 64]);

pub fn configure_current_thread() {
    CONFIGURED.with(|c| {
        if c.get() {
            return;
        }
        let mut cfg = TileCfg([0; 64]);
        cfg.0[0] = 1; // palette 1
        for t in 0..8 {
            // colsb[t] (u16 at 16 + 2t) = 64 bytes, rows[t] (u8 at 48 + t) = 16.
            cfg.0[16 + 2 * t] = 64;
            cfg.0[48 + t] = 16;
        }
        // SAFETY: loads a valid palette-1 configuration; AMX permission was granted.
        unsafe { asm!("ldtilecfg [{}]", in(reg) cfg.0.as_ptr(), options(nostack, readonly)) };
        c.set(true);
    });
}

/// C[16×64] += A[16×kp] · B for one 16-row tile and one 4-panel group, stored to `c` as four
/// consecutive 16×16 f32 tiles.
#[inline(never)]
unsafe fn tile_row_group(a: *const u16, a_stride: usize, b: *const u16, chunks: usize, panel_stride: usize, c: *mut f32) {
    asm!("tilezero tmm0", "tilezero tmm1", "tilezero tmm2", "tilezero tmm3", options(nostack, nomem));
    let b1 = b.add(panel_stride);
    let b2 = b.add(2 * panel_stride);
    let b3 = b.add(3 * panel_stride);
    for ch in 0..chunks {
        asm!(
            "tileloadd tmm4, [{a} + {sa}*1]",
            "tileloadd tmm5, [{b0} + {s64}*1]",
            "tileloadd tmm6, [{b1} + {s64}*1]",
            "tdpbf16ps tmm0, tmm4, tmm5",
            "tileloadd tmm7, [{b2} + {s64}*1]",
            "tdpbf16ps tmm1, tmm4, tmm6",
            "tileloadd tmm5, [{b3} + {s64}*1]",
            "tdpbf16ps tmm2, tmm4, tmm7",
            "tdpbf16ps tmm3, tmm4, tmm5",
            a = in(reg) a.add(ch * 32),
            sa = in(reg) a_stride * 2,
            b0 = in(reg) b.add(ch * 512),
            b1 = in(reg) b1.add(ch * 512),
            b2 = in(reg) b2.add(ch * 512),
            b3 = in(reg) b3.add(ch * 512),
            s64 = in(reg) 64usize,
            options(nostack, readonly),
        );
    }
    asm!(
        "tilestored [{c0} + {s}*1], tmm0",
        "tilestored [{c1} + {s}*1], tmm1",
        "tilestored [{c2} + {s}*1], tmm2",
        "tilestored [{c3} + {s}*1], tmm3",
        c0 = in(reg) c,
        c1 = in(reg) c.add(256),
        c2 = in(reg) c.add(512),
        c3 = in(reg) c.add(768),
        s = in(reg) 64usize,
        options(nostack),
    );
}

pub fn run(x: &[f32], m: usize, w: &PackedWeight, y: &mut [f32]) {
    let (n, k) = (w.n, w.k);
    assert!(x.len() >= m * k && y.len() >= m * n);
    assert!(amx_enabled(), "AMX is not available");
    let kp = kpad(k);
    let chunks = kp / 32;
    let mt = m.div_ceil(16);
    // Activations → bf16, padded to whole 16-row tiles and to kp.
    let mut a = vec![0u16; mt * 16 * kp];
    {
        let ap = SendPtr(a.as_mut_ptr());
        par_ranges(m, 16, |r0, r1| {
            for r in r0..r1 {
                // SAFETY: each row is written by one task.
                let dst = unsafe { std::slice::from_raw_parts_mut(ap.add(r * kp), k) };
                for (d, &s) in dst.iter_mut().zip(&x[r * k..(r + 1) * k]) {
                    *d = f32_to_bf16(s);
                }
            }
        });
    }
    let groups = npanels(n) / GROUP;
    let panel_stride = chunks * 512;
    let yp = SendPtr(y.as_mut_ptr());
    let ap = SendPtr(a.as_ptr() as *mut u16);
    let bp = SendPtr(w.data.as_ptr() as *mut u16);
    let chunk = (groups / (num_threads() * 3)).max(1);
    par_ranges(groups, chunk, |g0, g1| {
        configure_current_thread();
        let mut c = [0f32; 1024];
        for g in g0..g1 {
            let j0 = g * GROUP * 16;
            if j0 >= n {
                break;
            }
            let cols = (n - j0).min(64);
            for t in 0..mt {
                // SAFETY: tile loads stay inside the padded A buffer and the packed weights;
                // each task writes only its own output columns.
                unsafe {
                    tile_row_group(ap.ptr().add(t * 16 * kp), kp, bp.ptr().add(g * GROUP * panel_stride), chunks, panel_stride, c.as_mut_ptr());
                }
                let rows = (m - t * 16).min(16);
                for r in 0..rows {
                    let i = t * 16 + r;
                    for cc in 0..cols {
                        let (tile, col) = (cc / 16, cc % 16);
                        // SAFETY: (i, j0 + cc) is inside y and owned by this task.
                        unsafe { *yp.add(i * n + j0 + cc) = c[tile * 256 + r * 16 + col] };
                    }
                }
            }
        }
    });
}
