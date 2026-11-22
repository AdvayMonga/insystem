//! CPU kernels.
//!
//! Activations are f32. Weights are bf16, held in whatever layout the chosen matmul variant packs
//! them into. Every op that has more than one implementation is a [`registry`] entry, selected by
//! name at load time and replaceable while the engine runs.

pub mod attention;
pub mod matmul;
pub mod matmul_amx;
pub mod matmul_x86;
pub mod ops;
pub mod par;
pub mod peaks;
pub mod registry;

pub use registry::{KernelSet, MatmulKernel, PackedWeight};

#[inline]
pub fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

#[inline]
pub fn f32_to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return 0x7fc0;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}
