//! Named kernel variants. A [`KernelSet`] picks one variant per op; the runtime packs weights
//! with the chosen matmul variant's `pack`, and can switch variants on a live engine by
//! repacking.

use crate::attention::{self, AttnArgs};
use crate::attention_x86;
use crate::{matmul, matmul_amx, matmul_x86};

/// A bf16 weight `[n, k]` (out features × in features) in a variant's packed layout.
pub struct PackedWeight {
    pub n: usize,
    pub k: usize,
    pub data: Vec<u16>,
    /// Name of the variant whose layout `data` is in.
    pub layout: &'static str,
}

pub struct MatmulKernel {
    pub name: &'static str,
    pub about: &'static str,
    /// Packs a row-major `[n, k]` bf16 matrix.
    pub pack: fn(&[u16], usize, usize) -> Vec<u16>,
    /// `y[m, n] = x[m, k] · wᵀ`.
    pub run: fn(&[f32], usize, &PackedWeight, &mut [f32]),
}

pub struct AttentionKernel {
    pub name: &'static str,
    pub about: &'static str,
    pub run: fn(&AttnArgs),
}

pub static MATMUL_KERNELS: &[MatmulKernel] = &[
    MatmulKernel { name: "naive", about: "scalar triple loop; the correctness reference", pack: matmul::pack_rowmajor, run: matmul::naive },
    MatmulKernel { name: "tile_f32", about: "AVX-512: 16-column panels, MRx32 f32 register tile, bf16 widened once per k step", pack: matmul_x86::pack_panels16, run: matmul_x86::tile_f32 },
    MatmulKernel { name: "amx", about: "Intel AMX: activations rounded to bf16, 16x64 output per tile group, tdpbf16ps over VNNI-packed 1 KiB weight tiles", pack: matmul_amx::pack, run: matmul_amx::run },
];

pub static ATTENTION_KERNELS: &[AttentionKernel] = &[
    AttentionKernel { name: "flash_avx512", about: "flash with AVX-512: query row and output row held in registers per KV block; head_dim 64/128/256, else falls back to flash", run: attention_x86::flash_avx512 },
    AttentionKernel { name: "flash", about: "per (seq, kv head, 16 query tokens): K/V rows shared across the GQA group and query chunk, online softmax per KV block", run: attention::flash },
    AttentionKernel { name: "naive", about: "per (token, head): full scores, softmax, weighted sum", run: attention::naive },
];

pub fn matmul_kernel(name: &str) -> Option<&'static MatmulKernel> {
    MATMUL_KERNELS.iter().find(|k| k.name == name)
}

pub fn attention_kernel(name: &str) -> Option<&'static AttentionKernel> {
    ATTENTION_KERNELS.iter().find(|k| k.name == name)
}

#[derive(Clone, Copy)]
pub struct KernelSet {
    pub matmul: &'static MatmulKernel,
    pub attention: &'static AttentionKernel,
}

impl KernelSet {
    pub fn by_names(matmul: &str, attention: &str) -> Result<Self, String> {
        Ok(Self {
            matmul: matmul_kernel(matmul).ok_or_else(|| format!("no matmul kernel `{matmul}`"))?,
            attention: attention_kernel(attention).ok_or_else(|| format!("no attention kernel `{attention}`"))?,
        })
    }

    pub fn default_set() -> Self {
        Self::by_names(DEFAULT_MATMUL, DEFAULT_ATTENTION).unwrap()
    }
}

pub const DEFAULT_MATMUL: &str = "amx";
pub const DEFAULT_ATTENTION: &str = "flash_avx512";

impl PackedWeight {
    pub fn new(kernel: &MatmulKernel, rowmajor: &[u16], n: usize, k: usize) -> Self {
        assert_eq!(rowmajor.len(), n * k);
        Self { n, k, data: (kernel.pack)(rowmajor, n, k), layout: kernel.name }
    }
}
