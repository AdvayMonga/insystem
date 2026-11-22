//! Kernel microbenchmarks on synthetic data, outside the engine.

use std::time::Instant;

use clap::Args;
use insystem_kernels::attention::{AttnArgs, AttnSeq};
use insystem_kernels::par::SendPtr;
use insystem_kernels::registry::{attention_kernel, matmul_kernel};
use insystem_kernels::PackedWeight;
use serde_json::json;

#[derive(Args, Debug)]
pub struct KbenchArgs {
    /// `matmul` or `attention`.
    #[arg(long)]
    pub op: String,
    #[arg(long, default_value = insystem_kernels::registry::DEFAULT_MATMUL)]
    pub matmul: String,
    #[arg(long, default_value = insystem_kernels::registry::DEFAULT_ATTENTION)]
    pub attention: String,
    #[arg(long, default_value_t = 16)]
    pub m: usize,
    #[arg(long, default_value_t = 12288)]
    pub n: usize,
    #[arg(long, default_value_t = 2048)]
    pub k: usize,
    /// Attention: sequences, new tokens per sequence, context per sequence.
    #[arg(long, default_value_t = 16)]
    pub seqs: usize,
    #[arg(long, default_value_t = 1)]
    pub q_len: usize,
    #[arg(long, default_value_t = 160)]
    pub ctx: usize,
    #[arg(long, default_value_t = 16)]
    pub heads: usize,
    #[arg(long, default_value_t = 8)]
    pub kv_heads: usize,
    #[arg(long, default_value_t = 128)]
    pub head_dim: usize,
    #[arg(long, default_value_t = 200)]
    pub iters: usize,
}

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

pub fn run(a: KbenchArgs) -> anyhow::Result<()> {
    match a.op.as_str() {
        "matmul" => {
            let kern = matmul_kernel(&a.matmul).ok_or_else(|| anyhow::anyhow!("no matmul {}", a.matmul))?;
            let w: Vec<u16> = fill(a.n * a.k, 1).iter().map(|&x| insystem_kernels::f32_to_bf16(x)).collect();
            // Enough copies (~1 GiB) that the weights stream from DRAM as in the model, not from
            // the last-level cache.
            let copies = ((1usize << 30) / (a.n * a.k * 2)).clamp(1, 64);
            let pws: Vec<PackedWeight> = (0..copies).map(|_| PackedWeight::new(kern, &w, a.n, a.k)).collect();
            let x = fill(a.m * a.k, 2);
            let mut y = vec![0f32; a.m * a.n];
            (kern.run)(&x, a.m, &pws[0], &mut y);
            let t = Instant::now();
            for i in 0..a.iters {
                (kern.run)(&x, a.m, &pws[i % copies], &mut y);
            }
            let dt = t.elapsed().as_secs_f64() / a.iters as f64;
            let flops = 2.0 * (a.m * a.n * a.k) as f64;
            println!("{}", json!({"op": "matmul", "kernel": kern.name, "m": a.m, "n": a.n, "k": a.k, "us": dt * 1e6,
                "tflops": flops / dt / 1e12, "weight_gb_s": (a.n * a.k * 2) as f64 / dt / 1e9}));
        }
        "attention" => {
            let kern = attention_kernel(&a.attention).ok_or_else(|| anyhow::anyhow!("no attention {}", a.attention))?;
            let (h, kvh, d, bs) = (a.heads, a.kv_heads, a.head_dim, 16);
            let blocks_per = a.ctx.div_ceil(bs);
            let nblocks = a.seqs * blocks_per;
            let k: Vec<u16> = fill(nblocks * bs * kvh * d, 3).iter().map(|&x| insystem_kernels::f32_to_bf16(x)).collect();
            let v: Vec<u16> = fill(nblocks * bs * kvh * d, 4).iter().map(|&x| insystem_kernels::f32_to_bf16(x)).collect();
            let t_tokens = a.seqs * a.q_len;
            let q = fill(t_tokens * h * d, 5);
            let mut out = vec![0f32; t_tokens * h * d];
            let bts: Vec<Vec<u32>> = (0..a.seqs).map(|s| ((s * blocks_per) as u32..((s + 1) * blocks_per) as u32).collect()).collect();
            let seqs: Vec<AttnSeq> = (0..a.seqs).map(|s| AttnSeq { q_start: s * a.q_len, q_len: a.q_len, ctx_len: a.ctx, block_table: &bts[s] }).collect();
            let args = AttnArgs {
                q: &q,
                q_stride: h * d,
                k_cache: &k,
                v_cache: &v,
                out: SendPtr(out.as_mut_ptr()),
                seqs: &seqs,
                n_heads: h,
                n_kv_heads: kvh,
                head_dim: d,
                block_size: bs,
                scale: 1.0 / (d as f32).sqrt(),
                window: None,
            };
            (kern.run)(&args);
            let t = Instant::now();
            for _ in 0..a.iters {
                (kern.run)(&args);
            }
            let dt = t.elapsed().as_secs_f64() / a.iters as f64;
            let kv_bytes = (a.seqs * a.ctx * kvh * d * 2 * 2) as f64;
            let flops = 4.0 * (t_tokens * h * d * a.ctx) as f64;
            println!("{}", json!({"op": "attention", "kernel": kern.name, "seqs": a.seqs, "q_len": a.q_len, "ctx": a.ctx, "us": dt * 1e6,
                "kv_gb_s": kv_bytes / dt / 1e9, "gflops": flops / dt / 1e9}));
        }
        other => anyhow::bail!("unknown op {other}"),
    }
    Ok(())
}
