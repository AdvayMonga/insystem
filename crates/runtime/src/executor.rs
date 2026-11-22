use std::path::Path;
use std::time::Instant;

use anyhow::Context;
use insystem_core::{BatchPlan, CacheSpec, Executor, Sampled, SamplerSpec, SeqSlice, StepOutput};
use insystem_kernels::attention::{AttnArgs, AttnSeq};
use insystem_kernels::ops::{self, RopeTable};
use insystem_kernels::par::{par_for, SendPtr};
use insystem_kernels::{KernelSet, PackedWeight};
use insystem_model::{FfnSpec, ModelSpec, WeightStore};

use crate::profile::Profile;
use crate::program::{compile, Buf, Instr, Lin, Program};
use crate::sampler::sample;
use crate::weights::{FfnW, ModelWeights};

#[derive(Clone, Debug)]
pub struct ExecutorConfig {
    pub kv_cache_bytes: usize,
    pub block_size: usize,
    pub max_num_batched_tokens: usize,
    pub max_num_seqs: usize,
    pub max_model_len: usize,
    pub matmul: String,
    pub attention: String,
    pub profile: bool,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            kv_cache_bytes: 2 << 30,
            block_size: 16,
            max_num_batched_tokens: 512,
            max_num_seqs: 64,
            max_model_len: 4096,
            matmul: insystem_kernels::registry::DEFAULT_MATMUL.into(),
            attention: insystem_kernels::registry::DEFAULT_ATTENTION.into(),
            profile: false,
        }
    }
}

#[derive(Default)]
struct Bufs {
    resid: Vec<f32>,
    normed: Vec<f32>,
    qkv: Vec<f32>,
    attn: Vec<f32>,
    delta: Vec<f32>,
    gate_up: Vec<f32>,
    act: Vec<f32>,
    last: Vec<f32>,
    logits: Vec<f32>,
}

impl Bufs {
    fn get(&mut self, b: Buf) -> &mut Vec<f32> {
        match b {
            Buf::Resid => &mut self.resid,
            Buf::Normed => &mut self.normed,
            Buf::Qkv => &mut self.qkv,
            Buf::Attn => &mut self.attn,
            Buf::Delta => &mut self.delta,
            Buf::GateUp => &mut self.gate_up,
            Buf::Act => &mut self.act,
            Buf::Last => &mut self.last,
            Buf::Logits => &mut self.logits,
        }
    }
}

pub struct CpuExecutor {
    pub spec: ModelSpec,
    pub program: Program,
    pub weights: ModelWeights,
    pub kernels: KernelSet,
    store: WeightStore,
    cfg: ExecutorConfig,
    k_cache: Vec<Vec<f32>>,
    v_cache: Vec<Vec<f32>>,
    num_blocks: usize,
    probe_blocks: usize,
    rope: RopeTable,
    bufs: Bufs,
    pub profile: Option<Profile>,
}

const PROBE_TOKENS: usize = 48;

/// A zeroed buffer whose pages are all touched now, so no step pays first-touch page faults.
fn prefaulted(n: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let mut v: Vec<f32> = Vec::with_capacity(n);
    // SAFETY: every element is written below before the vector is read.
    unsafe { v.set_len(n) };
    v.par_chunks_mut(1 << 16).for_each(|c| c.fill(0.0));
    v
}

fn ensure_len(v: &mut Vec<f32>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}

impl CpuExecutor {
    pub fn load(dir: &Path, cfg: ExecutorConfig) -> anyhow::Result<Self> {
        let spec = insystem_model::hf::load_spec(dir)?;
        let store = WeightStore::open(dir)?;
        Self::from_parts(spec, store, cfg)
    }

    pub fn from_parts(spec: ModelSpec, store: WeightStore, cfg: ExecutorConfig) -> anyhow::Result<Self> {
        let kernels = KernelSet::by_names(&cfg.matmul, &cfg.attention).map_err(anyhow::Error::msg)?;
        let weights = ModelWeights::load(&store, &spec, kernels.matmul).context("loading weights")?;
        let per_token = spec.kv_bytes_per_token(4);
        let probe_blocks = PROBE_TOKENS.div_ceil(cfg.block_size);
        let total_blocks = (cfg.kv_cache_bytes / (per_token * cfg.block_size)).max(probe_blocks + 1);
        let num_blocks = total_blocks - probe_blocks;
        let mut k_cache = Vec::new();
        let mut v_cache = Vec::new();
        for l in &spec.layers {
            let n = total_blocks * cfg.block_size * l.attn.n_kv_heads * l.attn.head_dim;
            k_cache.push(prefaulted(n));
            v_cache.push(prefaulted(n));
        }
        let d = spec.layers[0].attn.head_dim;
        let rope = RopeTable::new(spec.rope.rotary_dim.unwrap_or(d), spec.rope.theta, cfg.max_model_len.min(spec.max_position).max(PROBE_TOKENS), spec.rope.llama3_scaling);
        let program = compile(&spec);
        let profile = cfg.profile.then(Profile::default);
        Ok(Self { spec, program, weights, kernels, store, cfg, k_cache, v_cache, num_blocks, probe_blocks, rope, bufs: Bufs::default(), profile })
    }

    pub fn config(&self) -> &ExecutorConfig {
        &self.cfg
    }

    fn lin(&self, layer: usize, lin: Lin) -> &PackedWeight {
        match lin {
            Lin::Qkv => &self.weights.layers[layer].qkv,
            Lin::O => &self.weights.layers[layer].o,
            Lin::LmHead => &self.weights.lm_head,
            Lin::GateUp => match &self.weights.layers[layer].ffn {
                FfnW::Dense { gate_up, .. } => gate_up,
                FfnW::Moe { .. } => unreachable!("dense op on MoE layer"),
            },
            Lin::Down => match &self.weights.layers[layer].ffn {
                FfnW::Dense { down, .. } => down,
                FfnW::Moe { .. } => unreachable!("dense op on MoE layer"),
            },
        }
    }

    fn run_instr(&mut self, ins: &Instr, plan: &BatchPlan, attn_seqs: &[AttnSeq], sampled: &[(usize, SamplerSpec)], out: &mut Vec<Sampled>) {
        let t = plan.num_tokens();
        let s = sampled.len();
        let h = self.spec.hidden;
        let eps = self.spec.norm_eps;
        match *ins {
            Instr::Embed => {
                ensure_len(&mut self.bufs.resid, t * h);
                ops::embedding(&plan.token_ids, &self.weights.embed, h, &mut self.bufs.resid[..t * h]);
            }
            Instr::RmsNorm { layer } => {
                ensure_len(&mut self.bufs.normed, t * h);
                ops::rmsnorm(&self.bufs.resid[..t * h], &self.weights.layers[layer].attn_norm, eps, &mut self.bufs.normed[..t * h]);
            }
            Instr::AddRmsNorm { layer, ffn } => {
                ensure_len(&mut self.bufs.normed, t * h);
                let lw = &self.weights.layers[layer];
                let w = if ffn { &lw.ffn_norm } else { &lw.attn_norm };
                ops::add_rmsnorm(&mut self.bufs.resid[..t * h], &self.bufs.delta[..t * h], w, eps, &mut self.bufs.normed[..t * h]);
            }
            Instr::Linear { layer, lin, src, dst } => {
                let m = if lin == Lin::LmHead { s } else { t };
                if m == 0 {
                    return;
                }
                let mut d = std::mem::take(self.bufs.get(dst));
                let x = std::mem::take(self.bufs.get(src));
                let w = self.lin(layer, lin);
                ensure_len(&mut d, m * w.n);
                (self.kernels.matmul.run)(&x[..m * w.k], m, w, &mut d[..m * w.n]);
                if lin == Lin::Qkv {
                    if let Some(b) = &self.weights.layers[layer].qkv_bias {
                        for row in d[..m * w.n].chunks_exact_mut(w.n) {
                            ops::add_inplace(row, b);
                        }
                    }
                }
                *self.bufs.get(src) = x;
                *self.bufs.get(dst) = d;
            }
            Instr::QkNormRope { layer } => {
                let a = &self.spec.layers[layer].attn;
                let (nh, nkv, d) = (a.n_heads, a.n_kv_heads, a.head_dim);
                let stride = (nh + 2 * nkv) * d;
                let lw = &self.weights.layers[layer];
                let qkv = &mut self.bufs.qkv[..t * stride];
                let norms = lw.q_norm.as_deref().zip(lw.k_norm.as_deref());
                ops::qk_norm_rope(qkv, stride, nh, nkv, d, norms, eps, &plan.positions, &self.rope);
            }
            Instr::KvWrite { layer } => {
                let a = &self.spec.layers[layer].attn;
                let (nh, nkv, d) = (a.n_heads, a.n_kv_heads, a.head_dim);
                let stride = (nh + 2 * nkv) * d;
                let w = nkv * d;
                let (kc, vc) = (&mut self.k_cache[layer], &mut self.v_cache[layer]);
                for (r, &slot) in plan.slot_mapping.iter().enumerate() {
                    let row = &self.bufs.qkv[r * stride..(r + 1) * stride];
                    let slot = slot as usize;
                    for g in 0..nkv {
                        let o = insystem_kernels::attention::kv_offset(slot, g, nkv, d, self.cfg.block_size);
                        kc[o..o + d].copy_from_slice(&row[nh * d + g * d..nh * d + (g + 1) * d]);
                        vc[o..o + d].copy_from_slice(&row[nh * d + w + g * d..nh * d + w + (g + 1) * d]);
                    }
                }
            }
            Instr::Attention { layer } => {
                let a = &self.spec.layers[layer].attn;
                let (nh, nkv, d) = (a.n_heads, a.n_kv_heads, a.head_dim);
                let stride = (nh + 2 * nkv) * d;
                ensure_len(&mut self.bufs.attn, t * nh * d);
                (self.kernels.attention.run)(&AttnArgs {
                    q: &self.bufs.qkv[..t * stride],
                    q_stride: stride,
                    k_cache: &self.k_cache[layer],
                    v_cache: &self.v_cache[layer],
                    out: SendPtr(self.bufs.attn.as_mut_ptr()),
                    seqs: attn_seqs,
                    n_heads: nh,
                    n_kv_heads: nkv,
                    head_dim: d,
                    block_size: self.cfg.block_size,
                    scale: 1.0 / (d as f32).sqrt(),
                    window: a.sliding_window,
                });
            }
            Instr::SiluMul { inter } => {
                ensure_len(&mut self.bufs.act, t * inter);
                ops::silu_mul(&self.bufs.gate_up[..t * 2 * inter], inter, &mut self.bufs.act[..t * inter]);
            }
            Instr::Moe { layer } => self.moe(layer, t),
            Instr::GatherLast => {
                ensure_len(&mut self.bufs.last, s * h);
                for (i, &(row, _)) in sampled.iter().enumerate() {
                    let dst = &mut self.bufs.last[i * h..(i + 1) * h];
                    for ((o, &a), &b) in dst.iter_mut().zip(&self.bufs.resid[row * h..(row + 1) * h]).zip(&self.bufs.delta[row * h..(row + 1) * h]) {
                        *o = a + b;
                    }
                }
            }
            Instr::FinalNorm => {
                if s == 0 {
                    return;
                }
                let src = self.bufs.last[..s * h].to_vec();
                ops::rmsnorm(&src, &self.weights.final_norm, eps, &mut self.bufs.last[..s * h]);
            }
            Instr::Sample => {
                let v = self.spec.vocab_size;
                let mut res = vec![Sampled { token: 0, logprob: 0.0 }; s];
                let rp = SendPtr(res.as_mut_ptr());
                let logits = &self.bufs.logits;
                par_for(s, |i| {
                    let r = sample(&logits[i * v..(i + 1) * v], &sampled[i].1);
                    // SAFETY: each index is written by one task.
                    unsafe { *rp.add(i) = r };
                });
                *out = res;
            }
        }
    }

    fn moe(&mut self, layer: usize, t: usize) {
        let h = self.spec.hidden;
        let FfnSpec::Moe { n_experts, top_k, intermediate, norm_topk_prob } = self.spec.layers[layer].ffn else { unreachable!() };
        let FfnW::Moe { router, experts } = &self.weights.layers[layer].ffn else { unreachable!() };
        let mm = self.kernels.matmul.run;
        let normed = &self.bufs.normed[..t * h];
        let mut logits = vec![0f32; t * n_experts];
        mm(normed, t, router, &mut logits);
        let mut assign: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_experts];
        for r in 0..t {
            let row = &logits[r * n_experts..(r + 1) * n_experts];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let p: Vec<f32> = row.iter().map(|v| (v - mx).exp()).collect();
            let z: f32 = p.iter().sum();
            let mut idx: Vec<usize> = (0..n_experts).collect();
            idx.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
            let top = &idx[..top_k];
            let norm = if norm_topk_prob { top.iter().map(|&e| p[e]).sum::<f32>() } else { z };
            for &e in top {
                assign[e].push((r, p[e] / norm));
            }
        }
        ensure_len(&mut self.bufs.delta, t * h);
        let delta = &mut self.bufs.delta[..t * h];
        delta.fill(0.0);
        for (e, toks) in assign.iter().enumerate() {
            if toks.is_empty() {
                continue;
            }
            let m = toks.len();
            let mut xs = vec![0f32; m * h];
            for (j, &(r, _)) in toks.iter().enumerate() {
                xs[j * h..(j + 1) * h].copy_from_slice(&normed[r * h..(r + 1) * h]);
            }
            let mut gu = vec![0f32; m * 2 * intermediate];
            mm(&xs, m, &experts[e].gate_up, &mut gu);
            let mut act = vec![0f32; m * intermediate];
            ops::silu_mul(&gu, intermediate, &mut act);
            let mut y = vec![0f32; m * h];
            mm(&act, m, &experts[e].down, &mut y);
            for (j, &(r, wt)) in toks.iter().enumerate() {
                for (d, &v) in delta[r * h..(r + 1) * h].iter_mut().zip(&y[j * h..(j + 1) * h]) {
                    *d += wt * v;
                }
            }
        }
    }

    fn forward(&mut self, plan: &BatchPlan) -> StepOutput {
        let bt = &plan.block_tables;
        let attn_seqs: Vec<AttnSeq> = plan
            .seqs
            .iter()
            .map(|s: &SeqSlice| AttnSeq {
                q_start: s.q_start as usize,
                q_len: s.q_len as usize,
                ctx_len: s.ctx_len as usize,
                block_table: &bt[s.bt_start as usize..(s.bt_start + s.bt_len) as usize],
            })
            .collect();
        let sampled: Vec<(usize, SamplerSpec)> =
            plan.seqs.iter().filter_map(|s| s.sample.map(|sp| ((s.q_start + s.q_len - 1) as usize, sp))).collect();
        let mut out = Vec::new();
        let instrs = std::mem::take(&mut self.program.instrs);
        let mut prof = self.profile.take();
        let phase = if plan.num_prefill_tokens > 0 { "mixed" } else { "decode" };
        for ins in &instrs {
            if let Some(p) = prof.as_mut() {
                let t0 = Instant::now();
                self.run_instr(ins, plan, &attn_seqs, &sampled, &mut out);
                p.add(format!("{}.{phase}", ins.label()), t0.elapsed());
            } else {
                self.run_instr(ins, plan, &attn_seqs, &sampled, &mut out);
            }
        }
        if let Some(p) = prof.as_mut() {
            p.steps += 1;
            p.tokens += plan.num_tokens() as u64;
        }
        self.profile = prof;
        self.program.instrs = instrs;
        StepOutput { sampled: out }
    }

    /// A one-sequence plan over the reserved probe blocks.
    fn probe_plan(&self) -> BatchPlan {
        let bs = self.cfg.block_size;
        let n = PROBE_TOKENS;
        let blocks: Vec<u32> = (0..self.probe_blocks as u32).map(|b| self.num_blocks as u32 + b).collect();
        BatchPlan {
            step: u64::MAX,
            token_ids: (0..n as u32).map(|i| (i * 7919 + 13) % self.spec.vocab_size as u32).collect(),
            positions: (0..n as u32).collect(),
            slot_mapping: (0..n).map(|p| blocks[p / bs] * bs as u32 + (p % bs) as u32).collect(),
            seqs: vec![SeqSlice {
                seq_id: u64::MAX,
                q_start: 0,
                q_len: n as u32,
                ctx_len: n as u32,
                bt_start: 0,
                bt_len: blocks.len() as u32,
                sample: Some(SamplerSpec { temperature: 0.0, top_k: 0, top_p: 1.0, rng_seed: 0 }),
            }],
            block_tables: blocks,
            num_prefill_tokens: n as u32,
        }
    }

    /// Logits of the probe batch's last token under the current kernels.
    pub fn probe_logits(&mut self) -> Vec<f32> {
        let plan = self.probe_plan();
        let prof = self.profile.take();
        self.forward(&plan);
        self.profile = prof;
        self.bufs.logits[..self.spec.vocab_size].to_vec()
    }

    /// Switches kernels on the live executor. Weights are repacked for the new matmul; the probe
    /// batch's logits under the new kernels must match the current ones within `tol` (max abs
    /// difference relative to the largest logit), or the switch is undone.
    pub fn swap_kernels(&mut self, new: KernelSet, tol: f32) -> anyhow::Result<f32> {
        let before = self.probe_logits();
        let old = self.kernels;
        let repack = new.matmul.name != self.weights.lm_head.layout;
        let old_weights = if repack {
            let w = ModelWeights::load(&self.store, &self.spec, new.matmul)?;
            Some(std::mem::replace(&mut self.weights, w))
        } else {
            None
        };
        self.kernels = new;
        let after = self.probe_logits();
        let scale = before.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        let err = before.iter().zip(&after).fold(0f32, |m, (a, b)| m.max((a - b).abs())) / scale;
        if !(err <= tol) {
            self.kernels = old;
            if let Some(w) = old_weights {
                self.weights = w;
            }
            anyhow::bail!("kernels {}/{} differ from {}/{} by {err:.2e} on the probe batch (tolerance {tol:.1e}); kept the old ones", new.matmul.name, new.attention.name, old.matmul.name, old.attention.name);
        }
        self.cfg.matmul = new.matmul.name.into();
        self.cfg.attention = new.attention.name.into();
        Ok(err)
    }
}

impl Executor for CpuExecutor {
    fn cache_spec(&self) -> CacheSpec {
        CacheSpec { block_size: self.cfg.block_size, num_blocks: self.num_blocks }
    }

    fn execute(&mut self, plan: &BatchPlan) -> anyhow::Result<StepOutput> {
        anyhow::ensure!(plan.num_tokens() > 0, "empty plan");
        Ok(self.forward(plan))
    }
}
