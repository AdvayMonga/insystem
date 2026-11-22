//! Weights for every op in the program, packed for the selected matmul kernel.

use anyhow::Context;
use insystem_kernels::{MatmulKernel, PackedWeight};
use insystem_model::{FfnSpec, ModelSpec, WeightStore};
use rayon::prelude::*;

pub struct ExpertW {
    pub gate_up: PackedWeight,
    pub down: PackedWeight,
}

pub enum FfnW {
    Dense { gate_up: PackedWeight, down: PackedWeight },
    Moe { router: PackedWeight, experts: Vec<ExpertW> },
}

pub struct LayerW {
    pub attn_norm: Vec<f32>,
    pub qkv: PackedWeight,
    pub qkv_bias: Option<Vec<f32>>,
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub o: PackedWeight,
    pub ffn_norm: Vec<f32>,
    pub ffn: FfnW,
}

pub struct ModelWeights {
    /// bf16 `[vocab, hidden]`.
    pub embed: Vec<u16>,
    pub final_norm: Vec<f32>,
    pub lm_head: PackedWeight,
    pub layers: Vec<LayerW>,
}

/// Row-concatenates bf16 matrices that share `k`.
fn cat_rows(store: &WeightStore, names: &[String]) -> anyhow::Result<(Vec<u16>, usize, usize)> {
    let mut out = Vec::new();
    let mut n = 0;
    let mut k = 0;
    for name in names {
        let (v, shape) = store.bf16(name)?;
        anyhow::ensure!(shape.len() == 2, "{name}: expected 2-D, got {shape:?}");
        if k == 0 {
            k = shape[1];
        }
        anyhow::ensure!(shape[1] == k, "{name}: inner dim {} != {k}", shape[1]);
        n += shape[0];
        out.extend_from_slice(&v);
    }
    Ok((out, n, k))
}

fn packed(store: &WeightStore, kern: &MatmulKernel, names: &[String]) -> anyhow::Result<PackedWeight> {
    let (w, n, k) = cat_rows(store, names)?;
    Ok(PackedWeight::new(kern, &w, n, k))
}

fn f32v(store: &WeightStore, name: &str) -> anyhow::Result<Vec<f32>> {
    Ok(store.f32(name)?.0)
}

fn load_layer(store: &WeightStore, spec: &ModelSpec, kern: &MatmulKernel, i: usize) -> anyhow::Result<LayerW> {
    let p = format!("{}layers.{i}.", spec.weight_prefix);
    let l = &spec.layers[i];
    let a = &l.attn;
    let qkv_names: Vec<String> = ["q", "k", "v"].iter().map(|x| format!("{p}self_attn.{x}_proj.weight")).collect();
    let qkv = packed(store, kern, &qkv_names)?;
    anyhow::ensure!(qkv.n == (a.n_heads + 2 * a.n_kv_heads) * a.head_dim && qkv.k == spec.hidden, "layer {i}: qkv shape mismatch");
    let qkv_bias = if a.qkv_bias {
        let mut b = Vec::new();
        for x in ["q", "k", "v"] {
            b.extend(f32v(store, &format!("{p}self_attn.{x}_proj.bias"))?);
        }
        Some(b)
    } else {
        None
    };
    let (q_norm, k_norm) = if a.qk_norm {
        (Some(f32v(store, &format!("{p}self_attn.q_norm.weight"))?), Some(f32v(store, &format!("{p}self_attn.k_norm.weight"))?))
    } else {
        (None, None)
    };
    let o = packed(store, kern, &[format!("{p}self_attn.o_proj.weight")])?;
    let ffn = match l.ffn {
        FfnSpec::Dense { .. } => FfnW::Dense {
            gate_up: packed(store, kern, &[format!("{p}mlp.gate_proj.weight"), format!("{p}mlp.up_proj.weight")])?,
            down: packed(store, kern, &[format!("{p}mlp.down_proj.weight")])?,
        },
        FfnSpec::Moe { n_experts, .. } => {
            let router = packed(store, kern, &[format!("{p}mlp.gate.weight")])?;
            let experts = (0..n_experts)
                .map(|e| {
                    let ep = format!("{p}mlp.experts.{e}.");
                    Ok(ExpertW {
                        gate_up: packed(store, kern, &[format!("{ep}gate_proj.weight"), format!("{ep}up_proj.weight")])?,
                        down: packed(store, kern, &[format!("{ep}down_proj.weight")])?,
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            FfnW::Moe { router, experts }
        }
    };
    Ok(LayerW {
        attn_norm: f32v(store, &format!("{p}input_layernorm.weight"))?,
        qkv,
        qkv_bias,
        q_norm,
        k_norm,
        o,
        ffn_norm: f32v(store, &format!("{p}post_attention_layernorm.weight"))?,
        ffn,
    })
}

impl ModelWeights {
    pub fn load(store: &WeightStore, spec: &ModelSpec, kern: &MatmulKernel) -> anyhow::Result<Self> {
        let p = &spec.weight_prefix;
        let (embed, shape) = store.bf16(&format!("{p}embed_tokens.weight"))?;
        anyhow::ensure!(shape == vec![spec.vocab_size, spec.hidden], "embed shape {shape:?}");
        let lm_name = if spec.tie_embeddings || !store.contains("lm_head.weight") { format!("{p}embed_tokens.weight") } else { "lm_head.weight".into() };
        let lm_head = packed(store, kern, &[lm_name])?;
        let layers = (0..spec.layers.len())
            .into_par_iter()
            .map(|i| load_layer(store, spec, kern, i).with_context(|| format!("layer {i}")))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self { embed, final_norm: f32v(store, &format!("{p}norm.weight"))?, lm_head, layers })
    }

    pub fn bytes(&self) -> usize {
        let pw = |w: &PackedWeight| w.data.len() * 2;
        let mut n = self.embed.len() * 2 + pw(&self.lm_head);
        for l in &self.layers {
            n += pw(&l.qkv) + pw(&l.o);
            n += match &l.ffn {
                FfnW::Dense { gate_up, down } => pw(gate_up) + pw(down),
                FfnW::Moe { router, experts } => pw(router) + experts.iter().map(|e| pw(&e.gate_up) + pw(&e.down)).sum::<usize>(),
            };
        }
        n
    }
}
