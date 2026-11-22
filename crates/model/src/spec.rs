use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RopeSpec {
    pub theta: f32,
    /// Llama-3 style frequency scaling: (factor, low_freq_factor, high_freq_factor, original_max_pos).
    #[serde(default)]
    pub llama3_scaling: Option<(f32, f32, f32, f32)>,
    /// Rotary dimensions per head; `None` means the whole head.
    #[serde(default)]
    pub rotary_dim: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AttnSpec {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    #[serde(default)]
    pub qkv_bias: bool,
    #[serde(default)]
    pub o_bias: bool,
    /// Per-head RMSNorm on q and k before RoPE (Qwen3).
    #[serde(default)]
    pub qk_norm: bool,
    #[serde(default)]
    pub sliding_window: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FfnSpec {
    /// SwiGLU: down(silu(gate(x)) * up(x)).
    Dense { intermediate: usize },
    /// Top-k routed SwiGLU experts.
    Moe { n_experts: usize, top_k: usize, intermediate: usize, norm_topk_prob: bool },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LayerSpec {
    pub attn: AttnSpec,
    pub ffn: FfnSpec,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelSpec {
    pub name: String,
    pub vocab_size: usize,
    pub hidden: usize,
    pub norm_eps: f32,
    pub tie_embeddings: bool,
    pub rope: RopeSpec,
    pub max_position: usize,
    pub layers: Vec<LayerSpec>,
    #[serde(default)]
    pub eos_token_ids: Vec<u32>,
    /// Tensor-name prefix of the decoder (`model.` for most Hugging Face checkpoints).
    #[serde(default = "default_prefix")]
    pub weight_prefix: String,
}

fn default_prefix() -> String {
    "model.".into()
}

impl ModelSpec {
    pub fn num_params(&self) -> usize {
        let mut n = self.vocab_size * self.hidden * if self.tie_embeddings { 1 } else { 2 } + self.hidden;
        for l in &self.layers {
            let a = &l.attn;
            n += self.hidden * (a.n_heads + 2 * a.n_kv_heads) * a.head_dim + a.n_heads * a.head_dim * self.hidden;
            n += 2 * self.hidden;
            if a.qk_norm {
                n += 2 * a.head_dim;
            }
            n += match l.ffn {
                FfnSpec::Dense { intermediate } => 3 * self.hidden * intermediate,
                FfnSpec::Moe { n_experts, intermediate, .. } => n_experts * 3 * self.hidden * intermediate + n_experts * self.hidden,
            };
        }
        n
    }

    /// Bytes of K plus V per cached token, summed over layers, at `bytes_per_elem`.
    pub fn kv_bytes_per_token(&self, bytes_per_elem: usize) -> usize {
        self.layers.iter().map(|l| 2 * l.attn.n_kv_heads * l.attn.head_dim * bytes_per_elem).sum()
    }
}
