//! Hugging Face `config.json` → [`ModelSpec`].

use anyhow::{bail, Context};
use serde_json::Value;

use crate::spec::{AttnSpec, FfnSpec, LayerSpec, ModelSpec, RopeSpec};

fn u(v: &Value, k: &str) -> anyhow::Result<usize> {
    v.get(k).and_then(Value::as_u64).map(|x| x as usize).with_context(|| format!("config.json: missing integer `{k}`"))
}

fn f(v: &Value, k: &str, default: f64) -> f32 {
    v.get(k).and_then(Value::as_f64).unwrap_or(default) as f32
}

fn b(v: &Value, k: &str, default: bool) -> bool {
    v.get(k).and_then(Value::as_bool).unwrap_or(default)
}

fn eos_ids(v: &Value) -> Vec<u32> {
    match v.get("eos_token_id") {
        Some(Value::Number(n)) => vec![n.as_u64().unwrap_or(0) as u32],
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_u64).map(|x| x as u32).collect(),
        _ => vec![],
    }
}

/// Builds a spec from a parsed `config.json`. Supported: Llama, Mistral, Qwen2, Qwen3, Qwen3-MoE.
pub fn spec_from_config(cfg: &Value) -> anyhow::Result<ModelSpec> {
    let arch = cfg
        .get("architectures")
        .and_then(|a| a.get(0))
        .and_then(Value::as_str)
        .context("config.json: missing `architectures`")?
        .to_string();
    let hidden = u(cfg, "hidden_size")?;
    let n_heads = u(cfg, "num_attention_heads")?;
    let n_kv_heads = cfg.get("num_key_value_heads").and_then(Value::as_u64).map(|x| x as usize).unwrap_or(n_heads);
    let head_dim = cfg.get("head_dim").and_then(Value::as_u64).map(|x| x as usize).unwrap_or(hidden / n_heads);
    let n_layers = u(cfg, "num_hidden_layers")?;
    let inter = u(cfg, "intermediate_size")?;

    let (qkv_bias, qk_norm) = match arch.as_str() {
        "LlamaForCausalLM" | "MistralForCausalLM" => (b(cfg, "attention_bias", false), false),
        "Qwen2ForCausalLM" => (true, false),
        "Qwen3ForCausalLM" | "Qwen3MoeForCausalLM" => (b(cfg, "attention_bias", false), true),
        other => bail!("unsupported architecture `{other}`; write a ModelSpec JSON for it"),
    };

    let use_sw = b(cfg, "use_sliding_window", arch == "MistralForCausalLM");
    let sw = cfg.get("sliding_window").and_then(Value::as_u64).map(|x| x as usize).filter(|_| use_sw);
    let max_window_layers = cfg.get("max_window_layers").and_then(Value::as_u64).map(|x| x as usize).unwrap_or(n_layers);

    // Newer configs put theta and scaling together under `rope_parameters`.
    let rope_params = cfg.get("rope_parameters").filter(|r| r.is_object());
    let rope_theta = rope_params.and_then(|r| r.get("rope_theta")).and_then(Value::as_f64).map(|x| x as f32).unwrap_or_else(|| f(cfg, "rope_theta", 10000.0));
    let scaling = rope_params.filter(|r| r.get("rope_type").and_then(Value::as_str).is_some_and(|t| t != "default")).or(cfg.get("rope_scaling"));
    let llama3 = scaling.filter(|r| r.get("rope_type").and_then(Value::as_str) == Some("llama3")).map(|r| {
        (f(r, "factor", 8.0), f(r, "low_freq_factor", 1.0), f(r, "high_freq_factor", 4.0), f(r, "original_max_position_embeddings", 8192.0))
    });
    if let Some(r) = scaling {
        if !r.is_null() && llama3.is_none() {
            bail!("unsupported rope_scaling {r}");
        }
    }

    let mut layers = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let ffn = if arch == "Qwen3MoeForCausalLM" {
            let step = cfg.get("decoder_sparse_step").and_then(Value::as_u64).unwrap_or(1) as usize;
            let mlp_only: Vec<usize> = cfg
                .get("mlp_only_layers")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_u64).map(|x| x as usize).collect())
                .unwrap_or_default();
            if !mlp_only.contains(&i) && (i + 1) % step == 0 {
                FfnSpec::Moe {
                    n_experts: u(cfg, "num_experts").or_else(|_| u(cfg, "num_local_experts"))?,
                    top_k: u(cfg, "num_experts_per_tok")?,
                    intermediate: u(cfg, "moe_intermediate_size")?,
                    norm_topk_prob: b(cfg, "norm_topk_prob", true),
                }
            } else {
                FfnSpec::Dense { intermediate: inter }
            }
        } else {
            FfnSpec::Dense { intermediate: inter }
        };
        layers.push(LayerSpec {
            attn: AttnSpec {
                n_heads,
                n_kv_heads,
                head_dim,
                qkv_bias,
                o_bias: false,
                qk_norm,
                sliding_window: if i < max_window_layers { sw } else { None },
            },
            ffn,
        });
    }

    Ok(ModelSpec {
        name: cfg.get("_name_or_path").and_then(Value::as_str).unwrap_or(&arch).to_string(),
        vocab_size: u(cfg, "vocab_size")?,
        hidden,
        norm_eps: f(cfg, "rms_norm_eps", 1e-6),
        tie_embeddings: b(cfg, "tie_word_embeddings", false),
        rope: RopeSpec { theta: rope_theta, llama3_scaling: llama3, rotary_dim: None },
        max_position: cfg.get("max_position_embeddings").and_then(Value::as_u64).unwrap_or(4096) as usize,
        layers,
        eos_token_ids: eos_ids(cfg),
        weight_prefix: "model.".into(),
    })
}

/// Loads a spec from a model directory: `insystem.json` if present (a hand-written spec),
/// otherwise `config.json`. EOS ids from `generation_config.json` are added.
pub fn load_spec(dir: &std::path::Path) -> anyhow::Result<ModelSpec> {
    let custom = dir.join("insystem.json");
    let mut spec = if custom.exists() {
        serde_json::from_str(&std::fs::read_to_string(&custom)?).context("insystem.json")?
    } else {
        let cfg: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).context("reading config.json")?)?;
        spec_from_config(&cfg)?
    };
    if let Ok(s) = std::fs::read_to_string(dir.join("generation_config.json")) {
        if let Ok(g) = serde_json::from_str::<Value>(&s) {
            for id in eos_ids(&g) {
                if !spec.eos_token_ids.contains(&id) {
                    spec.eos_token_ids.push(id);
                }
            }
        }
    }
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen3_config() {
        let cfg: Value = serde_json::from_str(
            r#"{"architectures":["Qwen3ForCausalLM"],"hidden_size":2048,"num_attention_heads":16,"num_key_value_heads":8,
            "head_dim":128,"num_hidden_layers":28,"intermediate_size":6144,"vocab_size":151936,"rms_norm_eps":1e-6,
            "rope_theta":1000000,"tie_word_embeddings":true,"eos_token_id":151645,"rope_scaling":null}"#,
        )
        .unwrap();
        let s = spec_from_config(&cfg).unwrap();
        assert_eq!(s.layers.len(), 28);
        assert!(s.layers[0].attn.qk_norm);
        let p = s.num_params();
        assert!(p > 1_700_000_000 && p < 1_750_000_000, "{p}");
    }
}
