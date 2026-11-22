"""Writes small random-weight checkpoints of each supported architecture, for equivalence tests.

    python tools/make_tiny.py --out DIR
"""

import argparse
import os

import torch
from transformers import (LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM, Qwen3Config,
                          Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM)

COMMON = dict(vocab_size=1024, hidden_size=128, intermediate_size=256, num_hidden_layers=3, num_attention_heads=4,
              num_key_value_heads=2, max_position_embeddings=512, rms_norm_eps=1e-6, eos_token_id=1)

MODELS = {
    "tiny-qwen3": (Qwen3Config, Qwen3ForCausalLM, dict(head_dim=32, tie_word_embeddings=True, rope_theta=10000.0)),
    "tiny-qwen3-moe": (Qwen3MoeConfig, Qwen3MoeForCausalLM, dict(head_dim=32, num_experts=8, num_experts_per_tok=2,
                                                                  moe_intermediate_size=64, norm_topk_prob=True,
                                                                  decoder_sparse_step=1, mlp_only_layers=[0])),
    "tiny-llama": (LlamaConfig, LlamaForCausalLM, dict(rope_theta=10000.0)),
    "tiny-qwen2": (Qwen2Config, Qwen2ForCausalLM, dict(rope_theta=10000.0)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for name, (cfg_cls, model_cls, extra) in MODELS.items():
        torch.manual_seed(0)
        cfg = cfg_cls(**COMMON, **extra)
        model = model_cls(cfg).to(torch.bfloat16)
        with torch.no_grad():
            for p in model.parameters():
                p.normal_(0, 0.08)
            for n, p in model.named_parameters():
                if "norm" in n:
                    p.fill_(1.0).add_(torch.randn_like(p) * 0.1)
        d = os.path.join(a.out, name)
        model.save_pretrained(d, safe_serialization=True)
        print("wrote", d)


if __name__ == "__main__":
    main()
