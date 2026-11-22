"""Writes the equivalence reference: greedy outputs of Hugging Face transformers in fp32.

For each prompt it records the greedy tokens, their log-probabilities, and the gap between the
top two log-probabilities at each step (`insystem check` allows a divergence only where that gap
is small). Prompts are random token ids of mixed lengths, plus the same prefix shared by some of
them so the engine's prefix cache is exercised.

    python tools/reference.py --model DIR --out ref.json [--prompts 8 --max-tokens 24]
"""

import argparse
import json
import random

import torch
from transformers import AutoModelForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    model.eval()
    vocab = min(model.config.vocab_size, 150000)
    shared = [rng.randrange(vocab) for _ in range(40)]
    prompts = []
    for i in range(a.prompts):
        n = rng.choice([5, 17, 33, 64, 100])
        p = [rng.randrange(vocab) for _ in range(n)]
        prompts.append(shared + p if i % 2 else p)

    outputs = []
    with torch.no_grad():
        for p in prompts:
            ids = torch.tensor([p])
            toks, lps, margins = [], [], []
            past = None
            cur = ids
            for _ in range(a.max_tokens):
                out = model(cur, past_key_values=past, use_cache=True)
                past = out.past_key_values
                lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
                top = torch.topk(lp, 2)
                t = int(top.indices[0])
                toks.append(t)
                lps.append(float(top.values[0]))
                margins.append(float(top.values[0] - top.values[1]))
                cur = torch.tensor([[t]])
            outputs.append({"tokens": toks, "logprobs": lps, "margins": margins})
    json.dump({"model": a.model, "prompts": prompts, "outputs": outputs}, open(a.out, "w"))
    print(f"wrote {len(prompts)} prompts x {a.max_tokens} tokens to {a.out}")


if __name__ == "__main__":
    main()
