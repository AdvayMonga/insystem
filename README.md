# insystem

An LLM inference engine, written from scratch in Rust.

- **Core/executor split:** a model-independent core (continuous batching, one token budget for chunked prefill and decode, paged KV with prefix caching, preemption by recompute) hands the executor one read-only batch plan per step and gets sampled tokens back.
- **Forward pass generated from a model spec:** Hugging Face configs (Llama, Mistral, Qwen2, Qwen3, Qwen3-MoE) or a hand-written `insystem.json` compile into an inspectable, per-op-profiled program.
- **Kernel registry with live swapping:** kernels are named variants. `POST /admin/kernels` repacks the weights, checks the new kernels against the current ones on a probe batch, and keeps them only if they match.
- **OpenAI-compatible server:** completions and chat, streaming, token-id prompts, `ignore_eos`, `/metrics`.
- **Measurement built in:** `insystem bench` (offline throughput), `bench/serve_bench.py` (InferenceX fixed-length sweep and an AgentX-like multi-turn load against any OpenAI server), `insystem check` (greedy equivalence against transformers fp32).
- **Autoresearch loop:** `tools/autoresearch.py` runs one experiment, `autoresearch/LOG.md` has one line per hypothesis and outcome, and `autoresearch/progress.png` charts the results.

```
cargo build --release
./target/release/insystem serve --model /path/to/Qwen3-1.7B
./target/release/insystem bench --model /path/to/Qwen3-1.7B --num-prompts 16 --input-len 128 --output-len 64
python tools/reference.py --model /path/to/Qwen3-1.7B --out ref.json && ./target/release/insystem check --model /path/to/Qwen3-1.7B --reference ref.json
```

Design decisions and their reasons: [DECISIONS.md](DECISIONS.md).
