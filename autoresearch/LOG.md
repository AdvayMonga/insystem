# Autoresearch log

Benchmark (fixed): `insystem bench`, Qwen3-1.7B, 16 prompts, ISL 128 / OSL 64, range ratio 0.8, all at once,
total tok/s. Every run must pass `insystem check` (greedy tokens vs transformers fp32). Reference point:
llama.cpp (bf16 GGUF, 4 threads, flash-attn) on the same shape = 82.4 tok/s (`llama-batched-bench -npp 115 -ntg 58 -npl 16`).

| # | hypothesis | outcome |
|---|---|---|
| 0 | baseline: row-major matmul (`rowdot`), naive attention | 21.1 tok/s; matmuls 96% of op time (gate/up 43%, down 25%, qkv 15%) |
| 1 | AVX-512 register-tiled matmul | crash: raw pointers captured by the parallel closure did not compile; fixed |
| 2 | AVX-512 register-tiled matmul: 6×32 f32 tile, bf16 widened once per k step, should raise FLOP/s several-fold | **83.7 tok/s (4.0×), kept**; check passes; matmuls now 83%, attention 8%, kv write 4% |
| 3 | AMX bf16 tiles: tdpbf16ps does 16× the MACs of AVX-512 FMA; bf16 activations should stay inside the check tolerance | **174.8 tok/s (2.1×), kept**; max log-prob diff 0.023 (was 0.0015). `rowdot` removed. Now: matmuls 63%, attention 16%, kv write 11% |
