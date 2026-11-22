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
| 4 | prefault the KV cache: its pages are first touched by the KV write, so each step page-faults | 177.8, within noise of 174.8; the kv-write op disappeared from the profile but matmul time rose |
| 5 | control: rerun of 4 | 169.3 |
| 6 | control: rerun of 3 | 147.0, so single runs vary by ±15% on this shared VM. **Method change**: best of 3 runs, 3% margin. Also added `insystem peaks`: 51.7 GB/s read, 12.2 TFLOP/s AMX bf16, so the weight-bandwidth floor puts the ceiling for this benchmark near 600 tok/s |
| 7 | control: AMX alone, best of 3 | 168.5 (168, 147, 140) |
| 8 | prefault KV cache, best of 3 | **183.2 (+8.7% over its control), kept** |
| 9 | reuse scratch buffers (AMX activation buffer, attention q) to cut the 6-8% sys time | 179.2, no gain, reverted |
| 10 | flash attention: K/V rows shared across the GQA group and a 16-token query chunk, online softmax per block, q read in place | **195.5 (+6.7%), kept**; attention 2.5s → 2.1s, less than expected |
| 11 | attention scratch sized to rows used (each task zero-filled ~160 KB) | 193.8, no gain (attention 2.1 → 1.9s); reverted. Recount: prefill attention is ~48 GFLOP of q·k dots, each a 128-wide dot with a horizontal sum, so it is compute-bound |
| 12 | K cache transposed per block so 16 scores come from 128 vector FMAs with no horizontal sums | 192.0, no gain; attention unchanged at 2.1s, so scoring is not where its time goes. Reverted; profiler now splits labels by step type to find out |
| 13 | AMX: tile-contiguous activations (no 4 KB-stride loads) and 2×2 tile blocking for m>16 | **209.0 (+6.9%), kept**; prefill matmuls still ~1.55 TFLOP/s, now limited by every core re-streaming the 1.8 MB activation buffer from L3 for each weight panel pair |
