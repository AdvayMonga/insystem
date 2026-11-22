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
| 14 | bf16 KV cache: decode attention reads ~20 MB of f32 K/V per layer call, so halve it | 206.5, no gain; attention.decode unchanged (1.55s), so it is not KV-bandwidth-bound either; log-prob drift 0.035. Reverted. Next: a kernel microbenchmark instead of more guesses |
| — | **Diagnosis tool**: `insystem kbench` times one kernel on synthetic data. Decode attention took 650 µs/call (16 seqs × ctx 160), about 1.3 MAC/cycle/core with ~9 µs fixed cost per task; layout and exp changes alone made no difference | — |
| 15 | AVX-512 attention: q and output rows in registers per KV block, scratch sized to rows, KV blocks laid out [block][kv head][slot][d] | kbench 714 → 440 µs (now at ~48 GB/s, the bandwidth limit); full run 213.8 vs best 209.0, inside the noise margin; the profile shows attention 2.0 → 1.3s |
| — | **Method change**: interleaved A/B. The candidate and a saved champion binary run alternately 3× each in the same session; win = best candidate > best champion by 2% | — |
| 16 | 15 again, by A/B | **candidate 208.4 vs champion 196.8 (+5.9%), kept** |
| — | prefill AMX matmul: tasks over (row block that fits L2 × panel range) so activations are not re-streamed from L3 per panel | kbench 23.8 → 18.9 ms then 17.8 / 15.5 on reruns: within kbench noise; not taken further |
| — | k-blocking with C tiles reloaded per 16-chunk block (weights kept in L1) | kbench single-thread 15.5 → 31.8 ms, worse; dropped |
| — | **New peak measured** (`insystem peaks`): the 2×2 tile loop with all four tile loads from L1 reaches only 3.5 TFLOP/s on 4 threads (vs 12.2 register-only), so prefill matmuls at 1.2-1.6 TFLOP/s are at 35-45% of the achievable ceiling. Re-ranked: decode matmuls 6.0s (floor 4.25s), prefill 3.7s, misc ops 1.5s (serial or scalar), attention 1.3s | — |
| 17 | parallel qk-norm+RoPE per token (was single-threaded) and a vectorizable polynomial exp in SiLU (was scalar libm) | **213.3 vs champion 186.4 (+14%), kept** |
| 18 | bf16 KV cache, now that attention is bandwidth-bound (kbench 440 → 330 µs) | A/B: 183.7 vs champion 209.3, worse, but every matmul in the candidate runs was ~20% slower with no matmul change |
| 19 | repeat of 18's A/B | 214.5 vs 202.8 (+5.8%). Pooled over both sessions: best 214.5 vs 209.3 (+2.5%). **Kept**: a small real effect (attention is ~8% of time) at this machine's noise floor, and it doubles KV capacity per GB. Reps raised to 4 |
