# Decisions

Each entry: what was decided, and why. Newest decisions are appended at the end of their section.

## Goal and scope

**D1. Build the engine from scratch in Rust.**
The engine is a fresh codebase, not a fork of Resin, vLLM or SGLang, so every core decision is
ours to understand and change. Rust gives a host side with no interpreter, no GIL and predictable
latency, and still allows `unsafe`, inline assembly and SIMD intrinsics where kernels need them.

**D2. The win condition is saturated throughput on the InferenceX method, with equivalence.**
InferenceX (SemiAnalysis) measures total token throughput (input + output) per GPU against
interactivity (1 / inter-token latency) over a concurrency sweep with random prompts of fixed
lengths (range ratio 0.8, 10 × concurrency requests, `ignore_eos`). AgentX adds multi-turn
agent traces where prefix caching dominates. Our benchmark client (`bench/serve_bench.py`) runs
both shapes against any OpenAI-compatible server, so the same numbers come out for insystem and
for llama.cpp / vLLM / SGLang. A speedup counts only if `insystem check` still passes.

**D3. This build targets CPU (Intel Sapphire Rapids, 4 cores, 16 GB, no GPU).**
That is the hardware this environment has. The architecture is device-independent (the core never
sees tensors), so a CUDA executor is a second implementation of the same `Executor` trait, not a
rewrite. CPU also makes a good test bed for the autoresearch loop: the same bottlenecks appear
(weight bandwidth at small batch, compute at large batch, attention at long context, host
overhead) on a machine where every experiment is cheap.

**D4. First model: Qwen3-1.7B (1.72B parameters), real weights from Hugging Face.**
Dense, well known, the same architecture family as the Qwen3-30B-A3B target (which adds MoE,
already supported). It is large enough that kernels, not overheads, dominate.

## Architecture

**D5. Core and executor meet at one read-only `BatchPlan` per step; the executor returns one
sampled token and its log-probability per sampling row** (the Resin split, decision 1 of the
design walk-through, option A). The core owns requests, the scheduler and the KV block
allocator; the executor owns weights, the KV memory, kernels and sampling. The plan is flat
struct-of-arrays: token ids, positions, the KV slot of every new token, one `SeqSlice` per
sequence (query range, context length, block-table range, sampler settings), block tables in CSR
form. The executor keeps no request→cache map: everything it needs to read or write the cache is
in the plan, which is what makes the executor replaceable (CPU now, CUDA later) and testable
alone. Sampling inside the executor means only token ids and log-probs cross the boundary, never
logits; on a GPU that keeps the vocab-sized tensor on the device.

**D6. One token budget for prefill and decode (vLLM V1), not separate prefill/decode batches
(Resin).** Each sequence just wants `len(tokens) - num_computed` more tokens; decodes are served
first, then prefills fill the remaining budget, chunked if needed. Mixing them keeps decode
latency steady while long prompts are processed and gives the matmuls more rows, which on CPU
(and GPU) raises arithmetic intensity. The attention kernel handles ragged mixed batches, so
nothing else needs to know which kind of token it is.

**D7. Paged KV cache with hash-chained prefix caching and LRU eviction of free blocks.**
Blocks of 16 tokens; a full block's id is a hash of its parent's hash and its tokens; freed
blocks keep their hash at the tail of a free list and can be taken back by a later request with
the same prefix; allocation takes from the head. Uncached freed blocks go to the head so they are
reused first. This is vLLM V1's design; it is O(1) per block and needs no tree. AgentX-style
traffic (a long context re-sent every turn) is mostly prefix hits, so this is the first-order
feature for that benchmark.

**D8. Preemption by recompute.** When the cache cannot grow a running sequence, the most recently
admitted one is released and re-queued at the front with its tokens; its prefix blocks are often
still cached, so the recompute is mostly a prefix hit. No swap space to manage.

**D9. Sampling is deterministic per request.** A request's random stream depends only on its seed
and output position, so the tokens it samples do not depend on what it was batched with. That
makes batch-invariance testable and reruns reproducible.

**D10. Models are data: a `ModelSpec` (JSON-serializable) built from Hugging Face `config.json`
by small adapters, or written by hand.** Supported today: Llama, Mistral, Qwen2, Qwen3,
Qwen3-MoE (per-layer dense or MoE FFN, GQA, q/k norm, QKV bias, sliding window, Llama-3 RoPE
scaling). Customizing a model (different depth, MoE layers, windows) is writing an
`insystem.json` next to the weights, with no code change.

**D11. The forward pass is generated: `compile(spec)` emits a flat `Program` of ops over named
scratch buffers.** Per layer: norm, fused QKV projection, q/k norm + RoPE, KV write, attention,
output projection, residual add fused into the next norm, fused gate/up projection, SiLU-mul,
down projection (or a routed MoE op). After the last layer only the sampled rows are gathered
and projected to the vocabulary, so prefill tokens that are not sampled never touch the LM head.
The program is printable (`insystem inspect`) and every op is timed by label when profiling is
on, which is what the autoresearch loop reads to find bottlenecks. Fusion decisions live in
`compile` and in weight loading (QKV and gate/up weights are row-concatenated at load).

**D12. Kernels are named variants in a registry; weights are packed by the chosen matmul
variant.** A matmul variant is `pack` (row-major bf16 → its layout) plus `run`. Variants are
selected at load (`--matmul`, `--attention`) and can be switched on a running engine
(`POST /admin/kernels`): weights are repacked, a fixed probe batch is run under the old and new
kernels in reserved KV blocks, and the switch is undone if logits differ by more than the
tolerance. That is the Together-Compiler idea (agents try kernels in a live engine and only
verified ones stay) on this hardware: an experiment costs a repack, not an engine restart.

**D13. Weights stay bf16 in memory; activations, accumulation and KV are f32 at first.**
bf16 is the checkpoint's own precision, so storing it is lossless and halves weight bandwidth
against f32. f32 activations make the first version match the fp32 reference to ~1e-3 in
log-probability, which gives a clean equivalence baseline that later numeric changes (bf16 KV,
bf16 activations into AMX) are measured against.

**D14. Built with `target-cpu=native`.** The engine is built on the machine it serves from (as
llama.cpp's GGML_NATIVE does), so the compiler may use AVX-512, AVX512-BF16 and FMA everywhere.

**D15. Threads: rayon's global pool, one thread per core, parallel-for over output columns
(matmul), (token, head) pairs (attention) and rows (norms).** Simple and correct first; the
synchronization cost per op is a measured question for the autoresearch loop.

**D16. The engine runs on its own OS thread; the HTTP front end (axum/tokio, two worker threads)
talks to it over channels and detokenizes per request.** Tokenization and SSE formatting never
block a forward step. No overlap scheduling (SGLang's CPU-schedules-N+1-while-GPU-runs-N): on
CPU the forward pass uses every core, so there is nothing idle to overlap with; scheduling costs
are measured in `EngineStats::sched_time`.

**D17. OpenAI-compatible API including token-id prompts and `ignore_eos`.** The benchmark sends
random token ids (as InferenceX does), so no tokenizer differences enter the comparison. Chat
uses the model's own Jinja chat template (minijinja) with Qwen3's thinking turned off.

## Measurement

**D18. Equivalence reference = Hugging Face transformers in fp32, greedy, with log-probs and the
top-2 margin at every step** (`tools/reference.py`). `insystem check` runs all prompts in one
batch (so batching, chunking and prefix caching are exercised), requires every greedy token to
match except after a near-tie in the reference (margin < 0.02), and log-probs to match within
0.05. Tiny random-weight models of each architecture (`tools/make_tiny.py`) cover the adapters
and the MoE path.

**D19. The autoresearch metric is offline saturated throughput** (`insystem bench`, as
`vllm bench throughput`): 16 prompts, input 128, output 64, range ratio 0.8, all submitted at
once, total (input + output) tokens per second. It is fixed for the whole loop so every point on
the chart is comparable, and short enough to run each iteration. The serving sweep (InferenceX
shape) and the agentic workload are run at the start and the end against llama.cpp.
