#!/usr/bin/env bash
# Serving comparison: insystem vs llama.cpp on the same machine, same client, same loads.
#   bench/compare.sh MODEL_DIR GGUF OUT.jsonl
# Each server runs alone; the fixed-length sweep (InferenceX method, scaled to CPU) and the
# multi-turn agentic load are run against it, then it is stopped.
set -euo pipefail
MODEL=$1 GGUF=$2 OUT=$3
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-python3}
LLAMA=${LLAMA:?set LLAMA to the llama-server binary}
CONC=${CONC:-"1 4 16 32"}
PORT=8011

wait_up() { for _ in $(seq 300); do curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }

run_loads() {
  local label=$1
  $PY "$ROOT/bench/serve_bench.py" --base-url "http://127.0.0.1:$PORT" --label "$label" --workload fixed \
    --isl 128 --osl 64 --range-ratio 0.8 --prompts-per-conc 4 --concurrency $CONC --out "$OUT"
  $PY "$ROOT/bench/serve_bench.py" --base-url "http://127.0.0.1:$PORT" --label "$label" --workload agentic \
    --prefix 1024 --turns 4 --turn-tokens 128 --osl 32 --concurrency 4 8 --out "$OUT"
}

"$ROOT/target/release/insystem" serve --model "$MODEL" --addr "127.0.0.1:$PORT" --max-num-seqs 32 \
  --max-num-batched-tokens 512 --kv-cache-gb 3 --max-model-len 4096 > "$OUT.insystem.log" 2>&1 &
PID=$!
wait_up && run_loads insystem
kill $PID; wait $PID 2>/dev/null || true

"$ROOT/target/release/insystem" serve --model "$MODEL" --addr "127.0.0.1:$PORT" --max-num-seqs 32 \
  --max-num-batched-tokens 512 --kv-cache-gb 3 --max-model-len 4096 --num-speculative-tokens 2 > "$OUT.insystem-spec.log" 2>&1 &
PID=$!
wait_up && run_loads insystem+ngram-spec
kill $PID; wait $PID 2>/dev/null || true

"$LLAMA" -m "$GGUF" --host 127.0.0.1 --port $PORT -c 65536 -np 32 -t 4 -b 512 -ub 512 -fa on --cache-prompt > "$OUT.llama.log" 2>&1 &
PID=$!
wait_up && run_loads llama.cpp
kill $PID; wait $PID 2>/dev/null || true
echo done
