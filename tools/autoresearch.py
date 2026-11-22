"""One autoresearch iteration: build, check equivalence, benchmark, record.

    python tools/autoresearch.py --label "avx512 matmul" --hypothesis "..." [--engine "--matmul x"]

The benchmark is fixed for the whole loop (see DECISIONS.md D19). A run is `best` when its total
throughput beats the best so far by more than the noise margin, `no-gain` otherwise,
`failed-check` when `insystem check` fails, `crash` when the build or a run fails. The row is
appended to autoresearch/results.jsonl; keeping or reverting the code is done by hand in git.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "autoresearch", "results.jsonl")
MODEL = os.environ.get("INSYSTEM_MODEL", "/home/user/models/Qwen3-1.7B")
REF = os.environ.get("INSYSTEM_REF", os.path.join(ROOT, "local", "ref_qwen3_1.7b.json"))
BIN = os.path.join(ROOT, "target", "release", "insystem")
BENCH = ["--num-prompts", "16", "--input-len", "128", "--output-len", "64", "--range-ratio", "0.8",
         "--max-num-seqs", "16", "--max-num-batched-tokens", "512", "--kv-cache-gb", "1.5", "--profile"]
NOISE = 0.02


def rows():
    if not os.path.exists(RESULTS):
        return []
    return [json.loads(l) for l in open(RESULTS) if l.strip()]


def git_head():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--hypothesis", required=True)
    ap.add_argument("--engine", default="", help="extra engine arguments for check and bench")
    ap.add_argument("--skip-check", action="store_true")
    a = ap.parse_args()
    extra = shlex.split(a.engine)
    prev = rows()
    best = max([r["total_tok_s"] for r in prev if r["status"] in ("best", "baseline")], default=None)
    row = {"iter": len(prev), "label": a.label, "hypothesis": a.hypothesis, "engine_args": a.engine, "commit": git_head()}
    if not prev:
        row.update(bench="qwen3_1.7b_cpu4_offline",
                   subtitle="Qwen3ForCausalLM · 1.72B params · Xeon (Sapphire Rapids) 4 cores · 16 prompts ISL 128 OSL 64 ratio 0.8")

    b = subprocess.run(["cargo", "build", "--release", "-q"], cwd=ROOT, capture_output=True, text=True)
    if b.returncode != 0:
        row.update(status="crash", error=b.stderr[-2000:])
        return finish(row)
    if not a.skip_check:
        c = subprocess.run([BIN, "check", "--model", MODEL, "--reference", REF, "--kv-cache-gb", "1", *extra], capture_output=True, text=True)
        try:
            row["check"] = json.loads(c.stdout.strip().splitlines()[-1])
        except Exception:
            row["check"] = {"pass": False, "stderr": c.stderr[-2000:]}
        if c.returncode != 0 or not row["check"].get("pass"):
            row.update(status="crash" if "pass" not in row["check"] or c.returncode not in (0, 1) else "failed-check")
            return finish(row)
    r = subprocess.run([BIN, "bench", "--model", MODEL, *BENCH, *extra], capture_output=True, text=True)
    if r.returncode != 0:
        row.update(status="crash", error=r.stderr[-2000:])
        return finish(row)
    res = json.loads(r.stdout.strip().splitlines()[-1])
    row.update({k: res[k] for k in ("total_tok_s", "output_tok_s", "elapsed_s", "mean_tokens_per_step", "steps", "kernels", "profile")})
    if best is None:
        row["status"] = "baseline"
    elif res["total_tok_s"] > best * (1 + NOISE):
        row["status"] = "best"
    else:
        row["status"] = "no-gain"
    row["best_before"] = best
    finish(row)


def finish(row):
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(row) + "\n")
    prof = row.get("profile") or {}
    ops = sorted(((k, v["seconds"]) for k, v in (prof.get("ops") or {}).items()), key=lambda x: -x[1])
    tot = sum(s for _, s in ops) or 1
    print(json.dumps({k: row.get(k) for k in ("iter", "label", "status", "total_tok_s", "output_tok_s", "best_before", "check")}))
    for k, s in ops[:8]:
        print(f"  {k:<16} {s:8.2f}s {100 * s / tot:5.1f}%")
    if row["status"] in ("crash",):
        print(row.get("error", ""), file=sys.stderr)


if __name__ == "__main__":
    main()
