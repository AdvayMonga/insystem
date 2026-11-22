"""Load generator for any OpenAI-compatible server (insystem, llama.cpp, vLLM, SGLang).

Two workloads:

* `fixed`: InferenceX's fixed-sequence method. For each concurrency C, 10*C requests of random
  token ids are sent with at most C in flight and no pacing; lengths are uniform in
  [range_ratio * len, len]; ignore_eos is set so every request makes its full output.
* `agentic`: a scaled-down AgentX-like load. C sessions run at once; each session is a chain of
  turns whose prompt is the whole history so far (a unique per-session prefix, then each turn's
  new user/tool tokens and the previous outputs), so the server's prefix cache decides most of
  the prefill cost.

Metrics per concurrency: total token throughput (input + output) / wall time, output
throughput, interactivity (1 / mean inter-token latency, as InferenceX reports it), TTFT, TPOT
and end-to-end latency percentiles, and the server-reported cached prompt tokens when given.

    python bench/serve_bench.py --base-url http://127.0.0.1:8000 --workload fixed \
        --isl 256 --osl 128 --concurrency 1 4 16 --out results.jsonl --label insystem
"""

import argparse
import asyncio
import json
import random
import statistics
import time

import aiohttp


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


async def one_request(session, url, model, prompt, max_tokens):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
            "ignore_eos": True, "stream_options": {"include_usage": True}, "cache_prompt": True}
    t0 = time.perf_counter()
    times = []
    out_ids = []
    cached = None
    n_chunks = 0
    async with session.post(url + "/v1/completions", json=body) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {await resp.text()}")
        buf = b""
        async for chunk in resp.content.iter_any():
            buf += chunk
            while b"\n\n" in buf:
                ev, buf = buf.split(b"\n\n", 1)
                for line in ev.split(b"\n"):
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        continue
                    j = json.loads(data)
                    u = j.get("usage")
                    if u:
                        d = u.get("prompt_tokens_details") or {}
                        cached = d.get("cached_tokens", cached)
                    for c in j.get("choices", []):
                        ids = c.get("token_ids")
                        txt = c.get("text", "")
                        if ids:
                            out_ids.extend(ids)
                            times.extend([time.perf_counter()] * len(ids))
                        elif txt:
                            times.append(time.perf_counter())
                        n_chunks += 1
    t1 = time.perf_counter()
    n_out = len(out_ids) if out_ids else len(times)
    ttft = times[0] - t0 if times else t1 - t0
    itl = [b - a for a, b in zip(times, times[1:])]
    return {"ttft": ttft, "e2e": t1 - t0, "itl": itl, "n_out": n_out, "out_ids": out_ids, "n_in": len(prompt), "cached": cached}


def summarize(label, workload, conc, results, wall, extra):
    n_in = sum(r["n_in"] for r in results)
    n_out = sum(r["n_out"] for r in results)
    itl = [x for r in results for x in r["itl"]]
    tpot = [(r["e2e"] - r["ttft"]) / (r["n_out"] - 1) for r in results if r["n_out"] > 1]
    cached = [r["cached"] for r in results if r["cached"] is not None]
    mean_itl = statistics.mean(itl) if itl else None
    return {
        "label": label, "workload": workload, "concurrency": conc, "requests": len(results), **extra,
        "wall_s": wall, "input_tokens": n_in, "output_tokens": n_out,
        "total_tok_s": (n_in + n_out) / wall, "output_tok_s": n_out / wall,
        "interactivity_tok_s_user": 1 / mean_itl if mean_itl else None,
        "interactivity_p90": 1 / pct(itl, 90) if itl else None,
        "ttft_p50": pct([r["ttft"] for r in results], 50), "ttft_p90": pct([r["ttft"] for r in results], 90),
        "tpot_p50": pct(tpot, 50), "e2e_p50": pct([r["e2e"] for r in results], 50),
        "cached_prompt_tokens": sum(cached) if cached else None,
        "cache_hit_rate": sum(cached) / n_in if cached else None,
    }


async def run_fixed(a, conc, rng):
    lo = lambda n: max(1, round(n * a.range_ratio))
    n = a.prompts_per_conc * conc
    reqs = [([rng.randrange(a.vocab) for _ in range(rng.randint(lo(a.isl), a.isl))], rng.randint(lo(a.osl), a.osl)) for _ in range(n)]
    sem = asyncio.Semaphore(conc)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as s:
        async def go(p, o):
            async with sem:
                return await one_request(s, a.base_url, a.model, p, o)
        # Warm-up: one request per slot, not measured.
        await asyncio.gather(*[go([rng.randrange(a.vocab) for _ in range(16)], 4) for _ in range(min(conc, 4))])
        t0 = time.perf_counter()
        res = await asyncio.gather(*[go(p, o) for p, o in reqs])
        wall = time.perf_counter() - t0
    return summarize(a.label, "fixed", conc, res, wall, {"isl": a.isl, "osl": a.osl, "range_ratio": a.range_ratio})


async def run_agentic(a, conc, rng):
    results = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as s:
        async def session_task(sid):
            hist = [rng.randrange(a.vocab) for _ in range(a.prefix)]
            for turn in range(a.turns):
                hist += [rng.randrange(a.vocab) for _ in range(rng.randint(a.turn_tokens // 2, a.turn_tokens))]
                r = await one_request(s, a.base_url, a.model, list(hist), a.osl)
                results.append(r)
                hist += r["out_ids"] or [rng.randrange(a.vocab) for _ in range(r["n_out"])]
                if a.think_ms:
                    await asyncio.sleep(a.think_ms / 1000)
        t0 = time.perf_counter()
        await asyncio.gather(*[session_task(i) for i in range(conc)])
        wall = time.perf_counter() - t0
    return summarize(a.label, "agentic", conc, results, wall, {"prefix": a.prefix, "turns": a.turns, "turn_tokens": a.turn_tokens, "osl": a.osl})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="model")
    ap.add_argument("--label", default="server")
    ap.add_argument("--workload", choices=["fixed", "agentic"], default="fixed")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--isl", type=int, default=256)
    ap.add_argument("--osl", type=int, default=128)
    ap.add_argument("--range-ratio", type=float, default=0.8)
    ap.add_argument("--prompts-per-conc", type=int, default=10)
    ap.add_argument("--prefix", type=int, default=1024, help="agentic: tokens of each session's initial context")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--turn-tokens", type=int, default=128)
    ap.add_argument("--think-ms", type=int, default=0)
    ap.add_argument("--vocab", type=int, default=150000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    for conc in a.concurrency:
        rng = random.Random(a.seed * 1000 + conc)
        r = asyncio.run(run_fixed(a, conc, rng) if a.workload == "fixed" else run_agentic(a, conc, rng))
        print(json.dumps(r), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
