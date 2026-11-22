"""Plots serving results from bench/serve_bench.py: total throughput against interactivity for the
fixed-length sweep (the InferenceX view), and throughput and TTFT for the agentic load.

    python tools/plot_serving.py results.jsonl out.png
"""

import json
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"insystem": "#3b5bdb", "insystem+ngram-spec": "#12b886", "llama.cpp": "#c07a35"}


def main():
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    fixed, agent = defaultdict(list), defaultdict(list)
    for r in rows:
        (fixed if r["workload"] == "fixed" else agent)[r["label"]].append(r)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 5.2), dpi=110, gridspec_kw={"width_ratios": [1.5, 1, 1]})
    for label, rs in fixed.items():
        rs.sort(key=lambda r: r["concurrency"])
        x = [r["interactivity_tok_s_user"] for r in rs]
        y = [r["total_tok_s"] for r in rs]
        a1.plot(x, y, "-o", color=COLORS.get(label), label=label)
        for r, xi, yi in zip(rs, x, y):
            a1.annotate(f"c={r['concurrency']}", (xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=8, color="#666")
    a1.set_xlabel("interactivity (tok/s/user, 1 / mean inter-token latency)")
    a1.set_ylabel("total throughput (input + output tok/s)")
    r0 = next(iter(fixed.values()))[0]
    a1.set_title(f"fixed ISL {r0['isl']} / OSL {r0['osl']} (ratio {r0['range_ratio']}), concurrency sweep", loc="left", fontsize=11)
    a1.legend()
    a1.grid(color="#eee")
    labels = list(agent)
    concs = sorted({r["concurrency"] for rs in agent.values() for r in rs})
    w = 0.8 / max(1, len(labels))
    for i, label in enumerate(labels):
        by = {r["concurrency"]: r for r in agent[label]}
        xs = [j + i * w for j in range(len(concs))]
        a2.bar(xs, [by[c]["total_tok_s"] if c in by else 0 for c in concs], w, color=COLORS.get(label), label=label)
        a3.bar(xs, [by[c]["ttft_p50"] if c in by else 0 for c in concs], w, color=COLORS.get(label), label=label)
    for ax, t in ((a2, "agentic: total tok/s"), (a3, "agentic: TTFT p50 (s)")):
        ax.set_xticks([j + w * (len(labels) - 1) / 2 for j in range(len(concs))], [f"{c} sessions" for c in concs])
        ax.set_title(t, loc="left", fontsize=11)
        ax.grid(axis="y", color="#eee")
    a2.legend(fontsize=8)
    for ax in (a1, a2, a3):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Qwen3-1.7B bf16 · Xeon (Sapphire Rapids) 4 cores · same client, same machine", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(sys.argv[2], bbox_inches="tight")
    print("wrote", sys.argv[2])


if __name__ == "__main__":
    main()
