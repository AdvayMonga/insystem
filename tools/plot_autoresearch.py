"""Plots autoresearch/results.jsonl as a staircase of best throughput over iterations.

    python tools/plot_autoresearch.py autoresearch/results.jsonl autoresearch/progress.png
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, RED, GREY = "#3b5bdb", "#c07a35", "#a33a3a", "#8a8f98"


def main():
    src, out = sys.argv[1], sys.argv[2]
    rows = [json.loads(l) for l in open(src) if l.strip()]
    base = rows[0]["total_tok_s"]
    best = base
    stair_x, stair_y = [0], [base]
    fig, ax = plt.subplots(figsize=(13, 6.2), dpi=110)
    for i, r in enumerate(rows):
        v = r.get("total_tok_s")
        st = r["status"]
        if st in ("crash", "failed-check"):
            ax.scatter([i], [base * 0.8], marker="x", s=70, color=RED, zorder=3)
            continue
        if st == "best" or i == 0:
            stair_x += [i, i]
            stair_y += [best, v]
            best = v
            ax.scatter([i], [v], s=70, color=BLUE, edgecolor="white", linewidth=1.2, zorder=4)
            if i > 0 and r.get("label"):
                ax.annotate(f"{r['label']}\n{v:.1f}", (i, v), xytext=(-10, 18), textcoords="offset points", fontsize=8.5,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#dddddd"))
        else:
            ax.scatter([i], [v], s=28, color=ORANGE, zorder=3)
    stair_x.append(len(rows) - 1 + 0.4)
    stair_y.append(best)
    ax.step(stair_x, stair_y, where="post", color=BLUE, linewidth=2, zorder=2)
    ax.axhline(base, color=GREY, linestyle="--", linewidth=1)
    ax.text(len(rows) - 0.6, base * 1.04, f"baseline {base:.1f}", color=GREY, ha="right", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("total tokens / second  (log)", color="#555")
    ax.set_xlabel("experiment", color="#555")
    ax.grid(axis="y", color="#e6e6e6")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    m = rows[0]
    ax.set_title(f"autoresearch · {m.get('bench', 'bench')} — {base:.1f} → {best:.1f} tok/s  ({best / base:.1f}×)",
                 loc="left", fontsize=14, fontweight="bold")
    fig.text(0.125, 0.905, m.get("subtitle", ""), color=GREY, fontsize=9.5, family="monospace")
    handles = [
        plt.Line2D([], [], marker="o", color="w", markerfacecolor=BLUE, markersize=9, label="new best (staircase steps)"),
        plt.Line2D([], [], marker="o", color="w", markerfacecolor=ORANGE, markersize=7, label="ran, did not take the lead"),
        plt.Line2D([], [], marker="x", color=RED, linestyle="none", markersize=8, label="crash or failed equivalence check"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=9)
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
