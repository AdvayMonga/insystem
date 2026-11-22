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
        control = r["label"].startswith(("repeat", "control")) and st != "best"
        if (st == "best" or i == 0) and not control:
            level = max(best, v)
            stair_x += [i, i]
            stair_y += [best, level]
            best = level
            ax.scatter([i], [v], s=70, color=BLUE, edgecolor="white", linewidth=1.2, zorder=4)
            if i > 0 and r.get("label"):
                n_best = sum(1 for x in rows[1:i + 1] if x["status"] == "best")
                dy = 30 + 26 * (n_best % 3)
                ax.annotate(f"{r['label']}\n{v:.1f}", (i, v), xytext=(4, dy), textcoords="offset points", fontsize=7.5,
                            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#dddddd"),
                            arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.8))
        elif control:
            ax.scatter([i], [v], s=50, facecolor="none", edgecolor=ORANGE, linewidth=1.8, zorder=3)
        else:
            ax.scatter([i], [v], s=28, color=ORANGE, zorder=3)
    stair_x.append(len(rows) - 1 + 0.4)
    stair_y.append(best)
    ax.step(stair_x, stair_y, where="post", color=BLUE, linewidth=2, zorder=2)
    ax.axhline(base, color=GREY, linestyle="--", linewidth=1)
    ax.text(len(rows) - 0.6, base * 1.04, f"baseline {base:.1f}", color=GREY, ha="right", fontsize=9)
    ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    ticks = [t for t in (5, 10, 20, 50, 100, 200, 500, 1000, 2000) if base * 0.5 <= t <= best * 3]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(base * 0.7, best * 2.6)
    ax.set_xlim(-0.8, len(rows) + 3)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_ylabel("total tokens / second  (log)", color="#555")
    ax.set_xlabel("experiment", color="#555")
    ax.grid(axis="y", color="#e6e6e6")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    m = rows[0]
    ax.set_title(f"autoresearch · {m.get('bench', 'bench')} — {base:.1f} → {best:.1f} tok/s  ({best / base:.1f}×)\n",
                 loc="left", fontsize=14, fontweight="bold")
    ax.text(0, 1.015, m.get("subtitle", ""), transform=ax.transAxes, color=GREY, fontsize=9, family="monospace")
    handles = [
        plt.Line2D([], [], marker="o", color="w", markerfacecolor=BLUE, markersize=9, label="new best (staircase steps)"),
        plt.Line2D([], [], marker="o", color="w", markerfacecolor=ORANGE, markersize=7, label="ran, did not take the lead"),
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="none", markeredgecolor=ORANGE, markeredgewidth=1.8, markersize=8, label="control / repeat"),
        plt.Line2D([], [], marker="x", color=RED, linestyle="none", markersize=8, label="crash or failed equivalence check"),
    ]
    ax.legend(handles=handles, loc="center right", frameon=True, fontsize=9)
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
