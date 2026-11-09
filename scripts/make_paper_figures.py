#!/usr/bin/env python3
"""Paper figures from per-seed held-out results and the deployment benchmark."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reports/paper_figs"
FIGS = ROOT / "figures"
DATASETS = ("market1501", "msmt17", "dukemtmc", "occludedduke")
TITLES = {"market1501": "Market-1501", "msmt17": "MSMT17", "dukemtmc": "DukeMTMC-reID", "occludedduke": "Occluded-Duke"}
EXCLUDED = set()

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def runs(results, condition):
    keys = sorted(k for k in results if k.rsplit("_seed", 1)[0] == condition and k not in EXCLUDED)
    return [results[k] for k in keys]


def macro(entry, idx):
    return statistics.mean(entry["r1_r5_map"][d][idx] for d in DATASETS)


def check_tables(results):
    print(f"{'cond':<14}{'n':>2}  avg mAP          avg R1")
    for cond in ("R0", "B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "OSNet_R0", "ConvNeXtS_R0"):
        rs = runs(results, cond)
        m = [macro(r, 2) for r in rs]
        r1 = [macro(r, 0) for r in rs]
        print(f"{cond:<14}{len(rs):>2}  {statistics.mean(m):5.1f} +- {statistics.stdev(m):.1f}   "
              f"{statistics.mean(r1):5.1f} +- {statistics.stdev(r1):.1f}")


def fig_cmc(results):
    series = [
        ("B2", "ArcFace (selected)", "#1f5aa6", "-"),
        ("R0", "Circle-softmax + triplet (default)", "#c0392b", "--"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0), sharex=True)
    ranks = np.arange(1, 21)
    for ax, ds in zip(axes, DATASETS):
        for cond, label, color, ls in series:
            curves = np.array([r["cmc"][ds][:20] for r in runs(results, cond)])
            mean, std = curves.mean(0), curves.std(0, ddof=1)
            ax.plot(ranks, mean, color=color, ls=ls, lw=1.3, label=label)
            ax.fill_between(ranks, mean - std, mean + std, color=color, alpha=0.18, lw=0)
        ax.set_title(TITLES[ds])
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlabel("Rank")
    axes[0].set_ylabel("Matching rate (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=0.6)
    fig.savefig(FIGS / "cmc_sbs.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "cmc_sbs.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_triplet(results):
    heads = [("Linear\nsoftmax", "B0", "B1"), ("ArcFace", "B2", "B3"),
             ("Sub-center\nArcFace", "B4", "B5"), ("SphereFace2", "B6", "B7")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))
    for ax, (idx, ylabel) in zip(axes, ((2, "Avg. mAP (%)"), (0, "Avg. Rank-1 (%)"))):
        x = np.arange(len(heads))
        w = 0.36
        for off, which, color, label in ((-w / 2, 1, "#7f8c8d", "without triplet"), (w / 2, 2, "#1f5aa6", "with triplet")):
            means, stds = [], []
            for h in heads:
                vals = [macro(r, idx) for r in runs(results, h[which])]
                means.append(statistics.mean(vals))
                stds.append(statistics.stdev(vals))
            bars = ax.bar(x + off, means, w, yerr=stds, color=color, capsize=2, label=label,
                          error_kw={"lw": 0.7})
            for b, m, s in zip(bars, means, stds):
                ax.text(b.get_x() + b.get_width() / 2, m + s + 0.4, f"{m:.1f}", ha="center", va="bottom", fontsize=6)
        r0 = [macro(r, idx) for r in runs(results, "R0")]
        ax.axhline(statistics.mean(r0), color="#c0392b", ls="--", lw=0.9, label="Circle-softmax + triplet (default)")
        ax.set_xticks(x)
        ax.set_xticklabels([h[0] for h in heads])
        ax.set_ylabel(ylabel)
        lo = 25 if idx == 2 else 40
        ax.set_ylim(lo, max(means) + 7)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    fig.savefig(FIGS / "triplet_effect.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "triplet_effect.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_cost(results, cost):
    points = [
        ("SBS R50-IBN", "R0", "Circle-softmax + triplet", "o", "#c0392b"),
        ("SBS R50-IBN", "B2", "ArcFace", "s", "#1f5aa6"),
        ("OSNet", "OSNet_R0", "Circle-softmax + triplet", "o", "#c0392b"),
        ("ConvNeXt-S", "ConvNeXtS_R0", "Circle-softmax + triplet", "o", "#c0392b"),
    ]
    extra = {"OSNet": "OSNet_B2", "ConvNeXt-S": "ConvNeXtS_B2"}
    for backbone, cond in extra.items():
        if runs(results, cond):
            points.append((backbone, cond, "ArcFace", "s", "#1f5aa6"))
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    seen = set()
    for backbone, cond, obj, marker, color in points:
        vals = [macro(r, 2) for r in runs(results, cond)]
        c = cost["models"][backbone]
        size = 20 + 6 * c["params_deployed_M"]
        label = obj if obj not in seen else None
        seen.add(obj)
        ax.errorbar(c["gmacs"], statistics.mean(vals), yerr=statistics.stdev(vals), fmt="none",
                    ecolor=color, elinewidth=0.8, capsize=2)
        ax.scatter(c["gmacs"], statistics.mean(vals), s=size, marker=marker, color=color,
                   edgecolor="black", linewidth=0.5, label=label, zorder=3)
        if cond in ("R0", "OSNet_R0", "ConvNeXtS_R0"):
            offset = {"ConvNeXtS_R0": (8, -16), "OSNet_R0": (10, 6)}.get(cond, (8, -14))
            ax.annotate(f"{backbone}\n{c['params_deployed_M']:.1f}M params", (c["gmacs"], statistics.mean(vals)),
                        textcoords="offset points", xytext=offset, fontsize=6)
    ax.set_xlabel("Inference cost (GMACs per image)")
    ax.set_ylabel("Avg. held-out mAP (%)")
    ax.set_xlim(0, 12)
    ax.grid(alpha=0.25, lw=0.5)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.55, 1.09), markerscale=0.6)
    fig.tight_layout()
    fig.savefig(FIGS / "accuracy_cost.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "accuracy_cost.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    FIGS.mkdir(exist_ok=True)
    results = json.loads((DATA / "cmc_from_logs.json").read_text(encoding="utf-8"))
    cost = json.loads((DATA / "deployment_cost.json").read_text(encoding="utf-8"))
    check_tables(results)
    fig_cmc(results)
    fig_triplet(results)
    fig_cost(results, cost)
    print("figures ->", FIGS)


if __name__ == "__main__":
    main()
