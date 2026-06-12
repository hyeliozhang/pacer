#!/usr/bin/env python3
"""Generate publication-quality PACER figures and compact LaTeX tables.

The experimental plots use matplotlib. The two conceptual figures are rendered
through LaTeX/TikZ when TeX is available, with a deterministic matplotlib
fallback. All outputs are vector PDFs tuned for IEEE two-column rendering.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

# Print-oriented palette: muted, color-blind aware, and still distinct in grayscale.
COLORS = {
    "ExactSecure": "#374151",
    "PostFilter-IVF": "#B85C38",
    "AdaptivePostFilter-IVF": "#C9902E",
    "ExactPrefix-PostFilter": "#6B7280",
    "TenantPartition-IVF": "#8B5E83",
    "PreFilter-Exact": "#4E88A8",
    "BitmapSlice-Exact": "#2C6E91",
    "PreFilter-IVF": "#508A6D",
    "PACER-A": "#1F5A85",
    "PACER-C": "#287D69",
    "PACER-X": "#111827",
    "PACER-A-NoSlices": "#D6B54D",
    "PACER-A-NoBounds": "#6A9FB5",
    "StaleNoRecheck": "#B85C38",
    "NaiveNoPolicy": "#8F1D2C",
    "ScalarOnly": "#7C8490",
}

LABEL = {
    "ExactSecure": "Exact",
    "PostFilter-IVF": "Post-IVF",
    "AdaptivePostFilter-IVF": "Adapt-IVF",
    "ExactPrefix-PostFilter": "ExactPref",
    "TenantPartition-IVF": "Tenant-IVF",
    "PreFilter-Exact": "PreExact",
    "BitmapSlice-Exact": "SliceExact",
    "PreFilter-IVF": "Pre-IVF",
    "PACER-A": "PACER-A",
    "PACER-C": "PACER-C",
    "PACER-X": "PACER-X",
    "PACER-A-NoSlices": "NoSlices",
    "PACER-A-NoBounds": "NoBounds",
    "StaleNoRecheck": "Stale",
    "NaiveNoPolicy": "NoPolicy",
    "ScalarOnly": "Scalar",
}

MARKERS = {
    "ExactSecure": "*",
    "PostFilter-IVF": "o",
    "AdaptivePostFilter-IVF": "s",
    "ExactPrefix-PostFilter": "P",
    "TenantPartition-IVF": "^",
    "PreFilter-Exact": "X",
    "BitmapSlice-Exact": "D",
    "PreFilter-IVF": "v",
    "PACER-A": "v",
    "PACER-C": "P",
    "PACER-X": "X",
    "PACER-A-NoSlices": "h",
    "PACER-A-NoBounds": "d",
}
LINESTYLES = ["-", "--", ":", "-."]
HATCHES = ["", "", "", "", "", ""]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.2,
    "axes.titlesize": 7.6,
    "axes.labelsize": 7.4,
    "xtick.labelsize": 6.6,
    "ytick.labelsize": 6.6,
    "legend.fontsize": 6.3,
    "axes.linewidth": 0.55,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "xtick.minor.width": 0.45,
    "ytick.minor.width": 0.45,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 400,
})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def mean(vals: Iterable[float | str]) -> float:
    xs = [float(v) for v in vals]
    return float(np.mean(xs)) if xs else 0.0


def tex(s: str) -> str:
    return s.replace("_", "\\_").replace("%", "\\%")


def polish_axes(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, linewidth=0.35, alpha=0.28, color="#6B7280")
    ax.set_axisbelow(True)


def savefig(path: Path, fig: plt.Figure) -> None:
    try:
        fig.tight_layout(pad=0.25)
    except Exception:
        pass
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.045, metadata={"Creator": "PACER artifact"})
    plt.close(fig)


def _rounded(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str, *,
             face: str, edge: str, fs: float = 7.2, weight: str = "normal") -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.020",
                                linewidth=0.75, edgecolor=edge, facecolor=face))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color="#111827", weight=weight, linespacing=1.05)


def _arrow(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, *,
           color: str = "#334155", lw: float = 0.85, rad: float = 0.0) -> None:
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, shrinkA=3, shrinkB=3,
                                mutation_scale=8, connectionstyle=f"arc3,rad={rad}"))


def architecture(path: Path) -> None:
    """Clean full-width PACER system diagram optimized after visual review."""
    fig, ax = plt.subplots(figsize=(7.16, 1.96))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    cards = [
        (0.035, 0.575, 0.190, "QUERY", "$q, k$, policy, epoch", "#F8FAFC", "#64748B"),
        (0.290, 0.575, 0.225, "SUMMARY PLANNER", "possible units\nscore bounds", "#F0F9FF", "#2C7A7B"),
        (0.585, 0.575, 0.215, "ROW VERIFIER", "visibility facts\nrow witness", "#FFF7ED", "#B45309"),
        (0.860, 0.575, 0.115, "STATUS", "top-k\n+ label", "#F8FAFC", "#475569"),
    ]
    for x, y, w, title, body, face, edge in cards:
        ax.add_patch(FancyBboxPatch((x, y), w, 0.245, boxstyle="round,pad=0.012,rounding_size=0.022",
                                    linewidth=0.85, edgecolor=edge, facecolor=face))
        ax.text(x + w / 2, y + 0.172, title, fontsize=6.7, color=edge, weight="bold", ha="center", va="center")
        ax.text(x + w / 2, y + 0.082, body, fontsize=6.9, color="#111827", weight="bold", ha="center", va="center", linespacing=1.0)
    _arrow(ax, 0.225, 0.700, 0.290, 0.700, color="#334155")
    _arrow(ax, 0.515, 0.700, 0.585, 0.700, color="#334155")
    _arrow(ax, 0.800, 0.700, 0.860, 0.700, color="#334155")

    lower = [
        (0.055, 0.300, 0.145, "selection\nthen top-k", "#FFFFFF", "#94A3B8"),
        (0.280, 0.300, 0.105, "no-FN\nsummaries", "#FFFFFF", "#2C7A7B"),
        (0.405, 0.300, 0.105, "policy\nrow slices", "#FFFFFF", "#2C7A7B"),
        (0.600, 0.300, 0.110, "tenant / scalar\n/provenance", "#FFFFFF", "#B45309"),
        (0.730, 0.300, 0.110, "insertion /\ndeletion", "#FFFFFF", "#B45309"),
        (0.860, 0.300, 0.110, "bounds or\nexhaustion", "#FFFFFF", "#475569"),
    ]
    for x, y, w, text, face, edge in lower:
        ax.add_patch(FancyBboxPatch((x, y), w, 0.145, boxstyle="round,pad=0.010,rounding_size=0.018",
                                    linewidth=0.65, edgecolor=edge, facecolor=face))
        ax.text(x + w / 2, y + 0.072, text, fontsize=6.2, color="#111827", ha="center", va="center", linespacing=1.00)
    _arrow(ax, 0.128, 0.445, 0.128, 0.575, color="#64748B", lw=0.68)
    _arrow(ax, 0.333, 0.445, 0.350, 0.575, color="#2C7A7B", lw=0.68)
    _arrow(ax, 0.458, 0.445, 0.455, 0.575, color="#2C7A7B", lw=0.68)
    _arrow(ax, 0.655, 0.445, 0.680, 0.575, color="#B45309", lw=0.68)
    _arrow(ax, 0.785, 0.445, 0.705, 0.575, color="#B45309", lw=0.68)
    _arrow(ax, 0.915, 0.445, 0.918, 0.575, color="#475569", lw=0.68)

    for x, label, fill in [(0.245, "checked approximate", "#E0F2FE"), (0.425, "contract-refined", "#DCFCE7"), (0.605, "certified exact", "#E5E7EB")]:
        ax.add_patch(FancyBboxPatch((x, 0.135), 0.155, 0.075, boxstyle="round,pad=0.008,rounding_size=0.030",
                                    linewidth=0.55, edgecolor="#64748B", facecolor=fill))
        ax.text(x + 0.0775, 0.172, label, ha="center", va="center", fontsize=6.3, color="#111827")
    ax.text(0.035, 0.045,
            "Invariant: summaries guide work only; row facts authorize output; exact labels require bounds or exhaustion.",
            fontsize=6.7, color="#334155")
    savefig(path, fig)

def executor_contract(path: Path) -> None:
    """Single-column visual replacement for the old raw algorithmic listing."""
    fig, ax = plt.subplots(figsize=(3.48, 2.30))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    steps = [
        (0.06, 0.790, "1", "order units", "upper-bound priority"),
        (0.06, 0.595, "2", "slice candidates", "conservative SQL + exact row slice"),
        (0.06, 0.400, "3", "verify and score", "visibility, epoch, witness, top-k heap"),
        (0.06, 0.205, "4", "stop or continue", "budget / refinement / certificate"),
    ]
    for x, y, num, title, subtitle in steps:
        ax.add_patch(FancyBboxPatch((x, y), 0.88, 0.125, boxstyle="round,pad=0.012,rounding_size=0.022",
                                    linewidth=0.70, edgecolor="#CBD5E1", facecolor="#FFFFFF"))
        ax.add_patch(plt.Circle((x + 0.047, y + 0.063), 0.029, facecolor="#1F5A85", edgecolor="none"))
        ax.text(x + 0.047, y + 0.063, num, ha="center", va="center", fontsize=6.3, color="white", weight="bold")
        ax.text(x + 0.090, y + 0.080, title, ha="left", va="center", fontsize=7.2, weight="bold", color="#111827")
        ax.text(x + 0.090, y + 0.043, subtitle, ha="left", va="center", fontsize=6.35, color="#475569")
    for y0, y1 in [(0.790, 0.720), (0.595, 0.525), (0.400, 0.330)]:
        _arrow(ax, 0.50, y0, 0.50, y1, color="#64748B", lw=0.65)
    # Termination/status capsule.
    status_x = [0.08, 0.37, 0.66]
    labels = [("PACER-A", "checked", "#E0F2FE"), ("PACER-C", "refined", "#DCFCE7"), ("PACER-X", "certified", "#E5E7EB")]
    for x, (name, state, fill) in zip(status_x, labels):
        ax.add_patch(FancyBboxPatch((x, 0.040), 0.235, 0.075, boxstyle="round,pad=0.010,rounding_size=0.030",
                                    linewidth=0.55, edgecolor="#64748B", facecolor=fill))
        ax.text(x + 0.118, 0.079, name, ha="center", va="center", fontsize=6.8, weight="bold", color="#111827")
        ax.text(x + 0.118, 0.052, state, ha="center", va="center", fontsize=5.9, color="#475569")
    ax.text(0.06, 0.145, "Only verified row facts may authorize output.", fontsize=5.9, color="#B45309", va="center")
    savefig(path, fig)

def recall_latency(path: Path, summary: dict) -> None:
    methods = ["PostFilter-IVF", "AdaptivePostFilter-IVF", "TenantPartition-IVF", "BitmapSlice-Exact", "PACER-A", "PACER-C", "PACER-X", "ExactSecure"]
    fig, ax = plt.subplots(figsize=(3.48, 2.45))
    for m in methods:
        v = summary["overall"][m]
        size = 58 if m.startswith("PACER") else 42
        ax.scatter(v["median_latency_ms"], v["recall_at_k"], marker=MARKERS.get(m, "o"), s=size,
                   color=COLORS.get(m, "#666666"), edgecolors="white", linewidths=0.65, label=LABEL[m], zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Median latency (ms, log)")
    ax.set_ylabel("Secure recall@10")
    ax.set_ylim(0.0, 1.06)
    ax.set_xlim(0.35, 12.5)
    polish_axes(ax)
    ax.axhline(1.0, color="#475569", linewidth=0.55, linestyle=":", zorder=1)
    # Direct labels for the main contract states reduce legend-reading load.
    offsets = {
        "PostFilter-IVF": (8, -8),
        "PACER-A": (-30, -16),
        "PACER-C": (-54, 13),
        "PACER-X": (10, -18),
        "ExactSecure": (6, 8),
    }
    for m, (dx, dy) in offsets.items():
        v = summary["overall"][m]
        ax.annotate(LABEL[m], (v["median_latency_ms"], v["recall_at_k"]), textcoords="offset points", xytext=(dx, dy),
                    fontsize=6.4, color=COLORS.get(m, "#111827"), weight="bold" if m.startswith("PACER") else "normal")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.96,
              ncol=2, loc="lower right", handletextpad=0.3, columnspacing=0.55, borderpad=0.25)
    savefig(path, fig)


def budget_recall(path: Path, budget: list[dict[str, str]], summary: dict | None = None) -> None:
    fig, ax = plt.subplots(figsize=(3.48, 2.38))
    for i, m in enumerate(["PostFilter-IVF", "AdaptivePostFilter-IVF", "PACER-A"]):
        xs = sorted({int(r["budget"]) for r in budget if r["method"] == m})
        ys = [mean(r["recall_at_k"] for r in budget if r["method"] == m and int(r["budget"]) == b) for b in xs]
        ax.plot(xs, ys, label=LABEL[m], marker=MARKERS.get(m, "o"), linestyle=LINESTYLES[i], linewidth=1.35,
                markersize=4.3, color=COLORS.get(m, "#666666"), markeredgecolor="white", markeredgewidth=0.45)
    if summary:
        ax.axhline(summary["overall"]["PACER-C"]["recall_at_k"], color=COLORS["PACER-C"], linewidth=0.9, linestyle="--", alpha=0.85)
        ax.text(160, 1.01, "PACER-C contract", color=COLORS["PACER-C"], fontsize=6.3, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("Raw-row budget")
    ax.set_ylabel("Secure recall@10")
    ax.set_ylim(0.0, 1.06)
    polish_axes(ax)
    ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.96, loc="lower right", borderpad=0.25)
    savefig(path, fig)


def grouped_regime_candidates(path: Path, summary: dict, regimes: list[str]) -> None:
    methods = ["PostFilter-IVF", "TenantPartition-IVF", "PreFilter-IVF", "PACER-A"]
    fig, ax = plt.subplots(figsize=(7.16, 2.18))
    x = np.arange(len(regimes))
    width = 0.18
    for i, m in enumerate(methods):
        vals = [summary["by_regime"][reg][m]["raw_candidates"] for reg in regimes]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=LABEL[m], color=COLORS.get(m, "#999999"),
                      edgecolor="white", linewidth=0.4, hatch=HATCHES[i])
    ax.set_xticks(x)
    ax.set_xticklabels([r.capitalize() for r in regimes])
    ax.set_ylabel("Raw identifiers")
    ax.set_xlabel("Policy regime")
    polish_axes(ax, "y")
    ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18), borderpad=0.25)
    savefig(path, fig)


def ablation_recall(path: Path, summary: dict) -> None:
    methods = ["PostFilter-IVF", "PACER-A-NoSlices", "PACER-A-NoBounds", "PACER-A", "PACER-C"]
    vals = [summary["overall"][m]["recall_at_k"] for m in methods]
    fig, ax = plt.subplots(figsize=(3.48, 2.28))
    y = np.arange(len(methods))
    bars = ax.barh(y, vals, color=[COLORS.get(m, "#888888") for m in methods], edgecolor="white", linewidth=0.45)
    for i, b in enumerate(bars):
        b.set_hatch(HATCHES[i % len(HATCHES)])
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[m] for m in methods])
    ax.set_xlabel("Secure recall@10")
    ax.set_xlim(0.0, 1.06)
    polish_axes(ax, "x")
    for i, v in enumerate(vals):
        ax.text(min(v + 0.018, 1.02), i, f"{v:.2f}", va="center", ha="left", fontsize=6.4, color="#111827")
    savefig(path, fig)


def failures(path: Path, failure: list[dict[str, str]]) -> None:
    methods = ["PostFilter-IVF", "AdaptivePostFilter-IVF", "PreFilter-IVF", "PACER-A", "PACER-C"]
    x = np.arange(len(methods))
    starve = np.array([mean(r["fraction"] for r in failure if r["method"] == m and r["failure_mode"] == "candidate_starvation") for m in methods])
    trunc = np.array([mean(r["fraction"] for r in failure if r["method"] == m and r["failure_mode"] == "ranking_truncation") for m in methods])
    fig, ax = plt.subplots(figsize=(3.48, 2.32))
    ax.bar(x, starve, 0.58, label="Starvation", color="#9ECAE1", edgecolor="white", linewidth=0.45, hatch="//")
    ax.bar(x, trunc, 0.58, bottom=starve, label="Truncation", color="#FDD0A2", edgecolor="white", linewidth=0.45, hatch="\\\\")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[m] for m in methods], rotation=18, ha="right")
    ax.set_ylabel("Fraction of queries")
    ax.set_ylim(0.0, 1.0)
    polish_axes(ax, "y")
    ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.96, loc="upper right", borderpad=0.25)
    savefig(path, fig)


def scalability(path: Path, scale_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.48, 2.28))
    if scale_path.exists():
        rows = read_csv(scale_path)
        for i, m in enumerate(["PostFilter-IVF", "BitmapSlice-Exact", "PACER-A", "PACER-C", "PACER-X", "ExactSecure"]):
            pairs = sorted((int(r["n"]), float(r["median_latency_ms"])) for r in rows if r["method"] == m)
            if pairs:
                ax.plot([p[0] for p in pairs], [p[1] for p in pairs], label=LABEL[m], marker=MARKERS.get(m, "o"),
                        linestyle=LINESTYLES[i % len(LINESTYLES)], linewidth=1.15, markersize=4.0,
                        color=COLORS.get(m, "#666666"), markeredgecolor="white", markeredgewidth=0.4)
    ax.set_xlabel("Vectors")
    ax.set_ylabel("Median latency (ms)")
    polish_axes(ax)
    ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=6.3, ncol=2, borderpad=0.25)
    savefig(path, fig)


def stress_correlation(path: Path, stress_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.16, 2.18))
    if stress_path.exists():
        rows = read_csv(stress_path)
        scenarios: list[str] = []
        for r in rows:
            if r["scenario"] not in scenarios:
                scenarios.append(r["scenario"])
        methods = ["PostFilter-IVF", "PreFilter-Exact", "PACER-A", "PACER-C", "PACER-X"]
        x = np.arange(len(scenarios))
        width = 0.16
        for i, m in enumerate(methods):
            vals = [mean(r["recall_at_k"] for r in rows if r["scenario"] == sc and r["method"] == m) for sc in scenarios]
            ax.bar(x + (i - 2) * width, vals, width, label=LABEL[m], color=COLORS.get(m, "#999999"),
                   edgecolor="white", linewidth=0.4, hatch=HATCHES[i])
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=10, ha="right")
    ax.set_ylabel("Secure recall@10")
    ax.set_ylim(0.0, 1.06)
    polish_axes(ax, "y")
    ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18), borderpad=0.25)
    savefig(path, fig)


def violations(path: Path, summary: dict) -> None:
    methods = ["ExactSecure", "PostFilter-IVF", "PreFilter-IVF", "PACER-A", "PACER-C", "PACER-X", "StaleNoRecheck", "NaiveNoPolicy"]
    vals = [summary["overall"][m]["policy_violations_per_query"] + summary["overall"][m]["deleted_violations_per_query"] + summary["overall"][m]["tenant_violations_per_query"] for m in methods]
    fig, ax = plt.subplots(figsize=(3.48, 2.24))
    bars = ax.bar(np.arange(len(methods)), vals, color=[COLORS.get(m, "#999999") for m in methods], edgecolor="white", linewidth=0.45)
    for i, b in enumerate(bars):
        b.set_hatch(HATCHES[i % len(HATCHES)])
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels([LABEL[m] for m in methods], rotation=24, ha="right")
    ax.set_ylabel("Returned violations/query")
    polish_axes(ax, "y")
    savefig(path, fig)


def write_tables(out: Path, summary: dict, regimes: list[str], stress_path: Path) -> None:
    selected = ["ExactSecure", "PreFilter-Exact", "BitmapSlice-Exact", "PostFilter-IVF", "AdaptivePostFilter-IVF", "ExactPrefix-PostFilter", "TenantPartition-IVF", "PreFilter-IVF", "PACER-A", "PACER-C", "PACER-X", "StaleNoRecheck", "NaiveNoPolicy"]
    with (out / "table_main_results.tex").open("w") as f:
        f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\nMethod & Rec. & Exact & Cert. & Raw & Dot & Viol.\\\\\n\\midrule\n")
        for m in selected:
            if m in ["PostFilter-IVF", "PACER-A", "StaleNoRecheck"]:
                f.write("\\midrule\n")
            v = summary["overall"][m]
            viol = v["policy_violations_per_query"] + v["deleted_violations_per_query"] + v["tenant_violations_per_query"]
            f.write(f"{tex(LABEL.get(m,m))} & {v['recall_at_k']:.3f} & {v['secure_topk_exact']:.3f} & {v['certified_fraction']:.2f} & {v['raw_candidates']:.0f} & {v['distance_evals']:.0f} & {viol:.2f}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    with (out / "table_regime_results.tex").open("w") as f:
        f.write("\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\nRegime & Visible & Post & Pre-IVF & PACER-A & PACER-X\\\\\n\\midrule\n")
        for reg in regimes:
            v = summary["by_regime"][reg]
            f.write(f"{reg.capitalize()} & {v['ExactSecure']['candidates']:.0f} & {v['PostFilter-IVF']['recall_at_k']:.3f} & {v['PreFilter-IVF']['recall_at_k']:.3f} & {v['PACER-A']['recall_at_k']:.3f} & {v['PACER-X']['secure_topk_exact']:.3f}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    with (out / "table_certificates.tex").open("w") as f:
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\nMethod & Cert. & Exact & BadCert & Gap\\\\\n\\midrule\n")
        for m in ["PACER-A", "PACER-C", "PACER-X", "PACER-A-NoSlices", "PACER-A-NoBounds"]:
            v = summary["overall"][m]
            f.write(f"{tex(LABEL.get(m,m))} & {v['certified_fraction']:.2f} & {v['secure_topk_exact']:.2f} & {v['certificate_errors']:.0f} & {v['mean_certificate_gap']:.3f}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    with (out / "table_deletion.tex").open("w") as f:
        f.write("\\begin{tabular}{rrrr}\n\\toprule\nEpoch & Tombstones & PACER-X del. & Stale del.\\\\\n\\midrule\n")
        for row in summary.get("deletion", []):
            if int(row["epoch"]) in [1, 3, 5, 7, 9]:
                f.write(f"{row['epoch']} & {row['stale_residues_in_index']} & {row['pacer_certify_deleted_results']} & {row['stale_no_recheck_deleted_results']}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    with (out / "table_overhead.tex").open("w") as f:
        o = summary["index_overhead"]
        f.write("\\begin{tabular}{lr}\n\\toprule\nComponent & Bytes/record\\\\\n\\midrule\n")
        for label, key in [("Assignment", "assignment_bytes_per_record"), ("Centroids", "centroid_bytes_per_record"), ("Radii", "radius_bytes_per_record"), ("Policy summaries", "summary_bytes_per_record"), ("Total structural metadata", "total_structural_bytes_per_record")]:
            f.write(f"{label} & {o[key]:.2f}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    with (out / "table_stress.tex").open("w") as f:
        f.write("\\begin{tabular}{llrrr}\n\\toprule\nScenario & Method & Rec. & Exact & Viol.\\\\\n\\midrule\n")
        if stress_path.exists():
            for r in read_csv(stress_path):
                if r["method"] in ["PostFilter-IVF", "PreFilter-Exact", "PACER-A", "PACER-C", "PACER-X"]:
                    viol = float(r["policy_violations_per_query"]) + float(r["deleted_violations_per_query"]) + float(r.get("tenant_violations_per_query", 0.0))
                    f.write(f"{tex(r['scenario'])} & {tex(LABEL.get(r['method'], r['method']))} & {float(r['recall_at_k']):.3f} & {float(r['secure_topk_exact']):.3f} & {viol:.2f}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    dyn_summary = stress_path.parent / "dynamic_overlay" / "dynamic_overlay_summary.json"
    with (out / "table_dynamic_overlay.tex").open("w") as f:
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\nMethod & Rec. & Exact & Raw & Rebuild ms\\\\\n\\midrule\n")
        if dyn_summary.exists():
            d = json.load(dyn_summary.open())
            labels = {"BaseOnly-Stale": "Stale-base", "DeltaPACER": "DeltaPACER", "FreshSlice-Rebuild": "Fresh-rebuild"}
            for m in ["BaseOnly-Stale", "DeltaPACER", "FreshSlice-Rebuild"]:
                v = d["methods"][m]
                f.write(f"{labels[m]} & {v['recall_at_k']:.3f} & {v['secure_topk_exact']:.3f} & {v['raw_checks']:.0f} & {v['rebuild_ms']:.1f}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    ap.add_argument("--figures", default=None)
    ap.add_argument("--n", type=int, default=15000)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    res = Path(args.results) if args.results else root / "results"
    out = Path(args.figures) if args.figures else root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    summary = json.load((res / f"summary_n{args.n}.json").open())
    budget = read_csv(res / f"budget_n{args.n}.csv")
    failure = read_csv(res / f"failure_modes_n{args.n}.csv")
    regimes = [r for r in ["broad", "medium", "narrow", "cold"] if r in summary["by_regime"]]

    try:
        from render_tikz_diagrams import render_architecture, render_executor
        render_architecture(out / "fig_semantics_architecture.pdf")
        render_executor(out / "fig_executor_contract.pdf")
        (res / "render_tikz_diagrams.log").write_text("TikZ architecture/executor figures rendered successfully.\n")
    except Exception as exc:  # pragma: no cover - artifact environments may lack TeX
        (res / "render_tikz_diagrams.log").write_text(f"TikZ rendering failed; using matplotlib fallback: {exc}\n")
        architecture(out / "fig_semantics_architecture.pdf")
        executor_contract(out / "fig_executor_contract.pdf")
    recall_latency(out / "fig_recall_latency.pdf", summary)
    budget_recall(out / "fig_budget_recall.pdf", budget, summary)
    grouped_regime_candidates(out / "fig_candidates_by_regime.pdf", summary, regimes)
    ablation_recall(out / "fig_ablation_recall.pdf", summary)
    failures(out / "fig_failure_modes.pdf", failure)
    scalability(out / "fig_scalability_latency.pdf", res / "scalability.csv")
    stress_correlation(out / "fig_stress_correlation.pdf", res / "stress_summary.csv")
    violations(out / "fig_policy_violations.pdf", summary)
    write_tables(out, summary, regimes, res / "stress_summary.csv")


if __name__ == "__main__":
    main()
