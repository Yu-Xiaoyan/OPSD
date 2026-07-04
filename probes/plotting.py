"""Plotting for probe aggregates (stage 2) — matplotlib, headless, savefig only.

Kept separate from aggregation so the analysis code returns pure data. These
helpers consume the dicts produced by probes/aggregate.py and write a PNG.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_position_curve(curve: dict, path: str, title: str = "",
                        ylabel: str = "per-token divergence (nats)",
                        label: str = "mean") -> str:
    """Plot a `relative_position_curve` dict (mean line + IQR band + median)."""
    x = curve["bin_centers"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if "q25" in curve and "q75" in curve:
        ax.fill_between(x, curve["q25"], curve["q75"], alpha=0.2,
                        color="#1f77b4", label="IQR (q25–q75)")
    ax.plot(x, curve["mean"], "-o", color="#1f77b4", label=label)
    if "median" in curve:
        ax.plot(x, curve["median"], "--", color="#d62728", label="median")
    ax.set_xlabel("relative position  t / T")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_group_stats(stats: dict, path: str, title: str = "",
                     ylabel: str = "per-token divergence (nats)") -> str:
    """Bar chart of per-category mean with IQR error bars (from `group_stats`)."""
    labels = list(stats.keys())
    means = [stats[l]["mean"] for l in labels]
    lo = [stats[l]["mean"] - stats[l].get("q25", stats[l]["mean"]) for l in labels]
    hi = [stats[l].get("q75", stats[l]["mean"]) - stats[l]["mean"] for l in labels]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(labels, means, yerr=[lo, hi], capsize=5,
           color=["#1f77b4", "#d62728", "#7f7f7f"][:len(labels)])
    for i, l in enumerate(labels):
        ax.text(i, means[i], f"  n={stats[l]['count']}", ha="center",
                va="bottom", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
