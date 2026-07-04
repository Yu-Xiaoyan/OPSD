"""Aggregation tools for token-level probe sequences (stage 2).

Pure data in, pure data out — every function returns plain dicts / ndarrays and
draws nothing. Plotting lives in probes/plotting.py. Nothing is persisted here;
inputs are in-memory `[T]` sequences (e.g. from probes/divergence.py). Never
cache `[T, V]` logits (CLAUDE.md §4).
"""
from __future__ import annotations

import numpy as np


def _as_1d(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a.ravel()


def relative_position_curve(sequences, n_bins: int = 20,
                            quantiles=(0.25, 0.5, 0.75)) -> dict:
    """Normalized position curve across variable-length `[T]` sequences.

    Each token at index t in a length-T sequence is placed at relative position
    t/T ∈ [0, 1) and dropped into one of `n_bins` equal-width bins. Values from
    all sequences are pooled per bin.

    Returns a dict of ndarrays (length n_bins unless noted):
      bin_edges  [n_bins+1], bin_centers, mean, count (int),
      qNN for each requested quantile (e.g. q25/q50/q75), and `median` (=q50 if
      0.5 was requested). Empty bins are NaN (count 0).
    """
    bins: list[list[float]] = [[] for _ in range(n_bins)]
    for seq in sequences:
        arr = _as_1d(seq)
        T = arr.shape[0]
        if T == 0:
            continue
        pos = np.arange(T) / T                      # [0, 1)
        idx = np.minimum((pos * n_bins).astype(int), n_bins - 1)
        for b, v in zip(idx, arr):
            bins[b].append(float(v))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=int)
    qcurves = {q: np.full(n_bins, np.nan) for q in quantiles}
    for b in range(n_bins):
        if bins[b]:
            a = np.asarray(bins[b])
            mean[b] = a.mean()
            count[b] = a.size
            for q in quantiles:
                qcurves[q][b] = np.quantile(a, q)

    out = {"bin_edges": edges, "bin_centers": centers, "mean": mean,
           "count": count}
    for q in quantiles:
        out[f"q{int(round(q * 100))}"] = qcurves[q]
    if 0.5 in quantiles:
        out["median"] = qcurves[0.5]
    return out


def group_stats(values, labels, quantiles=(0.25, 0.5, 0.75)) -> dict:
    """Per-category summary stats over pooled token values.

    values : `[N]` token divergences (any array-like, flattened).
    labels : `[N]` category strings aligned with `values`.

    Returns {label: {count, mean, std, median, qNN...}} sorted by label.
    """
    v = _as_1d(values)
    lab = np.asarray(labels)
    if v.shape[0] != lab.shape[0]:
        raise ValueError(f"values ({v.shape[0]}) and labels ({lab.shape[0]}) "
                         "must be the same length")
    out: dict[str, dict] = {}
    for label in sorted(set(lab.tolist())):
        a = v[lab == label]
        if a.size == 0:
            continue
        d = {"count": int(a.size), "mean": float(a.mean()),
             "std": float(a.std()), "median": float(np.median(a))}
        for q in quantiles:
            d[f"q{int(round(q * 100))}"] = float(np.quantile(a, q))
        out[label] = d
    return out
