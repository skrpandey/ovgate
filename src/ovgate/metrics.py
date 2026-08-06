"""Evaluation for selective-verification gates.

Burden and retained recall at a target auto-approval precision are the
quantities a deployment actually cares about; AUC is the summary. Intervals are
a cluster bootstrap over images, because detections within an image share a
scene and are not independent.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

__all__ = ["auc", "stratified_auc", "burden_at_precision", "bootstrap_ci"]


def auc(y, score):
    """Rank-based AUC. Ties get midranks."""
    y = np.asarray(y)
    npos, nneg = float(y.sum()), float(len(y) - y.sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def stratified_auc(y, score, strata):
    """Concordant pairs counted only within a stratum, pooled across strata.

    Use when scores are not comparable across groups, which is the case for raw
    image-text cosines across different label prompts. Pair-weighted, so it does
    not inherit the instability of averaging per-stratum AUCs.
    """
    y, score = np.asarray(y), np.asarray(score)
    # strata may be class names; np.diff needs an orderable numeric key
    _, strata = np.unique(np.asarray(strata), return_inverse=True)
    order = np.argsort(strata, kind="stable")
    y, score, strata = y[order], score[order], strata[order]
    num = den = 0.0
    for lo, hi in _spans(strata):
        a = auc(y[lo:hi], score[lo:hi])
        w = float(y[lo:hi].sum()) * float(len(y[lo:hi]) - y[lo:hi].sum())
        if w > 0 and np.isfinite(a):
            num, den = num + a * w, den + w
    return num / den if den else float("nan")


def _spans(sorted_keys):
    bounds = np.flatnonzero(np.diff(sorted_keys)) + 1
    return zip(np.r_[0, bounds], np.r_[bounds, len(sorted_keys)])


def burden_at_precision(score, y, target=0.95):
    """Lowest review burden whose auto-approved set reaches `target` precision.

    Auto-approve the top-k by score. Precision is not monotone in k, so the
    lowest-burden operating point is the largest k meeting the target, not the
    first one encountered.

    Returns (burden, recall_retained, threshold), or (nan, nan, nan) if the
    target is unreachable, which happens when the base rate is far below it.
    """
    score, y = np.asarray(score), np.asarray(y)
    order = np.argsort(-score, kind="stable")
    ys, ss = y[order], score[order]
    n, npos = len(ys), float(ys.sum())
    cum = np.cumsum(ys)
    k = np.arange(1, n + 1)
    ok = np.flatnonzero(cum / k >= target)
    if not len(ok) or npos == 0:
        return float("nan"), float("nan"), float("nan")
    kb = int(ok.max()) + 1
    return (n - kb) / n, cum[kb - 1] / npos, float(ss[kb - 1])


def bootstrap_ci(fn, groups, n_boot=1000, seed=0, alpha=0.05):
    """Percentile CI for `fn(index_array)` under a cluster bootstrap on `groups`.

    Pass the same resample indices to several statistics to compare them on
    identical data; unpaired intervals overstate uncertainty in a difference.
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    per = [np.flatnonzero(groups == g) for g in uniq]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(per), size=len(per))
        vals.append(fn(np.concatenate([per[i] for i in pick])))
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()), (float(np.percentile(a, 100 * alpha / 2)),
                             float(np.percentile(a, 100 * (1 - alpha / 2))))
