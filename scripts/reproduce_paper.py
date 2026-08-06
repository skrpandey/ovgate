#!/usr/bin/env python3
"""Reproduce the paper's gate table from the released per-detection data.

Runs in about a minute on a laptop. No GPU, no model inference, no images:
`data/*.csv` carry one row per detection with the signals and the
non-circular class-present label.

Exact reproduction note: single-signal numbers are deterministic in any
environment. The fused rows depend on cross-validation fold assignment,
which changed between scikit-learn versions; scikit-learn 1.9 reproduces the
paper's digits exactly (0.816 / 0.825 on COCO), 1.7 gives 0.812 / 0.820.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from ovgate import FusedGate, absolute_cosine  # noqa: F401  (package import check)
from ovgate.metrics import auc, burden_at_precision

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"

SINGLES = [("clip_score", "CLIP cosine (deployed)"),
           ("confidence", "detector confidence"),
           ("margin", "CLIP contrast margin"),
           ("softmax_scaled", "softmax over label set")]
FUSIONS = [("3-signal fusion", ["confidence", "margin", "truncated"]),
           ("4-signal fusion", ["confidence", "margin", "truncated", "clip_score"])]


def run(name: str, path: pathlib.Path, operating_points: bool) -> None:
    df = pd.read_csv(path)
    y = df.y.to_numpy()
    groups = pd.factorize(df.image)[0]
    print(f"\n{name}: {len(df)} detections, {df.image.nunique()} images, "
          f"base rate {y.mean():.3f}")
    print(f"{'signal':<26}{'AUC':>7}{'burden@P95':>12}{'recall kept':>13}")

    def line(label, score):
        a = auc(y, score)
        if operating_points:
            b, r, _ = burden_at_precision(score, y)
            print(f"{label:<26}{a:>7.3f}{b:>12.3f}{r:>13.3f}")
        else:
            print(f"{label:<26}{a:>7.3f}{'n/a':>12}{'n/a':>13}")

    for col, label in SINGLES:
        line(label, df[col].to_numpy(float))
    for label, feats in FUSIONS:
        gate = FusedGate().fit(df[feats].to_numpy(float), y, groups=groups)
        line(label, gate.oof_)


def main() -> None:
    run("COCO-2017 val", DATA / "detections_coco.csv", operating_points=True)
    # Open Images' 0.384 base rate puts the P=0.95 target out of reach for
    # every signal (burden 0.99+), so only AUC is meaningful there.
    run("Open Images V7 val", DATA / "detections_open_images.csv",
        operating_points=False)


if __name__ == "__main__":
    main()
