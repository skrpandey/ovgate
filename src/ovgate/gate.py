"""The fused selective-verification gate.

A logistic policy over signals a detect-then-verify pipeline already emits:
detector confidence, a verification read-out, and a truncation flag. Folds are
grouped by image so no image contributes to both fitting and evaluation.

Nothing here is specific to a detector, a matting model, a vision-language
encoder, or a hosting provider.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import burden_at_precision

__all__ = ["FusedGate"]


class FusedGate:
    """Fit a gate, choose an operating point, then route detections.

    >>> gate = FusedGate().fit(X, y, groups=image_ids)
    >>> gate.set_operating_point(target_precision=0.95)
    >>> approve = gate.approve(X_new)
    """

    def __init__(self, n_splits: int = 5, seed: int = 0):
        self.n_splits, self.seed = n_splits, seed
        self.model_ = None
        self.threshold_ = None
        self.oof_ = None

    def fit(self, X, y, groups):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        cv = StratifiedGroupKFold(self.n_splits, shuffle=True,
                                  random_state=self.seed)
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=1000))
        # out-of-fold scores are what the operating point must be chosen on;
        # choosing it on in-sample scores understates burden.
        self.oof_ = cross_val_predict(pipe, X, y, cv=cv, groups=groups,
                                      method="predict_proba")[:, 1]
        self.model_ = pipe.fit(X, y)
        self.y_ = y
        return self

    @property
    def coefficients_(self):
        """Standardized coefficients, so magnitudes are comparable."""
        return self.model_.named_steps["logisticregression"].coef_[0]

    def set_operating_point(self, target_precision: float = 0.95):
        burden, recall, thr = burden_at_precision(self.oof_, self.y_,
                                                  target_precision)
        if not np.isfinite(thr):
            raise ValueError(
                f"precision {target_precision} unreachable; the base rate is "
                f"{self.y_.mean():.3f}. Lower the target or collect labels "
                "closer to the deployment distribution.")
        self.threshold_ = thr
        return {"burden": burden, "recall_retained": recall, "threshold": thr}

    def score(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=float))[:, 1]

    def approve(self, X):
        """True where the detection is auto-approved, False where deferred."""
        if self.threshold_ is None:
            raise RuntimeError("call set_operating_point() first")
        return self.score(X) >= self.threshold_
