"""Drop-in usage: you already run a detector and a CLIP-like encoder.

Swapping `absolute_cosine` for `contrast_margin` on the line marked below is
the entire change the paper recommends.
"""
import numpy as np
from ovgate import FusedGate, contrast_margin
from ovgate.metrics import auc

rng = np.random.default_rng(0)
n, n_labels, dim = 2000, 20, 768

# whatever your pipeline already produces, per detection
cutout_emb = rng.normal(size=(n, dim)); cutout_emb /= np.linalg.norm(cutout_emb, axis=1, keepdims=True)
text_emb = rng.normal(size=(n_labels, dim)); text_emb /= np.linalg.norm(text_emb, axis=1, keepdims=True)
claimed = rng.integers(0, n_labels, size=n)
truncated = (rng.uniform(size=n) < 0.08).astype(float)
image_id = rng.integers(0, 300, size=n)

# a correct detection looks like its claimed label; an incorrect one looks like
# some other label, which is precisely the structure a contrast read-out sees
# and an absolute similarity does not
correct = (rng.uniform(size=n) < 0.80).astype(int)
looks_like = np.where(correct == 1, claimed, (claimed + 1) % n_labels)
cutout_emb = text_emb[looks_like] + 0.9 * rng.normal(size=(n, dim))
cutout_emb /= np.linalg.norm(cutout_emb, axis=1, keepdims=True)
confidence = np.clip(rng.normal(np.where(correct == 1, .72, .48), .12), .3, .95)

readout = contrast_margin(cutout_emb, text_emb, claimed)   # <- the change
X = np.column_stack([confidence, readout, truncated])

gate = FusedGate().fit(X, correct, groups=image_id)
op = gate.set_operating_point(target_precision=0.95)

print(f"gate AUC (out of fold): {auc(correct, gate.oof_):.3f}")
print(f"standardized coefficients: {np.round(gate.coefficients_, 3)}")
print(f"review burden: {op['burden']:.3f}   recall retained: {op['recall_retained']:.3f}")
print(f"auto-approved: {gate.approve(X).mean():.1%} of detections")
