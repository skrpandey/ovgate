# ovgate

Selective-verification gating for open-vocabulary detect-then-verify pipelines.

**The finding in one line:** do not threshold an absolute image-text similarity.
Read the same embeddings as a contrast over the candidate label set you already
have. On COCO-2017 val this moves gate AUC from 0.60 to 0.73 with no new model,
no training, and no additional inference.

## Why

A detect-then-verify stack proposes regions with an open-vocabulary detector,
then verifies each crop with a vision-language model. The conventional
verification step thresholds `cosine(cutout, "a photo of a {label}")`. That
read-out discards the only thing the model is actually good at, namely how the
claimed label compares against competing ones, and it makes a near-chance gate.

The detector was already prompted with a candidate label set, so the comparison
is free.

```python
from ovgate import contrast_margin, absolute_cosine

absolute_cosine(cutout_emb, text_emb, claimed)   # the usual read-out, AUC 0.60
contrast_margin(cutout_emb, text_emb, claimed)   # this one,           AUC 0.73
```

## Install

```bash
pip install -e .
```

Requires numpy, scipy and scikit-learn. No deep learning framework: this
operates on embeddings your pipeline has already computed.

## Use

```python
import numpy as np
from ovgate import FusedGate, contrast_margin

readout = contrast_margin(cutout_emb, text_emb, claimed_label_index)
X = np.column_stack([detector_confidence, readout, truncated_flag])

gate = FusedGate().fit(X, correct, groups=image_id)
op = gate.set_operating_point(target_precision=0.95)
print(op)            # {'burden': ..., 'recall_retained': ..., 'threshold': ...}

approve = gate.approve(X_new)     # True -> library, False -> human review
```

`groups=image_id` matters: detections from one image share a scene and are not
independent, so folds are grouped by image. Without it the reported AUC is
optimistic.

`set_operating_point` raises if the target precision is unreachable, which
happens when the base rate sits far below it. Report your base rate alongside
any burden number; "95% precision" means nothing without it.

## What is in here

| Module | Contents |
|---|---|
| `ovgate.readout` | `absolute_cosine`, `contrast_margin`, `softmax_probability`, `label_rank`, `is_argmax` |
| `ovgate.gate` | `FusedGate`: image-grouped CV, standardized coefficients, operating-point selection |
| `ovgate.metrics` | `auc`, `stratified_auc`, `burden_at_precision`, `bootstrap_ci` |

`stratified_auc` is worth knowing about: raw image-text cosines are not
comparable across label prompts, so a pooled AUC mixes within-class
discrimination with between-class scale offsets. It counts concordant pairs
within a stratum only, pair-weighted.

`burden_at_precision` returns the *largest* qualifying prefix, not the first.
Precision is not monotone in the number auto-approved, so the first crossing is
not the lowest-burden operating point.

## Reproducing the paper

```bash
python scripts/reproduce_paper.py
```

regenerates the paper's gate table from `data/`, which carries the released
per-detection signals: every number is reproducible from those two CSVs alone,
with no model inference and no access to the source imagery. Single-signal
numbers are deterministic everywhere; the fused rows match the paper's digits
exactly under scikit-learn 1.9 (fold assignment changed across sklearn
versions; 1.7 gives 0.812/0.820 instead of 0.816/0.825, same conclusions).

## Scope

This is the gate, not a pipeline. It assumes you already produce, per detection:
a confidence, a cutout embedding, the candidate label text embeddings, and a
truncation flag. It is agnostic to which detector, matting model, encoder, or
hosting you use.

## Citation

Paper under review; citation to follow.
