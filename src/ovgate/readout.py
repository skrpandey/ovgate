"""Verification read-outs over a contrastive image-text embedding space.

This module is the paper's contribution in code. Given a cutout embedding and
the text embeddings of the candidate label set, it computes the read-outs that
a detect-then-verify gate can threshold. All of them consume embeddings the
pipeline already produces; none requires additional model inference.

The deployed convention in most stacks is `absolute_cosine`, which discards the
comparison against competing labels. `contrast_margin` keeps it, and on our
benchmarks lifts gate AUC from 0.60 to 0.73 at no extra cost.

Conventions: embeddings are L2-normalised, and `labels` is the candidate set
the detector was prompted with, in a fixed order shared by `text_embeddings`.
"""
from __future__ import annotations

import numpy as np

__all__ = ["absolute_cosine", "contrast_margin", "softmax_probability",
           "label_rank", "is_argmax", "READOUTS"]


def _similarities(cutout: np.ndarray, text: np.ndarray) -> np.ndarray:
    """(n, d) x (m, d) -> (n, m) cosine similarities. Both must be unit norm."""
    if cutout.ndim == 1:
        cutout = cutout[None, :]
    return cutout @ text.T


def absolute_cosine(cutout, text, claimed):
    """The deployed read-out: similarity to the claimed label alone.

    This is what most detect-then-verify pipelines threshold. It ignores how
    well competing labels fit, which is where the discriminative signal is.
    """
    s = _similarities(cutout, text)
    return s[np.arange(len(s)), claimed]


def contrast_margin(cutout, text, claimed):
    """Claimed-label similarity minus the best competing label's.

    Positive when the claimed label fits better than any alternative. Requires
    no temperature and is the read-out we recommend.
    """
    s = _similarities(cutout, text)
    rows = np.arange(len(s))
    claimed_sim = s[rows, claimed]
    s = s.copy()
    s[rows, claimed] = -np.inf
    return claimed_sim - s.max(axis=1)


def softmax_probability(cutout, text, claimed, temperature=100.0):
    """Probability mass on the claimed label under a softmax over the label set.

    `temperature` is the model's learned logit scale (about 100 for CLIP
    ViT-L/14). Results depend on it, which is why `contrast_margin` is
    preferred when a temperature is not known.
    """
    z = _similarities(cutout, text) * temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True))[np.arange(len(z)), claimed]


def label_rank(cutout, text, claimed):
    """Rank of the claimed label among all candidates, higher is better."""
    s = _similarities(cutout, text)
    order = (-s).argsort(axis=1)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(s.shape[1])[None, :], axis=1)
    return 1.0 - rank[np.arange(len(s)), claimed] / s.shape[1]


def is_argmax(cutout, text, claimed):
    """Whether the claimed label wins outright. The crudest contrast, and it
    still beats a thresholded absolute cosine."""
    s = _similarities(cutout, text)
    return (s.argmax(axis=1) == claimed).astype(float)


READOUTS = {
    "absolute_cosine": absolute_cosine,
    "contrast_margin": contrast_margin,
    "softmax_probability": softmax_probability,
    "label_rank": label_rank,
    "is_argmax": is_argmax,
}
