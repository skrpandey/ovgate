"""Selective-verification gating for open-vocabulary detect-then-verify stacks.

The short version: do not threshold an absolute image-text similarity. Read the
same embeddings as a contrast over the candidate label set you already have.
"""
from .gate import FusedGate
from .readout import (READOUTS, absolute_cosine, contrast_margin, is_argmax,
                      label_rank, softmax_probability)

__version__ = "0.1.0"
__all__ = ["FusedGate", "READOUTS", "absolute_cosine", "contrast_margin",
           "is_argmax", "label_rank", "softmax_probability"]
