import numpy as np
from ovgate.readout import (absolute_cosine, contrast_margin, is_argmax,
                            softmax_probability)


def _fixture():
    text = np.eye(4)
    # cutout aligned with label 0, weakly with label 1
    cut = np.array([[0.9, 0.4, 0.0, 0.0], [0.5, 0.86, 0.0, 0.0]])
    cut /= np.linalg.norm(cut, axis=1, keepdims=True)
    return cut, text, np.array([0, 0])


def test_absolute_cosine_ignores_competitors():
    cut, text, claimed = _fixture()
    # both detections claim label 0; the absolute read-out sees only that column
    got = absolute_cosine(cut, text, claimed)
    assert np.allclose(got, cut[:, 0])


def test_margin_sign_tracks_whether_claimed_label_wins():
    cut, text, claimed = _fixture()
    m = contrast_margin(cut, text, claimed)
    assert m[0] > 0        # label 0 genuinely fits best
    assert m[1] < 0        # label 1 fits better, so the claim is suspect
    # and this is exactly what the absolute read-out cannot express
    assert absolute_cosine(cut, text, claimed)[1] > 0


def test_argmax_agrees_with_margin_sign():
    cut, text, claimed = _fixture()
    assert np.array_equal(is_argmax(cut, text, claimed) > 0,
                          contrast_margin(cut, text, claimed) > 0)


def test_softmax_is_a_probability():
    cut, text, claimed = _fixture()
    p = softmax_probability(cut, text, claimed, temperature=100.0)
    assert np.all((p >= 0) & (p <= 1))
