import numpy as np
from ovgate.metrics import auc, burden_at_precision, stratified_auc


def test_auc_of_perfect_and_random_scores():
    y = np.array([0, 0, 1, 1])
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_stratified_auc_removes_a_pure_group_offset():
    # identical within-group ordering, but group B is shifted far above A;
    # pooled AUC is corrupted by the offset, stratified AUC is not
    y = np.array([0, 1, 0, 1])
    grp = np.array(["a", "a", "b", "b"])
    score = np.array([0.0, 0.1, 10.0, 10.1])
    assert stratified_auc(y, score, grp) == 1.0
    assert auc(y, score) < 1.0


def test_burden_picks_the_lowest_burden_point_not_the_first():
    # precision is non-monotone in k; the largest qualifying k is the answer
    y = np.array([1, 1, 1, 0, 1, 1, 1, 1, 1, 1])
    score = np.arange(10)[::-1].astype(float)
    burden, recall, thr = burden_at_precision(score, y, target=0.9)
    assert burden == 0.0        # all ten qualify at 0.9 overall precision
    assert recall == 1.0


def test_unreachable_target_returns_nan():
    # the highest-scoring detection is wrong, so no prefix reaches 0.95
    y = np.array([0, 0, 0, 1])
    score = np.array([3.0, 2.0, 1.0, 0.0])
    burden, recall, thr = burden_at_precision(score, y, target=0.95)
    assert np.isnan(burden) and np.isnan(thr)


def test_stratified_auc_accepts_string_strata():
    y = np.array([0, 1, 0, 1])
    score = np.array([0.0, 0.1, 10.0, 10.1])
    assert stratified_auc(y, score, np.array(["cup", "cup", "mug", "mug"])) == 1.0
