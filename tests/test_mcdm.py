"""Kiểm tra cài đặt MCDM: đối chiếu với pymcdm + các tính chất lý thuyết.

    PYTHONPATH=src python -m pytest tests -q
"""
import numpy as np
import pytest

from drm import mcdm

rng = np.random.default_rng(42)
X = rng.uniform(1, 10, size=(8, 5))
TYPES = np.array([1, 1, -1, 1, -1])
W = np.array([0.3, 0.2, 0.15, 0.25, 0.1])

pymcdm = pytest.importorskip("pymcdm")
from pymcdm import methods as M, normalizations as N, weights as PW  # noqa: E402


def test_topsis_matches_pymcdm():
    ref = M.TOPSIS(N.vector_normalization)(X, W, TYPES)
    np.testing.assert_allclose(mcdm.score_topsis(X, W, TYPES), ref, rtol=1e-9)


def test_topsis_minmax_matches_pymcdm():
    ref = M.TOPSIS(N.minmax_normalization)(X, W, TYPES)
    np.testing.assert_allclose(mcdm.score_topsis(X, W, TYPES, norm="minmax"), ref, rtol=1e-9)


def test_vikor_matches_pymcdm():
    ref = M.VIKOR(v=0.5)(X, W, TYPES)  # pymcdm trả về Q
    np.testing.assert_allclose(-mcdm.score_vikor(X, W, TYPES), ref, rtol=1e-9)


def test_edas_matches_pymcdm():
    np.testing.assert_allclose(mcdm.score_edas(X, W, TYPES), M.EDAS()(X, W, TYPES), rtol=1e-9)


def test_moora_matches_pymcdm():
    np.testing.assert_allclose(mcdm.score_moora(X, W, TYPES), M.MOORA()(X, W, TYPES), rtol=1e-9)


def test_promethee_usual_matches_pymcdm():
    ref = M.PROMETHEE_II("usual")(X, W, TYPES)
    np.testing.assert_allclose(mcdm.score_promethee2(X, W, TYPES, pref="usual"), ref, rtol=1e-9)


def test_saw_matches_pymcdm():
    ref = M.WSM(N.minmax_normalization)(X, W, TYPES)
    np.testing.assert_allclose(mcdm.score_saw(X, W, TYPES), ref, rtol=1e-9)


def test_weights_match_pymcdm():
    np.testing.assert_allclose(mcdm.entropy_weights(X), PW.entropy_weights(X), rtol=1e-9)
    ones = np.ones(5)  # pymcdm CRITIC không xét loại tiêu chí -> so với toàn benefit
    np.testing.assert_allclose(mcdm.critic_weights(X, ones), PW.critic_weights(X), rtol=1e-9)


def test_ahp_consistent_matrix():
    w_true = np.array([0.5, 0.3, 0.2])
    A = w_true[:, None] / w_true[None, :]
    w, cr = mcdm.ahp_weights(A)
    np.testing.assert_allclose(w, w_true, rtol=1e-9)
    assert abs(cr) < 1e-9


def test_ahp_inconsistent_flagged():
    A = mcdm.ahp_from_upper({("a", "b"): 9, ("b", "c"): 9, ("a", "c"): 1 / 9}, ["a", "b", "c"])
    _, cr = mcdm.ahp_weights(A)
    assert cr > 0.1


def test_dominant_alternative_ranks_first_everywhere():
    X2 = X.copy()
    X2[0] = np.where(TYPES > 0, X.max(0) + 1, X.min(0) - 0.5)  # trội Pareto
    ranks = mcdm.rank_all(X2, W, TYPES, [f"a{i}" for i in range(8)])
    assert (ranks.loc["a0"] == 1).all()


def test_smaa_acceptability_rows_sum_to_one():
    A, first, _ = mcdm.smaa(X, TYPES, [f"a{i}" for i in range(8)], n_iter=500)
    np.testing.assert_allclose(A.sum(1), 1.0)
    assert abs(first.sum() - 1) < 1e-9
