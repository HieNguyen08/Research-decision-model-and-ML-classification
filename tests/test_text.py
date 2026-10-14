import numpy as np

from drm import text as T


def test_obfuscate_is_deterministic_and_changes_text():
    s = "please refund my order, this seller is an idiot and a scammer"
    assert T.obfuscate(s, 0.5, seed=1) == T.obfuscate(s, 0.5, seed=1)
    assert T.obfuscate(s, 1.0, seed=1) != s
    assert T.obfuscate(s, 0.0, seed=1) == s


def test_deobfuscate_reverses_common_tricks():
    assert T.deobfuscate("i.d.i.o.t") == "idiot"
    assert T.deobfuscate("h3ll0 sc@mmer") == "hello scammer"
    assert T.deobfuscate("stuuuupid") == "stupid"


def test_recall_at_fpr_and_threshold():
    y = np.array([0] * 90 + [1] * 10)
    s = np.concatenate([np.linspace(0, 0.5, 90), np.linspace(0.6, 1, 10)])  # tách hoàn hảo
    assert T.recall_at_fpr(y, s, 0.01) == 1.0
    thr = T.best_f1_threshold(y, s)
    assert 0.5 < thr <= 0.6
    m = T.metrics(y, s, thr)
    assert m["f1"] == 1.0 and abs(m["pr_auc"] - 1.0) < 1e-9


def test_paired_bootstrap_detects_better_model():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.3).astype(int)
    good = y + rng.normal(0, 0.3, 400)
    bad = y + rng.normal(0, 1.5, 400)
    d, lo, hi = T.paired_bootstrap(y, good, bad, n=200)
    assert d > 0 and lo > 0
