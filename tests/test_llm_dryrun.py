"""Dry-run pipeline LLM bằng một mô hình Qwen2 tí hon khởi tạo ngẫu nhiên (không cần tải trọng số).

Mục đích: bắt lỗi code (padding, chỉ số token, parse) trước khi chạy thật trên Colab GPU.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from drm import data, llm  # noqa: E402

@pytest.fixture(scope="module")
def scorer():
    return llm.tiny_random_scorer()


def _manual_logprob(scorer, prompt, label):
    p, l = scorer._ids(prompt), scorer._ids(label)
    ids = torch.tensor([p + l])
    with torch.no_grad():
        lp = torch.log_softmax(scorer.model(ids).logits.float(), -1)[0]
    return sum(lp[len(p) - 1 + t, l[t]].item() for t in range(len(l)))


def test_batched_scores_equal_manual(scorer):
    ds = data.load("penguins")
    prompts = [scorer.chat(llm.build_prompt(ds, ds.X.iloc[i])) for i in range(4)]
    labels = [llm.verbalizer(c) for c in ds.classes]
    got = scorer.label_logprobs(prompts, labels)
    ref = np.array([[_manual_logprob(scorer, p, l) for l in labels] for p in prompts])
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_classification_pipeline(scorer):
    ds = data.load("diabetes")
    res, extra = llm.evaluate_llm_classification(scorer, ds, shots=(0, 2), n_test=12)
    assert set(res["variant"]) == {"raw", "calibrated"}
    assert len(res) == 4 and res["roc_auc"].between(0, 1).all()
    P = extra["probs"][(2, "raw")]
    np.testing.assert_allclose(P.sum(1), 1, rtol=1e-6)


def test_baselines_and_stacking():
    ds = data.load("diabetes")
    tr, te = llm.split_for_llm(ds, n_test=50)
    base = llm.few_shot_ml_baselines(ds, tr, te, shots=(4, 8), n_rep=2)
    assert len(base) == 4
    rng = np.random.default_rng(0)
    st = llm.llm_prior_stacking(ds, rng.normal(size=len(tr)), rng.normal(size=len(te)), tr, te,
                                shots=(8, 16), n_rep=2)
    assert {"LLM zero-shot", "LogReg (X)", "LogReg (X + LLM prior)"} == set(st["method"])


def test_decision_maker_functions_do_not_crash(scorer):
    import pandas as pd
    M = pd.DataFrame({"auc": [0.9, 0.8], "time": [1.0, 0.1]}, index=["A", "B"])
    ranking, raw = scorer and llm.llm_rank_alternatives(scorer, M, {"auc": 1, "time": -1}, "test")
    assert ranking is None or sorted(ranking) == ["A", "B"]
    w, cr, raw = llm.llm_ahp_weights(scorer, {"auc": "accuracy", "time": "speed"}, "test")
    assert w is None or abs(w.sum() - 1) < 1e-9


def test_json_parsing():
    assert llm._extract_json('blah ["A", "B"] end') == ["A", "B"]
    assert llm._extract_json("{'a vs b': 3}") == {"a vs b": 3}


def test_left_padding_with_very_different_lengths(scorer):
    prompts = ["Yes", "The age is 39. " * 40, scorer.chat("Record: The age is 39.")]
    labels = ["Yes", "No", "Chinstrap"]
    got = scorer.label_logprobs(prompts, labels)
    ref = np.array([[_manual_logprob(scorer, p, l) for l in labels] for p in prompts])
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_left_padding_with_very_different_lengths(scorer):
    prompts = ["Yes", "The age is 39. " * 40, scorer.chat("Record: The age is 39.")]
    labels = ["Yes", "No", "Chinstrap"]
    got = scorer.label_logprobs(prompts, labels)
    ref = np.array([[_manual_logprob(scorer, p, l) for l in labels] for p in prompts])
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def _toy_text_task():
    import pandas as pd
    from drm.text import TextTask
    rng = np.random.default_rng(0)
    words = ["refund", "late", "idiot", "great", "scam", "thanks", "order", "awful"]
    X = pd.Series([" ".join(rng.choice(words, 6)) for _ in range(60)])
    y = np.array([int(("idiot" in t) or ("scam" in t)) for t in X])
    return TextTask("toy", X[:40].reset_index(drop=True), y[:40], X[40:].reset_index(drop=True), y[40:],
                    "toxic", "Is this message toxic?", "toy")


def test_llm_moderation_pipeline(scorer):
    task = _toy_text_task()
    res, sc = llm.evaluate_llm_moderation(scorer, task, shots=(0, 4), n_test=12)
    assert len(res) == 4 and res["pr_auc"].between(0, 1).all()
    assert sc["4|calibrated"].shape == (12,)


def test_guard_scoring_runs(scorer):
    task = _toy_text_task()
    res, sc = llm.evaluate_guard(scorer, "qwen3guard-0.6b", task, n_test=6)
    s = sc["0|raw"]
    assert s.shape == (6,) and np.all((s >= 0) & (s <= 1))
