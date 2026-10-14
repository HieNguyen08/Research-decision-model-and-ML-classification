"""Điểm của LLM rất tự tin không được bão hoà thành các giá trị hoà nhau (sai PR-AUC)."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from drm import llm
from drm.text import TextTask


class ConfidentScorer:
    """Giả lập mô hình rất tự tin: log P(Yes) - log P(No) từ 18 đến 40 nat, trả về float32 như HFScorer."""
    name = "confident"

    def __init__(self, margins):
        self.margins = dict(margins)

    def chat(self, text):
        return text

    def label_logprobs(self, prompts, labels):
        m = np.array([self.margins.get(p.rsplit('Message: "', 1)[-1].split('"')[0], 5.0) for p in prompts])
        return np.stack([-m, np.zeros_like(m)], 1).astype(np.float32)


def test_moderation_metrics_rank_by_logodds_not_saturated_probability():
    rng = np.random.default_rng(0)
    n = 60
    y = (np.arange(n) % 3 == 0).astype(int)
    margins = np.where(y == 1, rng.uniform(25, 40, n), rng.uniform(18, 33, n))  # chồng lấn, đều > 17 nat
    texts = pd.Series([f"msg {i}" for i in range(n)])
    task = TextTask("t", texts, y, texts, y, "toxic", "Is it toxic?", "test")
    scorer = ConfidentScorer(zip(texts, margins))  # prompt "N/A" (content-free) -> log-odds 5
    prompt = llm.build_moderation_prompt(task, texts[0])
    assert scorer.label_logprobs([prompt], ["No", "Yes"])[0, 0] == np.float32(-margins[0])

    res, scores = llm.evaluate_llm_moderation(scorer, task, shots=(0,), n_test=n)
    expected = average_precision_score(y, margins)
    got = res.set_index("variant")["pr_auc"]
    np.testing.assert_allclose(got["raw"], expected, rtol=1e-6)
    # contextual calibration là phép tịnh tiến log-odds -> không đổi thứ hạng với bài toán nhị phân
    np.testing.assert_allclose(got["calibrated"], expected, rtol=1e-6)
    assert np.unique(scores["0|raw|logit"]).size == n
    # bản cũ (softmax float32) cho toàn bộ điểm = 1.0 -> PR-AUC chỉ bằng tỉ lệ lớp dương
    old = np.exp(np.float32(0)) / (np.exp(np.float32(0)) + np.exp(-margins.astype(np.float32)))
    assert np.unique(old).size == 1 and average_precision_score(y, old) < expected
