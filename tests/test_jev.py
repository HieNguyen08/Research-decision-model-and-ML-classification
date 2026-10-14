"""Kiểm thử module Jev bằng transport giả lập (không gọi mạng, không cần API key)."""
import json

import numpy as np
import pandas as pd
import pytest

from drm import jev


class FakeTransport:
    """Trả lời như /v1/systemone: noul = 0.9 nếu state chứa 'idiot'/'free', ngược lại 0.1."""

    def __init__(self, fail_first=0, status=429):
        self.calls = []
        self.fail_first = fail_first
        self.status = status

    def __call__(self, url, headers, body, timeout):
        self.calls.append((url, headers, body))
        if self.fail_first > 0:
            self.fail_first -= 1
            return self.status, {"detail": "slow down"}
        text = json.dumps(body["state"]).lower()
        bad = any(w in text for w in ("idiot", "free"))
        answers = {}
        for name, q in body["questions"].items():
            if q["type"] == "noul":
                answers[name] = {"type": "noul", "noul": 0.9 if bad else 0.1}
            elif q["type"] == "choice":
                labels = list(q["criteria"])
                p = np.full(len(labels), 0.5 / max(len(labels) - 1, 1))
                p[-1 if bad else 0] = 0.5
                answers[name] = {"type": "choice", "choice": labels[int(p.argmax())], "confidence": float(p.max()),
                                 "probabilities": dict(zip(labels, p.tolist()))}
            else:
                k = len(q["criteria"])
                p = np.zeros(k)
                p[k // 2] = 1.0
                answers[name] = {"type": "score", "score": float(k // 2), "confidence": 1.0,
                                 "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                                 "probabilities": {str(i): float(v) for i, v in enumerate(p)}}
        return 200, {"model": "jev-test", "answers": answers, "usage": {"input_tokens": 50, "output_tokens": 0}}


def test_request_format_and_cache(tmp_path):
    t = FakeTransport()
    c = jev.JevClient(api_key="k", base_url="https://example.test/", cache_path=tmp_path / "c.jsonl", transport=t)
    q = {"toxic": jev.noul("Is it toxic?")}
    r1 = c.ask({"customer_message": "you idiot"}, q)
    url, headers, body = t.calls[0]
    assert url == "https://example.test/v1/systemone"
    assert headers["Authorization"] == "Bearer k"
    assert set(body) == {"state", "model", "questions"} and body["model"] == "jev-latest"
    assert r1["answers"]["toxic"]["noul"] == 0.9
    c.ask({"customer_message": "you idiot"}, q)  # cache hit trong bộ nhớ
    assert len(t.calls) == 1
    c2 = jev.JevClient(api_key="k", base_url="https://example.test", cache_path=tmp_path / "c.jsonl", transport=t)
    c2.ask({"customer_message": "you idiot"}, q)  # cache hit từ file
    assert len(t.calls) == 1 and c2.usage["cache_hits"] == 1


def test_retry_then_success_and_fatal_error(monkeypatch):
    monkeypatch.setattr(jev.time, "sleep", lambda s: None)
    t = FakeTransport(fail_first=2, status=429)
    c = jev.JevClient(api_key="k", transport=t)
    assert c.ask("hi", {"x": jev.noul("?")})["answers"]["x"]["noul"] == 0.1
    assert len(t.calls) == 3
    c_bad = jev.JevClient(api_key="k", transport=FakeTransport(fail_first=5, status=401))
    with pytest.raises(jev.JevError):
        c_bad.ask("hi", {"x": jev.noul("?")})


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(jev.JevError):
        jev.JevClient(transport=FakeTransport()).ask("hi", {"x": jev.noul("?")})


def test_evaluate_moderation_with_fake_client():
    from drm.text import TextTask
    X = pd.Series(["you idiot", "thanks a lot", "free prize now", "where is my order"] * 10)
    y = np.array([1, 0, 1, 0] * 10)
    task = TextTask("jigsaw_toxic", X, y, X, y, "toxic", "Is this toxic?", "toy")
    c = jev.JevClient(api_key="k", transport=FakeTransport())
    res, sc = jev.evaluate_moderation(c, task, n_test=20)
    assert set(res.wording) == {"generic", "task"}
    assert (res.roc_auc == 1.0).all()
    assert len(sc["action"]) == 20 and sc["severity"].shape == (20,)


def test_ahp_from_jev_equal_importance_is_consistent():
    crit = {"Quality": "", "Latency": "", "Cost": ""}
    qs, pairs = jev.ahp_questions(crit)
    assert len(qs) == 3
    resp = FakeTransport()("u", {}, {"state": "s", "questions": qs}, 1)[1]  # dồn xác suất vào "bằng nhau"
    w, cr, A = jev.ahp_from_jev(resp, crit)
    np.testing.assert_allclose(w.values, 1 / 3, atol=1e-9)
    assert abs(cr) < 1e-9


def test_order_state_is_json_serialisable():
    row = pd.Series({"account_age_days": 3.5, "phone_verified": np.int64(0), "prev_refusal_rate": np.nan})
    s = jev.order_state(row)
    json.dumps(s)
    assert s["order"]["prev_refusal_rate"]["value"] == "unknown"


def test_ece_perfectly_calibrated_is_small():
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)
    assert jev.expected_calibration_error(y, p) < 0.02
    assert jev.expected_calibration_error(y, np.clip(p * 0.3, 0, 1)) > 0.2


def test_local_adapter_same_format():
    pytest.importorskip("torch")
    from drm import llm
    a = jev.LocalJevAdapter(llm.tiny_random_scorer())
    r = a.ask({"customer_message": "hello"}, jev.MODERATION_QUESTIONS)
    assert 0 <= r["answers"]["toxic"]["noul"] <= 1
    assert abs(sum(r["answers"]["action"]["probabilities"].values()) - 1) < 1e-6
    assert 0 <= r["answers"]["severity"]["score"] <= 3


class FakeWaterSheep:
    """Giả lập watersheep.WaterSheep.ask (cùng định dạng {model, answers, usage})."""

    def ask(self, request):
        answers = FakeTransport()("u", {}, {"state": request["state"], "questions": request["questions"]}, 1)[1]["answers"]
        return {"model": "WaterSheep", "answers": answers, "usage": {"input_tokens": 12, "output_tokens": 0}}


def test_watersheep_adapter_same_interface():
    from drm.text import TextTask
    X = pd.Series(["you idiot", "thanks a lot", "free prize now", "where is my order"] * 10)
    y = np.array([1, 0, 1, 0] * 10)
    task = TextTask("sms_spam", X, y, X, y, "spam", "Is this spam?", "toy")
    ws = jev.WaterSheepAdapter(ws=FakeWaterSheep())
    res, sc = jev.evaluate_moderation(ws, task, n_test=20)
    assert (res.roc_auc == 1.0).all() and ws.usage["input_tokens"] == 12 * 40
    assert ws.cost_report()["est_cost_usd"] == 0.0


def test_watersheep_adapter_surfaces_question_errors():
    class Bad:
        def ask(self, request):
            return {"model": "WaterSheep", "answers": {"x": {"type": "score", "error": "score needs 2-10 levels"}},
                    "usage": {}}
    with pytest.raises(jev.JevError):
        jev.WaterSheepAdapter(ws=Bad()).ask("hi", {"x": jev.score("?", list("abcdefghijkl"))})
