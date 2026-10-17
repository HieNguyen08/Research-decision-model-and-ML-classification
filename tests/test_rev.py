"""Kiểm thử adapter Rev (notebook 07): toán pointer head / nhiệt độ, bố cục prompt, DRY_RUN – chạy trên CPU, không cần trọng số."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from drm import jev, rev as R, rev_finetune as RF
from drm.text import TextTask


def char_encode(s):
    """Tokenizer giả: mỗi ký tự là một token -> vị trí token kiểm tra được bằng mắt."""
    return [ord(c) for c in s]


# --------------------------------------------------------------------------- toán


def test_pointer_logits_match_rev_formula():
    r = np.random.default_rng(0)
    H, D, K = 12, 8, 3
    h_last, h_opts = r.normal(size=H), r.normal(size=(K, H))
    Wq, Wk = r.normal(size=(D, H)), r.normal(size=(D, H))
    z = R.pointer_logits(h_last, h_opts, Wq, Wk)
    q = Wq @ h_last
    ref = np.array([(Wk @ h_opts[j]) @ q / math.sqrt(D) for j in range(K)])
    np.testing.assert_allclose(z, ref)
    p = R.softmax(z, 1.7)
    assert p.shape == (K,) and abs(p.sum() - 1) < 1e-12
    np.testing.assert_allclose(p, np.exp(ref / 1.7) / np.exp(ref / 1.7).sum())


def test_temper_equals_dividing_logits():
    z = np.array([2.0, -1.0, 0.5])
    for T in (0.5, 1.0, 2.3):
        np.testing.assert_allclose(R.temper(R.softmax(z), T), R.softmax(z, T), atol=1e-12)
    p = np.array([0.9, 0.2, 0.5])
    two = np.array([R.temper([1 - x, x], 1.6)[1] for x in p])
    np.testing.assert_allclose(R.temper_binary(p, 1.6), two, atol=1e-12)
    np.testing.assert_allclose(R.temper_binary(p, 1.0), p, atol=1e-12)


def test_fit_temperature_recovers_known_temperature():
    r = np.random.default_rng(1)
    n, K, T_true = 3000, 3, 2.0
    z = r.normal(0, 3, (n, K))                         # logit "quá tự tin"
    y = np.array([r.choice(K, p=R.softmax(row, T_true)) for row in z])   # nhãn sinh từ phân phối với T = 2
    T, best = R.fit_temperature(list(R.softmax(z)), y)
    assert abs(T - T_true) < 0.25
    assert best <= R.nll(list(R.softmax(z)), y)
    assert R.multiclass_ece(list(R.softmax(z, T)), y) < R.multiclass_ece(list(R.softmax(z)), y)


def test_platt_improves_ece_on_overconfident_scores():
    r = np.random.default_rng(2)
    y = (r.random(600) < 0.3).astype(int)
    p = np.clip(0.5 + 0.45 * (2 * y - 1) * r.random(600) + r.normal(0, 0.25, 600), 0.01, 0.99)
    q = R.platt(p[:50], y[:50], p[50:])
    assert jev.expected_calibration_error(y[50:], q) < jev.expected_calibration_error(y[50:], p[50:])


def test_risk_coverage_and_coverage_at_risk():
    conf = np.array([0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    ok = np.array([1, 1, 1, 0, 1, 1, 0, 0], bool)
    c = R.risk_coverage(conf, ok)
    assert list(c.coverage.round(3)) == [round(k / 8, 3) for k in range(1, 9)]
    assert c.risk.iloc[2] == 0 and abs(c.risk.iloc[3] - 0.25) < 1e-12
    assert R.coverage_at_risk(c, 0.0) == 3 / 8
    assert R.coverage_at_risk(c, 0.2) == 6 / 8          # 1 lỗi / 6 ca = 0.167
    auto = np.array([1, 1, 0, 1, 1, 1, 1, 1], bool)     # ca thứ 3 model chọn "human" -> không tính vào vùng tự quyết
    c2 = R.risk_coverage(conf, ok, auto)
    assert len(c2) == 7 and R.coverage_at_risk(c2, 0.0) == 2 / 8


def test_symmetric_kl():
    lp = np.log(np.array([0.7, 0.2, 0.1]))
    assert R.symmetric_kl(lp, lp) == 0
    lq = np.log(np.array([0.1, 0.2, 0.7]))
    assert R.symmetric_kl(lp, lq) > 0 and abs(R.symmetric_kl(lp, lq) - R.symmetric_kl(lq, lp)) < 1e-12


# --------------------------------------------------------------------------- câu hỏi & prompt


def test_option_conversion_roundtrip():
    q = jev.MODERATION_QUESTIONS
    assert list(R.to_rev_options(q["toxic"])) == ["yes", "no"]
    assert R.to_rev_options(jev.noul("x?", true="T", false="F")) == {"yes": "T", "no": "F"}
    assert list(R.to_rev_options(q["action"])) == list(q["action"]["criteria"])
    assert list(R.to_rev_options(q["severity"])) == ["0", "1", "2", "3"]
    a = R.to_jev_answer(q["toxic"], {"yes": 0.8, "no": 0.2})
    assert jev.read_noul({"answers": {"t": a}}, "t") == 0.8
    s = R.to_jev_answer(q["severity"], {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4})
    assert abs(s["score"] - 2.0) < 1e-12 and s["legend"]["3"] == q["severity"]["criteria"][3]
    c = R.to_jev_answer(q["action"], {"reply_normally": 0.1, "reply_with_deescalation": 0.2,
                                      "escalate_to_supervisor": 0.6, "block_and_report": 0.1})
    assert c["choice"] == "escalate_to_supervisor" and abs(c["confidence"] - 0.6) < 1e-12


def test_encode_prompt_layout_and_positions():
    opts = {"yes": "bad", "no": "fine"}
    ids, pos, keys = R.encode_prompt(char_encode, {"m": "hi"}, "Is it bad?", opts, state_format="raw")
    text = "".join(map(chr, ids))
    assert text == 'State:\n{"m":"hi"}\nQuestion: Is it bad?\nOptions:\nyes: bad\nno: fine\nDecision:'
    assert keys == ["yes", "no"]
    assert [text[p] for p in pos] == ["\n", "\n"]                 # token cuối của mỗi dòng lựa chọn
    assert text[: pos[0] + 1].endswith("yes: bad\n") and text[: pos[1] + 1].endswith("no: fine\n")
    raw, _, _ = R.encode_prompt(char_encode, "plain text", "Q?", opts, state_format="raw")
    js, _, _ = R.encode_prompt(char_encode, "plain text", "Q?", opts, state_format="json")
    assert "".join(map(chr, raw)).startswith("State:\nplain text\n")
    assert "".join(map(chr, js)).startswith('State:\n"plain text"\n')


def test_encode_prompt_permutation_keeps_label_and_options():
    opts = {k: f"option {k}" for k in "abcde"}
    seen = set()
    for seed in range(20):
        ids, pos, keys = R.encode_prompt(char_encode, "s", "q", opts, seed=seed)
        assert sorted(keys) == list("abcde")
        text = "".join(map(chr, ids))
        for k, p in zip(keys, pos):
            assert text[: p + 1].endswith(f"{k}: option {k}\n")
        seen.add(tuple(keys))
        row = {"state": "s", "instructions": "q", "criteria": opts, "expected": "c"}
        e = RF.encode_row(char_encode, row, 10_000, seed=seed)
        assert e["keys"][e["label"]] == "c"
    assert len(seen) > 5
    assert RF.align_to(["b", "a", "c"], ["a", "b", "c"]) == [1, 0, 2]


def test_encode_row_truncates_long_messages():
    row = RF.binary_rows(["word " * 2000], [1], "Is it toxic?", RF.BINARY_OPTIONS["toxic"])[0]
    e = RF.encode_row(char_encode, row, 400)
    assert len(e["ids"]) <= 400 and e["keys"][e["label"]] == "yes"
    assert "".join(map(chr, e["ids"])).endswith("Decision:")


# --------------------------------------------------------------------------- DRY_RUN adapter


def toy_task(n=160, seed=0):
    rng = np.random.default_rng(seed)
    bad = ["you are an idiot", "win a free prize call now", "i hate them all"]
    good = ["thanks for the help", "my order is late", "where is my package"]
    y = (np.arange(n) % 4 == 0).astype(int)
    X = pd.Series([f"{rng.choice(bad if t else good)} {rng.choice(good)}" for t in y])
    return TextTask("jigsaw_toxic", X[: n // 2].reset_index(drop=True), y[: n // 2],
                    X[n // 2:].reset_index(drop=True), y[n // 2:], "toxic", "Is this message toxic?", "test")


def test_dry_run_adapter_answers_in_jev_format():
    ad = R.RevAdapter.dry_run(batch_size=4)
    states = [{"customer_message": "you idiot"}, {"customer_message": "thanks"}, {"customer_message": "free prize"}]
    out = ad.ask_many(states, jev.MODERATION_QUESTIONS, progress=False)
    assert len(out) == 3
    for r in out:
        assert set(r["answers"]) == set(jev.MODERATION_QUESTIONS)
        assert 0 <= jev.read_noul(r, "toxic") <= 1
        assert abs(sum(jev.read_choice_probs(r, "action").values()) - 1) < 1e-9
        assert 0 <= jev.read_score(r, "severity") <= 3
    # tất định, và batch không đổi kết quả
    again = R.RevAdapter.dry_run(batch_size=1).ask_many(states, jev.MODERATION_QUESTIONS, progress=False)
    for a, b in zip(out, again):
        assert a["answers"]["toxic"]["noul"] == pytest.approx(b["answers"]["toxic"]["noul"])
    rep = ad.cost_report()
    assert rep["calls"] == 3 and rep["input_tokens"] > 0 and rep["latency_p50_ms"] >= 0


def test_dry_run_adapter_temperature_changes_confidence_not_ranking():
    states = [{"customer_message": f"message {i}"} for i in range(30)]
    q = {"toxic": jev.noul("Is it toxic?")}
    p1 = np.array([jev.read_noul(r, "toxic") for r in R.RevAdapter.dry_run().ask_many(states, q, progress=False)])
    p3 = np.array([jev.read_noul(r, "toxic") for r in R.RevAdapter.dry_run(temperature=3.0).ask_many(states, q, progress=False)])
    assert (np.argsort(p1) == np.argsort(p3)).all()
    assert np.abs(p3 - 0.5).mean() < np.abs(p1 - 0.5).mean()


def test_dry_run_adapter_plugs_into_evaluate_moderation():
    task = toy_task()
    df, scores = jev.evaluate_moderation(R.RevAdapter.dry_run(), task, n_test=40, wording=("generic", "task"))
    assert set(df.wording) == {"generic", "task"}
    assert df.pr_auc.between(0, 1).all() and df.ece.between(0, 1).all()
    assert scores["generic|p"].shape == (40,) and scores["severity"].shape == (40,)


def test_bitext_routes_and_splits():
    assert RF.bitext_route("track_order") == "order_status"
    assert RF.bitext_route("check_refund_policy") == "policy_info"
    assert RF.bitext_route("cancel_order") == "human"
    intents = ["track_order", "cancel_order", "check_refund_policy", "complaint"] * 50
    df = pd.DataFrame({"instruction": [f"msg {i} {t}" for i, t in enumerate(intents)], "intent": intents})
    tr, va, te = RF.bitext_splits(df)
    assert len(tr) + len(va) + len(te) == len(df)
    assert not ({r["id"] for r in tr} & {r["id"] for r in te}) and not ({r["id"] for r in va} & {r["id"] for r in te})
    assert {r["expected"] for r in te} == {"order_status", "human", "policy_info"}
    assert list(te[0]["criteria"]) == list(RF.ROUTE_OPTIONS)


def test_rev_module_never_references_jev_answers_file():
    import inspect
    for mod in (R, RF):
        src = inspect.getsource(mod)
        assert "typesafe.ai" not in src and "JevClient" not in src
        assert "jev_official_answers" not in src.replace("``results/jev_official_answers.jsonl``", "")


# --------------------------------------------------------------------------- torch (bỏ qua nếu không có torch)


def test_lora_wrapper_merge_equivalence_and_tiny_training():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from drm import finetune as FT

    tok, model = FT.tiny_random_encoder(decoder=True)
    backbone = model.model
    backbone.requires_grad_(False)
    rows = [(tok.encode("State:\nhello\nQuestion: q\nOptions:\nyes: a\n", add_special_tokens=False), [3, 6])]
    torch.manual_seed(0)
    adapters = {}
    for path, m in backbone.named_modules():
        if isinstance(m, torch.nn.Linear) and path.split(".")[-1] in R.LORA_TARGETS:
            adapters[path] = {"a": torch.randn(R.LORA_RANK, m.in_features) * 0.1,
                              "b": torch.randn(m.out_features, R.LORA_RANK) * 0.1}
    hq, hk = R.make_heads(backbone.config.hidden_size, "cpu", head_dim=8)
    import copy
    merged = copy.deepcopy(backbone)
    R.merge_lora(merged, adapters)
    R.attach_lora(backbone, adapters, trainable=False)
    with torch.no_grad():
        z1 = R.head_logits(R.forward_rows(merged, rows, 0, "cpu"), rows, hq, hk)[0]
        z2 = R.head_logits(R.forward_rows(backbone, rows, 0, "cpu"), rows, hq, hk)[0]
    torch.testing.assert_close(z1, z2, atol=1e-4, rtol=1e-4)

    # huấn luyện tí hon theo công thức Rev: chạy hết, ra xác suất hợp lệ, T trong lưới
    tok, model = FT.tiny_random_encoder(decoder=True)
    bb = model.model
    bb.requires_grad_(False)
    task = toy_task(120)
    cfg = RF.RevFTConfig(batch_size=8, micro_batch=8, eval_every=4, epochs=1, max_tokens=128, grad_checkpointing=False,
                         adapter_lr=3e-3, head_lr=3e-3, eval_batch=16)
    res = RF.finetune_task(task, tok, bb, cfg, extra_tests={"obf": list(task.X_test)}, log=lambda *_: None)
    assert res.p_test.shape == (len(task.y_test),) and np.isfinite(res.p_test).all()
    assert res.p_val.shape == res.y_val.shape and 0.4 <= res.temperature <= 3.0
    assert res.p_extra["obf"].shape == res.p_test.shape
    assert res.trainable_params > 0
    assert RF.detach_lora(bb) > 0
    assert not any(hasattr(m, "a") and hasattr(m, "base") for m in bb.modules())


def test_state_text_matches_serve_local_json_encoding():
    # serve_local.py dùng json.dumps(state, separators=(',', ':')) với ensure_ascii mặc định
    s = {"customer_message": "Where is my order? 😡 café"}
    assert R.state_text(s, "json") == json.dumps(s, separators=(",", ":"))
    assert "\\ud83d" in R.state_text(s, "json")
    assert R.state_text("raw text 😡", "raw") == "raw text 😡"
