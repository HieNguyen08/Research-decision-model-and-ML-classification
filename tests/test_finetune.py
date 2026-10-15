"""Chạy thử pipeline fine-tune (encoder + LoRA decoder) và ANN CPU bằng mô hình tí hon, dữ liệu tổng hợp."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from drm import finetune as FT, nn_text as N  # noqa: E402
from drm.text import TextTask  # noqa: E402


def toy_task(n=240, seed=0):
    rng = np.random.default_rng(seed)
    bad = ["you are an idiot", "win a free prize call now", "i hate them all", "click the link spam"]
    good = ["thanks for the help", "my order is late", "please send the refund", "where is my package"]
    y = (np.arange(n) % 4 == 0).astype(int)
    X = pd.Series([f"{rng.choice(bad if t else good)} {rng.choice(good)}" for t in y])
    return TextTask("toy", X[: n * 3 // 4].reset_index(drop=True), y[: n * 3 // 4],
                    X[n * 3 // 4:].reset_index(drop=True), y[n * 3 // 4:], "toxic", "Is it bad?", "test")


def test_finetune_encoder_learns_toy_task():
    task = toy_task()
    tok, model = FT.tiny_random_encoder()
    res = FT.finetune(task, cfg=FT.FTConfig(lr=3e-3, epochs=6, batch_size=16, bf16=False, patience=20),
                      model=model, tokenizer=tok, device="cpu", log=lambda *_: None)
    assert res.p_test.shape == (len(task.y_test),) and np.isfinite(res.p_test).all()
    assert res.best_val_pr_auc > 0.9
    from sklearn.metrics import average_precision_score
    assert average_precision_score(task.y_test, res.p_test) > 0.9


def test_finetune_decoder_with_lora():
    pytest.importorskip("peft")
    task = toy_task()
    tok, model = FT.tiny_random_encoder(decoder=True)
    cfg = FT.FTConfig(lr=5e-3, epochs=2, batch_size=16, bf16=False, lora={"r": 4, "alpha": 8, "dropout": 0.0})
    res = FT.finetune(task, cfg=cfg, model=model, tokenizer=tok, device="cpu", log=lambda *_: None)
    total = sum(p.numel() for p in model.parameters())
    assert 0 < res.trainable_params < total  # chỉ LoRA + đầu phân loại được huấn luyện
    assert np.isfinite(res.p_test).all()


def test_greedy_ensemble_prefers_good_models():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.2).astype(int)
    good = y + rng.normal(0, 0.3, 400)
    noise = rng.normal(0, 1, 400)
    w, s, ap = FT.greedy_ensemble({"good": good, "noise": noise}, y, {"good": good, "noise": noise})
    assert w.get("good", 0) > w.get("noise", 0) and ap > 0.8 and len(s) == 400


@pytest.mark.parametrize("name", N.NN_MODELS)
def test_cpu_ann_models_run(name):
    task = toy_task(160)
    r = N.fit_nn(task, name)
    assert r.p_test.shape == (len(task.y_test),) and np.isfinite(r.p_test).all() and r.n_params > 0


def test_embedding_classifier_runs():
    task = toy_task()
    tok, model = FT.tiny_random_encoder(decoder=True)
    res = FT.embedding_classifier(task, model=model.model, tokenizer=tok, device="cpu",
                                  extra_tests={"same": list(task.X_test)})
    assert res.p_test.shape == (len(task.y_test),) and np.isfinite(res.p_test).all()
    np.testing.assert_allclose(res.p_extra["same"], res.p_test, rtol=1e-5)


def test_full_finetune_loads_fp32_weights_and_lora_bf16(monkeypatch):
    """transformers >= 5 nạp mặc định theo kiểu đã lưu (DeBERTa-v3: fp16) -> phải ép fp32 khi fine-tune toàn phần."""
    import transformers
    seen = []
    tok, tiny = FT.tiny_random_encoder()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tok)
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification, "from_pretrained",
                        lambda *a, **k: seen.append(k) or tiny)
    FT.load_model("any/encoder", FT.FTConfig(), "cpu")
    dt = seen[-1].get("dtype", seen[-1].get("torch_dtype"))
    assert dt == torch.float32
    monkeypatch.setattr(FT, "add_lora", lambda m, lora: m)
    FT.load_model("any/llm", FT.FTConfig(lora={"r": 4}), "cpu")
    assert seen[-1].get("dtype", seen[-1].get("torch_dtype")) == torch.bfloat16


def _fake_result(best, y):
    return FT.FTResult("", np.zeros(len(y)), y, np.zeros(3), best_val_pr_auc=best)


@pytest.mark.parametrize("failure", ["nan", "collapse"])
def test_finetune_retries_once_with_lower_lr(monkeypatch, failure):
    y = np.array([0, 0, 0, 1] * 25)
    tok, tiny = FT.tiny_random_encoder()
    monkeypatch.setattr(FT, "load_model", lambda mid, cfg, dev: (tok, tiny))
    calls = []

    def fake_train(task, model, tok_, cfg, device, log, extra, attempt=0):
        calls.append((cfg.lr, cfg.bf16, attempt))
        if attempt == 0:
            if failure == "nan":
                raise FT.TrainingDiverged("loss = nan")
            return _fake_result(0.26, y)  # ≈ tỉ lệ dương 0.25 -> sập
        return _fake_result(0.9, y)

    monkeypatch.setattr(FT, "_train", fake_train)
    res = FT.finetune(toy_task(), "any/model", FT.FTConfig(lr=2e-5, bf16=True), name="m", log=lambda *_: None)
    assert [c[2] for c in calls] == [0, 1] and calls[1][0] == 1e-5
    assert calls[1][1] is (failure != "nan")  # ra NaN -> tắt bf16 cho encoder
    assert res.config["attempts"] == 2 and res.best_val_pr_auc == 0.9 and res.name == "m"
    assert FT.collapsed(0.26, y) and not FT.collapsed(0.9, y)
