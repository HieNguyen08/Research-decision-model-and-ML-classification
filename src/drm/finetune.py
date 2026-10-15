"""Fine-tune mạng nơ-ron tiền huấn luyện (transformer) cho kiểm duyệt tin nhắn – chạy trên Colab GPU (notebook 06).

* Encoder (DeBERTa-v3, RoBERTa, BERTweet, ModernBERT...): fine-tune toàn bộ, đầu ra 1 logit, loss BCE.
* Decoder LLM (Qwen, Llama...): LoRA + đầu phân loại (``AutoModelForSequenceClassification``), trọng số bf16.

Chia train/validation giống ``text.fit_eval`` (10% stratified, cùng seed) để so sánh công bằng với TF-IDF và ANN CPU.
Dừng sớm theo PR-AUC trên validation; trả về xác suất trên validation (để ghép mô hình) và trên toàn bộ test.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split


@dataclass
class FTConfig:
    lr: float = 2e-5
    epochs: float = 3
    batch_size: int = 32
    max_len: int = 128
    warmup: float = 0.06
    weight_decay: float = 0.01
    eval_per_epoch: int = 2
    patience: int = 3            # số lần đánh giá liên tiếp không cải thiện thì dừng
    grad_accum: int = 1
    bf16: bool = True
    seed: int = 0                # khởi tạo đầu phân loại + thứ tự batch
    split_seed: int = 0          # cách chia train/validation: giữ cố định để ghép nhiều seed/mô hình trên cùng validation
    lora: dict | None = None     # vd. {"r": 16, "alpha": 32, "dropout": 0.05} -> LoRA cho decoder
    grad_checkpointing: bool = False
    pos_weight: float | None = None


@dataclass
class FTResult:
    name: str
    p_val: np.ndarray
    y_val: np.ndarray
    p_test: np.ndarray
    history: list = field(default_factory=list)
    fit_time_s: float = 0.0
    trainable_params: int = 0
    best_val_pr_auc: float = 0.0
    ms_per_msg: float = 0.0       # suy luận trên test (batch), GPU hiện tại
    p_extra: dict = field(default_factory=dict)  # điểm trên các bản test biến đổi (vd. obfuscation)
    config: dict = field(default_factory=dict)   # cấu hình thực tế (sau khi tự huấn luyện lại, nếu có)


class TrainingDiverged(RuntimeError):
    """Loss hoặc điểm dự đoán không hữu hạn (NaN/inf) trong lúc huấn luyện."""


# Mô hình đề xuất (Hugging Face id, cấu hình). Đều công khai, không cần chấp nhận license, trừ khi ghi chú.
ENCODERS = {
    "deberta-v3-base": ("microsoft/deberta-v3-base", FTConfig(lr=3e-5)),
    "deberta-v3-large": ("microsoft/deberta-v3-large", FTConfig(lr=1e-5, batch_size=16)),
    "roberta-large": ("FacebookAI/roberta-large", FTConfig(lr=1e-5, batch_size=16)),
    "bertweet-large": ("vinai/bertweet-large", FTConfig(lr=1e-5, batch_size=16)),
    "twitter-roberta-large-2022": ("cardiffnlp/twitter-roberta-large-2022-154m", FTConfig(lr=1e-5, batch_size=16)),
    "fbert": ("diptanu/fBERT", FTConfig(lr=2e-5)),  # BERT-base huấn luyện tiếp trên 1.4M tweet xúc phạm (SOLID)
    "modernbert-base": ("answerdotai/ModernBERT-base", FTConfig(lr=5e-5, max_len=256)),
    "modernbert-large": ("answerdotai/ModernBERT-large", FTConfig(lr=2e-5, batch_size=16, max_len=256)),
    # Khởi tạo từ mô hình zero-shot NLI. Dùng bản "-c" (chỉ dữ liệu tổng hợp/được phép thương mại):
    # bản thường v2.0 đã học trên Davidson, HateXplain, Jigsaw, spam -> rò rỉ test.
    "deberta-v3-large-zeroshot-c": ("MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c", FTConfig(lr=1e-5, batch_size=16)),
    # Decision model mở họ Jev (ModernBERT-base + đầu quyết định RLCD): fine-tune có giám sát, so với modernbert-base.
    # Nếu repo không nạp được như encoder chuẩn, vòng lặp sẽ báo lỗi và bỏ qua.
    "watersheep-ft": ("samratduttaofficial/WaterSheep", FTConfig(lr=5e-5, max_len=256)),
}
LORA_LLMS = {
    "qwen2.5-1.5b-lora": ("Qwen/Qwen2.5-1.5B-Instruct", FTConfig(lr=2e-4, epochs=2, batch_size=16,
                                                                 lora={"r": 16, "alpha": 32, "dropout": 0.05})),
    "llama-3.2-3b-lora": ("meta-llama/Llama-3.2-3B-Instruct", FTConfig(lr=1e-4, epochs=2, batch_size=16,
                                                                       lora={"r": 16, "alpha": 32, "dropout": 0.05})),
    "qwen3-8b-lora": ("Qwen/Qwen3-8B", FTConfig(lr=1e-4, epochs=1, batch_size=8, grad_accum=2, grad_checkpointing=True,
                                                lora={"r": 16, "alpha": 32, "dropout": 0.05})),
}
# Mô hình embedding (đóng băng) + LogReg: rẻ, mạnh khi ít nhãn (MTEB ToxicConversations)
EMBEDDERS = {
    "qwen3-emb-0.6b-lr": "Qwen/Qwen3-Embedding-0.6B",
    "qwen3-emb-4b-lr": "Qwen/Qwen3-Embedding-4B",
    "qwen3-emb-8b-lr": "Qwen/Qwen3-Embedding-8B",
}


def split_train_val(task, seed=0, val_size=0.1):
    Xtr, Xva, ytr, yva = train_test_split(task.X_train, task.y_train, test_size=val_size,
                                          stratify=task.y_train, random_state=seed)
    return list(Xtr), list(Xva), np.asarray(ytr), np.asarray(yva)


def _dtype_kw(dtype) -> dict:
    """``dtype=`` (transformers >= 4.56) hoặc ``torch_dtype=`` (bản cũ)."""
    import transformers
    from packaging.version import Version

    return {"dtype": dtype} if Version(transformers.__version__) >= Version("4.56") else {"torch_dtype": dtype}


def load_model(model_id: str, cfg: FTConfig, device):
    """Tải tokenizer + mô hình phân loại 1 logit; gắn LoRA nếu ``cfg.lora``.

    Fine-tune toàn phần luôn giữ trọng số ở **fp32** (tính toán vẫn bf16 qua autocast). Lý do: transformers >= 5
    mặc định ``dtype="auto"``, tức nạp đúng kiểu đã lưu; DeBERTa-v3 lưu ở fp16, nên AdamW chạy trên trọng số fp16
    và ra NaN ngay. LoRA: mô hình gốc bf16 (đóng băng), adapter + đầu phân loại fp32."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"num_labels": 1, "ignore_mismatched_sizes": True,
          **_dtype_kw(torch.bfloat16 if cfg.lora else torch.float32)}
    model = AutoModelForSequenceClassification.from_pretrained(model_id, **kw)
    model.config.pad_token_id = tok.pad_token_id
    if cfg.lora:
        model = add_lora(model, cfg.lora)
    if cfg.grad_checkpointing:
        model.gradient_checkpointing_enable()
        if cfg.lora:
            model.enable_input_require_grads()
    return tok, model.to(device)


def add_lora(model, lora: dict):
    from peft import LoraConfig, TaskType, get_peft_model

    conf = LoraConfig(task_type=TaskType.SEQ_CLS, r=lora.get("r", 16), lora_alpha=lora.get("alpha", 32),
                      lora_dropout=lora.get("dropout", 0.05), target_modules=lora.get("targets", "all-linear"))
    model = get_peft_model(model, conf)  # đầu phân loại ("score") được peft tự để huấn luyện toàn phần
    for p in model.parameters():  # adapter + đầu phân loại ở fp32 (ổn định hơn khi cập nhật bằng AdamW)
        if p.requires_grad:
            p.data = p.data.float()
    return model


def collapsed(best_val_pr_auc: float, y_val, min_lift=0.05) -> bool:
    """Mô hình "sập" (đoán gần như hằng số): PR-AUC validation chỉ nhỉnh hơn tỉ lệ dương rất ít."""
    base = float(np.mean(y_val))
    return (best_val_pr_auc - base) / max(1 - base, 1e-9) < min_lift


def finetune(task, model_id: str | None = None, cfg: FTConfig | None = None, name: str | None = None,
             model=None, tokenizer=None, device=None, log=print, extra_tests: dict | None = None,
             retry: bool = True) -> FTResult:
    """Fine-tune một mô hình trên ``task`` (hoặc truyền sẵn ``model``/``tokenizer`` – dùng cho chạy thử).

    ``extra_tests``: {tên: danh sách tin} được chấm thêm sau khi huấn luyện (vd. test đã làm nhiễu), cùng thứ tự với test.
    ``retry``: nếu huấn luyện ra NaN hoặc mô hình sập, tự nạp lại và huấn luyện lại **một lần** với learning rate
    giảm một nửa (và tắt bf16 nếu ra NaN với encoder), giữ nguyên cách chia train/validation."""
    import gc

    import torch

    cfg = cfg or FTConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    can_reload = model_id is not None and model is None
    for attempt in range(2 if retry and can_reload else 1):
        if model is None:
            tokenizer, model = load_model(model_id, cfg, device)
        else:
            model = model.to(device)
            if cfg.lora:
                model = add_lora(model, cfg.lora)
        try:
            res = _train(task, model, tokenizer, cfg, device, log, extra_tests, attempt)
        except TrainingDiverged as e:
            if attempt or not (retry and can_reload):
                raise
            log(f"  !! {e} -> huấn luyện lại với lr {cfg.lr / 2:g}" + ("" if cfg.lora else ", tắt bf16"))
            cfg = FTConfig(**{**cfg.__dict__, "lr": cfg.lr / 2, "bf16": cfg.bf16 if cfg.lora else False})
        else:
            if attempt == 0 and retry and can_reload and collapsed(res.best_val_pr_auc, res.y_val):
                log(f"  !! mô hình sập (PR-AUC validation {res.best_val_pr_auc:.3f} ≈ tỉ lệ dương "
                    f"{np.mean(res.y_val):.3f}) -> huấn luyện lại với lr {cfg.lr / 2:g}")
                cfg = FTConfig(**{**cfg.__dict__, "lr": cfg.lr / 2})
            else:
                res.name = name or model_id or type(model).__name__
                res.config = {**{k: v for k, v in cfg.__dict__.items() if k != "lora"}, "lora": cfg.lora,
                              "attempts": attempt + 1}
                return res
        del model
        model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return res  # chỉ tới đây khi lần huấn luyện lại vẫn sập: trả về kết quả lần cuối để người dùng tự đánh giá


def _train(task, model, tok, cfg: FTConfig, device, log, extra_tests, attempt=0) -> FTResult:
    import torch
    from transformers import get_linear_schedule_with_warmup

    torch.manual_seed(cfg.seed + 1000 * attempt)
    Xtr, Xva, ytr, yva = split_train_val(task, cfg.split_seed)  # cách chia không đổi giữa các lần thử / seed
    enc = lambda xs: tok([str(x) for x in xs], truncation=True, max_length=cfg.max_len)["input_ids"]
    Itr, Iva, Ite = enc(Xtr), enc(Xva), enc(list(task.X_test))
    use_bf16 = cfg.bf16 and device == "cuda" and torch.cuda.is_bf16_supported()

    def batch(ids_list, idx):
        b = tok.pad({"input_ids": [ids_list[i] for i in idx]}, return_tensors="pt")
        return {k: v.to(device) for k, v in b.items()}

    @torch.no_grad()
    def predict(ids_list, bs=None):
        model.eval()
        bs = bs or cfg.batch_size * 4
        order = np.argsort([len(x) for x in ids_list])  # gom câu cùng độ dài -> ít padding
        out = np.empty(len(ids_list), dtype=np.float32)
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logit = model(**batch(ids_list, idx)).logits[:, 0]
            out[idx] = logit.float().cpu().numpy()
        if not np.isfinite(out).all():
            raise TrainingDiverged("điểm dự đoán có NaN/inf")
        return out

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(Itr) / (cfg.batch_size * cfg.grad_accum))
    total = max(1, int(steps_per_epoch * cfg.epochs))
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup * total), total)
    pw = torch.tensor(cfg.pos_weight, device=device) if cfg.pos_weight else None
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    yt = torch.tensor(ytr, dtype=torch.float32)
    eval_every = max(1, steps_per_epoch // cfg.eval_per_epoch)
    rng = np.random.default_rng(cfg.seed + 1000 * attempt)
    best, best_state, bad, step, hist = -1.0, None, 0, 0, []
    t0 = time.perf_counter()
    done = False
    while not done:
        order = rng.permutation(len(Itr))
        model.train()
        for k in range(0, len(order), cfg.batch_size):
            idx = order[k:k + cfg.batch_size]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logit = model(**batch(Itr, idx)).logits[:, 0]
            loss = loss_fn(logit.float(), yt[idx].to(device)) / cfg.grad_accum
            if not torch.isfinite(loss):
                raise TrainingDiverged(f"loss = {loss.item()} ở bước {step}")
            loss.backward()
            if (k // cfg.batch_size + 1) % cfg.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % eval_every == 0 or step >= total:
                s = predict(Iva)
                ap = average_precision_score(yva, s)
                hist.append({"step": step, "epoch": round(step / steps_per_epoch, 2), "val_pr_auc": ap,
                             "loss": loss.item() * cfg.grad_accum})
                log(f"  step {step}/{total} epoch {step / steps_per_epoch:.2f} val PR-AUC {ap:.4f}")
                if ap > best + 1e-4:
                    best, bad = ap, 0
                    best_state = {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters()
                                  if p.requires_grad}
                else:
                    bad += 1
                model.train()
                if bad >= cfg.patience or step >= total:
                    done = True
                    break
    if best_state is not None:
        params = dict(model.named_parameters())
        with torch.no_grad():
            for n, v in best_state.items():
                params[n].copy_(v.to(params[n].device))
    fit_time = time.perf_counter() - t0
    sig = lambda z: 1 / (1 + np.exp(-z.astype(np.float64)))
    p_val = sig(predict(Iva))
    t1 = time.perf_counter()
    p_test = sig(predict(Ite))
    ms = 1000 * (time.perf_counter() - t1) / max(len(Ite), 1)
    p_extra = {k: sig(predict(enc(list(v)))) for k, v in (extra_tests or {}).items()}
    return FTResult("", p_val, yva, p_test, hist, fit_time, sum(p.numel() for p in trainable), best, ms, p_extra)


def tfidf_val_test(task, seed=0):
    """TF-IDF (từ + ký tự) + LogReg như notebook 01, trả về xác suất trên validation và test (để ghép mô hình)."""
    from .text import _proba, make_text_models, make_vectorizer

    Xtr, Xva, ytr, yva = split_train_val(task, seed)
    vec = make_vectorizer()
    Vtr = vec.fit_transform(Xtr).astype(np.float32)
    m = make_text_models(include=["LogReg"])["LogReg"].fit(Vtr, ytr)
    return (_proba(m, vec.transform(Xva).astype(np.float32)), yva,
            _proba(m, vec.transform(list(task.X_test)).astype(np.float32)))


def greedy_ensemble(val_scores: dict, y_val, test_scores: dict, max_size=10):
    """Chọn mô hình để ghép theo Caruana et al. (2004): thêm dần (có lặp lại) mô hình làm tăng PR-AUC
    trên validation của trung bình hạng. Trả về (trọng số, điểm test đã ghép, PR-AUC validation)."""
    from scipy.stats import rankdata

    rv = {k: rankdata(v) / len(v) for k, v in val_scores.items()}
    rt = {k: rankdata(v) / len(v) for k, v in test_scores.items()}
    chosen, best = [], -1.0
    for _ in range(max_size):
        cand = {k: average_precision_score(y_val, np.mean([rv[c] for c in chosen + [k]], 0)) for k in rv}
        k, ap = max(cand.items(), key=lambda kv: kv[1])
        if ap <= best + 1e-5:
            break
        chosen.append(k)
        best = ap
    w = {k: chosen.count(k) / len(chosen) for k in set(chosen)}
    return w, sum(w[k] * rt[k] for k in w), best


def tiny_random_encoder(seed=0, decoder=False):
    """Encoder BERT (hoặc decoder Qwen2) tí hon, trọng số ngẫu nhiên – chỉ để chạy thử pipeline trên CPU."""
    import torch
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    from transformers import (BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast, Qwen2Config,
                              Qwen2ForSequenceClassification)

    tk = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tk.normalizer = normalizers.BertNormalizer(lowercase=True)
    tk.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    corpus = ["you are an idiot win a free prize call now hate them all thanks for the help " * 3,
              "please send the refund my order is late this is spam click the link"]
    tk.train_from_iterator(corpus, trainers.WordPieceTrainer(vocab_size=300, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", unk_token="[UNK]",
                                  cls_token="[CLS]", sep_token="[SEP]")
    torch.manual_seed(seed)
    if decoder:
        cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, num_labels=1, pad_token_id=tok.pad_token_id)
        return tok, Qwen2ForSequenceClassification(cfg)
    cfg = BertConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                     num_attention_heads=4, num_labels=1, pad_token_id=tok.pad_token_id)
    return tok, BertForSequenceClassification(cfg)


def embed_texts(texts, model, tok, instruction="", batch_size=32, max_len=256, device=None):
    """Embedding kiểu decoder (Qwen3-Embedding): đệm trái, lấy trạng thái ẩn của token cuối, chuẩn hoá L2."""
    import torch

    device = device or next(model.parameters()).device
    tok.padding_side = "left"
    texts = [instruction + str(t) for t in texts]
    order = np.argsort([len(t) for t in texts])
    out = None
    model.eval()
    for i in range(0, len(texts), batch_size):
        idx = order[i:i + batch_size]
        enc = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt").to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"
                                              and str(device).startswith("cuda")):
            h = model(**enc).last_hidden_state[:, -1].float()
        e = torch.nn.functional.normalize(h, dim=-1).cpu().numpy()
        if out is None:
            out = np.empty((len(texts), e.shape[1]), dtype=np.float32)
        out[idx] = e
    return out


def embedding_classifier(task, model_id: str | None = None, name: str | None = None, model=None, tokenizer=None,
                         device=None, extra_tests: dict | None = None, seed=0, batch_size=32) -> FTResult:
    """Embedding đóng băng + LogReg (C chọn theo PR-AUC trên validation)."""
    import torch
    from sklearn.linear_model import LogisticRegression

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id, **_dtype_kw(torch.bfloat16 if device == "cuda" else torch.float32))
    model = model.to(device)
    instr = f"Instruct: {task.question}\nQuery: "
    Xtr, Xva, ytr, yva = split_train_val(task, 0)  # cùng validation với các mô hình khác (LogReg tất định, seed không đổi kết quả)
    t0 = time.perf_counter()
    emb = lambda xs: embed_texts(xs, model, tokenizer, instr, batch_size=batch_size, device=device)
    Etr, Eva = emb(Xtr), emb(Xva)
    best = (-1.0, None)
    for C in (0.1, 0.3, 1.0, 3.0, 10.0):
        clf = LogisticRegression(C=C, max_iter=3000).fit(Etr, ytr)
        ap = average_precision_score(yva, clf.predict_proba(Eva)[:, 1])
        if ap > best[0]:
            best = (ap, clf)
    clf = best[1]
    p_val = clf.predict_proba(Eva)[:, 1]
    fit_time = time.perf_counter() - t0
    t1 = time.perf_counter()
    p_test = clf.predict_proba(emb(list(task.X_test)))[:, 1]
    ms = 1000 * (time.perf_counter() - t1) / max(len(task.X_test), 1)
    p_extra = {k: clf.predict_proba(emb(list(v)))[:, 1] for k, v in (extra_tests or {}).items()}
    return FTResult(name or model_id or "embedding", p_val, yva, p_test, [{"C": clf.C, "val_pr_auc": best[0]}],
                    fit_time, int(clf.coef_.size), best[0], ms, p_extra)
