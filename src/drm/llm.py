"""LLM cho (1) phân loại dữ liệu bảng và (2) ra quyết định đa tiêu chí.

Thiết kế để chạy trên Google Colab (GPU T4/L4/A100). Mọi thí nghiệm dùng chung lớp ``HFScorer``:

* ``label_logprobs``: chấm điểm log P(nhãn | prompt) cho từng nhãn -> xác suất lớp
  (ổn định hơn so với sinh văn bản rồi parse; cho phép tính ROC-AUC).
* ``generate``: sinh văn bản (dùng cho xếp hạng MCDM, ma trận AHP, kiểm tra "học thuộc").
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import mcdm
from .data import Dataset, serialize_row

# Mô hình mở cỡ nhỏ/vừa. Dung lượng VRAM ước tính ở fp16; "4bit" = nên nạp lượng tử hoá trên T4 (16GB).
MODEL_ZOO = {
    # Qwen (Alibaba)
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B-Instruct", "1 GB"),
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B-Instruct", "3 GB"),
    "qwen2.5-3b": ("Qwen/Qwen2.5-3B-Instruct", "6 GB"),
    "qwen2.5-7b": ("Qwen/Qwen2.5-7B-Instruct", "15 GB / 4bit"),
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B", "1.2 GB"),
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", "3.4 GB"),
    "qwen3-4b": ("Qwen/Qwen3-4B", "8 GB"),
    "qwen3-8b": ("Qwen/Qwen3-8B", "16 GB / 4bit"),
    # Phi "mini" (Microsoft)
    "phi-3.5-mini": ("microsoft/Phi-3.5-mini-instruct", "7.6 GB"),
    "phi-4-mini": ("microsoft/Phi-4-mini-instruct", "7.6 GB"),
    # Gemma (Google) – cần chấp nhận license trên Hugging Face + HF_TOKEN
    "gemma-2-2b": ("google/gemma-2-2b-it", "5 GB"),
    "gemma-3-1b": ("google/gemma-3-1b-it", "2 GB"),
    # Llama (Meta) – cần chấp nhận license + HF_TOKEN
    "llama-3.2-1b": ("meta-llama/Llama-3.2-1B-Instruct", "2.5 GB"),
    "llama-3.2-3b": ("meta-llama/Llama-3.2-3B-Instruct", "6.5 GB"),
    # Khác
    "smollm2-1.7b": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "3.4 GB"),
    "mistral-7b": ("mistralai/Mistral-7B-Instruct-v0.3", "14.5 GB / 4bit"),
}


def verbalizer(c: str) -> str:
    return c[:1].upper() + c[1:]


# --------------------------------------------------------------------------- model wrapper


class HFScorer:
    def __init__(self, model_id: str | None = None, model=None, tokenizer=None, load_in_4bit: bool = False,
                 batch_size: int = 8, max_length: int = 4096, device: str | None = None, name: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.name = name or model_id or "custom"
        if model is None:
            if model_id in MODEL_ZOO:
                model_id = MODEL_ZOO[model_id][0]
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if torch.cuda.is_available():  # T4 không hỗ trợ bf16 gốc -> fp16
                dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dt = torch.float32
            kw = {"dtype": dt, "device_map": "auto"}
            if load_in_4bit:
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
            try:
                model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
            except TypeError:  # transformers cũ dùng torch_dtype
                kw["torch_dtype"] = kw.pop("dtype")
                model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        self.model = model.eval()
        self.tok = tokenizer
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.device = device or next(self.model.parameters()).device
        self.batch_size = batch_size
        self.max_length = max_length

    # ------------------------------------------------------------------ prompting
    def chat(self, user_text: str) -> str:
        """Bọc nội dung bằng chat template (không dùng system role vì Gemma không hỗ trợ)."""
        if getattr(self.tok, "chat_template", None):
            msgs = [{"role": "user", "content": user_text}]
            try:  # Qwen3: tắt chế độ "thinking" để trả lời trực tiếp
                return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False)
            except TypeError:
                return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return user_text + "\nAnswer: "

    def _ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    # ------------------------------------------------------------------ scoring
    def label_logprobs(self, prompts: list[str], labels: list[str]) -> np.ndarray:
        """log P(label | prompt) (tổng log-prob các token của nhãn). Kết quả: (n_prompts, n_labels).

        Left-padding + position_ids tường minh, và chỉ tính logits ở K vị trí cuối
        (``logits_to_keep``) -> tránh tensor logits khổng lồ (batch x seq x vocab ~150k) trên GPU nhỏ.
        """
        torch = self.torch
        label_ids = [self._ids(l) for l in labels]
        K = max(len(l) for l in label_ids) + 1
        seqs, meta = [], []
        for i, p in enumerate(prompts):
            pid = self._ids(p)
            for j, lid in enumerate(label_ids):
                ids = (pid + lid)[-self.max_length:]
                seqs.append(ids)
                meta.append((i, j, len(lid)))
        out = np.zeros((len(prompts), len(labels)))
        order = np.argsort([len(s) for s in seqs])  # gom các chuỗi dài bằng nhau -> ít padding
        pad = self.tok.pad_token_id
        for b in range(0, len(order), self.batch_size):
            idx = order[b:b + self.batch_size]
            L = max(len(seqs[k]) for k in idx)
            input_ids = torch.full((len(idx), L), pad, dtype=torch.long)
            attn = torch.zeros((len(idx), L), dtype=torch.long)
            for r, k in enumerate(idx):
                input_ids[r, L - len(seqs[k]):] = torch.tensor(seqs[k])
                attn[r, L - len(seqs[k]):] = 1
            pos = (attn.cumsum(-1) - 1).clamp(min=0)
            args = dict(input_ids=input_ids.to(self.device), attention_mask=attn.to(self.device),
                        position_ids=pos.to(self.device), use_cache=False)
            with torch.no_grad():
                try:
                    logits = self.model(**args, logits_to_keep=K).logits
                except TypeError:  # transformers cũ
                    logits = self.model(**args).logits[:, -K:]
            logp = torch.log_softmax(logits[:, -K:].float(), -1).cpu()
            for r, k in enumerate(idx):
                i, j, n = meta[k]
                tgt = input_ids[r, L - n:]
                out[i, j] = logp[r, torch.arange(K - n - 1, K - 1), tgt].sum().item()
        return out

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0, seed: int = 0) -> str:
        torch = self.torch
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        kw = {"max_new_tokens": max_new_tokens, "pad_token_id": self.tok.pad_token_id}
        if temperature > 0:
            torch.manual_seed(seed)
            kw.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            kw.update(do_sample=False)
        with torch.no_grad():
            out = self.model.generate(**enc, **kw)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def free(self):
        import gc
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


# --------------------------------------------------------------------------- (1) tabular classification


def build_prompt(ds: Dataset, row: pd.Series, shots: list[tuple[pd.Series, int]] = (), style: str = "text") -> str:
    labels = [verbalizer(c) for c in ds.classes]
    opts = ", ".join(labels)
    s = [f"You will see a record from a tabular dataset. Task: {ds.task}",
         f"Answer with exactly one word from: {opts}.", ""]
    for i, (r, y) in enumerate(shots, 1):
        s += [f"Example {i}:", serialize_row(r, style), f"Answer: {labels[y]}", ""]
    s += ["Record:" if not shots else "Now classify this record:", serialize_row(row, style),
          f"{ds.task} Answer with one of: {opts}."]
    return "\n".join(s)


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def split_for_llm(ds: Dataset, n_test=200, seed=0):
    from sklearn.model_selection import train_test_split

    tr, te = train_test_split(np.arange(len(ds.y)), test_size=0.3, stratify=ds.y, random_state=seed)
    if len(te) > n_test:
        te, _ = train_test_split(te, train_size=n_test, stratify=ds.y.iloc[te], random_state=seed)
    return np.sort(tr), np.sort(te)


def sample_shots(ds: Dataset, pool_idx, k: int, seed=0) -> list[int]:
    """Chọn k ví dụ cân bằng lớp (round-robin) từ tập train."""
    if k == 0:
        return []
    rng = np.random.default_rng(seed)
    by_c = {c: list(rng.permutation(pool_idx[ds.y.iloc[pool_idx].to_numpy() == c])) for c in range(ds.n_classes)}
    out = []
    while len(out) < k:
        for c in range(ds.n_classes):
            if by_c[c] and len(out) < k:
                out.append(int(by_c[c].pop()))
    return list(rng.permutation(out))


def content_free_row(ds: Dataset) -> pd.Series:
    return pd.Series({c: "N/A" for c in ds.X.columns})


def evaluate_llm_classification(scorer: HFScorer, ds: Dataset, shots=(0, 4, 8), n_test=200, seed=0,
                                style="text", calibrate=True) -> tuple[pd.DataFrame, dict]:
    """Zero-/few-shot classification theo kiểu TabLLM. Trả về (bảng metric, dict xác suất)."""
    from .benchmark import _scores

    tr, te = split_for_llm(ds, n_test, seed)
    labels = [verbalizer(c) for c in ds.classes]
    y_te = ds.y.iloc[te].to_numpy()
    rows, probs = [], {}
    for k in shots:
        shot_idx = sample_shots(ds, tr, k, seed)
        shots_rs = [(ds.X.iloc[i], int(ds.y.iloc[i])) for i in shot_idx]
        prompts = [scorer.chat(build_prompt(ds, ds.X.iloc[i], shots_rs, style)) for i in te]
        t0 = time.perf_counter()
        lp = scorer.label_logprobs(prompts, labels)
        dt = time.perf_counter() - t0
        variants = {"raw": _softmax(lp)}
        if calibrate:  # Contextual calibration (Zhao et al., 2021)
            cf = scorer.label_logprobs([scorer.chat(build_prompt(ds, content_free_row(ds), shots_rs, style))], labels)
            variants["calibrated"] = _softmax(lp - cf)
        for v, P in variants.items():
            rows.append({"model": scorer.name, "dataset": ds.name, "shots": k, "variant": v,
                         **_scores(y_te, P, np.arange(ds.n_classes)), "sec_per_row": dt / len(te),
                         "pred_class_share": np.round(np.bincount(P.argmax(1), minlength=ds.n_classes) / len(te), 3).tolist()})
            probs[(k, v)] = P
    return pd.DataFrame(rows), {"test_idx": te, "train_idx": tr, "probs": probs}


def few_shot_ml_baselines(ds: Dataset, train_idx, test_idx, shots=(4, 8, 16, 32, 64), seed=0, n_rep=5):
    """Baseline ML cổ điển chỉ được huấn luyện trên k mẫu (cùng chế độ few-shot với LLM)."""
    import warnings

    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    from .benchmark import _scores, make_pipeline_for

    models = {"LogReg": LogisticRegression(max_iter=2000), "RandomForest": RandomForestClassifier(300, random_state=seed)}
    y_te = ds.y.iloc[test_idx].to_numpy()
    rows = []
    for k in shots:
        for rep in range(n_rep):
            idx = sample_shots(ds, train_idx, k, seed + rep)
            for name, m in models.items():
                pipe = make_pipeline_for(name, clone(m), ds)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe.fit(ds.X.iloc[idx], ds.y.iloc[idx])
                    P = np.zeros((len(test_idx), ds.n_classes))
                    P[:, pipe.classes_] = pipe.predict_proba(ds.X.iloc[test_idx])
                rows.append({"model": name, "dataset": ds.name, "shots": k, "rep": rep,
                             **_scores(y_te, P, np.arange(ds.n_classes))})
    return pd.DataFrame(rows).groupby(["model", "dataset", "shots"]).mean(numeric_only=True).reset_index()


def llm_prior_stacking(ds: Dataset, llm_logit_tr: np.ndarray, llm_logit_te: np.ndarray, train_idx, test_idx,
                       shots=(8, 16, 32, 64, 128), seed=0, n_rep=5):
    """Hướng cải tiến: dùng điểm zero-shot của LLM làm đặc trưng 'tri thức tiên nghiệm' cho mô hình ML.

    So sánh trên cùng k mẫu: LogReg(X)  vs  LogReg(X + llm_logit)  vs  LLM zero-shot đơn thuần.
    ``llm_logit_*``: log-odds lớp dương (bài toán nhị phân) cho mọi dòng trong train_idx / test_idx.
    """
    import warnings

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    from .benchmark import make_pipeline_for

    assert ds.n_classes == 2
    y_te = ds.y.iloc[test_idx].to_numpy()
    pos = {i: p for p, i in enumerate(train_idx)}
    Xte = ds.X.iloc[test_idx].copy()
    Xte["llm_logit"] = llm_logit_te
    rows = [{"dataset": ds.name, "shots": 0, "method": "LLM zero-shot", "roc_auc": roc_auc_score(y_te, llm_logit_te)}]
    for k in shots:
        for rep in range(n_rep):
            idx = sample_shots(ds, train_idx, k, seed + rep)
            Xtr = ds.X.iloc[idx].copy()
            Xtr["llm_logit"] = llm_logit_tr[[pos[i] for i in idx]]
            ytr = ds.y.iloc[idx]
            for name, cols in [("LogReg (X)", list(ds.X.columns)), ("LogReg (X + LLM prior)", list(Xtr.columns))]:
                sub = Dataset(ds.name, Xtr[cols], ytr, ds.classes, ds.task, cat_cols=ds.cat_cols)
                pipe = make_pipeline_for("LogReg", LogisticRegression(max_iter=2000, C=0.5), sub)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe.fit(Xtr[cols], ytr)
                    p = pipe.predict_proba(Xte[cols])[:, list(pipe.classes_).index(1)]
                rows.append({"dataset": ds.name, "shots": k, "method": name, "rep": rep,
                             "roc_auc": roc_auc_score(y_te, p)})
    return pd.DataFrame(rows).groupby(["dataset", "shots", "method"], as_index=False)["roc_auc"].mean()


# --------------------------------------------------------------------------- (2) LLM as decision maker


def _extract_json(text: str):
    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    if not m:
        return None
    s = m.group(1)
    for cand in (s, s.replace("'", '"')):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None


def _table(df: pd.DataFrame) -> str:
    try:
        return df.round(4).to_markdown()
    except ImportError:  # thiếu tabulate
        return df.round(4).to_string()


def llm_rank_alternatives(scorer: HFScorer, matrix: pd.DataFrame, types: dict, scenario: str,
                          temperature=0.0, seed=0) -> tuple[list[str] | None, str]:
    """Cho LLM bảng quyết định + kịch bản, yêu cầu xếp hạng phương án (trả về JSON list)."""
    desc = ", ".join(f"{c} ({'higher is better' if t > 0 else 'lower is better'})" for c, t in types.items())
    prompt = (f"You are a machine-learning engineer choosing a classification model.\nScenario: {scenario}\n"
              f"Criteria: {desc}.\n\nDecision matrix:\n{_table(matrix)}\n\n"
              "Rank ALL models from best to worst for this scenario. Reply ONLY with a JSON list of model names, "
              'e.g. ["A", "B", ...].')
    out = scorer.generate(scorer.chat(prompt), max_new_tokens=300, temperature=temperature, seed=seed)
    js = _extract_json(out)
    alts = list(matrix.index)
    if isinstance(js, list):
        ranking = [a for a in js if a in alts]
        ranking += [a for a in alts if a not in ranking]  # thiếu -> xếp cuối
        return ranking, out
    return None, out


def llm_ahp_weights(scorer: HFScorer, criteria: dict[str, str], scenario: str, temperature=0.0, seed=0):
    """LLM đóng vai chuyên gia AHP: sinh so sánh cặp (thang Saaty 1-9) -> trọng số + CR."""
    names = list(criteria)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    lst = "\n".join(f'- "{a} vs {b}"' for a, b in pairs)
    crit = "\n".join(f"- {k}: {v}" for k, v in criteria.items())
    prompt = (f"You are an expert using the Analytic Hierarchy Process (AHP).\nScenario: {scenario}\n"
              f"Criteria:\n{crit}\n\nFor each pair 'X vs Y' give how much more important X is than Y on Saaty's "
              "scale: 1 (equal), 3 (moderate), 5 (strong), 7 (very strong), 9 (extreme); use reciprocals "
              "like 1/3, 1/5 if Y is more important.\nPairs:\n" + lst +
              '\nReply ONLY with JSON mapping each pair string to a number, e.g. {"A vs B": 3, "A vs C": 0.2}.')
    out = scorer.generate(scorer.chat(prompt), max_new_tokens=400, temperature=temperature, seed=seed)
    js = _extract_json(out.replace("1/9", "0.1111").replace("1/7", "0.1429").replace("1/5", "0.2")
                       .replace("1/3", "0.3333"))
    if not isinstance(js, dict):
        return None, None, out
    upper = {}
    for a, b in pairs:
        v = js.get(f"{a} vs {b}")
        if v is None:
            v2 = js.get(f"{b} vs {a}")
            v = 1 / float(v2) if v2 else 1.0
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = 1.0
        upper[(a, b)] = min(max(v, 1 / 9), 9)
    A = mcdm.ahp_from_upper(upper, names)
    w, cr = mcdm.ahp_weights(A)
    return pd.Series(w, index=names), cr, out


# --------------------------------------------------------------------------- data contamination check


def row_completion_test(scorer: HFScorer, csv_lines: list[str], n_trials=10, n_prefix=15, seed=0) -> dict:
    """Kiểm tra 'học thuộc' (Bordt et al., 2024): cho n dòng liên tiếp của file CSV gốc,
    yêu cầu sinh dòng kế tiếp; đo tỉ lệ trường khớp chính xác. Tỉ lệ cao -> nguy cơ rò rỉ dữ liệu."""
    rng = np.random.default_rng(seed)
    scores, exact = [], 0
    for _ in range(n_trials):
        s = int(rng.integers(0, len(csv_lines) - n_prefix - 1))
        prefix = "\n".join(csv_lines[s:s + n_prefix])
        truth = csv_lines[s + n_prefix].strip()
        prompt = ("Continue this CSV file with the next row exactly. Reply with the next row only.\n\n" + prefix)
        out = scorer.generate(scorer.chat(prompt), max_new_tokens=80).strip().splitlines()
        guess = out[0].strip() if out else ""
        t, g = truth.split(","), guess.split(",")
        scores.append(np.mean([a.strip() == b.strip() for a, b in zip(t, g)]) if g else 0.0)
        exact += guess == truth
    return {"field_match": float(np.mean(scores)), "exact_rows": exact, "trials": n_trials}


def tiny_random_scorer(seed: int = 0) -> HFScorer:
    """Mô hình Qwen2 tí hon, trọng số ngẫu nhiên – chỉ để chạy thử pipeline trên CPU (kết quả vô nghĩa)."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    corpus = ["The age is 39. Answer: Yes No Good Bad Adelie Chinstrap Gentoo Example Record " * 5,
              "Does this patient have diabetes? classify tabular dataset JSON vs AHP criteria"]
    tk.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=400, special_tokens=["<|pad|>", "<|im_start|>", "<|im_end|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|pad|>")
    tok.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
                         "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8192,
                      pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return HFScorer(model=Qwen2ForCausalLM(cfg), tokenizer=tok, batch_size=8, name="tiny-random", device="cpu")


@dataclass
class RunConfig:
    models: tuple = ("qwen2.5-0.5b", "qwen2.5-1.5b", "qwen3-1.7b", "phi-4-mini")
    datasets: tuple = ("diabetes", "german_credit", "adult", "breast_cancer", "haberman", "titanic", "penguins")
    shots: tuple = (0, 4, 8)
    n_test: int = 200
    seed: int = 0
