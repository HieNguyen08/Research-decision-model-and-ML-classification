"""Rev / Kev: decision model kiểu "Qwen + LoRA + pointer head" (notebook 07).

Rev (github.com/robbalian/rev, MIT) lấy công thức của Kev (Jared Palmer): Qwen3.5/3.8 + LoRA hạng 16 trên các phép chiếu
attention/DeltaNet + một *pointer head* 256 chiều. Mô hình không sinh chữ: prompt có dạng

    State:\\n{state}\\nQuestion: {instructions}\\nOptions:\\n{key}: {desc}\\n ... Decision:

head lấy trạng thái ẩn ở token cuối (``Decision:``) làm *query* và trạng thái ẩn ở token cuối của từng dòng lựa chọn làm *key*;
logit của lựa chọn j = <headq(h_last), headk(h_opt_j)> / 16 / T, softmax trên các lựa chọn. Code ở đây làm lại đúng
``serve_local.py`` của Rev (nạp checkpoint, gộp LoRA, chấm điểm), nhưng trả về **cùng định dạng với Jev/WaterSheep**
(``noul`` / ``choice`` / ``score``) để dùng chung hàm đánh giá của ``jev.py``.

Module gồm:
* phần toán thuần numpy (chạy CPU, có test): logit pointer, softmax có nhiệt độ, tempering ``p^(1/T)``, fit T trên validation,
  ECE đa lớp, Platt, đường risk–coverage;
* chuyển câu hỏi Jev -> lựa chọn Rev và ngược lại (``to_rev_options`` / ``to_jev_answer``);
* ``encode_prompt``: bố cục prompt + vị trí token của từng lựa chọn, có tuỳ chọn xáo thứ tự lựa chọn (dùng khi huấn luyện);
* ``TorchRevScorer`` / ``load_rev``: nạp base Qwen từ Hub + LoRA + head từ checkpoint Rev (bf16 gộp LoRA, hoặc 4-bit không gộp);
* ``DryRunScorer``: bản giả lập numpy tí hon, cùng giao diện, để chạy thử notebook và test trên CPU không cần trọng số;
* ``RevAdapter``: ``ask`` / ``ask_many`` (gom nhiều câu hỏi × nhiều tin vào một batch) / ``cost_report`` như ``WaterSheepAdapter``.

Không gọi TypeSafe/Jev, không dùng output Jev làm nhãn (điều khoản TypeSafe cấm chưng cất). File
``results/jev_official_answers.jsonl`` trong repo Rev là output Jev: **không** dùng để huấn luyện hay chọn mô hình.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
import zlib

import numpy as np

VERSION = "1"

REV_REPOS = {
    "rev-4b": "robbalian/rev-qwen3.5-4b",
    "rev-9b": "robbalian/rev-qwen3.5-9b",
    "rev-27b": "robbalian/rev-qwen3.8-27b",
}
# Revision base model được Rev ghim (copy từ serve_local.py / train/train.py). Chỉ dùng khi checkpoint không ghi revision.
BASE_REVISIONS = {
    "Qwen/Qwen3.8-27B": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
}
LORA_TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"}
LORA_RANK, LORA_ALPHA = 16, 32
LORA_SCALE = LORA_ALPHA / LORA_RANK      # = 2, như Rev
HEAD_DIM = 256                            # logit chia sqrt(256) = 16, như Rev
MAX_ROW_TOKENS = 16384


# =========================================================================== toán thuần numpy


def pointer_logits(h_last, h_opts, Wq, Wk) -> np.ndarray:
    """Logit của pointer head (chưa chia nhiệt độ). ``h_last``: (H,), ``h_opts``: (K, H), ``Wq``/``Wk``: (D, H) như
    ``torch.nn.Linear(H, D, bias=False).weight``. Trả về (K,) = (Wk h_opt · Wq h_last) / sqrt(D)."""
    q = np.asarray(Wq, float) @ np.asarray(h_last, float)
    k = np.asarray(h_opts, float) @ np.asarray(Wk, float).T
    return k @ q / math.sqrt(q.shape[0])


def softmax(z, T: float = 1.0) -> np.ndarray:
    z = np.asarray(z, float) / T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def temper(p, T: float) -> np.ndarray:
    """Đổi nhiệt độ của phân phối đã có: ``p^(1/T)`` chuẩn hoá lại (= chia logit cho T). Như ``fit_temperature.py``."""
    lp = np.log(np.clip(np.asarray(p, float), 1e-12, 1.0)) / T
    return softmax(lp)


def temper_binary(p_yes, T: float) -> np.ndarray:
    """``temper`` cho bài nhị phân khi chỉ lưu P(yes)."""
    p = np.clip(np.asarray(p_yes, float), 1e-12, 1 - 1e-12)
    z = (np.log(p) - np.log1p(-p)) / T
    return 1 / (1 + np.exp(-z))


def nll(probs: list, labels) -> float:
    """Negative log-likelihood trung bình; ``probs``: danh sách mảng xác suất (số lựa chọn có thể khác nhau)."""
    return float(-np.mean([math.log(max(float(p[int(y)]), 1e-12)) for p, y in zip(probs, labels)]))


def fit_temperature(probs: list, labels, grid=None) -> tuple[float, float]:
    """Tìm T (lưới 0.40..3.00, bước 0.01 như Rev) làm nhỏ NLL của nhãn đúng sau khi tempering. Trả về (T, NLL).
    Chỉ gọi trên **validation**; argmax không đổi nên accuracy/PR-AUC giữ nguyên."""
    grid = np.round(np.arange(0.40, 3.0001, 0.01), 2) if grid is None else np.asarray(grid, float)
    labels = [int(y) for y in labels]
    best = min(((nll([temper(p, T) for p in probs], labels), float(T)) for T in grid))
    return best[1], best[0]


def multiclass_ece(probs: list, labels, n_bins: int = 10) -> float:
    """ECE theo độ tự tin của lựa chọn cao nhất (cùng định nghĩa với fit_temperature.py của Rev)."""
    conf = np.array([float(np.max(p)) for p in probs])
    hit = np.array([int(np.argmax(p)) == int(y) for p, y in zip(probs, labels)], float)
    e = 0.0
    for b in range(n_bins):
        m = (conf > b / n_bins) & (conf <= (b + 1) / n_bins)
        if m.any():
            e += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(e)


def platt(p_cal, y_cal, p_eval) -> np.ndarray:
    """Hiệu chỉnh Platt (LogReg trên log-odds) như notebook 05: học trên ``p_cal`` có nhãn, áp cho ``p_eval``."""
    from sklearn.linear_model import LogisticRegression

    lo = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    y_cal = np.asarray(y_cal)
    if len(np.unique(y_cal)) < 2:
        return np.asarray(p_eval, float)
    return LogisticRegression().fit(lo(np.asarray(p_cal)), y_cal).predict_proba(lo(np.asarray(p_eval)))[:, 1]


def risk_coverage(confidence, correct, auto_mask=None) -> "pd.DataFrame":
    """Đường risk–coverage cho vùng tự quyết: xếp ca theo độ tự tin giảm dần; coverage = tỉ lệ ca được tự xử lý
    (trên **toàn bộ** ca), risk = tỉ lệ sai trong số ca tự xử lý. ``auto_mask``: chỉ những ca model muốn tự xử lý
    (vd. không chọn "human") mới được tính vào vùng tự quyết."""
    import pandas as pd

    c = np.asarray(confidence, float)
    ok = np.asarray(correct, bool)
    auto = np.ones(len(c), bool) if auto_mask is None else np.asarray(auto_mask, bool)
    order = np.argsort(-np.where(auto, c, -np.inf), kind="stable")
    order = order[auto[order]]
    n = len(c)
    err = np.cumsum(~ok[order])
    k = np.arange(1, len(order) + 1)
    return pd.DataFrame({"threshold": c[order], "coverage": k / n, "risk": err / k})


def coverage_at_risk(curve, max_risk: float = 0.05) -> float:
    """Coverage lớn nhất mà risk ≤ ``max_risk`` (0 nếu không đạt ở bất kỳ ngưỡng nào)."""
    ok = curve[curve["risk"] <= max_risk]
    return float(ok["coverage"].max()) if len(ok) else 0.0


def symmetric_kl(log_p, log_q) -> float:
    """KL đối xứng giữa hai phân phối (log-xác suất), dùng cho loss nhất quán khi đổi thứ tự lựa chọn."""
    p, q = np.exp(log_p), np.exp(log_q)
    return float(((p * (log_p - log_q)).sum() + (q * (log_q - log_p)).sum()) / 2)


# =========================================================================== câu hỏi Jev <-> lựa chọn Rev

NOUL_DEFAULT = {"yes": "Yes, the answer to the question is yes", "no": "No, the answer to the question is no"}


def to_rev_options(q: dict) -> dict:
    """Câu hỏi có kiểu (định dạng Jev, xem ``jev.noul/choice/score``) -> ``{key: mô tả}`` cho prompt Rev.

    noul -> {"yes", "no"} (dùng ``criteria.true/false`` nếu có); choice -> giữ nguyên; score -> {"0": bậc 0, "1": ...}."""
    t = q["type"]
    if t == "noul":
        c = q.get("criteria") or {}
        return {"yes": c.get("true") or NOUL_DEFAULT["yes"], "no": c.get("false") or NOUL_DEFAULT["no"]}
    if t == "choice":
        return {str(k): (v or str(k)) for k, v in q["criteria"].items()}
    if t == "score":
        return {str(i): str(d) for i, d in enumerate(q["criteria"])}
    raise ValueError(f"loại câu hỏi không hỗ trợ: {t}")


def to_jev_answer(q: dict, probs: dict) -> dict:
    """Phân phối trên lựa chọn Rev -> câu trả lời cùng định dạng Jev/WaterSheep (``jev.read_noul`` ... đọc được)."""
    t = q["type"]
    if t == "noul":
        return {"type": "noul", "noul": float(probs["yes"])}
    keys = list(probs)
    p = np.array([probs[k] for k in keys], float)
    if t == "choice":
        return {"type": "choice", "choice": keys[int(p.argmax())], "confidence": float(p.max()),
                "probabilities": {k: float(v) for k, v in zip(keys, p)}}
    levels = list(q["criteria"])
    return {"type": "score", "score": float(np.dot(p, np.arange(len(keys)))), "confidence": float(p.max()),
            "legend": {str(i): d for i, d in enumerate(levels)}, "probabilities": {k: float(v) for k, v in zip(keys, p)}}


def state_text(state, state_format: str = "raw") -> str:
    """Như serve_local.py: chuỗi giữ nguyên khi checkpoint dùng state 'raw', còn lại JSON gọn."""
    if state_format == "raw" and isinstance(state, str):
        return state
    return json.dumps(state, separators=(",", ":"))  # ensure_ascii mặc định như serve_local: emoji/dấu -> \uXXXX


def encode_prompt(encode, state, instructions, options: dict, state_format: str = "raw", seed: int | None = None):
    """Bố cục prompt Rev. ``encode``: hàm str -> list[int] (``tok.encode(s, add_special_tokens=False)``).

    Trả về (ids, positions, keys): ``positions[j]`` là chỉ số token **cuối** của dòng lựa chọn ``keys[j]``.
    ``seed`` khác None -> xáo thứ tự lựa chọn (Rev xáo khi huấn luyện để head không học vị trí)."""
    keys = list(options)
    if seed is not None:
        random.Random(seed).shuffle(keys)
    ins = instructions if isinstance(instructions, str) else json.dumps(instructions, separators=(",", ":"))
    ids = list(encode("State:\n" + state_text(state, state_format) + "\nQuestion: " + ins + "\nOptions:\n"))
    positions = []
    for k in keys:
        ids += list(encode(str(k) + ": " + str(options[k] or k) + "\n"))
        positions.append(len(ids) - 1)
    ids += list(encode("Decision:"))
    return ids, positions, keys


# =========================================================================== scorer giả lập (DRY_RUN, numpy)


class DryRunScorer:
    """Mô hình giả lập tí hon bằng numpy, cùng giao diện với ``TorchRevScorer``: tokenizer băm từ, trạng thái ẩn ở vị trí i
    là trung bình cộng dồn embedding ngẫu nhiên, pointer head ngẫu nhiên. Tất định theo ``seed``. Chỉ để kiểm tra luồng code."""

    name = "rev-dry-run"

    def __init__(self, seed: int = 0, vocab: int = 4096, hidden: int = 32, head_dim: int = 16, temperature: float = 1.0):
        r = np.random.default_rng(seed)
        self.E = r.normal(0, 1, (vocab, hidden))
        self.Wq = r.normal(0, 1 / math.sqrt(hidden), (head_dim, hidden))
        self.Wk = r.normal(0, 1 / math.sqrt(hidden), (head_dim, hidden))
        self.vocab = vocab
        self.temperature = temperature
        self.info = {"model": "dry-run", "state_format": "raw", "temperature": temperature, "device": "cpu",
                     "dtype": "fp64", "precision": "numpy", "server": "drm.rev.DryRunScorer"}
        self.state_format = "raw"
        self.device_type = "cpu"

    def encode(self, s: str) -> list:
        import re
        return [zlib.crc32(w.encode("utf-8")) % self.vocab for w in re.findall(r"\w+|[^\w\s]|\n", s)] or [0]

    def logits(self, rows: list, batch_size: int = 32) -> list:
        out = []
        for ids, positions in rows:
            h = np.cumsum(self.E[np.asarray(ids)], 0) / np.arange(1, len(ids) + 1)[:, None]
            out.append(pointer_logits(h[len(ids) - 1], h[positions], self.Wq, self.Wk))
        return out

    def peak_memory_gb(self) -> float:
        return 0.0


# =========================================================================== nạp checkpoint Rev thật (torch, GPU)


def _lora_wrapper_class():
    import torch

    class LoRALinear(torch.nn.Module):
        """``base(x) + scale * B(A x)`` như ``train/train.py`` của Rev; A, B ở fp32. Dùng khi huấn luyện, và khi base
        đã lượng tử hoá 4-bit (không gộp được LoRA vào trọng số)."""

        def __init__(self, base, rank: int = LORA_RANK, scale: float = LORA_SCALE, a=None, b=None, trainable=True):
            super().__init__()
            self.base, self.scale = base, scale
            dev = base.weight.device
            A = torch.empty(rank, base.in_features, device=dev, dtype=torch.float32)
            if a is None:
                torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            else:
                A.copy_(a.float())
            B = torch.zeros(base.out_features, rank, device=dev, dtype=torch.float32)
            if b is not None:
                B.copy_(b.float())
            self.a = torch.nn.Parameter(A, requires_grad=trainable)
            self.b = torch.nn.Parameter(B, requires_grad=trainable)

        def forward(self, x):
            upd = torch.nn.functional.linear(torch.nn.functional.linear(x, self.a.to(x.dtype)), self.b.to(x.dtype))
            return self.base(x) + self.scale * upd

    return LoRALinear


def attach_lora(backbone, adapters: dict | None = None, rank: int = LORA_RANK, scale: float = LORA_SCALE,
                trainable: bool = True, targets=LORA_TARGETS) -> dict:
    """Thay mọi ``nn.Linear`` có tên lá thuộc ``targets`` bằng ``LoRALinear``. ``adapters`` (từ checkpoint Rev:
    {đường dẫn: {"a", "b"}}) -> nạp sẵn; None -> khởi tạo mới (B = 0, tức mô hình ban đầu = base). Trả về {đường dẫn: module}."""
    import torch

    LoRALinear = _lora_wrapper_class()
    out = {}
    for path, module in list(backbone.named_modules()):
        if isinstance(module, torch.nn.Linear) and path.split(".")[-1] in targets:
            if adapters is not None and path not in adapters:
                continue
            w = (adapters or {}).get(path, {})
            parent, _, attr = path.rpartition(".")
            wrap = LoRALinear(module, rank, scale, w.get("a"), w.get("b"), trainable)
            setattr(backbone.get_submodule(parent) if parent else backbone, attr, wrap)
            out[path] = wrap
    if adapters is not None and len(out) != len(adapters):
        missing = sorted(set(adapters) - set(out))[:5]
        raise RuntimeError(f"checkpoint có adapter không khớp backbone: {missing}")
    if not out:
        raise RuntimeError("không tìm thấy lớp Linear nào để gắn LoRA")
    return out


def merge_lora(backbone, adapters: dict, scale: float = LORA_SCALE) -> int:
    """Gộp LoRA vào trọng số (``W += scale * B @ A``), đúng như serve_local.py. Trả về số lớp đã gộp."""
    import torch

    modules = dict(backbone.named_modules())
    n = 0
    for path, w in adapters.items():
        m = modules[path]
        assert isinstance(m, torch.nn.Linear) and path.split(".")[-1] in LORA_TARGETS, path
        with torch.no_grad():
            m.weight.add_((scale * (w["b"].float() @ w["a"].float())).to(m.weight.dtype).to(m.weight.device))
        n += 1
    assert n == len(adapters) and n > 0
    return n


def make_heads(hidden: int, device, head_dim: int = HEAD_DIM, state: tuple | None = None):
    """headq, headk: ``Linear(hidden, 256, bias=False)`` ở fp32 như Rev."""
    import torch

    hq = torch.nn.Linear(hidden, head_dim, bias=False, device=device, dtype=torch.float32)
    hk = torch.nn.Linear(hidden, head_dim, bias=False, device=device, dtype=torch.float32)
    if state is not None:
        hq.load_state_dict(state[0])
        hk.load_state_dict(state[1])
    return hq, hk


def head_logits(hidden, rows, headq, headk):
    """Logit pointer cho từng hàng trong batch: ``hidden`` (B, L, H); ``rows``: [(ids, positions)]. Trả về list tensor."""
    d = math.sqrt(headq.out_features)
    out = []
    for i, (ids, positions) in enumerate(rows):
        q = headq(hidden[i, len(ids) - 1].float())
        k = headk(hidden[i, list(positions)].float())
        out.append((k * q).sum(-1) / d)
    return out


def forward_rows(backbone, rows, pad_id: int, device, pad_multiple: int = 16, attention_mask: bool = False):
    """Một lượt forward, đệm phải. Như serve_local.py: không cần attention mask vì mô hình nhân quả (token thật không
    nhìn thấy token đệm phía sau); khi huấn luyện truyền mask như train.py."""
    import torch

    length = ((max(len(ids) for ids, _ in rows) + pad_multiple - 1) // pad_multiple) * pad_multiple
    inp = torch.full((len(rows), length), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(inp)
    for i, (ids, _) in enumerate(rows):
        inp[i, :len(ids)] = torch.tensor(ids, device=device)
        mask[i, :len(ids)] = 1
    return backbone(input_ids=inp, attention_mask=mask if attention_mask else None, use_cache=False).last_hidden_state


class TorchRevScorer:
    """Backbone (đã gộp hoặc gắn LoRA) + pointer head. ``logits(rows)`` chạy theo batch, gom hàng cùng độ dài."""

    def __init__(self, tok, backbone, headq, headk, info: dict):
        self.tok, self.backbone, self.headq, self.headk, self.info = tok, backbone, headq, headk, info
        self.device = next(backbone.parameters()).device
        self.device_type = self.device.type
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        self.state_format = info.get("state_format", "raw")
        self.temperature = float(info.get("temperature", 1.0))
        self.name = info.get("name") or info.get("model", "rev")

    def encode(self, s: str) -> list:
        return self.tok.encode(s, add_special_tokens=False)

    def logits(self, rows: list, batch_size: int = 32, max_batch_tokens: int = 65536) -> list:
        import torch

        out = [None] * len(rows)
        order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]))
        self.backbone.eval()
        with torch.inference_mode():
            s = 0
            while s < len(order):
                n = max(1, min(batch_size, max_batch_tokens // max(len(rows[order[min(s + batch_size, len(order)) - 1]][0]), 1)))
                idx = order[s:s + n]
                chunk = [rows[i] for i in idx]
                hidden = forward_rows(self.backbone, chunk, self.pad_id, self.device)
                for i, z in zip(idx, head_logits(hidden, chunk, self.headq, self.headk)):
                    out[i] = z.float().cpu().numpy().astype(np.float64)
                s += n
        return out

    def peak_memory_gb(self) -> float:
        import torch
        return torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0


def resolve_checkpoint(repo: str):
    """Thư mục/file local chứa checkpoint.pt, hoặc repo Hugging Face (chỉ tải checkpoint.pt, 60–200 MB)."""
    from pathlib import Path

    p = Path(repo)
    if p.is_file():
        return p
    if p.is_dir():
        if not (p / "checkpoint.pt").exists():
            raise FileNotFoundError(f"không có checkpoint.pt trong {p}")
        return p / "checkpoint.pt"
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, "checkpoint.pt"))


def load_checkpoint(path, allow_pickle: bool = False) -> dict:
    """Nạp checkpoint Rev. Mặc định ``weights_only=True`` (không chạy pickle tuỳ ý); serve_local.py dùng False.
    Chỉ đặt ``allow_pickle=True`` nếu tin nguồn file."""
    import torch

    try:
        ck = torch.load(path, map_location="cpu", weights_only=not allow_pickle)
    except Exception as e:
        if allow_pickle:
            raise
        raise RuntimeError(f"không nạp được checkpoint với weights_only=True ({type(e).__name__}: {str(e)[:200]}). "
                           "Kiểm tra nguồn rồi gọi lại với allow_pickle=True.") from e
    for key in ("adapters", "headq", "headk", "metadata"):
        if key not in ck:
            raise KeyError(f"checkpoint thiếu '{key}'")
    return ck


def load_base(name: str, revision: str | None = None, dtype: str = "bf16", device: str = "cuda",
              load_in_4bit: bool = False):
    """Base Qwen3.5/3.8 (mô hình thị giác–ngôn ngữ) -> (tokenizer, backbone ngôn ngữ), như serve_local.py.
    ``load_in_4bit``: NF4 qua bitsandbytes, tính toán bf16 (cho L4/T4 hoặc 9B trên GPU 24 GB)."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    revision = revision or BASE_REVISIONS.get(name)
    tdt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    kw = {"revision": revision, "dtype": tdt, "attn_implementation": "sdpa"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                        bnb_4bit_compute_dtype=torch.bfloat16)
        kw["device_map"] = {"": 0}
    else:
        kw["device_map"] = device
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    full = AutoModelForImageTextToText.from_pretrained(name, **kw).eval()
    backbone = full.model.language_model
    del full
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    backbone.requires_grad_(False)
    return tok, backbone


def load_rev(repo: str = REV_REPOS["rev-4b"], base: str | None = None, revision: str | None = None, dtype: str = "bf16",
             device: str = "cuda", load_in_4bit: bool = False, allow_pickle: bool = False,
             temperature: float | None = None) -> TorchRevScorer:
    """Nạp một checkpoint Rev: base Qwen từ Hub + LoRA (gộp nếu bf16, gắn rời nếu 4-bit) + pointer head fp32.
    ``temperature``: ghi đè nhiệt độ trong metadata (vd. T fit lại trên validation)."""
    import torch

    born = time.perf_counter()
    path = resolve_checkpoint(repo)
    ck = load_checkpoint(path, allow_pickle)
    meta = dict(ck["metadata"])
    name = base or meta.get("model")
    if not name:
        raise ValueError("metadata checkpoint không có tên base model; truyền base=")
    revision = revision or meta.get("revision") or BASE_REVISIONS.get(name)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)  # như serve_local.py: tránh autotune lại mỗi shape mới
    tok, backbone = load_base(name, revision, dtype, device, load_in_4bit)
    if load_in_4bit:
        attach_lora(backbone, ck["adapters"], trainable=False)
        n = len(ck["adapters"])
    else:
        n = merge_lora(backbone, ck["adapters"])
    dev = next(backbone.parameters()).device
    hq, hk = make_heads(backbone.config.hidden_size, dev, state=(ck["headq"], ck["headk"]))
    T = float(temperature if temperature is not None else meta.get("temperature", 1.0))
    info = {"name": repo.rstrip("/").split("/")[-1], "repo": repo, "model": name, "revision": revision,
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "state_format": meta.get("state_format", "json"),
            "temperature": T, "temperature_metadata": float(meta.get("temperature", 1.0)), "dtype": dtype,
            "load_in_4bit": load_in_4bit, "merged_adapters": 0 if load_in_4bit else n, "attached_adapters": n if load_in_4bit else 0,
            "load_seconds": time.perf_counter() - born, "training_examples": ck.get("examples")}
    return TorchRevScorer(tok, backbone, hq, hk, info)


# =========================================================================== adapter cùng giao diện Jev/WaterSheep


class RevAdapter:
    """Trả lời câu hỏi có kiểu (noul / choice / score) bằng một scorer Rev, xuất cùng định dạng response như Jev.

    ``ask_many`` gom mọi (tin × câu hỏi) thành các hàng và chấm theo batch (``batch_size`` hàng/forward).
    ``temperature``: nhiệt độ dùng để ra xác suất (mặc định của scorer); ``logits`` thô được giữ trong ``last_logits``."""

    def __init__(self, scorer, name: str | None = None, batch_size: int = 32, temperature: float | None = None):
        self.scorer = scorer
        self.name = name or getattr(scorer, "name", "rev")
        self.batch_size = batch_size
        self.temperature = float(temperature if temperature is not None else getattr(scorer, "temperature", 1.0))
        self.usage = {"calls": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0, "latency_s": [], "rows": 0,
                      "wall_s": 0.0}

    @classmethod
    def dry_run(cls, seed: int = 0, **kw) -> "RevAdapter":
        return cls(DryRunScorer(seed), name=kw.pop("name", "rev-dry-run"), **kw)

    def _rows(self, state, questions: dict):
        enc = []
        for qname, q in questions.items():
            ids, pos, keys = encode_prompt(self.scorer.encode, state, q.get("instructions") or "", to_rev_options(q),
                                           self.scorer.state_format)
            if len(ids) > MAX_ROW_TOKENS:
                raise ValueError(f"hàng {len(ids)} token vượt {MAX_ROW_TOKENS}")
            enc.append((qname, ids, pos, keys))
        return enc

    def score_rows(self, rows: list) -> list:
        """rows: [(ids, positions)] -> logit thô (chưa chia nhiệt độ)."""
        return self.scorer.logits(rows, batch_size=self.batch_size)

    def ask_many(self, states: list, questions: dict, progress: bool = True) -> list[dict]:
        t0 = time.perf_counter()
        enc = [self._rows(s, questions) for s in states]
        flat = [(ids, pos) for e in enc for _, ids, pos, _ in e]
        z = self.score_rows(flat)
        if getattr(self.scorer, "device_type", "cpu") == "cuda":
            import torch
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        out, j = [], 0
        for e in enc:
            answers = {}
            for qname, ids, pos, keys in e:
                p = softmax(z[j], self.temperature)
                answers[qname] = to_jev_answer(questions[qname], dict(zip(keys, p.tolist())))
                answers[qname]["logits"] = dict(zip(keys, map(float, z[j])))
                j += 1
            out.append({"model": self.name, "answers": answers, "usage": {"input_tokens": sum(len(i) for _, i, _, _ in e)}})
        self.usage["calls"] += len(states)
        self.usage["rows"] += len(flat)
        self.usage["input_tokens"] += sum(len(r[0]) for r in flat)
        self.usage["wall_s"] += dt
        self.usage["latency_s"].extend([dt / max(len(states), 1)] * len(states))
        if progress:
            print(f"  {self.name}: {len(states)} state × {len(questions)} câu hỏi, {1000 * dt / max(len(states), 1):.1f} ms/state",
                  flush=True)
        return out

    def ask(self, state, questions: dict) -> dict:
        return self.ask_many([state], questions, progress=False)[0]

    def cost_report(self) -> dict:
        lat = np.array(self.usage["latency_s"]) if self.usage["latency_s"] else np.array([np.nan])
        return {"calls": self.usage["calls"], "cache_hits": 0, "input_tokens": self.usage["input_tokens"],
                "est_cost_usd": 0.0, "latency_p50_ms": float(np.nanmedian(lat) * 1000),
                "latency_p95_ms": float(np.nanpercentile(lat, 95) * 1000),
                "ms_per_state_batched": 1000 * self.usage["wall_s"] / max(self.usage["calls"], 1),
                "peak_vram_gb": float(self.scorer.peak_memory_gb()) if hasattr(self.scorer, "peak_memory_gb") else np.nan}
