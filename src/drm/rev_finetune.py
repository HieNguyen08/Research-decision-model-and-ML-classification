"""Fine-tune theo công thức Rev (LoRA + pointer head trên token lựa chọn) cho kiểm duyệt và định tuyến intent – notebook 07.

Công thức lấy từ ``train/train.py`` của Rev (MIT):
* LoRA hạng 16, alpha 32 trên q/k/v/o_proj và các phép chiếu DeltaNet; base đóng băng bf16; adapter + head fp32;
* pointer head 256 chiều (``rev.make_heads``), logit = <q, k> / 16;
* mỗi ví dụ được chấm ở **hai thứ tự lựa chọn** (gốc đã xáo + một thứ tự khác), loss = trung bình cross-entropy hai bản
  + 0.1 × KL đối xứng giữa hai phân phối đã căn theo khoá (nhất quán khi đổi thứ tự);
* AdamW: adapter lr 1e-4, head lr 5e-4, weight decay 0.01; lr nhân ``0.2 + 0.8 · cos`` theo tiến độ; clip grad 1.0.

Khác Rev ở chỗ:
* dữ liệu: task kiểm duyệt của notebook 06 (cùng loader, cùng cách chia train/validation ``finetune.split_train_val``
  seed 0, cùng tập test) và tuỳ chọn intent Bitext; **không** dùng dữ liệu train của Rev, **không** dùng output Jev;
* có đánh giá giữa chừng trên validation, giữ trạng thái tốt nhất theo metric validation (PR-AUC cho nhị phân);
  test chỉ được chấm **một lần** ở cuối với trạng thái đã chọn;
* nhiệt độ T fit trên validation (``rev.fit_temperature``) và lưu kèm;
* 8 ví dụ mỗi bước (Rev: 4), để chạy nhanh hơn trên A100 với tin ngắn;
* hàng dài hơn ``max_tokens`` (512) được **cắt bớt tin nhắn** thay vì bỏ hàng như Rev (Rev bỏ hàng > 4096 token);
  tin kiểm duyệt ngắn nên hầu như không bị cắt.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import rev as R

VERSION = "1"


@dataclass
class RevFTConfig:
    rank: int = R.LORA_RANK
    alpha: int = R.LORA_ALPHA
    adapter_lr: float = 1e-4
    head_lr: float = 5e-4
    weight_decay: float = 0.01
    epochs: float = 1.0
    batch_size: int = 8               # số ví dụ / bước (mỗi ví dụ = 2 thứ tự lựa chọn)
    micro_batch: int = 8              # ví dụ / lượt forward (giảm khi thiếu VRAM; tự giảm khi OOM)
    consistency_weight: float = 0.1
    permutation_pairs: bool = True
    max_tokens: int = 512             # cắt bớt state (tin nhắn) để hàng ≤ max_tokens
    eval_every: int = 200             # số bước giữa hai lần đánh giá validation
    patience: int = 4
    max_train: int | None = None      # lấy mẫu phân tầng tối đa n ví dụ train (None = toàn bộ)
    max_val: int | None = 2000        # validation dùng để chọn checkpoint giữa chừng (toàn bộ validation chấm ở cuối)
    seed: int = 814
    grad_checkpointing: bool = True
    eval_batch: int = 32


@dataclass
class RevFTResult:
    name: str
    p_val: np.ndarray                 # xác suất validation (nhị phân: P(yes); đa lớp: ma trận n × K), T = 1
    y_val: np.ndarray
    p_test: np.ndarray
    temperature: float = 1.0          # fit trên validation
    history: list = field(default_factory=list)
    fit_time_s: float = 0.0
    trainable_params: int = 0
    best_val_metric: float = 0.0
    ms_per_msg: float = 0.0
    peak_vram_gb: float = 0.0
    p_extra: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    keys: list = field(default_factory=list)


# --------------------------------------------------------------------------- dữ liệu -> hàng Rev

BINARY_OPTIONS = {
    "toxic": {"yes": "toxic: rude, insulting, threatening, obscene or hateful", "no": "not toxic: ordinary message"},
    "offensive": {"yes": "offensive, profane or abusive towards someone", "no": "not offensive"},
    "hate": {"yes": "attacks or demeans people for a protected attribute", "no": "no hate speech"},
    "spam": {"yes": "spam or scam: unsolicited ads, phishing, prize bait, moving payment off the platform",
             "no": "a normal personal or customer message"},
}


def binary_options(task) -> dict:
    return BINARY_OPTIONS.get(task.category, {"yes": "yes", "no": "no"})


def binary_rows(texts, labels, question: str, options: dict, ids=None) -> list[dict]:
    """Tin nhắn + nhãn 0/1 -> hàng {state, instructions, criteria, expected} theo định dạng dữ liệu của Rev."""
    labels = [None] * len(texts) if labels is None else list(labels)
    return [{"id": (ids[i] if ids is not None else i), "state": {"customer_message": str(t)}, "instructions": question,
             "criteria": dict(options), "expected": (None if y is None else ("yes" if int(y) else "no"))}
            for i, (t, y) in enumerate(zip(texts, labels))]


def _stratified_cap(y, n, seed):
    if n is None or len(y) <= n:
        return np.arange(len(y))
    from sklearn.model_selection import train_test_split
    idx, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=seed)
    return np.sort(idx)


# Bitext (bitext/Bitext-customer-support-llm-chatbot-training-dataset, CDLA-Sharing-1.0): 27 intent CS.
# Gom thành 3 hướng cho điểm quyết định #2 (chuyển sang bot tự phục vụ) của guard CS.
BITEXT_ROUTES = {
    "order_status": ["track_order", "delivery_period", "track_refund"],
    "policy_info": ["check_refund_policy", "check_cancellation_fee", "check_payment_methods", "delivery_options"],
}
ROUTE_OPTIONS = {
    "order_status": "the customer asks where an order or refund is, or when it will arrive: an order-tracking bot can answer",
    "policy_info": "the customer asks about a policy, fee, payment method or delivery option: an FAQ bot can answer",
    "human": "anything else (cancel or change an order, complaints, account problems, invoices, payment errors): needs a person",
}
ROUTE_QUESTION = "Where should this customer-support request be routed?"


def bitext_route(intent: str) -> str:
    for r, intents in BITEXT_ROUTES.items():
        if intent in intents:
            return r
    return "human"


def bitext_splits(df, seed: int = 0, val_size: float = 0.15, test_size: float = 0.15):
    """DataFrame Bitext (cột ``instruction``, ``intent``) -> (train, val, test) hàng Rev 3 hướng, chia phân tầng theo intent."""
    from sklearn.model_selection import train_test_split

    df = df.dropna(subset=["instruction", "intent"]).reset_index(drop=True)
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=test_size, stratify=df["intent"], random_state=seed)
    tr, va = train_test_split(tr, test_size=val_size / (1 - test_size), stratify=df["intent"].iloc[tr], random_state=seed)

    def rows(ix):
        return [{"id": int(i), "state": {"customer_message": str(df["instruction"].iloc[i])}, "instructions": ROUTE_QUESTION,
                 "criteria": dict(ROUTE_OPTIONS), "expected": bitext_route(df["intent"].iloc[i]),
                 "intent": df["intent"].iloc[i]} for i in ix]

    return rows(np.sort(tr)), rows(np.sort(va)), rows(np.sort(te))


# --------------------------------------------------------------------------- mã hoá

def encode_row(encode, row: dict, max_tokens: int, state_format: str = "raw", seed=None) -> dict:
    """Mã hoá một hàng; nếu dài quá ``max_tokens`` thì cắt bớt tin nhắn (theo token) rồi mã hoá lại."""
    ids, pos, keys = R.encode_prompt(encode, row["state"], row["instructions"], row["criteria"], state_format, seed)
    if len(ids) > max_tokens and isinstance(row["state"], dict) and "customer_message" in row["state"]:
        msg = row["state"]["customer_message"]
        over = len(ids) - max_tokens
        toks = encode(msg)
        keep = max(len(toks) - over - 8, 16)
        cut = msg[: max(16, int(len(msg) * keep / max(len(toks), 1)))]
        ids, pos, keys = R.encode_prompt(encode, {"customer_message": cut}, row["instructions"], row["criteria"],
                                         state_format, seed)
    label = keys.index(row["expected"]) if row.get("expected") is not None else -1
    return {"ids": ids, "positions": pos, "keys": keys, "label": label, "row": row}


def align_to(keys_from, keys_to) -> list:
    """Chỉ số để sắp lại vector theo ``keys_from`` thành thứ tự ``keys_to``."""
    return [keys_from.index(k) for k in keys_to]


# --------------------------------------------------------------------------- huấn luyện (torch)


def pair_loss_terms(z, z_other, label, label_other, keys, keys_other, consistency_weight):
    """Loss Rev cho một ví dụ: CE trung bình hai thứ tự + w · KL đối xứng (sau khi căn ``z_other`` theo ``keys``)."""
    import torch
    import torch.nn.functional as F

    ce = (F.cross_entropy(z[None], torch.tensor([label], device=z.device))
          + F.cross_entropy(z_other[None], torch.tensor([label_other], device=z.device))) / 2
    aligned = z_other[align_to(keys_other, keys)]
    lp, lq = z.log_softmax(-1), aligned.log_softmax(-1)
    kl = ((lp.exp() * (lp - lq)).sum() + (lq.exp() * (lq - lp)).sum()) / 2
    return ce + consistency_weight * kl


def train_rev(tok, backbone, train_rows: list, val_rows: list, test_sets: dict, cfg: RevFTConfig | None = None,
              select: str = "pr_auc", positive: str = "yes", name: str = "rev-ft", log=print,
              state_format: str = "raw") -> RevFTResult:
    """Huấn luyện LoRA + pointer head mới trên ``backbone`` (đã đóng băng) theo công thức Rev.

    ``val_rows``: chọn checkpoint (``select`` = "pr_auc" cho nhị phân với lớp dương ``positive``, hoặc "nll"/"accuracy")
    và fit nhiệt độ. ``test_sets``: {tên: hàng} chỉ chấm một lần ở cuối (``"test"`` là tập chính).
    Nhị phân: ``p_*`` là P(positive); đa lớp: ma trận xác suất theo thứ tự khoá của hàng đầu tiên."""
    import gc

    import torch
    from sklearn.metrics import average_precision_score

    cfg = cfg or RevFTConfig()
    device = next(backbone.parameters()).device
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    encode = lambda s: tok.encode(s, add_special_tokens=False)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    adapters = R.attach_lora(backbone, None, cfg.rank, cfg.alpha / cfg.rank, trainable=True)
    hq, hk = R.make_heads(backbone.config.hidden_size, device)
    if cfg.grad_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(backbone, "config"):
        backbone.config.use_cache = False
    ap = [p for a in adapters.values() for p in (a.a, a.b)]
    hp = list(hq.parameters()) + list(hk.parameters())
    params = ap + hp
    opt = torch.optim.AdamW([{"params": ap, "lr": cfg.adapter_lr}, {"params": hp, "lr": cfg.head_lr}],
                            weight_decay=cfg.weight_decay)

    keys0 = list(train_rows[0]["criteria"])
    tr = [encode_row(encode, r, cfg.max_tokens, state_format, seed=cfg.seed + i) for i, r in enumerate(train_rows)]
    va = [encode_row(encode, r, cfg.max_tokens, state_format) for r in val_rows]
    y_val_keys = [r["expected"] for r in val_rows]
    if cfg.max_val and len(va) > cfg.max_val:
        sel_idx = _stratified_cap(np.array(y_val_keys), cfg.max_val, cfg.seed)
    else:
        sel_idx = np.arange(len(va))

    def probs_of(encoded, bs=cfg.eval_batch):
        """Xác suất (T = 1) theo thứ tự ``keys0`` cho các hàng đã mã hoá."""
        backbone.eval()
        out = [None] * len(encoded)
        order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]["ids"]))
        with torch.no_grad():
            for s in range(0, len(order), bs):
                idx = order[s:s + bs]
                chunk = [(encoded[i]["ids"], encoded[i]["positions"]) for i in idx]
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    hidden = R.forward_rows(backbone, chunk, pad_id, device)
                for i, z in zip(idx, R.head_logits(hidden, chunk, hq, hk)):
                    p = z.float().softmax(-1).cpu().numpy().astype(np.float64)
                    out[i] = p[align_to(encoded[i]["keys"], keys0)]
        return np.stack(out)

    def metric(P, y_keys):
        y_idx = np.array([keys0.index(k) for k in y_keys])
        if select == "pr_auc":
            return float(average_precision_score((y_idx == keys0.index(positive)).astype(int), P[:, keys0.index(positive)]))
        if select == "accuracy":
            return float(np.mean(P.argmax(1) == y_idx))
        return -R.nll(list(P), y_idx)

    def loss_for(xs, ids_):
        views = []
        for x, i in zip(xs, ids_):
            views.append(x)
            if cfg.permutation_pairs:
                views.append(encode_row(encode, x["row"], cfg.max_tokens, state_format, seed=900000 + int(i)))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hidden = R.forward_rows(backbone, [(v["ids"], v["positions"]) for v in views], pad_id, device,
                                    pad_multiple=16, attention_mask=True)
        zs = R.head_logits(hidden, [(v["ids"], v["positions"]) for v in views], hq, hk)
        losses = []
        for j in range(len(xs)):
            if cfg.permutation_pairs:
                a, b = views[2 * j], views[2 * j + 1]
                losses.append(pair_loss_terms(zs[2 * j], zs[2 * j + 1], a["label"], b["label"], a["keys"], b["keys"],
                                              cfg.consistency_weight))
            else:
                losses.append(torch.nn.functional.cross_entropy(zs[j][None], torch.tensor([xs[j]["label"]], device=device)))
        return torch.stack(losses).mean()

    n = len(tr)
    steps_per_epoch = math.ceil(n / cfg.batch_size)
    total = max(1, int(steps_per_epoch * cfg.epochs))
    best, best_state, bad, step, hist = -np.inf, None, 0, 0, []
    micro = cfg.micro_batch
    t0 = time.perf_counter()
    done = False
    while not done:
        order = rng.permutation(n)
        # gom ví dụ gần độ dài trong cửa sổ 256 (như Rev) để giảm đệm, rồi xáo các nhóm
        groups = []
        for s in range(0, n, 256):
            w = sorted(order[s:s + 256], key=lambda i: len(tr[i]["ids"]))
            groups += [w[j:j + cfg.batch_size] for j in range(0, len(w), cfg.batch_size)]
        rng.shuffle(groups)
        for g in groups:
            backbone.train()
            while True:
                try:
                    for o in range(0, len(g), micro):
                        ids_ = g[o:o + micro]
                        loss = loss_for([tr[i] for i in ids_], ids_)
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f"loss = {loss.item()} ở bước {step}")
                        (loss * len(ids_) / len(g)).backward()
                    break
                except torch.cuda.OutOfMemoryError:
                    opt.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    if micro == 1:
                        raise
                    micro = max(1, micro // 2)
                    log(f"  !! hết VRAM -> micro_batch {micro}")
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            step += 1
            f = 0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * min(step / total, 1.0)))
            opt.param_groups[0]["lr"], opt.param_groups[1]["lr"] = cfg.adapter_lr * f, cfg.head_lr * f
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step % cfg.eval_every == 0 or step >= total:
                P = probs_of([va[i] for i in sel_idx])
                m = metric(P, [y_val_keys[i] for i in sel_idx])
                hist.append({"step": step, "epoch": round(step / steps_per_epoch, 3), f"val_{select}": m,
                             "loss": float(loss.detach()), "micro_batch": micro})
                log(f"  step {step}/{total} val {select} {m:.4f} loss {float(loss.detach()):.4f}")
                if m > best + 1e-4:
                    best, bad = m, 0
                    best_state = {"a": {k: (v.a.detach().cpu().clone(), v.b.detach().cpu().clone()) for k, v in adapters.items()},
                                  "hq": {k: v.detach().cpu().clone() for k, v in hq.state_dict().items()},
                                  "hk": {k: v.detach().cpu().clone() for k, v in hk.state_dict().items()}}
                else:
                    bad += 1
                if bad >= cfg.patience or step >= total:
                    done = True
                    break
    if best_state is not None:
        with torch.no_grad():
            for k, (a, b) in best_state["a"].items():
                adapters[k].a.copy_(a.to(device))
                adapters[k].b.copy_(b.to(device))
            hq.load_state_dict(best_state["hq"])
            hk.load_state_dict(best_state["hk"])
    fit_time = time.perf_counter() - t0

    P_val = probs_of(va)
    y_val_idx = np.array([keys0.index(k) for k in y_val_keys])
    T, _ = R.fit_temperature(list(P_val), y_val_idx)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    scored, ms = {}, 0.0
    for tname, rows in test_sets.items():
        enc = [encode_row(encode, r, cfg.max_tokens, state_format) for r in rows]
        t1 = time.perf_counter()
        scored[tname] = probs_of(enc)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if tname == "test":
            ms = 1000 * (time.perf_counter() - t1) / max(len(rows), 1)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
    binary = select == "pr_auc"
    col = (lambda P: P[:, keys0.index(positive)]) if binary else (lambda P: P)
    yv = (y_val_idx == keys0.index(positive)).astype(int) if binary else y_val_idx
    self_state = {"adapters": {k: {"a": v.a.detach().cpu(), "b": v.b.detach().cpu()} for k, v in adapters.items()},
                  "headq": hq.state_dict(), "headk": hk.state_dict()}
    res = RevFTResult(name, col(P_val), yv, col(scored.get("test", np.zeros((0, len(keys0))))), T, hist, fit_time,
                      sum(p.numel() for p in params), float(best), ms, peak,
                      {k: col(v) for k, v in scored.items() if k != "test"},
                      {**cfg.__dict__, "select": select, "positive": positive, "n_train": n, "steps": step}, keys0)
    res.state = self_state  # để lưu checkpoint cùng định dạng Rev nếu muốn (không lưu mặc định)
    return res


def finetune_task(task, tok, backbone, cfg: RevFTConfig | None = None, name: str = "rev-ft", extra_tests: dict | None = None,
                  log=print, split_seed: int = 0) -> RevFTResult:
    """Task kiểm duyệt nhị phân của notebook 06 -> hàng Rev -> ``train_rev``. Cùng cách chia train/validation với
    ``finetune.split_train_val`` (seed 0) nên p_val/p_test khớp thứ tự với file điểm của notebook 06."""
    from .finetune import split_train_val

    cfg = cfg or RevFTConfig()
    Xtr, Xva, ytr, yva = split_train_val(task, split_seed)
    keep = _stratified_cap(ytr, cfg.max_train, cfg.seed)
    opts = binary_options(task)
    tr = binary_rows([Xtr[i] for i in keep], ytr[keep], task.question, opts)
    va = binary_rows(Xva, yva, task.question, opts)
    tests = {"test": binary_rows(list(task.X_test), task.y_test, task.question, opts)}
    for k, v in (extra_tests or {}).items():
        tests[k] = binary_rows(list(v), None, task.question, opts)
    return train_rev(tok, backbone, tr, va, tests, cfg, select="pr_auc", positive="yes", name=name, log=log)


def detach_lora(backbone) -> int:
    """Gỡ các ``LoRALinear`` khỏi backbone (trả lại lớp Linear gốc) để huấn luyện task kế tiếp trên cùng base."""
    import torch

    n = 0
    for path, module in list(backbone.named_modules()):
        if hasattr(module, "base") and hasattr(module, "a") and hasattr(module, "b") and isinstance(module.base, torch.nn.Linear):
            parent, _, attr = path.rpartition(".")
            setattr(backbone.get_submodule(parent) if parent else backbone, attr, module.base)
            n += 1
    if hasattr(backbone, "gradient_checkpointing_disable"):
        backbone.gradient_checkpointing_disable()
    return n
