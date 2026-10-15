"""Mạng nơ-ron (ANN) phân loại tin nhắn, huấn luyện từ đầu bằng PyTorch, chạy được trên CPU.

Không dùng mô hình tiền huấn luyện (môi trường nghiên cứu không tải được Hugging Face), nên đây là
nhóm ANN "cổ điển" để so với TF-IDF + mô hình tuyến tính:

* ``MLP (TF-IDF)``: đầu vào TF-IDF từ + ký tự (cùng vectorizer với notebook 01), 1 lớp ẩn.
* ``fastText-style``: trung bình embedding của n-gram từ (1–2) và ký tự (3–5) đã băm (Joulin et al., 2017).
* ``TextCNN``: embedding từ + tích chập cửa sổ 3/4/5 + max-pooling (Kim, 2014).
* ``BiGRU``: embedding từ + GRU hai chiều + gộp max/mean.

Transformer tiền huấn luyện (DeBERTa, RoBERTa, ModernBERT...) được fine-tune trong notebook 06 trên Colab GPU.
"""
from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from torch import nn

TOKEN = re.compile(r"[a-z0-9']+|[^\sa-z0-9]")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(str(text).lower())


class Vocab:
    PAD, UNK = 0, 1

    def __init__(self, texts, max_size=30000, min_freq=2):
        c = Counter(t for x in texts for t in tokenize(x))
        words = [w for w, n in c.most_common(max_size) if n >= min_freq]
        self.index = {w: i + 2 for i, w in enumerate(words)}

    def __len__(self):
        return len(self.index) + 2

    def encode(self, texts, max_len=200) -> list[list[int]]:
        return [[self.index.get(t, self.UNK) for t in tokenize(x)[:max_len]] or [self.UNK] for x in texts]


def _pad(seqs) -> torch.Tensor:
    L = max(len(s) for s in seqs)
    out = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s)
    return out


def _bag(csr) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CSR (scipy) -> (chỉ số, offsets, trọng số) cho nn.EmbeddingBag."""
    csr = csr.tocsr()
    return (torch.from_numpy(csr.indices.astype(np.int64)), torch.from_numpy(csr.indptr[:-1].astype(np.int64)),
            torch.from_numpy(csr.data.astype(np.float32)))


class SparseMLP(nn.Module):
    """Lớp đầu tiên là EmbeddingBag(mode='sum') có trọng số = phép nhân ma trận thưa X @ W, rồi MLP."""

    def __init__(self, n_features, hidden=256, dropout=0.3, mode="sum", out_hidden=True):
        super().__init__()
        self.bag = nn.EmbeddingBag(n_features, hidden, mode=mode)
        self.head = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)) if out_hidden \
            else nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, batch):
        idx, off, w = batch
        w = w if self.bag.mode == "sum" else None  # mode="mean": đặc trưng nhị phân, không cần trọng số
        return self.head(self.bag(idx, off, per_sample_weights=w)).squeeze(-1)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb=128, filters=100, windows=(3, 4, 5), dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(emb, filters, k, padding=k // 2) for k in windows)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(filters * len(windows), 1))

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        mask = (x != 0).unsqueeze(1)
        h = [torch.relu(c(e))[..., :x.shape[1]].masked_fill(~mask, -1e4).max(-1).values for c in self.convs]
        return self.head(torch.cat(h, -1)).squeeze(-1)


class BiGRU(nn.Module):
    def __init__(self, vocab_size, emb=128, hidden=96, dropout=0.4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(emb, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(4 * hidden, 1)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)
        h, _ = self.gru(self.drop(self.emb(x)))
        mx = h.masked_fill(~mask, -1e4).max(1).values
        mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(self.drop(torch.cat([mx, mean], -1))).squeeze(-1)


@dataclass
class NNResult:
    name: str
    p_val: np.ndarray
    y_val: np.ndarray
    p_test: np.ndarray
    fit_time_s: float
    epochs: int
    n_params: int


def _batches(n, bs, rng=None):
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, bs):
        yield order[i:i + bs]


def _fit(model, make_batch, y_tr, n_tr, make_val, y_val, lr=2e-3, epochs=8, patience=2, bs=64, seed=0,
         weight_decay=1e-4):
    """Huấn luyện BCE, dừng sớm theo PR-AUC trên validation, giữ trọng số tốt nhất."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    yt = torch.tensor(y_tr, dtype=torch.float32)
    best, best_state, bad, ep = -1.0, None, 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        for b in _batches(n_tr, bs, rng):
            opt.zero_grad()
            loss = loss_fn(model(make_batch(b)), yt[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        p = _predict(model, make_val, len(y_val))
        score = average_precision_score(y_val, p)
        if score > best + 1e-4:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return ep


@torch.no_grad()
def _predict(model, make_batch, n, bs=256):
    model.eval()
    return np.concatenate([torch.sigmoid(model(make_batch(b))).numpy() for b in _batches(n, bs)])


NN_MODELS = ("MLP (TF-IDF)", "fastText-style", "TextCNN", "BiGRU")


def fit_nn(task, name: str, seed=0, val_size=0.1, max_len=200, threads=None) -> NNResult:
    """Huấn luyện một ANN trên ``task.X_train`` (giữ 10% làm validation, chia cùng cách với ``text.fit_eval``)."""
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.pipeline import FeatureUnion

    from .text import make_vectorizer

    if threads:
        torch.set_num_threads(threads)
    Xtr, Xva, ytr, yva = train_test_split(task.X_train, task.y_train, test_size=val_size,
                                          stratify=task.y_train, random_state=seed)
    Xtr, Xva, Xte = list(Xtr), list(Xva), list(task.X_test)
    ytr, yva = np.asarray(ytr), np.asarray(yva)
    t0 = time.perf_counter()
    if name in ("MLP (TF-IDF)", "fastText-style"):
        if name == "MLP (TF-IDF)":
            vec = make_vectorizer("word+char")
            Mtr = vec.fit_transform(Xtr)
            model = SparseMLP(Mtr.shape[1], hidden=256, dropout=0.3)
            lr = 1e-3
        else:
            vec = FeatureUnion([
                ("word", HashingVectorizer(ngram_range=(1, 2), n_features=2 ** 17, alternate_sign=False, norm=None,
                                           binary=True, lowercase=True)),
                ("char", HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2 ** 17,
                                           alternate_sign=False, norm=None, binary=True, lowercase=True))])
            Mtr = vec.transform(Xtr)
            model = SparseMLP(Mtr.shape[1], hidden=64, dropout=0.2, mode="mean", out_hidden=False)
            lr = 5e-3
        Mva, Mte = vec.transform(Xva), vec.transform(Xte)

        def mk(M):
            return lambda b: _bag(M[b])
        ep = _fit(model, mk(Mtr), ytr, len(ytr), mk(Mva), yva, lr=lr, seed=seed)
        p_va, p_te = _predict(model, mk(Mva), len(yva)), _predict(model, mk(Mte), len(Xte))
    elif name in ("TextCNN", "BiGRU"):
        voc = Vocab(Xtr)
        Str, Sva, Ste = voc.encode(Xtr, max_len), voc.encode(Xva, max_len), voc.encode(Xte, max_len)
        model = TextCNN(len(voc)) if name == "TextCNN" else BiGRU(len(voc))

        def mk(S):
            return lambda b: _pad([S[i] for i in b])
        ep = _fit(model, mk(Str), ytr, len(ytr), mk(Sva), yva, lr=2e-3, seed=seed)
        p_va, p_te = _predict(model, mk(Sva), len(yva)), _predict(model, mk(Ste), len(Xte))
    else:
        raise ValueError(f"không có mô hình {name!r}; chọn trong {NN_MODELS}")
    return NNResult(name, p_va, yva, p_te, time.perf_counter() - t0, ep,
                    sum(p.numel() for p in model.parameters()))
