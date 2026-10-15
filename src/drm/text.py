"""Kiểm duyệt văn bản cho CS e-commerce: toxic / hate / offensive / spam.

Dữ liệu công khai tiếng Anh (tải từ GitHub raw, cache ở ``data/raw``):

| key                 | nguồn                                   | positive                    |
|---------------------|-----------------------------------------|-----------------------------|
| jigsaw_toxic        | Jigsaw Toxic Comment (Wikipedia talk)   | bất kỳ nhãn toxic nào       |
| tweeteval_offensive | TweetEval / OffensEval-2019 (OLID)      | offensive                   |
| tweeteval_hate      | TweetEval / HatEval-2019                | hate (nhập cư, phụ nữ)      |
| davidson_hate       | Davidson et al. 2017                    | hate (phân biệt với chửi tục)|
| hatexplain_toxic    | HateXplain (Mathew et al. 2021)         | hatespeech + offensive      |
| sms_spam            | SMS Spam Collection (Almeida et al.)    | spam                        |

``load_internal_csv`` cho phép cắm dữ liệu chat CS nội bộ vào cùng pipeline.
"""
from __future__ import annotations

import io
import json
import pickle
import re
import time
import urllib.request
import warnings
import zlib
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split

from .data import RAW_DIR

TE = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/"
URLS = {
    "jigsaw": "https://raw.githubusercontent.com/ipcplusplus/toxic-comments-classification/master/train.csv",
    "davidson": "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv",
    "hatexplain": "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json",
    "hatexplain_split": "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/post_id_divisions.json",
    "sms": "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
}


@dataclass
class TextTask:
    name: str
    X_train: pd.Series
    y_train: np.ndarray
    X_test: pd.Series
    y_test: np.ndarray
    category: str        # toxic | hate | offensive | spam
    question: str        # câu hỏi cho LLM
    domain: str
    labels: tuple = ("no", "yes")

    def summary(self):
        return {"task": self.name, "category": self.category, "domain": self.domain,
                "n_train": len(self.y_train), "n_test": len(self.y_test),
                "pos_rate_train": round(float(np.mean(self.y_train)), 3),
                "pos_rate_test": round(float(np.mean(self.y_test)), 3),
                "median_chars": int(pd.concat([self.X_train, self.X_test]).str.len().median())}


def _get(key: str, name: str | None = None, url: str | None = None) -> bytes:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / (name or f"{key}.dat")
    if not path.exists():
        with urllib.request.urlopen(url or URLS[key], timeout=120) as r:
            path.write_bytes(r.read())
    return path.read_bytes()


def _split(name, X, y, category, question, domain, test_size=0.2, seed=0, max_rows=None):
    X = pd.Series(X).reset_index(drop=True)
    y = np.asarray(y).astype(int)
    if max_rows and len(X) > max_rows:
        X, _, y, _ = train_test_split(X, y, train_size=max_rows, stratify=y, random_state=seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    return TextTask(name, Xtr.reset_index(drop=True), ytr, Xte.reset_index(drop=True), yte,
                    category, question, domain)


def load_jigsaw_toxic(max_rows=30000, seed=0) -> TextTask:
    df = pd.read_csv(io.BytesIO(_get("jigsaw", "jigsaw_train.csv")))
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = (df[labels].sum(1) > 0).astype(int)
    return _split("jigsaw_toxic", df["comment_text"], y, "toxic",
                  "Is this message toxic (rude, disrespectful, insulting, threatening, obscene or hateful)?",
                  "Wikipedia talk comments", max_rows=max_rows, seed=seed)


def _tweeteval(task: str) -> tuple:
    out = {}
    for split in ["train", "val", "test"]:
        txt = _get(f"te_{task}_{split}_text", f"tweeteval_{task}_{split}_text.txt",
                   f"{TE}{task}/{split}_text.txt").decode("utf-8").rstrip("\n").split("\n")
        lab = _get(f"te_{task}_{split}_labels", f"tweeteval_{task}_{split}_labels.txt",
                   f"{TE}{task}/{split}_labels.txt").decode("utf-8").rstrip("\n").split("\n")
        assert len(txt) == len(lab), (task, split, len(txt), len(lab))
        out[split] = (pd.Series(txt), np.array([int(l) for l in lab]))
    Xtr = pd.concat([out["train"][0], out["val"][0]], ignore_index=True)
    ytr = np.concatenate([out["train"][1], out["val"][1]])
    return Xtr, ytr, out["test"][0], out["test"][1]


def load_tweeteval_offensive() -> TextTask:
    Xtr, ytr, Xte, yte = _tweeteval("offensive")
    return TextTask("tweeteval_offensive", Xtr, ytr, Xte, yte, "offensive",
                    "Is this message offensive (insulting, profane, or targeted abuse)?", "Twitter (OLID)")


def load_tweeteval_hate() -> TextTask:
    Xtr, ytr, Xte, yte = _tweeteval("hate")
    return TextTask("tweeteval_hate", Xtr, ytr, Xte, yte, "hate",
                    "Is this message hate speech against a group (e.g. immigrants or women)?", "Twitter (HatEval)")


def load_davidson_hate(seed=0) -> TextTask:
    df = pd.read_csv(io.BytesIO(_get("davidson", "davidson_labeled_data.csv")))
    # class: 0 = hate speech, 1 = offensive, 2 = neither -> phân biệt "thù ghét" với "chửi tục"
    return _split("davidson_hate", df["tweet"], (df["class"] == 0).astype(int), "hate",
                  "Is this message hate speech (attacking people for their race, religion, gender, "
                  "sexual orientation, etc.), as opposed to merely profane or neutral?", "Twitter", seed=seed)


def load_hatexplain_toxic() -> TextTask:
    data = json.loads(_get("hatexplain", "hatexplain_dataset.json"))
    div = json.loads(_get("hatexplain_split", "hatexplain_divisions.json"))
    rows = {}
    for pid, d in data.items():
        votes = Counter(a["label"] for a in d["annotators"])
        lab, c = votes.most_common(1)[0]
        if c < 2:  # không có đa số -> bỏ (theo cách làm của bài báo gốc)
            continue
        rows[pid] = (" ".join(d["post_tokens"]), int(lab in ("hatespeech", "offensive")))
    tr = [rows[p] for p in div["train"] + div["val"] if p in rows]
    te = [rows[p] for p in div["test"] if p in rows]
    return TextTask("hatexplain_toxic", pd.Series([t for t, _ in tr]), np.array([l for _, l in tr]),
                    pd.Series([t for t, _ in te]), np.array([l for _, l in te]), "toxic",
                    "Is this message hateful or offensive?", "Twitter + Gab")


def load_sms_spam(seed=0) -> TextTask:
    df = pd.read_csv(io.BytesIO(_get("sms", "sms_spam.tsv")), sep="\t", header=None, names=["label", "text"])
    return _split("sms_spam", df["text"], (df["label"] == "spam").astype(int), "spam",
                  "Is this message spam (unsolicited advertising, scam, phishing or prize/lottery bait)?",
                  "SMS", seed=seed)


def load_davidson_abusive(seed=0) -> TextTask:
    """Davidson với định nghĩa sát nhu cầu CS: (hate + offensive) so với bình thường.
    Đồng thuận người gán nhãn cao hơn nhiều (Fleiss κ ≈ 0.73, so với ≈ 0.25 của hate so với phần còn lại)."""
    df = pd.read_csv(io.BytesIO(_get("davidson", "davidson_labeled_data.csv")))
    return _split("davidson_abusive", df["tweet"], (df["class"] != 2).astype(int), "offensive",
                  "Is this message abusive (hateful, offensive or profane), as opposed to neutral?", "Twitter", seed=seed)


def load_sms_spam_dedup(seed=0) -> TextTask:
    """SMS Spam Collection sau khi bỏ tin trùng (414 dòng) – tránh cùng một tin nằm ở cả train lẫn test."""
    df = pd.read_csv(io.BytesIO(_get("sms", "sms_spam.tsv")), sep="\t", header=None, names=["label", "text"])
    df = df.drop_duplicates(subset="text")
    return _split("sms_spam_dedup", df["text"], (df["label"] == "spam").astype(int), "spam",
                  "Is this message spam (unsolicited advertising, scam, phishing or prize/lottery bait)?",
                  "SMS", seed=seed)


def load_internal_csv(path, text_col="text", label_col="label", name="internal", category="toxic",
                      question="Is this customer message toxic or abusive?", test_size=0.2, seed=0) -> TextTask:
    """Cắm dữ liệu nội bộ (vd. log chat CS đã gán nhãn 0/1) vào cùng pipeline."""
    df = pd.read_csv(path)
    return _split(name, df[text_col].astype(str), df[label_col].astype(int), category, question,
                  "internal CS chat", test_size=test_size, seed=seed)


TASKS = {
    "jigsaw_toxic": load_jigsaw_toxic,
    "tweeteval_offensive": load_tweeteval_offensive,
    "tweeteval_hate": load_tweeteval_hate,
    "davidson_hate": load_davidson_hate,
    "hatexplain_toxic": load_hatexplain_toxic,
    "sms_spam": load_sms_spam,
}


# biến thể bổ sung (không nằm trong 6 task mặc định của notebook 01–05)
EXTRA_TASKS = {
    "davidson_abusive": load_davidson_abusive,
    "sms_spam_dedup": load_sms_spam_dedup,
}


def load_tasks(names=None) -> dict[str, TextTask]:
    reg = {**TASKS, **EXTRA_TASKS}
    return {n: reg[n]() for n in (names or TASKS)}


# --------------------------------------------------------------------------- obfuscation (né bộ lọc)

LEET = {"a": "@", "e": "3", "i": "1", "o": "0", "s": "$", "t": "7"}


def obfuscate(text: str, p: float = 0.3, seed: int = 0) -> str:
    """Mô phỏng cách người dùng né bộ lọc: leetspeak, chèn ký tự giữa các chữ, kéo dài, tách chữ.

    Mỗi từ (≥ 3 ký tự chữ) bị biến đổi với xác suất p bằng một trong 4 kiểu.
    (Hosseini et al., 2017 cho thấy các thay đổi nhỏ như vậy đủ đánh lừa Perspective API.)
    """
    rng = np.random.default_rng(seed + zlib.crc32(text.encode("utf-8")))

    def tf(w):
        if len(w) < 3 or not w.isalpha() or rng.random() > p:
            return w
        k = rng.integers(4)
        if k == 0:
            return "".join(LEET.get(c.lower(), c) if rng.random() < 0.6 else c for c in w)
        if k == 1:
            return rng.choice([".", "*", "-", "_"]).join(w)
        if k == 2:
            i = int(rng.integers(len(w)))
            return w[:i] + w[i] * int(rng.integers(2, 4)) + w[i + 1:]
        i = int(rng.integers(1, len(w)))
        return w[:i] + " " + w[i:]

    return " ".join(tf(w) for w in text.split(" "))


_REV_LEET = {v: k for k, v in LEET.items()}


def deobfuscate(text: str) -> str:
    """Chuẩn hoá ngược (cải tiến đề xuất): bỏ ký tự chèn, đảo leetspeak, gộp ký tự lặp."""
    t = re.sub(r"(?:[A-Za-z0-9@$][.*\-_]){2,}[A-Za-z0-9@$]",            # f.u.c.k -> fuck
               lambda m: re.sub(r"[.*\-_]", "", m.group(0)), text)

    def fix(tok):
        if re.search(r"[A-Za-z]", tok) and re.search(r"[@310$7]", tok):   # h3ll0 -> hello
            return "".join(_REV_LEET.get(c, c) for c in tok)
        return tok

    t = " ".join(fix(w) for w in t.split(" "))
    return re.sub(r"([A-Za-z])\1{2,}", r"\1", t)                         # fuuuck -> fuck


# --------------------------------------------------------------------------- features & models


def make_vectorizer(kind="word+char", max_word=30000, max_char=50000):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion

    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_word, sublinear_tf=True,
                           lowercase=True, strip_accents="unicode")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=max_char,
                           sublinear_tf=True, lowercase=True)
    if kind == "word":
        return word
    if kind == "char":
        return char
    return FeatureUnion([("word", word), ("char", char)])


def make_text_models(seed=0, n_jobs=1, include=None):
    from lightgbm import LGBMClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.naive_bayes import ComplementNB
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.pipeline import make_pipeline

    from .benchmark import PrunedTree

    def sel(model, k=5000):  # cây trên ~80k đặc trưng thưa rất chậm -> chọn top-k theo chi²
        return make_pipeline(SelectKBest(chi2, k=k), model)

    m = {
        "NaiveBayes": ComplementNB(alpha=0.3),
        "LogReg": LogisticRegression(C=4.0, max_iter=3000, solver="liblinear"),
        "LinearSVM": CalibratedClassifierCV(LinearSVC(C=0.5), method="sigmoid", cv=3),
        "SGD-online": SGDClassifier(loss="modified_huber", alpha=2e-5, random_state=seed),
        "DT-CART": sel(DecisionTreeClassifier(min_samples_leaf=2, random_state=seed)),
        "DT-pruned": sel(PrunedTree(n_alphas=8, cv=3, random_state=seed)),
        "RandomForest": sel(RandomForestClassifier(n_estimators=200, n_jobs=n_jobs, random_state=seed)),
        "ExtraTrees": sel(ExtraTreesClassifier(n_estimators=200, n_jobs=n_jobs, random_state=seed)),
        "LightGBM": LGBMClassifier(n_estimators=400, learning_rate=0.1, num_leaves=63, colsample_bytree=0.3,
                                   n_jobs=n_jobs, random_state=seed, verbose=-1),
        "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.2, max_depth=8, tree_method="hist",
                                 colsample_bytree=0.3, n_jobs=n_jobs, random_state=seed, verbosity=0),
        "Stacking": StackingClassifier(
            estimators=[("lr", LogisticRegression(C=4.0, max_iter=3000, solver="liblinear")),
                        ("nb", ComplementNB(alpha=0.3)),
                        ("lgbm", LGBMClassifier(n_estimators=300, colsample_bytree=0.3, n_jobs=n_jobs,
                                                random_state=seed, verbose=-1))],
            final_estimator=LogisticRegression(max_iter=2000), cv=3, n_jobs=1),
    }
    if include:
        m = {k: v for k, v in m.items() if k in include}
    return m


TEXT_INTERPRETABILITY = {"NaiveBayes": 4, "LogReg": 4, "LinearSVM": 3, "SGD-online": 3, "DT-CART": 4,
                         "DT-pruned": 5, "RandomForest": 2, "ExtraTrees": 2, "LightGBM": 2, "XGBoost": 2,
                         "Stacking": 1}


def _proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def recall_at_fpr(y, s, fpr_max=0.01):
    """Tỉ lệ nội dung xấu bắt được khi chỉ chấp nhận gắn cờ nhầm ≤ fpr_max tin sạch."""
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= fpr_max].max()) if (fpr <= fpr_max).any() else 0.0


def best_f1_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p * r / np.maximum(p + r, 1e-12)
    i = int(np.nanargmax(f[:-1]))
    return float(t[i])


def metrics(y, s, thr):
    pred = (s >= thr).astype(int)
    return {"pr_auc": average_precision_score(y, s), "roc_auc": roc_auc_score(y, s),
            "f1": f1_score(y, pred), "recall@1%fpr": recall_at_fpr(y, s, 0.01),
            "recall@5%fpr": recall_at_fpr(y, s, 0.05),
            "precision": float(((pred == 1) & (y == 1)).sum() / max(pred.sum(), 1)),
            "recall": float(((pred == 1) & (y == 1)).sum() / max(y.sum(), 1))}


def fit_eval(task: TextTask, name: str, model, vec_kind="word+char", seed=0, obf_p=(0.3,),
             return_scores=False):
    """Fit vectorizer + model trên train (90%), chọn ngưỡng F1 trên validation (10%), đánh giá test.

    Đồng thời đo độ bền trước obfuscation và hiệu quả của bước deobfuscate.
    """
    Xtr, Xva, ytr, yva = train_test_split(task.X_train, task.y_train, test_size=0.1,
                                          stratify=task.y_train, random_state=seed)
    vec = make_vectorizer(vec_kind)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        Vtr = vec.fit_transform(Xtr).astype(np.float32)
        mdl = clone(model).fit(Vtr, ytr)
        fit_t = time.perf_counter() - t0
        thr = best_f1_threshold(yva, _proba(mdl, vec.transform(Xva).astype(np.float32)))
        t0 = time.perf_counter()
        s = _proba(mdl, vec.transform(task.X_test).astype(np.float32))
        pred_t = time.perf_counter() - t0
    rec = {"task": task.name, "model": name, "features": vec_kind, **metrics(task.y_test, s, thr),
           "threshold": thr, "fit_time_s": fit_t, "ms_per_1k_msgs": 1000 * pred_t / len(s) * 1000,
           "model_size_mb": len(pickle.dumps((vec, mdl))) / 1e6}
    scores = {"clean": s}
    for p in obf_p:
        Xo = task.X_test.map(lambda t: obfuscate(t, p))
        so = _proba(mdl, vec.transform(Xo).astype(np.float32))
        sd = _proba(mdl, vec.transform(Xo.map(deobfuscate)).astype(np.float32))
        mo, md = metrics(task.y_test, so, thr), metrics(task.y_test, sd, thr)
        rec[f"pr_auc_obf{p}"] = mo["pr_auc"]
        rec[f"recall_obf{p}"] = mo["recall"]
        rec[f"pr_auc_obf{p}_deobf"] = md["pr_auc"]
        rec[f"recall_obf{p}_deobf"] = md["recall"]
        scores[f"obf{p}"] = so
    return (rec, scores, (vec, mdl)) if return_scores else rec


def paired_bootstrap(y, s_a, s_b, metric=average_precision_score, n=1000, seed=0):
    """Khoảng tin cậy 95% của metric(a) - metric(b) trên cùng tập test (bootstrap theo mẫu)."""
    rng = np.random.default_rng(seed)
    d = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if y[b].min() == y[b].max():
            continue
        d.append(metric(y[b], s_a[b]) - metric(y[b], s_b[b]))
    d = np.array(d)
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
