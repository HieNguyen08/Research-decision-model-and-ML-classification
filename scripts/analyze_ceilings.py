"""Mức trần thực tế của từng task kiểm duyệt: nhiễu nhãn, lệch phân phối, độ bất định của tập test nhỏ.

    PYTHONPATH=src python scripts/analyze_ceilings.py

Đầu ra (results/ceiling/):
    human_agreement.csv   – Davidson, HateXplain: một người gán nhãn so với đa số những người còn lại,
                            và PR-AUC "gán nhãn lại" của một mô hình hoàn hảo (biết đúng xác suất người gán nhãn)
    shift.csv             – PR-AUC kiểm định chéo trên train so với trên test; AUC phân biệt train/test
    subset_ci.csv         – khoảng tin cậy bootstrap của PR-AUC (TF-IDF LogReg) trên toàn bộ test và tập con 300 tin
    label_definition.csv  – Davidson: "hate" so với "hate hoặc offensive" (định nghĩa sát nhu cầu CS hơn)
    duplicates.csv        – tin test trùng (sau chuẩn hoá) với tin train: tỉ lệ, độ nhất quán nhãn, PR-AUC khi bỏ chúng
"""
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from drm import llm, text as T
from drm.data import RAW_DIR

warnings.filterwarnings("ignore")
OUT = Path("results/ceiling")
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)


def votes_davidson():
    d = pd.read_csv(RAW_DIR / "davidson_labeled_data.csv")
    return d["hate_speech"].to_numpy(), d["count"].to_numpy(), (d["class"] == 0).astype(int).to_numpy()


def votes_hatexplain():
    data = json.loads((RAW_DIR / "hatexplain_dataset.json").read_text())
    div = json.loads((RAW_DIR / "hatexplain_divisions.json").read_text())
    pos, n, y = [], [], []
    for pid in div["test"]:
        labs = [a["label"] for a in data[pid]["annotators"]]
        lab, c = Counter(labs).most_common(1)[0]
        if c < 2:
            continue
        k = sum(lab_ in ("hatespeech", "offensive") for lab_ in labs)
        pos.append(k), n.append(len(labs)), y.append(int(k * 2 > len(labs)))
    return np.array(pos), np.array(n), np.array(y)


def human_agreement(name, pos, n, y, n_sim=200):
    """(1) Một người gán nhãn ngẫu nhiên so với đa số của những người còn lại (bỏ các ca hoà).
    (2) Trần "gán nhãn lại": mô hình biết đúng p_i = P(một người gán nhãn nói 'dương'), chấm điểm bằng p_i,
        so với nhãn đa số của một nhóm người gán nhãn MỚI cùng số lượng (mô phỏng), giả định người gán nhãn
        hoán đổi được. p_i ước lượng bằng trung bình hậu nghiệm Beta (prior khớp moment trên cả dataset)."""
    tp = fp = fn = 0
    for _ in range(n_sim):
        held = rng.random(len(n)) < pos / n  # phiếu của người bị giữ lại
        rest_pos = pos - held
        rest_n = n - 1
        maj = rest_pos * 2 > rest_n
        tie = rest_pos * 2 == rest_n
        keep = ~tie
        tp += np.sum(held & maj & keep)
        fp += np.sum(held & ~maj & keep)
        fn += np.sum(~held & maj & keep)
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    f = pos / n
    m, v = f.mean(), f.var()
    common = m * (1 - m) / max(v, 1e-9) - 1
    a, b = m * common, (1 - m) * common
    p_hat = (pos + a) / (n + a + b)
    aps, aucs = [], []
    for _ in range(n_sim):
        p = rng.beta(pos + a, n - pos + b)  # hậu nghiệm của p_i
        fresh = rng.binomial(n, p) * 2 > n
        if fresh.min() == fresh.max():
            continue
        aps.append(average_precision_score(fresh, p_hat))
        aucs.append(roc_auc_score(fresh, p_hat))
    return {"task": name, "n_items": len(n), "pos_rate": y.mean(),
            "share_pos_items_unanimous": float(np.mean(pos[y == 1] == n[y == 1])),
            "share_neg_items_with_some_pos_vote": float(np.mean(pos[y == 0] > 0)),
            "human_precision": prec, "human_recall": rec, "human_f1": 2 * prec * rec / (prec + rec),
            "reannotation_pr_auc_ceiling": float(np.mean(aps)), "reannotation_pr_auc_p5": float(np.percentile(aps, 5)),
            "reannotation_roc_auc_ceiling": float(np.mean(aucs))}


rows = [human_agreement("davidson_hate (toàn bộ)", *votes_davidson()),
        human_agreement("hatexplain_toxic (test)", *votes_hatexplain())]
pd.DataFrame(rows).to_csv(OUT / "human_agreement.csv", index=False)
print(pd.DataFrame(rows).round(3).T.to_string())

tasks = T.load_tasks()
lr = LogisticRegression(C=4.0, max_iter=2000, class_weight=None)
shift = []
for name, task in tasks.items():
    vec = T.make_vectorizer()
    Vtr = vec.fit_transform(task.X_train).astype(np.float32)
    cv = cross_val_predict(lr, Vtr, task.y_train, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                           method="predict_proba", n_jobs=4)[:, 1]
    te = np.load(f"results/text/scores_{name}.npz")["LogReg|clean"]
    # adversarial validation: phân biệt được tin train với tin test đến mức nào (0.5 = cùng phân phối)
    Xall = pd.concat([task.X_train, task.X_test], ignore_index=True)
    dom = np.r_[np.zeros(len(task.X_train)), np.ones(len(task.X_test))]
    Vall = T.make_vectorizer().fit_transform(Xall).astype(np.float32)
    pd_ = cross_val_predict(LogisticRegression(max_iter=2000), Vall, dom, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                            method="predict_proba", n_jobs=4)[:, 1]
    shift.append({"task": name, "pr_auc_cv_train": average_precision_score(task.y_train, cv),
                  "pr_auc_test": average_precision_score(task.y_test, te),
                  "pos_rate_train": task.y_train.mean(), "pos_rate_test": task.y_test.mean(),
                  "train_vs_test_auc": roc_auc_score(dom, pd_)})
    print(shift[-1], flush=True)
pd.DataFrame(shift).to_csv(OUT / "shift.csv", index=False)

ci = []
for name, task in tasks.items():
    y, s = task.y_test, np.load(f"results/text/scores_{name}.npz")["LogReg|clean"]
    for label, idx in [("full", np.arange(len(y))), ("subset300", llm.text_test_subset(task, 300))]:
        bs = []
        for _ in range(1000):
            b = rng.choice(idx, len(idx), replace=True)
            if y[b].min() != y[b].max():
                bs.append(average_precision_score(y[b], s[b]))
        ci.append({"task": name, "eval_set": label, "n": len(idx), "n_pos": int(y[idx].sum()),
                   "pr_auc": average_precision_score(y[idx], s[idx]),
                   "ci95_low": np.percentile(bs, 2.5), "ci95_high": np.percentile(bs, 97.5)})
pd.DataFrame(ci).to_csv(OUT / "subset_ci.csv", index=False)
print(pd.DataFrame(ci).round(3).to_string())

d = pd.read_csv(RAW_DIR / "davidson_labeled_data.csv")
lab = []
for lname, yy in [("hate vs (offensive + neither)", (d["class"] == 0)), ("(hate + offensive) vs neither", (d["class"] != 2))]:
    t = T._split("davidson", d["tweet"], yy.astype(int), "hate", "", "Twitter")
    rec = T.fit_eval(t, "LogReg", T.make_text_models(include=["LogReg"])["LogReg"], obf_p=())
    lab.append({"definition": lname, "pos_rate": float(yy.mean()), "pr_auc": rec["pr_auc"], "roc_auc": rec["roc_auc"],
                "f1": rec["f1"]})
pd.DataFrame(lab).to_csv(OUT / "label_definition.csv", index=False)
print(pd.DataFrame(lab).round(3).to_string())

norm = lambda t: " ".join(str(t).lower().split())
dup = []
for name, task in tasks.items():
    tr = {}
    for x, yy in zip(task.X_train.map(norm), task.y_train):
        tr.setdefault(x, []).append(yy)
    te = task.X_test.map(norm)
    is_dup = te.isin(tr).to_numpy()
    same = [np.mean(tr[x]) == yy for x, yy in zip(te[is_dup], task.y_test[is_dup])]
    s_ = np.load(f"results/text/scores_{name}.npz")["LogReg|clean"]
    keep = ~is_dup
    dup.append({"task": name, "test_dup_share": float(is_dup.mean()), "n_dup": int(is_dup.sum()),
                "dup_label_consistent": float(np.mean(same)) if same else np.nan,
                "pr_auc_all": average_precision_score(task.y_test, s_),
                "pr_auc_without_dups": average_precision_score(task.y_test[keep], s_[keep])})
pd.DataFrame(dup).to_csv(OUT / "duplicates.csv", index=False)
print(pd.DataFrame(dup).round(3).to_string())
