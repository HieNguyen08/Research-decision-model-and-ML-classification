"""Benchmark kiểm duyệt văn bản (notebook 01 nạp lại kết quả).

    PYTHONPATH=src python scripts/run_text_benchmark.py --jobs 4

Đầu ra (results/text/):
    benchmark.csv        – metric cho mỗi task x model (TF-IDF word+char)
    ablation.csv         – word / char / word+char cho LogReg & LinearSVM
    cross_domain.csv     – huấn luyện trên task A, kiểm tra trên task B
    scores_<task>.npz    – điểm dự đoán trên test (dùng cho bootstrap & notebook 04)
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from drm import text as T

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--tasks", nargs="*", default=None)
ap.add_argument("--models", nargs="*", default=None)
ap.add_argument("--jobs", type=int, default=4)
ap.add_argument("--out", default="results/text")
ap.add_argument("--skip-main", action="store_true")
a = ap.parse_args()
OUT = Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)

tasks = T.load_tasks(a.tasks)
models = T.make_text_models(n_jobs=1, include=a.models)


def run_task(task):
    import warnings
    warnings.filterwarnings("ignore")
    recs, scores = [], {"y_test": task.y_test}
    for name, m in models.items():
        rec, sc, _ = T.fit_eval(task, name, m, return_scores=True)
        recs.append(rec)
        for k, v in sc.items():
            scores[f"{name}|{k}"] = v
        print(f"[{task.name}] {name}: pr_auc={rec['pr_auc']:.3f}", flush=True)
    np.savez_compressed(OUT / f"scores_{task.name}.npz", **scores)
    return recs


def run_ablation(task):
    import warnings
    warnings.filterwarnings("ignore")
    m = T.make_text_models(include=["LogReg", "LinearSVM"])
    return [T.fit_eval(task, name, mdl, vec_kind=k, obf_p=(0.3,))
            for name, mdl in m.items() for k in ["word", "char", "word+char"]]


def run_cross(src, dst_names):
    import warnings
    from sklearn.base import clone
    warnings.filterwarnings("ignore")
    rows = []
    for name, mdl in T.make_text_models(include=["LogReg", "LinearSVM", "NaiveBayes"]).items():
        _, _, (vec, fitted) = T.fit_eval(tasks[src], name, mdl, return_scores=True, obf_p=())
        for dst in dst_names:
            d = tasks[dst]
            s = T._proba(fitted, vec.transform(d.X_test).astype(np.float32))
            met = T.metrics(d.y_test, s, 0.5)
            rows.append({"train_on": src, "test_on": dst, "model": name,
                         "pr_auc": met["pr_auc"], "roc_auc": met["roc_auc"], "recall@5%fpr": met["recall@5%fpr"]})
    return rows


if not a.skip_main:
    res = Parallel(n_jobs=a.jobs)(delayed(run_task)(t) for t in tasks.values())
    pd.DataFrame([r for rs in res for r in rs]).to_csv(OUT / "benchmark.csv", index=False)
    print("saved benchmark.csv")

abl = Parallel(n_jobs=a.jobs)(delayed(run_ablation)(t) for t in tasks.values())
pd.DataFrame([r for rs in abl for r in rs]).to_csv(OUT / "ablation.csv", index=False)
print("saved ablation.csv")

tox = [n for n in ["jigsaw_toxic", "tweeteval_offensive", "hatexplain_toxic"] if n in tasks]
cross = Parallel(n_jobs=a.jobs)(delayed(run_cross)(s, tox) for s in tox)
pd.DataFrame([r for rs in cross for r in rs]).to_csv(OUT / "cross_domain.csv", index=False)
print("saved cross_domain.csv")
