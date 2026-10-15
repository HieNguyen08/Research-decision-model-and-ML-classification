"""Benchmark ANN huấn luyện từ đầu (CPU) trên 6 task kiểm duyệt, so với TF-IDF + LogReg của notebook 01.

    PYTHONPATH=src python scripts/run_nn_benchmark.py --jobs 4

Đánh giá trên **toàn bộ** tập test (không phải tập con 300 tin như notebook 03/05) và ghi thêm PR-AUC trên
tập con 300 tin để đối chiếu. Đầu ra (results/nn/):
    benchmark.csv      – metric cho mỗi task x mô hình (gồm cả ghép với TF-IDF LogReg bằng trung bình hạng)
    scores_<task>.npz  – điểm dự đoán trên test của từng ANN
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

from drm import llm, nn_text as N, text as T

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--tasks", nargs="*", default=None)
ap.add_argument("--models", nargs="*", default=list(N.NN_MODELS))
ap.add_argument("--jobs", type=int, default=4)
ap.add_argument("--out", default="results/nn")
a = ap.parse_args()
OUT = Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)


def run(task_name, model):
    import warnings
    warnings.filterwarnings("ignore")
    task = T.load_tasks([task_name])[task_name]
    return task_name, N.fit_nn(task, model, threads=1)


tasks = T.load_tasks(a.tasks)
done = Parallel(n_jobs=a.jobs, verbose=5)(delayed(run)(t, m) for t in tasks for m in a.models)

rows = []
for name, task in tasks.items():
    y = task.y_test
    sub = llm.text_test_subset(task, 300)
    lr = np.load(f"results/text/scores_{name}.npz")["LogReg|clean"]
    res = {r.name: r for t, r in done if t == name}
    scores = {"y_test": y}

    def add(model, s, thr, extra=None):
        rows.append({"task": name, "model": model, **T.metrics(y, s, thr),
                     "pr_auc_subset300": float(average_precision_score(y[sub], s[sub])), **(extra or {})})

    add("TF-IDF LogReg (nb 01)", lr, 0.5)
    for m, r in res.items():
        scores[m] = r.p_test
        add(m, r.p_test, T.best_f1_threshold(r.y_val, r.p_val),
            {"fit_time_s": round(r.fit_time_s, 1), "epochs": r.epochs, "n_params": r.n_params})
        ens = 0.5 * rankdata(r.p_test) / len(y) + 0.5 * rankdata(lr) / len(y)
        add(f"{m} ⊕ TF-IDF LogReg", ens, np.quantile(ens, 1 - y.mean()))
    allr = np.mean([rankdata(r.p_test) for r in res.values()] + [rankdata(lr)], 0) / len(y)
    add("ANN×4 ⊕ TF-IDF LogReg", allr, np.quantile(allr, 1 - y.mean()))
    np.savez_compressed(OUT / f"scores_{name}.npz", **scores)

df = pd.DataFrame(rows)
df.to_csv(OUT / "benchmark.csv", index=False)
piv = df.pivot_table(index="model", columns="task", values="pr_auc")
piv["MEAN"] = piv.mean(1)
print(piv.sort_values("MEAN", ascending=False).round(3).to_string())
