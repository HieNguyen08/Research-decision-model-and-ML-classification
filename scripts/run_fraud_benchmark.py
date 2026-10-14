"""Benchmark gian lận / chargeback / bùng hàng (notebook 02 nạp lại kết quả).

    PYTHONPATH=src python scripts/run_fraud_benchmark.py
    PYTHONPATH=src python scripts/run_fraud_benchmark.py --ieee data/raw/ieee   # thêm IEEE-CIS (Colab)

Đầu ra (results/fraud/): benchmark.csv, scores_<task>.npz (y_test, amount, điểm từng mô hình)
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from drm import fraud as F

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="*", default=None)
ap.add_argument("--ieee", default=None, help="thư mục chứa train_transaction.csv của IEEE-CIS")
ap.add_argument("--internal", default=None, help="CSV đơn hàng nội bộ")
ap.add_argument("--internal-label", default="label")
ap.add_argument("--internal-amount", default=None)
ap.add_argument("--jobs", type=int, default=4)
ap.add_argument("--out", default="results/fraud")
a = ap.parse_args()
OUT = Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)

tasks = {**F.sim_tasks(), "creditcard": F.load_creditcard()}
if a.ieee:
    tasks["ieee_cis_chargeback"] = F.load_ieee_cis(a.ieee)
if a.internal:
    tasks["internal"] = F.load_internal_orders(a.internal, a.internal_label, a.internal_amount)

rows = []
for tname, task in tasks.items():
    print(task.summary(), flush=True)
    scores = {"y_test": task.test.y.to_numpy(), "amount": task.test.X[task.amount_col].to_numpy()}
    for name, model in F.make_fraud_models(n_jobs=a.jobs, include=a.models).items():
        rec, s, _ = F.fit_eval_fraud(task, name, model, return_scores=True)
        rows.append(rec)
        scores[name] = s
        print(f"  {name:26s} pr_auc={rec['pr_auc']:.3f} cost_saving={rec['cost_saving_%']:.1f}% "
              f"fit={rec['fit_time_s']:.1f}s", flush=True)
    np.savez_compressed(OUT / f"scores_{tname}.npz", **scores)
    pd.DataFrame(rows).to_csv(OUT / "benchmark.csv", index=False)
print("saved", OUT / "benchmark.csv")
