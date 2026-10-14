"""Chạy toàn bộ benchmark ML (notebook 01 có thể nạp lại kết quả này).

    PYTHONPATH=src python scripts/run_benchmark.py --repeats 2 --jobs 4
"""
import argparse
import warnings
from pathlib import Path

from drm import benchmark as B
from drm import data

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="*", default=None)
ap.add_argument("--models", nargs="*", default=None)
ap.add_argument("--splits", type=int, default=5)
ap.add_argument("--repeats", type=int, default=2)
ap.add_argument("--jobs", type=int, default=4)
ap.add_argument("--out", default="results/benchmark_folds.csv")
a = ap.parse_args()

ds = data.load_all(a.datasets)
res = B.run_benchmark(ds, B.make_models(include=a.models), n_splits=a.splits, n_repeats=a.repeats, n_jobs=a.jobs)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
res.to_csv(a.out, index=False)
print("saved", a.out, res.shape)
