"""Trích kết quả từ output đã lưu của notebook 03 và 05 (chạy trên Colab) ra CSV.

Dùng khi notebook được lưu lên GitHub kèm output nhưng các file trong `results/` không được đẩy lên
(ô cuối để `PUSH_RESULTS = False`). Số liệu lấy từ bảng hiển thị (đã làm tròn 3 chữ số) và từ log in ra
trong lúc chạy, nên đủ để báo cáo và cho notebook 04 (B7), nhưng không thay được file điểm `.npz`.

    python scripts/extract_notebook_results.py
"""
import base64
import io
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# (mô hình, task) mà dữ liệu test nằm (hoặc nghi nằm) trong dữ liệu huấn luyện của classifier fine-tune sẵn
LEAKAGE = {
    ("toxic-bert", "jigsaw_toxic"): "rò rỉ: huấn luyện trên Jigsaw",
    ("bert-tiny-finetuned-sms-spam-detection", "sms_spam"): "nghi rò rỉ: huấn luyện trên SMS Spam Collection",
    ("twitter-roberta-base-hate-latest", "tweeteval_hate"): "nghi rò rỉ: tập huấn luyện gộp 13 dataset hate",
    ("twitter-roberta-base-hate-latest", "hatexplain_toxic"): "nghi rò rỉ: tập huấn luyện gộp 13 dataset hate",
    ("twitter-roberta-base-hate-latest", "davidson_hate"): "nghi rò rỉ: tập huấn luyện gộp 13 dataset hate",
}


def load(nb_path):
    return json.loads((ROOT / nb_path).read_text())["cells"]


def find(cells, marker):
    hits = [c for c in cells if c["cell_type"] == "code" and marker in "".join(c["source"])]
    if len(hits) != 1:
        raise ValueError(f"marker {marker!r}: {len(hits)} ô khớp")
    return hits[0]


def table(cell, k=0):
    """Bảng HTML thứ k trong output của ô."""
    htmls = ["".join(o["data"]["text/html"]) for o in cell.get("outputs", [])
             if "data" in o and "text/html" in o["data"]]
    return pd.read_html(io.StringIO(htmls[k]), flavor="lxml")[0]


def stream(cell):
    return "".join("".join(o["text"]) for o in cell.get("outputs", []) if o["output_type"] == "stream")


def save_png(cell, path):
    for o in cell.get("outputs", []):
        if "data" in o and "image/png" in o["data"]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(base64.b64decode(o["data"]["image/png"]))
            return


def flat(df, n_index=1):
    """Bảng pandas đọc từ HTML -> cột phẳng. `n_index` cột đầu là chỉ số của DataFrame gốc
    (lấy tên chỉ số ở tầng tiêu đề cuối); các cột dữ liệu ghép các tầng tiêu đề bằng "|".
    Với bảng có chỉ số số nguyên mặc định (`n_index=0`) thì bỏ cột chỉ số đó."""
    named = lambda tup: [str(p) for p in (tup if isinstance(tup, tuple) else (tup,)) if not str(p).startswith("Unnamed")]
    cols = [((named(c) or ["index"])[-1] if i < n_index else "|".join(named(c))) for i, c in enumerate(df.columns)]
    df = df.copy()
    df.columns = cols
    if n_index == 0 and cols and cols[0] == "":
        df = df.iloc[:, 1:]
    return df.dropna(how="all").reset_index(drop=True)


def nb03(out):
    cells = load("notebooks/03_llm_moderation_colab.ipynb")
    rows = []
    for line in stream(find(cells, "for key in LLM_MODELS:")).splitlines():
        m = re.match(r"^(\S+) (\S+) \[\[0\.0, ([\d.]+), ([\d.]+)\], \[8\.0, ([\d.]+), ([\d.]+)\]\]$", line.strip())
        if m:
            key, task, p0, f0, p8, f8 = m.groups()
            rows += [{"model": key, "task": task, "shots": 0, "variant": "calibrated", "pr_auc": float(p0), "f1": float(f0)},
                     {"model": key, "task": task, "shots": 8, "variant": "calibrated", "pr_auc": float(p8), "f1": float(f8)}]
    pd.DataFrame(rows).to_csv(out / "llm_by_task.csv", index=False)

    rows = []
    for line in stream(find(cells, "for key in GUARDS:")).splitlines():
        m = re.match(r"^(\S+) (\S+) ([\d.]+) \| ví dụ output: (.*)$", line.strip())
        if m:
            rows.append({"model": m[1], "task": m[2], "pr_auc": float(m[3]), "example_output": m[4]})
    pd.DataFrame(rows).to_csv(out / "guard_by_task.csv", index=False)

    rows = []
    for line in stream(find(cells, "for mid in SPECIALIZED:")).splitlines():
        m = re.match(r"^(\S+/\S+) (\S+) ([\d.]+)$", line.strip())
        if m:
            rows.append({"model": m[1].split("/")[-1], "model_id": m[1], "task": m[2], "pr_auc": float(m[3]),
                         "leakage": LEAKAGE.get((m[1].split("/")[-1], m[2]), "")})
    pd.DataFrame(rows).to_csv(out / "spec_by_task.csv", index=False)

    flat(table(find(cells, "base_rows = []")), 4).to_csv(out / "summary_by_config.csv", index=False)

    best = find(cells, "best_cfg = cfg.loc")
    piv = flat(table(best), 1)
    long = piv.drop(columns=["MEAN"]).melt(id_vars=["model", "kind"], var_name="task", value_name="pr_auc").dropna()
    long["leakage"] = [LEAKAGE.get((m, t), "") for m, t in zip(long.model, long.task)]
    long.sort_values(["kind", "model", "task"]).to_csv(out / "pr_auc_best_config_by_task.csv", index=False)
    save_png(best, out / "pr_auc_heatmap.png")
    save_png(find(cells, 'lat = allres[allres["sec_per_msg"]'), out / "quality_vs_latency.png")

    flat(table(find(cells, "llm_only = allres")), 1).to_csv(out / "llm_shots_variant.csv", index=False)
    ens = flat(table(find(cells, 'for f in sorted(OUT.glob("llm_*_scores.npz"))')), 1)
    ens.melt(id_vars="llm", var_name="w_llm", value_name="pr_auc_mean_6_tasks").to_csv(out / "ens_llm_tfidf_mean.csv", index=False)
    flat(table(find(cells, "CRITERIA = {")), 0).to_csv(out / "dm_llm_ahp.csv", index=False)


def nb05(out):
    cells = load("notebooks/05_jev_zero_shot_decisions.ipynb")
    e1 = find(cells, "rows, SC = [], {}")
    rows = []
    for line in stream(e1).splitlines():
        m = re.match(r"^(.+) (\S+) \[\['generic', ([\d.]+), ([\d.]+)\], \['task', ([\d.]+), ([\d.]+)\]\] \| obf ([\d.]+)$",
                     line.strip())
        if m:
            b, t, pg, eg, pt, et, po = m.groups()
            rows.append({"backend": b, "task": t, "pr_auc_generic": float(pg), "ece_generic": float(eg),
                         "pr_auc_task_wording": float(pt), "ece_task_wording": float(et), "pr_auc_obf0.3": float(po)})
    pd.DataFrame(rows).to_csv(out / "e1_by_task.csv", index=False)
    flat(table(e1), 1).to_csv(out / "e1_pr_auc_by_task.csv", index=False)
    flat(table(find(cells, 'summ = clean.groupby("model")')), 1).to_csv(out / "e1_summary.csv", index=False)

    e2 = find(cells, "from sklearn.calibration import calibration_curve")
    cal = flat(table(e2), 1)
    cal.columns = [c.replace("|", "@labels=") for c in cal.columns]
    cal.to_csv(out / "e2_recalibration.csv", index=False)
    save_png(e2, out / "figures/e2_reliability.png")

    flat(table(find(cells, "w * rankdata(pj)")), 1).to_csv(out / "e4_ensemble.csv", index=False)
    flat(table(find(cells, 'act = pd.Series(s["action"])')), 1).to_csv(out / "e4_severity_action.csv", index=False)
    flat(table(find(cells, "st = F.sim_tasks()")), 0).to_csv(out / "e5_orders.csv", index=False)
    cold = flat(table(find(cells, "# Cold-start")), 2)
    cold.columns = list(cold.columns[:2]) + [f"roc_auc@{c}_labels" for c in cold.columns[2:]]
    cold.to_csv(out / "e5_cold_start.csv", index=False)
    flat(table(find(cells, "cases = D.build_cases")), 1).to_csv(out / "e6_triage.csv", index=False)
    flat(table(find(cells, "qs, _ = jev.ahp_questions(CRIT)")), 1).to_csv(out / "e7_ahp_weights.csv", index=False)
    rep = flat(table(find(cells, "rep = pd.DataFrame({n: c.cost_report()")), 1).rename(columns={"index": "backend"})
    rep.to_csv(out / "cost_latency.csv", index=False)


if __name__ == "__main__":
    for fn, sub in [(nb03, "results/llm"), (nb05, "results/jev")]:
        out = ROOT / sub
        out.mkdir(parents=True, exist_ok=True)
        fn(out)
        print(sub, sorted(p.name for p in out.rglob("*") if p.is_file()), file=sys.stderr)
