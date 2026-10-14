"""Gian lận đơn hàng / thanh toán: fraud, chargeback (hoàn tiền gian lận), bùng hàng COD.

Nguồn dữ liệu:
* ``creditcard``  – ULB Credit Card Fraud (Dal Pozzolo et al., 2015): 284 807 giao dịch, 0.17% gian lận.
                    Chia **theo thời gian** (80% đầu train, 20% cuối test) để mô phỏng triển khai thực.
* ``ieee_cis``    – IEEE-CIS Fraud Detection (Vesta, Kaggle 2019), giao dịch e-commerce thật. Ban tổ chức mô tả
                    (trên diễn đàn cuộc thi) nhãn isFraud gắn với **chargeback** được báo cáo -> gần nhất với bài
                    toán chargeback. Cần Kaggle API (chạy trên Colab).
* ``simulate_orders`` – **DỮ LIỆU MÔ PHỎNG** đơn hàng e-commerce với hai nhãn: ``cod_refusal`` (bùng hàng COD)
                    và ``chargeback`` (khách thanh toán trước rồi đòi hoàn tiền gian lận). Cơ chế sinh dữ liệu được
                    công bố đầy đủ bên dưới; chỉ dùng để minh hoạ pipeline – **thay bằng dữ liệu nội bộ** qua
                    ``load_internal_orders``.
"""
from __future__ import annotations

import io
import time
import urllib.request
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from .data import RAW_DIR, Dataset

CC_URL = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"


@dataclass
class FraudTask:
    name: str
    train: Dataset
    test: Dataset
    amount_col: str | None  # cột giá trị đơn/giao dịch để tính chi phí
    description: str

    def summary(self):
        return {"task": self.name, "n_train": len(self.train.y), "n_test": len(self.test.y),
                "n_features": self.train.X.shape[1], "pos_rate_train": round(self.train.y.mean(), 4),
                "pos_rate_test": round(self.test.y.mean(), 4), "description": self.description}


def _ds(name, X, y, task):
    X = X.reset_index(drop=True)
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    for c in cat:
        X[c] = X[c].astype(str).astype("category")
    return Dataset(name, X, pd.Series(np.asarray(y), name="target").astype(int), ["legit", "fraud"], task, cat_cols=cat)


def load_creditcard(test_frac=0.2) -> FraudTask:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / "creditcard.csv"
    if not path.exists():
        with urllib.request.urlopen(CC_URL, timeout=300) as r:
            path.write_bytes(r.read())
    df = pd.read_csv(path).sort_values("Time").reset_index(drop=True)
    cut = int(len(df) * (1 - test_frac))
    y = df.pop("Class")
    df["hour"] = (df["Time"] // 3600) % 24
    df = df.drop(columns=["Time"])
    return FraudTask("creditcard", _ds("creditcard", df.iloc[:cut], y.iloc[:cut], "card fraud"),
                     _ds("creditcard", df.iloc[cut:], y.iloc[cut:], "card fraud"), "Amount",
                     "ULB card transactions (PCA features V1-V28), temporal split")


def load_ieee_cis(folder: str, max_rows: int | None = 200_000, test_frac=0.2) -> FraudTask:
    """IEEE-CIS (Kaggle): tải bằng `kaggle competitions download -c ieee-fraud-detection` (cần chấp nhận luật thi)."""
    tr = pd.read_csv(f"{folder}/train_transaction.csv", nrows=max_rows)
    try:
        idf = pd.read_csv(f"{folder}/train_identity.csv")
        tr = tr.merge(idf, on="TransactionID", how="left")
    except FileNotFoundError:
        pass
    tr = tr.sort_values("TransactionDT").reset_index(drop=True)
    y = tr.pop("isFraud")
    tr = tr.drop(columns=["TransactionID", "TransactionDT"])
    keep = [c for c in tr.columns if tr[c].isna().mean() < 0.9]  # bỏ cột gần như trống
    tr = tr[keep]
    for c in tr.columns:
        if not pd.api.types.is_numeric_dtype(tr[c]):
            tr[c] = tr[c].fillna("NA")
    cut = int(len(tr) * (1 - test_frac))
    return FraudTask("ieee_cis_chargeback", _ds("ieee", tr.iloc[:cut], y.iloc[:cut], "chargeback"),
                     _ds("ieee", tr.iloc[cut:], y.iloc[cut:], "chargeback"), "TransactionAmt",
                     "IEEE-CIS (label = reported chargeback), temporal split")


def load_internal_orders(path, label_col, amount_col=None, time_col=None, drop_cols=(), test_frac=0.2,
                         name="internal") -> FraudTask:
    """Cắm dữ liệu đơn hàng nội bộ (vd. nhãn bùng hàng / chargeback) vào pipeline."""
    df = pd.read_csv(path)
    if time_col:
        df = df.sort_values(time_col)
    y = df.pop(label_col)
    df = df.drop(columns=[c for c in (*drop_cols, time_col) if c and c in df.columns])
    cut = int(len(df) * (1 - test_frac))
    return FraudTask(name, _ds(name, df.iloc[:cut], y.iloc[:cut], label_col),
                     _ds(name, df.iloc[cut:], y.iloc[cut:], label_col), amount_col, f"internal: {path}")


# --------------------------------------------------------------------------- simulator (MÔ PHỎNG)

SIM_DOC = """
Cơ chế sinh dữ liệu mô phỏng (giả định, có thể thay đổi):
* Mỗi khách hàng có một kiểu ẩn: normal 89.5% | impulsive 7% (đặt bốc đồng, hay đổi ý) |
  prankster 2.3% (cố ý bùng/phá, "boom hàng") | fraudster 1.2% (thẻ ăn cắp / friendly fraud).
* Đặc trưng quan sát được tương quan *có nhiễu* với kiểu ẩn: tài khoản mới, chưa xác minh SĐT, nhiều tài khoản
  chung thiết bị, nhiều đơn trong 24h, mua đêm khuya, flash-sale giảm sâu, đổi địa chỉ, email dùng một lần,
  địa chỉ thanh toán ≠ giao hàng, lịch sử bùng/chargeback, điểm toxic cao nhất trong chat với CS...
* cod_refusal chỉ xảy ra ở đơn COD; chargeback chỉ ở đơn trả trước (thẻ/ví).
  Xác suất = sigmoid(tổ hợp tuyến tính + tương tác phi tuyến + nhiễu) -> mô hình cây có lợi thế vừa phải.
"""


def simulate_orders(n: int = 60_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    kind = rng.choice(["normal", "impulsive", "prankster", "fraudster"], size=n, p=[0.895, 0.07, 0.023, 0.012])
    imp, pra, fra = (kind == "impulsive"), (kind == "prankster"), (kind == "fraudster")
    bad = pra | fra

    age = np.where(bad, rng.lognormal(2.0, 1.2, n), rng.lognormal(5.2, 1.0, n)).clip(0, 3000)
    n_prev = rng.poisson(np.clip(age / 60, 0, 40))
    p_ref = np.where(pra, rng.beta(4, 3, n), np.where(imp, rng.beta(2, 6, n), rng.beta(1, 25, n)))
    prev_ref = rng.binomial(n_prev, p_ref)
    prev_cb = rng.binomial(n_prev, np.where(fra, 0.15, 0.003))
    pay = np.where(fra, rng.choice(["card", "ewallet", "cod"], n, p=[0.6, 0.3, 0.1]),
                   np.where(pra, rng.choice(["cod", "card", "ewallet"], n, p=[0.9, 0.05, 0.05]),
                            rng.choice(["cod", "card", "ewallet"], n, p=[0.55, 0.25, 0.20])))
    value = rng.lognormal(np.where(fra, 4.4, 3.5), 0.8, n).round(2)
    flash = rng.random(n) < np.where(imp, 0.6, 0.2)
    discount = np.clip(rng.normal(np.where(flash, 45, 10), 12, n), 0, 80).round(0)
    night = rng.random(n) < np.where(bad, 0.35, np.where(imp, 0.25, 0.08))
    phone_ok = rng.random(n) < np.where(bad, 0.35, 0.92)
    addr_chg = rng.poisson(np.where(bad, 1.5, 0.15))
    acc_dev = 1 + rng.poisson(np.where(bad, 2.5, 0.1))
    orders24 = 1 + rng.poisson(np.where(pra, 2.0, np.where(imp, 0.8, 0.2)))
    distance = rng.gamma(2, np.where(pra, 120, 60), n).round(1)
    disp_email = rng.random(n) < np.where(fra, 0.4, 0.03)
    mismatch = (pay != "cod") & (rng.random(n) < np.where(fra, 0.55, 0.06))
    toxic = np.clip(rng.beta(1, 12, n) + np.where(bad, rng.beta(2, 5, n), 0), 0, 1).round(3)
    cs_contacts = rng.poisson(np.where(bad, 1.2, 0.3))
    n_items = 1 + rng.poisson(np.where(pra, 2.5, 1.0))

    def sig(z):
        return 1 / (1 + np.exp(-z))

    z_ref = (-3.4 + 1.6 * imp + 3.2 * pra
             + 3.0 * np.where(n_prev > 0, prev_ref / np.maximum(n_prev, 1), 0.1)
             + 0.9 * (flash & (discount > 40)) + 0.5 * night + 0.9 * ~phone_ok
             + 0.45 * np.log1p(orders24 - 1) + 0.25 * addr_chg + 0.002 * distance
             + 0.7 * ((age < 14) & (value > 60)) + rng.normal(0, 0.6, n))
    z_cb = (-7.0 + 4.6 * fra + 1.1 * mismatch + 0.9 * disp_email + 0.45 * np.log(value)
            + 1.2 * np.minimum(prev_cb, 2) + 0.7 * (age < 7) + 0.8 * toxic * (cs_contacts > 0)
            + 0.6 * (night & (value > 100)) + rng.normal(0, 0.6, n))
    cod = pay == "cod"
    df = pd.DataFrame({
        "account_age_days": age.round(1), "n_prev_orders": n_prev, "prev_refusals": prev_ref,
        "prev_refusal_rate": np.where(n_prev > 0, prev_ref / np.maximum(n_prev, 1), np.nan).round(3),
        "prev_chargebacks": prev_cb, "payment_method": pay, "order_value": value, "n_items": n_items,
        "discount_pct": discount, "is_flash_sale": flash.astype(int), "night_order": night.astype(int),
        "phone_verified": phone_ok.astype(int), "address_changes_30d": addr_chg,
        "accounts_on_device": acc_dev, "orders_last_24h": orders24, "shipping_distance_km": distance,
        "disposable_email": disp_email.astype(int), "billing_shipping_mismatch": mismatch.astype(int),
        "max_chat_toxicity": toxic, "cs_contacts_before_delivery": cs_contacts,
        "cod_refusal": np.where(cod, rng.random(n) < sig(z_ref), 0).astype(int),
        "chargeback": np.where(~cod, rng.random(n) < sig(z_cb), 0).astype(int),
        "_latent_kind": kind,
    })
    return df


def sim_tasks(n=60_000, seed=0, test_frac=0.25) -> dict[str, FraudTask]:
    df = simulate_orders(n, seed)
    out = {}
    for target, mask, desc in [("cod_refusal", df.payment_method == "cod", "SYNTHETIC: COD refusal (bùng hàng)"),
                               ("chargeback", df.payment_method != "cod", "SYNTHETIC: chargeback / friendly fraud")]:
        d = df[mask].drop(columns=["_latent_kind", "cod_refusal", "chargeback"]).reset_index(drop=True)
        y = df.loc[mask, target].reset_index(drop=True)
        if target == "cod_refusal":
            d = d.drop(columns=["payment_method", "billing_shipping_mismatch"])
        tr, te = train_test_split(np.arange(len(d)), test_size=test_frac, stratify=y, random_state=seed)
        out[f"sim_{target}"] = FraudTask(f"sim_{target}", _ds(target, d.iloc[tr], y.iloc[tr], target),
                                         _ds(target, d.iloc[te], y.iloc[te], target), "order_value", desc)
    return out


# --------------------------------------------------------------------------- models & evaluation


def make_fraud_models(seed=0, n_jobs=4, include=None):
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier, IsolationForest,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    from .benchmark import PrunedTree

    m = {
        "LogReg": LogisticRegression(max_iter=3000),
        "LogReg-balanced": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "DT-CART": DecisionTreeClassifier(min_samples_leaf=5, random_state=seed),
        "DT-pruned": PrunedTree(n_alphas=8, cv=3, random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=n_jobs, random_state=seed),
        "RandomForest-balanced": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                                                        n_jobs=n_jobs, random_state=seed),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=n_jobs, random_state=seed),
        "HistGB": HistGradientBoostingClassifier(random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                                 n_jobs=n_jobs, random_state=seed, verbosity=0),
        "LightGBM": LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.8, n_jobs=n_jobs, random_state=seed, verbose=-1),
        # Bản regularize: mặc định của LightGBM/HistGB không ổn định khi lớp dương cực hiếm (lá chứa quá ít mẫu dương)
        "LightGBM-reg": LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=100,
                                       reg_lambda=5, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                       n_jobs=n_jobs, random_state=seed, verbose=-1),
        "HistGB-reg": HistGradientBoostingClassifier(l2_regularization=5, min_samples_leaf=100, random_state=seed),
        "CatBoost": CatBoostClassifier(iterations=600, thread_count=n_jobs, random_seed=seed, verbose=False,
                                       allow_writing_files=False),
        "IsolationForest (unsup.)": IsolationForest(n_estimators=300, random_state=seed, n_jobs=n_jobs),
    }
    if include:
        m = {k: v for k, v in m.items() if k in include}
    return m


FRAUD_INTERPRETABILITY = {"LogReg": 4, "LogReg-balanced": 4, "DT-CART": 4, "DT-pruned": 5, "RandomForest": 2,
                          "RandomForest-balanced": 2, "ExtraTrees": 2, "HistGB": 2, "XGBoost": 2, "LightGBM": 2,
                          "LightGBM-reg": 2, "HistGB-reg": 2, "CatBoost": 2, "IsolationForest (unsup.)": 2}


def _score(pipe, X):
    m = pipe.named_steps["model"]
    if hasattr(m, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return -pipe.decision_function(X)  # IsolationForest: điểm càng thấp càng bất thường


def precision_at_budget(y, s, budget=0.005):
    """Precision khi chỉ được gửi top `budget` (vd. 0.5%) đơn/giao dịch sang review thủ công."""
    k = max(1, int(round(budget * len(s))))
    top = np.argsort(-s)[:k]
    return float(y[top].mean()), float(y[top].sum() / max(y.sum(), 1))


def expected_cost(y, s, thr, amount, fn_cost, fp_cost):
    """Tổng chi phí khi chặn/review mọi đơn có điểm ≥ thr. fn_cost/fp_cost: hàm của amount."""
    flag = s >= thr
    return float(np.sum(fn_cost(amount) * ((~flag) & (y == 1))) + np.sum(fp_cost(amount) * (flag & (y == 0))))


def best_cost_threshold(y, s, amount, fn_cost, fp_cost, grid=200):
    qs = np.unique(np.quantile(s, np.linspace(0, 1, grid)))
    costs = [expected_cost(y, s, t, amount, fn_cost, fp_cost) for t in qs]
    i = int(np.argmin(costs))
    return float(qs[i]), float(costs[i])


# Chi phí mặc định (giả định, chỉnh theo số liệu thực của doanh nghiệp):
COSTS = {
    # bùng hàng COD: bỏ sót = phí ship 2 chiều + xử lý + hao hụt (~$4 + 8% giá trị);
    # báo động nhầm = gọi xác minh/yêu cầu cọc làm mất ~ 3% khả năng chốt đơn -> ~ $0.8 + 2% giá trị
    "cod_refusal": (lambda a: 4.0 + 0.08 * a, lambda a: 0.8 + 0.02 * a),
    # chargeback: bỏ sót = mất hàng + tiền + phí chargeback $15; báo động nhầm = review thủ công $3 + 1% giá trị
    "chargeback": (lambda a: a + 15.0, lambda a: 3.0 + 0.01 * a),
    "card fraud": (lambda a: a + 15.0, lambda a: 3.0 + 0.01 * a),
}


def fit_eval_fraud(task: FraudTask, name: str, model, seed=0, return_scores=False):
    from .benchmark import make_pipeline_for

    tr = task.train
    itr, iva = train_test_split(np.arange(len(tr.y)), test_size=0.2, stratify=tr.y, random_state=seed)
    pipe = make_pipeline_for(name if name in {"LogReg"} else "LogReg" if name.startswith("LogReg") else name,
                             clone(model), tr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        if name.startswith("IsolationForest"):
            pipe.fit(tr.X.iloc[itr])
        else:
            pipe.fit(tr.X.iloc[itr], tr.y.iloc[itr])
        fit_t = time.perf_counter() - t0
        s_va = _score(pipe, tr.X.iloc[iva])
        t0 = time.perf_counter()
        s = _score(pipe, task.test.X)
        pred_t = time.perf_counter() - t0
    y = task.test.y.to_numpy()
    fn_c, fp_c = COSTS[task.train.task]
    amt_va = tr.X.iloc[iva][task.amount_col].to_numpy()
    amt = task.test.X[task.amount_col].to_numpy()
    thr, _ = best_cost_threshold(tr.y.iloc[iva].to_numpy(), s_va, amt_va, fn_c, fp_c)
    cost = expected_cost(y, s, thr, amt, fn_c, fp_c)
    cost_none = expected_cost(y, s, np.inf, amt, fn_c, fp_c)  # không can thiệp gì
    prec_b, rec_b = precision_at_budget(y, s, 0.005)
    from .text import recall_at_fpr
    is_prob = not name.startswith("IsolationForest")
    rec = {"task": task.name, "model": name, "pr_auc": average_precision_score(y, s), "roc_auc": roc_auc_score(y, s),
           "recall@1%fpr": recall_at_fpr(y, s, 0.01), "precision@0.5%budget": prec_b, "recall@0.5%budget": rec_b,
           "brier": brier_score_loss(y, s) if is_prob else np.nan,
           "cost_threshold": thr, "expected_cost": cost, "cost_no_action": cost_none,
           "cost_saving_%": 100 * (1 - cost / cost_none), "flag_rate": float((s >= thr).mean()),
           "fit_time_s": fit_t, "ms_per_1k": 1000 * pred_t / len(s) * 1000}
    return (rec, s, pipe) if return_scores else rec
