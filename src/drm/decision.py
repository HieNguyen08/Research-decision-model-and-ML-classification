"""Mô hình ra quyết định vận hành cho hàng đợi CS (triage).

Mỗi *ca* (case) đến hàng đợi gồm một tin nhắn của khách và đơn hàng liên quan. Các mô hình dự báo (notebook 01, 02)
cho xác suất: tin nhắn độc hại / spam, đơn có rủi ro (bùng hàng hoặc chargeback). Hệ thống phải chọn **một hành động**:

    auto      – xử lý tự động: tin nhắn tới nhân viên/bot bình thường, đơn được giao
    review    – chuyển chuyên viên xem xét (giới hạn năng lực K ca/ngày), chuyên viên quyết định đúng với xác suất q
    restrict  – hạn chế ngay: ẩn/chặn tin nhắn + giữ đơn / yêu cầu trả trước

So sánh các *chính sách quyết định*:
    rules     – ngưỡng cố định trên từng điểm (thực tế phổ biến)
    bayes     – tối thiểu chi phí kỳ vọng (lý thuyết quyết định Bayes) + phân bổ năng lực review theo lợi ích kỳ vọng
    mcdm      – điểm ưu tiên đa tiêu chí (AHP -> TOPSIS) như một "scorecard" vận hành
    threshold – một ngưỡng duy nhất trên điểm rủi ro lớn nhất, tối ưu trên validation
    oracle    – biết trước sự thật (cận dưới chi phí)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import mcdm


@dataclass
class CostModel:
    """Chi phí (đơn vị: USD tương đương) – GIẢ ĐỊNH, cần thay bằng số liệu doanh nghiệp."""
    harm_toxic: float = 6.0          # tin độc hại tới nhân viên (thời gian, sức khoẻ tinh thần, nghỉ việc)
    harm_spam: float = 4.0           # spam/lừa đảo lọt qua (rủi ro cho khách, uy tín)
    false_restrict: float = 25.0     # chặn/giữ nhầm khách tốt (khiếu nại, mất khách)
    review_cost: float = 2.0         # 1 lần chuyên viên xem xét
    reviewer_accuracy: float = 0.95  # xác suất chuyên viên xử lý đúng
    order_fn: callable = field(default=lambda a: 4.0 + 0.08 * a)   # bỏ sót đơn bùng hàng
    order_fp: callable = field(default=lambda a: 0.8 + 0.02 * a)   # giữ/yêu cầu cọc đơn tốt (cộng thêm)

    def loss(self, action, toxic, spam, risky, amount):
        """Chi phí thực tế (vector hoá) của một hành động khi biết trạng thái thật."""
        harm = self.harm_toxic * toxic + self.harm_spam * spam + self.order_fn(amount) * risky
        innocent = (toxic == 0) & (spam == 0) & (risky == 0)
        if action == "auto":
            return harm
        if action == "restrict":
            return innocent * (self.false_restrict + self.order_fp(amount))
        if action == "review":
            return self.review_cost + (1 - self.reviewer_accuracy) * harm
        raise ValueError(action)

    def expected_loss(self, action, p_toxic, p_spam, p_risky, amount):
        """Chi phí kỳ vọng với xác suất dự báo (giả định các rủi ro độc lập có điều kiện)."""
        e_harm = self.harm_toxic * p_toxic + self.harm_spam * p_spam + self.order_fn(amount) * p_risky
        p_innocent = (1 - p_toxic) * (1 - p_spam) * (1 - p_risky)
        if action == "auto":
            return e_harm
        if action == "restrict":
            return p_innocent * (self.false_restrict + self.order_fp(amount))
        if action == "review":
            return self.review_cost + (1 - self.reviewer_accuracy) * e_harm
        raise ValueError(action)


ACTIONS = ("auto", "review", "restrict")


def build_cases(msg_pool: pd.DataFrame, order_pool: pd.DataFrame, n: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Ghép ngẫu nhiên tin nhắn (cột: toxic, spam, p_toxic, p_spam) với đơn (cột: risky, p_risky, amount)."""
    rng = np.random.default_rng(seed)
    m = msg_pool.iloc[rng.integers(0, len(msg_pool), n)].reset_index(drop=True)
    o = order_pool.iloc[rng.integers(0, len(order_pool), n)].reset_index(drop=True)
    return pd.concat([m, o], axis=1)


def _apply_capacity(want_review: np.ndarray, priority: np.ndarray, capacity: int, fallback: np.ndarray):
    """Chỉ `capacity` ca có priority cao nhất được review; phần dư nhận hành động fallback."""
    actions = np.where(want_review, "review", fallback).astype(object)
    idx = np.where(want_review)[0]
    if len(idx) > capacity:
        keep = idx[np.argsort(-priority[idx])[:capacity]]
        drop = np.setdiff1d(idx, keep)
        actions[drop] = fallback[drop]
    return actions


def policy_rules(c: pd.DataFrame, capacity: int, hi=0.9, lo=0.5):
    s = c[["p_toxic", "p_spam", "p_risky"]].max(1).to_numpy()
    restrict = s >= hi
    want = (s >= lo) & ~restrict
    # quy tắc thực tế: hàng đợi FIFO -> ai đến trước được review trước (priority = -thứ tự)
    acts = _apply_capacity(want, -np.arange(len(c)), capacity, np.array(["auto"] * len(c), dtype=object))
    acts[restrict] = "restrict"
    return acts


def policy_bayes(c: pd.DataFrame, capacity: int, cost: CostModel):
    args = (c.p_toxic.to_numpy(), c.p_spam.to_numpy(), c.p_risky.to_numpy(), c.amount.to_numpy())
    L = {a: cost.expected_loss(a, *args) for a in ACTIONS}
    best_no_review = np.where(L["restrict"] < L["auto"], "restrict", "auto").astype(object)
    gain = np.minimum(L["auto"], L["restrict"]) - L["review"]  # lợi ích kỳ vọng nếu được review
    want = gain > 0
    return _apply_capacity(want, gain, capacity, best_no_review)


def policy_threshold(c: pd.DataFrame, capacity: int, t_review: float, t_restrict: float):
    s = c[["p_toxic", "p_spam", "p_risky"]].max(1).to_numpy()
    restrict = s >= t_restrict
    want = (s >= t_review) & ~restrict
    acts = _apply_capacity(want, s, capacity, np.array(["auto"] * len(c), dtype=object))
    acts[restrict] = "restrict"
    return acts


def mcdm_priority(c: pd.DataFrame, weights: dict, method="TOPSIS"):
    """Điểm ưu tiên đa tiêu chí: mỗi ca là một phương án, tiêu chí = các rủi ro + giá trị đơn."""
    crit = list(weights)
    X = c[crit].to_numpy(float)
    w = np.array([weights[k] for k in crit])
    types = np.ones(len(crit))  # mọi tiêu chí: càng lớn càng cần ưu tiên
    if method == "TOPSIS":
        return mcdm.score_topsis(X, w, types, norm="minmax")
    return mcdm.METHODS[method](X, w, types)


def policy_mcdm(c: pd.DataFrame, capacity: int, weights: dict, restrict_top: float = 0.01, review_min: float = 0.3):
    """Scorecard MCDM: top `restrict_top` theo điểm -> restrict, các ca kế tiếp (điểm ≥ review_min) -> review theo năng lực."""
    s = mcdm_priority(c, weights)
    order = np.argsort(-s)
    n_restrict = int(restrict_top * len(c))
    acts = np.array(["auto"] * len(c), dtype=object)
    acts[order[:n_restrict]] = "restrict"
    want = (s >= np.quantile(s, 1 - review_min)) & (acts != "restrict")
    acts2 = _apply_capacity(want, s, capacity, acts.copy())
    return acts2


def policy_oracle(c: pd.DataFrame, capacity: int, cost: CostModel):
    args = (c.toxic.to_numpy(), c.spam.to_numpy(), c.risky.to_numpy(), c.amount.to_numpy())
    L = {a: cost.loss(a, *args) for a in ACTIONS}
    return np.where(L["restrict"] < L["auto"], "restrict", "auto").astype(object)


def evaluate(c: pd.DataFrame, actions: np.ndarray, cost: CostModel, name: str = "") -> dict:
    args = (c.toxic.to_numpy(), c.spam.to_numpy(), c.risky.to_numpy(), c.amount.to_numpy())
    tot = np.zeros(len(c))
    for a in ACTIONS:
        m = actions == a
        tot[m] = cost.loss(a, *(x[m] for x in args))
    harmful = (c.toxic + c.spam) > 0
    innocent = (c.toxic == 0) & (c.spam == 0) & (c.risky == 0)
    return {"policy": name, "cost_per_1k_cases": 1000 * tot.mean(),
            "harmful_msgs_to_agents_%": 100 * float((harmful & (actions == "auto")).sum() / max(harmful.sum(), 1)),
            "risky_orders_shipped_%": 100 * float(((c.risky == 1) & (actions == "auto")).sum() / max(c.risky.sum(), 1)),
            "false_restrict_%": 100 * float((innocent & (actions == "restrict")).sum() / max(innocent.sum(), 1)),
            "review_%": 100 * float((actions == "review").mean()),
            "restrict_%": 100 * float((actions == "restrict").mean())}


def tune_threshold_policy(c_val: pd.DataFrame, capacity: int, cost: CostModel, grid=None):
    grid = grid if grid is not None else np.linspace(0.05, 0.99, 25)
    best = (None, None, np.inf)
    for tr in grid:
        for tv in grid[grid < tr]:
            v = evaluate(c_val, policy_threshold(c_val, capacity, tv, tr), cost)["cost_per_1k_cases"]
            if v < best[2]:
                best = (tv, tr, v)
    return best


def tune_mcdm_policy(c_val: pd.DataFrame, capacity: int, cost: CostModel, weights: dict):
    best = (None, None, np.inf)
    for rt in [0.0, 0.005, 0.01, 0.02, 0.04, 0.08]:
        for rm in [0.05, 0.1, 0.2, 0.3, 0.5]:
            v = evaluate(c_val, policy_mcdm(c_val, capacity, weights, rt, rm), cost)["cost_per_1k_cases"]
            if v < best[2]:
                best = (rt, rm, v)
    return best


def risk_coverage(y: np.ndarray, p: np.ndarray, thr: float = 0.5):
    """Selective classification (Chow, 1970; Geifman & El-Yaniv, 2017): tự động xử lý các ca *tự tin* nhất,
    chuyển phần còn lại cho người. Trả về DataFrame coverage -> error rate trên phần tự động."""
    conf = np.abs(p - thr)
    order = np.argsort(-conf)
    pred = (p >= thr).astype(int)
    err = (pred[order] != y[order]).astype(float)
    cov = np.arange(1, len(y) + 1) / len(y)
    return pd.DataFrame({"coverage": cov, "error": np.cumsum(err) / np.arange(1, len(y) + 1)})
