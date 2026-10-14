"""Các phương pháp ra quyết định đa tiêu chí (MCDM) – cài đặt NumPy thuần.

Quy ước:
    X      : ma trận quyết định (m phương án x n tiêu chí)
    w      : vector trọng số (tổng = 1)
    types  : +1 = benefit (càng lớn càng tốt), -1 = cost (càng nhỏ càng tốt)
Mọi hàm ``score_*`` trả về điểm "càng lớn càng tốt" (VIKOR được đổi dấu) để dễ so sánh.
Các cài đặt được kiểm tra chéo với thư viện ``pymcdm`` trong ``tests/test_mcdm.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------- normalisation


def minmax_norm(X, types):
    X = np.asarray(X, float)
    lo, hi = X.min(0), X.max(0)
    rng = np.where(hi - lo == 0, 1, hi - lo)
    N = (X - lo) / rng
    return np.where(np.asarray(types) > 0, N, 1 - N)


def vector_norm(X):
    X = np.asarray(X, float)
    d = np.sqrt((X ** 2).sum(0))
    return X / np.where(d == 0, 1, d)


# --------------------------------------------------------------------------- weighting


def equal_weights(X):
    n = np.asarray(X).shape[1]
    return np.full(n, 1.0 / n)


def entropy_weights(X, types=None):
    """Shannon entropy weights: tiêu chí càng phân tán -> càng nhiều thông tin -> trọng số lớn."""
    X = np.asarray(X, float)
    if types is not None:  # đưa về dạng benefit dương để tính tỉ lệ
        X = minmax_norm(X, types) + 1e-12
    P = X / X.sum(0)
    m = X.shape[0]
    E = -(P * np.log(P + 1e-300)).sum(0) / np.log(m)
    d = 1 - E
    return d / d.sum()


def critic_weights(X, types):
    """CRITIC (Diakoulaki et al., 1995): độ lệch chuẩn x mức độ xung đột (1 - tương quan)."""
    N = minmax_norm(X, types)
    std = N.std(0, ddof=1)
    R = np.corrcoef(N, rowvar=False)
    R = np.nan_to_num(R, nan=0.0)
    C = std * (1 - R).sum(0)
    return C / C.sum()


RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


def ahp_weights(A):
    """AHP (Saaty, 1980): trọng số = vector riêng chính; trả về (w, CR).

    CR < 0.10 được coi là nhất quán chấp nhận được.
    """
    A = np.asarray(A, float)
    n = A.shape[0]
    vals, vecs = np.linalg.eig(A)
    i = np.argmax(vals.real)
    w = np.abs(vecs[:, i].real)
    w = w / w.sum()
    lam = vals[i].real
    ci = (lam - n) / (n - 1) if n > 2 else 0.0
    cr = ci / RI[n] if RI.get(n, 0) > 0 else 0.0
    return w, cr


def ahp_from_upper(upper: dict, criteria: list[str]):
    """Tạo ma trận so sánh cặp từ dict {(ci, cj): a_ij} chỉ cần nửa trên."""
    n = len(criteria)
    A = np.ones((n, n))
    idx = {c: i for i, c in enumerate(criteria)}
    for (a, b), v in upper.items():
        A[idx[a], idx[b]] = v
        A[idx[b], idx[a]] = 1.0 / v
    return A


# --------------------------------------------------------------------------- ranking methods


def score_saw(X, w, types):
    """Simple Additive Weighting / WSM với chuẩn hoá min-max."""
    return minmax_norm(X, types) @ np.asarray(w)


def score_topsis(X, w, types, norm="vector"):
    """TOPSIS (Hwang & Yoon, 1981). norm="vector" (kinh điển) hoặc "minmax"."""
    if norm == "minmax":  # sau min-max mọi tiêu chí đã thành benefit
        V = minmax_norm(X, types) * np.asarray(w)
        ideal, anti = V.max(0), V.min(0)
        dp = np.sqrt(((V - ideal) ** 2).sum(1))
        dm = np.sqrt(((V - anti) ** 2).sum(1))
        return dm / (dp + dm)
    V = vector_norm(X) * np.asarray(w)
    t = np.asarray(types)
    ideal = np.where(t > 0, V.max(0), V.min(0))
    anti = np.where(t > 0, V.min(0), V.max(0))
    dp = np.sqrt(((V - ideal) ** 2).sum(1))
    dm = np.sqrt(((V - anti) ** 2).sum(1))
    return dm / (dp + dm)


def vikor_SRQ(X, w, types, v=0.5):
    X = np.asarray(X, float)
    t = np.asarray(types)
    best = np.where(t > 0, X.max(0), X.min(0))
    worst = np.where(t > 0, X.min(0), X.max(0))
    rng = np.where(best - worst == 0, 1, best - worst)
    D = np.asarray(w) * (best - X) / rng
    S, R = D.sum(1), D.max(1)
    sr = (S.max() - S.min()) or 1.0
    rr = (R.max() - R.min()) or 1.0
    Q = v * (S - S.min()) / sr + (1 - v) * (R - R.min()) / rr
    return S, R, Q


def score_vikor(X, w, types, v=0.5):
    """VIKOR (Opricovic, 1998): Q càng nhỏ càng tốt -> trả về -Q."""
    return -vikor_SRQ(X, w, types, v)[2]


def score_promethee2(X, w, types, pref="linear"):
    """PROMETHEE II (Brans & Vincke, 1985): net outranking flow.

    pref="usual": P(d)=1 nếu d>0; pref="linear" (V-shape) với p = khoảng giá trị của tiêu chí.
    """
    X = np.asarray(X, float) * np.asarray(types)  # đổi cost -> benefit
    m = X.shape[0]
    D = X[:, None, :] - X[None, :, :]  # d_j(a,b)
    if pref == "usual":
        P = (D > 0).astype(float)
    else:
        p = X.max(0) - X.min(0)
        p = np.where(p == 0, 1, p)
        P = np.clip(D / p, 0, 1)
    Pi = (P * np.asarray(w)).sum(2)
    phi_plus = Pi.sum(1) / (m - 1)
    phi_minus = Pi.sum(0) / (m - 1)
    return phi_plus - phi_minus


def score_edas(X, w, types):
    """EDAS (Keshavarz Ghorabaee et al., 2015): khoảng cách tới nghiệm trung bình."""
    X = np.asarray(X, float)
    t = np.asarray(types)
    AV = X.mean(0)
    AVs = np.where(AV == 0, 1, AV)
    pda = np.where(t > 0, np.maximum(0, X - AV), np.maximum(0, AV - X)) / AVs
    nda = np.where(t > 0, np.maximum(0, AV - X), np.maximum(0, X - AV)) / AVs
    SP, SN = pda @ w, nda @ w
    NSP = SP / (SP.max() or 1)
    NSN = 1 - SN / (SN.max() or 1)
    return (NSP + NSN) / 2


def score_moora(X, w, types):
    """MOORA – ratio system (Brauers & Zavadskas, 2006)."""
    V = vector_norm(X) * np.asarray(w)
    return (V * np.asarray(types)).sum(1)


METHODS = {
    "SAW": score_saw,
    "TOPSIS": score_topsis,
    "VIKOR": score_vikor,
    "PROMETHEE II": score_promethee2,
    "EDAS": score_edas,
    "MOORA": score_moora,
}


# --------------------------------------------------------------------------- helpers & analysis


def to_rank(scores):
    """Điểm cao -> hạng 1."""
    return stats.rankdata(-np.asarray(scores), method="min").astype(int)


def rank_all(X, w, types, alternatives, methods=None) -> pd.DataFrame:
    methods = methods or METHODS
    return pd.DataFrame({k: to_rank(f(X, w, types)) for k, f in methods.items()}, index=alternatives)


def rank_correlations(ranks: pd.DataFrame, kind="spearman") -> pd.DataFrame:
    cols = ranks.columns
    f = stats.spearmanr if kind == "spearman" else stats.kendalltau
    M = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            M.loc[a, b] = M.loc[b, a] = f(ranks[a], ranks[b])[0]
    return M


def borda(ranks: pd.DataFrame) -> pd.Series:
    """Đồng thuận Borda: trung bình hạng qua các phương pháp."""
    return ranks.mean(1).rank(method="min").astype(int).sort_values()


def copeland(ranks: pd.DataFrame) -> pd.Series:
    """Đồng thuận Copeland: số trận thắng - số trận thua theo đa số phương pháp."""
    alts = ranks.index
    R = ranks.to_numpy()
    wins = np.zeros(len(alts))
    for i in range(len(alts)):
        for j in range(len(alts)):
            if i != j:
                better = (R[i] < R[j]).sum()
                worse = (R[i] > R[j]).sum()
                wins[i] += np.sign(better - worse)
    return pd.Series(stats.rankdata(-wins, method="min").astype(int), index=alts).sort_values()


def smaa(X, types, alternatives, method="TOPSIS", n_iter=5000, X_samples=None, w_center=None,
         concentration=None, seed=0):
    """SMAA-2 (Lahdelma et al., 1998) bằng Monte Carlo.

    * Trọng số ngẫu nhiên đều trên simplex (Dirichlet(1)) hoặc tập trung quanh ``w_center``
      với tham số ``concentration`` (Dirichlet(c * w_center)).
    * ``X_samples``: tuỳ chọn, mảng (S, m, n) các ma trận quyết định có nhiễu (vd. từng fold CV)
      -> lan truyền bất định của phép đo vào thứ hạng (cải tiến đề xuất).
    Trả về: rank acceptability (m x m), xác suất đứng đầu, central weight vector.
    """
    rng = np.random.default_rng(seed)
    f = METHODS[method] if isinstance(method, str) else method
    X = np.asarray(X, float)
    m, n = X.shape
    acc = np.zeros((m, m))
    cw = np.zeros((m, n))
    for it in range(n_iter):
        if w_center is not None and concentration:
            w = rng.dirichlet(np.asarray(w_center) * concentration + 1e-6)
        else:
            w = rng.dirichlet(np.ones(n))
        Xi = X if X_samples is None else X_samples[rng.integers(len(X_samples))]
        r = to_rank(f(Xi, w, types)) - 1
        acc[np.arange(m), r] += 1
        cw[r.argmin()] += w
    acc /= n_iter
    first = acc[:, 0]
    cw = cw / np.maximum(first[:, None] * n_iter, 1)
    A = pd.DataFrame(acc, index=alternatives, columns=[f"rank_{i + 1}" for i in range(m)])
    return A, pd.Series(first, index=alternatives).sort_values(ascending=False), pd.DataFrame(cw, index=alternatives)


def rank_reversal_test(X, w, types, alternatives, methods=None):
    """Loại lần lượt từng phương án; đếm số cặp còn lại bị đảo thứ tự (rank reversal)."""
    methods = methods or METHODS
    X = np.asarray(X, float)
    m = X.shape[0]
    out = {}
    for name, f in methods.items():
        base = f(X, w, types)
        flips, trials = 0, 0
        for k in range(m):
            keep = np.array([i for i in range(m) if i != k])
            new = f(X[keep], w, types)
            b = base[keep]
            for i in range(len(keep)):
                for j in range(i + 1, len(keep)):
                    trials += 1
                    if np.sign(b[i] - b[j]) != np.sign(new[i] - new[j]) and b[i] != b[j]:
                        flips += 1
        out[name] = {"pair_reversals": flips, "pairs_checked": trials, "reversal_rate": flips / trials}
    return pd.DataFrame(out).T


def weight_sensitivity(X, w, types, alternatives, criteria, method="TOPSIS", deltas=(-0.5, -0.25, 0.25, 0.5)):
    """One-at-a-time: tăng/giảm trọng số tiêu chí j theo tỉ lệ delta (renormalise), đo Kendall tau."""
    f = METHODS[method]
    base = to_rank(f(X, w, types))
    rows = []
    for j, c in enumerate(criteria):
        for d in deltas:
            w2 = np.array(w, float)
            w2[j] *= 1 + d
            w2 /= w2.sum()
            r = to_rank(f(X, w2, types))
            rows.append({"criterion": c, "delta": d, "kendall_tau": stats.kendalltau(base, r)[0],
                         "top1": alternatives[int(np.argmin(r))]})
    return pd.DataFrame(rows)
