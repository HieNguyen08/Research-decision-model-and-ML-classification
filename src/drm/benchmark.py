"""Model zoo + cross-validation runner + kiểm định thống kê (Demšar, 2006)."""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

# --------------------------------------------------------------------------- custom trees


class PrunedTree(BaseEstimator, ClassifierMixin):
    """CART + cost-complexity pruning, chọn ccp_alpha bằng CV nội bộ (Breiman et al., 1984)."""

    def __init__(self, n_alphas=25, cv=5, criterion="gini", random_state=0):
        self.n_alphas = n_alphas
        self.cv = cv
        self.criterion = criterion
        self.random_state = random_state

    def fit(self, X, y):
        base = DecisionTreeClassifier(criterion=self.criterion, random_state=self.random_state)
        path = base.cost_complexity_pruning_path(X, y)
        alphas = np.unique(path.ccp_alphas[:-1])  # bỏ alpha cuối (cây chỉ còn gốc)
        if len(alphas) > self.n_alphas:
            alphas = np.quantile(alphas, np.linspace(0, 1, self.n_alphas))
        cv = StratifiedKFold(self.cv, shuffle=True, random_state=self.random_state)
        scores = [cross_val_score(clone(base).set_params(ccp_alpha=a), X, y, cv=cv,
                                  scoring="balanced_accuracy").mean() for a in alphas]
        # "1-SE rule" đơn giản hoá: chọn alpha lớn nhất trong 0.5% của điểm tốt nhất -> cây gọn hơn
        best = np.max(scores)
        self.ccp_alpha_ = max(a for a, s in zip(alphas, scores) if s >= best - 0.005)
        self.tree_ = clone(base).set_params(ccp_alpha=self.ccp_alpha_).fit(X, y)
        self.classes_ = self.tree_.classes_
        return self

    def predict(self, X):
        return self.tree_.predict(X)

    def predict_proba(self, X):
        return self.tree_.predict_proba(X)

    def get_n_leaves(self):
        return self.tree_.get_n_leaves()


class ObliqueTree(BaseEstimator, ClassifierMixin):
    """Cây 'xiên' rẻ tiền: CART trên [X, PCA(X)] -> cho phép đường cắt chéo trục.

    Xấp xỉ ý tưởng oblique tree (OC1, Murthy et al., 1994) mà không cần tối ưu siêu phẳng.
    """

    def __init__(self, max_depth=None, min_samples_leaf=2, random_state=0):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def _aug(self, X):
        return np.hstack([X, self.pca_.transform(self.scaler_.transform(X))])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.scaler_ = StandardScaler().fit(X)
        self.pca_ = PCA(random_state=self.random_state).fit(self.scaler_.transform(X))
        self.tree_ = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                            random_state=self.random_state).fit(self._aug(X), y)
        self.classes_ = self.tree_.classes_
        return self

    def predict_proba(self, X):
        return self.tree_.predict_proba(self._aug(np.asarray(X, dtype=float)))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

    def get_n_leaves(self):
        return self.tree_.get_n_leaves()


class RotationForest(BaseEstimator, ClassifierMixin):
    """Rotation Forest (Rodríguez, Kuncheva & Alonso, 2006) – bản rút gọn.

    Mỗi cây: chia ngẫu nhiên đặc trưng thành nhóm K, PCA trên bootstrap 75% cho từng nhóm,
    ghép thành ma trận xoay rồi huấn luyện CART trên dữ liệu đã xoay.
    """

    def __init__(self, n_estimators=50, group_size=3, random_state=0):
        self.n_estimators = n_estimators
        self.group_size = group_size
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        n, d = Xs.shape
        self.classes_ = np.unique(y)
        self.rotations_, self.trees_ = [], []
        for _ in range(self.n_estimators):
            perm = rng.permutation(d)
            R = np.zeros((d, d))
            for g in np.array_split(perm, max(1, int(np.ceil(d / self.group_size)))):
                idx = rng.choice(n, size=max(len(g) + 1, int(0.75 * n)), replace=True)
                pca = PCA().fit(Xs[np.ix_(idx, g)])
                comp = pca.components_
                R[np.ix_(g, g[: comp.shape[0]])] = comp.T
            tree = DecisionTreeClassifier(random_state=int(rng.integers(1 << 31))).fit(Xs @ R, y)
            self.rotations_.append(R)
            self.trees_.append(tree)
        return self

    def predict_proba(self, X):
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        P = np.zeros((len(Xs), len(self.classes_)))
        for R, t in zip(self.rotations_, self.trees_):
            P[:, np.searchsorted(self.classes_, t.classes_)] += t.predict_proba(Xs @ R)
        return P / len(self.trees_)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


# --------------------------------------------------------------------------- model zoo

# Điểm "khả năng diễn giải" (1 = hộp đen, 5 = minh bạch) – đánh giá chuyên gia, dùng cho MCDM.
INTERPRETABILITY = {
    "DT-CART": 5, "DT-entropy": 5, "DT-pruned": 5, "DT-oblique": 4,
    "LogReg": 4, "NaiveBayes": 4, "kNN": 3,
    "RandomForest": 2, "ExtraTrees": 2, "RotationForest": 2,
    "HistGB": 2, "XGBoost": 2, "LightGBM": 2, "CatBoost": 2,
    "SVM-RBF": 1, "MLP": 1, "Stacking": 1,
}

FAMILY = {
    "DT-CART": "single tree", "DT-entropy": "single tree", "DT-pruned": "single tree", "DT-oblique": "single tree",
    "RandomForest": "bagging", "ExtraTrees": "bagging", "RotationForest": "bagging",
    "HistGB": "boosting", "XGBoost": "boosting", "LightGBM": "boosting", "CatBoost": "boosting",
    "LogReg": "linear", "NaiveBayes": "probabilistic", "kNN": "instance-based",
    "SVM-RBF": "kernel", "MLP": "neural net", "Stacking": "meta-ensemble",
}


def make_models(seed: int = 0, n_jobs: int = 1, include=None) -> dict[str, BaseEstimator]:
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    m = {
        "DT-CART": DecisionTreeClassifier(random_state=seed),
        "DT-entropy": DecisionTreeClassifier(criterion="entropy", min_samples_leaf=2, random_state=seed),
        "DT-pruned": PrunedTree(random_state=seed),
        "DT-oblique": ObliqueTree(min_samples_leaf=2, random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=n_jobs, random_state=seed),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=300, n_jobs=n_jobs, random_state=seed),
        "RotationForest": RotationForest(n_estimators=50, random_state=seed),
        "HistGB": HistGradientBoostingClassifier(random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, n_jobs=n_jobs,
                                 random_state=seed, verbosity=0),
        "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, n_jobs=n_jobs,
                                   random_state=seed, verbose=-1),
        "CatBoost": CatBoostClassifier(iterations=500, thread_count=n_jobs, random_seed=seed,
                                       verbose=False, allow_writing_files=False),
        "LogReg": LogisticRegression(max_iter=2000),
        "NaiveBayes": GaussianNB(),
        "kNN": KNeighborsClassifier(n_neighbors=7),
        "SVM-RBF": SVC(probability=True, random_state=seed),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True,
                             random_state=seed),
        "Stacking": StackingClassifier(
            estimators=[("rf", RandomForestClassifier(n_estimators=200, n_jobs=n_jobs, random_state=seed)),
                        ("xgb", XGBClassifier(n_estimators=200, n_jobs=n_jobs, random_state=seed, verbosity=0)),
                        ("lr", LogisticRegression(max_iter=2000))],
            final_estimator=LogisticRegression(max_iter=2000), cv=5, n_jobs=1),
    }
    if include:
        m = {k: v for k, v in m.items() if k in include}
    return m


# Mô hình cần chuẩn hoá đặc trưng
NEEDS_SCALING = {"LogReg", "kNN", "SVM-RBF", "MLP", "NaiveBayes", "Stacking"}


def make_pipeline_for(name: str, model, ds) -> Pipeline:
    num = ds.num_cols
    cat = ds.cat_cols
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if name in NEEDS_SCALING:
        num_steps.append(("scale", StandardScaler()))
    pre = ColumnTransformer(
        [("num", Pipeline(num_steps), num),
         ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                           ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                                 min_frequency=5))]), cat)],
        sparse_threshold=0.0, verbose_feature_names_out=False)
    return Pipeline([("pre", pre), ("model", model)])


# --------------------------------------------------------------------------- evaluation


def _scores(y_true, proba, classes):
    pred = classes[proba.argmax(1)]
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "f1_macro": f1_score(y_true, pred, average="macro"),
    }
    try:
        if len(classes) == 2:
            out["roc_auc"] = roc_auc_score(y_true, proba[:, 1])
        else:
            out["roc_auc"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=classes)
    except ValueError:
        out["roc_auc"] = np.nan
    return out


def _n_leaves(model):
    m = model.named_steps["model"]
    if hasattr(m, "get_n_leaves"):
        try:
            return m.get_n_leaves()
        except Exception:
            return np.nan
    return np.nan


def evaluate_fold(name, model, ds, train_idx, test_idx, fold):
    pipe = make_pipeline_for(name, clone(model), ds)
    Xtr, Xte = ds.X.iloc[train_idx], ds.X.iloc[test_idx]
    ytr, yte = ds.y.iloc[train_idx].to_numpy(), ds.y.iloc[test_idx].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        pipe.fit(Xtr, ytr)
        fit_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        proba = pipe.predict_proba(Xte)
        pred_t = time.perf_counter() - t0
    classes = pipe.classes_
    full = np.zeros((len(Xte), ds.n_classes))
    full[:, classes] = proba
    rec = {"dataset": ds.name, "model": name, "fold": fold,
           **_scores(yte, full, np.arange(ds.n_classes)),
           "fit_time_s": fit_t, "predict_ms_per_1k": 1000 * pred_t / len(Xte) * 1000,
           "n_leaves": _n_leaves(pipe)}
    return rec


def run_benchmark(datasets: dict, models: dict, n_splits=5, n_repeats=1, seed=0, n_jobs=4,
                  verbose=True) -> pd.DataFrame:
    from joblib import Parallel, delayed

    jobs = []
    for ds in datasets.values():
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        for f, (tr, te) in enumerate(cv.split(ds.X, ds.y)):
            for name, model in models.items():
                jobs.append((name, model, ds, tr, te, f))
    if verbose:
        print(f"{len(jobs)} fits ({len(datasets)} datasets x {len(models)} models x {n_splits * n_repeats} folds)")
    recs = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
        delayed(evaluate_fold)(*j) for j in jobs)
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- statistics


def mean_table(res: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Bảng dataset x model (trung bình qua fold)."""
    return res.pivot_table(index="dataset", columns="model", values=metric, aggfunc="mean")


def average_ranks(table: pd.DataFrame, higher_is_better=True) -> pd.Series:
    r = table.rank(axis=1, ascending=not higher_is_better, method="average")
    return r.mean().sort_values()


def friedman_nemenyi(table: pd.DataFrame, alpha=0.05, higher_is_better=True):
    """Friedman test + critical difference của Nemenyi (Demšar, JMLR 2006)."""
    t = table.dropna()
    N, k = t.shape
    stat, p = stats.friedmanchisquare(*[t[c].to_numpy() for c in t.columns])
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * N))
    return {"N_datasets": N, "k_models": k, "friedman_chi2": stat, "p_value": p, "CD": cd,
            "avg_rank": average_ranks(t, higher_is_better)}


def wilcoxon_vs_best(table: pd.DataFrame, higher_is_better=True) -> pd.DataFrame:
    """So sánh cặp mô hình tốt nhất (theo rank) với các mô hình khác, hiệu chỉnh Holm."""
    ranks = average_ranks(table, higher_is_better)
    best = ranks.index[0]
    rows = []
    for m in ranks.index[1:]:
        d = (table[best] - table[m]).dropna()
        try:
            p = stats.wilcoxon(d).pvalue
        except ValueError:
            p = 1.0
        rows.append({"best": best, "other": m, "mean_diff": d.mean(), "p_raw": p})
    df = pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)
    m = len(df)
    adj = np.maximum.accumulate([(m - i) * p for i, p in enumerate(df["p_raw"])])
    df["p_holm"] = np.minimum(adj, 1.0)
    df["significant_0.05"] = df["p_holm"] < 0.05
    return df


def plot_cd_diagram(avg_rank: pd.Series, cd: float, ax=None, title=None):
    """Critical-difference diagram tối giản."""
    import matplotlib.pyplot as plt

    avg_rank = avg_rank.sort_values()
    k = len(avg_rank)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 0.35 * k + 1.5))
    y = np.arange(k)[::-1]
    ax.scatter(avg_rank.values, y, color="#3b6ea8", zorder=3)
    for yi, (name, r) in zip(y, avg_rank.items()):
        ax.text(r + 0.15, yi, f"{name} ({r:.2f})", va="center", fontsize=9)
    best = avg_rank.iloc[0]
    ax.axvspan(best, best + cd, color="#3b6ea8", alpha=0.12,
               label=f"CD = {cd:.2f} (không khác biệt có ý nghĩa với #1)")
    ax.set_yticks([])
    ax.set_ylim(-1.2, k - 0.3)
    ax.set_xlabel("Average rank (thấp hơn = tốt hơn)")
    ax.set_xlim(0.5, k + 3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(title or "Critical difference diagram")
    ax.grid(axis="x", alpha=0.3)
    return ax
