"""Bộ nạp dữ liệu benchmark.

Tất cả dataset được tải từ GitHub raw (hoặc có sẵn trong scikit-learn) và
cache tại ``data/raw``. Mỗi loader trả về một ``Dataset`` gồm:

* ``X``: DataFrame đặc trưng (giữ nguyên kiểu category để còn serialize cho LLM)
* ``y``: Series nhãn đã mã hoá 0..K-1
* ``classes``: tên lớp dạng ngôn ngữ tự nhiên (dùng làm verbalizer cho LLM)
* ``task``: câu hỏi phân loại bằng tiếng Anh (dùng trong prompt LLM)
* ``llm_ok``: tên cột có ý nghĩa ngữ nghĩa -> phù hợp để thử với LLM
"""
from __future__ import annotations

import io
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(os.environ.get("DRM_DATA_DIR", Path(__file__).resolve().parents[2] / "data" / "raw"))

JB = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
URLS = {
    "adult": JB + "adult-all.csv",
    "german_credit": JB + "german.csv",
    "diabetes": JB + "pima-indians-diabetes.data.csv",
    "ionosphere": JB + "ionosphere.csv",
    "sonar": JB + "sonar.csv",
    "banknote": JB + "banknote_authentication.csv",
    "haberman": JB + "haberman.csv",
    "glass": JB + "glass.csv",
    "titanic": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    "penguins": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv",
    "heart": "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv",
}


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: pd.Series
    classes: list[str]
    task: str
    llm_ok: bool = False
    notes: str = ""
    cat_cols: list[str] = field(default_factory=list)

    @property
    def num_cols(self) -> list[str]:
        return [c for c in self.X.columns if c not in self.cat_cols]

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def summary(self) -> dict:
        return {
            "dataset": self.name,
            "n_rows": len(self.X),
            "n_features": self.X.shape[1],
            "n_categorical": len(self.cat_cols),
            "n_classes": self.n_classes,
            "minority_ratio": round(self.y.value_counts(normalize=True).min(), 3),
            "llm_ok": self.llm_ok,
        }


def _fetch(name: str) -> bytes:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{name}.csv"
    if not path.exists():
        with urllib.request.urlopen(URLS[name], timeout=60) as r:
            path.write_bytes(r.read())
    return path.read_bytes()


def _csv(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(_fetch(name)), **kw)


def _finish(name, df, target, classes, task, mapping=None, **kw) -> Dataset:
    y_raw = df.pop(target)
    if mapping is not None:
        y = y_raw.map(mapping)
        assert y.notna().all(), f"unmapped labels in {name}"
    else:
        y = y_raw
    y = y.astype(int).rename("target").reset_index(drop=True)
    X = df.reset_index(drop=True)
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    for c in cat_cols:
        X[c] = X[c].astype("category")
    return Dataset(name, X, y, classes, task, cat_cols=cat_cols, **kw)


# --------------------------------------------------------------------------- loaders
ADULT_COLS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
              "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
              "hours_per_week", "native_country", "income"]


def load_adult(max_rows: int | None = 8000, seed: int = 0) -> Dataset:
    df = _csv("adult", header=None, names=ADULT_COLS, na_values="?", skipinitialspace=True)
    df = df.drop(columns=["fnlwgt", "education_num"])  # trọng số mẫu & trùng education
    for c in ["workclass", "occupation", "native_country"]:
        df[c] = df[c].fillna("unknown")
    if max_rows and len(df) > max_rows:
        from sklearn.model_selection import train_test_split
        df, _ = train_test_split(df, train_size=max_rows, stratify=df["income"], random_state=seed)
    return _finish("adult", df, "income", ["no", "yes"],
                   "Does this person earn more than 50,000 dollars per year?",
                   mapping={"<=50K": 0, ">50K": 1}, llm_ok=True,
                   notes=f"UCI Adult/Census income, stratified subsample {max_rows} rows")


GERMAN_COLS = ["checking_account", "duration_months", "credit_history", "purpose", "credit_amount",
               "savings", "employment_since", "installment_rate_pct", "personal_status_sex",
               "other_debtors", "residence_since", "property", "age", "other_installment_plans",
               "housing", "existing_credits", "job", "people_liable", "telephone", "foreign_worker",
               "risk"]
GERMAN_CODES = {
    "A11": "< 0 DM", "A12": "0 to 200 DM", "A13": ">= 200 DM", "A14": "no checking account",
    "A30": "no credits taken / all paid back duly", "A31": "all credits at this bank paid back duly",
    "A32": "existing credits paid back duly till now", "A33": "delay in paying off in the past",
    "A34": "critical account / other credits existing",
    "A40": "car (new)", "A41": "car (used)", "A42": "furniture/equipment", "A43": "radio/television",
    "A44": "domestic appliances", "A45": "repairs", "A46": "education", "A47": "vacation",
    "A48": "retraining", "A49": "business", "A410": "others",
    "A61": "< 100 DM", "A62": "100 to 500 DM", "A63": "500 to 1000 DM", "A64": ">= 1000 DM",
    "A65": "unknown / no savings account",
    "A71": "unemployed", "A72": "< 1 year", "A73": "1 to 4 years", "A74": "4 to 7 years", "A75": ">= 7 years",
    "A91": "male, divorced/separated", "A92": "female, divorced/separated/married",
    "A93": "male, single", "A94": "male, married/widowed", "A95": "female, single",
    "A101": "none", "A102": "co-applicant", "A103": "guarantor",
    "A121": "real estate", "A122": "building society savings / life insurance", "A123": "car or other",
    "A124": "unknown / no property",
    "A141": "bank", "A142": "stores", "A143": "none",
    "A151": "rent", "A152": "own", "A153": "for free",
    "A171": "unemployed / unskilled non-resident", "A172": "unskilled resident",
    "A173": "skilled employee / official", "A174": "management / self-employed / highly qualified",
    "A191": "none", "A192": "yes, registered", "A201": "yes", "A202": "no",
}


def load_german_credit() -> Dataset:
    df = _csv("german_credit", header=None, names=GERMAN_COLS)
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].map(GERMAN_CODES)
            assert df[c].notna().all(), c
    return _finish("german_credit", df, "risk", ["good", "bad"],
                   "Is this applicant a bad credit risk?", mapping={1: 0, 2: 1}, llm_ok=True,
                   notes="Statlog German credit; codes decoded to text")


def load_diabetes() -> Dataset:
    cols = ["pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin", "bmi",
            "diabetes_pedigree", "age", "outcome"]
    df = _csv("diabetes", header=None, names=cols)
    return _finish("diabetes", df, "outcome", ["no", "yes"],
                   "Does this patient have diabetes?", llm_ok=True,
                   notes="Pima Indians diabetes; zeros in glucose/bp/skin/insulin/bmi mean missing")


def load_ionosphere() -> Dataset:
    df = _csv("ionosphere", header=None)
    df.columns = [f"a{i}" for i in range(df.shape[1] - 1)] + ["label"]
    return _finish("ionosphere", df, "label", ["bad", "good"], "Is this radar return good?",
                   mapping={"b": 0, "g": 1})


def load_sonar() -> Dataset:
    df = _csv("sonar", header=None)
    df.columns = [f"band{i}" for i in range(df.shape[1] - 1)] + ["label"]
    return _finish("sonar", df, "label", ["rock", "mine"], "Is the object a metal mine?",
                   mapping={"R": 0, "M": 1})


def load_banknote() -> Dataset:
    df = _csv("banknote", header=None, names=["wavelet_variance", "wavelet_skewness",
                                               "wavelet_curtosis", "image_entropy", "label"])
    return _finish("banknote", df, "label", ["genuine", "forged"], "Is this banknote forged?")


def load_haberman() -> Dataset:
    df = _csv("haberman", header=None, names=["age_at_operation", "operation_year_1900s",
                                               "positive_axillary_nodes", "status"])
    return _finish("haberman", df, "status", ["survived", "died"],
                   "Did the breast-cancer patient die within 5 years after surgery?",
                   mapping={1: 0, 2: 1}, llm_ok=True)


def load_glass() -> Dataset:
    df = _csv("glass", header=None, names=["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "type"])
    codes = sorted(df["type"].unique())
    return _finish("glass", df, "type", [f"type_{c}" for c in codes], "What type of glass is this?",
                   mapping={c: i for i, c in enumerate(codes)})


def load_titanic() -> Dataset:
    df = _csv("titanic").drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    df.columns = [c.lower() for c in df.columns]
    df["pclass"] = df["pclass"].astype(int)
    df["embarked"] = df["embarked"].map({"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}).fillna("unknown")
    return _finish("titanic", df, "survived", ["no", "yes"],
                   "Did this passenger survive the Titanic disaster?", llm_ok=True,
                   notes="Rất phổ biến -> nguy cơ LLM đã 'thuộc' dữ liệu")


def load_penguins() -> Dataset:
    df = _csv("penguins").dropna()
    species = ["Adelie", "Chinstrap", "Gentoo"]
    return _finish("penguins", df, "species", species, "Which species is this penguin?",
                   mapping={s: i for i, s in enumerate(species)}, llm_ok=True)


def load_heart() -> Dataset:
    df = _csv("heart", encoding="utf-8-sig")
    return _finish("heart", df, "target", ["class_0", "class_1"], "Heart disease (Kaggle version)",
                   notes="Phiên bản Kaggle của Cleveland; ngữ nghĩa nhãn không rõ ràng -> không dùng cho LLM")


def _sk(name, loader, task, classes, llm_ok=False):
    def f() -> Dataset:
        b = loader(as_frame=True)
        df = b.frame.copy()
        names = classes or [str(c) for c in b.target_names]
        return _finish(name, df, "target", names, task, llm_ok=llm_ok)
    return f


def _breast_cancer():
    from sklearn.datasets import load_breast_cancer
    # sklearn: 0 = malignant, 1 = benign -> đảo lại để 1 = malignant (lớp "dương tính")
    b = load_breast_cancer(as_frame=True)
    df = b.frame.copy()
    df["target"] = 1 - df["target"]
    return _finish("breast_cancer", df, "target", ["benign", "malignant"],
                   "Is this breast tumour malignant?", llm_ok=True)


def _wine():
    from sklearn.datasets import load_wine
    return _sk("wine", load_wine, "Which cultivar produced this wine?", ["cultivar_1", "cultivar_2", "cultivar_3"])()


def _digits():
    from sklearn.datasets import load_digits
    return _sk("digits", load_digits, "Which digit is this?", [str(i) for i in range(10)])()


LOADERS = {
    "adult": load_adult,
    "german_credit": load_german_credit,
    "diabetes": load_diabetes,
    "breast_cancer": _breast_cancer,
    "heart": load_heart,
    "titanic": load_titanic,
    "haberman": load_haberman,
    "ionosphere": load_ionosphere,
    "sonar": load_sonar,
    "banknote": load_banknote,
    "penguins": load_penguins,
    "wine": _wine,
    "glass": load_glass,
    "digits": _digits,
}


def load(name: str, **kw) -> Dataset:
    return LOADERS[name](**kw)


def load_all(names=None) -> dict[str, Dataset]:
    return {n: load(n) for n in (names or LOADERS)}


def serialize_row(row: pd.Series, style: str = "text") -> str:
    """Chuyển một dòng bảng thành văn bản (TabLLM, Hegselmann et al., 2023).

    style="text":  "The age is 39. The workclass is State-gov. ..."
    style="list":  "- age: 39\n- workclass: State-gov\n..."
    """
    parts = []
    for k, v in row.items():
        if isinstance(v, (float, np.floating)):
            v = "unknown" if np.isnan(v) else (f"{v:.4g}")
        key = k.replace("_", " ")
        parts.append(f"The {key} is {v}." if style == "text" else f"- {key}: {v}")
    return (" " if style == "text" else "\n").join(parts)
