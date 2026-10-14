"""Jev (TypeSafe AI) – "System One" decision model – và các thí nghiệm ra quyết định zero-shot.

Jev nhận một *state* (văn bản / JSON) và nhiều *câu hỏi có kiểu*, trả về xác suất đã hiệu chỉnh
(huấn luyện bằng RLCD – Reinforcement Learning for Calibrated Decisions) trong **một lần gọi**:

    noul    câu hỏi có/không                -> {"noul": P(yes)}
    choice  chọn một trong các nhãn          -> {"choice", "confidence", "probabilities": {nhãn: p}}
    score   thang bậc 0..k (rubric có thứ tự) -> {"score": E[bậc], "confidence", "probabilities", "legend"}

Giao thức (đối chiếu từ SDK chính thức `typesafe-sdk` 0.7.4, MIT):
    POST {base_url}/v1/systemone   Authorization: Bearer <TYPESAFE_API_KEY>
    body = {"state": ..., "model": "jev-latest", "questions": {tên: {"type": ..., "instructions": ..., "criteria": ...}}}

Module gồm:
* ``JevClient``       – client HTTP tối giản (không phụ thuộc SDK), cache JSONL, gọi song song, đếm token/độ trễ.
                       ``base_url`` đổi được -> dùng với các server mở tương thích `/v1/systemone`.
* ``LocalJevAdapter`` – cùng giao diện nhưng chạy bằng LLM mở (``llm.HFScorer``) qua log-likelihood của nhãn
                       -> baseline "Jev-like không có RLCD", và để chạy thử không cần API key.
* ``WaterSheepAdapter`` – decision model mở, miễn phí (ModernBERT-base, Apache-2.0), cùng định dạng câu hỏi/trả lời
                       -> thay thế Jev khi không có credit.
* Bộ câu hỏi cho kiểm duyệt tin nhắn, bùng hàng/chargeback, AHP; hàm đánh giá & ECE.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"
PRICE_PER_MTOK_INPUT = 0.042  # USD, theo bài ra mắt của TypeSafe (output miễn phí) – cần kiểm tra lại


# --------------------------------------------------------------------------- question builders


def noul(instructions: str, true: str | None = None, false: str | None = None) -> dict:
    q = {"type": "noul", "instructions": instructions}
    if true is not None or false is not None:
        q["criteria"] = {k: v for k, v in (("true", true), ("false", false)) if v is not None}
    return q


def choice(instructions: str, criteria: dict) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}


def score(instructions: str, levels: list) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


# --------------------------------------------------------------------------- HTTP client


class JevError(RuntimeError):
    pass


def _urllib_transport(url: str, headers: dict, body: dict, timeout: float):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read().decode("utf-8"))
        except Exception:
            payload = {"detail": str(e)}
        return e.code, payload


class JevClient:
    """Client tối giản cho `/v1/systemone`.

    transport: hàm (url, headers, body, timeout) -> (status, json); thay được trong test.
    cache_path: file JSONL lưu {key: response} để chạy lại notebook không phải trả tiền lần nữa.
    """

    RETRY_STATUS = {408, 429, 500, 502, 503, 504}

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str = DEFAULT_MODEL,
                 cache_path: str | Path | None = None, max_workers: int = 8, timeout: float = 30.0,
                 max_retries: int = 4, transport=None, name: str = "jev"):
        self.api_key = (api_key or os.environ.get("TYPESAFE_API_KEY", "")).strip()
        self.base_url = (base_url or os.environ.get("TYPESAFE_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.model = model
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_retries = max_retries
        self.transport = transport or _urllib_transport
        self.name = name
        self.usage = {"calls": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0, "latency_s": []}
        self._lock = threading.Lock()
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, dict] = {}
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._cache[rec["key"]] = rec["response"]

    def _key(self, state, questions) -> str:
        blob = json.dumps({"m": self.model, "s": state, "q": questions}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def ask(self, state, questions: dict) -> dict:
        key = self._key(state, questions)
        if key in self._cache:
            with self._lock:
                self.usage["cache_hits"] += 1
            return self._cache[key]
        if not self.api_key:
            raise JevError("Thiếu TYPESAFE_API_KEY (Colab: thêm vào 🔑 Secrets).")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                   "Accept": "application/json"}
        body = {"state": state, "model": self.model, "questions": questions}
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                status, payload = self.transport(f"{self.base_url}/v1/systemone", headers, body, self.timeout)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                status, payload = 599, {"detail": str(e)}
            dt = time.perf_counter() - t0
            if status == 200:
                break
            if status not in self.RETRY_STATUS or attempt == self.max_retries:
                raise JevError(f"HTTP {status}: {str(payload)[:300]}")
            time.sleep(delay)
            delay = min(delay * 2, 30)
        with self._lock:
            self.usage["calls"] += 1
            self.usage["latency_s"].append(dt)
            u = payload.get("usage") or {}
            self.usage["input_tokens"] += u.get("input_tokens") or 0
            self.usage["output_tokens"] += u.get("output_tokens") or 0
            self._cache[key] = payload
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "response": payload}, ensure_ascii=False) + "\n")
        return payload

    def ask_many(self, states: list, questions: dict, progress: bool = True) -> list[dict]:
        out = [None] * len(states)

        def run(i):
            out[i] = self.ask(states[i], questions)

        with ThreadPoolExecutor(self.max_workers) as ex:
            for k, _ in enumerate(ex.map(run, range(len(states)))):
                if progress and (k + 1) % 200 == 0:
                    print(f"  {self.name}: {k + 1}/{len(states)}", flush=True)
        return out

    def cost_report(self) -> dict:
        lat = np.array(self.usage["latency_s"]) if self.usage["latency_s"] else np.array([np.nan])
        return {"calls": self.usage["calls"], "cache_hits": self.usage["cache_hits"],
                "input_tokens": self.usage["input_tokens"],
                "est_cost_usd": self.usage["input_tokens"] / 1e6 * PRICE_PER_MTOK_INPUT,
                "latency_p50_ms": float(np.nanmedian(lat) * 1000), "latency_p95_ms": float(np.nanpercentile(lat, 95) * 1000)}


# --------------------------------------------------------------------------- local Jev-like adapter


class LocalJevAdapter:
    """Trả lời câu hỏi có kiểu bằng LLM mở (log-likelihood nhãn), xuất cùng định dạng response như Jev.

    Không có huấn luyện RLCD -> dùng làm baseline để đo phần đóng góp của RLCD, hoặc chạy thử offline.
    """

    def __init__(self, scorer, max_state_chars: int = 1500):
        self.scorer = scorer
        self.name = f"jev-like ({scorer.name})"
        self.max_state_chars = max_state_chars
        self.usage = {"calls": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0, "latency_s": []}

    def _state(self, state) -> str:
        s = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)
        return s[: self.max_state_chars]

    @staticmethod
    def _softmax(z):
        z = np.asarray(z, float) - np.max(z)
        e = np.exp(z)
        return e / e.sum()

    def _prompt(self, state_txt, body):
        return self.scorer.chat(f"Input:\n{state_txt}\n\n{body}")

    def ask(self, state, questions: dict) -> dict:
        t0 = time.perf_counter()
        st = self._state(state)
        answers = {}
        for name, q in questions.items():
            instr = q.get("instructions") or ""
            if q["type"] == "noul":
                crit = q.get("criteria") or {}
                extra = "".join(f"\n- {'Yes' if k == 'true' else 'No'} means: {v}" for k, v in crit.items())
                lp = self.scorer.label_logprobs([self._prompt(st, f"Question: {instr}{extra}\nAnswer Yes or No.")],
                                                ["No", "Yes"])[0]
                answers[name] = {"type": "noul", "noul": float(self._softmax(lp)[1])}
            elif q["type"] == "choice":
                labels = list(q["criteria"])
                opts = "\n".join(f"- {l}" + (f": {d}" if d else "") for l, d in q["criteria"].items())
                lp = self.scorer.label_logprobs([self._prompt(st, f"Question: {instr}\nOptions:\n{opts}\n"
                                                                  "Answer with exactly one option name.")], labels)[0]
                p = self._softmax(lp)
                answers[name] = {"type": "choice", "choice": labels[int(p.argmax())], "confidence": float(p.max()),
                                 "probabilities": dict(zip(labels, map(float, p)))}
            elif q["type"] == "score":
                levels = list(q["criteria"])
                rub = "\n".join(f"{i}: {d}" for i, d in enumerate(levels))
                lp = self.scorer.label_logprobs([self._prompt(st, f"Question: {instr}\nRubric:\n{rub}\n"
                                                                  "Answer with the level number only.")],
                                                [str(i) for i in range(len(levels))])[0]
                p = self._softmax(lp)
                answers[name] = {"type": "score", "score": float(np.dot(p, np.arange(len(levels)))),
                                 "confidence": float(p.max()), "legend": {str(i): d for i, d in enumerate(levels)},
                                 "probabilities": {str(i): float(v) for i, v in enumerate(p)}}
        self.usage["calls"] += 1
        self.usage["latency_s"].append(time.perf_counter() - t0)
        return {"model": self.name, "answers": answers, "usage": {}}

    def ask_many(self, states, questions, progress=True):
        out = []
        for i, s in enumerate(states):
            out.append(self.ask(s, questions))
            if progress and (i + 1) % 100 == 0:
                print(f"  {self.name}: {i + 1}/{len(states)}", flush=True)
        return out

    def cost_report(self):
        lat = np.array(self.usage["latency_s"]) if self.usage["latency_s"] else np.array([np.nan])
        return {"calls": self.usage["calls"], "cache_hits": 0, "input_tokens": np.nan, "est_cost_usd": 0.0,
                "latency_p50_ms": float(np.nanmedian(lat) * 1000), "latency_p95_ms": float(np.nanpercentile(lat, 95) * 1000)}


# --------------------------------------------------------------------------- WaterSheep (mở, miễn phí)

WATERSHEEP_REPO = "samratduttaofficial/WaterSheep"
WATERSHEEP_PIP = "git+https://github.com/SamratDuttaOfficial/WaterSheep"


class WaterSheepAdapter:
    """WaterSheep (Apache-2.0): decision model mở trên ModernBERT-base, hiệu chỉnh xác suất bằng temperature.

    Nhận cùng request {state, questions} và trả cùng định dạng answers như Jev (noul / choice / score),
    nên mọi hàm đánh giá trong module dùng được ngay. Chạy tốt trên CPU, nhanh hơn trên GPU.
    Cài: ``pip install git+https://github.com/SamratDuttaOfficial/WaterSheep`` (yêu cầu transformers>=5).
    Có thể thay bằng server tương thích: ``watersheep --model ... --serve`` rồi
    ``JevClient(api_key="local", base_url="http://127.0.0.1:8766")``.
    """

    def __init__(self, model: str = WATERSHEEP_REPO, device: str | None = None, ws=None, name: str = "WaterSheep"):
        if ws is None:
            from watersheep import WaterSheep
            ws = WaterSheep.load(model, device=device)
        self.ws = ws
        self.name = name
        self.usage = {"calls": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0, "latency_s": []}

    def ask(self, state, questions: dict) -> dict:
        t0 = time.perf_counter()
        resp = self.ws.ask({"state": state, "questions": questions})
        errors = {k: a["error"] for k, a in resp.get("answers", {}).items() if isinstance(a, dict) and "error" in a}
        if errors:
            raise JevError(f"WaterSheep từ chối câu hỏi: {errors}")
        self.usage["calls"] += 1
        self.usage["input_tokens"] += (resp.get("usage") or {}).get("input_tokens") or 0
        self.usage["latency_s"].append(time.perf_counter() - t0)
        return resp

    def ask_many(self, states, questions, progress=True):
        out = []
        for i, s in enumerate(states):
            out.append(self.ask(s, questions))
            if progress and (i + 1) % 200 == 0:
                print(f"  {self.name}: {i + 1}/{len(states)}", flush=True)
        return out

    def cost_report(self):
        lat = np.array(self.usage["latency_s"]) if self.usage["latency_s"] else np.array([np.nan])
        return {"calls": self.usage["calls"], "cache_hits": 0, "input_tokens": self.usage["input_tokens"],
                "est_cost_usd": 0.0, "latency_p50_ms": float(np.nanmedian(lat) * 1000),
                "latency_p95_ms": float(np.nanpercentile(lat, 95) * 1000)}


# --------------------------------------------------------------------------- answer readers


def read_noul(resp: dict, name: str) -> float:
    return float(resp["answers"][name]["noul"])


def read_choice_probs(resp: dict, name: str) -> dict:
    return resp["answers"][name]["probabilities"]


def read_score(resp: dict, name: str) -> float:
    return float(resp["answers"][name]["score"])


# --------------------------------------------------------------------------- moderation (CS chat)

MODERATION_QUESTIONS = {
    "toxic": noul("Is this customer message toxic: rude, disrespectful, insulting, threatening, obscene or hateful?"),
    "offensive": noul("Is this message offensive, profane or abusive towards someone?"),
    "hate": noul("Does this message attack or demean people because of a protected attribute such as race, "
                 "ethnicity, religion, gender, sexual orientation, nationality or disability?"),
    "spam": noul("Is this message spam or a scam: unsolicited advertising, phishing, prize or lottery bait, "
                 "or an attempt to move payment or contact off the platform?"),
    "severity": score("How severe is this message for a customer-service agent who receives it?",
                      ["harmless", "impolite or mildly rude", "abusive, harassing or obscene",
                       "threatening, hateful or dangerous"]),
    "action": choice("What should the customer-service team do with this message?",
                     {"reply_normally": "ordinary request or feedback",
                      "reply_with_deescalation": "frustrated or rude customer, answer calmly",
                      "escalate_to_supervisor": "abusive, threatening or legally sensitive",
                      "block_and_report": "spam, scam or severe abuse"}),
}

# câu hỏi "generic" dùng cho mỗi task (Just Ask Jev: một câu hỏi chung đã đủ mạnh)
TASK_QUESTION = {"jigsaw_toxic": "toxic", "hatexplain_toxic": "toxic", "tweeteval_offensive": "offensive",
                 "tweeteval_hate": "hate", "davidson_hate": "hate", "sms_spam": "spam"}


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 0, 1)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def evaluate_moderation(client, task, n_test=300, seed=0, wording=("generic", "task"), obfuscate_p=None):
    """Jev (hoặc LocalJevAdapter) zero-shot trên tập con test của notebook 01.

    wording="generic": bộ câu hỏi chung MODERATION_QUESTIONS (mọi câu hỏi trong 1 lần gọi)
    wording="task":    dùng nguyên câu hỏi của task làm Noul (kiểm tra độ nhạy theo cách diễn đạt)
    """
    from .llm import text_test_subset
    from .text import metrics, obfuscate

    idx = text_test_subset(task, n_test, seed)
    y = task.y_test[idx]
    texts = [str(t) for t in task.X_test.iloc[idx]]
    if obfuscate_p:
        texts = [obfuscate(t, obfuscate_p) for t in texts]
    rows, scores = [], {"idx": idx}
    for w in wording:
        qs = MODERATION_QUESTIONS if w == "generic" else {"task": noul(task.question)}
        qname = TASK_QUESTION[task.name] if w == "generic" else "task"
        t0 = time.perf_counter()
        resps = client.ask_many([{"customer_message": t} for t in texts], qs, progress=False)
        dt = time.perf_counter() - t0
        s = np.array([read_noul(r, qname) for r in resps])
        m = metrics(y, s, 0.5)
        rows.append({"model": client.name, "task": task.name, "wording": w, "obfuscated": bool(obfuscate_p),
                     **m, "f1@0.5": m["f1"], "ece": expected_calibration_error(y, s),
                     "brier": float(np.mean((s - y) ** 2)), "wall_s_per_msg": dt / len(texts)})
        scores[f"{w}|p"] = s
        if w == "generic":
            scores["severity"] = np.array([read_score(r, "severity") for r in resps])
            scores["action"] = np.array([max(read_choice_probs(r, "action").items(), key=lambda kv: kv[1])[0]
                                         for r in resps])
    return pd.DataFrame(rows), scores


# --------------------------------------------------------------------------- orders (COD refusal / chargeback)

ORDER_FIELD_DOC = {
    "account_age_days": "days since the customer account was created",
    "n_prev_orders": "number of previous orders", "prev_refusals": "previous deliveries refused",
    "prev_refusal_rate": "share of previous deliveries refused", "prev_chargebacks": "previous chargebacks",
    "order_value": "order value in USD", "discount_pct": "discount percent", "is_flash_sale": "1 if flash sale",
    "night_order": "1 if ordered between midnight and 5am", "phone_verified": "1 if phone number verified",
    "address_changes_30d": "shipping address changes in last 30 days",
    "accounts_on_device": "customer accounts seen on the same device", "orders_last_24h": "orders in last 24 hours",
    "shipping_distance_km": "shipping distance in km", "disposable_email": "1 if disposable email domain",
    "billing_shipping_mismatch": "1 if billing and shipping addresses differ",
    "max_chat_toxicity": "max toxicity score (0-1) of the customer's chat messages",
    "cs_contacts_before_delivery": "customer-service contacts before delivery",
}

ORDER_QUESTIONS = {
    "cod_refusal": {
        "refuse": noul("Will this customer refuse to accept or pay for this cash-on-delivery order when it arrives?"),
        "risk": score("How risky is it to ship this cash-on-delivery order without verification?",
                      ["safe", "slightly risky", "risky", "very risky"]),
    },
    "chargeback": {
        "chargeback": noul("Will this customer later dispute this prepaid payment with their bank (chargeback) "
                           "or claim the purchase was unauthorized?"),
        "risk": score("How risky is this prepaid order for the shop?", ["safe", "slightly risky", "risky", "very risky"]),
    },
}


def order_state(row: pd.Series) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            v = "unknown"
        elif isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = round(float(v), 3)
        out[k] = {"value": v if not hasattr(v, "item") else v.item(), "meaning": ORDER_FIELD_DOC.get(k, k)}
    return {"order": out}


# --------------------------------------------------------------------------- Jev as AHP expert (one call)

SAATY = [1 / 9, 1 / 7, 1 / 5, 1 / 3, 1, 3, 5, 7, 9]


def ahp_questions(criteria: dict[str, str]) -> tuple[dict, list]:
    names = list(criteria)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    levels = lambda a, b: [f"{b} is extremely more important than {a}", f"{b} is very strongly more important than {a}",
                           f"{b} is strongly more important than {a}", f"{b} is moderately more important than {a}",
                           f"{a} and {b} are equally important", f"{a} is moderately more important than {b}",
                           f"{a} is strongly more important than {b}", f"{a} is very strongly more important than {b}",
                           f"{a} is extremely more important than {b}"]
    qs = {f"{a}__vs__{b}": score(f"For this scenario, compare the importance of '{a}' versus '{b}'.", levels(a, b))
          for a, b in pairs}
    return qs, pairs


def ahp_from_jev(resp: dict, criteria: dict[str, str]):
    """Ma trận so sánh cặp từ phân phối xác suất của Score: trung bình nhân theo xác suất trên thang Saaty."""
    from . import mcdm
    _, pairs = ahp_questions(criteria)
    upper = {}
    for a, b in pairs:
        p = resp["answers"][f"{a}__vs__{b}"]["probabilities"]
        probs = np.array([p[str(i)] if str(i) in p else p[i] for i in range(9)], float)
        probs = probs / probs.sum()
        upper[(a, b)] = float(math.exp(np.dot(probs, np.log(SAATY))))
    A = mcdm.ahp_from_upper(upper, list(criteria))
    w, cr = mcdm.ahp_weights(A)
    return pd.Series(w, index=list(criteria)), cr, A
