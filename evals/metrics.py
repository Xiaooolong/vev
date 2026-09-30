import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

N_BINS = 10
EPS = 1e-12
ONE_TOL = 1e-9
THRESHOLDS = [round(0.50 + 0.01 * i, 2) for i in range(50)]


@dataclass
class Row:
    record_id: str
    qname: str
    qtype: str  # "choice" | "noul" | "score"
    pred: Any  # choice: {label: p}; noul: P(true); score: {"i": p}
    target: Any  # same shape as pred
    confidence: float
    latency_ms: float | None = None
    labels: list[str] = field(default_factory=list)  # criteria order; used for argmax tie-breaks
    pred_reversed: Any = None  # already mapped back to the original label/level space
    pred_separate: Any = None
    pred_repeat: Any = None
    pred_jitter: Any = None  # answer on a near-duplicate frame (frame-pair stability)
    pred_negated: float | None = None

    def __post_init__(self):
        if not self.labels:
            if self.qtype == "noul":
                self.labels = ["true", "false"]
            elif self.qtype == "score":
                self.labels = [str(i) for i in range(len(self.target))]
            else:
                self.labels = list(self.target)


def as_dist(qtype: str, x: Any) -> dict[str, float]:
    if qtype == "noul":
        return {"true": float(x), "false": 1.0 - float(x)}
    return {str(k): float(v) for k, v in x.items()}


def argmax(dist: dict[str, float], labels: list[str]) -> str:
    # first label in criteria order wins ties (spec 5.1 for choice; lowest level for score;
    # "true" for noul, i.e. p >= 0.5 counts as true)
    best = labels[0]
    for lab in labels[1:]:
        if dist.get(lab, 0.0) > dist.get(best, 0.0):
            best = lab
    return best


def confidence_of(qtype: str, dist: dict[str, float]) -> float:
    if qtype == "noul":
        return max(dist["true"], dist["false"])
    k = len(dist)
    if k <= 1:
        return 1.0
    return (max(dist.values()) - 1.0 / k) / (1.0 - 1.0 / k)


def is_correct(row: Row) -> bool:
    return argmax(as_dist(row.qtype, row.pred), row.labels) == argmax(as_dist(row.qtype, row.target), row.labels)


def brier(row: Row) -> float:
    p, t = as_dist(row.qtype, row.pred), as_dist(row.qtype, row.target)
    return sum((p.get(k, 0.0) - t[k]) ** 2 for k in t)


def nll(row: Row) -> float:
    p, t = as_dist(row.qtype, row.pred), as_dist(row.qtype, row.target)
    return -sum(tk * math.log(max(p.get(k, 0.0), EPS)) for k, tk in t.items() if tk > 0)


def _bin(c: float) -> int:
    return min(max(int(c * N_BINS), 0), N_BINS - 1)


def ece(confs: list[float], correct: list[bool]) -> tuple[float, list[list]]:
    n = len(confs)
    sums = [[0.0, 0.0, 0] for _ in range(N_BINS)]
    for c, ok in zip(confs, correct):
        b = sums[_bin(c)]
        b[0] += c
        b[1] += ok
        b[2] += 1
    total, reliability = 0.0, []
    for cs, acc, m in sums:
        if m == 0:
            reliability.append([None, None, 0])
            continue
        total += m / n * abs(acc / m - cs / m)
        reliability.append([cs / m, acc / m, m])
    return total, reliability


def ece_bootstrap_ci(confs: list[float], correct: list[bool], n_boot: int = 1000, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    n = len(confs)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(ece([confs[i] for i in idx], [correct[i] for i in idx])[0])
    vals.sort()
    return [_quantile(vals, 0.025), _quantile(vals, 0.975)]


def ece_noise_floor(rows: list[Row], reps: int = 200, seed: int = 0) -> float:
    # A perfectly calibrated model predicts the target distribution itself; its "true" label is
    # drawn from that same distribution. The mean ECE over draws is the floor attainable at this n.
    rng = random.Random(seed)
    prepared = []
    for r in rows:
        t = as_dist(r.qtype, r.target)
        prepared.append((r.labels, [t.get(lab, 0.0) for lab in r.labels], confidence_of(r.qtype, t), argmax(t, r.labels)))
    confs = [p[2] for p in prepared]
    total = 0.0
    for _ in range(reps):
        correct = []
        for labels, probs, _, top in prepared:
            u, acc, drawn = rng.random(), 0.0, labels[-1]
            for lab, p in zip(labels, probs):
                acc += p
                if u < acc:
                    drawn = lab
                    break
            correct.append(drawn == top)
        total += ece(confs, correct)[0]
    return total / reps


def coverage_accuracy(confs: list[float], correct: list[bool]) -> list[list]:
    n = len(confs)
    out = []
    for th in THRESHOLDS:
        sel = [ok for c, ok in zip(confs, correct) if c >= th]
        out.append([th, len(sel) / n, (sum(sel) / len(sel)) if sel else None])
    return out


def auroc(scores: list[float], positive: list[bool]) -> float | None:
    n_pos = sum(positive)
    n_neg = len(positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-based average rank of the tie group
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum = sum(r for r, pos in zip(ranks, positive) if pos)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _f1_per_class(pairs: list[tuple[str, str]]) -> dict[str, float]:
    tp, fp, fn = Counter(), Counter(), Counter()
    for y, yhat in pairs:
        if y == yhat:
            tp[y] += 1
        else:
            fp[yhat] += 1
            fn[y] += 1
    out = {}
    for c in set(tp) | set(fp) | set(fn):
        denom = 2 * tp[c] + fp[c] + fn[c]
        out[c] = 2 * tp[c] / denom if denom else 0.0
    return out


def macro_f1(rows: list[Row]) -> tuple[float | None, dict[str, float]]:
    # each type keeps its own class space: choice labels, noul true/false, score levels;
    # the overall value averages over all (type, class) pairs that occur in targets or predictions
    per_type: dict[str, list[tuple[str, str]]] = {}
    for r in rows:
        y = argmax(as_dist(r.qtype, r.target), r.labels)
        yhat = argmax(as_dist(r.qtype, r.pred), r.labels)
        per_type.setdefault(r.qtype, []).append((y, yhat))
    by_type, pooled = {}, []
    for qtype, pairs in per_type.items():
        f1s = _f1_per_class(pairs)
        by_type[qtype] = sum(f1s.values()) / len(f1s)
        pooled.extend(f1s.values())
    return (sum(pooled) / len(pooled) if pooled else None), by_type


def max_abs_dp(rows: list[Row], attr: str) -> float | None:
    best = None
    for r in rows:
        other = getattr(r, attr)
        if other is None:
            continue
        a, b = as_dist(r.qtype, r.pred), as_dist(r.qtype, other)
        d = max(abs(a[k] - b.get(k, 0.0)) for k in a)
        best = d if best is None else max(best, d)
    return best


def order_flip_rate(rows: list[Row]) -> float | None:
    rel = [r for r in rows if r.qtype in ("choice", "score") and r.pred_reversed is not None]
    if not rel:
        return None
    flips = sum(argmax(as_dist(r.qtype, r.pred), r.labels) != argmax(as_dist(r.qtype, r.pred_reversed), r.labels)
                for r in rel)
    return flips / len(rel)


def argmax_flip_rate(rows: list[Row], attr: str) -> float | None:
    rel = [r for r in rows if getattr(r, attr) is not None]
    if not rel:
        return None
    flips = sum(argmax(as_dist(r.qtype, r.pred), r.labels) != argmax(as_dist(r.qtype, getattr(r, attr)), r.labels)
                for r in rel)
    return flips / len(rel)


def negation_gap(rows: list[Row]) -> float | None:
    rel = [r for r in rows if r.qtype == "noul" and r.pred_negated is not None]
    if not rel:
        return None
    return sum(abs(r.pred + r.pred_negated - 1.0) for r in rel) / len(rel)


def _quantile(sorted_vals: list[float], q: float) -> float:
    pos = (len(sorted_vals) - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def latency_summary(latencies: list[float]) -> dict:
    if not latencies:
        return {"p50": None, "p95": None, "mean": None}
    s = sorted(latencies)
    return {"p50": _quantile(s, 0.5), "p95": _quantile(s, 0.95), "mean": sum(s) / len(s)}


def errors_summary(errors: list[dict]) -> dict:
    by_status = Counter(str(e.get("status")) for e in errors)
    return {"count": len(errors), "by_status": dict(by_status), "samples": errors[:5]}


def compute_metrics(rows: list[Row], request_latencies: list[float] | None = None,
                    errors: list[dict] | None = None, n_boot: int = 1000, noise_reps: int = 200,
                    seed: int = 0) -> dict:
    if request_latencies is None:
        # one main request per record: count its latency once
        seen: dict[str, float] = {}
        for r in rows:
            if r.latency_ms is not None:
                seen.setdefault(r.record_id, r.latency_ms)
        request_latencies = list(seen.values())
    out: dict[str, Any] = {
        "n_records": len({r.record_id for r in rows}),
        "n_questions": len(rows),
        "by_type": dict(Counter(r.qtype for r in rows)),
        "latency_ms": latency_summary(request_latencies),
        "errors": errors_summary(errors or []),
        "order_flip_rate": order_flip_rate(rows),
        "batched_vs_separate_max_dp": max_abs_dp(rows, "pred_separate"),
        "repeat_max_dp": max_abs_dp(rows, "pred_repeat"),
        "jitter_max_dp": max_abs_dp(rows, "pred_jitter"),
        "jitter_flip_rate": argmax_flip_rate(rows, "pred_jitter"),
        "negation_gap": negation_gap(rows),
    }
    if not rows:
        for k in ("accuracy", "macro_f1", "brier", "nll", "ece", "ece_maxprob", "ece_ci95", "ece_noise_floor", "reliability",
                  "coverage_accuracy", "auroc_confidence", "confidence_distinct"):
            out[k] = None
        out["macro_f1_by_type"] = {}
        out["confidence_one_bucket"] = {"n": 0, "errors": 0}
        return out

    confs = [float(r.confidence) for r in rows]
    correct = [is_correct(r) for r in rows]
    e, reliability = ece(confs, correct)
    # ECE against max-probability: the spec's rescaled `confidence` is not P(correct) for choice/score,
    # so this is the number to read when judging calibration of the distribution itself.
    maxp = [max(as_dist(r.qtype, r.pred).values()) for r in rows]
    e_maxp, _ = ece(maxp, correct)
    f1, f1_by_type = macro_f1(rows)
    ones = [ok for c, ok in zip(confs, correct) if c >= 1.0 - ONE_TOL]
    out.update({
        "accuracy": sum(correct) / len(rows),
        "macro_f1": f1,
        "macro_f1_by_type": f1_by_type,
        "brier": sum(brier(r) for r in rows) / len(rows),
        "nll": sum(nll(r) for r in rows) / len(rows),
        "ece": e,
        "ece_maxprob": e_maxp,
        "ece_ci95": ece_bootstrap_ci(confs, correct, n_boot, seed),
        "ece_noise_floor": ece_noise_floor(rows, noise_reps, seed),
        "reliability": reliability,
        "coverage_accuracy": coverage_accuracy(confs, correct),
        "auroc_confidence": auroc(confs, correct),
        "confidence_distinct": len({round(c, 12) for c in confs}),
        "confidence_one_bucket": {"n": len(ones), "errors": len(ones) - sum(ones)},
    })
    return out
