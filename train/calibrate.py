"""Fit per-type temperatures from evals results (candidate A: zero-training readout + temperature).

    python -m train.calibrate --results results/<target>/calib_*.json --records data/build/v1/calibration.jsonl \
        --out train/temperatures/<target>.json

The results must come from a server that applies no temperature (plain softmax over logits), so a
distribution p can be re-tempered as softmax(log p / T). Temperatures are chosen on a 121-point log grid
0.25..4 by minimum mean NLL per question type (choice / noul / score), the kev recipe.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from evals.metrics import Row, compute_metrics
from evals.schema import iter_records

GRID = [0.25 * (16 ** (i / 120)) for i in range(121)]  # 0.25 .. 4.0, log-spaced
EPS = 1e-12


def as_dist(qtype: str, pred) -> dict[str, float]:
    if qtype == "noul":
        p = min(max(float(pred), EPS), 1 - EPS)
        return {"true": p, "false": 1 - p}
    return {k: max(float(v), EPS) for k, v in pred.items()}


def temper(dist: dict[str, float], t: float) -> dict[str, float]:
    logits = {k: math.log(v) / t for k, v in dist.items()}
    m = max(logits.values())
    z = sum(math.exp(v - m) for v in logits.values())
    return {k: math.exp(v - m) / z for k, v in logits.items()}


def target_dist(qtype: str, target) -> dict[str, float]:
    if qtype == "noul":
        return {"true": float(target), "false": 1 - float(target)}
    return {k: float(v) for k, v in target.items()}


def nll(pred: dict[str, float], target: dict[str, float]) -> float:
    return -sum(t * math.log(max(pred.get(k, EPS), EPS)) for k, t in target.items() if t > 0)


def collect(results_paths: list[Path], records_path: Path) -> dict[str, list[tuple]]:
    records = {r["id"]: r for r in iter_records(records_path)}
    by_type: dict[str, list[tuple]] = defaultdict(list)
    missing = 0
    for path in results_paths:
        res = json.loads(path.read_text(encoding="utf-8"))
        for entry in res["raw"]:
            rec = records.get(entry["id"])
            if rec is None or "error" in entry:
                missing += 1
                continue
            for name, qraw in entry["questions"].items():
                if "pred" not in qraw or qraw.get("error"):
                    continue
                q = rec["questions"].get(name)
                if q is None or q["type"] != qraw.get("type", q["type"]):
                    missing += 1
                    continue
                qtype = q["type"]
                by_type[qtype].append((entry["id"], name, q, as_dist(qtype, qraw["pred"]),
                                       target_dist(qtype, rec["targets"][name]), qraw.get("confidence")))
    if missing:
        print(f"[calibrate] {missing} result entries had no record or an error", file=sys.stderr)
    return by_type


def fit_type(items: list[tuple]) -> tuple[float, dict]:
    best_t, best_nll = 1.0, float("inf")
    for t in GRID:
        total = sum(nll(temper(p, t), tgt) for _, _, _, p, tgt, _ in items) / len(items)
        if total < best_nll:
            best_t, best_nll = t, total
    return best_t, {"nll_before": sum(nll(p, tgt) for _, _, _, p, tgt, _ in items) / len(items), "nll_after": best_nll}


def rows_for(items: list[tuple], qtype: str, t: float | None) -> list[Row]:
    rows = []
    for rid, name, q, p, tgt, _ in items:
        d = temper(p, t) if t else p
        labels = list(q["criteria"]) if qtype == "choice" else ([str(i) for i in range(len(q["criteria"]))] if qtype == "score" else ["true", "false"])
        pred = d["true"] if qtype == "noul" else {k: d[k] for k in labels}
        target = tgt["true"] if qtype == "noul" else {k: tgt.get(k, 0.0) for k in labels}
        if qtype == "noul":
            conf = max(pred, 1 - pred)
        else:
            k = len(labels)
            conf = 1.0 if k <= 1 else (max(pred.values()) - 1 / k) / (1 - 1 / k)
        rows.append(Row(rid, name, qtype, pred, target, conf, labels=labels))
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True, help="evals result JSON files (server run at T=1)")
    ap.add_argument("--records", required=True, help="the records JSONL those results were produced from")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    by_type = collect([Path(p) for p in a.results], Path(a.records))
    if not by_type:
        sys.exit("[calibrate] no usable questions")
    temps, report = {}, {}
    for qtype, items in sorted(by_type.items()):
        t, fit = fit_type(items)
        before = compute_metrics(rows_for(items, qtype, None), n_boot=50, noise_reps=5)
        after = compute_metrics(rows_for(items, qtype, t), n_boot=50, noise_reps=5)
        temps[qtype] = round(t, 4)
        report[qtype] = {"n": len(items), "temperature": round(t, 4), **{k: round(v, 5) for k, v in fit.items()},
                         "ece_before": before["ece"], "ece_after": after["ece"],
                         "brier_before": before["brier"], "brier_after": after["brier"],
                         "accuracy": after["accuracy"]}
        print(f"[calibrate] {qtype}: n={len(items)} T={t:.3f} nll {fit['nll_before']:.4f}->{fit['nll_after']:.4f} "
              f"ece {before['ece']:.4f}->{after['ece']:.4f} brier {before['brier']:.4f}->{after['brier']:.4f}", file=sys.stderr)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"temperatures": temps, "fit": report, "results": a.results, "records": a.records},
                              indent=2), encoding="utf-8")
    print(f"[calibrate] wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
