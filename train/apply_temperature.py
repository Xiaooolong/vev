"""Apply per-type temperatures to evals result files post hoc (candidate A = zero-shot readout + temperature).

    python -m train.apply_temperature --temperatures results/calib/zeroshot-qwen3.5-4b-temperatures.json \
        --records evals/data/text/jevbench.jsonl --results results/zeroshot-qwen3.5-4b/jevbench.json \
        --out results/zeroshot-qwen3.5-4b-calibrated/jevbench.json

Softmax(logits / T) equals p^(1/T) renormalized, so the server need not be re-run. Perturbation predictions
(order / separate / repeat / jitter) are tempered the same way, so the stability metrics stay comparable."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from evals.metrics import Row, compute_metrics
from evals.run import labels_of
from evals.schema import iter_records
from train.calibrate import EPS, temper


def _temper_pred(qtype: str, pred: Any, t: float) -> Any:
    if pred is None:
        return None
    if qtype == "noul":
        p = min(max(float(pred), EPS), 1 - EPS)
        return temper({"true": p, "false": 1 - p}, t)["true"]
    return temper({k: max(float(v), EPS) for k, v in pred.items()}, t)


def _confidence(qtype: str, pred: Any) -> float:
    if qtype == "noul":
        return max(pred, 1 - pred)
    k = len(pred)
    return 1.0 if k <= 1 else (max(pred.values()) - 1 / k) / (1 - 1 / k)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--temperatures", required=True, help="JSON from train.calibrate")
    ap.add_argument("--records", required=True)
    ap.add_argument("--results", required=True, help="evals result JSON produced at T=1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default=None, help="target label for the output (default: <old>-calibrated)")
    a = ap.parse_args(argv)

    temps = json.loads(Path(a.temperatures).read_text(encoding="utf-8"))["temperatures"]
    res = json.loads(Path(a.results).read_text(encoding="utf-8"))
    records = {r["id"]: r for r in iter_records(a.records)}
    rows: list[Row] = []
    latencies: list[float] = []
    errors: list[dict] = []
    skipped = 0
    for entry in res["raw"]:
        rec = records.get(entry["id"])
        if rec is None or "error" in entry:
            errors.append({"id": entry["id"], "error": entry.get("error", "no record")})
            continue
        latencies.append(entry.get("latency_ms", 0.0))
        for name, qraw in entry["questions"].items():
            q = rec["questions"].get(name)
            if q is None or "pred" not in qraw or qraw.get("error") or q["type"] != qraw.get("type", q["type"]):
                skipped += 1
                continue
            qtype = q["type"]
            t = float(temps.get(qtype, 1.0))
            pred = _temper_pred(qtype, qraw["pred"], t)
            qraw["pred"], qraw["confidence"] = pred, _confidence(qtype, pred)
            for key in ("order", "separate", "repeat", "jitter"):  # evals.run stores perturbed predictions under these
                if qraw.get(key) is not None:
                    qraw[key] = _temper_pred(qtype, qraw[key], t)
            rows.append(Row(record_id=entry["id"], qname=name, qtype=qtype, pred=pred, target=rec["targets"][name],
                            confidence=qraw["confidence"], latency_ms=entry.get("latency_ms"), labels=labels_of(q),
                            pred_reversed=qraw.get("order"), pred_separate=qraw.get("separate"),
                            pred_repeat=qraw.get("repeat"), pred_jitter=qraw.get("jitter")))
    if skipped:
        print(f"[apply_temperature] skipped {skipped} questions (errors / type mismatch)", file=sys.stderr)
    res["metrics"] = compute_metrics(rows, latencies, errors)
    res["target"] = a.target or f"{res.get('target', 'model')}-calibrated"
    res["temperatures"] = temps
    res["tempered_from"] = str(a.results)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    m = res["metrics"]
    print(f"[apply_temperature] {out}: n={len(rows)} accuracy={m.get('accuracy')} ece={m.get('ece')} brier={m.get('brier')}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
