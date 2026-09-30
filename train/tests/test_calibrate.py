import json
import math
import random
from pathlib import Path

from train import calibrate


def _softmax(logits):
    m = max(logits)
    z = sum(math.exp(v - m) for v in logits)
    return [math.exp(v - m) / z for v in logits]


def _synth(tmp_path: Path, n: int = 300, true_t: float = 2.5, seed: int = 0):
    """Records + a results file whose predictions are over-confident by a known factor true_t."""
    rng = random.Random(seed)
    records, raw = [], []
    for i in range(n):
        labels = ["A", "B", "C", "D"]
        # calibrated logits -> sample the label from them -> report over-sharpened probabilities
        base = [rng.gauss(0, 1) for _ in labels]
        p_true = _softmax(base)
        ans = rng.choices(labels, weights=p_true)[0]
        p_over = _softmax([v * true_t for v in base])
        p_yes_true = 1 / (1 + math.exp(-rng.gauss(0, 1.2)))
        yes = rng.random() < p_yes_true
        lo = math.log(p_yes_true / (1 - p_yes_true)) * true_t
        p_yes_over = 1 / (1 + math.exp(-lo))
        rid = f"syn/{i}"
        records.append({"id": rid, "source": "syn", "split": "calibration", "license": "commercial-ok", "lang": "en",
                        "state": f"item {i}",
                        "questions": {"pick": {"type": "choice", "instructions": "pick", "criteria": {k: None for k in labels}},
                                      "yes": {"type": "noul", "instructions": "yes?"}},
                        "targets": {"pick": {k: 1.0 if k == ans else 0.0 for k in labels}, "yes": 1.0 if yes else 0.0},
                        "meta": {"grid_id": rid}})
        raw.append({"id": rid, "questions": {
            "pick": {"type": "choice", "pred": dict(zip(labels, p_over)), "confidence": max(p_over)},
            "yes": {"type": "noul", "pred": p_yes_over, "confidence": max(p_yes_over, 1 - p_yes_over)}}})
    rec_path = tmp_path / "calib.jsonl"
    rec_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    res_path = tmp_path / "res.json"
    res_path.write_text(json.dumps({"target": "syn", "raw": raw, "metrics": {}}), encoding="utf-8")
    return rec_path, res_path


def test_recovers_overconfidence_temperature(tmp_path):
    rec, res = _synth(tmp_path)
    out = tmp_path / "t.json"
    calibrate.main(["--results", str(res), "--records", str(rec), "--out", str(out)])
    fit = json.loads(out.read_text(encoding="utf-8"))
    for qtype in ("choice", "noul"):
        t = fit["temperatures"][qtype]
        r = fit["fit"][qtype]
        assert 1.8 <= t <= 3.4, (qtype, t)  # true sharpening factor was 2.5
        assert r["nll_after"] < r["nll_before"]
        assert r["brier_after"] < r["brier_before"]
    # ECE on the spec's rescaled confidence is not a proper calibration measure for choice, so only
    # the noul ECE (confidence = max(p, 1-p) = P(correct) when calibrated) must improve.
    assert fit["fit"]["noul"]["ece_after"] < fit["fit"]["noul"]["ece_before"]


def test_temper_identity_and_direction():
    d = {"A": 0.7, "B": 0.2, "C": 0.1}
    same = calibrate.temper(d, 1.0)
    assert all(abs(same[k] - d[k]) < 1e-9 for k in d)
    flatter = calibrate.temper(d, 3.0)
    assert flatter["A"] < d["A"] and abs(sum(flatter.values()) - 1) < 1e-9
