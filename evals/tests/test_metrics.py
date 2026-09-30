import math
import random

import pytest

from evals.metrics import Row, auroc, compute_metrics, ece, ece_noise_floor, is_correct, macro_f1


def choice_row(i, pred, target, conf, **kw):
    return Row(record_id=f"r{i}", qname="q", qtype="choice", pred=pred, target=target, confidence=conf, **kw)


def test_all_correct_full_confidence():
    rows = [
        choice_row(0, {"A": 1.0, "B": 0.0}, {"A": 1.0, "B": 0.0}, 1.0, latency_ms=10),
        choice_row(1, {"A": 0.0, "B": 1.0}, {"A": 0.0, "B": 1.0}, 1.0, latency_ms=20),
        Row("r2", "q", "noul", 1.0, 1.0, 1.0, latency_ms=30),
        Row("r3", "q", "score", {"0": 0.0, "1": 0.0, "2": 1.0}, {"0": 0.0, "1": 0.0, "2": 1.0}, 1.0, latency_ms=40),
    ]
    m = compute_metrics(rows, n_boot=50, noise_reps=20)
    assert m["accuracy"] == 1.0
    assert m["ece"] == 0.0
    assert m["brier"] == 0.0
    assert m["nll"] == pytest.approx(0.0, abs=1e-9)
    assert m["ece_ci95"] == [0.0, 0.0]
    assert m["ece_noise_floor"] == 0.0
    assert m["confidence_one_bucket"] == {"n": 4, "errors": 0}
    assert m["confidence_distinct"] == 1
    assert m["macro_f1"] == 1.0
    assert m["auroc_confidence"] is None  # no negatives
    assert m["by_type"] == {"choice": 2, "noul": 1, "score": 1}
    assert m["latency_ms"]["p50"] == 25.0
    assert m["reliability"][9] == [1.0, 1.0, 4]
    assert m["errors"]["count"] == 0


def test_hand_computed_values():
    rows = [
        Row("r0", "q", "noul", 0.8, 1.0, 0.8),  # correct, conf .8
        Row("r1", "q", "noul", 0.3, 1.0, 0.7),  # wrong, conf .7
        Row("r2", "q", "noul", 0.1, 0.0, 0.9),  # correct, conf .9
        Row("r3", "q", "noul", 0.6, 0.0, 0.6),  # wrong, conf .6
    ]
    m = compute_metrics(rows, n_boot=50, noise_reps=20)
    assert m["accuracy"] == 0.5
    # brier summed over both classes: 2 * (p - t)^2
    assert m["brier"] == pytest.approx((2 * 0.04 + 2 * 0.49 + 2 * 0.01 + 2 * 0.36) / 4)
    assert m["nll"] == pytest.approx(-(math.log(0.8) + math.log(0.3) + math.log(0.9) + math.log(0.4)) / 4)
    # each confidence sits alone in its bin: .6 wrong, .7 wrong, .8 right, .9 right
    assert m["ece"] == pytest.approx((0.6 + 0.7 + 0.2 + 0.1) / 4)
    assert m["auroc_confidence"] == 1.0
    cov = {t: (c, a) for t, c, a in m["coverage_accuracy"]}
    assert len(m["coverage_accuracy"]) == 50 and m["coverage_accuracy"][-1][0] == 0.99
    assert cov[0.5] == (1.0, 0.5)
    assert cov[0.75] == (0.5, 1.0)
    assert cov[0.95] == (0.0, None)
    # true: tp1 fp1 fn1 -> .5 ; false: tp1 fp1 fn1 -> .5
    assert m["macro_f1"] == pytest.approx(0.5)


def test_auroc_ties_and_ordering():
    assert auroc([0.1, 0.2, 0.3, 0.4], [False, False, True, True]) == 1.0
    assert auroc([0.1, 0.2, 0.3, 0.4], [True, True, False, False]) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [True, False, True, False]) == 0.5
    assert auroc([0.1, 0.4, 0.35, 0.8], [False, False, True, True]) == pytest.approx(0.75)


def test_macro_f1_choice_labels():
    rows = [
        choice_row(0, {"A": 0.9, "B": 0.1, "C": 0.0}, {"A": 1, "B": 0, "C": 0}, 0.8),
        choice_row(1, {"A": 0.9, "B": 0.1, "C": 0.0}, {"A": 0, "B": 1, "C": 0}, 0.8),
        choice_row(2, {"A": 0.1, "B": 0.1, "C": 0.8}, {"A": 0, "B": 0, "C": 1}, 0.8),
    ]
    f1, by_type = macro_f1(rows)
    # A: tp1 fp1 -> 2/3 ; B: fn1 -> 0 ; C: 1
    assert f1 == pytest.approx((2 / 3 + 0 + 1) / 3)
    assert by_type["choice"] == pytest.approx(f1)


def test_argmax_tie_breaks_by_criteria_order():
    # predicted A/B tie resolves to A (first in criteria order), which matches the target
    r = choice_row(0, {"B": 0.5, "A": 0.5}, {"A": 1.0, "B": 0.0}, 0.0, labels=["A", "B"])
    assert compute_metrics([r], n_boot=10, noise_reps=5)["accuracy"] == 1.0


def test_perturbation_metrics():
    rows = [
        choice_row(0, {"A": 0.7, "B": 0.3}, {"A": 1, "B": 0}, 0.4,
                   pred_reversed={"A": 0.4, "B": 0.6}, pred_separate={"A": 0.69, "B": 0.31},
                   pred_repeat={"A": 0.7, "B": 0.3}),
        Row("r1", "q", "score", {"0": 0.1, "1": 0.9}, {"0": 0, "1": 1}, 0.8,
            pred_reversed={"0": 0.2, "1": 0.8}, pred_separate={"0": 0.1, "1": 0.9}, pred_repeat={"0": 0.15, "1": 0.85}),
        Row("r2", "q", "noul", 0.7, 1.0, 0.7, pred_separate=0.7, pred_negated=0.4),
        Row("r3", "q", "noul", 0.2, 0.0, 0.8, pred_negated=0.7),
    ]
    m = compute_metrics(rows, n_boot=10, noise_reps=5)
    assert m["order_flip_rate"] == 0.5
    assert m["batched_vs_separate_max_dp"] == pytest.approx(0.01)
    assert m["repeat_max_dp"] == pytest.approx(0.05)
    assert m["negation_gap"] == pytest.approx((0.1 + 0.1) / 2)


def test_perturbations_absent_are_none():
    m = compute_metrics([Row("r0", "q", "noul", 0.7, 1.0, 0.7)], n_boot=10, noise_reps=5)
    for k in ("order_flip_rate", "batched_vs_separate_max_dp", "repeat_max_dp", "negation_gap"):
        assert m[k] is None


def test_ece_bins_and_bootstrap_ci_contains_point():
    rng = random.Random(1)
    rows = []
    for i in range(300):
        p = rng.uniform(0.5, 1.0)
        rows.append(Row(f"r{i}", "q", "noul", p, 1.0 if rng.random() < 0.8 else 0.0, p))
    m = compute_metrics(rows, n_boot=200, noise_reps=20)
    lo, hi = m["ece_ci95"]
    assert lo <= m["ece"] <= hi
    assert sum(b[2] for b in m["reliability"]) == 300
    e, _ = ece([0.05, 1.0], [False, True])
    assert e == pytest.approx(0.025)


def _draw(rng: random.Random, dist: dict[str, float]) -> str:
    u, acc = rng.random(), 0.0
    for k, p in dist.items():
        acc += p
        if u < acc:
            return k
    return k


@pytest.mark.parametrize("qtype", ["noul", "choice"])
def test_calibrated_model_ece_matches_noise_floor(qtype):
    # Latent distributions q_i. A calibrated model predicts q_i and the observed label is drawn
    # from q_i; its expected ECE must match the noise floor computed with q_i as targets.
    rng = random.Random(7)
    latent = []
    for _ in range(120):
        if qtype == "noul":
            latent.append(rng.random())
        else:
            w = [rng.random() ** 2 for _ in range(3)]
            latent.append({k: v / sum(w) for k, v in zip("ABC", w)})

    def conf(d):
        if qtype == "noul":
            return max(d, 1 - d)
        return (max(d.values()) - 1 / 3) / (1 - 1 / 3)

    floor = ece_noise_floor([Row(f"r{i}", "q", qtype, d, d, conf(d)) for i, d in enumerate(latent)], reps=400, seed=3)
    draws = []
    for _ in range(400):
        rows = []
        for i, d in enumerate(latent):
            if qtype == "noul":
                target = 1.0 if rng.random() < d else 0.0
            else:
                lab = _draw(rng, d)
                target = {k: float(k == lab) for k in "ABC"}
            rows.append(Row(f"r{i}", "q", qtype, d, target, conf(d)))
        draws.append(ece([r.confidence for r in rows], [is_correct(r) for r in rows])[0])
    model_ece = sum(draws) / len(draws)
    assert floor > 0.01
    assert model_ece == pytest.approx(floor, rel=0.1)


def test_noise_floor_shrinks_with_n():
    rng = random.Random(0)
    small = [Row(f"r{i}", "q", "noul", p, p, max(p, 1 - p)) for i, p in enumerate(rng.random() for _ in range(30))]
    big = [Row(f"r{i}", "q", "noul", p, p, max(p, 1 - p)) for i, p in enumerate(rng.random() for _ in range(3000))]
    assert ece_noise_floor(big, reps=30) < ece_noise_floor(small, reps=30)


def test_empty_rows():
    m = compute_metrics([], errors=[{"status": 500, "id": "x"}])
    assert m["accuracy"] is None and m["errors"]["count"] == 1 and m["n_questions"] == 0
