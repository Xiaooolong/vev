"""Identical requests must return identical probabilities."""

from __future__ import annotations

from conftest import DETERMINISM_TOL, Q3, STATE_TICKET, record, safe_json


def _flat(answers: dict) -> list[float]:
    """Probabilities in a key-stable order (the API does not guarantee key order)."""
    out: list[float] = []
    for name in sorted(answers):
        a = answers[name]
        if a["type"] == "noul":
            out.append(a["noul"])
        else:
            probs = a["probabilities"]
            out.extend(probs[k] for k in sorted(probs))
    return out


def test_repeat_five_times(post):
    runs = []
    latencies = []
    for _ in range(5):
        r = post(STATE_TICKET, Q3)
        assert r.status_code == 200, safe_json(r)
        runs.append(_flat(r.json()["answers"]))
        latencies.append(r.elapsed_ms)
    worst = round(max(abs(x - y) for run in runs[1:] for x, y in zip(runs[0], run, strict=True)), 6)
    distinct = len({tuple(run) for run in runs})
    record("determinism.max_delta", worst)
    record("determinism.distinct_outputs", distinct)
    record("determinism.latencies_ms", latencies)
    assert worst <= DETERMINISM_TOL, f"max |dp| across repeats {worst} > {DETERMINISM_TOL}"
