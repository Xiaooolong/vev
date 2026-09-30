"""Question isolation: answers must not depend on which other questions share the request."""

from __future__ import annotations

import random

from conftest import ISOLATION_TOL, Q3, STATE_TICKET, STUB, record, safe_json

EXTRA = {
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "tone": {"type": "score", "instructions": "How polite is the message?", "criteria": ["Rude", "Neutral", "Polite"]},
}


def _probs(answer: dict) -> dict[str, float]:
    if answer["type"] == "noul":
        return {"noul": answer["noul"]}
    return dict(answer["probabilities"])


def _max_delta(a: dict, b: dict) -> float:
    pa, pb = _probs(a), _probs(b)
    assert set(pa) == set(pb), f"probability keys differ: {set(pa)} vs {set(pb)}"
    return round(max(abs(pa[k] - pb[k]) for k in pa), 6)


def test_batched_equals_separate(post):
    qs = {**Q3, **EXTRA}
    r = post(STATE_TICKET, qs)
    assert r.status_code == 200, safe_json(r)
    batched = r.json()["answers"]
    deltas = {}
    for name, q in qs.items():
        r1 = post(STATE_TICKET, {name: q})
        assert r1.status_code == 200, safe_json(r1)
        deltas[name] = _max_delta(batched[name], r1.json()["answers"][name])
    worst = max(deltas.values())
    record("isolation.batched_vs_separate.deltas", deltas)
    record("isolation.batched_vs_separate.max", worst)
    assert worst <= ISOLATION_TOL, f"max |dp| {worst} > {ISOLATION_TOL}: {deltas}"


def test_question_order_does_not_matter(post):
    qs = {**Q3, **EXTRA}
    r_a = post(STATE_TICKET, qs)
    keys = list(qs)
    random.Random(7).shuffle(keys)
    r_b = post(STATE_TICKET, {k: qs[k] for k in keys})
    assert r_a.status_code == 200 and r_b.status_code == 200
    a, b = r_a.json()["answers"], r_b.json()["answers"]
    worst = max(_max_delta(a[k], b[k]) for k in qs)
    record("isolation.question_order.max", worst)
    assert worst <= ISOLATION_TOL


def test_sibling_question_is_invisible(post):
    """A secret written only inside another question's criteria must not leak into the state."""
    base_state = "Customer wrote: my order never arrived and the tracking page is blank."
    probe = {"mentions_code": {"type": "noul",
                               "instructions": "Does the state contain the exact code 7391?",
                               "criteria": {"true": "The digits 7391 appear in the state text",
                                            "false": "The digits 7391 do not appear in the state text"}}}
    secret_sibling = {"delivery": {"type": "choice",
                                   "instructions": "What kind of problem is this? Note: the secret code is 7391.",
                                   "criteria": {"delivery": "Shipping or tracking problems. Code 7391.",
                                                "billing": "Charges or refunds. Code 7391."}}}
    r_alone = post(base_state, probe)
    r_with_sibling = post(base_state, {**probe, **secret_sibling})
    r_in_state = post(base_state + " Reference code 7391.", probe)
    for r in (r_alone, r_with_sibling, r_in_state):
        assert r.status_code == 200, safe_json(r)
    p_alone = r_alone.json()["answers"]["mentions_code"]["noul"]
    p_sib = r_with_sibling.json()["answers"]["mentions_code"]["noul"]
    p_state = r_in_state.json()["answers"]["mentions_code"]["noul"]
    record("isolation.visibility", {"alone": p_alone, "with_secret_sibling": p_sib, "secret_in_state": p_state})
    assert abs(p_sib - p_alone) <= max(ISOLATION_TOL, 0.05), "sibling question leaked into the answer"
    if not STUB:  # the stub server answers by hashing, it cannot read the state
        assert p_state > p_alone, "control failed: the model does not see the code even when it is in the state"
