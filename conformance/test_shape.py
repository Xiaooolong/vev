"""Request/response shape: what every /v1/systemone implementation must get right."""

from __future__ import annotations

import pytest
from conftest import (
    Q3,
    STATE_TICKET,
    SUPPORTS_IMAGES,
    check_response,
    record,
    safe_json,
    tiny_png_data_url,
)


def test_models_endpoint(client):
    r = client.get("/v1/models")
    record("models.status", r.status_code)
    record("models.body", safe_json(r))
    assert r.status_code == 200
    body = r.json()
    cards = body["models"] if isinstance(body, dict) and "models" in body else body
    assert isinstance(cards, list) and cards, "no model cards"
    assert all(isinstance(c.get("name"), str) for c in cards)


def test_healthz(client):
    r = client.get("/healthz")
    record("healthz.status", r.status_code)
    if r.status_code == 404:
        pytest.skip("no /healthz (optional)")
    assert r.status_code == 200


def test_three_types(post):
    r = post(STATE_TICKET, Q3)
    record("three_types.status", r.status_code)
    record("three_types.latency_ms", r.elapsed_ms)
    record("three_types.body", safe_json(r))
    assert r.status_code == 200, safe_json(r)
    check_response(r.json(), Q3)


def test_request_id_header(post):
    r = post(STATE_TICKET, {"is_urgent": Q3["is_urgent"]})
    rid = r.headers.get("x-typesafe-request-id")
    record("request_id_header", rid)
    assert r.status_code == 200
    assert rid, "x-typesafe-request-id header missing"


def test_model_alias_jev_latest(post):
    r = post(STATE_TICKET, {"is_urgent": Q3["is_urgent"]}, model="jev-latest")
    record("alias.jev-latest.status", r.status_code)
    assert r.status_code == 200, safe_json(r)
    record("alias.jev-latest.model_echo", r.json().get("model"))


def test_model_field_missing(post):
    r = post(STATE_TICKET, {"is_urgent": Q3["is_urgent"]}, model=None)
    record("model_missing.status", r.status_code)
    assert r.status_code in (200, 422), safe_json(r)


def test_state_object(post):
    state = {"ticket": {"subject": "Payouts failing", "body": STATE_TICKET}, "plan": "pro"}
    r = post(state, Q3)
    record("state_object.status", r.status_code)
    assert r.status_code == 200, safe_json(r)
    check_response(r.json(), Q3)


def test_state_array(post):
    state = [{"from": "customer", "text": STATE_TICKET}, {"from": "agent", "text": "Looking into it."}]
    r = post(state, Q3)
    record("state_array.status", r.status_code)
    assert r.status_code == 200, safe_json(r)
    check_response(r.json(), Q3)


def test_state_null(post):
    r = post(None, {"is_urgent": Q3["is_urgent"]})
    record("state_null.status", r.status_code)
    record("state_null.body", safe_json(r))
    assert r.status_code in (200, 422)


def test_missing_instructions(post):
    q = {"is_urgent": {"type": "noul", "criteria": Q3["is_urgent"]["criteria"]}}
    r = post(STATE_TICKET, q)
    record("missing_instructions.status", r.status_code)
    record("missing_instructions.body", safe_json(r))
    # The spec makes instructions optional; the official docs mark it required. Both are recorded.
    assert r.status_code in (200, 422)


def test_noul_without_criteria(post):
    q = {"is_urgent": {"type": "noul", "instructions": "Does this message convey urgency?"}}
    r = post(STATE_TICKET, q)
    record("noul_no_criteria.status", r.status_code)
    assert r.status_code == 200, safe_json(r)
    check_response(r.json(), q)


def test_structured_criteria(post):
    q = {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": {"what": "Payments, invoicing, refunds", "examples": ["double charge", "refund"]},
                "technical": {"what": "Bugs, outages, integrations", "not_for": "pricing questions"},
                "sales": None,
            },
        }
    }
    r = post(STATE_TICKET, q)
    record("structured_criteria.status", r.status_code)
    assert r.status_code == 200, safe_json(r)
    check_response(r.json(), q)


def test_choice_single_option(post):
    q = {"only": {"type": "choice", "instructions": "Pick.", "criteria": {"a": "the only option"}}}
    r = post(STATE_TICKET, q)
    record("choice_single.status", r.status_code)
    record("choice_single.body", safe_json(r))
    assert r.status_code in (200, 422)
    if r.status_code == 200:
        ans = r.json()["answers"]["only"]
        assert abs(ans["probabilities"]["a"] - 1.0) <= 1e-6


def test_choice_fifty_options(post):
    labels = {f"team_{i:02d}": f"Team number {i}" for i in range(50)}
    labels["technical"] = "Bugs, outages, integrations, failed payouts"
    q = {"dept": {"type": "choice", "instructions": "Which team should handle this?", "criteria": labels}}
    r = post(STATE_TICKET, q)
    record("choice_fifty.status", r.status_code)
    record("choice_fifty.latency_ms", r.elapsed_ms)
    assert r.status_code == 200, safe_json(r)
    check_response(r.json(), q)
    record("choice_fifty.top", r.json()["answers"]["dept"]["choice"])


def test_score_ten_levels(post):
    q = {"anger": {"type": "score", "instructions": "How angry is the customer, 0 = calm?", "criteria": [f"level {i}" for i in range(10)]}}
    r = post(STATE_TICKET, q)
    record("score_ten.status", r.status_code)
    assert r.status_code == 200, safe_json(r)
    check_response(r.json(), q)


def test_many_questions_one_request(post):
    qs = {f"q{i}": {"type": "noul", "instructions": f"Does the message mention topic number {i}?"} for i in range(20)}
    qs.update(Q3)
    r = post(STATE_TICKET, qs)
    record("many_questions.status", r.status_code)
    record("many_questions.latency_ms", r.elapsed_ms)
    assert r.status_code == 200, safe_json(r)
    check_response(r.json(), qs)


def test_image_object_in_state(post):
    """A text-only server must still accept the reserved image object as ordinary JSON."""
    state = {"screenshot": {"image": {"url": tiny_png_data_url()}}, "note": "a black square on grey"}
    q = {"has_square": {"type": "noul", "instructions": "Is there a black square in the image?"}}
    r = post(state, q)
    record("image_object.status", r.status_code)
    record("image_object.body", safe_json(r))
    assert r.status_code == 200, safe_json(r)
    check_response(r.json(), q)
    record("image_object.supports_images", SUPPORTS_IMAGES)
