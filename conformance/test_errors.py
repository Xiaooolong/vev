"""Error handling. Status codes are asserted where the spec is firm; bodies are recorded."""

from __future__ import annotations

import httpx
import pytest
from conftest import BASE_URL, EXPECT_AUTH, MODEL, Q3, STATE_TICKET, record, safe_json


def _rec(key, r):
    record(f"{key}.status", r.status_code)
    record(f"{key}.body", safe_json(r))


def test_missing_questions(post):
    r = post(STATE_TICKET, None, raw_body={"state": STATE_TICKET, "model": MODEL})
    _rec("err.missing_questions", r)
    assert 400 <= r.status_code < 500


def test_empty_questions(post):
    r = post(STATE_TICKET, {})
    _rec("err.empty_questions", r)
    assert r.status_code in (200, 422)


def test_unknown_question_type(post):
    r = post(STATE_TICKET, {"q": {"type": "rank", "instructions": "Rank these."}})
    _rec("err.unknown_type", r)
    assert r.status_code == 422 or 400 <= r.status_code < 500


def test_choice_zero_options(post):
    r = post(STATE_TICKET, {"q": {"type": "choice", "instructions": "Pick.", "criteria": {}}})
    _rec("err.choice_zero", r)
    assert 400 <= r.status_code < 500


def test_choice_missing_criteria(post):
    r = post(STATE_TICKET, {"q": {"type": "choice", "instructions": "Pick."}})
    _rec("err.choice_no_criteria", r)
    assert 400 <= r.status_code < 500


def test_score_one_level(post):
    r = post(STATE_TICKET, {"q": {"type": "score", "instructions": "Rate.", "criteria": ["only"]}})
    _rec("err.score_one_level", r)
    # official API accepts a single level (probability 1.0); docs say "at least two"
    assert r.status_code in (200, 400, 422)
    if r.status_code == 200:
        assert abs(r.json()["answers"]["q"]["probabilities"]["0"] - 1.0) <= 1e-6


def test_score_eleven_levels(post):
    r = post(STATE_TICKET, {"q": {"type": "score", "instructions": "Rate.", "criteria": [f"l{i}" for i in range(11)]}})
    _rec("err.score_eleven", r)
    # the official API and Vev cap at 10 levels and answer 400
    assert r.status_code in (200, 400, 422)


def test_choice_256_options(post):
    labels = {f"o{i}": f"option {i}" for i in range(256)}
    r = post(STATE_TICKET, {"q": {"type": "choice", "instructions": "Pick.", "criteria": labels}})
    _rec("err.choice_256", r)
    assert r.status_code in (400, 422)


def test_unknown_model(post):
    r = post(STATE_TICKET, {"is_urgent": Q3["is_urgent"]}, model="no-such-model-xyz")
    _rec("err.unknown_model", r)
    assert 400 <= r.status_code < 500


def test_malformed_json(client):
    r = client.post("/v1/systemone", content=b'{"state": "x", "questions": {', headers={"Content-Type": "application/json"})
    _rec("err.malformed_json", r)
    assert 400 <= r.status_code < 500


def test_wrong_api_key():
    with httpx.Client(base_url=BASE_URL, timeout=60) as c:
        r = c.post(
            "/v1/systemone",
            json={"state": STATE_TICKET, "model": MODEL, "questions": {"is_urgent": Q3["is_urgent"]}},
            headers={"Authorization": "Bearer definitely-wrong-key-000"},
        )
    _rec("err.wrong_key", r)
    if EXPECT_AUTH:
        assert r.status_code == 401
    else:
        assert r.status_code in (200, 401)


def test_error_body_is_sdk_parsable(post):
    """The official SDK extracts messages from error.message / message / detail."""
    r = post(STATE_TICKET, {"q": {"type": "choice", "instructions": "Pick.", "criteria": {}}})
    body = safe_json(r)
    record("err.body_shape", body)
    if r.status_code == 200:
        pytest.skip("target accepted an empty choice; nothing to parse")
    assert isinstance(body, dict), "error body is not JSON"
    err, msg, detail = body.get("error"), body.get("message"), body.get("detail")
    ok = (
        isinstance(err, str)
        or (isinstance(err, dict) and isinstance(err.get("message"), str))
        or isinstance(msg, str)
        or isinstance(detail, str)
        or (isinstance(detail, dict) and isinstance(detail.get("message"), str))
        or isinstance(detail, list)
    )
    assert ok, f"SDK cannot extract a message from {body!r}"
