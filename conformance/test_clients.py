"""Real clients against the target: official Python SDK and langchain-typesafe."""

from __future__ import annotations

import pytest
from conftest import API_KEY, BASE_URL, MODEL, STATE_TICKET, record


def test_official_python_sdk():
    typesafe_sdk = pytest.importorskip("typesafe_sdk")
    from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

    client = TypeSafeClient(api_key=API_KEY, base_url=BASE_URL, model=MODEL)
    resp = client.system_one(
        state=STATE_TICKET,
        questions={
            "is_urgent": Noul(instructions="Does this message convey urgency?",
                              criteria={"true": "Explicitly time-sensitive", "false": "No urgency expressed"}),
            "department": Choice(instructions="Which team should handle this?",
                                 criteria={"billing": "Payments, invoicing, refunds",
                                           "technical": "Bugs, outages, integrations",
                                           "sales": "Pricing, upgrades, new accounts"}),
            "frustration": Score(instructions="How frustrated is the customer?",
                                 criteria=["Calm", "Frustrated", "Very angry"]),
        },
    )
    answers = resp.answers
    urgent = answers["is_urgent"]
    dept = answers["department"]
    frus = answers["frustration"]
    record("sdk.python.version", getattr(typesafe_sdk, "__version__", "?"))
    record("sdk.python.answers", {
        "is_urgent": float(urgent.noul),
        "department": {"choice": dept.choice, "confidence": float(dept.confidence), "probabilities": dict(dept.probabilities)},
        "frustration": {"score": float(frus.score), "confidence": float(frus.confidence)},
    })
    assert 0.0 <= float(urgent.noul) <= 1.0
    assert dept.choice in dept.probabilities
    assert 0.0 <= float(frus.score) <= 2.0


def test_official_python_sdk_list_models():
    pytest.importorskip("typesafe_sdk")
    from typesafe_sdk import TypeSafeClient

    client = TypeSafeClient(api_key=API_KEY, base_url=BASE_URL, model=MODEL)
    fn = getattr(client, "list_models", None) or getattr(client, "models", None)
    if fn is None:
        pytest.skip("SDK has no list_models")
    resp = fn() if callable(fn) else fn
    record("sdk.python.models", str(resp)[:500])
    assert resp is not None


def test_langchain_typesafe(monkeypatch):
    lc = pytest.importorskip("langchain_typesafe")
    monkeypatch.setenv("TYPESAFE_BASE_URL", BASE_URL)
    monkeypatch.setenv("TYPESAFE_API_KEY", API_KEY)
    from langchain_typesafe import TypeSafeClassifier

    Noul = getattr(lc, "Noul", None)
    Choice = getattr(lc, "Choice", None)
    Score = getattr(lc, "Score", None)
    if Noul is None:
        from langchain_typesafe.types import Choice, Noul, Score  # type: ignore[no-redef]

    clf = TypeSafeClassifier(model=MODEL)
    resp = clf.invoke({
        "state": STATE_TICKET,
        "questions": {
            "is_urgent": Noul(instructions="Does this message convey urgency?"),
            "department": Choice(instructions="Which team should handle this?",
                                 criteria={"billing": "Payments, invoicing, refunds",
                                           "technical": "Bugs, outages, integrations",
                                           "sales": "Pricing, upgrades, new accounts"}),
            "frustration": Score(instructions="How frustrated is the customer?",
                                 criteria=["Calm", "Frustrated", "Very angry"]),
        },
    })
    record("langchain.answers", {
        "is_urgent": float(resp.nouls["is_urgent"].noul),
        "department": resp.choices["department"].choice,
        "frustration": float(resp.scores["frustration"].score),
    })
    assert resp.choices["department"].choice in ("billing", "technical", "sales")
