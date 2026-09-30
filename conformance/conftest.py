"""Conformance suite for /v1/systemone servers.

Runs against any implementation over plain HTTP. Configure the target with env vars:

    SO_BASE_URL          default http://127.0.0.1:8009
    SO_API_KEY           default: TYPESAFE_API_KEY, else "local"
    SO_MODEL             default jev-latest
    SO_TARGET            label used for the results directory, default "unnamed"
    SO_RESULTS_DIR       default conformance/results
    SO_SUPPORTS_IMAGES   1 = the target is expected to *see* images (our server); 0 = text-only
    SO_EXPECT_AUTH       1 = the target rejects a wrong key with 401
    SO_ISOLATION_TOL     max |dp| allowed between batched and separate questions, default 1e-4
    SO_DETERMINISM_TOL   max |dp| allowed between repeated identical requests, default 0
    SO_TIMEOUT           per-request timeout seconds, default 120
    SO_STUB              1 = the target is conformance/stub_server.py: skip checks that need a real model

Every test records what it saw into a JSON file under SO_RESULTS_DIR/SO_TARGET/, so a run
against a target that fails some tests still leaves a usable observation log.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

BASE_URL = os.environ.get("SO_BASE_URL", "http://127.0.0.1:8009").rstrip("/")
API_KEY = os.environ.get("SO_API_KEY") or os.environ.get("TYPESAFE_API_KEY") or "local"
MODEL = os.environ.get("SO_MODEL", "jev-latest")
TARGET = os.environ.get("SO_TARGET", "unnamed")
RESULTS_DIR = Path(os.environ.get("SO_RESULTS_DIR", str(Path(__file__).parent / "results")))
SUPPORTS_IMAGES = os.environ.get("SO_SUPPORTS_IMAGES", "0") == "1"
EXPECT_AUTH = os.environ.get("SO_EXPECT_AUTH", "0") == "1"
ISOLATION_TOL = float(os.environ.get("SO_ISOLATION_TOL", "1e-4"))
DETERMINISM_TOL = float(os.environ.get("SO_DETERMINISM_TOL", "0"))
TIMEOUT = float(os.environ.get("SO_TIMEOUT", "120"))
STUB = os.environ.get("SO_STUB", "0") == "1"

OBS: dict[str, Any] = {
    "target": TARGET,
    "base_url": BASE_URL,
    "model": MODEL,
    "started": datetime.now(UTC).isoformat(),
    "observations": {},
}


def record(key: str, value: Any) -> None:
    OBS["observations"][key] = value


# --- canonical fixtures shared by several test files --------------------------------

STATE_TICKET = (
    "Help! My payouts have been failing for 3 days. I already emailed support twice and "
    "nobody answered. If this is not fixed by tomorrow I am closing my account."
)

Q3: dict[str, dict[str, Any]] = {
    "is_urgent": {
        "type": "noul",
        "instructions": "Does this message convey urgency?",
        "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
    },
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "billing": "Payments, invoicing, refunds",
            "technical": "Bugs, outages, integrations",
            "sales": "Pricing, upgrades, new accounts",
        },
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated", "Very angry"],
    },
}


def tiny_png_data_url(size: int = 64) -> str:
    """A small solid-colour PNG with a black square, as a data: URI."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), (240, 240, 240))
    ImageDraw.Draw(img).rectangle([size // 4, size // 4, 3 * size // 4, 3 * size // 4], fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# --- answer shape checks ----------------------------------------------------------------

def check_answer(name: str, question: dict[str, Any], answer: dict[str, Any]) -> None:
    qtype = question["type"]
    assert answer.get("type") == qtype, f"{name}: type {answer.get('type')!r} != {qtype!r}"
    if qtype == "noul":
        p = answer["noul"]
        assert isinstance(p, (int, float)) and 0.0 <= p <= 1.0, f"{name}: noul={p!r}"
        return
    probs = answer["probabilities"]
    assert isinstance(probs, dict), f"{name}: probabilities not a dict"
    total = sum(probs.values())
    assert abs(total - 1.0) <= 1e-3, f"{name}: probabilities sum to {total}"
    conf = answer["confidence"]
    assert isinstance(conf, (int, float)) and -1e-9 <= conf <= 1.0 + 1e-9, f"{name}: confidence={conf!r}"
    # Key order of `probabilities` is not guaranteed by the official API (observed to vary
    # between otherwise identical responses), so keys are compared as sets and looked up by name.
    if qtype == "choice":
        labels = set(question["criteria"].keys())
        assert set(probs.keys()) == labels, f"{name}: probability keys {set(probs)} != {labels}"
        best = max(probs.items(), key=lambda kv: kv[1])[0]
        assert answer["choice"] in probs, f"{name}: choice {answer['choice']!r} not a label"
        assert abs(probs[answer["choice"]] - probs[best]) <= 1e-9, f"{name}: choice is not the argmax"
    elif qtype == "score":
        n = len(question["criteria"])
        keys = {str(i) for i in range(n)}
        assert set(probs.keys()) == keys, f"{name}: score probability keys {set(probs)} != {keys}"
        legend = answer["legend"]
        assert set(legend.keys()) == keys, f"{name}: legend keys {set(legend)} != {keys}"
        for i, level in enumerate(question["criteria"]):
            assert legend[str(i)] == level, f"{name}: legend[{i}] {legend[str(i)]!r} != {level!r}"
        expected = sum(i * probs[str(i)] for i in range(n))
        # The official API rounds probabilities to 0.01 but computes `score` from unrounded
        # values; allow the worst-case rounding drift for n levels.
        tol = 0.005 * n * (n - 1) / 2 + 1e-3
        assert abs(answer["score"] - expected) <= tol, f"{name}: score {answer['score']} != E[i]={expected} (tol {tol})"


def check_response(body: dict[str, Any], questions: dict[str, dict[str, Any]]) -> None:
    assert isinstance(body.get("model"), str) and body["model"], "model missing"
    answers = body["answers"]
    assert set(answers.keys()) == set(questions.keys()), f"answer keys {set(answers)} != {set(questions)}"
    for name, q in questions.items():
        check_answer(name, q, answers[name])
    usage = body["usage"]
    assert isinstance(usage["input_tokens"], int) and usage["input_tokens"] >= 0
    assert isinstance(usage["output_tokens"], int) and usage["output_tokens"] >= 0


# --- pytest plumbing ----------------------------------------------------------------------

@pytest.fixture(scope="session")
def client() -> httpx.Client:
    with httpx.Client(
        base_url=BASE_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        timeout=TIMEOUT,
    ) as c:
        yield c


@pytest.fixture(scope="session")
def post(client: httpx.Client):
    def _post(
        state: Any,
        questions: dict[str, Any] | None,
        model: str | None = MODEL,
        headers: dict[str, str] | None = None,
        raw_body: Any = None,
    ) -> httpx.Response:
        if raw_body is None:
            body: dict[str, Any] = {"state": state, "questions": questions}
            if model is not None:
                body["model"] = model
        else:
            body = raw_body
        t0 = time.perf_counter()
        r = client.post("/v1/systemone", json=body, headers=headers)
        r.elapsed_ms = round((time.perf_counter() - t0) * 1000)  # type: ignore[attr-defined]
        return r

    return _post


def safe_json(r: httpx.Response) -> Any:
    try:
        return r.json()
    except ValueError:
        return r.text[:500]


def pytest_sessionfinish(session, exitstatus):
    OBS["finished"] = datetime.now(UTC).isoformat()
    OBS["exit_status"] = int(exitstatus)
    out_dir = RESULTS_DIR / TARGET
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{stamp}.json"
    path.write_text(json.dumps(OBS, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nobservations written to {path}")
