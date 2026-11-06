"""Decision Index text engine for Vev (github.com/apolinario/decision-index, public 0.3 suite): the kit's
HttpSystemOne against a running `vev serve`, plus three recorded, mechanical changes.

    PYTHONPATH=evals/decision_index decision-index run --engine text_engine:VevHttp \
        --option base_url=http://127.0.0.1:PORT --option model=jev-latest ...

1. Non-string `instructions` are sent as the kit's own text rendering (decision_index.engines.base.text), because
   Vev accepts only strings there.
2. Vev's three capacity refusals (state/request token limits, too many choices) become Unsupported; any other 4xx
   stays an error.
3. After 10 consecutive 5xx or connection failures the run stops with SystemExit (the runner's `except Exception`
   does not catch it), so a dead server cannot turn the rest of the suite into errors; the job restarts the server
   and resumes."""

from __future__ import annotations

import copy

import httpx
from decision_index.engines.base import Unsupported, text
from decision_index.engines.http import HttpSystemOne

CAPACITY = ("tokens (max 32768)", "tokens (max 65536)", "too many choices")
MAX_SERVER_FAILURES = 10


def capacity_message(r: httpx.Response) -> str:
    """Vev's own error message ({"detail": {"error_type", "message"}}); pydantic 422 bodies echo the request and are
    never read for markers."""
    try:
        detail = r.json().get("detail")
    except ValueError:
        return ""
    return detail.get("message", "") if isinstance(detail, dict) else ""


class VevHttp(HttpSystemOne):
    name = "vev-http"

    def __init__(self, **options):
        super().__init__(**options)
        self.failures = 0
        self.provenance = {**self.provenance, "policy": (
            "Kit HttpSystemOne with three mechanical changes: non-string question instructions are rendered with "
            "decision_index.engines.base.text before sending (Vev takes strings only); Vev's capacity refusals "
            f"{list(CAPACITY)} are recorded as unsupported; the run stops after {MAX_SERVER_FAILURES} consecutive "
            "server or connection failures and is resumed after a server restart."),
            "server_models": self.client.get("/v1/models").json()["models"]}

    @staticmethod
    def adapt(questions: dict) -> dict:
        out = copy.deepcopy(questions)
        for q in out.values():
            if "instructions" in q and q["instructions"] is not None and not isinstance(q["instructions"], str):
                q["instructions"] = text(q["instructions"])
        return out

    def __call__(self, state, questions):
        try:
            r = self.client.post("/v1/systemone", json={"model": self.model, "state": state,
                                                        "questions": self.adapt(questions), **self.extra})
        except httpx.TransportError:
            self._fail()
            raise
        if r.status_code >= 500:
            self._fail()
            r.raise_for_status()
        self.failures = 0
        if r.status_code == 400 and any(s in capacity_message(r) for s in CAPACITY):
            raise Unsupported(r.text)
        r.raise_for_status()
        raw = r.json()
        return {k: v for k, v in raw.items() if k != "evaluation_trace"}, raw

    def _fail(self) -> None:
        self.failures += 1
        if self.failures >= MAX_SERVER_FAILURES:
            raise SystemExit(f"{MAX_SERVER_FAILURES} consecutive server failures: stopping so the job can restart the server")
