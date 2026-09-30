"""Run the real Vev HTTP layer on top of a model-free engine, so the conformance suite can run without a GPU (CI).

    python conformance/stub_server.py --port 8009 &
    SO_TARGET=stub SO_SUPPORTS_IMAGES=1 pytest conformance

Answers are deterministic hashes of (state, question): the suite checks the API contract, not judgment quality."""

from __future__ import annotations

import argparse
import hashlib
import json
import math

import uvicorn

from vev.readout import Result, confidence
from vev.serve import create_app
from vev.state import serialize_state


class StubEngine:
    base_id = "stub"

    def warmup(self) -> None:
        pass

    def run(self, state, questions, max_images=8, max_pixels=1024 * 1024, allow_remote=False, max_state_tokens=32768,
            max_request_tokens=65536) -> Result:
        serialize_state(state, max_images=max_images, max_pixels=max_pixels, allow_remote=allow_remote)
        key = json.dumps(state, sort_keys=True, ensure_ascii=False, default=str)
        answers = {name: self._answer(key, q) for name, q in questions.items()}
        n_in = len(key) // 4 + sum(len(json.dumps(q, ensure_ascii=False)) // 4 for q in questions.values())
        return Result(answers=answers, input_tokens=n_in, output_tokens=len(answers),
                      extensions={"logits": {}, "timing": {}, "tokens": {}})

    @staticmethod
    def _logits(key: str, q: dict, labels: list) -> list[float]:
        out = []
        for lab in labels:
            h = hashlib.sha256(json.dumps([key, q.get("instructions"), str(lab)], ensure_ascii=False).encode()).digest()
            out.append(int.from_bytes(h[:4], "big") / 2**32 * 4 - 2)
        return out

    def _answer(self, key: str, q: dict) -> dict:
        if q["type"] == "noul":
            z = self._logits(key, q, ["yes"])[0]
            return {"type": "noul", "noul": 1 / (1 + math.exp(-z))}
        labels = list(q["criteria"]) if q["type"] == "choice" else list(range(len(q["criteria"])))
        z = self._logits(key, q, labels)
        m = max(z)
        e = [math.exp(x - m) for x in z]
        p = [x / sum(e) for x in e]
        best = max(range(len(p)), key=lambda i: (p[i], -i))
        conf = confidence(p[best], len(p))
        if q["type"] == "choice":
            return {"type": "choice", "choice": labels[best], "confidence": conf, "probabilities": dict(zip(labels, p))}
        return {"type": "score", "score": sum(i * v for i, v in enumerate(p)), "confidence": conf,
                "probabilities": {str(i): v for i, v in enumerate(p)},
                "legend": {str(i): d for i, d in enumerate(q["criteria"])}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8009)
    a = ap.parse_args()
    uvicorn.run(create_app(StubEngine(), "vev-stub", "stub", False), host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
