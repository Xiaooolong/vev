"""Latency curves: questions per request with a short state, a long state and a ~1 MP image, and state length.
Writes JSON, asserts nothing. Each cell reports the first request of that shape separately (first_ms), since kernels
are tuned per shape on first use.

    python conformance/curves.py --out conformance/results/<target>/curves.json

Uses the same SO_* environment variables as the test suite.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from conftest import API_KEY, BASE_URL, MODEL, TARGET, TIMEOUT, tiny_png_data_url

FILLER = (
    "The customer opened the ticket on Monday, attached two screenshots, and asked for a call back. "
    "The account has been active for three years with no previous disputes. "
)


def state_of_words(n_words: int) -> str:
    words = FILLER.split()
    out = []
    while len(out) < n_words:
        out.extend(words)
    return "Help! My payouts have been failing for 3 days. " + " ".join(out[:n_words])


def questions(n: int) -> dict:
    qs = {}
    for i in range(n):
        kind = i % 3
        if kind == 0:
            qs[f"q{i}"] = {"type": "noul", "instructions": f"Does the state mention topic {i}?"}
        elif kind == 1:
            qs[f"q{i}"] = {"type": "choice", "instructions": f"Which team for topic {i}?",
                           "criteria": {"billing": "money", "technical": "bugs", "sales": "pricing"}}
        else:
            qs[f"q{i}"] = {"type": "score", "instructions": f"Severity of topic {i}?",
                           "criteria": ["low", "medium", "high"]}
    return qs


def image_state() -> dict:
    return {"note": "checkout page", "screen": {"image": {"url": tiny_png_data_url(1024)}}}


def measure(client: httpx.Client, state, qs: dict, repeats: int) -> dict:
    lat = []
    tokens = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        r = client.post("/v1/systemone", json={"state": state, "model": MODEL, "questions": qs})
        lat.append((time.perf_counter() - t0) * 1000)
        if r.status_code != 200:
            return {"error": r.status_code, "body": r.text[:300]}
        tokens = r.json().get("usage", {}).get("input_tokens")
    rest = lat[1:] or lat
    return {"p50_ms": round(statistics.median(rest), 1), "min_ms": round(min(rest), 1),
            "max_ms": round(max(rest), 1), "first_ms": round(lat[0], 1), "input_tokens": tokens, "n": len(rest)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / TARGET / "curves.json"))
    ap.add_argument("--repeats", type=int, default=6, help="requests per cell; the first is reported as first_ms")
    ap.add_argument("--questions", default="1,5,10,25,50,100")
    ap.add_argument("--words", default="60,500,2000,6000")
    a = ap.parse_args()

    with httpx.Client(base_url=BASE_URL, headers={"Authorization": f"Bearer {API_KEY}"}, timeout=TIMEOUT) as c:
        c.post("/v1/systemone", json={"state": state_of_words(60), "model": MODEL, "questions": questions(1)})  # warm-up
        counts = [int(x) for x in a.questions.split(",")]
        curves = {"latency_vs_questions": (state_of_words(60), counts),
                  "latency_vs_questions_long_state": (state_of_words(6000), counts),
                  "latency_vs_questions_image": (image_state(), [n for n in counts if n <= 50])}
        found = {}
        for key, (state, ns) in curves.items():
            found[key] = {}
            for n in ns:
                found[key][n] = measure(c, state, questions(n), a.repeats)
                print(f"{key} questions={n:4d}  {found[key][n]}")
        by_state = {}
        for w in (int(x) for x in a.words.split(",")):
            by_state[w] = measure(c, state_of_words(w), questions(3), a.repeats)
            print(f"state_words={w:5d}  {by_state[w]}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "target": TARGET, "base_url": BASE_URL, "model": MODEL,
        "measured": datetime.now(UTC).isoformat(),
        **found, "latency_vs_state_words": by_state,
    }, indent=2), encoding="utf-8")
    print(f"written {out}")


if __name__ == "__main__":
    main()
