"""Old per-row engine vs prefix-shared engine (sequential and batched branches) on the same requests; optional latency.

PYTHONPATH=. .venv/Scripts/python.exe train/tests/prefix_share_check.py --ckpt runs/prior-only-0.8b --dtype fp32
PYTHONPATH=. .venv/Scripts/python.exe train/tests/prefix_share_check.py --ckpt runs/prior-only-0.8b --dtype bf16 --bench
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import time
from pathlib import Path

from PIL import Image

from vev.pointer import PointerEngine

ROOT = Path(__file__).resolve().parents[2]

QS = {
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?", "criteria": None},
    "topic": {"type": "choice", "instructions": "What is the message mainly about?",
              "criteria": {"delivery": "Shipping or tracking", "billing": "Charges, refunds or invoices",
                           "product": "The item itself", "other": None}},
    "tone": {"type": "score", "instructions": "How polite is the message?", "criteria": ["Rude", "Neutral", "Polite"]},
    "angry": {"type": "noul", "instructions": "", "criteria": {"true": "The writer is clearly upset", "false": "Calm"}},
    "urgency": {"type": "score", "instructions": "How urgent is it?", "criteria": ["Not", "Somewhat", "Very", "Critical"]},
    "lang": {"type": "choice", "instructions": "Language?", "criteria": {"en": None, "zh": None, "fr": None}},
    "bare": {"type": "noul", "instructions": "", "criteria": None},
    "name": {"type": "noul", "instructions": "Does the message include the customer's name?",
             "criteria": {"true": "A personal name appears"}},
}
LONG_Q = {"type": "choice",
          "instructions": "Read the whole message carefully and decide which department should own the follow-up. " * 6,
          "criteria": {f"dept_{i}": f"Department {i}: handles cases of kind {i} " * 3 for i in range(12)}}
TICKET = ("Hi, I ordered a blue kettle (order #58213) two weeks ago. The tracking page has been blank since Monday and "
          "nobody answers the phone. I was charged twice as well. Please sort this out or refund me. - Dana")
JSON_STATE = {"ticket": {"id": 58213, "channel": "email", "body": TICKET, "tags": ["shipping", "billing"]},
              "customer": {"tier": "gold", "orders": 14, "last_contact_days": 3.5, "vip": True}}


def image_url() -> str:
    src = ROOT / "data/check/v1-research/montage/ai2d.png"
    img = Image.open(src).convert("RGB") if src.exists() else Image.new("RGB", (256, 256), (200, 30, 30))
    img.thumbnail((256, 256))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def cases() -> dict[str, tuple[object, dict]]:
    names = list(QS)
    return {
        "text_1q": (TICKET, {"topic": QS["topic"]}),
        "text_3q": (TICKET, {k: QS[k] for k in names[:3]}),
        "text_8q": (TICKET, dict(QS)),
        "json_4q": (JSON_STATE, {k: QS[k] for k in names[:4]}),
        "image_3q": ({"note": "A diagram from a science textbook.", "image": {"url": image_url()}},
                     {"what": {"type": "choice", "instructions": "What does the image show?",
                               "criteria": {"diagram": None, "photo": None, "text only": None}},
                      "colorful": {"type": "noul", "instructions": "Is the image colorful?", "criteria": None},
                      "busy": {"type": "score", "instructions": "How cluttered?", "criteria": ["clean", "some", "busy"]}}),
        "long_suffix": (TICKET, {"topic": QS["topic"], "long": LONG_Q, "bare": QS["bare"]}),
    }


def probs(result) -> dict[str, list[float]]:
    out = {}
    for name, a in result.answers.items():
        out[name] = [a["noul"], 1 - a["noul"]] if a["type"] == "noul" else list(a["probabilities"].values())
    return out


def run_mode(eng: PointerEngine, mode: str, state, qs):
    eng.prefix_share, eng.branch_batch = mode != "old", mode == "batched"
    return eng.run(state, qs)


def compare(eng: PointerEngine) -> dict[str, dict[str, float]]:
    table = {}
    for case, (state, qs) in cases().items():
        ref = run_mode(eng, "old", state, qs)
        rep = run_mode(eng, "old", state, qs)
        row = {"old_repeat": max_diff(probs(ref), probs(rep))}
        for mode in ("sequential", "batched"):
            r = run_mode(eng, mode, state, qs)
            assert r.input_tokens == ref.input_tokens and r.extensions["tokens"] == ref.extensions["tokens"]
            row[mode] = max_diff(probs(ref), probs(r))
        # isolation: each question alone vs together, within the same mode
        for mode in ("sequential", "batched"):
            together = probs(run_mode(eng, mode, state, qs))
            alone = {k: probs(run_mode(eng, mode, state, {k: q}))[k] for k, q in qs.items()}
            row[mode[:3] + "_isolation"] = max_diff(together, alone)
        table[case] = row
        print(case, json.dumps({k: f"{v:.2e}" for k, v in row.items()}), flush=True)
    return table


def max_diff(a: dict, b: dict) -> float:
    return max(abs(x - y) for k in a for x, y in zip(a[k], b[k]))


def bench(eng: PointerEngine, reps: int, repeats: list[int]) -> None:
    names = list(QS)
    for k in repeats:
        state = (TICKET + " ") * k
        st = eng.run(state, {'bare': QS['bare']}).extensions['tokens']['state_text']
        for n in (1, 4, 8):
            qs = {q: QS[q] for q in names[:n]}
            line = [f"state {st} tok | {n} q"]
            for mode in ("old", "sequential", "batched"):
                run_mode(eng, mode, state, qs)  # warm
                ts = []
                for _ in range(reps):
                    t0 = time.perf_counter()
                    run_mode(eng, mode, state, qs)
                    ts.append((time.perf_counter() - t0) * 1000)
                line.append(f"{mode} {statistics.median(ts):.0f} ms")
            print(" | ".join(line), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/prior-only-0.8b")
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--bench-only", action="store_true")
    ap.add_argument("--state-repeats", default="9", help="comma list; the state is the ticket text repeated k times (9 ~ 500 tokens)")
    ap.add_argument("--vs-fp32", action="store_true", help="also report each mode's error against the fp32 old engine")
    a = ap.parse_args()
    eng = PointerEngine(a.ckpt, dtype=a.dtype, prefix_min_tokens=0)
    print(f"== {a.ckpt} {a.dtype}")
    table = compare(eng) if not a.bench_only else {}
    if a.vs_fp32:
        ref = PointerEngine(a.ckpt, dtype="fp32", prefix_min_tokens=0)
        worst = {m: 0.0 for m in ("old", "sequential", "batched")}
        for case, (state, qs) in cases().items():
            truth = probs(run_mode(ref, "old", state, qs))
            errs = {m: max_diff(truth, probs(run_mode(eng, m, state, qs))) for m in worst}
            worst = {m: max(worst[m], errs[m]) for m in worst}
            print(case, "vs fp32", json.dumps({k: f"{v:.2e}" for k, v in errs.items()}), flush=True)
        print("max vs fp32:", json.dumps({k: f"{v:.2e}" for k, v in worst.items()}))
    for mode in ("sequential", "batched", "seq_isolation", "bat_isolation", "old_repeat") if table else ():
        print(f"max {mode}: {max(r[mode] for r in table.values()):.2e}")
    if a.bench or a.bench_only:
        bench(eng, a.reps, [int(x) for x in a.state_repeats.split(",")])


if __name__ == "__main__":
    main()
