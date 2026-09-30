"""Unlabeled KL-anchor set: inputs on which training must keep the base model's zero-shot distribution.

    python -m data.anchor.build --synth <records.jsonl> --out data/build/anchor-v1

The input is a JSONL of records with a question q1 and meta.domain (synthetic gate questions in our runs; their
generator is not shipped). Each record becomes a bare-label question on the same state: choice keeps only the option
labels, noul drops its criteria (score is skipped, a level scale without descriptions has no meaning). These rows
carry no usable target; train.py only computes the KL to the adapter-disabled base on them (--anchor-build)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def bare(q: dict) -> dict | None:
    if q["type"] == "choice":
        return {"type": "choice", "instructions": q["instructions"], "criteria": {k: None for k in q["criteria"]}}
    if q["type"] == "noul":
        return {"type": "noul", "instructions": q["instructions"], "criteria": None}
    return None


def dummy_target(q: dict):
    if q["type"] == "noul":
        return 0.5
    return {k: 1.0 / len(q["criteria"]) for k in q["criteria"]}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with (out / "train.jsonl").open("w", encoding="utf-8") as fh:
        for line in Path(a.synth).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            q = bare(rec["questions"]["q1"])
            if q is None:
                continue
            fh.write(json.dumps({"id": f"anchor/{rec['id']}", "source": "anchor_bare", "split": "train", "lang": rec.get("lang"),
                                 "state": rec["state"], "questions": {"q1": q}, "targets": {"q1": dummy_target(q)},
                                 "meta": {"from": rec["id"], "domain": rec["meta"]["domain"]}}, ensure_ascii=False) + "\n")
            n += 1
    print(f"[anchor] {n} rows -> {out / 'train.jsonl'}")


if __name__ == "__main__":
    main()
