"""Criteria-flip probe: records that share a state but ask with different criteria, whose targets are opposite.
Records are paired by meta.flip_group (each group = one image, two noul records: target 1 and target 0, question
q1). Reports whether the model's P(true) follows the criteria rather than the image alone.

    python -m evals.criteria_flip --records evals/data/image/ui_toggle.jsonl --result results/<model>/ui_toggle.json

  margin   mean of P(true | target-true record) - P(true | target-false record); 1 = perfect, 0 = criteria ignored
  ordered  share of pairs with margin > 0
  both_hi  share of pairs answered > 0.5 on both (says yes regardless of the criterion)
  both_lo  share answered < 0.5 on both
Grouped by meta.flip_kind when present."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    groups: dict[str, dict] = defaultdict(dict)
    kinds: dict[str, str] = {}
    for line in Path(a.records).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        meta = r.get("meta") or {}
        g = meta.get("flip_group")
        if g is None or r["questions"]["q1"]["type"] != "noul":
            continue
        role = "t" if float(r["targets"]["q1"]) > 0.5 else "f"
        groups[g][role] = r["id"]
        kinds[g] = meta.get("flip_kind", "all")
    preds = {}
    for e in json.loads(Path(a.result).read_text(encoding="utf-8"))["raw"]:
        if "error" not in e and "pred" in e["questions"].get("q1", {}):
            preds[e["id"]] = float(e["questions"]["q1"]["pred"])
    stats = defaultdict(lambda: {"n": 0, "margin": 0.0, "ordered": 0, "both_hi": 0, "both_lo": 0})
    for g, ids in groups.items():
        if "t" not in ids or "f" not in ids or ids["t"] not in preds or ids["f"] not in preds:
            continue
        pt, pf = preds[ids["t"]], preds[ids["f"]]
        for key in {kinds[g], "all"}:
            s = stats[key]
            s["n"] += 1
            s["margin"] += pt - pf
            s["ordered"] += pt > pf
            s["both_hi"] += pt > 0.5 and pf > 0.5
            s["both_lo"] += pt < 0.5 and pf < 0.5
    lines = [f"# Criteria-flip probe: {Path(a.result).parent.name}/{Path(a.result).name}", "",
             "| group | n pairs | margin | ordered | both_hi | both_lo |", "|---|---|---|---|---|---|"]
    for k, s in sorted(stats.items(), key=lambda kv: (kv[0] != "all", kv[0])):
        n = max(1, s["n"])
        lines.append(f"| {k} | {s['n']} | {s['margin'] / n:.3f} | {s['ordered'] / n:.2f} | {s['both_hi'] / n:.2f} | {s['both_lo'] / n:.2f} |")
    text = "\n".join(lines) + "\n"
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
