"""anchor-v3 = anchor-v1 + anchor-short (half A; its builder is not shipped) + the bare-label short rows again with a generic instruction that does not
state the task (wording deliberately differs from every eval set).

    python -m data.anchor.add_generic --v1 data/build/anchor-v1/train.jsonl --short data/build/anchor-short-v1/train.jsonl \
        --out data/build/anchor-v3/train.jsonl"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

GENERIC = {"zh": ["请给这段文字打标签。", "文本分类任务。", "判断类别。", "请选择合适的标签。"],
           "en": ["Classify the text.", "Assign a label.", "Pick the right category.", "Label this text."]}
BARE = ("bare_pos_first", "bare_neg_first", "bare", "bare_shuffled")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True)
    ap.add_argument("--short", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    short = [json.loads(l) for l in Path(a.short).read_text(encoding="utf-8").splitlines() if l.strip()]
    gen = []
    for r in short:
        if r["meta"]["variant"] not in BARE:
            continue
        q = {**r["questions"]["q1"], "instructions": rng.choice(GENERIC[r["lang"]])}
        gen.append({**r, "id": r["id"].replace("anchor_short/", "anchor_generic/"), "source": "anchor_generic",
                    "questions": {"q1": q}, "meta": {**r["meta"], "variant": r["meta"]["variant"] + "_generic"}})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(Path(a.v1).read_text(encoding="utf-8"))
        fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in short + gen)
    print(f"[anchor] {a.out}: + {len(short)} short + {len(gen)} generic")


if __name__ == "__main__":
    main()
