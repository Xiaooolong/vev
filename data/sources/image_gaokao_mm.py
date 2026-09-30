"""GAOKAO-MM (OpenMOSS/GAOKAO-MM): 2010-2023 Gaokao image MCQs -> choice A/B/C/D, lang=zh.

Source: GitHub, pinned commit; Data/<subject>.json (8 files, <=300 KB each) and the images they list
(Data/<subject>/<name>.png), fetched one by one from raw.githubusercontent.com. Options are embedded in the question
text; they are split off at the last in-order "A. ... D." (or "... E.") marker sequence. Skipped: multi-answer questions,
questions whose options are pictures only (no parseable A-D text). All pictures go into state in listed order
(one: state["image"]; several: "<image n>").

python -m data.sources.image_gaokao_mm --out data/raw/image/gaokao_mm.jsonl --limit 30
"""
import json
import random
import re

from data.sources._common import RawWriter, image_from_bytes, one_hot, record, save_image, source_args
from data.sources._ranged import fetch, fetch_bytes, gh_raw

NAME, REPO, COMMIT = "gaokao_mm", "OpenMOSS/GAOKAO-MM", "76d5193c89655192d053dcf86ead743e0423ec95"
FILES = ["2010-2023_Biology_MCQs", "2010-2023_Chemistry_MCQs", "2010-2023_Chinese_Pratical_Lit",
         "2010-2023_Geography_MCQs", "2010-2023_History_MCQs", "2010-2023_Math_MCQs", "2010-2023_Physics_MCQs",
         "2010-2023_Political_Science_MCQs"]
INSTR = "回答高考选择题：选出唯一正确的选项。"
MARK = re.compile(r"([A-E])\s*[\.．、:：]")


def split_inline(text: str):
    """(stem, {A..D or A..E: text}) from the last in-order option markers, else None."""
    return _split(text, "ABCDE") or _split(text, "ABCD")


def _split(text: str, letters: str):
    pos = {k: [m.start() for m in MARK.finditer(text) if m.group(1) == k] for k in letters}
    end, picked = len(text) + 1, {}
    for k in reversed(letters):
        cands = [p for p in pos[k] if p < end]
        if not cands:
            return None
        picked[k] = end = cands[-1]
    order = [picked[k] for k in letters] + [len(text)]
    opts = {}
    for i, k in enumerate(letters):
        body = MARK.sub("", text[order[i]:order[i + 1]], count=1).strip()
        if not body:
            return None
        opts[k] = body
    return text[:order[0]].strip(), opts


def main():
    args = source_args(NAME)
    w = RawWriter(NAME, args.out)
    items = []
    for f in FILES:
        data = json.loads(fetch(gh_raw(REPO, COMMIT, f"Data/{f}.json"), f"gaokao_{f}.json").read_text(encoding="utf-8"))
        items += [(f, ex) for ex in data["example"]]
    random.Random(args.seed).shuffle(items)
    for f, ex in items:
        if args.limit and w.n >= args.limit:
            break
        if len(ex["answer"]) != 1 or len(ex["answer"][0].strip()) != 1:
            w.skip("multi_answer")
            continue
        parsed = split_inline(ex["question"])
        if parsed is None:
            w.skip("options_not_in_text")
            continue
        stem, crit = parsed
        if ex["answer"][0].strip() not in crit:
            w.skip("answer_not_in_options")
            continue
        state = {"question_context": stem}
        pics = ex["picture"]
        for n, pic in enumerate(pics, 1):
            rel = pic.split("../", 1)[-1]
            path = save_image(image_from_bytes(fetch_bytes(gh_raw(REPO, COMMIT, rel))), NAME,
                              rel.rsplit("/", 1)[-1].rsplit(".", 1)[0])
            if len(pics) == 1:
                state["image"] = {"path": path}
            else:
                state[f"<image {n}>"] = {"image": {"path": path}}
        w.write(record(NAME, w.n, state, {"q1": {"type": "choice", "instructions": INSTR, "criteria": crit}},
                       {"q1": one_hot(crit, ex["answer"][0].strip())}, lang="zh", grid_id=f"{NAME}/{f}_{ex['index']}",
                       source_split="test", meta={"subject_file": f, "index": ex["index"], "year": ex["year"],
                                                  "category": ex["category"], "n_images": len(pics)}))
    w.close()


if __name__ == "__main__":
    main()
