"""MMStar val (1,500 4-choice VQA) -> choice records.

python -m evals.datasets.image_mmstar --out evals/data/image/mmstar.jsonl --limit 30
"""
import re

from evals.datasets._common import ImageSetWriter, blank, hf_stream, image_args, one_hot

NAME = "mmstar"
HF_ID, CONFIG, SPLIT = "Lin-Chen/MMStar", "val", "val"
REVISION = "bc98d668301da7b14f648724866e57302778ab27"
N_FULL = 1498  # 1,500 rows; 2 have answer A whose option text is "nan"
INSTRUCTIONS = "Answer the multiple-choice question about the image: pick the correct option."
BOILERPLATE = re.compile(r"^Hint: Please answer the question[^\n]*at the end\.\n(?:Question: )?")


def parse_question(text: str):
    """MMStar packs options into the question text in two styles; returns (stem, {label: text}) or None."""
    lines = list(re.finditer(r"^\(([A-D])\) ?(.*)$", text, re.M))
    if len(lines) >= 2:  # MathVista-style: "[Choices:]\n(A) x\n(B) y"
        opts = {m.group(1): m.group(2) for m in lines}
        stem = re.sub(r"\n?Choices:\s*$", "", text[:lines[0].start()].rstrip("\n"))
        stem = BOILERPLATE.sub("", stem)
    elif "Options:" in text:  # "Options: A: x, B: y, C: z, D: w"
        stem, tail = text.rsplit("Options:", 1)
        tail, marks, start = tail.strip(), [], 0
        for label in "ABCD":
            pat = "A: " if label == "A" else f", {label}: "
            i = tail.find(pat, start)
            if i < 0:
                break
            marks.append((label, i, i + len(pat)))
            start = i + len(pat)
        if not marks or marks[0][1] != 0:
            return None
        opts = {lab: tail[s:(marks[k + 1][1] if k + 1 < len(marks) else len(tail))].strip()
                for k, (lab, _, s) in enumerate(marks)}
    else:
        return None
    opts = {k: v.strip() for k, v in opts.items() if not blank(v)}
    return stem.strip(), opts


def main():
    args = image_args()
    w = ImageSetWriter(NAME, args)
    for row in hf_stream(HF_ID, CONFIG, SPLIT, REVISION, image_cols=["image"]):
        if w.full:
            break
        parsed = parse_question(row["question"])
        if parsed is None or len(parsed[1]) < 2:
            w.skip("options_unparsed")
            continue
        stem, opts = parsed
        if row["answer"] not in opts:
            w.skip("answer_not_in_options")
            continue
        rid = w.next_id(SPLIT)
        seq = rid.rsplit("/", 1)[1]
        meta_info = row.get("meta_info") or {}
        w.add({
            "id": rid, "source": NAME, "split": SPLIT, "license": "unknown", "lang": "en",
            "state": {"context": stem, "image": w.image(row["image"], seq)},
            "questions": {"q1": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": opts}},
            "targets": {"q1": one_hot(opts, row["answer"])},
            "meta": {"grid_id": f"{NAME}/img/{row['index']}", "orig_index": row["index"], "orig_answer_type": "mc",
                     "category": row["category"], "l2_category": row["l2_category"],
                     "orig_source": meta_info.get("source"), "orig_source_split": meta_info.get("split")},
        })
    w.finish({"hf_id": HF_ID, "hf_config": CONFIG, "hf_split": SPLIT, "hf_revision": REVISION, "n_records_full": N_FULL,
              "license": "unknown", "lang": "en",
              "notes": "Options parsed out of the question text (two styles: 'Options: A: .., B: ..' and "
                       "'[Choices:]\\n(A) ..'); 'nan' options dropped; MathVista answer-format hints are stripped. "
                       "License not stated on HF card or GitHub; samples are drawn from other benchmarks "
                       "(MMBench/SEED/AI2D/ScienceQA/MathVista/MMMU). Held-out."})


if __name__ == "__main__":
    main()
