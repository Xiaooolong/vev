"""MMBench v1.0 EN dev (lmms-lab mirror) -> choice records. Shared by image_mmbench_cn / image_ccbench.

python -m evals.datasets.image_mmbench_en --out evals/data/image/mmbench_en.jsonl --limit 30
"""
from evals.datasets._common import ImageSetWriter, blank, hf_stream, image_args, one_hot

HF_ID = "lmms-lab-encoder/MMBench"  # lmms-lab/MMBench now redirects here
REVISION = "56ba1af8954932c4804bd3f522e05ed96e63b654"
CIRCULAR_OFFSET = 1_000_000  # rows with index >= 1e6 are CircularEval option rotations of index % 1e6
INSTRUCTIONS = {"en": "Answer the multiple-choice question about the image: pick the correct option.",
                "zh": "根据图片回答单选题，选出正确选项。"}
LICENSE_NOTE = ("License: open-compass/MMBench GitHub is Apache-2.0; HF card has no license field; "
                "images come from third-party sources. ")


def run(name: str, config: str, split: str, lang: str, n_full: int, notes: str) -> None:
    args = image_args()
    w = ImageSetWriter(name, args)
    for row in hf_stream(HF_ID, config, split, REVISION, image_cols=["image"]):
        if w.full:
            break
        index = int(row["index"])
        if index >= CIRCULAR_OFFSET:
            w.skip("circular_rotation")
            continue
        opts = {k: str(row[k]).strip() for k in "ABCD" if not blank(row.get(k))}
        if row["answer"] not in opts:
            w.skip("answer_not_in_options")
            continue
        hint = row.get("hint")
        context = row["question"].strip() if blank(hint) else f"{str(hint).strip()}\n\n{row['question'].strip()}"
        rid = w.next_id(split)
        meta = {"grid_id": f"{name}/img/{index}", "orig_index": index, "orig_answer_type": "mc",
                "category": row.get("category"), "orig_source": None if blank(row.get("source")) else row["source"]}
        for k in ("L2-category", "comment"):
            if not blank(row.get(k)):
                meta[k.replace("-", "_").lower()] = row[k]
        w.add({
            "id": rid, "source": name, "split": split, "license": "commercial-ok", "lang": lang,
            "state": {"context": context, "image": w.image(row["image"], rid.rsplit("/", 1)[1])},
            "questions": {"q1": {"type": "choice", "instructions": INSTRUCTIONS[lang], "criteria": opts}},
            "targets": {"q1": one_hot(opts, row["answer"])},
            "meta": meta,
        })
    w.finish({"hf_id": HF_ID, "hf_config": config, "hf_split": split, "hf_revision": REVISION,
              "n_records_full": n_full, "license": "commercial-ok", "lang": lang, "notes": LICENSE_NOTE + notes})


if __name__ == "__main__":
    run("mmbench_en", "en", "dev", "en", 1164,
        "dev has 4,329 rows = 1,164 questions + CircularEval rotations (index >= 1e6); rotations skipped "
        "(use run.py --perturb order instead). 'nan' options dropped (2-/3-option questions). "
        "hint prepended to the question in state.context. test split has no answers.")
