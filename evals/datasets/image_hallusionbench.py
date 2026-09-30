"""HallusionBench yes/no questions with an image -> noul records; the 178 text-only (non_image) questions are skipped.

python -m evals.datasets.image_hallusionbench --out evals/data/image/hallusionbench.jsonl --limit 30
"""
import re

from evals.datasets._common import ImageSetWriter, hf_stream, image_args

NAME = "hallusionbench"
HF_ID, CONFIG = "lmms-lab-encoder/HallusionBench", "default"  # lmms-lab/HallusionBench now redirects here
SPLITS = ("image", "non_image")
REVISION = "cd417161857aefb23d878d42cf1bb53aa9dd646f"
N_FULL = 951
TARGET = {"1": 1.0, "0": 0.0}


def main():
    args = image_args()
    w = ImageSetWriter(NAME, args)
    for split in SPLITS:
        if w.full:
            break
        for row in hf_stream(HF_ID, CONFIG, split, REVISION, image_cols=["image"]):
            if w.full:
                break
            if row.get("image") is None or str(row["visual_input"]) == "0":
                w.skip("no_image")
                continue
            if str(row["gt_answer"]) not in TARGET:
                w.skip("answer_not_0_1")
                continue
            rid = w.next_id("test")
            fig = f"{row['category']}/{row['subcategory']}/{row['set_id']}_{row['figure_id']}"
            w.add({
                "id": rid, "source": NAME, "split": "test", "license": "commercial-ok", "lang": "en",
                # state is the bare image object (spec §4); figures are shared by several questions: one file per figure
                "state": w.image(row["image"], re.sub(r"[^\w]+", "_", fig)),
                "questions": {"q1": {"type": "noul", "instructions": row["question"].strip()}},
                "targets": {"q1": TARGET[str(row["gt_answer"])]},
                "meta": {"grid_id": f"{NAME}/{fig}", "category": row["category"], "subcategory": row["subcategory"],
                         "set_id": row["set_id"], "figure_id": row["figure_id"], "question_id": row["question_id"],
                         "visual_input": row["visual_input"], "sample_note": row["sample_note"],
                         "gt_answer_details": row["gt_answer_details"], "orig_filename": row["filename"],
                         "orig_answer_type": "yes_no"},
            })
    w.finish({"hf_id": HF_ID, "hf_config": CONFIG, "hf_split": "image (non_image read and skipped)",
              "hf_revision": REVISION, "n_records_full": N_FULL, "license": "commercial-ok", "lang": "en",
              "notes": "1,129 questions = 951 with a figure (split 'image', visual_input 1 = original figure, "
                       "2 = edited/illusion figure) + 178 text-only (split 'non_image', visual_input 0) which are "
                       "skipped. gt_answer 1=yes, 0=no. Pair structure kept in meta.set_id/figure_id/question_id "
                       "(same set_id+question_id across figure_id = the original/edited control pair); records "
                       "id split is 'test'. One file per figure (346 figures). License BSD-3 (GitHub)."})


if __name__ == "__main__":
    main()
