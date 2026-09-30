"""POPE (COCO val2014 object-existence yes/no, 3 negative-sampling settings) -> noul records.

python -m evals.datasets.image_pope --out evals/data/image/pope.jsonl --limit 30
"""
import re

from evals.datasets._common import ImageSetWriter, hf_stream, image_args

NAME = "pope"
HF_ID, CONFIG, SPLIT = "lmms-lab-encoder/POPE", "default", "test"  # lmms-lab/POPE now redirects here
REVISION = "4db1276663dfa5eb8ad16a52d24c31a09e470896"
N_FULL = 9000
PATTERN = re.compile(r"Is there an? (.+) in the image\?")
TARGET = {"yes": 1.0, "no": 0.0}


def main():
    args = image_args()
    w = ImageSetWriter(NAME, args)
    for row in hf_stream(HF_ID, CONFIG, SPLIT, REVISION, image_cols=["image"]):
        if w.full:
            break
        answer = str(row["answer"]).strip().lower()
        if answer not in TARGET:
            w.skip("answer_not_yes_no")
            continue
        if row.get("image") is None:
            w.skip("no_image")
            continue
        question = row["question"].strip()
        rid = w.next_id(SPLIT)
        meta = {"grid_id": f"{NAME}/{row['image_source']}", "category": row["category"],
                "image_source": row["image_source"], "orig_question_id": row["question_id"],
                "orig_answer_type": "yes_no"}
        m = PATTERN.fullmatch(question)
        if m:
            meta["negated_questions"] = {"q1": {"type": "noul", "instructions": f"Is there no {m.group(1)} in the image?"}}
        w.add({
            "id": rid, "source": NAME, "split": SPLIT, "license": "unknown", "lang": "en",
            # one file per COCO image: the 500 images are shared by 9,000 questions
            "state": w.image(row["image"], row["image_source"]),  # state is the bare image object (spec §4)
            "questions": {"q1": {"type": "noul", "instructions": question}},
            "targets": {"q1": TARGET[answer]},
            "meta": meta,
        })
    w.finish({"hf_id": HF_ID, "hf_config": CONFIG, "hf_split": SPLIT, "hf_revision": REVISION, "n_records_full": N_FULL,
              "license": "unknown", "lang": "en",
              "notes": "3 x 3,000 questions (meta.category = adversarial/popular/random) on 500 COCO val2014 "
                       "images; each image is stored once (images/pope/<COCO name>.jpg) and shared. yes=1, no=0. "
                       "meta.negated_questions gives 'Is there no X in the image?'. License: POPE code is MIT, no "
                       "data license stated; images are COCO (Flickr terms) -> unknown."})


if __name__ == "__main__":
    main()
