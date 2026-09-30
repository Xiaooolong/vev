"""MMMU validation, multiple-choice only (847 of 900) -> choice records; multi-image questions keep every image.

python -m evals.datasets.image_mmmu --out evals/data/image/mmmu.jsonl --limit 30 [--subjects Art_Theory,Math]
"""
import ast
import re

from evals.datasets._common import ImageSetWriter, hf_stream, image_args, one_hot

NAME = "mmmu"
HF_ID, SPLIT = "MMMU/MMMU", "validation"
REVISION = "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68"
N_FULL = 847
MAX_IMAGES = 7
SUBJECTS = [
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art", "Art_Theory", "Basic_Medical_Science",
    "Biology", "Chemistry", "Clinical_Medicine", "Computer_Science", "Design", "Diagnostics_and_Laboratory_Medicine",
    "Economics", "Electronics", "Energy_and_Power", "Finance", "Geography", "History", "Literature", "Manage",
    "Marketing", "Materials", "Math", "Mechanical_Engineering", "Music", "Pharmacy", "Physics", "Psychology",
    "Public_Health", "Sociology",
]
INSTRUCTIONS = ("Answer the multiple-choice question. '<image N>' in the question or options refers to the "
                "state entry with that key. Pick the correct option.")
TAG = re.compile(r"<image (\d+)>")


def ordered_image_keys(question: str, options: list[str], present: list[int]) -> tuple[list[int], list[int]]:
    """Image numbers in order of first appearance (question, then options), then unreferenced ones."""
    seen = []
    for n in (int(m) for m in TAG.findall(question + "\n" + "\n".join(options))):
        if n not in seen:
            seen.append(n)
    rest = [n for n in present if n not in seen]
    return seen, rest


def main():
    args = image_args(lambda ap: ap.add_argument("--subjects", default="", help="comma list; default all 30"))
    subjects = args.subjects.split(",") if args.subjects else SUBJECTS
    w = ImageSetWriter(NAME, args)
    cols = [f"image_{i}" for i in range(1, MAX_IMAGES + 1)]
    for subject in subjects:
        if w.full:
            break
        for row in hf_stream(HF_ID, subject, SPLIT, REVISION, image_cols=cols):
            if w.full:
                break
            if row["question_type"] != "multiple-choice":
                w.skip(f"question_type={row['question_type']}")
                continue
            options = [str(o) for o in ast.literal_eval(row["options"])]
            labels = [chr(ord("A") + i) for i in range(len(options))]
            criteria = dict(zip(labels, options))
            if row["answer"] not in criteria:
                w.skip("answer_not_in_options")
                continue
            present = [i for i in range(1, MAX_IMAGES + 1) if row.get(f"image_{i}") is not None]
            referenced, unreferenced = ordered_image_keys(row["question"], options, present)
            if any(n not in present for n in referenced):
                w.skip("referenced_image_missing")
                continue
            rid = w.next_id(SPLIT)
            seq = rid.rsplit("/", 1)[1]
            state = {"context": row["question"]}
            for n in referenced + unreferenced:
                state[f"<image {n}>"] = w.image(row[f"image_{n}"], f"{seq}_{n}")
            meta = {"grid_id": f"{NAME}/{row['id']}", "orig_id": row["id"], "orig_answer_type": "mc",
                    "subject": subject, "subfield": row["subfield"], "img_type": row["img_type"],
                    "topic_difficulty": row["topic_difficulty"], "n_images": len(present)}
            if unreferenced:
                meta["unreferenced_images"] = unreferenced
            w.add({
                "id": rid, "source": NAME, "split": SPLIT, "license": "commercial-ok", "lang": "en",
                "state": state,
                "questions": {"q1": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}},
                "targets": {"q1": one_hot(criteria, row["answer"])},
                "meta": meta,
            })
    w.finish({"hf_id": HF_ID, "hf_config": ",".join(subjects) if args.subjects else "all 30 subjects",
              "hf_split": SPLIT, "hf_revision": REVISION, "n_records_full": N_FULL,
              "license": "commercial-ok", "lang": "en",
              "notes": "validation only (test has no public answers); 53 open questions skipped -> 847. "
                       "2-9 options, labels A.. in option order. Images go into state under keys '<image N>' "
                       "matching the tags in the text, ordered by first appearance in question then options "
                       "(16 questions have images only inside options); images present but never referenced are "
                       "appended after and listed in meta.unreferenced_images. 39 questions are multi-image. "
                       "HF card: Apache-2.0."})


if __name__ == "__main__":
    main()
