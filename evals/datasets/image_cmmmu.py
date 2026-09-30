"""CMMMU val: multiple choice (选择) -> choice, true/false (判断) -> noul; fill-in (填空) skipped. lang=zh.

python -m evals.datasets.image_cmmmu --out evals/data/image/cmmmu.jsonl --limit 30
"""
import re

from evals.datasets._common import ImageSetWriter, blank, hf_stream, image_args, one_hot

NAME = "cmmmu"
HF_ID, SPLIT = "m-a-p/CMMMU", "val"
REVISION = "d7ba0c8d5b73a45a0897ec29a9e841fab8acdde2"
N_FULL = 674  # 590 multiple-choice - 4 multi-answer = 586 choice, + 88 true/false noul
CONFIGS = ["art_and_design", "business", "health_and_medicine", "humanities_and_social_sciences",
           "science", "technology_and_engineering"]
MAX_IMAGES = 5
CHOICE_INSTR = "回答选择题：'<image N>' 指 state 中同名键的图片。选出唯一正确的选项。"
JUDGE_INSTR = "判断题：'<image N>' 指 state 中同名键的图片。题干中的说法是否正确？"
JUDGE_TARGET = {"对": 1.0, "错": 0.0}
TAG = re.compile(r'<img="([^"]+)">')


def main():
    args = image_args()
    w = ImageSetWriter(NAME, args)
    cols = [f"image_{i}" for i in range(1, MAX_IMAGES + 1)]
    for config in CONFIGS:
        if w.full:
            break
        for row in hf_stream(HF_ID, config, SPLIT, REVISION, image_cols=cols):
            if w.full:
                break
            qtype = row["type"]
            if qtype not in ("选择", "判断"):
                w.skip(f"type={qtype}")
                continue
            by_file = {row[f"image_{i}_filename"]: i for i in range(1, MAX_IMAGES + 1)
                       if row.get(f"image_{i}") is not None and not blank(row.get(f"image_{i}_filename"))}
            options = [str(row[f"option{i}"]).strip() for i in range(1, 5) if not blank(row.get(f"option{i}"))]
            order, missing = [], []

            def retag(text: str) -> str:
                def sub(m):
                    n = by_file.get(m.group(1))
                    if n is None:
                        missing.append(m.group(1))
                        return m.group(0)
                    if n not in order:
                        order.append(n)
                    return f"<image {n}>"
                return TAG.sub(sub, text)

            question = retag(row["question"])
            options = [retag(o) for o in options]
            if missing:
                w.skip("img_tag_without_image")
                continue
            unreferenced = [n for n in sorted(by_file.values()) if n not in order]
            if not order and not unreferenced:
                w.skip("no_image")
                continue
            if qtype == "选择":
                answer = str(row["answer"]).strip()
                criteria = {chr(ord("A") + i): o for i, o in enumerate(options)}
                if len(answer) != 1:
                    w.skip("multi_answer")
                    continue
                if answer not in criteria or len(criteria) < 2:
                    w.skip("answer_not_in_options")
                    continue
                question_obj = {"type": "choice", "instructions": CHOICE_INSTR, "criteria": criteria}
                target = one_hot(criteria, answer)
            else:
                if row["answer"] not in JUDGE_TARGET:
                    w.skip("judge_answer_not_对错")
                    continue
                question_obj = {"type": "noul", "instructions": JUDGE_INSTR}
                target = JUDGE_TARGET[row["answer"]]
            rid = w.next_id(SPLIT)
            seq = rid.rsplit("/", 1)[1]
            state = {"context": question}
            for n in order + unreferenced:
                state[f"<image {n}>"] = w.image(row[f"image_{n}"], f"{seq}_{n}")
            meta = {"grid_id": f"{NAME}/{row['id']}", "orig_id": row["id"], "orig_type": qtype,
                    "orig_answer_type": "mc" if qtype == "选择" else "judge", "config": config,
                    "category": row["category"], "subcategory": row["subcategory"],
                    "difficulty_level": row["difficulty_level"], "img_type": row["img_type"]}
            if unreferenced:
                meta["unreferenced_images"] = unreferenced
            w.add({
                "id": rid, "source": NAME, "split": SPLIT, "license": "commercial-ok", "lang": "zh",
                "state": state, "questions": {"q1": question_obj}, "targets": {"q1": target}, "meta": meta,
            })
    w.finish({"hf_id": HF_ID, "hf_config": "all 6 configs", "hf_split": SPLIT, "hf_revision": REVISION,
              "n_records_full": N_FULL, "license": "commercial-ok", "lang": "zh",
              "notes": "900 val rows: 590 multiple-choice, 88 true/false, 222 fill-in (skipped). 4 multiple-choice with "
                       "multi-letter answers (e.g. 'ABCD') skipped -> 586 choice + 88 noul (对=1, 错=0). Inline tags "
                       "<img=\"file\"> in question/options rewritten to '<image N>' and images put in state under "
                       "those keys in order of appearance; unreferenced images appended (meta.unreferenced_images). "
                       "License: GitHub Apache-2.0, HF card has no field. n_records_full excludes rows whose "
                       "images turn out missing (not checkable without downloading)."})


if __name__ == "__main__":
    main()
