"""VL-RewardBench (1,247 image+query, two responses, human-verified preference) -> choice records.

python -m evals.datasets.image_vl_rewardbench --out evals/data/image/vl_rewardbench.jsonl --limit 30
"""
from evals.datasets._common import ImageSetWriter, hf_stream, image_args, one_hot, stable_bit

NAME = "vl_rewardbench"
HF_ID, CONFIG, SPLIT = "MMInstruction/VL-RewardBench", "default", "test"
REVISION = "0e6e62701eba92818a69ce95af0ed7aa0648b176"
N_FULL = 1247
INSTRUCTIONS = ("Two candidate responses answer the user's query about the image. "
                "Which response is better (more accurate, helpful and free of hallucination)?")


def main():
    args = image_args()
    w = ImageSetWriter(NAME, args)
    for row in hf_stream(HF_ID, CONFIG, SPLIT, REVISION, image_cols=["image"]):
        if w.full:
            break
        resp, rank = list(row["response"]), list(row["human_ranking"])
        if len(resp) != 2 or sorted(rank) != [0, 1]:
            w.skip("not_a_strict_pair")
            continue
        if row.get("image") is None:
            w.skip("no_image")
            continue
        best = rank.index(0)  # rank 0 = preferred
        # upstream puts the preferred response first in 1,226/1,247 rows; swap half (hash of id) so
        # "always pick A" scores ~50% and position bias is not rewarded
        swapped = bool(stable_bit(row["id"]))
        if swapped:
            resp, best = resp[::-1], 1 - best
        criteria = {"response_a": resp[0], "response_b": resp[1]}
        rid = w.next_id(SPLIT)
        meta = {"grid_id": f"{NAME}/{row['id']}", "orig_id": row["id"], "orig_answer_type": "pairwise",
                "query_source": row["query_source"], "models": list(row["models"]), "judge": row["judge"],
                "swapped": swapped}
        for k in ("rationale", "ground_truth"):
            if row.get(k):
                meta[k] = row[k]
        w.add({
            "id": rid, "source": NAME, "split": SPLIT, "license": "non-commercial", "lang": "en",
            "state": {"context": row["query"], "image": w.image(row["image"], rid.rsplit("/", 1)[1])},
            "questions": {"q1": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}},
            "targets": {"q1": one_hot(criteria, ["response_a", "response_b"][best])},
            "meta": meta,
        })
    w.finish({"hf_id": HF_ID, "hf_config": CONFIG, "hf_split": SPLIT, "hf_revision": REVISION, "n_records_full": N_FULL,
              "license": "non-commercial", "lang": "en",
              "notes": "human_ranking [0,1] = first response preferred. Upstream order is 1,226:21 in favour of "
                       "the first response, so the pair is swapped for ~half the rows by a stable hash of the id "
                       "(meta.swapped). License: HF tag says MIT but the card says 'Research use only ... "
                       "restricted by the license agreements of GPT-4o and Claude' -> non-commercial."})


if __name__ == "__main__":
    main()
