"""MJ-Bench 'alignment' subset (prompt + two generated images, which follows the prompt better) -> choice records.

python -m evals.datasets.image_mjbench --out evals/data/image/mjbench.jsonl --limit 30
"""
from evals.datasets._common import ImageSetWriter, hf_stream, image_args, one_hot, stable_bit

NAME = "mjbench"
HF_ID, SUBSET = "MJ-Bench/MJ-Bench", "alignment"
DATA_FILES = {"train": f"data/{SUBSET}.parquet"}  # repo has no configs; each subset is one parquet file
REVISION = "4b9667374c71463f63ae663ddd88373d3ffc5ba9"
N_FULL = 729
INSTRUCTIONS = "Which of the two images better matches the text-to-image prompt?"
CRITERIA_DESC = {"image_a": "the image under state.image_a", "image_b": "the image under state.image_b"}


def main():
    args = image_args()
    w = ImageSetWriter(NAME, args)
    for i, row in enumerate(hf_stream(HF_ID, None, "train", REVISION, data_files=DATA_FILES,
                                      image_cols=["image0", "image1"])):
        if w.full:
            break
        if row["label"] not in (0, 1):
            w.skip("label_not_0_1")
            continue
        if row.get("image0") is None or row.get("image1") is None:
            w.skip("missing_image")
            continue
        imgs, best = [row["image0"], row["image1"]], row["label"]  # label = index of the preferred image
        # upstream label is 643:86 in favour of image0; swap half by a stable hash so position bias is not rewarded
        swapped = bool(stable_bit(f"{SUBSET}/{i}/{row['caption']}"))
        if swapped:
            imgs, best = imgs[::-1], 1 - best
        rid = w.next_id(SUBSET)
        seq = rid.rsplit("/", 1)[1]
        meta = {"grid_id": f"{NAME}/{SUBSET}/{i}", "orig_row": i, "subset": SUBSET, "swapped": swapped,
                "orig_answer_type": "pairwise"}
        if row.get("info"):
            meta["info"] = row["info"]
        w.add({
            "id": rid, "source": NAME, "split": SUBSET, "license": "commercial-ok", "lang": "en",
            "state": {"prompt": row["caption"], "image_a": w.image(imgs[0], f"{seq}_a"),
                      "image_b": w.image(imgs[1], f"{seq}_b")},
            "questions": {"q1": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": dict(CRITERIA_DESC)}},
            "targets": {"q1": one_hot(CRITERIA_DESC, ["image_a", "image_b"][best])},
            "meta": meta,
        })
    w.finish({"hf_id": HF_ID, "hf_config": f"data_files=data/{SUBSET}.parquet", "hf_split": "train",
              "hf_revision": REVISION, "n_records_full": N_FULL, "license": "commercial-ok", "lang": "en",
              "notes": "Subset 'alignment' (729 pairs) chosen because it is exactly 'which image follows the prompt'; "
                       "safety (672) / quality (1,120, ~7 GB) / bias (1,548, different task) not converted. The "
                       "file is a single 206 MB row group, so even --limit 30 downloads all of it. label = index of "
                       "the preferred image (per MJ-Bench eval code); upstream 643:86 favours image0, so ~half the "
                       "pairs are swapped by a stable hash (meta.swapped). Record split = 'alignment'. HF card: "
                       "Apache-2.0."})


if __name__ == "__main__":
    main()
