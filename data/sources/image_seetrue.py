"""SeeTRUE (image-text pairs from DrawBench / EditBench / COCO-t2i, human binary "text matches image") -> noul.

The upstream card says SeeTRUE "should be used as a TEST SET, not as a training set" (gated prompt); converted as
asked, the license entry decides whether build.py lets it through. Real COCO images (dataset_source coco_t2i, file
names '*_coco_val_*') are COCO val with no COCO id upstream: flagged with meta.coco_val = true.

python -m data.sources.image_seetrue --limit 30
"""
import random
import sys

import pandas as pd

from data.sources._common import RawWriter, image_from_bytes, record, save_image
from data.sources._remote import RemoteZip, hf_download, hf_open, image_source_args

NAME = "seetrue"
HF_ID, REVISION, SPLIT = "yonatanbitton/SeeTRUE", "6cc46a6e5d9388f44fdf9096d5387cd3b7988676", "test"
INSTR = 'Does the following description accurately match the image? Description: "{}"'


def main():
    args = image_source_args(NAME)
    w = RawWriter(NAME, args.out)
    df = pd.read_csv(hf_download(HF_ID, "SeeTRUE.csv", REVISION))
    z = RemoteZip(lambda: hf_open(HF_ID, "images.zip", REVISION, block_size=1 << 18), "images.zip")
    members = {m.rsplit("/", 1)[-1]: m for m in z.namelist() if not m.endswith("/")}
    order = list(range(len(df)))
    if args.limit:
        random.Random(args.seed).shuffle(order)
    idx = 0
    for i in order:
        if args.limit and idx >= args.limit:
            break
        r = df.iloc[i]
        if r["label"] not in (0, 1) or not isinstance(r["text"], str):
            w.skip("bad_row")
            continue
        if r["image"] not in members:
            w.skip("image_missing_in_zip")
            continue
        stem = r["image"].rsplit(".", 1)[0]
        raw = z.read(members[r["image"]])
        try:
            path = save_image(image_from_bytes(raw), NAME, stem)
        except (OSError, SyntaxError) as e:
            print(f"  undecodable {r['image']}: {e}", file=sys.stderr)
            w.skip("undecodable_image")
            continue
        meta = {"dataset_source": r["dataset_source"], "original_dataset_id": r["original_dataset_id"],
                "orig_image": r["image"]}
        if "coco_val" in r["image"]:
            meta["coco_val"] = True
        q = {"type": "noul", "instructions": INSTR.format(r["text"].strip())}
        w.write(record(NAME, idx, {"image": {"path": path}}, {"q1": q}, {"q1": float(r["label"])}, lang="en",
                       grid_id=f"{NAME}/{stem}", source_split=SPLIT, meta=meta))
        idx += 1
    w.close()


if __name__ == "__main__":
    main()
