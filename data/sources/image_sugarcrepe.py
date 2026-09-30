"""SugarCrepe (7 hard-negative caption subsets on COCO val2017) -> noul "does this description match the image".
Each item has a positive and a hard-negative caption; one of the two is used per item, chosen by a stable hash,
so positives and negatives are ~half each and never sit next to each other in one record.

NOTE: every image is COCO val2017, so data/dedup.py's COCO-val rule drops this whole source (meta.coco_id is
recorded for exactly that check).

python -m data.sources.image_sugarcrepe --limit 30
"""
import json
import random

from data.sources._common import RawWriter, image_from_bytes, record, save_image, stable_bit
from data.sources._remote import image_source_args, http_bytes, http_cached

NAME = "sugarcrepe"
COMMIT = "0047054b243992f0fad63d6f64f7544862daf846"  # github.com/RAIVNLab/sugar-crepe
SUBSETS = ["add_att", "add_obj", "replace_att", "replace_obj", "replace_rel", "swap_att", "swap_obj"]
COCO_URL = "http://images.cocodataset.org/val2017/{}"
INSTR = 'Does the following description accurately match the image? Description: "{}"'


def main():
    args = image_source_args(NAME)
    w = RawWriter(NAME, args.out)
    items = []
    for sub in SUBSETS:
        url = f"https://raw.githubusercontent.com/RAIVNLab/sugar-crepe/{COMMIT}/data/{sub}.json"
        data = json.loads(http_cached(url, NAME, f"{sub}.json").read_text(encoding="utf-8"))
        items += [(sub, k, v) for k, v in data.items()]
    if args.limit:
        items = random.Random(args.seed).sample(items, min(args.limit, len(items)))
    for idx, (sub, key, it) in enumerate(items):
        coco_id = int(it["filename"].rsplit(".", 1)[0])
        path = save_image(image_from_bytes(http_bytes(COCO_URL.format(it["filename"]))), NAME, f"{coco_id:012d}")
        positive = bool(stable_bit(f"{NAME}/{sub}/{key}"))
        caption = it["caption"] if positive else it["negative_caption"]
        q = {"type": "noul", "instructions": INSTR.format(caption.strip())}
        meta = {"coco_id": coco_id, "coco_split": "val2017", "subset": sub, "orig_key": key,
                "positive": positive, "other_caption": it["negative_caption"] if positive else it["caption"]}
        w.write(record(NAME, idx, {"image": {"path": path}}, {"q1": q}, {"q1": 1.0 if positive else 0.0},
                       lang="en", grid_id=f"{NAME}/coco_{coco_id}", source_split=sub, meta=meta))
    w.close()


if __name__ == "__main__":
    main()
