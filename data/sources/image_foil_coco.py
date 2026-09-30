"""FOIL-COCO (COCO captions with one word swapped for a wrong "foil" word) -> noul "does this description match
the image". Uses the train file, whose images are COCO train2014 (the test file uses val2014 and would be
dropped by dedup). Per original caption id one of {original, foil} is used, chosen by a stable hash.

Annotations: foilv1.0_train_2017.json (86 MB, Dropbox link from foilunitn.github.io), downloaded once into the
source cache. Images: images.cocodataset.org/train2014, one file each.

python -m data.sources.image_foil_coco --limit 30
"""
from pathlib import Path
from PIL import Image
import os
import json
import random
from collections import defaultdict

from data.sources._common import RawWriter, image_from_bytes, record, save_image, stable_bit
from data.sources._remote import http_bytes, http_cached, image_source_args

NAME = "foil_coco"
URL = "https://www.dropbox.com/s/bsmowpgz43pwkyd/foilv1.0_train_2017.json?dl=1"
COCO_URL = "http://images.cocodataset.org/train2014/COCO_train2014_{:012d}.jpg"
INSTR = 'Does the following description accurately match the image? Description: "{}"'


def main():
    args = image_source_args(NAME)
    w = RawWriter(NAME, args.out)
    data = json.loads(http_cached(URL, NAME, "foilv1.0_train_2017.json").read_text(encoding="utf-8"))
    groups = defaultdict(lambda: {"orig": None, "foils": []})  # original caption id -> original + foil captions
    for a in data["annotations"]:
        g = groups[(a["image_id"], a["id"])]
        if a["foil"]:
            g["foils"].append(a)
        else:
            g["orig"] = a
    keys = sorted(k for k, g in groups.items() if g["orig"] and g["foils"])
    if args.limit:
        keys = random.Random(args.seed).sample(keys, min(args.limit, len(keys)))
    for idx, (image_id, cap_id) in enumerate(keys):
        g = groups[(image_id, cap_id)]
        positive = bool(stable_bit(f"{NAME}/{cap_id}"))
        a = g["orig"] if positive else g["foils"][0]
        local_dir = os.environ.get("VEV_COCO_DIR")
        local = Path(local_dir) / f"COCO_train2014_{image_id:012d}.jpg" if local_dir else None
        if local is not None and local.exists():
            img = Image.open(local)
        else:
            img = image_from_bytes(http_bytes(COCO_URL.format(image_id)))
        path = save_image(img, NAME, f"{image_id:012d}")
        meta = {"coco_id": image_id, "coco_split": "train2014", "caption_id": cap_id, "foil_id": a["foil_id"],
                "positive": positive}
        if not positive:
            meta.update(target_word=a["target_word"], foil_word=a["foil_word"])
        q = {"type": "noul", "instructions": INSTR.format(a["caption"].strip())}
        w.write(record(NAME, idx, {"image": {"path": path}}, {"q1": q}, {"q1": 1.0 if positive else 0.0},
                       lang="en", grid_id=f"{NAME}/coco_{image_id}", source_split="train", meta=meta))
    w.close()


if __name__ == "__main__":
    main()
