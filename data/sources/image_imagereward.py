"""ImageRewardDB (THUDM): generated image + prompt, expert overall rating 1-7 -> score (7 levels).

python -m data.sources.image_imagereward --limit 30
"""
import random
import sys
from collections import defaultdict

import pandas as pd

from data.sources._common import RawWriter, image_from_bytes, record, save_image, target_for
from data.sources._remote import image_source_args, RemoteZip, hf_download, hf_open

NAME = "imagereward"
HF_ID, REVISION, SPLIT = "THUDM/ImageRewardDB", "c493721e19e296eb615420036f2a2eed08412bb4", "train"
LEVELS = ["1 - very poor", "2 - poor", "3 - somewhat poor", "4 - fair", "5 - good", "6 - very good", "7 - excellent"]
QUESTION = {"type": "score", "criteria": LEVELS,
            "instructions": "Overall, how good is this generated image for the text-to-image prompt "
                            "(image quality and how well it follows the prompt together)?"}


def main():
    args = image_source_args(NAME)
    w = RawWriter(NAME, args.out)
    df = pd.read_parquet(hf_download(HF_ID, f"metadata-{SPLIT}.parquet", REVISION))
    # image_path = images/train/train_7/<uuid>.webp -> member <uuid>.webp of images/train/train_7.zip
    by_zip = defaultdict(list)
    for i, zip_dir in enumerate(df["image_path"].str.rsplit("/", n=1).str[0]):
        by_zip[zip_dir].append(i)
    rng = random.Random(args.seed)
    zips = sorted(by_zip)
    per_zip = None
    if args.limit:  # zips are the shards: visit them in seeded order, a bounded number of rows from each
        rng.shuffle(zips)
        per_zip = max(10, args.limit // 50)
    idx = 0
    for zip_dir in zips:
        rows = by_zip[zip_dir]
        if args.limit:
            rng.shuffle(rows)
            rows = rows[:per_zip]
        z = RemoteZip(lambda p=f"{zip_dir}.zip": hf_open(HF_ID, p, REVISION, block_size=1 << 18), zip_dir)
        for i in rows:
            if args.limit and idx >= args.limit:
                break
            r = df.iloc[i]
            rating = int(r["overall_rating"])
            if not 1 <= rating <= 7:
                w.skip("rating_out_of_range")
                continue
            member = r["image_path"].rsplit("/", 1)[1]
            image_id = member.rsplit(".", 1)[0]
            raw = z.read(member)
            try:
                path = save_image(image_from_bytes(raw), NAME, image_id)
            except (OSError, SyntaxError) as e:  # a few members are not decodable images
                print(f"  undecodable {zip_dir}/{member} ({len(raw)} bytes, head {raw[:12]!r}): {e}", file=sys.stderr)
                w.skip("undecodable_image")
                continue
            meta = {"prompt_id": r["prompt_id"], "classification": r["classification"], "rank": int(r["rank"]),
                    "n_images_for_prompt": int(r["image_amount_in_total"]),
                    "alignment_rating": int(r["image_text_alignment_rating"]),
                    "fidelity_rating": int(r["fidelity_rating"]), "orig_path": r["image_path"]}
            w.write(record(NAME, idx, {"prompt": r["prompt"], "image": {"path": path}}, {"q1": dict(QUESTION)},
                           {"q1": target_for(QUESTION, rating - 1)}, lang="en", grid_id=f"{NAME}/{image_id}",
                           source_split=SPLIT, meta=meta))
            idx += 1
        if args.limit and idx >= args.limit:
            break
    w.close()


if __name__ == "__main__":
    main()
