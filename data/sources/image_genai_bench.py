"""GenAI-Bench (1,600 compositional prompts x 6 text-to-image models, three human 1-5 alignment ratings each)
-> score (5 levels), target = the three raters' rating distribution.

Sampling unit = (shard, row group, model column): each unit reads one image column chunk (DALL-E 3 chunks are
~260 MB, the others 1-16 MB).

python -m data.sources.image_genai_bench --limit 30
"""
import random

import pyarrow.parquet as pq

from data.sources._common import RawWriter, image_from_bytes, record, save_image
from data.sources._remote import hf_open, image_source_args, parquet_units, retry

NAME = "genai_bench"
HF_ID, REVISION, SPLIT = "BaiqiL/GenAI-Bench", "94e543b0801feade1419873335191de90b68d164", "train"
FILES = [f"data/train-{i:05d}-of-00012.parquet" for i in range(12)]
MODELS = ["DALLE_3", "DeepFloyd_I_XL_v1", "Midjourney_6", "SDXL_2_1", "SDXL_Base", "SDXL_Turbo"]
Q = {"type": "score", "instructions": "How well does this generated image match the text-to-image prompt?",
     "criteria": ["1 - does not match at all", "2 - has significant discrepancies", "3 - has several minor "
                  "discrepancies", "4 - has a few minor discrepancies", "5 - matches exactly"]}


def main():
    args = image_source_args(NAME)
    w = RawWriter(NAME, args.out)
    rng = random.Random(args.seed)
    units = [(f, g, md, m) for f, g, _, md in parquet_units(HF_ID, FILES, REVISION) for m in MODELS]
    if args.limit:
        rng.shuffle(units)
    per_unit = max(10, args.limit // 50) if args.limit else None
    idx = 0
    for f, g, md, model in units:
        if args.limit and idx >= args.limit:
            break

        def read():
            with hf_open(HF_ID, f, REVISION) as fh:
                return pq.ParquetFile(fh, metadata=md).read_row_group(
                    g, columns=["Index", "Prompt", "Tags", "HumanRatings", model]).to_pylist()
        rows = retry(read, f"{f} rg {g} {model}")
        order = list(range(len(rows)))
        if args.limit:
            rng.shuffle(order)
            order = order[:per_unit]
        for i in order:
            if args.limit and idx >= args.limit:
                break
            r = rows[i]
            ratings = [int(x) for x in (r["HumanRatings"] or {}).get(model) or [] if 1 <= int(x) <= 5]
            if not ratings or not r.get(model) or not r[model].get("bytes"):
                w.skip("no_rating_or_image")
                continue
            image_id = f"{r['Index']:04d}_{model}"
            path = save_image(image_from_bytes(r[model]["bytes"]), NAME, image_id)
            target = {str(k): ratings.count(k + 1) / len(ratings) for k in range(5)}
            meta = {"prompt_index": r["Index"], "model": model, "ratings": ratings,
                    "tags_basic": list(r["Tags"]["basic"] or []), "tags_advanced": list(r["Tags"]["advanced"] or [])}
            w.write(record(NAME, idx, {"prompt": r["Prompt"], "image": {"path": path}}, {"q1": dict(Q)},
                           {"q1": target}, lang="en", grid_id=f"{NAME}/{image_id}", source_split=SPLIT, meta=meta))
            idx += 1
    w.close()


if __name__ == "__main__":
    main()
