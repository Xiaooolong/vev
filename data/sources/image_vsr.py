"""Cauldron vsr (Visual Spatial Reasoning: is this spatial caption true of the image) -> noul.

python -m data.sources.image_vsr --out data/raw/image/vsr.jsonl --limit 30
"""
import re

from data.sources._common import RawWriter, record, source_args
from data.sources._openqa import yes_no
from data.sources._parquet import cauldron_image, cauldron_row_groups

NAME = CONFIG = "vsr"
QUOTED = re.compile(r'"(.+)"', re.S)  # Cauldron wraps the caption in one of ~10 paraphrased templates


def main():
    args = source_args(NAME)
    w = RawWriter(NAME, args.out)
    done = lambda: args.limit and w.n >= args.limit  # noqa: E731
    for rows, where in cauldron_row_groups(CONFIG, args.seed):
        for row in rows:
            image = None
            for t in row["texts"]:
                if done():
                    break
                m, y = QUOTED.search(t["user"]), yes_no(t["assistant"])
                if not m or y is None:
                    w.skip("unparsed")
                    continue
                caption = m.group(1).strip()
                if image is None:
                    image = cauldron_image(row["images"][0], NAME)
                path, image_id, extra = image
                q = f'Is this statement about the image true? "{caption}"'
                w.write(record(NAME, w.n, {"image": {"path": path}}, {"q1": {"type": "noul", "instructions": q}},
                               {"q1": y}, lang="en", grid_id=f"{NAME}/{image_id}", source_split="train",
                               meta={"upstream": f"the_cauldron/{CONFIG}", "shard": where, "caption": caption, **extra}))
            if done():
                break
        if done():
            break
    w.close()


if __name__ == "__main__":
    main()
