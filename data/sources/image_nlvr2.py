"""Cauldron nlvr2 (two images + a statement, true/false) -> noul.
state = {"left": {"image"}, "right": {"image"}}; the statement goes into the question instructions, so all
statements about one image pair share the pair's grid.

python -m data.sources.image_nlvr2 --out data/raw/image/nlvr2.jsonl --limit 30
"""
import re

from data.sources._common import RawWriter, record, source_args
from data.sources._openqa import yes_no
from data.sources._parquet import cauldron_image, cauldron_row_groups

NAME = CONFIG = "nlvr2"
QUOTED = re.compile(r'"(.+)"', re.S)
TEMPLATE = "Is this statement true of the two images (left and right)? {}"


def main():
    args = source_args(NAME)
    w = RawWriter(NAME, args.out)
    done = lambda: args.limit and w.n >= args.limit  # noqa: E731
    for rows, where in cauldron_row_groups(CONFIG, args.seed):
        for row in rows:
            if len(row["images"]) != 2:
                w.skip("not_two_images")
                continue
            images = None
            for t in row["texts"]:
                if done():
                    break
                m, y = QUOTED.search(t["user"]), yes_no(t["assistant"])
                if not m or y is None:
                    w.skip("unparsed")
                    continue
                statement = m.group(1).strip()
                if images is None:
                    images = [cauldron_image(im, NAME) for im in row["images"]]
                (lp, lid, _), (rp, rid, _) = images
                pair = re.sub(r"-img0$", "", lid)
                split = pair.split("-", 1)[0] if pair.startswith(("train-", "dev-", "test1-")) else "train"
                state = {"left": {"image": {"path": lp}}, "right": {"image": {"path": rp}}}
                w.write(record(NAME, w.n, state, {"q1": {"type": "noul", "instructions": TEMPLATE.format(statement)}},
                               {"q1": y}, lang="en", grid_id=f"{NAME}/{pair}", source_split=split,
                               meta={"upstream": f"the_cauldron/{CONFIG}", "shard": where, "statement": statement,
                                     "left_id": lid, "right_id": rid}))
            if done():
                break
        if done():
            break
    w.close()


if __name__ == "__main__":
    main()
