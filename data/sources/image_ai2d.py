"""Cauldron ai2d (science diagrams, 4-way MC) -> choice.

python -m data.sources.image_ai2d --out data/raw/image/ai2d.jsonl --limit 30
"""
from data.sources._common import RawWriter, one_hot, record, source_args
from data.sources._openqa import parse_lettered
from data.sources._parquet import cauldron_image, cauldron_row_groups

NAME = CONFIG = "ai2d"


def run(name: str, config: str):
    args = source_args(name)
    w = RawWriter(name, args.out)
    done = lambda: args.limit and w.n >= args.limit  # noqa: E731
    for rows, where in cauldron_row_groups(config, args.seed):
        for row in rows:
            image = None
            for t in row["texts"]:
                if done():
                    break
                p = parse_lettered(t["user"], t["assistant"])
                if isinstance(p, str):
                    w.skip(p)
                    continue
                q, opts, ans = p
                if image is None:
                    image = cauldron_image(row["images"][0], name)
                path, image_id, extra = image
                w.write(record(name, w.n, {"image": {"path": path}},
                               {"q1": {"type": "choice", "instructions": q, "criteria": opts}},
                               {"q1": one_hot(opts, ans)}, lang="en", grid_id=f"{name}/{image_id}",
                               source_split="train", meta={"upstream": f"the_cauldron/{config}", "shard": where, **extra}))
            if done():
                break
        if done():
            break
    w.close()


if __name__ == "__main__":
    run(NAME, CONFIG)
