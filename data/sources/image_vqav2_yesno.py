"""Cauldron vqav2, yes/no questions -> noul. Images are shared with vqav2_mc (images/vqav2/, same grid_id).

python -m data.sources.image_vqav2_yesno --out data/raw/image/vqav2_yesno.jsonl --limit 30
"""
from data.sources._common import RawWriter, record, source_args
from data.sources._openqa import question_line, yes_no
from data.sources._parquet import cauldron_image, cauldron_row_groups

NAME, CONFIG, IMAGE_DIR = "vqav2_yesno", "vqav2", "vqav2"


def main():
    args = source_args(NAME)
    w = RawWriter(NAME, args.out)
    for rows, where in cauldron_row_groups(CONFIG, args.seed):
        for row in rows:
            qa = [(question_line(u), yes_no(a)) for u, a in ((t["user"], t["assistant"]) for t in row["texts"])]
            qa = [(q, y) for q, y in qa if y is not None]
            if not qa:
                continue
            path, image_id, extra = cauldron_image(row["images"][0], IMAGE_DIR)
            for q, y in qa:
                if args.limit and w.n >= args.limit:
                    break
                w.write(record(NAME, w.n, {"image": {"path": path}}, {"q1": {"type": "noul", "instructions": q}},
                               {"q1": y}, lang="en", grid_id=f"{IMAGE_DIR}/{image_id}", source_split="train",
                               meta={"upstream": f"the_cauldron/{CONFIG}", "shard": where, **extra}))
            if args.limit and w.n >= args.limit:
                break
        if args.limit and w.n >= args.limit:
            break
    w.close()


if __name__ == "__main__":
    main()
