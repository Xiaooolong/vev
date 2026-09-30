"""Cauldron vqav2, non-yes/no short answers -> 4-way choice with rule-based distractors (data/README.md).
Images are shared with vqav2_yesno (images/vqav2/, same grid_id).

python -m data.sources.image_vqav2_mc --out data/raw/image/vqav2_mc.jsonl --limit 30
"""
import random

from data.sources._common import RawWriter, record, source_args
from data.sources._openqa import AnswerPool, clean_answer, pick_distractors, question_line, to_choice, yes_no
from data.sources._parquet import cauldron_image, cauldron_row_groups

NAME, CONFIG, IMAGE_DIR = "vqav2_mc", "vqav2", "vqav2"


def main():
    args = source_args(NAME)
    w = RawWriter(NAME, args.out)
    rng, pool = random.Random(args.seed), AnswerPool()
    done = lambda: args.limit and w.n >= args.limit  # noqa: E731
    for rows, where in cauldron_row_groups(CONFIG, args.seed):
        parsed = [[(question_line(t["user"]), clean_answer(t["assistant"])) for t in row["texts"]] for row in rows]
        for qa in parsed:  # pool first so the first rows of a shard already have typed distractors
            for q, a in qa:
                if yes_no(a) is None:
                    pool.add(q, a)
        for row, qa in zip(rows, parsed):
            image = None
            for i, (q, a) in enumerate(qa):
                if done():
                    break
                if yes_no(a) is not None:
                    continue
                picked = pick_distractors(q, a, qa[:i] + qa[i + 1:], pool, rng)
                if picked is None:
                    w.skip("too_few_distractors")
                    continue
                if image is None:
                    image = cauldron_image(row["images"][0], IMAGE_DIR)
                path, image_id, extra = image
                crit, target, key = to_choice(a, picked[0], rng)
                w.write(record(NAME, w.n, {"image": {"path": path}},
                               {"q1": {"type": "choice", "instructions": q, "criteria": crit}}, {"q1": target},
                               lang="en", grid_id=f"{IMAGE_DIR}/{image_id}", source_split="train",
                               meta={"upstream": f"the_cauldron/{CONFIG}", "shard": where, "orig_answer": a,
                                     "distractor_source": picked[1], **extra}))
            if done():
                break
        if done():
            break
    w.close()


if __name__ == "__main__":
    main()
