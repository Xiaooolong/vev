"""Cauldron textvqa (open OCR answers) -> 4-way choice with rule-based distractors, plus with p=0.5 a derived noul
"Is the answer: X?" (X correct / a distractor, half each). Yes/no questions go straight to noul.

python -m data.sources.image_textvqa --out data/raw/image/textvqa.jsonl --limit 30
"""
import random

from data.sources._common import RawWriter, record, source_args
from data.sources._openqa import AnswerPool, clean_answer, pick_distractors, question_line, to_choice, yes_no
from data.sources._parquet import cauldron_image, cauldron_row_groups

NAME = CONFIG = "textvqa"


def run(name: str, config: str, *, per_image: int = 0, fix_answer=None, derive_p: float = 0.0):
    """Shared by textvqa / plotqa. per_image: cap questions kept per image (0 = all)."""
    args = source_args(name)
    w = RawWriter(name, args.out)
    rng, pool = random.Random(args.seed), AnswerPool()
    fix = fix_answer or (lambda a: a)
    done = lambda: args.limit and w.n >= args.limit  # noqa: E731
    for rows, where in cauldron_row_groups(config, args.seed):
        parsed = [[(question_line(t["user"]), fix(clean_answer(t["assistant"]))) for t in row["texts"]] for row in rows]
        for qa in parsed:  # pool first so the first rows of a shard already have typed distractors
            for q, a in qa:
                if yes_no(a) is None:
                    pool.add(q, a)
        for row, qa in zip(rows, parsed):
            order = list(range(len(qa)))
            rng.shuffle(order)
            if per_image:
                order = order[:per_image]
            image = None
            for i in order:
                if done():
                    break
                q, a = qa[i]
                if not a:
                    w.skip("empty_answer")
                    continue
                yn = yes_no(a)
                picked = None
                if yn is None:
                    picked = pick_distractors(q, a, qa[:i] + qa[i + 1:], pool, rng)
                    if picked is None:
                        w.skip("too_few_distractors")
                        continue
                if image is None:
                    image = cauldron_image(row["images"][0], name)
                path, image_id, extra = image
                state = {"image": {"path": path}}
                meta = {"upstream": f"the_cauldron/{config}", "shard": where, "orig_answer": a, **extra}
                grid = f"{name}/{image_id}"
                if yn is not None:
                    w.write(record(name, w.n, state, {"q1": {"type": "noul", "instructions": q}}, {"q1": yn},
                                   lang="en", grid_id=grid, source_split="train", meta=meta))
                    continue
                crit, target, _ = to_choice(a, picked[0], rng)
                w.write(record(name, w.n, state, {"q1": {"type": "choice", "instructions": q, "criteria": crit}},
                               {"q1": target}, lang="en", grid_id=grid, source_split="train",
                               meta={**meta, "distractor_source": picked[1]}))
                if derive_p and rng.random() < derive_p and not done():
                    pos = rng.random() < 0.5
                    x = a if pos else rng.choice(picked[0])
                    w.write(record(name, w.n, state,
                                   {"q1": {"type": "noul", "instructions": f"{q} Is the answer: {x}?"}},
                                   {"q1": 1.0 if pos else 0.0}, lang="en", grid_id=grid, source_split="train",
                                   meta={**meta, "derived": ["q1"], "derived_from": f"{name}/raw/{w.n - 1:06d}"}))
            if done():
                break
        if done():
            break
    w.close()


if __name__ == "__main__":
    run(NAME, CONFIG, derive_p=0.5)
