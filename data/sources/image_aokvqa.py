"""Cauldron aokvqa (4 options in the prompt, some answers carry a rationale) -> choice.

python -m data.sources.image_aokvqa --out data/raw/image/aokvqa.jsonl --limit 30
"""
import re

from data.sources._common import RawWriter, one_hot, record, source_args
from data.sources._openqa import question_line
from data.sources._parquet import cauldron_image, cauldron_row_groups

NAME = CONFIG = "aokvqa"


def parse(user: str, assistant: str):
    """'q\n<hint>\nOptions: A, b, c, d.' + 'D.' or 'Answer: d.\nRationale: …'. Options are comma-separated, so an
    option containing ', ' makes the split ambiguous: anything but exactly 4 distinct parts is skipped."""
    if "\nOptions: " not in user:
        return "no_options_line"
    tail = user.rsplit("\nOptions: ", 1)[1].strip()
    tail = tail[:-1] if tail.endswith(".") else tail
    opts = [o.strip() for o in tail.split(", ")]
    if opts and opts[0][:2].istitle():  # Cauldron capitalised the first option only; A-OKVQA choices are lowercase
        opts[0] = opts[0][0].lower() + opts[0][1:]
    if len(opts) != 4 or len({o.lower() for o in opts}) != 4 or not all(opts):
        return "options_not_4"
    m = re.match(r"^(?:Answer: )?(.*?)\.?(?:\nRationale: (.*))?$", assistant.strip(), re.S)
    ans = m.group(1).strip().lower()
    hits = [i for i, o in enumerate(opts) if o.lower() == ans]
    if len(hits) != 1:
        return "answer_not_in_options"
    crit = {chr(65 + i): o for i, o in enumerate(opts)}
    return question_line(user), crit, chr(65 + hits[0]), (m.group(2) or "").strip() or None


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
                p = parse(t["user"], t["assistant"])
                if isinstance(p, str):
                    w.skip(p)
                    continue
                q, crit, ans, rationale = p
                if image is None:
                    image = cauldron_image(row["images"][0], NAME)
                path, image_id, extra = image
                meta = {"upstream": f"the_cauldron/{CONFIG}", "shard": where, **extra}
                if rationale:
                    meta["rationale"] = rationale
                w.write(record(NAME, w.n, {"image": {"path": path}},
                               {"q1": {"type": "choice", "instructions": q, "criteria": crit}},
                               {"q1": one_hot(crit, ans)}, lang="en", grid_id=f"{NAME}/{image_id}",
                               source_split="train", meta=meta))
            if done():
                break
        if done():
            break
    w.close()


if __name__ == "__main__":
    main()
