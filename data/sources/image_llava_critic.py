"""LLaVA-Critic-113k (GPT-4o judgments on image + instruction + model responses).

- 0-100 "Final Score" judgments (pointwise config, svit-detail captions) -> score, 5 buckets of 20 points.
- Every two-response judgment -> choice a/b: pairwise config ("The better response: [first|second|1|2|A|B]",
  "equally good" = 0.5/0.5), pointwise config's LLaVA-bench "s1 s2" 1-10 score lines and MT-bench "[[A>B]]"
  verdicts. Strict preference -> one-hot, tie -> 0.5/0.5; the verdict strength (">>" vs ">", the two scores) is
  kept in meta, not baked into the target. The pair is swapped for half the rows by a stable hash of the id.
Rows in other formats are skipped and counted.

python -m data.sources.image_llava_critic --limit 30
"""
import re

from data.sources._common import RawWriter, image_from_bytes, one_hot, record, save_image, stable_bit
from data.sources._remote import image_source_args, iter_parquet

NAME = "llava_critic"
HF_ID, REVISION = "lmms-lab/LLaVA-Critic-113k", "7aa893ac47197c8300d7399d921836e12ceb5499"
FILES = [f"pointwise/train-{i:05d}-of-00006.parquet" for i in range(6)] + \
        [f"pairwise/train-{i:05d}-of-00005.parquet" for i in range(5)]
PAIR_INSTR = ("Two candidate responses answer the user's question about the image. "
              "Which response is better (more accurate, helpful and free of hallucination)?")
BUCKETS = ["0-19 - very poor", "20-39 - poor", "40-59 - fair", "60-79 - good", "80-100 - excellent"]

BRACKET = re.compile(r"Question: \[(.*?)\]\s*\n\s*(?:Response 1|Response A|The first response): \[(.*?)\]\s*\n\s*"
                     r"(?:Response 2|Response B|The second response): \[(.*)\]", re.S)
BETTER = re.compile(r"The better response: \[(\w+)\]")
ARENA = re.compile(r"<\|User Prompt\|>\s*(.*?)\s*<\|The Start of Assistant A's Answer\|>\s*(.*?)\s*"
                   r"<\|The End of Assistant A's Answer\|>\s*<\|The Start of Assistant B's Answer\|>\s*(.*?)\s*"
                   r"<\|The End of Assistant B's Answer\|>", re.S)
VERDICT = re.compile(r"\[\[(A>>B|A>B|A=B|B>A|B>>A)\]\]")
LLAVA_BENCH = re.compile(r"\[Question\]\s*(.*?)\s*\[Assistant 1\]\s*(.*?)\s*\[End of Assistant 1\]\s*"
                         r"\[Assistant 2\]\s*(.*?)\s*\[End of Assistant 2\]", re.S)
TWO_SCORES = re.compile(r"^\s*(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*\n")
CAPTION = re.compile(r"Text Caption:\s*(.*?)\s*\n(From 0 to 100, [^\n]*)", re.S)
FINAL = re.compile(r"Final Score:\s*(\d+(?:\.\d+)?)")


def parse(human: str, gpt: str):
    """-> ("pair", question, resp1, resp2, p_first, verdict) | ("score", caption, rubric, value) | None"""
    human = human.replace("<image>", "").strip()
    m = BRACKET.search(human)
    if m:
        b = BETTER.search(gpt)
        if b:
            first = b.group(1).lower() in ("first", "1", "a")
            if b.group(1).lower() not in ("first", "second", "1", "2", "a", "b"):
                return None
            return "pair", m.group(1), m.group(2), m.group(3).rstrip(), 1.0 if first else 0.0, b.group(1)
        if "equally good" in gpt[-300:]:
            return "pair", m.group(1), m.group(2), m.group(3).rstrip(), 0.5, "equal"
        return None
    m = ARENA.search(human)
    if m:
        v = VERDICT.findall(gpt)
        if not v:
            return None
        p = {"A>>B": 1.0, "A>B": 1.0, "A=B": 0.5, "B>A": 0.0, "B>>A": 0.0}[v[-1]]
        return "pair", m.group(1), m.group(2), m.group(3), p, v[-1]
    m = LLAVA_BENCH.search(human)
    if m:
        s = TWO_SCORES.match(gpt)
        if not s:
            return None
        s1, s2 = float(s.group(1)), float(s.group(2))
        return "pair", m.group(1), m.group(2), m.group(3), 1.0 if s1 > s2 else 0.0 if s1 < s2 else 0.5, f"{s1:g} {s2:g}"
    m = CAPTION.search(human)
    f = FINAL.findall(gpt)
    if m and f and 0 <= float(f[-1]) <= 100:
        return "score", m.group(1), m.group(2).strip(), float(f[-1])
    return None


def main():
    args = image_source_args(NAME)
    w = RawWriter(NAME, args.out)
    per_unit = max(10, args.limit // 50) if args.limit else None
    idx = 0
    for row in iter_parquet(HF_ID, FILES, REVISION, ["id", "source", "conversations", "image"], seed=args.seed,
                            limit=args.limit, per_unit=per_unit):
        if args.limit and idx >= args.limit:
            break
        conv = row["conversations"]
        if len(conv) != 2 or not row.get("image") or not row["image"].get("bytes"):
            w.skip("not_single_turn_with_image")
            continue
        parsed = parse(conv[0]["value"], conv[1]["value"])
        if parsed is None:
            w.skip("unparsed_format")
            continue
        config = row["_file"].split("/")[0]
        meta = {"orig_id": row["id"], "upstream_source": row["source"], "config": config}
        try:
            path = save_image(image_from_bytes(row["image"]["bytes"]), NAME, f"{config}_{row['id']}")
        except (OSError, SyntaxError):
            w.skip("undecodable_image")
            continue
        if parsed[0] == "pair":
            _, question, r1, r2, p_first, verdict = parsed
            swapped = bool(stable_bit(f"{NAME}/{row['id']}"))
            if swapped:
                r1, r2, p_first = r2, r1, 1.0 - p_first
            crit = {"a": r1.strip(), "b": r2.strip()}
            q = {"type": "choice", "instructions": PAIR_INSTR, "criteria": crit}
            target = {"a": p_first, "b": 1.0 - p_first}
            state = {"question": question.strip(), "image": {"path": path}}
            meta.update(verdict=verdict, swapped=swapped, orig_answer_type="pairwise")
        else:
            _, caption, rubric, value = parsed
            q = {"type": "score", "criteria": list(BUCKETS),
                 "instructions": "How correct and comprehensive is the caption under state.caption as a "
                                 "description of the image? (" + rubric + ")"}
            target = one_hot(range(5), min(4, int(value // 20)))
            state = {"caption": caption, "image": {"path": path}}
            meta.update(final_score=value, orig_answer_type="0-100")
        w.write(record(NAME, idx, state, {"q1": q}, {"q1": target}, lang="en", grid_id=f"{NAME}/{row['id']}",
                       source_split=f"{config}/train", meta=meta))
        idx += 1
    w.close()


if __name__ == "__main__":
    main()
