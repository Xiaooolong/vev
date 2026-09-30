"""NaturalBench (paired design: 2 images x 2 questions, answers flip across the pair) -> yes/no -> noul, MC -> choice.
Each (image, question) cell is one record; meta.pair records the cell and its partners.

python -m data.sources.image_naturalbench --out data/raw/image/naturalbench.jsonl --limit 30
"""
import re

from data.sources._common import RawWriter, image_from_bytes, one_hot, record, save_image, source_args
from data.sources._openqa import YES_NO
from data.sources._parquet import iter_row_groups, list_files

NAME = "naturalbench"
HF_ID, REVISION, SPLIT = "BaiqiL/NaturalBench", "ba41a7d564877a9b64c094b08015ca493cc3e54b", "train"
OPTION_TAIL = re.compile(r"\n\s*Options?:\s*(.*)$", re.S)
OPTION = re.compile(r"([A-Z])\s*:\s*(.*?)\s*(?:;|$)")


def parse_mc(question: str):
    """'What …?\\nOption: A:To the right; B:To the left;' -> (stem, {A: .., B: ..}) or None."""
    m = OPTION_TAIL.search(question)
    if not m:
        return None
    opts = {k: v for k, v in OPTION.findall(m.group(1).strip()) if v}
    if list(opts) != [chr(65 + i) for i in range(len(opts))] or len(opts) < 2:
        return None
    return question[:m.start()].strip(), opts


def main():
    args = source_args(NAME)
    w = RawWriter(NAME, args.out)
    done = lambda: args.limit and w.n >= args.limit  # noqa: E731
    for rows, where in iter_row_groups(list_files(HF_ID, REVISION, "data/*.parquet"), args.seed):
        for row in rows:
            idx, qtype = row["Index"], row["Question_Type"]
            paths = {}
            for i in (0, 1):
                for j in (0, 1):
                    if done():
                        break
                    q, ans = row[f"Question_{j}"], str(row[f"Image_{i}_Question_{j}"]).strip()
                    if qtype == "yes_no":
                        if ans.lower() not in YES_NO:
                            w.skip("answer_not_yes_no")
                            continue
                        question, target = {"type": "noul", "instructions": q.strip()}, YES_NO[ans.lower()]
                    elif qtype == "multiple_choice":
                        p = parse_mc(q)
                        if p is None or ans not in p[1]:
                            w.skip("options_unparsed" if p is None else "answer_not_in_options")
                            continue
                        question = {"type": "choice", "instructions": p[0], "criteria": p[1]}
                        target = one_hot(p[1], ans)
                    else:
                        w.skip(f"type_{qtype}")
                        continue
                    if i not in paths:
                        paths[i] = save_image(image_from_bytes(row[f"Image_{i}"]["bytes"]), NAME, f"{idx}_img{i}")
                    pair = {"index": idx, "image": i, "question": j, "question_type": qtype,
                            "partner_image_grid": f"{NAME}/{idx}_img{1 - i}", "group": f"{NAME}/{idx}"}
                    w.write(record(NAME, w.n, {"image": {"path": paths[i]}}, {"q1": question}, {"q1": target},
                                   lang="en", grid_id=f"{NAME}/{idx}_img{i}", source_split=SPLIT,
                                   meta={"shard": where, "pair": pair, "orig_source": row["Source"]}))
            if done():
                break
        if done():
            break
    w.close()


if __name__ == "__main__":
    main()
