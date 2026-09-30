"""CharXiv (arXiv charts) descriptive questions -> 4-way choice; reasoning questions are skipped.

Question text = CharXiv's own template (src/constants.py DESCRIPTIVE_RESP_INST) with its subplot prefix.
Distractors (data/README.md open->choice rule): answers to the chart's other descriptive questions of the same
kind (number vs text) first, then a same-type perturbation (numbers +-10-50 %, "n by m" layouts, Yes/No/Not
Applicable), then answers to the same template on other charts of the same row group, then "Not Applicable".
Only the val split has public answers (test answers are withheld).

python -m data.sources.image_charxiv --limit 30
"""
import ast
import math
import random
import re

from data.sources._common import RawWriter, image_from_bytes, one_hot, record, save_image
from data.sources._remote import hf_open, http_cached, image_source_args, parquet_units, retry

NAME = "charxiv"
HF_ID, REVISION, SPLIT = "princeton-nlp/CharXiv", "f441eb632fc62f6f777830a0f47619e6e86459b0", "val"
GH_COMMIT = "7ebe88f78dee387691551f071abcb2b9e1a8025b"  # github.com/princeton-nlp/CharXiv
LABELS = "ABCD"
NA = "Not Applicable"
NUM = re.compile(r"^-?\d+(?:\.\d+)?$")
LAYOUT = re.compile(r"^(\d+) by (\d+)$")


def templates() -> dict:
    src = http_cached(f"https://raw.githubusercontent.com/princeton-nlp/CharXiv/{GH_COMMIT}/src/constants.py",
                      NAME, "constants.py").read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and node.targets[0].id == "DESCRIPTIVE_RESP_INST":
            return {k: "\n".join(line.strip() for line in v.strip().splitlines())
                    for k, v in ast.literal_eval(node.value).items()}
    raise ValueError("DESCRIPTIVE_RESP_INST not found")


def prefix(row) -> str:  # CharXiv descriptive_utils.descriptive_query_helper
    loc = row.get("subplot_loc")
    if isinstance(loc, str) and loc.strip():
        return f"For {loc.strip()}, "
    r, c = row.get("subplot_row"), row.get("subplot_col")
    if r is None or (isinstance(r, float) and math.isnan(r)) or int(r) == 0:
        return "For the current plot, "
    return f"For the subplot at row {int(r)} and column {int(c)}, "


def perturb(ans: str, rng: random.Random) -> list[str]:
    if NUM.match(ans):
        x = float(ans)
        dec = len(ans.split(".")[1]) if "." in ans else 0
        out = []
        for _ in range(12):
            y = x * (1 + rng.choice([-1, 1]) * rng.uniform(0.1, 0.5)) if x != 0 else rng.choice([-1, 1]) * rng.randint(1, 5)
            if x >= 0 and y < 0:
                y = -y
            s = f"{y:.{dec}f}" if dec else str(int(round(y)))
            if s != ans and s not in out and s not in ("-0", "0" if x == 0 else ""):
                out.append(s)
        return out
    m = LAYOUT.match(ans)
    if m:
        n, k = int(m.group(1)), int(m.group(2))
        return [f"{a} by {b}" for a, b in [(k, n), (n + 1, k), (n, k + 1), (max(1, n - 1), k), (n, max(1, k - 1))]
                if (a, b) != (n, k)]
    if ans in ("Yes", "No"):
        return ["No" if ans == "Yes" else "Yes"]
    return []


def kind(a: str) -> str:
    return "num" if NUM.match(a) else "layout" if LAYOUT.match(a) else "bool" if a in ("Yes", "No") else "text"


def options_for(ans: str, same_chart: list[str], pool: list[str], rng: random.Random) -> list[str]:
    seen, opts = {ans.lower()}, [ans]

    def add(cands):
        for c in cands:
            if len(opts) == 4:
                return
            if c and c.lower() not in seen:
                seen.add(c.lower())
                opts.append(c)

    add([a for a in same_chart if kind(a) == kind(ans) and a != NA])
    add(perturb(ans, rng))
    add(rng.sample(pool, len(pool)))
    add([NA])
    return opts


def main():
    args = image_source_args(NAME)
    w = RawWriter(NAME, args.out)
    tmpl = templates()
    import pyarrow.parquet as pq
    rng = random.Random(args.seed)
    units = parquet_units(HF_ID, [f"{SPLIT}.parquet"], REVISION)
    if args.limit:
        rng.shuffle(units)
    per_unit = max(4, args.limit // 40) if args.limit else None
    idx = n_q = 0
    for f, g, _, md in units:
        if args.limit and n_q >= args.limit:
            break

        def read():
            with hf_open(HF_ID, f, REVISION) as fh:
                return pq.ParquetFile(fh, metadata=md).read_row_group(g).to_pylist()
        rows = retry(read, f"{f} row group {g}")
        pool = {}  # template id -> answers on the charts of this row group
        for r in rows:
            for k in range(1, 5):
                if r[f"descriptive_a{k}"]:
                    pool.setdefault(r[f"descriptive_q{k}"], []).append(str(r[f"descriptive_a{k}"]).strip())
        order = list(range(len(rows)))
        if args.limit:
            rng.shuffle(order)
            order = order[:per_unit]
        for i in order:
            if args.limit and n_q >= args.limit:
                break
            r = rows[i]
            answers = [(r[f"descriptive_q{k}"], str(r[f"descriptive_a{k}"]).strip()) for k in range(1, 5)
                       if r[f"descriptive_a{k}"] is not None]
            questions, targets = {}, {}
            for n, (qid, ans) in enumerate(answers, 1):
                others = [a for q, a in answers if q != qid]
                opts = options_for(ans, others, [a for a in pool.get(qid, []) if a != ans], rng)
                if len(opts) < 2:
                    w.skip("no_distractor")
                    continue
                rng.shuffle(opts)
                crit = dict(zip(LABELS, opts))
                questions[f"q{n}"] = {"type": "choice", "instructions": tmpl[qid].format(prefix(r)), "criteria": crit}
                targets[f"q{n}"] = one_hot(crit, LABELS[opts.index(ans)])
            if not questions:
                continue
            stem = r["figure_path"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
            path = save_image(image_from_bytes(r["image"]["bytes"]), NAME, f"{SPLIT}_{stem}")
            meta = {"arxiv_id": r["original_id"], "category": r["category"], "year": r["year"],
                    "template_ids": [q for q, _ in answers], "orig_answer_type": "open", "row_group": g}
            w.write(record(NAME, idx, {"image": {"path": path}}, questions, targets, lang="en",
                           grid_id=f"{NAME}/{SPLIT}/{stem}", source_split=SPLIT, meta=meta))
            idx += 1
            n_q += len(questions)
    w.close()


if __name__ == "__main__":
    main()
