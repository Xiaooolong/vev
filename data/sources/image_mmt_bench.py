"""MMT-Bench VAL (single-image MC, up to 9 options, base64 images in a TSV) -> choice (<= 8 options kept).
Only VAL carries answers (the ALL / test TSV has no answer column). The TSV is sorted by task, so a --limit run
samples rows at seeded random byte offsets (HTTP range reads); --limit 0 streams the whole file.

python -m data.sources.image_mmt_bench --out data/raw/image/mmt_bench.jsonl --limit 30
"""
import base64
import csv
import io
import random
import sys

from data.sources._common import RawWriter, image_from_bytes, one_hot, record, save_image, source_args
from data.sources._parquet import _fs, _retry, bytes_id

NAME = "mmt_bench"
HF_ID, REVISION, FILE = "Kaining/MMT-Bench", "498043370ba1dd8f89bbea509b6b981c8d53f07c", "MMT-Bench_VAL.tsv"
LABELS = "ABCDEFGHI"
MAX_OPTIONS = 8
csv.field_size_limit(2 ** 31 - 1)


def parse_line(line: str, header: list[str]):
    fields = next(csv.reader([line], delimiter="\t"), None)
    return dict(zip(header, fields)) if fields and len(fields) == len(header) else None


def random_rows(fh, size: int, start: int, header: list[str], rng, want: int):
    """Rows at random byte offsets: seek, drop the partial line, parse the next. Duplicates are left to the caller."""
    for _ in range(want * 5):
        fh.seek(rng.randrange(start, size))
        fh.readline()
        line = fh.readline().decode("utf-8").rstrip("\r\n")
        row = parse_line(line, header) if line else None
        yield row


def seq_rows(fh, header):
    text = io.TextIOWrapper(fh, encoding="utf-8", newline="")
    for fields in csv.reader(text, delimiter="\t"):
        yield dict(zip(header, fields)) if len(fields) == len(header) else None


def main():
    args = source_args(NAME)
    w = RawWriter(NAME, args.out)
    fs, rng = _fs(), random.Random(args.seed)
    path = f"datasets/{HF_ID}@{REVISION}/{FILE}"
    size = _retry(lambda: fs.info(path)["size"])
    fh = fs.open(path, block_size=256 << 10)
    header = fh.readline().decode("utf-8").rstrip("\r\n").split("\t")
    rows = random_rows(fh, size, fh.tell(), header, rng, args.limit) if args.limit else seq_rows(fh, header)
    seen, saved = set(), {}
    for row in rows:
        if args.limit and w.n >= args.limit:
            break
        if row is None:
            w.skip("row_unparsed")
            continue
        if row["index"] in seen:
            continue
        seen.add(row["index"])
        opts = {k: row[k].strip() for k in LABELS if row.get(k, "").strip() not in ("", "nan")}
        ans = row["answer"].strip()
        if list(opts) != list(LABELS[:len(opts)]) or len(opts) < 2:
            w.skip("labels_not_consecutive")
            continue
        if len(opts) > MAX_OPTIONS:
            w.skip("more_than_8_options")
            continue
        if ans not in opts:
            w.skip("answer_not_in_options")
            continue
        img = row["image"].strip()
        if img.isdigit():  # VLMEvalKit dedup: a reference to another row's image
            if img not in saved:
                w.skip("image_ref_unresolved")
                continue
            image_path, image_id = saved[img]
        elif img.startswith("["):
            w.skip("multi_image")
            continue
        else:
            raw = base64.b64decode(img)
            image_id = bytes_id(raw)
            image_path = save_image(image_from_bytes(raw), NAME, image_id)
            saved[row["index"]] = (image_path, image_id)
        w.write(record(NAME, w.n, {"image": {"path": image_path}},
                       {"q1": {"type": "choice", "instructions": row["question"].strip(), "criteria": opts}},
                       {"q1": one_hot(opts, ans)}, lang="en", grid_id=f"{NAME}/{image_id}", source_split="val",
                       meta={"orig_index": int(row["index"]), "category": row["category"],
                             "l2_category": row["l2-category"], "image_sha1": image_id}))
    if args.limit and w.n < args.limit:
        print(f"{NAME}: only {w.n} usable rows in {args.limit * 5} random probes", file=sys.stderr)
    w.close()


if __name__ == "__main__":
    main()
