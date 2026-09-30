"""Shared loop for the text_* converters: stream an HF split, convert rows, seeded sample, write. See data/README.md."""
import random
import time

from data.sources._common import RawWriter, hf_stream, record, reservoir, source_args


def _retry(fn, tries=6):
    for k in range(tries):
        try:
            return fn()
        except Exception:  # transient network errors
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def _row_groups(files):
    """[(file, row group, upstream index of its first row)] over the split's parquet files, from the footers only."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs, out, base = HfFileSystem(), [], 0
    for f in dict.fromkeys(files):
        md = _retry(lambda: pq.ParquetFile(fs.open(f.removeprefix("hf://"), cache_type="none")).metadata)
        for i in range(md.num_row_groups):
            out.append((f, i, base))
            base += md.row_group(i).num_rows
    return fs, out


def upstream(ds, hf_id, seed, scan):
    """Yield (row_index, row) from a streaming split. scan == 0: the whole split in upstream order. scan > 0: parquet
    row groups in seeded random order (one range request per column chunk, nothing else fetched), stopping after
    `scan` rows; a local shortcut for big sets on a slow link. Row indices are the same either way."""
    if not scan:
        yield from enumerate(ds)
        return
    import pyarrow.parquet as pq

    files = [f for f in getattr(ds._ex_iterable, "kwargs", {}).get("files") or [] if f.endswith(".parquet")]
    if not files:
        raise SystemExit(f"{hf_id}: --scan needs parquet files; run without --scan")
    fs, groups = _row_groups(files)
    random.Random(seed).shuffle(groups)
    read = 0
    for f, i, base in groups:
        rows = _retry(lambda: pq.ParquetFile(fs.open(f.removeprefix("hf://"), cache_type="none")).read_row_group(i).to_pylist())
        for j, row in enumerate(rows):
            yield base + j, row
            read += 1
            if read >= scan:
                return


def run(name, hf_id, convert, *, lang, config=None, split="train", revision=None, id_field=None, label_col=None):
    """convert(row, rng, names) -> (state, questions, targets) or a skip-reason string.
    names: the ClassLabel names of `label_col` (None when not given)."""
    args = source_args(name, lambda ap: ap.add_argument(
        "--scan", type=int, default=0, help="read at most N upstream rows, random row groups first; 0 = whole split"))
    w = RawWriter(name, args.out)

    ds = hf_stream(hf_id, config, split, revision)
    names = ds.features[label_col].names if label_col else None

    def converted():
        for i, row in upstream(ds, hf_id, args.seed, args.scan):
            uid = row[id_field] if id_field else i
            grid_id = f"{name}/{split}/{uid}"
            out = convert(row, random.Random(f"{args.seed}:{grid_id}"), names)
            if isinstance(out, str):
                w.skip(out)
                continue
            yield grid_id, out

    kept = list(reservoir(converted(), args.limit, args.seed))
    random.Random(args.seed).shuffle(kept)
    n_q = 0
    for grid_id, (state, questions, targets) in kept:
        if args.limit and n_q + len(questions) > args.limit:
            break
        w.write(record(name, w.n, state, questions, targets, lang=lang, grid_id=grid_id, source_split=split))
        n_q += len(questions)
    w.close()


def words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])
