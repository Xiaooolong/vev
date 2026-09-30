"""Row-group streaming over HF-hosted parquet: seeded shard order, one HTTP range per column chunk, no full download."""
import fnmatch
import hashlib
import io
import random
import re
import sys
import time

BIG_FILE_BYTES = 1024 ** 3  # shards this large (e.g. MME-RealWorld's 2 GB single-row-group files) are visited last
CAULDRON, CAULDRON_REV = "HuggingFaceM4/the_cauldron", "847a98a779b1652d65111daf20c972dfcd333605"


def _fs():
    from huggingface_hub import HfFileSystem
    return HfFileSystem()


def list_files(repo: str, revision: str, pattern: str) -> list[tuple[str, int]]:
    """[(hf path, size)] of the files matching `pattern` (relative to the repo root)."""
    fs = _fs()
    root = f"datasets/{repo}@{revision}"
    sub = pattern.rsplit("/", 1)[0] if "/" in pattern else ""
    found = _retry(lambda: fs.find(f"{root}/{sub}" if sub else root, detail=True))
    return sorted((p, d["size"]) for p, d in found.items() if fnmatch.fnmatch(p[len(root) + 1:], pattern))


def _retry(fn, tries: int = 4):
    for k in range(tries):
        try:
            return fn()
        except Exception:  # transient network errors
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


class ChunkedFile(io.RawIOBase):
    """Seekable read-only view of an HF file that fetches in CHUNK-sized range requests, each retried on its own.
    Long single GETs are sometimes cut near the end ("peer closed connection") and restart from 0; small ranges make
    a dropped connection cost one chunk."""
    CHUNK = 2 << 20

    def __init__(self, fs, path: str, size: int | None = None):
        from data.sources._localhf import local_for_hf_path
        self.fs, self.path, self.pos = fs, path, 0
        self.local = local_for_hf_path(path)  # VEV_HF_LOCAL=1: whole file in the HF cache, read from disk
        if self.local is not None:
            self.size = self.local.stat().st_size
            return
        self.size = size if size is not None else _retry(lambda: fs.info(path)["size"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        parts = []
        while self.pos < end:
            a, b = self.pos, min(end, self.pos + self.CHUNK)
            parts.append(_retry(lambda: self._range(a, b), tries=8))
            self.pos = b
        return b"".join(parts)

    def _range(self, a: int, b: int) -> bytes:
        if self.local is not None:
            with self.local.open("rb") as fh:
                fh.seek(a)
                return fh.read(b - a)
        with self.fs.open(self.path, cache_type="none") as fh:  # the default readahead cache fetches MBs per read
            fh.seek(a)
            return fh.read(b - a)

    def readinto(self, buf):
        data = self.read(len(buf))
        buf[:len(data)] = data
        return len(data)


def iter_row_groups(files: list[tuple[str, int]], seed: int, columns=None):
    """Yield (rows, where) per row group: files and row groups in seeded order, rows shuffled within the group.
    Each row group costs one range request per column chunk; a caller that stops early never touches the rest.
    `where` = "<file basename>#rg<i>", a stable pointer back to the upstream shard."""
    import pyarrow.parquet as pq

    fs, rng = _fs(), random.Random(seed)
    order = files[:]
    rng.shuffle(order)
    order.sort(key=lambda f: f[1] >= BIG_FILE_BYTES)  # stable: seeded order kept inside each class
    for f, size in order:
        n_rg = _retry(lambda: pq.ParquetFile(ChunkedFile(fs, f, size)).metadata.num_row_groups)
        idx = list(range(n_rg))
        rng.shuffle(idx)
        for i in idx:
            def read():
                pf = pq.ParquetFile(ChunkedFile(fs, f, size))
                return pf.read_row_group(i, columns=columns).to_pylist()
            t0 = time.time()
            rows = _retry(read)
            print(f"  read {f.rsplit('/', 1)[1]}#rg{i}: {len(rows)} rows, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
            rng.shuffle(rows)
            yield rows, f"{f.rsplit('/', 1)[1]}#rg{i}"


def cauldron_row_groups(config: str, seed: int):
    return iter_row_groups(list_files(CAULDRON, CAULDRON_REV, f"{config}/*.parquet"), seed)


def bytes_id(raw: bytes) -> str:
    return hashlib.sha1(raw).hexdigest()[:16]


def cauldron_qa(row) -> list[tuple[str, str]]:
    """(user, assistant) pairs of a Cauldron row."""
    return [(t["user"], t["assistant"]) for t in row["texts"]]


def cauldron_image(im: dict, image_dir: str):
    """Save one Cauldron image struct; id = upstream file stem when Cauldron kept it, else sha1 of the bytes.
    Returns (record path, image_id, extra meta)."""
    from data.sources._common import image_from_bytes, save_image

    raw, name = im["bytes"], im.get("path")
    meta = {}
    if name:
        image_id = name.rsplit(".", 1)[0]
        m = re.match(r"COCO_(train|val)(\d{4})_0*(\d+)$", image_id)
        if m:
            meta["coco_id"], meta["coco_split"] = int(m.group(3)), f"{m.group(1)}{m.group(2)}"
    else:
        image_id = bytes_id(raw)
        meta["image_sha1"] = image_id
    return save_image(image_from_bytes(raw), image_dir, image_id), image_id, meta
