"""Shared helpers for the training-data converters (data/sources/*). See data/README.md."""
import argparse
import collections
import hashlib
import io
import json
import random
import sys
from pathlib import Path

from evals.datasets._common import hf_stream, one_hot, stable_bit, target_for  # noqa: F401  (re-exported)
from evals.schema import validate_record

DATA_DIR = Path(__file__).resolve().parent.parent
LICENSES = DATA_DIR / "licenses.json"
IMAGES_DIR = DATA_DIR / "images"
MAX_SIDE = 1024
JPEG_QUALITY = 90
DEFAULT_LIMIT = 30000


def _bucket_of_caller() -> str:
    """image_*.py converters write to raw/image, text_*.py to raw/text (decided by the entry script name)."""
    return "image" if Path(sys.argv[0]).stem.startswith("image_") else "text"


def source_args(name: str, extra=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=f"convert {name} into data/raw records")
    ap.add_argument("--out", default=str(DATA_DIR / "raw" / _bucket_of_caller() / f"{name}.jsonl"))
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="max questions to write; 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    if extra:
        extra(ap)
    return ap.parse_args()


def license_tier(source: str) -> str:
    """Tier from data/licenses.json; 'unknown' when the file or the source is missing."""
    if not LICENSES.exists():
        return "unknown"
    entry = json.loads(LICENSES.read_text(encoding="utf-8")).get(source)
    return entry.get("tier", "unknown") if entry else "unknown"


def reservoir(iterable, limit: int, seed: int):
    """Seeded reservoir sample of `limit` items from a stream (all items when limit == 0)."""
    if limit <= 0:
        yield from iterable
        return
    rng = random.Random(seed)
    kept: list = []
    for i, item in enumerate(iterable):
        if i < limit:
            kept.append(item)
        else:
            j = rng.randint(0, i)
            if j < limit:
                kept[j] = item
    yield from kept


def take(iterable, limit: int):
    """First `limit` items (all when limit == 0). Use when the stream is already shuffled upstream."""
    for i, item in enumerate(iterable):
        if limit and i >= limit:
            return
        yield item


def save_image(img, source: str, image_id: str) -> str:
    """Store a PIL image under data/images/<source>/, max side 1024, JPEG q90. Returns the record path."""
    from PIL import Image

    d = IMAGES_DIR / source
    d.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(image_id))[:120]
    path = d / f"{safe}.jpg"
    if not path.exists():
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        w, h = img.size
        scale = MAX_SIDE / max(w, h)
        if scale < 1.0:
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
        img.save(path, format="JPEG", quality=JPEG_QUALITY)
    return f"../../images/{source}/{safe}.jpg"


def image_from_bytes(raw: bytes):
    from PIL import Image

    return Image.open(io.BytesIO(raw))


def record(source: str, idx: int, state, questions: dict, targets: dict, *, lang: str, grid_id: str,
           source_split: str, meta: dict | None = None) -> dict:
    m = {"grid_id": grid_id, "source_split": source_split}
    if meta:
        m.update(meta)
    return {"id": f"{source}/raw/{idx:06d}", "source": source, "split": "raw", "license": license_tier(source),
            "lang": lang, "state": state, "questions": questions, "targets": targets, "meta": m}


class RawWriter:
    """Validates and writes records; prints the stats block every converter must end with."""

    def __init__(self, source: str, out: str):
        self.source, self.out = source, Path(out)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.n = 0
        self.by_type: collections.Counter = collections.Counter()
        self.by_lang: collections.Counter = collections.Counter()
        self.skipped: collections.Counter = collections.Counter()
        self._fh = self.out.open("w", encoding="utf-8")

    def skip(self, reason: str) -> None:
        self.skipped[reason] += 1

    def write(self, rec: dict) -> None:
        errs = validate_record(rec)
        if errs:
            raise ValueError(f"{rec.get('id')}: {errs}")
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.n += 1
        self.by_lang[rec["lang"]] += 1
        for q in rec["questions"].values():
            self.by_type[q["type"]] += 1

    def close(self) -> None:
        self._fh.close()
        sha = hashlib.sha256(self.out.read_bytes()).hexdigest()[:16]
        print(f"{self.source}: {self.n} records -> {self.out} (sha256 {sha}…)")
        print(f"  by_type  {dict(self.by_type)}")
        print(f"  by_lang  {dict(self.by_lang)}")
        print(f"  license  {license_tier(self.source)}")
        if self.skipped:
            print(f"  skipped  {dict(self.skipped)}")
        if self.n == 0:
            sys.exit(f"{self.source}: wrote no records")
