"""Shared helpers for the text_* converters: source checkout, one-hot targets, writing, stats, manifest."""
import argparse
import collections
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from evals.schema import validate_record

SCRATCH = Path(tempfile.gettempdir()) / "vev-sources"
MANIFEST = Path(__file__).resolve().parent.parent / "manifest" / "text.json"


def parse_args(name: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=str(SCRATCH / name), help="local clone of the upstream repo")
    ap.add_argument("--manifest", default=str(MANIFEST))
    return ap.parse_args()


def ensure_repo(url: str, commit: str, dest: str) -> Path:
    """Shallow-fetch the pinned commit if the clone is missing; refuse a clone at another commit."""
    d = Path(dest)
    if not (d / ".git").exists():
        d.mkdir(parents=True, exist_ok=True)
        for cmd in (["init", "-q"], ["remote", "add", "origin", url],
                    ["fetch", "-q", "--depth", "1", "origin", commit], ["checkout", "-q", "FETCH_HEAD"]):
            subprocess.run(["git", "-C", str(d), *cmd], check=True)
    head = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    if head != commit:
        sys.exit(f"{d} is at {head}, expected pinned {commit}")
    return d


def read_jsonl(path: Path, sha256: str | None = None) -> list[dict]:
    raw = path.read_bytes().replace(b"\r\n", b"\n")  # git autocrlf on Windows rewrites line endings
    if sha256 and hashlib.sha256(raw).hexdigest() != sha256:
        sys.exit(f"{path}: sha256 mismatch vs upstream-declared {sha256}")
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


def one_hot(keys, hit) -> dict:
    keys = [str(k) for k in keys]
    if str(hit) not in keys:
        raise ValueError(f"target {hit!r} not in {keys}")
    return {k: 1.0 if k == str(hit) else 0.0 for k in keys}


def target_for(q: dict, answer) -> object:
    """Hard answer -> distribution. choice: label; noul: bool; score: level index."""
    if q["type"] == "choice":
        return one_hot(q["criteria"], answer)
    if q["type"] == "noul":
        if not isinstance(answer, bool):
            raise ValueError(f"noul answer must be bool, got {answer!r}")
        return 1.0 if answer else 0.0
    return one_hot(range(len(q["criteria"])), int(answer))


def write_and_report(name: str, records: list[dict], out: str, repo_url: str, commit: str, manifest: str) -> None:
    bad = [(r["id"], e) for r in records for e in [validate_record(r)] if e]
    if bad:
        for rid, errs in bad[:10]:
            print(rid, errs, file=sys.stderr)
        sys.exit(f"{len(bad)} records failed validation")
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids)), "duplicate ids"

    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(out_p.read_bytes()).hexdigest()

    by_type = collections.Counter(q["type"] for r in records for q in r["questions"].values())
    lic = collections.Counter(r["license"] for r in records)
    lang = collections.Counter(r["lang"] for r in records)
    by_split = collections.Counter(r["split"] for r in records)
    n_q = sum(by_type.values())
    print(f"{name}: {len(records)} records, {n_q} questions -> {out_p}")
    print(f"  by_type  {dict(sorted(by_type.items()))}")
    print(f"  by_split {dict(by_split)}")
    print(f"  lang     {dict(lang)}")
    print(f"  license  {dict(lic)}")
    print(f"  sha256   {sha}")

    m_p = Path(manifest)
    m_p.parent.mkdir(parents=True, exist_ok=True)
    m = json.loads(m_p.read_text(encoding="utf-8")) if m_p.exists() else {}
    m[name] = {
        "file": out_p.as_posix() if not out_p.is_absolute() else out_p.name,
        "n_records": len(records),
        "n_questions": n_q,
        "by_type": dict(sorted(by_type.items())),
        "by_split": dict(by_split),
        "sha256": sha,
        "source_repo": repo_url,
        "source_commit": commit,
        # a single string when uniform, else {license: n_records}
        "license": next(iter(lic)) if len(lic) == 1 else dict(lic),
        "lang": next(iter(lang)) if len(lang) == 1 else dict(lang),
    }
    with open(m_p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(dict(sorted(m.items())), f, ensure_ascii=False, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# image_* converters: HF streaming, image files, streaming JSONL writer, stats, manifest/image.json
# ---------------------------------------------------------------------------
import io
import math
import os

IMAGE_MANIFEST = Path(__file__).resolve().parent.parent / "manifest" / "image.json"
KEEP_FORMATS = {"JPEG": ".jpg", "MPO": ".jpg", "PNG": ".png", "WEBP": ".webp"}  # spec §4 MIME set


def image_args(extra=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="evals/data/image/<set>.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="records to write; 0 = all")
    ap.add_argument("--images-dir", default=None, help="default: <out dir>/../images/<set>")
    ap.add_argument("--manifest", default=str(IMAGE_MANIFEST), help="'' to skip the manifest update")
    if extra:
        extra(ap)
    return ap.parse_args()


def hf_stream(hf_id: str, config=None, split="train", revision=None, data_files=None, image_cols=()):
    """Streaming IterableDataset with image columns left as raw {"bytes","path"} (no decode, no full download)."""
    from datasets import Image, load_dataset
    ds = load_dataset(hf_id, name=config, split=split, streaming=True, revision=revision, data_files=data_files)
    for c in image_cols:
        ds = ds.cast_column(c, Image(decode=False))
    return ds


def blank(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and v.strip() in ("", "nan", "None", "-"))


def stable_bit(key: str) -> int:
    """Deterministic coin flip for order balancing (same key -> same side on every run/machine)."""
    return hashlib.sha1(key.encode("utf-8")).digest()[0] & 1


class ImageSetWriter:
    """Streams records to <out>.tmp, stores images, counts; finish() validates-as-it-goes, renames, prints, updates manifest."""

    def __init__(self, name: str, args: argparse.Namespace):
        self.name, self.limit = name, args.limit
        self.out = Path(args.out)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.img_dir = Path(args.images_dir) if args.images_dir else self.out.parent.parent / "images" / name
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = args.manifest
        self.tmp = self.out.with_suffix(self.out.suffix + ".tmp")
        self.f = open(self.tmp, "w", encoding="utf-8", newline="\n")
        self.n = 0
        self.ids = set()
        self.by_type = collections.Counter()
        self.skips = collections.Counter()
        self.images_written = 0
        self.image_refs = 0
        self.converted = collections.Counter()
        self._saved = {}

    @property
    def full(self) -> bool:
        return self.limit > 0 and self.n >= self.limit

    def next_id(self, split: str) -> str:
        return f"{self.name}/{split}/{self.n:06d}"

    def skip(self, reason: str) -> None:
        self.skips[reason] += 1

    def image(self, raw, key: str) -> dict:
        """raw: {"bytes","path"} from a non-decoded Image column (or PIL image). key: file stem, reused if already saved."""
        self.image_refs += 1
        if key in self._saved:
            return {"image": {"path": self._saved[key]}}
        from PIL import Image as PILImage
        if isinstance(raw, dict):
            data = raw.get("bytes")
            if data is None and raw.get("path"):
                data = Path(raw["path"]).read_bytes()
            if data is None:
                raise ValueError("image has neither bytes nor path")
            im = PILImage.open(io.BytesIO(data))
        else:
            im, data = raw, None
        fmt = im.format
        if data is not None and fmt in KEEP_FORMATS:
            ext = KEEP_FORMATS[fmt]
        else:  # unsupported MIME (GIF/BMP/TIFF/...) -> PNG
            self.converted[str(fmt)] += 1
            if im.mode not in ("1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16"):
                im = im.convert("RGBA" if "A" in im.mode else "RGB")
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            data, ext = buf.getvalue(), ".png"
        dest = self.img_dir / f"{key}{ext}"
        if not dest.exists() or dest.stat().st_size != len(data):
            dest.write_bytes(data)
        self.images_written += 1
        rel = Path(os.path.relpath(dest, self.out.parent)).as_posix()
        self._saved[key] = rel
        return {"image": {"path": rel}}

    def add(self, rec: dict) -> None:
        errs = validate_record(rec)
        if errs:
            raise ValueError(f"{rec.get('id')}: {errs}")
        if rec["id"] in self.ids:
            raise ValueError(f"duplicate id {rec['id']}")
        self.ids.add(rec["id"])
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.n += 1
        self.by_type.update(q["type"] for q in rec["questions"].values())

    def finish(self, entry: dict) -> None:
        """entry: hf_id, hf_config, hf_split, hf_revision, n_records_full, license, lang, notes."""
        self.f.close()
        os.replace(self.tmp, self.out)
        sha = hashlib.sha256(self.out.read_bytes()).hexdigest()
        print(f"{self.name}: {self.n} records -> {self.out}")
        print(f"  sha256   {sha}")
        print(f"  by_type  {dict(sorted(self.by_type.items()))}")
        print(f"  images   {self.images_written} files ({self.image_refs} refs) in {self.img_dir}"
              + (f"; converted to PNG: {dict(self.converted)}" if self.converted else ""))
        print(f"  skipped  {dict(self.skips.most_common()) or 0}")
        if not self.manifest:
            return
        m_p = Path(self.manifest)
        m_p.parent.mkdir(parents=True, exist_ok=True)
        m = json.loads(m_p.read_text(encoding="utf-8")) if m_p.exists() else {}
        try:
            file = self.out.resolve().relative_to(m_p.resolve().parents[2]).as_posix()
        except ValueError:
            file = self.out.as_posix()
        m[self.name] = {
            "file": file,
            "hf_id": entry["hf_id"],
            "hf_config": entry.get("hf_config"),
            "hf_split": entry["hf_split"],
            "hf_revision": entry.get("hf_revision"),
            "n_records_local": self.n,
            "n_records_full": entry.get("n_records_full"),
            "n_questions": sum(self.by_type.values()),
            "sha256": sha,
            "by_type": dict(sorted(self.by_type.items())),
            "license": entry["license"],
            "lang": entry["lang"],
            "notes": entry.get("notes", ""),
        }
        with open(m_p, "w", encoding="utf-8", newline="\n") as f:
            json.dump(dict(sorted(m.items())), f, ensure_ascii=False, indent=2)
            f.write("\n")
