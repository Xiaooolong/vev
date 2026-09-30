"""Decontamination against held-out eval sets: image dHash, text MinHash, COCO val ids. Contract: data/README.md."""
import base64
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HELDOUT_RECORDS = (ROOT / "evals/data/image", ROOT / "evals/data/text")
HELDOUT_IMAGES = ROOT / "evals/data/images"

HAMMING_MAX = 4
JACCARD_MIN = 0.8
NUM_PERM = 128
SHINGLE = 3
# records that carry images only go through the text pass when their text is this long; short templated
# questions ("Is there a dog in the image?") are shared across benchmarks and say nothing about the image
IMAGE_RECORD_MIN_TEXT = 200
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
# 64 bits split into 5 blocks: two hashes within Hamming 4 agree exactly on at least one block
BLOCKS = ((0, 13), (13, 26), (26, 39), (39, 52), (52, 64))

HELDOUT_COCO_RE = re.compile(r"COCO_val20(?:14|17)_0*(\d+)")
ANY_COCO_RE = re.compile(r"COCO_(?:train|val|test)20\d\d_0*(\d+)|(?:train|val)20(?:14|17)[/\\]0*(\d+)\.")


def _dhash(img) -> int:
    import imagehash

    return int(str(imagehash.dhash(img.convert("RGB"), hash_size=8)), 16)


def _open_image(obj: dict, base: Path | None):
    from PIL import Image

    if "url" in obj:
        url = obj["url"]
        if not url.startswith("data:"):
            return None
        return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    p = Path(obj["path"])
    p = p if p.is_absolute() or base is None else base / p
    return Image.open(p) if p.exists() else None


def image_objects(node, found: list) -> list:
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "image" and isinstance(v, dict) and ({"path", "url"} & set(v)):
                found.append(v)
            else:
                image_objects(v, found)
    elif isinstance(node, list):
        for v in node:
            image_objects(v, found)
    return found


def _strings(node, out: list) -> list:
    if isinstance(node, str):
        out.append(node)
    elif isinstance(node, dict):
        for k, v in node.items():
            if k != "image":
                _strings(v, out)
    elif isinstance(node, list):
        for v in node:
            _strings(v, out)
    return out


def record_text(rec: dict) -> str:
    parts = _strings(rec.get("state"), [])
    parts += [q.get("instructions", "") for q in rec.get("questions", {}).values() if isinstance(q, dict)]
    return re.sub(r"\s+", " ", " ".join(parts)).strip().lower()


def _minhash(text: str):
    from datasketch import MinHash

    m = MinHash(num_perm=NUM_PERM)
    for i in range(max(1, len(text) - SHINGLE + 1)):
        m.update(text[i:i + SHINGLE].encode("utf-8"))
    return m


def _coco_ids(rec: dict, pattern: re.Pattern) -> set[int]:
    ids = set()
    cid = (rec.get("meta") or {}).get("coco_id")
    if cid is not None and str(cid).strip().isdigit():
        ids.add(int(cid))
    blob = json.dumps({"state": rec.get("state"), "meta": rec.get("meta")}, ensure_ascii=False)
    for m in pattern.finditer(blob):
        ids.add(int(next(g for g in m.groups() if g)))
    return ids


class Dedup:
    def __init__(self, heldout_records_dirs=HELDOUT_RECORDS, heldout_images_dir=HELDOUT_IMAGES,
                 coco_ids_file: str | Path | None = None):
        from datasketch import MinHashLSH
        from PIL import Image

        self.hashes: list[int] = []
        self.index = [dict() for _ in BLOCKS]
        d = Path(heldout_images_dir) if heldout_images_dir else None
        if d and d.exists():
            for p in sorted(d.rglob("*")):
                if p.suffix.lower() in IMAGE_EXTS:
                    with Image.open(p) as img:
                        self._add_hash(_dhash(img))

        self.lsh = MinHashLSH(threshold=JACCARD_MIN, num_perm=NUM_PERM)
        self.minhashes: dict[str, object] = {}
        self.coco: set[int] = set()
        n = 0
        for rd in heldout_records_dirs or ():
            for f in sorted(Path(rd).glob("*.jsonl")):
                with open(f, encoding="utf-8") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        self.coco |= _coco_ids(rec, HELDOUT_COCO_RE)
                        text = record_text(rec)
                        if text:
                            key = f"h{n}"
                            n += 1
                            mh = _minhash(text)
                            self.lsh.insert(key, mh)
                            self.minhashes[key] = mh
        if coco_ids_file:
            raw = Path(coco_ids_file).read_text(encoding="utf-8")
            ids = json.loads(raw) if raw.lstrip().startswith("[") else raw.split()
            self.coco |= {int(x) for x in ids}
        self._cache: dict[str, bool] = {}

    def _add_hash(self, h: int) -> None:
        self.hashes.append(h)
        for idx, (a, b) in zip(self.index, BLOCKS):
            idx.setdefault((h >> a) & ((1 << (b - a)) - 1), []).append(h)

    def image_hit(self, h: int) -> bool:
        for idx, (a, b) in zip(self.index, BLOCKS):
            for c in idx.get((h >> a) & ((1 << (b - a)) - 1), ()):
                if bin(h ^ c).count("1") <= HAMMING_MAX:
                    return True
        return False

    def _image_pass(self, rec: dict, base: Path | None) -> bool:
        # any image of a multi-image record hitting drops the whole record
        for obj in image_objects(rec.get("state"), []):
            key = obj.get("path") and str((base / obj["path"]).resolve() if base else obj["path"])
            if key and key in self._cache:
                hit = self._cache[key]
            else:
                img = _open_image(obj, base)
                hit = img is not None and self.image_hit(_dhash(img))
                if key:
                    self._cache[key] = hit
            if hit:
                return True
        return False

    def _text_pass(self, rec: dict) -> bool:
        text = record_text(rec)
        if not text or (image_objects(rec.get("state"), []) and len(text) < IMAGE_RECORD_MIN_TEXT):
            return False
        mh = _minhash(text)
        return any(mh.jaccard(self.minhashes[k]) >= JACCARD_MIN for k in self.lsh.query(mh))

    def check(self, record: dict, base_dir: str | Path | None = None) -> str | None:
        """Name of the first pass that hits ('image' / 'text' / 'coco'), or None.
        base_dir: directory that relative image paths resolve against (the record's JSONL directory)."""
        base = Path(base_dir) if base_dir else None
        if self.hashes and self._image_pass(record, base):
            return "image"
        if self.minhashes and self._text_pass(record):
            return "text"
        if self.coco and _coco_ids(record, ANY_COCO_RE) & self.coco:
            return "coco"
        return None
