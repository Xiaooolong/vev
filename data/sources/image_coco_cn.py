"""COCO-CN (AIMClab-RUC/COCO-CN): Chinese human-written tags -> noul "图里是否有{tag}？", lang=zh.

Per image: up to 2 positives (its own tags) and the same number of negatives (frequent tags of the whole set that
are neither among its tags nor substrings of its Chinese captions, which cuts false negatives from incomplete
tagging), shuffled. Images are COCO train2014 (meta.coco_id / coco_split); COCO val2014 images are skipped here since
dedup drops them anyway. Images come from --coco-dir (local train2014/) when given, else one GET per image from
images.cocodataset.org.

Source: coco-cn-version1805v1.1.tar.gz (15 MB, fetched once and cached); the tag file is the member whose name
contains "tag", lines "<COCO file stem><sep><tags>"; captions are read from members containing "caption" (when
present) only for the negative filter.

python -m data.sources.image_coco_cn --out data/raw/image/coco_cn.jsonl --limit 30
"""
import collections
import random
import re
import tarfile
from pathlib import Path

from data.sources._common import RawWriter, image_from_bytes, record, save_image, source_args
from data.sources._ranged import fetch, fetch_bytes, hf_url

NAME, REPO, REV = "coco_cn", "AIMClab-RUC/COCO-CN", "d40a7aac6fdfa88f5e30c0cd89516feaa8014739"
TAR = "coco-cn-version1805v1.1.tar.gz"
COCO_URL = "http://images.cocodataset.org/{split}/{stem}.jpg"
STEM = re.compile(r"(COCO_(train|val)(2014)_0*(\d+))")
SEP = re.compile(r"[\s,;，；、]+")
N_POS, FREQ_POOL = 2, 300
INSTR = "图里是否有{tag}？"


def extra(ap):
    ap.add_argument("--coco-dir", default=None, help="directory holding COCO_train2014_*.jpg")


def read_tar():
    tags, caps = {}, collections.defaultdict(str)
    with tarfile.open(fetch(hf_url(REPO, TAR, REV))) as t:
        for m in t.getmembers():
            if not m.isfile():
                continue
            low = m.name.lower()
            if "tag" not in low and "caption" not in low:
                continue
            for line in t.extractfile(m).read().decode("utf-8", "replace").splitlines():
                mm = STEM.search(line)
                if not mm:
                    continue
                rest = line[mm.end():].lstrip("#0123456789").strip()
                if "tag" in low:
                    got = [x for x in SEP.split(rest) if x]
                    if got:
                        tags.setdefault(mm.group(1), [])
                        tags[mm.group(1)] += [x for x in got if x not in tags[mm.group(1)]]
                else:
                    caps[mm.group(1)] += rest
    return tags, caps


def main():
    args = source_args(NAME, extra)
    w = RawWriter(NAME, args.out)
    rng = random.Random(args.seed)
    tags, caps = read_tar()
    if not tags:
        raise SystemExit("coco_cn: no tag file found in the archive")
    freq = [t for t, _ in collections.Counter(t for ts in tags.values() for t in ts).most_common(FREQ_POOL)]
    stems = sorted(tags)
    rng.shuffle(stems)
    for stem in stems:
        if args.limit and sum(w.by_type.values()) >= args.limit:
            break
        m = STEM.match(stem)
        split = f"{m.group(2)}{m.group(3)}"
        if m.group(2) == "val":
            w.skip("coco_val_image")
            continue
        own = tags[stem]
        neg_pool = [t for t in freq if t not in own and t not in caps.get(stem, "")]
        k = min(N_POS, len(own), len(neg_pool))
        if k == 0:
            w.skip("no_tags")
            continue
        items = [(t, 1.0) for t in rng.sample(own, k)] + [(t, 0.0) for t in rng.sample(neg_pool, k)]
        rng.shuffle(items)
        local = Path(args.coco_dir) / f"{stem}.jpg" if args.coco_dir else None
        raw = local.read_bytes() if local and local.exists() else fetch_bytes(COCO_URL.format(split=split, stem=stem))
        path = save_image(image_from_bytes(raw), NAME, stem)
        qs = {f"q{i + 1}": {"type": "noul", "instructions": INSTR.format(tag=t)} for i, (t, _) in enumerate(items)}
        ts = {f"q{i + 1}": y for i, (_, y) in enumerate(items)}
        w.write(record(NAME, w.n, {"image": {"path": path}}, qs, ts, lang="zh", grid_id=f"{NAME}/{stem}",
                       source_split="coco-cn", meta={"coco_id": int(m.group(4)), "coco_split": split,
                                                     "tags": own, "asked": [t for t, _ in items]}))
    w.close()


if __name__ == "__main__":
    main()
