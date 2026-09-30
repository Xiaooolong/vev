"""Build manifest: data/manifest/<version>.json. Contract: data/README.md, section "Manifest"."""
import collections
import datetime
import hashlib
from pathlib import Path

from data.split import SPLITS

PASSES = ("image", "text", "coco")


def sha256_file(path: str | Path) -> str | None:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def build_manifest(version: str, sources: dict[str, dict], records: list[dict], augment_config: dict,
                   args: dict | None = None) -> dict:
    """sources: name -> {bucket, raw_file, converter_file, license, allowed, license_dropped, coco_license_dropped,
    raw_records, dedup: {pass: n}}.
    records: the final (gridded, split) records."""
    split_counts = collections.defaultdict(collections.Counter)
    by_bucket, by_type, by_lang = collections.Counter(), collections.Counter(), collections.Counter()
    for r in records:
        split_counts[r["source"]][r["split"]] += 1
        by_bucket[sources.get(r["source"], {}).get("bucket", "?")] += 1
        by_lang[r["lang"]] += 1
        for q in r["questions"].values():
            by_type[q["type"]] += 1

    out_sources = {}
    for name, s in sorted(sources.items()):
        out_sources[name] = {
            "bucket": s["bucket"],
            "raw_records": s["raw_records"],
            "raw_sha256": sha256_file(s["raw_file"]),
            "converter_sha256": sha256_file(s["converter_file"]),
            "license": s["license"],
            "allowed": s["allowed"],
            "license_dropped": s.get("license_dropped", 0),
            "coco_license_dropped": s.get("coco_license_dropped", 0),
            "dedup_dropped": {p: s["dedup"].get(p, 0) for p in PASSES},
            "splits": {sp: split_counts[name][sp] for sp in SPLITS},
        }
    return {
        "version": version,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "args": args or {},
        "sources": out_sources,
        "totals": {
            "records": len(records),
            "splits": {sp: sum(c[sp] for c in split_counts.values()) for sp in SPLITS},
            "by_bucket_records": dict(sorted(by_bucket.items())),
            "by_type_questions": dict(sorted(by_type.items())),
            "by_lang_records": dict(sorted(by_lang.items())),
        },
        "augment_config": augment_config,
    }
