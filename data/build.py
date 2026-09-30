"""CLI: data/raw -> data/build/<version>/ + data/manifest/<version>.json. See data/README.md."""
import argparse
import collections
import json
import os
import random
import sys
from pathlib import Path

from data.augment import DEFAULT_CONFIG, Augmenter
from data.coco_licenses import KEEP as COCO_KEEP, lookup as coco_license
from data.dedup import HELDOUT_IMAGES, HELDOUT_RECORDS, Dedup, _strings, image_objects
from data.grid import build_grids
from data.manifest import build_manifest
from data.split import SPLITS, ensure_coverage
from evals.schema import LICENSES, iter_records, validate_record

DATA_DIR = Path(__file__).resolve().parent
BUCKETS = ("text", "image")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def load_tiers(path: Path) -> dict | None:
    if not path.exists():
        return None
    return {k: v.get("tier", "unknown") for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def normalize_tier(tier: str) -> str:
    if tier == "copyleft":
        return "non-commercial"
    return tier if tier in LICENSES else "unknown"


def rebase_images(rec: dict, src_dir: Path, dst_dir: Path) -> None:
    """Rewrite relative image paths from the raw JSONL directory to the build directory."""
    for obj in image_objects(rec["state"], []):
        p = obj.get("path")
        if p and not Path(p).is_absolute():
            obj["path"] = os.path.relpath((src_dir / p).resolve(), dst_dir.resolve()).replace("\\", "/")


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _summary(state, limit: int = 300) -> str:
    s = json.dumps(state, ensure_ascii=False)
    return s if len(s) <= limit else s[:limit] + f"… ({len(s)} chars)"


def write_preview(out: Path, records: list[dict]) -> None:
    write_jsonl(out / "preview.jsonl", records)
    lines = [f"# preview ({len(records)} records)\n"]
    for r in records:
        lines.append(f"## {r['id']}\n")
        lines.append(f"- source: `{r['source']}` lang: `{r['lang']}` aug: `{', '.join(r['meta']['aug']) or '-'}`")
        lines.append(f"- derived: `{', '.join(r['meta'].get('derived', [])) or '-'}`")
        lines.append(f"- state: `{_summary(r['state'])}`\n")
        for name, q in r["questions"].items():
            lines.append(f"**{name}** ({q['type']}) {q.get('instructions', '')}")
            if q["type"] == "choice":
                for k, v in q["criteria"].items():
                    lines.append(f"- `{k}` {v} → **{r['targets'][name][k]:.3f}**")
            elif q["type"] == "score":
                for i, v in enumerate(q["criteria"]):
                    lines.append(f"- `{i}` {v} → **{r['targets'][name][str(i)]:.3f}**")
            else:
                lines.append(f"- P(true) → **{r['targets'][name]:.3f}**")
            lines.append("")
    (out / "preview.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="build a training data version from data/raw")
    ap.add_argument("--version", required=True)
    ap.add_argument("--allow", default="commercial-ok", help="comma-separated license tiers to keep")
    ap.add_argument("--preview", type=int, default=0, help="render N augmented train records")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--raw-dir", default=str(DATA_DIR / "raw"))
    ap.add_argument("--out-dir", default=str(DATA_DIR / "build"))
    ap.add_argument("--manifest-dir", default=str(DATA_DIR / "manifest"))
    ap.add_argument("--licenses", default=str(DATA_DIR / "licenses.json"))
    ap.add_argument("--sources-dir", default=str(DATA_DIR / "sources"))
    ap.add_argument("--heldout-records", default=",".join(str(p) for p in HELDOUT_RECORDS),
                    help="comma-separated held-out JSONL directories")
    ap.add_argument("--heldout-images", default=str(HELDOUT_IMAGES))
    ap.add_argument("--coco-ids", default=None, help="extra COCO val image ids (one per line or a JSON list)")
    ap.add_argument("--coco-licenses", default=None,
                    help="COCO image id/file name -> license id JSON (data/coco_licenses.py); drops records whose "
                         "meta.coco_id is not under CC BY / BY-SA / BY-ND / no-restriction / US-gov")
    ap.add_argument("--exclude-sources", default="", help="comma-separated source names to skip entirely")
    ap.add_argument("--coco-id-required", default="",
                    help="comma-separated COCO-based sources whose records must carry meta.coco_id when "
                         "--coco-licenses is on; records without it are dropped (Cauldron copies lost the file names)")
    args = ap.parse_args(argv)

    allow = {normalize_tier(t.strip()) for t in args.allow.split(",") if t.strip()}
    exclude = {s.strip() for s in args.exclude_sources.split(",") if s.strip()}
    coco_required = {s.strip() for s in args.coco_id_required.split(",") if s.strip()}
    raw_dir, out = Path(args.raw_dir), Path(args.out_dir) / args.version
    out.mkdir(parents=True, exist_ok=True)
    tiers = load_tiers(Path(args.licenses))
    if tiers is None:
        log(f"{args.licenses} not found: keeping the license written by each converter")
    coco_map = json.loads(Path(args.coco_licenses).read_text(encoding="utf-8")) if args.coco_licenses else None

    log("loading held-out sets for dedup …")
    dedup = Dedup([p for p in args.heldout_records.split(",") if p], args.heldout_images, args.coco_ids)
    log(f"  {len(dedup.hashes)} held-out images, {len(dedup.minhashes)} held-out texts, {len(dedup.coco)} COCO ids")

    sources: dict[str, dict] = {}
    kept: list[dict] = []
    for bucket in BUCKETS:
        for f in sorted((raw_dir / bucket).glob("*.jsonl")):
            n = 0
            for rec in iter_records(f):
                src = rec["source"]
                s = sources.setdefault(src, {
                    "bucket": bucket, "raw_file": str(f),
                    "converter_file": str(Path(args.sources_dir) / f"{bucket}_{src}.py"),
                    "license": None, "allowed": False, "raw_records": 0, "dedup": collections.Counter(),
                    "license_dropped": 0, "coco_license_dropped": 0, "excluded": src in exclude})
                s["raw_records"] += 1
                n += 1
                if src in exclude:
                    continue
                if tiers is not None:
                    rec["license"] = normalize_tier(tiers.get(src, "unknown"))
                s["license"] = rec["license"]
                if rec["license"] not in allow:
                    s["license_dropped"] += 1
                    continue
                s["allowed"] = True
                cid = rec.get("meta", {}).get("coco_id")
                if coco_map is not None and cid is not None and coco_license(coco_map, cid) not in COCO_KEEP:
                    s["coco_license_dropped"] += 1
                    continue
                if coco_map is not None and cid is None and src in coco_required:
                    s["coco_license_dropped"] += 1
                    continue
                hit = dedup.check(rec, f.parent)
                if hit:
                    s["dedup"][hit] += 1
                    continue
                rebase_images(rec, f.parent, out)
                kept.append(rec)
            log(f"{f.name}: {n} raw records")
    dropped = collections.Counter()
    for s in sources.values():
        dropped.update(s["dedup"])
    log(f"license filter dropped {sum(s['license_dropped'] for s in sources.values())}, "
        f"COCO image license dropped {sum(s['coco_license_dropped'] for s in sources.values())}, "
        f"dedup dropped {dict(dropped)}, kept {len(kept)}")

    rng = random.Random(args.seed)
    grids = build_grids(kept, rng)
    ensure_coverage(grids)
    for r in grids:
        errs = validate_record(r)
        if errs:
            raise ValueError(f"{r['id']}: {errs}")
    by_split = {sp: [r for r in grids if r["split"] == sp] for sp in SPLITS}
    for sp, recs in by_split.items():
        write_jsonl(out / f"{sp}.jsonl", recs)
        log(f"{sp}: {len(recs)} records -> {out / (sp + '.jsonl')}")
    (out / "augment_config.json").write_text(json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = build_manifest(args.version, sources, grids, DEFAULT_CONFIG,
                              {"allow": sorted(allow), "seed": args.seed,
                               "coco_licenses": args.coco_licenses})
    mpath = Path(args.manifest_dir) / f"{args.version}.json"
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"manifest -> {mpath}")

    if args.preview:
        train = by_split["train"]
        text_pool = [" ".join(_strings(r["state"], [])) for r in train if not image_objects(r["state"], [])]
        image_pool = sorted({o["path"] for r in train for o in image_objects(r["state"], []) if "path" in o})
        aug = Augmenter(DEFAULT_CONFIG, random.Random(args.seed), text_pool, image_pool, out / "aug_images")
        picked = random.Random(args.seed).sample(train, min(args.preview, len(train)))
        rendered = [aug.apply(r) for r in picked]
        for r in rendered:
            errs = validate_record(r)
            if errs:
                raise ValueError(f"augmented {r['id']}: {errs}")
        write_preview(out, rendered)
        log(f"preview: {len(rendered)} records -> {out / 'preview.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
