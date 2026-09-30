"""COCO per-image license map for build.py --coco-licenses. Contract: data/README.md, section "License tiers".

Reads the images[] table (id, file_name, license) of COCO annotation files and writes
{"<image id>": license_id, "<file_name>": license_id}. Any annotation file with that table works;
captions_train20xx.json carries the same images[] and is far smaller than instances_train20xx.json.

    python -m data.coco_licenses --annotations instances_train2014.json instances_train2017.json \
        --out data/coco_licenses.json
"""
import argparse
import collections
import json
import sys
from pathlib import Path

KEEP = {4, 5, 6, 7, 8}  # CC BY, CC BY-SA, CC BY-ND, no known restrictions, US government work
DROP = {1, 2, 3}  # CC BY-NC-SA, CC BY-NC, CC BY-NC-ND


def license_map(annotation_files) -> dict[str, int]:
    out: dict[str, int] = {}
    for f in annotation_files:
        with open(f, encoding="utf-8") as fh:
            images = json.load(fh)["images"]
        for img in images:
            out[str(img["id"])] = int(img["license"])
            out[img["file_name"]] = int(img["license"])
    return out


def lookup(mapping: dict[str, int], coco_id) -> int | None:
    """License id for a meta.coco_id value: a numeric id (int or zero-padded string) or a file name."""
    s = str(coco_id).strip()
    if s.isdigit() and str(int(s)) in mapping:
        return mapping[str(int(s))]
    return mapping.get(s, mapping.get(Path(s).name))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="COCO image id / file name -> license id map")
    ap.add_argument("--annotations", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    mapping = license_map(args.annotations)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(mapping), encoding="utf-8")
    by_license = collections.Counter(v for k, v in mapping.items() if k.isdigit())
    print(f"{sum(by_license.values())} images -> {args.out}; by license id {dict(sorted(by_license.items()))}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
