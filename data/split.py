"""train / calibration / validation split by a stable hash of grid_id. See data/README.md."""
import collections
import hashlib

SPLITS = ("train", "calibration", "validation")
CALIBRATION = 0.03
VALIDATION = 0.03
MIN_SOURCE = 50


def _grid(record: dict) -> str:
    return (record.get("meta") or {}).get("grid_id") or record["id"]


def grid_hash(grid_id: str) -> float:
    return int(hashlib.sha256(grid_id.encode("utf-8")).hexdigest()[:16], 16) / 2 ** 64


def assign_split(record: dict) -> str:
    h = grid_hash(_grid(record))
    if h < CALIBRATION:
        return "calibration"
    if h < CALIBRATION + VALIDATION:
        return "validation"
    return "train"


def ensure_coverage(records: list[dict]) -> list[dict]:
    """Set record['split'] in place. Sources under MIN_SOURCE records go entirely to train; otherwise a split
    left empty by the hash takes the train grid(s) with the lowest hash, whole grids only."""
    by_source = collections.defaultdict(list)
    for r in records:
        by_source[r["source"]].append(r)
    for recs in by_source.values():
        if len(recs) < MIN_SOURCE:
            for r in recs:
                r["split"] = "train"
            continue
        grids = collections.defaultdict(list)
        for r in recs:
            grids[_grid(r)].append(r)
        split_of = {g: assign_split(rs[0]) for g, rs in grids.items()}
        for want in ("calibration", "validation"):
            if want in split_of.values():
                continue
            train = sorted((g for g, s in split_of.items() if s == "train"), key=grid_hash)
            if len(train) >= 2:
                split_of[train[0]] = want
        for g, rs in grids.items():
            for r in rs:
                r["split"] = split_of[g]
    return records
