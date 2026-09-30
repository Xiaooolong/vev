"""JevBench v1.2 public items (easy 48 + standard 72 + hard 111) -> records.

python -m evals.datasets.text_jevbench --out evals/data/text/jevbench.jsonl
"""
from evals.datasets._common import ensure_repo, parse_args, read_jsonl, target_for, write_and_report

NAME = SOURCE = "jevbench"
REPO = "https://github.com/fstandhartinger/jevbench"
COMMIT = "f79a1cab94ab9a5879383b7ef9ee1805b9dc2d84"
# tier <- public file; sha256 from datasets/manifest.json (LF bytes)
FILES = [
    ("easy", "easy.jsonl", "231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b"),
    ("standard", "original.jsonl", "5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180"),
    ("hard", "hard.jsonl", "89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb"),
]
LICENSE = {"MIT": "commercial-ok"}


def answer_of(row: dict):
    """JevBench `expected`: label for choice, "yes"/"no" for noul, int level for score."""
    t = row["question"]["type"]
    if t == "noul":
        return {"yes": True, "no": False}[row["expected"]]
    return row["expected"]


def convert(row: dict, tier: str, idx: int) -> dict:
    q = row["question"]
    prov = row["provenance"]
    meta = {
        "tier": tier,
        "family": row["family"],
        "orig_id": row["id"],
        "grid_id": f"{SOURCE}/{row['id']}",
        "orig_answer_type": {"choice": "label", "noul": "yes_no", "score": "level_index"}[q["type"]],
    }
    if row.get("group"):
        meta["group"] = row["group"]  # standard tier: paraphrase pair (different text, same answer)
    for k in ("author_model", "surface_answer", "why_hard", "rationale"):
        if prov.get(k):
            meta[k] = prov[k]
    return {
        "id": f"{SOURCE}/public/{idx:06d}",
        "source": SOURCE,
        "split": "public",
        "license": LICENSE.get(prov.get("license"), "unknown"),
        "lang": "en",
        "state": row["state"],
        "questions": {"q1": q},
        "targets": {"q1": target_for(q, answer_of(row))},
        "meta": meta,
    }


def main() -> None:
    args = parse_args("jevbench")
    root = ensure_repo(REPO, COMMIT, args.src) / "datasets" / "public"
    records = []
    for tier, fname, sha in FILES:
        for row in read_jsonl(root / fname, sha):
            assert row["split"] == "public"
            records.append(convert(row, tier, len(records)))
    write_and_report(NAME, records, args.out, REPO, COMMIT, args.manifest)


if __name__ == "__main__":
    main()
