"""kev transfer-v4 (dev + locked test) -> records.

python -m evals.datasets.text_kev_transfer_v4 --out evals/data/text/kev_transfer_v4.jsonl
"""
from evals.datasets._common import ensure_repo, parse_args, read_jsonl, target_for, write_and_report

NAME = "kev_transfer_v4"
SOURCE = "kev-transfer-v4"
REPO = "https://github.com/jaredpalmer/kev"
COMMIT = "557598fced1dada75dfbf36ed144dce309ac6ceb"
# sha256 from evals/v4/transfer-v4/manifest.json (LF bytes)
SPLITS = {
    "dev": ("development.jsonl", "ff374c49c6c9f15f8a56fb274b4a4857d20497eb8dd1ac07ce01560e682a5f2e"),
    "test": ("test.jsonl", "c30a91274f9b483aac9e4f02ada5dea953b3f1a2829456e0bc07806c4e73b517"),
}
# per upstream HF dataset card; synthetic policy items (repo=None) are kev's own, Apache-2.0
LICENSE_BY_REPO = {
    "cais/mmlu": "commercial-ok",                     # MIT
    "allenai/sciq": "non-commercial",                 # CC BY-NC 3.0
    "dair-ai/emotion": "unknown",                     # card: other
    "cardiffnlp/tweet_eval": "unknown",               # card: unknown
    "nyu-mll/glue": "unknown",                        # card: other (QNLI)
    "google-research-datasets/paws": "unknown",       # card: other
    None: "commercial-ok",
}


def convert(row: dict, split: str, idx: int) -> dict:
    m = row["_meta"]
    questions, targets, src = {}, {}, {}
    for key, q in row["questions"].items():
        q = dict(q)
        label = q.pop("label")
        src[key] = q.pop("src")
        questions[key] = q
        targets[key] = target_for(q, label)
    meta = {
        "subset": m["source"],
        "src": src[key] if len(src) == 1 else src,
        "variant": m["variant"],
        "orig_id": m["id"],
        "grid_id": f"{SOURCE}/{m['group_id']}",
        "orig_answer_type": {"choice": "label", "noul": "bool", "score": "level_index"}[q["type"]],
    }
    for k in ("repo", "revision", "split", "row", "family", "pair_id", "sibling", "pair_kind", "none_key"):
        if m.get(k) is not None:
            meta[f"upstream_{k}" if k in ("split", "row") else k] = m[k]
    return {
        "id": f"{SOURCE}/{split}/{idx:06d}",
        "source": SOURCE,
        "split": split,
        "license": LICENSE_BY_REPO[m.get("repo")],
        "lang": "en",
        "state": row["state"],
        "questions": questions,
        "targets": targets,
        "meta": meta,
    }


def main() -> None:
    args = parse_args("kev")
    root = ensure_repo(REPO, COMMIT, args.src) / "evals" / "v4" / "transfer-v4"
    records = []
    for split, (fname, sha) in SPLITS.items():
        rows = read_jsonl(root / fname, sha)
        records += [convert(r, split, i) for i, r in enumerate(rows)]
    write_and_report(NAME, records, args.out, REPO, COMMIT, args.manifest)


if __name__ == "__main__":
    main()
