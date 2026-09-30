"""nimble frozen held-out eval (324 contrastive records, 162 base/counterfactual pairs) -> records.

python -m evals.datasets.text_nimble --out evals/data/text/nimble.jsonl
"""
from evals.datasets._common import ensure_repo, parse_args, read_jsonl, target_for, write_and_report

NAME = SOURCE = "nimble"
REPO = "https://github.com/bespokelabsai/nimble"
COMMIT = "54c2e82e8f613a7ab3a4d4072c50dfb9376db4e3"
EVAL_SHA256 = "8e9e48b8de5206593912ae01ddc95bd77e40ad2ecf4c9292c1711290eca0d896"  # declared in the nimble repo


def convert(row: dict, idx: int) -> dict:
    inp = row["input"]
    questions = inp["questions"]
    answer = row["reference"]["target"]
    (key, q), = questions.items()
    return {
        "id": f"{SOURCE}/heldout/{idx:06d}",
        "source": SOURCE,
        "split": "heldout",
        "license": "unknown",  # repo has no LICENSE file and the data docs state none
        "lang": "en",
        "state": inp["state"],
        "questions": questions,
        "targets": {key: target_for(q, answer)},
        "meta": {
            "domain": row["domain"],
            "source_family": row["source_family"],
            "family": row["family"],
            "variant": row["variant"],
            "orig_id": row["id"],
            "orig_split": row["split"],
            "grid_id": f"{SOURCE}/{row['id']}",
            "orig_answer_type": {"choice": "label", "noul": "bool", "score": "level_index"}[q["type"]],
            "human_reviewed": row["reference"]["human_reviewed"],
        },
    }


def main() -> None:
    args = parse_args("nimble")
    path = ensure_repo(REPO, COMMIT, args.src) / "data" / "eval.jsonl"
    rows = read_jsonl(path, EVAL_SHA256)
    records = [convert(r, i) for i, r in enumerate(rows)]
    write_and_report(NAME, records, args.out, REPO, COMMIT, args.manifest)


if __name__ == "__main__":
    main()
