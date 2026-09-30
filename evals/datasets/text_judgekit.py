"""judgekit zh mini-benchmark (130 hand-labelled items, 4 tasks) -> records.

python -m evals.datasets.text_judgekit --out evals/data/text/judgekit.jsonl

Mirrors benchmarks/run_bench.py: the label set is each file's labels in order of first
appearance, the question is a choice with instructions "中文分类基准测评。", and the
state is {"text": ...}. The bench defines no per-label descriptions (its typesafe
provider falls back to the label string itself), so criteria descriptions are null.
"""
from evals.datasets._common import ensure_repo, parse_args, read_jsonl, target_for, write_and_report

NAME = SOURCE = "judgekit"
REPO = "https://github.com/lexingtonhibiki/judgekit"
COMMIT = "da010de3a7ec3d672471c5738449f0f5ed46d419"
TASKS = ["intent_zh", "sentiment_zh", "spam_zh", "urgency_zh"]
INSTRUCTIONS = "中文分类基准测评。"  # run_bench.py Task(instruction=...), sent as the question instructions


def main() -> None:
    args = parse_args("judgekit")
    root = ensure_repo(REPO, COMMIT, args.src) / "benchmarks" / "data"
    records = []
    for task in TASKS:
        rows = read_jsonl(root / f"{task}.jsonl")
        labels = list(dict.fromkeys(r["label"] for r in rows))
        q = {"type": "choice", "instructions": INSTRUCTIONS, "criteria": {lb: None for lb in labels}}
        for r in rows:
            records.append({
                "id": f"{SOURCE}/test/{len(records):06d}",
                "source": SOURCE,
                "split": "test",
                "license": "commercial-ok",  # repo MIT; samples are hand-written in-repo
                "lang": "zh",
                "state": {"text": r["text"]},
                "questions": {"d": q},
                "targets": {"d": target_for(q, r["label"])},
                "meta": {"task": task, "orig_id": r["id"], "grid_id": f"{SOURCE}/{task}/{r['id']}",
                         "orig_answer_type": "label"},
            })
    write_and_report(NAME, records, args.out, REPO, COMMIT, args.manifest)


if __name__ == "__main__":
    main()
