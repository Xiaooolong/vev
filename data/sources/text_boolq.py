"""boolq (google/boolq, train) -> yes/no noul over the passage, as kev/data.py _boolq.

python -m data.sources.text_boolq --out data/raw/text/boolq.jsonl --limit 30000
"""
from data.sources._text import run


def convert(row, rng, names):
    q = {"type": "noul", "instructions": row["question"].strip().rstrip("?") + "?"}
    return row["passage"], {"answer": q}, {"answer": 1.0 if row["answer"] else 0.0}


if __name__ == "__main__":
    run("boolq", "google/boolq", convert, lang="en")
