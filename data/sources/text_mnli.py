"""mnli (nyu-mll/glue config mnli, train: the same pairs as nyu-mll/multi_nli at a quarter of the bytes) -> 3-way choice,
as kev/data.py _mnli.

python -m data.sources.text_mnli --out data/raw/text/mnli.jsonl --limit 30000
"""
from data.sources._common import one_hot
from data.sources._text import run

MNLI = {"entailment": "The hypothesis follows from the premise", "neutral": "The hypothesis may or may not be true given the premise",
        "contradiction": "The hypothesis contradicts the premise"}
KEYS = list(MNLI)  # upstream label order


def convert(row, rng, names):
    if row["label"] < 0:
        return "no gold label"
    q = {"type": "choice", "instructions": f'Hypothesis: "{row["hypothesis"]}" How does it relate to the premise?',
         "criteria": dict(MNLI)}
    return row["premise"], {"relation": q}, {"relation": one_hot(KEYS, KEYS[row["label"]])}


if __name__ == "__main__":
    run("mnli", "nyu-mll/glue", convert, lang="en", config="mnli", id_field="idx")
