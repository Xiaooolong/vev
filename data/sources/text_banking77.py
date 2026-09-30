"""banking77 (legacy-datasets/banking77, train) -> 77-intent choice, as kev/data.py _banking.

python -m data.sources.text_banking77 --out data/raw/text/banking77.jsonl --limit 30000
"""
from data.sources._common import one_hot
from data.sources._text import run


def convert(row, rng, names):
    q = {"type": "choice", "instructions": "Which banking intent best describes this customer message?",
         "criteria": {k: None for k in names}}
    return row["text"], {"intent": q}, {"intent": one_hot(names, names[row["label"]])}


if __name__ == "__main__":
    run("banking77", "legacy-datasets/banking77", convert, lang="en", label_col="label")
