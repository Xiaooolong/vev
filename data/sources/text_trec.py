"""trec (CogComp/trec, Hub parquet branch, train) -> 6-way coarse answer-type choice, as kev/data.py _trec.

python -m data.sources.text_trec --out data/raw/text/trec.jsonl --limit 30000
"""
from data.sources._common import one_hot
from data.sources._text import run

TREC = {"abbreviation": "Asks what an abbreviation stands for", "entity": "Asks about a thing, object, animal, product, or creative work",
        "description": "Asks for a definition, description, reason, or manner", "human": "Asks about a person, group, or organisation",
        "location": "Asks about a place", "number": "Asks for a number, date, count, or other numeric value"}
KEYS = list(TREC)  # upstream coarse_label order: ABBR, ENTY, DESC, HUM, LOC, NUM


def convert(row, rng, names):
    q = {"type": "choice", "instructions": "What kind of answer does this question ask for?", "criteria": dict(TREC)}
    return row["text"], {"answer_type": q}, {"answer_type": one_hot(KEYS, KEYS[row["coarse_label"]])}


if __name__ == "__main__":
    run("trec", "CogComp/trec", convert, lang="en", revision="refs/convert/parquet")
