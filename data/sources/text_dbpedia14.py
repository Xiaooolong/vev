"""dbpedia14 (fancyzhx/dbpedia_14, train) -> 14-category choice, as kev/data.py _dbpedia.

python -m data.sources.text_dbpedia14 --out data/raw/text/dbpedia14.jsonl --limit 30000
"""
from data.sources._common import one_hot
from data.sources._text import run, words


def convert(row, rng, names):
    keys = [x.lower().replace(" ", "_") for x in names]
    q = {"type": "choice", "instructions": "Which category does the subject of this encyclopedia text belong to?",
         "criteria": {k: None for k in keys}}
    return words(row["content"], 200), {"category": q}, {"category": one_hot(keys, keys[row["label"]])}


if __name__ == "__main__":
    run("dbpedia14", "fancyzhx/dbpedia_14", convert, lang="en", label_col="label")
