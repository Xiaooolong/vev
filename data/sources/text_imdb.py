"""imdb (stanfordnlp/imdb, train) -> "is this review positive" noul, as kev/data.py _imdb.

python -m data.sources.text_imdb --out data/raw/text/imdb.jsonl --limit 30000
"""
from data.sources._text import run, words


def convert(row, rng, names):
    q = {"type": "noul", "instructions": "Is this movie review positive?",
         "criteria": {"true": "The reviewer liked the film overall", "false": "The reviewer disliked the film overall"}}
    return words(row["text"].replace("<br />", " "), 220), {"positive": q}, {"positive": 1.0 if row["label"] == 1 else 0.0}


if __name__ == "__main__":
    run("imdb", "stanfordnlp/imdb", convert, lang="en")
