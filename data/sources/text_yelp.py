"""yelp (Yelp/yelp_review_full, train) -> 5-star score + "would recommend" noul, as kev/data.py _yelp.

python -m data.sources.text_yelp --out data/raw/text/yelp.jsonl --limit 30000
"""
from data.sources._common import one_hot
from data.sources._text import run, words

YELP = ["1 star: terrible experience", "2 stars: poor", "3 stars: average", "4 stars: good", "5 stars: excellent"]


def convert(row, rng, names):
    qs = {"rating": {"type": "score", "instructions": "How many stars did this reviewer give?", "criteria": list(YELP)},
          "recommend": {"type": "noul", "instructions": "Would this reviewer recommend the business?",
                        "criteria": {"true": "Clearly positive overall", "false": "Negative or mixed"}}}
    ts = {"rating": one_hot(range(5), row["label"]), "recommend": 1.0 if row["label"] >= 3 else 0.0}
    return words(row["text"], 220), qs, ts


if __name__ == "__main__":
    run("yelp", "Yelp/yelp_review_full", convert, lang="en")
