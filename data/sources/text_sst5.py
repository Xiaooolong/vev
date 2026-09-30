"""sst5 (SetFit/sst5, train) -> 5-level sentiment score, as kev/data.py _sst5.

python -m data.sources.text_sst5 --out data/raw/text/sst5.jsonl --limit 30000
"""
from data.sources._common import one_hot
from data.sources._text import run

SST5 = ["very negative", "negative", "neutral", "positive", "very positive"]


def convert(row, rng, names):
    q = {"type": "score", "instructions": "What is the sentiment of this review sentence?", "criteria": list(SST5)}
    return row["text"], {"sentiment": q}, {"sentiment": one_hot(range(5), row["label"])}


if __name__ == "__main__":
    run("sst5", "SetFit/sst5", convert, lang="en")
