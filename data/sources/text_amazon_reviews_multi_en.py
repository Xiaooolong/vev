"""amazon_reviews_multi_en (SetFit/amazon_reviews_multi_en, Hub parquet branch, train; the original
amazon_reviews_multi is gone from the Hub) -> 5-star score, as kev/data.py _amazon.

python -m data.sources.text_amazon_reviews_multi_en --out data/raw/text/amazon_reviews_multi_en.jsonl --limit 30000
"""
from data.sources._common import one_hot
from data.sources._text import run, words

AMAZON = ["1 star: very negative", "2 stars: negative", "3 stars: mixed", "4 stars: positive", "5 stars: very positive"]


def convert(row, rng, names):
    q = {"type": "score", "instructions": "How many stars did this product reviewer give?", "criteria": list(AMAZON)}
    return words(row["text"], 220), {"stars": q}, {"stars": one_hot(range(5), row["label"])}


if __name__ == "__main__":
    run("amazon_reviews_multi_en", "SetFit/amazon_reviews_multi_en", convert, lang="en",
        revision="refs/convert/parquet", id_field="id")
