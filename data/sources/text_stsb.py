"""stsb (nyu-mll/glue config stsb, train) -> 6-level similarity score; the 0-5 gold mean is rounded half-up to a level.
Levels follow the STS annotation guidelines (Agirre et al. 2013).

python -m data.sources.text_stsb --out data/raw/text/stsb.jsonl --limit 30000
"""
import math

from data.sources._common import one_hot
from data.sources._text import run

LEVELS = ["The two sentences are completely dissimilar",
          "Not equivalent, but on the same topic",
          "Not equivalent, but they share some details",
          "Roughly equivalent, but some important information differs or is missing",
          "Mostly equivalent, but some unimportant details differ",
          "Completely equivalent: they mean the same thing"]


def convert(row, rng, names):
    if row["label"] < 0:
        return "no gold label"
    q = {"type": "score", "instructions": f'How close in meaning is this sentence to: "{row["sentence2"]}"', "criteria": list(LEVELS)}
    level = min(5, math.floor(row["label"] + 0.5))
    return row["sentence1"], {"similarity": q}, {"similarity": one_hot(range(6), level)}


if __name__ == "__main__":
    run("stsb", "nyu-mll/glue", convert, lang="en", config="stsb", id_field="idx")
