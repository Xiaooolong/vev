"""ag_news (fancyzhx/ag_news, train) -> 4-topic choice + two "is it about X" noul, as kev/data.py _agnews.

python -m data.sources.text_ag_news --out data/raw/text/ag_news.jsonl --limit 30000
"""
from data.sources._common import one_hot
from data.sources._text import run

AG = {"world": "World news: politics, international affairs, conflicts", "sports": "Sports: games, athletes, teams, results",
      "business": "Business: companies, markets, economy, finance", "scitech": "Science and technology: research, gadgets, software, space"}
KEYS = list(AG)  # upstream label order: World, Sports, Business, Sci/Tech


def convert(row, rng, names):
    y = KEYS[row["label"]]
    qs = {"topic": {"type": "choice", "instructions": "What is the topic of this article?", "criteria": dict(AG)}}
    ts = {"topic": one_hot(KEYS, y)}
    for k in rng.sample(KEYS, 2):
        qs[f"is_{k}"] = {"type": "noul", "instructions": f"Is this article about {AG[k].split(':')[0].lower()}?"}
        ts[f"is_{k}"] = 1.0 if k == y else 0.0
    return row["text"], qs, ts


if __name__ == "__main__":
    run("ag_news", "fancyzhx/ag_news", convert, lang="en")
