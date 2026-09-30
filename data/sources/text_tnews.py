"""tnews (clue/clue config tnews, train) -> 15-category news-headline choice in Chinese.

python -m data.sources.text_tnews --out data/raw/text/tnews.jsonl --limit 30000
"""
from data.sources._common import one_hot
from data.sources._text import run

# CLUE labels.json: code -> news_<x>; keys are the usual Chinese category names
CODES = {"100": "故事", "101": "文化", "102": "娱乐", "103": "体育", "104": "财经", "106": "房产", "107": "汽车", "108": "教育",
         "109": "科技", "110": "军事", "112": "旅游", "113": "国际", "114": "股票", "115": "农业", "116": "电竞"}


def convert(row, rng, names):
    if row["label"] < 0:
        return "no gold label"
    q = {"type": "choice", "instructions": "这条新闻标题属于哪个类别？", "criteria": {k: None for k in CODES.values()}}
    return row["sentence"], {"category": q}, {"category": one_hot(CODES.values(), CODES[names[row["label"]]])}


if __name__ == "__main__":
    run("tnews", "clue/clue", convert, lang="zh", config="tnews", id_field="idx", label_col="label")
