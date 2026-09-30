"""ocnli (clue/clue config ocnli, train) -> 3-way NLI choice in Chinese, same shape as text_mnli.

python -m data.sources.text_ocnli --out data/raw/text/ocnli.jsonl --limit 30000
"""
from data.sources._common import one_hot
from data.sources._text import run

ZH = {"entailment": ("蕴含", "假设可以从前提推出"), "neutral": ("中立", "根据前提，假设可能成立也可能不成立"),
      "contradiction": ("矛盾", "假设与前提相矛盾")}
CRITERIA = {k: d for k, d in (ZH[n] for n in ("entailment", "neutral", "contradiction"))}


def convert(row, rng, names):
    if row["label"] < 0:
        return "no gold label"
    q = {"type": "choice", "instructions": f"假设：“{row['sentence2']}”它与前提是什么关系？", "criteria": dict(CRITERIA)}
    return row["sentence1"], {"relation": q}, {"relation": one_hot(CRITERIA, ZH[names[row["label"]]][0])}


if __name__ == "__main__":
    run("ocnli", "clue/clue", convert, lang="zh", config="ocnli", id_field="idx", label_col="label")
