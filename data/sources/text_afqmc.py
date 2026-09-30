"""afqmc (clue/clue config afqmc, train) -> "same meaning" noul in Chinese, shaped like kev's paws question.

python -m data.sources.text_afqmc --out data/raw/text/afqmc.jsonl --limit 30000
"""
from data.sources._text import run


def convert(row, rng, names):
    if row["label"] < 0:
        return "no gold label"
    q = {"type": "noul", "instructions": f"这句话和“{row['sentence2']}”意思相同吗？",
         "criteria": {"true": "意思相同，只是说法不同", "false": "意思不同，即使用词相近"}}
    return row["sentence1"], {"paraphrase": q}, {"paraphrase": 1.0 if names[row["label"]] == "1" else 0.0}


if __name__ == "__main__":
    run("afqmc", "clue/clue", convert, lang="zh", config="afqmc", id_field="idx", label_col="label")
