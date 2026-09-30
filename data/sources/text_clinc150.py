"""clinc150 (clinc/clinc_oos config plus, train) -> 150-intent choice; the out-of-scope class is kept as one more label
("none of the above"), so the source carries its own abstain examples.

python -m data.sources.text_clinc150 --out data/raw/text/clinc150.jsonl --limit 30000
"""
from data.sources._common import one_hot
from data.sources._text import run

OOS_KEY = "out_of_scope"
OOS_DESC = "None of the above: the request matches none of the listed intents"


def convert(row, rng, names):
    keys = [OOS_KEY if n == "oos" else n for n in names]
    q = {"type": "choice", "instructions": "Which intent best describes this user request?",
         "criteria": {k: OOS_DESC if k == OOS_KEY else None for k in keys}}
    return row["text"], {"intent": q}, {"intent": one_hot(keys, keys[row["intent"]])}


if __name__ == "__main__":
    run("clinc150", "clinc/clinc_oos", convert, lang="en", config="plus", label_col="intent")
