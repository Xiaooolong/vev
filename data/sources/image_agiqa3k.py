"""AGIQA-3K (AI-generated image quality): perceptual-quality MOS and prompt-alignment MOS (both 0-5) -> two
score questions (5 levels) on the same image. Soft target = the rater distribution N(mos, std) discretised
onto the five unit-wide bins.

python -m data.sources.image_agiqa3k --limit 30
"""
import math
import random

import pandas as pd

from data.sources._common import RawWriter, image_from_bytes, record, save_image
from data.sources._remote import image_source_args, hf_download, hf_open, retry

NAME = "agiqa3k"
# HF mirror of github.com/lcysyzxdxc/AGIQA-3k-Database (data.csv = the upstream MOS file, images/ = the 2,982 images)
HF_ID, REVISION, SPLIT = "strawhat/agiqa-3k", "bf9b7bb8d633f44832d6ba51ff496a25d8e8e2ae", "all"
EDGES = [1.0, 2.0, 3.0, 4.0]  # MOS scale 0-5 -> bins [0,1) [1,2) [2,3) [3,4) [4,5]
Q_QUALITY = {"type": "score", "instructions": "How good is the perceptual quality of this AI-generated image "
                                              "(sharpness, artifacts, naturalness, aesthetics), ignoring the prompt?",
             "criteria": ["1 - bad", "2 - poor", "3 - fair", "4 - good", "5 - excellent"]}
Q_ALIGN = {"type": "score", "instructions": "How well does this AI-generated image match its text prompt?",
           "criteria": ["1 - does not match the prompt at all", "2 - matches a small part of the prompt",
                        "3 - partly matches the prompt", "4 - mostly matches the prompt",
                        "5 - fully matches the prompt"]}


def binned_normal(mu: float, sd: float) -> dict:
    cdf = [0.0] + [0.5 * (1 + math.erf((e - mu) / (max(sd, 1e-3) * math.sqrt(2)))) for e in EDGES] + [1.0]
    p = [max(0.0, cdf[i + 1] - cdf[i]) for i in range(5)]
    s = sum(p)
    return {str(i): v / s for i, v in enumerate(p)}


def main():
    args = image_source_args(NAME)
    w = RawWriter(NAME, args.out)
    df = pd.read_csv(hf_download(HF_ID, "data.csv", REVISION))
    order = list(range(len(df)))
    if args.limit:
        random.Random(args.seed).shuffle(order)
    idx = n_q = 0
    for i in order:
        if args.limit and n_q >= args.limit:
            break
        r = df.iloc[i]
        name = r["name"]
        raw = retry(lambda: hf_open(HF_ID, f"images/{name}", REVISION).read(), name)
        stem = name.rsplit(".", 1)[0]
        path = save_image(image_from_bytes(raw), NAME, stem)
        meta = {"generator": name.split("_")[0], "mos_quality": float(r["mos_quality"]),
                "std_quality": float(r["std_quality"]), "mos_align": float(r["mos_align"]),
                "std_align": float(r["std_align"]), "orig_answer_type": "mos"}
        for k in ("adj1", "adj2", "style"):
            if isinstance(r[k], str):
                meta[k] = r[k]
        w.write(record(NAME, idx, {"prompt": r["prompt"], "image": {"path": path}},
                       {"q1": dict(Q_QUALITY), "q2": dict(Q_ALIGN)},
                       {"q1": binned_normal(r["mos_quality"], r["std_quality"]),
                        "q2": binned_normal(r["mos_align"], r["std_align"])},
                       lang="en", grid_id=f"{NAME}/{stem}", source_split=SPLIT, meta=meta))
        idx += 1
        n_q += 2
    w.close()


if __name__ == "__main__":
    main()
