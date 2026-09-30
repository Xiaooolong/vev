"""HPDv2 (human preference between two generated images for one prompt) -> choice a/b.

State {"prompt", "a": {"image"}, "b": {"image"}}; half of the pairs are swapped by a stable hash of the pair key.
train: every annotation row is one vote on a pair; repeated rows of the same unordered pair are pooled and the
target is the vote share. test (--split test): 9 images per prompt ranked by ~10 annotators; every image pair
becomes a record, target = share of annotators ranking it higher (rank 0 = best).

Images live in one gzip tar per split (train.tar.gz = 31.7 GB, members in shuffled order), which cannot be
range-read: the converter samples pairs from the annotation json first, then streams the tar once and keeps only
the needed members, stopping as soon as all are found. On the cluster that is one sequential pass over the tar.

python -m data.sources.image_hpdv2 --limit 30000            # cluster
"""
import itertools
import json
import random
import tarfile
from collections import defaultdict

from data.sources._common import RawWriter, image_from_bytes, one_hot, record, save_image, stable_bit
from data.sources._remote import hf_download, hf_open, image_source_args

NAME = "hpdv2"
HF_ID, REVISION = "ymhao/HPDv2", "bb0f2563e451e11c3f376b2ba93615db00e042e1"
INSTR = "Which of the two generated images do people prefer for this text-to-image prompt?"
CRIT = {"a": "the image under state.a", "b": "the image under state.b"}


def pairs_from(split: str, data: list) -> dict:
    """{(img_x, img_y): {"prompt", "votes": [n_x, n_y]}} with img_x < img_y."""
    pairs = defaultdict(lambda: {"prompt": None, "votes": [0, 0]})
    for row in data:
        paths = row["image_path"]
        if split == "train":
            (x, y), pref = paths, row["human_preference"]
            if sorted(pref) != [0, 1]:
                continue
            key, won = (x, y) if x < y else (y, x), paths[pref.index(1)]
            p = pairs[key]
            p["prompt"] = row["prompt"]
            p["votes"][key.index(won)] += 1
        else:
            for i, j in itertools.combinations(range(len(paths)), 2):
                key = tuple(sorted((paths[i], paths[j])))
                p = pairs[key]
                p["prompt"] = row["prompt"]
                for ann in row["raw_annotations"]:
                    r = ann["annotation"]
                    if r[i] != r[j]:
                        best = paths[i] if r[i] < r[j] else paths[j]
                        p["votes"][key.index(best)] += 1
    return {k: v for k, v in pairs.items() if sum(v["votes"])}


def main():
    args = image_source_args(NAME, lambda ap: ap.add_argument("--split", default="train", choices=["train", "test"]))
    w = RawWriter(NAME, args.out)
    data = json.loads(open(hf_download(HF_ID, f"{args.split}.json", REVISION), encoding="utf-8").read())
    pairs = pairs_from(args.split, data)
    keys = sorted(pairs)
    if args.limit:
        keys = random.Random(args.seed).sample(keys, min(args.limit, len(keys)))
    needed = {p for k in keys for p in k}
    saved = {}
    with hf_open(HF_ID, f"{args.split}.tar.gz", REVISION, block_size=8 << 20) as fh:
        with tarfile.open(fileobj=fh, mode="r|gz") as tar:
            for m in tar:
                base = m.name.rsplit("/", 1)[-1]
                if m.isfile() and base in needed and base not in saved:
                    img = image_from_bytes(tar.extractfile(m).read())
                    saved[base] = save_image(img, NAME, f"{args.split}_{base.rsplit('.', 1)[0]}")
                    if len(saved) == len(needed):
                        break
    for idx, key in enumerate(k for k in keys if k[0] in saved and k[1] in saved):
        x, y = key
        nx, ny = pairs[key]["votes"]
        swapped = bool(stable_bit(f"{NAME}/{args.split}/{x}+{y}"))
        if swapped:
            x, y, nx, ny = y, x, ny, nx
        share = nx / (nx + ny)
        target = one_hot(CRIT, "a") if share == 1 else one_hot(CRIT, "b") if share == 0 else {"a": share, "b": 1 - share}
        sx, sy = (s.rsplit(".", 1)[0] for s in (x, y))
        w.write(record(NAME, idx, {"prompt": pairs[key]["prompt"], "a": {"image": {"path": saved[x]}},
                                   "b": {"image": {"path": saved[y]}}},
                       {"q1": {"type": "choice", "instructions": INSTR, "criteria": dict(CRIT)}}, {"q1": target},
                       lang="en", grid_id=f"{NAME}/{args.split}/{sx}+{sy}", source_split=args.split,
                       meta={"votes": [nx, ny], "swapped": swapped, "orig_answer_type": "pairwise"}))
    missing = len(keys) - w.n
    if missing:
        w.skip("image_not_in_tar")
        w.skipped["image_not_in_tar"] = missing
    w.close()


if __name__ == "__main__":
    main()
