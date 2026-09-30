"""MME-RealWorld, English part (high-res real-world MC, 5 options "(A) …") -> choice.
Read from the author's parquet re-upload yifanzhang114/MME-RealWorld-Lmms-eval (English only; the CN part is the
separate …-CN-Lmms-eval repo). The original yifanzhang114/MME-RealWorld ships images only as multi-GB tar.gz parts,
which cannot be sampled without downloading them whole.

python -m data.sources.image_mme_realworld --out data/raw/image/mme_realworld.jsonl --limit 30
"""
import base64
import re

from data.sources._common import RawWriter, image_from_bytes, one_hot, record, save_image, source_args
from data.sources._parquet import bytes_id, iter_row_groups, list_files

NAME = "mme_realworld"
HF_ID, REVISION, SPLIT = "yifanzhang114/MME-RealWorld-Lmms-eval", "f157390129e8d2b60757eb68d340aade9446b52b", "train"
OPTION = re.compile(r"^\(([A-Z])\)\s*(.*)$", re.S)


def main():
    args = source_args(NAME)
    w = RawWriter(NAME, args.out)
    done = lambda: args.limit and w.n >= args.limit  # noqa: E731
    for rows, where in iter_row_groups(list_files(HF_ID, REVISION, "data/*.parquet"), args.seed):
        for row in rows:
            if done():
                break
            ms = [OPTION.match(o.strip()) for o in row["multi-choice options"] or []]
            if not ms or not all(ms):
                w.skip("options_unparsed")
                continue
            opts = {m.group(1): m.group(2).strip() for m in ms}
            if list(opts) != [chr(65 + i) for i in range(len(opts))] or len(opts) < 2:
                w.skip("labels_not_consecutive")
                continue
            ans = (row["answer"] or "").strip()
            if ans not in opts:
                w.skip("answer_not_in_options")
                continue
            raw = base64.b64decode(row["bytes"])
            image_id = bytes_id(raw)
            path = save_image(image_from_bytes(raw), NAME, image_id)
            w.write(record(NAME, w.n, {"image": {"path": path}},
                           {"q1": {"type": "choice", "instructions": row["question"].strip(), "criteria": opts}},
                           {"q1": one_hot(opts, ans)}, lang="en", grid_id=f"{NAME}/{image_id}", source_split=SPLIT,
                           meta={"shard": where, "orig_index": row["index"], "category": row["category"],
                                 "l2_category": row["l2-category"], "image_sha1": image_id}))
        if done():
            break
    w.close()


if __name__ == "__main__":
    main()
