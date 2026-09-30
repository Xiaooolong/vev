"""GUICourse / GUIAct (HF yiye2023/GUIAct): screenshot + task -> next-action-type choice.

Three subsets: smartphone (multi-step, one action per row), web-multi (multi-step), web-single (single-step; a row
may carry several actions, the first one is the answer). The action vocabulary is built per subset from the names
that occur in its data JSON (so the choice set matches the data), e.g. smartphone: tap / swipe / input / enter /
answer ...; "answer" is GUIAct's terminal action (answers the question / reports the task complete).
lang is detected per row (any CJK character in the task -> zh, else en).

Source: <subset>_<split>_data.json (1-54 MB, fetched whole) + <subset>_<split>_images.parquet (148 MB - 5.6 GB,
columns base64 / elements / __index_level_0__ = image_id). Every images parquet is a single row group, so there is
no partial read: the file is downloaded to the HF cache and scanned in record batches, emitting the rows whose
image_id is in the (seed-shuffled, limit-capped) selection. Cluster only: at 150 KB/s even the 148 MB test file
of the smallest subset is out of reach locally. --split test uses the test files (same format).

python -m data.sources.image_guicourse --out data/raw/image/guicourse.jsonl --limit 30
"""
import base64
import json
import random
import re

from data.sources._common import RawWriter, image_from_bytes, one_hot, record, save_image, source_args

NAME, REPO = "guicourse", "yiye2023/GUIAct"
REV = "9f3341d4670624aa6bf4a6eb71544d1b09961c0e"
SUBSETS = ["smartphone", "web-multi", "web-single"]
DESCR = {
    "tap": "tap a UI element", "click": "click a UI element", "swipe": "swipe the screen", "scroll": "scroll the page",
    "input": "type text into a field", "enter": "press enter", "answer": "stop and give the answer / report done",
    "hover": "hover over an element", "select": "select an option", "select_text": "select text",
    "copy": "copy the selection", "drag": "drag an element", "press": "press a key",
}
INSTR = "Given the task and the current screenshot, which kind of action should the agent take next?"
CJK = re.compile(r"[一-鿿]")


def extra(ap):
    ap.add_argument("--split", default="train", choices=["train", "test"])


def action_name(row):
    lab = row["actions_label"]
    if isinstance(lab, list):
        lab = lab[0] if lab else None
    name = (lab or {}).get("name")
    return name.lower() if isinstance(name, str) else None  # the data mixes 'Click' and 'click'


def main():
    args = source_args(NAME, extra)
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    w = RawWriter(NAME, args.out)
    rng = random.Random(args.seed)
    per_subset = -(-args.limit // len(SUBSETS)) if args.limit else 0
    for subset in SUBSETS:
        data = json.loads(open(hf_hub_download(REPO, f"{subset}_{args.split}_data.json", repo_type="dataset",
                                               revision=REV), encoding="utf-8").read())
        names = sorted({n for n in map(action_name, data) if n})
        criteria = {n: DESCR.get(n, n.replace("_", " ")) for n in names}
        rng.shuffle(data)
        want = {}
        for row in data:
            if per_subset and len(want) >= per_subset:
                break
            if action_name(row) and row["image_id"] not in want:
                want[row["image_id"]] = row
        pq_path = hf_hub_download(REPO, f"{subset}_{args.split}_images.parquet", repo_type="dataset", revision=REV)
        for batch in pq.ParquetFile(pq_path).iter_batches(columns=["__index_level_0__", "base64"], batch_size=256):
            for image_id, b64 in zip(*(c.to_pylist() for c in batch.columns)):
                row = want.pop(image_id, None)
                if row is None:
                    continue
                path = save_image(image_from_bytes(base64.b64decode(b64)), NAME, f"{subset}_{image_id}")
                state = {"task": row["question"], "image": {"path": path}}
                if row.get("actions_history"):
                    state["history"] = row["actions_history"]
                act = action_name(row)
                w.write(record(NAME, w.n, state,
                               {"q1": {"type": "choice", "instructions": INSTR, "criteria": criteria}},
                               {"q1": one_hot(criteria, act)}, lang="zh" if CJK.search(row["question"]) else "en",
                               grid_id=f"{NAME}/{subset}/{image_id}", source_split=f"{subset}_{args.split}",
                               meta={"uid": row["uid"], "subset": subset, "actions_label": row["actions_label"]}))
            if not want:
                break
        for _ in want:
            w.skip("image_not_in_parquet")
    w.close()


if __name__ == "__main__":
    main()
