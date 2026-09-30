"""EvalMuse-40K (generated images, 3-annotator element-level 0/1 and 1-5 alignment scores, BSD-3) -> three sets on a
prompt-level held-out block:

  t2i_elem       per image, one noul per element the 3 annotators agreed on (3/3 present -> 1, 0/3 -> 0)
  t2i_elem_soft  the split-vote elements (1/3, 2/3) as soft targets; Brier only
  t2i_score      per image, 5-level score; target = histogram of the 3 votes

state = {"prompt", "image"}; meta.cluster = meta.prompt_id (the gate resamples prompts; --shuffle-group prompt_id swaps
images among generators of the same prompt). Held-out prompts are drawn after removing prompts that also occur in the
training text-to-image sources (--exclude-prompts, normalised strings); the chosen prompt_ids are written next to the
jsonl so a future training build can exclude them. Images are read straight out of the 6-part images.zip over HTTP
range requests (byte-split parts concatenated into one seekable file), only the needed members.

python -m evals.datasets.image_t2i_align --out-dir evals/data/image --prompts 120 --per-prompt 4 --limit 20"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import zipfile
from pathlib import Path

from evals.datasets._common import IMAGE_MANIFEST, ImageSetWriter

HF_ID = "DY-Evalab/EvalMuse"
REVISION = "main"
LEVELS = ["1 - the image does not match the prompt at all",
          "2 - the image matches only a small part of the prompt",
          "3 - the image matches about half of the prompt",
          "4 - the image matches most of the prompt with minor errors",
          "5 - the image matches the prompt completely"]
SCORE_INSTR = "How well does the generated image match its text prompt (state.prompt)?"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def elem_instr(element: str) -> str:
    name, _, cat = element.rpartition(" (")
    cat = cat.rstrip(")")
    what = f"'{name}' ({cat})" if name else f"'{element}'"
    return f"The image was generated from the prompt in state.prompt. Does the image show the element {what} as the prompt describes?"


class Concat(io.RawIOBase):
    """The byte-split zip parts as one seekable read-only file over HfFileSystem."""

    def __init__(self, fs, paths: list[str], sizes: list[int]):
        self.fhs = [fs.open(p, "rb", block_size=4 * 1024 * 1024) for p in paths]
        self.sizes = sizes
        self.starts = [sum(sizes[:i]) for i in range(len(sizes))]
        self.total = sum(sizes)
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.total + off}[whence]
        return self.pos

    def readinto(self, b):
        need, out = len(b), bytearray()
        while need > 0 and self.pos < self.total:
            i = max(j for j, s in enumerate(self.starts) if s <= self.pos)
            local = self.pos - self.starts[i]
            take = min(need, self.sizes[i] - local)
            fh = self.fhs[i]
            fh.seek(local)
            chunk = fh.read(take)
            if not chunk:
                break
            out += chunk
            self.pos += len(chunk)
            need -= len(chunk)
        b[: len(out)] = out
        return len(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="evals/data/image")
    ap.add_argument("--prompts", type=int, default=120)
    ap.add_argument("--per-prompt", type=int, default=4, help="images per prompt, distinct generators")
    ap.add_argument("--limit", type=int, default=0, help="images cap for smoke runs; 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-prompts", default=None, help="text file, one normalised prompt per line, from the training sources")
    ap.add_argument("--manifest", default=str(IMAGE_MANIFEST))
    a = ap.parse_args(argv)
    from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
    api = HfApi()
    rev = api.dataset_info(HF_ID, revision=REVISION).sha
    rows = json.load(open(hf_hub_download(HF_ID, "train_list.json", repo_type="dataset", revision=rev), encoding="utf-8"))
    excl = set()
    if a.exclude_prompts:
        excl = {l.strip() for l in Path(a.exclude_prompts).read_text(encoding="utf-8").splitlines() if l.strip()}
    by_prompt: dict[str, list] = {}
    for r in rows:
        if not r.get("element_score") or not r.get("total_score"):
            continue
        by_prompt.setdefault(r["prompt_id"], []).append(r)
    ids = sorted(p for p, rs in by_prompt.items() if norm(rs[0]["prompt"]) not in excl)
    n_excluded = len(by_prompt) - len(ids)
    rng = random.Random(a.seed)
    rng.shuffle(ids)
    chosen = ids[: a.prompts]
    picks = []
    for pid in chosen:
        rs = by_prompt[pid][:]
        rng.shuffle(rs)
        gens = set()
        for r in rs:
            g = r["img_path"].split("/")[0]
            if g in gens:
                continue
            gens.add(g)
            picks.append(r)
            if len(gens) >= a.per_prompt:
                break
    if a.limit:
        picks = picks[: a.limit]
    # zip over range requests
    tree = {f.path: getattr(f, "size", None) for f in api.list_repo_tree(HF_ID, repo_type="dataset", revision=rev)}
    parts = sorted(p for p in tree if p.startswith("images.zip.part-"))
    fs = HfFileSystem()
    zf = zipfile.ZipFile(io.BufferedReader(Concat(fs, [f"datasets/{HF_ID}@{rev}/{p}" for p in parts], [tree[p] for p in parts]),
                                           buffer_size=4 * 1024 * 1024))
    members = set(zf.namelist())
    ns = argparse.Namespace
    img_dir = f"{a.out_dir}/../images/t2i_align"
    writers = {n: ImageSetWriter(n, ns(out=f"{a.out_dir}/{n}.jsonl", limit=0, images_dir=img_dir, manifest=a.manifest))
               for n in ("t2i_elem", "t2i_elem_soft", "t2i_score")}
    n_img = 0
    for r in picks:
        member = f"dataset/images/{r['img_path']}"
        if member not in members:
            for w in writers.values():
                w.skip("image_missing")
            continue
        data = zf.read(member)
        key = r["img_path"].replace("/", "_").rsplit(".", 1)[0]
        pid, gen = r["prompt_id"], r["img_path"].split("/")[0]
        elems = json.loads(r["element_score"]) if isinstance(r["element_score"], str) else r["element_score"]
        hard, soft = {}, {}
        for el, votes in elems.items():
            if not isinstance(votes, list) or len(votes) != 3 or any(v is None for v in votes):
                continue
            s = sum(int(v) for v in votes)
            (hard if s in (0, 3) else soft)[el] = s / 3
        meta = {"grid_id": f"t2i_align/{key}", "cluster": pid, "prompt_id": pid, "generator": gen, "prompt_type": r.get("type"),
                "orig_answer_type": "votes3"}
        n_img += 1
        for name, els in (("t2i_elem", hard), ("t2i_elem_soft", soft)):
            if not els:
                continue
            w = writers[name]
            st = {"prompt": r["prompt"], "image": w.image({"bytes": data, "path": None}, key)["image"]}
            qs = {f"q{i + 1}": {"type": "noul", "instructions": elem_instr(el)} for i, el in enumerate(els)}
            w.add({"id": f"{name}/heldout/{key}", "source": name, "split": "heldout", "license": "commercial-ok", "lang": "en",
                   "state": st, "questions": qs, "targets": {f"q{i + 1}": els[el] for i, el in enumerate(els)},
                   "meta": {**meta, "elements": list(els)}})
        votes = [int(v) for v in r["total_score"] if v is not None]
        if votes:
            w = writers["t2i_score"]
            st = {"prompt": r["prompt"], "image": w.image({"bytes": data, "path": None}, key)["image"]}
            hist = {str(i): 0.0 for i in range(5)}
            for v in votes:
                hist[str(min(max(v, 1), 5) - 1)] += 1 / len(votes)
            w.add({"id": f"t2i_score/heldout/{key}", "source": "t2i_score", "split": "heldout", "license": "commercial-ok", "lang": "en",
                   "state": st, "questions": {"q1": {"type": "score", "instructions": SCORE_INSTR, "criteria": list(LEVELS)}},
                   "targets": {"q1": hist}, "meta": {**meta, "votes": votes}})
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{a.out_dir}/t2i_align_heldout_prompts.json").write_text(
        json.dumps({"hf_revision": rev, "seed": a.seed, "prompt_ids": chosen, "n_prompts_excluded_as_in_training": n_excluded},
                   indent=1), encoding="utf-8")
    notes = (f"Held-out block of {len(chosen)} prompt_ids (seed {a.seed}, {n_excluded} prompts removed because their text occurs in "
             f"the training t2i sources), up to {a.per_prompt} generators per prompt, {n_img} images. Element targets = 3-vote share; "
             f"t2i_elem keeps unanimous elements only, t2i_elem_soft the split votes (Brier only). Score target = vote histogram over "
             f"5 levels. Prompt ids in t2i_align_heldout_prompts.json (exclude from any EvalMuse training use). BSD-3.")
    for w in writers.values():
        w.finish({"hf_id": HF_ID, "hf_config": "train_list.json + images.zip", "hf_split": "train", "hf_revision": rev,
                  "n_records_full": len(rows), "license": "commercial-ok", "lang": "en", "notes": notes})


if __name__ == "__main__":
    main()
