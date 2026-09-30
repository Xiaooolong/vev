"""Training rows from a build directory: record -> augment -> sample <= max_q questions -> render -> encode.
Every question is its own row carrying the full state prefix."""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info

from data.augment import Augmenter
from vev.pointer import (LabelTokens, collate_rows, encode_prior_row, encode_row, image_loader_for, option_spans,
                         prior_row, row_text, special_ids, target_vector)
from vev.state import serialize_state

_SOURCE_RE = re.compile(rb'"source": "([^"]+)"')


def option_ids(q: dict[str, Any]) -> list[str]:
    """Identity of each rendered option: choice label; score level index in the question as given."""
    if q["type"] == "choice":
        return list(q["criteria"])
    return [str(i) for i in range(len(q["criteria"]))]


def permuted(q: dict[str, Any], ids: list[str], rng: random.Random, reverse: bool = False) -> tuple[dict[str, Any], list[str]]:
    """Same question with its options in another order: choice shuffled (or reversed), score levels reversed (a score
    scale is only ever shown ascending or descending). Returns the question and the option identity per position."""
    if q["type"] == "score":
        return {**q, "criteria": list(reversed(q["criteria"]))}, list(reversed(ids))
    keys = list(q["criteria"])
    order = list(range(len(keys)))
    if reverse:
        order.reverse()
    else:
        while len(keys) > 1 and order == list(range(len(keys))):
            rng.shuffle(order)
    return {**q, "criteria": {keys[i]: q["criteria"][keys[i]] for i in order}}, [ids[i] for i in order]


def _images(node, out: list) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "image" and isinstance(v, dict) and "path" in v:
                out.append(v["path"])
            else:
                _images(v, out)
    elif isinstance(node, list):
        for v in node:
            _images(v, out)


class RowDataset(IterableDataset):
    def __init__(self, build_dir: str | Path, split: str, processor, *, limit: int = 0, seed: int = 0, max_q: int = 4,
                 augment: bool = True, max_pixels: int = 1024 * 1024, max_row_tokens: int = 4096,
                 sources: set[str] | None = None, max_images: int = 8, prior: bool = False,
                 source_weights: dict[str, float] | None = None, views: str | None = None, view_frac: float = 1.0):
        """views (prior mode, choice/score questions): 'consistency' adds, for a view_frac share of rows, one copy with
        the options permuted (row['views'], trained to agree with the row); 'anchor_sym' shows the row itself in a random
        order and attaches the question in its given and reversed order as teacher views (their base distributions
        are averaged, so the anchor does not pin the base model's position bias). Each view carries 'idx': for
        every option of the row, its position in the view."""
        self.build = Path(build_dir)
        self.views, self.view_frac = views, view_frac
        self.processor = processor
        self.tok = processor.tokenizer
        self.sid = special_ids(self.tok)
        self.labels = LabelTokens(self.tok) if prior else None
        self.seed, self.max_q, self.augment = seed, max_q, augment
        self.max_pixels, self.max_row_tokens, self.max_images = max_pixels, max_row_tokens, max_images
        lines = []
        with (self.build / f"{split}.jsonl").open("rb") as fh:
            for line in fh:
                if not line.strip():
                    continue
                if sources is not None:
                    m = _SOURCE_RE.search(line)
                    if not m or m.group(1).decode() not in sources:
                        continue
                lines.append(line)
        rng = random.Random(f"{seed}:records")
        rng.shuffle(lines)
        lines = lines[:limit] if limit else lines
        if source_weights:  # up/down-sample sources: weight w keeps floor(w) copies plus one more with prob frac(w)
            out = []
            for line in lines:
                m = _SOURCE_RE.search(line)
                w = source_weights.get(m.group(1).decode(), 1.0) if m else 1.0
                k = int(w) + (1 if rng.random() < (w - int(w)) else 0)
                out.extend([line] * k)
            rng.shuffle(out)
            lines = out
        self.lines = lines
        cfg_path = self.build / "augment_config.json"
        self.aug_cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else None
        self.text_pool: list[str] = []
        self.image_pool: list[str] = []
        for line in self.lines[:3000]:
            rec = json.loads(line)
            if isinstance(rec["state"], str):
                self.text_pool.append(rec["state"])
            _images(rec["state"], self.image_pool)
        self.epoch = 0
        self.skipped: dict[str, int] = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.lines)

    def _skip(self, why: str) -> None:
        self.skipped[why] = self.skipped.get(why, 0) + 1

    def _encode(self, segments, images, q: dict[str, Any], k: int) -> dict[str, Any]:
        if self.labels:
            text, ends = prior_row(self.processor, segments, q, self.labels.letters)
            return encode_prior_row(self.processor, text, images, ends)
        return encode_row(self.processor, row_text(self.processor, segments, q), images, self.sid, k)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        order = list(range(len(self.lines)))
        random.Random(f"{self.seed}:{self.epoch}:order").shuffle(order)
        loader = image_loader_for(self.build, self.max_pixels)
        aug = Augmenter(self.aug_cfg, random.Random(0), self.text_pool, self.image_pool, self.build / "aug_images") \
            if self.augment else None
        for idx in order[wid::nw]:
            rec = json.loads(self.lines[idx])
            rng = random.Random(f"{self.seed}:{self.epoch}:{idx}")
            if aug is not None:
                aug.rng = rng
                try:
                    rec = aug.apply(rec)
                except Exception as e:  # noqa: BLE001
                    self._skip(f"augment:{type(e).__name__}")
                    continue
            names = list(rec["questions"])
            rng.shuffle(names)
            try:
                segments, images = serialize_state(rec["state"], max_images=self.max_images, max_pixels=self.max_pixels,
                                                   image_loader=loader)
            except Exception as e:  # noqa: BLE001
                self._skip(f"state:{type(e).__name__}")
                continue
            for name in names[: self.max_q]:
                q = rec["questions"][name]
                k = len(option_spans(q))
                if k < 2:
                    self._skip("single_option")
                    continue
                target = rec["targets"][name]
                view_qs: list[tuple[dict[str, Any], list[str]]] = []
                if self.views and self.labels and q["type"] in ("choice", "score"):
                    ids = option_ids(q)
                    if self.views == "consistency" and rng.random() < self.view_frac:
                        view_qs = [permuted(q, ids, rng)]
                        prim_ids = ids
                    elif self.views == "anchor_sym":
                        view_qs = [(q, ids), permuted(q, ids, rng, reverse=True)]
                        if q["type"] == "choice" or rng.random() < 0.5:
                            q, prim_ids = permuted(q, ids, rng)
                        else:
                            prim_ids = ids
                        target = None
                try:
                    row = self._encode(segments, images, q, k)
                    views = []
                    for vq, vids in view_qs:
                        v = self._encode(segments, images, vq, k)
                        v["answer_ids"] = self.labels.answer_ids(vq)
                        v["idx"] = torch.tensor([vids.index(i) for i in prim_ids])
                        views.append(v)
                except Exception as e:  # noqa: BLE001
                    self._skip(f"encode:{type(e).__name__}")
                    continue
                if int(row["input_ids"].shape[0]) > self.max_row_tokens:
                    self._skip("too_long")
                    continue
                if views:
                    row["views"] = views
                if target is None:  # anchor rows: placeholder target, only the KL is used
                    row["target"] = torch.full((k,), 1.0 / k)
                else:
                    row["target"] = torch.tensor(target_vector(q, target), dtype=torch.float32)
                if self.labels:
                    row["answer_ids"] = self.labels.answer_ids(q)
                row["qtype"] = q["type"]
                row["source"] = rec["source"]
                row["id"] = f"{rec['id']}:{name}"
                yield row


def micro_batches(rows: Iterator[dict[str, Any]], micro_tokens: int, pad_id: int, buffer: int = 64):
    """Group rows into right-padded batches of at most micro_tokens padded tokens; a small buffer is sorted by length
    first so padding stays low. Yields (batch tensors, readouts, rows)."""
    buf: list[dict[str, Any]] = []

    def flush(items: list[dict[str, Any]]):
        items.sort(key=lambda r: int(r["input_ids"].shape[0]))
        cur: list[dict[str, Any]] = []
        for r in items:
            L = int(r["input_ids"].shape[0])
            if cur and L * (len(cur) + 1) > micro_tokens:
                batch, readouts = collate_rows(cur, pad_id)
                yield batch, readouts, cur
                cur = []
            cur.append(r)
        if cur:
            batch, readouts = collate_rows(cur, pad_id)
            yield batch, readouts, cur

    for row in rows:
        buf.append(row)
        if len(buf) >= buffer:
            yield from flush(buf)
            buf = []
    if buf:
        yield from flush(buf)


def describe(ds: RowDataset, n: int = 200) -> dict[str, Any]:
    """Token statistics over the first n rows (for planning budgets)."""
    lens, types = [], {}
    for i, row in enumerate(ds):
        if i >= n:
            break
        lens.append(int(row["input_ids"].shape[0]))
        types[row["qtype"]] = types.get(row["qtype"], 0) + 1
    lens.sort()
    return {"rows": len(lens), "tokens_mean": sum(lens) / max(1, len(lens)), "tokens_p50": lens[len(lens) // 2] if lens else 0,
            "tokens_max": lens[-1] if lens else 0, "types": types, "skipped": dict(ds.skipped)}


if __name__ == "__main__":  # quick look: python -m train.data <build_dir> [base]
    from transformers import AutoProcessor

    build = sys.argv[1]
    base = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3.5-0.8B"
    proc = AutoProcessor.from_pretrained(base)
    ds = RowDataset(build, "train", proc, limit=100)
    print(json.dumps(describe(ds), indent=2))
