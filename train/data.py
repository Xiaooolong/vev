"""Training rows from a build directory: record -> augment -> sample <= max_q questions -> render -> encode.
Every question is its own row carrying the full state prefix."""

from __future__ import annotations

import json
import math
import random
import re
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from data.augment import Augmenter
from vev.model import collate_rows, encode_row, render_row
from vev.readout import LabelTokens
from vev.state import serialize_state

_SOURCE_RE = re.compile(rb'"source": "([^"]+)"')


def target_vector(q: dict[str, Any], target: Any) -> list[float]:
    """Soft target over the options in the order they are shown."""
    if q["type"] == "noul":
        p = float(target)
        return [p, 1.0 - p]
    if q["type"] == "choice":
        return [float(target[label]) for label in q["criteria"]]
    return [float(target[str(i)]) for i in range(len(q["criteria"]))]


def image_loader_for(base_dir: Path, max_pixels: int) -> Callable[[dict], Image.Image]:
    """Images of a build are stored as files next to the jsonl ({"image": {"path": ...}})."""
    def load(value: dict) -> Image.Image:
        img = Image.open(base_dir / value["path"])
        img.load()
        img = img.convert("RGB")
        w, h = img.size
        if w * h > max_pixels:
            s = math.sqrt(max_pixels / (w * h))
            img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.Resampling.BICUBIC)
        return img
    return load


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
                 sources: set[str] | None = None, max_images: int = 8):
        self.build = Path(build_dir)
        self.processor = processor
        self.labels = LabelTokens(processor.tokenizer)
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
        random.Random(f"{seed}:records").shuffle(lines)
        self.lines = lines[:limit] if limit else lines
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
                answer_ids = self.labels.answer_ids(q)
                if len(answer_ids) < 2:
                    self._skip("single_option")
                    continue
                try:
                    row = encode_row(self.processor, render_row(self.processor, segments, q, self.labels), images)
                except Exception as e:  # noqa: BLE001
                    self._skip(f"encode:{type(e).__name__}")
                    continue
                if int(row["input_ids"].shape[0]) > self.max_row_tokens:
                    self._skip("too_long")
                    continue
                row["target"] = torch.tensor(target_vector(q, rec["targets"][name]), dtype=torch.float32)
                row["answer_ids"] = answer_ids
                row["qtype"] = q["type"]
                row["source"] = rec["source"]
                row["id"] = f"{rec['id']}:{name}"
                yield row


def micro_batches(rows: Iterator[dict[str, Any]], micro_tokens: int, pad_id: int, buffer: int = 64):
    """Group rows into right-padded batches of at most micro_tokens padded tokens; a small buffer is sorted by length
    first so padding stays low. Yields (batch tensors, decide positions, rows)."""
    buf: list[dict[str, Any]] = []

    def flush(items: list[dict[str, Any]]):
        items.sort(key=lambda r: int(r["input_ids"].shape[0]))
        cur: list[dict[str, Any]] = []
        for r in items:
            L = int(r["input_ids"].shape[0])
            if cur and L * (len(cur) + 1) > micro_tokens:
                batch, decide = collate_rows(cur, pad_id)
                yield batch, decide, cur
                cur = []
            cur.append(r)
        if cur:
            batch, decide = collate_rows(cur, pad_id)
            yield batch, decide, cur

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
