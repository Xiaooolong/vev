"""Startup warm-up of `vev serve` for Vev checkpoints: a fixed list of requests, run before the server starts listening.

flash-linear-attention's Triton kernels are compiled and autotuned the first time they meet a new autotune key, which
takes from under 1 s to about 11 s per kernel. Two keys depend on the request: the L2-norm kernel autotunes once per
2,048 tokens of batch rows x row length (key NB = ceil(rows x length / 2048)), and the cumulative-sum kernel once per
batch size. The list below covers the shapes the Decision Index boards send (text, one image, two images, at 1.6 MP,
one choice question, in the checkpoint's reading layout, so images and short texts are read twice), every NB from 1 to
8 with one question (rows up to 16,384 tokens), and batches of 2 to 64 questions. A longer row or another batch size
still pays its first-time autotune on a cold Triton cache.

The answers are discarded, and later answers do not depend on the warm-up: requests share no state, and an autotuned
kernel keeps the config Triton picked, also across restarts (it is stored in TRITON_CACHE_DIR). With an empty cache,
Triton picks each config by timing, with or without the warm-up, so the pick can differ between fresh machines."""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any

from PIL import Image, ImageDraw

log = logging.getLogger("vev")

BOARD_PIXELS = 1280 * 1280  # 1,638,400: the vision board's 1.6 MP
OPTIONS = {"A": "The first one", "B": "The second one", "C": "Both", "D": "Neither"}
MIXED = {
    "n": {"type": "noul", "instructions": "Is anything marked as late?", "criteria": None},
    "c": {"type": "choice", "instructions": "Which depot appears most often?",
          "criteria": {"north": None, "south": None}},
    "s": {"type": "score", "instructions": "How complete is the log?", "criteria": ["sparse", "partial", "complete"]},
}


def image_url(w: int, h: int, seed: int) -> str:
    """A deterministic synthetic JPEG of w x h pixels."""
    bands = [Image.linear_gradient("L").rotate(90 * ((seed + i) % 4)).resize((w, h)) for i in range(3)]
    img = Image.merge("RGB", bands)
    d = ImageDraw.Draw(img)
    for i in range(6):
        x, y = (w * (i + 1 + seed)) // 9 % w, (h * (2 * i + 1)) // 13
        d.ellipse([x, y, x + w // 8, y + h // 8], fill=((60 * i + 40 * seed) % 256, 30 * i, 200 - 25 * i))
        d.rectangle([w - x - w // 10, h - y - h // 12, w - x, h - y], outline=(20, 20, 20), width=4)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def text_state(tok, n_tokens: int) -> str:
    """A shipping log of about n_tokens tokens."""
    line = "Record {i}: parcel {p} left depot {d} on day {k}, status {s}.\n"
    lines = [line.format(i=i, p=(37 * i) % 1000, d="north" if i % 3 else "south", k=i % 30 + 1,
                         s="late" if i % 7 == 0 else "on time") for i in range(1, 2001)]
    per = len(tok.encode("".join(lines[:200]), add_special_tokens=False)) / 200
    return "".join(lines[: max(1, round(n_tokens / per))])


def choice(instructions: str) -> dict[str, dict[str, Any]]:
    return {"answer": {"type": "choice", "instructions": instructions, "criteria": OPTIONS}}


def requests(tok) -> list[tuple[str, Any, dict[str, dict[str, Any]]]]:
    """(label, state, questions). The comments give the NB of each one-question row in the v0.2 layout (read twice)."""
    big = [image_url(1280, 1280, s) for s in range(4)]
    out: list[tuple[str, Any, dict[str, dict[str, Any]]]] = [
        ("text, ~100 tokens", "Order #4411 still shows 'label created' after 9 days.",
         choice("Which order is late?")),  # 1
        ("text, ~2,500 tokens", text_state(tok, 2500), choice("Which depot sent more parcels?")),  # 3
        ("text, ~11,000 tokens", text_state(tok, 11000), choice("Which depot sent more parcels?")),  # 6
        ("text, ~15,000 tokens", text_state(tok, 15000), choice("Which depot sent more parcels?")),  # 8
        ("1 image, 0.3 MP", {"image": {"url": image_url(640, 480, 0)}}, choice("Which shape is larger?")),  # 1
        ("1 image, 1.6 MP", {"image": {"url": big[0]}}, choice("Which shape is larger?")),  # 2
    ]
    for n in (2, 3, 4):  # NB 4, 5, 7
        st = {f"image_{i + 1}": {"image": {"url": big[i]}} for i in range(n)}
        out.append((f"{n} images, 1.6 MP each", st, choice("Which image has more circles?")))
    out.append(("1 image, 1.6 MP, 3 questions", {"note": "depot camera", "image": {"url": big[1]}}, MIXED))
    for k in (2, 3, 4, 6, 8, 16, 64):
        qs = {f"q{i}": {"type": "noul", "instructions": f"Is parcel {i} mentioned?", "criteria": None}
              for i in range(k)}
        out.append((f"text, {k} questions", "Record 1: parcel 37 left depot north on day 2.", qs))
    return out


def run(engine) -> None:
    """Run the warm-up requests on engine, one log line each."""
    max_pixels = getattr(engine, "max_image_pixels", 1024 * 1024)
    todo = requests(engine.tok)
    for i, (label, state, questions) in enumerate(todo, 1):
        t0 = time.perf_counter()
        r = engine.run(state, questions, max_pixels=max_pixels)
        log.info("warmup %d/%d: %s, %d tokens, %.0f ms", i, len(todo), label, r.input_tokens,
                 (time.perf_counter() - t0) * 1000)
