"""Synthetic records and images for the data pipeline tests."""
import random
from pathlib import Path

from PIL import Image, ImageDraw


def draw_image(path: Path, seed: int, size: int = 96) -> Path:
    rng = random.Random(seed)
    img = Image.new("RGB", (size, size), tuple(rng.randrange(256) for _ in range(3)))
    d = ImageDraw.Draw(img)
    for _ in range(12):
        x0, y0 = rng.randrange(size), rng.randrange(size)
        d.rectangle([x0, y0, x0 + rng.randrange(8, 40), y0 + rng.randrange(8, 40)],
                    fill=tuple(rng.randrange(256) for _ in range(3)))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def choice_q(n: int = 4, instructions: str = "Which option fits?") -> dict:
    return {"type": "choice", "instructions": instructions,
            "criteria": {chr(65 + i): f"option text {chr(65 + i)}" for i in range(n)}}


def score_q(n: int = 5) -> dict:
    return {"type": "score", "instructions": "How good is it?", "criteria": [f"level {i} quality" for i in range(n)]}


def noul_q(text: str = "Is the sky blue?") -> dict:
    return {"type": "noul", "instructions": text}


def rec(rid: str, state, questions: dict, targets: dict, *, source: str = "synth", grid_id: str | None = None,
        lang: str = "en", license: str = "commercial-ok", meta: dict | None = None) -> dict:
    m = {"grid_id": grid_id or rid, "source_split": "train"}
    m.update(meta or {})
    return {"id": rid, "source": source, "split": "raw", "license": license, "lang": lang, "state": state,
            "questions": questions, "targets": targets, "meta": m}


def mixed_record(rid: str, state="some text", **kw) -> dict:
    return rec(rid, state,
               {"c": choice_q(), "s": score_q(), "n": noul_q()},
               {"c": {"A": 0.1, "B": 0.6, "C": 0.2, "D": 0.1},
                "s": {"0": 0.05, "1": 0.1, "2": 0.2, "3": 0.25, "4": 0.4},
                "n": 0.8}, **kw)
