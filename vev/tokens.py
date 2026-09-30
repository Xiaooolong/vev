"""Token accounting for usage (spec §5.2)."""

from __future__ import annotations

import json
from typing import Any

from vev.state import ImageSlot


def count_state_tokens(tok, segments: list, enc: dict, merge_size: int) -> tuple[int, int]:
    """(text tokens, image tokens) of the state; image tokens include vision_start/vision_end."""
    text = sum(len(tok.encode(s, add_special_tokens=False)) for s in segments if not isinstance(s, ImageSlot))
    grid = enc.get("image_grid_thw")
    image = 0 if grid is None else int(sum(int(g.prod()) // merge_size**2 + 2 for g in grid))
    return text, image


def count_output_tokens(tok, answers: dict[str, Any]) -> int:
    return len(tok.encode(json.dumps(answers, ensure_ascii=False, separators=(",", ":")), add_special_tokens=False))
