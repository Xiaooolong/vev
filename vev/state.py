"""Serialize a request `state` into interleaved text/image segments (spec §4)."""

from __future__ import annotations

import base64
import binascii
import io
import json
import math
import re
from collections.abc import Callable
from typing import Any

from PIL import Image

ALLOWED_MIME = {"image/png", "image/jpeg", "image/webp"}
DATA_URL_RE = re.compile(r"^data:([\w/+.-]+)((?:;[\w=.-]+)*);base64,(.*)$", re.DOTALL)
INDENT = "  "


class InvalidRequest(Exception):
    def __init__(self, status: int, error_type: str, message: str):
        super().__init__(message)
        self.status = status
        self.error_type = error_type
        self.message = message


class ImageSlot(int):
    """Placeholder in the segment stream; the value is the index into `images`."""


def is_image_value(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("url"), str)


def _scalar(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _fetch_remote(url: str) -> bytes:
    import httpx

    r = httpx.get(url, timeout=10.0, follow_redirects=True)
    r.raise_for_status()
    return r.content


def decode_image(url: str, n: int, path: str, max_pixels: int, allow_remote: bool) -> Image.Image:
    where = f"image #{n} at {path}"
    if url.startswith(("http://", "https://")):
        if not allow_remote:
            raise InvalidRequest(400, "invalid_request_error", f"{where}: remote image URLs are disabled; use a data: URI")
        try:
            raw = _fetch_remote(url)
        except Exception as e:
            raise InvalidRequest(400, "invalid_request_error", f"{where}: fetch failed: {e}") from e
    else:
        m = DATA_URL_RE.match(url)
        if not m:
            raise InvalidRequest(400, "invalid_request_error", f"{where}: url must be a base64 data: URI")
        mime = m.group(1).lower()
        if mime not in ALLOWED_MIME:
            raise InvalidRequest(400, "invalid_request_error", f"{where}: unsupported MIME type {mime!r}; allowed: {sorted(ALLOWED_MIME)}")
        try:
            raw = base64.b64decode(m.group(3), validate=True)
        except (binascii.Error, ValueError) as e:
            raise InvalidRequest(400, "invalid_request_error", f"{where}: invalid base64 ({e})") from e
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as e:
        raise InvalidRequest(400, "invalid_request_error", f"{where}: cannot decode image data ({type(e).__name__})") from e
    img = img.convert("RGB")
    w, h = img.size
    if w * h > max_pixels:
        s = math.sqrt(max_pixels / (w * h))
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.Resampling.BICUBIC)
    return img


def serialize_state(
    state: Any,
    max_images: int = 8,
    max_pixels: int = 1024 * 1024,
    allow_remote: bool = False,
    image_loader: Callable[[dict], Image.Image] | None = None,
) -> tuple[list[str | ImageSlot], list[Image.Image]]:
    """Render `state` in tree order. Returns (segments, images); segments mix text and ImageSlot.

    image_loader (training only): also treats {"image": {"path": ...}} as an image and loads it through the callable;
    the server never passes one, so clients cannot point at local files."""
    segments: list[str | ImageSlot] = []
    images: list[Image.Image] = []
    buf: list[str] = []

    def is_img(v: Any) -> bool:
        return is_image_value(v) or (image_loader is not None and isinstance(v, dict) and isinstance(v.get("path"), str))

    def emit(text: str) -> None:
        buf.append(text)

    def emit_image(value: dict, path: str) -> None:
        n = len(images) + 1
        if n > max_images:
            raise InvalidRequest(400, "invalid_request_error", f"image #{n} at {path}: too many images (max {max_images})")
        if image_loader is not None and "url" not in value:
            img = image_loader(value)
        else:
            img = decode_image(value["url"], n, path, max_pixels, allow_remote)
        if buf:
            segments.append("".join(buf))
            buf.clear()
        segments.append(ImageSlot(len(images)))
        images.append(img)

    def render(value: Any, depth: int, path: str) -> None:
        pad = INDENT * depth
        if isinstance(value, dict):
            for key, v in value.items():
                sub = f"{path}.{key}"
                if key == "image" and is_img(v):
                    emit(pad)
                    emit_image(v, sub)
                    emit("\n")
                elif isinstance(v, (dict, list)) and v:
                    emit(f"{pad}{key}:\n")
                    render(v, depth + 1, sub)
                else:
                    emit(f"{pad}{key}:{_inline(v, depth)}\n")
        elif isinstance(value, list):
            for i, v in enumerate(value):
                sub = f"{path}[{i}]"
                if isinstance(v, (dict, list)) and v:
                    emit(f"{pad}{i + 1}.\n")
                    render(v, depth + 1, sub)
                else:
                    emit(f"{pad}{i + 1}.{_inline(v, depth)}\n")
        else:
            emit(pad + _block(value, depth) + "\n")

    if isinstance(state, str):
        emit(state)
    else:
        render(state, 0, "state")
    if buf:
        segments.append("".join(buf).rstrip("\n"))
    return segments, images


def _block(value: Any, depth: int) -> str:
    """A scalar (or empty container); multi-line strings keep their lines, indented one level."""
    text = _scalar(value)
    if "\n" in text:
        pad = INDENT * (depth + 1)
        return "\n" + "\n".join(pad + line for line in text.split("\n"))
    return text


def _inline(value: Any, depth: int) -> str:
    text = _block(value, depth)
    return text if text.startswith("\n") else " " + text


def render_desc(value: Any) -> str:
    """Text rendering of a criteria description (string/object/array/null); images are not decoded here."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return _scalar(value)
