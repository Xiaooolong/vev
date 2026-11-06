"""Decision Index vision engine for Vev (github.com/apolinario/decision-index, public vision suite): the kit's
HttpDecide, speaking Vev's /v1/systemone instead, against a running `vev serve`.

    PYTHONPATH=evals/decision_index decision-index vision pipeline --dir work/vision --engine vision_engine:VevVision \
        --option base_url=http://127.0.0.1:PORT --option model=jev-latest --workers 4 --out runs/vev-4b

The mapping is mechanical and the same for every row:
- The row's images become image objects in the state ({"image": {"url": data URL}}, the Vev image extension), in the
  row's order: one image as {"image": ...}; several as {"image_1": {"image": ...}, "image_2": ...}. They are capped
  at 1.6 MP exactly as the kit's own encoder does for its decide engine (aspect kept, LANCZOS, original bytes when
  small enough), except that a PNG it has to downscale is saved at zlib level 1 instead of Pillow's default 6: the
  same pixels, about half the encoding time.
- The row's state ({} in the 0.3 suite) is kept; its single choice question is sent unchanged (lettered criteria).
- The row's `prompt` (options_in_image, readout prefix) is not used: Vev reads the option probabilities directly.
- Vev's capacity refusals ("tokens (max ...)", "too many choices", too many images) are recorded as unsupported."""

from __future__ import annotations

import base64
import io

from decision_index.engines.base import Unsupported
from decision_index.vision import images as img
from decision_index.vision.engines import HttpDecide, pixels, single_choice

CAPACITY = ("tokens (max ", "too many choices", "too many images")


def data_url(ref, root=None, max_pixels=None):
    """The kit's images.data_url, with PNGs over max_pixels re-encoded at zlib level 1 (lossless, so the server decodes
    the same pixels as from the kit's level-6 file)."""
    from PIL import Image

    raw = img.load(ref, root)
    with Image.open(io.BytesIO(raw)) as im:
        size = img.fit(im.width, im.height, max_pixels) if max_pixels else im.size
        if size == im.size or im.format != "PNG":
            return img.data_url(ref, root, max_pixels)
        mode = "RGBA" if ("A" in im.mode or "transparency" in im.info) else "RGB"
        out = im.convert(mode).resize(size, Image.LANCZOS)
    buf = io.BytesIO()
    out.save(buf, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class VevVision(HttpDecide):
    name = "vev-vision"

    def __init__(self, base_url=None, model="jev-latest", **options):
        super().__init__(base_url=base_url, path="/v1/systemone", protocol="decisions", model=model, thinking=None, **options)
        self.provenance = {**self.provenance, "kind": "vev-systemone-vision", "policy": (
            "Images go into the state as Vev image objects in row order ({'image': ...} for one, image_1..image_n for "
            "several), capped at max_pixels as the kit's encoder does (PNGs it downscales saved at zlib level 1); the "
            "choice question is sent unchanged; the row's prompt options are not used; Vev capacity refusals are "
            "recorded as unsupported.")}

    def body(self, state, questions, images):
        key, q, keys = single_choice(questions)
        urls = [data_url(p, self.image_root, self.max_pixels) for p in images or []]
        st = dict(state or {})
        if len(urls) == 1:
            st["image"] = {"url": urls[0]}
        else:
            for i, u in enumerate(urls, 1):
                st[f"image_{i}"] = {"image": {"url": u}}
        return {"model": self.model, "state": st, "questions": {key: q}, **self.extra}, key, keys

    def __call__(self, state, questions, images=(), prompt=None):
        body, key, keys = self.body(state, questions, images)
        r = self.clients.get().post(self.path, json=body)
        if r.status_code == 400 and any(m in r.text for m in CAPACITY):
            raise Unsupported(r.text[:300])
        r.raise_for_status()
        d = r.json()
        d.pop("evaluation_trace", None)
        d["image_pixels"] = [pixels(p, self.image_root, self.max_pixels) for p in images or []]
        return self.parse(d, key, keys), d
