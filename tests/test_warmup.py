"""vev.warmup: the requests reach the engine with its own pixel limit, and with the v0.2 reading layout the one-question
rows cover every L2-norm autotune bucket (NB = ceil(rows x length / 2048)) from 1 to 8."""

import math
import os

import pytest

from vev import warmup
from vev.readout import Result
from vev.state import serialize_state


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text) // 4))


class FakeEngine:
    tok = CharTokenizer()
    max_image_pixels = 1638400

    def __init__(self):
        self.calls = []

    def run(self, state, questions, max_pixels=1024 * 1024, **kwargs):
        self.calls.append((state, questions, max_pixels))
        return Result(answers={}, input_tokens=1, output_tokens=0)


def test_run_sends_every_request_with_the_engine_pixel_limit():
    engine = FakeEngine()
    warmup.run(engine)
    assert len(engine.calls) == len(warmup.requests(engine.tok))
    assert {c[2] for c in engine.calls} == {1638400}
    sizes = {im.size for state, _, _ in engine.calls for im in serialize_state(state, max_pixels=1638400)[1]}
    assert (1280, 1280) in sizes  # 1.6 MP images are not downscaled


@pytest.mark.skipif(os.environ.get("VEV_TEST_TOKENIZER") is None, reason="set VEV_TEST_TOKENIZER=<hf id> to run")
def test_one_question_rows_cover_nb_1_to_8():
    from transformers import AutoProcessor

    from vev.model import encode_rows, render_row
    from vev.readout import LabelTokens

    proc = AutoProcessor.from_pretrained(os.environ["VEV_TEST_TOKENIZER"])
    labels = LabelTokens(proc.tokenizer)
    nb = set()
    for _, state, questions in warmup.requests(proc.tokenizer):
        if len(questions) != 1:
            continue
        segments, images = serialize_state(state, max_pixels=warmup.BOARD_PIXELS)
        mode = "twice" if images else ""
        text = render_row(proc, segments, next(iter(questions.values())), labels, mode, "full", 3000)
        row = encode_rows(proc, [text], images * (2 if mode else 1))[0]
        nb.add(math.ceil(int(row["input_ids"].shape[0]) / 2048))
    assert nb == set(range(1, 9))
