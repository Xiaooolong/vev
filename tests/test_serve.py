"""The HTTP layer of `vev serve` on a model-free engine: /v1/models and the pixel limit passed to engine.run, for an
engine with its own max_image_pixels (Vev checkpoints) and one without (zero-shot base models, the conformance stub)."""

import pytest
from fastapi.testclient import TestClient

from vev.readout import Result
from vev.serve import MAX_IMAGE_PIXELS, create_app


class FakeEngine:
    def __init__(self, max_image_pixels=None):
        if max_image_pixels is not None:
            self.max_image_pixels = max_image_pixels
        self.max_pixels_seen = []

    def run(self, state, questions, max_images=8, max_pixels=1024 * 1024, allow_remote=False, max_state_tokens=32768,
            max_request_tokens=65536):
        self.max_pixels_seen.append(max_pixels)
        answers = {name: {"type": "noul", "noul": 0.5} for name in questions}
        return Result(answers=answers, input_tokens=1, output_tokens=len(answers), extensions={})


@pytest.mark.parametrize("limit, expected", [(1638400, 1638400), (None, MAX_IMAGE_PIXELS)])
def test_models_and_systemone_use_the_engine_pixel_limit(limit, expected):
    engine = FakeEngine(limit)
    client = TestClient(create_app(engine, "vev-test", "Qwen/Qwen3.5-4B", allow_remote_images=False))
    r = client.get("/v1/models")
    assert r.status_code == 200, r.text
    card = r.json()["models"][0]
    assert card["name"] == "vev-test"
    assert card["limits"]["max_image_pixels"] == expected
    body = {"model": "vev-test", "state": "hello", "questions": {"q": {"type": "noul", "instructions": "Is it?"}}}
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 200, r.text
    assert engine.max_pixels_seen == [expected]


@pytest.mark.parametrize("date, expected", [("2026-11-06T00:00:00Z", "2026-11-06T00:00:00Z"), (None, "2026-10-01T00:00:00Z")])
def test_models_report_the_checkpoint_release_date(date, expected):
    engine = FakeEngine()
    engine.release_date = date
    client = TestClient(create_app(engine, "vev-test", "Qwen/Qwen3.5-4B", allow_remote_images=False))
    assert client.get("/v1/models").json()["models"][0]["release_date"] == expected
