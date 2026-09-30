import base64
import io

import pytest
from PIL import Image

from evals.client import SystemOneClient, SystemOneError

QS = {"q": {"type": "noul", "instructions": "is it dark?"}}


def make_client(fake, **kw) -> SystemOneClient:
    kw.setdefault("backoff", 0.01)
    return SystemOneClient(fake.url, "local", "jev-latest", timeout=10, **kw)


def test_ask_returns_json_with_latency(fake):
    with make_client(fake) as c:
        out = c.ask("hello", QS)
    assert out["model"] == "fake-1"
    assert 0.0 <= out["answers"]["q"]["noul"] <= 1.0
    assert out["latency_ms"] > 0
    assert fake.config.requests[0]["model"] == "jev-latest"


def test_image_paths_are_inlined_relative_to_base_dir(fake, tmp_path):
    (tmp_path / "images").mkdir()
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (0, 0, 0)).save(buf, format="PNG")
    (tmp_path / "images" / "a.png").write_bytes(buf.getvalue())
    state = {"text": "look", "pics": [{"image": {"path": "images/a.png"}}, {"image": {"url": "data:image/png;base64,AA"}}],
             "other": {"image": {"path": "x", "note": 1}}}
    with make_client(fake, base_dir=tmp_path) as c:
        c.ask(state, QS)
    sent = fake.config.requests[0]["state"]
    url = sent["pics"][0]["image"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == buf.getvalue()
    assert sent["pics"][1] == {"image": {"url": "data:image/png;base64,AA"}}
    assert sent["other"] == {"image": {"path": "x", "note": 1}}  # not the reserved shape: untouched
    assert state["pics"][0] == {"image": {"path": "images/a.png"}}  # caller's object not mutated


@pytest.mark.parametrize("status", [429, 529, 503])
def test_retries_on_transient_status(fake, status):
    fake.reset(fail_first=2, fail_status=status)
    with make_client(fake, max_retries=3) as c:
        out = c.ask("x", QS)
    assert out["answers"]["q"]["type"] == "noul"
    assert len(fake.config.requests) == 3


def test_gives_up_after_max_retries(fake):
    fake.reset(fail_first=100)
    with make_client(fake, max_retries=2) as c, pytest.raises(SystemOneError) as ei:
        c.ask("x", QS)
    assert ei.value.status == 529 and ei.value.attempts == 3
    assert len(fake.config.requests) == 3


def test_no_retry_on_client_error(fake):
    fake.reset(fail_first=1, fail_status=400)
    with make_client(fake, max_retries=3) as c, pytest.raises(SystemOneError) as ei:
        c.ask("x", QS)
    assert ei.value.status == 400 and ei.value.attempts == 1
    assert ei.value.body["detail"]["error_type"] == "overloaded_error"


def test_ask_many_keeps_order_and_captures_errors(fake):
    fake.reset(fail_instructions={"boom"}, delay_ms=20)
    items = [(f"s{i}", QS) for i in range(12)] + [("s", {"q": {"type": "noul", "instructions": "boom"}})]
    seen = []
    with make_client(fake, max_retries=0) as c:
        out = c.ask_many(items, concurrency=6, on_done=lambda i, r: seen.append(i))
    assert sorted(seen) == list(range(13))
    assert all(isinstance(r, dict) for r in out[:12])
    assert isinstance(out[12], SystemOneError) and out[12].status == 500
    # deterministic mode: same state -> same answer, and results line up with inputs
    with make_client(fake) as c:
        again = c.ask("s3", QS)
    assert again["answers"] == out[3]["answers"]
