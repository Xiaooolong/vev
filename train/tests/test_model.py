import os

import pytest
import torch

from train.data import target_vector
from train.train import row_loss
from vev.model import collate_rows, question_text, sanitize
from vev.readout import answer_format


def test_sanitize_defuses_special_tokens():
    assert sanitize("a <|box_end|> b <|im_end|>") == "a <¦box_end¦> b <¦im_end¦>"
    assert sanitize("plain <text>") == "plain <text>"


@pytest.mark.parametrize("q", [
    {"type": "choice", "instructions": "Pick one", "criteria": {"a": "first", "b": None, "c": {"x": 1}}},
    {"type": "score", "instructions": " How much? ", "criteria": ["none", "", "a lot"]},
    {"type": "noul", "instructions": "Is it?", "criteria": {"true": "yes it is", "false": None}},
    {"type": "noul", "instructions": "", "criteria": None},
])
def test_question_text_matches_zero_shot_prompt(q):
    """A trained checkpoint is read with exactly the zero-shot prompt (plain text is unchanged by sanitize)."""
    letters = ["A", "B", "C"]
    instructions = (q.get("instructions") or "").strip()
    expected = "\n\n" + (f"Question: {instructions}\n\n" if instructions else "") + answer_format(q, letters)
    assert question_text(q, letters) == expected


def test_target_vector():
    assert target_vector({"type": "noul"}, 0.25) == [0.25, 0.75]
    assert target_vector({"type": "choice", "criteria": {"x": None, "y": None}}, {"y": 0.7, "x": 0.3}) == [0.3, 0.7]
    assert target_vector({"type": "score", "criteria": ["a", "b", "c"]}, {"0": 0.2, "1": 0.3, "2": 0.5}) == [0.2, 0.3, 0.5]


def test_collate_right_pads_and_keeps_decide():
    rows = [{"input_ids": torch.tensor([1, 2, 3]), "decide": 2}, {"input_ids": torch.tensor([4, 5]), "decide": 1}]
    batch, decide = collate_rows(rows, pad_id=0)
    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 5, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert decide == [2, 1]


def test_row_loss_is_minimal_at_target():
    t = torch.tensor([0.8, 0.2])
    good = row_loss(torch.log(t), t, "choice")
    bad = row_loss(torch.log(torch.tensor([0.2, 0.8])), t, "choice")
    assert good < bad


@pytest.mark.skipif(os.environ.get("VEV_TEST_TOKENIZER") is None, reason="set VEV_TEST_TOKENIZER=<hf id> to run")
def test_render_row_ends_at_the_answer_position():
    from transformers import AutoProcessor

    from vev.model import encode_row, render_row
    from vev.readout import SYSTEM_PROMPT, LabelTokens
    from vev.state import serialize_state

    proc = AutoProcessor.from_pretrained(os.environ["VEV_TEST_TOKENIZER"])
    labels = LabelTokens(proc.tokenizer)
    segments, images = serialize_state({"note": "hello"})
    q = {"type": "choice", "instructions": "Which?", "criteria": {"a": "first", "b": "second", "c": None}}
    text = render_row(proc, segments, q, labels)
    assert SYSTEM_PROMPT in text and "Options:\nA. a - first\nB. b - second\nC. c\n\nAnswer with the letter" in text
    assert text.endswith("<think>\n\n</think>\n\n")
    row = encode_row(proc, text, images)
    assert row["decide"] == int(row["input_ids"].shape[0]) - 1
    assert labels.answer_ids(q) == [labels.ids["A"], labels.ids["B"], labels.ids["C"]]


@pytest.mark.skipif(os.environ.get("VEV_TEST_TOKENIZER") is None, reason="set VEV_TEST_TOKENIZER=<hf id> to run")
def test_encode_rows_matches_encode_row_with_an_image():
    import base64
    import io

    from PIL import Image
    from transformers import AutoProcessor

    from vev.model import encode_row, encode_rows, render_row
    from vev.readout import LabelTokens
    from vev.state import serialize_state

    proc = AutoProcessor.from_pretrained(os.environ["VEV_TEST_TOKENIZER"])
    labels = LabelTokens(proc.tokenizer)
    buf = io.BytesIO()
    Image.new("RGB", (700, 500), (10, 120, 200)).save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    state = {"a": {"image": {"url": url}}, "note": "two images", "b": {"image": {"url": url}}}
    segments, images = serialize_state(state)
    qs = [{"type": "noul", "instructions": "Is it blue?"},
          {"type": "choice", "instructions": "Which?", "criteria": {"x": "first", "y": None}},
          {"type": "score", "instructions": "How much?", "criteria": ["low", "high"]}]
    texts = [render_row(proc, segments, q, labels) for q in qs]
    plain = [render_row(proc, ["plain text"], q, labels) for q in qs]
    for rows_texts, imgs in ((texts, images), (plain, [])):
        for text, got in zip(rows_texts, encode_rows(proc, rows_texts, imgs)):
            want = encode_row(proc, text, imgs)
            assert set(got) == set(want)
            for k, v in want.items():
                if isinstance(v, torch.Tensor):
                    assert got[k].dtype == v.dtype and torch.equal(got[k], v), k
                else:
                    assert got[k] == v, k
