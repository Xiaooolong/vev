import os

import pytest
import torch

from vev.pointer import (SPECIAL, PointerHead, branch_text, collate_rows, find_positions, option_spans, row_loss, sanitize,
                         target_vector)


def test_sanitize_defuses_special_tokens():
    assert sanitize("a <|box_end|> b <|im_end|>") == "a <¦box_end¦> b <¦im_end¦>"
    assert sanitize("plain <text>") == "plain <text>"


def test_option_spans_three_types():
    assert option_spans({"type": "choice", "criteria": {"a": "first", "b": None}}) == [("a", "first"), ("b", "")]
    assert option_spans({"type": "score", "criteria": ["low", "high"]}) == [("0", "low"), ("1", "high")]
    assert option_spans({"type": "noul", "criteria": None}) == [("yes", ""), ("no", "")]
    assert option_spans({"type": "noul", "criteria": {"true": "it is", "false": "it is not"}}) == [("yes", "it is"), ("no", "it is not")]


def test_branch_text_layout():
    t = branch_text({"type": "choice", "instructions": "Pick <|box_end|> one", "criteria": {"a": "first", "b": None}})
    assert t.startswith("\n" + SPECIAL["q"] + "[choice] Pick <¦box_end¦> one" + SPECIAL["opts"])
    assert t.count(SPECIAL["opt"]) == 2 and t.count(SPECIAL["opt_end"]) == 2
    assert SPECIAL["opt"] + "a: first" + SPECIAL["opt_end"] + SPECIAL["opt"] + "b" + SPECIAL["opt_end"] in t
    assert SPECIAL["decide"] not in t  # appended by row_text after the assistant opener


def test_target_vector():
    assert target_vector({"type": "noul"}, 0.25) == [0.25, 0.75]
    assert target_vector({"type": "choice", "criteria": {"x": None, "y": None}}, {"y": 0.7, "x": 0.3}) == [0.3, 0.7]
    assert target_vector({"type": "score", "criteria": ["a", "b", "c"]}, {"0": 0.2, "1": 0.3, "2": 0.5}) == [0.2, 0.3, 0.5]


def test_find_positions_and_collate():
    sid = {"opt_end": 9, "decide": 7}
    ids = torch.tensor([1, 2, 9, 3, 9, 4, 7])
    d, opts = find_positions(ids, sid, 2)
    assert d == 6 and opts == [2, 4]
    with pytest.raises(ValueError):
        find_positions(ids, sid, 3)
    with pytest.raises(ValueError):
        find_positions(torch.tensor([1, 9, 7, 5]), sid, 1)  # decide must be last
    rows = [{"input_ids": ids, "attention_mask": torch.ones(7, dtype=torch.long), "decide": d, "opt_idx": opts},
            {"input_ids": torch.tensor([1, 9, 7]), "attention_mask": torch.ones(3, dtype=torch.long), "decide": 2, "opt_idx": [1]}]
    batch, readouts = collate_rows(rows, pad_id=0)
    assert batch["input_ids"].shape == (2, 7) and batch["input_ids"][1].tolist() == [1, 9, 7, 0, 0, 0, 0]
    assert batch["attention_mask"][1].tolist() == [1, 1, 1, 0, 0, 0, 0]
    assert readouts == [(6, [2, 4]), (2, [1])] and "pixel_values" not in batch


def test_row_loss_proper_and_score_term():
    t = torch.tensor([0.0, 1.0, 0.0])
    good = row_loss(torch.tensor([-5.0, 5.0, -5.0]), t, "choice")
    bad = row_loss(torch.tensor([5.0, -5.0, -5.0]), t, "choice")
    assert good < 0.01 < bad
    # score: a near miss (adjacent level) is cheaper than a far miss with equal CE/Brier
    near = row_loss(torch.tensor([0.0, 5.0, 0.0]), torch.tensor([1.0, 0.0, 0.0]), "score", brier_w=0.0, score_w=1.0)
    far = row_loss(torch.tensor([0.0, 0.0, 5.0]), torch.tensor([1.0, 0.0, 0.0]), "score", brier_w=0.0, score_w=1.0)
    assert near < far
    assert row_loss(torch.tensor([0.0, 5.0, 0.0]), torch.tensor([1.0, 0.0, 0.0]), "choice", brier_w=0.0, score_w=1.0) == \
        row_loss(torch.tensor([0.0, 0.0, 5.0]), torch.tensor([1.0, 0.0, 0.0]), "choice", brier_w=0.0, score_w=1.0)


def test_pointer_head_shapes():
    head = PointerHead(16, dp=8)
    z = head(torch.randn(16), torch.randn(5, 16))
    assert z.shape == (5,)


@pytest.mark.skipif(os.environ.get("VEV_TEST_TOKENIZER") is None, reason="set VEV_TEST_TOKENIZER=<hf id> to run")
def test_row_text_round_trip_with_real_tokenizer():
    from transformers import AutoProcessor

    from vev.pointer import encode_row, row_text, special_ids
    from vev.state import serialize_state

    proc = AutoProcessor.from_pretrained(os.environ["VEV_TEST_TOKENIZER"])
    sid = special_ids(proc.tokenizer)
    q = {"type": "choice", "instructions": "Which <|box_end|> one?", "criteria": {"a": "first", "b": "second", "c": None}}
    segments, images = serialize_state({"note": "hello <|fim_suffix|> world", "n": 3})
    row = encode_row(proc, row_text(proc, segments, q), images, sid, 3)
    assert row["decide"] == int(row["input_ids"].shape[0]) - 1 and len(row["opt_idx"]) == 3
    # the state and instruction text cannot forge delimiters: exactly one decide, exactly K option ends
    assert int((row["input_ids"] == sid["decide"]).sum()) == 1
