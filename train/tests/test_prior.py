import os

import pytest
import torch

from vev.pointer import SPECIAL, PointerHead, branch_text, prior_labels


def test_prior_labels_per_type():
    letters = ["A", "B", "C", "D"]
    assert prior_labels({"type": "choice", "criteria": {"x": None, "y": None, "z": None}}, letters) == ["A", "B", "C"]
    assert prior_labels({"type": "noul", "criteria": None}, letters) == ["Yes", "No"]
    assert prior_labels({"type": "score", "criteria": ["lo", "mid", "hi"]}, letters) == ["0", "1", "2"]


def test_branch_text_prior_mode_prefixes_and_hint():
    letters = ["A", "B", "C"]
    q = {"type": "choice", "instructions": "Pick", "criteria": {"x": "first", "y": None}}
    t = branch_text(q, letters)
    assert SPECIAL["opt"] + "A. x: first" + SPECIAL["opt_end"] + SPECIAL["opt"] + "B. y" + SPECIAL["opt_end"] in t
    assert t.endswith("\nAnswer with the letter of the single best option.")
    n = {"type": "noul", "instructions": "Is it?", "criteria": {"true": "yes it is", "false": "nope"}}
    t = branch_text(n, letters)
    assert SPECIAL["opt"] + "Yes: yes it is" + SPECIAL["opt_end"] + SPECIAL["opt"] + "No: nope" + SPECIAL["opt_end"] in t
    assert branch_text({"type": "noul", "instructions": "Is it?", "criteria": None}, letters).count(SPECIAL["opt"] + "Yes" + SPECIAL["opt_end"]) == 1
    s = {"type": "score", "instructions": "How much?", "criteria": ["none", "some"]}
    assert SPECIAL["opt"] + "0: none" + SPECIAL["opt_end"] + SPECIAL["opt"] + "1: some" + SPECIAL["opt_end"] in branch_text(s, letters)
    # without letters the old layout is unchanged
    assert "A. " not in branch_text(q) and "Answer with" not in branch_text(q)


def test_zero_output_head_gives_zero_logits_but_nonzero_gradient():
    head = PointerHead(16, dp=8)
    head.zero_output()
    hd, ho = torch.randn(16), torch.randn(3, 16)
    z = head(hd, ho)
    assert torch.all(z == 0)
    z.sum().backward()
    assert head.q.weight.grad is not None and float(head.q.weight.grad.abs().sum()) > 0


@pytest.mark.skipif(os.environ.get("VEV_TEST_TOKENIZER") is None, reason="set VEV_TEST_TOKENIZER=<hf id> to run")
def test_label_tokens_and_row_positions_with_real_tokenizer():
    from transformers import AutoProcessor

    from vev.pointer import LabelTokens, encode_row, row_text, special_ids
    from vev.state import serialize_state

    proc = AutoProcessor.from_pretrained(os.environ["VEV_TEST_TOKENIZER"])
    labels = LabelTokens(proc.tokenizer)
    sid = special_ids(proc.tokenizer)
    q = {"type": "choice", "instructions": "Which?", "criteria": {"a": "first", "b": "second", "c": None}}
    segments, images = serialize_state({"note": "hello"})
    row = encode_row(proc, row_text(proc, segments, q, labels.letters), images, sid, 3)
    assert len(row["opt_idx"]) == 3 and row["decide"] == int(row["input_ids"].shape[0]) - 1
    assert labels.answer_ids(q) == [labels.ids["A"], labels.ids["B"], labels.ids["C"]]


@pytest.mark.skipif(os.environ.get("VEV_TEST_TOKENIZER") is None, reason="set VEV_TEST_TOKENIZER=<hf id> to run")
def test_prior_row_matches_zero_shot_prompt_and_positions():
    from transformers import AutoProcessor

    from vev.pointer import LabelTokens, encode_prior_row, prior_row
    from vev.readout import SYSTEM_PROMPT as ZS_PROMPT
    from vev.state import serialize_state

    proc = AutoProcessor.from_pretrained(os.environ["VEV_TEST_TOKENIZER"])
    labels = LabelTokens(proc.tokenizer)
    segments, images = serialize_state({"note": "hello"})
    q = {"type": "choice", "instructions": "Which?", "criteria": {"a": "first", "b": "second", "c": None}}
    text, ends = prior_row(proc, segments, q, labels.letters)
    assert ZS_PROMPT in text and "Options:\nA. a - first\nB. b - second\nC. c\n\nAnswer with the letter" in text
    assert text.endswith("<think>\n\n</think>\n\n") and SPECIAL["decide"] not in text and SPECIAL["opt"] not in text
    row = encode_prior_row(proc, text, images, ends)
    tok = proc.tokenizer
    assert [tok.decode([int(row["input_ids"][i])]).strip() for i in row["opt_idx"]] == ["first", "second", "c"]
    assert row["decide"] == int(row["input_ids"].shape[0]) - 1
    n = {"type": "noul", "instructions": "Is it?", "criteria": None}
    text, ends = prior_row(proc, segments, n, labels.letters)
    row = encode_prior_row(proc, text, images, ends)
    assert [tok.decode([int(row["input_ids"][i])]).strip() for i in row["opt_idx"]] == ["Yes", "No"]
