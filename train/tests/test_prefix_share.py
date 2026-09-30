import os
from pathlib import Path

import pytest
import torch
from transformers.cache_utils import DynamicCache, DynamicLayer, LinearAttentionLayer

from vev.model import common_prefix_len, fork_cache


def test_common_prefix_len():
    assert common_prefix_len([[1, 2, 3], [1, 2, 4], [1, 2]]) == 2
    assert common_prefix_len([[5, 6], [5, 6]]) == 2
    assert common_prefix_len([[1], [2]]) == 0


def test_fork_cache_isolates_branches():
    att, lin = DynamicLayer(), LinearAttentionLayer()
    att.update(torch.ones(1, 2, 3, 4), torch.ones(1, 2, 3, 4))
    lin.update_conv_state(torch.ones(1, 5, 4), conv_kernel_size=4)
    lin.update_recurrent_state(torch.ones(1, 2, 4, 4))
    cache = DynamicCache()
    cache.layers = [att, lin]
    f = fork_cache(cache)
    f.layers[1].update_recurrent_state(torch.zeros(1, 2, 4, 4))  # in place, in the fork only
    f.layers[1].update_conv_state(torch.zeros(1, 5, 1))
    f.layers[0].update(torch.zeros(1, 2, 1, 4), torch.zeros(1, 2, 1, 4))
    assert lin.recurrent_states[0].eq(1).all() and lin.conv_states[0].eq(1).all()
    assert att.keys.shape == (1, 2, 3, 4) and f.layers[0].keys.shape == (1, 2, 4, 4) and cache.layers[1] is lin


def test_fork_cache_n_rows():
    att, lin = DynamicLayer(), LinearAttentionLayer()
    att.update(torch.ones(1, 2, 3, 4), torch.ones(1, 2, 3, 4))
    lin.update_conv_state(torch.ones(1, 5, 4), conv_kernel_size=4)
    lin.update_recurrent_state(torch.ones(1, 2, 4, 4))
    cache = DynamicCache()
    cache.layers = [att, lin]
    f = fork_cache(cache, 3)
    assert f.layers[0].keys.shape == (3, 2, 3, 4) and f.layers[1].recurrent_states[0].shape == (3, 2, 4, 4)
    f.layers[1].update_recurrent_state(torch.zeros(3, 2, 4, 4))
    f.layers[0].update(torch.zeros(3, 2, 2, 4), torch.zeros(3, 2, 2, 4))
    assert f.layers[0].keys.shape == (3, 2, 5, 4)
    assert lin.recurrent_states[0].eq(1).all() and att.keys.shape == (1, 2, 3, 4)


def test_chunks_respect_the_budget_and_keep_order():
    from vev.model import CheckpointEngine

    names = ["a", "b", "c", "d", "e"]
    length = {"a": 10, "b": 20, "c": 30, "d": 40, "e": 90}
    chunks = CheckpointEngine._chunks(names, length, lambda b, s: b * s <= 100)
    assert chunks == [["a", "b", "c"], ["d"], ["e"]]
    assert [k for c in chunks for k in c] == names


CKPT = os.environ.get("VEV_TEST_CKPT")


@pytest.mark.skipif(not CKPT or not torch.cuda.is_available() or not Path(CKPT or ".").is_dir(),
                    reason="set VEV_TEST_CKPT=<checkpoint dir> (needs CUDA)")
def test_batched_questions_match_single_questions_fp32():
    from vev.model import CheckpointEngine

    eng = CheckpointEngine(CKPT, dtype="fp32")
    state = {"ticket": "My kettle never arrived and I was charged twice. Refund please.", "tier": "gold"}
    qs = {"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?", "criteria": None},
          "topic": {"type": "choice", "instructions": "Topic?", "criteria": {"delivery": None, "billing": "Charges"}},
          "tone": {"type": "score", "instructions": "How polite?", "criteria": ["Rude", "Neutral", "Polite"]}}

    def probs(r):
        return {k: [a["noul"]] if a["type"] == "noul" else list(a["probabilities"].values()) for k, a in r.answers.items()}

    def gap(a, b):
        return max(abs(x - y) for k in a for x, y in zip(a[k], b[k]))

    def alone(prefix_min_tokens):
        eng.prefix_min_tokens = prefix_min_tokens
        return {k: probs(eng.run(state, {k: q}))[k] for k, q in qs.items()}

    def together(share_min_saved_tokens, questions=qs):
        eng.share_min_saved_tokens = share_min_saved_tokens
        return probs(eng.run(state, questions))

    single_shared, single_rows = alone(0), alone(10**9)
    batch_shared, batch_rows = together(0), together(10**9)
    # fp32 is not bit-exact across kernel paths and batch shapes (the DeltaNet Triton kernels): the two single-question
    # paths already differ by about 1.4e-4 on vev-4b and 2.5e-4 on vev-9b; a real bug shows up above 1e-2
    assert gap(batch_shared, single_shared) <= 1e-3
    assert gap(batch_rows, single_rows) <= 1e-3
    assert gap(batch_shared, batch_rows) <= 1e-3
    reordered = together(0, dict(reversed(list(qs.items()))))
    assert reordered == batch_shared  # rows are ordered by (length, name), not by request order
