import os
from pathlib import Path

import pytest
import torch
from transformers.cache_utils import DynamicCache, DynamicLayer, LinearAttentionLayer

from vev.pointer import common_prefix_len, fork_cache


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
    f = fork_cache(cache, 3)
    assert f.layers[0].keys.shape == (3, 2, 3, 4) and f.layers[1].recurrent_states[0].shape == (3, 2, 4, 4)
    f.layers[1].update_recurrent_state(torch.zeros(3, 2, 4, 4))  # in-place in the fork only
    f.layers[1].update_conv_state(torch.zeros(3, 5, 1))
    f.layers[0].update(torch.zeros(3, 2, 1, 4), torch.zeros(3, 2, 1, 4))
    assert lin.recurrent_states[0].eq(1).all() and lin.conv_states[0].eq(1).all()
    assert att.keys.shape == (1, 2, 3, 4) and cache.layers[1] is lin


CKPT = os.environ.get("VEV_TEST_POINTER_CKPT")


@pytest.mark.skipif(not CKPT or not torch.cuda.is_available() or not Path(CKPT or ".").is_dir(),
                    reason="set VEV_TEST_POINTER_CKPT=<checkpoint dir> (needs CUDA)")
def test_prefix_share_matches_per_row_fp32():
    from vev.pointer import PointerEngine

    eng = PointerEngine(CKPT, dtype="fp32", prefix_min_tokens=0)
    state = {"ticket": "My kettle never arrived and I was charged twice. Refund please.", "tier": "gold"}
    qs = {"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?", "criteria": None},
          "topic": {"type": "choice", "instructions": "Topic?", "criteria": {"delivery": None, "billing": "Charges"}},
          "tone": {"type": "score", "instructions": "How polite?", "criteria": ["Rude", "Neutral", "Polite"]}}

    def probs(r):
        return {k: [a["noul"]] if a["type"] == "noul" else list(a["probabilities"].values()) for k, a in r.answers.items()}

    eng.prefix_share = False
    old = eng.run(state, qs)
    eng.prefix_share = True
    new = eng.run(state, qs)
    assert new.input_tokens == old.input_tokens and new.extensions["tokens"] == old.extensions["tokens"]
    po, pn = probs(old), probs(new)
    assert max(abs(x - y) for k in po for x, y in zip(po[k], pn[k])) <= 1e-4
    alone = probs(eng.run(state, {"tone": qs["tone"]}))["tone"]
    assert alone == pn["tone"]  # the shared prefix depends on the state only: exact isolation
