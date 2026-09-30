import random

from train.data import option_ids, permuted


def test_choice_permutation_maps_back():
    q = {"type": "choice", "instructions": "x", "criteria": {"a": "A", "b": "B", "c": "C", "d": None}}
    ids = option_ids(q)
    for seed in range(20):
        vq, vids = permuted(q, ids, random.Random(seed))
        assert vids != ids and sorted(vids) == sorted(ids)
        assert list(vq["criteria"]) == vids and all(vq["criteria"][k] == q["criteria"][k] for k in vids)
        idx = [vids.index(i) for i in ids]
        assert [vids[j] for j in idx] == ids


def test_reverse_and_score():
    q = {"type": "choice", "instructions": "x", "criteria": {"a": None, "b": None, "c": None}}
    vq, vids = permuted(q, option_ids(q), random.Random(0), reverse=True)
    assert vids == ["c", "b", "a"]
    s = {"type": "score", "instructions": "x", "criteria": ["low", "mid", "high"]}
    vs, sids = permuted(s, option_ids(s), random.Random(0))
    assert vs["criteria"] == ["high", "mid", "low"] and sids == ["2", "1", "0"]
    idx = [sids.index(i) for i in option_ids(s)]
    assert [vs["criteria"][j] for j in idx] == s["criteria"]
