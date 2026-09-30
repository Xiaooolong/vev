import random

import pytest

from data.augment import DEFAULT_CONFIG, Augmenter
from data.tests.synth import choice_q, mixed_record, noul_q, rec, score_q
from evals.schema import validate_record

TEXT_POOL = ["alpha beta gamma delta epsilon " * 40, "zeta eta theta iota kappa " * 40]


def only(name: str, **extra) -> dict:
    cfg = {"p": {k: (1.0 if k == name else 0.0) for k in DEFAULT_CONFIG["p"]}}
    cfg.update(extra)
    return cfg


def aug(name: str, seed: int = 0, image_pool=(), aug_dir=None, **extra) -> Augmenter:
    return Augmenter(only(name, **extra), random.Random(seed), TEXT_POOL, list(image_pool), aug_dir)


def image_record():
    return rec("img", {"image": {"path": "../../images/s/1.jpg"}},
               {"c": choice_q(), "s": score_q(4), "n": noul_q()},
               {"c": {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.1}, "s": {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4},
                "n": 0.9})


@pytest.mark.parametrize("seed", range(6))
def test_wrap_reshapes_string_state_only(seed):
    r = mixed_record("r")
    r["state"] = "the customer says the parcel never arrived"
    out = aug("wrap", seed=seed).apply(r)
    assert out["meta"]["aug"] == ["wrap"]
    assert not isinstance(out["state"], str)
    assert "the parcel never arrived" in str(out["state"])
    assert out["targets"] == r["targets"] and out["questions"] == r["questions"]
    assert not validate_record(out)
    img = aug("wrap", seed=seed).apply(image_record())
    assert img["meta"]["aug"] == [] and img["state"] == image_record()["state"]


def test_shuffle_choice_target_follows_labels():
    r = mixed_record("r")
    orders = set()
    for seed in range(30):
        out = aug("shuffle", seed, score_reverse=0.0).apply(r)
        crit, tgt = out["questions"]["c"]["criteria"], out["targets"]["c"]
        assert list(crit) == list(tgt)
        assert crit == r["questions"]["c"]["criteria"] and tgt == r["targets"]["c"]  # same label -> text/prob
        assert out["meta"]["aug"] == ["shuffle"]
        orders.add(tuple(crit))
    assert len(orders) > 5


def test_shuffle_keeps_none_last():
    r = aug("abstain", abstain_delete=0.0).apply(mixed_record("r"))
    for seed in range(20):
        out = aug("shuffle", seed).apply(r)
        assert list(out["questions"]["c"]["criteria"])[-1] == "none"


def test_shuffle_score_reverse():
    r = mixed_record("r")
    out = aug("shuffle", score_reverse=1.0).apply(r)
    assert out["questions"]["s"]["criteria"] == list(reversed(r["questions"]["s"]["criteria"]))
    for i in range(5):
        assert out["targets"]["s"][str(i)] == r["targets"]["s"][str(4 - i)]
    assert validate_record(out) == []


def test_abstain_delete_moves_mass_to_none():
    r = mixed_record("r")
    out = aug("abstain", abstain_delete=1.0).apply(r)
    crit, tgt = out["questions"]["c"]["criteria"], out["targets"]["c"]
    assert "B" not in crit and crit["none"] == "None of the above"
    assert tgt == {"A": 0.1, "C": 0.2, "D": 0.1, "none": 0.6}
    assert validate_record(out) == []


def test_abstain_keep_gives_none_zero():
    r = mixed_record("r", lang="zh")
    out = aug("abstain", abstain_delete=0.0).apply(r)
    assert out["questions"]["c"]["criteria"]["none"] == "以上都不是"
    assert out["targets"]["c"] == {**r["targets"]["c"], "none": 0.0}


def test_negation_flips_target():
    r = mixed_record("r")
    out = aug("negation").apply(r)
    assert out["questions"]["n"]["instructions"] == "Is it not the case that Is the sky blue?"
    assert abs(out["targets"]["n"] - 0.2) < 1e-12
    assert out["targets"]["c"] == r["targets"]["c"]
    zh = aug("negation").apply(rec("z", "s", {"n": noul_q("天是蓝的吗？")}, {"n": 0.3}, lang="zh"))
    assert zh["questions"]["n"]["instructions"] == "是否并非如此：天是蓝的吗"
    assert abs(zh["targets"]["n"] - 0.7) < 1e-12


def test_nonsense_uniform_targets_and_generated_image(tmp_path):
    r = image_record()
    out = aug("nonsense", aug_dir=tmp_path / "aug_images", nonsense_kinds=["noise"]).apply(r)
    assert out["targets"]["c"] == {k: 0.25 for k in "ABCD"}
    assert out["targets"]["s"] == {str(i): 0.25 for i in range(4)}
    assert out["targets"]["n"] == 0.5
    path = out["state"]["image"]["path"]
    assert set(out["state"]) == {"image"} and set(out["state"]["image"]) == {"path"}
    assert path.startswith("aug_images/") and (tmp_path / path).exists()
    assert validate_record(out) == []


def test_nonsense_other_image_from_pool():
    pool = ["../../images/s/1.jpg", "../../images/s/2.jpg"]
    out = aug("nonsense", image_pool=pool, nonsense_kinds=["other"]).apply(image_record())
    assert out["state"] == {"image": {"path": "../../images/s/2.jpg"}}


def test_nonsense_skips_text_records():
    out = aug("nonsense").apply(mixed_record("r"))
    assert out["meta"]["aug"] == [] and out["targets"]["n"] == 0.8


@pytest.mark.parametrize("seed", range(10))
def test_trap_keeps_target(seed):
    r = mixed_record("r")
    out = aug("trap", seed).apply(r)
    assert out["targets"] == r["targets"]
    before, after = r["questions"]["c"]["criteria"], out["questions"]["c"]["criteria"]
    changed = [k for k in after if after[k] != before[k]]
    assert len(changed) == 1
    k = changed[0]
    assert after[k].startswith(before[k] + " ")
    assert any(after[k].endswith(s) for s in DEFAULT_CONFIG["trap_pool"]["en"])


def test_trap_half_on_correct():
    r = mixed_record("r")
    hits = sum(aug("trap", s).apply(r)["questions"]["c"]["criteria"]["B"] != "option text B" for s in range(1000))
    assert 430 < hits < 570


def test_injection_keeps_target():
    r = image_record()
    out = aug("injection").apply(r)
    assert out["targets"] == r["targets"]
    assert out["state"]["image"] == r["state"]["image"]
    assert out["state"]["text"] in DEFAULT_CONFIG["injection_pool"]["en"]
    assert validate_record(out) == []
    s = aug("injection").apply(mixed_record("t", state="hello world"))
    head, tail = s["state"].split("\n\n")
    assert head == "hello world" and tail in DEFAULT_CONFIG["injection_pool"]["en"]


@pytest.mark.parametrize("seed", range(10))
def test_long_state_keeps_original_text(seed):
    original = "The original state text that must survive."
    out = aug("long_state", seed).apply(mixed_record("r", state={"context": original, "n": 3}))
    ctx = out["state"]["context"]
    assert original in ctx and out["state"]["n"] == 3
    added = ctx.replace(original, "").strip()
    assert 200 <= len(added.split()) <= 800
    assert out["targets"] == mixed_record("r")["targets"]


def test_default_config_order_and_validity(tmp_path):
    a = Augmenter(DEFAULT_CONFIG, random.Random(0), TEXT_POOL, ["../../images/s/2.jpg"], tmp_path / "aug_images")
    seen = set()
    for i in range(600):
        r = image_record() if i % 2 else mixed_record(f"r{i}")
        out = a.apply(r)
        assert validate_record(out) == [], validate_record(out)
        names = out["meta"]["aug"]
        assert names == [n for n in DEFAULT_CONFIG["order"] if n in names]
        seen.update(names)
    assert seen == set(DEFAULT_CONFIG["order"])
