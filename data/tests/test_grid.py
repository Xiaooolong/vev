import random
import statistics

from data.grid import MAX_QUESTIONS, build_grids, derive_choice, derive_score, sample_size
from data.tests.synth import choice_q, mixed_record, noul_q, rec, score_q
from evals.schema import validate_record


def test_derive_choice_target_is_option_probability():
    q = choice_q()
    t = {"A": 0.1, "B": 0.6, "C": 0.2, "D": 0.1}
    picks = set()
    for seed in range(200):
        dq, dt = derive_choice(q, t, "en", random.Random(seed))
        label = next(k for k, v in q["criteria"].items() if dq["instructions"].endswith(f"Is the answer: {v}?"))
        assert dt == t[label]
        assert dq["instructions"].startswith(q["instructions"])
        picks.add(label)
    assert picks == {"A", "B", "C", "D"}


def test_derive_choice_half_correct():
    q, t = choice_q(), {"A": 0.0, "B": 1.0, "C": 0.0, "D": 0.0}
    pos = sum(derive_choice(q, t, "en", random.Random(s))[1] for s in range(2000))
    assert 900 < pos < 1100


def test_derive_score_target_is_tail_sum():
    q = score_q(5)
    t = {"0": 0.05, "1": 0.1, "2": 0.2, "3": 0.25, "4": 0.4}
    seen = set()
    for seed in range(200):
        dq, dt = derive_score(q, t, "en", random.Random(seed))
        k = next(i for i, d in enumerate(q["criteria"]) if dq["instructions"].endswith(f"Is it at least: {d}?"))
        assert k >= 1
        assert abs(dt - sum(t[str(i)] for i in range(k, 5))) < 1e-12
        seen.add(k)
    assert seen == {1, 2, 3, 4}


def test_sample_size_geometric():
    rng = random.Random(0)
    xs = [sample_size(rng) for _ in range(20000)]
    assert min(xs) == 1 and max(xs) == MAX_QUESTIONS
    # truncated geometric mean: sum_{k=0}^{15} (1-p)^k
    expect = sum(0.65 ** k for k in range(MAX_QUESTIONS))
    assert abs(statistics.mean(xs) - expect) < 0.05
    assert abs(xs.count(1) / len(xs) - 0.35) < 0.02


def test_build_grids_merges_and_keeps_every_question():
    records = [mixed_record(f"r{i}", grid_id=f"g{i % 3}", state=f"state {i % 3}") for i in range(30)]
    grids = build_grids(records, random.Random(1))
    for g in grids:
        assert validate_record(g) == [], validate_record(g)
        assert len(g["questions"]) <= MAX_QUESTIONS
        assert set(g["meta"]["derived"]) <= set(g["questions"])
        for name in g["meta"]["derived"]:
            assert g["questions"][name]["type"] == "noul"
    assert {g["meta"]["grid_id"] for g in grids} == {"g0", "g1", "g2"}
    original = sum(len(g["questions"]) - len(g["meta"]["derived"]) for g in grids)
    assert original == 30 * 3
    assert len({g["id"] for g in grids}) == len(grids)
    # a chunk only holds questions of its own grid: states never mix
    for g in grids:
        assert g["state"] == f"state {g['meta']['grid_id'][1]}"


def test_build_grids_derivation_rates():
    records = [mixed_record(f"r{i}") for i in range(2000)]
    grids = build_grids(records, random.Random(2))
    derived = [g["questions"][n]["instructions"] for g in grids for n in g["meta"]["derived"]]
    from_choice = sum("Is the answer:" in d for d in derived)
    from_score = sum("Is it at least:" in d for d in derived)
    assert abs(from_choice / 2000 - 0.5) < 0.04
    assert abs(from_score / 2000 - 0.3) < 0.04


def test_build_grids_derived_target_consistent_with_parent():
    records = [mixed_record(f"r{i}") for i in range(300)]
    t = records[0]["targets"]
    crit_c, crit_s = records[0]["questions"]["c"]["criteria"], records[0]["questions"]["s"]["criteria"]
    for g in build_grids(records, random.Random(3)):
        for name in g["meta"]["derived"]:
            ins, p = g["questions"][name]["instructions"], g["targets"][name]
            if "Is the answer:" in ins:
                label = next(k for k, v in crit_c.items() if ins.endswith(f"{v}?"))
                assert p == t["c"][label]
            else:
                k = next(i for i, d in enumerate(crit_s) if ins.endswith(f"{d}?"))
                assert abs(p - sum(t["s"][str(i)] for i in range(k, 5))) < 1e-12


def test_build_grids_skips_two_option_choice_and_keeps_negations():
    q = noul_q()
    r = rec("x", "s", {"c": choice_q(2), "n": q}, {"c": {"A": 1.0, "B": 0.0}, "n": 1.0},
            meta={"negated_questions": {"n": noul_q("Is the sky not blue?")}})
    for seed in range(20):
        grids = build_grids([r], random.Random(seed))
        assert sum(len(g["questions"]) for g in grids) == 2
        for g in grids:
            assert g["meta"]["derived"] == []
            assert validate_record(g) == []
        (g,) = [g for g in grids if q in g["questions"].values()]
        name = next(k for k, v in g["questions"].items() if v == q)
        assert g["meta"]["negated_questions"] == {name: noul_q("Is the sky not blue?")}


def test_build_grids_same_grid_different_state_not_merged():
    a = mixed_record("a", grid_id="g", state="one")
    b = mixed_record("b", grid_id="g", state="two")
    grids = build_grids([a, b], random.Random(0))
    assert {g["state"] for g in grids} == {"one", "two"}
    assert all(g["meta"]["grid_id"] == "g" for g in grids)


def test_derived_template_language_follows_question_text():
    rng = random.Random(0)
    q = {"type": "choice", "instructions": "Which action comes next?", "criteria": {"a": "click", "b": "type", "c": "scroll"}}
    dq, _ = derive_choice(q, {"a": 0.8, "b": 0.1, "c": 0.1}, "zh", rng)
    assert "答案是否是" not in dq["instructions"] and "Is the answer:" in dq["instructions"]
    q = {"type": "score", "instructions": "这张图的质量如何？", "criteria": ["差", "一般", "好"]}
    dq, _ = derive_score(q, {"0": 0.2, "1": 0.3, "2": 0.5}, "en", rng)
    assert "是否至少达到" in dq["instructions"] and " Is it" not in dq["instructions"]


def test_derived_uses_label_when_description_is_none():
    rng = random.Random(0)
    q = {"type": "choice", "instructions": "Which intent", "criteria": {"card_lost": None, "refund": "", "other": {"k": 1}}}
    seen = set()
    for seed in range(30):
        dq, _ = derive_choice(q, {"card_lost": 0.6, "refund": 0.3, "other": 0.1}, "en", random.Random(seed))
        seen.add(dq["instructions"])
    assert "None" not in "".join(seen)
    assert any(s.endswith("Is the answer: card_lost?") for s in seen)
    assert any(s.endswith("Is the answer: refund?") for s in seen)
    assert any(s.endswith('Is the answer: {"k": 1}?') for s in seen)
    assert all(s.startswith("Which intent. ") for s in seen)  # missing terminal punctuation is added


def test_record_ids_unique_across_sources_sharing_grid_ids():
    rng = random.Random(0)
    a = rec("a", "x", {"q1": noul_q()}, {"q1": 1.0}, source="s_a", grid_id="shared/img1")
    b = rec("b", "x", {"q1": noul_q("Other?")}, {"q1": 0.0}, source="s_b", grid_id="shared/img1")
    ids = [r["id"] for r in build_grids([a, b], rng)]
    assert len(ids) == len(set(ids)) == 2 and all(i.startswith(("s_a/shared", "s_b/shared")) for i in ids)
    c = rec("c", "x", {"q1": noul_q()}, {"q1": 1.0}, source="s_c", grid_id="s_c/x")
    assert build_grids([c], rng)[0]["id"].startswith("s_c/x#")
