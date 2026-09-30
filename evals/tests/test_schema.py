import copy
import json

from evals.schema import iter_records, validate_record

GOOD = {
    "id": "t/1", "source": "t", "split": "val", "license": "unknown", "lang": "en",
    "state": {"context": "x", "image": {"path": "images/1.png"}, "more": [{"image": {"url": "data:image/png;base64,AA"}}]},
    "questions": {
        "q1": {"type": "choice", "instructions": "pick", "criteria": {"A": "a", "B": "b", "C": "c"}},
        "q2": {"type": "noul", "instructions": "yes?"},
        "q3": {"type": "score", "instructions": "how", "criteria": ["lo", "mid", "hi"]},
    },
    "targets": {"q1": {"A": 0.0, "B": 1.0, "C": 0.0}, "q2": 0.87, "q3": {"0": 0.2, "1": 0.5, "2": 0.3}},
    "meta": {"grid_id": "g1", "negated_questions": {"q2": {"type": "noul", "instructions": "no?"}}},
}


def bad(mutate) -> list[str]:
    rec = copy.deepcopy(GOOD)
    mutate(rec)
    return validate_record(rec)


def test_valid_record():
    assert validate_record(GOOD) == []


def test_target_keys_must_match_questions():
    assert bad(lambda r: r["targets"].pop("q2"))
    assert bad(lambda r: r["targets"].update(q9=0.5))


def test_choice_target_keys_must_match_criteria():
    assert bad(lambda r: r["targets"].update(q1={"A": 0.0, "B": 1.0}))
    assert bad(lambda r: r["targets"].update(q1={"A": 0.0, "B": 1.0, "D": 0.0}))


def test_noul_target_range():
    assert bad(lambda r: r["targets"].update(q2=1.2))
    assert bad(lambda r: r["targets"].update(q2=-0.1))
    assert bad(lambda r: r["targets"].update(q2={"true": 1.0}))
    assert bad(lambda r: r["targets"].update(q2=True))


def test_score_target_keys_are_zero_based_strings():
    assert bad(lambda r: r["targets"].update(q3={"1": 0.2, "2": 0.5, "3": 0.3}))
    assert bad(lambda r: r["targets"].update(q3={"lo": 0.2, "mid": 0.5, "hi": 0.3}))


def test_distribution_must_sum_to_one():
    assert bad(lambda r: r["targets"].update(q1={"A": 0.1, "B": 0.8, "C": 0.0}))
    assert not bad(lambda r: r["targets"].update(q1={"A": 0.1, "B": 0.9 + 5e-7, "C": 0.0}))


def test_image_object_shape():
    assert bad(lambda r: r["state"].update(image={"path": "a.png", "url": "data:image/png;base64,AA"}))
    assert bad(lambda r: r["state"]["more"].append({"image": {"file": "a.png"}}))
    assert bad(lambda r: r["state"].update(image={"path": "a.png", "alt": "x"}))
    # next to other keys, an "image" value without path/url is ordinary state data
    assert not bad(lambda r: r["state"].update(image={"caption": "a cat"}))
    assert bad(lambda r: r["state"]["more"].append({"image": {}}))


def test_required_fields_and_license():
    assert bad(lambda r: r.update(license="free"))
    assert bad(lambda r: r.update(state=None))
    assert bad(lambda r: r.update(questions={}))
    assert bad(lambda r: r["questions"]["q1"].update(type="rank"))


def test_negated_questions_must_be_noul():
    assert bad(lambda r: r["meta"].update(negated_questions={"q1": {"type": "noul"}}))
    assert bad(lambda r: r["meta"].update(negated_questions={"zz": {"type": "noul"}}))


def test_iter_records(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps(GOOD) + "\n\n" + json.dumps({**GOOD, "id": "t/2"}) + "\n", encoding="utf-8")
    assert [r["id"] for r in iter_records(p)] == ["t/1", "t/2"]
