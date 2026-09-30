import collections

from data.split import assign_split, ensure_coverage
from data.tests.synth import mixed_record


def test_assign_split_ratios_and_stability():
    counts = collections.Counter(assign_split(mixed_record(f"r{i}", grid_id=f"g{i}")) for i in range(50000))
    assert abs(counts["calibration"] / 50000 - 0.03) < 0.004
    assert abs(counts["validation"] / 50000 - 0.03) < 0.004
    r = mixed_record("x", grid_id="stable")
    assert len({assign_split(r) for _ in range(5)}) == 1


def test_same_grid_same_split():
    recs = [mixed_record(f"r{i}", grid_id=f"g{i // 4}") for i in range(4000)]
    ensure_coverage(recs)
    by_grid = collections.defaultdict(set)
    for r in recs:
        by_grid[r["meta"]["grid_id"]].add(r["split"])
    assert all(len(s) == 1 for s in by_grid.values())


def test_coverage_every_source_has_all_splits():
    recs = [mixed_record(f"a{i}", source="small_ok", grid_id=f"a{i}") for i in range(50)]
    recs += [mixed_record(f"b{i}", source="big", grid_id=f"b{i}") for i in range(3000)]
    ensure_coverage(recs)
    for src in ("small_ok", "big"):
        assert {r["split"] for r in recs if r["source"] == src} == {"train", "calibration", "validation"}


def test_tiny_source_all_train():
    recs = [mixed_record(f"a{i}", source="tiny", grid_id=f"a{i}") for i in range(49)]
    ensure_coverage(recs)
    assert {r["split"] for r in recs} == {"train"}
