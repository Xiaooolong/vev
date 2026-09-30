import json

from evals import report
from evals.metrics import Row, compute_metrics


def result(target: str, rows: list[Row]) -> dict:
    return {"target": target, "base_url": "http://x", "model": "m", "records_file": "f", "records_sha256": "0",
            "n": len({r.record_id for r in rows}), "metrics": compute_metrics(rows, n_boot=20, noise_reps=10), "raw": []}


def test_report_markdown(tmp_path):
    rows_a = [Row("r0", "q", "noul", 0.9, 1.0, 0.9, latency_ms=100.0, pred_repeat=0.9),
              Row("r1", "q", "noul", 1.0, 0.0, 1.0, latency_ms=300.0)]
    rows_b = [Row("r0", "q", "choice", {"A": 1.0, "B": 0.0}, {"A": 1.0, "B": 0.0}, 1.0, latency_ms=50.0,
                  pred_reversed={"A": 0.0, "B": 1.0})]
    (tmp_path / "setA.json").write_text(json.dumps(result("tgt", rows_a)), encoding="utf-8")
    (tmp_path / "setB.json").write_text(json.dumps(result("tgt", rows_b)), encoding="utf-8")
    (tmp_path / "summary.json").write_text(json.dumps({"metrics": {}}), encoding="utf-8")
    (tmp_path / "junk.json").write_text("not json", encoding="utf-8")
    out = tmp_path / "summary.md"
    report.main([str(tmp_path), "--out", str(out)])
    md = out.read_text(encoding="utf-8")
    lines = [ln for ln in md.splitlines() if ln.startswith("|")]
    assert md.startswith("# tgt")
    assert len(lines) == 4  # header, separator, two sets
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    assert header[0] == "set" and "ece_noise_floor" in header and "p50 latency ms" in header
    a = [c.strip() for c in lines[2].strip("|").split("|")]
    b = [c.strip() for c in lines[3].strip("|").split("|")]
    assert a[0] == "setA" and b[0] == "setB"
    assert a[1] == "2" and a[2] == "0.500"
    assert a[8] == "1/1"  # one confidence==1.0 answer, and it is wrong
    assert a[9] == "—" and a[11] == "0.0000"
    assert a[header.index("p50 latency ms")] == "200"
    assert b[9] == "1.000"
    assert "[" in a[4]
