import io
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from evals import run

ROOT = Path(__file__).resolve().parents[2]


def write_records(tmp_path: Path, n: int = 6) -> Path:
    (tmp_path / "images").mkdir(exist_ok=True)
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 0, 0)).save(buf, format="PNG")
    (tmp_path / "images" / "red.png").write_bytes(buf.getvalue())
    lines = []
    for i in range(n):
        lines.append(json.dumps({
            "id": f"syn/{i}", "source": "syn", "split": "test", "license": "unknown", "lang": "en",
            "state": {"context": f"item {i}", "image": {"path": "images/red.png"}},
            "questions": {
                "pick": {"type": "choice", "instructions": f"pick {i}", "criteria": {"A": "a", "B": "b", "C": "c", "D": "d"}},
                "yes": {"type": "noul", "instructions": f"yes {i}"},
                "level": {"type": "score", "instructions": f"level {i}", "criteria": ["low", "mid", "high"]},
            },
            "targets": {"pick": {"A": 0.0, "B": 1.0, "C": 0.0, "D": 0.0}, "yes": 0.8,
                        "level": {"0": 0.1, "1": 0.3, "2": 0.6}},
            "meta": {"grid_id": f"g{i}", "negated_questions": {"yes": {"type": "noul", "instructions": f"not yes {i}"}}},
        }))
    p = tmp_path / "set.jsonl"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def run_cli(fake, records: Path, out: Path, *extra: str) -> dict:
    run.main(["--records", str(records), "--base-url", fake.url, "--target", "fake", "--out", str(out),
              "--concurrency", "4", *extra])
    return json.loads(out.read_text(encoding="utf-8"))


def test_jitter_perturbation_on_image_records(fake, tmp_path):
    records = write_records(tmp_path, n=4)
    res = run_cli(fake, records, tmp_path / "j.json", "--perturb", "jitter")
    m = res["metrics"]
    assert m["jitter_max_dp"] is not None and 0.0 <= m["jitter_max_dp"] <= 1.0
    assert m["jitter_flip_rate"] is not None and 0.0 <= m["jitter_flip_rate"] <= 1.0
    assert all("jitter" in q for entry in res["raw"] for q in entry["questions"].values())


def test_end_to_end_deterministic(fake, tmp_path):
    records = write_records(tmp_path)
    res = run_cli(fake, records, tmp_path / "out" / "set.json", "--perturb", "order,separate,repeat,negation")
    assert set(res) >= {"target", "base_url", "model", "records_file", "records_sha256", "n", "metrics", "raw"}
    assert res["n"] == 6 and len(res["raw"]) == 6
    m = res["metrics"]
    assert m["n_records"] == 6 and m["n_questions"] == 18
    assert m["by_type"] == {"choice": 6, "noul": 6, "score": 6}
    assert m["errors"]["count"] == 0
    # the fake is order-invariant, isolated and deterministic
    assert m["order_flip_rate"] == 0.0
    assert m["batched_vs_separate_max_dp"] == 0.0
    assert m["repeat_max_dp"] == 0.0
    assert m["negation_gap"] is not None
    for k in ("accuracy", "macro_f1", "brier", "nll", "ece", "ece_ci95", "ece_noise_floor", "reliability",
              "coverage_accuracy", "auroc_confidence", "confidence_distinct", "confidence_one_bucket", "latency_ms"):
        assert k in m
    q = res["raw"][0]["questions"]
    assert set(q["pick"]) >= {"pred", "confidence", "order", "separate", "repeat"}
    assert list(q["level"]["pred"]) == ["0", "1", "2"]
    assert q["level"]["order"] == q["level"]["pred"]  # reversed request mapped back to original levels
    assert "negation" in q["yes"]
    # 6 main + 6 order + 18 separate + 6 repeat + 6 negation
    assert len(fake.config.requests) == 42
    assert all(r["state"]["image"]["url"].startswith("data:image/png;base64,") for r in fake.config.requests)
    order_reqs = [r for r in fake.config.requests
                  if "pick" in r["questions"] and list(r["questions"]["pick"]["criteria"]) == ["D", "C", "B", "A"]]
    assert len(order_reqs) == 6
    assert order_reqs[0]["questions"]["level"]["criteria"] == ["high", "mid", "low"]


def test_oracle_target_matches_perfectly(fake, tmp_path):
    records = write_records(tmp_path, n=3)
    oracle = {}
    for i in range(3):
        oracle[f"pick {i}"] = {"A": 0.0, "B": 1.0, "C": 0.0, "D": 0.0}
        oracle[f"yes {i}"] = 1.0
        oracle[f"level {i}"] = {"0": 0.0, "1": 0.0, "2": 1.0}
    fake.reset(mode="oracle", oracle=oracle)
    m = run_cli(fake, records, tmp_path / "o.json")["metrics"]
    assert m["accuracy"] == 1.0 and m["ece"] == 0.0
    assert m["confidence_one_bucket"] == {"n": 9, "errors": 0}


def test_nondeterministic_and_order_sensitive_targets(fake, tmp_path):
    records = write_records(tmp_path, n=8)
    fake.reset(mode="random")
    m = run_cli(fake, records, tmp_path / "r.json", "--perturb", "repeat,separate")["metrics"]
    assert m["repeat_max_dp"] > 0 and m["batched_vs_separate_max_dp"] > 0
    fake.reset(order_bias=8.0)
    m = run_cli(fake, records, tmp_path / "b.json", "--perturb", "order")["metrics"]
    assert m["order_flip_rate"] == 1.0  # first-listed option always wins, reversal always flips


def test_http_errors_do_not_stop_the_run(fake, tmp_path):
    records = write_records(tmp_path, n=4)
    fake.reset(fail_instructions={"pick 1", "not yes 2"})
    res = run_cli(fake, records, tmp_path / "e.json", "--perturb", "negation", "--max-retries", "0")
    m = res["metrics"]
    assert m["n_records"] == 3
    assert m["errors"]["count"] == 2 and m["errors"]["by_status"] == {"500": 2}
    raw = {r["id"]: r for r in res["raw"]}
    assert raw["syn/1"]["error"]["status"] == 500
    assert raw["syn/2"]["perturb_errors"]["negation"]["status"] == 500
    assert "negation" not in raw["syn/2"]["questions"]["yes"]


def test_limit_and_module_entrypoint(fake, tmp_path):
    records = write_records(tmp_path, n=5)
    out = tmp_path / "cli.json"
    proc = subprocess.run(
        [sys.executable, "-m", "evals.run", "--records", str(records), "--base-url", fake.url,
         "--api-key", "local", "--model", "jev-latest", "--target", "fake", "--out", str(out), "--limit", "2"],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "[run]" in proc.stderr
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["n"] == 2 and res["metrics"]["n_records"] == 2


def test_invalid_records_abort(fake, tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"id": "x", "questions": {}}) + "\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "evals.run", "--records", str(p), "--base-url", fake.url, "--target", "t",
         "--out", str(tmp_path / "x.json")],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode != 0 and "invalid" in proc.stderr
    assert fake.config.requests == []
