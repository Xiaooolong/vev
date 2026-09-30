import argparse
import json
import sys
from pathlib import Path

COLUMNS = ["set", "n", "accuracy", "brier", "ece (95% CI)", "ece_noise_floor", "auroc", "conf_distinct",
           "one_bucket errors/n", "order_flip", "batched_vs_separate", "repeat", "jitter_flip", "p50 latency ms"]


def _f(x, digits: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def row_for(name: str, res: dict) -> list[str]:
    m = res["metrics"]
    ci = m.get("ece_ci95")
    ece = _f(m.get("ece"))
    if ci:
        ece += f" [{_f(ci[0])}, {_f(ci[1])}]"
    one = m.get("confidence_one_bucket") or {}
    return [
        name,
        _f(res.get("n")),
        _f(m.get("accuracy")),
        _f(m.get("brier")),
        ece,
        _f(m.get("ece_noise_floor")),
        _f(m.get("auroc_confidence")),
        _f(m.get("confidence_distinct")),
        f"{one.get('errors', 0)}/{one.get('n', 0)}",
        _f(m.get("order_flip_rate")),
        _f(m.get("batched_vs_separate_max_dp"), 4),
        _f(m.get("repeat_max_dp"), 4),
        _f(m.get("jitter_flip_rate")),
        _f((m.get("latency_ms") or {}).get("p50"), 0),
    ]


def build_report(result_dir: Path) -> str:
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    targets = set()
    for path in sorted(result_dir.glob("*.json")):
        if path.stem.startswith("summary"):
            continue
        try:
            res = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"[report] skip {path.name}: {e}", file=sys.stderr)
            continue
        if not isinstance(res, dict) or "metrics" not in res:
            print(f"[report] skip {path.name}: no metrics", file=sys.stderr)
            continue
        targets.add(res.get("target"))
        lines.append("| " + " | ".join(row_for(path.stem, res)) + " |")
    title = ", ".join(sorted(str(t) for t in targets)) or result_dir.name
    return f"# {title}\n\n" + "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m evals.report")
    ap.add_argument("result_dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    md = build_report(Path(args.result_dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"[report] wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
