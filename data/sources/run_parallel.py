"""Run image/text converters in parallel (one subprocess per source), skipping sources that already have output.

    python -m data.sources.run_parallel --groups group1,group2,group3 --jobs 10 --limit 30000

Each source writes data/raw/<bucket>/<src>.jsonl and a log under data/raw/_logs/<src>.log. A source is
skipped when its output exists and is non-empty (delete the file to force a re-run).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent

GROUPS = {
    "text": ("text", ["banking77", "boolq", "ag_news", "mnli", "sst5", "yelp", "trec", "dbpedia14",
                      "amazon_reviews_multi_en", "imdb", "clinc150", "stsb", "ocnli", "tnews", "afqmc"]),
    "group1": ("image", ["vqav2_yesno", "vqav2_mc", "aokvqa", "ai2d", "visual7w", "nlvr2", "vsr", "textvqa",
                         "plotqa", "naturalbench", "mmt_bench", "mme_realworld"]),
    "group2": ("image", ["imagereward", "hpdv2", "llava_critic", "agiqa3k", "genai_bench", "sugarcrepe",
                         "seetrue", "foil_coco", "charxiv"]),
    "group3": ("image", ["androidcontrol", "amex", "gui_odyssey", "guicourse", "procgen", "alfred",
                         "multi_benchmark", "cmm_math", "gaokao_mm", "coco_cn"]),
}
DEFAULT_EXCLUDE = {"alfred", "seetrue"}  # alfred needs a 100 GB 7z; seetrue is test-only per its authors


def run_one(bucket: str, src: str, limit: int, seed: int, log_dir: Path) -> tuple[str, int, float, int]:
    out = DATA_DIR / "raw" / bucket / f"{src}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{src}.log"
    cmd = [sys.executable, "-m", f"data.sources.{bucket}_{src}", "--out", str(out), "--limit", str(limit), "--seed", str(seed)]
    t0 = time.time()
    with log.open("w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(DATA_DIR.parent)).returncode
    n = sum(1 for _ in out.open(encoding="utf-8")) if out.exists() else 0
    return src, rc, time.time() - t0, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="group1,group2,group3")
    ap.add_argument("--only", default="", help="comma-separated sources (overrides --groups)")
    ap.add_argument("--exclude", default=",".join(sorted(DEFAULT_EXCLUDE)))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="re-run sources that already have output")
    a = ap.parse_args()

    exclude = {s.strip() for s in a.exclude.split(",") if s.strip()}
    todo: list[tuple[str, str]] = []
    if a.only:
        wanted = {s.strip() for s in a.only.split(",") if s.strip()}
        for bucket, names in GROUPS.values():
            todo += [(bucket, s) for s in names if s in wanted]
    else:
        for g in a.groups.split(","):
            bucket, names = GROUPS[g.strip()]
            todo += [(bucket, s) for s in names]
    todo = [(b, s) for b, s in todo if s not in exclude]

    log_dir = DATA_DIR / "raw" / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    skipped, queue = [], []
    for bucket, src in todo:
        out = DATA_DIR / "raw" / bucket / f"{src}.jsonl"
        if not a.force and out.exists() and out.stat().st_size > 0:
            skipped.append(src)
        else:
            queue.append((bucket, src))
    print(f"[parallel] {len(queue)} to run, {len(skipped)} skipped (existing): {skipped}", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futs = {pool.submit(run_one, b, s, a.limit, a.seed, log_dir): s for b, s in queue}
        for fut in as_completed(futs):
            src, rc, dt, n = fut.result()
            results.append((src, rc, dt, n))
            print(f"[parallel] {src}: rc={rc} records={n} {dt / 60:.1f} min", flush=True)
    failed = [s for s, rc, _, n in results if rc != 0 or n == 0]
    print(f"[parallel] done: {len(results) - len(failed)} ok, {len(failed)} failed: {failed}", flush=True)
    for src, rc, _, n in results:
        if rc != 0 or n == 0:
            print(f"--- tail of {src}.log ---")
            print("".join((log_dir / f"{src}.log").read_text(encoding="utf-8", errors="replace").splitlines(True)[-15:]))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
