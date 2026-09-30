"""Run every text converter in order, each in its own process, into data/raw/text/<src>.jsonl.

python -m data.sources.text_all                  # default per-source limit (30000)
python -m data.sources.text_all --limit 300 --only boolq,trec
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SOURCES = ["banking77", "boolq", "ag_news", "mnli", "sst5", "yelp", "trec", "dbpedia14", "amazon_reviews_multi_en", "imdb",
           "clinc150", "stsb", "ocnli", "tnews", "afqmc"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="passed through; default = each converter's own")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--only", default="", help="comma-separated subset of " + ",".join(SOURCES))
    args = ap.parse_args()
    only = [s for s in args.only.split(",") if s]
    unknown = set(only) - set(SOURCES)
    if unknown:
        sys.exit(f"unknown sources: {sorted(unknown)}")
    failed = []
    for src in SOURCES:
        if only and src not in only:
            continue
        cmd = [sys.executable, "-m", f"data.sources.text_{src}", "--out", f"data/raw/text/{src}.jsonl"]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]
        t0 = time.time()
        print(f"== {src}", flush=True)
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        print(f"== {src}: exit {rc}, {time.time() - t0:.0f}s", flush=True)
        if rc:
            failed.append(src)
    if failed:
        sys.exit(f"failed: {failed}")


if __name__ == "__main__":
    main()
