"""Run the group-1 image converters in order, each in its own process; a failure does not stop the rest.

python -m data.sources.image_group1_all --limit 30
"""
import argparse
import subprocess
import sys
import time

from data.sources._common import DATA_DIR

SOURCES = ["vqav2_yesno", "vqav2_mc", "aokvqa", "ai2d", "visual7w", "nlvr2", "vsr", "textvqa", "plotqa",
           "naturalbench", "mmt_bench", "mme_realworld"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="passed through; omitted = each converter's default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated subset of source names")
    args = ap.parse_args()
    names = [s for s in SOURCES if not args.only or s in args.only.split(",")]
    failed = []
    for name in names:
        cmd = [sys.executable, "-m", f"data.sources.image_{name}",
               "--out", str(DATA_DIR / "raw" / "image" / f"{name}.jsonl"), "--seed", str(args.seed)]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        t0 = time.time()
        print(f"== {name}", flush=True)
        rc = subprocess.call(cmd, cwd=DATA_DIR.parent)
        print(f"== {name}: rc={rc} {time.time() - t0:.0f}s", flush=True)
        if rc:
            failed.append(name)
    print(f"done: {len(names) - len(failed)}/{len(names)} ok" + (f"; failed: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
