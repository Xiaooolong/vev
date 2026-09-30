"""Run the group-2 image converters (scoring / preference, image-text consistency, charts) one after another.

python -m data.sources.image_group2_all --limit 10
"""
import argparse
import subprocess
import sys

from data.sources._common import DATA_DIR, DEFAULT_LIMIT

SOURCES = ["imagereward", "hpdv2", "llava_critic", "agiqa3k", "genai_bench",
           "sugarcrepe", "seetrue", "foil_coco", "charxiv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated subset of " + ",".join(SOURCES))
    args = ap.parse_args()
    todo = [s for s in SOURCES if not args.only or s in args.only.split(",")]
    failed = []
    for name in todo:
        cmd = [sys.executable, "-m", f"data.sources.image_{name}", "--limit", str(args.limit),
               "--out", str(DATA_DIR / "raw" / "image" / f"{name}.jsonl"), "--seed", str(args.seed)]
        print(f"== {name}", flush=True)
        if subprocess.run(cmd).returncode != 0:
            failed.append(name)
    print(f"done: {len(todo) - len(failed)}/{len(todo)} ok" + (f"; failed: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
