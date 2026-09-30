"""MMBench v1.0 CN dev -> choice records (lang=zh).

python -m evals.datasets.image_mmbench_cn --out evals/data/image/mmbench_cn.jsonl --limit 30
"""
from evals.datasets.image_mmbench_en import run

if __name__ == "__main__":
    run("mmbench_cn", "cn", "dev", "zh", 1164,
        "Chinese translation of the EN dev set (same 1,164 questions/images). dev has 4,329 rows incl. "
        "CircularEval rotations (index >= 1e6), skipped. 'nan' options dropped. hint prepended to state.context. "
        "test split has no answers.")
