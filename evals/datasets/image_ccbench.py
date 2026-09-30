"""CCBench (MMBench 'cc' config, Chinese culture, 4-choice) -> choice records (lang=zh).

python -m evals.datasets.image_ccbench --out evals/data/image/ccbench.jsonl --limit 30
"""
from evals.datasets.image_mmbench_en import run

if __name__ == "__main__":
    run("ccbench", "cc", "test", "zh", 509,
        "Only split is 'test' but it carries answers. 2,040 rows = 510 questions + CircularEval rotations "
        "(index >= 1e6), skipped. 1 question has answer 'CC' (not a label) and is skipped -> 509. "
        "lmms-lab/CCBench does not exist on HF; the 'cc' config of the MMBench mirror is used.")
