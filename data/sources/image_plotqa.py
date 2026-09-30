"""Cauldron plotqa (numeric / categorical answers on synthetic plots) -> 4-way choice; yes/no -> noul.
Numeric distractors: same-plot numbers within 5x of the answer, then ±10–50% perturbation in the same notation.

python -m data.sources.image_plotqa --out data/raw/image/plotqa.jsonl --limit 30
"""
import re

from data.sources.image_textvqa import run

NAME = CONFIG = "plotqa"
PER_IMAGE = 8  # a Cauldron plotqa row carries ~180 questions on one plot; keep the grids spread across plots


def fix_exponent(a: str) -> str:
    """Cauldron stripped trailing zeros from '%.2e' answers: '1.22e+1' is 1.22e+10 (Python never writes a
    one-digit exponent). Restore the dropped zero."""
    return re.sub(r"^(-?\d+(?:\.\d+)?e[+-])(\d)$", r"\g<1>\g<2>0", a)


if __name__ == "__main__":
    run(NAME, CONFIG, per_image=PER_IMAGE, fix_answer=fix_exponent)
