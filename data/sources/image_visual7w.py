"""Cauldron visual7w (telling questions, 4-way MC) -> choice.

python -m data.sources.image_visual7w --out data/raw/image/visual7w.jsonl --limit 30
"""
from data.sources.image_ai2d import run

NAME = CONFIG = "visual7w"

if __name__ == "__main__":
    run(NAME, CONFIG)
