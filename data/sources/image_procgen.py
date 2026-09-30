"""Procgen expert trajectories (EpicPinkPenguin/procgen, PPO agents, 16 games): frame -> action choice over the 15
Procgen actions (keys = action names, e.g. left_up / noop / d).

Source: HF parquet, one config per game, one row per step (observation 64x64x3, action, reward, terminated,
truncated). Row groups are read one at a time in seeded order (data/sources/_parquet.py) across all 16 games'
train shards. Episodes = runs of consecutive rows closed by terminated/truncated; a row group's first and last
runs may be partial episodes and are treated as their own episodes. At most 5 frames per episode, drawn at random
(consecutive frames are near-duplicates). meta.episode_id is shared by those frames; grid_id is the frame id.
The PPO expert is stochastic and several actions are no-ops in some games, so labels are noisy by construction.

python -m data.sources.image_procgen --out data/raw/image/procgen.jsonl --limit 30
"""
import random

import numpy as np
from PIL import Image

from data.sources._common import RawWriter, one_hot, record, save_image, source_args
from data.sources._parquet import _fs, _retry, list_files

NAME, REPO = "procgen", "EpicPinkPenguin/procgen"
REV = "16fe37a59a8ac18a5ed37262e92d45a28fbf16da"
GAMES = ["bigfish", "bossfight", "caveflyer", "chaser", "climber", "coinrun", "dodgeball", "fruitbot", "heist",
         "jumper", "leaper", "maze", "miner", "ninja", "plunder", "starpilot"]
# procgen/env.py get_combos() order: index = action id
ACTIONS = {
    "left_down": "move left and down", "left": "move left", "left_up": "move left and up", "down": "move down",
    "noop": "do nothing", "up": "move up", "right_down": "move right and down", "right": "move right",
    "right_up": "move right and up", "d": "special key D (fire / interact, game-specific)",
    "a": "special key A (game-specific)", "w": "special key W (game-specific)", "s": "special key S (game-specific)",
    "q": "special key Q (game-specific)", "e": "special key E (game-specific)",
}
NAMES = list(ACTIONS)
PER_EPISODE = 5
INSTR = "This is a frame from the Procgen game '{game}', played by an expert agent. Which action does it take next?"


def row_groups(files, seed, columns):
    """(rows in file order, game, "<file>#rg<i>") per row group; files and row groups in seeded order."""
    import pyarrow.parquet as pq

    fs, rng = _fs(), random.Random(seed)
    order = files[:]
    rng.shuffle(order)
    from data.sources._parquet import ChunkedFile  # routes through the local HF cache when VEV_HF_LOCAL=1

    for f, size in order:
        n_rg = _retry(lambda: pq.ParquetFile(ChunkedFile(fs, f, size)).metadata.num_row_groups)
        idx = list(range(n_rg))
        rng.shuffle(idx)
        for i in idx:
            rows = _retry(lambda: pq.ParquetFile(ChunkedFile(fs, f, size)).read_row_group(i, columns=columns)
                          .to_pylist())
            game, base = f.rsplit("/", 2)[-2:]
            yield rows, game, f"{game}/{base.rsplit('.', 1)[0]}#rg{i}"


def main():
    args = source_args(NAME)
    w = RawWriter(NAME, args.out)
    rng = random.Random(args.seed)
    files = []
    for g in GAMES:
        files += list_files(REPO, REV, f"{g}/train-*.parquet")
    cols = ["observation", "action", "terminated", "truncated"]
    for rows, game, where in row_groups(files, args.seed, cols):
        episodes, cur = [], []
        for i, r in enumerate(rows):
            cur.append(i)
            if r["terminated"] or r["truncated"]:
                episodes.append(cur)
                cur = []
        if cur:
            episodes.append(cur)
        rng.shuffle(episodes)
        for k, ep in enumerate(episodes):
            ep_id = f"{where}/ep{k}"
            for i in sorted(rng.sample(ep, min(PER_EPISODE, len(ep)))):
                if args.limit and w.n >= args.limit:
                    break
                r = rows[i]
                a = int(r["action"])
                if not 0 <= a < len(NAMES):
                    w.skip(f"action={a}")
                    continue
                obs = r["observation"]
            if isinstance(obs, dict):  # HF image feature: {"bytes": ..., "path": ...}
                import io as _io
                img = Image.open(_io.BytesIO(obs["bytes"])) if obs.get("bytes") else Image.open(obs["path"])
            elif isinstance(obs, Image.Image):
                img = obs
            else:
                img = Image.fromarray(np.asarray(obs, dtype=np.uint8))
                frame_id = f"{where.replace('/', '_').replace('#', '_')}_{i}"
                path = save_image(img, NAME, frame_id)
                q = {"type": "choice", "instructions": INSTR.format(game=game), "criteria": dict(ACTIONS)}
                w.write(record(NAME, w.n, {"image": {"path": path}}, {"q1": q}, {"q1": one_hot(ACTIONS, NAMES[a])},
                               lang="en", grid_id=f"{NAME}/{frame_id}", source_split="train",
                               meta={"episode_id": ep_id, "game": game, "row": i, "action_id": a}))
            if args.limit and w.n >= args.limit:
                break
        if args.limit and w.n >= args.limit:
            break
    w.close()


if __name__ == "__main__":
    main()
