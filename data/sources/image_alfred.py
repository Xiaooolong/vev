"""ALFRED (askforalfred/alfred): egocentric AI2-THOR frame + task instruction -> next low-level action choice.

Needs the extracted "full" release (trajectory JSONs + raw_images), which ships only as one 7z archive:
    wget https://ai2-vision-alfred.s3-us-west-2.amazonaws.com/full_2.1.0.7z   # ~100+ GB, 7z = no partial read
    7z x full_2.1.0.7z -o$PROJECT_DIR/alfred                                    # -> $PROJECT_DIR/alfred/full_2.1.0/
and then  python -m data.sources.image_alfred --alfred-dir $PROJECT_DIR/alfred/full_2.1.0
(json_2.1.0.7z / json_feat_2.1.0.7z carry no frames; the HF mirrors checked either lack frames or are a single
11 GB parquet row group.) Nothing is sampled locally.

Per trajectory (<split>/<task>/<trial>/traj_data.json): plan.low_actions[i].api_action.action is the action taken
from the frame of images[] whose low_idx == i (first frame of that action); the frame after the last action gets
"stop". Vocabulary = the 12 ALFRED interaction/navigation actions + stop. State = {"task": turk task_desc,
"image": frame}. At most 5 frames per trajectory (random, seeded) to limit near-duplicates; meta.episode_id is the
trajectory, grid_id the frame. Splits: train, valid_seen, valid_unseen (tests have no actions).

python -m data.sources.image_alfred --out data/raw/image/alfred.jsonl --limit 30 --alfred-dir <full_2.1.0>
"""
import json
import random
import sys
from pathlib import Path

from PIL import Image

from data.sources._common import RawWriter, one_hot, record, save_image, source_args

NAME = "alfred"
ACTIONS = {
    "MoveAhead": "move forward", "RotateLeft": "turn left", "RotateRight": "turn right", "LookUp": "look up",
    "LookDown": "look down", "PickupObject": "pick up an object", "PutObject": "put the held object somewhere",
    "OpenObject": "open an object", "CloseObject": "close an object", "ToggleObjectOn": "turn an object on",
    "ToggleObjectOff": "turn an object off", "SliceObject": "slice an object", "stop": "stop: the task is done",
}
SPLITS = ["train", "valid_seen", "valid_unseen"]
PER_TRAJ = 5
INSTR = "Given the household task and the robot's current view, which low-level action does it take next?"


def extra(ap):
    ap.add_argument("--alfred-dir", default=None, help="extracted full_2.1.0 directory")


def main():
    args = source_args(NAME, extra)
    if not args.alfred_dir or not Path(args.alfred_dir).is_dir():
        sys.exit("alfred: pass --alfred-dir <extracted full_2.1.0>; see the module docstring for the download")
    root = Path(args.alfred_dir)
    w = RawWriter(NAME, args.out)
    rng = random.Random(args.seed)
    trajs = sorted((s, p) for s in SPLITS for p in (root / s).glob("*/*/traj_data.json"))
    rng.shuffle(trajs)
    for split, tj in trajs:
        if args.limit and w.n >= args.limit:
            break
        t = json.loads(tj.read_text(encoding="utf-8"))
        lows = [a["api_action"]["action"] for a in t["plan"]["low_actions"]]
        first = {}
        for im in t["images"]:
            first.setdefault(im["low_idx"], im["image_name"])
        steps = [(i, lows[i] if i < len(lows) else "stop") for i in sorted(first) if i <= len(lows)]
        anns = t.get("turk_annotations", {}).get("anns") or [{}]
        task = anns[0].get("task_desc") or t.get("task_type", "")
        ep = f"{split}/{tj.parent.parent.name}/{tj.parent.name}"
        for i, act in sorted(rng.sample(steps, min(PER_TRAJ, len(steps)))):
            if args.limit and w.n >= args.limit:
                break
            if act not in ACTIONS:
                w.skip(f"action={act}")
                continue
            frame = tj.parent / "raw_images" / first[i].replace(".png", ".jpg")
            if not frame.exists():
                w.skip("frame_missing")
                continue
            fid = f"{tj.parent.parent.name}_{tj.parent.name}_{first[i].rsplit('.', 1)[0]}"
            path = save_image(Image.open(frame), NAME, fid)
            w.write(record(NAME, w.n, {"task": task, "image": {"path": path}},
                           {"q1": {"type": "choice", "instructions": INSTR, "criteria": dict(ACTIONS)}},
                           {"q1": one_hot(ACTIONS, act)}, lang="en", grid_id=f"{NAME}/{fid}", source_split=split,
                           meta={"episode_id": ep, "low_idx": i, "task_type": t.get("task_type")}))
    w.close()


if __name__ == "__main__":
    main()
