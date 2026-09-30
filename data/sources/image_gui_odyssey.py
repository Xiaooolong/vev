"""GUIOdyssey (hflqf88888/GUIOdyssey, the updated release): each step -> next_action choice + task_complete noul.

Source: per-episode annotation JSONs (annotations/<episode_id>.json) + split lists (splits/random_split.json) +
screenshots in a 93 GB split zip (screenshots.z01-.z08 + .zip). Episodes are visited in seeded order; only the
zip central directory (cached) and the screenshots a run needs are range-read.

Action vocabulary (card: CLICK, SCROLL, LONG_PRESS, TYPE, COMPLETE, IMPOSSIBLE, HOME, BACK; the data also writes
TEXT for typing and CLICK with info KEY_HOME / KEY_BACK / KEY_RECENT for system keys) -> click, long_press, scroll,
type, navigate_home, navigate_back, recent_apps, done, impossible. task_complete = 1 only on COMPLETE.

python -m data.sources.image_gui_odyssey --out data/raw/image/gui_odyssey.jsonl --limit 30
"""
import json
import random

from data.sources._common import RawWriter, image_from_bytes, one_hot, record, save_image, source_args
from data.sources._ranged import RemoteZip, fetch, fetch_bytes, hf_url

NAME, REPO = "gui_odyssey", "hflqf88888/GUIOdyssey"
REV = "61632d0f3f4d51d7e9561ce4f84347dd06b2019d"
SHOT_PART_SIZES = [10737418240] * 8 + [6740164517]
ACTIONS = {
    "click": "tap a UI element on the screen",
    "long_press": "long-press a UI element",
    "scroll": "swipe / scroll the screen",
    "type": "type text into the focused field",
    "navigate_home": "go to the home screen",
    "navigate_back": "press the back button",
    "recent_apps": "open the recent-apps switcher",
    "done": "stop: the task is complete",
    "impossible": "stop: the task cannot be completed",
}
KEYS = {"KEY_HOME": "navigate_home", "KEY_BACK": "navigate_back", "KEY_RECENT": "recent_apps"}
PLAIN = {"CLICK": "click", "LONG_PRESS": "long_press", "SCROLL": "scroll", "TYPE": "type", "TEXT": "type",
         "HOME": "navigate_home", "BACK": "navigate_back", "COMPLETE": "done", "IMPOSSIBLE": "impossible"}
ACTION_INSTR = "Given the task and the current phone screenshot, which action should the agent take next?"
DONE_INSTR = "Given the task and the current phone screenshot, has the task already been completed?"


def action_of(step):
    a, info = step["action"], step.get("info")
    if a == "CLICK" and isinstance(info, str) and info in KEYS:
        return KEYS[info]
    return PLAIN.get(a)


def main():
    args = source_args(NAME)
    w = RawWriter(NAME, args.out)
    splits = json.loads(fetch(hf_url(REPO, "splits/random_split.json", REV), "gui_odyssey_random_split.json")
                        .read_text(encoding="utf-8"))
    split_of = {e.rsplit(".", 1)[0]: s for s, ids in splits.items() for e in ids}
    shot_zip = RemoteZip([hf_url(REPO, f"screenshots/screenshots.z{i:02d}", REV) for i in range(1, 9)]
                         + [hf_url(REPO, "screenshots/screenshots.zip", REV)], SHOT_PART_SIZES)
    shot_of = {n.rsplit("/", 1)[-1]: n for n in shot_zip.names()}
    eps = sorted(split_of)
    random.Random(args.seed).shuffle(eps)
    n_q = lambda: sum(w.by_type.values())  # noqa: E731
    for ep_id in eps:
        if args.limit and n_q() >= args.limit:
            break
        ep = json.loads(fetch_bytes(hf_url(REPO, f"annotations/{ep_id}.json", REV)))
        task = ep["task_info"]["instruction"]
        for step in ep["steps"]:
            if args.limit and n_q() >= args.limit:
                break
            act = action_of(step)
            if act is None:
                w.skip(f"action={step['action']}")
                continue
            if step["screenshot"] not in shot_of:
                w.skip("screenshot_missing")
                continue
            stem = step["screenshot"].rsplit(".", 1)[0]
            path = save_image(image_from_bytes(shot_zip.read(shot_of[step["screenshot"]])), NAME, stem)
            state = {"task": task, "step": step["step"], "screenshot": {"image": {"path": path}}}
            qs = {"next_action": {"type": "choice", "instructions": ACTION_INSTR, "criteria": dict(ACTIONS)},
                  "task_complete": {"type": "noul", "instructions": DONE_INSTR}}
            ts = {"next_action": one_hot(ACTIONS, act), "task_complete": 1.0 if act == "done" else 0.0}
            meta = {"episode_id": ep_id, "step": step["step"], "orig_action": step["action"], "info": step.get("info"),
                    "category": ep["task_info"]["category"], "apps": ep["task_info"]["app"],
                    "device": ep["device_info"]["device_name"],
                    "low_level_instruction": step.get("low_level_instruction")}
            w.write(record(NAME, w.n, state, qs, ts, lang="en", grid_id=f"{NAME}/{stem}",
                           source_split=split_of[ep_id], meta=meta))
    w.close()


if __name__ == "__main__":
    main()
