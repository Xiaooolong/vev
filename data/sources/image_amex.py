"""AMEX (Yuxiang007/AMEX): each step -> next_action choice + task_complete noul, and for TAP steps a
"which element to tap" choice over <=8 clickable elements of that screen (e1..e8, order random).

Source: three zips on HF (instruction_anno.zip 3 MB, element_anno.zip 115 MB, screenshot.z01-.z08+.zip 93 GB split
archive). Only the central directories (~27 MB, cached) and the members a run needs are range-read.

Action vocabulary (AMEX L3 action space): TAP, SWIPE, TYPE, PRESS_BACK, PRESS_HOME, PRESS_ENTER, TASK_COMPLETE,
TASK_IMPOSSIBLE -> click, scroll, type, navigate_back, navigate_home, enter, done, impossible.
Element target = the smallest clickable element whose bbox contains touch_coord; candidates = the target plus up to
7 other clickable elements with a non-empty, distinct description (xml_desc / functionality) that do not contain the
touch point. Descriptions carry the bbox in 0-1000 screen coordinates so equal texts stay distinguishable.

python -m data.sources.image_amex --out data/raw/image/amex.jsonl --limit 30
"""
import json
import random

from data.sources._common import RawWriter, image_from_bytes, one_hot, record, save_image, source_args
from data.sources._ranged import RemoteZip, hf_url

NAME, REPO = "amex", "Yuxiang007/AMEX"
REV = "17196b29c88dd48a7fb90ef9131bc5c7bf39f26e"
SHOT_PART_SIZES = [10737418240] * 8 + [7581961664]
ACTION_MAP = {"TAP": "click", "SWIPE": "scroll", "TYPE": "type", "PRESS_BACK": "navigate_back",
              "PRESS_HOME": "navigate_home", "PRESS_ENTER": "enter", "TASK_COMPLETE": "done",
              "TASK_IMPOSSIBLE": "impossible"}
ACTIONS = {
    "click": "tap a UI element on the screen",
    "scroll": "swipe / scroll the screen",
    "type": "type text into the focused field",
    "navigate_back": "press the back button",
    "navigate_home": "go to the home screen",
    "enter": "press the enter key",
    "done": "stop: the task is complete",
    "impossible": "stop: the task cannot be completed",
}
ACTION_INSTR = "Given the task and the current phone screenshot, which action should the agent take next?"
DONE_INSTR = "Given the task and the current phone screenshot, has the task already been completed?"
ELEM_INSTR = ("Given the task and the current phone screenshot, which element should the agent tap next? "
              "Boxes are [x1, y1, x2, y2] in 0-1000 screen coordinates.")
MAX_ELEMS = 8


def _desc(el) -> str:
    parts = [el.get("functionality") or ""] + list(dict.fromkeys(d for d in el.get("xml_desc") or [] if d))
    return " / ".join(p.strip() for p in parts if p and p.strip())


def _norm_box(b, w, h):
    return [round(b[0] * 1000 / w), round(b[1] * 1000 / h), round(b[2] * 1000 / w), round(b[3] * 1000 / h)]


def element_question(step, elems, rng):
    x, y = step["touch_coord"]
    w, h = step["device_dim"]
    inside = [e for e in elems if e["bbox"][0] <= x <= e["bbox"][2] and e["bbox"][1] <= y <= e["bbox"][3]]
    if not inside:
        return None, "tap_outside_elements"
    target = min(inside, key=lambda e: (e["bbox"][2] - e["bbox"][0]) * (e["bbox"][3] - e["bbox"][1]))
    if not _desc(target):
        return None, "target_without_description"
    seen, others = {_desc(target)}, []
    for e in elems:
        d = _desc(e)
        if e in inside or not d or d in seen:
            continue
        seen.add(d)
        others.append(e)
    if not others:
        return None, "no_other_elements"
    cands = rng.sample(others, min(MAX_ELEMS - 1, len(others))) + [target]
    rng.shuffle(cands)
    crit = {f"e{i + 1}": f"{_desc(e)} at {_norm_box(e['bbox'], w, h)}" for i, e in enumerate(cands)}
    key = f"e{cands.index(target) + 1}"
    return ({"type": "choice", "instructions": ELEM_INSTR, "criteria": crit}, one_hot(crit, key)), None


def main():
    args = source_args(NAME)
    w = RawWriter(NAME, args.out)
    rng = random.Random(args.seed)
    instr_zip = RemoteZip([hf_url(REPO, "AMEX/instruction_anno.zip", REV)])
    elem_zip = RemoteZip([hf_url(REPO, "AMEX/element_anno.zip", REV)])
    shot_zip = RemoteZip([hf_url(REPO, f"AMEX/screenshot.z{i:02d}", REV) for i in range(1, 9)]
                         + [hf_url(REPO, "AMEX/screenshot.zip", REV)], SHOT_PART_SIZES)
    shot_of = {n.rsplit("/", 1)[-1]: n for n in shot_zip.names()}
    elem_of = {n.rsplit("/", 1)[-1]: n for n in elem_zip.names()}
    episodes = sorted(n for n in instr_zip.names() if n.endswith(".json"))
    rng.shuffle(episodes)
    n_q = lambda: sum(w.by_type.values())  # noqa: E731
    for ep_name in episodes:
        if args.limit and n_q() >= args.limit:
            break
        ep = json.loads(instr_zip.read(ep_name))
        for step in ep["steps"]:
            if args.limit and n_q() >= args.limit:
                break
            act = ACTION_MAP.get(step["action"])
            if act is None:
                w.skip(f"action={step['action']}")
                continue
            img = step["image_path"]
            if img not in shot_of:
                w.skip("screenshot_missing")
                continue
            stem = img.rsplit(".", 1)[0]
            path = save_image(image_from_bytes(shot_zip.read(shot_of[img])), NAME, stem)
            state = {"task": ep["instruction"], "step": step["step_id"], "screenshot": {"image": {"path": path}}}
            qs = {"next_action": {"type": "choice", "instructions": ACTION_INSTR, "criteria": dict(ACTIONS)},
                  "task_complete": {"type": "noul", "instructions": DONE_INSTR}}
            ts = {"next_action": one_hot(ACTIONS, act), "task_complete": 1.0 if act == "done" else 0.0}
            if act == "click":
                anno = elem_of.get(f"{stem}.json")
                if anno is None:
                    w.skip("element_anno_missing")
                else:
                    elems = json.loads(elem_zip.read(anno)).get("clickable_elements") or []
                    got, why = element_question(step, elems, rng)
                    if got:
                        qs["tap_element"], ts["tap_element"] = got
                    else:
                        w.skip(why)
            meta = {"episode_id": ep["episode_id"], "step": step["step_id"], "orig_action": step["action"],
                    "touch_coord": step["touch_coord"], "lift_coord": step["lift_coord"],
                    "device_dim": step["device_dim"], "type_text": step["type_text"],
                    "package_name": step["package_name"]}
            w.write(record(NAME, w.n, state, qs, ts, lang="en", grid_id=f"{NAME}/{stem}",
                           source_split="train", meta=meta))
    w.close()


if __name__ == "__main__":
    main()
