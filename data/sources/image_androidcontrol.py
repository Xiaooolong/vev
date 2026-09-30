"""AndroidControl (google-research): each screenshot of an episode -> next_action choice + task_complete noul.

Source: the original gzip TFRecord shards on GCS (gs://gresearch/android_control, 20 shards, ~50 GB), streamed
from the start of each shard in seeded shard order and parsed without TensorFlow; a run stops downloading as soon
as --limit questions are written. The HF mirror smolagents/android-control stores each shard as one ~200 MB row
group, so it cannot be read partially.

Episode = goal + N actions + N+1 screenshots. Screenshot i < N gets its action as the answer; the final screenshot
(after the last action) gets "done" and task_complete = 1. Action vocabulary = the 8 AndroidControl action types
+ done.

python -m data.sources.image_androidcontrol --out data/raw/image/androidcontrol.jsonl --limit 30
"""
import json
import random
import struct
import zlib

from data.sources._common import RawWriter, image_from_bytes, one_hot, record, save_image, source_args
from data.sources._ranged import fetch, stream

NAME = "androidcontrol"
GCS = "https://storage.googleapis.com/gresearch/android_control"
N_SHARDS = 20
ACTIONS = {
    "click": "click (tap) a UI element on the screen",
    "long_press": "long-press a UI element",
    "scroll": "scroll the screen in some direction",
    "input_text": "type text into the focused field",
    "open_app": "open an app",
    "navigate_home": "go to the home screen",
    "navigate_back": "press the back button",
    "wait": "wait for the screen to update",
    "done": "nothing more to do: the task is complete",
}
ACTION_INSTR = "Given the task and the current phone screenshot, which action should the agent take next?"
DONE_INSTR = "Given the task and the current phone screenshot, has the task already been completed?"


def _varint(b, i):
    s = v = 0
    while True:
        x = b[i]
        i += 1
        v |= (x & 0x7F) << s
        s += 7
        if x < 0x80:
            return v, i


def _fields(b):
    i = 0
    while i < len(b):
        k, i = _varint(b, i)
        wt, fn = k & 7, k >> 3
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 2:
            n, i = _varint(b, i)
            v, i = b[i:i + n], i + n
        elif wt == 1:
            v, i = b[i:i + 8], i + 8
        elif wt == 5:
            v, i = b[i:i + 4], i + 4
        else:
            raise ValueError(f"wire type {wt}")
        yield fn, wt, v


def parse_example(b: bytes) -> dict:
    """tf.train.Example -> {key: [bytes...] | [int...]} (float lists are not used by AndroidControl)."""
    out = {}
    for _, _, feats in _fields(b):
        for _, _, entry in _fields(feats):
            key, val = None, b""
            for fn, _, v in _fields(entry):
                if fn == 1:
                    key = v.decode()
                else:
                    val = v
            vals = []
            for kind, _, lst in _fields(val):
                if kind == 1:
                    vals = [v for _, _, v in _fields(lst)]
                elif kind == 3:
                    for _, wt, v in _fields(lst):
                        if wt == 2:
                            j = 0
                            while j < len(v):
                                x, j = _varint(v, j)
                                vals.append(x)
                        else:
                            vals.append(v)
            out[key] = vals
    return out


def episodes(url: str):
    """Stream one gzip TFRecord shard; yield parsed Examples."""
    d, buf = zlib.decompressobj(16 + zlib.MAX_WBITS), b""
    for chunk in stream(url):
        buf += d.decompress(chunk)
        while len(buf) >= 12:
            n = struct.unpack("<Q", buf[:8])[0]
            if len(buf) < 16 + n:
                break
            yield parse_example(buf[12:12 + n])
            buf = buf[16 + n:]


def main():
    args = source_args(NAME)
    w = RawWriter(NAME, args.out)
    split_of = {}
    for split, ids in json.loads(fetch(f"{GCS}/splits.json", "androidcontrol_splits.json").read_text()).items():
        for i in ids:
            split_of[int(i)] = split
    shards = list(range(N_SHARDS))
    random.Random(args.seed).shuffle(shards)
    n_q = lambda: sum(w.by_type.values())  # noqa: E731
    for shard in shards:
        url = f"{GCS}/android_control-{shard:05d}-of-{N_SHARDS:05d}"
        for ex in episodes(url):
            if not all(k in ex for k in ("episode_id", "goal", "actions", "screenshots")):
                w.skip("missing " + ",".join(k for k in ("episode_id", "goal", "actions", "screenshots") if k not in ex))
                continue
            ep = ex["episode_id"][0]
            goal = ex["goal"][0].decode("utf-8")
            acts = [json.loads(a) for a in ex["actions"]]
            shots = ex["screenshots"]
            instr = [s.decode("utf-8") for s in ex.get("step_instructions", [])]
            if len(shots) != len(acts) + 1:
                w.skip("screenshots != actions + 1")
                continue
            for i, png in enumerate(shots):
                if args.limit and n_q() >= args.limit:
                    break
                act = acts[i]["action_type"] if i < len(acts) else "done"
                if act not in ACTIONS:
                    w.skip(f"action={act}")
                    continue
                path = save_image(image_from_bytes(png), NAME, f"{ep}_{i}")
                state = {"task": goal, "step": i, "screenshot": {"image": {"path": path}}}
                qs = {"next_action": {"type": "choice", "instructions": ACTION_INSTR, "criteria": dict(ACTIONS)},
                      "task_complete": {"type": "noul", "instructions": DONE_INSTR}}
                ts = {"next_action": one_hot(ACTIONS, act), "task_complete": 1.0 if act == "done" else 0.0}
                meta = {"episode_id": ep, "step": i, "n_steps": len(acts), "shard": shard,
                        "action": acts[i] if i < len(acts) else None,
                        "step_instruction": instr[i] if i < len(instr) else None}
                w.write(record(NAME, w.n, state, qs, ts, lang="en", grid_id=f"{NAME}/{ep}_{i}",
                               source_split=split_of.get(ep, "unknown"), meta=meta))
            if args.limit and n_q() >= args.limit:
                break
        if args.limit and n_q() >= args.limit:
            break
    w.close()


if __name__ == "__main__":
    main()
